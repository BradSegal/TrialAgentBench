"""The rule-based developer and the fixed TrialDev policies, acting only on public designs and raw readouts.

Its planning models were fitted on 192 development worlds, shared with neither split, using those worlds' true effects.
Run it on a released programme:

    uv run python examples/rule_based_developer.py --task td-87d6661cdfed --output results/rule-based
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from types import MappingProxyType
from typing import Literal, NamedTuple, get_args

import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.api as sm
from pydantic import BaseModel, ConfigDict, Field
from rule_readout import (
    TrialDevReadoutValue,
    observed_trialdev_readout_values,
)
from scipy.special import ndtr
from trialagentbench_verifier.resource_accounting import (
    reconstruct_trialdev_terminal_value,
    verify_trialdev_terminal_value,
)
from trialagentbench_verifier.response import TrialDevFrozenProgrammeValue

from trialagentbench.contracts.trialdev.pools import TrialDevFrozenPool
from trialagentbench.io import read_json_model, sha256_file, write_json_model
from trialagentbench.trialdev.actions import (
    TrialDevActionSession,
    TrialDevAdvance,
    TrialDevReadout,
    TrialDevStartStudy,
    TrialDevStop,
)
from trialagentbench.trialdev.design import (
    TrialDevAvailableTreatment,
    TrialDevDesignSpace,
    TrialDevRequestedArm,
    trialdev_design_space,
)
from trialagentbench.trialdev.release import TrialDevAuthoredProgrammeConfiguration
from trialagentbench.trialdev.resources import (
    TRIALDEV_HORIZON_DAYS,
    TrialDevBudgetLedger,
    TrialDevLook,
    TrialDevPhaseDesignLimits,
)
from trialagentbench.trialdev.value import (
    TrialDevSponsorPreferences,
    grade_trialdev_finished_programme,
    trialdev_market_years,
)

TrialDevPolicyId = Literal["stop", "fixed_first", "readout_rank", "reference"]
CRITICAL_Z = 1.959963984540054
CONFIRMATION_SIZES = (300, 400, 600, 800, 1000, 1500, 2000, 3000)
# R7: exploration sizes searched by the reference's preposterior plan, and the network
# fractions every planned study chooses its site count from.
EXPLORATION_SIZES = (30, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 1500)
SITE_FRACTIONS = (1.0, 0.75, 0.5, 0.375, 0.25, 0.125)
RECRUITMENT_FORECAST_BUFFER = 1.5
POLICY_STUDY_IDS = {"early_safety": "study-101", "exploration": "study-201", "confirmation": "study-301"}
# Probabilists' Gauss-Hermite rule for posterior expectations over a normal benefit.
_HERMITE_NODES, _HERMITE_WEIGHTS = np.polynomial.hermite_e.hermegauss(21)
_HERMITE_WEIGHTS = _HERMITE_WEIGHTS / _HERMITE_WEIGHTS.sum()
_Floats = npt.NDArray[np.float64]
_Integers = npt.NDArray[np.int64]


class _Evidence(NamedTuple):
    """Linear model of one evidence source: ``E[estimate | truth] = slope * truth``, with excess error.

    The excess standard deviation is error beyond the estimate's sampling error, such as
    confounding bias in an observed association.
    """

    slope: float
    excess_sd: float


class _PlanningModel(NamedTuple):
    """The reference's model of one clinical programme, calibrated on development worlds.

    The value benefit is the valued all-comer contrast in benefit units and the safety risk its
    risk difference, both for a candidate's standard regimen. Across candidates the value benefit
    follows a normal mixture of (weight, mean, SD) components, which keeps the right tail of
    effective candidates that drives the value of exploration; the safety risk is normal (mean, SD). Each evidence source relates linearly to the value it measures: the initial
    observed association, and the exploration readout at the longest exploration follow-up,
    whose sampling error is ``readout_precision`` times the planning SD times ``sqrt(2 / n)``.
    A confirmation at follow-up ``f`` tests ``confirmation_benefit[f]`` times the value benefit
    with an outcome SD of ``confirmation_spread[f]`` times the planning SD.
    """

    benefit_prior: tuple[tuple[float, float, float], ...]
    safety_prior: tuple[float, float]
    association: _Evidence
    association_safety: _Evidence
    readout: _Evidence
    readout_safety: _Evidence
    readout_precision: float
    confirmation_benefit: Mapping[int, float]
    confirmation_spread: Mapping[int, float]


# Refit on paired development worlds.
REFERENCE_PLANNING_MODELS: Mapping[str, _PlanningModel] = MappingProxyType(
    {
        "easi_score_end_of_follow_up": _PlanningModel(  # atopic_dermatitis
            benefit_prior=(
                (0.7668, 0.02392, 0.133),
                (0.2332, 0.4527, 0.07408),
            ),
            safety_prior=(0.02001, 0.02956),
            association=_Evidence(1.747, 0.11),
            association_safety=_Evidence(0.4762, 0.0),
            readout=_Evidence(1.334, 0.04762),
            readout_safety=_Evidence(0.5872, 0.0),
            readout_precision=1.108,
            confirmation_benefit=MappingProxyType({112: 1.304, 224: 1.0}),
            confirmation_spread=MappingProxyType({112: 1.009, 224: 1.062}),
        ),
        "moderate_or_severe_exacerbations_per_year": _PlanningModel(  # copd_exacerbation
            benefit_prior=(
                (0.7591, 0.1277, 0.2868),
                (0.2409, 1.053, 0.1874),
            ),
            safety_prior=(0.02666, 0.04393),
            association=_Evidence(1.209, 0.43),
            association_safety=_Evidence(0.5451, 0.0),
            readout=_Evidence(1.13, 0.0),
            readout_safety=_Evidence(0.5373, 0.0),
            readout_precision=1.02,
            confirmation_benefit=MappingProxyType({364: 1.132, 728: 1.0}),
            confirmation_spread=MappingProxyType({364: 1.064, 728: 0.9799}),
        ),
        "monthly_migraine_days_end_of_follow_up": _PlanningModel(  # migraine_prevention
            benefit_prior=(
                (0.5453, -0.02246, 0.08383),
                (0.2557, 0.2361, 0.04726),
                (0.199, 0.6514, 0.1178),
            ),
            safety_prior=(0.03515, 0.04613),
            association=_Evidence(1.344, 0.26),
            association_safety=_Evidence(0.2953, 0.0),
            readout=_Evidence(1.228, 0.03375),
            readout_safety=_Evidence(0.5379, 0.0),
            readout_precision=1.124,
            confirmation_benefit=MappingProxyType({168: 1.202, 336: 1.0}),
            confirmation_spread=MappingProxyType({168: 1.058, 336: 1.087}),
        ),
        "das28_score_end_of_follow_up": _PlanningModel(  # rheumatoid_arthritis
            benefit_prior=(
                (0.4835, -0.02477, 0.07118),
                (0.2251, 0.2042, 0.03434),
                (0.2915, 0.5221, 0.112),
            ),
            safety_prior=(0.02131, 0.03365),
            association=_Evidence(1.642, 0.21),
            association_safety=_Evidence(0.5446, 0.0),
            readout=_Evidence(1.219, 0.0931),
            readout_safety=_Evidence(0.5067, 0.002308),
            readout_precision=1.094,
            confirmation_benefit=MappingProxyType({168: 1.269, 336: 1.0}),
            confirmation_spread=MappingProxyType({168: 1.011, 336: 1.078}),
        ),
    }
)
"""Frozen planning models keyed by the public endpoint field of the programme's initial cohort."""


class TrialDevPolicyRun(BaseModel):
    """One completed declared control policy and its realized development value."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    world_id: str
    task_id: str | None = None
    budget_usd: float = Field(gt=0)
    policy_id: TrialDevPolicyId
    pool_manifest_sha256: str
    value: TrialDevFrozenProgrammeValue
    readout_diagnostics: tuple[TrialDevReadoutValue, ...]
    planned_expected_value_usd: float | None = Field(
        default=None,
        description="Reference: expected net value of the executed plan before any study is sampled.",
    )


def _initial_cohort(public_root: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Link the public initial cohort and return its endpoint and covariate columns."""
    initial = public_root / "data" / "initial"
    data = pd.read_parquet(initial / "participants.parquet")
    endpoint_fields = []
    for name in ("treatments", "clinical_assessments", "safety_assessments"):
        frame = pd.read_parquet(initial / f"{name}.parquet")
        if name != "treatments":
            fields = [column for column in frame.columns if column != "participant_id"]
            if len(fields) != 1:
                raise ValueError("Initial policy assessment requires one declared endpoint")
            endpoint_fields.append(fields[0])
        data = data.merge(frame, on="participant_id", validate="one_to_one", how="outer")
    if data.isna().any().any():
        raise ValueError("Initial policy requires complete linked public cohort records")
    dictionary = pd.read_csv(public_root / "documents" / "observational_data_dictionary.csv")
    covariates = dictionary.loc[dictionary.role.eq("covariate"), "variable"].tolist() + ["site_id"]
    return data, endpoint_fields, covariates


def _initial_policy_values(
    public_root: Path, preferences: TrialDevSponsorPreferences
) -> tuple[TrialDevReadoutValue, ...]:
    """Screen assets using adjusted associations, without claiming causal identification."""
    data, endpoint_fields, covariates = _initial_cohort(public_root)
    assets = sorted(set(data.asset_id) - {"control"})
    treatments = pd.DataFrame({asset: data.asset_id.eq(asset).astype(float) for asset in assets})
    covariate_frame = pd.get_dummies(data[covariates], drop_first=True, dtype=float)
    design = sm.add_constant(pd.concat([treatments, covariate_frame], axis=1).astype(float))
    clinical_price = preferences.annual_profit_per_benefit_unit_usd / preferences.benefit_unit
    safety_price = preferences.annual_cost_per_safety_percentage_point_usd * 100
    clinical = sm.OLS(data[endpoint_fields[0]], design).fit(cov_type="HC1")
    safety = sm.OLS(data[endpoint_fields[1]], design).fit(cov_type="HC1")
    # Fitting joint utility retains clinical/safety covariance in its HC1 error; HC1 stays finite when a
    # site of the large network contributes one patient (leverage one), where HC3 is infinite.
    utility = -clinical_price * data[endpoint_fields[0]] - safety_price * data[endpoint_fields[1]]
    fit = sm.OLS(utility, design).fit(cov_type="HC1")
    return tuple(
        TrialDevReadoutValue(
            study_id="initial-observed-association",
            arm_id=asset,
            clinical_difference=float(clinical.params[asset]),
            clinical_standard_error=float(clinical.bse[asset]),
            safety_difference=float(safety.params[asset]),
            safety_standard_error=float(safety.bse[asset]),
            estimated_annual_profit_usd=float(fit.params[asset]),
            standard_error_usd=float(fit.bse[asset]),
        )
        for asset in assets
    )


class _StudyForecast(BaseModel):
    """Public planning forecast for one candidate study design."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    cost_usd: float = Field(ge=0)
    deadline_day: int = Field(ge=1)


def policy_readout_folder(phase: str) -> str:
    """Return the evidence folder of a declared policy's single readout of one phase.

    Parameters
    ----------
    phase
        Development phase of the policy study.

    Returns
    -------
    str
        Folder under the run's ``evidence`` directory.
    """
    return f"{POLICY_STUDY_IDS[phase]}-look-1"


def _protocol_visits(design: TrialDevDesignSpace, follow_up: int) -> tuple[int, ...]:
    """Return the complete protocol visit grid through a committed follow-up.

    Every declared study assesses and surveils the full grid, so its readout releases
    the visit history on which missing-at-random observation loss can depend.
    """
    return (0, *(day for day in design.visit_days if day <= follow_up))


def _forecast_study(
    design: TrialDevDesignSpace,
    limits: TrialDevPhaseDesignLimits,
    *,
    arms: int,
    per_arm: int,
    follow_up: int,
    sites: int,
    start_day: float,
) -> _StudyForecast:
    """Forecast a study's cost and completion day from public tariffs and site rates."""
    cost, duration = _forecast_cost_and_duration(
        design, limits, arms=arms, per_arm=per_arm, follow_up=follow_up, sites=sites
    )
    return _StudyForecast(cost_usd=cost, deadline_day=_deadline_day(start_day, duration, follow_up))


def _deadline_day(start_day: float, duration: float, follow_up: int) -> int:
    """Return the readout deadline of a study starting on ``start_day`` with this forecast duration."""
    return ceil(start_day + duration + follow_up + 30)


def _forecast_cost_and_duration(
    design: TrialDevDesignSpace,
    limits: TrialDevPhaseDesignLimits,
    *,
    arms: int,
    per_arm: int,
    follow_up: int,
    sites: int,
) -> tuple[float, float]:
    """Return a study's forecast cost and its days from start to the last enrolment."""
    network = design.recruitment.sites[:sites]
    total = arms * per_arm
    passing = 1 - design.recruitment.screen_failure_fraction
    # R3': twice the forecast recruitment duration and 30 days allow for arrival and
    # reporting variation; observed accrual took 0.99-1.61 times the unbuffered forecast
    # . Never a private accrual lookup; shared by the policies.
    enrolled_per_month = passing * sum(site.screening_candidates_per_month for site in network)
    recruitment_days = RECRUITMENT_FORECAST_BUFFER * 30 * total / enrolled_per_month
    duration = limits.tariff.startup_days + max(site.opening_day for site in network) + recruitment_days
    tariff = limits.tariff
    # Screen failures are random, so plan for an upper bound of about three standard deviations
    # on the candidates screened per enrolled patient; the commitment must fit the budget.
    failures = design.recruitment.screen_failure_fraction
    screened = (total + 3 * (total * failures) ** 0.5) / passing
    cost = tariff.setup + sites * tariff.per_site + screened * tariff.screening
    cost += total * (
        tariff.enrollment
        + follow_up * tariff.followup_day
        + len(_protocol_visits(design, follow_up)) * tariff.assessment
    )
    return cost, duration


def _run_study(
    session: TrialDevActionSession,
    design: TrialDevDesignSpace,
    limits: TrialDevPhaseDesignLimits,
    treatments: tuple[TrialDevAvailableTreatment, ...],
    *,
    per_arm: int,
    follow_up: int,
    sites: int,
    deadline: int,
    output: Path,
    preferences: TrialDevSponsorPreferences,
) -> tuple[TrialDevBudgetLedger, tuple[TrialDevReadoutValue, ...]]:
    """Commit one control-plus-actives study, read it at its deadline and summarise it."""
    arm_names = ("control", *(f"active-{index}" for index in range(len(treatments) - 1)))
    study_id = POLICY_STUDY_IDS[limits.phase]
    session.execute(
        TrialDevStartStudy(
            action="start_study",
            study_id=study_id,
            phase=limits.phase,
            sites=sites,
            arms=tuple(
                TrialDevRequestedArm(arm_id=name, asset_id=item.asset_id, dose=item.dose, participant_count=per_arm)
                for name, item in zip(arm_names, treatments, strict=True)
            ),
            follow_up_days=follow_up,
            assessment_days=_protocol_visits(design, follow_up),
            surveillance_days=_protocol_visits(design, follow_up),
            looks=(TrialDevLook(deadline_day=deadline),),
        ),
        action_id=study_id,
    )
    folder = policy_readout_folder(limits.phase)
    ledger = session.execute(
        TrialDevReadout(action="readout", study_id=study_id, evidence_folder=folder), action_id=study_id + "-readout"
    )
    if limits.phase == "early_safety":
        return ledger, ()
    return ledger, observed_trialdev_readout_values(output / "evidence" / folder, preferences)


def _expected_approval_value(
    relative_benefit: npt.ArrayLike,
    posterior_sd: npt.ArrayLike,
    relative_sd: npt.ArrayLike,
    per_arm: npt.ArrayLike,
    annual_profit: npt.ArrayLike,
    price: float,
) -> _Floats:
    """Posterior-expected approval power times annual profit (R1 assurance sizing).

    The benefit is uncertain, ``b ~ N(relative_benefit, posterior_sd**2)``; profit moves
    with it at ``price`` per benefit unit. The expectation of the normal-approximation
    power of the two-sided 5% test times the profit uses a 21-node Gauss-Hermite rule.
    A zero ``posterior_sd`` is exactly point power times profit. Arguments
    broadcast together, so one call values many beliefs against many designs.
    """
    mean = np.asarray(relative_benefit, dtype=float)[..., None]
    benefits = mean + np.asarray(posterior_sd, dtype=float)[..., None] * _HERMITE_NODES
    scale = np.sqrt(np.asarray(per_arm, dtype=float)[..., None] / 2) / np.asarray(relative_sd, dtype=float)[..., None]
    power = ndtr(benefits * scale - CRITICAL_Z)
    profit = np.asarray(annual_profit, dtype=float)[..., None] + price * (benefits - mean)
    return np.asarray((power * profit) @ _HERMITE_WEIGHTS)


def _site_options(network: int) -> tuple[int, ...]:
    """Return the declared site counts, largest first: fixed fractions of the network, at least two."""
    return tuple(sorted({max(2, round(fraction * network)) for fraction in SITE_FRACTIONS}, reverse=True))


@dataclass(frozen=True)
class _ConfirmationOptions:
    """Every admissible two-arm confirmation design: follow-up, size and site count, with its forecast."""

    follow_up: _Integers
    per_arm: _Integers
    sites: _Integers
    cost_usd: _Floats
    duration_days: _Floats


class _ConfirmationPlan(NamedTuple):
    """The confirmation design with the largest forecast expected net value."""

    expected_value_usd: float
    per_arm: int
    follow_up: int
    sites: int
    forecast: _StudyForecast


def _confirmation_options(design: TrialDevDesignSpace) -> _ConfirmationOptions:
    """Forecast every declared confirmation follow-up, size and site count once."""
    limits = design.phases[2]
    rows = [
        (follow_up, per_arm, sites)
        for follow_up in limits.follow_up_days
        for per_arm in CONFIRMATION_SIZES
        if limits.minimum_participants_per_arm <= per_arm <= limits.maximum_participants_per_arm
        for sites in _site_options(len(design.recruitment.sites))
    ]
    forecasts = [
        _forecast_cost_and_duration(design, limits, arms=2, per_arm=per_arm, follow_up=follow_up, sites=sites)
        for follow_up, per_arm, sites in rows
    ]
    follow_up, per_arm, sites = (np.array(column) for column in zip(*rows, strict=True))
    cost, duration = (np.array(column) for column in zip(*forecasts, strict=True))
    return _ConfirmationOptions(
        follow_up=follow_up, per_arm=per_arm, sites=sites, cost_usd=cost, duration_days=duration
    )


def _confirmation_values(
    options: _ConfirmationOptions,
    preferences: TrialDevSponsorPreferences,
    *,
    budget_usd: float,
    committed_usd: float,
    start_day: float,
    relative_benefit: _Floats,
    posterior_sd: npt.ArrayLike,
    relative_sd: dict[int, float],
    annual_profit: _Floats,
) -> tuple[_Floats, _Integers]:
    """Forecast expected net value of every confirmation design for each belief (R1).

    Parameters
    ----------
    options
        Declared confirmation designs.
    preferences
        Sponsor economics deciding the value.
    budget_usd, committed_usd
        Programme budget and the spend already committed before the confirmation.
    start_day
        Programme day the confirmation would start.
    relative_benefit, posterior_sd, annual_profit
        One entry per belief: benefit mean and standard deviation, and its annual profit.
    relative_sd
        Planning outcome standard deviation in benefit units by follow-up; designs at
        other follow-ups are not considered.

    Returns
    -------
    tuple of numpy.ndarray
        Expected net values (beliefs by designs, minus infinity where a design is
        unaffordable, untimely or unplanned) and each design's deadline day.
    """
    deadlines = np.array(
        [
            _deadline_day(start_day, duration, int(follow_up))
            for duration, follow_up in zip(options.duration_days, options.follow_up, strict=True)
        ]
    )
    planned = np.isin(options.follow_up, list(relative_sd))
    feasible = planned & (committed_usd + options.cost_usd <= budget_usd) & (deadlines <= TRIALDEV_HORIZON_DAYS)
    years = np.array([trialdev_market_years(preferences, float(day)) for day in deadlines])
    spread = np.array([relative_sd.get(int(day), 1.0) for day in options.follow_up])
    approval = _expected_approval_value(
        np.asarray(relative_benefit, dtype=float)[:, None],
        np.broadcast_to(posterior_sd, np.shape(relative_benefit))[:, None],
        spread[None, :],
        options.per_arm[None, :],
        np.asarray(annual_profit, dtype=float)[:, None],
        preferences.annual_profit_per_benefit_unit_usd,
    )
    values = approval * years - options.cost_usd - (deadlines - start_day) * preferences.programme_cost_per_day_usd
    return np.where(feasible, values, -np.inf), deadlines


def _best_confirmation(
    options: _ConfirmationOptions,
    preferences: TrialDevSponsorPreferences,
    *,
    budget_usd: float,
    committed_usd: float,
    start_day: float,
    relative_benefit: float,
    posterior_sd: float,
    relative_sd: dict[int, float],
    annual_profit: float,
) -> _ConfirmationPlan | None:
    """Choose the confirmation follow-up, size and site count maximizing forecast expected net value.

    The budget and horizon only constrain the choice; the site count is part of the
    argmax, so a slower, cheaper network is bought whenever it is worth more.

    Returns
    -------
    _ConfirmationPlan or None
        The best design, or None when no planned design is affordable and timely.
    """
    values, deadlines = _confirmation_values(
        options,
        preferences,
        budget_usd=budget_usd,
        committed_usd=committed_usd,
        start_day=start_day,
        relative_benefit=np.array([relative_benefit]),
        posterior_sd=posterior_sd,
        relative_sd=relative_sd,
        annual_profit=np.array([annual_profit]),
    )
    best = int(np.argmax(values[0]))
    if not np.isfinite(values[0, best]):
        return None
    return _ConfirmationPlan(
        expected_value_usd=float(values[0, best]),
        per_arm=int(options.per_arm[best]),
        follow_up=int(options.follow_up[best]),
        sites=int(options.sites[best]),
        forecast=_StudyForecast(cost_usd=float(options.cost_usd[best]), deadline_day=int(deadlines[best])),
    )


class _Belief(NamedTuple):
    """A normal-mixture belief: component weights, means and standard deviations."""

    weights: _Floats
    means: _Floats
    sds: _Floats

    @property
    def mean(self) -> float:
        """Return the belief's mean."""
        return float(self.weights @ self.means)

    @property
    def sd(self) -> float:
        """Return the belief's standard deviation."""
        return float(np.sqrt(max(float(self.weights @ (self.sds**2 + self.means**2)) - self.mean**2, 0.0)))


def _posterior(
    prior: tuple[tuple[float, float, float], ...], evidence: tuple[tuple[float, float, _Evidence], ...]
) -> _Belief:
    """Update a normal-mixture prior with linear normal evidence (R6).

    Each item is an estimate, its sampling standard error and its evidence model; the
    estimate's variance adds the model's excess error, so a confounded association informs
    selection without overriding randomized data. Each component updates as a normal prior,
    and its weight by the marginal likelihood of the estimate under that component.

    Parameters
    ----------
    prior
        Prior components: weight, mean and standard deviation.
    evidence
        Estimates, sampling standard errors and evidence models on the prior's scale.

    Returns
    -------
    _Belief
        Posterior mixture.
    """
    log_weights = np.log([component[0] for component in prior])
    means = np.array([component[1] for component in prior], dtype=float)
    sds = np.array([component[2] for component in prior], dtype=float)
    for estimate, error, model in evidence:
        variance = error**2 + model.excess_sd**2
        if variance == 0:
            # A readout without sampling error (for example no events in any arm) is decisive.
            return _Belief(np.ones(1), np.array([estimate / model.slope]), np.zeros(1))
        predictive = model.slope**2 * sds**2 + variance
        log_weights = log_weights - 0.5 * ((estimate - model.slope * means) ** 2 / predictive + np.log(predictive))
        precision = sds**-2 + model.slope**2 / variance
        means = (means / sds**2 + model.slope * estimate / variance) / precision
        sds = precision**-0.5
    weights = np.exp(log_weights - log_weights.max())
    return _Belief(weights / weights.sum(), means, sds)


def _readout_preposterior(belief: _Belief, error: float, model: _Evidence) -> tuple[_Floats, _Floats, _Floats]:
    """Return the posterior means, SDs and probabilities a readout of this precision can produce (R7).

    The readout's predictive law is a mixture over the belief's components; each component
    contributes Gauss-Hermite nodes, and each node's posterior is moment-matched for R1.
    """
    variance = error**2 + model.excess_sd**2
    predictive = np.sqrt(model.slope**2 * belief.sds**2 + variance)
    readouts = (model.slope * belief.means)[:, None] + predictive[:, None] * _HERMITE_NODES
    probabilities = (belief.weights[:, None] * _HERMITE_WEIGHTS).ravel()
    log_weights = np.log(belief.weights)[:, None] - 0.5 * (
        (readouts.ravel()[None, :] - model.slope * belief.means[:, None]) ** 2 / predictive[:, None] ** 2
        + 2 * np.log(predictive)[:, None]
    )
    weights = np.exp(log_weights - log_weights.max(axis=0))
    weights /= weights.sum(axis=0)
    precision = belief.sds**-2 + model.slope**2 / variance
    component_means = (
        (belief.means / belief.sds**2)[:, None] + model.slope * readouts.ravel()[None, :] / variance
    ) / precision[:, None]
    means = (weights * component_means).sum(axis=0)
    second = (weights * (component_means**2 + (1 / precision)[:, None])).sum(axis=0)
    return means, np.sqrt(np.maximum(second - means**2, 0.0)), probabilities


class _CandidateBelief(NamedTuple):
    """An asset's R6 belief from its observed association, before any study."""

    asset_id: str
    benefit: _Belief
    risk: float


def _candidate_belief(
    row: TrialDevReadoutValue, preferences: TrialDevSponsorPreferences, model: _PlanningModel
) -> _CandidateBelief:
    """Return the R6 posterior of one asset's observed association with no randomized readout yet."""
    observed = -row.clinical_difference / preferences.benefit_unit
    observed_error = row.clinical_standard_error / preferences.benefit_unit
    benefit = _posterior(model.benefit_prior, ((observed, observed_error, model.association),))
    risk = _posterior(
        ((1.0, *model.safety_prior),),
        ((row.safety_difference, row.safety_standard_error, model.association_safety),),
    )
    return _CandidateBelief(row.arm_id, benefit, risk.mean)


def _expected_maximum(values: _Floats, weights: _Floats) -> float:
    """Return ``E[max(0, X_1, ..., X_k)]`` for independent discrete ``X_i`` with atoms ``values[i]``.

    ``weights`` are the atoms' probabilities, shared by every ``X_i`` or one row per ``X_i``.
    """
    atoms = np.maximum(values, 0.0)
    levels = np.unique(atoms)
    below = ((atoms[:, :, None] <= levels) * np.broadcast_to(weights, atoms.shape)[:, :, None]).sum(axis=1)
    return float(levels @ np.diff(below.prod(axis=0), prepend=0.0))


class _ScreeningPlan(NamedTuple):
    """The early-safety and exploration design with the largest preposterior expected net value."""

    expected_value_usd: float
    candidates: int
    per_arm: int
    sites: int
    forecasts: tuple[_StudyForecast, _StudyForecast]


def _plan_screening(
    design: TrialDevDesignSpace,
    preferences: TrialDevSponsorPreferences,
    *,
    budget_usd: float,
    beliefs: tuple[_CandidateBelief, ...],
    relative_sd: float,
    early_size: int,
    model: _PlanningModel,
) -> _ScreeningPlan | None:
    """Choose candidates, exploration size and sites by preposterior expected net value (R7).

    Each plan explores the ``k`` leading candidates with ``n`` patients per arm on ``s``
    sites. Its exploration readout of a candidate follows the calibrated evidence model, so
    the candidate's R6 posterior mean after the readout is normal around its current belief,
    with the variance the readout removes.
    After the readout the sponsor confirms the best candidate or stops: each candidate
    is worth its best R1 confirmation under the remaining budget, or zero. The plan's
    value is the expectation of that maximum over independent candidates (21-node
    Gauss-Hermite rule) less the screening cost and programme time. Budget only bounds
    the plans considered, so planned value cannot fall as the budget grows.

    Parameters
    ----------
    design, preferences
        Public design space and sponsor economics.
    budget_usd
        Programme budget.
    beliefs
        Candidates in rank order with their R6 beliefs before any study.
    relative_sd
        Planning outcome standard deviation in benefit units.
    early_size
        Early-safety participants per arm.
    model
        The programme's calibrated planning model.

    Returns
    -------
    _ScreeningPlan or None
        The best plan, or None when no screening is affordable and timely. A best plan
        with nonpositive value means the sample information is not worth its cost.
    """
    early, exploration, confirmation = design.phases
    options = _confirmation_options(design)
    clinical_price = preferences.annual_profit_per_benefit_unit_usd
    safety_price = preferences.annual_cost_per_safety_percentage_point_usd * 100
    planning_sd = {
        day: relative_sd * model.confirmation_spread[day] / model.confirmation_benefit[day]
        for day in confirmation.follow_up_days
    }
    sizes = [
        size
        for size in EXPLORATION_SIZES
        if exploration.minimum_participants_per_arm <= size <= exploration.maximum_participants_per_arm
    ]
    best: _ScreeningPlan | None = None
    for count in range(1, len(beliefs) + 1):
        leading = beliefs[:count]
        for per_arm in sizes:
            readout_error = model.readout_precision * relative_sd * (2 / per_arm) ** 0.5
            means, updated_sd, probabilities = (
                np.array(column)
                for column in zip(
                    *(_readout_preposterior(belief.benefit, readout_error, model.readout) for belief in leading),
                    strict=True,
                )
            )
            risks = np.array([belief.risk for belief in leading])[:, None]
            for sites in _site_options(len(design.recruitment.sites)):
                first = _forecast_study(
                    design,
                    early,
                    arms=count + 1,
                    per_arm=early_size,
                    follow_up=min(early.follow_up_days),
                    sites=sites,
                    start_day=0,
                )
                second = _forecast_study(
                    design,
                    exploration,
                    arms=count + 1,
                    per_arm=per_arm,
                    follow_up=max(exploration.follow_up_days),
                    sites=sites,
                    start_day=first.deadline_day,
                )
                screening_cost = first.cost_usd + second.cost_usd
                if screening_cost > budget_usd or second.deadline_day > TRIALDEV_HORIZON_DAYS:
                    continue
                values, _ = _confirmation_values(
                    options,
                    preferences,
                    budget_usd=budget_usd,
                    committed_usd=screening_cost,
                    start_day=second.deadline_day,
                    relative_benefit=means.ravel(),
                    posterior_sd=updated_sd.ravel(),
                    relative_sd=planning_sd,
                    annual_profit=(clinical_price * means - safety_price * risks).ravel(),
                )
                decision = values.max(axis=1).reshape(means.shape)
                expected = (
                    _expected_maximum(decision, probabilities)
                    - screening_cost
                    - second.deadline_day * preferences.programme_cost_per_day_usd
                )
                if best is None or expected > best.expected_value_usd:
                    best = _ScreeningPlan(expected, count, per_arm, sites, (first, second))
    return best


def _run_reference_policy(
    session: TrialDevActionSession,
    design: TrialDevDesignSpace,
    configuration: TrialDevAuthoredProgrammeConfiguration,
    public_root: Path,
    output: Path,
) -> tuple[TrialDevBudgetLedger, tuple[TrialDevReadoutValue, ...], float]:
    """Plan screening by preposterior value, select on the R6 posterior and confirm on expected value.

    The policy uses public information only: the initial observed associations,
    released readouts, tariffs, site forecasts and the declared asset prior. It is
    the decision-analytic baseline. Before any study it chooses how many leading
    candidates to explore, how many patients per arm and on how many sites by the
    expected net value of the decision the exploration readout will inform (R7); it
    stops at once when no screening is worth its cost. After exploration, selection,
    go/no-go and sizing use one posterior that keeps the observed association as biased
    evidence (R6); confirmation follow-up, size and sites maximize posterior-expected
    power times value (R1), and the programme stops if and only if the best
    confirmation's expected net value is not positive. The budget is only a constraint.

    Returns
    -------
    tuple
        Terminal ledger, observed-data diagnostics and the planned expected net value.
    """
    preferences = configuration.preferences
    initial = _initial_policy_values(public_root, preferences)
    diagnostics = list(initial)
    data, endpoints, _ = _initial_cohort(public_root)
    relative_sd = float(data.loc[data.asset_id.eq("control"), endpoints[0]].std()) / preferences.benefit_unit
    if endpoints[0] not in REFERENCE_PLANNING_MODELS:
        raise ValueError(f"no calibrated reference planning model for endpoint {endpoints[0]}")
    model = REFERENCE_PLANNING_MODELS[endpoints[0]]
    clinical_price = preferences.annual_profit_per_benefit_unit_usd
    safety_price = preferences.annual_cost_per_safety_percentage_point_usd * 100
    standard = {item.asset_id: item for item in design.treatments if item.dose == "standard"}
    beliefs = tuple(
        sorted(
            (_candidate_belief(row, preferences, model) for row in initial),
            key=lambda belief: (
                -(clinical_price * belief.benefit.mean - safety_price * belief.risk),
                belief.asset_id,
            ),
        )
    )
    early, exploration, _ = design.phases
    early_size = max(20, early.minimum_participants_per_arm)
    screening = _plan_screening(
        design,
        preferences,
        budget_usd=configuration.budget_usd,
        beliefs=beliefs,
        relative_sd=relative_sd,
        early_size=early_size,
        model=model,
    )
    if screening is None or screening.expected_value_usd <= 0:
        return session.execute(TrialDevStop(action="stop"), action_id="policy-stop"), tuple(diagnostics), 0.0
    treatments = (design.treatments[0], *(standard[belief.asset_id] for belief in beliefs[: screening.candidates]))
    _run_study(
        session,
        design,
        early,
        treatments,
        per_arm=early_size,
        follow_up=min(early.follow_up_days),
        sites=screening.sites,
        deadline=screening.forecasts[0].deadline_day,
        output=output,
        preferences=preferences,
    )
    ledger, estimates = _run_study(
        session,
        design,
        exploration,
        treatments,
        per_arm=screening.per_arm,
        follow_up=max(exploration.follow_up_days),
        sites=screening.sites,
        deadline=max(
            screening.forecasts[1].deadline_day,
            ceil(read_json_model(TrialDevBudgetLedger, output / "budget.json").current_day) + 1,
        ),
        output=output,
        preferences=preferences,
    )
    diagnostics.extend(estimates)
    planned = screening.expected_value_usd
    observed_association = {row.arm_id: row for row in initial}
    candidates = []
    for row in estimates:
        association = observed_association[treatments[int(row.arm_id.removeprefix("active-")) + 1].asset_id]
        belief = _posterior(
            model.benefit_prior,
            (
                (
                    -association.clinical_difference / preferences.benefit_unit,
                    association.clinical_standard_error / preferences.benefit_unit,
                    model.association,
                ),
                (
                    -row.clinical_difference / preferences.benefit_unit,
                    row.clinical_standard_error / preferences.benefit_unit,
                    model.readout,
                ),
            ),
        )
        risk = _posterior(
            ((1.0, *model.safety_prior),),
            (
                (association.safety_difference, association.safety_standard_error, model.association_safety),
                (row.safety_difference, row.safety_standard_error, model.readout_safety),
            ),
        ).mean
        candidates.append((clinical_price * belief.mean - safety_price * risk, belief.mean, belief.sd, row))
    if not candidates:
        return session.execute(TrialDevStop(action="stop"), action_id="policy-stop"), tuple(diagnostics), planned
    annual_profit, benefit, posterior_sd, selected = max(candidates, key=lambda item: (item[0], item[3].arm_id))
    chosen = treatments[int(selected.arm_id.removeprefix("active-")) + 1]
    # The readout's implied outcome SD, rescaled to each confirmation follow-up's planning spread.
    readout_sd = (
        selected.clinical_standard_error / (2 / screening.per_arm) ** 0.5 / preferences.benefit_unit
    ) / model.readout_precision
    plan = _best_confirmation(
        _confirmation_options(design),
        preferences,
        budget_usd=configuration.budget_usd,
        committed_usd=ledger.spent_usd + ledger.remaining_liability_usd,
        start_day=ledger.current_day,
        relative_benefit=benefit,
        posterior_sd=posterior_sd,
        relative_sd={
            day: readout_sd * model.confirmation_spread[day] / model.confirmation_benefit[day]
            for day in design.phases[2].follow_up_days
        },
        annual_profit=annual_profit,
    )
    if plan is None or plan.expected_value_usd <= 0:
        return session.execute(TrialDevStop(action="stop"), action_id="policy-stop"), tuple(diagnostics), planned
    ledger, confirmed = _run_study(
        session,
        design,
        design.phases[2],
        (design.treatments[0], chosen),
        per_arm=plan.per_arm,
        follow_up=plan.follow_up,
        sites=plan.sites,
        deadline=plan.forecast.deadline_day,
        output=output,
        preferences=preferences,
    )
    diagnostics.extend(confirmed)
    if (
        len(confirmed) == 1
        and confirmed[0].clinical_standard_error > 0
        and -confirmed[0].clinical_difference / confirmed[0].clinical_standard_error >= CRITICAL_Z
        and confirmed[0].estimated_annual_profit_usd > 0
        and chosen.dose is not None
    ):
        ledger = session.execute(
            TrialDevAdvance(action="advance", asset_id=chosen.asset_id, dose=chosen.dose), action_id="policy-advance"
        )
        return ledger, tuple(diagnostics), planned
    return session.execute(TrialDevStop(action="stop"), action_id="policy-stop"), tuple(diagnostics), planned


def run_trialdev_control_policy(
    *,
    pool_root: Path,
    configuration: TrialDevAuthoredProgrammeConfiguration,
    output: Path,
    policy: TrialDevPolicyId,
    task_id: str | None = None,
    public_root: Path | None = None,
    profit_multiplier: float = 1.0,
) -> TrialDevPolicyRun:
    """Execute a declared control through the public authored action interface.

    Parameters
    ----------
    pool_root, configuration
        Authenticated fixed pool and task-owned resources and sponsor preferences.
    output
        New directory retaining requests, raw readouts, ledger and control result.
    policy
        Immediate stop, fixed-first three-phase advance, study-evidence ranking
        or the evidence-responsive reference policy.
        Fixed and ranking controls run phase-minimum arms at the shortest follow-up
        on the full network, each read at its forecast deadline.
    task_id
        Released task identity when this run belongs to a control panel.
    public_root
        Participant world directory, required for the reference policy's initial
        observed-association screen. Private reference fields never enter its
        candidate selection or study planning.
    profit_multiplier
        Sponsor-economics run setting: scales the annual profit per benefit unit
        and the safety price together, leaving study costs, overhead and clocks
        unchanged. Every policy decides and is graded under the scaled
        economics, so a sensitivity band re-decides rather than revalues.

    Returns
    -------
    TrialDevPolicyRun
        Actual terminal value and observed-data diagnostics. The reference records
        its plan's expected value before sampling.

    Raises
    ------
    ValueError
        If input identity, public data, study admission or independent terminal
        reconstruction fails. A failed execution is not replaced by a zero stop.
    """
    if policy not in get_args(TrialDevPolicyId):
        raise ValueError("Unknown declared TrialDev control policy")
    if not math.isfinite(profit_multiplier) or profit_multiplier <= 0:
        raise ValueError("profit multiplier must be finite and positive")
    if profit_multiplier != 1.0:
        preferences = configuration.preferences
        configuration = configuration.model_copy(
            update={
                "preferences": TrialDevSponsorPreferences.model_validate(
                    preferences.model_dump()
                    | {
                        "annual_profit_per_benefit_unit_usd": preferences.annual_profit_per_benefit_unit_usd
                        * profit_multiplier,
                        "annual_cost_per_safety_percentage_point_usd": (
                            preferences.annual_cost_per_safety_percentage_point_usd * profit_multiplier
                        ),
                    }
                )
            }
        )
    if output.resolve().is_relative_to(pool_root.resolve()):
        raise ValueError("Policy output cannot change its frozen pool")
    if sha256_file(pool_root / "manifest.json") != configuration.pool_manifest_sha256:
        raise ValueError("Policy configuration identifies a different frozen pool")
    output.mkdir(parents=True, exist_ok=False)
    write_json_model(output / "configuration.json", configuration)
    budget = output / "budget.json"
    write_json_model(
        budget,
        TrialDevBudgetLedger(
            budget_usd=configuration.budget_usd,
            pool_manifest_sha256=configuration.pool_manifest_sha256,
            recruitment_seed=configuration.recruitment_seed,
        ),
    )
    session = TrialDevActionSession(
        pool_root, budget, output / "evidence", output / "commitments", configuration.recruitment_seed
    )
    manifest = read_json_model(TrialDevFrozenPool, pool_root / "manifest.json")
    design = trialdev_design_space(manifest)
    candidates = {design.treatments[1].asset_id: design.treatments[1]}
    for item in design.treatments[1:]:
        candidates.setdefault(item.asset_id, item)
    uses_readout = policy not in {"stop", "fixed_first"}
    active = tuple(candidates.values()) if uses_readout else (design.treatments[1],)
    diagnostics: list[TrialDevReadoutValue] = []
    planned: float | None = None
    if policy == "reference":
        if public_root is None:
            raise ValueError("Reference policy requires its participant world directory")
        ledger, reference_diagnostics, planned = _run_reference_policy(
            session, design, configuration, public_root, output
        )
        diagnostics.extend(reference_diagnostics)
    elif policy == "stop":
        ledger = session.execute(TrialDevStop(action="stop"), action_id="policy-stop")
    else:
        sites = len(design.recruitment.sites)
        # Each study reads out at its forecast deadline on the full network, as the lean programme does.
        for limits in design.phases:
            treatments = (design.treatments[0], *active)
            arm_names = ("control", *(f"active-{index}" for index in range(len(active))))
            followup = min(limits.follow_up_days)
            per_arm = limits.minimum_participants_per_arm
            current = read_json_model(TrialDevBudgetLedger, budget)
            forecast = _forecast_study(
                design,
                limits,
                arms=len(treatments),
                per_arm=per_arm,
                follow_up=followup,
                sites=sites,
                start_day=current.current_day,
            )
            if (
                current.spent_usd + current.remaining_liability_usd + forecast.cost_usd > configuration.budget_usd
                or forecast.deadline_day > TRIALDEV_HORIZON_DAYS
            ):
                # A fixed control that cannot afford or finish its next study stops instead of requesting it.
                ledger = session.execute(TrialDevStop(action="stop"), action_id="policy-stop")
                break
            ledger, _ = _run_study(
                session,
                design,
                limits,
                treatments,
                per_arm=per_arm,
                follow_up=followup,
                sites=sites,
                deadline=forecast.deadline_day,
                output=output,
                preferences=configuration.preferences,
            )
            if uses_readout and limits.phase != "early_safety":
                estimates = observed_trialdev_readout_values(
                    output / "evidence" / policy_readout_folder(limits.phase), configuration.preferences
                )
                diagnostics.extend(estimates)
                best = max(estimates, key=lambda row: row.estimated_annual_profit_usd, default=None)
                if best is None or best.estimated_annual_profit_usd - 1.645 * best.standard_error_usd <= 0:
                    ledger = session.execute(TrialDevStop(action="stop"), action_id="policy-stop")
                    break
                active = (treatments[arm_names.index(best.arm_id)],)
        else:
            selected = active[0]
            if selected.dose is None:
                raise ValueError("Control policy selected no active regimen")
            ledger = session.execute(
                TrialDevAdvance(action="advance", asset_id=selected.asset_id, dose=selected.dose),
                action_id="policy-advance",
            )
    value = grade_trialdev_finished_programme(
        pool_root=pool_root, ledger=ledger, preferences=configuration.preferences
    )
    independent = reconstruct_trialdev_terminal_value(
        ledger_path=budget, configuration_path=output / "configuration.json", pool_path=pool_root / "manifest.json"
    )
    verify_trialdev_terminal_value(independent, value)
    result = TrialDevPolicyRun(
        world_id=manifest.world_id,
        task_id=task_id,
        budget_usd=configuration.budget_usd,
        policy_id=policy,
        pool_manifest_sha256=configuration.pool_manifest_sha256,
        value=value,
        readout_diagnostics=tuple(diagnostics),
        planned_expected_value_usd=planned,
    )
    write_json_model(output / "result.json", result)
    return result


def main() -> None:
    """Run one policy on one released programme and print its development value."""
    import argparse
    import json

    from trialagentbench.contracts.release.archive_safety import materialized_release_archive

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory for the run")
    parser.add_argument("--policy", default="reference", choices=get_args(TrialDevPolicyId))
    parser.add_argument("--release", type=Path, default=Path("benchmark/full"))
    args = parser.parse_args()
    catalog = args.release / "trialdev-evaluator.assets.json"
    digests = {name: digest for digest, name in map(str.split, (args.release / "SHA256SUMS").read_text().splitlines())}
    with materialized_release_archive(
        catalog, expected_sha256=digests[catalog.name], task_ids=(args.task,)
    ) as materialized:
        root = materialized.root
        case = json.loads((root / "participant/tasks" / args.task / "task.json").read_text())["case_id"]
        run = run_trialdev_control_policy(
            pool_root=root / "evaluation/worlds" / case / "pool",
            configuration=read_json_model(
                TrialDevAuthoredProgrammeConfiguration,
                root / "evaluation/tasks" / args.task / "authored_programme.json",
            ),
            output=args.output,
            policy=args.policy,
            task_id=args.task,
            public_root=root / "participant/worlds" / case,
        )
    print(f"{args.task}: {args.policy}; development value US${run.value.net_value_usd / 1e6:.1f}M")


if __name__ == "__main__":
    main()
