"""Declared MAR-valid public analysis of TrialDev study readouts.

One analysis serves every declared control policy. Every contrast is the
intention-to-treat difference between randomised arms (the treatment-policy
estimand under conduct), estimated from the released records only:

* continuous endpoints: a mixed model for repeated measures fitted in each arm
  (maximum likelihood by EM, unstructured covariance, visit means by arm) on the
  primary baseline field and the scheduled visits, with a robust HC1 sandwich
  variance and a t reference on the smaller arm's size minus one, so the interval
  stays valid when the residual variance grows with the outcome level;
* recurrent-event counts: negative binomial (NB2) regression with a log-exposure
  offset and the primary baseline field, standardised to the readout's patients,
  with a robust (sandwich) covariance;
* safety: Kaplan-Meier risk of a first released event by the committed follow-up,
  censoring each patient at the readout cutoff, so an early readout does not count
  unfinished follow-up as event-free.

These analyses are valid when observation loss depends on baseline, released
history and reported events (missing at random). They are not valid under
outcome-dependent (MNAR) loss, which the construct does not score.
"""

from __future__ import annotations

import math
from collections.abc import Hashable
from pathlib import Path
from statistics import NormalDist

import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.api as sm
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import t

from trialagentbench.io import read_json_model
from trialagentbench.trialdev.materialization import TrialDevReadoutManifest
from trialagentbench.trialdev.value import TrialDevSponsorPreferences, trialdev_annual_net_profit_usd

CRITICAL_Z = NormalDist().inv_cdf(0.975)
EM_ITERATIONS = 5_000
EM_TOLERANCE = 1e-10
_EXCLUDED_BASELINE = {"participant_id", "enrollment_day", "consent_time"}

FloatArray = npt.NDArray[np.float64]


class TrialDevReadoutValue(BaseModel):
    """Declared reference contrast of one active arm against control in one readout."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    study_id: str
    arm_id: str
    clinical_difference: float
    clinical_standard_error: float = Field(ge=0)
    safety_difference: float
    safety_standard_error: float = Field(ge=0)
    estimated_annual_profit_usd: float
    standard_error_usd: float = Field(ge=0)


def _visit_patterns(missing: npt.NDArray[np.bool_]) -> list[tuple[npt.NDArray[np.bool_], npt.NDArray[np.intp]]]:
    """Group patients by their pattern of observed variables."""
    groups: dict[bytes, list[int]] = {}
    for row, mask in enumerate(missing):
        groups.setdefault(mask.tobytes(), []).append(row)
    return [(~missing[rows[0]], np.asarray(rows)) for rows in groups.values()]


def _maximum_likelihood(x: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return the maximum-likelihood mean and covariance of a normal sample missing at random (EM).

    Raises
    ------
    ValueError
        If EM does not converge.
    """
    p = x.shape[1]
    patterns = _visit_patterns(np.isnan(x))
    mean = np.nanmean(x, axis=0)
    covariance = np.diag(np.nanvar(x, axis=0))
    for _ in range(EM_ITERATIONS):
        total = np.zeros(p)
        cross = np.zeros((p, p))
        for observed, rows in patterns:
            missing = ~observed
            block = np.empty((len(rows), p))
            block[:, observed] = x[np.ix_(rows, observed)]
            if missing.any():
                coefficient = np.linalg.solve(
                    covariance[np.ix_(observed, observed)], covariance[np.ix_(observed, missing)]
                ).T
                block[:, missing] = mean[missing] + (x[np.ix_(rows, observed)] - mean[observed]) @ coefficient.T
                conditional = (
                    covariance[np.ix_(missing, missing)] - coefficient @ covariance[np.ix_(observed, missing)]
                )
                cross[np.ix_(missing, missing)] += conditional * len(rows)
            total += block.sum(axis=0)
            cross += block.T @ block
        updated_mean = total / len(x)
        updated_covariance = cross / len(x) - np.outer(updated_mean, updated_mean)
        change = max(np.max(np.abs(updated_mean - mean)), np.max(np.abs(updated_covariance - covariance)))
        mean, covariance = updated_mean, updated_covariance
        if change < EM_TOLERANCE:
            return mean, covariance
    raise ValueError("Readout mixed-model likelihood did not converge")


def _observed_scores(x: FloatArray, mean: FloatArray, covariance: FloatArray) -> FloatArray:
    """Return each patient's observed-data score for the mean and the covariance's upper triangle."""
    n, p = x.shape
    upper = np.triu_indices(p)
    scores = np.zeros((n, p + len(upper[0])))
    for observed, rows in _visit_patterns(np.isnan(x)):
        if not observed.any():
            continue
        inverse = np.zeros((p, p))
        inverse[np.ix_(observed, observed)] = np.linalg.inv(covariance[np.ix_(observed, observed)])
        weighted = np.nan_to_num(x[rows] - mean) @ inverse
        scores[rows, :p] = weighted
        # d log L / d sigma_jk counts an off-diagonal element twice.
        gradient = weighted[:, :, None] * weighted[:, None, :] - inverse
        gradient *= np.where(np.eye(p, dtype=bool), 0.5, 1.0)
        scores[rows, p:] = gradient[:, upper[0], upper[1]]
    return scores


def _robust_mean(data: FloatArray, column: int) -> tuple[float, float]:
    """Return the likelihood mean of one variable and its HC1 sandwich variance.

    The sandwich ``A^-1 B A^-1`` uses the observed information ``A`` (a central difference of the
    analytic total score) and the outer product ``B`` of the per-patient scores, scaled by
    ``n / (n - 1)``. It remains valid when the residual variance depends on the outcome level, and with
    no missing data it equals the sample variance of the mean.
    """
    centre = np.nanmean(data, axis=0)
    scale = np.nanstd(data, axis=0)
    scale[~(scale > 0)] = 1.0
    x = (data - centre) / scale
    n, p = x.shape
    mean, covariance = _maximum_likelihood(x)
    upper = np.triu_indices(p)
    parameters = np.concatenate([mean, covariance[upper]])

    def total_score(values: FloatArray) -> FloatArray:
        trial = np.zeros((p, p))
        trial[upper] = values[p:]
        total: FloatArray = _observed_scores(x, values[:p], trial + np.triu(trial, 1).T).sum(axis=0)
        return total

    hessian = np.empty((len(parameters), len(parameters)))
    for position, value in enumerate(parameters):
        step = 1e-5 * max(1.0, abs(value))
        up, down = parameters.copy(), parameters.copy()
        up[position] += step
        down[position] -= step
        hessian[:, position] = (total_score(up) - total_score(down)) / (2 * step)
    bread = np.linalg.inv(-(hessian + hessian.T) / 2)
    scores = _observed_scores(x, mean, covariance)
    variance = (bread @ scores.T @ scores @ bread)[column, column] * n / (n - 1)
    if not (np.isfinite(variance) and variance > 0):
        raise ValueError("Readout mixed-model variance is not finite and positive")
    return float(mean[column] * scale[column] + centre[column]), float(variance * scale[column] ** 2)


def _arm_visits(root: Path, study: TrialDevReadoutManifest) -> tuple[pd.DataFrame, list[int]]:
    """Return one row per enrolled patient with baseline and scheduled visit outcomes."""
    baseline = pd.read_parquet(root / "baseline.parquet")
    if not baseline.participant_id.is_unique:
        raise ValueError("Readout baseline has duplicate participant identifiers")
    records = pd.read_parquet(root / "measurements.parquet")
    if not set(records.participant_id) <= set(baseline.participant_id):
        raise ValueError("Readout clinical records identify unenrolled patients")
    enrollment = baseline.set_index("participant_id").enrollment_day
    records = records.loc[records.outcome_id.eq("clinical_outcome")]
    visits = [day for day in study.assessment_days if day <= study.follow_up_days]
    wide = pd.DataFrame(index=baseline.participant_id)
    for day in visits:
        # Released visit dates are enrolment plus the integer protocol day.
        at_visit = records.loc[
            records.occurrence_time.eq(records.participant_id.map(enrollment) + day) & records.observed
        ]
        if not at_visit.participant_id.is_unique:
            raise ValueError("Readout repeats a scheduled clinical measurement")
        wide[f"visit_{day}"] = at_visit.set_index("participant_id").observed_outcome.astype(float)
    primary = next(
        (
            column
            for column in baseline.columns
            if column not in _EXCLUDED_BASELINE and pd.api.types.is_numeric_dtype(baseline[column])
        ),
        None,
    )
    if primary is not None:
        wide.insert(0, "baseline", baseline.set_index("participant_id")[primary].astype(float))
    return wide, visits


def _distinct_columns(wide: pd.DataFrame, columns: list[Hashable], *, keep: Hashable) -> list[Hashable]:
    """Drop each column that copies another on their jointly observed rows, keeping the more complete one.

    A day-0 visit is a copy of the baseline wherever it is observed; when some patients miss it, both
    would enter the mixed model as perfectly collinear variables and make its covariance singular.
    """
    retained: list[Hashable] = []
    for column in columns:
        values = wide[column]
        duplicate = next(
            (
                other
                for other in retained
                if (jointly := values.notna() & wide[other].notna()).sum() >= 2
                and np.allclose(values[jointly], wide[other][jointly])
            ),
            None,
        )
        if duplicate is None:
            retained.append(column)
        elif column == keep or (duplicate != keep and values.notna().sum() >= wide[duplicate].notna().sum()):
            retained[retained.index(duplicate)] = column
    return retained


def _repeated_measures_contrasts(readout_root: Path, study: TrialDevReadoutManifest) -> dict[str, tuple[float, float]]:
    """Per-arm MMRM final-visit means contrasted with control, with a z-equivalent robust standard error."""
    means = {}
    for arm in study.arms:
        wide, _ = _arm_visits(readout_root / "arms" / arm.arm_id, study)
        final = f"visit_{study.follow_up_days}"
        if final not in wide or wide[final].notna().sum() < 2:
            continue
        # A variable with fewer than two released values (an unreached visit) carries
        # no information and cannot be modelled; the final visit is always retained.
        columns = [
            column
            for column in wide
            if wide[column].notna().sum() >= 2 and (column == final or wide[column].nunique() > 1)
        ]
        columns = _distinct_columns(wide, columns, keep=final)
        means[arm.arm_id] = (*_robust_mean(wide[columns].to_numpy(dtype=float), columns.index(final)), len(wide))
    if "control" not in means:
        return {}
    control_mean, control_variance, control_count = means.pop("control")
    contrasts = {}
    for arm_id, (mean, variance, count) in means.items():
        # The t reference at the smaller arm's degrees of freedom is expressed as a z-equivalent error.
        degrees = min(count, control_count) - 1
        contrasts[arm_id] = (
            mean - control_mean,
            math.sqrt(variance + control_variance) * float(t.ppf(0.975, degrees)) / CRITICAL_Z,
        )
    return contrasts


def _recurrent_rate_contrasts(
    readout_root: Path, study: TrialDevReadoutManifest, *, programme_clock: bool
) -> dict[str, tuple[float, float]]:
    """NB2 annual-rate contrasts with exposure offset, baseline adjustment and sandwich covariance."""
    frames = []
    for arm in study.arms:
        root = readout_root / "arms" / arm.arm_id
        baseline = pd.read_parquet(root / "baseline.parquet")
        if not baseline.participant_id.is_unique:
            raise ValueError("Readout baseline has duplicate participant identifiers")
        records = pd.read_parquet(root / "recurrent_events.parquet")
        if not set(records.participant_id) <= set(baseline.participant_id):
            raise ValueError("Readout clinical records identify unenrolled patients")
        enrollment = baseline.set_index("participant_id").enrollment_day
        ends = records.loc[~records.event.astype(bool)].set_index("participant_id").occurrence_time
        if not ends.index.is_unique or set(ends.index) != set(enrollment.index):
            raise ValueError("Readout recurrent records lack one observation end per patient")
        # Released times are programme days; the adverse control deliberately keeps them
        # as the invalid programme-clock exposure.
        exposure = (ends if programme_clock else ends - enrollment).reindex(enrollment.index)
        counts = (
            records.loc[records.event.astype(bool)]
            .groupby("participant_id")
            .size()
            .reindex(enrollment.index, fill_value=0)
        )
        primary = next(
            column
            for column in baseline.columns
            if column not in _EXCLUDED_BASELINE and pd.api.types.is_numeric_dtype(baseline[column])
        )
        frames.append(
            pd.DataFrame(
                {
                    "arm_id": arm.arm_id,
                    "count": counts.to_numpy(dtype=float),
                    "years": exposure.to_numpy(dtype=float) / 365.0,
                    "baseline": baseline.set_index("participant_id")[primary]
                    .reindex(enrollment.index)
                    .to_numpy(float),
                }
            )
        )
    data = pd.concat(frames, ignore_index=True)
    data = data.loc[data.years.gt(0)]
    # An arm needs two followed patients and an observed event for a finite log rate.
    supported = [
        arm.arm_id
        for arm in study.arms
        if data.arm_id.eq(arm.arm_id).sum() >= 2 and data.loc[data.arm_id.eq(arm.arm_id), "count"].sum() > 0
    ]
    if "control" not in supported:
        return {}
    data = data.loc[data.arm_id.isin(supported)]
    actives = [arm_id for arm_id in supported if arm_id != "control"]
    spread = float(data.baseline.std())
    standardised = (data.baseline - data.baseline.mean()) / (spread if spread > 0 else 1.0)
    design = np.column_stack(
        [np.ones(len(data)), *(data.arm_id.eq(arm_id).to_numpy(float) for arm_id in actives), standardised]
    )
    coefficients, covariance = _negative_binomial(data["count"].to_numpy(), design, data.years.to_numpy())

    def marginal(position: int | None) -> tuple[float, FloatArray]:
        counterfactual = design.copy()
        counterfactual[:, 1 : 1 + len(actives)] = 0.0
        if position is not None:
            counterfactual[:, 1 + position] = 1.0
        rates = np.exp(counterfactual @ coefficients)
        return float(rates.mean()), (counterfactual * rates[:, None]).mean(axis=0)

    control_rate, control_gradient = marginal(None)
    contrasts = {}
    for position, arm_id in enumerate(actives):
        rate, gradient = marginal(position)
        difference = gradient - control_gradient
        contrasts[arm_id] = (rate - control_rate, math.sqrt(float(difference @ covariance @ difference)))
    return contrasts


def _negative_binomial(counts: FloatArray, design: FloatArray, years: FloatArray) -> tuple[FloatArray, FloatArray]:
    """NB2 log-rate coefficients with a sandwich covariance, started from the Poisson fit.

    When the overdispersion estimate reaches its boundary (no extra-Poisson variation
    in a small readout), the NB2 maximum is the Poisson fit, which is returned with its
    own sandwich covariance.
    """
    poisson = sm.GLM(counts, design, family=sm.families.Poisson(), exposure=years).fit(cov_type="HC0")
    if not np.isfinite(poisson.params).all():
        raise ValueError("Readout Poisson rate analysis is not finite")
    model = sm.NegativeBinomial(counts, design, exposure=years)
    columns = design.shape[1]
    for method in ("newton", "bfgs"):
        fit = model.fit(
            start_params=np.append(poisson.params, 0.5), disp=0, method=method, maxiter=500, cov_type="HC0"
        )
        parameters, covariance = np.asarray(fit.params), np.asarray(fit.cov_params())
        if (
            fit.mle_retvals["converged"]
            and np.isfinite(parameters).all()
            and np.isfinite(covariance).all()
            and parameters[-1] > 1e-8
        ):
            return parameters[:columns], covariance[:columns, :columns]
    return np.asarray(poisson.params), np.asarray(poisson.cov_params())


def _kaplan_meier_risk(root: Path, study: TrialDevReadoutManifest) -> tuple[float, float] | None:
    """Kaplan-Meier risk of a first released safety event by follow-up, with Greenwood variance."""
    baseline = pd.read_parquet(root / "baseline.parquet")
    if not baseline.participant_id.is_unique:
        raise ValueError("Readout baseline has duplicate participant identifiers")
    if len(baseline) < 2:
        return None
    safety = pd.read_parquet(root / "safety_events.parquet")
    if not set(safety.participant_id) <= set(baseline.participant_id):
        raise ValueError("Readout safety events identify unenrolled patients")
    enrollment = baseline.set_index("participant_id").enrollment_day
    first = (safety.groupby("participant_id").occurrence_time.min() - enrollment).reindex(enrollment.index)
    follow_up = np.minimum(study.follow_up_days, study.readout_day - enrollment.to_numpy(dtype=float))
    event = first.notna().to_numpy() & (first.fillna(np.inf).to_numpy() <= follow_up)
    time = np.where(event, first.fillna(np.inf).to_numpy(), follow_up)
    survival, greenwood = 1.0, 0.0
    for moment in np.unique(time[event]):
        at_risk = int(np.sum(time >= moment))
        events = int(np.sum(event & (time == moment)))
        survival *= 1 - events / at_risk
        if at_risk > events:
            greenwood += events / (at_risk * (at_risk - events))
    if not np.any(time >= study.follow_up_days) and survival > 0:
        # No patient was followed to the committed horizon: the risk is not estimable.
        return None
    return 1 - survival, survival**2 * greenwood


def observed_trialdev_readout_values(
    readout_root: Path,
    preferences: TrialDevSponsorPreferences,
    *,
    programme_clock: bool = False,
) -> tuple[TrialDevReadoutValue, ...]:
    """Estimate each active arm's benefit, safety and value contrast with the declared analysis.

    Parameters
    ----------
    readout_root
        Raw acquired participant files and their public study manifest.
    preferences
        Public sponsor coefficients, with no private treatment contrasts.
    programme_clock
        Deliberately invalid recurrent-event exposure for the declared clock
        negative control. Ordinary policies use elapsed follow-up.

    Returns
    -------
    tuple of TrialDevReadoutValue
        Available active-arm contrasts. An arm without two released final
        outcomes (or a readout without a supported control) is omitted, so a
        policy can stop for inadequate information.

    Raises
    ------
    ValueError
        If the readout records are internally inconsistent or the count model fails.

    Examples
    --------
    ``observed_trialdev_readout_values(Path("evidence/study-301-look-1"), preferences)``.
    """
    study = read_json_model(TrialDevReadoutManifest, readout_root / "study.json")
    if "control" not in {arm.arm_id for arm in study.arms}:
        raise ValueError("Policy readout has no randomized control")
    recurrent = (readout_root / "arms" / "control" / "recurrent_events.parquet").is_file()
    clinical = (
        _recurrent_rate_contrasts(readout_root, study, programme_clock=programme_clock)
        if recurrent
        else _repeated_measures_contrasts(readout_root, study)
    )
    risks = {arm.arm_id: _kaplan_meier_risk(readout_root / "arms" / arm.arm_id, study) for arm in study.arms}
    control_risk = risks["control"]
    if control_risk is None:
        return ()
    clinical_price = preferences.annual_profit_per_benefit_unit_usd / preferences.benefit_unit
    safety_price = preferences.annual_cost_per_safety_percentage_point_usd * 100
    results = []
    for arm in study.arms:
        risk = risks[arm.arm_id]
        if arm.arm_id == "control" or arm.arm_id not in clinical or risk is None:
            continue
        difference, clinical_error = clinical[arm.arm_id]
        safety_difference = risk[0] - control_risk[0]
        safety_variance = risk[1] + control_risk[1]
        results.append(
            TrialDevReadoutValue(
                study_id=study.study_id,
                arm_id=arm.arm_id,
                clinical_difference=difference,
                clinical_standard_error=clinical_error,
                safety_difference=safety_difference,
                safety_standard_error=math.sqrt(safety_variance),
                estimated_annual_profit_usd=trialdev_annual_net_profit_usd(
                    preferences, outcome_mean_difference=difference, safety_risk_difference=safety_difference
                ),
                standard_error_usd=math.sqrt(
                    clinical_price**2 * clinical_error**2 + safety_price**2 * safety_variance
                ),
            )
        )
    return tuple(results)


__all__ = ["TrialDevReadoutValue", "observed_trialdev_readout_values"]
