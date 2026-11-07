"""Independent arithmetic for the authored programme's sponsor accounting."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_verifier.io import sha256_file
from trialagentbench_verifier.response import TrialDevFrozenProgrammeValue


def verify_trialdev_terminal_value(
    reconstructed: TrialDevFrozenProgrammeValue,
    reported: TrialDevFrozenProgrammeValue | None,
) -> None:
    """Check terminal components at the existing absolute dollar precision.

    Parameters
    ----------
    reconstructed, reported
        Independently reconstructed and reported sponsor accounting. Missing
        reports cannot agree with a completed terminal calculation.

    Raises
    ------
    ValueError
        If decisions differ or any monetary component differs by more than
        1e-6 dollars. Relative tolerance is zero, including for large programmes.
    """
    fields = (
        "clinical_value_usd",
        "safety_cost_usd",
        "study_cost_usd",
        "time_cost_usd",
        "net_value_usd",
    )
    if (
        reported is None
        or reconstructed.decision != reported.decision
        or any(
            not math.isclose(getattr(reconstructed, field), getattr(reported, field), rel_tol=0, abs_tol=1e-6)
            for field in fields
        )
    ):
        raise ValueError("Terminal value differs from independent cost and payoff reconstruction")


class TrialDevAccountingCase(BaseModel):
    """Executed cost calculation with enough inputs for independent reconstruction."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    phase: Literal["early_safety", "exploration", "confirmation"]
    cost_multiplier: float = Field(gt=0)
    programme_day: float = Field(ge=0)
    enrollment_days: tuple[float, ...]
    sites: int = Field(ge=2)
    screened: int = Field(ge=0)
    follow_up_days: int = Field(gt=0)
    assessment_days: tuple[int, ...]
    surveillance_days: tuple[int, ...]
    setup: float = Field(ge=0)
    per_site: float = Field(ge=0)
    screening: float = Field(ge=0)
    enrollment: float = Field(ge=0)
    followup_day: float = Field(ge=0)
    assessment: float = Field(ge=0)
    accrued_usd: float = Field(ge=0)
    remaining_liability_usd: float = Field(ge=0)

    @model_validator(mode="after")
    def verify_arithmetic(self) -> Self:
        """Reconstruct each patient's bill without importing the simulator or harness."""
        if self.screened < len(self.enrollment_days) or any(
            not math.isfinite(day) or day < 0 or day > self.programme_day for day in self.enrollment_days
        ):
            raise ValueError("Resource account has invalid screening or enrollment dates")
        schedules = (self.assessment_days, self.surveillance_days)
        if any(
            not schedule
            or tuple(sorted(set(schedule))) != schedule
            or schedule[0] < 0
            or schedule[-1] > self.follow_up_days
            for schedule in schedules
        ):
            raise ValueError("Resource account has an invalid assessment schedule")
        visits = set(self.assessment_days) | set(self.surveillance_days)
        fixed = self.setup + self.sites * self.per_site + self.screened * self.screening
        accrued = fixed
        remaining = 0.0
        for enrolled in self.enrollment_days:
            elapsed = min(self.programme_day - enrolled, self.follow_up_days)
            observed_visits = sum(day <= elapsed for day in visits)
            accrued += self.enrollment + elapsed * self.followup_day + observed_visits * self.assessment
            remaining += (self.follow_up_days - elapsed) * self.followup_day
            remaining += (len(visits) - observed_visits) * self.assessment
        if not math.isclose(accrued, self.accrued_usd, rel_tol=0, abs_tol=1e-8) or not math.isclose(
            remaining, self.remaining_liability_usd, rel_tol=0, abs_tol=1e-8
        ):
            raise ValueError("Recorded costs differ from independent participant-level accounting")
        return self


class TrialDevResourceAccountingV1(BaseModel):
    """Finite checks of cost calculations; neither clinical calibration nor release admission."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_id: Literal["trialdev.authored-resource-accounting/v1"] = "trialdev.authored-resource-accounting/v1"
    basis: Literal["declared_sponsor_scenarios_nominal_usd"] = "declared_sponsor_scenarios_nominal_usd"
    cases: tuple[TrialDevAccountingCase, ...]

    @model_validator(mode="after")
    def verify_coverage(self) -> Self:
        """Require all phases, cost sensitivities and partial/full follow-up cuts."""
        expected = {
            (phase, multiplier, day)
            for phase in ("early_safety", "exploration", "confirmation")
            for multiplier in (0.6, 1.0, 1.8)
            for day in (28.0, 70.0, 112.0)
        }
        identities = {(case.phase, case.cost_multiplier, case.programme_day) for case in self.cases}
        if identities != expected or len(identities) != len(self.cases):
            raise ValueError("Resource accounting must cover the declared phase, sensitivity and cutoff cases")
        monetary_fields = ("setup", "per_site", "screening", "enrollment", "followup_day", "assessment")
        for case in self.cases:
            base = next(
                row
                for row in self.cases
                if row.phase == case.phase and row.cost_multiplier == 1 and row.programme_day == 28
            )
            if any(
                not math.isclose(
                    getattr(case, field), getattr(base, field) * case.cost_multiplier, rel_tol=0, abs_tol=1e-8
                )
                for field in monetary_fields
            ):
                raise ValueError("Resource accounting prices differ from their declared cost-sensitivity scale")
            if (
                case.enrollment_days != (0.0, 14.0, 28.0)
                or case.sites != 2
                or case.screened != 5
                or case.follow_up_days != 84
                or case.assessment_days != (28, 84)
                or case.surveillance_days != (14, 28, 84)
            ):
                raise ValueError("Resource accounting case differs from the declared accrued-cohort probe")
        return self


# The public fixed programme horizon, restated independently of the runtime.
_TRIALDEV_HORIZON_DAYS = 2190


class _Projection(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, allow_inf_nan=False)


class _Review(_Projection):
    completed_treated: int = Field(ge=0)
    benefit_z_statistic: float | None


class _Terminal(_Projection):
    decision: Literal["stop", "advance"]
    programme_day: float = Field(ge=0)
    asset_id: str | None = None
    dose: str | None = None
    regulatory_review: _Review | None = None


class _Patient(_Projection):
    participant_id: int
    enrollment_day: float = Field(ge=0)


class _Enrollment(_Projection):
    study_id: str
    screened: int = Field(ge=0)
    patients: tuple[_Patient, ...]


class _Commitment(_Projection):
    study_id: str
    phase: Literal["early_safety", "exploration", "confirmation"]
    sites: int = Field(ge=2)
    follow_up_days: int = Field(gt=0)
    assessment_days: tuple[int, ...]
    surveillance_days: tuple[int, ...]


class _Ledger(_Projection):
    budget_usd: float = Field(ge=0)
    resource_day: float = Field(ge=0)
    terminal: _Terminal | None
    pool_manifest_sha256: str
    enrollments: tuple[_Enrollment, ...]
    recruitment_commitments: tuple[_Commitment, ...]


class _Preferences(_Projection):
    outcome_unit: str
    benefit_unit: float = Field(gt=0)
    annual_profit_per_benefit_unit_usd: float = Field(ge=0)
    annual_cost_per_safety_percentage_point_usd: float = Field(ge=0)
    discount_rate_per_year: float = Field(ge=0, lt=1)
    exclusivity_end_day: int = Field(gt=0)
    regulatory_review_days: int = Field(ge=0)
    programme_cost_per_day_usd: float = Field(ge=0)


class _Configuration(_Projection):
    pool_manifest_sha256: str
    budget_usd: float = Field(gt=0)
    preferences: _Preferences


class _Contrast(_Projection):
    asset_id: str
    dose: str
    outcome_unit: str
    outcome_mean_difference: float
    safety_risk_difference: float = Field(ge=-1, le=1)


class _Pool(_Projection):
    targets: tuple[_Contrast, ...]


def reconstruct_trialdev_terminal_value(
    *, ledger_path: Path, configuration_path: Path, pool_path: Path
) -> TrialDevFrozenProgrammeValue:
    """Reconstruct development value after trajectory authentication, without receipt totals.

    Parameters
    ----------
    ledger_path
        Authenticated trajectory with committed studies and accrued enrollment.
        An unfinished trajectory is valued as a stop on its resource day, which
        every accepted action reconciles to the programme clock.
    configuration_path, pool_path
        Release-owned sponsor preferences and fixed-population contrasts.

    Returns
    -------
    TrialDevFrozenProgrammeValue
        Independent discounted market payoff, nominal study obligations and overhead.

    Raises
    ------
    ValueError
        If identities, resources, enrollment or the explicit terminal choice disagree.

    Notes
    -----
    This checks arithmetic after trajectory authentication. It does not establish
    that supplied recruitment records were produced by lawful participant actions.
    """
    ledger = _Ledger.model_validate_json(ledger_path.read_bytes())
    configuration = _Configuration.model_validate_json(configuration_path.read_bytes())
    if (
        ledger.pool_manifest_sha256 != configuration.pool_manifest_sha256
        or sha256_file(pool_path) != configuration.pool_manifest_sha256
    ):
        raise ValueError("Terminal accounting identifies a different frozen pool")
    terminal, preferences, day = ledger.terminal, configuration.preferences, ledger.resource_day
    if ledger.budget_usd != configuration.budget_usd:
        raise ValueError("Terminal accounting changed released resource limits")
    if (terminal is not None and terminal.programme_day != day) or day > _TRIALDEV_HORIZON_DAYS:
        raise ValueError("Terminal accounting has an inconsistent programme clock")
    commitments = {row.study_id: row for row in ledger.recruitment_commitments}
    enrollments = {row.study_id: row for row in ledger.enrollments}
    if (
        len(commitments) != len(ledger.recruitment_commitments)
        or len(enrollments) != len(ledger.enrollments)
        or commitments.keys() != enrollments.keys()
    ):
        raise ValueError("Terminal accounting changed the committed study census")
    identities = [patient.participant_id for row in ledger.enrollments for patient in row.patients]
    if len(set(identities)) != len(identities) or len(identities) > 16000:
        raise ValueError("Terminal accounting repeats patients or exceeds programme capacity")
    # Independent reference prices for the released v1 sponsor scenarios.
    prices = {
        "early_safety": (1_000_000, 50_000, 1_000, 12_000, 60, 1_500),
        "exploration": (2_000_000, 75_000, 1_500, 10_000, 40, 600),
        "confirmation": (4_000_000, 100_000, 2_000, 12_000, 40, 800),
    }
    costs = []
    for study_id, design in commitments.items():
        enrolled = enrollments[study_id]
        if enrolled.screened < len(enrolled.patients) or any(
            patient.enrollment_day > day for patient in enrolled.patients
        ):
            raise ValueError("Terminal accounting includes unenrolled or unscreened patients")
        for grid in (design.assessment_days, design.surveillance_days):
            if not grid or tuple(sorted(set(grid))) != grid or grid[0] < 0 or grid[-1] != design.follow_up_days:
                raise ValueError("Terminal accounting has an invalid observation schedule")
        setup, site, screen, enroll, follow, visit = prices[design.phase]
        total = setup + design.sites * site + enrolled.screened * screen
        total += len(enrolled.patients) * (
            enroll
            + design.follow_up_days * follow
            + len(set(design.assessment_days) | set(design.surveillance_days)) * visit
        )
        costs.append(float(total))
    study_cost = math.fsum(costs)
    if study_cost > ledger.budget_usd:
        raise ValueError("Terminal study obligations exceed the released budget")
    clinical = safety = 0.0
    approved = False
    if terminal is not None and terminal.decision == "advance":
        pool = _Pool.model_validate_json(pool_path.read_bytes())
        matches = [row for row in pool.targets if (row.asset_id, row.dose) == (terminal.asset_id, terminal.dose)]
        if terminal.asset_id is None or terminal.dose is None or len(matches) != 1:
            raise ValueError("Terminal advancement has no unique released contrast")
        if terminal.regulatory_review is None:
            raise ValueError("Terminal advancement has no regulatory review")
        target = matches[0]
        if target.outcome_unit != preferences.outcome_unit:
            raise ValueError("Terminal contrast and sponsor units disagree")
        review = terminal.regulatory_review
        # Released rule: two-sided 5% benefit test and 300 completed treated patients.
        approved = (
            review.benefit_z_statistic is not None
            and review.benefit_z_statistic >= 1.959963984540054
            and review.completed_treated >= 300
        )
        if approved:
            launch = (day + preferences.regulatory_review_days) / 365.25
            end = preferences.exclusivity_end_day / 365.25
            rate = math.log1p(preferences.discount_rate_per_year)
            years = 0.0
            if launch < end:
                years = end - launch if rate == 0 else (math.exp(-rate * launch) - math.exp(-rate * end)) / rate
            benefit = -target.outcome_mean_difference / preferences.benefit_unit
            clinical = benefit * preferences.annual_profit_per_benefit_unit_usd * years
            safety = (
                target.safety_risk_difference * 100 * preferences.annual_cost_per_safety_percentage_point_usd * years
            )
    elif terminal is not None and (
        terminal.asset_id is not None or terminal.dose is not None or terminal.regulatory_review is not None
    ):
        raise ValueError("Terminal stop cannot select an asset, regimen or regulatory review")
    time_cost = day * preferences.programme_cost_per_day_usd
    return TrialDevFrozenProgrammeValue(
        decision="stop" if terminal is None else terminal.decision,
        approved=approved,
        clinical_value_usd=clinical,
        safety_cost_usd=safety,
        study_cost_usd=study_cost,
        time_cost_usd=time_cost,
        net_value_usd=clinical - safety - study_cost - time_cost,
    )
