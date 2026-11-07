"""Quantity-based study costs and unavoidable accrued-patient liabilities."""

import math
from bisect import bisect_right
from typing import Annotated, Literal, Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench.contracts.trialdev.pools import TrialDevArmId, TrialDevAssignedArm
from trialagentbench.contracts.trialdev.release import TrialDevDose
from trialagentbench.io import canonical_payload_sha256
from trialagentbench.trialdev.errors import TrialDevActionRejectedError
from trialagentbench.trialdev.recruitment import (
    TrialDevCommittedPatient,
    TrialDevEnrollmentPlan,
    TrialDevStudyEnrollment,
    TrialDevStudyReservation,
    allocate_trialdev_patients,
)


class TrialDevStudyTariff(BaseModel):
    """Public fictional sponsor prices per executed study resource unit, in USD."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    setup: float = Field(ge=0)
    per_site: float = Field(ge=0)
    screening: float = Field(ge=0)
    enrollment: float = Field(ge=0)
    followup_day: float = Field(ge=0)
    assessment: float = Field(ge=0)
    startup_days: int = Field(ge=0)
    closeout_days: int = Field(ge=0)


TrialDevPhase = Literal["early_safety", "exploration", "confirmation"]
MINIMUM_PHASE_PARTICIPANTS_PER_ARM = 2
_PRECEDING_PHASE: dict[TrialDevPhase, TrialDevPhase] = {"exploration": "early_safety", "confirmation": "exploration"}


def trialdev_phase_tariff(phase: TrialDevPhase) -> TrialDevStudyTariff:
    """Return the declared sponsor tariff for a development phase.

    Parameters
    ----------
    phase
        Early safety, efficacy exploration or confirmation.

    Returns
    -------
    TrialDevStudyTariff
        Public unit costs and startup/closeout days selected for this benchmark.

    Raises
    ------
    ValueError
        If the phase is unsupported.

    Examples
    --------
    Prices are directional estimates, not quotes. With typical arm sizes they give
    about $2-4M for early safety, $8-20M for exploration and $25-100M for one
    confirmation study (Sertkaya et al. 2016; Moore et al. 2018; Martin et al. 2017).
    The per-visit assessment price keeps a study that assesses the full protocol
    visit grid at the calibrated per-patient cost; screening is priced per screened
    candidate, so screen failures add to the cost per randomised patient.
    """
    values = {
        "early_safety": (1_000_000, 50_000, 1_000, 12_000, 60, 1_500, 60, 14),
        "exploration": (2_000_000, 75_000, 1_500, 10_000, 40, 600, 90, 30),
        "confirmation": (4_000_000, 100_000, 2_000, 12_000, 40, 800, 120, 45),
    }
    if phase not in values:
        raise ValueError("unsupported development phase")
    setup, site, screen, enrollment, followup, assessment, startup, closeout = values[phase]
    return TrialDevStudyTariff(
        setup=setup,
        per_site=site,
        screening=screen,
        enrollment=enrollment,
        followup_day=followup,
        assessment=assessment,
        startup_days=startup,
        closeout_days=closeout,
    )


# Every programme has the same six-year horizon for its terminal decision. Time is
# already costed through discounting and exclusivity erosion in the value model, so
# the horizon is a fixed rule, not a resource condition.
TRIALDEV_HORIZON_DAYS = 2190
TRIALDEV_APPROVAL_TWO_SIDED_ALPHA = 0.05
TRIALDEV_APPROVAL_CRITICAL_Z = 1.959963984540054
TRIALDEV_APPROVAL_MINIMUM_COMPLETED_TREATED = 300

# Accrued and remaining costs are separately rounded floats, so an unchanged obligation
# can sum a few units in the last place differently between receipts (in one live run
# the interim parts of $10,166,000 summed to 10166000.000000002).
_COST_REPRESENTATION_RELATIVE_TOLERANCE = 1e-12


def _cost_regresses(current: float, prior: float) -> bool:
    """Return whether a cumulative cost falls by more than its float representation error."""
    return current < prior * (1 - _COST_REPRESENTATION_RELATIVE_TOLERANCE)


class TrialDevRegulatoryReview(BaseModel):
    """Evaluator-only primary analysis of completed confirmation evidence at submission.

    The regulator analyses every patient randomized in confirmation studies of the
    advanced asset/regimen whose committed follow-up completed by the submission
    day, using their complete recorded outcomes. Approval requires a two-sided 5%
    benefit test and at least 300 completed treated patients.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    completed_treated: int = Field(ge=0, strict=True)
    completed_control: int = Field(ge=0, strict=True)
    benefit_z_statistic: float | None
    approved: bool

    @model_validator(mode="after")
    def validate_rule(self) -> Self:
        """Bind approval to the declared significance and exposure rule."""
        expected = (
            self.benefit_z_statistic is not None
            and self.benefit_z_statistic >= TRIALDEV_APPROVAL_CRITICAL_Z
            and self.completed_treated >= TRIALDEV_APPROVAL_MINIMUM_COMPLETED_TREATED
        )
        if self.approved != expected:
            raise ValueError("regulatory approval must follow the declared significance and exposure rule")
        return self


class TrialDevRecruitmentPause(BaseModel):
    """An optional cohort enrollment halt and subsequent continuation date."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    after_enrolled: int = Field(ge=1, le=7999, strict=True)
    resume_day: float | None = Field(default=None, ge=0)


class TrialDevAccruedRecruitment(BaseModel):
    """Actual screening and enrollment at a programme calendar cutoff."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    screened: int = Field(ge=0, strict=True)
    patients: tuple[TrialDevCommittedPatient, ...]


def trialdev_accrued_recruitment(
    reservation: TrialDevStudyReservation,
    tariff: TrialDevStudyTariff,
    *,
    start_day: float,
    programme_day: float,
    pause: TrialDevRecruitmentPause | None = None,
) -> TrialDevAccruedRecruitment:
    """Accrue reserved patients through an outcome-blind screening queue.

    Parameters
    ----------
    reservation
        Frozen patient assignments and screening ordinals from study commitment.
    tariff
        Public study startup duration and resource prices.
    start_day
        Programme day on which study startup begins.
    programme_day
        Calendar cutoff for actual screening and enrollment.
    pause
        Optional prespecified cohort halt. Without a continuation date, later
        screening remains paused. Continuation changes only subsequent arrivals;
        previously enrolled patients retain their original follow-up clock.

    Returns
    -------
    TrialDevAccruedRecruitment
        Screening count and patients enrolled by the cutoff. Rejected candidates
        still consume screening effort. Screening stops
        when the final reserved patient enrolls.

    Raises
    ------
    ValueError
        If the calendar or site count is invalid, or recruitment is already closed.

    Examples
    --------
    Site activation changes the reserved candidate set and its frozen screening
    clock. A readout never redraws arrivals or assigns new dates to past enrollment.
    """
    if not math.isfinite(start_day) or not math.isfinite(programme_day) or start_day < 0 or programme_day < start_day:
        raise ValueError("recruitment calendar must be finite and follow study startup")
    if reservation.closed:
        raise ValueError("closed recruitment must use its retained actual enrollment")
    opening_day = start_day + tariff.startup_days
    ordered = sorted(reservation.patients, key=lambda item: item.screening_ordinal)
    final_screen = ordered[-1].screening_ordinal
    pause_screen: int | None = None
    if pause is not None:
        if pause.after_enrolled >= len(ordered):
            raise ValueError("an interim recruitment pause must precede full enrollment")
        pause_screen = ordered[pause.after_enrolled - 1].screening_ordinal
        pause_day = opening_day + reservation.screening_days[pause_screen - 1]
        if pause.resume_day is not None and pause.resume_day < pause_day:
            raise ValueError("recruitment cannot resume before the committed cohort halt")

    def screen_day(ordinal: int) -> float:
        if pause_screen is not None and ordinal > pause_screen:
            assert pause is not None
            if pause.resume_day is None:
                return math.inf
            return (
                pause.resume_day
                + reservation.screening_days[ordinal - 1]
                - reservation.screening_days[pause_screen - 1]
            )
        return opening_day + reservation.screening_days[ordinal - 1]

    screened = bisect_right(range(1, final_screen + 1), programme_day, key=screen_day)
    patients = tuple(
        TrialDevCommittedPatient(
            participant_id=patient.participant_id,
            participant_label=f"{reservation.plan.study_id}:screen-{patient.screening_ordinal}",
            arm_id=patient.arm_id,
            enrollment_day=screen_day(patient.screening_ordinal),
        )
        for patient in ordered
        if patient.screening_ordinal <= screened
    )
    return TrialDevAccruedRecruitment(screened=screened, patients=patients)


class TrialDevStudyCost(BaseModel):
    """Accrued expenditure and remaining obligations for enrolled participants."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    accrued_usd: float = Field(ge=0)
    remaining_liability_usd: float = Field(ge=0)

    @property
    def committed_usd(self) -> float:
        """Return the amount owed even if all future enrollment stops now."""
        return self.accrued_usd + self.remaining_liability_usd


def trialdev_study_cost(
    tariff: TrialDevStudyTariff,
    *,
    sites: int,
    screened: int,
    enrollment_days: tuple[float, ...],
    programme_day: float,
    follow_up_days: int,
    assessment_days: tuple[int, ...],
    surveillance_days: tuple[int, ...],
) -> TrialDevStudyCost:
    """Account for actual enrollment and its retained follow-up obligations.

    Parameters
    ----------
    tariff
        Public prices fixed before the study decision.
    sites
        Activated sites, whose setup cost has been incurred.
    screened
        Candidates actually screened, including those not enrolled.
    enrollment_days
        Programme enrollment dates of patients already accrued, not reserved slots.
    programme_day
        Current programme day, independently of record reporting delays.
    follow_up_days
        Committed follow-up for each enrolled patient.
    assessment_days
        Committed primary assessment visits, in patient-relative days.
    surveillance_days
        Committed safety visits. A coincident primary/safety visit is charged once.

    Returns
    -------
    TrialDevStudyCost
        Accrued expenditure and unavoidable remaining patient obligations. This
        pure calculation does not debit a ledger; the runtime applies deltas once.

    Raises
    ------
    ValueError
        If resource quantities, enrollment dates or schedules are invalid.

    Examples
    --------
    Stopping recruitment preserves ``cost.committed_usd`` for already accrued patients.
    """
    if (
        isinstance(sites, bool)
        or not isinstance(sites, int)
        or sites < 2
        or isinstance(screened, bool)
        or not isinstance(screened, int)
        or screened < len(enrollment_days)
        or isinstance(follow_up_days, bool)
        or not isinstance(follow_up_days, int)
        or follow_up_days <= 0
        or not math.isfinite(programme_day)
        or programme_day < 0
    ):
        raise ValueError("study resource quantities are invalid")
    dates = np.asarray(enrollment_days, dtype=float)
    if not np.isfinite(dates).all() or np.any(dates < 0) or np.any(dates > programme_day):
        raise ValueError("costs require dates of already enrolled patients")
    for schedule in (assessment_days, surveillance_days):
        if (
            not schedule
            or tuple(sorted(set(schedule))) != schedule
            or any(isinstance(day, bool) or not isinstance(day, int) for day in schedule)
            or schedule[0] < 0
            or schedule[-1] > follow_up_days
        ):
            raise ValueError("resource schedules must be unique increasing dates within follow-up")
    visits = np.asarray(sorted(set(assessment_days) | set(surveillance_days)), dtype=float)
    elapsed = np.minimum(programme_day - dates, follow_up_days)
    completed_visits = int(np.sum(elapsed[:, None] >= visits[None, :]))
    total_visits = len(dates) * len(visits)
    fixed = tariff.setup + sites * tariff.per_site + screened * tariff.screening + len(dates) * tariff.enrollment
    # Exactly rounded sums keep a readout's cost identity independent of patient order.
    accrued = fixed + math.fsum(elapsed.tolist()) * tariff.followup_day + completed_visits * tariff.assessment
    remaining = math.fsum((follow_up_days - elapsed).tolist()) * tariff.followup_day
    remaining += (total_visits - completed_visits) * tariff.assessment
    return TrialDevStudyCost(accrued_usd=accrued, remaining_liability_usd=remaining)


class TrialDevCostReceipt(BaseModel):
    """Unique readout or resource-only update of cumulative study expenditure."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    study_id: str = Field(min_length=1)
    readout_id: str = Field(min_length=1)
    kind: Literal["readout", "resource"] = "readout"
    cost: TrialDevStudyCost
    programme_day: float = Field(default=0, ge=0, allow_inf_nan=False)
    supported_arms: tuple[TrialDevArmId, ...] = ()

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        """Only published readouts can establish unique arm evidence."""
        if len(set(self.supported_arms)) != len(self.supported_arms):
            raise ValueError("readout evidence arms must be unique")
        if self.kind != "readout" and self.supported_arms:
            raise ValueError("resource receipts cannot establish readout evidence")
        return self


class TrialDevLook(BaseModel):
    """Validated observation timing for a planned readout."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        json_schema_extra={
            "description": "One readout: on deadline_day (a programme day), or earlier once completed_participants have their primary outcome recorded at the end of follow-up."
        },
    )
    deadline_day: float = Field(
        ge=0,
        description=(
            "Programme day of the readout, counted from programme start (day 0), not a duration after the study "
            "starts; recruitment begins after the study's start-up and follow-up runs from each enrolment."
        ),
    )
    completed_participants: int | None = Field(
        default=None,
        ge=1,
        le=8000,
        strict=True,
        description="Optional earlier trigger: read out once this many participants have completed follow-up.",
    )


class TrialDevRecruitmentCommitment(BaseModel):
    """Prospective recruitment clock bound before any study evidence is released."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    study_id: str = Field(min_length=1)
    phase: TrialDevPhase
    tariff: TrialDevStudyTariff
    sites: int = Field(ge=2, strict=True)
    start_day: float = Field(ge=0)
    pause_after_enrolled: int | None = Field(default=None, ge=1, le=7999, strict=True)
    looks: tuple[TrialDevLook, ...] = Field(min_length=1, max_length=2)
    arms: tuple[TrialDevAssignedArm, ...] = Field(min_length=2, max_length=7)
    follow_up_days: int = Field(gt=0, strict=True)
    assessment_days: tuple[Annotated[int, Field(ge=0, strict=True)], ...] = Field(min_length=1)
    surveillance_days: tuple[Annotated[int, Field(ge=0, strict=True)], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_design(self) -> Self:
        """Require unique treatments, one shared control and ordered observation grids."""
        deadlines = tuple(look.deadline_day for look in self.looks)
        if tuple(sorted(set(deadlines))) != deadlines or deadlines[0] <= self.start_day:
            raise ValueError("committed look dates must increase after study start")
        ids = [arm.arm_id for arm in self.arms]
        controls = [arm for arm in self.arms if arm.partition.dose is None]
        treatments = [(arm.partition.asset_id, arm.partition.dose) for arm in self.arms]
        if len(set(ids)) != len(ids) or len(set(treatments)) != len(treatments):
            raise ValueError("committed study arm IDs and asset/regimen pairs must be unique")
        if len(controls) != 1 or controls[0].arm_id != "control":
            raise ValueError("committed study requires exactly one shared control")
        for grid in (self.assessment_days, self.surveillance_days):
            if tuple(sorted(set(grid))) != grid or grid[-1] != self.follow_up_days:
                raise ValueError("committed observation grids must increase and end at follow-up")
        return self


class TrialDevRecruitmentContinuation(BaseModel):
    """Immutable calendar decision to resume a previously paused study."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    study_id: str = Field(min_length=1)
    resume_day: float = Field(ge=0)


class TrialDevTerminalDecision(BaseModel):
    """Explicit terminal stop or asset/regimen advancement, separate from noncompletion."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    decision: Literal["stop", "advance"]
    programme_day: float = Field(ge=0)
    asset_id: str | None = Field(default=None, min_length=1)
    dose: TrialDevDose | None = None
    regulatory_review: TrialDevRegulatoryReview | None = None

    @model_validator(mode="after")
    def validate_choice(self) -> Self:
        """Require an asset and regimen exactly when advancement is selected."""
        if self.decision == "advance":
            if self.asset_id is None or self.dose is None:
                raise ValueError("advancement requires an asset and regimen")
        elif self.asset_id is not None or self.dose is not None or self.regulatory_review is not None:
            raise ValueError("stopping cannot select an asset, regimen or regulatory review")
        return self


class TrialDevBudgetLedger(BaseModel):
    """Immutable programme accounting with replayable readout receipts."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    budget_usd: float = Field(ge=0)
    resource_day: float = Field(default=0, ge=0)
    terminal: TrialDevTerminalDecision | None = None
    settlement_usd: float = Field(default=0, ge=0)
    receipts: tuple[TrialDevCostReceipt, ...] = ()
    pool_manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    enrollments: tuple[TrialDevStudyEnrollment, ...] = ()
    recruitment_seed: int | None = Field(default=None, ge=0, strict=True)
    reservations: tuple[TrialDevStudyReservation, ...] = ()
    recruitment_commitments: tuple[TrialDevRecruitmentCommitment, ...] = ()
    recruitment_continuations: tuple[TrialDevRecruitmentContinuation, ...] = ()

    @model_validator(mode="after")
    def validate_history(self) -> Self:
        """Reject invalid persisted histories as well as invalid new charges."""
        studies = [study.study_id for study in self.enrollments]
        patients = [patient for study in self.enrollments for patient in study.patients]
        identities = [patient.participant_id for patient in patients]
        labels = [patient.participant_label for patient in patients]
        if (
            len(set(studies)) != len(studies)
            or len(set(identities)) != len(identities)
            or len(set(labels)) != len(labels)
        ):
            raise ValueError("programme enrollment cannot reuse studies, patients or public labels")
        if len(patients) > 16000 or (self.enrollments and self.pool_manifest_sha256 is None):
            raise ValueError("programme enrollment exceeds capacity or lacks its frozen population identity")
        reservation_ids = [item.plan.study_id for item in self.reservations]
        if len(reservation_ids) != len(set(reservation_ids)):
            raise ValueError("study reservation identities must be unique")
        reserved_studies = {item.plan.study_id: item for item in self.reservations}
        commitments = {item.study_id: item for item in self.recruitment_commitments}
        if len(commitments) != len(self.recruitment_commitments) or not set(commitments).issubset(reserved_studies):
            raise ValueError("recruitment commitments require unique reserved study identities")
        for identity, commitment in commitments.items():
            if commitment.sites != len(reserved_studies[identity].plan.site_ids):
                raise ValueError("committed site count must match activated recruitment sites")
            if {arm.arm_id for arm in commitment.arms} != {q.arm_id for q in reserved_studies[identity].plan.arms}:
                raise ValueError("committed treatment arms must match reserved allocation")
            pause_after = commitment.pause_after_enrolled
            if pause_after is not None and pause_after >= len(reserved_studies[identity].patients):
                raise ValueError("committed interim pause must precede full enrollment")
        continuations = {item.study_id: item for item in self.recruitment_continuations}
        if len(continuations) != len(self.recruitment_continuations):
            raise ValueError("study continuation identities must be unique")
        for identity, continuation in continuations.items():
            pause_commitment = commitments.get(identity)
            if pause_commitment is None or pause_commitment.pause_after_enrolled is None:
                raise ValueError("continuation requires a committed cohort pause")
            ordered = sorted(reserved_studies[identity].patients, key=lambda item: item.screening_ordinal)
            halt_screen = ordered[pause_commitment.pause_after_enrolled - 1].screening_ordinal
            halt_day = (
                pause_commitment.start_day
                + pause_commitment.tariff.startup_days
                + reserved_studies[identity].screening_days[halt_screen - 1]
            )
            if continuation.resume_day < halt_day:
                raise ValueError("continuation cannot precede the cohort halt")
        for study in self.enrollments:
            reservation = reserved_studies.get(study.study_id)
            if reservation is not None:
                assigned = {patient.participant_id: patient.arm_id for patient in reservation.patients}
                if any(assigned.get(patient.participant_id) != patient.arm_id for patient in study.patients):
                    raise ValueError("actual enrollment must preserve its reserved patients and arms")
        owners = {patient.participant_id: study.study_id for study in self.enrollments for patient in study.patients}
        for reservation in self.reservations:
            for patient in reservation.active_patients:
                if owners.get(patient.participant_id, reservation.plan.study_id) != reservation.plan.study_id:
                    raise ValueError("a reserved patient belongs to another study")
                owners[patient.participant_id] = reservation.plan.study_id
        if len(owners) > 16000 or (
            self.reservations and (self.recruitment_seed is None or self.pool_manifest_sha256 is None)
        ):
            raise ValueError("reservations exceed programme capacity or lack their fixed seed/population")
        seen: set[str] = set()
        look_counts: dict[str, int] = {}
        latest: dict[str, TrialDevStudyCost] = {}
        for receipt in self.receipts:
            design = commitments.get(receipt.study_id)
            if receipt.supported_arms and (
                design is None or not set(receipt.supported_arms).issubset(arm.arm_id for arm in design.arms)
            ):
                raise ValueError("readout evidence must identify committed study arms")
            if design is not None and receipt.kind == "readout":
                count = look_counts.get(receipt.study_id, 0)
                if count >= len(design.looks):
                    raise ValueError("study receipts must follow the committed look schedule")
                look = design.looks[count]
                previous_day = max(
                    (
                        item.programme_day
                        for item in self.receipts[: len(seen)]
                        if item.study_id == receipt.study_id and item.kind == "readout"
                    ),
                    default=design.start_day,
                )
                if (
                    receipt.programme_day <= previous_day
                    # Delayed requests retain the already accrued programme
                    # clock; a receipt cannot invent an unaccounted late cutoff.
                    or (receipt.programme_day > look.deadline_day and receipt.programme_day > self.resource_day)
                    or (look.completed_participants is None and receipt.programme_day < look.deadline_day)
                ):
                    raise ValueError("study receipts must follow the committed look schedule")
                look_counts[receipt.study_id] = count + 1
            if receipt.readout_id in seen:
                raise ValueError("budget history contains duplicate readout identities")
            seen.add(receipt.readout_id)
            prior = latest.get(receipt.study_id)
            if prior is not None and (
                _cost_regresses(receipt.cost.accrued_usd, prior.accrued_usd)
                or _cost_regresses(receipt.cost.committed_usd, prior.committed_usd)
            ):
                raise ValueError("cumulative study costs cannot erase expenditure or enrolled-patient obligations")
            latest[receipt.study_id] = receipt.cost
        if sum(cost.committed_usd for cost in latest.values()) > self.budget_usd:
            raise ValueError("study expenditure and obligations exceed the programme budget")
        if self.current_day > TRIALDEV_HORIZON_DAYS:
            raise ValueError("programme activity exceeds its time horizon")
        if self.terminal is None:
            if self.settlement_usd != 0:
                raise ValueError("liability settlement requires an explicit terminal decision")
        elif (
            self.terminal.programme_day != self.resource_day
            or self.current_day != self.resource_day
            or any(not item.closed for item in self.reservations)
            or self.settlement_usd != sum(cost.remaining_liability_usd for cost in latest.values())
        ):
            raise ValueError("terminal state must freeze current resources and settle every remaining liability")
        return self

    @property
    def current_day(self) -> float:
        """Return the latest programme time established by resources or accepted actions."""
        return max(
            [
                self.resource_day,
                *(item.programme_day for item in self.receipts),
                *(item.resume_day for item in self.recruitment_continuations),
                *(item.start_day for item in self.recruitment_commitments),
            ]
        )

    @property
    def latest_study_costs(self) -> tuple[TrialDevStudyCost, ...]:
        """Return the latest cumulative cost for each distinct study."""
        latest = {receipt.study_id: receipt.cost for receipt in self.receipts}
        return tuple(latest.values())

    @property
    def spent_usd(self) -> float:
        """Return incurred expenditure without counting repeated readouts twice."""
        return self.settlement_usd + sum(cost.accrued_usd for cost in self.latest_study_costs)

    @property
    def remaining_liability_usd(self) -> float:
        """Return the unpaid obligations of already enrolled participants."""
        return (
            0.0 if self.terminal is not None else sum(cost.remaining_liability_usd for cost in self.latest_study_costs)
        )

    @property
    def available_usd(self) -> float:
        """Return budget available after both expenditure and patient obligations."""
        return self.budget_usd - self.spent_usd - self.remaining_liability_usd


def reconcile_trialdev_cost(ledger: TrialDevBudgetLedger, receipt: TrialDevCostReceipt) -> TrialDevBudgetLedger:
    """Apply one cumulative study cost exactly once within the programme budget.

    Parameters
    ----------
    ledger
        Current immutable accounting state.
    receipt
        Readout identity and cumulative cost from its committed study inputs.

    Returns
    -------
    TrialDevBudgetLedger
        Updated accounting state, or the original state for an identical retry.
        A runtime transaction must persist this returned state with its action.

    Raises
    ------
    ValueError
        If an identity is reused with different inputs, obligations are erased,
        cumulative spending regresses or the programme budget is exceeded.

    Examples
    --------
    Reapplying the same receipt returns the same ledger without another charge.
    """
    for previous in ledger.receipts:
        if previous.readout_id == receipt.readout_id:
            if previous != receipt:
                raise ValueError("readout cost identity is already bound to different inputs")
            return ledger
    if ledger.terminal is not None:
        raise ValueError("terminal programme cannot incur a new resource or readout receipt")
    costs = {item.study_id: item.cost for item in ledger.receipts}
    costs[receipt.study_id] = receipt.cost
    committed = sum(cost.committed_usd for cost in costs.values())
    if committed > ledger.budget_usd:
        raise TrialDevActionRejectedError(
            f"spending and follow-up obligations would reach ${committed:,.0f}, above the ${ledger.budget_usd:,.0f} "
            f"programme budget; this study's committed cost is ${receipt.cost.committed_usd:,.0f}"
        )
    return TrialDevBudgetLedger(
        budget_usd=ledger.budget_usd,
        resource_day=ledger.resource_day,
        terminal=ledger.terminal,
        settlement_usd=ledger.settlement_usd,
        recruitment_commitments=ledger.recruitment_commitments,
        recruitment_continuations=ledger.recruitment_continuations,
        receipts=(*ledger.receipts, receipt),
        pool_manifest_sha256=ledger.pool_manifest_sha256,
        enrollments=ledger.enrollments,
        recruitment_seed=ledger.recruitment_seed,
        reservations=ledger.reservations,
    )


def reconcile_trialdev_enrollment(
    ledger: TrialDevBudgetLedger, *, pool_manifest_sha256: str, enrollment: TrialDevStudyEnrollment
) -> TrialDevBudgetLedger:
    """Retain accrued patients without cross-study reuse or retrospective reassignment.

    Parameters
    ----------
    ledger
        Persisted programme budget and actual enrollment history.
    pool_manifest_sha256
        Immutable population identity used for every study in the run.
    enrollment
        Current study's complete accrued patient history and fixed design identity.

    Returns
    -------
    TrialDevBudgetLedger
        Updated immutable state for the same transaction as readout accounting.

    Raises
    ------
    ValueError
        If the population/design changes, prior patients are erased or reassigned,
        or patients are reused across studies.

    Examples
    --------
    Later readouts may add patients but must preserve every earlier enrollment record.
    """
    if ledger.terminal is not None:
        raise ValueError("terminal programme cannot change enrollment")
    if ledger.pool_manifest_sha256 not in (None, pool_manifest_sha256):
        raise ValueError("programme cannot change its frozen population")
    previous = next((study for study in ledger.enrollments if study.study_id == enrollment.study_id), None)
    reservation = next((item for item in ledger.reservations if item.plan.study_id == enrollment.study_id), None)
    if reservation is not None and (reservation.closed or reservation.closed_arms):
        prior_ids = {patient.participant_id for patient in previous.patients} if previous is not None else set()
        if any(
            patient.participant_id not in prior_ids
            and (reservation.closed or patient.arm_id in reservation.closed_arms)
            for patient in enrollment.patients
        ):
            raise ValueError("a closed reservation or arm cannot accrue more patients")
    if reservation is not None and reservation.retained_control_ids is not None:
        if any(
            patient.arm_id == "control" and patient.participant_id not in reservation.retained_control_ids
            for patient in enrollment.patients
        ):
            raise ValueError("enrollment cannot restore released control patients")
    if previous is not None:
        if enrollment.screened < previous.screened:
            raise ValueError("study cannot erase incurred screening")
        current = {patient.participant_id: patient for patient in enrollment.patients}
        if previous.design_sha256 != enrollment.design_sha256 or any(
            current.get(patient.participant_id) != patient for patient in previous.patients
        ):
            raise ValueError("study cannot change its design or erase/reassign accrued patients")
    studies = tuple(study for study in ledger.enrollments if study.study_id != enrollment.study_id)
    return TrialDevBudgetLedger(
        budget_usd=ledger.budget_usd,
        resource_day=ledger.resource_day,
        terminal=ledger.terminal,
        settlement_usd=ledger.settlement_usd,
        recruitment_commitments=ledger.recruitment_commitments,
        recruitment_continuations=ledger.recruitment_continuations,
        receipts=ledger.receipts,
        pool_manifest_sha256=pool_manifest_sha256,
        enrollments=tuple(sorted((*studies, enrollment), key=lambda study: study.study_id)),
        recruitment_seed=ledger.recruitment_seed,
        reservations=ledger.reservations,
    )


def reserve_trialdev_patients(
    ledger: TrialDevBudgetLedger,
    baseline: pd.DataFrame,
    plan: TrialDevEnrollmentPlan,
    *,
    pool_manifest_sha256: str,
    seed: int,
) -> TrialDevBudgetLedger:
    """Reserve a replayable allocation without treating selected patients as enrolled.

    Parameters
    ----------
    ledger
        Current actual enrollment, reservations and budget state.
    baseline
        Verified frozen patient identities and observable clinical group memberships.
    plan
        Complete integer arm caps.
    pool_manifest_sha256
        Immutable population identity for the programme.
    seed
        Private run seed, fixed by the first reservation.

    Returns
    -------
    TrialDevBudgetLedger
        State with the allocation reserved and no enrollment or cost added.
        The run transaction must persist this state before exposing a commitment.

    Raises
    ------
    ValueError
        If a study identity changes, patients are unavailable, or run identity changes.

    Examples
    --------
    Repeating a reservation preserves the original patients even after other studies reserve patients.
    """
    if ledger.pool_manifest_sha256 not in (None, pool_manifest_sha256) or ledger.recruitment_seed not in (None, seed):
        raise ValueError("reservation cannot change the programme population or seed")
    for existing in ledger.reservations:
        if existing.plan.study_id == plan.study_id:
            if existing.plan != plan:
                raise ValueError("study identity already has a different reservation")
            return ledger
    if ledger.terminal is not None:
        raise ValueError("terminal programme cannot reserve a new study")
    unavailable = {patient.participant_id for study in ledger.enrollments for patient in study.patients}
    unavailable.update(patient.participant_id for item in ledger.reservations for patient in item.active_patients)
    reservation = allocate_trialdev_patients(
        baseline,
        plan,
        seed=seed,
        unavailable_participant_ids=unavailable,
        screening_round=len(ledger.reservations),
    )
    return TrialDevBudgetLedger(
        budget_usd=ledger.budget_usd,
        resource_day=ledger.resource_day,
        terminal=ledger.terminal,
        settlement_usd=ledger.settlement_usd,
        recruitment_commitments=ledger.recruitment_commitments,
        recruitment_continuations=ledger.recruitment_continuations,
        receipts=ledger.receipts,
        enrollments=ledger.enrollments,
        pool_manifest_sha256=pool_manifest_sha256,
        recruitment_seed=seed,
        reservations=(*ledger.reservations, reservation),
    )


def close_trialdev_reservation(ledger: TrialDevBudgetLedger, study_id: str) -> TrialDevBudgetLedger:
    """Release unaccrued capacity while retaining the original reservation and enrolled patients.

    Parameters
    ----------
    ledger
        Programme reservation, actual enrollment and accounting state.
    study_id
        Study whose future recruitment is stopped.

    Returns
    -------
    TrialDevBudgetLedger
        State with the reservation closed. Actual enrollment and costs are unchanged.
        Repeating closure returns the original state.

    Raises
    ------
    ValueError
        If the study has no reservation.

    Examples
    --------
    Subsequent studies may reserve unused slots, but never previously enrolled patients.
    """
    existing = next((item for item in ledger.reservations if item.plan.study_id == study_id), None)
    if existing is None:
        raise TrialDevActionRejectedError(f"unknown study {study_id!r}")
    if existing.closed:
        return ledger
    return TrialDevBudgetLedger(
        budget_usd=ledger.budget_usd,
        resource_day=ledger.resource_day,
        terminal=ledger.terminal,
        settlement_usd=ledger.settlement_usd,
        recruitment_commitments=ledger.recruitment_commitments,
        recruitment_continuations=ledger.recruitment_continuations,
        receipts=ledger.receipts,
        enrollments=ledger.enrollments,
        pool_manifest_sha256=ledger.pool_manifest_sha256,
        recruitment_seed=ledger.recruitment_seed,
        reservations=tuple(
            item.model_copy(update={"closed": True}) if item.plan.study_id == study_id else item
            for item in ledger.reservations
        ),
    )


def trialdev_observed_treatments(ledger: TrialDevBudgetLedger, phase: TrialDevPhase) -> set[tuple[str, TrialDevDose]]:
    """Return regimens with published arm and control endpoint evidence in a phase.

    Parameters
    ----------
    ledger
        Persisted study commitments and outcome-blind evidence receipts.
    phase
        Development phase whose readouts support a subsequent decision.

    Returns
    -------
    set[tuple[str, TrialDevDose]]
        Asset/regimen pairs supported in the same readout as their control.
        No treatment effect, statistical test or agent analysis is inspected.

    Examples
    --------
    A standard-regimen readout does not qualify an untested intensified regimen.
    """
    supported: set[tuple[str, TrialDevDose]] = set()
    for study in ledger.recruitment_commitments:
        if study.phase != phase:
            continue
        for receipt in ledger.receipts:
            if receipt.study_id != study.study_id or "control" not in receipt.supported_arms:
                continue
            supported.update(
                (arm.partition.asset_id, arm.partition.dose)
                for arm in study.arms
                if arm.partition.dose is not None and arm.arm_id in receipt.supported_arms
            )
    return supported


def commit_trialdev_recruitment(
    ledger: TrialDevBudgetLedger, commitment: TrialDevRecruitmentCommitment
) -> TrialDevBudgetLedger:
    """Bind the study recruitment clock before enrollment or spending.

    Parameters
    ----------
    ledger
        Current private programme state, including the reserved allocation.
    commitment
        Prospective startup, site capacity, tariffs and optional cohort halt.

    Returns
    -------
    TrialDevBudgetLedger
        Validated state with the immutable commitment; identical retries return
        the existing state without changing recruitment or expenditure.

    Raises
    ------
    ValueError
        If a commitment changes or arrives after study activity has started.

    Examples
    --------
    Commit before releasing any accrued participant records to the analyst.
    """
    for existing in ledger.recruitment_commitments:
        if existing.study_id == commitment.study_id:
            if existing != commitment:
                raise ValueError("study recruitment is already committed to different inputs")
            return ledger
    if ledger.terminal is not None:
        raise ValueError("terminal programme cannot admit a new study")
    latest_day = max(
        [
            0.0,
            *(receipt.programme_day for receipt in ledger.receipts),
            *(item.resume_day for item in ledger.recruitment_continuations),
        ]
    )
    if commitment.start_day < latest_day:
        raise ValueError("a new study cannot be backdated before programme evidence or decisions")
    predecessor = _PRECEDING_PHASE.get(commitment.phase)
    if predecessor is not None:
        supported = trialdev_observed_treatments(ledger, predecessor)
        requested = {
            (arm.partition.asset_id, arm.partition.dose) for arm in commitment.arms if arm.partition.dose is not None
        }
        if not requested.issubset(supported):
            eligible = ", ".join(f"{asset} {dose.value}" for asset, dose in sorted(supported, key=str)) or "none yet"
            raise TrialDevActionRejectedError(
                f"each active compound and dose in {commitment.phase} needs a readout from {predecessor} with at "
                f"least {MINIMUM_PHASE_PARTICIPANTS_PER_ARM} participants observed in its arm and in control; "
                f"eligible now: {eligible}"
            )
    if (
        any(study.study_id == commitment.study_id for study in ledger.enrollments)
        or any(receipt.study_id == commitment.study_id for receipt in ledger.receipts)
        or any(item.plan.study_id == commitment.study_id and item.closed for item in ledger.reservations)
    ):
        raise ValueError("recruitment must be committed before study activity")
    return TrialDevBudgetLedger.model_validate(
        {
            **ledger.model_dump(),
            "recruitment_commitments": (*ledger.recruitment_commitments, commitment),
        }
    )


def trialdev_committed_accrual(
    ledger: TrialDevBudgetLedger, study_id: str, *, programme_day: float
) -> TrialDevAccruedRecruitment:
    """Recover actual recruitment from the persisted study clock and allocation.

    Parameters
    ----------
    ledger
        Private programme state containing the prospective recruitment commitment.
    study_id
        Reserved and committed study identity.
    programme_day
        Cutoff for available enrolled patients and incurred screening effort.

    Returns
    -------
    TrialDevAccruedRecruitment
        Deterministic current accrual, or retained enrollment after closure.

    Raises
    ------
    ValueError
        If the commitment is absent or the cutoff precedes retained closed enrollment.

    Examples
    --------
    Use this result for readout input rather than accepting caller-authored enrollment dates.
    """
    commitment = next((item for item in ledger.recruitment_commitments if item.study_id == study_id), None)
    if commitment is None:
        raise ValueError("study requires a prospective recruitment commitment")
    reservation = next(item for item in ledger.reservations if item.plan.study_id == study_id)
    if reservation.closed:
        existing = next((item for item in ledger.enrollments if item.study_id == study_id), None)
        patients = () if existing is None else tuple(sorted(existing.patients, key=lambda item: item.enrollment_day))
        if (
            not math.isfinite(programme_day)
            or programme_day < commitment.start_day
            or any(patient.enrollment_day > programme_day for patient in patients)
        ):
            raise ValueError("closed enrollment cannot be reconstructed at an earlier cutoff")
        return TrialDevAccruedRecruitment(screened=0 if existing is None else existing.screened, patients=patients)
    accrued = trialdev_accrued_recruitment(
        reservation,
        commitment.tariff,
        start_day=commitment.start_day,
        programme_day=programme_day,
        pause=(
            None
            if commitment.pause_after_enrolled is None
            else TrialDevRecruitmentPause(
                after_enrolled=commitment.pause_after_enrolled,
                resume_day=next(
                    (item.resume_day for item in ledger.recruitment_continuations if item.study_id == study_id), None
                ),
            )
        ),
    )

    if not reservation.closed_arms and reservation.retained_control_ids is None:
        return accrued
    existing = next((item for item in ledger.enrollments if item.study_id == study_id), None)
    retained = () if existing is None else existing.patients
    if any(patient.enrollment_day > programme_day for patient in retained):
        raise ValueError("closed-arm enrollment cannot be reconstructed at an earlier cutoff")
    active_ids = {patient.participant_id for patient in reservation.active_patients}
    active = tuple(patient for patient in accrued.patients if patient.participant_id in active_ids)
    kept = {patient.participant_id: patient for patient in (*active, *retained)}
    final_screen = max(
        [
            0 if existing is None else existing.screened,
            *(patient.screening_ordinal for patient in reservation.active_patients),
        ]
    )
    return TrialDevAccruedRecruitment(
        screened=min(accrued.screened, final_screen),
        patients=tuple(sorted(kept.values(), key=lambda item: item.enrollment_day)),
    )


def continue_trialdev_recruitment(
    ledger: TrialDevBudgetLedger, continuation: TrialDevRecruitmentContinuation
) -> TrialDevBudgetLedger:
    """Record one immutable continuation after a paused cohort readout.

    Parameters
    ----------
    ledger
        Current programme state with the accrued cohort and its readout receipt.
    continuation
        Decision date, no earlier than evidence already released for the study.

    Returns
    -------
    TrialDevBudgetLedger
        State preserving allocation, earlier enrollment and cost history.

    Raises
    ------
    ValueError
        If continuation changes, is backdated, or lacks an open paused study readout.

    Examples
    --------
    Repeating the same decision returns the existing state without a second charge.
    """
    for existing in ledger.recruitment_continuations:
        if existing.study_id == continuation.study_id:
            if existing != continuation:
                raise ValueError("study continuation is already committed to a different date")
            return ledger
    reservation = next((item for item in ledger.reservations if item.plan.study_id == continuation.study_id), None)
    commitment = next(
        (item for item in ledger.recruitment_commitments if item.study_id == continuation.study_id), None
    )
    enrollment = next((item for item in ledger.enrollments if item.study_id == continuation.study_id), None)
    receipts = [item for item in ledger.receipts if item.study_id == continuation.study_id]
    if (
        reservation is None
        or reservation.closed
        or commitment is None
        or commitment.pause_after_enrolled is None
        or enrollment is None
        or len(enrollment.patients) != commitment.pause_after_enrolled
        or not any(item.kind == "readout" for item in receipts)
    ):
        raise TrialDevActionRejectedError("continuation requires an open paused cohort with a published readout")
    if continuation.resume_day < max(item.programme_day for item in receipts):
        raise ValueError("continuation cannot be backdated before released evidence")
    return TrialDevBudgetLedger.model_validate(
        {**ledger.model_dump(), "recruitment_continuations": (*ledger.recruitment_continuations, continuation)}
    )


class TrialDevPhaseDesignLimits(BaseModel):
    """Public phase-specific design limits and the tariff enforced at admission."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    phase: TrialDevPhase
    prerequisite_phase: TrialDevPhase | None
    minimum_completed_participants_per_arm: int = Field(default=MINIMUM_PHASE_PARTICIPANTS_PER_ARM)
    minimum_participants_per_arm: int
    maximum_participants_per_arm: int
    follow_up_days: tuple[int, ...]
    tariff: TrialDevStudyTariff


def trialdev_phase_design_limits(phase: TrialDevPhase, collection_days: tuple[int, ...]) -> TrialDevPhaseDesignLimits:
    """Return the public phase rules used to admit a numerical study design.

    Parameters
    ----------
    phase
        Development phase selected by the analyst.
    collection_days
        Supported clinical programme observation grid.

    Returns
    -------
    TrialDevPhaseDesignLimits
        Enrollment range, follow-up choices and quantity-based sponsor tariff.

    Raises
    ------
    ValueError
        If the programme observation grid is unsupported.
    """
    if collection_days not in ((28, 56, 84, 168, 336), (28, 56, 112, 224), (28, 56, 182, 364, 728)):
        raise ValueError("unsupported native programme collection grid")
    # A confirmation arm must be able to meet the approval exposure rule on its own.
    minimum, maximum = {
        "early_safety": (10, 100),
        "exploration": (30, 1500),
        "confirmation": (TRIALDEV_APPROVAL_MINIMUM_COMPLETED_TREATED, 3000),
    }[phase]
    allowed = {
        "early_safety": (28, 56),
        "exploration": collection_days[-3:-1],
        "confirmation": collection_days[-2:],
    }[phase]
    return TrialDevPhaseDesignLimits(
        phase=phase,
        prerequisite_phase=_PRECEDING_PHASE.get(phase),
        minimum_participants_per_arm=minimum,
        maximum_participants_per_arm=maximum,
        follow_up_days=allowed,
        tariff=trialdev_phase_tariff(phase),
    )


def validate_trialdev_phase_design(
    commitment: TrialDevRecruitmentCommitment, plan: TrialDevEnrollmentPlan, collection_days: tuple[int, ...]
) -> None:
    """Check an authored study against the selected phase and native schedule envelope.

    Parameters
    ----------
    commitment
        Prospective treatment, follow-up and resource design.
    plan
        Exact reserved arm caps.
    collection_days
        Native programme assessment grid from the verified frozen manifest.

    Returns
    -------
    None
        Valid designs may proceed to durable commitment.

    Raises
    ------
    ValueError
        If sizes, follow-up or tariffs do not match the supported phase.

    Examples
    --------
    Lower precision at the supported minimum is a task consequence, not grounds for rejection.
    """
    limits = trialdev_phase_design_limits(commitment.phase, collection_days)
    counts = {arm.arm_id: arm.participant_count for arm in plan.arms}
    if any(
        not limits.minimum_participants_per_arm <= count <= limits.maximum_participants_per_arm
        for count in counts.values()
    ):
        raise TrialDevActionRejectedError(
            f"{commitment.phase} arms must have {limits.minimum_participants_per_arm}-"
            f"{limits.maximum_participants_per_arm} participants each, control included"
        )
    if commitment.follow_up_days not in limits.follow_up_days:
        raise TrialDevActionRejectedError(
            f"{commitment.phase} follow-up must be one of {', '.join(map(str, limits.follow_up_days))} days"
        )
    if commitment.tariff != limits.tariff:
        raise ValueError("study tariff must match its declared development phase")


def close_trialdev_arm(ledger: TrialDevBudgetLedger, study_id: str, arm_id: str) -> TrialDevBudgetLedger:
    """Stop an active arm's future enrollment while retaining its accrued obligations.

    Parameters
    ----------
    ledger
        Programme state after an accrued study readout.
    study_id
        Reserved study whose active arm is dropped.
    arm_id
        Active treatment arm; the shared control cannot be dropped independently.

    Returns
    -------
    TrialDevBudgetLedger
        State releasing unused arm reservations and preserving actual patients,
        allocations, screening history and all expenditure/follow-up obligations.

    Raises
    ------
    ValueError
        If the arm is unknown, is control, or lacks an open study readout.

    Examples
    --------
    Drop an unpromising arm without reassigning its already randomized participants.
    """
    reservation = next((item for item in ledger.reservations if item.plan.study_id == study_id), None)
    if reservation is None or arm_id == "control" or arm_id not in {q.arm_id for q in reservation.plan.arms}:
        active = sorted({q.arm_id for q in reservation.plan.arms} - {"control"}) if reservation is not None else []
        raise TrialDevActionRejectedError(
            f"{arm_id!r} is not an active arm of {study_id}; active arms: {', '.join(active) or 'none'}"
        )
    if arm_id in reservation.closed_arms:
        return ledger
    if reservation.closed or not any(
        receipt.study_id == study_id and receipt.kind == "readout" for receipt in ledger.receipts
    ):
        raise TrialDevActionRejectedError("an arm can be dropped only from an open study after its first readout")
    closed_arms = tuple(sorted((*reservation.closed_arms, arm_id)))
    active_arms = {q.arm_id for q in reservation.plan.arms} - {"control"}
    changed = reservation.model_copy(update={"closed_arms": closed_arms, "closed": set(closed_arms) == active_arms})
    return TrialDevBudgetLedger.model_validate(
        {
            **ledger.model_dump(),
            "reservations": tuple(changed if item.plan.study_id == study_id else item for item in ledger.reservations),
        }
    )


def reduce_trialdev_control(
    ledger: TrialDevBudgetLedger, study_id: str, participant_count: int
) -> TrialDevBudgetLedger:
    """Reduce future control enrollment after an active-arm drop without changing enrolled patients.

    Parameters
    ----------
    ledger
        Programme state after a study readout and active-arm closure.
    study_id
        Still-active multi-arm study.
    participant_count
        Reduced total control cap, including controls already enrolled.

    Returns
    -------
    TrialDevBudgetLedger
        Original controls retained by accrued identity, then original screening order.
        Unselected future controls are released.

    Raises
    ------
    ValueError
        If the cap increases, erases accrued controls,
        or is requested before an arm drop.

    Examples
    --------
    Reduction changes only future enrollment; no outcomes enter control selection.
    """
    if isinstance(participant_count, bool) or not isinstance(participant_count, int) or participant_count < 1:
        raise TrialDevActionRejectedError("control_participants must be a positive integer")
    reservation = next((item for item in ledger.reservations if item.plan.study_id == study_id), None)
    if reservation is None or reservation.closed or not reservation.closed_arms:
        raise TrialDevActionRejectedError("control can be reduced only in an open study after an arm was dropped")
    controls = sorted(
        (p for p in reservation.active_patients if p.arm_id == "control"), key=lambda p: p.screening_ordinal
    )
    if participant_count > len(controls):
        raise TrialDevActionRejectedError(f"control_participants must be at most the current cap of {len(controls)}")
    if participant_count == len(controls):
        return ledger
    existing = next((item for item in ledger.enrollments if item.study_id == study_id), None)
    retained = {p.participant_id for p in (() if existing is None else existing.patients) if p.arm_id == "control"}
    if len(retained) > participant_count:
        raise TrialDevActionRejectedError(
            f"control_participants must be at least the {len(retained)} controls already enrolled"
        )
    for patient in controls:
        if len(retained) == participant_count:
            break
        retained.add(patient.participant_id)
    changed = reservation.model_copy(
        update={"retained_control_ids": tuple(p.participant_id for p in controls if p.participant_id in retained)}
    )
    return TrialDevBudgetLedger.model_validate(
        {
            **ledger.model_dump(),
            "reservations": tuple(changed if item.plan.study_id == study_id else item for item in ledger.reservations),
        }
    )


def advance_trialdev_clock(ledger: TrialDevBudgetLedger, programme_day: float) -> TrialDevBudgetLedger:
    """Accrue every admitted study's patients and obligations without exposing a new look.

    Parameters
    ----------
    ledger
        Current private study commitments, decisions and accounting.
    programme_day
        New programme clock, no earlier than recorded resource or decision activity.

    Returns
    -------
    TrialDevBudgetLedger
        Fully reconciled enrollment and quantity-based costs at the new clock.
        Identical updates preserve receipt identity and do not charge twice.

    Raises
    ------
    ValueError
        If the clock regresses or actual obligations exceed the programme budget.

    Examples
    --------
    Advance the clock before terminal valuation; no participant observation file is generated.
    """
    latest = ledger.current_day
    if isinstance(programme_day, bool) or not math.isfinite(programme_day):
        raise ValueError("programme resource clock cannot regress or be nonfinite")
    if programme_day < latest:
        raise TrialDevActionRejectedError(f"the programme is already at day {latest:g}; choose a later day")
    if programme_day > TRIALDEV_HORIZON_DAYS:
        raise TrialDevActionRejectedError(
            f"day {programme_day:g} is beyond the programme horizon, day {TRIALDEV_HORIZON_DAYS}"
        )
    if ledger.terminal is not None:
        if programme_day == ledger.terminal.programme_day:
            return ledger
        raise ValueError("terminal programme clock is immutable")
    programme_day = float(programme_day)
    updated = ledger
    for design in sorted(ledger.recruitment_commitments, key=lambda item: item.study_id):
        if design.start_day > programme_day:
            continue
        reservation = next(item for item in ledger.reservations if item.plan.study_id == design.study_id)
        previous = next((item for item in ledger.enrollments if item.study_id == design.study_id), None)
        if reservation.closed and previous is None:
            continue
        accrued = trialdev_committed_accrual(updated, design.study_id, programme_day=programme_day)
        assert ledger.pool_manifest_sha256 is not None
        updated = reconcile_trialdev_enrollment(
            updated,
            pool_manifest_sha256=ledger.pool_manifest_sha256,
            enrollment=TrialDevStudyEnrollment(
                study_id=design.study_id,
                design_sha256=(
                    previous.design_sha256
                    if previous is not None
                    else canonical_payload_sha256(design.model_dump(mode="json"))
                ),
                patients=tuple(sorted(accrued.patients, key=lambda item: item.participant_label)),
                screened=accrued.screened,
            ),
        )
        cost = trialdev_study_cost(
            design.tariff,
            sites=design.sites,
            screened=accrued.screened,
            enrollment_days=tuple(patient.enrollment_day for patient in accrued.patients),
            programme_day=programme_day,
            follow_up_days=design.follow_up_days,
            assessment_days=design.assessment_days,
            surveillance_days=design.surveillance_days,
        )
        identity = canonical_payload_sha256({"study_id": design.study_id, "programme_day": programme_day})
        updated = reconcile_trialdev_cost(
            updated,
            TrialDevCostReceipt(
                study_id=design.study_id,
                readout_id=f"resource:{identity}",
                kind="resource",
                cost=cost,
                programme_day=programme_day,
            ),
        )
    return TrialDevBudgetLedger.model_validate({**updated.model_dump(), "resource_day": programme_day})


def finish_trialdev_programme(
    ledger: TrialDevBudgetLedger, decision: TrialDevTerminalDecision
) -> TrialDevBudgetLedger:
    """Freeze an explicit terminal choice after accruing resources and settling liabilities.

    Parameters
    ----------
    ledger
        Evaluator-owned programme state.
    decision
        Explicit stop or advancement and actual programme day.

    Returns
    -------
    TrialDevBudgetLedger
        Frozen terminal state; original receipts and allocation histories remain intact.

    Raises
    ------
    ValueError
        If the decision changes, exceeds resources or advances an unconfirmed regimen.

    Examples
    --------
    Stopping never requires an efficacy threshold. Advancing requires a confirmation
    readout for the selected treatment; approval is then decided by the attached
    evaluator review, not by the participant's own analysis.
    """
    if ledger.terminal is not None:
        if ledger.terminal != decision:
            raise ValueError("terminal programme decision is immutable")
        return ledger
    if decision.decision == "advance":
        if (decision.asset_id, decision.dose) not in trialdev_observed_treatments(ledger, "confirmation"):
            confirmed = trialdev_observed_treatments(ledger, "confirmation")
            raise TrialDevActionRejectedError(
                "only a compound and dose with a confirmation readout (arm and control observed) can be advanced; "
                f"eligible now: {', '.join(f'{a} {d.value}' for a, d in sorted(confirmed, key=str)) or 'none'}"
            )
        if decision.regulatory_review is None:
            raise ValueError("advancement requires the evaluator's regulatory review at submission")
    current = advance_trialdev_clock(ledger, decision.programme_day)
    return TrialDevBudgetLedger.model_validate(
        {
            **current.model_dump(),
            "terminal": decision,
            "settlement_usd": current.remaining_liability_usd,
            "reservations": tuple(item.model_copy(update={"closed": True}) for item in current.reservations),
        }
    )
