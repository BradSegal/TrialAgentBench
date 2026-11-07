"""Public authored study choices and their private frozen-pool binding."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from trialagentbench.contracts.trialdev.pools import TrialDevAssignedArm, TrialDevFrozenPool, TrialDevRecruitmentFrame
from trialagentbench.contracts.trialdev.release import TrialDevDose
from trialagentbench.trialdev.errors import TrialDevActionRejectedError
from trialagentbench.trialdev.recruitment import (
    TrialDevArmEnrollment,
    TrialDevEnrollmentPlan,
    TrialDevRecruitmentSelection,
    plan_trialdev_enrollment,
)
from trialagentbench.trialdev.resources import (
    TRIALDEV_APPROVAL_MINIMUM_COMPLETED_TREATED,
    TRIALDEV_APPROVAL_TWO_SIDED_ALPHA,
    TrialDevLook,
    TrialDevPhase,
    TrialDevPhaseDesignLimits,
    TrialDevRecruitmentCommitment,
    trialdev_phase_design_limits,
    trialdev_phase_tariff,
)


class TrialDevAvailableTreatment(BaseModel):
    """Participant-visible treatment identity without private partition locations."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    asset_id: str
    dose: TrialDevDose | None


class TrialDevApprovalRule(BaseModel):
    """Public regulatory rule applied to an advanced treatment at submission."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    two_sided_alpha: float = TRIALDEV_APPROVAL_TWO_SIDED_ALPHA
    minimum_completed_treated: int = TRIALDEV_APPROVAL_MINIMUM_COMPLETED_TREATED
    description: str = (
        "This is a simplified benchmark approval rule, not regulatory practice. At advancement the "
        "benchmark reviewer uses fully ascertained frozen outcomes for every participant randomized to "
        "the advanced compound/dose or control in confirmation studies whose planned follow-up has elapsed "
        "by the programme clock, including outcomes unavailable in participant readouts after observation loss. "
        "It applies a two-sided test of benefit at the stated alpha, pooled by inverse variance across studies, "
        "and requires at least the stated number of calendar-matured treated participants. Calendar maturity "
        "is distinct from observed endpoint completion. This approval is required for benchmark market value."
    )


class TrialDevDesignSpace(BaseModel):
    """Available treatments, recruitment frame, executable phase rules and approval rule."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    treatments: tuple[TrialDevAvailableTreatment, ...]
    recruitment: TrialDevRecruitmentFrame
    visit_days: tuple[int, ...] = Field(
        min_length=1,
        description=(
            "Available assessment and safety-contact days after enrolment (day 0), through the selected follow-up."
        ),
    )
    phases: tuple[TrialDevPhaseDesignLimits, ...]
    target_population_participants: int = Field(
        ge=1,
        description=(
            "Size of the fixed programme population. Study and terminal targets are treatment contrasts over "
            "this complete finite population at the stated follow-up; studies sample patients from it."
        ),
    )
    approval: TrialDevApprovalRule = Field(default_factory=TrialDevApprovalRule)


def trialdev_design_space(manifest: TrialDevFrozenPool) -> TrialDevDesignSpace:
    """Project public design information from a verified frozen pool.

    Parameters
    ----------
    manifest
        Evaluator-owned pool inventory. Potential outcomes, private paths and
        construction provenance are excluded from the returned design space.

    Returns
    -------
    TrialDevDesignSpace
        Treatment/regimen choices, recruitment and phase constraints, the finite
        target population size and the regulatory approval rule.

    Raises
    ------
    ValueError
        If the pool's clinical schedule is unsupported by runtime admission.
    """
    phases: tuple[TrialDevPhase, ...] = ("early_safety", "exploration", "confirmation")
    return TrialDevDesignSpace(
        treatments=tuple(
            TrialDevAvailableTreatment(asset_id=row.asset_id, dose=row.dose) for row in manifest.partitions
        ),
        recruitment=manifest.recruitment,
        visit_days=manifest.collection_days,
        phases=tuple(trialdev_phase_design_limits(phase, manifest.collection_days) for phase in phases),
        target_population_participants=manifest.participant_count,
    )


class TrialDevRequestedArm(TrialDevArmEnrollment):
    """Public asset, regimen and numerical enrollment cap for one study arm."""

    participant_count: int = Field(ge=10, le=3000, strict=True)
    asset_id: str = Field(min_length=1)
    dose: TrialDevDose | None


class TrialDevStudyRequest(BaseModel):
    """Agent-authored numerical study design without private patients, paths or prices."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    phase: TrialDevPhase
    arms: tuple[TrialDevRequestedArm, ...] = Field(
        min_length=2,
        max_length=7,
        description="One shared control and distinct asset/regimen pairs, each with a unique arm_id and enrollment cap.",
    )
    sites: int = Field(ge=2, strict=True)
    recruitment: TrialDevRecruitmentSelection = Field(default_factory=TrialDevRecruitmentSelection)
    follow_up_days: int = Field(gt=0, strict=True)
    assessment_days: tuple[int, ...]
    surveillance_days: tuple[int, ...]
    looks: tuple[TrialDevLook, ...] = Field(min_length=1, max_length=2)
    pause_after_enrolled: int | None = Field(
        default=None,
        ge=1,
        le=7999,
        strict=True,
        description="Optional interim recruitment pause before the total arm enrollment cap; omit for uninterrupted recruitment.",
    )

    @model_validator(mode="after")
    def validate_interim_pause(self) -> Self:
        """Reject impossible pauses at the request boundary before committing state."""
        if self.pause_after_enrolled is not None and self.pause_after_enrolled >= sum(
            arm.participant_count for arm in self.arms
        ):
            raise ValueError("pause_after_enrolled must be below total enrollment; omit it for full recruitment")
        return self


def bind_trialdev_study_request(
    request: TrialDevStudyRequest,
    manifest: TrialDevFrozenPool,
    *,
    study_id: str,
    start_day: float,
) -> tuple[TrialDevEnrollmentPlan, TrialDevRecruitmentCommitment]:
    """Resolve public design choices into exact arm caps and private treatment assignments.

    Parameters
    ----------
    request
        Agent-selected assets, regimen, sizes, recruitment, schedules and looks.
    manifest
        Verified evaluator-owned frozen pool inventory and clinical recruitment frame.
    study_id
        Runner-owned identity for the new study.
    start_day
        Current programme day supplied by the runner.

    Returns
    -------
    tuple of TrialDevEnrollmentPlan and TrialDevRecruitmentCommitment
        Exact quota allocation and prospective design for atomic admission.

    Raises
    ------
    ValueError
        If a treatment is unavailable, arms conflict or recruitment support is invalid.

    Examples
    --------
    Agent inputs never select frozen patient identities, private partitions or unit prices.
    """
    treatments = {(item.asset_id, item.dose): item for item in manifest.partitions}
    assigned = []
    for arm in request.arms:
        partition = treatments.get((arm.asset_id, arm.dose))
        if partition is None:
            available = ", ".join(f"{row.asset_id} {row.dose.value}" for row in manifest.partitions if row.dose)
            raise TrialDevActionRejectedError(f"{arm.asset_id} {arm.dose} is not available; available: {available}")
        assigned.append(TrialDevAssignedArm(arm_id=arm.arm_id, partition=partition))
    try:
        plan = plan_trialdev_enrollment(
            manifest.recruitment,
            study_id=study_id,
            arms=tuple(
                TrialDevArmEnrollment(arm_id=arm.arm_id, participant_count=arm.participant_count)
                for arm in sorted(request.arms, key=lambda item: treatments[(item.asset_id, item.dose)].directory)
            ),
            recruitment=request.recruitment,
        )
        if request.sites > len(manifest.recruitment.sites):
            raise TrialDevActionRejectedError(f"sites must be at most {len(manifest.recruitment.sites)}")
        plan = plan.model_copy(
            update={"site_ids": tuple(site.site_id for site in manifest.recruitment.sites[: request.sites])}
        )
        commitment = TrialDevRecruitmentCommitment(
            study_id=study_id,
            phase=request.phase,
            tariff=trialdev_phase_tariff(request.phase),
            sites=request.sites,
            start_day=start_day,
            pause_after_enrolled=request.pause_after_enrolled,
            looks=request.looks,
            arms=tuple(assigned),
            follow_up_days=request.follow_up_days,
            assessment_days=request.assessment_days,
            surveillance_days=request.surveillance_days,
        )
    except ValidationError as error:
        # These are participant design relationships, checked once by the
        # commitment contract before any state is persisted. Exclude private
        # partition inputs from the correctable public error.
        messages = "; ".join(item["msg"] for item in error.errors(include_input=False, include_url=False))
        raise TrialDevActionRejectedError(messages) from error
    return plan, commitment
