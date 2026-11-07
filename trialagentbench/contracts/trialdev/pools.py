"""Standalone wire contracts for private frozen TrialDev patient pools."""

import math
from datetime import date
from typing import Annotated, Literal, Self

from clinical_data_tables import validate_clinical_epochs
from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench.contracts.trialdev.release import TrialDevDose, TrialDevReleaseArtifactV1

TrialDevPoolParticipantCount = Annotated[int, Field(strict=True, ge=40, le=64000)]

_Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
TrialDevArmId = Annotated[str, Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$")]


class TrialDevPoolContrast(BaseModel):
    """Exact asset-control contrasts over one fixed finite patient population."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    asset_id: str
    dose: TrialDevDose
    participant_count: int = Field(ge=1)
    reference_days: float = Field(gt=0)
    outcome_unit: str
    outcome_mean_difference: float
    safety_risk_difference: float = Field(ge=-1, le=1)


class TrialDevPoolPartition(BaseModel):
    """Treatment identity of one immutable private record directory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_id: str
    dose: TrialDevDose | None
    directory: str = Field(pattern=r"^partitions/p[0-9]{3}$")


class TrialDevAssignedArm(BaseModel):
    """Committed mapping from a public randomized arm to its private pool treatment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    arm_id: TrialDevArmId
    partition: TrialDevPoolPartition


class TrialDevRecruitmentCondition(BaseModel):
    """One public recorded-baseline comparison in a clinical recruitment group."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    field_id: str = Field(min_length=1)
    operator: Literal["equal", "not_equal", "greater_than", "greater_than_or_equal", "less_than", "less_than_or_equal"]
    value: float | str

    @model_validator(mode="after")
    def validate_comparison(self) -> Self:
        """Require finite numerical thresholds for ordered comparisons."""
        if self.operator not in {"equal", "not_equal"} and not math.isfinite(float(self.value)):
            raise ValueError("ordered recruitment comparisons require finite thresholds")
        return self


class TrialDevRecruitmentGroup(BaseModel):
    """Possibly overlapping clinical group, defined by all its recorded conditions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    group_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    label: str = Field(min_length=1)
    conditions: tuple[TrialDevRecruitmentCondition, ...] = Field(min_length=1)
    participant_count: int = Field(ge=0, strict=True)

    @property
    def membership_column(self) -> str:
        """Return the frozen membership field consumed by the standalone allocator."""
        return f"recruitment_{self.group_id}"


class TrialDevRecruitmentSite(BaseModel):
    """Public site identity, forecast rate and finite candidate capacity."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    site_id: str = Field(min_length=1)
    country_code: str = Field(pattern=r"^[A-Z]{3}$")
    opening_day: float = Field(ge=0, description="Opening delay after study startup is complete")
    screening_candidates_per_month: float = Field(
        gt=0, description="Expected screened candidates per 30 active days, including screen failures"
    )
    candidate_count: int = Field(ge=0, strict=True, description="Initial capacity before enrolments and reservations")


class TrialDevRecruitmentFrame(BaseModel):
    """Clinical recruitment rules for the unchanged finite target population."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    groups: tuple[TrialDevRecruitmentGroup, ...] = Field(min_length=1, max_length=8)
    sites: tuple[TrialDevRecruitmentSite, ...] = Field(min_length=2)
    screen_failure_fraction: float = Field(
        ge=0,
        lt=1,
        description=(
            "Share of screened candidates who fail screening independently of their recorded data; failures "
            "consume screening effort and arrival time but are not enrolled."
        ),
    )

    @property
    def participant_count(self) -> int:
        """Return the complete frame size; overlapping group counts are not additive."""
        return sum(site.candidate_count for site in self.sites)

    @model_validator(mode="after")
    def validate_groups(self) -> Self:
        """Require unique groups and sites with counts inside the complete frame."""
        if len({site.site_id for site in self.sites}) != len(self.sites):
            raise ValueError("recruitment site identities must be unique")
        if len({group.group_id for group in self.groups}) != len(self.groups):
            raise ValueError("recruitment group identities must be unique")
        if any(group.participant_count > self.participant_count for group in self.groups):
            raise ValueError("recruitment group count exceeds the complete population")
        return self


class TrialDevFrozenPool(BaseModel):
    """Byte-bound private pool readable without native generation packages."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    calendar_origin: date | None = Field(default=None, description="Authored date corresponding to programme day zero")
    schema_id: Literal["trialdev.frozen-pool/v8"] = "trialdev.frozen-pool/v8"
    audience: Literal["evaluation"] = "evaluation"
    world_id: str
    outcome_object: str = Field(min_length=1)
    participant_count: TrialDevPoolParticipantCount
    collection_days: tuple[Annotated[int, Field(gt=0)], ...] = Field(min_length=1)
    epochs: tuple[tuple[str, float, float], ...] = ()
    monotone_dropout: bool = False
    recruitment: TrialDevRecruitmentFrame
    baseline_fields: tuple[Annotated[str, Field(min_length=1)], ...] = Field(min_length=1)
    source_provenance_sha256: _Sha256 = Field(
        description="Acquisition-time source provenance only; reuse requires the per-world generation_fingerprint.json."
    )
    partitions: tuple[TrialDevPoolPartition, ...] = Field(min_length=2)
    targets: tuple[TrialDevPoolContrast, ...] = Field(min_length=1)
    artifacts: tuple[TrialDevReleaseArtifactV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_inventory(self) -> Self:
        """Require one control, unique treatments and matching fixed targets."""
        validate_clinical_epochs(self.epochs, horizon=self.collection_days[-1])
        public = self.baseline_fields
        if len(set(public)) != len(public):
            raise ValueError("pool recorded baseline fields must be unique")
        if {
            "participant_id",
            "enrollment_day",
            "site_id",
            "country_code",
            "screening_day",
            "site_candidate_index",
        } & set(public):
            raise ValueError("clinical field names cannot overwrite patient or recruitment identifiers")
        if {group.membership_column for group in self.recruitment.groups} & set(public):
            raise ValueError("clinical fields cannot overwrite recruitment memberships")
        if any(
            condition.field_id not in public for group in self.recruitment.groups for condition in group.conditions
        ):
            raise ValueError("recruitment groups must use recorded public baseline fields")
        if self.recruitment.participant_count != self.participant_count:
            raise ValueError("recruitment counts must cover the complete fixed population")
        if tuple(sorted(set(self.collection_days))) != self.collection_days:
            raise ValueError("pool collection days must be unique and increasing")
        if self.partitions[0].dose is not None or any(item.dose is None for item in self.partitions[1:]):
            raise ValueError("pool inventory requires exactly one control, first")
        pairs = [(item.asset_id, item.dose) for item in self.partitions[1:]]
        directories = [item.directory for item in self.partitions]
        paths = [item.relative_path for item in self.artifacts]
        if len(set(pairs)) != len(pairs) or len(set(directories)) != len(directories) or len(set(paths)) != len(paths):
            raise ValueError("pool inventory contains duplicate treatments, directories or artifacts")
        if [(item.asset_id, item.dose) for item in self.targets] != pairs:
            raise ValueError("pool targets must match every active treatment in order")
        if any(
            item.participant_count != self.participant_count or item.reference_days != self.collection_days[-1]
            for item in self.targets
        ):
            raise ValueError("pool targets must use the complete fixed population and final date")
        return self
