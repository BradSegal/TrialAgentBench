"""Wire contracts for role-separated TrialDev releases."""

from __future__ import annotations

from enum import Enum
from pathlib import PurePosixPath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from trialagentbench_verifier.response import (
    TRIALDEV_ASSET_ID_PATTERN as TRIALDEV_ASSET_ID_PATTERN,
)
from trialagentbench_verifier.response import (
    TrialDevAction as TrialDevAction,
)
from trialagentbench_verifier.response import (
    TrialDevActionArgumentsV1 as TrialDevActionArgumentsV1,
)

from trialagentbench.io import canonical_payload_sha256

TRIALDEV_MINIMUM_CLUSTER_UNITS = 12


class _ReleaseModel(BaseModel):
    """Strict immutable base for released TrialDev records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class TrialDevStream(str, Enum):
    """Separately evaluated TrialDev tasks."""

    EVIDENCE_ANALYSIS = "evidence_analysis"
    PORTFOLIO_MANAGEMENT = "portfolio_management"


class ProspectivePhase(str, Enum):
    """Prospective evidence stages in a development programme."""

    EXPLORATORY = "exploratory"
    CONFIRMATORY = "confirmatory"


class TrialDevPopulation(str, Enum):
    """Participant populations available for prospective study."""

    ALL_ELIGIBLE = "all_eligible"
    HIGHER_BASELINE_RISK = "higher_baseline_risk"


class TrialDevDose(str, Enum):
    # Retain this docstring: its presence preserves the frozen provider JSON key order.
    """Internal dose identifiers used by the released programme contracts."""

    STANDARD = "standard"
    INTENSIFIED = "intensified"


class TrialDevStudyDesign(str, Enum):
    """Randomized designs supported by the released environment."""

    PARALLEL_GROUP = "parallel_group"
    CLUSTER_PARALLEL = "cluster_parallel"


class TrialDevEndpoint(str, Enum):
    """Decision-relevant outcomes that a prospective study can collect."""

    CLINICAL_OUTCOME = "clinical_outcome"
    SAFETY_RISK = "safety_risk"


class TrialDevProspectiveDesign(_ReleaseModel):
    """Scientific design inputs for native prospective clinical generation."""

    study_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    phase: ProspectivePhase
    population: TrialDevPopulation
    dose: TrialDevDose
    design: TrialDevStudyDesign
    endpoints: tuple[TrialDevEndpoint, ...] = Field(min_length=1)
    participant_count: int = Field(ge=40)
    endpoint_assessment_days: int = Field(gt=0)
    duration_days: int = Field(gt=0)
    cluster_size: int | None = Field(default=None, ge=2)
    intracluster_correlation: float | None = Field(default=None, ge=0, lt=1)

    @model_validator(mode="after")
    def validate_design(self) -> Self:
        """Require coherent outcomes and cluster-design parameters."""

        if len(set(self.endpoints)) != len(self.endpoints):
            raise ValueError("TrialDev prospective study endpoints must be unique")
        if self.endpoint_assessment_days > self.duration_days:
            raise ValueError("TrialDev endpoint assessment cannot occur after study completion")
        clustered = self.design is TrialDevStudyDesign.CLUSTER_PARALLEL
        complete_cluster_parameters = self.cluster_size is not None and self.intracluster_correlation is not None
        if clustered != complete_cluster_parameters:
            raise ValueError("TrialDev cluster studies require cluster size and intracluster correlation")
        if self.cluster_size is not None and self.participant_count % self.cluster_size:
            raise ValueError("TrialDev participant count must contain complete clusters")
        if (
            self.cluster_size is not None
            and self.participant_count // self.cluster_size < TRIALDEV_MINIMUM_CLUSTER_UNITS
        ):
            raise ValueError(
                f"TrialDev cluster study requires at least {TRIALDEV_MINIMUM_CLUSTER_UNITS} independent units"
            )
        return self


class ProspectiveStudyV1(TrialDevProspectiveDesign):
    """Priced prospective study option."""

    cost: float = Field(gt=0)


class TrialDevReleaseArtifactV1(_ReleaseModel):
    """Byte identity of one file in a released role."""

    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_length: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        """Reject paths that are not portable role-relative identities."""

        path = PurePosixPath(self.relative_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in self.relative_path
            or path.as_posix() != self.relative_path
        ):
            raise ValueError("TrialDev artifact path is not portable")
        return self


class TrialDevRoleManifestV1(_ReleaseModel):
    """Complete byte inventory for one TrialDev release role."""

    schema_id: Literal["ariadne.trialdev.role_manifest/v1"] = "ariadne.trialdev.role_manifest/v1"
    release_id: str = Field(pattern=r"^release_[0-9a-f]{64}$")
    role: Literal["participant", "evaluation"]
    world_count: int = Field(ge=1)
    task_count: int = Field(ge=1)
    streams: tuple[TrialDevStream, TrialDevStream]
    artifacts: tuple[TrialDevReleaseArtifactV1, ...] = Field(min_length=1)
    checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_inventory(self) -> Self:
        """Require exact role containment, ordering, coverage, and checksum."""

        paths = tuple(artifact.relative_path for artifact in self.artifacts)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("TrialDev role artifact paths must be ordered and unique")
        if any(PurePosixPath(path).parts[0] != self.role for path in paths):
            raise ValueError("TrialDev role manifest crosses its release boundary")
        if set(self.streams) != set(TrialDevStream):
            raise ValueError("TrialDev role manifest must cover both streams")
        expected = canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))
        if self.checksum is not None and self.checksum != expected:
            raise ValueError("TrialDev role manifest checksum mismatch")
        object.__setattr__(self, "checksum", expected)
        return self


class TrialDevReleaseManifestV1(_ReleaseModel):
    """Identity of a complete participant-and-evaluation TrialDev release.

    ``qualification_sha256`` binds the resource-accounting check artifact.
    Scientific admission is separate, in the candidate-bound public
    qualification report bundle, and is required before release archive export.
    Explicit development packaging does not assert scientific admission.
    """

    schema_id: Literal["ariadne.trialdev.release/v1"] = "ariadne.trialdev.release/v1"
    data_provenance: Literal["generated_under_declared_trial_mechanisms"] = "generated_under_declared_trial_mechanisms"
    release_id: str = Field(pattern=r"^release_[0-9a-f]{64}$")
    construct_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    qualification_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    world_count: int = Field(ge=1)
    task_count: int = Field(ge=1)
    participant_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        """Bind release identity and checksum to the declared construct."""

        if self.release_id != f"release_{self.construct_sha256}":
            raise ValueError("TrialDev release identity differs from its construct")
        expected = canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))
        if self.checksum is not None and self.checksum != expected:
            raise ValueError("TrialDev release manifest checksum mismatch")
        object.__setattr__(self, "checksum", expected)
        return self


class TrialDevMaterialV1(_ReleaseModel):
    """Participant-visible input named by a TrialDev task."""

    relative_path: str = Field(min_length=1)
    description: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        """Reject paths outside the participant world directory."""

        path = PurePosixPath(self.relative_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in self.relative_path
            or path.as_posix() != self.relative_path
        ):
            raise ValueError("TrialDev material path is not portable")
        return self


class TrialDevProgrammeResources(_ReleaseModel):
    """Available money for an agent-designed programme; every programme has the same fixed horizon."""

    budget_usd: float = Field(gt=0, allow_inf_nan=False)


class TrialDevTaskV1(_ReleaseModel):
    """One participant-visible TrialDev question and its required output fields."""

    task_id: str = Field(pattern=r"^td-[0-9a-f]{12}$")
    case_id: str = Field(pattern=r"^tdw-[0-9a-f]{4,16}$")
    stream: TrialDevStream
    data_view: Literal["raw", "analysis_ready"] = "raw"
    resources: TrialDevProgrammeResources | None = None
    title: str = Field(min_length=1)
    clinical_question: str = Field(min_length=1)
    available_materials: tuple[TrialDevMaterialV1, ...] = Field(min_length=1)
    requested_outputs: tuple[str, ...] = Field(min_length=1)
    initial_target_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_materials(self) -> Self:
        """Require one unambiguous description for every staged input."""

        if self.stream is TrialDevStream.EVIDENCE_ANALYSIS and self.resources is not None:
            raise ValueError("Standalone evidence analysis has no development resource budget")
        if len(set(self.initial_target_ids)) != len(self.initial_target_ids) or any(
            not target for target in self.initial_target_ids
        ):
            raise ValueError("Initial numerical target identifiers must be unique and nonempty")
        paths = tuple(material.relative_path for material in self.available_materials)
        if len(set(paths)) != len(paths):
            raise ValueError("TrialDev task material paths must be unique")
        return self


class TrialDevActionResourceV1(_ReleaseModel):
    """Public resource consequence for one portfolio action."""

    action: TrialDevAction
    cost: float = Field(ge=0)
    elapsed_days: int = Field(ge=0)


__all__ = [
    "TRIALDEV_ASSET_ID_PATTERN",
    "TRIALDEV_MINIMUM_CLUSTER_UNITS",
    "ProspectivePhase",
    "ProspectiveStudyV1",
    "TrialDevAction",
    "TrialDevActionResourceV1",
    "TrialDevDose",
    "TrialDevEndpoint",
    "TrialDevMaterialV1",
    "TrialDevPopulation",
    "TrialDevProspectiveDesign",
    "TrialDevReleaseArtifactV1",
    "TrialDevReleaseManifestV1",
    "TrialDevRoleManifestV1",
    "TrialDevStudyDesign",
    "TrialDevStream",
    "TrialDevTaskV1",
]
