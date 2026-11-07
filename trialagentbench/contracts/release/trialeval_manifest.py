"""TrialEval participant-release manifest contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench.contracts.core.trialeval_factors import TrialEvalEvidenceFactorsV1
from trialagentbench.contracts.trialeval_diagnostics import (
    TrialEvalParticipantDiagnosticDictionaryV1,
)
from trialagentbench.contracts.trialeval_methods import (
    TrialEvalParticipantMethodDictionaryV1,
    TrialEvalParticipantMethodV1,
)
from trialagentbench.io.checksums import canonical_payload_sha256


class TrialEvalParticipantArtifactV1(BaseModel):
    """One checksummed file in a TrialEval participant release."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rel_path: str = Field(..., min_length=1)
    sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(..., ge=0)


class TrialEvalParticipantManifestV1(BaseModel):
    """Task order, evidence factors, and file identities for a participant release."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["v1"] = "v1"
    schema_id: Literal["trial_analysis_public_bundle_manifest/v1"] = "trial_analysis_public_bundle_manifest/v1"
    bundle_rel_root: Literal["public"] = "public"
    applied_baseline_profile_id: str | None = Field(...)
    applied_baseline_profile_sha256: str | None = Field(
        ...,
        pattern=r"^[0-9a-f]{64}$",
    )
    task_ids: tuple[str, ...] = Field(..., min_length=1)
    task_evidence_factors: dict[str, TrialEvalEvidenceFactorsV1]
    artifacts: tuple[TrialEvalParticipantArtifactV1, ...] = ()
    checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_manifest(self) -> TrialEvalParticipantManifestV1:
        if (self.applied_baseline_profile_id is None) != (self.applied_baseline_profile_sha256 is None):
            raise ValueError("applied baseline profile id and SHA-256 must be supplied together")
        task_ids = tuple(self.task_ids)
        if len(task_ids) != len(set(task_ids)) or any(not task_id for task_id in task_ids):
            raise ValueError("TrialEval participant task_ids must be unique and non-empty.")
        if set(self.task_evidence_factors) != set(task_ids):
            raise ValueError("task_evidence_factors must contain exactly one entry for every task_id.")
        artifacts = tuple(sorted(self.artifacts, key=lambda row: row.rel_path))
        paths = [row.rel_path for row in artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("TrialEval participant artifact paths must be unique.")
        object.__setattr__(self, "artifacts", artifacts)
        payload = self.model_dump(mode="json", exclude={"checksum"})
        digest = canonical_payload_sha256(payload)
        if self.checksum is not None and self.checksum != digest:
            raise ValueError("TrialEval participant manifest checksum does not match its payload.")
        object.__setattr__(self, "checksum", digest)
        return self


__all__ = [
    "TrialEvalParticipantArtifactV1",
    "TrialEvalParticipantDiagnosticDictionaryV1",
    "TrialEvalParticipantManifestV1",
    "TrialEvalParticipantMethodDictionaryV1",
    "TrialEvalParticipantMethodV1",
]
