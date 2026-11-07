"""Shared construct and released-content identities for TrialAgentBench."""

from __future__ import annotations

from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from trialagentbench.io import canonical_payload_sha256


class BenchmarkReleaseIdentityV1(BaseModel):
    """Exact outer identity shared by participant and evaluator projections."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["ariadne.trialeval.release/v1", "ariadne.trialdev.release/v1"]
    data_provenance: Literal["generated_under_declared_trial_mechanisms"]
    release_id: str = Field(pattern=r"^release_[0-9a-f]{64}$")
    construct_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    qualification_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_count: int = Field(ge=1)
    participant_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trial_count: int | None = Field(default=None, ge=1)
    world_count: int | None = Field(default=None, ge=1)
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        """Bind the construct, stream census, and exact manifest content."""

        is_trialeval = self.schema_id == "ariadne.trialeval.release/v1"
        if is_trialeval != (self.trial_count is not None) or is_trialeval == (self.world_count is not None):
            raise ValueError("benchmark release has incompatible stream counts")
        if self.release_id != f"release_{self.construct_sha256}":
            raise ValueError("benchmark release identity differs from its construct")
        expected = canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}, exclude_none=True))
        if self.checksum != expected:
            raise ValueError("benchmark release manifest checksum mismatch")
        return self


class PairedParticipantArchiveIdentityV1(BaseModel):
    """Participant archive identity carried by its paired evaluator ZIP."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["ariadne.trialeval.participant_archive_identity/v1"] = (
        "ariadne.trialeval.participant_archive_identity/v1"
    )
    release_id: str = Field(pattern=r"^release_[0-9a-f]{64}$")
    participant_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    participant_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        release_id: str,
        participant_archive_sha256: str,
        participant_manifest_checksum: str,
    ) -> PairedParticipantArchiveIdentityV1:
        """Construct the checksum-bound identity from one participant ZIP."""

        payload = {
            "schema_id": "ariadne.trialeval.participant_archive_identity/v1",
            "release_id": release_id,
            "participant_archive_sha256": participant_archive_sha256,
            "participant_manifest_checksum": participant_manifest_checksum,
        }
        return cls(**payload, checksum=canonical_payload_sha256(cast(JsonValue, payload)))

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        """Require the checksum to bind the release, archive, and role manifest."""

        expected = canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))
        if self.checksum != expected:
            raise ValueError("participant archive identity checksum mismatch")
        return self


__all__ = ["BenchmarkReleaseIdentityV1", "PairedParticipantArchiveIdentityV1"]
