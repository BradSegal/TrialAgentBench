"""Typed contract for deterministic portable result exports."""

from __future__ import annotations

from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.types import JsonValue

from trialagentbench.io import canonical_payload_sha256

ResultArtifactKind = Literal["run", "grade", "verification", "analysis"]
ResultExportProfile = Literal["full_custody", "analysis_reproduction"]


class ResultExportMemberV1(BaseModel):
    """One immutable file included in a portable result archive."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class ResultExportSourceV1(BaseModel):
    """One caller-owned artifact tree included in a result archive."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ResultArtifactKind
    label: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    archive_prefix: str = Field(min_length=1)
    file_count: int = Field(ge=1)
    custody_file_count: int | None = Field(default=None, ge=1)
    custody_byte_length: int | None = Field(default=None, ge=1)
    custody_tree_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_custody_identity(self) -> ResultExportSourceV1:
        """Require the complete external tree identity as one all-or-none record."""

        values = (self.custody_file_count, self.custody_byte_length, self.custody_tree_sha256)
        if any(value is None for value in values) != all(value is None for value in values):
            raise ValueError("result-export custody identity must be complete when present")
        if self.custody_file_count is not None and self.file_count > self.custody_file_count:
            raise ValueError("included result files cannot exceed the external custody tree")
        return self


class ResultExportBundleManifestV1(BaseModel):
    """Checksummed identity of one portable result archive."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["ariadne.result_export_bundle/v1"] = "ariadne.result_export_bundle/v1"
    release_id: str = Field(min_length=1)
    release_stage: Literal["benchmark_publication"]
    release_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    release_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    export_profile: ResultExportProfile = "full_custody"
    sources: tuple[ResultExportSourceV1, ...] = Field(min_length=2)
    members: tuple[ResultExportMemberV1, ...] = Field(min_length=1)
    member_count: int = Field(ge=1)
    checksum: str = ""

    @model_validator(mode="after")
    def _validate_manifest(self) -> ResultExportBundleManifestV1:
        if self.member_count != len(self.members):
            raise ValueError("member_count must equal the result member inventory")
        source_kinds = {source.kind for source in self.sources}
        required = {"run", "grade"}
        if not required.issubset(source_kinds):
            raise ValueError("result export requires run, grade, and verification artifacts")
        if self.export_profile == "full_custody" and any(
            source.file_count != source.custody_file_count
            for source in self.sources
            if source.custody_file_count is not None
        ):
            raise ValueError("full-custody exports cannot omit files from declared source trees")
        source_prefixes = [source.archive_prefix for source in self.sources]
        if len(source_prefixes) != len(set(source_prefixes)):
            raise ValueError("result export source prefixes must be unique")
        member_paths = [member.path for member in self.members]
        if member_paths != sorted(member_paths) or len(member_paths) != len(set(member_paths)):
            raise ValueError("result export member paths must be sorted and unique")
        expected = canonical_payload_sha256(cast(JsonValue, self.model_dump(mode="json", exclude={"checksum"})))
        if self.checksum and self.checksum != expected:
            raise ValueError("result export manifest checksum mismatch")
        object.__setattr__(self, "checksum", expected)
        return self


__all__ = [
    "ResultArtifactKind",
    "ResultExportProfile",
    "ResultExportBundleManifestV1",
    "ResultExportMemberV1",
    "ResultExportSourceV1",
]
