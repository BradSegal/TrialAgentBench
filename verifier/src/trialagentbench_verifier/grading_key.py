"""Minimal grading keys: the evaluator files that local scoring reads.

A public evaluator archive holds the participant role, the unchanged release manifest and a grading
key in place of the evaluation role. The key manifest binds the key to that release manifest and
inventories every other evaluator-side file of the archive, so a key archive has a complete byte
census without the files that stay with the operator. An operator archive carries the same key
beside its complete evaluation role, so both grade from identical bytes.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator

from trialagentbench_verifier.io import (
    VerificationInputError,
    canonical_json_bytes,
    canonical_sha256,
    read_json_lines,
    read_json_object,
    sha256_file,
)
from trialagentbench_verifier.trialeval import TrialEvalGradingKeyRecordV1

if TYPE_CHECKING:
    from trialagentbench_verifier.assets import ReleaseAssetScope

GRADING_KEY_MANIFEST: Final = "grading_key/manifest.json"
TRIALEVAL_GRADING_KEY_RECORDS: Final = "grading_key/records.jsonl"
_SCHEMAS: Final[dict[str, Literal["trialeval", "trialdev"]]] = {
    "trialagentbench.trialeval_grading_key/v1": "trialeval",
    "trialagentbench.trialdev_grading_key/v1": "trialdev",
}
_RELEASE_SCHEMAS: Final[dict[str, Literal["trialeval", "trialdev"]]] = {
    "ariadne.trialeval.release/v1": "trialeval",
    "ariadne.trialdev.release/v1": "trialdev",
}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GradingKeyArtifactV1(_Model):
    """Byte identity of one evaluator-side file carried with a grading key."""

    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_length: int = Field(ge=0)

    @model_validator(mode="after")
    def _portable(self) -> Self:
        path = PurePosixPath(self.relative_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != self.relative_path
            or path.parts[0] not in {"evaluation", "grading_key"}
            or self.relative_path == GRADING_KEY_MANIFEST
        ):
            raise ValueError("grading-key artifacts are portable evaluation or key files other than the manifest")
        return self


class GradingKeyManifestV1(_Model):
    """Release binding, census and byte inventory of one grading key."""

    schema_id: Literal["trialagentbench.trialeval_grading_key/v1", "trialagentbench.trialdev_grading_key/v1"]
    release_id: str = Field(pattern=r"^release_[0-9a-f]{64}$")
    release_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    derivation_id: str = Field(min_length=1)
    task_count: int = Field(ge=1)
    world_count: int = Field(ge=1)
    cell_count: int | None = Field(default=None, ge=1)
    series_count: int | None = Field(default=None, ge=1)
    artifacts: tuple[GradingKeyArtifactV1, ...] = Field(min_length=1)
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def benchmark(self) -> Literal["trialeval", "trialdev"]:
        """Return the benchmark stream the key grades."""
        return _SCHEMAS[self.schema_id]

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        paths = tuple(item.relative_path for item in self.artifacts)
        if paths != tuple(sorted(set(paths))):
            raise ValueError("grading-key artifacts must be sorted and unique")
        if (self.benchmark == "trialeval") != (self.cell_count is not None and self.series_count is not None):
            raise ValueError("TrialEval keys, and only they, count cells and series")
        if self.checksum != canonical_sha256(self.model_dump(mode="json", exclude={"checksum"})):
            raise ValueError("grading-key manifest checksum mismatch")
        return self

    @classmethod
    def create(
        cls,
        *,
        schema_id: Literal["trialagentbench.trialeval_grading_key/v1", "trialagentbench.trialdev_grading_key/v1"],
        release_id: str,
        release_manifest_sha256: str,
        derivation_id: str,
        task_count: int,
        world_count: int,
        artifacts: tuple[GradingKeyArtifactV1, ...],
        cell_count: int | None = None,
        series_count: int | None = None,
    ) -> GradingKeyManifestV1:
        """Return a manifest whose checksum covers every other field."""
        unsigned: dict[str, JsonValue] = {
            "schema_id": schema_id,
            "release_id": release_id,
            "release_manifest_sha256": release_manifest_sha256,
            "derivation_id": derivation_id,
            "task_count": task_count,
            "world_count": world_count,
            "cell_count": cell_count,
            "series_count": series_count,
            "artifacts": [item.model_dump(mode="json") for item in artifacts],
        }
        return cls.model_validate({**unsigned, "checksum": canonical_sha256(unsigned)})

    def artifact(self, relative_path: str) -> GradingKeyArtifactV1:
        """Return the inventoried identity of one key file."""
        matches = tuple(item for item in self.artifacts if item.relative_path == relative_path)
        if len(matches) != 1:
            raise VerificationInputError(f"grading key does not inventory {relative_path}")
        return matches[0]


def grading_key_lines(records: Iterable[BaseModel]) -> bytes:
    """Encode key records as canonical JSON lines, one per record.

    Parameters
    ----------
    records
        Validated public key records in their released order.

    Returns
    -------
    bytes
        Each record's sorted-key JSON object followed by a newline.
    """
    return b"".join(canonical_json_bytes(record.model_dump(mode="json")) + b"\n" for record in records)


def _verify_file(root: Path, artifact: GradingKeyArtifactV1, *, verify_files: bool = True) -> Path:
    path = root / artifact.relative_path
    if (
        not path.is_file()
        or path.is_symlink()
        or not path.resolve(strict=True).is_relative_to(root.resolve(strict=True))
        or path.stat().st_size != artifact.byte_length
        or (verify_files and sha256_file(path) != artifact.sha256)
    ):
        raise VerificationInputError(f"grading-key artifact is missing or changed: {artifact.relative_path}")
    return path


def _read_manifest(root: Path) -> GradingKeyManifestV1:
    try:
        return GradingKeyManifestV1.model_validate(read_json_object(root / GRADING_KEY_MANIFEST))
    except ValidationError as error:
        raise VerificationInputError(f"grading-key manifest violates its public contract: {error}") from error


def read_trialeval_grading_key(
    root: Path, *, verify_files: bool = True
) -> tuple[GradingKeyManifestV1, tuple[TrialEvalGradingKeyRecordV1, ...]]:
    """Read and verify the canonical TrialEval key records of one release root.

    Parameters
    ----------
    root
        Extracted evaluator release containing ``grading_key/``.

    Returns
    -------
    tuple
        The key manifest and its records, sorted by task.

    Raises
    ------
    VerificationInputError
        If the records differ from their inventoried bytes, are not canonical, or disagree with
        the manifest's census.
    """
    manifest = _read_manifest(root)
    if manifest.benchmark != "trialeval":
        raise VerificationInputError("grading key does not grade TrialEval")
    path = _verify_file(root, manifest.artifact(TRIALEVAL_GRADING_KEY_RECORDS), verify_files=verify_files)
    try:
        records = tuple(TrialEvalGradingKeyRecordV1.model_validate(row) for row in read_json_lines(path))
    except ValidationError as error:
        raise VerificationInputError(f"grading-key record violates its public contract: {error}") from error
    if grading_key_lines(records) != path.read_bytes():
        raise VerificationInputError("grading-key records are not canonical JSON lines")
    task_ids = tuple(record.task_id for record in records)
    cells: dict[str, str] = {}
    for record in records:
        if cells.setdefault(record.cell_ref, record.series_ref) != record.series_ref:
            raise VerificationInputError("a grading-key cell spans more than one series")
    if (
        task_ids != tuple(sorted(set(task_ids)))
        or {record.release_id for record in records} != {manifest.release_id}
        or len(records) != manifest.task_count
        or len(cells) != manifest.cell_count
        or len(set(cells.values())) != manifest.series_count
        or len({record.base_world_id for record in records}) != manifest.world_count
    ):
        raise VerificationInputError("grading-key records differ from the manifest census")
    return manifest, records


def verify_grading_key(
    root: Path,
    *,
    scope: ReleaseAssetScope | None = None,
    verify_files: bool = True,
    verified_paths: frozenset[str] = frozenset(),
    release_manifest_sha256: str | None = None,
) -> GradingKeyManifestV1:
    """Verify a grading key against its release manifest and inventoried bytes.

    Parameters
    ----------
    root
        Extracted evaluator release containing ``grading_key/manifest.json``.
    scope
        Selected asset view; files outside it are bound to the catalog but not read.

    Returns
    -------
    GradingKeyManifestV1
        The verified manifest; its artifacts and the manifest itself complete the census.

    Raises
    ------
    VerificationInputError
        If the key belongs to another release or stream, or any inventoried file differs.
    """
    manifest = _read_manifest(root)
    release_path = root / "release_manifest.json"
    release = read_json_object(release_path)
    if (
        _RELEASE_SCHEMAS.get(str(release.get("schema_id"))) != manifest.benchmark
        or release.get("release_id") != manifest.release_id
        or release.get("task_count") != manifest.task_count
        or (release_manifest_sha256 if release_manifest_sha256 is not None else sha256_file(release_path))
        != manifest.release_manifest_sha256
    ):
        raise VerificationInputError("grading key belongs to a different release")
    for artifact in manifest.artifacts:
        if scope is None or scope.requires(artifact.relative_path, artifact.sha256, artifact.byte_length):
            _verify_file(
                root,
                artifact,
                verify_files=verify_files and scope is None and artifact.relative_path not in verified_paths,
            )
    if manifest.benchmark == "trialeval":
        read_trialeval_grading_key(root, verify_files=False)
    else:
        read_trialdev_world_strata(root, verify_files=False)
    return manifest


def grading_key_manifest_bytes(manifest: GradingKeyManifestV1) -> bytes:
    """Return the released bytes of a key manifest."""
    return (json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n").encode("utf-8")


TRIALDEV_WORLD_STRATA: Final = "grading_key/world_strata.json"


class TrialDevWorldStratumV1(_Model):
    """The ranking stratum of one TrialDev world: clinical profile, candidate count and scenario reference."""

    world_id: str = Field(pattern=r"^tdw-[0-9a-f]{4,16}$")
    clinical_profile_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    candidate_count: int = Field(ge=1)
    scenario_ref: str = Field(pattern=r"^scenario_[0-9a-f]{20}$")


class TrialDevWorldStrataV1(_Model):
    """Every world's ranking stratum, sorted by world."""

    schema_id: Literal["trialagentbench.trialdev_world_strata/v1"]
    release_id: str = Field(pattern=r"^release_[0-9a-f]{64}$")
    worlds: tuple[TrialDevWorldStratumV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _sorted_worlds(self) -> Self:
        world_ids = tuple(row.world_id for row in self.worlds)
        if world_ids != tuple(sorted(set(world_ids))):
            raise ValueError("world strata must be sorted by unique world")
        return self

    def stratum(self, world_id: str) -> TrialDevWorldStratumV1:
        """Return one world's stratum or fail on an unknown world."""
        matches = tuple(row for row in self.worlds if row.world_id == world_id)
        if len(matches) != 1:
            raise VerificationInputError(f"world strata do not contain {world_id}")
        return matches[0]


def trialdev_world_strata_bytes(strata: TrialDevWorldStrataV1) -> bytes:
    """Return the canonical released bytes of TrialDev world strata."""
    return canonical_json_bytes(strata.model_dump(mode="json")) + b"\n"


def read_trialdev_world_strata(root: Path, *, verify_files: bool = True) -> TrialDevWorldStrataV1:
    """Read and verify the canonical TrialDev world strata of one release root.

    Parameters
    ----------
    root
        Extracted evaluator release containing ``grading_key/``.

    Returns
    -------
    TrialDevWorldStrataV1
        Every released world's stratum.

    Raises
    ------
    VerificationInputError
        If the strata differ from their inventoried bytes, are not canonical, or disagree with the
        manifest's release and world census.
    """
    manifest = _read_manifest(root)
    if manifest.benchmark != "trialdev":
        raise VerificationInputError("grading key does not grade TrialDev")
    path = _verify_file(root, manifest.artifact(TRIALDEV_WORLD_STRATA), verify_files=verify_files)
    try:
        strata = TrialDevWorldStrataV1.model_validate(read_json_object(path))
    except ValidationError as error:
        raise VerificationInputError("TrialDev world strata violate their public contract") from error
    if trialdev_world_strata_bytes(strata) != path.read_bytes():
        raise VerificationInputError("TrialDev world strata are not canonical JSON")
    if strata.release_id != manifest.release_id or len(strata.worlds) != manifest.world_count:
        raise VerificationInputError("TrialDev world strata differ from the manifest census")
    return strata


__all__ = [
    "GRADING_KEY_MANIFEST",
    "TRIALDEV_WORLD_STRATA",
    "TRIALEVAL_GRADING_KEY_RECORDS",
    "GradingKeyArtifactV1",
    "GradingKeyManifestV1",
    "TrialDevWorldStrataV1",
    "TrialDevWorldStratumV1",
    "grading_key_lines",
    "grading_key_manifest_bytes",
    "read_trialdev_world_strata",
    "read_trialeval_grading_key",
    "trialdev_world_strata_bytes",
    "verify_grading_key",
]
