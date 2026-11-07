"""Authenticate explicitly unqualified archive inputs for development campaigns."""

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_verifier.contracts import CheckResultV1, VerificationReportV1
from trialagentbench_verifier.io import (
    bounded_archive_file,
    materialized_release,
    read_json_object,
    release_archive,
    sha256_file,
)
from trialagentbench_verifier.release import _report, _verify_materialized_integrity


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DevelopmentArchiveV1(_Model):
    """One externally pinned archive consumed by the native runtime or grader."""

    filename: Literal["trialeval-participant.zip", "trialeval-evaluator.zip", "trialdev-evaluator.zip"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_length: int = Field(gt=0)


class DevelopmentExecutorV1(_Model):
    """Exact isolated execution image and its portable environment identity."""

    image_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    environment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DevelopmentCampaignInputsV1(_Model):
    """Unqualified inputs, deliberately incompatible with a release-set manifest."""

    schema_id: Literal["trialagentbench.development_campaign_inputs/v1"] = (
        "trialagentbench.development_campaign_inputs/v1"
    )
    status: Literal["unqualified"] = "unqualified"
    artifacts: tuple[DevelopmentArchiveV1, ...] = Field(min_length=1, max_length=3)
    executor: DevelopmentExecutorV1

    @model_validator(mode="after")
    def validate_archives(self) -> Self:
        """Require unique archive names and both TrialEval roles when present."""
        names = {item.filename for item in self.artifacts}
        if len(names) != len(self.artifacts):
            raise ValueError("Development archive names must be unique")
        if ("trialeval-participant.zip" in names) != ("trialeval-evaluator.zip" in names):
            raise ValueError("Development TrialEval requires participant and evaluator archives")
        return self


def authenticate_development_inputs(root: Path, *, expected_manifest_sha256: str) -> DevelopmentCampaignInputsV1:
    """Authenticate development bytes without asserting scientific qualification.

    Native preparation still validates the archive's internal contracts before
    any provider request. This operation cannot produce a qualified release set.
    """
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Development input root must be a regular directory")
    path = root / "development-inputs.json"
    if sha256_file(path) != expected_manifest_sha256:
        raise ValueError("Development input manifest differs from its trusted digest")
    manifest = DevelopmentCampaignInputsV1.model_validate(read_json_object(path))
    for artifact in manifest.artifacts:
        source = bounded_archive_file(root / artifact.filename)
        if source.stat().st_size != artifact.byte_length:
            raise ValueError(f"Development archive changed: {artifact.filename}")
    return manifest


def verify_development_evaluator(
    path: Path, *, expected_sha256: str, benchmark: Literal["trialeval", "trialdev"] | None = None
) -> VerificationReportV1:
    """Return a stable, explicitly unqualified report of evaluator archive integrity.

    Qualification is deliberately not inferred. The caller must retain development
    provenance; qualified publication continues to use verify_release_set.
    """
    with (
        release_archive(path, expected_sha256=expected_sha256) as archive,
        materialized_release(archive) as root,
    ):
        stream, candidate, digest = _verify_materialized_integrity(root, require_paired_participant_identity=True)
        if benchmark is not None and stream != benchmark:
            raise ValueError("Development evaluator belongs to a different benchmark")
        return _report(
            stream=stream,
            operation="release",
            candidate_digest=candidate,
            release_digest=digest,
            input_kind="archive",
            input_sha256=sha256_file(archive),
            expected_input_sha256=expected_sha256,
            report_digests=(),
            development=True,
            checks=(
                CheckResultV1(
                    check_id="release.development_integrity",
                    scope_id=candidate,
                    status="pass",
                    observed="Declared archive bytes and role identities agree",
                    expected="Declared archive bytes and role identities agree; scientific qualification is not assessed",
                    artifact_paths=("release_manifest.json", "participant/manifest.json", "evaluation/manifest.json"),
                ),
            ),
        )
