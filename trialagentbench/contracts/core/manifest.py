"""Artifact manifest for a canonically graded TrialEval run."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from trialagentbench_verifier.scorer import CurrentScorer

from trialagentbench.io import canonical_payload_sha256
from trialagentbench.util.source_identity import require_harness_scorer_v1


class GradeManifestV1(BaseModel):
    """Content identity and denominator of one canonical TrialEval grade."""

    scorer: CurrentScorer

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trialagentbench_grade_manifest_v1"] = "trialagentbench_grade_manifest_v1"
    schema_version: Literal[1] = 1
    suite: Literal["trialeval"] = "trialeval"
    harness_version: str = Field(min_length=1)
    harness_git_sha: str | None = Field(min_length=1)
    harness_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    timestamp_utc: datetime
    input_run_dir: str = Field(min_length=1)
    output_run_dir: str = Field(min_length=1)
    input_run_sha256: str | None = Field(
        default=None, exclude_if=lambda value: value is None, pattern=r"^[0-9a-f]{64}$"
    )
    evaluator_release_sha256: str | None = Field(
        default=None, exclude_if=lambda value: value is None, pattern=r"^[0-9a-f]{64}$"
    )
    release_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    score_profile_id: Literal["oracle_outcome_components_v1"] = "oracle_outcome_components_v1"
    scorer_surface_sha256: str | None = Field(
        default=None, exclude_if=lambda value: value is None, pattern=r"^[0-9a-f]{64}$"
    )
    suite_task_count: int = Field(ge=1)
    graded_task_count: int = Field(ge=0)
    notes: tuple[str, ...] = ()
    checksum: str | None = Field(default=None, exclude_if=lambda value: value is None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        """Require a bounded denominator and bind the complete manifest."""

        require_harness_scorer_v1()
        if self.graded_task_count > self.suite_task_count:
            raise ValueError("Graded TrialEval tasks cannot exceed the scheduled denominator")
        if self.checksum is not None and self.checksum != canonical_payload_sha256(
            self.model_dump(mode="json", exclude={"checksum"})
        ):
            raise ValueError("grade manifest checksum mismatch")
        return self
