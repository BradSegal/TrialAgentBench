"""Immutable result inventory for canonical TrialDev task runs."""

from __future__ import annotations

from typing import Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench.contracts.core.config import (
    ExperimentConditionV1,
    ProviderRetryPolicyV1,
)
from trialagentbench.contracts.trialdev.release import (
    TrialDevReleaseArtifactV1,
    TrialDevStream,
)
from trialagentbench.io import canonical_payload_sha256
from trialagentbench.ports import CodeExecutionLimitsV1

TrialDevExecutionStatusV1: TypeAlias = Literal["completed", "model_noncompletion", "evaluator_error"]


class TrialDevRunInvocationV1(BaseModel):
    """Immutable execution identity required to continue one TrialDev run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["ariadne.trialdev.run_invocation/v1"] = "ariadne.trialdev.run_invocation/v1"
    release_id: str = Field(pattern=r"^release_[0-9a-f]{64}$")
    release_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    release_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    harness_source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    scheduled_task_ids: tuple[str, ...] = Field(min_length=1)
    scheduled_task_streams: dict[str, TrialDevStream]
    model: str = Field(min_length=1)
    provider: Literal["openai", "openai_responses", "openrouter", "offline_control"]
    openrouter_provider: str | None = None
    experiment_condition: ExperimentConditionV1
    decoding_seed: int | None = Field(default=None, ge=0)
    retry: ProviderRetryPolicyV1 = Field(default_factory=ProviderRetryPolicyV1)
    send_temperature: bool
    maximum_completion_tokens: int = Field(ge=1)
    maximum_context_characters: int = Field(ge=1)
    watchdog_seconds: int = Field(ge=1)
    request_timeout_seconds: float = Field(gt=0)
    reported_cost_stop_usd: float | None = Field(default=None, gt=0)
    workers: int | None = Field(default=None, ge=1)
    executor_image: str = Field(min_length=1)
    executor_limits: CodeExecutionLimitsV1
    checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_invocation(self) -> Self:
        """Require exact provider routing, unique tasks, and bound content."""

        if len(self.scheduled_task_ids) != len(set(self.scheduled_task_ids)):
            raise ValueError("TrialDev scheduled task identifiers must be unique")
        if set(self.scheduled_task_streams) != set(self.scheduled_task_ids):
            raise ValueError("TrialDev scheduled task streams must cover the task denominator")
        if (self.provider == "openrouter") != (self.openrouter_provider is not None):
            raise ValueError("TrialDev analysis provider routing is incomplete")
        capability = self.experiment_condition.provider_capability
        if capability is not None:
            capability.validate_route(
                provider=self.provider, model=self.model, upstream_provider=self.openrouter_provider
            )
        expected = canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}, exclude_none=True))
        if self.checksum is not None and self.checksum != expected:
            raise ValueError("TrialDev run invocation checksum mismatch")
        object.__setattr__(self, "checksum", expected)
        return self


class TrialDevRunResultV1(BaseModel):
    """Terminal status and custody digest for one scheduled task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(pattern=r"^td-[0-9a-f]{12}$")
    stream: TrialDevStream
    execution_status: TrialDevExecutionStatusV1
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")


class TrialDevReportedCostV1(BaseModel):
    """Provider-reported spend observed under an optional run stop."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    threshold_usd: float = Field(gt=0)
    observed_usd: float = Field(ge=0)
    response_count: int = Field(ge=0)
    threshold_reached: bool
    cost_complete: bool
    overshoot_usd: float = Field(ge=0)


class TrialDevRunIndexV1(BaseModel):
    """Complete portable result inventory for one TrialDev invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["ariadne.trialdev.run_index/v1"] = "ariadne.trialdev.run_index/v1"
    release_id: str = Field(pattern=r"^release_[0-9a-f]{64}$")
    scheduled_task_ids: tuple[str, ...] = Field(min_length=1)
    model: str = Field(min_length=1)
    invocation_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    results: tuple[TrialDevRunResultV1, ...] = Field(min_length=1)
    reported_cost: TrialDevReportedCostV1 | None = None
    checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_index(self) -> Self:
        """Require complete ordered coverage and bind the result inventory."""

        if tuple(result.task_id for result in self.results) != self.scheduled_task_ids:
            raise ValueError("TrialDev results must cover the scheduled tasks in order")
        expected = canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))
        if self.checksum is not None and self.checksum != expected:
            raise ValueError("TrialDev run-index checksum mismatch")
        object.__setattr__(self, "checksum", expected)
        return self


class TrialDevTaskRunV1(BaseModel):
    """Numerical analysis or authored-programme custody for one task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["ariadne.trialdev.task_run/v1"] = "ariadne.trialdev.task_run/v1"
    release_id: str = Field(pattern=r"^release_[0-9a-f]{64}$")
    release_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_id: str = Field(pattern=r"^td-[0-9a-f]{12}$")
    stream: TrialDevStream
    execution_status: TrialDevExecutionStatusV1
    # These members must be empty and remain serialized to preserve retained-run checksums.
    report_relative_paths: tuple[()]
    ingress_relative_paths: tuple[()]
    grade_relative_paths: tuple[()]
    artifacts: tuple[TrialDevReleaseArtifactV1, ...] = ()
    programme_relative_path: Literal["programme_run.json"] | None = None
    analysis_relative_path: Literal["analysis_run.json"] | None = None
    final_state: None = None
    error: str | None = None
    checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        """Require aligned custody and bind the exact result inventory."""

        artifact_paths = tuple(artifact.relative_path for artifact in self.artifacts)
        if artifact_paths != tuple(sorted(set(artifact_paths))):
            raise ValueError("TrialDev task-run artifacts must be sorted and unique")
        numerical_paths = tuple(
            path for path in (self.programme_relative_path, self.analysis_relative_path) if path is not None
        )
        if len(numerical_paths) > 1:
            raise ValueError("A task cannot contain both standalone and programme results")
        if self.analysis_relative_path is not None and self.stream is not TrialDevStream.EVIDENCE_ANALYSIS:
            raise ValueError("Standalone numerical results require the evidence-analysis stream")
        if self.programme_relative_path is not None and self.stream is not TrialDevStream.PORTFOLIO_MANAGEMENT:
            raise ValueError("Authored programme results require the portfolio-management stream")
        if not set(numerical_paths).issubset(artifact_paths):
            raise ValueError("TrialDev task-run artifacts do not bind every declared result")
        if self.execution_status == "completed":
            if self.error is not None or not numerical_paths:
                raise ValueError("Completed TrialDev tasks require numerical results and no error")
        elif self.error is None:
            raise ValueError("Incomplete TrialDev tasks require an error")
        payload = self.model_dump(mode="json", exclude={"checksum"})
        if self.programme_relative_path is None:
            payload.pop("programme_relative_path")
        if self.analysis_relative_path is None:
            payload.pop("analysis_relative_path")
        expected = canonical_payload_sha256(payload)
        if self.checksum is not None and self.checksum != expected:
            raise ValueError("TrialDev task-run checksum mismatch")
        object.__setattr__(self, "checksum", expected)
        return self


__all__ = [
    "TrialDevExecutionStatusV1",
    "TrialDevReportedCostV1",
    "TrialDevRunIndexV1",
    "TrialDevRunInvocationV1",
    "TrialDevRunResultV1",
    "TrialDevTaskRunV1",
]
