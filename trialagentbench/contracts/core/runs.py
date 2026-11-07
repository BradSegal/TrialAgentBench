"""Run- and result-artifact contracts (TrialEval/TrialDev harness outputs)."""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator
from trialagentbench_verifier.execution_contracts import (
    ExecutorEnvironmentV1 as ExecutorEnvironmentV1,
)
from trialagentbench_verifier.execution_contracts import (
    ExecutorPackageV1 as ExecutorPackageV1,
)
from trialagentbench_verifier.execution_contracts import (
    ReportingInterfaceV1,
)
from trialagentbench_verifier.run_contracts import ProviderUnavailabilityReceiptV1

from trialagentbench.contracts.core.config import (
    DecodingConfigV1,
    ExperimentConditionV1,
    ProviderModelCapabilityV1,
    ProviderRetryPolicyV1,
    RoutingConfigV1,
    TrialEvalAgentStatusV1,
)
from trialagentbench.contracts.core.trialeval_conditions import (
    ProcedureAssistanceV1,
    procedure_assistance_exposure_v1,
)
from trialagentbench.contracts.core.trialeval_factors import (
    TrialEvalAnalysisSpecificationV1,
    TrialEvalEvidenceFactorsV1,
)
from trialagentbench.contracts.scoring.trialeval_scores import TrialEvalItemScoresV1
from trialagentbench.contracts.trace.observable import BenchmarkRuntimeFailureTypeV1
from trialagentbench.io.checksums import canonical_payload_sha256


class _AuthoritativeRunConfigV1(BaseModel):
    """Base for persisted run configs with a required canonical self-identity."""

    run_identity_sha256: str = Field(..., min_length=64, max_length=64)
    _identity_excluded_fields: ClassVar[set[str]]

    @classmethod
    def create(cls, **data: object) -> Self:
        """Normalize fields, compute the authoritative identity, and revalidate."""

        if "run_identity_sha256" in data:
            raise ValueError("Run identity is computed by the authoritative config factory.")
        provisional = cls.model_validate(
            {**data, "run_identity_sha256": "0" * 64},
            context={"skip_run_identity_validation": True},
        )
        payload = provisional.model_dump(
            mode="json",
            exclude=cls._identity_excluded_fields | {"run_identity_sha256"},
        )
        return cls.model_validate(
            {
                **provisional.model_dump(mode="python"),
                "run_identity_sha256": canonical_payload_sha256(payload),
            }
        )

    @model_validator(mode="after")
    def validate_authoritative_identity(self, info: ValidationInfo) -> Self:
        """Reject persisted configs whose identity does not match their content."""

        if info.context and info.context.get("skip_run_identity_validation") is True:
            return self
        payload = self.model_dump(
            mode="json",
            exclude=self._identity_excluded_fields | {"run_identity_sha256"},
        )
        if self.run_identity_sha256 != canonical_payload_sha256(payload):
            raise ValueError("Run identity does not match its authoritative configuration.")
        return self


class TrialEvalRunConfigV1(_AuthoritativeRunConfigV1):
    """TrialEval live-run configuration persisted to `run_config.json`."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench_trialeval_run_config_v1"] = "trialagentbench_trialeval_run_config_v1"
    schema_version: Literal[1] = 1
    timestamp_utc: datetime

    model: str
    budget_profile_id: Literal["trialeval_release_default_v1"] = "trialeval_release_default_v1"
    item_watchdog_seconds: int = Field(..., ge=1)
    maximum_context_characters: int = Field(default=120_000, ge=1)

    participant_archive: str
    participant_archive_sha256: str = Field(..., min_length=64, max_length=64)
    participant_manifest_checksum: str = Field(..., min_length=64, max_length=64)
    prompt_set_sha256: str = Field(..., min_length=64, max_length=64)
    scorer_source_sha256: str = Field(..., min_length=64, max_length=64)
    harness_source_sha256: str = Field(..., min_length=64, max_length=64)
    experiment_condition: ExperimentConditionV1
    task_ids: list[str] = Field(..., min_length=1)
    task_evidence_factors: dict[str, TrialEvalEvidenceFactorsV1]

    decoding: DecodingConfigV1
    routing: RoutingConfigV1
    retry: ProviderRetryPolicyV1 = Field(default_factory=ProviderRetryPolicyV1)
    executor: ExecutorEnvironmentV1

    workers: int = Field(default=1, ge=1)
    n_items: int = Field(..., ge=1)

    @model_validator(mode="after")
    def validate_provider_route(self) -> Self:
        """Keep both direct and CLI run creation bound to the same provider source."""

        capability = self.experiment_condition.provider_capability
        if capability is not None:
            capability.validate_route(
                provider=self.routing.provider, model=self.model, upstream_provider=self.routing.openrouter_provider
            )
        return self

    _identity_excluded_fields: ClassVar[set[str]] = {
        "timestamp_utc",
        "participant_archive",
        "workers",
    }

    @model_validator(mode="after")
    def validate_task_denominator(self) -> TrialEvalRunConfigV1:
        """Require a unique task denominator consistent with the run size."""

        condition = self.experiment_condition
        procedure_assistance_exposure_v1(
            suite="trialeval",
            procedure_assistance=condition.procedure_assistance,
        )
        if condition.tool_choice != "auto":
            raise ValueError("TrialEval owns tool selection within the agent loop.")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("task_ids must be unique.")
        if any(not task_id.strip() for task_id in self.task_ids):
            raise ValueError("task_ids must not contain empty values.")
        if self.n_items != len(self.task_ids):
            raise ValueError("n_items must equal the number of task_ids.")
        if set(self.task_evidence_factors) != set(self.task_ids):
            raise ValueError("task_evidence_factors must contain exactly one entry for every task_id.")
        return self


class RunCoverageV1(BaseModel):
    """Prospective immutable schedule and atomically maintained completion set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["ariadne.run_coverage/v1"] = "ariadne.run_coverage/v1"
    run_identity_sha256: str = Field(..., min_length=64, max_length=64)
    schedule_sha256: str = Field(..., min_length=64, max_length=64)
    unit_ids: tuple[str, ...] = Field(..., min_length=1)
    completed_unit_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_denominator(self) -> RunCoverageV1:
        """Reject duplicate, reordered, or out-of-schedule completion identities."""

        if len(set(self.unit_ids)) != len(self.unit_ids):
            raise ValueError("Run coverage unit_ids must be unique.")
        if len(set(self.completed_unit_ids)) != len(self.completed_unit_ids):
            raise ValueError("Run coverage completed_unit_ids must be unique.")
        expected_order = tuple(unit_id for unit_id in self.unit_ids if unit_id in self.completed_unit_ids)
        if self.completed_unit_ids != expected_order:
            raise ValueError("Completed units must be an ordered subset of the prospective schedule.")
        return self


# Failure diagnostics keep the head of the provider's message and response body.
PROVIDER_ERROR_DETAIL_CHARACTERS = 2000


class ProviderRequestEventV1(BaseModel):
    """One append-only lifecycle event for a provider request."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["ariadne.provider_request_event/v1"] = "ariadne.provider_request_event/v1"
    request_id: str = Field(..., min_length=1)
    status: Literal["started", "succeeded", "failed"]
    benchmark: Literal["trialeval", "trialdev"]
    unit_id: str = Field(..., min_length=1)
    phase_id: str = Field(..., min_length=1)
    step_id: str = Field(..., min_length=1)
    turn_index: int = Field(..., ge=1)
    elapsed_seconds: float | None = Field(..., ge=0.0)
    requested_model: str = Field(..., min_length=1)
    provider_route: str = Field(..., min_length=1)
    context_management: Literal["unreported", "provider_managed", "local_complete", "local_archived"] = "unreported"
    source_context_messages: int = Field(default=0, ge=0)
    transmitted_context_messages: int = Field(default=0, ge=0)
    source_context_characters: int = Field(default=0, ge=0)
    transmitted_context_characters: int = Field(default=0, ge=0)
    requested_completion_tokens: int = Field(default=0, ge=0)
    provider_capability: ProviderModelCapabilityV1 | None = None
    response_received: bool = False
    response_id: str | None = None
    returned_model: str | None = None
    upstream_provider: str | None = None
    finish_reason: str | None = None
    completion_limit_reached: bool | None = None
    created_unix: int | None = Field(default=None, ge=0)
    usage_status: Literal["not_applicable", "reported", "not_reported", "invalid"]
    usage_error: str | None = Field(default=None, min_length=1, max_length=PROVIDER_ERROR_DETAIL_CHARACTERS)
    prompt_tokens: int = Field(..., ge=0)
    completion_tokens: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)
    compaction_requests: int = Field(default=0, ge=0)
    compaction_input_tokens: int = Field(default=0, ge=0)
    compaction_output_tokens: int = Field(default=0, ge=0)
    cached_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    reasoning: str | None = None
    model_outcome: Literal["refusal", "malformed_response", "empty_response"] | None = None
    reported_cost_usd: float | None = Field(default=None, ge=0.0)
    request_attempts: int = Field(..., ge=0)
    transient_failure_count: int = Field(..., ge=0)
    backoff_seconds: float = Field(..., ge=0.0)
    failure_type: (
        Literal[
            "timeout",
            "provider_error",
            "cancelled",
        ]
        | None
    ) = None
    exception_type: str | None = Field(default=None, min_length=1)
    http_status_code: int | None = Field(default=None, ge=100, le=599)
    error_detail: str | None = Field(default=None, min_length=1, max_length=PROVIDER_ERROR_DETAIL_CHARACTERS)
    unavailability_policy: Literal["defer_unanswered_v1"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    scientific_clock_excluded_seconds: float = Field(
        default=0.0, ge=0.0, allow_inf_nan=False, exclude_if=lambda value: value == 0.0
    )
    retry_after_seconds: float | None = Field(
        default=None, ge=0.0, allow_inf_nan=False, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def validate_lifecycle_state(self) -> ProviderRequestEventV1:
        """Require complete, status-consistent route and usage telemetry."""

        if self.unavailability_policy is not None:
            if self.status != "failed" or self.response_received or self.model_outcome is not None:
                raise ValueError("Provider deferral requires a failed request without a received model response.")
            if self.elapsed_seconds is None or self.scientific_clock_excluded_seconds > self.elapsed_seconds:
                raise ValueError("Excluded scientific time cannot exceed the observed failed request interval.")
        elif self.scientific_clock_excluded_seconds or self.retry_after_seconds is not None:
            raise ValueError("Availability clock accounting requires an explicit unavailability policy.")

        if self.usage_status == "invalid":
            if self.status != "succeeded" or self.usage_error is None:
                raise ValueError("Invalid usage requires a successful response and a usage diagnostic.")
            if any(
                (
                    self.prompt_tokens,
                    self.completion_tokens,
                    self.total_tokens,
                    self.compaction_requests,
                    self.compaction_input_tokens,
                    self.compaction_output_tokens,
                    self.cached_tokens,
                    self.reasoning_tokens,
                )
            ):
                raise ValueError("Invalid usage must use zero placeholders; original counts belong in the raw log.")
        elif self.usage_error is not None:
            raise ValueError("Only invalid usage can contain a usage diagnostic.")
        context_counts = (
            self.source_context_messages,
            self.transmitted_context_messages,
            self.source_context_characters,
            self.transmitted_context_characters,
            self.requested_completion_tokens,
        )
        if self.context_management == "unreported":
            if any(context_counts):
                raise ValueError("Unreported context accounting requires explicit zero counts.")
        else:
            if min(context_counts) < 1:
                raise ValueError("Reported context accounting requires positive counts and a completion limit.")
            if self.transmitted_context_messages > self.source_context_messages + 1:
                raise ValueError("Transmitted context can add at most one explicit archive reference.")
            if self.context_management in {"provider_managed", "local_complete"} and (
                self.source_context_messages != self.transmitted_context_messages
                or self.source_context_characters != self.transmitted_context_characters
            ):
                raise ValueError("Complete context accounting must preserve the source context exactly.")
            if self.context_management == "local_archived" and (
                self.source_context_messages == self.transmitted_context_messages
                and self.source_context_characters == self.transmitted_context_characters
            ):
                raise ValueError("Archived context accounting must record a changed transmitted context.")
        if self.model_outcome is not None and self.status != "succeeded":
            raise ValueError("A model outcome requires a retained successful provider response.")
        if self.status == "started":
            if self.response_received:
                raise ValueError("A started request cannot contain a received response.")
            if self.elapsed_seconds is not None or self.request_attempts != 0:
                raise ValueError("Started provider requests cannot contain terminal timing or attempts.")
            if self.usage_status != "not_applicable" or any(
                (
                    self.prompt_tokens,
                    self.completion_tokens,
                    self.total_tokens,
                    self.compaction_requests,
                    self.compaction_input_tokens,
                    self.compaction_output_tokens,
                    self.cached_tokens,
                    self.reasoning_tokens,
                )
            ):
                raise ValueError("Started provider requests require explicit non-applicable zero usage.")
            if self.failure_type is not None:
                raise ValueError("Started provider requests cannot contain a failure_type.")
            if self.completion_limit_reached is not None:
                raise ValueError("Started provider requests cannot contain a completion outcome.")
            if self.exception_type is not None or self.http_status_code is not None or self.error_detail is not None:
                raise ValueError("Started provider requests cannot contain failure diagnostics.")
            if self.reasoning is not None:
                raise ValueError("Only succeeded provider requests carry returned reasoning.")
            return self
        if self.elapsed_seconds is None or self.request_attempts < 1:
            raise ValueError("Terminal provider requests require elapsed_seconds and request_attempts.")
        if self.status == "failed":
            if self.failure_type is None:
                raise ValueError("Failed provider requests require a typed failure_type.")
            if self.usage_status != "not_applicable" or any(
                (
                    self.prompt_tokens,
                    self.completion_tokens,
                    self.total_tokens,
                    self.compaction_requests,
                    self.compaction_input_tokens,
                    self.compaction_output_tokens,
                    self.cached_tokens,
                    self.reasoning_tokens,
                )
            ):
                raise ValueError("Failed provider requests require explicit non-applicable zero usage.")
            if self.transient_failure_count > self.request_attempts:
                raise ValueError("Failed request transient_failure_count cannot exceed request_attempts.")
            if self.completion_limit_reached is not None:
                raise ValueError("Failed provider requests cannot contain a completion outcome.")
            if self.reasoning is not None:
                raise ValueError("Only succeeded provider requests carry returned reasoning.")
            return self
        if (
            self.failure_type is not None
            or self.exception_type is not None
            or self.http_status_code is not None
            or self.error_detail is not None
        ):
            raise ValueError("Succeeded provider requests cannot contain failure diagnostics.")
        if self.usage_status == "not_applicable":
            raise ValueError("Succeeded provider requests require reported, not_reported or invalid usage status.")
        if self.usage_status == "not_reported" and any(
            (self.prompt_tokens, self.completion_tokens, self.total_tokens)
        ):
            raise ValueError("Unreported provider usage must use explicit zero token counts.")
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("total_tokens must equal prompt_tokens + completion_tokens.")
        if self.compaction_requests == 0 and any((self.compaction_input_tokens, self.compaction_output_tokens)):
            raise ValueError("Compaction token usage requires at least one compaction request.")
        if self.compaction_input_tokens > self.prompt_tokens:
            raise ValueError("Compaction input tokens cannot exceed total prompt tokens.")
        if self.compaction_output_tokens > self.completion_tokens:
            raise ValueError("Compaction output tokens cannot exceed total completion tokens.")
        if self.cached_tokens > self.prompt_tokens or self.reasoning_tokens > self.completion_tokens:
            raise ValueError("Cached and reasoning tokens cannot exceed prompt and completion tokens.")
        if self.request_attempts != self.transient_failure_count + 1:
            raise ValueError("request_attempts must equal transient_failure_count + 1.")
        if self.finish_reason is None and self.completion_limit_reached is not None:
            raise ValueError("Completion-limit status requires a provider finish reason.")
        return self


class ProviderLogRecoveryV1(BaseModel):
    """Unobserved request endings and untouched torn bytes in one source log."""

    model_config = ConfigDict(extra="forbid")

    source_file: str
    interrupted_request_ids: tuple[str, ...] = ()
    torn_tail_bytes: int = Field(default=0, ge=0)
    torn_tail_sha256: str | None = Field(default=None, min_length=64, max_length=64)


class ProviderTelemetrySummaryV1(BaseModel):
    """Run-level provider telemetry totals derived from response records."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["ariadne.provider_telemetry_summary/v1"] = "ariadne.provider_telemetry_summary/v1"
    run_identity_sha256: str = Field(..., min_length=64, max_length=64)
    schedule_sha256: str = Field(..., min_length=64, max_length=64)
    unit_ids: tuple[str, ...] = Field(..., min_length=1)
    completed_unit_ids: tuple[str, ...]
    response_count: int = Field(..., ge=0)
    locally_archived_request_count: int = Field(default=0, ge=0)
    completion_limit_reached_count: int = Field(default=0, ge=0)
    source_context_characters: int = Field(default=0, ge=0)
    transmitted_context_characters: int = Field(default=0, ge=0)
    requested_completion_tokens: int = Field(default=0, ge=0)
    failed_request_count: int = Field(..., ge=0)
    failure_type_counts: dict[Literal["timeout", "provider_error", "cancelled"], int]
    responses_with_usage: int = Field(..., ge=0)
    responses_with_invalid_usage: int = Field(default=0, ge=0)
    responses_with_response_id: int = Field(..., ge=0)
    responses_with_reported_cost: int = Field(..., ge=0)
    prompt_tokens: int = Field(..., ge=0)
    completion_tokens: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)
    compaction_requests: int = Field(default=0, ge=0)
    compaction_input_tokens: int = Field(default=0, ge=0)
    compaction_output_tokens: int = Field(default=0, ge=0)
    reported_cost_usd: float = Field(..., ge=0.0)
    elapsed_seconds: float = Field(..., ge=0.0)
    request_attempt_count: int = Field(..., ge=0)
    transient_failure_count: int = Field(..., ge=0)
    backoff_seconds: float = Field(..., ge=0.0)
    requested_models: list[str]
    returned_models: list[str]
    upstream_providers: list[str]
    source_files: list[str]
    archived_request_count: int = Field(..., ge=0)
    archived_failed_request_count: int = Field(..., ge=0)
    archived_failure_type_counts: dict[Literal["timeout", "provider_error", "cancelled"], int]
    archived_request_attempt_count: int = Field(..., ge=0)
    archived_transient_failure_count: int = Field(..., ge=0)
    archived_backoff_seconds: float = Field(..., ge=0.0)
    archived_elapsed_seconds: float = Field(..., ge=0.0)
    archived_source_files: list[str]
    # Totals above describe observed terminal records only. Unknown charges stay
    # reserved in requests.sqlite; these records must never be treated as free.
    interrupted_request_count: int = Field(default=0, ge=0)
    archived_interrupted_request_count: int = Field(default=0, ge=0)
    log_recoveries: tuple[ProviderLogRecoveryV1, ...] = ()

    @model_validator(mode="after")
    def validate_failure_totals(self) -> ProviderTelemetrySummaryV1:
        """Require typed failure counts to agree with their totals."""

        expected_order = tuple(unit_id for unit_id in self.unit_ids if unit_id in self.completed_unit_ids)
        if self.completed_unit_ids != expected_order:
            raise ValueError("Provider telemetry completed units must be an ordered schedule subset.")
        if sum(self.failure_type_counts.values()) != self.failed_request_count:
            raise ValueError("Provider failure_type_counts do not equal failed_request_count.")
        if sum(self.archived_failure_type_counts.values()) != self.archived_failed_request_count:
            raise ValueError("Archived failure_type_counts do not equal archived_failed_request_count.")
        if self.responses_with_usage + self.responses_with_invalid_usage > self.response_count:
            raise ValueError("Reported and invalid usage counts cannot exceed successful responses.")
        return self


class TrialEvalConditionProvenanceV1(BaseModel):
    """Exact participant-facing prompt and interface provenance for one item."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    procedure_assistance: ProcedureAssistanceV1
    # Absent in 1.0 items, which used the v1 interface.
    reporting_interface: ReportingInterfaceV1 = Field(default="v1", exclude_if=lambda value: value == "v1")
    analysis_specification: TrialEvalAnalysisSpecificationV1
    analysis_surface_sha256: str = Field(..., min_length=64, max_length=64)
    max_turns: int = Field(..., ge=1)
    prompt_set_sha256: str = Field(..., min_length=64, max_length=64)
    rendered_system_prompt_sha256: str = Field(..., min_length=64, max_length=64)
    tool_schema_sha256: str = Field(..., min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_trialeval_assistance(self) -> TrialEvalConditionProvenanceV1:
        """Reject assistance levels that are defined only for TrialDev."""

        procedure_assistance_exposure_v1(
            suite="trialeval",
            procedure_assistance=self.procedure_assistance,
        )
        return self


class TrialEvalAgentOutputV1(BaseModel):
    """Minimal agent output persisted alongside scoring results."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench_trialeval_agent_output_v1"] = "trialagentbench_trialeval_agent_output_v1"
    schema_version: Literal[1] = 1
    status: TrialEvalAgentStatusV1
    failure_type: BenchmarkRuntimeFailureTypeV1 | None = None
    provider_unavailability: ProviderUnavailabilityReceiptV1 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    turns_used: int | None = Field(default=None, ge=0)
    report: str | None = None
    condition_provenance: TrialEvalConditionProvenanceV1

    @model_validator(mode="after")
    def validate_turns(self) -> TrialEvalAgentOutputV1:
        """Require a positive turn count whenever the agent returned normally."""

        if self.status != "failed" and (self.turns_used is None or self.turns_used < 1):
            raise ValueError("Completed and turn-limited TrialEval outputs require turns_used.")
        if (self.status == "success") != (self.failure_type is None):
            raise ValueError("TrialEval terminal failure provenance is required exactly for unsuccessful outputs.")
        if self.status == "success" and (self.report is None or not self.report.strip()):
            raise ValueError("Successful TrialEval outputs require a non-empty report.")
        if self.status != "success" and self.report is not None:
            raise ValueError("Unsuccessful TrialEval outputs cannot contain a report.")
        if (self.failure_type == "provider_unavailable") != (self.provider_unavailability is not None):
            raise ValueError(
                "Provider unavailability requires its retained typed request receipt, and only that failure may carry one."
            )
        if self.provider_unavailability is not None:
            ProviderRequestEventV1.model_validate(self.provider_unavailability.request_event)
        return self


class TrialEvalItemResultV1(BaseModel):
    """Per-item TrialEval artifact persisted under `items/<item_id>.json`."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench_trialeval_item_result_v1"] = "trialagentbench_trialeval_item_result_v1"
    schema_version: Literal[1] = 1
    item_id: str
    timestamp_utc: datetime

    run_config: TrialEvalRunConfigV1
    agent_output: TrialEvalAgentOutputV1
    scores: TrialEvalItemScoresV1 | None = None

    @model_validator(mode="after")
    def validate_condition_provenance(self) -> TrialEvalItemResultV1:
        """Bind the participant-facing condition to the immutable run config."""

        provenance = self.agent_output.condition_provenance
        if self.agent_output.provider_unavailability is not None:
            self.agent_output.provider_unavailability.require_run_binding(
                item_id=self.item_id,
                model=self.run_config.model,
                policy=self.run_config.retry.unavailability_policy,
            )
        expected = {
            "procedure_assistance": self.run_config.experiment_condition.procedure_assistance,
            "reporting_interface": self.run_config.experiment_condition.reporting_interface,
            "analysis_specification": self.run_config.task_evidence_factors[self.item_id].analysis_specification,
            "max_turns": self.run_config.experiment_condition.maximum_turns_per_step,
            "prompt_set_sha256": self.run_config.prompt_set_sha256,
        }
        observed = {
            "procedure_assistance": provenance.procedure_assistance,
            "reporting_interface": provenance.reporting_interface,
            "analysis_specification": provenance.analysis_specification,
            "max_turns": provenance.max_turns,
            "prompt_set_sha256": provenance.prompt_set_sha256,
        }
        if observed != expected:
            raise ValueError("TrialEval condition provenance does not match the immutable run configuration.")
        return self
