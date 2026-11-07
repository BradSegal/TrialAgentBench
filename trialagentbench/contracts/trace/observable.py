"""Contracts for observable benchmark events and evidence-linked features."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.types import JsonValue

from trialagentbench.contracts.core.trialeval_conditions import ProcedureAssistanceV1
from trialagentbench.contracts.core.trialeval_factors import (
    TrialEvalAnalysisSpecificationV1,
    TrialEvalContextConfigurationV1,
    TrialEvalDataPreparationV1,
    TrialEvalEvidenceFactorsV1,
)

BenchmarkNameV1 = Literal["trialeval", "trialdev"]
BenchmarkRuntimeFailureTypeV1 = Literal[
    "turn_limit_no_submission",
    "turn_limit_no_terminal_decision",
    "submission_limit",
    "timeout",
    "provider_error",
    "provider_unavailable",
    "contract_error",
    "participant_session_terminated",
    "runtime_error",
    "evaluator_error",
    "pre_run_identity_error",
    "cancelled",
]
RuntimeEventTypeV1 = Literal[
    "step_started",
    "prompt",
    "assistant_message",
    "tool_call",
    "tool_result",
    "code_execution",
    "file_inspection",
    "submission",
    "step_terminal",
]
TraceCoverageStatusV1 = Literal[
    "full_conversation_trace",
    "submission_only_trace",
    "scorer_only_trace",
    "missing_trace",
]
TrialEvalSubmissionTransportV1 = Literal["direct", "file", "not_observed"]
TraceEndpointStateV1 = Literal[
    "valid",
    "failed",
    "evaluator_error",
    "not_reached_after_stop",
    "not_attempted_noncompletion",
    "score_export_absent",
    "not_scored_by_design",
    "submission_absent",
    "submission_present_score_absent",
    "score_present_submission_absent",
    "source_inconsistent_requires_adjudication",
    "not_scoreable_trace_only",
]
EvidenceCategoryV1 = Literal[
    "observational_extract",
    "baseline_covariates",
    "time_to_event",
    "safety_events",
    "longitudinal_markers",
    "randomization_or_treatment_assignment",
    "missingness_or_data_quality",
    "public_catalog",
    "data_dictionary_or_schema",
    "protocol_or_program_contract",
    "trial_design_request",
    "trial_population_table",
    "trial_output_summary",
    "analysis_or_submission_workfile",
    "cost_or_budget",
    "phase_state_or_prior_decision",
    "simulator_output_public_summary",
    "scratch_or_diagnostic_file",
    "scratch_schema_dump",
    "scratch_summary_table",
    "scratch_model_result",
    "scratch_survival_result",
    "scratch_ci_or_uncertainty_result",
    "scratch_required_fields_or_contract_copy",
    "unclassified_public_scratch",
    "absolute_or_run_internal_path",
    "shell_literal_or_pseudo_path",
    "protected_reference_or_grader_artifact",
]


class ModelActionTraceEventV1(BaseModel):
    """One observable model or harness action in a benchmark run."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["ariadne.action_trace_event/v1"] = "ariadne.action_trace_event/v1"
    event_id: str = Field(min_length=1)
    timestamp: datetime
    benchmark: BenchmarkNameV1
    model_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    task_id: str | None = None
    assignment_id: str | None = None
    program_id: str | None = None
    scenario_id: str | None = None
    objective_id: str | None = None
    phase_id: str | None = None
    step_id: str | None = None
    turn_index: int | None = None
    event_index: int = Field(ge=0)
    event_type: Literal[
        "prompt",
        "assistant_message",
        "tool_call",
        "tool_result",
        "code_execution",
        "file_inspection",
        "submission",
        "grade_link",
        "state_transition",
        "missing_artifact",
    ]
    source_path: str
    source_artifact_path: str = Field(min_length=1)
    source_payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tool_call_id: str | None = None
    tool_name: str | None = None
    file_accessed: str | None = None
    elapsed_sec: float | None = None
    execution_status: Literal["success", "execution_error", "timeout", "session_terminated"] | None = None
    execution_language: Literal["python", "r"] | None = None
    cpu_seconds: float | None = Field(default=None, ge=0.0)
    session_peak_memory_bytes: int | None = Field(default=None, ge=0)
    session_peak_processes: int | None = Field(default=None, ge=0)
    output_truncated: bool | None = None
    content_chars: int | None = None
    status: Literal["observed", "missing", "invalid", "not_available"]
    failure_code: str | None = None

    @field_validator("timestamp")
    @classmethod
    def _require_utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
            raise ValueError("Trace event timestamps must be timezone-aware UTC values.")
        return value

    @model_validator(mode="after")
    def _require_trialdev_context(self) -> ModelActionTraceEventV1:
        if self.benchmark == "trialdev" and not all((self.program_id, self.scenario_id, self.objective_id)):
            raise ValueError("TrialDev action events require program_id, scenario_id, and objective_id.")
        return self


class BenchmarkRuntimeTraceEventV1(BaseModel):
    """One runner-native event linked to an immutable conversation payload."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["ariadne.runtime_trace_event/v1"] = "ariadne.runtime_trace_event/v1"
    event_id: str = Field(min_length=1)
    timestamp: datetime
    source_artifact_path: str = Field(min_length=1)
    source_payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark: BenchmarkNameV1
    event_index: int = Field(ge=0)
    task_id: str | None = Field(default=None, min_length=1)
    program_id: str | None = Field(default=None, min_length=1)
    scenario_id: str | None = Field(default=None, min_length=1)
    objective_id: str | None = Field(default=None, min_length=1)
    phase_id: str = Field(min_length=1)
    step_id: str = Field(min_length=1)
    event_type: RuntimeEventTypeV1
    conversation_message_index: int | None = Field(default=None, ge=0)
    tool_call_index: int | None = Field(default=None, ge=0)
    tool_call_id: str | None = None
    tool_name: str | None = None
    file_accessed: str | None = None
    status: Literal["observed", "invalid"] = "observed"
    execution_status: Literal["success", "execution_error", "timeout", "session_terminated"] | None = None
    execution_language: Literal["python", "r"] | None = None
    elapsed_seconds: float | None = Field(default=None, ge=0.0)
    cpu_seconds: float | None = Field(default=None, ge=0.0)
    session_peak_memory_bytes: int | None = Field(default=None, ge=0)
    session_peak_processes: int | None = Field(default=None, ge=0)
    output_truncated: bool | None = None
    terminal_status: Literal["completed", "failed"] | None = None
    failure_type: BenchmarkRuntimeFailureTypeV1 | None = None

    @field_validator("timestamp")
    @classmethod
    def _require_utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
            raise ValueError("Runtime event timestamps must be timezone-aware UTC values.")
        return value

    @model_validator(mode="after")
    def _validate_event(self) -> BenchmarkRuntimeTraceEventV1:
        execution_event = self.event_type in {"code_execution", "file_inspection", "tool_result"}
        if self.execution_status is not None and not execution_event:
            raise ValueError("Execution status is only valid for execution and tool-result events.")
        execution_observations = (
            self.execution_language,
            self.elapsed_seconds,
            self.cpu_seconds,
            self.session_peak_memory_bytes,
            self.session_peak_processes,
            self.output_truncated,
        )
        if any(observation is not None for observation in execution_observations) and self.execution_status is None:
            raise ValueError("Execution resource observations require an execution status.")
        if self.benchmark == "trialdev" and not all((self.program_id, self.scenario_id, self.objective_id)):
            raise ValueError("TrialDev runtime events require program_id, scenario_id, and objective_id.")
        if self.benchmark == "trialeval" and not self.task_id:
            raise ValueError("TrialEval runtime events require task_id.")
        if self.event_type != "step_terminal":
            if self.terminal_status is not None or self.failure_type is not None:
                raise ValueError("Terminal status is only valid for step_terminal events.")
            return self
        if self.conversation_message_index is not None:
            raise ValueError("step_terminal must not reference a conversation message.")
        if self.terminal_status is None:
            raise ValueError("step_terminal requires terminal_status.")
        if self.terminal_status == "failed" and not self.failure_type:
            raise ValueError("Failed step_terminal events require failure_type.")
        if self.terminal_status == "completed" and self.failure_type is not None:
            raise ValueError("Completed step_terminal events cannot contain failure_type.")
        return self


def is_successful_file_read(event: BenchmarkRuntimeTraceEventV1 | ModelActionTraceEventV1) -> bool:
    """Return whether an event is a file read that succeeded (traces before outcomes were kept count).

    A read rejected as an invalid tool call never executed, so it carries no execution status; its ``invalid``
    status excludes it.
    """
    return (
        event.event_type == "file_inspection"
        and event.status == "observed"
        and event.execution_status in {None, "success"}
    )


def runtime_event_source_payload_v1(
    *,
    benchmark: BenchmarkNameV1,
    task_id: str | None,
    program_id: str | None,
    scenario_id: str | None,
    objective_id: str | None,
    phase_id: str,
    step_id: str,
    event_type: RuntimeEventTypeV1,
    terminal_status: Literal["completed", "failed"] | None,
    failure_type: BenchmarkRuntimeFailureTypeV1 | None,
    conversation_message: JsonValue | None,
) -> JsonValue:
    """Return the immutable payload hashed by one runner-native event."""

    if conversation_message is not None:
        return conversation_message
    if event_type not in {"step_started", "step_terminal"}:
        raise ValueError("Conversation-linked runtime events require their exact indexed message payload.")
    return {
        "benchmark": benchmark,
        "task_id": task_id,
        "program_id": program_id,
        "scenario_id": scenario_id,
        "objective_id": objective_id,
        "phase_id": phase_id,
        "step_id": step_id,
        "event_type": event_type,
        "terminal_status": terminal_status,
        "failure_type": failure_type,
    }


class TraceFeatureRowV1(BaseModel):
    """One benchmark unit summarized as observable action features."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["ariadne.trace_feature_row/v1"] = "ariadne.trace_feature_row/v1"
    benchmark: BenchmarkNameV1
    model_id: str
    run_id: str
    task_id: str | None = None
    assignment_id: str | None = None
    program_id: str | None = None
    scenario_id: str | None = None
    objective_id: str | None = None
    phase_id: str | None = None
    trace_coverage_status: TraceCoverageStatusV1
    inspected_public_data: bool
    executed_code: bool
    submitted_answer: bool = False
    submission_transport: TrialEvalSubmissionTransportV1 | None = None
    context_tier: TrialEvalContextConfigurationV1 | None = None
    data_preparation: TrialEvalDataPreparationV1 | None = None
    analysis_specification: TrialEvalAnalysisSpecificationV1 | None = None
    procedure_assistance: ProcedureAssistanceV1
    score_link_id: str | None = None
    endpoint_valid: bool | None = None
    endpoint_state: TraceEndpointStateV1 = "not_scoreable_trace_only"
    submission_artifact_paths: tuple[str, ...] = ()
    execution_status: str | None = None
    asset_count: int | None = Field(default=None, ge=2)
    observational_loss: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    development_value: float | None = None
    probability_exceeding_safety_threshold: float | None = Field(default=None, ge=0, le=1)
    ranking_source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _require_trialdev_context(self) -> TraceFeatureRowV1:
        if self.benchmark == "trialdev" and not all((self.program_id, self.scenario_id, self.objective_id)):
            raise ValueError("TrialDev trace features require program_id, scenario_id, and objective_id.")
        if self.benchmark == "trialdev" and any(
            value is not None for value in (self.context_tier, self.data_preparation, self.analysis_specification)
        ):
            raise ValueError("TrialDev trace features cannot carry TrialEval evidence factors.")
        if self.benchmark == "trialeval":
            TrialEvalEvidenceFactorsV1(
                context_configuration=self.context_tier,
                data_preparation=self.data_preparation,
                analysis_specification=self.analysis_specification,
            )
        return self


class TrialEvalTraceInputV1(BaseModel):
    """One typed participant output before trace features are derived."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["ariadne.trialeval_trace_input/v1"] = "ariadne.trialeval_trace_input/v1"
    run_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    assignment_id: str | None = Field(default=None, min_length=1)
    context_tier: TrialEvalContextConfigurationV1
    data_preparation: TrialEvalDataPreparationV1
    analysis_specification: TrialEvalAnalysisSpecificationV1
    procedure_assistance: ProcedureAssistanceV1
    submission_transport: TrialEvalSubmissionTransportV1
    source_path: str = Field(min_length=1)
    narrative_report: str | None = None

    @model_validator(mode="after")
    def _evidence_factors_are_consistent(self) -> TrialEvalTraceInputV1:
        TrialEvalEvidenceFactorsV1(
            context_configuration=self.context_tier,
            data_preparation=self.data_preparation,
            analysis_specification=self.analysis_specification,
        )
        return self

    @property
    def answer_present(self) -> bool:
        """Return whether the participant supplied the declared output form."""

        return bool(self.narrative_report and self.narrative_report.strip())


class EvidenceUseRowV1(BaseModel):
    """One evidence category accepted or rejected for an observable unit."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["ariadne.evidence_use_row/v1"] = "ariadne.evidence_use_row/v1"
    benchmark: BenchmarkNameV1
    model_id: str
    run_id: str
    task_id: str | None = None
    assignment_id: str | None = None
    program_id: str | None = None
    phase_id: str | None = None
    evidence_category: EvidenceCategoryV1
    source: Literal["tool_call", "code_path", "structured_field", "text_citation", "validator"]
    artifact_path: str
    participant_facing: bool
    leakage_violation: bool = False


class TrialDevDecisionRowV1(BaseModel):
    """One observed submission or committed decision; structure never implies scientific success."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_id: Literal["trialagentbench.trialdev_decision_row/v1"] = "trialagentbench.trialdev_decision_row/v1"
    model_id: str
    run_id: str
    condition_id: str
    task_id: str
    world_id: str
    stream: Literal["evidence_analysis", "portfolio_management"]
    asset_count: int | None = Field(default=None, ge=2)
    decision_index: int = Field(ge=1)
    event_index: int | None = Field(default=None, ge=0)
    submission_status: Literal["rejected", "accepted", "committed", "graded", "ingress_without_grade"]
    evidence_cutoff: str | None = None
    action: str | None = None
    selected_asset_id: str | None = None
    active_asset_id: str | None = None
    study_id: str | None = None
    evidence_paths: tuple[str, ...] = ()
    released_evidence_paths: tuple[str, ...] = ()
    day_before: float | None = Field(default=None, ge=0)
    day_after: float | None = Field(default=None, ge=0)
    budget_before: float | None = Field(default=None, ge=0)
    budget_after: float | None = Field(default=None, ge=0)
    incurred_cost: float | None = Field(default=None, ge=0)
    elapsed_days: float | None = Field(default=None, ge=0)
    terminal_reason: str | None = None
    structurally_valid: bool | None = None
    observational_loss: float | None = Field(default=None, ge=0)
    source_relative_path: str
    source_binding: Literal["file", "event_payload"] = "file"
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ingress_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    score_link_id: str


ObservableFailureTypeV1 = Literal[
    "missing_output",
    "endpoint_invalid_or_unusable",
    "hidden_or_grader_access",
    "model_noncompletion",
    "evaluator_failure",
    "none_observed",
]


class FailureCascadeRowV1(BaseModel):
    """First observable failure and downstream outcome for one unit."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["ariadne.failure_cascade_row/v1"] = "ariadne.failure_cascade_row/v1"
    benchmark: BenchmarkNameV1
    model_id: str
    run_id: str
    task_id: str | None = None
    assignment_id: str | None = None
    program_id: str | None = None
    first_failure_phase: str | None = None
    first_failure_type: ObservableFailureTypeV1
    downstream_endpoint_failed: bool
    score_link_id: str | None = None


__all__ = [
    "is_successful_file_read",
    "BenchmarkNameV1",
    "BenchmarkRuntimeFailureTypeV1",
    "BenchmarkRuntimeTraceEventV1",
    "EvidenceCategoryV1",
    "EvidenceUseRowV1",
    "FailureCascadeRowV1",
    "ObservableFailureTypeV1",
    "ModelActionTraceEventV1",
    "TraceEndpointStateV1",
    "RuntimeEventTypeV1",
    "TraceFeatureRowV1",
    "TrialEvalTraceInputV1",
    "runtime_event_source_payload_v1",
]
