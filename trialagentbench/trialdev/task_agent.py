"""Bounded analysis agent used by canonical TrialDev tasks."""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field
from pydantic.types import JsonValue
from trialagentbench_verifier.execution_contracts import AnalysisMode, ReplayInputArtifactV1
from trialagentbench_verifier.executor import DockerCodeSession, analysis_project_files
from trialagentbench_verifier.numerical import (
    NumericalReportingContext,
    read_numerical_table,
    stage_reporting_interface,
)

from trialagentbench.contracts.core.config import ProviderModelCapabilityV1, ToolChoiceV1
from trialagentbench.contracts.trace.observable import (
    BenchmarkRuntimeFailureTypeV1,
    BenchmarkRuntimeTraceEventV1,
    runtime_event_source_payload_v1,
)
from trialagentbench.execution_policy import (
    TRIALDEV_RELEASE_BUDGET_V1,
)
from trialagentbench.io import canonical_payload_sha256, read_json_model, sha256_file, write_json_model
from trialagentbench.io.directory import staged_directory
from trialagentbench.io.json import append_jsonl_model, read_jsonl_models
from trialagentbench.ports import (
    CodeExecutionLimitsV1,
    CodeExecutionResultV1,
    CodeExecutionSession,
    LLMProvider,
    ModelResponseNoncompletion,
    ToolCall,
)
from trialagentbench.ports.llm_provider import ModelResponseOutcome
from trialagentbench.ports.tool_input import ToolInputError, parse_tool_arguments
from trialagentbench.tools.analysis import ANALYSIS_TOOLS as COMMON_ANALYSIS_TOOLS
from trialagentbench.tools.analysis import (
    analysis_instructions,
    analysis_languages,
    analysis_tools,
    check_analysis_tool_v1,
    execute_analysis_tool_v1,
)
from trialagentbench.tools.workspace import (
    WORKSPACE_TOOLS,
    handle_workspace_tool,
    read_workspace_submission_text,
)
from trialagentbench.trialdev.actions import (
    TrialDevActionRequest,
    TrialDevActionSession,
    TrialDevProgrammeCall,
    TrialDevProgrammeStatus,
    TrialDevReadout,
    TrialDevStartStudy,
    trialdev_action_names,
    trialdev_programme_status,
    trialdev_programme_tools,
    trialdev_terminal_tools,
)
from trialagentbench.trialdev.analysis_ready import (
    trialdev_baseline_fields_from_dictionary,
    write_trialdev_analysis_ready,
)
from trialagentbench.trialdev.errors import TrialDevActionRejectedError
from trialagentbench.trialdev.letters import decision_letter, design_letter, readout_letter, status_letter
from trialagentbench.trialdev.materialization import TrialDevReadoutManifest
from trialagentbench.trialdev.resources import TrialDevBudgetLedger
from trialagentbench.util.agent_runtime import AgentTurn, offered_tool_names_v1, run_agent_turn_v1
from trialagentbench.util.provider_telemetry import read_provider_request_log_v1
from trialagentbench.util.runtime_context import (
    FINAL_TURN_SUBMISSION_REMINDER,
    RuntimeDeadline,
    persist_bulky_tool_output,
    turn_budget_tag,
)
from trialagentbench.util.runtime_failures import common_runtime_failure_type_v1
from trialagentbench.util.terminal import terminal_text

DEFAULT_MAX_TURNS_PER_STEP = TRIALDEV_RELEASE_BUDGET_V1.maximum_turns
_REPORT_TOOL = "submit_analysis_report"
_REPORT_FILE_TOOL = "submit_analysis_report_file"

_ACCEPT_INCOMPLETE = (
    "Submit even if the saved analysis fails or omits a current target; the incomplete result is kept."
)
# Appended to the final-turn reminder: no analysis turn remains to repair a failing script.
_INCOMPLETE_SUBMISSION_NOTE = " If your saved script fails or omits a result, submit with accept_incomplete=true."
_FINAL_PROGRAMME_TURN = (
    "This is your final turn and a rejected call cannot be retried: call stop_development, or advance_candidate "
    "only if the brief's advance rules are met. Set accept_incomplete=true unless check_analysis has just "
    "reproduced every current target."
)

_SUBMIT_REPORT: dict[str, JsonValue] = {
    "type": "function",
    "function": {
        "name": _REPORT_TOOL,
        "description": (
            "Submit the completed report and numerical results. Input errors can be corrected and retried "
            "before the final turn. Accepted results are saved."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "content": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Completed Markdown report.",
                },
                "accept_incomplete": {
                    "type": "boolean",
                    "default": False,
                    "description": _ACCEPT_INCOMPLETE,
                },
            },
            "required": ["content"],
        },
    },
}

_SUBMIT_REPORT_FILE: dict[str, JsonValue] = {
    "type": "function",
    "function": {
        "name": _REPORT_FILE_TOOL,
        "description": (
            "Submit a Markdown report stored under scratch/ and the numerical result file. Input "
            "errors can be corrected and retried before the final turn. Accepted results are saved."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "path": {
                    "type": "string",
                    "description": "UTF-8 file path relative to scratch/.",
                },
                "accept_incomplete": {
                    "type": "boolean",
                    "default": False,
                    "description": _ACCEPT_INCOMPLETE,
                },
            },
            "required": ["path"],
        },
    },
}

ANALYSIS_TOOLS: tuple[dict[str, JsonValue], ...] = (
    *COMMON_ANALYSIS_TOOLS,
    *WORKSPACE_TOOLS,
    _SUBMIT_REPORT,
    _SUBMIT_REPORT_FILE,
)

PROGRAMME_TOOLS = frozenset(trialdev_action_names())
# Tools whose wall time the harness owns: saved-script replays, readouts and decisions.
_HARNESS_TOOLS = PROGRAMME_TOOLS | {"check_analysis"}
_CUTOFF_ACTIONS = frozenset({"readout", "stop", "advance"})

_CHANGE_TITLES = {
    "close_study": "{study_id} closed",
    "continue_recruitment": "{study_id} recruitment resumed",
    "drop_arm": "{arm_id} dropped from {study_id}",
    "reduce_control": "{study_id} control cap reduced to {participant_count}",
    "wait": "Waited until programme day {programme_day:,.0f}",
}

RuntimeEventType = Literal[
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


class AgentExecutionTerminated(RuntimeError):
    """The participant code exhausted or terminated its isolated executor."""


class AgentTurnLimitExceeded(RuntimeError):
    """Raised when an analysis is not submitted within its turn budget."""


def _runtime_failure_type(error: BaseException | None) -> BenchmarkRuntimeFailureTypeV1:
    """Map terminal exceptions to the stable public runtime taxonomy."""

    if isinstance(error, AgentTurnLimitExceeded):
        return "turn_limit_no_submission"
    return common_runtime_failure_type_v1(error)


class TrialDevAgentTimeCheckpoint(BaseModel):
    """Remaining charged time; null means the original task had no time limit."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    remaining_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    charged_operation_in_progress: bool = False


class TrialDevAgentCheckpoint(BaseModel):
    """Durable conversation and pending-call position for one authored attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    identity: str
    messages: list[dict[str, JsonValue]]
    phase_id: str
    step_id: str
    active_prompt_index: int = Field(ge=0)
    event_index: int = Field(ge=0)
    report_index: int = Field(ge=0)
    turns_used: int = Field(ge=0)
    pending_calls: tuple[ToolCall, ...]
    pending_message_index: int = Field(ge=0)
    pending_tool_index: int = Field(ge=0)
    executed_code: list[dict[str, JsonValue]]
    execution_outputs: list[dict[str, JsonValue]]
    reporting_context: NumericalReportingContext | None
    # Checkpoints without agent-time accounting remain readable for audit.
    agent_time: TrialDevAgentTimeCheckpoint | None = None
    model_noncompletion: ModelResponseOutcome | None = None


@dataclass(frozen=True)
class ReportCapture:
    """One raw report and the provider tool call that transported it."""

    content: str
    tool_call_id: str
    tool_name: str
    path: Path


@dataclass
class TrialDevTaskAgent:
    """Run one persistent, bounded analysis conversation for a TrialDev task."""

    provider: LLMProvider
    workdir: Path
    system_prompt: str
    program_id: str
    scenario_id: str
    objective_id: str
    max_tokens: int = TRIALDEV_RELEASE_BUDGET_V1.maximum_completion_tokens_per_turn
    max_turns_per_step: int = DEFAULT_MAX_TURNS_PER_STEP
    tool_choice: ToolChoiceV1 = "auto"
    max_tool_output_chars: int = 4_000
    max_context_chars: int = TRIALDEV_RELEASE_BUDGET_V1.maximum_context_characters
    provider_capability: ProviderModelCapabilityV1 | None = None
    analysis_language: AnalysisMode = "python_r"
    authored_programme: TrialDevActionSession | None = None
    reporting_context: NumericalReportingContext | None = None
    verbose: bool = False
    resume: bool = False
    data_view: Literal["raw", "analysis_ready"] = "raw"
    conversation_log_path: Path | None = None
    event_log_path: Path | None = None
    provider_log_path: Path | None = None
    executor_image: str | None = None
    executor_limits: CodeExecutionLimitsV1 | None = None
    deadline_monotonic: float | None = None
    agent_time_budget_seconds: float | None = None
    provider_unavailability_policy: Literal["charge_all_v1", "defer_unanswered_v1"] = "charge_all_v1"
    messages: list[dict[str, Any]] = field(init=False)
    session: CodeExecutionSession | None = field(init=False)
    _executor_image_id: str = field(init=False)
    phase_id: str | None = field(default=None, init=False)
    step_id: str | None = field(default=None, init=False)
    active_prompt_index: int | None = field(default=None, init=False)
    event_index: int = field(default=0, init=False)
    report_index: int = field(default=0, init=False)
    executed_code: list[dict[str, JsonValue]] = field(default_factory=list, init=False)
    execution_outputs: list[dict[str, JsonValue]] = field(default_factory=list, init=False)
    turns_used: int = field(default=0, init=False)
    pending_calls: tuple[ToolCall, ...] = field(default=(), init=False)
    pending_message_index: int = field(default=0, init=False)
    pending_tool_index: int = field(default=0, init=False)
    step_terminal: bool = field(default=True, init=False)
    _programme_state: TrialDevProgrammeStatus | None = field(default=None, init=False)
    deadline: RuntimeDeadline = field(init=False)
    _charged_operation_in_progress: bool = field(default=False, init=False)
    _time_notice_shown: bool = field(default=False, init=False)
    model_noncompletion: ModelResponseOutcome | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Validate budgets and open the isolated analysis session."""

        for field_name, value in (
            ("max_tokens", self.max_tokens),
            ("max_turns_per_step", self.max_turns_per_step),
            ("max_tool_output_chars", self.max_tool_output_chars),
            ("max_context_chars", self.max_context_chars),
        ):
            if value < 1:
                raise ValueError(f"{field_name} must be positive")
        self.executor_limits = self.executor_limits or TRIALDEV_RELEASE_BUDGET_V1.code_execution
        self.system_prompt += "\n" + analysis_instructions(self.analysis_language, limits=self.executor_limits)
        self.system_prompt += self._run_rules()
        self.deadline = RuntimeDeadline(
            monotonic=self.deadline_monotonic,
            label="TrialDev task",
            budget_seconds=self.agent_time_budget_seconds,
            provider_unavailability_policy=self.provider_unavailability_policy,
        )
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.report_index = len(tuple((self.workdir.parent / "analysis_sources").glob("submission-*")))
        if self.authored_programme is not None:
            workspace = self.workdir.resolve()
            programme = self.authored_programme
            programme.evidence_root.resolve().relative_to(workspace)
            for private in (programme.pool_root, programme.budget_path, programme.commitment_root):
                if private.resolve().is_relative_to(workspace):
                    raise ValueError("Authored programme private state cannot be inside the agent workspace")
        if self.resume:
            self._restore_checkpoint()
        with self.deadline.harness_work():
            session = DockerCodeSession(
                cwd=self.workdir,
                image=self.executor_image,
                limits=self.executor_limits,
            )
            # Replay checks reuse the immutable image, even if its tag moves mid-episode.
            self._executor_image_id = session.image_id
            self.session = session
        if not self.resume:
            self.messages = [{"role": "system", "content": self.system_prompt}]
        self._persist_conversation()

    def _run_rules(self) -> str:
        """State the turn budget, its final-turn restriction and the rules the tools do not announce."""

        last = self.max_turns_per_step
        if self.authored_programme is not None:
            budget = (
                f"You have {last} turns for the whole programme; turn {last} offers only stop_development and "
                "advance_candidate, and a rejected call on that turn cannot be retried. A programme without a terminal "
                "decision when turns or agent time run out is valued as a stop on its current day."
            )
        else:
            budget = (
                f"You have {last} turns for each report; turn {last} offers only the report submission tools, and a "
                "rejected submission on that turn cannot be retried. A report not submitted when turns or agent time "
                "run out is recorded as missing."
            )
        entrypoints = " or ".join(
            "scratch/analysis.py" if name == "python" else "scratch/analysis.R"
            for name in analysis_languages(self.analysis_language)
        )
        limits = self.executor_limits or TRIALDEV_RELEASE_BUDGET_V1.code_execution
        return (
            f"{budget} Model requests and the analysis and workspace tools use agent time; check_analysis, "
            "saved-analysis reruns and programme tools do not. Keep exactly one saved script, "
            f"{entrypoints}. Tool results longer than {self.max_tool_output_chars:,} characters show only their first "
            f"and last parts; the full text (up to {limits.output_bytes // 1024:,} KiB per execution) is saved under "
            "scratch/outputs/.\n"
        )

    def _time_notice(self) -> str | None:
        """Disclose the agent-time budget in full once, then as a one-line remainder."""

        if not self._time_notice_shown:
            notice = self.deadline.model_notice()
            self._time_notice_shown = notice is not None
            return notice
        remaining = self.deadline.remaining()
        if remaining is None:
            return None
        total = "" if self.agent_time_budget_seconds is None else f" of {self.agent_time_budget_seconds:g}"
        return f"[Agent time: {math.floor(remaining)}{total} seconds remaining.]"

    def _checkpoint_identity(self) -> str:
        programme = self.authored_programme
        if programme is None and self.reporting_context is None:
            raise ValueError("Numerical reporting or an authored programme is required for resumable state")
        ledger = None if programme is None else read_json_model(TrialDevBudgetLedger, programme.budget_path)
        return canonical_payload_sha256(
            {
                "program_id": self.program_id,
                "scenario_id": self.scenario_id,
                "objective_id": self.objective_id,
                "system_prompt": self.system_prompt,
                "analysis_language": self.analysis_language,
                "max_turns": self.max_turns_per_step,
                "max_tokens": self.max_tokens,
                "max_context": self.max_context_chars,
                "agent_time_budget_seconds": self.agent_time_budget_seconds,
                **(
                    {"provider_unavailability_policy": self.provider_unavailability_policy}
                    if self.provider_unavailability_policy != "charge_all_v1"
                    else {}
                ),
                "data_view": self.data_view,
                "seed": None if programme is None else programme.recruitment_seed,
                "pool": None if ledger is None else ledger.pool_manifest_sha256,
                "analysis_context": (
                    self.reporting_context.model_dump(mode="json")
                    if programme is None and self.reporting_context is not None
                    else None
                ),
                "image": self.executor_image,
                "limits": None if self.executor_limits is None else self.executor_limits.model_dump(mode="json"),
            }
        )

    def _checkpoint_path(self) -> Path:
        path = self.workdir.parent / "private" / "agent_state.json"
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError("Agent checkpoint cannot use symbolic links")
        return path

    def _persist_checkpoint(self) -> None:
        if (self.authored_programme is None and self.reporting_context is None) or self.active_prompt_index is None:
            return
        with self.deadline.harness_work():
            payload = {
                name: getattr(self, name)
                for name in TrialDevAgentCheckpoint.model_fields
                if name not in {"identity", "agent_time"}
            }
            payload["agent_time"] = TrialDevAgentTimeCheckpoint(
                remaining_seconds=self.deadline.snapshot_remaining(),
                charged_operation_in_progress=self._charged_operation_in_progress,
            )
            checkpoint = TrialDevAgentCheckpoint.model_validate({"identity": self._checkpoint_identity(), **payload})
            write_json_model(self._checkpoint_path(), checkpoint)

    def _restore_checkpoint(self) -> None:
        checkpoint = read_json_model(TrialDevAgentCheckpoint, self._checkpoint_path())
        if checkpoint.agent_time is None:
            raise ValueError(
                "Cannot resume this checkpoint: it has no saved agent-time budget. "
                "Elapsed time cannot be preserved; start a separately labelled new attempt instead."
            )
        if checkpoint.agent_time.charged_operation_in_progress:
            raise ValueError(
                "Cannot resume after an unclean interruption during a charged operation: its elapsed agent time "
                "was not saved. Start a separately labelled new attempt instead of resetting the time budget."
            )
        if checkpoint.identity != self._checkpoint_identity():
            raise ValueError("Agent checkpoint differs from the requested programme or execution limits")
        if (
            checkpoint.active_prompt_index >= len(checkpoint.messages)
            or checkpoint.pending_message_index >= len(checkpoint.messages)
            or checkpoint.turns_used > self.max_turns_per_step
        ):
            raise ValueError("Agent checkpoint has an invalid conversation position")
        for name in TrialDevAgentCheckpoint.model_fields:
            if name not in {"identity", "agent_time"}:
                setattr(self, name, getattr(checkpoint, name))
        if self.event_log_path is not None:
            if self.event_log_path.is_symlink():
                raise ValueError("Agent event log cannot be a symbolic link")
            events = tuple(read_jsonl_models(BenchmarkRuntimeTraceEventV1, self.event_log_path))
            if len(events) < self.event_index or any(
                event.event_index != index
                or event.program_id != self.program_id
                or event.scenario_id != self.scenario_id
                or event.objective_id != self.objective_id
                for index, event in enumerate(events)
            ):
                raise ValueError("Agent event log differs from the checkpoint identity or sequence")
            self.event_index = len(events)
        if self.pending_calls and self.pending_calls[0].name in {"execute_code", "check_analysis"}:
            transaction = self._code_transaction(self.pending_calls[0])
            snapshot = transaction / ("after" if (transaction / "after").exists() else "before") / "scratch"
            if snapshot.exists():
                self._restore_scratch(snapshot)
        if self.provider_log_path is not None:
            for path in self.provider_log_path.parent.glob(f"{self.provider_log_path.stem}*.jsonl"):
                provider_events, torn = read_provider_request_log_v1(path)
                outcomes = [event for event in provider_events if event.model_outcome is not None]
                if outcomes:
                    self.model_noncompletion = outcomes[-1].model_outcome
                elif any(
                    (event.status == "succeeded" or event.response_received)
                    and event.phase_id == self.phase_id
                    and event.step_id == self.step_id
                    and event.turn_index > self.turns_used
                    for event in provider_events
                ):
                    raise ValueError(
                        "Cannot resume a received provider response before its conversation checkpoint; "
                        "start a separately labelled new attempt instead of repeating a paid primary"
                    )
                elif torn or sum(event.status == "started" for event in provider_events) != sum(
                    event.status != "started" for event in provider_events
                ):
                    raise ValueError("Cannot resume an interrupted provider request with an unknown paid outcome")
        self.deadline.restore_remaining(checkpoint.agent_time.remaining_seconds)
        self.step_terminal = False

    @contextmanager
    def _charged_work(self) -> Iterator[None]:
        """Save active work before starting, then its consumed time even on a handled failure."""

        self._charged_operation_in_progress = True
        self._persist_checkpoint()
        try:
            yield
        finally:
            self._charged_operation_in_progress = False
            self._persist_checkpoint()

    def close(self) -> None:
        """Close the execution session without changing task outcome state."""

        if self.session is not None:
            with self.deadline.harness_work():
                self.session.close()
                self.session = None
                self._persist_checkpoint()

    def finish_step(
        self,
        *,
        status: Literal["completed", "failed"],
        failure_type: BenchmarkRuntimeFailureTypeV1 | None = None,
    ) -> None:
        """Record the terminal outcome owned by the host evaluation boundary."""

        self._finish_step(status=status, failure_type=failure_type)

    def begin_step(self, *, phase_id: str, step_id: str) -> None:
        """Start one decision step after the prior step has terminated."""

        if self.phase_id is not None and not self.step_terminal:
            raise RuntimeError("TrialDev cannot start a step before the prior step has a recorded outcome")
        self.phase_id = phase_id
        self.step_id = step_id
        self.active_prompt_index = None
        self.turns_used = 0
        self.pending_calls = ()
        self.pending_tool_index = 0
        self.executed_code = []
        self.execution_outputs = []
        self.step_terminal = False
        self._emit_event(event_type="step_started")

    def append_user_message(self, content: str) -> None:
        """Append participant-visible task or correction context."""

        with self.deadline.harness_work():
            self.messages.append({"role": "user", "content": content})
            if self.active_prompt_index is None:
                self.active_prompt_index = len(self.messages) - 1
            self._persist_conversation()
            self._emit_event(event_type="prompt", conversation_message_index=len(self.messages) - 1)

    def append_tool_reply(
        self,
        tool_call_id: str,
        content: str,
        *,
        tool_name: str,
        execution: CodeExecutionResultV1 | None = None,
        status: Literal["observed", "invalid"] = "observed",
    ) -> None:
        """Append the observed result of one provider tool call."""

        if self.pending_calls and self.pending_calls[0].id == tool_call_id:
            self.pending_calls = self.pending_calls[1:]
            self.pending_tool_index += 1
        self.messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})
        self._persist_conversation()
        self._emit_event(
            event_type="tool_result",
            conversation_message_index=len(self.messages) - 1,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            status=status,
            execution=execution,
        )

    def run_until_report(self) -> ReportCapture:
        """Run analysis turns until the model submits one non-empty report."""

        if self.active_prompt_index is None:
            raise RuntimeError("A TrialDev step prompt is required before execution")
        completed = False
        try:
            capture = self._run_until_report()
            if capture is None:
                raise RuntimeError("Report-only execution returned a terminal programme action")
            completed = True
            return capture
        finally:
            if not completed and not self.step_terminal:
                error = sys.exception()
                self._finish_step(status="failed", failure_type=_runtime_failure_type(error))

    def run_programme(self) -> None:
        """Continue until the agent stops or advances the authored programme."""
        if self.authored_programme is None or self.active_prompt_index is None:
            raise RuntimeError("An authored programme and active prompt are required")
        completed = False
        try:
            ledger = read_json_model(TrialDevBudgetLedger, self.authored_programme.budget_path)
            if ledger.terminal is None or self.pending_calls:
                if self._run_until_report(allow_terminal=True) is not None:
                    raise RuntimeError("Programme runs offer no report tool")
            if not self.step_terminal:
                self._finish_step(status="completed")
            completed = True
        finally:
            if not completed and not self.step_terminal:
                error = sys.exception()
                self._finish_step(
                    status="failed",
                    failure_type=(
                        "turn_limit_no_terminal_decision"
                        if isinstance(error, AgentTurnLimitExceeded)
                        else _runtime_failure_type(error)
                    ),
                )

    def _run_until_report(self, *, allow_terminal: bool = False) -> ReportCapture | None:
        if self.session is None:
            raise RuntimeError("TrialDev task agent is closed")
        if self.phase_id is None or self.step_id is None or self.active_prompt_index is None:
            raise RuntimeError("TrialDev trace context is incomplete")
        if self.model_noncompletion is not None:
            raise ModelResponseNoncompletion(self.model_noncompletion)
        first_turn = self.turns_used if self.pending_calls else self.turns_used + 1
        for turn in range(first_turn, self.max_turns_per_step + 1):
            self.deadline.remaining()
            request_tools = [*analysis_tools(self.analysis_language), *WORKSPACE_TOOLS]
            request_tools += (
                [_SUBMIT_REPORT, _SUBMIT_REPORT_FILE]
                if self.authored_programme is None
                else trialdev_programme_tools()
            )
            if turn == self.max_turns_per_step:
                if not self.pending_calls:
                    self.append_user_message(
                        _FINAL_PROGRAMME_TURN
                        if allow_terminal
                        else FINAL_TURN_SUBMISSION_REMINDER + _INCOMPLETE_SUBMISSION_NOTE
                    )
                request_tools = trialdev_terminal_tools() if allow_terminal else [_SUBMIT_REPORT, _SUBMIT_REPORT_FILE]
            if not self.pending_calls:
                notice = self._time_notice()
                if notice is not None:
                    self.append_user_message(notice)
                # The checkpoint that clears the charge must already hold the response; otherwise a
                # crash before the conversation is saved leaves a clean-looking, unresumable checkpoint.
                with self._charged_work():
                    turn_result = self._provider_turn(
                        tools=request_tools, turn=turn, timeout_seconds=self.deadline.remaining()
                    )
                    self.turns_used = turn
                    self.messages.append(turn_result.message)
                    self.pending_calls = tuple(turn_result.response.tool_calls)
                    self.pending_message_index = len(self.messages) - 1
                    self.pending_tool_index = 0
                self._persist_conversation()
                self._emit_event(event_type="assistant_message", conversation_message_index=self.pending_message_index)
                if turn_result.response.model_outcome is not None:
                    self.model_noncompletion = turn_result.response.model_outcome
                    self._persist_checkpoint()
                    raise ModelResponseNoncompletion(self.model_noncompletion)
            message_index = self.pending_message_index
            counter = turn_budget_tag(turn=turn, maximum=self.max_turns_per_step)
            if not self.pending_calls:
                self.append_user_message(
                    (
                        "Continue with the tools; development ends only with stop_development or advance_candidate."
                        if allow_terminal
                        else "Continue the analysis or submit the completed report."
                    )
                    + counter
                )
                continue
            offered = offered_tool_names_v1(request_tools)
            for tool_call_index, call in enumerate(self.pending_calls, start=self.pending_tool_index):
                self.deadline.remaining()
                event_type = self._event_type(call.name)
                file_accessed: str | None = None
                try:
                    if call.name not in offered:
                        raise ToolInputError(f"tool {call.name!r} was not offered on this turn")
                    arguments = parse_tool_arguments(call.arguments, tool_name=call.name)
                    file_accessed = self._file_access(arguments=arguments, tool_name=call.name)
                    if call.name in {_REPORT_TOOL, _REPORT_FILE_TOOL}:
                        with self._charged_work():
                            report = self._report_text(call=call, arguments=arguments)
                        accept_incomplete = arguments.get("accept_incomplete", False)
                        if not isinstance(accept_incomplete, bool):
                            raise ToolInputError("accept_incomplete must be a boolean")
                        with self.deadline.harness_work():
                            self._check_cutoff(accept_incomplete=accept_incomplete)
                            self.session.snapshot_scratch()
                            path = self._capture_report(report, call=call)
                            if not accept_incomplete:
                                self._validate_public_result(path.parent)
                            self._commit_numerical_report(path.parent)
                        self._emit_event(
                            event_type="submission",
                            conversation_message_index=message_index,
                            tool_call_index=tool_call_index,
                            tool_call_id=call.id,
                            tool_name=call.name,
                            file_accessed=file_accessed,
                        )
                        return ReportCapture(
                            content=report,
                            tool_call_id=call.id,
                            tool_name=call.name,
                            path=path,
                        )
                    with self.deadline.harness_work() if call.name in _HARNESS_TOOLS else self._charged_work():
                        execution = self._execute_checkpointed_tool(call)
                except ToolInputError as error:
                    self._emit_event(
                        event_type=event_type,
                        conversation_message_index=message_index,
                        tool_call_index=tool_call_index,
                        tool_call_id=call.id,
                        tool_name=call.name,
                        file_accessed=file_accessed,
                        status="invalid",
                    )
                    self.append_tool_reply(
                        call.id,
                        f"Tool input rejected: {error}" + counter,
                        tool_name=call.name,
                        status="invalid",
                    )
                    continue
                if call.name == "execute_code":
                    self.executed_code.append(
                        {
                            "tool_call_id": call.id,
                            "code": cast(str, arguments["code"]),
                            "language": cast(JsonValue, arguments.get("language", "python")),
                            "purpose": cast(JsonValue, arguments.get("purpose")),
                        }
                    )
                if call.name in {"execute_code", "check_analysis"} | PROGRAMME_TOOLS:
                    record: dict[str, JsonValue] = {
                        "tool_call_id": call.id,
                        "result": cast(JsonValue, execution.model_dump(mode="json")),
                    }
                    if call.name in PROGRAMME_TOOLS and self._programme_state is not None:
                        # Structured state for action traces; the agent sees only the Markdown letter.
                        record["programme_state"] = cast(JsonValue, self._programme_state.model_dump(mode="json"))
                    self.execution_outputs.append(record)
                self._emit_event(
                    event_type=event_type,
                    conversation_message_index=message_index,
                    tool_call_index=tool_call_index,
                    tool_call_id=call.id,
                    tool_name=call.name,
                    file_accessed=file_accessed,
                    # A read's success travels with its event, so failed reads are not evidence use.
                    execution=execution if event_type == "file_inspection" else None,
                )
                with self.deadline.harness_work():
                    reply = persist_bulky_tool_output(
                        execution.output
                        or ("(completed without output)" if execution.status == "success" else execution.status),
                        session=self.session,
                        artifact_id=f"turn-{turn}-{tool_call_index + 1}",
                        inline_chars=self.max_tool_output_chars,
                    )
                self.append_tool_reply(
                    call.id,
                    reply + counter,
                    tool_name=call.name,
                    execution=execution,
                )
                if execution.status == "session_terminated":
                    raise AgentExecutionTerminated(execution.output or "Participant execution session terminated")
                if allow_terminal and call.name in PROGRAMME_TOOLS and self.authored_programme is not None:
                    ledger = read_json_model(TrialDevBudgetLedger, self.authored_programme.budget_path)
                    if ledger.terminal is not None:
                        self.pending_calls = ()
                        return None
        required = "terminal development decision" if allow_terminal else "report"
        raise AgentTurnLimitExceeded(f"No {required} was submitted within {self.max_turns_per_step} turns")

    def _provider_turn(
        self,
        *,
        tools: list[dict[str, JsonValue]],
        turn: int,
        timeout_seconds: float | None,
    ) -> AgentTurn:
        required: tuple[int, ...] = (len(self.messages) - 1,) if turn == self.max_turns_per_step else ()
        if self.deadline.monotonic is not None:
            required = (
                (len(self.messages) - 2, len(self.messages) - 1)
                if turn == self.max_turns_per_step
                else (len(self.messages) - 1,)
            )
        return run_agent_turn_v1(
            provider=self.provider,
            messages=self.messages,
            session=cast(CodeExecutionSession, self.session),
            active_prompt_index=cast(int, self.active_prompt_index),
            max_context_chars=self.max_context_chars,
            provider_capability=self.provider_capability,
            required_message_indices=required,
            tools=tools,
            temperature=0.0,
            max_tokens=self.max_tokens,
            timeout_seconds=timeout_seconds,
            deadline=self.deadline,
            tool_choice=self.tool_choice,
            provider_log_path=self.provider_log_path,
            benchmark="trialdev",
            unit_id=self.program_id,
            phase_id=self.phase_id or "",
            step_id=self.step_id or "",
            turn_index=turn,
        )

    def _report_text(self, *, call: ToolCall, arguments: dict[str, object]) -> str:
        if self.session is None:
            raise RuntimeError("TrialDev task agent is closed")
        if call.name == _REPORT_FILE_TOOL:
            if set(arguments) - {"accept_incomplete"} != {"path"}:
                raise ToolInputError(f"{_REPORT_FILE_TOOL} accepts only path")
            report = self.deadline.run_blocking(
                partial(read_workspace_submission_text, path=arguments.get("path"), session=self.session),
                operation_name="TrialDev report read",
                on_timeout=self.session.close,
            )
        else:
            if set(arguments) - {"accept_incomplete"} != {"content"} or not isinstance(arguments.get("content"), str):
                raise ToolInputError(f"{_REPORT_TOOL} requires one Markdown report")
            report = str(arguments["content"])
        if not report.strip():
            raise ToolInputError("The submitted report is empty")
        return report

    def _check_cutoff(self, *, accept_incomplete: bool) -> None:
        """Capture reproducible saved-script output before a reporting decision."""
        if self.reporting_context is None or self._reporting_cutoff_closed(self.reporting_context.checkpoint_id):
            return
        context_identity = canonical_payload_sha256(self.reporting_context.model_dump(mode="json"))
        if (self.workdir.parent / "analysis_sources" / "submitted" / context_identity).exists():
            return
        if self.session is None:
            raise RuntimeError("TrialDev task agent is closed")
        result, self.session = check_analysis_tool_v1(
            ToolCall(id="cutoff-check", name="check_analysis", arguments="{}"),
            session=self.session,
            workspace=self.workdir,
            context=self.reporting_context,
            executor_limits=self.executor_limits or TRIALDEV_RELEASE_BUDGET_V1.code_execution,
            executor_image=self._executor_image_id,
            analysis_language=self.analysis_language,
            reporting_cutoff=True,
        )
        if result.status != "success" and not accept_incomplete:
            raise ToolInputError(
                "Analysis capture rejected: " + result.output + " Fix the saved script or set accept_incomplete=true."
            )

    def _validate_public_result(self, captured: Path) -> tuple[str, ...]:
        """Reject malformed results and report omitted targets without grading."""
        context = read_json_model(NumericalReportingContext, captured / "reporting_context.json")
        try:
            rows = read_numerical_table(captured / "estimates.csv", target_ids=context.target_ids)
        except (FileNotFoundError, ValueError, csv.Error, UnicodeDecodeError) as error:
            raise ToolInputError(f"Invalid scratch/estimates.csv: {str(error)[:1200]}") from error
        present = {row.target_id for row in rows}
        return tuple(target for target in context.target_ids if target not in present)

    def _capture_report(self, report: str, *, call: ToolCall | None = None) -> Path:
        if (self.authored_programme is not None or self.reporting_context is not None) and call is not None:
            identity = canonical_payload_sha256(
                {
                    "step": self.step_id,
                    "turn": self.turns_used,
                    "call_id": call.id,
                    "arguments": call.arguments,
                }
            )
            root = self.workdir.parent / "analysis_sources" / f"submission-{identity}"
            if root.exists():
                if root.is_symlink() or (root / "report.md").read_text(encoding="utf-8") != report:
                    raise ValueError("Repeated report identity differs from its committed capture")
                return root / "report.md"
            with staged_directory(root) as staging:
                (staging / "report.md").write_text(report, encoding="utf-8")
                (staging / "capture_id.txt").write_text(identity, encoding="utf-8")
                self._capture_analysis(staging)
            self.report_index += 1
            return root / "report.md"
        self.report_index += 1
        root = self.workdir.parent / "analysis_sources" / f"submission-{self.report_index:04d}"
        root.mkdir(parents=True, exist_ok=False)
        path = root / "report.md"
        path.write_text(report, encoding="utf-8")
        self._capture_analysis(root)
        return path

    def _capture_analysis(self, root: Path, *, allow_invalid_program: bool = False) -> None:
        """Copy the current source, output and reporting context without inventing a report."""
        if self.reporting_context is not None:
            write_json_model(root / "reporting_context.json", self.reporting_context)
        scratch = self.workdir / "scratch"
        programs = tuple(path for path in (scratch / "analysis.py", scratch / "analysis.R") if path.is_file())
        if programs:
            try:
                sources = analysis_project_files(programs[0])
            except ValueError as error:
                if not allow_invalid_program:
                    raise ToolInputError(str(error)) from error
                sources = ()
                (root / "source_error.txt").write_text(str(error), encoding="utf-8")
            for source in sources:
                destination = root / source.relative_to(scratch)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        result_path = scratch / "estimates.csv"
        if result_path.is_file() and not result_path.is_symlink():
            shutil.copy2(result_path, root / result_path.name)
        if self.executed_code:
            code_path = root / "executed_code.json"
            code_path.write_text(
                json.dumps(self.executed_code, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if self.execution_outputs:
            output_path = root / "execution_output.json"
            output_path.write_text(
                json.dumps(self.execution_outputs, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    def _reporting_cutoff_closed(self, checkpoint_id: str) -> bool:
        """Read accepted cutoff records; rejected actions do not close a report."""
        root = self.workdir.parent / "analysis_sources" / "cutoffs"
        return any(
            read_json_model(NumericalReportingContext, path).checkpoint_id == checkpoint_id
            for path in sorted(root.glob("*/reporting_context.json"))
        )

    def _commit_numerical_report(self, captured: Path) -> None:
        """Keep the first syntax-valid explicit report for its reporting context."""
        if self.reporting_context is None:
            return
        context = read_json_model(NumericalReportingContext, captured / "reporting_context.json")
        if self._reporting_cutoff_closed(context.checkpoint_id):
            raise ToolInputError("The reporting cutoff has already closed for this checkpoint")
        identity = canonical_payload_sha256(context.model_dump(mode="json"))
        committed = self.workdir.parent / "analysis_sources" / "submitted" / identity
        if committed.exists():
            capture_id = captured / "capture_id.txt"
            committed_id = committed / "capture_id.txt"
            if (
                capture_id.is_file()
                and committed_id.is_file()
                and capture_id.read_bytes() == committed_id.read_bytes()
            ):
                return
            raise ToolInputError("A numerical report has already been submitted for this checkpoint")
        with staged_directory(committed) as staging:
            shutil.copytree(captured, staging, dirs_exist_ok=True)

    def _prepare_cutoff(
        self, identity: str, *, published: str | None, reject_invalid_result: bool = False
    ) -> Path | None:
        """Preserve the pre-action analysis; failed actions leave only pending captures."""
        root = self.workdir.parent / "analysis_sources"
        if (root / "cutoffs" / identity).exists():
            return None
        pending = root / "pending_cutoffs" / identity
        if pending.exists():
            context = read_json_model(NumericalReportingContext, pending / "reporting_context.json")
            return None if self._reporting_cutoff_closed(context.checkpoint_id) else pending
        if self.reporting_context is None or self._reporting_cutoff_closed(self.reporting_context.checkpoint_id):
            return None
        if (
            self.authored_programme is not None
            and published is not None
            and (self.authored_programme.evidence_root / published).exists()
        ):
            return None
        if self.session is None:
            raise RuntimeError("TrialDev task agent is closed")
        context_identity = canonical_payload_sha256(self.reporting_context.model_dump(mode="json"))
        committed = root / "submitted" / context_identity
        if committed.exists():
            with staged_directory(pending) as staging:
                shutil.copytree(committed, staging, dirs_exist_ok=True)
        else:
            self.session.snapshot_scratch()
            with staged_directory(pending) as staging:
                self._capture_analysis(staging, allow_invalid_program=True)
                if reject_invalid_result and (staging / "estimates.csv").exists():
                    self._validate_public_result(staging)
        return pending

    def _assign_readout_context(self, readout_path: Path) -> TrialDevReadoutManifest:
        """Bind newly published raw records and public targets to the next report."""
        root = self.workdir / readout_path
        dictionary = self.workdir / "documents/observational_data_dictionary.csv"
        projection = (
            write_trialdev_analysis_ready(
                root,
                self.workdir / "analysis_ready" / readout_path.name,
                baseline_fields=trialdev_baseline_fields_from_dictionary(dictionary) if dictionary.is_file() else (),
            )
            if self.data_view == "analysis_ready"
            else None
        )
        manifest = read_json_model(TrialDevReadoutManifest, root / "study.json")
        evidence = (
            {}
            if self.reporting_context is None
            else {item.relative_path: item for item in self.reporting_context.evidence_artifacts}
        )
        already_acquired = (readout_path / "study.json").as_posix() in evidence
        paths = list(root.rglob("*")) + ([] if projection is None else list(projection.rglob("*")))
        for path in sorted(paths):
            if path.is_symlink():
                raise ValueError("published readout evidence cannot contain symbolic links")
            if not path.is_file():
                continue
            relative = path.relative_to(self.workdir).as_posix()
            artifact = ReplayInputArtifactV1(
                relative_path=relative, sha256=sha256_file(path), byte_length=path.stat().st_size
            )
            if relative in evidence and evidence[relative] != artifact:
                raise ValueError("published readout evidence changed after acquisition")
            evidence[relative] = artifact
        if already_acquired:
            return manifest
        self.reporting_context = NumericalReportingContext(
            checkpoint_id=readout_path.as_posix(),
            target_ids=tuple(target.target_id for target in manifest.targets),
            evidence_artifacts=tuple(evidence[name] for name in sorted(evidence)),
        )
        stage_reporting_interface(self.workdir, self.reporting_context.target_ids)
        return manifest

    def _code_transaction(self, call: ToolCall) -> Path:
        identity = canonical_payload_sha256(
            {
                "step": self.step_id,
                "turn": self.turns_used,
                "call_id": call.id,
                "name": call.name,
                "arguments": call.arguments,
            }
        )
        path = self._checkpoint_path().parent / "code_calls" / identity
        if path.parent.is_symlink() or path.is_symlink():
            raise ValueError("Code recovery records cannot use symbolic links")
        return path

    @staticmethod
    def _copy_scratch(source: Path, destination: Path) -> None:
        if source.is_symlink() or any(path.is_symlink() for path in source.rglob("*")):
            raise ValueError("Scratch recovery cannot follow symbolic links")
        shutil.copytree(source, destination)

    def _restore_scratch(self, snapshot: Path) -> None:
        scratch = self.workdir / "scratch"
        if scratch.is_symlink():
            raise ValueError("Scratch recovery destination cannot be a symbolic link")
        with TemporaryDirectory(prefix=".scratch-recovery-", dir=self.workdir.parent) as temporary:
            temporary_path = Path(temporary)
            self._copy_scratch(snapshot, temporary_path / "restored")
            if scratch.exists():
                scratch.rename(temporary_path / "interrupted")
            (temporary_path / "restored").rename(scratch)

    def _execute_checkpointed_tool(self, call: ToolCall) -> CodeExecutionResultV1:
        if self.session is None:
            raise RuntimeError("TrialDev task agent is closed")
        transaction = None
        if (self.authored_programme is not None or self.reporting_context is not None) and call.name in {
            "execute_code",
            "check_analysis",
        }:
            with self.deadline.harness_work():
                transaction = self._code_transaction(call)
                committed = transaction / "after"
                if committed.exists():
                    return read_json_model(CodeExecutionResultV1, committed / "result.json")
                before = transaction / "before"
                if not before.exists():
                    self.session.snapshot_scratch()
                    with staged_directory(before) as staging:
                        self._copy_scratch(self.workdir / "scratch", staging / "scratch")
        # Start the watchdog after recovery preparation. A pause in another thread
        # cannot extend a wait() timeout that was already computed.
        result = self.deadline.run_blocking(
            partial(self._dispatch_local_tool, call),
            operation_name=f"TrialDev tool {call.name}",
            on_timeout=self.session.close,
        )
        if transaction is not None:
            with self.deadline.harness_work():
                if result.status != "session_terminated":
                    self.session.snapshot_scratch()
                # A dead executor retains the last durable scratch and its terminal result.
                with staged_directory(transaction / "after") as staging:
                    self._copy_scratch(self.workdir / "scratch", staging / "scratch")
                    write_json_model(staging / "result.json", result)
        return result

    def _programme_tool(self, call: ToolCall, arguments: dict[str, object]) -> CodeExecutionResultV1:
        """Decode, execute and letter one programme tool call."""
        programme = self.authored_programme
        if programme is None:
            raise ToolInputError("No authored programme is available")
        identity = "action-" + canonical_payload_sha256({"step": self.step_id, "call": call.id})[:24]
        request_path = programme.commitment_root / f"{identity}.request.json"
        started = time.monotonic()
        self._programme_state = None
        try:
            if request_path.exists():
                # A resumed call keeps the action it committed before the interruption.
                decoded = TrialDevProgrammeCall(
                    action=read_json_model(TrialDevActionRequest, request_path).request,
                    accept_incomplete=arguments.get("accept_incomplete") is True,
                )
            else:
                decoded = programme.decode(call.name, arguments)
            action = decoded.action
            if decoded.preview:
                assert isinstance(action, TrialDevStartStudy)
                self._programme_state = trialdev_programme_status(programme.preview_study(action)).model_copy(
                    update={"preview": True}
                )
                return CodeExecutionResultV1(
                    status="success",
                    output=design_letter(action, self._programme_state, preview=True),
                    elapsed_seconds=time.monotonic() - started,
                )
        except TrialDevActionRejectedError as error:
            raise ToolInputError(f"{error}. Nothing was committed.") from error
        folder = action.evidence_folder if isinstance(action, TrialDevReadout) else None
        report_error = None
        pending = None
        if action.action in _CUTOFF_ACTIONS:
            if not (self.workdir.parent / "analysis_sources" / "pending_cutoffs" / identity).exists():
                self._check_cutoff(accept_incomplete=decoded.accept_incomplete)
            pending = self._prepare_cutoff(
                identity, published=folder, reject_invalid_result=not decoded.accept_incomplete
            )
            captured = pending or self.workdir.parent / "analysis_sources" / "cutoffs" / identity
            if captured.exists():
                try:
                    missing = self._validate_public_result(captured)
                    if missing:
                        report_error = "missing targets " + ", ".join(missing)
                except ToolInputError as error:
                    report_error = str(error)
        try:
            ledger = programme.execute(action, action_id=identity)
        except TrialDevActionRejectedError as error:
            raise ToolInputError(f"{error}. Nothing was committed.") from error
        if pending is not None:
            accepted = pending.parent.parent / "cutoffs" / identity
            accepted.parent.mkdir(parents=True, exist_ok=True)
            pending.rename(accepted)
        status = trialdev_programme_status(ledger)
        if isinstance(action, TrialDevReadout):
            readout_path = (
                (programme.evidence_root / action.evidence_folder).resolve().relative_to(self.workdir.resolve())
            )
            manifest = self._assign_readout_context(readout_path)
            status = status.model_copy(update={"readout_path": readout_path.as_posix()})
            text = readout_letter(
                manifest,
                folder=readout_path.as_posix(),
                look=int(action.evidence_folder.rsplit("-", 1)[1]),
                status=status,
                previous_capture_error=report_error,
            )
        elif status.terminal is not None:
            text = decision_letter(status, previous_capture_error=report_error)
        elif isinstance(action, TrialDevStartStudy):
            text = design_letter(action, status, preview=False)
        else:
            text = status_letter(status, change=_CHANGE_TITLES[action.action].format(**action.model_dump()))
        self._programme_state = status
        return CodeExecutionResultV1(status="success", output=text, elapsed_seconds=time.monotonic() - started)

    def _dispatch_local_tool(self, call: ToolCall) -> CodeExecutionResultV1:
        if self.session is None:
            raise RuntimeError("TrialDev task agent is closed")
        arguments = parse_tool_arguments(call.arguments, tool_name=call.name)
        if call.name in PROGRAMME_TOOLS:
            return self._programme_tool(call, arguments)
        if call.name == "check_analysis":
            if self.reporting_context is None:
                raise ToolInputError("No numerical reporting context is currently assigned")
            result, self.session = check_analysis_tool_v1(
                call,
                session=self.session,
                workspace=self.workdir,
                context=self.reporting_context,
                executor_limits=self.executor_limits or TRIALDEV_RELEASE_BUDGET_V1.code_execution,
                executor_image=self._executor_image_id,
                analysis_language=self.analysis_language,
            )
            return result
        if call.name in {"execute_code", "inspect_parquet"}:
            result = execute_analysis_tool_v1(call, session=self.session, analysis_language=self.analysis_language)
            if self.verbose:
                print(terminal_text(f"  [{call.name}] {result.output[:300]}"))
            return result
        if call.name in {"write_workspace_file", "read_workspace_file", "list_workspace_files"}:
            return handle_workspace_tool(name=call.name, arguments=arguments, session=self.session)
        raise ToolInputError(f"Unknown local tool: {call.name}")

    @staticmethod
    def _event_type(tool_name: str) -> RuntimeEventType:
        if tool_name in {_REPORT_TOOL, _REPORT_FILE_TOOL}:
            return "submission"
        if tool_name in {"execute_code", "check_analysis"}:
            return "code_execution"
        if tool_name in {"inspect_parquet", "read_workspace_file"}:
            return "file_inspection"
        return "tool_call"

    def _file_access(self, *, arguments: dict[str, object], tool_name: str) -> str | None:
        raw = arguments.get("path")
        if not isinstance(raw, str) or not raw:
            return None
        if tool_name == "write_workspace_file" or tool_name == _REPORT_FILE_TOOL:
            return raw if raw.startswith("scratch/") else f"scratch/{raw}"
        return raw

    def _persist_conversation(self) -> None:
        with self.deadline.harness_work():
            self._persist_checkpoint()
            if self.conversation_log_path is None:
                return
            self.conversation_log_path.parent.mkdir(parents=True, exist_ok=True)
            self.conversation_log_path.write_text(json.dumps(self.messages, indent=2, default=str), encoding="utf-8")

    def _emit_event(
        self,
        *,
        event_type: RuntimeEventType,
        conversation_message_index: int | None = None,
        tool_call_index: int | None = None,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        file_accessed: str | None = None,
        status: Literal["observed", "invalid"] = "observed",
        execution: CodeExecutionResultV1 | None = None,
        terminal_status: Literal["completed", "failed"] | None = None,
        failure_type: BenchmarkRuntimeFailureTypeV1 | None = None,
    ) -> None:
        if self.event_log_path is None:
            return
        if self.phase_id is None or self.step_id is None:
            raise RuntimeError("TrialDev trace context is incomplete")
        message = (
            cast(JsonValue, self.messages[conversation_message_index])
            if conversation_message_index is not None
            else None
        )
        payload = runtime_event_source_payload_v1(
            benchmark="trialdev",
            task_id=None,
            program_id=self.program_id,
            scenario_id=self.scenario_id,
            objective_id=self.objective_id,
            phase_id=self.phase_id,
            step_id=self.step_id,
            event_type=event_type,
            terminal_status=terminal_status,
            failure_type=failure_type,
            conversation_message=message,
        )
        event = BenchmarkRuntimeTraceEventV1(
            event_id=f"trialdev:{self.program_id}:{self.event_index:06d}",
            timestamp=datetime.now(UTC),
            source_artifact_path=(
                self.conversation_log_path.as_posix()
                if self.conversation_log_path is not None
                else f"run://trialdev/{self.program_id}/conversation"
            ),
            source_payload_sha256=canonical_payload_sha256(payload),
            benchmark="trialdev",
            event_index=self.event_index,
            program_id=self.program_id,
            scenario_id=self.scenario_id,
            objective_id=self.objective_id,
            phase_id=self.phase_id,
            step_id=self.step_id,
            event_type=event_type,
            conversation_message_index=conversation_message_index,
            tool_call_index=tool_call_index,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            file_accessed=file_accessed,
            status=status,
            execution_status=None if execution is None else execution.status,
            execution_language=None if execution is None else execution.language,
            elapsed_seconds=None if execution is None else execution.elapsed_seconds,
            cpu_seconds=None if execution is None else execution.cpu_seconds,
            session_peak_memory_bytes=None if execution is None else execution.session_peak_memory_bytes,
            session_peak_processes=None if execution is None else execution.session_peak_processes,
            output_truncated=None if execution is None else execution.output_truncated,
            terminal_status=terminal_status,
            failure_type=failure_type,
        )
        with self.deadline.harness_work():
            append_jsonl_model(self.event_log_path, event)
        self.event_index += 1

    def _finish_step(
        self,
        *,
        status: Literal["completed", "failed"],
        failure_type: BenchmarkRuntimeFailureTypeV1 | None = None,
    ) -> None:
        if self.step_terminal:
            raise RuntimeError("TrialDev step emitted more than one terminal event")
        self._emit_event(
            event_type="step_terminal",
            terminal_status=status,
            failure_type=failure_type,
        )
        self.step_terminal = True


__all__ = [
    "ANALYSIS_TOOLS",
    "AgentTurnLimitExceeded",
    "DEFAULT_MAX_TURNS_PER_STEP",
    "ReportCapture",
    "TrialDevTaskAgent",
]
