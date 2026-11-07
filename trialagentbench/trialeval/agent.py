"""Run the TrialEval agent loop.

This module gives an LLM the visible surface of a benchmark item, a persistent
Python session, and tools for clinical trial analysis.

The report explains the analysis; the required executable program supplies the
typed numerical result used for grading.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import subprocess
import sys
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Literal, cast

from pydantic.types import JsonValue
from trialagentbench_verifier.execution_contracts import AnalysisMode, ReportingInterfaceV1
from trialagentbench_verifier.executor import CodeExecutionSessionTerminated, DockerCodeSession
from trialagentbench_verifier.numerical import NumericalReportingContext

from trialagentbench.contracts.core.config import ProviderModelCapabilityV1
from trialagentbench.contracts.core.trialeval_conditions import (
    ProcedureAssistanceV1,
)
from trialagentbench.contracts.core.trialeval_factors import TrialEvalAnalysisSpecificationV1
from trialagentbench.contracts.release.trialeval_integrity import (
    TrialEvalPublicIntegrityPolicyV1,
)
from trialagentbench.contracts.trace.observable import (
    BenchmarkRuntimeFailureTypeV1,
    BenchmarkRuntimeTraceEventV1,
    RuntimeEventTypeV1,
    runtime_event_source_payload_v1,
)
from trialagentbench.execution_policy import TRIALEVAL_RELEASE_BUDGET_V1
from trialagentbench.io.checksums import canonical_payload_sha256
from trialagentbench.io.json import append_jsonl_model
from trialagentbench.ports import (
    CodeExecutionLimitsV1,
    CodeExecutionResultV1,
    CodeExecutionSession,
    LLMProvider,
    ModelResponseNoncompletion,
    ToolCall,
)
from trialagentbench.ports.tool_input import HarnessConfigurationError, ToolInputError, parse_tool_arguments
from trialagentbench.release import load_participant_release
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
from trialagentbench.trialeval.conditions import (
    INCOMPLETE_SUBMISSION_NOTE,
    procedure_assistance_v1,
    prompt_set_sha256_v1,
    stage_numerical_result_template,
    submission_instruction_v1,
    submission_tools_v1,
)
from trialagentbench.trialeval.data import (
    load_participant_diagnostic_dictionary,
    load_visible_context,
    participant_analysis_surface_sha256,
    participant_evidence_artifacts,
    stage_participant_evidence,
)
from trialagentbench.trialeval.data_integrity import (
    DATA_INTEGRITY_POLICY_FILENAME_V1,
)
from trialagentbench.trialeval.schema import BenchmarkItem
from trialagentbench.util.agent_runtime import offered_tool_names_v1, run_agent_turn_v1
from trialagentbench.util.runtime_context import (
    FINAL_TURN_SUBMISSION_REMINDER,
    SUBMISSION_WINDOW_REMINDER,
    RuntimeDeadline,
    RuntimeDeadlineExceeded,
    persist_bulky_tool_output,
)
from trialagentbench.util.runtime_failures import common_runtime_failure_type_v1
from trialagentbench.util.terminal import terminal_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MAX_TURNS = TRIALEVAL_RELEASE_BUDGET_V1.maximum_turns
# The final turns of every run offer only the submission tools.
SUBMISSION_ONLY_TURNS = 2
# Under the clarified reporting interface, a failed replay at submission reopens the analysis tools for
# this many turns, once per task, when no analysis turn remains.
REPAIR_TURNS = 2
# Longer tool results are saved under scratch/outputs/ and shown as head and tail.
_TOOL_OUTPUT_INLINE_CHARS = 4000


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


def _declares_data_integrity(item: BenchmarkItem) -> bool:
    """Return whether the participant item includes a source-data quality plan."""

    policy_path = item.visible_dir / DATA_INTEGRITY_POLICY_FILENAME_V1
    if not policy_path.is_file() and item.suite_dir is not None:
        manifest_path = item.suite_dir / "manifest.json"
        if manifest_path.is_file():
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                isinstance(manifest_payload, dict)
                and manifest_payload.get("schema_id") == "ariadne.benchmark_role_manifest/v1"
            ):
                release = load_participant_release(item.suite_dir)
                task_id = item.task_id or item.item_id
                task = next((candidate for candidate in release.tasks if candidate.task_id == task_id), None)
                if task is None:
                    raise ValueError(f"TrialEval task is absent from its participant release: {task_id}")
                policies = tuple(
                    material
                    for material in task.available_materials
                    if material.artifact_id == "data_integrity_policy"
                )
                if len(policies) > 1:
                    raise ValueError(f"TrialEval task declares multiple data-integrity policies: {task_id}")
                if policies:
                    policy_path = item.suite_dir / PurePosixPath(policies[0].relative_path).with_suffix(".json")
    if not policy_path.is_file():
        return False
    policy = TrialEvalPublicIntegrityPolicyV1.model_validate_json(policy_path.read_text(encoding="utf-8"))
    if policy.task_id != (item.task_id or item.item_id):
        raise ValueError("TrialEval data-integrity policy does not match the selected task.")
    return True


def _get_tools(analysis_language: AnalysisMode = "python_r") -> list[dict[str, JsonValue]]:
    """Return the shared analysis, workspace and submission tools."""

    return [*analysis_tools(analysis_language), *WORKSPACE_TOOLS, *submission_tools_v1()]


def _get_submission_tools() -> list[dict[str, JsonValue]]:
    """Return the only tools exposed on the terminal provider turn."""

    return list(submission_tools_v1())


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def _turn_budget_instruction(max_turns: int, reporting_interface: ReportingInterfaceV1 = "v1") -> str:
    """State the turn budget and when the analysis tools close, as the run loop enforces it."""

    last_analysis_turn = max_turns - SUBMISSION_ONLY_TURNS
    if last_analysis_turn < 1:
        return (
            f"You have {max_turns} turns total, all reserved for submitting your final report; "
            "you cannot run code or modify files.\n"
        )
    return (
        f"You have {max_turns} turns total. Analysis tools are available only during "
        + (f"turns 1-{last_analysis_turn}. " if last_analysis_turn > 1 else "turn 1. ")
        + f"Save your analysis script and numerical results before turn {last_analysis_turn} ends. "
        f"Turns {last_analysis_turn + 1}-{max_turns} are reserved for submitting your final report; "
        "you cannot run code or modify files then.\n"
        + (
            ""
            if reporting_interface == "v1"
            else f"If your saved script fails when it is rerun at submission and no analysis turn remains, the "
            f"analysis tools reopen for {REPAIR_TURNS} more turns, once, so you can fix it and submit again.\n"
        )
    )


def _run_rules(analysis_language: AnalysisMode, executor_limits: CodeExecutionLimitsV1) -> str:
    """State the run rules the harness enforces but the tools do not announce."""

    entrypoints = " or ".join(
        "scratch/analysis.py" if name == "python" else "scratch/analysis.R"
        for name in analysis_languages(analysis_language)
    )
    return (
        f"Keep exactly one saved script, {entrypoints}. Its rerun sees the supplied files and your .py/.R files in "
        f"scratch/ only, not tables you saved, and must finish within one {executor_limits.timeout_seconds:g}-second "
        "execution. The task ends if an interactive execution exceeds the memory limit or the turns or agent time "
        "run out; a report not yet submitted is then recorded as missing. Model requests and the analysis and "
        "workspace tools use agent time; check_analysis and the checks at submission do not. Tool results longer "
        f"than {_TOOL_OUTPUT_INLINE_CHARS:,} characters show only their first and last parts; the full text (up to "
        f"{executor_limits.output_bytes // 1024:,} KiB per execution) is saved under scratch/outputs/, so print "
        "compact summaries.\n"
    )


def _turn_tag(turn: int, max_turns: int, repair_until: int = 0) -> str:
    """Render the turn counter with the turn after which the analysis tools close."""

    last_analysis_turn = max_turns - SUBMISSION_ONLY_TURNS
    if turn < repair_until:
        window = f"; repair: analysis tools close after turn {repair_until}"
    else:
        window = (
            f"; analysis tools close after turn {last_analysis_turn}"
            if turn <= last_analysis_turn
            else "; submission only"
        )
    return f"\n[turn {turn}/{max_turns}, {max_turns - turn} left{window}]"


def _agent_time_notice(deadline: RuntimeDeadline, *, first: bool) -> str | None:
    """State the agent-time budget in full once, then as a one-line remainder."""

    remaining = deadline.remaining()
    if remaining is None:
        return None
    budget = deadline.budget_seconds
    if not first:
        total = "" if budget is None else f" of {budget:g}"
        return f"[Agent time: {math.floor(remaining)}{total} seconds remaining.]"
    total = "" if budget is None else f"{budget:g} seconds total; "
    return (
        f"[Agent-time budget: {total}{math.floor(remaining)} seconds remaining. Model requests and the analysis and "
        "workspace tools use this time; check_analysis and the checks at submission do not. Time can run out before all "
        "turns are used; a report not submitted by then is recorded as missing.]"
    )


def _build_system_prompt(
    item: BenchmarkItem,
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    procedure_assistance: ProcedureAssistanceV1 = "output_contract_only",
    analysis_specification: TrialEvalAnalysisSpecificationV1 = "locked_sap",
    analysis_language: AnalysisMode = "python_r",
    executor_limits: CodeExecutionLimitsV1 = TRIALEVAL_RELEASE_BUDGET_V1.code_execution,
    reporting_interface: ReportingInterfaceV1 = "v1",
) -> str:
    """Build a method-neutral prompt from the participant-visible surface."""

    if analysis_specification != item.analysis_specification:
        raise ValueError(
            "Requested analysis_specification does not match the task's immutable participant evidence surface."
        )
    _declares_data_integrity(item)
    assistance = procedure_assistance_v1(procedure_assistance)
    if item.workspace_data_paths:
        data_files = list(item.workspace_data_paths)
    elif item.raw_data_dir and item.raw_data_dir.exists():
        data_files = sorted(f"data/raw/{path.name}" for path in item.raw_data_dir.resolve().glob("*.parquet"))
    else:
        data_files = sorted(f"data/{path.name}" for path in item.data_dir.resolve().glob("*.parquet"))
    return (
        "You are the trial statistician for the study described in the documents below. Answer the analysis "
        "request: estimate each requested result with its 95% confidence interval and write a statistical report.\n"
        f"{analysis_instructions(analysis_language, limits=executor_limits, reporting_interface=reporting_interface)}"
        "The analysis request names each result, its measure and unit, in the order of interface/targets.csv. "
        f"Data files: {', '.join(data_files) or 'none'}.\n"
        + (f"{assistance.strip()}\n" if assistance.strip() else "")
        + f"{submission_instruction_v1(analysis_language).strip()}\n"
        + _turn_budget_instruction(max_turns, reporting_interface)
        + _run_rules(analysis_language, executor_limits)
    )


def condition_provenance_v1(
    item: BenchmarkItem,
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    procedure_assistance: ProcedureAssistanceV1 = "output_contract_only",
    analysis_specification: TrialEvalAnalysisSpecificationV1 = "locked_sap",
    analysis_language: AnalysisMode = "python_r",
    executor_limits: CodeExecutionLimitsV1 = TRIALEVAL_RELEASE_BUDGET_V1.code_execution,
    reporting_interface: ReportingInterfaceV1 = "v1",
) -> dict[str, JsonValue]:
    """Return exact hashes for one rendered participant condition."""

    if item.analysis_specification != analysis_specification:
        raise ValueError("Requested analysis specification differs from the immutable participant task.")
    system_prompt = _build_system_prompt(
        item,
        max_turns=max_turns,
        procedure_assistance=procedure_assistance,
        analysis_specification=analysis_specification,
        analysis_language=analysis_language,
        executor_limits=executor_limits,
        reporting_interface=reporting_interface,
    )
    tools = _get_tools(analysis_language)
    return {
        **({} if reporting_interface == "v1" else {"reporting_interface": reporting_interface}),
        "procedure_assistance": procedure_assistance,
        "analysis_specification": analysis_specification,
        "analysis_surface_sha256": participant_analysis_surface_sha256(item),
        "max_turns": max_turns,
        "prompt_set_sha256": prompt_set_sha256_v1(),
        "rendered_system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "tool_schema_sha256": canonical_payload_sha256(cast(JsonValue, tools)),
    }


def _decode_response_v1(
    text: str,
    *,
    label: str,
) -> dict[str, object]:
    """Decode one non-empty Markdown report transport payload."""

    if not text.strip():
        raise ToolInputError(f"{label} requires one non-empty Markdown report.")
    return {"report": text}


def _handle_tool_call(
    tool_call: ToolCall,
    session: CodeExecutionSession,
    verbose: bool,
    analysis_language: AnalysisMode = "python_r",
) -> tuple[str, dict[str, object] | None, CodeExecutionResultV1 | None]:
    """Execute a tool call and preserve any isolated execution result."""
    name = tool_call.name
    args = parse_tool_arguments(tool_call.arguments, tool_name=name)
    if name in {"submit_response", "submit_response_file"}:
        if not isinstance(args.pop("accept_incomplete", False), bool):
            raise ToolInputError("accept_incomplete must be a boolean")

    if name in {"execute_code", "inspect_parquet"}:
        execution = execute_analysis_tool_v1(tool_call, session=session, analysis_language=analysis_language)
        output = _agent_facing_execution_output(execution)
        if verbose:
            detail = args.get("purpose") if name == "execute_code" else args.get("path")
            print(terminal_text(f"  [{name}] {detail or ''}".rstrip()))
            preview = output[:600] + ("..." if len(output) > 600 else "")
            print(terminal_text(f"  Output: {preview}"))
        return output, None, execution

    if name in {"write_workspace_file", "read_workspace_file", "list_workspace_files"}:
        execution = handle_workspace_tool(name=name, arguments=args, session=session)
        return execution.output, None, execution

    elif name == "submit_response":
        if set(args) != {"content"} or not isinstance(args.get("content"), str):
            raise ToolInputError("submit_response accepts only one string content field.")
        answer = _decode_response_v1(cast(str, args["content"]), label="submit_response")
        if verbose:
            print("  [submit_response] Agent submitting report")
        return "Response submitted.", answer, None

    elif name == "submit_response_file":
        if set(args) != {"path"}:
            raise ToolInputError("submit_response_file accepts only the path field.")
        text = read_workspace_submission_text(path=args.get("path"), session=session)
        answer = _decode_response_v1(
            text,
            label="submit_response_file",
        )
        if verbose:
            print("  [submit_response_file] Agent submitting report from scratch/")
        return "Response submitted.", answer, None

    raise ToolInputError(f"Unknown TrialEval tool: {name!r}.")


def _agent_facing_execution_output(result: CodeExecutionResultV1) -> str:
    """Render one execution result without suppressing its failure state."""

    if result.output:
        return result.output
    if result.status == "success":
        return "(code executed successfully; no stdout produced)"
    return f"[{result.status}]"


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------


def run_agent(
    item: BenchmarkItem,
    provider: LLMProvider,
    max_turns: int | None = None,
    temperature: float = 0.0,
    max_tokens: int = TRIALEVAL_RELEASE_BUDGET_V1.maximum_completion_tokens_per_turn,
    verbose: bool = True,
    procedure_assistance: ProcedureAssistanceV1 = "output_contract_only",
    analysis_specification: TrialEvalAnalysisSpecificationV1 | None = None,
    provider_log_path: Path | None = None,
    conversation_log_path: Path | None = None,
    event_log_path: Path | None = None,
    executor_image: str | None = None,
    executor_limits: CodeExecutionLimitsV1 | None = None,
    max_elapsed_seconds: float | None = None,
    item_workspace: Path | None = None,
    max_context_chars: int = TRIALEVAL_RELEASE_BUDGET_V1.maximum_context_characters,
    provider_capability: ProviderModelCapabilityV1 | None = None,
    analysis_language: AnalysisMode = "python_r",
    provider_unavailability_policy: Literal["charge_all_v1", "defer_unanswered_v1"] = "charge_all_v1",
    reporting_interface: ReportingInterfaceV1 = "v1",
) -> dict[str, object]:
    """
    Run the analysis agent against a benchmark item.

    Returns the execution status, canonical submission, turn count, and
    observable conversation trace.
    """
    if max_turns is None:
        max_turns = DEFAULT_MAX_TURNS
    if max_turns < 1:
        raise ValueError("max_turns must be at least 1.")
    if max_elapsed_seconds is not None and max_elapsed_seconds <= 0.0:
        raise ValueError("max_elapsed_seconds must be positive when provided.")
    if max_context_chars < 1:
        raise ValueError("max_context_chars must be at least 1.")
    if analysis_specification is None:
        analysis_specification = item.analysis_specification
    executor_limits = executor_limits or TRIALEVAL_RELEASE_BUDGET_V1.code_execution
    deadline = RuntimeDeadline.after(
        max_elapsed_seconds, label="TrialEval item", provider_unavailability_policy=provider_unavailability_policy
    )

    system_prompt = _build_system_prompt(
        item,
        max_turns=max_turns,
        procedure_assistance=procedure_assistance,
        analysis_specification=analysis_specification,
        analysis_language=analysis_language,
        executor_limits=executor_limits,
        reporting_interface=reporting_interface,
    )
    visible_context = load_visible_context(item)
    _declares_data_integrity(item)
    condition_provenance: dict[str, JsonValue] = {}

    messages: list[dict[str, JsonValue]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": visible_context},
    ]
    # Snapshot the rendered prompts so the saved transcript stays self-
    # contained even if agent.py or the bundle drifts later. These are turn-0
    # entries, never sent to the LLM separately (they're already in `messages`).
    conversation_log: list[dict[str, object]] = [
        {"role": "system", "turn": 0, "content": system_prompt},
        {"role": "user", "turn": 0, "content": visible_context},
    ]
    task_id = item.task_id or item.item_id
    source_artifact_path = (
        conversation_log_path.as_posix()
        if conversation_log_path is not None
        else f"runner://trialeval/{task_id}/conversation"
    )
    runtime_events: list[BenchmarkRuntimeTraceEventV1] = []
    event_log_initialized = False

    def _event(**values: object) -> None:
        event_index = len(runtime_events)
        message_index = values.get("conversation_message_index")
        conversation_message = (
            cast(JsonValue, conversation_log[message_index]) if isinstance(message_index, int) else None
        )
        event_type = cast(RuntimeEventTypeV1, values["event_type"])
        terminal_status = values.get("terminal_status")
        failure_type = values.get("failure_type")
        source_payload = runtime_event_source_payload_v1(
            benchmark="trialeval",
            task_id=task_id,
            program_id=None,
            scenario_id=None,
            objective_id=None,
            phase_id="task",
            step_id="analysis",
            event_type=event_type,
            terminal_status=cast("Literal['completed', 'failed'] | None", terminal_status),
            failure_type=cast(BenchmarkRuntimeFailureTypeV1 | None, failure_type),
            conversation_message=conversation_message,
        )
        source_record = {
            "benchmark": "trialeval",
            "task_id": task_id,
            "event_index": event_index,
            "phase_id": "task",
            "step_id": "analysis",
            **values,
        }
        event = BenchmarkRuntimeTraceEventV1.model_validate(
            {
                **source_record,
                "event_id": f"trialeval:{task_id}:{event_index:06d}",
                "timestamp": datetime.now(UTC),
                "source_artifact_path": source_artifact_path,
                "source_payload_sha256": canonical_payload_sha256(source_payload),
            }
        )
        runtime_events.append(event)
        if event_log_path is not None and event_log_initialized:
            with deadline.harness_work():
                append_jsonl_model(event_log_path, event)

    _event(event_type="step_started")
    _event(event_type="prompt", conversation_message_index=1)
    if conversation_log_path is not None:
        conversation_log_path.parent.mkdir(parents=True, exist_ok=True)
        conversation_log_path.write_text(
            json.dumps(conversation_log, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if event_log_path is not None:
        event_log_path.parent.mkdir(parents=True, exist_ok=True)
        event_log_path.write_text(
            "".join(json.dumps(event.model_dump(mode="json"), sort_keys=True) + "\n" for event in runtime_events),
            encoding="utf-8",
        )
        event_log_initialized = True

    terminal_emitted = False

    def _terminal(
        *,
        status: Literal["completed", "failed"],
        failure_type: BenchmarkRuntimeFailureTypeV1 | None = None,
    ) -> None:
        nonlocal terminal_emitted
        if terminal_emitted:
            raise RuntimeError("TrialEval assignment emitted more than one terminal runtime event.")
        _event(
            event_type="step_terminal",
            terminal_status=status,
            failure_type=failure_type,
        )
        terminal_emitted = True

    def _persist_conversation() -> None:
        if conversation_log_path is None:
            return
        with deadline.harness_work():
            conversation_log_path.write_text(
                json.dumps(conversation_log, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    if item_workspace is None:
        log_path = conversation_log_path or event_log_path or provider_log_path
        if log_path is None:
            raise ValueError("item_workspace is required when no persistent item log path is configured.")
        log_stem = Path(log_path).stem
        for suffix in ("_conversation", "_events", "_provider_responses"):
            log_stem = log_stem.removesuffix(suffix)
        item_workspace = Path(log_path).parent / f"{log_stem}_workspace"
    workspace_root = Path(item_workspace)
    evidence_root = workspace_root / "item"
    session: CodeExecutionSession | None = None
    result_dict: dict[str, object] | None = None
    in_flight: BaseException | None = None
    try:
        try:
            with deadline.harness_work():
                condition_provenance = condition_provenance_v1(
                    item,
                    max_turns=max_turns,
                    procedure_assistance=procedure_assistance,
                    analysis_specification=analysis_specification,
                    analysis_language=analysis_language,
                    executor_limits=executor_limits,
                    reporting_interface=reporting_interface,
                )
                tools = _get_tools(analysis_language)
                workspace_root.mkdir(parents=True, exist_ok=True)
                if evidence_root.exists():
                    if not evidence_root.is_dir():
                        raise FileExistsError(f"TrialEval item workspace is not a directory: {evidence_root}")
                else:
                    stage_participant_evidence(item, evidence_root)
                stage_numerical_result_template(evidence_root)
                from trialagentbench_verifier.numerical import stage_reporting_interface

                stage_reporting_interface(
                    evidence_root,
                    tuple(cast(list[str], item.task.get("target_ids", []))),
                    interface=reporting_interface,
                )
                diagnostic_path = evidence_root / "diagnostic_dictionary.json"
                if diagnostic_path.is_file():
                    load_participant_diagnostic_dictionary(evidence_root)
                session = DockerCodeSession(
                    cwd=evidence_root,
                    image=executor_image,
                    limits=executor_limits,
                )
                # Replay checks reuse the immutable image, even if its tag moves mid-item.
                executor_image = session.image_id
                deadline.run_blocking(
                    partial(session.execute, "DATA_DIR = 'data'"),
                    operation_name="TrialEval executor initialization",
                    on_timeout=session.close,
                )
                if item.raw_data_dir:
                    deadline.run_blocking(
                        partial(session.execute, "RAW_DATA_DIR = 'data/raw'"),
                        operation_name="TrialEval executor initialization",
                        on_timeout=session.close,
                    )

            # Under the clarified interface a failed replay at submission can extend the analysis window once.
            turn_limit, repair_until, repair_granted = max_turns, 0, False
            turn = -1
            while turn + 1 < turn_limit:
                turn += 1
                notice = _agent_time_notice(deadline, first=turn == 0)
                if notice is not None:
                    messages.append({"role": "user", "content": notice})
                    conversation_log.append({"role": "user", "turn": turn + 1, "content": notice})
                    _persist_conversation()
                    _event(event_type="prompt", conversation_message_index=len(conversation_log) - 1)
                submission_only_turn = turn >= max(0, max_turns - SUBMISSION_ONLY_TURNS, repair_until)
                if submission_only_turn:
                    reminder = (
                        FINAL_TURN_SUBMISSION_REMINDER if turn == turn_limit - 1 else SUBMISSION_WINDOW_REMINDER
                    ) + INCOMPLETE_SUBMISSION_NOTE
                    messages.append(
                        {
                            "role": "user",
                            "content": reminder,
                        }
                    )
                    conversation_log.append(
                        {
                            "role": "user",
                            "turn": turn + 1,
                            "content": reminder,
                        }
                    )
                    _persist_conversation()
                    _event(
                        event_type="prompt",
                        conversation_message_index=len(conversation_log) - 1,
                    )
                if verbose:
                    print(f"\n{'=' * 60}")
                    print(f"Turn {turn + 1}/{max_turns}")
                    print(f"{'=' * 60}")

                request_tools = _get_submission_tools() if submission_only_turn else tools
                request_tool_names = offered_tool_names_v1(request_tools)
                required: tuple[int, ...] = (len(messages) - 1,) if submission_only_turn else ()
                if notice is not None:
                    required = (len(messages) - 2, len(messages) - 1) if submission_only_turn else (len(messages) - 1,)
                turn_result = run_agent_turn_v1(
                    provider=provider,
                    messages=messages,
                    session=session,
                    active_prompt_index=1,
                    max_context_chars=max_context_chars,
                    provider_capability=provider_capability,
                    required_message_indices=required,
                    tools=request_tools,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                    timeout_seconds=deadline.remaining(),
                    deadline=deadline,
                    tool_choice="required" if submission_only_turn else "auto",
                    provider_log_path=provider_log_path,
                    benchmark="trialeval",
                    unit_id=task_id,
                    phase_id="task",
                    step_id="analysis",
                    turn_index=turn + 1,
                )
                response = turn_result.response
                turn_elapsed = turn_result.elapsed_seconds

                messages.append(turn_result.message)

                # Log the assistant turn with usage + wall time and, as TrialDev does, the raw
                # assistant message (tool calls included) exactly as it re-enters the context.
                conversation_log.append(
                    {
                        "role": "assistant",
                        "turn": turn + 1,
                        "content": response.content,
                        "message": turn_result.message,
                        "usage": response.usage,
                        "elapsed_sec": round(turn_elapsed, 3),
                    }
                )
                _persist_conversation()
                _event(event_type="assistant_message", conversation_message_index=len(conversation_log) - 1)
                if response.model_outcome is not None:
                    raise ModelResponseNoncompletion(response.model_outcome)
                if response.content and verbose:
                    preview = response.content[:400] + ("..." if len(response.content) > 400 else "")
                    print(terminal_text(f"[Reasoning]: {preview}"))

                if response.tool_calls:
                    for tool_call_index, tc in enumerate(response.tool_calls):
                        deadline.remaining()
                        tool_started = time.monotonic()
                        parsed_arguments: dict[str, object] = {}
                        execution: CodeExecutionResultV1 | None
                        try:
                            if tc.name not in request_tool_names:
                                raise ToolInputError(f"tool {tc.name!r} was not offered on this turn")
                            parsed_arguments = parse_tool_arguments(tc.arguments, tool_name=tc.name)
                            if tc.name == "check_analysis":
                                context = NumericalReportingContext(
                                    checkpoint_id=task_id,
                                    target_ids=tuple(cast(list[str], item.task.get("target_ids", []))),
                                    evidence_artifacts=participant_evidence_artifacts(item),
                                )
                                with deadline.harness_work():
                                    execution, session = check_analysis_tool_v1(
                                        tc,
                                        session=session,
                                        workspace=evidence_root,
                                        context=context,
                                        executor_limits=executor_limits,
                                        executor_image=executor_image,
                                        analysis_language=analysis_language,
                                        reporting_interface=reporting_interface,
                                    )
                                result_str, answer = execution.output, None
                            else:
                                result_str, answer, execution = deadline.run_blocking(
                                    partial(
                                        _handle_tool_call,
                                        tc,
                                        session,
                                        verbose,
                                        analysis_language=analysis_language,
                                    ),
                                    operation_name=f"TrialEval tool {tc.name}",
                                    on_timeout=session.close,
                                )
                                if execution is not None and execution.status == "session_terminated":
                                    raise CodeExecutionSessionTerminated(
                                        execution.output or "Participant execution session terminated"
                                    )
                            if tc.name in {"submit_response", "submit_response_file"}:
                                accept_incomplete = parsed_arguments.get("accept_incomplete", False)
                                if not isinstance(accept_incomplete, bool):
                                    raise ToolInputError("accept_incomplete must be a boolean")
                                with deadline.harness_work():
                                    execution, session = check_analysis_tool_v1(
                                        ToolCall(id=tc.id, name="check_analysis", arguments="{}"),
                                        session=session,
                                        workspace=evidence_root,
                                        context=NumericalReportingContext(
                                            checkpoint_id=task_id,
                                            target_ids=tuple(cast(list[str], item.task.get("target_ids", []))),
                                            evidence_artifacts=participant_evidence_artifacts(item),
                                        ),
                                        executor_limits=executor_limits,
                                        executor_image=executor_image,
                                        analysis_language=analysis_language,
                                        reporting_cutoff=True,
                                        reporting_interface=reporting_interface,
                                    )
                                no_analysis_turn = turn + 1 >= max(max_turns - SUBMISSION_ONLY_TURNS, repair_until)
                                if (
                                    execution.status != "success"
                                    and not accept_incomplete
                                    and reporting_interface == "clarified"
                                    and no_analysis_turn
                                    and not repair_granted
                                ):
                                    repair_granted = True
                                    repair_until = turn + 1 + REPAIR_TURNS
                                    turn_limit = max(turn_limit, repair_until + 1)
                                    raise ToolInputError(
                                        execution.output
                                        + f" The analysis tools reopen for {REPAIR_TURNS} repair turns: fix the saved "
                                        "script, run check_analysis, then submit again."
                                    )
                                if execution.status != "success" and not accept_incomplete:
                                    raise ToolInputError(
                                        execution.output
                                        + (
                                            " No analysis turn remains: resubmit with accept_incomplete=true, or the "
                                            "report is recorded as missing."
                                            if turn + 1 >= max_turns - SUBMISSION_ONLY_TURNS
                                            else " Fix the saved script or submit with accept_incomplete=true."
                                        )
                                    )
                            tool_status = "observed"
                        except ToolInputError as exc:
                            result_str = f"Tool input rejected: {exc}"
                            answer = None
                            execution = None
                            tool_status = "invalid"
                        tool_elapsed = time.monotonic() - tool_started
                        with deadline.harness_work():
                            result_str = persist_bulky_tool_output(
                                result_str,
                                session=session,
                                artifact_id=f"turn-{turn + 1}-{tool_call_index + 1}",
                                inline_chars=_TOOL_OUTPUT_INLINE_CHARS,
                            )
                        result_str += _turn_tag(turn + 1, turn_limit, repair_until)

                        conversation_log.append(
                            {
                                "role": "tool",
                                "turn": turn + 1,
                                "tool_call_id": tc.id,
                                "tool": tc.name,
                                "args": parsed_arguments,
                                "output": result_str,
                                "elapsed_sec": round(tool_elapsed, 3),
                            }
                        )
                        _persist_conversation()
                        message_index = len(conversation_log) - 1
                        event_type = "submission" if answer is not None else "tool_call"
                        file_accessed = None
                        if tc.name in {"execute_code", "check_analysis"}:
                            event_type = "code_execution"
                        elif tc.name == "inspect_parquet":
                            event_type = "file_inspection"
                            path_value = parsed_arguments.get("path")
                            file_accessed = path_value if isinstance(path_value, str) and path_value else None
                        elif tc.name == "read_workspace_file":
                            event_type = "file_inspection"
                            workspace_path_value = parsed_arguments.get("path")
                            file_accessed = (
                                workspace_path_value
                                if isinstance(workspace_path_value, str) and workspace_path_value
                                else None
                            )
                        elif tc.name == "write_workspace_file":
                            workspace_path_value = parsed_arguments.get("path")
                            file_accessed = (
                                (
                                    workspace_path_value
                                    if workspace_path_value.startswith("scratch/")
                                    else f"scratch/{workspace_path_value}"
                                )
                                if isinstance(workspace_path_value, str) and workspace_path_value
                                else None
                            )
                        elif tc.name == "submit_response_file":
                            workspace_path_value = parsed_arguments.get("path")
                            file_accessed = (
                                (
                                    workspace_path_value
                                    if workspace_path_value.startswith("scratch/")
                                    else f"scratch/{workspace_path_value}"
                                )
                                if isinstance(workspace_path_value, str) and workspace_path_value
                                else None
                            )
                        # A read's success travels with its event, so failed reads are not evidence use.
                        event_execution = execution if event_type in {"code_execution", "file_inspection"} else None
                        _event(
                            event_type=event_type,
                            conversation_message_index=message_index,
                            tool_call_index=tool_call_index,
                            tool_call_id=tc.id,
                            tool_name=tc.name,
                            file_accessed=file_accessed,
                            status=tool_status,
                            execution_status=event_execution.status if event_execution is not None else None,
                            execution_language=event_execution.language if event_execution is not None else None,
                            elapsed_seconds=event_execution.elapsed_seconds if event_execution is not None else None,
                            cpu_seconds=event_execution.cpu_seconds if event_execution is not None else None,
                            session_peak_memory_bytes=(
                                event_execution.session_peak_memory_bytes if event_execution is not None else None
                            ),
                            session_peak_processes=(
                                event_execution.session_peak_processes if event_execution is not None else None
                            ),
                            output_truncated=event_execution.output_truncated if event_execution is not None else None,
                        )
                        _event(
                            event_type="tool_result",
                            conversation_message_index=message_index,
                            tool_call_id=tc.id,
                            tool_name=tc.name,
                            status=tool_status,
                            execution_status=execution.status if execution is not None else None,
                            execution_language=execution.language if execution is not None else None,
                            elapsed_seconds=execution.elapsed_seconds if execution is not None else None,
                            cpu_seconds=execution.cpu_seconds if execution is not None else None,
                            session_peak_memory_bytes=(
                                execution.session_peak_memory_bytes if execution is not None else None
                            ),
                            session_peak_processes=(
                                execution.session_peak_processes if execution is not None else None
                            ),
                            output_truncated=execution.output_truncated if execution is not None else None,
                        )

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": result_str,
                            }
                        )

                        if answer is not None:
                            _terminal(status="completed")
                            result_dict = {
                                "status": "success",
                                "turns_used": turn + 1,
                                "conversation": conversation_log,
                                "report": answer["report"],
                                "condition_provenance": condition_provenance,
                                "events": [event.model_dump(mode="json") for event in runtime_events],
                            }
                            return result_dict

                elif not response.content:
                    messages.append(
                        {
                            "role": "user",
                            "content": "Please continue your analysis using the available tools.",
                        }
                    )

            _terminal(status="failed", failure_type="turn_limit_no_submission")
            result_dict = {
                "status": "max_turns_reached",
                "failure_type": "turn_limit_no_submission",
                "report": None,
                "turns_used": turn_limit,
                "conversation": conversation_log,
                "condition_provenance": condition_provenance,
                "events": [event.model_dump(mode="json") for event in runtime_events],
            }
            return result_dict

        finally:
            in_flight = sys.exception()
            if session is not None:
                try:
                    session.close()
                except subprocess.TimeoutExpired as close_error:
                    if common_runtime_failure_type_v1(close_error) != "evaluator_error":
                        raise
                    if in_flight is None and not terminal_emitted:
                        raise
                    logger.warning("Qualified executor teardown timeout suppressed; preserving the existing outcome.")

    except (
        HarnessConfigurationError,
        CodeExecutionSessionTerminated,
        RuntimeDeadlineExceeded,
        ModelResponseNoncompletion,
        subprocess.TimeoutExpired,
    ) as exception:
        # Participant session loss (including its own code/OOM kill) is scored
        # noncompletion, never an infrastructure retry. The item's time limit
        # is also scored. A typed host-control timeout retains an infrastructure
        # primary; an unqualified subprocess timeout cannot acquire that origin.
        failure_type = common_runtime_failure_type_v1(exception)
        if isinstance(exception, subprocess.TimeoutExpired) and failure_type != "evaluator_error":
            raise
        if terminal_emitted:
            # A completed submission survives a qualified host-control teardown fault.
            if not isinstance(exception, subprocess.TimeoutExpired):
                raise
            assert result_dict is not None
            return result_dict
        result_dict = {
            "status": "failed",
            "failure_type": failure_type,
            "report": None,
            "turns_used": sum(1 for entry in conversation_log if entry.get("role") == "assistant"),
            "conversation": conversation_log,
            "condition_provenance": condition_provenance,
            "events": [event.model_dump(mode="json") for event in runtime_events],
        }
        return result_dict
    finally:
        if not terminal_emitted:
            error = in_flight if in_flight is not None else sys.exception()
            # An in-flight outcome retains its classification even if unqualified cleanup fails.
            if error is not None:
                failure_type = common_runtime_failure_type_v1(error)
            else:
                assert result_dict is not None
                failure_type = cast(BenchmarkRuntimeFailureTypeV1, result_dict["failure_type"])
            _terminal(status="failed", failure_type=failure_type)
        if result_dict is not None:
            result_dict["events"] = [event.model_dump(mode="json") for event in runtime_events]
