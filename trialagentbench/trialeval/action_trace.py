"""Action-trace projection for stored TrialEval results."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from pydantic.types import JsonValue

from trialagentbench.analysis.evidence_use import evidence_rows_from_events
from trialagentbench.analysis.failure_cascade import phase_failure_cascade
from trialagentbench.analysis.run_identity import require_unique_run_ids
from trialagentbench.analysis.trialeval_score_rows import iter_trialeval_score_rows
from trialagentbench.contracts.core.runs import TrialEvalItemResultV1
from trialagentbench.contracts.trace.observable import (
    BenchmarkRuntimeTraceEventV1,
    EvidenceUseRowV1,
    FailureCascadeRowV1,
    ModelActionTraceEventV1,
    TraceFeatureRowV1,
    TrialEvalTraceInputV1,
    is_successful_file_read,
    runtime_event_source_payload_v1,
)
from trialagentbench.io import read_json_model
from trialagentbench.io.checksums import canonical_payload_sha256, sha256_file


def _runtime_events(path: Path) -> tuple[BenchmarkRuntimeTraceEventV1, ...]:
    events: list[BenchmarkRuntimeTraceEventV1] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                events.append(BenchmarkRuntimeTraceEventV1.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"Invalid TrialEval runtime event at {path}:{line_number}") from exc
    if [event.event_index for event in events] != list(range(len(events))):
        raise ValueError(f"TrialEval runtime event indices must be contiguous: {path}")
    if len({event.event_id for event in events}) != len(events):
        raise ValueError(f"TrialEval runtime event IDs must be unique: {path}")
    return tuple(events)


def _validate_runtime_events(
    events: tuple[BenchmarkRuntimeTraceEventV1, ...],
    *,
    task_id: str,
    conversation_path: Path,
) -> None:
    if not events:
        raise ValueError(f"TrialEval runtime event stream is empty: {conversation_path}")
    if events[0].event_type != "step_started":
        raise ValueError(f"TrialEval runtime events must start with step_started: {conversation_path}")
    for event in events:
        if event.benchmark != "trialeval" or event.task_id != task_id:
            raise ValueError(f"TrialEval runtime event identity mismatch: {conversation_path}")
    terminals = [event for event in events if event.event_type == "step_terminal"]
    if len(terminals) != 1 or events[-1] is not terminals[0]:
        raise ValueError(f"TrialEval runtime events require exactly one final step_terminal: {conversation_path}")
    messages = json.loads(conversation_path.read_text(encoding="utf-8"))
    if not isinstance(messages, list):
        raise ValueError(f"TrialEval conversation must be a list: {conversation_path}")
    for event in events:
        if event.phase_id != "task" or event.step_id != "analysis":
            raise ValueError(f"TrialEval runtime event route mismatch: {conversation_path}")
        if event.event_type in {"step_started", "step_terminal"}:
            if event.conversation_message_index is not None:
                raise ValueError("TrialEval lifecycle events must not reference a conversation message.")
            source_payload = runtime_event_source_payload_v1(
                benchmark=event.benchmark,
                task_id=event.task_id,
                program_id=event.program_id,
                scenario_id=event.scenario_id,
                objective_id=event.objective_id,
                phase_id=event.phase_id,
                step_id=event.step_id,
                event_type=event.event_type,
                terminal_status=event.terminal_status,
                failure_type=event.failure_type,
                conversation_message=None,
            )
            if canonical_payload_sha256(source_payload) != event.source_payload_sha256:
                raise ValueError(f"TrialEval runtime lifecycle payload hash mismatch: {conversation_path}")
            continue
        index = event.conversation_message_index
        if index is None or index >= len(messages) or not isinstance(messages[index], dict):
            raise ValueError(f"TrialEval runtime event references a missing conversation message: {conversation_path}")
        message = messages[index]
        source_payload = runtime_event_source_payload_v1(
            benchmark=event.benchmark,
            task_id=event.task_id,
            program_id=event.program_id,
            scenario_id=event.scenario_id,
            objective_id=event.objective_id,
            phase_id=event.phase_id,
            step_id=event.step_id,
            event_type=event.event_type,
            terminal_status=event.terminal_status,
            failure_type=event.failure_type,
            conversation_message=cast(JsonValue, message),
        )
        if canonical_payload_sha256(source_payload) != event.source_payload_sha256:
            raise ValueError(f"TrialEval runtime conversation payload hash mismatch: {conversation_path}")
        expected_role = "user" if event.event_type == "prompt" else "assistant"
        if event.event_type in {"tool_call", "tool_result", "code_execution", "file_inspection", "submission"}:
            expected_role = "tool"
            if str(message.get("tool_call_id") or "") != str(event.tool_call_id or ""):
                raise ValueError(f"TrialEval runtime tool id mismatch: {conversation_path}")
            if str(message.get("tool") or "") != str(event.tool_name or ""):
                raise ValueError(f"TrialEval runtime tool name mismatch: {conversation_path}")
        if str(message.get("role") or "") != expected_role:
            raise ValueError(f"TrialEval runtime event role mismatch: {conversation_path}")


def _require_submission_terminal(
    events: tuple[BenchmarkRuntimeTraceEventV1, ...],
    *,
    submission_expected: bool,
    path: Path,
) -> None:
    terminal = events[-1]
    submissions = [event for event in events if event.event_type == "submission" and event.status == "observed"]
    if submission_expected:
        if len(submissions) != 1:
            raise ValueError(f"TrialEval submitted result requires exactly one runner-native submission event: {path}")
        if terminal.terminal_status != "completed":
            raise ValueError(f"TrialEval submitted result requires a completed runner-native terminal event: {path}")
    elif submissions:
        raise ValueError(f"TrialEval scorer state says no submission but runner telemetry contains one: {path}")


def _submission_transport(run_dir: Path, *unit_ids: str) -> str:
    candidates = tuple(
        dict.fromkeys(
            parent / f"{unit_id}_events.jsonl"
            for unit_id in unit_ids
            for parent in (run_dir / "logs", run_dir / "events")
        )
    )
    observed_paths = [path for path in candidates if path.is_file()]
    if not observed_paths:
        return "not_observed"
    if len(observed_paths) > 1:
        raise ValueError(f"TrialEval unit has multiple runtime event sources: {observed_paths}")
    runtime_path = observed_paths[0]
    submission_tools = {
        event.tool_name
        for event in _runtime_events(runtime_path)
        if event.event_type == "submission" and event.tool_name is not None
    }
    if len(submission_tools) > 1:
        raise ValueError(f"TrialEval unit has conflicting submission transports: {runtime_path}")
    if not submission_tools:
        return "not_observed"
    tool_name = next(iter(submission_tools))
    if tool_name == "submit_response":
        return "direct"
    if tool_name == "submit_response_file":
        return "file"
    raise ValueError(f"TrialEval unit has an unknown submission tool {tool_name!r}: {runtime_path}")


def collect_trialeval_trace_inputs(run_dirs: list[Path]) -> tuple[TrialEvalTraceInputV1, ...]:
    """Load report outputs without deriving prose semantics."""

    run_ids = require_unique_run_ids(run_dirs, suite="trialeval")
    inputs: list[TrialEvalTraceInputV1] = []
    for raw_run_dir in sorted(run_dirs):
        run_dir = raw_run_dir.resolve()
        run_id = run_ids[run_dir]
        paths = sorted((run_dir / "items").glob("*.json"))
        for path in paths:
            item_record = read_json_model(TrialEvalItemResultV1, path)
            scored_task_id = item_record.scores.task_id if item_record.scores is not None else None
            task_id = scored_task_id or item_record.item_id
            evidence_factors = item_record.run_config.task_evidence_factors[item_record.item_id]
            provenance = item_record.agent_output.condition_provenance
            inputs.append(
                TrialEvalTraceInputV1(
                    run_id=run_id,
                    model_id=item_record.run_config.model,
                    task_id=task_id,
                    context_tier=evidence_factors.context_configuration,
                    data_preparation=evidence_factors.data_preparation,
                    analysis_specification=evidence_factors.analysis_specification,
                    procedure_assistance=provenance.procedure_assistance,
                    submission_transport=_submission_transport(run_dir, task_id, item_record.item_id),
                    source_path=path.as_posix(),
                    narrative_report=item_record.agent_output.report,
                )
            )
    return tuple(inputs)


def collect_trialeval_action_trace(
    run_dirs: list[Path],
) -> tuple[
    list[ModelActionTraceEventV1],
    list[TraceFeatureRowV1],
    list[EvidenceUseRowV1],
    list[FailureCascadeRowV1],
]:
    """Collect TrialEval action-trace rows from stored result directories."""
    run_ids = require_unique_run_ids(run_dirs, suite="trialeval")
    trace_inputs = {
        (trace_input.run_id, trace_input.task_id): trace_input
        for trace_input in collect_trialeval_trace_inputs(run_dirs)
        if trace_input.assignment_id is None
    }
    events: list[ModelActionTraceEventV1] = []
    features: list[TraceFeatureRowV1] = []
    evidence: list[EvidenceUseRowV1] = []
    cascades: list[FailureCascadeRowV1] = []
    for raw_run_dir in sorted(run_dirs):
        run_dir = raw_run_dir.resolve()
        run_label = run_ids[run_dir]
        for row in iter_trialeval_score_rows(run_dir):
            task_id = row.task_id
            model_id = row.model_id
            validates = (
                row.outcome.question_alignment.status == "observed"
                and row.outcome.evidence_eligibility.status == "satisfied"
                and row.outcome.reproducibility.status == "satisfied"
            )
            complete_scoreable = row.outcome.estimand_performance.status == "observed"
            endpoint_state = "valid" if complete_scoreable else "failed"
            source_path = row.source_json_path
            source_artifact = Path(source_path)
            if not source_artifact.is_file():
                raise FileNotFoundError(f"TrialEval score source artifact is missing: {source_artifact}")
            event = ModelActionTraceEventV1(
                event_id=f"trialeval:{run_label}:{task_id}:grade",
                timestamp=datetime.fromtimestamp(0, tz=UTC),
                benchmark="trialeval",
                model_id=model_id,
                run_id=run_label,
                task_id=task_id,
                event_index=0,
                event_type="grade_link",
                source_path=source_path,
                source_artifact_path=source_path,
                source_payload_sha256=sha256_file(source_artifact),
                status="observed",
                content_chars=len(source_artifact.read_text(encoding="utf-8")),
            )
            events.append(event)

            trace_input = trace_inputs.get((run_label, task_id))
            if trace_input is None:
                raise ValueError(f"TrialEval score lacks a typed trace input: {run_label}:{task_id}")
            runtime_path = run_dir / "logs" / f"{task_id}_events.jsonl"
            conversation_path = run_dir / "logs" / f"{task_id}_conversation.json"
            if not runtime_path.is_file():
                raise FileNotFoundError(f"TrialEval score row requires runner-native runtime events: {runtime_path}")
            if not conversation_path.is_file():
                raise FileNotFoundError(f"TrialEval runtime events require a conversation: {conversation_path}")
            runtime_events = _runtime_events(runtime_path)
            _validate_runtime_events(runtime_events, task_id=task_id, conversation_path=conversation_path)
            _require_submission_terminal(
                runtime_events,
                submission_expected=trace_input.submission_transport != "not_observed",
                path=runtime_path,
            )
            event_type_map = {
                "step_started": "state_transition",
                "prompt": "prompt",
                "assistant_message": "assistant_message",
                "tool_call": "tool_call",
                "tool_result": "tool_result",
                "code_execution": "code_execution",
                "file_inspection": "file_inspection",
                "submission": "submission",
                "step_terminal": "state_transition",
            }
            for runtime_event in runtime_events:
                events.append(
                    ModelActionTraceEventV1(
                        event_id=runtime_event.event_id,
                        timestamp=runtime_event.timestamp,
                        benchmark="trialeval",
                        model_id=model_id,
                        run_id=run_label,
                        task_id=task_id,
                        phase_id=runtime_event.phase_id,
                        step_id=runtime_event.step_id,
                        turn_index=runtime_event.conversation_message_index,
                        event_index=runtime_event.event_index,
                        event_type=event_type_map[runtime_event.event_type],
                        source_path=runtime_path.as_posix(),
                        source_artifact_path=runtime_event.source_artifact_path,
                        source_payload_sha256=runtime_event.source_payload_sha256,
                        tool_call_id=runtime_event.tool_call_id,
                        tool_name=runtime_event.tool_name,
                        file_accessed=runtime_event.file_accessed,
                        elapsed_sec=runtime_event.elapsed_seconds,
                        execution_status=runtime_event.execution_status,
                        execution_language=runtime_event.execution_language,
                        cpu_seconds=runtime_event.cpu_seconds,
                        session_peak_memory_bytes=runtime_event.session_peak_memory_bytes,
                        session_peak_processes=runtime_event.session_peak_processes,
                        output_truncated=runtime_event.output_truncated,
                        status=runtime_event.status,
                        failure_code=runtime_event.failure_type,
                    )
                )

            unit_events = [e for e in events if e.run_id == run_label and e.task_id == task_id]
            feature = TraceFeatureRowV1(
                benchmark="trialeval",
                model_id=model_id,
                run_id=run_label,
                task_id=task_id,
                phase_id="task",
                trace_coverage_status=(
                    "full_conversation_trace"
                    if runtime_events
                    else ("submission_only_trace" if validates else "scorer_only_trace")
                ),
                inspected_public_data=any(is_successful_file_read(event) for event in unit_events),
                executed_code=any(event.event_type == "code_execution" for event in unit_events),
                submitted_answer=trace_input.answer_present or validates,
                submission_transport=trace_input.submission_transport,
                context_tier=trace_input.context_tier,
                data_preparation=trace_input.data_preparation,
                analysis_specification=trace_input.analysis_specification,
                procedure_assistance=trace_input.procedure_assistance,
                score_link_id=f"{run_label}:{task_id}",
                endpoint_valid=complete_scoreable,
                endpoint_state=endpoint_state,
                execution_status=row.observed_attempt_outcome,
            )
            features.append(feature)
            cascades.append(phase_failure_cascade(feature))
    evidence = evidence_rows_from_events(events)
    return events, features, evidence, cascades


def discover_trialeval_run_dirs(root: Path) -> list[Path]:
    """Discover persisted TrialEval run directories."""

    configured = {path.parent for path in root.rglob("run_config.json") if (path.parent / "items").is_dir()}
    return sorted(configured)


__all__ = [
    "collect_trialeval_action_trace",
    "collect_trialeval_trace_inputs",
    "discover_trialeval_run_dirs",
]
