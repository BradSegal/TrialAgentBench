"""Observable action traces for canonical TrialDev task runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from pydantic.types import JsonValue
from trialagentbench_verifier.io import trialdev_task_run_paths
from trialagentbench_verifier.response import TrialDevRunRankingV1

from trialagentbench.analysis.evidence_use import evidence_rows_from_events
from trialagentbench.contracts.trace.observable import (
    BenchmarkRuntimeTraceEventV1,
    EvidenceUseRowV1,
    ModelActionTraceEventV1,
    TraceEndpointStateV1,
    TraceFeatureRowV1,
    TrialDevDecisionRowV1,
    is_successful_file_read,
    runtime_event_source_payload_v1,
)
from trialagentbench.contracts.trialdev.run import TrialDevRunInvocationV1, TrialDevTaskRunV1
from trialagentbench.io import canonical_payload_sha256, read_json, read_json_model, read_jsonl_models, sha256_file
from trialagentbench.ports import CodeExecutionResultV1
from trialagentbench.trialdev.actions import TrialDevActionRequest, TrialDevProgrammeStatus, trialdev_action_names
from trialagentbench.trialdev.resources import TrialDevBudgetLedger
from trialagentbench.trialdev.run import (
    TrialDevAnalysisRun,
    TrialDevProgrammeRun,
    verify_trialdev_task_run_artifacts_v1,
)
from trialagentbench.trialdev.task_agent import TrialDevAgentCheckpoint


def _json(path: Path) -> JsonValue:
    return cast(JsonValue, read_json(path))


def _conversation(path: Path) -> list[dict[str, Any]]:
    payload = _json(path)
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError(f"TrialDev conversation is not a message array: {path}")
    return cast(list[dict[str, Any]], payload)


def _read_runtime_events(path: Path) -> tuple[BenchmarkRuntimeTraceEventV1, ...]:
    events = tuple(read_jsonl_models(BenchmarkRuntimeTraceEventV1, path))
    if tuple(event.event_index for event in events) != tuple(range(len(events))):
        raise ValueError(f"TrialDev runtime event indices are not contiguous: {path}")
    if len({event.event_id for event in events}) != len(events):
        raise ValueError(f"TrialDev runtime event IDs must be unique: {path}")
    return events


def _validate_runtime_event_link(
    event: BenchmarkRuntimeTraceEventV1,
    *,
    messages: list[dict[str, Any]],
    path: Path,
) -> None:
    index = event.conversation_message_index
    message: JsonValue | None = None
    if index is not None:
        if index >= len(messages):
            raise ValueError(f"TrialDev event references a missing conversation message: {path}")
        message = cast(JsonValue, messages[index])
    expected = runtime_event_source_payload_v1(
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
        conversation_message=message,
    )
    if canonical_payload_sha256(expected) != event.source_payload_sha256:
        raise ValueError(f"TrialDev runtime conversation payload hash mismatch: {path}")


def _trace_event(
    event: BenchmarkRuntimeTraceEventV1,
    *,
    model_id: str,
    run_id: str,
    messages: list[dict[str, Any]],
    path: Path,
    event_index: int,
) -> ModelActionTraceEventV1:
    _validate_runtime_event_link(event, messages=messages, path=path)
    message_index = event.conversation_message_index
    content_chars = None
    if message_index is not None:
        content = messages[message_index].get("content")
        content_chars = len(content) if isinstance(content, str) else 0
    event_type = "state_transition" if event.event_type in {"step_started", "step_terminal"} else event.event_type
    return ModelActionTraceEventV1(
        event_id=event.event_id,
        timestamp=event.timestamp,
        benchmark="trialdev",
        model_id=model_id,
        run_id=run_id,
        task_id=event.task_id,
        program_id=event.program_id,
        scenario_id=event.scenario_id,
        objective_id=event.objective_id,
        phase_id=event.phase_id,
        step_id=event.step_id,
        turn_index=message_index,
        event_index=event_index,
        event_type=event_type,
        source_path=path.as_posix(),
        source_artifact_path=event.source_artifact_path,
        source_payload_sha256=event.source_payload_sha256,
        tool_call_id=event.tool_call_id,
        tool_name=event.tool_name,
        file_accessed=event.file_accessed,
        elapsed_sec=event.elapsed_seconds,
        execution_status=event.execution_status,
        execution_language=event.execution_language,
        cpu_seconds=event.cpu_seconds,
        session_peak_memory_bytes=event.session_peak_memory_bytes,
        session_peak_processes=event.session_peak_processes,
        output_truncated=event.output_truncated,
        content_chars=content_chars,
        status=event.status,
        failure_code=event.failure_type,
    )


def discover_trialdev_run_dirs(root: Path) -> list[Path]:
    """Return canonical run directories below ``root`` in stable order."""

    root = Path(root)
    candidates = {path.parent.resolve() for path in root.rglob("invocation.json")}
    return sorted(candidates)


def _trace_inputs(task_root: Path) -> tuple[tuple[Path, Path], ...]:
    """Return every event/conversation pair consumed by trace analysis."""

    pairs: list[tuple[Path, Path]] = []
    for event_path in sorted(task_root.glob("events*.jsonl")):
        suffix = event_path.name.removeprefix("events").removesuffix(".jsonl")
        conversation_path = task_root / f"conversation{suffix}.json"
        if not conversation_path.is_file():
            raise FileNotFoundError(f"TrialDev event log has no conversation: {event_path}")
        pairs.append((event_path, conversation_path))
    return tuple(pairs)


def _task_events(
    inputs: tuple[tuple[Path, Path], ...],
    *,
    model_id: str,
    run_id: str,
) -> list[ModelActionTraceEventV1]:
    rows: list[ModelActionTraceEventV1] = []
    for event_path, conversation_path in inputs:
        messages = _conversation(conversation_path)
        for event in _read_runtime_events(event_path):
            rows.append(
                _trace_event(
                    event,
                    model_id=model_id,
                    run_id=run_id,
                    messages=messages,
                    path=event_path,
                    event_index=len(rows),
                )
            )
    return rows


def _endpoint_result(
    task_root: Path,
    run: TrialDevTaskRunV1,
) -> tuple[bool | None, TraceEndpointStateV1]:
    """Return the terminal result without converting evaluator failure into participant failure."""

    if run.execution_status == "model_noncompletion":
        return None, "not_attempted_noncompletion"
    if run.execution_status == "evaluator_error":
        return None, "evaluator_error"
    if run.programme_relative_path is not None:
        result = read_json_model(TrialDevProgrammeRun, task_root / run.programme_relative_path)
        passed = result.evaluation.programme_complete
    elif run.analysis_relative_path is not None:
        analysis = read_json_model(TrialDevAnalysisRun, task_root / run.analysis_relative_path)
        passed = analysis.replay.reproducible
    else:
        raise ValueError("TrialDev trace requires a current numerical or authored-programme result")
    return passed, "valid" if passed else "failed"


def collect_trialdev_action_trace(
    run_dirs: list[Path],
    *,
    trialdev_release_root: Path | None = None,
) -> tuple[
    list[ModelActionTraceEventV1],
    list[TraceFeatureRowV1],
    list[EvidenceUseRowV1],
]:
    """Collect public action and evidence-use records from canonical run custody."""

    all_events: list[ModelActionTraceEventV1] = []
    features: list[TraceFeatureRowV1] = []
    run_ids: set[str] = set()
    for run_dir in sorted(Path(path).resolve() for path in run_dirs):
        invocation = read_json_model(TrialDevRunInvocationV1, run_dir / "invocation.json")
        ranking_path = run_dir / "ranking.json"
        ranking = read_json_model(TrialDevRunRankingV1, ranking_path) if ranking_path.is_file() else None
        if ranking is not None and (
            ranking.invocation_checksum != invocation.checksum
            or ranking.model_id != invocation.model
            or ranking.release_id != invocation.release_id
            or ranking.release_manifest_sha256 != invocation.release_manifest_sha256
            or {row.task_id for row in ranking.items} != set(invocation.scheduled_task_ids)
        ):
            raise ValueError("TrialDev trace ranking belongs to a different invocation")
        ranked = {} if ranking is None else {row.task_id: row for row in ranking.items}
        run_id = f"trialdev:{invocation.checksum}"
        if run_id in run_ids:
            raise ValueError(f"TrialDev run identifier is not unique: {run_id}")
        run_ids.add(run_id)
        task_paths = trialdev_task_run_paths(run_dir)
        if not task_paths.keys() <= set(invocation.scheduled_task_ids):
            raise ValueError("TrialDev trace contains unscheduled task results")
        for task_id in invocation.scheduled_task_ids:
            if task_id not in task_paths:
                item = ranked.get(task_id)
                unfinished = any((parent / task_id).exists() for parent in (run_dir, run_dir / "tasks"))
                features.append(
                    TraceFeatureRowV1(
                        benchmark="trialdev",
                        model_id=invocation.model,
                        run_id=run_id,
                        task_id=task_id,
                        program_id=task_id,
                        scenario_id=item.world_id if item else "unobserved",
                        objective_id=invocation.scheduled_task_streams[task_id].value,
                        trace_coverage_status="missing_trace",
                        inspected_public_data=False,
                        executed_code=False,
                        procedure_assistance="output_contract_only",
                        execution_status="incomplete_run" if unfinished else "not_started",
                        endpoint_state="not_scoreable_trace_only" if unfinished else "not_attempted_noncompletion",
                        asset_count=None if item is None else item.asset_count,
                        observational_loss=None if unfinished or item is None else item.observational_loss,
                        ranking_source_sha256=None if ranking is None else sha256_file(ranking_path),
                    )
                )
                continue
            task_root = task_paths[task_id].parent
            run = read_json_model(TrialDevTaskRunV1, task_root / "run.json")
            if (
                run.task_id != task_id
                or run.stream != invocation.scheduled_task_streams[task_id]
                or run.release_id != invocation.release_id
                or run.release_manifest_sha256 != invocation.release_manifest_sha256
            ):
                raise ValueError("TrialDev trace task differs from its invocation")
            verify_trialdev_task_run_artifacts_v1(task_root=task_root, run=run)
            trace_inputs = _trace_inputs(task_root)
            bound = {artifact.relative_path for artifact in run.artifacts}
            unbound = tuple(
                path.relative_to(task_root).as_posix()
                for pair in trace_inputs
                for path in pair
                if path.relative_to(task_root).as_posix() not in bound
            )
            if unbound:
                raise ValueError(f"TrialDev task-run artifacts do not bind trace input: {unbound[0]}")
            events = _task_events(trace_inputs, model_id=invocation.model, run_id=run_id)
            all_events.extend(events)
            contexts = {(event.program_id, event.scenario_id, event.objective_id, event.phase_id) for event in events}
            if len(contexts) > 1:
                raise ValueError(f"TrialDev task trace has inconsistent context: {task_id}")
            item = ranked.get(task_id)
            if item is not None and item.run_checksum != run.checksum:
                raise ValueError("TrialDev trace ranking belongs to a different task result")
            program_id, scenario_id, objective_id, phase_id = (
                contexts.pop()
                if contexts
                else (
                    task_id,
                    item.world_id if item else "unobserved",
                    run.stream.value,
                    "development" if run.programme_relative_path is not None else run.stream.value,
                )
            )
            if program_id is None or scenario_id is None or objective_id is None:
                raise ValueError(f"TrialDev task trace lacks its programme context: {task_id}")
            endpoint_valid, endpoint_state = _endpoint_result(task_root, run)
            feature = TraceFeatureRowV1(
                benchmark="trialdev",
                model_id=invocation.model,
                run_id=run_id,
                task_id=task_id,
                program_id=program_id,
                scenario_id=scenario_id,
                objective_id=objective_id,
                phase_id=phase_id,
                trace_coverage_status="full_conversation_trace" if events else "missing_trace",
                inspected_public_data=any(is_successful_file_read(event) for event in events),
                executed_code=any(event.event_type == "code_execution" for event in events),
                submitted_answer=any(
                    event.event_type == "submission" and event.status == "observed" for event in events
                ),
                procedure_assistance="output_contract_only",
                score_link_id=f"{run_id}:{task_id}",
                endpoint_valid=endpoint_valid,
                endpoint_state=endpoint_state,
                submission_artifact_paths=tuple(
                    artifact.relative_path
                    for artifact in run.artifacts
                    if artifact.relative_path.startswith("analysis_sources/")
                    and artifact.relative_path.endswith(("/report.md", "/estimates.csv"))
                ),
                execution_status=run.execution_status,
                asset_count=None if item is None else item.asset_count,
                observational_loss=None if item is None else item.observational_loss,
                development_value=None if item is None else item.net_development_value_usd,
                ranking_source_sha256=None if ranking is None else sha256_file(ranking_path),
            )
            features.append(feature)
    evidence = evidence_rows_from_events(all_events, trialdev_release_root=trialdev_release_root)
    return all_events, features, evidence


def trialdev_decision_rows(run_dir: Path, events: list[ModelActionTraceEventV1]) -> list[TrialDevDecisionRowV1]:
    """Project accepted analyses and authored operations from bound execution records."""
    run_dir = run_dir.resolve(strict=True)
    invocation = read_json_model(TrialDevRunInvocationV1, run_dir / "invocation.json")
    paths = trialdev_task_run_paths(run_dir)
    rows = []
    for task_id in invocation.scheduled_task_ids:
        if task_id not in paths:
            continue
        task_root = paths[task_id].parent
        run = read_json_model(TrialDevTaskRunV1, paths[task_id])
        verify_trialdev_task_run_artifacts_v1(task_root=task_root, run=run)
        bound = {artifact.relative_path for artifact in run.artifacts}
        task_events = [event for event in events if event.program_id == task_id]
        checkpoint_path = task_root / "private/agent_state.json"
        checkpoint = read_json_model(TrialDevAgentCheckpoint, checkpoint_path) if checkpoint_path.is_file() else None
        outputs = (
            {}
            if checkpoint is None
            else {
                str(record["tool_call_id"]): CodeExecutionResultV1.model_validate(record["result"])
                for record in checkpoint.execution_outputs
            }
        )
        states = (
            {}
            if checkpoint is None
            else {
                str(record["tool_call_id"]): TrialDevProgrammeStatus.model_validate(record["programme_state"])
                for record in checkpoint.execution_outputs
                if "programme_state" in record
            }
        )
        programme_tools = frozenset(trialdev_action_names())
        ledger_path = task_root / "private/budget.json"
        ledger = read_json_model(TrialDevBudgetLedger, ledger_path) if ledger_path.is_file() else None
        previous_day = 0.0
        previous_spent = 0.0
        previous_available = None if ledger is None else ledger.budget_usd
        decision_index = 0
        for event in task_events:
            if event.event_type != "submission" and event.tool_name not in programme_tools:
                continue
            if event.event_type not in {"submission", "tool_call"}:
                continue
            source = Path(event.source_path)
            relative = source.relative_to(task_root).as_posix()
            if relative not in bound:
                raise ValueError("Decision event is not bound by task custody")
            if event.tool_name in programme_tools and event.status == "observed":
                if event.tool_call_id is None:
                    raise ValueError("Programme operation lacks its tool call identity")
                execution = outputs.get(event.tool_call_id)
                if execution is None or execution.status != "success" or event.tool_call_id not in states:
                    raise ValueError("Programme operation lacks its successful retained output")
                if states[event.tool_call_id].preview:
                    identity = (
                        "action-" + canonical_payload_sha256({"step": event.step_id, "call": event.tool_call_id})[:24]
                    )
                    if (task_root / "private/commitments" / f"{identity}.request.json").exists():
                        raise ValueError("A study preview cannot have a committed action")
                    continue  # Retained in tool traces, not counted as an acquired study or decision.
            decision_index += 1
            action = "report" if event.event_type == "submission" else None
            study_id = None
            selected_asset = None
            current = None
            request_path = None
            if event.tool_name in programme_tools and event.status == "observed":
                if checkpoint is None or "private/agent_state.json" not in bound:
                    raise ValueError("Programme decision lacks its bound agent checkpoint")
                identity = (
                    "action-" + canonical_payload_sha256({"step": event.step_id, "call": event.tool_call_id})[:24]
                )
                request_path = task_root / "private/commitments" / f"{identity}.request.json"
                if request_path.relative_to(task_root).as_posix() not in bound:
                    raise ValueError("Programme decision lacks its bound request")
                request = read_json_model(TrialDevActionRequest, request_path).request
                action = request.action
                study_id = getattr(request, "study_id", None)
                selected_asset = getattr(request, "asset_id", None)
                if event.tool_call_id not in outputs:
                    raise ValueError("Programme decision lacks its retained full tool output")
                execution = outputs[event.tool_call_id]
                if execution.status != "success":
                    raise ValueError("Committed programme event has no successful tool output")
                current = states[event.tool_call_id]
            rejected = event.status == "invalid"
            rows.append(
                TrialDevDecisionRowV1(
                    model_id=invocation.model,
                    run_id=f"trialdev:{invocation.checksum}",
                    condition_id=invocation.experiment_condition.condition_id,
                    task_id=task_id,
                    world_id=event.scenario_id or "unobserved",
                    stream=run.stream.value,
                    decision_index=decision_index,
                    event_index=event.event_index,
                    submission_status="rejected" if rejected else "committed" if current is not None else "accepted",
                    action=None if rejected else action,
                    study_id=study_id,
                    selected_asset_id=selected_asset,
                    evidence_cutoff=event.step_id,
                    day_before=previous_day if ledger is not None else None,
                    day_after=(
                        previous_day
                        if current is None and ledger is not None
                        else None if current is None else current.programme_day
                    ),
                    budget_before=previous_available,
                    budget_after=previous_available if current is None else current.available_usd,
                    incurred_cost=None if current is None else current.spent_usd - previous_spent,
                    elapsed_days=None if current is None else current.programme_day - previous_day,
                    terminal_reason=None if current is None or current.terminal is None else current.terminal.decision,
                    released_evidence_paths=(
                        () if current is None or current.readout_path is None else (current.readout_path,)
                    ),
                    structurally_valid=not rejected,
                    source_relative_path=source.relative_to(run_dir).as_posix(),
                    source_binding="event_payload",
                    source_sha256=event.source_payload_sha256,
                    ingress_sha256=None if request_path is None else sha256_file(request_path),
                    score_link_id=f"{run_dir.name}:{task_id}:{decision_index:03d}",
                )
            )
            if current is not None:
                previous_day, previous_spent, previous_available = (
                    current.programme_day,
                    current.spent_usd,
                    current.available_usd,
                )
    return rows


__all__ = ["collect_trialdev_action_trace", "discover_trialdev_run_dirs", "trialdev_decision_rows"]
