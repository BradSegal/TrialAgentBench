"""Focused runtime-event provenance and streaming-ingestion tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic.types import JsonValue

from trialagentbench.contracts.trace.observable import (
    BenchmarkNameV1,
    BenchmarkRuntimeFailureTypeV1,
    BenchmarkRuntimeTraceEventV1,
    runtime_event_source_payload_v1,
)
from trialagentbench.io.checksums import canonical_payload_sha256
from trialagentbench.trialdev.action_trace import (
    _read_runtime_events as read_trialdev_runtime_events,
)
from trialagentbench.trialdev.action_trace import (
    _validate_runtime_event_link as validate_trialdev_runtime_event_link,
)
from trialagentbench.trialeval.action_trace import (
    _runtime_events as read_trialeval_runtime_events,
)
from trialagentbench.trialeval.action_trace import (
    _validate_runtime_events as validate_trialeval_runtime_events,
)

RuntimeEventReader = Callable[[Path], tuple[BenchmarkRuntimeTraceEventV1, ...]]


def _event(
    *,
    benchmark: BenchmarkNameV1,
    event_index: int,
    event_type: str,
    conversation_message: JsonValue | None = None,
    failure_type: BenchmarkRuntimeFailureTypeV1 | None = None,
) -> dict[str, object]:
    unit = "TASK1" if benchmark == "trialeval" else "program-1"
    event: dict[str, object] = {
        "event_id": f"{benchmark}:{unit}:{event_index:06d}",
        "timestamp": "2026-07-19T12:00:00Z",
        "source_artifact_path": f"/runs/{unit}/conversation.json",
        "benchmark": benchmark,
        "event_index": event_index,
        "phase_id": "task" if benchmark == "trialeval" else "phase1",
        "step_id": "analysis",
        "event_type": event_type,
    }
    if benchmark == "trialeval":
        event["task_id"] = unit
    else:
        event.update(program_id=unit, scenario_id="s01", objective_id="benefit_risk")
    if event_type == "step_terminal":
        event["terminal_status"] = "failed" if failure_type else "completed"
        event["failure_type"] = failure_type
    if conversation_message is not None:
        event["conversation_message_index"] = 0
    source_payload = runtime_event_source_payload_v1(
        benchmark=benchmark,
        task_id=unit if benchmark == "trialeval" else None,
        program_id=unit if benchmark == "trialdev" else None,
        scenario_id="s01" if benchmark == "trialdev" else None,
        objective_id="benefit_risk" if benchmark == "trialdev" else None,
        phase_id=str(event["phase_id"]),
        step_id="analysis",
        event_type=event_type,
        terminal_status=str(event["terminal_status"]) if event.get("terminal_status") is not None else None,
        failure_type=failure_type,
        conversation_message=conversation_message,
    )
    event["source_payload_sha256"] = canonical_payload_sha256(source_payload)
    return event


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_runtime_event_contract_requires_native_provenance() -> None:
    payload = _event(benchmark="trialeval", event_index=0, event_type="step_started")
    del payload["event_id"]

    with pytest.raises(ValidationError, match="event_id"):
        BenchmarkRuntimeTraceEventV1.model_validate(payload)


def test_runtime_event_contract_rejects_non_utc_timestamp() -> None:
    payload = _event(benchmark="trialeval", event_index=0, event_type="step_started")
    payload["timestamp"] = "2026-07-19T13:00:00+01:00"

    with pytest.raises(ValidationError, match="timezone-aware UTC"):
        BenchmarkRuntimeTraceEventV1.model_validate(payload)


def test_runtime_event_contract_rejects_unclassified_terminal_failure() -> None:
    """Keep runtime failure attribution inside the public closed taxonomy."""

    payload = _event(benchmark="trialeval", event_index=0, event_type="step_terminal")
    payload.update(terminal_status="failed", failure_type="SomeInternalException")

    with pytest.raises(ValidationError, match="failure_type"):
        BenchmarkRuntimeTraceEventV1.model_validate(payload)


@pytest.mark.parametrize(
    ("benchmark", "reader"),
    [
        ("trialeval", read_trialeval_runtime_events),
        ("trialdev", read_trialdev_runtime_events),
    ],
)
def test_runtime_jsonl_ingestion_preserves_native_ids(
    tmp_path: Path,
    benchmark: BenchmarkNameV1,
    reader: RuntimeEventReader,
) -> None:
    path = tmp_path / "events.jsonl"
    rows = [
        _event(benchmark=benchmark, event_index=0, event_type="step_started"),
        _event(benchmark=benchmark, event_index=1, event_type="step_terminal"),
    ]
    _write_jsonl(path, rows)

    events = reader(path)

    assert [event.event_id for event in events] == [row["event_id"] for row in rows]
    assert [event.source_payload_sha256 for event in events] == [row["source_payload_sha256"] for row in rows]


@pytest.mark.parametrize(
    ("benchmark", "reader"),
    [
        ("trialeval", read_trialeval_runtime_events),
        ("trialdev", read_trialdev_runtime_events),
    ],
)
def test_runtime_jsonl_ingestion_rejects_duplicate_native_ids(
    tmp_path: Path,
    benchmark: BenchmarkNameV1,
    reader: RuntimeEventReader,
) -> None:
    path = tmp_path / "events.jsonl"
    rows = [
        _event(benchmark=benchmark, event_index=0, event_type="step_started"),
        _event(benchmark=benchmark, event_index=1, event_type="step_terminal"),
    ]
    rows[1]["event_id"] = rows[0]["event_id"]
    _write_jsonl(path, rows)

    with pytest.raises(ValueError, match="event IDs must be unique"):
        reader(path)


def test_trialeval_ingestion_rejects_tampered_indexed_conversation_payload(tmp_path: Path) -> None:
    conversation_path = tmp_path / "conversation.json"
    original_message: JsonValue = {"role": "user", "content": "original prompt"}
    rows = [
        _event(benchmark="trialeval", event_index=0, event_type="step_started"),
        _event(
            benchmark="trialeval",
            event_index=1,
            event_type="prompt",
            conversation_message=original_message,
        ),
        _event(benchmark="trialeval", event_index=2, event_type="step_terminal"),
    ]
    conversation_path.write_text(
        json.dumps([{"role": "user", "content": "tampered prompt"}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="conversation payload hash mismatch"):
        validate_trialeval_runtime_events(
            tuple(BenchmarkRuntimeTraceEventV1.model_validate(row) for row in rows),
            task_id="TASK1",
            conversation_path=conversation_path,
        )


def test_trialdev_ingestion_rejects_tampered_indexed_conversation_payload(tmp_path: Path) -> None:
    original_message: JsonValue = {"role": "user", "content": "original prompt"}
    row = _event(
        benchmark="trialdev",
        event_index=0,
        event_type="prompt",
        conversation_message=original_message,
    )
    event = BenchmarkRuntimeTraceEventV1.model_validate(row)

    with pytest.raises(ValueError, match="conversation payload hash mismatch"):
        validate_trialdev_runtime_event_link(
            event,
            messages=[{"role": "user", "content": "tampered prompt"}],
            path=tmp_path / "events.jsonl",
        )


@pytest.mark.parametrize("failure", ["provider_error", "timeout", "runtime_error", "turn_limit_no_submission"])
def test_trialeval_action_handoff_preserves_runtime_failure(tmp_path, monkeypatch, failure):
    """Project authenticated runtime events without losing their terminal cause."""
    from types import SimpleNamespace

    from trialagentbench.contracts.trace.observable import TrialEvalTraceInputV1
    from trialagentbench.trialeval import action_trace as owner

    run = tmp_path / "run"
    logs = run / "logs"
    logs.mkdir(parents=True)
    source = run / "score.json"
    source.write_text("{}")
    (logs / "TASK1_conversation.json").write_text("[]")
    _write_jsonl(
        logs / "TASK1_events.jsonl",
        [
            _event(benchmark="trialeval", event_index=0, event_type="step_started"),
            _event(benchmark="trialeval", event_index=1, event_type="step_terminal", failure_type=failure),
        ],
    )
    trace_input = TrialEvalTraceInputV1(
        run_id="run",
        model_id="model",
        task_id="TASK1",
        context_tier="C1",
        data_preparation="analysis_ready",
        analysis_specification="locked_sap",
        procedure_assistance="output_contract_only",
        submission_transport="not_observed",
        source_path=str(source),
    )
    unavailable = SimpleNamespace(status="not_evaluable")
    row = SimpleNamespace(
        task_id="TASK1",
        model_id="model",
        source_json_path=str(source),
        observed_attempt_outcome="failed",
        outcome=SimpleNamespace(
            question_alignment=unavailable,
            evidence_eligibility=unavailable,
            reproducibility=unavailable,
            estimand_performance=unavailable,
        ),
    )
    monkeypatch.setattr(owner, "require_unique_run_ids", lambda *args, **kwargs: {run: "run"})
    monkeypatch.setattr(owner, "collect_trialeval_trace_inputs", lambda _: (trace_input,))
    monkeypatch.setattr(owner, "iter_trialeval_score_rows", lambda _: iter((row,)))
    events, _, _, _ = owner.collect_trialeval_action_trace([run])
    grade = next(event for event in events if event.event_type == "grade_link")
    assert grade.content_chars == len(source.read_text(encoding="utf-8"))
    terminal = next(event for event in events if event.event_id.endswith(":000001"))
    assert terminal.failure_code == failure
    assert terminal.event_type == "state_transition"

    import csv

    from trialagentbench.tools.build import build_trace_analysis_bundle as builder
    from trialagentbench.tools.validate.validate_trace_analysis_bundle import validate_trace_analysis_bundle

    monkeypatch.setattr(builder, "discover_trialeval_run_dirs", lambda _: [run])
    output = builder.build_trace_analysis_bundle(out_dir=tmp_path / "handoff", trialeval_root=run)
    validate_trace_analysis_bundle(output)
    with (output / "action_events.csv").open() as handle:
        exported = list(csv.DictReader(handle))
    assert [event["failure_code"] for event in exported if event["failure_code"]] == [failure]

    from trialagentbench.trialdev.action_trace import _trace_event

    development_event = BenchmarkRuntimeTraceEventV1.model_validate(
        _event(benchmark="trialdev", event_index=0, event_type="step_terminal", failure_type=failure)
    )
    development = _trace_event(
        development_event,
        model_id="model",
        run_id="run",
        messages=[],
        path=tmp_path / "development_events.jsonl",
        event_index=0,
    )
    assert development.model_dump(mode="json")["failure_code"] == failure
