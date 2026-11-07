"""Observable evidence-use linkage tests."""

from __future__ import annotations

from datetime import UTC, datetime

from trialagentbench.analysis.evidence_use import evidence_rows_from_events
from trialagentbench.contracts.trace.observable import ModelActionTraceEventV1


def _event(
    *,
    event_type: str,
    tool_call_id: str,
    execution_status: str | None = None,
    file_accessed: str | None = None,
    status: str = "observed",
    source_path: str = "events.jsonl",
) -> ModelActionTraceEventV1:
    return ModelActionTraceEventV1.model_validate(
        {
            "event_id": f"{event_type}:{tool_call_id}",
            "timestamp": datetime(2026, 7, 22, tzinfo=UTC),
            "benchmark": "trialeval",
            "model_id": "model",
            "run_id": "run",
            "task_id": "TASK1",
            "assignment_id": "assignment",
            "phase_id": "task",
            "step_id": "analysis",
            "event_index": 0,
            "event_type": event_type,
            "source_path": source_path,
            "source_artifact_path": "conversation.json",
            "source_payload_sha256": "a" * 64,
            "tool_call_id": tool_call_id,
            "tool_name": "inspect_parquet",
            "file_accessed": file_accessed,
            "execution_status": execution_status,
            "status": status,
        }
    )


def test_evidence_use_requires_successful_file_inspection() -> None:
    events = [
        _event(event_type="file_inspection", tool_call_id="successful", file_accessed="data/ADSL.parquet"),
        _event(
            event_type="file_inspection",
            tool_call_id="failed",
            file_accessed="scratch/missing.json",
            status="invalid",
        ),
    ]

    rows = evidence_rows_from_events(events)

    assert len(rows) == 1
    assert rows[0].artifact_path == "data/ADSL.parquet"
    assert rows[0].assignment_id == "assignment"


def test_evidence_use_does_not_treat_workspace_writes_as_inspection() -> None:
    events = [
        _event(event_type="tool_call", tool_call_id="write", file_accessed="scratch/result.json"),
        _event(event_type="tool_result", tool_call_id="write", execution_status="success"),
    ]

    assert evidence_rows_from_events(events) == []


def test_failed_reads_are_not_evidence_use() -> None:
    """A read whose execution failed (for example a missing file) is traced but not counted."""
    events = [
        _event(
            event_type="file_inspection",
            tool_call_id="read",
            file_accessed="data/ADSL.parquet",
            execution_status="success",
        ),
        _event(
            event_type="file_inspection",
            tool_call_id="missing",
            file_accessed="data/ADAE.parquet",
            execution_status="execution_error",
        ),
    ]

    assert [row.artifact_path for row in evidence_rows_from_events(events)] == ["data/ADSL.parquet"]


def test_trialeval_scratch_reads_resolve_in_the_item_workspace(tmp_path) -> None:
    logs = tmp_path / "run" / "logs"
    scratch = logs / "TASK1_workspace" / "item" / "scratch"
    (scratch / "outputs").mkdir(parents=True)
    (scratch / "fit.csv").write_text("coef,p_value\n1,0.1\n", encoding="utf-8")
    (scratch / "outputs" / "turn-3-1.txt").write_text("long output\n", encoding="utf-8")
    events = [
        _event(
            event_type="file_inspection",
            tool_call_id=name,
            file_accessed=name,
            execution_status="success",
            source_path=(logs / "TASK1_events.jsonl").as_posix(),
        )
        for name in ("scratch/fit.csv", "fit.csv", "scratch/outputs/turn-3-1.txt")
    ]

    rows = evidence_rows_from_events(events)

    assert [row.evidence_category for row in rows] == [
        "scratch_model_result",
        "scratch_model_result",
        "analysis_or_submission_workfile",
    ]


def test_invalid_read_is_neither_evidence_nor_a_public_data_inspection() -> None:
    """A read rejected as an invalid tool call never executed: it carries no execution status."""
    from trialagentbench.contracts.trace.observable import is_successful_file_read

    invalid = _event(
        event_type="file_inspection", tool_call_id="bad", file_accessed="data/ADSL.parquet", status="invalid"
    )
    valid = _event(event_type="file_inspection", tool_call_id="ok", file_accessed="data/ADSL.parquet")

    assert not is_successful_file_read(invalid)
    assert is_successful_file_read(valid)
    assert len(evidence_rows_from_events([invalid, valid])) == 1


def test_failed_unclassifiable_read_is_excluded_without_error() -> None:
    """A failed read of an unresolvable path ('document') does not stop the behaviour report."""
    failed = _event(
        event_type="file_inspection", tool_call_id="r", file_accessed="document", execution_status="execution_error"
    )

    assert evidence_rows_from_events([failed]) == []


def test_runner_staged_reporting_interface_reads_are_classified(tmp_path) -> None:
    """Runner-created interface files are recognised in the item scope."""
    from trialagentbench_verifier.numerical import stage_reporting_interface

    stage_reporting_interface(tmp_path, ("task:clinical_outcome",))
    assert (tmp_path / "interface" / "targets.csv").is_file()
    events = [
        _event(
            event_type="file_inspection",
            tool_call_id=name,
            file_accessed=f"interface/{name}",
            execution_status="success",
        )
        for name in ("targets.csv", "reporting.py")
    ]

    assert [row.evidence_category for row in evidence_rows_from_events(events)] == [
        "protocol_or_program_contract",
        "analysis_or_submission_workfile",
    ]
