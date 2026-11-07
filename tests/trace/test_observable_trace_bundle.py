"""Black-box tests for the public observable trace bundle."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from trialagentbench.analysis.failure_cascade import phase_failure_cascade
from trialagentbench.contracts.trace.observable import (
    EvidenceUseRowV1,
    FailureCascadeRowV1,
    ModelActionTraceEventV1,
    TraceFeatureRowV1,
)
from trialagentbench.tools.build import build_trace_analysis_bundle as builder
from trialagentbench.tools.validate.validate_trace_analysis_bundle import (
    validate_trace_analysis_bundle,
)


@pytest.mark.parametrize("loss", [-1.0, float("inf"), float("nan")])
def test_trace_rejects_invalid_numerical_loss(loss: float) -> None:
    with pytest.raises(ValidationError, match="observational_loss"):
        TraceFeatureRowV1(
            benchmark="trialdev",
            model_id="model",
            run_id="run",
            program_id="programme",
            scenario_id="world",
            objective_id="development",
            trace_coverage_status="missing_trace",
            inspected_public_data=False,
            executed_code=False,
            procedure_assistance="output_contract_only",
            observational_loss=loss,
        )


def _trialeval_rows(
    *,
    run_id: str = "run-001",
    task_id: str = "task-001",
) -> tuple[
    list[ModelActionTraceEventV1],
    list[TraceFeatureRowV1],
    list[EvidenceUseRowV1],
    list[FailureCascadeRowV1],
]:
    feature = TraceFeatureRowV1(
        benchmark="trialeval",
        model_id="arbitrary/model-id",
        run_id=run_id,
        task_id=task_id,
        phase_id="task",
        trace_coverage_status="full_conversation_trace",
        inspected_public_data=True,
        executed_code=True,
        submitted_answer=True,
        submission_transport="direct",
        context_tier="C1",
        data_preparation="analysis_ready",
        analysis_specification="locked_sap",
        procedure_assistance="output_contract_only",
        score_link_id=f"score:{run_id}:{task_id}",
        endpoint_valid=True,
        endpoint_state="valid",
    )
    return (
        [
            ModelActionTraceEventV1(
                event_id=f"event:{run_id}:{task_id}",
                timestamp=datetime(2026, 7, 22, tzinfo=UTC),
                benchmark="trialeval",
                model_id=feature.model_id,
                run_id=feature.run_id,
                task_id=feature.task_id,
                phase_id="task",
                event_index=0,
                event_type="file_inspection",
                source_path="items/task-001/events.jsonl",
                source_artifact_path="items/task-001/events.jsonl",
                source_payload_sha256="a" * 64,
                file_accessed="public/adtte.parquet",
                status="observed",
            )
        ],
        [feature],
        [
            EvidenceUseRowV1(
                benchmark="trialeval",
                model_id=feature.model_id,
                run_id=feature.run_id,
                task_id=feature.task_id,
                phase_id="task",
                evidence_category="time_to_event",
                source="tool_call",
                artifact_path="public/adtte.parquet",
                participant_facing=True,
            )
        ],
        [
            FailureCascadeRowV1(
                benchmark="trialeval",
                model_id=feature.model_id,
                run_id=feature.run_id,
                task_id=feature.task_id,
                first_failure_phase="task",
                first_failure_type="none_observed",
                downstream_endpoint_failed=False,
                score_link_id=f"score:{run_id}:{task_id}",
            )
        ],
    )


def test_observable_trace_bundle_is_deterministic_and_model_agnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public bundle is deterministic and accepts arbitrary model IDs."""

    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(builder, "discover_trialeval_run_dirs", lambda _root: [runs / "run-001"])
    monkeypatch.setattr(builder, "collect_trialeval_action_trace", lambda _runs: _trialeval_rows())

    first = builder.build_trace_analysis_bundle(out_dir=tmp_path / "first", trialeval_root=runs)
    moved = tmp_path / "moved-runs"
    moved.mkdir()

    def relocated_rows(run_dirs: list[Path]):
        events, features, evidence, cascades = _trialeval_rows()
        events = [
            event.model_copy(
                update={
                    "source_path": str(run_dirs[0].parent / event.source_path),
                    "source_artifact_path": str(run_dirs[0].parent / event.source_artifact_path),
                }
            )
            for event in events
        ]
        return events, features, evidence, cascades

    monkeypatch.setattr(builder, "discover_trialeval_run_dirs", lambda root: [root / "run-001"])
    monkeypatch.setattr(builder, "collect_trialeval_action_trace", relocated_rows)
    second = builder.build_trace_analysis_bundle(out_dir=tmp_path / "second", trialeval_root=moved)

    manifest = validate_trace_analysis_bundle(first)
    assert manifest.model_ids == ("arbitrary/model-id",)
    assert manifest.benchmark_suites == ("trialeval",)
    assert {row.path for row in manifest.tables} == {
        "action_events.csv",
        "evidence_use.csv",
        "trialdev_decisions.csv",
        "unit_features.csv",
    }
    assert {path.name: path.read_bytes() for path in first.iterdir() if path.is_file()} == {
        path.name: path.read_bytes() for path in second.iterdir() if path.is_file()
    }


def test_event_rows_retain_numeric_chronology_instead_of_json_lexical_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    events, features, evidence, cascades = _trialeval_rows()
    shuffled = [
        events[0].model_copy(update={"event_index": index, "event_id": f"event-{index}"}) for index in (10, 2, 1)
    ]
    monkeypatch.setattr(builder, "discover_trialeval_run_dirs", lambda _: [runs / "run-001"])
    monkeypatch.setattr(builder, "collect_trialeval_action_trace", lambda _: (shuffled, features, evidence, cascades))
    output = builder.build_trace_analysis_bundle(out_dir=tmp_path / "trace", trialeval_root=runs)
    with (output / "action_events.csv").open() as handle:
        assert [int(row["event_index"]) for row in csv.DictReader(handle)] == [1, 2, 10]


def test_observable_trace_bundle_fails_on_unknown_or_mutated_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown members and checksum drift fail rather than being ignored."""

    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(builder, "discover_trialeval_run_dirs", lambda _root: [runs / "run-001"])
    monkeypatch.setattr(builder, "collect_trialeval_action_trace", lambda _runs: _trialeval_rows())
    bundle = builder.build_trace_analysis_bundle(out_dir=tmp_path / "bundle", trialeval_root=runs)

    (bundle / "author_case_selection.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="member mismatch"):
        validate_trace_analysis_bundle(bundle)
    (bundle / "author_case_selection.json").unlink()
    with (bundle / "unit_features.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_trace_analysis_bundle(bundle)


def test_observable_trace_bundle_requires_completed_runs(tmp_path: Path) -> None:
    """An empty requested root is an invalid analysis input."""

    runs = tmp_path / "runs"
    runs.mkdir()
    with pytest.raises(ValueError, match="contains no completed runs"):
        builder.build_trace_analysis_bundle(out_dir=tmp_path / "bundle", trialeval_root=runs)


@pytest.mark.parametrize(
    ("endpoint_state", "expected_failure", "downstream_failed"),
    (
        ("not_attempted_noncompletion", "model_noncompletion", False),
        ("evaluator_error", "evaluator_failure", False),
        ("failed", "endpoint_invalid_or_unusable", True),
        ("valid", "none_observed", False),
        ("not_reached_after_stop", "none_observed", False),
    ),
)
def test_trialdev_failure_cascade_separates_measurement_boundaries(
    endpoint_state: str,
    expected_failure: str,
    downstream_failed: bool,
) -> None:
    """Keep model, evaluator, and scientific endpoints in distinct result states."""

    feature = TraceFeatureRowV1.model_validate(
        {
            "benchmark": "trialdev",
            "model_id": "model",
            "run_id": "run",
            "task_id": "task",
            "program_id": "program",
            "scenario_id": "scenario",
            "objective_id": "objective",
            "phase_id": "evidence_analysis",
            "trace_coverage_status": "full_conversation_trace",
            "inspected_public_data": True,
            "executed_code": True,
            "submitted_answer": endpoint_state not in {"not_attempted_noncompletion", "evaluator_error"},
            "procedure_assistance": "output_contract_only",
            "score_link_id": "run:task",
            "endpoint_valid": True if endpoint_state == "valid" else False if endpoint_state == "failed" else None,
            "endpoint_state": endpoint_state,
        }
    )

    cascade = phase_failure_cascade(feature)

    assert cascade.first_failure_type == expected_failure
    assert cascade.downstream_endpoint_failed is downstream_failed
    assert phase_failure_cascade(feature, hidden_access=True).first_failure_type == "hidden_or_grader_access"


def test_observable_trace_bundle_collects_one_run_at_a_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trace collection is bounded by one persisted run rather than the full panel."""

    runs = tmp_path / "runs"
    runs.mkdir()
    run_dirs = [runs / f"run-{index:03d}" for index in range(3)]
    observed_batches: list[tuple[Path, ...]] = []

    def _collect(
        batch: list[Path],
        *,
        expanded_report: Path | None = None,
    ) -> tuple[
        list[ModelActionTraceEventV1],
        list[TraceFeatureRowV1],
        list[EvidenceUseRowV1],
        list[FailureCascadeRowV1],
    ]:
        assert expanded_report is None
        observed_batches.append(tuple(batch))
        run_id = batch[0].name
        return _trialeval_rows(run_id=run_id, task_id=f"task:{run_id}")

    monkeypatch.setattr(builder, "discover_trialeval_run_dirs", lambda _root: run_dirs)
    monkeypatch.setattr(builder, "collect_trialeval_action_trace", _collect)

    bundle = builder.build_trace_analysis_bundle(out_dir=tmp_path / "bundle", trialeval_root=runs)
    manifest = validate_trace_analysis_bundle(bundle)

    assert observed_batches == [(run_dir,) for run_dir in run_dirs]
    assert manifest.run_ids == tuple(run_dir.name for run_dir in run_dirs)
    assert next(table for table in manifest.tables if table.path == "unit_features.csv").row_count == 3
