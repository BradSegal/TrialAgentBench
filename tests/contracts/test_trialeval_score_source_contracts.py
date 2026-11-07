"""Tests for canonical TrialEval outcome-row ingestion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from trialagentbench_verifier.scorer import current_scorer

from trialagentbench.analysis.trialeval_score_rows import (
    TrialEvalScoreSourceError,
    iter_trialeval_score_rows,
)
from trialagentbench.contracts.core.runs import TrialEvalRunConfigV1
from trialagentbench.grading.outcome import (
    TrialEvalOutcomeGradingInputV1,
    TrialEvalPointOracleTargetV1,
    TrialEvalQuestionBindingV1,
    grade_trialeval_outcome_v1,
)


def _run_config(*, task_ids: list[str]) -> TrialEvalRunConfigV1:
    return TrialEvalRunConfigV1.create(
        timestamp_utc="2026-07-15T00:00:00Z",
        model="test-model",
        item_watchdog_seconds=3600,
        participant_archive="participant.zip",
        participant_archive_sha256="d" * 64,
        participant_manifest_checksum="e" * 64,
        prompt_set_sha256="a" * 64,
        scorer_source_sha256="s" * 64,
        harness_source_sha256="g" * 64,
        experiment_condition={
            "condition_id": "primary",
            "request_replicate_id": "request-1",
            "procedure_assistance": "output_contract_only",
            "maximum_turns_per_step": 25,
            "tool_choice": "auto",
        },
        task_evidence_factors={
            task_id: {
                "context_configuration": "C1",
                "data_preparation": "analysis_ready",
                "analysis_specification": "locked_sap",
            }
            for task_id in task_ids
        },
        task_ids=task_ids,
        n_items=len(task_ids),
        decoding={"temperature": 0.0, "max_tokens": 512, "send_temperature": True},
        routing={"provider": "openai", "request_timeout_seconds": 300.0},
        executor={
            "image_reference": "trialagentbench/executor:test",
            "image_id": f"sha256:{'e' * 64}",
            "python_version": "3.12.10",
            "packages": [
                {"name": "lifelines", "version": "0.30.0"},
                {"name": "numpy", "version": "2.4.4"},
                {"name": "pandas", "version": "2.3.3"},
                {"name": "pyarrow", "version": "23.0.1"},
                {"name": "scipy", "version": "1.17.1"},
                {"name": "statsmodels", "version": "0.14.6"},
            ],
            "limits": {},
        },
    )


def _missing_outcome() -> dict[str, object]:
    binding = TrialEvalQuestionBindingV1(
        analysis_population_id="itt",
        estimand_id="estimand",
        treatment_id="active",
        comparator_id="control",
        endpoint_id="endpoint",
        effect_measure="mean_difference",
        analysis_unit="participant",
        population_basis="superpopulation",
        identification_status="point_identified",
        result_unit="points",
        uncertainty_type="central_confidence_interval",
        confidence_level=0.95,
        sidedness="two_sided_central",
        null_value=0.0,
        favorable_direction="higher",
    )
    target = TrialEvalPointOracleTargetV1(
        binding=binding,
        value=1.0,
        confidence_level=0.95,
        reference_sampling_standard_deviation=0.2,
        reference_sampling_basis_id="finite_world_oracle_contrast",
    )
    return grade_trialeval_outcome_v1(
        TrialEvalOutcomeGradingInputV1(
            context="C1",
            target=target,
            attempt_status="missing_required_output",
        )
    ).model_dump(mode="json")


def _write_missing_outcome(path: Path, *, task_ids: list[str] | None = None, item_id: str = "TASK1") -> None:
    task_ids = task_ids or [item_id]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "item_id": item_id,
                "timestamp_utc": "2026-07-15T00:00:00Z",
                "run_config": _run_config(task_ids=task_ids).model_dump(mode="json"),
                "agent_output": {
                    "status": "success",
                    "turns_used": 1,
                    "report": "Analysis report.",
                    "condition_provenance": {
                        "procedure_assistance": "output_contract_only",
                        "analysis_specification": "locked_sap",
                        "analysis_surface_sha256": "d" * 64,
                        "max_turns": 25,
                        "prompt_set_sha256": "a" * 64,
                        "rendered_system_prompt_sha256": "b" * 64,
                        "tool_schema_sha256": "c" * 64,
                    },
                },
                "scores": {
                    "scorer": current_scorer().model_dump(mode="json"),
                    "item_id": item_id,
                    "task_id": item_id,
                    "release_id": "release-test",
                    "trial_id": "trial_aaaaaaaaaaaaaaaaaaaa",
                    "base_world_id": "world_bbbbbbbbbbbbbbbbbbbb",
                    "trial_name": "trial_aaaaaaaaaaaaaaaaaaaa",
                    "cell_ref": "cell_cccccccccccccccccccc",
                    "effect_state": "favourable",
                    "design_tier": "D1",
                    "design_subtype": "individual_randomized",
                    "context_tier": "C1",
                    "effect_measure": "mean_difference",
                    "result_unit": "points",
                    "model": "test-model",
                    "turns_used": 1,
                    "agent_status": "success",
                    "outcome": _missing_outcome(),
                    "target": TrialEvalPointOracleTargetV1(
                        binding=TrialEvalQuestionBindingV1(
                            analysis_population_id="itt",
                            estimand_id="estimand",
                            treatment_id="active",
                            comparator_id="control",
                            endpoint_id="endpoint",
                            effect_measure="mean_difference",
                            analysis_unit="participant",
                            population_basis="superpopulation",
                            identification_status="point_identified",
                            result_unit="points",
                            uncertainty_type="central_confidence_interval",
                            confidence_level=0.95,
                            sidedness="two_sided_central",
                            null_value=0.0,
                            favorable_direction="higher",
                        ),
                        value=1.0,
                        confidence_level=0.95,
                        reference_sampling_standard_deviation=0.2,
                        reference_sampling_basis_id="finite_world_oracle_contrast",
                    ).model_dump(mode="json"),
                    "oracle_worlds": 400,
                    "oracle_monte_carlo_standard_error": 0.001,
                    "maximum_oracle_monte_carlo_standard_error": 0.01,
                    "replay": {
                        "status": "missing_result",
                        "entrypoint": "analysis.py",
                        "target_ids": ["estimand"],
                        "evidence_artifacts": [],
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_reader_requires_fresh_canonical_outcome_payload(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_missing_outcome(run_dir / "items" / "TASK1.json")

    row = next(iter_trialeval_score_rows(run_dir))

    assert row.task_id == "TASK1"
    assert row.release_id == "release-test"
    assert row.trial_id == "trial_aaaaaaaaaaaaaaaaaaaa"
    assert row.participant_manifest_checksum == "e" * 64
    assert row.run_identity_sha256 == _run_config(task_ids=["TASK1"]).run_identity_sha256
    assert len(row.source_json_sha256) == 64
    assert row.cell_ref == "cell_cccccccccccccccccccc"
    assert row.condition_id == "primary"
    assert row.request_replicate_id == "request-1"
    assert row.reasoning_effort is None
    assert row.outcome.failure_origins == ("missing_required_output",)
    assert row.outcome.completion.status == "not_satisfied"
    assert row.replay.status == "missing_result"


def test_reader_rejects_historical_or_extra_score_fields(tmp_path: Path) -> None:
    path = tmp_path / "run" / "items" / "TASK1.json"
    _write_missing_outcome(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["scores"]["primary_score_normalised_max_recoverable"] = 0.5
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        list(iter_trialeval_score_rows(path.parents[1]))


def test_reader_rejects_incomplete_scheduled_surface(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_missing_outcome(
        run_dir / "items" / "TASK1.json",
        task_ids=["TASK1", "TASK2"],
    )

    with pytest.raises(TrialEvalScoreSourceError, match="missing=.*TASK2"):
        list(iter_trialeval_score_rows(run_dir))


@pytest.mark.parametrize(
    ("recovery_surface", "accepted"),
    ((["TASK2"], True), (["TASK2", "TASK3"], False), (["TASK2", "TASK1"], False)),
    ids=("subset", "outside-campaign", "reordered-campaign"),
)
def test_reader_admits_recovery_items_scheduled_on_a_subset_surface(
    tmp_path: Path, recovery_surface: list[str], accepted: bool
) -> None:
    """A recovery run reschedules some campaign tasks; any other surface disagreement is a conflict."""
    from trialagentbench.io import write_json_model

    run_dir = tmp_path / "run"
    write_json_model(run_dir / "run_config.json", _run_config(task_ids=["TASK1", "TASK2"]))
    _write_missing_outcome(run_dir / "items" / "TASK1.json", task_ids=["TASK1", "TASK2"])
    _write_missing_outcome(run_dir / "items" / "TASK2.json", task_ids=recovery_surface, item_id="TASK2")
    if accepted:
        assert [row.task_id for row in iter_trialeval_score_rows(run_dir)] == ["TASK1", "TASK2"]
    else:
        with pytest.raises(TrialEvalScoreSourceError, match="disagree on the scheduled task surface"):
            list(iter_trialeval_score_rows(run_dir))


@pytest.mark.parametrize(("graded", "censored"), ((1, True), (2, False)), ids=("declared", "undeclared"))
def test_reader_admits_only_a_declared_budget_censored_shortfall(tmp_path: Path, graded: int, censored: bool) -> None:
    """A spending limit that stops a run partway leaves unrun items its grade manifest declares."""
    from datetime import UTC, datetime

    from trialagentbench.contracts.core.manifest import GradeManifestV1
    from trialagentbench.io import write_json_model

    run_dir = tmp_path / "run"
    _write_missing_outcome(run_dir / "items" / "TASK1.json", task_ids=["TASK1", "TASK2"])
    write_json_model(
        run_dir / "GRADE_MANIFEST.json",
        GradeManifestV1(
            scorer=current_scorer(),
            harness_version="test",
            harness_git_sha=None,
            harness_source_sha256="a" * 64,
            timestamp_utc=datetime(2026, 10, 9, tzinfo=UTC),
            input_run_dir="run",
            output_run_dir="grade",
            input_run_sha256="b" * 64,
            evaluator_release_sha256="c" * 64,
            release_manifest_sha256="d" * 64,
            scorer_surface_sha256="e" * 64,
            suite_task_count=2,
            graded_task_count=graded,
        ),
    )
    if censored:
        assert [row.task_id for row in iter_trialeval_score_rows(run_dir)] == ["TASK1"]
    else:
        with pytest.raises(TrialEvalScoreSourceError, match="missing=.*TASK2"):
            list(iter_trialeval_score_rows(run_dir))


def test_reader_rejects_observed_outcome_without_replay_result(tmp_path: Path) -> None:
    path = tmp_path / "run" / "items" / "TASK1.json"
    _write_missing_outcome(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["scores"]["outcome"]["estimand_performance"] = {
        "component_id": "estimand_performance",
        "status": "observed",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="observed estimand performance"):
        list(iter_trialeval_score_rows(path.parents[1]))
