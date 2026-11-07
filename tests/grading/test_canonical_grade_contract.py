"""Tests for the canonical TrialEval grading boundary."""

from __future__ import annotations

import contextlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from trialagentbench_test_helpers import (
    FIXTURE_C5_INTEGRITY,
    FIXTURE_CELL_KEY,
    FIXTURE_SERIES_KEY,
    write_trialeval_grading_key,
)
from trialagentbench_verifier.scorer import current_scorer
from trialagentbench_verifier.trialeval import (
    TrialEvalDecisionRuleV1,
    TrialEvalGradingKeyRecordV1,
    TrialEvalGradingPlanV1,
)

from trialagentbench.contracts.core.config import DecodingConfigV1
from trialagentbench.contracts.core.runs import TrialEvalItemResultV1, TrialEvalRunConfigV1
from trialagentbench.contracts.release.identity import PairedParticipantArchiveIdentityV1
from trialagentbench.grading.outcome import (
    TrialEvalPointOracleTargetV1,
    TrialEvalQuestionBindingV1,
    TrialEvalScoreInputV1,
    grade_trialeval_outcome_v1,
)
from trialagentbench.io import read_json_model, staged_directory, write_json_model
from trialagentbench.release import ParticipantReleaseError, load_trialeval_grading_key
from trialagentbench.tools.grade import grade_trialeval
from trialagentbench.trialeval.outcome_submission import bind_trialeval_numerical_input
from trialagentbench.trialeval.schema import BenchmarkItem

TASK_ID = f"te_{'1' * 20}"
TRIAL_ID = f"trial_{'2' * 20}"


def _run_config_payload(
    *, task_id: str = "TASK1", task_ids: tuple[str, ...] | None = None, executor=None
) -> dict[str, object]:
    selected_ids = task_ids or (task_id,)
    config = TrialEvalRunConfigV1.create(
        timestamp_utc=datetime.now(UTC),
        model="fixture-model",
        item_watchdog_seconds=3600,
        participant_archive="participant.zip",
        participant_archive_sha256="a" * 64,
        participant_manifest_checksum="f" * 64,
        prompt_set_sha256="b" * 64,
        scorer_source_sha256="c" * 64,
        harness_source_sha256="d" * 64,
        experiment_condition={
            "condition_id": "primary",
            "request_replicate_id": "request-1",
            "procedure_assistance": "output_contract_only",
            "maximum_turns_per_step": 25,
            "tool_choice": "auto",
        },
        task_evidence_factors={
            selected_id: {
                "context_configuration": "C1",
                "data_preparation": "analysis_ready",
                "analysis_specification": "locked_sap",
            }
            for selected_id in selected_ids
        },
        task_ids=list(selected_ids),
        n_items=len(selected_ids),
        decoding=DecodingConfigV1(temperature=0.0, max_tokens=512, send_temperature=True),
        routing={"provider": "openai", "request_timeout_seconds": 300.0},
        executor=executor
        or {
            "image_reference": "executor:test",
            "image_id": f"sha256:{'e' * 64}",
            "python_version": "3.12",
            "packages": [{"name": "pandas", "version": "2.2"}],
            "limits": {},
        },
        workers=1,
    )
    return config.model_dump(mode="python")


def _evaluator_record() -> TrialEvalGradingKeyRecordV1:
    binding = TrialEvalQuestionBindingV1(
        analysis_population_id="itt",
        estimand_id="estimand",
        intercurrent_event_strategy_ids=("treatment_policy",),
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
    return TrialEvalGradingKeyRecordV1(
        schema_id="trialagentbench.trialeval_grading_key_record/v1",
        release_id=f"release_{'1' * 64}",
        task_id=TASK_ID,
        trial_id=TRIAL_ID,
        base_world_id=f"world_{'2' * 20}",
        cell_ref=FIXTURE_CELL_KEY,
        series_ref=FIXTURE_SERIES_KEY,
        context="C1",
        effect_state="favourable",
        design_tier="D1",
        design_subtype="Individually randomized parallel-group trial",
        design_family="individual_randomized",
        participant_data_sha256="3" * 64,
        participant_artifact_paths=("data/example.parquet",),
        target=TrialEvalPointOracleTargetV1(
            binding=binding,
            value=1.0,
            confidence_level=0.95,
            reference_sampling_standard_deviation=0.2,
            reference_sampling_basis_id="finite_world_oracle_contrast",
        ).model_dump(),
        oracle_worlds=10_000,
        oracle_monte_carlo_standard_error=0.001,
        maximum_oracle_monte_carlo_standard_error=0.01,
    )


def _c5_record() -> TrialEvalGradingKeyRecordV1:
    return TrialEvalGradingKeyRecordV1.model_validate(
        {**_evaluator_record().model_dump(), "context": "C5", "data_integrity": FIXTURE_C5_INTEGRITY}
    )


def test_grading_key_supports_a_prespecified_task_subset(tmp_path: Path) -> None:
    """Grade bounded panels from a complete grading key."""

    first = _evaluator_record()
    second = first.model_copy(update={"task_id": f"te_{'3' * 20}", "trial_id": f"trial_{'4' * 20}"})
    write_trialeval_grading_key(tmp_path, (first, second))

    key = load_trialeval_grading_key(tmp_path, expected_task_ids=(first.task_id,))

    assert key.for_task(first.task_id) == first
    assert key.manifest.cell_count == key.manifest.series_count == key.manifest.world_count == 1


def test_grading_key_rejects_a_missing_scheduled_task(tmp_path: Path) -> None:
    """Keep every scheduled task in the grading denominator."""

    write_trialeval_grading_key(tmp_path, (_evaluator_record(),))

    with pytest.raises(ParticipantReleaseError, match="missing scheduled tasks"):
        load_trialeval_grading_key(tmp_path, expected_task_ids=(f"te_{'5' * 20}",))


def test_grading_key_rejects_undeclared_evaluator_files(tmp_path: Path) -> None:
    """A key archive holds only its inventoried files beside the participant role."""

    write_trialeval_grading_key(tmp_path, (_evaluator_record(),))
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "evaluation" / "construct.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ParticipantReleaseError, match="differ from the grading-key inventory"):
        load_trialeval_grading_key(tmp_path)


def test_trial_eval_run_contract_rejects_superseded_identity() -> None:
    payload = _run_config_payload()
    payload["data_version"] = "invalid_release"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TrialEvalRunConfigV1.model_validate(payload)


def test_trial_eval_run_contract_rejects_judge_configuration() -> None:
    payload = _run_config_payload()
    payload["judge_model"] = "judge-model"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TrialEvalRunConfigV1.model_validate(payload)


def test_trial_eval_run_contract_requires_positive_item_watchdog() -> None:
    payload = _run_config_payload()
    payload["item_watchdog_seconds"] = 0

    with pytest.raises(ValidationError, match="item_watchdog_seconds"):
        TrialEvalRunConfigV1.model_validate(payload)


@pytest.mark.parametrize(
    "field",
    (
        "participant_archive_sha256",
        "participant_manifest_checksum",
        "prompt_set_sha256",
        "scorer_source_sha256",
        "harness_source_sha256",
        "run_identity_sha256",
    ),
)
def test_trial_eval_run_contract_rejects_omitted_authoritative_identity(field: str) -> None:
    payload = _run_config_payload()
    del payload[field]

    with pytest.raises(ValidationError, match=field):
        TrialEvalRunConfigV1.model_validate(payload)


def test_trial_eval_item_rejects_lossy_extracted_score_authority() -> None:
    payload = {
        "item_id": "TASK1",
        "timestamp_utc": datetime.now(UTC),
        "run_config": _run_config_payload(),
        "agent_output": {
            "status": "failed",
            "failure_type": "runtime_error",
            "condition_provenance": {
                "procedure_assistance": "output_contract_only",
                "analysis_specification": "locked_sap",
                "analysis_surface_sha256": "a" * 64,
                "max_turns": 25,
                "prompt_set_sha256": "b" * 64,
                "rendered_system_prompt_sha256": "b" * 64,
                "tool_schema_sha256": "c" * 64,
            },
        },
        "extracted": {"primary": {"estimator": "coxph_binary"}},
    }

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TrialEvalItemResultV1.model_validate(payload)


def test_trial_eval_item_rejects_analysis_specification_drift() -> None:
    payload = {
        "item_id": "TASK1",
        "timestamp_utc": datetime.now(UTC),
        "run_config": _run_config_payload(),
        "agent_output": {
            "status": "failed",
            "failure_type": "runtime_error",
            "condition_provenance": {
                "procedure_assistance": "output_contract_only",
                "analysis_specification": "protocol_only",
                "analysis_surface_sha256": "a" * 64,
                "max_turns": 25,
                "prompt_set_sha256": "b" * 64,
                "rendered_system_prompt_sha256": "c" * 64,
                "tool_schema_sha256": "d" * 64,
            },
        },
    }

    with pytest.raises(ValidationError, match="condition provenance"):
        TrialEvalItemResultV1.model_validate(payload)


@pytest.mark.parametrize("planned", (False, True))
@pytest.mark.parametrize(
    "local_replay,submitted", [(False, False), (True, False), pytest.param(False, True, marks=pytest.mark.executor)]
)
def test_trial_eval_offline_grader_retains_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    local_replay: bool,
    submitted: bool,
    planned: bool,
) -> None:
    suite = tmp_path / "suite"
    participant = suite / "participant"
    participant.mkdir(parents=True)
    write_json_model(
        suite / "participant_archive_identity.json",
        PairedParticipantArchiveIdentityV1.create(
            release_id=f"release_{'1' * 64}",
            participant_archive_sha256="a" * 64,
            participant_manifest_checksum="f" * 64,
        ),
    )
    record = _planned_record() if planned else _evaluator_record()
    write_trialeval_grading_key(suite, (record,))
    evaluator_archive = tmp_path / "checksum-bound-evaluator.zip"
    evaluator_archive.write_bytes(b"evaluator")
    run_dir = tmp_path / "run"
    executor = None
    if submitted:
        from trialagentbench_verifier.execution_contracts import RELEASE_EXECUTION_LIMITS_V1
        from trialagentbench_verifier.executor import resolve_executor_environment

        executor = resolve_executor_environment(limits=RELEASE_EXECUTION_LIMITS_V1).model_dump(mode="python")
    run_config = TrialEvalRunConfigV1.model_validate(_run_config_payload(task_id=TASK_ID, executor=executor))
    local_executor = run_config.executor.model_copy(
        update={"image_id": "sha256:" + "8" * 64, "image_reference": "sha256:" + "8" * 64}
    )

    def resolve_local_executor(*, image, expected_image_id, expected_environment_sha256, limits):
        assert image == expected_image_id == local_executor.image_id
        assert expected_environment_sha256 == run_config.executor.semantic_sha256()
        assert limits == run_config.executor.limits
        return local_executor

    monkeypatch.setattr(grade_trialeval, "resolve_trusted_executor_environment", resolve_local_executor)
    write_json_model(run_dir / "run_config.json", run_config)
    write_json_model(
        run_dir / "items" / f"{TASK_ID}.json",
        TrialEvalItemResultV1(
            item_id=TASK_ID,
            timestamp_utc=datetime.now(UTC),
            run_config=run_config,
            agent_output={
                "status": "success" if submitted else "failed",
                "failure_type": None if submitted else "turn_limit_no_submission",
                "report": "Estimated effect with uncertainty." if submitted else None,
                "turns_used": 25,
                "condition_provenance": {
                    "procedure_assistance": "output_contract_only",
                    "analysis_specification": "locked_sap",
                    "analysis_surface_sha256": "a" * 64,
                    "max_turns": 25,
                    "prompt_set_sha256": "b" * 64,
                    "rendered_system_prompt_sha256": "c" * 64,
                    "tool_schema_sha256": "d" * 64,
                },
            },
        ),
    )
    item = BenchmarkItem(
        item_id=TASK_ID,
        task_id=TASK_ID,
        trial_name="fixture",
        design_tier="undisclosed",
        design_subtype="individual_randomized",
        assumption_tier="undisclosed",
        context_tier="C1",
        data_preparation="analysis_ready",
        analysis_specification="locked_sap",
        visible_dir=participant,
        data_dir=participant,
        task={},
    )

    monkeypatch.setattr(
        grade_trialeval,
        "load_participant_release",
        lambda _: SimpleNamespace(
            root=participant,
            manifest=SimpleNamespace(release_id=f"release_{'1' * 64}", checksum="f" * 64),
        ),
    )
    monkeypatch.setattr(grade_trialeval, "discover_participant_items", lambda *_args, **_kwargs: {TASK_ID: item})
    evidence = _numerical_replay().evidence_artifacts
    if submitted:
        import hashlib

        import pandas as pd
        from trialagentbench_verifier.execution_contracts import ReplayInputArtifactV1

        workspace = run_dir / "logs" / f"{TASK_ID}_workspace" / "item"
        (workspace / "data").mkdir(parents=True)
        (workspace / "scratch").mkdir()
        data = workspace / "data" / "example.parquet"
        pd.DataFrame({"value": [0.5, 1.5]}).to_parquet(data)
        evidence = (
            ReplayInputArtifactV1(
                relative_path="data/example.parquet",
                sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
                byte_length=data.stat().st_size,
            ),
        )
        (workspace / "scratch" / "analysis.py").write_text(
            "import csv\nimport pandas as pd\n"
            "value = float(pd.read_parquet('data/example.parquet').value.mean())\n"
            "with open('scratch/estimates.csv', 'w', newline='') as handle:\n"
            "    writer = csv.writer(handle)\n"
            "    writer.writerow(['target_id','estimate','lower','upper'])\n"
            "    writer.writerow(['estimand',value,value-0.5,value+0.5])\n"
        )
        (workspace / "scratch" / "estimates.csv").write_text("target_id,estimate,lower,upper\nestimand,1,0.5,1.5\n")
    monkeypatch.setattr(grade_trialeval, "participant_evidence_artifacts", lambda _: evidence)

    @contextlib.contextmanager
    def materialized_release_archive(
        path: Path, *, expected_sha256: str, task_ids: tuple[str, ...] | None = None, in_place: bool = False
    ):
        assert path == evaluator_archive
        assert task_ids == (TASK_ID,)
        assert expected_sha256 == "9" * 64
        yield SimpleNamespace(root=suite, archive_sha256=expected_sha256)

    monkeypatch.setattr(grade_trialeval, "materialized_release_archive", materialized_release_archive)

    monkeypatch.setattr(grade_trialeval, "head_sha", lambda _: "")
    graded = tmp_path / "graded"
    assert (
        grade_trialeval.grade_trialeval_run(
            [
                str(run_dir),
                "--evaluator-release",
                str(evaluator_archive),
                "--expected-sha256",
                "9" * 64,
                "--out-dir",
                str(graded),
                *(["--replay-executor-image-id", local_executor.image_id] if local_replay else []),
            ]
        )
        == 0
    )
    manifest = json.loads((graded / "GRADE_MANIFEST.json").read_text())
    assert manifest["harness_git_sha"] is None
    assert manifest["harness_source_sha256"] == grade_trialeval.harness_source_sha256_v1()
    replay_environment = json.loads((graded / "replay_executor.json").read_text())
    assert replay_environment["image_id"] == (local_executor if local_replay else run_config.executor).image_id
    assert read_json_model(TrialEvalRunConfigV1, graded / "run_config.json") == run_config
    scored = read_json_model(TrialEvalItemResultV1, graded / "items" / f"{TASK_ID}.json")
    assert scored.scores is not None
    assert scored.scores.outcome.included_in_denominator is True
    assert scored.scores.outcome.completion.status == ("satisfied" if submitted else "not_satisfied")
    if submitted:
        from trialagentbench.analysis.trialeval_score_rows import iter_trialeval_score_rows

        assert scored.scores.replay.status == "reproduced"
        assert scored.scores.outcome.point_performance.native_absolute_error == 0
        row = next(iter_trialeval_score_rows(graded))
        assert row.replay.estimates[0].estimate == 1
        assert row.observed_attempt_outcome == "completed"
    else:
        assert scored.scores.outcome.failure_origins == ("missing_submission",)
        assert scored.scores.replay.status == "invalid_program"
    verification_input = read_json_model(
        TrialEvalScoreInputV1,
        graded / "verification_inputs" / f"{TASK_ID}.json",
    )
    assert verification_input.task_id == TASK_ID
    assert verification_input.attempt_status == ("complete" if submitted else "missing")
    from trialagentbench_verifier.campaign import _campaign_row
    from trialagentbench_verifier.trialeval import TrialEvalScoreInputV1 as IndependentScoreInput

    independent = _campaign_row(
        json.loads((graded / "items" / f"{TASK_ID}.json").read_text()),
        source_sha256="c" * 64,
        score_input_sha256="d" * 64,
        score_input=IndependentScoreInput.model_validate(verification_input.model_dump(mode="json")),
    )
    assert independent.observed_attempt_outcome == ("completed" if submitted else "model_noncompletion")
    # The unreported plan as executed is inaccurate, apart from the task's loss and decision.
    assert (scored.scores.plan_execution is not None) is planned
    from trialagentbench.analysis.trialeval_score_rows import iter_trialeval_score_rows

    assert next(iter_trialeval_score_rows(graded)).plan_execution == scored.scores.plan_execution
    assert (verification_input.plan_input is not None) is planned
    assert (independent.plan_execution is not None) is planned
    assert independent.plan_execution is None or independent.plan_execution.accurate is False
    from trialagentbench_verifier.campaign import _BoundAttempt, _row_metrics

    attempt = SimpleNamespace(
        attempt_id="a",
        model_id="m",
        task_id=TASK_ID,
        trial_id=TRIAL_ID,
        base_world_id=f"world_{'2' * 20}",
        cell_ref=FIXTURE_CELL_KEY,
        effect_state="favourable",
        context="C1",
        target_kind="point",
    )
    metrics = {row.metric.metric_id: row.value for row in _row_metrics(_BoundAttempt(attempt, independent))}
    assert metrics.get("plan_execution_accuracy") == (0.0 if planned else None)
    from trialagentbench_verifier.campaign import failure_cause_v1, reproduced_answer_v1

    # Replay scoring replays exactly the answers whose saved script reproduced them.
    assert reproduced_answer_v1(independent) == ((1.0, 0.5, 1.5) if submitted else None)
    assert failure_cause_v1(independent) == (None if submitted else "no_answer")
    if submitted:
        assert independent.outcome.point_performance.normalized_absolute_error == 0
        assert independent.replay_program_artifacts == scored.scores.replay.program_artifacts
        changed = verification_input.model_copy(
            update={"submitted_result": verification_input.submitted_result.model_copy(update={"value": 999})}
        )
        with pytest.raises(ValueError, match="certified CSV numbers"):
            _campaign_row(
                json.loads((graded / "items" / f"{TASK_ID}.json").read_text()),
                source_sha256="c" * 64,
                score_input_sha256="d" * 64,
                score_input=IndependentScoreInput.model_validate(changed.model_dump(mode="json")),
            )

    summary = json.loads((graded / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_items"] == 1
    assert summary["component_counts"]["completion"] == {"satisfied" if submitted else "not_satisfied": 1}

    write_json_model(
        suite / "participant_archive_identity.json",
        PairedParticipantArchiveIdentityV1.create(
            release_id=f"release_{'1' * 64}",
            participant_archive_sha256="b" * 64,
            participant_manifest_checksum="f" * 64,
        ),
    )
    with pytest.raises(ValueError, match="paired participant archive"):
        grade_trialeval.grade_trialeval_run(
            [
                str(run_dir),
                "--evaluator-release",
                str(evaluator_archive),
                "--expected-sha256",
                "9" * 64,
                "--out-dir",
                str(tmp_path / "regraded"),
            ]
        )


def test_staged_directory_publishes_complete_output(tmp_path: Path) -> None:
    destination = tmp_path / "graded"

    with staged_directory(destination) as staging:
        (staging / "GRADE_MANIFEST.json").write_text("{}", encoding="utf-8")
        assert not destination.exists()

    assert (destination / "GRADE_MANIFEST.json").is_file()


def test_staged_directory_removes_failed_output(tmp_path: Path) -> None:
    destination = tmp_path / "graded"

    with pytest.raises(RuntimeError, match="scoring failed"):
        with staged_directory(destination) as staging:
            (staging / "partial.json").write_text("{}", encoding="utf-8")
            raise RuntimeError("scoring failed")

    assert not destination.exists()
    assert not tuple(tmp_path.glob(".graded.*"))


def test_staged_directory_refuses_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "graded"
    destination.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        with staged_directory(destination):
            pytest.fail("existing output must not be entered")


def test_c5_grades_recovery_without_a_prescribed_intermediate_table() -> None:
    """Source access and numerical recovery are separate from the agent's table layout."""
    record = _c5_record()
    replay = _numerical_replay()
    bound = bind_trialeval_numerical_input(
        evaluator_record=record, replay=replay, expected_evidence=replay.evidence_artifacts
    )
    grade = grade_trialeval_outcome_v1(bound)
    assert bound.evidence_eligible is True
    assert grade.point_performance is not None


def test_c5_join_errors_change_grade_but_source_order_does_not() -> None:
    """Grade recovered numbers, without a join declaration or cleaned-table gate."""
    assignments = {0: "active", 1: "active", 2: "control", 3: "control"}
    endpoint_rows = [(2, 0.0), (3, 0.0), (0, 1.0), (1, 1.0)]
    record = _c5_record()

    def contrast(rows: list[tuple[int, float]]) -> float:
        active = [value for participant, value in rows if assignments[participant] == "active"]
        control = [value for participant, value in rows if assignments[participant] == "control"]
        return sum(active) / len(active) - sum(control) / len(control)

    def grade_value(value: float) -> float:
        from trialagentbench_verifier.numerical import NumericalEstimate

        replay = _numerical_replay().model_copy(
            update={
                "estimates": (
                    NumericalEstimate(target_id="estimand", estimate=value, lower=value - 0.2, upper=value + 0.2),
                )
            }
        )
        bound = bind_trialeval_numerical_input(
            evaluator_record=record, replay=replay, expected_evidence=replay.evidence_artifacts
        )
        assert bound.evidence_eligible
        return grade_trialeval_outcome_v1(bound).signal_loss

    correctly_joined = contrast(endpoint_rows)
    reordered = contrast(list(reversed(endpoint_rows)))
    positional_join = contrast(list(zip(assignments, [value for _, value in endpoint_rows], strict=True)))
    assert correctly_joined == reordered == 1.0
    assert positional_join == -1.0
    assert grade_value(correctly_joined) == grade_value(reordered)
    assert grade_value(correctly_joined) < grade_value(positional_join)


def _numerical_replay(*, missing: bool = False):
    from trialagentbench_verifier.execution_contracts import CodeExecutionResultV1, ReplayInputArtifactV1
    from trialagentbench_verifier.numerical import NumericalEstimate, NumericalReplay

    evidence = (ReplayInputArtifactV1(relative_path="data/example.parquet", sha256="a" * 64, byte_length=10),)
    execution = CodeExecutionResultV1(status="success", output="", elapsed_seconds=0.01)
    return NumericalReplay(
        status="reproduced",
        entrypoint="analysis.py",
        target_ids=("estimand",),
        evidence_artifacts=evidence,
        submitted_csv_sha256="b" * 64,
        replayed_csv_sha256s=("b" * 64, "b" * 64),
        executions=(execution, execution),
        estimates=() if missing else (NumericalEstimate(target_id="estimand", estimate=1, lower=0.5, upper=1.5),),
    )


def test_csv_binding_uses_task_metadata_and_existing_numerical_grade() -> None:
    """Minimal numerical output reaches the same scientific grade as a full result."""
    from trialagentbench.trialeval.outcome_submission import bind_trialeval_numerical_input

    record = _evaluator_record()
    replay = _numerical_replay()
    bound = bind_trialeval_numerical_input(
        evaluator_record=record, replay=replay, expected_evidence=replay.evidence_artifacts
    )
    assert bound.attempt_status == "complete"
    assert bound.submitted_result is not None
    assert bound.submitted_result.unit == "points"
    assert bound.submitted_result.conclusion == "favors_treatment"
    grade = grade_trialeval_outcome_v1(bound)
    assert grade.completion.status == "satisfied"


def _paired_evaluator_record() -> TrialEvalGradingKeyRecordV1:
    from trialagentbench.grading.outcome import TrialEvalGradingReferenceV1

    primary = _evaluator_record()
    payload = primary.model_dump(include=set(TrialEvalGradingReferenceV1.model_fields))
    payload["target"]["binding"]["estimand_id"] = "complement"
    complement = TrialEvalGradingReferenceV1.model_validate(payload)
    return TrialEvalGradingKeyRecordV1.model_validate(
        primary.model_dump() | {"additional_references": (complement.model_dump(),)}
    )


@pytest.mark.parametrize("reported_target", ["estimand", "complement"])
def test_csv_partial_pair_retains_reported_target_and_marks_missing_target(reported_target: str) -> None:
    """Either requested result can be missing without discarding the available result."""
    record = _paired_evaluator_record()
    replay = _numerical_replay()
    replay = replay.model_copy(
        update={
            "target_ids": record.target_ids,
            "estimates": (replay.estimates[0].model_copy(update={"target_id": reported_target}),),
        }
    )
    for target_id in record.target_ids:
        bound = bind_trialeval_numerical_input(
            evaluator_record=record, replay=replay, expected_evidence=replay.evidence_artifacts, target_id=target_id
        )
        grade = grade_trialeval_outcome_v1(bound)
        if target_id == reported_target:
            assert bound.attempt_status == "complete"
            assert bound.submitted_result.value == 1
            assert grade.signal_loss == pytest.approx(2.5)
        else:
            assert bound.attempt_status == "missing_required_output"
            assert bound.submitted_result is None
            assert grade.signal_loss is None
            assert grade.completion.status == "not_satisfied"
        assert grade.included_in_denominator


@pytest.mark.parametrize("invalid_ids", [("estimand", "estimand"), ("estimand", "unassigned")])
@pytest.mark.parametrize("selected_target", ["estimand", "complement"])
def test_csv_partial_pair_still_rejects_duplicate_or_unassigned_rows(
    invalid_ids: tuple[str, ...], selected_target: str
) -> None:
    """A missing selected target must not bypass validation of the submitted rows."""
    record = _paired_evaluator_record()
    replay = _numerical_replay()
    replay = replay.model_copy(
        update={
            "target_ids": record.target_ids,
            "estimates": tuple(replay.estimates[0].model_copy(update={"target_id": target}) for target in invalid_ids),
        }
    )
    with pytest.raises(ValueError, match="numerical estimates do not match"):
        bind_trialeval_numerical_input(
            evaluator_record=record,
            replay=replay,
            expected_evidence=replay.evidence_artifacts,
            target_id=selected_target,
        )


@pytest.mark.parametrize("reported_target", ["estimand", "complement"])
def test_offline_grader_retains_partial_pair_and_continues_to_next_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reported_target: str
) -> None:
    """A replayed partial result leaves task loss unavailable and does not abort the run."""
    from trialagentbench_verifier.campaign import _campaign_row
    from trialagentbench_verifier.execution_contracts import ReplayInputArtifactV1
    from trialagentbench_verifier.trialeval import TrialEvalScoreInputV1 as IndependentScoreInput

    first = _paired_evaluator_record()
    second = _evaluator_record().model_copy(update={"task_id": f"te_{'3' * 20}", "trial_id": f"trial_{'4' * 20}"})
    records = (first, second)
    task_ids = tuple(record.task_id for record in records)
    suite = tmp_path / "suite"
    participant = suite / "participant"
    participant.mkdir(parents=True)
    write_json_model(
        suite / "participant_archive_identity.json",
        PairedParticipantArchiveIdentityV1.create(
            release_id=first.release_id,
            participant_archive_sha256="a" * 64,
            participant_manifest_checksum="f" * 64,
        ),
    )
    write_trialeval_grading_key(suite, records)
    evaluator_archive = tmp_path / "evaluator.zip"
    evaluator_archive.write_bytes(b"evaluator")
    run_dir = tmp_path / "run"
    run_config = TrialEvalRunConfigV1.model_validate(_run_config_payload(task_ids=task_ids))
    write_json_model(run_dir / "run_config.json", run_config)
    for task_id in task_ids:
        write_json_model(
            run_dir / "items" / f"{task_id}.json",
            TrialEvalItemResultV1(
                item_id=task_id,
                timestamp_utc=datetime.now(UTC),
                run_config=run_config,
                agent_output={
                    "status": "success",
                    "report": "Estimated effect with uncertainty.",
                    "turns_used": 1,
                    "condition_provenance": {
                        "procedure_assistance": "output_contract_only",
                        "analysis_specification": "locked_sap",
                        "analysis_surface_sha256": "a" * 64,
                        "max_turns": 25,
                        "prompt_set_sha256": "b" * 64,
                        "rendered_system_prompt_sha256": "c" * 64,
                        "tool_schema_sha256": "d" * 64,
                    },
                },
            ),
        )
    monkeypatch.setattr(
        grade_trialeval,
        "load_participant_release",
        lambda _: SimpleNamespace(
            root=participant, manifest=SimpleNamespace(release_id=first.release_id, checksum="f" * 64)
        ),
    )
    monkeypatch.setattr(
        grade_trialeval,
        "discover_participant_items",
        lambda *_args, **_kwargs: {
            task_id: SimpleNamespace(trial_name="fixture", estimand_mode="") for task_id in task_ids
        },
    )
    full_replay = _numerical_replay().model_copy(
        update={
            "program_artifacts": (
                ReplayInputArtifactV1(relative_path="scratch/analysis.py", sha256="e" * 64, byte_length=100),
            )
        }
    )
    partial_replay = full_replay.model_copy(
        update={
            "target_ids": first.target_ids,
            "estimates": (full_replay.estimates[0].model_copy(update={"target_id": reported_target}),),
        }
    )
    monkeypatch.setattr(grade_trialeval, "participant_evidence_artifacts", lambda _: full_replay.evidence_artifacts)
    replayed_tasks = []

    def replay_program(**kwargs):
        replayed_tasks.append(kwargs["target_ids"])
        return partial_replay if kwargs["target_ids"] == first.target_ids else full_replay

    monkeypatch.setattr(grade_trialeval, "replay_numerical_program", replay_program)
    monkeypatch.setattr(grade_trialeval, "head_sha", lambda _: "")
    graded = tmp_path / "graded"
    assert (
        grade_trialeval._grade_materialized_run(
            source=run_dir, suite_dir=suite, suite_source=evaluator_archive, output=graded
        )
        == 0
    )
    partial = read_json_model(TrialEvalItemResultV1, graded / "items" / f"{first.task_id}.json").scores
    complete = read_json_model(TrialEvalItemResultV1, graded / "items" / f"{second.task_id}.json").scores
    assert partial.signal_loss is None
    assert sum(result.outcome.signal_loss is not None for result in partial.target_results) == 1
    assert all(result.outcome.included_in_denominator for result in partial.target_results)
    assert complete.signal_loss == pytest.approx(2.5)
    partial_row = grade_trialeval._flat_score_row(score=partial)
    assert partial_row["required_targets"] == 2
    assert partial_row["scored_targets"] == 1
    assert partial_row["signal_loss"] is None
    assert partial_row["completion"] == "not_satisfied"
    assert "missing_required_output" in partial_row["failure_origins"]
    independent = _campaign_row(
        json.loads((graded / "items" / f"{first.task_id}.json").read_text()),
        source_sha256="c" * 64,
        score_input_sha256="d" * 64,
        score_input=IndependentScoreInput.model_validate_json(
            (graded / "verification_inputs" / f"{first.task_id}.json").read_text()
        ),
    )
    assert independent.outcome.signal_loss is None
    assert independent.outcome.completion.status == "not_satisfied"
    assert independent.observed_attempt_outcome != "completed"
    assert len(independent.outcome.target_outcomes) == 2
    manifest = json.loads((graded / "GRADE_MANIFEST.json").read_text())
    assert manifest["suite_task_count"] == manifest["graded_task_count"] == 2
    summary = json.loads((graded / "summary.json").read_text())
    assert summary["n_items"] == 2
    assert summary["n_scored_items"] == 1
    assert summary["n_evaluator_errors"] == 0
    assert summary["component_counts"]["completion"] == {"not_satisfied": 1, "satisfied": 1}
    balanced = json.loads((graded / "balanced_summary.json").read_text())
    metrics = {metric["quantity_id"]: metric for metric in balanced["context_cells"][0]["comparable_metrics"]}
    assert metrics["signal_loss"]["scheduled_count"] == 2
    assert metrics["signal_loss"]["observed_count"] == 1
    assert metrics["signal_loss"]["mean"] is None
    assert metrics["observed_signal_loss"]["mean"] == pytest.approx(2.5)

    original_bytes = {path: path.read_bytes() for path in run_dir.rglob("*") if path.is_file()}
    selected_grade = tmp_path / "selected-grade"
    assert (
        grade_trialeval._grade_materialized_run(
            source=run_dir,
            suite_dir=suite,
            suite_source=evaluator_archive,
            output=selected_grade,
            selected_task_ids=frozenset({second.task_id}),
        )
        == 0
    )
    assert {path.stem for path in (selected_grade / "items").glob("*.json")} == {second.task_id}
    assert (selected_grade / "run_config.json").read_bytes() == original_bytes[run_dir / "run_config.json"]
    assert (
        read_json_model(TrialEvalItemResultV1, selected_grade / "items" / f"{second.task_id}.json").run_config
        == run_config
    )
    assert all(path.read_bytes() == raw for path, raw in original_bytes.items())
    selected_manifest = json.loads((selected_grade / "GRADE_MANIFEST.json").read_bytes())
    assert selected_manifest["suite_task_count"] == 2
    assert selected_manifest["graded_task_count"] == 1
    assert any("Explicit derived grade selection" in note for note in selected_manifest["notes"])
    for invalid in (frozenset(), frozenset({"unscheduled"})):
        with pytest.raises(ValueError, match="Selected TrialEval tasks"):
            grade_trialeval._grade_materialized_run(
                source=run_dir,
                suite_dir=suite,
                suite_source=evaluator_archive,
                output=tmp_path / "invalid-selection",
                selected_task_ids=invalid,
            )

    # Consume a derived cross-campaign selection through the actual independent
    # grade reader and attempt binder, without editing either native attempt.
    import shutil
    from datetime import timedelta

    from trialagentbench_verifier.campaign import (
        TrialEvalCampaignPlanV1,
        _bind_attempts,
        load_trial_eval_campaign_grade_source_v1,
    )

    from trialagentbench.contracts.campaign import CampaignRunV1
    from trialagentbench.io import sha256_file
    from trialagentbench.tools.primary_selection import (
        PRIMARY_SELECTION,
        TrialEvalEarlierCompletionV1,
        TrialEvalPrimarySelectionV1,
        select_trialeval_primary_sources,
        validate_earlier_completion,
    )

    later = tmp_path / "runs" / "later"
    shutil.copytree(run_dir, later)
    for task in task_ids:
        path = later / "items" / f"{task}.json"
        value = read_json_model(TrialEvalItemResultV1, path)
        write_json_model(path, value.model_copy(update={"timestamp_utc": value.timestamp_utc + timedelta(days=1)}))
    later_bytes = {path: path.read_bytes() for path in later.rglob("*") if path.is_file()}
    run = CampaignRunV1(
        job_id="later-primary",
        benchmark="trialeval",
        model_name="fixture",
        repetition=1,
        relative_run_root="runs/later",
        invocation_sha256="a" * 64,
        analysis_plan_path="plans/later.json",
        analysis_plan_sha256="b" * 64,
    )
    later_grade = tmp_path / "results" / run.job_id / "grade"
    grade_trialeval._grade_materialized_run(
        source=later, suite_dir=suite, suite_source=evaluator_archive, output=later_grade
    )
    audit = tmp_path / "condition-audit.json"
    audit.write_text('{"same_condition": true}\n')
    origin = TrialEvalEarlierCompletionV1(
        target_job_id=run.job_id,
        task_id=second.task_id,
        target_run_config_sha256=sha256_file(later / "run_config.json"),
        source_run_root=run_dir,
        source_run_config_sha256=sha256_file(run_dir / "run_config.json"),
        source_item_sha256=sha256_file(run_dir / "items" / f"{second.task_id}.json"),
    )
    selection = TrialEvalPrimarySelectionV1(audit_path=audit, audit_sha256=sha256_file(audit), selections=(origin,))
    write_json_model(tmp_path / PRIMARY_SELECTION, selection)
    combined = TrialEvalCampaignPlanV1.model_validate(
        {
            "release_id": first.release_id,
            "release_manifest_sha256": selected_manifest["release_manifest_sha256"],
            "participant_manifest_checksum": run_config.participant_manifest_checksum,
            "prospective_analysis_plan_sha256": "a" * 64,
            "cells": [{"cell_ref": first.cell_ref, "series_ref": first.series_ref}],
            "attempts": [
                {
                    "attempt_id": record.task_id,
                    "scheduled_unit_id": record.task_id,
                    "attempt_role": "primary",
                    "model_id": run_config.model,
                    "run_identity_sha256": run_config.run_identity_sha256,
                    "task_id": record.task_id,
                    "trial_id": record.trial_id,
                    "base_world_id": record.base_world_id,
                    "series_ref": record.series_ref,
                    "cell_ref": record.cell_ref,
                    "effect_state": record.effect_state,
                    "context": record.context,
                    "condition_id": run_config.experiment_condition.condition_id,
                    "request_replicate_id": run_config.experiment_condition.request_replicate_id,
                    "procedure_assistance": run_config.experiment_condition.procedure_assistance,
                    "analysis_specification": run_config.task_evidence_factors[record.task_id].analysis_specification,
                    "design_family": record.design_family,
                    "reconstruction_family": None,
                    "target_ids": record.target_ids,
                    "target_kind": "point",
                    "effect_measure": record.target.binding.effect_measure,
                    "result_unit": record.target.binding.result_unit,
                }
                for record in records
            ],
        }
    )

    def grade_selected(arguments):
        tasks = frozenset(arguments[index + 1] for index, value in enumerate(arguments) if value == "--task-id")
        return grade_trialeval._grade_materialized_run(
            source=Path(arguments[0]),
            suite_dir=suite,
            suite_source=evaluator_archive,
            output=Path(arguments[arguments.index("--out-dir") + 1]),
            selected_task_ids=tasks,
        )

    monkeypatch.setattr(grade_trialeval, "grade_trialeval_run", grade_selected)
    selected_plan, selected_roots, natives = select_trialeval_primary_sources(
        tmp_path,
        combined,
        (run,),
        views=tmp_path / "selection-views",
        evaluator=evaluator_archive,
        evaluator_sha256="e" * 64,
    )
    source = load_trial_eval_campaign_grade_source_v1(
        "",
        expected_release_id=combined.release_id,
        expected_release_manifest_sha256=combined.release_manifest_sha256,
        expected_archive_sha256=None,
        grade_roots=selected_roots,
    )
    bound = _bind_attempts(selected_plan, source)
    by_task = {attempt.plan.task_id: attempt.row for attempt in bound}
    assert len(bound) == 2
    assert by_task[first.task_id].source_json_sha256 == sha256_file(later_grade / "items" / f"{first.task_id}.json")
    assert by_task[second.task_id].source_json_sha256 == sha256_file(
        selected_roots[-1] / "items" / f"{second.task_id}.json"
    )
    assert natives == (later, run_dir)
    assert all(path.read_bytes() == raw for path, raw in original_bytes.items())
    assert all(path.read_bytes() == raw for path, raw in later_bytes.items())
    for invalid in (
        origin.model_copy(update={"source_item_sha256": "0" * 64}),
        origin.model_copy(
            update={
                "source_run_root": later,
                "source_item_sha256": sha256_file(later / "items" / f"{second.task_id}.json"),
            }
        ),
    ):
        with pytest.raises(ValueError, match="custody changed|earlier completed"):
            validate_earlier_completion(invalid, later)
    different = tmp_path / "different-condition"
    shutil.copytree(later, different)
    payload = run_config.model_dump(mode="json", exclude={"run_identity_sha256"})
    payload["item_watchdog_seconds"] = run_config.item_watchdog_seconds + 1
    different_config = TrialEvalRunConfigV1.create(**payload)
    write_json_model(different / "run_config.json", different_config)
    path = different / "items" / f"{second.task_id}.json"
    value = read_json_model(TrialEvalItemResultV1, path)
    write_json_model(path, value.model_copy(update={"run_config": different_config}))
    with pytest.raises(ValueError, match="scientific condition"):
        validate_earlier_completion(
            origin.model_copy(update={"target_run_config_sha256": sha256_file(different / "run_config.json")}),
            different,
        )


def test_paired_csv_rows_bind_by_target_identity_with_separate_reference_scales() -> None:
    """Row ordering cannot exchange the two questions or their normalization."""
    from trialagentbench_verifier.numerical import NumericalEstimate

    from trialagentbench.contracts.scoring.trialeval_scores import TrialEvalItemScoresV1, TrialEvalTargetScoreV1
    from trialagentbench.grading.outcome import TrialEvalGradingReferenceV1

    primary = _evaluator_record()
    reference = TrialEvalGradingReferenceV1.model_validate(
        primary.model_dump(include=set(TrialEvalGradingReferenceV1.model_fields))
    )
    payload = reference.model_dump()
    payload["target"]["binding"]["estimand_id"] = "complement"
    payload["target"]["reference_sampling_standard_deviation"] = 2.0
    payload["target"]["value"] = 3.0
    complement = TrialEvalGradingReferenceV1.model_validate(payload)
    record = TrialEvalGradingKeyRecordV1.model_validate(
        primary.model_dump() | {"additional_references": (complement.model_dump(),)}
    )
    replay = _numerical_replay()
    replay = replay.model_copy(
        update={
            "target_ids": record.target_ids,
            "estimates": (
                NumericalEstimate(target_id="complement", estimate=3, lower=2, upper=4),
                replay.estimates[0],
            ),
        }
    )
    target_scores = []
    for target_id, expected in (("estimand", 1.0), ("complement", 3.0)):
        bound = bind_trialeval_numerical_input(
            evaluator_record=record, replay=replay, expected_evidence=replay.evidence_artifacts, target_id=target_id
        )
        assert bound.submitted_result.value == expected
        assert bound.target.binding.estimand_id == target_id
        assert grade_trialeval_outcome_v1(bound).signal_loss is not None
        target_scores.append(
            TrialEvalTargetScoreV1(
                outcome=grade_trialeval_outcome_v1(bound),
                target=bound.target,
                oracle_worlds=record.oracle_worlds,
                oracle_monte_carlo_standard_error=record.oracle_monte_carlo_standard_error,
                maximum_oracle_monte_carlo_standard_error=record.maximum_oracle_monte_carlo_standard_error,
            )
        )
    score = TrialEvalItemScoresV1(
        scorer=current_scorer(),
        **target_scores[0].model_dump(),
        item_id=record.task_id,
        task_id=record.task_id,
        release_id=record.release_id,
        trial_id=record.trial_id,
        base_world_id=record.base_world_id,
        cell_ref=FIXTURE_CELL_KEY,
        effect_state="favourable",
        context_tier=record.context,
        effect_measure=record.target.binding.effect_measure,
        result_unit=record.target.binding.result_unit,
        model="test",
        turns_used=1,
        agent_status="success",
        replay=replay,
        additional_results=(target_scores[1],),
    )
    assert score.signal_loss == pytest.approx(sum(item.outcome.signal_loss for item in target_scores) / 2)
    missing = target_scores[1].model_copy(
        update={"outcome": target_scores[1].outcome.model_copy(update={"signal_loss": None})}
    )
    assert score.model_copy(update={"additional_results": (missing,)}).signal_loss is None
    extreme = score.model_copy(
        update={
            "outcome": score.outcome.model_copy(update={"signal_loss": 1e308}),
            "additional_results": (
                target_scores[1].model_copy(
                    update={"outcome": target_scores[1].outcome.model_copy(update={"signal_loss": 1e308})}
                ),
            ),
        }
    )
    assert extreme.signal_loss == 1e308
    duplicate = primary.model_dump() | {"additional_references": (reference.model_dump(),)}
    with pytest.raises(ValueError, match="repeat a result identifier"):
        TrialEvalGradingKeyRecordV1.model_validate(duplicate)


@pytest.mark.parametrize("missing", [True, False])
def test_csv_missing_or_failed_replay_retains_required_target(missing: bool) -> None:
    """Omitted rows and unreproduced outputs remain failed attempts."""
    from trialagentbench.trialeval.outcome_submission import bind_trialeval_numerical_input

    replay = _numerical_replay(missing=True)
    if not missing:
        replay = replay.model_copy(update={"status": "result_mismatch", "estimates": None})
    bound = bind_trialeval_numerical_input(
        evaluator_record=_evaluator_record(), replay=replay, expected_evidence=replay.evidence_artifacts
    )
    assert bound.attempt_status == "missing_required_output"
    assert bound.submitted_result is None


def test_partial_csv_grades_reported_targets_and_records_the_rest_as_missing() -> None:
    """A partial answer keeps its reported target and fails only the absent one."""
    from trialagentbench.grading.outcome import TrialEvalGradingReferenceV1
    from trialagentbench.trialeval.outcome_submission import bind_trialeval_numerical_input

    primary = _evaluator_record()
    payload = TrialEvalGradingReferenceV1.model_validate(
        primary.model_dump(include=set(TrialEvalGradingReferenceV1.model_fields))
    ).model_dump()
    payload["target"]["binding"]["estimand_id"] = "complement"
    record = TrialEvalGradingKeyRecordV1.model_validate(
        primary.model_dump()
        | {"additional_references": (TrialEvalGradingReferenceV1.model_validate(payload).model_dump(),)}
    )
    replay = _numerical_replay().model_copy(update={"target_ids": record.target_ids})
    bound = {
        target_id: bind_trialeval_numerical_input(
            evaluator_record=record, replay=replay, expected_evidence=replay.evidence_artifacts, target_id=target_id
        )
        for target_id in record.target_ids
    }
    assert bound["estimand"].attempt_status == "complete"
    assert grade_trialeval_outcome_v1(bound["estimand"]).signal_loss is not None
    assert bound["complement"].attempt_status == "missing_required_output"
    assert bound["complement"].submitted_result is None


@pytest.mark.parametrize("changed", ["target", "hash", "undeclared", "empty"])
def test_csv_rejects_wrong_task_or_cutoff(changed: str) -> None:
    """Cross-task or unauthenticated evidence cannot receive a scientific score."""
    from trialagentbench.trialeval.outcome_submission import bind_trialeval_numerical_input

    replay = _numerical_replay()
    expected = replay.evidence_artifacts
    if changed == "target":
        replay = replay.model_copy(update={"target_ids": ("other",)})
    elif changed == "hash":
        expected = (expected[0].model_copy(update={"sha256": "f" * 64}),)
    elif changed == "undeclared":
        expected = (expected[0].model_copy(update={"relative_path": "data/other.parquet"}),)
        replay = replay.model_copy(update={"evidence_artifacts": expected})
    else:
        expected = ()
        replay = replay.model_copy(update={"evidence_artifacts": expected})
    with pytest.raises(ValueError):
        bind_trialeval_numerical_input(evaluator_record=_evaluator_record(), replay=replay, expected_evidence=expected)


@pytest.mark.parametrize("estimate,lower,upper", [(0, 0.5, 1.5), (-1, 0.5, 1.5), (1, 0, 1.5), (1, -1, 1.5)])
def test_csv_invalid_ratio_is_a_failed_answer_not_an_evaluator_failure(
    estimate: float, lower: float, upper: float
) -> None:
    from trialagentbench_verifier.numerical import NumericalEstimate

    from trialagentbench.trialeval.outcome_submission import bind_trialeval_numerical_input

    record = _evaluator_record()
    binding = record.target.binding.model_copy(
        update={"effect_measure": "rate_ratio", "result_unit": "ratio", "null_value": 1.0}
    )
    target = record.target.model_copy(update={"binding": binding})
    record = record.model_copy(update={"target": target})
    replay = _numerical_replay().model_copy(
        update={"estimates": (NumericalEstimate(target_id="estimand", estimate=estimate, lower=lower, upper=upper),)}
    )
    bound = bind_trialeval_numerical_input(
        evaluator_record=record, replay=replay, expected_evidence=replay.evidence_artifacts
    )
    grade = grade_trialeval_outcome_v1(bound)
    assert grade.signal_loss is None
    assert bound.attempt_status == "missing_required_output"


@pytest.mark.parametrize("estimate,lower,upper", [(1e200, 0, 2), (1e308, -1e308, 1e308)])
def test_csv_extreme_finite_answers_preserve_credit_and_serializable_diagnostics(
    estimate: float, lower: float, upper: float
) -> None:
    import json
    import math

    from trialagentbench_verifier.numerical import NumericalEstimate

    from trialagentbench.trialeval.outcome_submission import bind_trialeval_numerical_input

    replay = _numerical_replay().model_copy(
        update={"estimates": (NumericalEstimate(target_id="estimand", estimate=estimate, lower=lower, upper=upper),)}
    )
    bound = bind_trialeval_numerical_input(
        evaluator_record=_evaluator_record(), replay=replay, expected_evidence=replay.evidence_artifacts
    )
    grade = grade_trialeval_outcome_v1(bound)
    assert bound.attempt_status == "complete"
    assert (
        grade.signal_loss is None
        if estimate == 1e308
        else math.isfinite(grade.signal_loss) and grade.signal_loss > 1e190
    )
    assert grade.point_performance is not None
    assert grade.point_performance.submitted_value == estimate
    assert grade.point_performance.native_squared_error is None
    json.dumps(grade.model_dump(mode="json"), allow_nan=False)


def _planned_record() -> TrialEvalGradingKeyRecordV1:
    """Return the C1 fixture with its locked plan's result on the task data."""
    return TrialEvalGradingKeyRecordV1.model_validate(
        _evaluator_record().model_dump()
        | {
            "plan_replay": TrialEvalGradingPlanV1(
                target_id="prespecified_primary_analysis", estimate=0.9, lower=0.4, upper=1.4, tolerance_sd=0.05
            ),
            "decision": TrialEvalDecisionRuleV1(kind="superiority", margin=0.0, reference_decisions=((True,),)),
        }
    )


@pytest.mark.parametrize(
    ("plan_row", "accurate", "point_deviation"),
    (
        ((0.9, 0.4, 1.4), True, 0.0),
        ((0.905, 0.4, 1.4), True, 0.025),
        ((0.0, -1.0, 1.0), False, 4.5),
        (None, False, None),
    ),
)
def test_plan_as_executed_is_graded_apart_from_the_loss_and_decision(
    plan_row: tuple[float, float, float] | None, accurate: bool, point_deviation: float | None
) -> None:
    """Executing the plan is scored separately; the acted-on answer alone drives loss and decision."""
    from trialagentbench_verifier.numerical import NumericalEstimate

    from trialagentbench.contracts.scoring.trialeval_scores import TrialEvalItemScoresV1
    from trialagentbench.grading.outcome import grade_trialeval_decision_v1, grade_trialeval_plan_execution_v1

    record = _planned_record()
    assert record.reported_target_ids == ("estimand", "prespecified_primary_analysis")
    replay = _numerical_replay()
    rows = (
        ()
        if plan_row is None
        else (
            NumericalEstimate(
                target_id="prespecified_primary_analysis", estimate=plan_row[0], lower=plan_row[1], upper=plan_row[2]
            ),
        )
    )
    replay = replay.model_copy(
        update={"target_ids": record.reported_target_ids, "estimates": (*(replay.estimates or ()), *rows)}
    )
    evidence = replay.evidence_artifacts
    bound = bind_trialeval_numerical_input(evaluator_record=record, replay=replay, expected_evidence=evidence)
    plan_input = bind_trialeval_numerical_input(
        evaluator_record=record, replay=replay, expected_evidence=evidence, target_id="prespecified_primary_analysis"
    )
    outcome = grade_trialeval_outcome_v1(bound)
    decision = grade_trialeval_decision_v1(record, (bound,), (outcome,))
    plan = grade_trialeval_plan_execution_v1(record, plan_input)
    assert plan is not None and plan.accurate is accurate
    assert plan.point_deviation == (None if point_deviation is None else pytest.approx(point_deviation))
    # The acted-on answer's loss and decision equal those of the same answer graded without a plan.
    unplanned = _evaluator_record().model_copy(update={"decision": record.decision})
    alone = bind_trialeval_numerical_input(
        evaluator_record=unplanned, replay=_numerical_replay(), expected_evidence=evidence
    )
    assert outcome == grade_trialeval_outcome_v1(alone)
    assert decision == grade_trialeval_decision_v1(unplanned, (alone,), (outcome,))
    assert decision is not None and decision.status == "satisfied"
    score = TrialEvalItemScoresV1(
        scorer=current_scorer(),
        outcome=outcome,
        target=record.target.model_dump(),
        oracle_worlds=record.oracle_worlds,
        oracle_monte_carlo_standard_error=record.oracle_monte_carlo_standard_error,
        maximum_oracle_monte_carlo_standard_error=record.maximum_oracle_monte_carlo_standard_error,
        item_id=record.task_id,
        task_id=record.task_id,
        release_id=record.release_id,
        trial_id=record.trial_id,
        base_world_id=record.base_world_id,
        cell_ref=FIXTURE_CELL_KEY,
        effect_state="favourable",
        context_tier=record.context,
        effect_measure=record.target.binding.effect_measure,
        result_unit=record.target.binding.result_unit,
        model="test",
        turns_used=1,
        agent_status="success",
        replay=replay,
        decision=decision,
        plan_execution=plan,
    )
    assert score.signal_loss == outcome.signal_loss
    with pytest.raises(ValidationError, match="complete replay inventory"):
        TrialEvalItemScoresV1.model_validate(score.model_dump() | {"replay": _numerical_replay()})


def test_plan_replay_is_bound_only_where_a_plan_is_given() -> None:
    """A plan lane outside C1/C3 is rejected, and its row must be in the replayed inventory."""
    record = _planned_record()
    with pytest.raises(ValidationError, match="plan replay is present outside the plan lane"):
        TrialEvalGradingKeyRecordV1.model_validate(record.model_dump() | {"context": "C2"})
    replay = _numerical_replay()
    with pytest.raises(ValueError, match="does not bind the assigned"):
        bind_trialeval_numerical_input(
            evaluator_record=record, replay=replay, expected_evidence=replay.evidence_artifacts
        )


def test_plan_answer_without_the_primary_answer_is_graded_not_raised() -> None:
    """A reported plan row alone leaves the primary target missing; scoring records it instead of raising.

    A complementary row alone can make replay 'reproducible' while the primary outcome is missing; the score
    wrapper must not raise and stop grading then, and the plan lane has the same shape.
    """
    from trialagentbench_verifier.numerical import NumericalEstimate

    from trialagentbench.contracts.scoring.trialeval_scores import TrialEvalItemScoresV1
    from trialagentbench.grading.outcome import grade_trialeval_plan_execution_v1

    record = _planned_record()
    template = _numerical_replay()
    primary = next(row for row in template.estimates or () if row.target_id == record.target.binding.estimand_id)
    replay = template.model_copy(
        update={
            "target_ids": record.reported_target_ids,
            "estimates": (
                NumericalEstimate(
                    target_id="prespecified_primary_analysis",
                    estimate=primary.estimate,
                    lower=primary.lower,
                    upper=primary.upper,
                ),
            ),
        }
    )
    evidence = replay.evidence_artifacts
    bound = bind_trialeval_numerical_input(evaluator_record=record, replay=replay, expected_evidence=evidence)
    assert bound.attempt_status == "missing_required_output"
    outcome = grade_trialeval_outcome_v1(bound)
    assert outcome.reproducibility.status != "satisfied"
    plan_input = bind_trialeval_numerical_input(
        evaluator_record=record, replay=replay, expected_evidence=evidence, target_id="prespecified_primary_analysis"
    )
    plan = grade_trialeval_plan_execution_v1(record, plan_input)
    score = TrialEvalItemScoresV1(
        scorer=current_scorer(),
        outcome=outcome,
        target=record.target.model_dump(),
        oracle_worlds=record.oracle_worlds,
        oracle_monte_carlo_standard_error=record.oracle_monte_carlo_standard_error,
        maximum_oracle_monte_carlo_standard_error=record.maximum_oracle_monte_carlo_standard_error,
        item_id=record.task_id,
        task_id=record.task_id,
        release_id=record.release_id,
        trial_id=record.trial_id,
        base_world_id=record.base_world_id,
        cell_ref=FIXTURE_CELL_KEY,
        effect_state="favourable",
        context_tier=record.context,
        effect_measure=record.target.binding.effect_measure,
        result_unit=record.target.binding.result_unit,
        model="test",
        turns_used=1,
        agent_status="success",
        replay=replay,
        decision=None,
        plan_execution=plan,
    )
    assert score.signal_loss is None
    assert score.plan_execution is not None


def test_reported_but_ineligible_primary_answer_is_graded_not_raised() -> None:
    """A primary row the binding rejects (for example a nonpositive rate ratio) is graded as missing.

    The provider-free end-to-end campaign found the offline control's zero rate ratio in a plan-given task made the
    score wrapper raise 'replay and reproducibility component disagree' and stop grading.
    """
    from trialagentbench.contracts.scoring.trialeval_scores import TrialEvalItemScoresV1
    from trialagentbench.grading.outcome import TrialEvalOutcomeGradingInputV1

    record = _evaluator_record()
    replay = _numerical_replay()
    assert any(row.target_id == record.target.binding.estimand_id for row in replay.estimates or ())
    ineligible = TrialEvalOutcomeGradingInputV1(
        context=record.context, target=record.target.model_dump(), attempt_status="missing_required_output"
    )
    outcome = grade_trialeval_outcome_v1(ineligible)
    assert outcome.completion.status != "satisfied"
    score = TrialEvalItemScoresV1(
        scorer=current_scorer(),
        outcome=outcome,
        target=record.target.model_dump(),
        oracle_worlds=record.oracle_worlds,
        oracle_monte_carlo_standard_error=record.oracle_monte_carlo_standard_error,
        maximum_oracle_monte_carlo_standard_error=record.maximum_oracle_monte_carlo_standard_error,
        item_id=record.task_id,
        task_id=record.task_id,
        release_id=record.release_id,
        trial_id=record.trial_id,
        base_world_id=record.base_world_id,
        cell_ref=FIXTURE_CELL_KEY,
        effect_state="favourable",
        context_tier=record.context,
        effect_measure=record.target.binding.effect_measure,
        result_unit=record.target.binding.result_unit,
        model="test",
        turns_used=1,
        agent_status="success",
        replay=replay,
        decision=None,
    )
    assert score.signal_loss is None


@pytest.mark.parametrize(
    "measure,unit",
    [
        ("mean_difference", "points"),
        ("risk_difference", "proportion"),
        ("rate_ratio", "ratio"),
        ("restricted_mean_survival_time_difference", "days"),
    ],
)
@pytest.mark.parametrize("point_scored", [True, False])
@pytest.mark.parametrize("scale", [0.1, 0.2, 2.0])
def test_final_scoring_grid_needs_only_the_portable_grading_key(tmp_path, measure, unit, point_scored, scale):
    """The public verifier and harness agree without any construction/evaluation tree."""
    import math
    from itertools import product

    from trialagentbench_verifier.trialeval import (
        TrialEvalScoreInputV1 as IndependentInput,
    )
    from trialagentbench_verifier.trialeval import (
        verify_trialeval_outcome_v1,
    )

    from trialagentbench.grading.outcome import TrialEvalConfidenceIntervalV1, TrialEvalPointEstimateV1

    record = _evaluator_record()
    binding = TrialEvalQuestionBindingV1.model_validate(
        record.target.binding.model_dump()
        | {"effect_measure": measure, "result_unit": unit, "null_value": float(measure == "rate_ratio")}
    )
    record = TrialEvalGradingKeyRecordV1.model_validate(
        record.model_dump()
        | {
            "target": record.target.model_dump()
            | {
                "binding": binding.model_dump(),
                "point_scored": point_scored,
                "value": float(measure == "rate_ratio"),
                "reference_sampling_standard_deviation": scale,
            }
        }
    )
    write_trialeval_grading_key(tmp_path, (record,))
    loaded = load_trialeval_grading_key(tmp_path).for_task(record.task_id)
    assert not (tmp_path / "evaluation").exists()
    for point, limits, eligibility in product(
        (-0.2, 0, 0.2),
        ((-0.1, 0.1), (-0.4, 0.4), (-1, 1), (-1, 0.2)),
        ("valid", "no_evidence", "no_replay"),
    ):

        def native(value):
            return math.exp(value) if measure == "rate_ratio" else value

        result = TrialEvalPointEstimateV1(
            value=native(point),
            effect_measure=measure,
            unit=unit,
            interval=TrialEvalConfidenceIntervalV1(
                lower=native(point + limits[0]), upper=native(point + limits[1]), confidence_level=0.95
            ),
            conclusion="no_clear_difference",
        )
        score_input = TrialEvalScoreInputV1(
            task_id=record.task_id,
            attempt_status="complete",
            submitted_result=result.model_dump(mode="json"),
            evidence_source_artifacts=() if eligibility == "no_evidence" else record.participant_artifact_paths,
            reproducible=eligibility != "no_replay",
        )
        from trialagentbench.grading.outcome import TrialEvalOutcomeGradingInputV1

        actual = grade_trialeval_outcome_v1(
            TrialEvalOutcomeGradingInputV1(
                context=loaded.context,
                target=loaded.target.model_dump(),
                attempt_status="complete",
                submitted_result=result,
                evidence_eligible=eligibility != "no_evidence",
                reproducible=eligibility != "no_replay",
            )
        )
        verified = verify_trialeval_outcome_v1(
            release_root=tmp_path,
            score_input=IndependentInput.model_validate_json(score_input.model_dump_json()),
            authenticated_execution_reproduced=None if eligibility == "no_replay" else True,
        )
        if eligibility == "valid":
            assert actual.signal_loss is not None
            assert actual.signal_loss == verified.signal_loss
            if measure != "rate_ratio" and point == 0 and limits == (-1, 0.2) and scale == 0.1:
                assert actual.signal_loss == 6.0
        else:
            assert actual.signal_loss is verified.signal_loss is None


def test_bundled_demo_baselines_have_exact_independent_loss(tmp_path: Path) -> None:
    """Regrade the public demo's baseline reports from its portable key."""
    from zipfile import ZipFile

    from trialagentbench_verifier.trialeval import (
        TrialEvalScoreInputV1 as IndependentInput,
    )
    from trialagentbench_verifier.trialeval import verify_trialeval_outcome_v1

    from trialagentbench.grading.outcome import (
        TrialEvalConfidenceIntervalV1,
        TrialEvalOutcomeGradingInputV1,
        TrialEvalPointEstimateV1,
    )

    archive_path = Path(__file__).resolve().parents[2] / "examples/clinical/demo-trialeval-evaluator.zip"
    with ZipFile(archive_path) as archive:
        for name in ("release_manifest.json", "grading_key/manifest.json", "grading_key/records.jsonl"):
            archive.extract(name, tmp_path)
    key = load_trialeval_grading_key(tmp_path)
    assert key.records
    for record in key.records:
        target = record.target
        truth, scale = target.value, target.reference_sampling_standard_deviation
        result = TrialEvalPointEstimateV1(
            value=truth,
            effect_measure=target.binding.effect_measure,
            unit=target.binding.result_unit,
            interval=TrialEvalConfidenceIntervalV1(
                lower=truth - 1.96 * scale,
                upper=truth + 1.96 * scale,
                confidence_level=target.confidence_level,
            ),
            conclusion="no_clear_difference",
        )
        actual = grade_trialeval_outcome_v1(
            TrialEvalOutcomeGradingInputV1(
                context=record.context,
                target=target.model_dump(),
                attempt_status="complete",
                submitted_result=result,
                evidence_eligible=True,
                reproducible=True,
            )
        )
        verified = verify_trialeval_outcome_v1(
            release_root=tmp_path,
            score_input=IndependentInput(
                task_id=record.task_id,
                attempt_status="complete",
                submitted_result=result.model_dump(mode="json"),
                evidence_source_artifacts=record.participant_artifact_paths,
                reproducible=True,
            ),
            authenticated_execution_reproduced=True,
        )
        assert actual.signal_loss is not None
        assert actual.signal_loss == verified.signal_loss


@pytest.mark.parametrize("field", ["estimate", "lower", "upper"])
@pytest.mark.parametrize("nonpositive", [0.0, -1.0])
def test_common_csv_refuses_nonpositive_ratio_before_grading(field, nonpositive):
    """An invalid ratio remains a failed required answer, even after successful replay."""
    from trialagentbench_verifier.numerical import NumericalEstimate

    record = _evaluator_record()
    binding = TrialEvalQuestionBindingV1.model_validate(
        record.target.binding.model_dump()
        | {"effect_measure": "rate_ratio", "result_unit": "ratio", "null_value": 1.0}
    )
    record = TrialEvalGradingKeyRecordV1.model_validate(
        record.model_dump() | {"target": record.target.model_dump() | {"binding": binding.model_dump()}}
    )
    numbers = {"estimate": 1.0, "lower": 0.5, "upper": 1.5}
    numbers[field] = nonpositive
    if field == "upper":
        numbers["lower"] = nonpositive - 1
    replay = _numerical_replay().model_copy(
        update={"estimates": (NumericalEstimate(target_id="estimand", **numbers),)}
    )
    bound = bind_trialeval_numerical_input(
        evaluator_record=record, replay=replay, expected_evidence=replay.evidence_artifacts
    )
    grade = grade_trialeval_outcome_v1(bound)
    assert bound.attempt_status == "missing_required_output"
    assert grade.signal_loss is None
    assert grade.completion.failure_origin == "missing_required_output"
