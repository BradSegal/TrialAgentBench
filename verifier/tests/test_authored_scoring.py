"""Hand-calculated fixed-population and recovery counterexamples."""

import json
from pathlib import Path

import pandas as pd
import pytest

from trialagentbench_verifier.authored_results import TrialDevNumericalRecovery
from trialagentbench_verifier.authored_scoring import _contrast, _ReferencePool, verify_numerical_recovery
from trialagentbench_verifier.io import sha256_file
from trialagentbench_verifier.numerical import NumericalEstimate


def _pool(root: Path, recurrent: bool = False, *, multiple_safety: bool = False) -> _ReferencePool:
    pd.DataFrame({"participant_id": [1, 2, 3]}).to_parquet(root / "baseline.parquet", index=False)
    for number, active in enumerate((False, True)):
        directory = root / f"partitions/p{number:03}"
        directory.mkdir(parents=True)
        if recurrent:
            events = [(1, 5.0, 1), (2, 20.0, 1), (2, 28.0, 1)] if not active else []
            frame = pd.DataFrame(
                [*events, *((patient, 28.0, 0) for patient in (1, 2, 3))],
                columns=["participant_id", "event_time", "event"],
            )
            frame["assigned_arm"] = "active" if active else "control"
            frame.to_parquet(directory / "recurrent_truth.parquet", index=False)
        else:
            frame = pd.DataFrame(
                {
                    "participant_id": [1, 2, 3] * 2,
                    "occurrence_time": [14] * 3 + [28] * 3,
                    "outcome_id": ["clinical_outcome"] * 6,
                    "complete_recorded_outcome": [3, 4, 5, 2, 3, 4] if active else [4, 5, 6, 6, 7, 8],
                }
            )
            frame.to_parquet(directory / "complete_measurements.parquet", index=False)
        safety = pd.DataFrame(
            {
                "participant_id": [1, 2, 3] * 2,
                "event_id": ["safety_event"] * 6,
                "_scheduled_contact_time": [14] * 3 + [28] * 3,
                "occurrence_time": [14, 14, 14, 28, 20, 28] if active else [5, 14, 14, 28, 28, 28],
                "event_at_contact": (
                    [False, False, False, False, True, False] if active else [True, False, False, False, False, False]
                ),
            }
        )
        if multiple_safety:
            secondary = safety.copy()
            secondary["event_id"] = "second_event"
            # The control patient already has an event; a second type must not
            # count twice. The active arm gains one additional affected patient.
            secondary["event_at_contact"] = [True, False, False, False, False, False]
            safety = pd.concat([safety, secondary], ignore_index=True)
        safety.to_parquet(directory / "safety_events.parquet", index=False)
    manifest = {
        "schema_id": "trialdev.frozen-pool/v7",
        "outcome_object": "recurrent_event" if recurrent else "clinical_outcome",
        "participant_count": 3,
        "collection_days": [14, 28],
        "baseline_fields": [],
        "partitions": [
            {"asset_id": "control", "dose": None, "directory": "partitions/p000"},
            {"asset_id": "candidate", "dose": "standard", "directory": "partitions/p001"},
        ],
        "targets": [
            {
                "asset_id": "candidate",
                "dose": "standard",
                "outcome_unit": "events per participant-year" if recurrent else "severity",
                "outcome_mean_difference": -4,
                "safety_risk_difference": 0,
            }
        ],
        "artifacts": [
            {
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "byte_length": path.stat().st_size,
            }
            for path in sorted(root.rglob("*.parquet"))
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return _ReferencePool.model_validate(manifest)


def test_study_window_changes_benefit_and_cumulative_safety(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    early = _contrast(tmp_path, pool, "candidate", "standard", 14)
    final = _contrast(tmp_path, pool, "candidate", "standard", 28)
    assert early.outcome_mean_difference == -1
    assert early.safety_risk_difference == pytest.approx(-1 / 3)
    assert final.outcome_mean_difference == -4
    assert final.safety_risk_difference == 0
    with pytest.raises(ValueError, match="collection grid"):
        _contrast(tmp_path, pool, "candidate", "standard", 21)


def test_recurrent_reference_includes_event_free_patients(tmp_path: Path) -> None:
    pool = _pool(tmp_path, recurrent=True)
    early = _contrast(tmp_path, pool, "candidate", "standard", 14)
    final = _contrast(tmp_path, pool, "candidate", "standard", 28)
    assert early.outcome_mean_difference == pytest.approx(-365 / 42)
    assert final.outcome_mean_difference == pytest.approx(-365 / 28)
    path = tmp_path / "partitions/p001/recurrent_truth.parquet"
    pd.read_parquet(path).iloc[:2].to_parquet(path, index=False)
    # Even rebinding file identity cannot turn two observed patients into the full population.
    rebound = pool.model_copy(
        update={
            "artifacts": tuple(
                (
                    row.model_copy(update={"sha256": sha256_file(path), "byte_length": path.stat().st_size})
                    if row.relative_path == "partitions/p001/recurrent_truth.parquet"
                    else row
                )
                for row in pool.artifacts
            )
        }
    )
    with pytest.raises(ValueError, match="event-free population"):
        _contrast(tmp_path, rebound, "candidate", "standard", 28)


def test_independent_loss_retains_missingness_and_rejects_substituted_credit() -> None:
    estimate = NumericalEstimate(target_id="effect", estimate=3.0, lower=1.0, upper=4.0)
    score = TrialDevNumericalRecovery(
        target_id="effect",
        status="reported",
        reference_value=2,
        normalization_scale=1,
        normalized_absolute_error=1,
        normalized_interval_loss=3,
    )
    assert verify_numerical_recovery(
        score=score, estimate=estimate, reference=2, scale=1, confidence=0.95, reproducible=True
    ) == pytest.approx(2)
    with pytest.raises(ValueError, match="independent recovery"):
        verify_numerical_recovery(
            score=score.model_copy(update={"normalized_absolute_error": 999}),
            estimate=estimate,
            reference=2,
            scale=1,
            confidence=0.95,
            reproducible=True,
        )
    with pytest.raises(ValueError, match="reproduced target"):
        verify_numerical_recovery(
            score=score, estimate=estimate, reference=2, scale=1, confidence=0.95, reproducible=False
        )


def test_independent_loss_uses_declared_confidence_and_zero_credit_for_absence() -> None:
    estimate = NumericalEstimate(target_id="effect", estimate=1.0, lower=1.0, upper=2.0)
    score = TrialDevNumericalRecovery(
        target_id="effect",
        status="reported",
        reference_value=0,
        normalization_scale=1,
        normalized_absolute_error=1,
        normalized_interval_loss=21,
    )
    assert verify_numerical_recovery(
        score=score, estimate=estimate, reference=0, scale=1, confidence=0.9, reproducible=True
    ) == pytest.approx(11)
    with pytest.raises(ValueError, match="independent recovery"):
        verify_numerical_recovery(
            score=score, estimate=estimate, reference=0, scale=1, confidence=0.95, reproducible=True
        )
    for status, reproduced in (("missing", True), ("invalid_replay", False)):
        missing = TrialDevNumericalRecovery(
            target_id="effect",
            status=status,
            reference_value=0,
            normalization_scale=1,
            normalized_absolute_error=None,
            normalized_interval_loss=None,
        )
        assert (
            verify_numerical_recovery(
                score=missing, estimate=None, reference=0, scale=1, confidence=0.95, reproducible=reproduced
            )
            is None
        )


def test_scalar_initial_recovery_uses_declared_reference_and_requires_execution(tmp_path):
    import json

    from trialagentbench_verifier.contracts import NormalizedSubmissionV1
    from trialagentbench_verifier.release import _score_trialdev

    task_id = "td-0123456789ab"
    target_id = "asset:clinical_outcome"
    config = tmp_path / "evaluation/tasks" / task_id / "analysis.json"
    config.parent.mkdir(parents=True)
    config.write_text(
        json.dumps(
            {
                "schema_id": "trialdev.numerical-analysis/v1",
                "preferences": {"outcome_unit": "points", "benefit_unit": 2},
                "initial_evidence": {
                    "context": {"checkpoint_id": "initial", "target_ids": [target_id], "evidence_artifacts": []},
                    "targets": [
                        {
                            "asset_id": "asset",
                            "endpoint": "clinical_outcome",
                            "unit": "points",
                            "decision_scale": 2,
                            "comparison_sample_size": 100,
                            "point_identified": True,
                            "oracle_value": 1,
                            "estimand": "standardized_observed_association",
                        }
                    ],
                },
            }
        )
    )
    task = tmp_path / "participant/tasks" / task_id / "task.json"
    task.parent.mkdir(parents=True)
    task.write_text(json.dumps({"initial_target_ids": [target_id]}))
    submission = NormalizedSubmissionV1(
        benchmark_stream="trialdev",
        task_id=task_id,
        submission_kind="evidence",
        payload={
            "task_id": task_id,
            "estimates": [{"target_id": target_id, "estimate": 1.0, "lower": 0.0, "upper": 2.0}],
        },
    )
    for authenticated in (True, False):
        check = _score_trialdev(
            tmp_path,
            tmp_path / "submission.json",
            submission,
            execution_authenticated=authenticated,
        )[0]
        assert check.status == ("pass" if authenticated else "fail")
        assert check.observed["mean_loss"] == (pytest.approx(0.5) if authenticated else None)
    missing = submission.model_copy(update={"payload": {"task_id": task_id, "estimates": []}})
    check = _score_trialdev(
        tmp_path,
        tmp_path / "submission.json",
        missing,
        execution_authenticated=True,
    )[0]
    assert check.status == "pass"
    assert check.observed["mean_loss"] is None
    assert [row["target_id"] for row in check.observed["scores"]] == [target_id]
    original_task = task.read_bytes()
    original_config = config.read_bytes()
    for defect in ("public_target", "reference_target", "causal_target", "wrong_unit"):
        changed = json.loads(original_config)
        if defect == "public_target":
            task.write_text(json.dumps({"initial_target_ids": ["other:clinical_outcome"]}))
        elif defect == "reference_target":
            changed["initial_evidence"]["context"]["target_ids"] = ["other:clinical_outcome"]
        elif defect == "causal_target":
            changed["initial_evidence"]["targets"][0]["estimand"] = "causal_effect"
        else:
            changed["initial_evidence"]["targets"][0]["unit"] = "events per participant-year"
        config.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="targets differ|observed association|standardized_observed_association"):
            _score_trialdev(
                tmp_path,
                tmp_path / "submission.json",
                submission,
                execution_authenticated=True,
            )
        task.write_bytes(original_task)
        config.write_bytes(original_config)


def test_independent_safety_union_counts_participants_and_requires_component_census(tmp_path):
    pool = _pool(tmp_path, multiple_safety=True)
    contrast = _contrast(tmp_path, pool, "candidate", "standard", 28)
    assert contrast.safety_risk_difference == pytest.approx(1 / 3)
    path = tmp_path / "partitions/p001/safety_events.parquet"
    records = pd.read_parquet(path)
    records = records.loc[
        ~(records.event_id.eq("second_event") & records.participant_id.eq(3) & records._scheduled_contact_time.eq(28))
    ]
    records.to_parquet(path, index=False)
    pool = pool.model_copy(
        update={
            "artifacts": tuple(
                (
                    entry.model_copy(update={"sha256": sha256_file(path), "byte_length": path.stat().st_size})
                    if entry.relative_path == "partitions/p001/safety_events.parquet"
                    else entry
                )
                for entry in pool.artifacts
            )
        }
    )
    with pytest.raises(ValueError, match="patient/event rows"):
        _contrast(tmp_path, pool, "candidate", "standard", 28)
