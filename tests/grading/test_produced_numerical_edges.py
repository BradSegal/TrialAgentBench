"""Numerical grading of a produced TrialDev clinical analysis."""

import json
from decimal import Decimal, localcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from trialagentbench_verifier.authored_scoring import compute_numerical_recovery
from trialagentbench_verifier.numerical import NumericalEstimate, NumericalReplay, NumericalReportingContext

from trialagentbench.contracts.trialdev.grading_key import TrialDevEvidenceTargetV1
from trialagentbench.contracts.trialdev.pools import TrialDevPoolContrast
from trialagentbench.io import sha256_file, write_json_model
from trialagentbench.ports import CodeExecutionLimitsV1
from trialagentbench.trialdev import run, target_grading, targets
from trialagentbench.trialdev.materialization import TrialDevReadoutCommitment
from trialagentbench.trialdev.release import TrialDevAnalysisConfiguration, TrialDevInitialEvidence
from trialagentbench.trialdev.target_grading import grade_trialdev_observational_numbers
from trialagentbench.trialdev.value import TrialDevSponsorPreferences

_CASE = json.loads((Path(__file__).parents[1] / "fixtures/produced_clinical_score.json").read_text())


@pytest.fixture(autouse=True)
def forbid_float_grading(monkeypatch):
    """Produced-artifact grading must never execute a frozen float grader."""

    def float_grading_called(*args, **kwargs):
        raise AssertionError("runtime grading reached frozen float implementation")

    for name in ("_numerical_recovery", "grade_trialdev_observational_numbers", "grade_trialdev_readout_numbers"):
        monkeypatch.setattr(targets, name, float_grading_called)


@pytest.mark.parametrize("bounds", [(-1e308, 1e308), (1e308, 1.1e308), (-1.1e308, -1e308)])
def test_finite_clinical_interval_keeps_representable_loss(bounds, tmp_path, monkeypatch):
    """Scale the mathematical loss before deciding whether it overflows."""
    source = _CASE["score"]
    estimate = NumericalEstimate.model_validate({**_CASE["estimate"], "lower": bounds[0], "upper": bounds[1]})
    replay = NumericalReplay.model_validate(_CASE["replay"]).model_copy(
        update={"target_ids": (estimate.target_id,), "estimates": (estimate,)}
    )
    (actual,) = grade_trialdev_observational_numbers(
        targets=(TrialDevEvidenceTargetV1.model_validate(_CASE["target"]),),
        replay=replay,
        preferences=TrialDevSponsorPreferences.model_validate(_CASE["preferences"]),
    )
    independent = compute_numerical_recovery(
        target_id=source["target_id"],
        reference=source["reference_value"],
        scale=source["normalization_scale"],
        confidence=0.95,
        estimate=estimate,
        reproducible=True,
    )
    assert actual == independent
    context = NumericalReportingContext(
        checkpoint_id=replay.checkpoint_id, target_ids=replay.target_ids, evidence_artifacts=replay.evidence_artifacts
    )
    capture = tmp_path / "analysis_sources/submission"
    write_json_model(capture / "reporting_context.json", context)
    monkeypatch.setattr(run, "replay_numerical_submission", lambda **kwargs: replay)
    runtime = run.evaluate_trialdev_analysis(
        workspace=tmp_path / "workspace",
        configuration=TrialDevAnalysisConfiguration(
            preferences=TrialDevSponsorPreferences.model_validate(_CASE["preferences"]),
            initial_evidence=TrialDevInitialEvidence(
                context=context, targets=(TrialDevEvidenceTargetV1.model_validate(_CASE["target"]),)
            ),
        ),
        report_relative_path="analysis_sources/submission/report.md",
        execution_status="completed",
        error=None,
        executor_limits=CodeExecutionLimitsV1(),
    )
    assert runtime.scores == (independent,)
    with localcontext() as context:
        context.prec = 80
        lo, hi, truth, scale, confidence = map(
            Decimal.from_float,
            (estimate.lower, estimate.upper, source["reference_value"], source["normalization_scale"], 0.95),
        )
        expected = float((hi - lo + 2 * max(lo - truth, truth - hi, Decimal(0)) / (1 - confidence)) / scale)
    assert actual.normalized_interval_loss == (expected if expected < float("inf") else None)


def test_acquired_runtime_grading_bypasses_frozen_float_implementation(tmp_path, monkeypatch):
    """Grade an actual acquired confirmation receipt through the runtime cutoff evaluator."""
    case = json.loads((Path(__file__).parents[1] / "fixtures/produced_confirmation_readout.json").read_text())
    commitment = TrialDevReadoutCommitment.model_validate(case["commitment"])
    preferences = TrialDevSponsorPreferences.model_validate(case["preferences"])
    workspace, pool = tmp_path / "workspace", tmp_path / "pool"
    checkpoint = "evidence/" + commitment.readout.study_id + "-look-1"
    study = workspace / checkpoint / "study.json"
    study.parent.mkdir(parents=True)
    study.write_text(case["study_manifest_json"])
    pool.mkdir()
    (pool / "manifest.json").write_text(case["pool_manifest_json"])
    write_json_model(tmp_path / "commitments" / (study.parent.name + ".json"), commitment)
    contrast = next(iter(case["contrasts"].values()))
    monkeypatch.setattr(
        target_grading, "trialdev_pool_contrast", lambda *args, **kwargs: TrialDevPoolContrast.model_validate(contrast)
    )
    declared = commitment.readout.targets
    estimates = tuple(
        NumericalEstimate(
            target_id=target.target_id,
            estimate=(
                contrast["outcome_mean_difference"]
                if target.endpoint == "clinical_outcome"
                else contrast["safety_risk_difference"]
            ),
            lower=-1e308,
            upper=1e308,
        )
        for target in declared
    )
    context = NumericalReportingContext(
        checkpoint_id=checkpoint,
        target_ids=tuple(target.target_id for target in declared),
        evidence_artifacts=(
            {
                "relative_path": checkpoint + "/study.json",
                "sha256": sha256_file(study),
                "byte_length": study.stat().st_size,
            },
        ),
    )
    replay = NumericalReplay.model_validate(_CASE["replay"]).model_copy(
        update={
            "checkpoint_id": checkpoint,
            "target_ids": context.target_ids,
            "evidence_artifacts": context.evidence_artifacts,
            "estimates": estimates,
        }
    )
    capture = tmp_path / "analysis_sources/cutoffs/submission"
    write_json_model(capture / "reporting_context.json", context)
    monkeypatch.setattr(run, "_replay_accepted_capture", lambda **kwargs: replay)
    _, scores = run.evaluate_trialdev_cutoff(
        workspace=workspace,
        captured=capture,
        programme=SimpleNamespace(
            pool_root=pool, evidence_root=workspace / "evidence", commitment_root=tmp_path / "commitments"
        ),
        preferences=preferences,
        executor_limits=CodeExecutionLimitsV1(),
    )
    for target, estimate, score in zip(declared, estimates, scores, strict=True):
        clinical = target.endpoint == "clinical_outcome"
        expected = compute_numerical_recovery(
            target_id=target.target_id,
            estimate=estimate,
            reference=contrast["outcome_mean_difference"] if clinical else contrast["safety_risk_difference"],
            scale=preferences.benefit_unit if clinical else 0.01,
            confidence=target.confidence_level,
            reproducible=True,
        )
        assert score == expected
        if clinical:
            assert score.normalized_interval_loss == 1e308
