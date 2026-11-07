"""Public released-byte verification and score-replay tests."""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from trialagentbench_verifier import ExecutionProofV1, NormalizedSubmissionV1, score_submission, verify_release
from trialagentbench_verifier import io as verifier_io
from trialagentbench_verifier import release as release_module
from trialagentbench_verifier import release_set as release_set_module
from trialagentbench_verifier.cli import _parser, main
from trialagentbench_verifier.io import (
    VerificationInputError,
    canonical_json_bytes,
    canonical_sha256,
    materialized_release,
    read_json_object,
)
from trialagentbench_verifier.methods import (
    _cluster_design_effect,
    _kaplan_meier_summary,
    _linear_contrast,
    _obrien_fleming_boundary,
    _weighted_mean_difference,
)
from trialagentbench_verifier.numerical import read_numerical_table
from trialagentbench_verifier.practice import verify_practice_suite
from trialagentbench_verifier.proof import numerical_score_projection_sha256, submission_score_projection_sha256
from trialagentbench_verifier.release import (
    EvaluationReleaseError,
    ScoreSubmissionV1,
    score_submissions,
)
from trialagentbench_verifier.trialeval import (
    TrialEvalGradingKeyRecordV1,
    TrialEvalScoreInputV1,
    verify_trialeval_outcome_v1,
)
from trialagentbench_verifier.units import compatible_units


@pytest.mark.parametrize(
    ("payload", "digest"),
    (
        (None, "74234e98afe7498fb5daf1f36ac2d78acc339464f950703b8c019892f982b90b"),
        ({"b": 2, "a": 1}, "43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777"),
        (
            {"label": "München", "items": [True, None, -2, 1.25]},
            "7a3fe5f88f89e0d3660c1c7efb08199d3c6b2243fca07bba1011ad67dc077597",
        ),
    ),
)
def test_release_serialization_conformance(payload: object, digest: str) -> None:
    assert canonical_sha256(payload) == digest  # type: ignore[arg-type]


def test_release_serialization_rejects_non_json_numbers() -> None:
    with pytest.raises(ValueError, match="JSON compliant"):
        canonical_sha256(float("nan"))


@pytest.mark.parametrize(
    ("reported", "reference"),
    (
        ("infection risk difference", "risk difference"),
        ("serious-adverse-event probability difference", "risk difference"),
        ("difference_in_proportions", "risk difference"),
        (
            "risk difference by Day 168 (proportion; candidate minus standard inhaled therapy)",
            "risk difference",
        ),
        ("clinical total score points", "clinical score points"),
        ("DAS28 difference (points)", "DAS28 points"),
    ),
)
def test_trialdev_unit_labels_accept_equivalent_descriptions(reported: str, reference: str) -> None:
    """Accept qualified labels that preserve the reference numerical scale."""

    assert compatible_units(reported, reference)


@pytest.mark.parametrize(
    ("reported", "reference"),
    (
        ("infection percentage points", "risk difference"),
        ("hazard ratio", "risk difference"),
        ("log risk difference", "risk difference"),
        ("odds risk difference", "risk difference"),
        ("risk difference", "clinical score points"),
        ("log DAS28 points", "DAS28 points"),
        ("ratio DAS28 points", "DAS28 points"),
        ("percent DAS28 points", "DAS28 points"),
    ),
)
def test_trialdev_unit_labels_reject_scale_changes(reported: str, reference: str) -> None:
    """Reject labels that require a numerical conversion or change the estimand scale."""

    assert not compatible_units(reported, reference)


def test_trialeval_score_input_allows_point_without_optional_interval() -> None:
    observed = TrialEvalScoreInputV1.model_validate(
        {
            "schema_id": "trialagentbench.trialeval_score_input/v1",
            "task_id": "task-1",
            "attempt_status": "incomplete",
            "submitted_result": {
                "kind": "point",
                "value": 2.0,
                "effect_measure": "mean_difference",
                "unit": "points",
                "interval": None,
                "conclusion": "no_clear_difference",
            },
        }
    )

    assert observed.submitted_result is not None
    assert observed.submitted_result.kind == "point"
    assert observed.submitted_result.interval is None


@pytest.mark.parametrize(
    "content",
    (
        '{"field": 1, "field": 2}',
        '{"field": NaN}',
        '{"field": Infinity}',
    ),
)
def test_json_reader_rejects_ambiguous_values(tmp_path: Path, content: str) -> None:
    """Reject duplicate fields and non-finite values at the public boundary."""

    path = tmp_path / "input.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(VerificationInputError):
        read_json_object(path)


def test_json_reader_rejects_oversized_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject JSON before reading content beyond the declared resource bound."""

    path = tmp_path / "input.json"
    path.write_text('{"field": 1}', encoding="utf-8")
    monkeypatch.setattr(verifier_io, "MAX_JSON_BYTES", path.stat().st_size - 1)
    with pytest.raises(VerificationInputError, match="size limit"):
        read_json_object(path)


@pytest.mark.parametrize(
    ("limit_name", "limit", "message"),
    (
        ("MAX_RELEASE_ARCHIVE_BYTES", 1, "file size limit"),
        ("MAX_RELEASE_ARCHIVE_MEMBERS", 1, "too many members"),
        ("MAX_RELEASE_UNCOMPRESSED_BYTES", 1, "uncompressed size limit"),
    ),
)
def test_archive_reader_enforces_resource_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    limit: int,
    message: str,
) -> None:
    """Reject archives that exceed any bounded extraction resource."""

    archive_path = tmp_path / "release.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("first.txt", "first")
        archive.writestr("second.txt", "second")
    monkeypatch.setattr(verifier_io, limit_name, limit)
    with pytest.raises(VerificationInputError, match=message), materialized_release(archive_path):
        pass


@pytest.mark.slow
def test_archive_readers_accept_a_large_custody_inventory(tmp_path: Path) -> None:
    """Inspect and extract the file counts produced by full native campaigns."""

    archive_path = tmp_path / "release.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("manifest.json", "{}")
        for index in range(100_000):
            archive.writestr(f"snapshots/{index}.txt", "")
    with zipfile.ZipFile(archive_path) as archive:
        assert len(release_set_module.validate_zip_members(archive)) == 100_001
    with materialized_release(archive_path) as extracted:
        assert (extracted / "snapshots/99999.txt").is_file()


def test_archive_inspector_retains_its_member_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A larger supported inventory still rejects an archive beyond its bound."""

    archive_path = tmp_path / "release.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("first.txt", "")
        archive.writestr("second.txt", "")
    monkeypatch.setattr(release_set_module, "MAX_RELEASE_ARCHIVE_MEMBERS", 1)
    with zipfile.ZipFile(archive_path) as archive, pytest.raises(ValueError, match="member-count limit"):
        release_set_module.validate_zip_members(archive)


@pytest.mark.parametrize("member", ("../escape", "/absolute", "C:/drive", "dir\\escape"))
def test_archive_reader_rejects_nonportable_paths(tmp_path: Path, member: str) -> None:
    """Reject traversing, absolute, drive-qualified, and backslash paths."""

    archive_path = tmp_path / "release.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(member, "content")
    with pytest.raises(VerificationInputError, match="non-portable"), materialized_release(archive_path):
        pass


def _canonical(payload: object) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(body).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact(root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    return {
        "relative_path": relative,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "byte_length": path.stat().st_size,
    }


RECORDS = "grading_key/records.jsonl"
KEY_MANIFEST = "grading_key/manifest.json"
C5_INTEGRITY = {
    "affected_domain": "data/raw/event_history.parquet",
    "compound_key_fields": ["participant_id", "event_id"],
    "resolution_action": "remove_exact_duplicate_copy",
    "resolution_status": "repaired",
    "observed_issue_group_count": 1,
    "observed_affected_row_count": 2,
    "selected_affected_row_count": 1,
    "selected_observed_compound_keys": ['[["int64","1"],["int64","2"]]'],
    "nonmaterial_to_primary_analysis": False,
    "resolved_domain_content_sha256": "a" * 64,
}


def _reseal_key(root: Path) -> None:
    """Rewrite the key records canonically and reseal the key manifest to the current release bytes."""

    records_path = root / RECORDS
    lines = []
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    for payload in records:
        with contextlib.suppress(ValidationError):  # A deliberately invalid record must reach its detector.
            payload = TrialEvalGradingKeyRecordV1.model_validate(payload).model_dump(mode="json")
        lines.append(canonical_json_bytes(payload) + b"\n")
    records_path.write_bytes(b"".join(lines))
    manifest = json.loads((root / KEY_MANIFEST).read_text(encoding="utf-8"))
    for count, field in (
        ("task_count", "task_id"),
        ("world_count", "base_world_id"),
        ("cell_count", "cell_ref"),
        ("series_count", "series_ref"),
    ):
        manifest[count] = len({str(record.get(field)) for record in records})
    manifest["artifacts"] = [_artifact(root, str(row["relative_path"])) for row in manifest["artifacts"]]
    manifest["release_manifest_sha256"] = hashlib.sha256((root / "release_manifest.json").read_bytes()).hexdigest()
    manifest.pop("checksum", None)
    manifest["checksum"] = _canonical(manifest)
    _write_json(root / KEY_MANIFEST, manifest)


def _write_records(root: Path, *records: dict[str, object]) -> None:
    """Replace the key records and reseal the key."""

    (root / RECORDS).write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    _reseal_key(root)


def _reseal_evaluation(root: Path, relative: str) -> None:
    """Reseal public manifests so a semantic mutation reaches its detector."""

    role_path = root / "evaluation" / "manifest.json"
    if not role_path.is_file():
        _reseal_key(root)
        return
    role = json.loads(role_path.read_text(encoding="utf-8"))
    role["artifacts"] = [
        _artifact(root, relative) if row["relative_path"] == relative else row for row in role["artifacts"]
    ]
    _reseal_role(root, role="evaluation", payload=role)


def _reseal_role(root: Path, *, role: str, payload: dict[str, object] | None = None) -> None:
    """Reseal one role inventory, its outer release identity and the grading key bound to it."""

    role_path = root / role / "manifest.json"
    manifest = json.loads(role_path.read_text(encoding="utf-8")) if payload is None else payload
    manifest.pop("checksum", None)
    manifest["checksum"] = _canonical(manifest)
    _write_json(role_path, manifest)
    release_path = root / "release_manifest.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    release[f"{role}_manifest_sha256"] = manifest["checksum"]
    release.pop("checksum")
    release["checksum"] = _canonical(release)
    _write_json(release_path, release)
    if (root / KEY_MANIFEST).is_file():
        _reseal_key(root)


def _release(root: Path, *, operator: bool = True) -> Path:
    """Write a one-task release: the operator layout (both roles and the key) or the public key layout."""

    construct = "a" * 64
    release_id = f"release_{construct}"
    (root / "participant" / "task.txt").parent.mkdir(parents=True)
    (root / "participant" / "task.txt").write_text("task\n", encoding="utf-8")
    qualification = {"schema_id": "qualification/v1", "accepted": True}
    qualification["checksum"] = _canonical(qualification)
    _write_json(root / "evaluation" / "qualification.json", qualification)
    record = {
        "schema_id": "trialagentbench.trialeval_grading_key_record/v1",
        "release_id": release_id,
        "task_id": "task_1",
        "trial_id": "trial_1",
        "base_world_id": "world_11111111111111111111",
        "cell_ref": f"cell_{'1' * 20}",
        "series_ref": f"series_{'1' * 20}",
        "context": "C2",
        "effect_state": "favourable",
        "design_tier": "D1",
        "design_subtype": "Individually randomized parallel-group trial",
        "design_family": "individual_randomized",
        "participant_data_sha256": "b" * 64,
        "participant_artifact_paths": ["data/analysis.parquet"],
        "target": {
            "kind": "point",
            "binding": {
                "schema_id": "trialagentbench.trialeval_question_binding/v1",
                "analysis_population_id": "all_randomized",
                "estimand_id": "risk_difference",
                "intercurrent_event_strategy_ids": [],
                "assessment_horizon_days": 28.0,
                "treatment_id": "active",
                "comparator_id": "control",
                "endpoint_id": "response",
                "effect_measure": "risk_difference",
                "analysis_unit": "participant",
                "population_basis": "finite_population",
                "identification_status": "point_identified",
                "identification_assumption_ids": [],
                "result_unit": "risk_difference",
                "uncertainty_type": "central_confidence_interval",
                "confidence_level": 0.95,
                "sidedness": "two_sided_central",
                "null_value": 0.0,
                "favorable_direction": "higher",
            },
            "value": 0.1,
            "confidence_level": 0.95,
            "reference_sampling_standard_deviation": 0.05,
            "reference_sampling_basis_id": "finite_world_oracle_contrast",
        },
        "oracle_worlds": 1,
        "oracle_monte_carlo_standard_error": 0.0,
        "maximum_oracle_monte_carlo_standard_error": 1e-8,
    }
    records = root / RECORDS
    records.parent.mkdir(parents=True)
    records.write_bytes(
        canonical_json_bytes(TrialEvalGradingKeyRecordV1.model_validate(record).model_dump(mode="json")) + b"\n"
    )
    report_identities = []
    report_paths = []
    for gate_id in ("G-EXT-01", "G-TAB-01", "X-19"):
        relative = f"evaluation/verification/reports/{gate_id}.json"
        report = {
            "schema_id": "trialagentbench.qualification_gate_result/v1",
            "gate_id": gate_id,
            "candidate_digest": construct,
            "accepted": True,
            "scheduled_count": 1,
            "passed_count": 1,
            "failed_count": 0,
        }
        _write_json(root / relative, report)
        report_paths.append(relative)
        report_identities.append(
            {
                "gate_id": gate_id,
                "relative_path": relative,
                "sha256": hashlib.sha256((root / relative).read_bytes()).hexdigest(),
                "candidate_digest": construct,
                "accepted": True,
                "scheduled_count": 1,
                "passed_count": 1,
                "failed_count": 0,
            }
        )
    bundle = {
        "schema_id": "trialagentbench.qualification_report_bundle/v1",
        "candidate_digest": construct,
        "reports": report_identities,
        "comparisons": [
            {
                "comparison_id": "fixture:mean",
                "scope_id": "task_1",
                "quantity_id": "mean",
                "candidate_value": 1.0,
                "reference_value": 1.0,
                "absolute_difference": 0.0,
                "absolute_margin": 1e-12,
                "matched": True,
                "tolerance_id": "fixture-exact",
                "artifact_paths": [report_paths[0], "evaluation/qualification.json"],
            }
        ],
    }
    bundle_relative = "evaluation/verification/qualification_reports.json"
    _write_json(root / bundle_relative, bundle)
    retained = [_artifact(root, bundle_relative), *[_artifact(root, relative) for relative in report_paths]]
    role_artifacts = {
        "participant": [_artifact(root, "participant/task.txt")],
        "evaluation": [_artifact(root, "evaluation/qualification.json"), *retained],
    }
    role_checksums: dict[str, str] = {}
    for role in ("participant", "evaluation"):
        manifest = {
            "schema_id": "ariadne.benchmark_role_manifest/v1",
            "release_id": release_id,
            "role": role,
            "trial_count": 1,
            "task_count": 1,
            "artifacts": role_artifacts[role],
        }
        manifest["checksum"] = _canonical(manifest)
        role_checksums[role] = str(manifest["checksum"])
        _write_json(root / role / "manifest.json", manifest)
    manifest = {
        "schema_id": "ariadne.trialeval.release/v1",
        "data_provenance": "generated_under_declared_trial_mechanisms",
        "release_id": release_id,
        "construct_sha256": construct,
        "qualification_sha256": qualification["checksum"],
        "trial_count": 1,
        "task_count": 1,
        "participant_manifest_sha256": role_checksums["participant"],
        "evaluation_manifest_sha256": role_checksums["evaluation"],
    }
    manifest["checksum"] = _canonical(manifest)
    _write_json(root / "release_manifest.json", manifest)
    key = {
        "schema_id": "trialagentbench.trialeval_grading_key/v1",
        "release_id": release_id,
        "release_manifest_sha256": "0" * 64,
        "derivation_id": "fixture",
        "task_count": 1,
        "world_count": 1,
        "cell_count": 1,
        "series_count": 1,
        "artifacts": sorted([_artifact(root, RECORDS), *retained], key=lambda row: str(row["relative_path"])),
        "checksum": "0" * 64,
    }
    _write_json(root / KEY_MANIFEST, key)
    _reseal_key(root)
    if not operator:
        (root / "evaluation" / "manifest.json").unlink()
        (root / "evaluation" / "qualification.json").unlink()
    return root


def _submission(path: Path) -> Path:
    binding = {
        "schema_id": "trialagentbench.trialeval_question_binding/v1",
        "analysis_population_id": "all_randomized",
        "estimand_id": "risk_difference",
        "intercurrent_event_strategy_ids": [],
        "assessment_horizon_days": 28.0,
        "treatment_id": "active",
        "comparator_id": "control",
        "endpoint_id": "response",
        "effect_measure": "risk_difference",
        "analysis_unit": "participant",
        "population_basis": "finite_population",
        "identification_status": "point_identified",
        "identification_assumption_ids": [],
        "result_unit": "risk_difference",
        "uncertainty_type": "central_confidence_interval",
        "confidence_level": 0.95,
        "sidedness": "two_sided_central",
        "null_value": 0.0,
        "favorable_direction": "higher",
    }
    normalized = NormalizedSubmissionV1(
        benchmark_stream="trialeval",
        task_id="task_1",
        submission_kind="analysis",
        payload={
            "schema_id": "trialagentbench.trialeval_score_input/v1",
            "task_id": "task_1",
            "attempt_status": "complete",
            "submitted_binding": binding,
            "submitted_result": {
                "kind": "point",
                "value": 0.1,
                "effect_measure": "risk_difference",
                "unit": "risk_difference",
                "interval": {"lower": -0.1, "upper": 0.3, "confidence_level": 0.95},
                "conclusion": "no_clear_difference",
            },
            "evidence_source_artifacts": ["data/analysis.parquet"],
            "reproducible": True,
        },
    )
    _write_json(path, normalized.model_dump(mode="json"))
    return path


def _set_release(root: Path) -> Path:
    """Project the compact release fixture onto a partially identified target."""

    release = _release(root)
    record_path = release / RECORDS
    record = read_json_object(record_path)
    binding = record["target"]["binding"]
    binding.update(
        identification_status="partially_identified",
        identification_assumption_ids=["declared_partial_identification"],
        uncertainty_type="simultaneous_outer_confidence_region",
        sidedness="simultaneous_outer",
    )
    record["target"] = {
        "kind": "identified_set",
        "binding": binding,
        "set_definition_id": "declared_set",
        "lower": 0.1,
        "upper": 0.2,
        "outer_confidence_level": 0.95,
        "effect_support_lower": -1.0,
        "effect_support_upper": 1.0,
    }
    _write_records(release, record)
    _reseal_key(release)
    return release


def _set_submission(path: Path, *, lower: float, upper: float, region_lower: float, region_upper: float) -> Path:
    """Write a valid set result that makes an invalid favorable claim."""

    point = read_json_object(_submission(path.with_name("point.json")))
    binding = point["payload"]["submitted_binding"]
    binding.update(
        identification_status="partially_identified",
        identification_assumption_ids=["declared_partial_identification"],
        uncertainty_type="simultaneous_outer_confidence_region",
        sidedness="simultaneous_outer",
    )
    point["payload"]["submitted_result"] = {
        "kind": "identified_set",
        "set_definition_id": "declared_set",
        "lower": lower,
        "upper": upper,
        "effect_measure": "risk_difference",
        "unit": "risk_difference",
        "conclusion": "favors_treatment",
        "outer_confidence_region": {
            "lower": region_lower,
            "upper": region_upper,
            "confidence_level": 0.95,
            "region_type": "simultaneous_outer_confidence_region",
            "coverage_scope": "identified_set",
        },
    }
    _write_json(path, point)
    return path


def _execution_evidence(release: Path, submission_path: Path) -> dict[str, object]:
    """Create internally coherent fixture evidence for score-proof boundary tests."""

    payload = read_json_object(submission_path)
    if payload.get("schema_id") == "trialagentbench.trialeval_score_input/v1":
        score_input = payload
        normalized = NormalizedSubmissionV1(
            benchmark_stream="trialeval",
            task_id=str(payload["task_id"]),
            submission_kind="analysis",
            payload=payload,
        )
    else:
        normalized = NormalizedSubmissionV1.model_validate(payload)
        score_input = normalized.payload
    submitted = dict(score_input["submitted_result"])
    if submitted.get("kind") == "non_identification":
        conclusion = submitted.pop("conclusion")
        submitted["reason"] = "Required endpoint observations are unavailable."
    else:
        conclusion = submitted.pop("conclusion")
    submitted.pop("set_definition_id", None)
    if submitted.get("kind") == "point":
        submitted["sampling_interval"] = submitted.pop("interval")
    result_payload = {
        "schema_id": "trialagentbench.trialeval_program_result/v2",
        "result": submitted,
        "source_artifacts": list(score_input["evidence_source_artifacts"]),
        "conclusion": conclusion,
    }
    program = submission_path.with_suffix(".analysis.py")
    program.write_text("# independently replayed fixture\n", encoding="utf-8")
    result = submission_path.with_suffix(".analysis_result.json")
    _write_json(result, result_payload)
    manifest = read_json_object(release / "release_manifest.json")
    digest = hashlib.sha256((release / "release_manifest.json").read_bytes()).hexdigest()
    result_digest = canonical_sha256(result_payload)
    from trialagentbench_verifier.executor import analysis_project_artifacts

    proof = ExecutionProofV1(
        status="pass",
        input_kind="directory",
        input_sha256=digest,
        expected_input_sha256=digest,
        release_digest=digest,
        candidate_digest=str(manifest["construct_sha256"]),
        benchmark_stream="trialeval",
        task_id=normalized.task_id,
        language="python",
        program_artifacts=analysis_project_artifacts(program),
        program_sha256=hashlib.sha256(program.read_bytes()).hexdigest(),
        program_result_sha256=result_digest,
        score_projection_sha256=submission_score_projection_sha256(normalized),
        replayed_result_sha256s=(result_digest, result_digest),
        replay_status="reproduced",
        execution_status="success",
        comparison_classification="exact",
        attempt_count=2,
        elapsed_seconds=0.1,
        cpu_seconds=0.05,
        session_peak_memory_bytes=1024,
        session_peak_processes=1,
        executor_image_reference="trialagentbench-executor:test",
        executor_image_id="sha256:" + "d" * 64,
        expected_executor_image_id="sha256:" + "d" * 64,
        executor_environment_sha256="e" * 64,
        expected_executor_environment_sha256="e" * 64,
    )
    proof_path = submission_path.with_suffix(".execution_proof.json")
    _write_json(proof_path, proof.model_dump(mode="json"))
    return {
        "execution_proof": proof_path,
        "expected_execution_proof_sha256": hashlib.sha256(proof_path.read_bytes()).hexdigest(),
        "program": program,
        "program_result": result,
    }


def _authenticated_score(release: Path, submission_path: Path):
    """Score one fixture through an externally anchored successful replay proof."""

    return score_submission(release, submission_path, **_execution_evidence(release, submission_path))


def test_release_and_score_share_one_stream_selected_report(tmp_path: Path) -> None:
    """Verify and score from released identities rather than command labels."""

    release = _release(tmp_path / "release")
    verified = verify_release(release)
    qualification = [row for row in verified.checks if row.check_id.startswith("qualification.")]
    assert {row.check_id for row in qualification} == {
        "qualification.external_anchor",
        "qualification.table_integrity",
        "qualification.execution_replay",
        "qualification.portable_comparison",
    }
    assert all(not any(code in row.scope_id for code in ("G-EXT-01", "G-TAB-01", "X-19")) for row in qualification)
    submission = _submission(tmp_path / "submission.json")
    evidence = _execution_evidence(release, submission)
    scored = score_submission(release, submission, graded_item=submission, **evidence)
    assert verified.status == scored.status == "pass"
    assert verified.input_kind == scored.input_kind == "directory"
    assert verified.input_sha256 == scored.input_sha256
    assert verified.expected_input_sha256 is None
    assert verified.benchmark_stream == scored.benchmark_stream == "trialeval"
    assert scored.operation == "score"
    assert verified.scheduled_check_count == 6
    assert scored.passed_check_count == scored.scheduled_check_count == 6
    assert scored.score_binding is not None
    assert scored.score_binding.task_id == "task_1"
    assert scored.score_binding.graded_item_sha256 == hashlib.sha256(submission.read_bytes()).hexdigest()
    assert scored.score_binding.submission_artifact_sha256 == hashlib.sha256(submission.read_bytes()).hexdigest()
    assert scored.score_binding.execution_proof_sha256 == evidence["expected_execution_proof_sha256"]
    from trialagentbench_verifier.trialeval_result import TrialEvalProgramResultV2

    result = TrialEvalProgramResultV2.model_validate(read_json_object(evidence["program_result"]))
    assert scored.score_binding.normalized_program_result_sha256 == canonical_sha256(result.model_dump(mode="json"))
    assert scored.score_binding.replayed_result_sha256s == (
        scored.score_binding.program_result_sha256,
        scored.score_binding.program_result_sha256,
    )


@pytest.mark.parametrize("retained_version", ["1.0.0", "1.0.1"])
def test_an_execution_proof_from_earlier_verifiers_still_scores(tmp_path: Path, retained_version: str) -> None:
    """Retained replay proofs verify under the current package, which stamps its own version."""
    from importlib.metadata import version

    from trialagentbench_verifier.contracts import VERIFIER_VERSION

    release = _release(tmp_path / "release")
    submission = _submission(tmp_path / "submission.json")
    evidence = _execution_evidence(release, submission)
    proof_path = evidence["execution_proof"]
    _write_json(proof_path, read_json_object(proof_path) | {"verifier_version": retained_version})
    evidence["expected_execution_proof_sha256"] = hashlib.sha256(proof_path.read_bytes()).hexdigest()
    scored = score_submission(release, submission, graded_item=submission, **evidence)
    assert scored.status == "pass"
    assert scored.verifier_version == VERIFIER_VERSION == version("trialagentbench-verifier")


def test_score_requires_an_authenticated_execution_proof_but_preserves_numerical_metrics(
    tmp_path: Path,
) -> None:
    """Fail closed on replay while retaining independently recomputed numeric information."""

    release = _release(tmp_path / "release")
    submission = _submission(tmp_path / "submission.json")
    payload = read_json_object(submission)
    payload["payload"]["reproducible"] = True
    _write_json(submission, payload)

    report = score_submission(release, submission)

    assert report.status == "fail"
    reproducibility = next(check for check in report.checks if check.check_id == "score.reproducibility")
    estimand = next(check for check in report.checks if check.check_id == "score.estimand_performance")
    assert reproducibility.failure_code == "execution_proof_required"
    assert estimand.status == "pass"
    assert estimand.observed["metrics"]["normalized_absolute_error"] == 0.0


def test_fresh_proof_replaces_the_untrusted_producer_reproducibility_flag(tmp_path: Path) -> None:
    """Use fresh replay facts regardless of a producer-carried Boolean."""

    release = _release(tmp_path / "release")
    submission = _submission(tmp_path / "submission.json")
    payload = read_json_object(submission)
    payload["payload"]["reproducible"] = False
    _write_json(submission, payload)

    report = _authenticated_score(release, submission)

    assert report.status == "pass"
    reproducibility = next(check for check in report.checks if check.check_id == "score.reproducibility")
    assert reproducibility.status == "pass"


def test_score_rejects_a_proof_changed_after_its_trusted_digest_was_recorded(tmp_path: Path) -> None:
    """Treat the verifier-custody proof digest as an external trust anchor."""

    release = _release(tmp_path / "release")
    submission = _submission(tmp_path / "submission.json")
    evidence = _execution_evidence(release, submission)
    proof = Path(evidence["execution_proof"])
    proof.write_text(proof.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(EvaluationReleaseError, match="trusted expected SHA-256"):
        score_submission(release, submission, **evidence)


@pytest.mark.parametrize("field", ("task_id", "candidate_digest", "release_digest"))
def test_score_rejects_wrong_task_candidate_or_release_proof(tmp_path: Path, field: str) -> None:
    """Reject an intact proof replayed against a different scoring identity."""

    release = _release(tmp_path / "release")
    submission = _submission(tmp_path / "submission.json")
    evidence = _execution_evidence(release, submission)
    proof_path = Path(evidence["execution_proof"])
    proof = read_json_object(proof_path)
    proof[field] = "different_task" if field == "task_id" else "f" * 64
    _write_json(proof_path, proof)
    evidence["expected_execution_proof_sha256"] = hashlib.sha256(proof_path.read_bytes()).hexdigest()

    with pytest.raises(EvaluationReleaseError, match="identity differs"):
        score_submission(release, submission, **evidence)


def test_independent_scoring_rejects_unsupported_set_encoding(tmp_path: Path) -> None:
    """The sole released result contract is an estimate and sampling interval."""

    release = _release(tmp_path / "release")
    submission = _submission(tmp_path / "point.json")
    payload = read_json_object(submission)
    result = payload["payload"]["submitted_result"]
    interval = result.pop("interval")
    value = result.pop("value")
    result.update(
        kind="identified_set",
        set_definition_id="participant_singleton",
        lower=value,
        upper=value,
        outer_confidence_region={
            **interval,
            "region_type": "simultaneous_outer_confidence_region",
            "coverage_scope": "identified_set",
        },
    )
    singleton = tmp_path / "singleton.json"
    _write_json(singleton, payload)
    with pytest.raises(ValueError):
        _authenticated_score(release, singleton)


def test_score_rejects_a_stale_proof_for_a_different_normalized_result(tmp_path: Path) -> None:
    """Bind proof facts to the exact score-bearing projection."""

    release = _release(tmp_path / "release")
    submission = _submission(tmp_path / "submission.json")
    evidence = _execution_evidence(release, submission)
    payload = read_json_object(submission)
    payload["payload"]["submitted_result"]["value"] = 0.2
    _write_json(submission, payload)

    with pytest.raises(EvaluationReleaseError, match="normalized score input"):
        score_submission(release, submission, **evidence)


def test_score_rejects_a_substituted_program_or_program_result(tmp_path: Path) -> None:
    """Bind both executable and declared ProgramResult bytes to the replay proof."""

    release = _release(tmp_path / "release")
    submission = _submission(tmp_path / "submission.json")
    evidence = _execution_evidence(release, submission)
    Path(evidence["program"]).write_text("# substituted\n", encoding="utf-8")
    with pytest.raises(EvaluationReleaseError, match="supplied program"):
        score_submission(release, submission, **evidence)

    evidence = _execution_evidence(release, submission)
    result = Path(evidence["program_result"])
    payload = read_json_object(result)
    payload["result"]["value"] = 0.2
    _write_json(result, payload)
    with pytest.raises(EvaluationReleaseError, match="supplied program result"):
        score_submission(release, submission, **evidence)


def test_portable_trialeval_score_input_runs_without_a_wrapper(tmp_path: Path) -> None:
    """Verify the exact score-input file emitted by the public grader."""

    wrapped = json.loads(_submission(tmp_path / "wrapped.json").read_text(encoding="utf-8"))
    direct = tmp_path / "score_input.json"
    _write_json(direct, wrapped["payload"])

    scored = _authenticated_score(_release(tmp_path / "release"), direct)

    assert scored.status == "pass"
    assert scored.scheduled_check_count == 6


def test_evidence_mutation_does_not_change_numerical_verification(tmp_path: Path) -> None:
    """Keep evidence trace separate from estimator-neutral numerical accuracy."""

    release = _release(tmp_path / "release")
    wrapped = json.loads(_submission(tmp_path / "wrapped.json").read_text(encoding="utf-8"))
    eligible = TrialEvalScoreInputV1.model_validate(wrapped["payload"])
    ineligible = eligible.model_copy(update={"evidence_source_artifacts": ("scratch/derived.csv",)})

    eligible_result = verify_trialeval_outcome_v1(
        release_root=release,
        score_input=eligible,
    )
    ineligible_result = verify_trialeval_outcome_v1(
        release_root=release,
        score_input=ineligible,
    )

    eligible_components = {component.component_id: component for component in eligible_result.components}
    ineligible_components = {component.component_id: component for component in ineligible_result.components}
    assert eligible_components["evidence_eligibility"].status == "satisfied"
    assert ineligible_components["evidence_eligibility"].status == "not_satisfied"
    assert eligible_components["estimand_performance"] == ineligible_components["estimand_performance"]
    assert eligible_components["uncertainty"] == ineligible_components["uncertainty"]
    assert eligible_result.metrics == ineligible_result.metrics


def test_interval_type_mismatch_preserves_point_metrics(tmp_path: Path) -> None:
    """Reject a repeated interval where the question requests an ordinary interval."""

    release = _release(tmp_path / "release")
    wrapped = json.loads(_submission(tmp_path / "wrapped.json").read_text(encoding="utf-8"))
    score_input = TrialEvalScoreInputV1.model_validate(wrapped["payload"])
    assert score_input.submitted_result is not None
    assert score_input.submitted_result.kind == "point"
    submitted = score_input.submitted_result.model_copy(
        update={
            "interval": score_input.submitted_result.interval.model_copy(
                update={"interval_type": "repeated_confidence_interval"}
            )
        }
    )

    result = verify_trialeval_outcome_v1(
        release_root=release,
        score_input=score_input.model_copy(update={"submitted_result": submitted}),
    )

    components = {component.component_id: component for component in result.components}
    assert components["estimand_performance"].status == "observed"
    assert components["uncertainty"].status == "not_evaluable"
    assert components["uncertainty"].failure_origin == "interval_type_mismatch"
    assert result.metrics is not None


def test_identification_declaration_does_not_suppress_portable_metrics(tmp_path: Path) -> None:
    """Report an identification mismatch without discarding the estimate."""

    release = _release(tmp_path / "release")
    wrapped = json.loads(_submission(tmp_path / "wrapped.json").read_text(encoding="utf-8"))
    score_input = TrialEvalScoreInputV1.model_validate(wrapped["payload"])
    assert score_input.submitted_binding is not None
    submitted = score_input.submitted_binding.model_copy(
        update={
            "identification_status": "identified_under_assumptions",
            "identification_assumption_ids": ("missingness_unrelated_to_outcome",),
        }
    )

    result = verify_trialeval_outcome_v1(
        release_root=release,
        score_input=score_input.model_copy(update={"submitted_binding": submitted}),
    )

    components = {component.component_id: component for component in result.components}
    assert components["question_alignment"].status == "not_satisfied"
    assert components["estimand_performance"].status == "observed"
    assert components["uncertainty"].status == "observed"
    assert result.metrics is not None


def test_score_input_rejects_a_fabricated_analysis_method() -> None:
    payload = {
        "schema_id": "trialagentbench.trialeval_score_input/v1",
        "task_id": "task-1",
        "attempt_status": "missing",
        "analysis_method_id": "claimed-plan-method",
    }

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        TrialEvalScoreInputV1.model_validate(payload)


def test_evaluator_archive_requires_an_external_trust_anchor(tmp_path: Path) -> None:
    """Do not treat a self-consistent evaluator archive as publisher-authentic."""

    release = _release(tmp_path / "release")
    release_manifest = json.loads((release / "release_manifest.json").read_text(encoding="utf-8"))
    participant_manifest = json.loads((release / "participant" / "manifest.json").read_text(encoding="utf-8"))
    participant_identity = {
        "schema_id": "ariadne.trialeval.participant_archive_identity/v1",
        "release_id": release_manifest["release_id"],
        "participant_archive_sha256": "0" * 64,
        "participant_manifest_checksum": participant_manifest["checksum"],
    }
    participant_identity["checksum"] = _canonical(participant_identity)
    _write_json(release / "participant_archive_identity.json", participant_identity)
    archive_path = tmp_path / "evaluator.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in sorted(release.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(release).as_posix())
    expected = hashlib.sha256(archive_path.read_bytes()).hexdigest()

    unanchored = verify_release(archive_path)
    anchored = verify_release(archive_path, expected_sha256=expected)

    assert unanchored.status == "fail"
    assert unanchored.checks[0].failure_code == "evaluator_trust_anchor_required"
    assert anchored.status == "pass"
    assert anchored.checks[1].check_id == "release.archive_identity"
    assert anchored.input_sha256 == anchored.expected_input_sha256 == expected
    with pytest.raises(EvaluationReleaseError, match="requires a trusted expected SHA-256"):
        score_submission(archive_path, _submission(tmp_path / "unanchored-submission.json"))
    scored = score_submission(
        archive_path,
        _submission(tmp_path / "anchored-submission.json"),
        expected_sha256=expected,
    )
    assert scored.status == "fail"
    reproducibility = next(check for check in scored.checks if check.check_id == "score.reproducibility")
    assert reproducibility.failure_code == "execution_proof_required"

    request = ScoreSubmissionV1(submission=tmp_path / "anchored-submission.json")
    batch = list(score_submissions(archive_path, (request, request), expected_sha256=expected))
    assert batch == [scored, scored]
    assert all(report.input_kind == "archive" and report.input_sha256 == expected for report in batch)
    with pytest.raises(EvaluationReleaseError):
        list(score_submissions(archive_path, (request,), expected_sha256="f" * 64))


def test_participant_archive_requires_and_records_an_external_trust_anchor(tmp_path: Path) -> None:
    """Separate participant integrity from provenance supplied by a trusted digest."""

    release = _release(tmp_path / "release")
    archive_path = tmp_path / "participant.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in sorted((release / "participant").rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(release / "participant").as_posix())
        archive.write(release / "release_manifest.json", "release_manifest.json")

    unanchored = verify_release(archive_path)
    expected = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    report = verify_release(archive_path, expected_sha256=expected)

    assert unanchored.status == "fail"
    assert unanchored.checks[1].failure_code == "participant_trust_anchor_required"
    assert report.status == "pass"
    assert report.input_sha256 == report.expected_input_sha256 == expected
    assert report.benchmark_stream == "trialeval"
    assert tuple(check.check_id for check in report.checks) == (
        "participant.inventory",
        "participant.candidate_identity",
    )


def test_participant_resealing_cannot_retain_the_outer_release_identity(tmp_path: Path) -> None:
    """Reject participant bytes re-manifested under an unchanged outer release."""

    release = _release(tmp_path / "release")
    participant = tmp_path / "participant"
    shutil.copytree(release / "participant", participant)
    shutil.copyfile(release / "release_manifest.json", participant / "release_manifest.json")
    task = participant / "task.txt"
    task.write_text("changed\n", encoding="utf-8")
    manifest_path = participant / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"] = [
        {
            "relative_path": "participant/task.txt",
            "sha256": hashlib.sha256(task.read_bytes()).hexdigest(),
            "byte_length": task.stat().st_size,
        }
    ]
    manifest.pop("checksum")
    manifest["checksum"] = _canonical(manifest)
    _write_json(manifest_path, manifest)

    with pytest.raises(EvaluationReleaseError, match="not bound by the outer release manifest"):
        verify_release(participant)


def test_trusted_archive_digest_is_checked_before_extraction(tmp_path: Path) -> None:
    """Reject a resealed archive when its transport identity is not trusted."""

    release = _release(tmp_path / "release")
    archive_path = tmp_path / "participant.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in sorted((release / "participant").rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(release / "participant").as_posix())
        archive.write(release / "release_manifest.json", "release_manifest.json")
    expected = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    report = verify_release(archive_path, expected_sha256=expected)
    assert report.status == "pass"
    assert report.input_kind == "archive"
    assert report.input_sha256 == expected
    assert report.expected_input_sha256 == expected

    rewritten = tmp_path / "rewritten.zip"
    with zipfile.ZipFile(archive_path) as source, zipfile.ZipFile(rewritten, "w") as target:
        for name in source.namelist():
            target.writestr(name, source.read(name))
        target.writestr("surplus.txt", "changed")
    with pytest.raises(EvaluationReleaseError, match="trusted expected SHA-256"):
        verify_release(rewritten, expected_sha256=expected)


def test_hash_valid_historical_score_does_not_assert_current_authorization(tmp_path: Path) -> None:
    """Offline agreement cannot establish current access or withdrawal status."""
    release = _release(tmp_path / "historical")
    report = _authenticated_score(release, _submission(tmp_path / "submission.json"))
    assert report.status == "pass"
    assert report.authorization_status == "not_assessed"
    # Once the owner supplies a different current trust anchor, old internally
    # consistent bytes cannot be relabelled as the current evaluation release.
    with pytest.raises(EvaluationReleaseError, match="trusted expected SHA-256"):
        verify_release(release, expected_sha256="0" * 64)


def test_trialeval_loss_does_not_adjudicate_report_interpretation(tmp_path: Path) -> None:
    """Report interpretation is behavioural evidence, not a separate scientific rubric."""

    release = _release(tmp_path / "release")
    submission_path = _submission(tmp_path / "submission.json")
    payload = json.loads(submission_path.read_text(encoding="utf-8"))
    payload["payload"]["submitted_result"]["conclusion"] = "favors_comparator"
    _write_json(submission_path, payload)

    scored = _authenticated_score(release, submission_path)

    assert scored.status == "pass"
    assert scored.failed_check_count == 0
    assert all(check.check_id != "score.interpretation" for check in scored.checks)


@pytest.mark.parametrize("failure_code", ("missing_primary_submission", "missing_required_deliverable"))
def test_trialeval_score_preserves_submission_failures(
    tmp_path: Path,
    failure_code: str,
) -> None:
    """Return a denominator-preserving score for every harness submission state."""

    release = _release(tmp_path / "release")
    submission = NormalizedSubmissionV1.model_validate(
        {
            "benchmark_stream": "trialeval",
            "task_id": "task_1",
            "submission_kind": "analysis",
            "submission_failure_code": failure_code,
            "payload": {},
        }
    )
    submission_path = tmp_path / "submission.json"
    _write_json(submission_path, submission.model_dump(mode="json"))

    scored = score_submission(release, submission_path)

    assert scored.status == "fail"
    assert scored.scheduled_check_count == 6
    assert scored.failed_check_count == 6
    expected_origin = (
        "missing_submission" if failure_code == "missing_primary_submission" else "missing_required_output"
    )
    assert {check.failure_code for check in scored.checks} == {expected_origin}


def test_byte_mutation_is_terminal(tmp_path: Path) -> None:
    """Reject a released artifact whose bytes no longer match its role manifest."""

    release = _release(tmp_path / "release")
    (release / "participant" / "task.txt").write_text("changed\n", encoding="utf-8")
    report = verify_release(release)
    assert report.status == "fail"
    assert report.failed_check_count == 1
    assert report.checks[0].failure_code == "release_integrity_failure"


@pytest.mark.parametrize("operation", ("byte_mutation", "omission", "replacement"))
def test_every_role_manifest_artifact_is_tamper_evident(tmp_path: Path, operation: str) -> None:
    """Reject each byte, omission, or replacement mutation at every declared artifact."""

    source = _release(tmp_path / "source")
    declared: list[str] = []
    for role in ("participant", "evaluation"):
        manifest = json.loads((source / role / "manifest.json").read_text(encoding="utf-8"))
        declared.extend(str(row["relative_path"]) for row in manifest["artifacts"])

    for index, relative in enumerate(declared):
        release = tmp_path / f"{operation}-{index}"
        shutil.copytree(source, release)
        target = release / relative
        original = target.read_bytes()
        if operation == "omission":
            target.unlink()
        elif operation == "replacement":
            target.write_bytes(b"x" * len(original))
        else:
            target.write_bytes(original + b"changed")
        report = verify_release(release)
        assert report.status == "fail", relative
        assert report.checks[0].failure_code == "release_integrity_failure", relative


@pytest.mark.parametrize(
    "relative",
    (
        "/tmp/task.txt",
        "participant/../evaluation/qualification.json",
        "evaluation/qualification.json",
        "participant/./task.txt",
    ),
)
def test_role_manifest_rejects_path_manipulation(tmp_path: Path, relative: str) -> None:
    """Reject absolute, traversing, cross-role, and non-normalized paths after resealing."""

    release = _release(tmp_path / "release")
    role_path = release / "participant" / "manifest.json"
    role = json.loads(role_path.read_text(encoding="utf-8"))
    role["artifacts"][0]["relative_path"] = relative
    _reseal_role(release, role="participant", payload=role)

    report = verify_release(release)
    assert report.status == "fail"
    assert report.checks[0].failure_code == "release_integrity_failure"


def test_release_rejects_undeclared_files_and_symlink_substitution(tmp_path: Path) -> None:
    """Require exact file inventory and regular in-tree released artifacts."""

    undeclared = _release(tmp_path / "undeclared")
    (undeclared / "participant" / "extra.txt").write_text("extra\n", encoding="utf-8")
    assert verify_release(undeclared).status == "fail"

    symlinked = _release(tmp_path / "symlinked")
    outside = tmp_path / "outside.txt"
    outside.write_text("task\n", encoding="utf-8")
    task = symlinked / "participant" / "task.txt"
    task.unlink()
    task.symlink_to(outside)
    assert verify_release(symlinked).status == "fail"


@pytest.mark.parametrize(
    "relative",
    ("release_manifest.json", "participant/manifest.json", "evaluation/manifest.json"),
)
def test_every_manifest_identity_is_tamper_evident(tmp_path: Path, relative: str) -> None:
    """Reject changed outer or role-manifest bytes before accepting the release."""

    release = _release(tmp_path / "release")
    path = release / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["task_count"] = 2
    _write_json(path, payload)
    assert verify_release(release).status == "fail"


def test_report_mutation_is_terminal(tmp_path: Path) -> None:
    """Reject a substituted qualification report through the ordinary reader."""

    release = _release(tmp_path / "release")
    report = release / "evaluation" / "verification" / "reports" / "G-TAB-01.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["candidate_digest"] = "b" * 64
    _write_json(report, payload)
    _reseal_evaluation(release, "evaluation/verification/reports/G-TAB-01.json")
    report = verify_release(release)
    assert report.status == "fail"
    assert report.checks[0].failure_code == "release_integrity_failure"


def test_assignment_mutation_is_terminal_after_manifest_reseal(tmp_path: Path) -> None:
    """Reject an evaluator record that no longer assigns the submitted task exactly once."""

    release = _release(tmp_path / "release")
    key_path = release / RECORDS
    key = json.loads(key_path.read_text(encoding="utf-8"))
    key["task_id"] = "different_task"
    _write_records(release, key)
    _reseal_key(release)
    with pytest.raises(ValueError, match="does not contain the task exactly once"):
        score_submission(release, _submission(tmp_path / "submission.json"))


def test_score_mutation_fails_without_compensation(tmp_path: Path) -> None:
    """Retain a wrong estimate as a continuous error rather than a route failure."""

    release = _release(tmp_path / "release")
    submission_path = _submission(tmp_path / "submission.json")
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    submission["payload"]["submitted_result"]["value"] = 0.2
    _write_json(submission_path, submission)
    report = _authenticated_score(release, submission_path)
    assert report.status == "pass"
    estimand = next(check for check in report.checks if check.check_id == "score.estimand_performance")
    assert estimand.observed["metrics"]["normalized_absolute_error"] == pytest.approx(2.0)


def test_score_input_has_no_estimator_gate(tmp_path: Path) -> None:
    """Score a complete outcome without declaring an estimator identity."""

    report = _authenticated_score(_release(tmp_path / "release"), _submission(tmp_path / "submission.json"))

    assert report.status == "pass"
    assert all("route" not in check.check_id for check in report.checks)


def test_alternative_effect_measure_is_a_wrong_question_not_a_schema_failure(tmp_path: Path) -> None:
    """Retain a valid but mismatched analysis scale as an interpretable failure."""

    release = _release(tmp_path / "release")
    submission_path = _submission(tmp_path / "submission.json")
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    binding = submission["payload"]["submitted_binding"]
    binding.update(effect_measure="hazard_ratio", result_unit="ratio", null_value=1.0)
    _write_json(submission_path, submission)

    report = _authenticated_score(release, submission_path)

    question = next(check for check in report.checks if check.check_id == "score.question_alignment")
    estimand = next(check for check in report.checks if check.check_id == "score.estimand_performance")
    assert question.status == "fail"
    assert question.failure_code == "wrong_question"
    assert estimand.status == "fail"
    assert estimand.failure_code == "wrong_question"


def test_portable_comparison_cannot_store_a_false_decision(tmp_path: Path) -> None:
    """Reject comparator bytes whose stored difference does not follow its values."""

    release = _release(tmp_path / "release")
    bundle = release / "evaluation" / "verification" / "qualification_reports.json"
    payload = json.loads(bundle.read_text(encoding="utf-8"))
    payload["comparisons"][0]["candidate_value"] = 2.0
    _write_json(bundle, payload)
    _reseal_evaluation(release, "evaluation/verification/qualification_reports.json")
    report = verify_release(release)
    assert report.status == "fail"
    assert report.checks[0].failure_code == "release_integrity_failure"


def test_portable_comparison_must_cite_a_released_artifact(tmp_path: Path) -> None:
    """Reject a comparator that cannot be traced to declared release bytes."""

    release = _release(tmp_path / "release")
    bundle = release / "evaluation" / "verification" / "qualification_reports.json"
    payload = json.loads(bundle.read_text(encoding="utf-8"))
    payload["comparisons"][0]["artifact_paths"] = ["evaluation/not-released.json"]
    _write_json(bundle, payload)
    _reseal_evaluation(release, "evaluation/verification/qualification_reports.json")
    report = verify_release(release)
    assert report.status == "fail"
    assert "missing or unsafe artifact" in str(report.checks[0].observed)


def test_cli_exposes_release_score_execution_and_practice(tmp_path: Path) -> None:
    """Expose the complete small public verification workflow."""

    assert set(_parser()._subparsers._group_actions[0].choices) == {
        "analyse-trialeval",
        "analyse-trialdev",
        "assets",
        "attest-executor",
        "behaviour",
        "compare-trialdev",
        "execute",
        "practice",
        "release",
        "release-set",
        "reproducibility",
        "score",
        "toolkit",
    }
    release = _release(tmp_path / "release")
    output = tmp_path / "report.json"
    assert main(["release", str(release), "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "pass"


def test_disclosed_practice_suite_covers_required_counterexamples() -> None:
    """Execute all package-contained known answers, for the shared numerical interface."""

    report = verify_practice_suite()

    assert report.status == "pass"
    assert report.scheduled_check_count == report.passed_check_count == len(report.checks)
    assert {check.scope_id for check in report.checks} >= {
        "loss.accurate",
        "loss.wide",
        "loss.narrow_wrong",
        "loss.point_outside_interval",
    }


def test_cli_writes_a_terminal_report_for_invalid_score_inputs(tmp_path: Path) -> None:
    """Return structured failure evidence instead of an internal traceback."""

    output = tmp_path / "score-failure.json"
    assert (
        main(
            [
                "score",
                str(tmp_path / "missing-release"),
                "--submission",
                str(tmp_path / "missing-submission.json"),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_id"] == "trialagentbench.verification_failure/v1"
    assert report["operation"] == "score"
    assert report["status"] == "fail"
    assert report["failure_code"] == "invalid_input"


def test_standard_method_adapters_match_analytic_fixtures() -> None:
    """Reproduce closed-form results through the maintained-library adapters."""

    outcome = np.array([1.0, 2.0, 4.0, 6.0])
    treatment = np.array([0, 0, 1, 1])
    estimate, standard_error = _linear_contrast(outcome, treatment)
    assert estimate == pytest.approx(3.5)
    assert standard_error > 0
    assert _weighted_mean_difference(outcome, treatment, np.ones(4)) == pytest.approx(3.5)
    survival, risk, rmst = _kaplan_meier_summary(
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([1, 1, 0, 0]),
        horizon=4.0,
    )
    assert survival == pytest.approx(0.5)
    assert risk == pytest.approx(0.5)
    assert rmst == pytest.approx(2.75)
    assert _cluster_design_effect(10.0, 0.1) == pytest.approx(1.9)
    assert _obrien_fleming_boundary(1.0, two_sided_alpha=0.05) == pytest.approx(1.9599639845)


def test_c5_grades_numerical_consequences_without_a_cleaned_table(tmp_path: Path) -> None:
    """No private input table is needed to score a source-derived numerical answer."""
    release = _release(tmp_path / "release")
    record_path = release / RECORDS
    record = json.loads(record_path.read_text())
    record.update(context="C5", data_integrity=C5_INTEGRITY)
    _write_records(release, record)
    submission = TrialEvalScoreInputV1.model_validate(
        read_json_object(_submission(tmp_path / "submission.json"))["payload"]
    )
    verified = verify_trialeval_outcome_v1(release_root=release, score_input=submission)
    assert next(row for row in verified.components if row.component_id == "evidence_eligibility").status == "satisfied"
    assert verified.metrics is not None
    # A bad numerical result remains a scientific error, not an evidence-access failure.
    payload = submission.model_dump(mode="json")
    payload["submitted_result"]["value"] += 1.0
    wrong = verify_trialeval_outcome_v1(
        release_root=release, score_input=TrialEvalScoreInputV1.model_validate(payload)
    )
    assert next(row for row in wrong.components if row.component_id == "evidence_eligibility").status == "satisfied"
    assert wrong.metrics != verified.metrics


def test_execution_proof_binds_retained_analysis_data_bytes(tmp_path: Path) -> None:
    """Replacing a correct table after replay invalidates its authenticated proof."""
    release = _release(tmp_path / "release")
    submission = _submission(tmp_path / "submission.json")
    evidence = _execution_evidence(release, submission)
    program_result = Path(evidence["program_result"])
    data = program_result.parent / "analysis_data.csv"
    data.write_text("participant_id,endpoint\np1,1\n")
    proof_path = Path(evidence["execution_proof"])
    proof = read_json_object(proof_path)
    proof.update(
        analysis_data_filename=data.name,
        analysis_data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
        analysis_data_reproduced=True,
    )
    _write_json(proof_path, proof)
    evidence["expected_execution_proof_sha256"] = hashlib.sha256(proof_path.read_bytes()).hexdigest()
    assert score_submission(release, submission, **evidence).status == "pass"
    data.write_text("participant_id,endpoint\np1,9\n")
    with pytest.raises(VerificationInputError, match="analysis data differs"):
        score_submission(release, submission, **evidence)


def test_batch_scoring_preserves_reports_and_releases_one_authenticated_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Amortize archive work without changing scores or hiding a later bad submission."""
    release = _release(tmp_path / "release")
    request = ScoreSubmissionV1(submission=_submission(tmp_path / "submission.json"))
    expected = score_submission(release, request.submission)
    original = release_module.materialized_release
    entered: list[Path] = []
    closed: list[Path] = []

    @contextmanager
    def observed(path: Path, *, task_ids: tuple[str, ...] | None = None) -> Iterator[Path]:
        with original(path, task_ids=task_ids) as root:
            entered.append(root)
            try:
                yield root
            finally:
                closed.append(root)

    monkeypatch.setattr(release_module, "materialized_release", observed)
    reports = list(score_submissions(release, (request, request)))
    assert reports == [expected, expected]
    assert entered == closed == [release]
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"schema_id":"unsupported"}')
    stream = score_submissions(release, (request, ScoreSubmissionV1(submission=invalid)))
    assert next(stream) == expected
    with pytest.raises(ValueError):
        next(stream)
    assert len(entered) == len(closed) == 2


def test_trialeval_csv_scores_like_normalized_input_and_rejects_changed_values(tmp_path: Path) -> None:
    """The simple CSV interface preserves independent score and proof binding."""
    release = _release(tmp_path / "release")
    submission = _submission(tmp_path / "submission.json")
    normalized = NormalizedSubmissionV1.model_validate(read_json_object(submission))
    task_path = release / "participant/tasks/task_1/task.json"
    _write_json(
        task_path,
        {
            "task_id": "task_1",
            "target_ids": [normalized.payload["submitted_binding"]["estimand_id"]],
            "available_materials": [{"relative_path": "data/analysis.parquet"}],
        },
    )
    role = read_json_object(release / "participant/manifest.json")
    role["artifacts"].append(_artifact(release, "participant/tasks/task_1/task.json"))
    _reseal_role(release, role="participant", payload=role)
    evidence = _execution_evidence(release, submission)
    csv_path = tmp_path / "estimates.csv"
    csv_path.write_text("target_id,estimate,lower,upper\nrisk_difference,0.1,-0.1,0.3\n")
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    proof_path = evidence["execution_proof"]
    proof = read_json_object(proof_path)
    proof.update(
        result_format="csv",
        target_ids=["risk_difference"],
        program_result_sha256=digest,
        replayed_result_sha256s=[digest, digest],
        evidence_artifacts=[
            {
                "relative_path": "data/analysis.parquet",
                "sha256": "b" * 64,
                "byte_length": 1,
            }
        ],
    )
    _write_json(proof_path, proof)
    evidence["expected_execution_proof_sha256"] = hashlib.sha256(proof_path.read_bytes()).hexdigest()
    evidence["program_result"] = csv_path
    expected = score_submission(release, submission, **evidence)
    observed = score_submission(release, csv_path, **evidence)
    assert observed.status == expected.status == "pass"
    assert observed.checks == expected.checks
    csv_path.write_text("target_id,estimate,lower,upper\nrisk_difference,0.9,-0.1,0.3\n")
    with pytest.raises(VerificationInputError, match="result differs"):
        score_submission(release, csv_path, **evidence)


def test_trialeval_csv_scores_a_nonpositive_ratio_as_missing_like_the_harness(tmp_path: Path) -> None:
    """A nonpositive rate ratio in the executed CSV is an ineligible answer, not invalid verifier input."""
    release = _release(tmp_path / "release")
    record_path = release / RECORDS
    record = json.loads(record_path.read_text())
    record["target"]["binding"].update(effect_measure="rate_ratio", result_unit="ratio", null_value=1.0)
    _write_records(release, record)
    _reseal_key(release)
    submission = _submission(tmp_path / "submission.json")
    _write_json(
        release / "participant/tasks/task_1/task.json",
        {
            "task_id": "task_1",
            "target_ids": ["risk_difference"],
            "available_materials": [{"relative_path": "data/analysis.parquet"}],
        },
    )
    role = read_json_object(release / "participant/manifest.json")
    role["artifacts"].append(_artifact(release, "participant/tasks/task_1/task.json"))
    _reseal_role(release, role="participant", payload=role)
    evidence = _execution_evidence(release, submission)
    csv_path = tmp_path / "estimates.csv"
    csv_path.write_text("target_id,estimate,lower,upper\nrisk_difference,0,0,0\n")
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    proof_path = evidence["execution_proof"]
    proof = read_json_object(proof_path)
    proof.update(
        result_format="csv",
        target_ids=["risk_difference"],
        program_result_sha256=digest,
        replayed_result_sha256s=[digest, digest],
        evidence_artifacts=[{"relative_path": "data/analysis.parquet", "sha256": "b" * 64, "byte_length": 1}],
        score_projection_sha256=numerical_score_projection_sha256(
            read_numerical_table(csv_path, target_ids=("risk_difference",)), (("data/analysis.parquet",),)
        ),
    )
    _write_json(proof_path, proof)
    evidence["expected_execution_proof_sha256"] = hashlib.sha256(proof_path.read_bytes()).hexdigest()
    evidence["program_result"] = csv_path
    report = score_submission(release, csv_path, **evidence)
    numerical = next(check for check in report.checks if check.check_id == "score.estimand_performance")
    assert numerical.observed["failure_origin"] == "missing_required_output"
    assert numerical.observed["status"] == "not_evaluable"


def test_bounds_target_evaluator_is_rejected(tmp_path: Path) -> None:
    """A bounds target cannot enter the point-loss release."""
    release = _set_release(tmp_path / "release")
    submission = _set_submission(tmp_path / "submission.json", lower=0.1, upper=0.2, region_lower=0, region_upper=0.3)
    with pytest.raises(ValueError):
        _authenticated_score(release, submission)


@pytest.mark.parametrize("estimate,expected", [(1e200, 1e201), (1e308, None)])
def test_independent_public_score_handles_extreme_finite_answers(
    tmp_path: Path, estimate: float, expected: float | None
) -> None:
    """Preserve completion and finite loss without serializing overflowing diagnostics."""
    release = _release(tmp_path / "release")
    path = _submission(tmp_path / "submission.json")
    payload = read_json_object(path)
    payload["payload"]["submitted_result"]["value"] = estimate
    _write_json(path, payload)
    report = _authenticated_score(release, path)
    report.model_dump_json()
    observed = verify_trialeval_outcome_v1(
        release_root=release, score_input=TrialEvalScoreInputV1.model_validate(payload["payload"])
    )
    assert observed.metrics is not None
    assert observed.metrics["normalized_squared_error"] is None
    assert observed.signal_loss == pytest.approx(expected) if expected is not None else observed.signal_loss is None
    assert observed.components[0].status == "satisfied"


def test_paired_results_are_independently_scored_and_bound_to_every_csv_row(tmp_path: Path) -> None:
    """Complementary results affect task loss and cannot escape execution custody."""
    from copy import deepcopy

    release = _release(tmp_path / "release")
    record_path = release / RECORDS
    record = json.loads(record_path.read_text())
    extra = {
        key: deepcopy(record[key])
        for key in (
            "target",
            "oracle_worlds",
            "oracle_monte_carlo_standard_error",
            "maximum_oracle_monte_carlo_standard_error",
        )
    }
    extra["target"]["binding"].update(
        estimand_id="event_free_time", effect_measure="restricted_mean_survival_time_difference", result_unit="days"
    )
    extra["target"].update(value=4.0, reference_sampling_standard_deviation=2.0)
    record["additional_references"] = [extra]
    _write_records(release, record)
    _reseal_key(release)
    submission_path = _submission(tmp_path / "paired.json")
    normalized = read_json_object(submission_path)
    primary = normalized["payload"]
    primary["target_id"] = "risk_difference"
    secondary = {key: deepcopy(value) for key, value in primary.items() if key not in {"schema_id", "task_id"}}
    secondary["target_id"] = "event_free_time"
    secondary["submitted_binding"] = extra["target"]["binding"]
    secondary["submitted_result"].update(
        value=4.0,
        effect_measure="restricted_mean_survival_time_difference",
        unit="days",
        interval={"lower": 2.0, "upper": 6.0, "confidence_level": 0.95},
    )
    primary["additional_inputs"] = [secondary]
    _write_json(submission_path, normalized)
    score_input = TrialEvalScoreInputV1.model_validate(primary)
    verified = verify_trialeval_outcome_v1(release_root=release, score_input=score_input)
    assert [item.signal_loss for item in verified.target_outcomes] == pytest.approx([4.0, 1.0])
    assert verified.signal_loss == pytest.approx(2.5)
    omitted = score_input.model_copy(update={"additional_inputs": ()})
    missing = verify_trialeval_outcome_v1(release_root=release, score_input=omitted)
    assert missing.signal_loss is None
    assert missing.target_outcomes[0].signal_loss == pytest.approx(4.0)
    assert missing.target_outcomes[1].signal_loss is None
    secondary_input = score_input.additional_inputs[0]
    assert secondary_input.submitted_result is not None
    wrong_unit = score_input.model_copy(
        update={
            "additional_inputs": (
                secondary_input.model_copy(
                    update={"submitted_result": secondary_input.submitted_result.model_copy(update={"unit": "years"})}
                ),
            )
        }
    )
    mislabelled = verify_trialeval_outcome_v1(release_root=release, score_input=wrong_unit)
    assert mislabelled.signal_loss is None
    assert len(mislabelled.target_outcomes) == 2
    assert mislabelled.target_outcomes[0].signal_loss == pytest.approx(4.0)
    assert mislabelled.target_outcomes[1].signal_loss is None
    assert any(item.failure_origin == "wrong_question" for item in mislabelled.target_outcomes[1].components)

    task_path = release / "participant/tasks/task_1/task.json"
    _write_json(
        task_path,
        {
            "task_id": "task_1",
            "target_ids": ["risk_difference", "event_free_time"],
            "available_materials": [{"relative_path": "data/analysis.parquet"}],
        },
    )
    role = read_json_object(release / "participant/manifest.json")
    role["artifacts"].append(_artifact(release, "participant/tasks/task_1/task.json"))
    _reseal_role(release, role="participant", payload=role)
    evidence = _execution_evidence(release, submission_path)
    with pytest.raises(EvaluationReleaseError, match="complete CSV"):
        score_submission(release, submission_path, **evidence)
    result = Path(evidence["program_result"])
    result.write_text("target_id,estimate,lower,upper\nevent_free_time,4,2,6\nrisk_difference,0.1,-0.1,0.3\n")
    proof_path = Path(evidence["execution_proof"])
    proof = read_json_object(proof_path)
    digest = hashlib.sha256(result.read_bytes()).hexdigest()
    proof.update(
        result_format="csv",
        target_ids=["risk_difference", "event_free_time"],
        evidence_artifacts=[{"relative_path": "data/analysis.parquet", "sha256": "b" * 64, "byte_length": 1}],
        program_result_sha256=digest,
        replayed_result_sha256s=[digest, digest],
    )
    _write_json(proof_path, proof)
    evidence["expected_execution_proof_sha256"] = hashlib.sha256(proof_path.read_bytes()).hexdigest()
    report = score_submission(release, submission_path, **evidence)
    assert report.status == "pass"
    numerical = next(check for check in report.checks if check.check_id == "score.estimand_performance")
    assert numerical.observed["signal_loss"] == pytest.approx(2.5)
    assert len(numerical.observed["targets"]) == 2
    csv_path = tmp_path / "estimates.csv"
    csv_path.write_bytes(result.read_bytes())
    direct = score_submission(release, csv_path, **evidence)
    direct_numerical = next(check for check in direct.checks if check.check_id == "score.estimand_performance")
    assert direct_numerical.observed == numerical.observed

    result.write_text("target_id,estimate,lower,upper\nevent_free_time,9,2,10\nrisk_difference,0.1,-0.1,0.3\n")
    digest = hashlib.sha256(result.read_bytes()).hexdigest()
    proof.update(program_result_sha256=digest, replayed_result_sha256s=[digest, digest])
    _write_json(proof_path, proof)
    evidence["expected_execution_proof_sha256"] = hashlib.sha256(proof_path.read_bytes()).hexdigest()
    with pytest.raises(EvaluationReleaseError, match="CSV numerical values"):
        score_submission(release, submission_path, **evidence)

    # A retained row the grader scored as missing (for example a nonpositive ratio) carries no answer: custody
    # still binds the CSV, and only committed answers must equal their rows.
    result.write_text("target_id,estimate,lower,upper\nevent_free_time,4,2,6\nrisk_difference,0.1,-0.1,0.3\n")
    digest = hashlib.sha256(result.read_bytes()).hexdigest()
    proof.update(program_result_sha256=digest, replayed_result_sha256s=[digest, digest])
    _write_json(proof_path, proof)
    evidence["expected_execution_proof_sha256"] = hashlib.sha256(proof_path.read_bytes()).hexdigest()
    scored_missing = read_json_object(submission_path)
    scored_missing["payload"]["additional_inputs"][0].update(
        attempt_status="missing_required_output",
        submitted_result=None,
        submitted_binding=None,
        evidence_source_artifacts=[],
        reproducible=False,
    )
    _write_json(submission_path, scored_missing)
    missing_report = score_submission(release, submission_path, **evidence)
    # Custody authenticates; the missing complementary answer is then a scored failure, not an error.
    assert missing_report.status == "fail"
    missing_numerical = next(
        check for check in missing_report.checks if check.check_id == "score.estimand_performance"
    )
    assert missing_numerical.observed["signal_loss"] is None


@pytest.mark.parametrize("defect", ["precision", "population", "duplicate_target"])
def test_additional_reference_requires_its_own_qualified_precision(tmp_path, defect):
    """A qualified primary cannot conceal an invalid secondary reference."""
    release = _release(tmp_path / "release")
    path = release / RECORDS
    record = json.loads(path.read_text())
    extra = {
        key: record[key]
        for key in (
            "target",
            "oracle_worlds",
            "oracle_monte_carlo_standard_error",
            "maximum_oracle_monte_carlo_standard_error",
        )
    }
    if defect == "precision":
        extra["oracle_monte_carlo_standard_error"] = 1
    elif defect == "population":
        extra["oracle_worlds"] = 10
    else:
        extra["oracle_worlds"] = 1  # A copy of the primary target repeats its result identifier.
    record["additional_references"] = [extra]
    _write_records(release, record)
    submission = read_json_object(_submission(tmp_path / "submission.json"))
    with pytest.raises(ValueError, match="Monte Carlo|finite-population|repeat a result identifier"):
        verify_trialeval_outcome_v1(
            release_root=release, score_input=TrialEvalScoreInputV1.model_validate(submission["payload"])
        )


def test_standalone_catalog_score_needs_only_submitted_task_shards(tmp_path: Path) -> None:
    """A returned task can be scored without acquiring a second task's data."""
    from trialagentbench_verifier.assets import pack_release_assets, read_asset_catalog
    from trialagentbench_verifier.io import sha256_file

    release = _release(tmp_path / "release")
    selected_id = "te_" + "1" * 20
    other_id = "te_" + "2" * 20
    record = json.loads((release / RECORDS).read_text())
    record.update(task_id=selected_id, trial_id="trial_" + "1" * 20)
    outer = read_json_object(release / "release_manifest.json")
    outer.update(task_count=2, trial_count=2)
    outer.pop("checksum")
    outer["checksum"] = _canonical(outer)
    _write_json(release / "release_manifest.json", outer)
    _write_records(release, record, {**record, "task_id": other_id, "trial_id": "trial_" + "2" * 20})
    unrelated = release / f"participant/tasks/{other_id}/data.csv"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("patient,value\n2,9\n")
    role = read_json_object(release / "participant/manifest.json")
    role["artifacts"].append(_artifact(release, f"participant/tasks/{other_id}/data.csv"))
    _reseal_role(release, role="participant", payload=role)
    selected = release / f"participant/tasks/{selected_id}/data.csv"
    selected.parent.mkdir(parents=True)
    selected.write_text("patient,value\n1,1\n")
    role = read_json_object(release / "participant/manifest.json")
    role["artifacts"].append(_artifact(release, selected.relative_to(release).as_posix()))
    _reseal_role(release, role="participant", payload=role)
    role = read_json_object(release / "evaluation/manifest.json")
    role["artifacts"] = [_artifact(release, row["relative_path"]) for row in role["artifacts"]]
    _reseal_role(release, role="evaluation", payload=role)
    identity = {
        "schema_id": "ariadne.trialeval.participant_archive_identity/v1",
        "release_id": read_json_object(release / "release_manifest.json")["release_id"],
        "participant_archive_sha256": "0" * 64,
        "participant_manifest_checksum": read_json_object(release / "participant/manifest.json")["checksum"],
    }
    identity["checksum"] = _canonical(identity)
    _write_json(release / "participant_archive_identity.json", identity)
    submission = _submission(tmp_path / "submission.json")
    submission.write_text(submission.read_text().replace("task_1", selected_id))
    evidence = _execution_evidence(release, submission)
    archive = tmp_path / "release.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for path in release.rglob("*"):
            if path.is_file():
                bundle.write(path, path.relative_to(release).as_posix())
    _, catalog_path = pack_release_assets(archive, expected_sha256=sha256_file(archive), output=tmp_path / "assets")
    catalog = read_asset_catalog(catalog_path, expected_sha256=sha256_file(catalog_path))
    proof_path = evidence["execution_proof"]
    proof = read_json_object(proof_path)
    participant_digest = hashlib.sha256(catalog.participant_catalog().to_bytes()).hexdigest()
    proof.update(input_kind="archive", input_sha256=participant_digest, expected_input_sha256=participant_digest)
    _write_json(proof_path, proof)
    evidence["expected_execution_proof_sha256"] = sha256_file(proof_path)
    shard = next(row for row in catalog.shards if row.group == f"participant-{other_id}")
    (catalog_path.parent / shard.filename).rename(tmp_path / "unselected.zip")
    report = score_submission(catalog_path, submission, expected_sha256=sha256_file(catalog_path), **evidence)
    assert report.status == "pass"
    assert report.score_binding.task_id == selected_id


@pytest.mark.parametrize(("reference", "status"), [([[False, False]], "satisfied"), ([[True, True]], "not_satisfied")])
def test_declared_fixed_sequence_decision_is_independently_scored(
    tmp_path: Path, reference: list[list[bool]], status: str
) -> None:
    """A gated secondary is not rejected after a retained primary, whatever its own interval shows."""
    from copy import deepcopy

    release = _release(tmp_path / "release")
    record_path = release / RECORDS
    record = json.loads(record_path.read_text())
    extra = {
        key: deepcopy(record[key])
        for key in (
            "target",
            "oracle_worlds",
            "oracle_monte_carlo_standard_error",
            "maximum_oracle_monte_carlo_standard_error",
        )
    }
    extra["target"]["binding"].update(
        estimand_id="event_free_time", effect_measure="restricted_mean_survival_time_difference", result_unit="days"
    )
    extra["target"].update(value=4.0, reference_sampling_standard_deviation=2.0)
    record["additional_references"] = [extra]
    record["decision"] = {"kind": "fixed_sequence", "margin": 0.0, "reference_decisions": reference}
    _write_records(release, record)
    _reseal_key(release)
    primary = read_json_object(_submission(tmp_path / "paired.json"))["payload"]
    primary["target_id"] = "risk_difference"
    secondary = {key: deepcopy(value) for key, value in primary.items() if key not in {"schema_id", "task_id"}}
    secondary["target_id"] = "event_free_time"
    secondary["submitted_binding"] = extra["target"]["binding"]
    secondary["submitted_result"].update(
        value=4.0,
        effect_measure="restricted_mean_survival_time_difference",
        unit="days",
        interval={"lower": 2.0, "upper": 6.0, "confidence_level": 0.95},
    )
    primary["additional_inputs"] = [secondary]
    verified = verify_trialeval_outcome_v1(
        release_root=release, score_input=TrialEvalScoreInputV1.model_validate(primary)
    )
    assert verified.decision is not None and verified.decision.status == status
    record["decision"]["reference_decisions"] = [[True]]
    _write_records(release, record)
    with pytest.raises(ValueError, match="do not test the declared targets"):
        verify_trialeval_outcome_v1(release_root=release, score_input=TrialEvalScoreInputV1.model_validate(primary))


@pytest.mark.parametrize(("reference", "status"), [([[False]], "satisfied"), ([[True]], "not_satisfied")])
def test_unscored_point_and_superiority_decision_are_independently_scored(
    tmp_path: Path, reference: list[list[bool]], status: str
) -> None:
    """The unscored point is reported but only the interval and its efficacy decision score."""
    release = _release(tmp_path / "release")
    record_path = release / RECORDS
    record = json.loads(record_path.read_text())
    record["target"]["point_scored"] = False
    record["decision"] = {"kind": "superiority", "margin": 0.0, "reference_decisions": reference}
    _write_records(release, record)
    _reseal_key(release)
    payload = read_json_object(_submission(tmp_path / "submission.json"))["payload"]
    payload["submitted_result"]["value"] = 0.2
    verified = verify_trialeval_outcome_v1(
        release_root=release, score_input=TrialEvalScoreInputV1.model_validate(payload)
    )
    # Interval [-0.1, 0.3] covers the 0.1 target: interval half (0.4) / (2 x 0.05); the 0.1 point error is not scored.
    assert verified.signal_loss == pytest.approx(4.0)
    assert verified.metrics is not None and verified.metrics["native_absolute_error"] == pytest.approx(0.1)
    # The interval includes the null, so efficacy is not concluded.
    assert verified.decision is not None and verified.decision.status == status
    record["target"]["point_scored"] = True
    _write_records(release, record)
    _reseal_key(release)
    scored = verify_trialeval_outcome_v1(
        release_root=release, score_input=TrialEvalScoreInputV1.model_validate(payload)
    )
    assert scored.signal_loss == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("plan_row", "accurate"), (((0.12, -0.08, 0.32), True), ((0.1, -0.1, 0.3), False), (None, False))
)
def test_plan_as_executed_is_verified_apart_from_the_loss_and_decision(
    tmp_path: Path, plan_row: tuple[float, float, float] | None, accurate: bool
) -> None:
    """A plan-given task's plan as executed is checked for execution only."""
    from copy import deepcopy

    release = _release(tmp_path / "release")
    record_path = release / RECORDS
    record = json.loads(record_path.read_text())
    payload = read_json_object(_submission(tmp_path / "submission.json"))["payload"]
    record["decision"] = {"kind": "superiority", "margin": 0.0, "reference_decisions": [[False]]}
    _write_records(release, record)
    unplanned = verify_trialeval_outcome_v1(
        release_root=release, score_input=TrialEvalScoreInputV1.model_validate(payload)
    )
    record.update(
        context="C1",
        plan_replay={
            "target_id": "prespecified_primary_analysis",
            "estimate": 0.12,
            "lower": -0.08,
            "upper": 0.32,
            "tolerance_sd": 0.05,
        },
    )
    _write_records(release, record)
    if plan_row is not None:
        plan_input = {key: deepcopy(value) for key, value in payload.items() if key not in {"schema_id", "task_id"}}
        plan_input["target_id"] = "prespecified_primary_analysis"
        plan_input["submitted_result"].update(
            value=plan_row[0], interval={"lower": plan_row[1], "upper": plan_row[2], "confidence_level": 0.95}
        )
        payload["plan_input"] = plan_input
    verified = verify_trialeval_outcome_v1(
        release_root=release, score_input=TrialEvalScoreInputV1.model_validate(payload)
    )
    assert verified.plan_execution is not None and verified.plan_execution.accurate is accurate
    assert (verified.plan_execution.point_deviation is None) is (plan_row is None)
    # The acted-on answer alone sets the loss, the components and the decision.
    assert (verified.signal_loss, verified.components, verified.decision) == (
        unplanned.signal_loss,
        unplanned.components,
        unplanned.decision,
    )
    from trialagentbench_verifier.release import _score_trialeval

    checks = _score_trialeval(
        release,
        NormalizedSubmissionV1(
            benchmark_stream="trialeval", task_id="task_1", submission_kind="analysis", payload=payload
        ),
        execution_authenticated=True,
    )
    plan_check = next(check for check in checks if check.check_id == "score.plan_execution")
    assert plan_check.status == ("pass" if accurate else "fail")
    assert plan_check.observed == verified.plan_execution.model_dump(mode="json")
    record.update(context="C2")
    _write_records(release, record)
    with pytest.raises(ValueError, match="outside the plan lane"):
        verify_trialeval_outcome_v1(release_root=release, score_input=TrialEvalScoreInputV1.model_validate(payload))


def test_plan_as_executed_is_part_of_the_proof_projection(tmp_path: Path) -> None:
    """The replay proof hashes every CSV row; a plan-given submission's projection includes its plan row."""
    from copy import deepcopy

    from trialagentbench_verifier.numerical import NumericalEstimate
    from trialagentbench_verifier.proof import numerical_score_projection_sha256

    normalized = NormalizedSubmissionV1.model_validate(read_json_object(_submission(tmp_path / "s.json")))
    payload = deepcopy(normalized.payload)
    payload["target_id"] = "risk_difference"
    plan = {key: deepcopy(value) for key, value in payload.items() if key not in {"schema_id", "task_id"}}
    plan["target_id"] = "prespecified_primary_analysis"
    plan["submitted_result"]["value"] = 0.2
    payload["plan_input"] = plan
    planned = normalized.model_copy(update={"payload": payload})
    rows = tuple(
        NumericalEstimate(
            target_id=item["target_id"],
            estimate=item["submitted_result"]["value"],
            lower=item["submitted_result"]["interval"]["lower"],
            upper=item["submitted_result"]["interval"]["upper"],
        )
        for item in (payload, plan)
    )
    evidence = tuple(payload["evidence_source_artifacts"])
    assert submission_score_projection_sha256(planned) == numerical_score_projection_sha256(rows, (evidence, evidence))
    assert submission_score_projection_sha256(planned) != submission_score_projection_sha256(normalized)


def test_key_release_verifies_and_scores_exactly_as_the_operator_release(tmp_path: Path) -> None:
    """The public key layout and the operator layout grade from the same key bytes."""

    operator = _release(tmp_path / "operator")
    key = _release(tmp_path / "key", operator=False)
    assert not (key / "evaluation" / "manifest.json").exists()
    assert sorted(path.relative_to(key).as_posix() for path in (key / "evaluation").rglob("*") if path.is_file()) == [
        "evaluation/verification/qualification_reports.json",
        "evaluation/verification/reports/G-EXT-01.json",
        "evaluation/verification/reports/G-TAB-01.json",
        "evaluation/verification/reports/X-19.json",
    ]
    reports = {name: verify_release(root) for name, root in (("operator", operator), ("key", key))}
    assert all(report.status == "pass" for report in reports.values())
    # The key layout accepts the cited private qualification file as committed by the release manifest.
    assert reports["operator"].model_dump(exclude={"checks"}) == reports["key"].model_dump(exclude={"checks"})
    scored = {
        name: score_submission(root, _submission(tmp_path / f"{name}.json"))
        for name, root in (("operator", operator), ("key", key))
    }
    assert scored["operator"] == scored["key"]
    assert {check.artifact_paths for check in scored["key"].checks} == {(RECORDS,)}


@pytest.mark.parametrize("operation", ("byte_mutation", "omission"))
def test_every_grading_key_file_is_tamper_evident(tmp_path: Path, operation: str) -> None:
    """Reject a changed or missing key manifest, key record file or retained report in a key release."""

    source = _release(tmp_path / "source", operator=False)
    inventory = json.loads((source / KEY_MANIFEST).read_text(encoding="utf-8"))["artifacts"]
    for index, relative in enumerate((KEY_MANIFEST, *(str(row["relative_path"]) for row in inventory))):
        release = tmp_path / f"{operation}-{index}"
        shutil.copytree(source, release)
        target = release / relative
        if operation == "omission":
            target.unlink()
        else:
            target.write_bytes(b"0" + target.read_bytes())
        report = verify_release(release)
        assert report.status == "fail", relative
        assert report.checks[0].failure_code == "release_integrity_failure", relative


@pytest.mark.parametrize("directory", ("evaluation", "grading_key"))
def test_key_release_rejects_undeclared_evaluator_files(tmp_path: Path, directory: str) -> None:
    """A key release holds only the files its key inventories beside the participant role."""

    release = _release(tmp_path / "release", operator=False)
    _write_json(release / directory / "construct.json", {"generation": {}})
    assert verify_release(release).checks[0].failure_code == "release_integrity_failure"


def test_grading_key_is_bound_to_its_release_manifest(tmp_path: Path) -> None:
    """A resealed key that names another release manifest, or census, is rejected."""

    for field, value in (("release_manifest_sha256", "f" * 64), ("cell_count", 2), ("task_count", 2)):
        release = _release(tmp_path / field, operator=False)
        manifest = read_json_object(release / KEY_MANIFEST)
        manifest[field] = value
        manifest.pop("checksum")
        manifest["checksum"] = _canonical(manifest)
        _write_json(release / KEY_MANIFEST, manifest)
        report = verify_release(release)
        assert report.status == "fail", field
        assert report.checks[0].failure_code == "release_integrity_failure", field


def test_grading_key_records_must_be_canonical_lines(tmp_path: Path) -> None:
    """Key records have one byte form, so equal keys are equal files."""

    release = _release(tmp_path / "release", operator=False)
    record = read_json_object(release / RECORDS)
    (release / RECORDS).write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    manifest = read_json_object(release / KEY_MANIFEST)
    manifest["artifacts"] = [_artifact(release, str(row["relative_path"])) for row in manifest["artifacts"]]
    manifest.pop("checksum")
    manifest["checksum"] = _canonical(manifest)
    _write_json(release / KEY_MANIFEST, manifest)
    with pytest.raises(VerificationInputError, match="not canonical|blank record|invalid UTF-8 JSONL"):
        verify_trialeval_outcome_v1(
            release_root=release,
            score_input=TrialEvalScoreInputV1.model_validate(
                read_json_object(_submission(tmp_path / "submission.json"))["payload"]
            ),
        )


def test_release_without_evaluation_role_or_grading_key_is_rejected(tmp_path: Path) -> None:
    """A participant role with neither evaluator surface cannot be verified as an evaluator release."""

    release = _release(tmp_path / "release", operator=False)
    shutil.rmtree(release / "grading_key")
    report = verify_release(release)
    assert report.status == "fail"
    assert "neither an evaluation role nor a grading key" in str(report.checks[0].observed)
