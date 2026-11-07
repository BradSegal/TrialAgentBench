"""Tests for the portable release result exporter."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import JsonValue
from trialagentbench_verifier.release_set import verify_result_export

from trialagentbench.contracts.release.identity import PairedParticipantArchiveIdentityV1
from trialagentbench.contracts.release.result_export import (
    ResultExportBundleManifestV1,
)
from trialagentbench.io import canonical_payload_sha256
from trialagentbench.tools import export_results
from trialagentbench.tools.export_results import (
    _analysis_reproduction_files,
    _projected_json_bytes,
    export_results_v1,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _release(root: Path) -> Path:
    archive = Path(__file__).parents[2] / "examples" / "clinical" / "demo-trialdev-evaluator.zip"
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(root)
    return root


def _archive_name(release: Path) -> str:
    return f"{json.loads((release / 'release_manifest.json').read_text())['release_id']}-results.zip"


def _trialdev_release(root: Path) -> Path:
    _write_json(root / "release_manifest.json", {"schema_id": "ariadne.trialdev.release/v1"})
    return root


@pytest.mark.parametrize("stage", ["candidate_single_seed", "paired_release"])
def test_result_export_refuses_superseded_bundle_stages(tmp_path: Path, stage: str) -> None:
    """Both the harness contract and independent verifier require the publication stage."""
    release = _release(tmp_path / "release")
    run, grade, verification, _ = _artifacts(tmp_path / "artifacts", release)
    output = export_results_v1(
        release_root=release,
        output_dir=tmp_path / "output",
        run_roots=[run],
        grade_roots=[grade],
        verification_roots=[verification],
    )
    with zipfile.ZipFile(output / _archive_name(release)) as archive:
        members = {name: archive.read(name) for name in archive.namelist() if name != "SHA256SUMS"}
    manifest = json.loads(members["manifest.json"])
    manifest["release_stage"] = stage
    manifest.pop("checksum")
    manifest["checksum"] = canonical_payload_sha256(manifest)
    with pytest.raises(ValueError, match="release_stage"):
        ResultExportBundleManifestV1.model_validate(manifest)
    members["manifest.json"] = json.dumps(manifest).encode()
    checksums = "".join(f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in sorted(members.items()))
    altered = tmp_path / "altered.zip"
    with zipfile.ZipFile(altered, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
        archive.writestr("SHA256SUMS", checksums)
    with pytest.raises(ValueError, match="release_stage"):
        verify_result_export(altered, required_profile="full_custody")


def test_result_export_requires_current_role_manifests(tmp_path: Path) -> None:
    """A release without the required role manifests cannot authorize export."""
    release = tmp_path / "release"
    contract = release / "export.json"
    _write_json(release / "schema.json", {"type": "object"})
    _write_json(release / "properties.json", {})
    _write_json(
        contract,
        {
            "schema_id": "ariadne.export_results_manifest/v1",
            "release_id": "incomplete-release",
            "status": "awaiting_model_runs",
            "canonical_result_schema_path": "schema.json",
            "simulation_properties_path": "properties.json",
            "expected_analysis_unit_count": 1,
            "join_key": "canonical_result_join_key",
            "denominator_policy": "retain_every_scheduled_assignment_with_typed_status",
            "result_export_command": "trialagentbench export results",
            "required_artifact_kinds": ["run", "grade", "verification"],
        },
    )
    _write_json(
        release / "RELEASE_MANIFEST.json",
        {
            "release_id": "incomplete-release",
            "release_stage": "benchmark_publication",
            "result_export_manifest": {"path": "export.json", "sha256": _sha256(contract)},
        },
    )
    run, grade, verification, _ = _artifacts(tmp_path / "artifacts")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="release_manifest.json|unsupported benchmark manifest"):
        export_results._release_contract(release)
    with pytest.raises(ValueError, match="release_manifest.json|unsupported benchmark manifest"):
        export_results_v1(
            release_root=release,
            output_dir=output,
            run_roots=[run],
            grade_roots=[grade],
            verification_roots=[verification],
        )
    assert not output.exists()


def _artifacts(root: Path, release: Path | None = None) -> tuple[Path, Path, Path, Path]:
    paths = (root / "run", root / "grade", root / "verification", root / "analysis")
    for index, path in enumerate(paths):
        (path / "nested").mkdir(parents=True)
        (path / "nested" / "record.json").write_text(f'{{"value": {index}}}\n', encoding="utf-8")
    if release is not None:
        _bind_trialdev_results(
            release=release,
            release_id=json.loads((release / "release_manifest.json").read_text())["release_id"],
            run=paths[0],
            grade=paths[1],
            verification=paths[2],
        )
    return paths


def _bind_trialdev_results(
    *,
    release: Path,
    release_id: str,
    run: Path,
    grade: Path,
    verification: Path,
) -> None:
    manifest_sha256 = _sha256(release / "release_manifest.json")
    _write_json(
        run / "invocation.json",
        {
            "release_id": release_id,
            "release_manifest_sha256": manifest_sha256,
        },
    )
    _write_json(
        grade / "tasks" / "task-001" / "run.json",
        {"release_manifest_sha256": manifest_sha256},
    )
    _write_json(
        verification / "verification_report.json",
        {
            "schema_id": "trialagentbench.verification_report/v1",
            "release_digest": manifest_sha256,
        },
    )


def _bind_trialeval_results(
    *,
    release: Path,
    run: Path,
    grade: Path,
    verification: Path,
) -> None:
    manifest_sha256 = _sha256(release / "release_manifest.json")
    _write_json(
        run / "run_config.json",
        {"participant_manifest_checksum": "2" * 64},
    )
    _write_json(
        grade / "GRADE_MANIFEST.json",
        {"release_manifest_sha256": manifest_sha256},
    )
    _write_json(
        verification / "verification_report.json",
        {
            "schema_id": "trialagentbench.verification_report/v1",
            "release_digest": manifest_sha256,
        },
    )


def test_result_export_rejects_unsupported_manifest_filename(tmp_path: Path) -> None:
    """Reject an export directory containing an unsupported manifest filename."""

    release = _release(tmp_path / "release")
    output = tmp_path / "output"
    output.mkdir()
    (output / "result_export_receipt.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="choose a new output directory"):
        export_results_v1(
            release_root=release,
            output_dir=output,
            run_roots=(),
            grade_roots=(),
            verification_roots=(),
        )


def test_result_export_is_deterministic_and_self_verifying(tmp_path: Path) -> None:
    """Equivalent inputs must produce an identical portable archive."""

    release = _release(tmp_path / "release")
    run, grade, verification, analysis = _artifacts(tmp_path / "artifacts", release)
    first = export_results_v1(
        release_root=release,
        output_dir=tmp_path / "first",
        run_roots=[run],
        grade_roots=[grade],
        verification_roots=[verification],
        analysis_roots=[analysis],
    )
    second = export_results_v1(
        release_root=release,
        output_dir=tmp_path / "second",
        run_roots=[run],
        grade_roots=[grade],
        verification_roots=[verification],
        analysis_roots=[analysis],
    )

    archive_name = _archive_name(release)
    first_archive = first / archive_name
    second_archive = second / archive_name
    assert first_archive.read_bytes() == second_archive.read_bytes()
    assert (first / f"{archive_name}.sha256").read_text(encoding="utf-8") == (
        f"{_sha256(first_archive)}  {archive_name}\n"
    )

    manifest = ResultExportBundleManifestV1.model_validate_json(
        (first / "result_export_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.release_id == json.loads((release / "release_manifest.json").read_text())["release_id"]
    assert {source.kind for source in manifest.sources} == {
        "run",
        "grade",
        "verification",
        "analysis",
    }
    assert manifest.release_contract_sha256 == json.loads((release / "release_manifest.json").read_text())["checksum"]
    serialized = json.dumps(manifest.model_dump(mode="json"))
    assert str(tmp_path) not in serialized

    with zipfile.ZipFile(first_archive) as archive:
        assert archive.read("manifest.json") == (first / "result_export_manifest.json").read_bytes()
        run_source = next(source for source in manifest.sources if source.kind == "run")
        assert (
            archive.read(f"{run_source.archive_prefix}/nested/record.json")
            == (run / "nested/record.json").read_bytes()
        )
        checksums = {
            line.split("  ", maxsplit=1)[1]: line.split("  ", maxsplit=1)[0]
            for line in archive.read("SHA256SUMS").decode().splitlines()
        }
        assert checksums["manifest.json"] == hashlib.sha256(archive.read("manifest.json")).hexdigest()
        for member in manifest.members:
            assert checksums[member.path] == member.sha256
            assert hashlib.sha256(archive.read(member.path)).hexdigest() == member.sha256


def test_full_custody_preserves_metadata_and_projection_remains_separate(tmp_path: Path) -> None:
    """Raw custody retains original bytes; analysis projection replaces host paths."""

    release = _release(tmp_path / "release")
    run, grade, verification, _ = _artifacts(tmp_path / "artifacts", release)
    _write_json(
        run / "run_config.json",
        {
            "participant_archive": str(tmp_path / "private" / "trialeval-participant.zip"),
            "nested": {"input_run_dir": str(tmp_path / "private" / "compiled-run")},
        },
    )
    output = export_results_v1(
        release_root=release,
        output_dir=tmp_path / "output",
        run_roots=[run],
        grade_roots=[grade],
        verification_roots=[verification],
    )

    with zipfile.ZipFile(output / _archive_name(release)) as archive:
        member = next(name for name in archive.namelist() if name.endswith("run_config.json"))
        assert archive.read(member) == (run / "run_config.json").read_bytes()
    projected = _projected_json_bytes(run / "run_config.json")
    assert projected is not None
    payload = json.loads(projected)

    assert payload["participant_archive"] == "trialeval-participant.zip"
    assert payload["nested"]["input_run_dir"] == "compiled-run"
    assert str(tmp_path) not in json.dumps(payload)


def test_full_custody_preserves_jsonl_and_projection_remains_separate(tmp_path: Path) -> None:
    """Raw event identity survives export and has an explicit portable projection."""

    release = _release(tmp_path / "release")
    run, grade, verification, _ = _artifacts(tmp_path / "artifacts", release)
    event_path = run / "events.jsonl"
    event_path.write_text(
        json.dumps(
            {
                "event_type": "prompt",
                "source_artifact_path": "build/private/conversation.json",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = export_results_v1(
        release_root=release,
        output_dir=tmp_path / "output",
        run_roots=[run],
        grade_roots=[grade],
        verification_roots=[verification],
    )

    with zipfile.ZipFile(output / _archive_name(release)) as archive:
        member = next(name for name in archive.namelist() if name.endswith("events.jsonl"))
        assert archive.read(member) == event_path.read_bytes()
    projected = _projected_json_bytes(event_path)
    assert projected is not None
    payload = json.loads(projected)

    assert payload["source_artifact_path"] == "conversation.json"
    assert str(tmp_path) not in json.dumps(payload)


def test_result_export_distinguishes_parallel_roots_with_same_name(tmp_path: Path) -> None:
    """Parallel timestamped runs retain distinct, location-independent custody."""
    release = _release(tmp_path / "release")
    run, grade, verification, _ = _artifacts(tmp_path / "first", release)
    other, _, _, _ = _artifacts(tmp_path / "second", release)
    (other / "nested" / "record.json").write_text('{"value": 10}\n', encoding="utf-8")
    result = export_results_v1(
        release_root=release,
        output_dir=tmp_path / "export",
        run_roots=[run, other],
        grade_roots=[grade],
        verification_roots=[verification],
    )
    manifest = ResultExportBundleManifestV1.model_validate_json(
        (result / "result_export_manifest.json").read_text(encoding="utf-8")
    )
    sources = [source for source in manifest.sources if source.kind == "run"]
    assert len({source.label for source in sources}) == 2
    with zipfile.ZipFile(result / _archive_name(release)) as archive:
        assert {
            json.loads(archive.read(f"{source.archive_prefix}/nested/record.json"))["value"] for source in sources
        } == {0, 10}
    moved = tmp_path / "relocated"
    shutil.copytree(tmp_path / "first", moved / "first")
    shutil.copytree(tmp_path / "second", moved / "second")
    repeated = export_results_v1(
        release_root=release,
        output_dir=tmp_path / "repeated",
        run_roots=[moved / "first" / "run", moved / "second" / "run"],
        grade_roots=[moved / "first" / "grade"],
        verification_roots=[moved / "first" / "verification"],
    )
    assert (repeated / _archive_name(release)).read_bytes() == (result / _archive_name(release)).read_bytes()
    with pytest.raises(ValueError, match="duplicate result artifact label"):
        export_results_v1(
            release_root=release,
            output_dir=tmp_path / "duplicate",
            run_roots=[run, run],
            grade_roots=[grade],
            verification_roots=[verification],
        )


@pytest.mark.parametrize("layout", ["direct", "tasks"])
def test_result_export_supports_trialdev_publication_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
) -> None:
    """TrialDev results must bind the canonical publication manifest."""

    release = _trialdev_release(tmp_path / "release")
    release_id = f"release_{'1' * 64}"
    monkeypatch.setattr(
        "trialagentbench.tools.export_results.load_trialdev_evaluator_release",
        lambda _: SimpleNamespace(manifest=SimpleNamespace(release_id=release_id, checksum="2" * 64)),
    )
    run, grade, verification, analysis = _artifacts(tmp_path / "artifacts")
    _bind_trialdev_results(
        release=release,
        release_id=release_id,
        run=run,
        grade=grade,
        verification=verification,
    )
    if layout == "direct":
        (grade / "tasks/task-001").rename(grade / "task-001")
    output = export_results_v1(
        release_root=release,
        output_dir=tmp_path / "output",
        run_roots=[run],
        grade_roots=[grade],
        verification_roots=[verification],
        analysis_roots=[analysis],
    )

    manifest = ResultExportBundleManifestV1.model_validate_json(
        (output / "result_export_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.release_id == release_id
    assert manifest.release_stage == "benchmark_publication"
    assert manifest.release_manifest_sha256 == _sha256(release / "release_manifest.json")
    assert manifest.release_contract_sha256 == "2" * 64


@pytest.mark.parametrize("layout", ["operator", "key"])
@pytest.mark.parametrize("paired_role", ["absent", "matching", "mismatched"])
def test_result_export_supports_current_trialeval_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    paired_role: str,
    layout: str,
) -> None:
    """Role-separated archives bind result exports."""

    release = tmp_path / "release"
    participant = release / "participant"
    evaluation = release / "evaluation"
    participant.mkdir(parents=True)
    evaluation.mkdir()
    (participant / "manifest.json").write_text("{}\n", encoding="utf-8")
    (release / "grading_key").mkdir()
    (release / "grading_key" / "manifest.json").write_text("{}\n", encoding="utf-8")
    if layout == "operator":
        _write_json(evaluation / "qualification.json", {"checksum": "4" * 64})
        (evaluation / "manifest.json").write_text("{}\n", encoding="utf-8")
    else:
        (evaluation / "verification").mkdir()
        (evaluation / "verification" / "qualification_reports.json").write_text("{}\n", encoding="utf-8")
    release_id = f"release_{'1' * 64}"
    manifest_payload: dict[str, JsonValue] = {
        "schema_id": "ariadne.trialeval.release/v1",
        "data_provenance": "generated_under_declared_trial_mechanisms",
        "release_id": release_id,
        "construct_sha256": "1" * 64,
        "qualification_sha256": "4" * 64,
        "task_count": 1,
        "participant_manifest_sha256": "2" * 64,
        "evaluation_manifest_sha256": "3" * 64,
        "trial_count": 1,
    }
    manifest_payload["checksum"] = canonical_payload_sha256(manifest_payload)
    _write_json(release / "release_manifest.json", manifest_payload)
    if paired_role != "absent":
        paired = PairedParticipantArchiveIdentityV1.create(
            release_id=release_id,
            participant_archive_sha256="a" * 64,
            participant_manifest_checksum=("2" if paired_role == "matching" else "9") * 64,
        )
        (release / "participant_archive_identity.json").write_text(paired.model_dump_json())
    monkeypatch.setattr(
        "trialagentbench.tools.export_results.load_participant_release",
        lambda root: SimpleNamespace(
            root=root,
            manifest=SimpleNamespace(release_id=release_id, checksum="2" * 64),
        ),
    )
    monkeypatch.setattr(
        "trialagentbench.tools.export_results.load_evaluation_release",
        lambda root: SimpleNamespace(
            root=root,
            manifest=SimpleNamespace(release_id=release_id, checksum="3" * 64),
        ),
    )
    monkeypatch.setattr(
        "trialagentbench.tools.export_results.load_trialeval_grading_key",
        lambda root: SimpleNamespace(root=root, manifest=SimpleNamespace(release_id=release_id)),
    )
    run, grade, verification, _ = _artifacts(tmp_path / "artifacts")
    _bind_trialeval_results(
        release=release,
        run=run,
        grade=grade,
        verification=verification,
    )

    if paired_role == "mismatched":
        with pytest.raises(ValueError, match="paired participant identity differs"):
            export_results_v1(
                release_root=release,
                output_dir=tmp_path / "rejected-output",
                run_roots=[run],
                grade_roots=[grade],
                verification_roots=[verification],
            )
        return

    output = export_results_v1(
        release_root=release,
        output_dir=tmp_path / "output",
        run_roots=[run],
        grade_roots=[grade],
        verification_roots=[verification],
    )

    manifest = ResultExportBundleManifestV1.model_validate_json(
        (output / "result_export_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.release_id == release_id
    assert manifest.release_stage == "benchmark_publication"
    assert manifest.release_contract_sha256 == manifest_payload["checksum"]


def test_result_export_rejects_result_identity_from_another_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Required result artifacts may not be relabelled for another release."""

    release = _trialdev_release(tmp_path / "release")
    release_id = f"release_{'1' * 64}"
    monkeypatch.setattr(
        "trialagentbench.tools.export_results.load_trialdev_evaluator_release",
        lambda _: SimpleNamespace(manifest=SimpleNamespace(release_id=release_id, checksum="2" * 64)),
    )
    run, grade, verification, _ = _artifacts(tmp_path / "artifacts")
    _bind_trialdev_results(
        release=release,
        release_id=release_id,
        run=run,
        grade=grade,
        verification=verification,
    )
    _write_json(run / "invocation.json", {"release_id": release_id, "release_manifest_sha256": "f" * 64})

    with pytest.raises(ValueError, match="different released bytes"):
        export_results_v1(
            release_root=release,
            output_dir=tmp_path / "output",
            run_roots=[run],
            grade_roots=[grade],
            verification_roots=[verification],
        )


def test_result_export_rejects_trialdev_artifact_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A publication member changed after manifest creation must be rejected."""

    release = _trialdev_release(tmp_path / "release")
    run, grade, verification, _ = _artifacts(tmp_path / "artifacts")

    def reject_changed_release(_: Path) -> None:
        raise ValueError("TrialDev evaluator artifact changed")

    monkeypatch.setattr(
        "trialagentbench.tools.export_results.load_trialdev_evaluator_release",
        reject_changed_release,
    )

    with pytest.raises(ValueError, match="artifact changed"):
        export_results_v1(
            release_root=release,
            output_dir=tmp_path / "output",
            run_roots=[run],
            grade_roots=[grade],
            verification_roots=[verification],
        )


def test_result_export_rejects_release_manifest_mutation(tmp_path: Path) -> None:
    """The release manifest must retain its content checksum."""

    release = _release(tmp_path / "release")
    run, grade, verification, _ = _artifacts(tmp_path / "artifacts", release)
    manifest = json.loads((release / "release_manifest.json").read_text())
    manifest["checksum"] = "0" * 64
    _write_json(release / "release_manifest.json", manifest)

    with pytest.raises(ValueError, match="checksum"):
        export_results_v1(
            release_root=release,
            output_dir=tmp_path / "output",
            run_roots=[run],
            grade_roots=[grade],
            verification_roots=[verification],
        )


@pytest.mark.parametrize("unsafe_name", [".env", "private.pem", "cached.pyc"])
def test_result_export_rejects_credentials_and_transient_files(
    tmp_path: Path,
    unsafe_name: str,
) -> None:
    """Credential-shaped and transient files must never enter result bundles."""

    release = _release(tmp_path / "release")
    run, grade, verification, _ = _artifacts(tmp_path / "artifacts", release)
    (run / unsafe_name).write_text("not-a-secret\n", encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden credential or transient"):
        export_results_v1(
            release_root=release,
            output_dir=tmp_path / "output",
            run_roots=[run],
            grade_roots=[grade],
            verification_roots=[verification],
        )


@pytest.mark.parametrize("unsafe_name", [".env", "private.pem", "__pycache__/cached.pyc"])
def test_participant_scratch_names_do_not_block_result_export(tmp_path: Path, unsafe_name: str) -> None:
    """An agent's own scratch file names are participant content, not operator credentials."""

    release = _release(tmp_path / "release")
    run, grade, verification, _ = _artifacts(tmp_path / "artifacts", release)
    planted = run / "logs" / "te_example_workspace" / "item" / "scratch" / unsafe_name
    planted.parent.mkdir(parents=True)
    planted.write_text("participant-written\n", encoding="utf-8")

    bundle = export_results_v1(
        release_root=release,
        output_dir=tmp_path / "output",
        run_roots=[run],
        grade_roots=[grade],
        verification_roots=[verification],
    )
    assert bundle.exists()


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
def test_result_export_rejects_records_containing_a_provider_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    """A run record that captured the exporter's provider key never enters a bundle."""

    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-test-key-123")
    release = _release(tmp_path / "release")
    run, grade, verification, _ = _artifacts(tmp_path / "artifacts", release)
    (run / f"provider_responses{suffix}").write_text(
        '{"error_detail": "401: Authorization: Bearer dummy-test-key-123"}\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="provider credential"):
        export_results_v1(
            release_root=release,
            output_dir=tmp_path / "output",
            run_roots=[run],
            grade_roots=[grade],
            verification_roots=[verification],
        )


def test_result_export_preserves_immutable_release_boundary(tmp_path: Path) -> None:
    """Neither caller results nor export output may be placed in the release."""

    release = _release(tmp_path / "release")
    run, grade, verification, _ = _artifacts(tmp_path / "artifacts", release)
    release_result = release / "evaluation"

    with pytest.raises(ValueError, match="outside the immutable release"):
        export_results_v1(
            release_root=release,
            output_dir=tmp_path / "output",
            run_roots=[release_result],
            grade_roots=[grade],
            verification_roots=[verification],
        )
    with pytest.raises(ValueError, match="outside the immutable release"):
        export_results_v1(
            release_root=release,
            output_dir=release / "output",
            run_roots=[run],
            grade_roots=[grade],
            verification_roots=[verification],
        )


@pytest.mark.parametrize("prefix", ["", "tasks/"])
@pytest.mark.parametrize("kind", ["run", "grade"])
def test_compact_export_retains_authored_results_without_raw_workspaces(tmp_path: Path, prefix: str, kind) -> None:
    retained = {"invocation.json", "run_index.json", "ranking.json"} | {
        f"{prefix}td-0123456789ab/{name}" for name in ("run.json", "programme_run.json", "analysis_run.json")
    }
    omitted = {
        f"{prefix}td-0123456789ab/workspace/evidence/study/participants.parquet",
        f"{prefix}td-0123456789ab/private/budget.json",
        "events.jsonl",
        "scratch/run.json",
    }
    for relative in retained | omitted:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    selected = _analysis_reproduction_files(
        kind, tmp_path, tuple(path for path in tmp_path.rglob("*") if path.is_file())
    )
    assert {path.relative_to(tmp_path).as_posix() for path in selected} == retained


def test_result_export_refuses_case_only_member_clashes(tmp_path: Path) -> None:
    """Analysis.py and analysis.py would overwrite each other on a case-insensitive filesystem."""

    release = _release(tmp_path / "release")
    run, grade, verification, _ = _artifacts(tmp_path / "artifacts", release)
    (run / "nested" / "Analysis.py").write_text("print(1)\n", encoding="utf-8")
    (run / "nested" / "analysis.py").write_text("print(2)\n", encoding="utf-8")
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="case-insensitive"):
        export_results_v1(
            release_root=release,
            output_dir=output,
            run_roots=[run],
            grade_roots=[grade],
            verification_roots=[verification],
        )
    assert not output.exists()
