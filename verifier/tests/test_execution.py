"""Public offline-execution boundary tests."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import sysconfig
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from trialagentbench_verifier import NormalizedSubmissionV1
from trialagentbench_verifier import execution as execution_module
from trialagentbench_verifier.cli import main
from trialagentbench_verifier.contracts import DiagnosticArtifactExecutionV1, ExecutionProofV1
from trialagentbench_verifier.execution import (
    ExecutionSubmissionV1,
    _stage_task_inputs,
    _task,
    execute_submission,
    execute_submissions,
)
from trialagentbench_verifier.execution_contracts import (
    RELEASE_EXECUTION_LIMITS_V1,
    ExecutorEnvironmentV1,
    ReportingInterfaceV1,
)
from trialagentbench_verifier.io import VerificationInputError
from trialagentbench_verifier.proof import authenticate_execution_proof


def _canonical(payload: object) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(body).hexdigest()


_FAKE_EXECUTOR_IMAGE_ID = "sha256:" + "d" * 64
_FAKE_EXECUTOR_ENVIRONMENT_SHA256 = _canonical(
    {"image_reference": "executor:test", "image_id": _FAKE_EXECUTOR_IMAGE_ID}
)


@pytest.fixture(scope="module")
def live_executor() -> ExecutorEnvironmentV1:
    """Bind container tests to the executor built on this host, without fixed host IDs."""
    from trialagentbench_verifier.executor import resolve_executor_environment

    return resolve_executor_environment(limits=RELEASE_EXECUTION_LIMITS_V1)


@pytest.mark.executor
def test_live_numerical_replay_cannot_read_its_captured_answer(
    tmp_path: Path, live_executor: ExecutorEnvironmentV1
) -> None:
    """A captured answer cannot substitute for executing the declared analysis."""
    from trialagentbench_verifier.execution_contracts import RELEASE_EXECUTION_LIMITS_V1
    from trialagentbench_verifier.numerical import replay_numerical_program

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    capture = tmp_path / "analysis_sources" / "submission-0001"
    capture.mkdir(parents=True)
    result = "target_id,estimate,lower,upper\neffect,1,0,2\n"
    (capture / "estimates.csv").write_text(result)
    (scratch / "estimates.csv").write_text(result)
    program = scratch / "analysis.py"
    program.write_text(
        "from pathlib import Path\nPath('scratch/estimates.csv').write_text("
        "Path('analysis_sources/submission-0001/estimates.csv').read_text())\n"
    )
    rejected = replay_numerical_program(
        workspace=tmp_path,
        entrypoint="analysis.py",
        target_ids=("effect",),
        evidence_artifacts=(),
        executor_image=live_executor.image_id,
        executor_limits=RELEASE_EXECUTION_LIMITS_V1,
    )
    assert rejected.status == "execution_error"
    assert not rejected.reproducible
    program.write_text(f"from pathlib import Path\nPath('scratch/estimates.csv').write_text({result!r})\n")
    recovered = replay_numerical_program(
        workspace=tmp_path,
        entrypoint="analysis.py",
        target_ids=("effect",),
        evidence_artifacts=(),
        executor_image=live_executor.image_id,
        executor_limits=RELEASE_EXECUTION_LIMITS_V1,
    )
    assert recovered.reproducible
    assert len(recovered.replayed_csv_sha256s) == 2
    assert (capture / "estimates.csv").read_text() == result


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _participant_release(
    root: Path, *, target_ids: tuple[str, ...] = ("effect",), analysis_bytes: bytes = b"practice-input\n"
) -> Path:
    candidate = "a" * 64
    release_id = f"release_{candidate}"
    task_id = "te_fixture"
    task = {
        "schema_id": "ariadne.trialeval.task/v1",
        "task_id": task_id,
        "target_ids": list(target_ids),
        "available_materials": [
            {
                "artifact_id": "analysis_data",
                "description": "Disclosed practice input.",
                "relative_path": f"tasks/{task_id}/data/analysis.parquet",
            }
        ],
    }
    _write_json(root / "tasks" / task_id / "task.json", task)
    data = root / "tasks" / task_id / "data" / "analysis.parquet"
    data.parent.mkdir(parents=True, exist_ok=True)
    data.write_bytes(analysis_bytes)
    artifacts = []
    for relative in (f"tasks/{task_id}/task.json", f"tasks/{task_id}/data/analysis.parquet"):
        path = root / relative
        artifacts.append(
            {
                "relative_path": f"participant/{relative}",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "byte_length": path.stat().st_size,
            }
        )
    manifest = {
        "schema_id": "ariadne.benchmark_role_manifest/v1",
        "release_id": release_id,
        "role": "participant",
        "trial_count": 1,
        "task_count": 1,
        "artifacts": artifacts,
    }
    manifest["checksum"] = _canonical(manifest)
    _write_json(root / "manifest.json", manifest)
    outer = {
        "schema_id": "ariadne.trialeval.release/v1",
        "data_provenance": "generated_under_declared_trial_mechanisms",
        "release_id": release_id,
        "construct_sha256": candidate,
        "qualification_sha256": "b" * 64,
        "trial_count": 1,
        "task_count": 1,
        "participant_manifest_sha256": manifest["checksum"],
        "evaluation_manifest_sha256": "c" * 64,
    }
    outer["checksum"] = _canonical(outer)
    _write_json(root / "release_manifest.json", outer)
    return root


def _submission(root: Path, *, language: str = "python") -> Path:
    root.mkdir()
    name = "analysis.py" if language == "python" else "analysis.R"
    (root / name).write_text("# submitted practice analysis\n", encoding="utf-8")
    _write_json(
        root / "analysis_result.json",
        {
            "schema_id": "trialagentbench.trialeval_program_result/v2",
            "result": {
                "kind": "point",
                "value": 0.25,
                "effect_measure": "risk_difference",
                "unit": "proportion",
            },
            "source_artifacts": ["data/analysis.parquet"],
            "conclusion": "favors_treatment",
        },
    )
    (root / "estimates.csv").write_text("target_id,estimate,lower,upper\neffect,0.25,0.1,0.4\n")
    return root


def _executable_submission(root: Path, language: str, *, submitted_value: float = 0.25) -> Path:
    root.mkdir()
    if language == "python":
        (root / "analysis.py").write_text(
            "from pathlib import Path\nimport csv\n"
            "assert Path('data/analysis.parquet').read_bytes() == b'practice-input\\n'\n"
            "with open('scratch/estimates.csv','w',newline='') as handle:\n"
            "    w=csv.writer(handle)\n    w.writerow(['target_id','estimate','lower','upper'])\n"
            "    w.writerow(['effect',0.25,0.1,0.4])\n"
        )
    else:
        (root / "analysis.R").write_text(
            "stopifnot(file.info('data/analysis.parquet')$size == 15)\n"
            "write.csv(data.frame(target_id='effect',estimate=0.25,lower=0.1,upper=0.4),"
            "'scratch/estimates.csv',row.names=FALSE)\n"
        )
    (root / "estimates.csv").write_text(f"target_id,estimate,lower,upper\neffect,{submitted_value},0.1,0.4\n")
    return root


def _install_fake_executor(
    monkeypatch: pytest.MonkeyPatch, *, expected_reporting_interface: ReportingInterfaceV1 = "v1"
) -> None:
    from trialagentbench_verifier import executor as docker_code_execution
    from trialagentbench_verifier import numerical as program_replay

    environment = SimpleNamespace(
        image_reference="executor:test",
        image_id="sha256:" + "d" * 64,
        limits=SimpleNamespace(),
        semantic_sha256=lambda: _FAKE_EXECUTOR_ENVIRONMENT_SHA256,
        model_dump=lambda **_: {"image_reference": "executor:test", "image_id": "sha256:" + "d" * 64},
    )
    monkeypatch.setattr(docker_code_execution, "resolve_executor_environment", lambda **_: environment)

    def replay(
        *,
        workspace,
        entrypoint,
        target_ids,
        evidence_artifacts,
        executor_limits,
        executor_image,
        reporting_interface="v1",
    ):
        from trialagentbench_verifier.execution_contracts import CodeExecutionResultV1
        from trialagentbench_verifier.executor import analysis_project_artifacts
        from trialagentbench_verifier.numerical import NumericalReplay, read_numerical_table

        assert reporting_interface == expected_reporting_interface
        assert (workspace / "data" / "analysis.parquet").read_bytes() == b"practice-input\n"
        assert not (workspace / "scratch" / "stale-answer.json").exists()
        result = workspace / "scratch" / "estimates.csv"
        digest = hashlib.sha256(result.read_bytes()).hexdigest()
        execution = CodeExecutionResultV1(status="success", output="", elapsed_seconds=0.05)
        return NumericalReplay(
            status="reproduced",
            entrypoint=entrypoint,
            target_ids=target_ids,
            evidence_artifacts=evidence_artifacts,
            program_artifacts=analysis_project_artifacts(workspace / "scratch" / entrypoint),
            submitted_csv_sha256=digest,
            replayed_csv_sha256s=(digest, digest),
            executions=(execution, execution),
            estimates=read_numerical_table(result, target_ids=target_ids),
        )

    monkeypatch.setattr(program_replay, "replay_numerical_program", replay)


def test_execution_uses_authenticated_task_inputs_and_ignores_stale_side_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = _participant_release(tmp_path / "participant")
    submission = _submission(tmp_path / "submission")
    _write_json(submission / "stale-answer.json", {"reproducible": True, "controlled_target": 0.25})
    expected = hashlib.sha256((release / "release_manifest.json").read_bytes()).hexdigest()
    _install_fake_executor(monkeypatch)

    proof = execute_submission(
        release,
        submission,
        task_id="te_fixture",
        expected_sha256=expected,
        executor_image="executor:test",
        expected_executor_image_id=_FAKE_EXECUTOR_IMAGE_ID,
        expected_executor_environment_sha256=_FAKE_EXECUTOR_ENVIRONMENT_SHA256,
    )

    assert proof.status == "pass"
    assert proof.schema_id == "trialagentbench.execution_proof/v1"
    assert proof.operation == "execute"
    assert proof.attempt_count == 2
    assert proof.language == "python"
    assert proof.program_result_sha256 is not None


@pytest.mark.parametrize("anchor", ("image", "environment"))
def test_execution_rejects_wrong_executor_trust_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    anchor: str,
) -> None:
    """Refuse execution before participant code runs in an untrusted environment."""

    release = _participant_release(tmp_path / "participant")
    expected = hashlib.sha256((release / "release_manifest.json").read_bytes()).hexdigest()
    _install_fake_executor(monkeypatch)
    arguments = {
        "expected_executor_image_id": _FAKE_EXECUTOR_IMAGE_ID,
        "expected_executor_environment_sha256": _FAKE_EXECUTOR_ENVIRONMENT_SHA256,
    }
    arguments["expected_executor_image_id" if anchor == "image" else "expected_executor_environment_sha256"] = (
        "sha256:" + "f" * 64 if anchor == "image" else "f" * 64
    )

    with pytest.raises(VerificationInputError, match=f"executor {anchor} differs"):
        execute_submission(
            release,
            _submission(tmp_path / "submission"),
            task_id="te_fixture",
            expected_sha256=expected,
            **arguments,
        )


def test_within_tolerance_execution_proof_binds_the_submitted_result(tmp_path, monkeypatch):
    """Authenticate submitted CSV bytes while retaining non-exact replay hashes."""
    from trialagentbench_verifier import numerical

    release = _participant_release(tmp_path / "participant")
    submission = _submission(tmp_path / "submission")
    helper = submission / "helper.py"
    helper.write_text("value = 0.25\n")
    _install_fake_executor(monkeypatch)
    replay = numerical.replay_numerical_program

    def within_tolerance(**kwargs):
        return replay(**kwargs).model_copy(
            update={"status": "within_tolerance", "replayed_csv_sha256s": ("f" * 64, "f" * 64)}
        )

    monkeypatch.setattr(numerical, "replay_numerical_program", within_tolerance)
    proof = execute_submission(
        release,
        submission,
        task_id="te_fixture",
        expected_sha256=hashlib.sha256((release / "release_manifest.json").read_bytes()).hexdigest(),
        expected_executor_image_id=_FAKE_EXECUTOR_IMAGE_ID,
        expected_executor_environment_sha256=_FAKE_EXECUTOR_ENVIRONMENT_SHA256,
        executor_image="executor:test",
    )
    proof_path = tmp_path / "proof.json"
    _write_json(proof_path, proof.model_dump(mode="json"))
    normalized = NormalizedSubmissionV1(
        benchmark_stream="trialeval",
        task_id="te_fixture",
        submission_kind="analysis",
        payload={
            "schema_id": "trialagentbench.trialeval_score_input/v1",
            "task_id": "te_fixture",
            "attempt_status": "complete",
            "reproducible": True,
            "submitted_result": {
                "kind": "point",
                "value": 0.25,
                "effect_measure": "risk_difference",
                "unit": "proportion",
                "interval": {
                    "lower": 0.1,
                    "upper": 0.4,
                    "confidence_level": 0.95,
                    "interval_type": "central_confidence_interval",
                },
                "conclusion": "favors_treatment",
            },
            "evidence_source_artifacts": ["data/analysis.parquet"],
        },
    )

    def authenticate():
        return authenticate_execution_proof(
            proof_path,
            expected_sha256=hashlib.sha256(proof_path.read_bytes()).hexdigest(),
            program_path=submission / "analysis.py",
            program_result_path=submission / "estimates.csv",
            root=release,
            stream="trialeval",
            candidate_digest=proof.candidate_digest,
            release_digest=proof.release_digest,
            submission=normalized,
        )

    facts = authenticate()
    assert facts.replay_status == "within_tolerance"
    assert facts.program_result_sha256 == hashlib.sha256((submission / "estimates.csv").read_bytes()).hexdigest()
    assert facts.normalized_program_result_sha256 is None
    assert facts.program_result_sha256 != facts.replayed_result_sha256s[0]
    helper.write_text("value = 999\n")
    with pytest.raises(VerificationInputError, match="source project"):
        authenticate()
    helper.write_text("value = 0.25\n")
    (submission / "estimates.csv").write_text("target_id,estimate,lower,upper\neffect,999,0.1,0.4\n")
    with pytest.raises(VerificationInputError, match="CSV numerical values"):
        authenticate()


def test_diagnostic_support_is_derived_from_replay_bound_artifact_content() -> None:
    """Reject byte mismatch and withhold support after scientific verification failure."""

    artifact = DiagnosticArtifactExecutionV1(
        asset_id="asset_alpha",
        relative_path="scratch/diagnostics.csv",
        submitted_sha256="a" * 64,
        replayed_sha256s=("a" * 64, "a" * 64),
        verification_status="verified",
    )
    common = {
        "status": "pass",
        "input_kind": "directory",
        "input_sha256": "b" * 64,
        "expected_input_sha256": "b" * 64,
        "release_digest": "c" * 64,
        "candidate_digest": "d" * 64,
        "benchmark_stream": "trialdev",
        "task_id": "td-fixture",
        "language": "python",
        "program_sha256": "e" * 64,
        "program_result_sha256": "f" * 64,
        "score_projection_sha256": "1" * 64,
        "replayed_result_sha256s": ("f" * 64, "f" * 64),
        "replay_status": "reproduced",
        "execution_status": "success",
        "comparison_classification": "exact",
        "attempt_count": 2,
        "elapsed_seconds": 0.1,
        "executor_image_reference": "executor:test",
        "executor_image_id": _FAKE_EXECUTOR_IMAGE_ID,
        "expected_executor_image_id": _FAKE_EXECUTOR_IMAGE_ID,
        "executor_environment_sha256": _FAKE_EXECUTOR_ENVIRONMENT_SHA256,
        "expected_executor_environment_sha256": _FAKE_EXECUTOR_ENVIRONMENT_SHA256,
    }
    proof = ExecutionProofV1(**common, diagnostic_artifacts=(artifact,))
    assert proof.verified_diagnostic_asset_ids == ("asset_alpha",)
    unverified = artifact.model_copy(update={"verification_status": "value_mismatch"})
    assert ExecutionProofV1(**common, diagnostic_artifacts=(unverified,)).verified_diagnostic_asset_ids == ()
    with pytest.raises(ValidationError, match="differs across fresh replays"):
        DiagnosticArtifactExecutionV1(
            asset_id="asset_alpha",
            relative_path="scratch/diagnostics.csv",
            submitted_sha256="a" * 64,
            replayed_sha256s=("a" * 64, "b" * 64),
            verification_status="verified",
        )


def test_execution_rejects_input_tamper_before_executor_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = _participant_release(tmp_path / "participant")
    expected = hashlib.sha256((release / "release_manifest.json").read_bytes()).hexdigest()
    release.joinpath("release_manifest.json").write_text("{}\n", encoding="utf-8")
    called = False

    def unexpected(**_: object) -> None:
        nonlocal called
        called = True

    from trialagentbench_verifier import executor as docker_code_execution

    monkeypatch.setattr(docker_code_execution, "resolve_executor_environment", unexpected)
    with pytest.raises(VerificationInputError, match="trusted expected SHA-256"):
        execute_submission(
            release,
            _submission(tmp_path / "submission"),
            task_id="te_fixture",
            expected_sha256=expected,
            expected_executor_image_id=_FAKE_EXECUTOR_IMAGE_ID,
            expected_executor_environment_sha256=_FAKE_EXECUTOR_ENVIRONMENT_SHA256,
        )
    assert not called


def test_task_staging_rejects_a_symlinked_parent_directory(tmp_path: Path) -> None:
    """Reject task material whose leaf is regular only through a directory link."""

    release = _participant_release(tmp_path / "participant")
    data = release / "tasks" / "te_fixture" / "data"
    relocated = release / "relocated"
    data.rename(relocated)
    data.symlink_to(relocated, target_is_directory=True)
    destination = tmp_path / "workspace"
    destination.mkdir()

    with pytest.raises(VerificationInputError, match="contains a symlink"):
        _stage_task_inputs(
            release,
            destination,
            stream="trialeval",
            task=_task(release, "te_fixture"),
        )


def test_execution_refuses_evaluator_release_and_unavailable_containment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = _participant_release(tmp_path / "participant")
    complete = tmp_path / "complete"
    complete.mkdir()
    shutil.copytree(release, complete / "participant")
    shutil.copy2(release / "release_manifest.json", complete / "release_manifest.json")
    expected_complete = hashlib.sha256((complete / "release_manifest.json").read_bytes()).hexdigest()
    with pytest.raises(VerificationInputError, match="participant release"):
        execute_submission(
            complete,
            _submission(tmp_path / "submission-complete"),
            task_id="te_fixture",
            expected_sha256=expected_complete,
            expected_executor_image_id=_FAKE_EXECUTOR_IMAGE_ID,
            expected_executor_environment_sha256=_FAKE_EXECUTOR_ENVIRONMENT_SHA256,
        )

    expected = hashlib.sha256((release / "release_manifest.json").read_bytes()).hexdigest()
    from trialagentbench_verifier import executor as docker_code_execution

    monkeypatch.setattr(
        docker_code_execution,
        "resolve_executor_environment",
        lambda **_: (_ for _ in ()).throw(RuntimeError("cgroup accounting unavailable")),
    )
    output = tmp_path / "refusal.json"
    exit_code = main(
        [
            "execute",
            str(release),
            "--expected-sha256",
            expected,
            "--task-id",
            "te_fixture",
            "--submission",
            str(_submission(tmp_path / "submission-refusal")),
            "--expected-executor-image-id",
            _FAKE_EXECUTOR_IMAGE_ID,
            "--expected-executor-environment-sha256",
            _FAKE_EXECUTOR_ENVIRONMENT_SHA256,
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert report["operation"] == "execute"
    assert report["failure_code"] == "executor_unavailable"


def test_execution_proof_is_distinct_from_score_verification() -> None:
    from trialagentbench_verifier.contracts import ExecutionProofV1, VerificationReportV1

    assert ExecutionProofV1.model_fields["schema_id"].default != VerificationReportV1.model_fields["schema_id"].default


def test_public_numerical_sources_do_not_import_controlled_or_production_code() -> None:
    """Keep controlled truth and producer grading outside the independent verifier."""

    source = Path(__file__).parents[1] / "src" / "trialagentbench_verifier"
    forbidden = ("trialagentbench_construction", "trialagentbench_qualification")
    for path in source.rglob("*.py"):
        imports: list[str] = []
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imports.append(node.module)
        assert not any(name.startswith(forbidden) for name in imports), path
        if path.name not in {"execution.py", "release_set.py"}:
            assert not any(name == "trialagentbench" or name.startswith("trialagentbench.") for name in imports), path


@pytest.mark.executor
@pytest.mark.parametrize("language", ("python", "r"))
def test_live_public_execution_replays_python_and_r_from_clean_inputs(
    tmp_path: Path, language: str, live_executor: ExecutorEnvironmentV1
) -> None:
    release = _participant_release(tmp_path / "participant")
    expected = hashlib.sha256((release / "release_manifest.json").read_bytes()).hexdigest()
    submission = _executable_submission(tmp_path / language, language)
    (submission / "methods").mkdir()
    if language == "python":
        helper = submission / "methods" / "effect.py"
        helper.write_text("value = 0.25\n")
        entrypoint = submission / "analysis.py"
        prefix = "from methods.effect import value\nassert value == 0.25\n"
    else:
        helper = submission / "methods" / "effect.R"
        helper.write_text("value <- 0.25\n")
        entrypoint = submission / "analysis.R"
        prefix = "source('scratch/methods/effect.R')\nstopifnot(value == 0.25)\n"
    entrypoint.write_text(prefix + entrypoint.read_text())
    _write_json(submission / "stale-answer.json", {"reproducible": True, "value": 99})

    proof = execute_submission(
        release,
        submission,
        task_id="te_fixture",
        expected_sha256=expected,
        expected_executor_image_id=live_executor.image_id,
        expected_executor_environment_sha256=live_executor.semantic_sha256(),
    )

    assert proof.status == "pass"
    assert proof.language == language
    assert proof.replay_status == "reproduced"
    assert proof.execution_status == "success"
    assert proof.attempt_count == 2
    assert {row.relative_path for row in proof.program_artifacts} == {
        "scratch/" + entrypoint.name,
        "scratch/methods/" + helper.name,
    }
    assert next(row.sha256 for row in proof.program_artifacts if row.relative_path.endswith(helper.name)) == (
        hashlib.sha256(helper.read_bytes()).hexdigest()
    )


@pytest.mark.executor
@pytest.mark.parametrize(
    "reporting_interface,dataframe,expected_status",
    [(None, False, "pass"), (None, True, "fail"), ("v1", False, "pass"), ("clarified", True, "pass")],
)
def test_live_execution_uses_declared_reporting_interface(
    tmp_path: Path,
    live_executor: ExecutorEnvironmentV1,
    reporting_interface: ReportingInterfaceV1 | None,
    dataframe: bool,
    expected_status: str,
) -> None:
    """Clarified replays DataFrame reports; omitted and explicit v1 retain the old helper."""
    release = _participant_release(tmp_path / "participant")
    submission = _submission(tmp_path / "submission")
    (submission / "estimates.csv").write_bytes(b"target_id,estimate,lower,upper\r\neffect,0.25,0.1,0.4\r\n")
    (submission / "analysis.py").write_text(
        "import sys\nimport pandas as pd\nsys.path.insert(0, 'interface')\nfrom reporting import report\n"
        "rows = [{'target_id': 'effect', 'estimate': 0.25, 'lower': 0.1, 'upper': 0.4}]\n"
        + ("report(pd.DataFrame(rows))\n" if dataframe else "report(rows)\n")
    )
    arguments = {} if reporting_interface is None else {"reporting_interface": reporting_interface}
    proof = execute_submission(
        release,
        submission,
        task_id="te_fixture",
        expected_sha256=hashlib.sha256((release / "release_manifest.json").read_bytes()).hexdigest(),
        expected_executor_image_id=live_executor.image_id,
        expected_executor_environment_sha256=live_executor.semantic_sha256(),
        **arguments,
    )
    assert proof.status == expected_status
    if expected_status == "pass":
        assert proof.replay_status == "reproduced"
        assert proof.execution_status == "success"
        assert proof.attempt_count == 2
        assert proof.replayed_result_sha256s == (proof.program_result_sha256,) * 2
    else:
        assert proof.replay_status == "execution_error"
        assert proof.execution_status == "execution_error"


@pytest.mark.parametrize("reporting_interface", (None, "v1", "clarified"))
def test_execute_cli_forwards_reporting_interface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reporting_interface: ReportingInterfaceV1 | None
) -> None:
    """The independent CLI defaults to v1 and forwards an explicit condition."""
    release = _participant_release(tmp_path / "participant")
    submission = _submission(tmp_path / "submission")
    _install_fake_executor(monkeypatch, expected_reporting_interface=reporting_interface or "v1")
    output = tmp_path / "proof.json"
    assert (
        main(
            [
                "execute",
                str(release),
                "--expected-sha256",
                hashlib.sha256((release / "release_manifest.json").read_bytes()).hexdigest(),
                "--task-id",
                "te_fixture",
                "--submission",
                str(submission),
                "--expected-executor-image-id",
                _FAKE_EXECUTOR_IMAGE_ID,
                "--expected-executor-environment-sha256",
                _FAKE_EXECUTOR_ENVIRONMENT_SHA256,
                "--output",
                str(output),
                *([] if reporting_interface is None else ["--reporting-interface", reporting_interface]),
            ]
        )
        == 0
    )
    assert ExecutionProofV1.model_validate_json(output.read_bytes()).status == "pass"


@pytest.mark.executor
@pytest.mark.parametrize("ordered", [True, False])
def test_live_csv_interval_contract(tmp_path, ordered, live_executor):
    """An estimate outside an ordered interval is allowed; reversed endpoints fail."""
    release = _participant_release(tmp_path / "participant")
    submission = _executable_submission(tmp_path / "submission", "python")
    csv_text = "target_id,estimate,lower,upper\neffect,0.25," + ("1,2\n" if ordered else "2,1\n")
    (submission / "estimates.csv").write_text(csv_text)
    (submission / "analysis.py").write_text(
        "from pathlib import Path\n"
        "assert Path('data/analysis.parquet').exists()\n"
        f"Path('scratch/estimates.csv').write_text({csv_text!r})\n"
    )
    proof = execute_submission(
        release,
        submission,
        task_id="te_fixture",
        expected_sha256=hashlib.sha256((release / "release_manifest.json").read_bytes()).hexdigest(),
        expected_executor_image_id=live_executor.image_id,
        expected_executor_environment_sha256=live_executor.semantic_sha256(),
    )
    assert proof.status == ("pass" if ordered else "fail")
    assert proof.replay_status == ("reproduced" if ordered else "invalid_result")
    assert proof.attempt_count == (2 if ordered else 0)


@pytest.mark.executor
def test_live_public_execution_rejects_output_tamper(tmp_path: Path, live_executor: ExecutorEnvironmentV1) -> None:
    release = _participant_release(tmp_path / "participant")
    expected = hashlib.sha256((release / "release_manifest.json").read_bytes()).hexdigest()
    submission = _executable_submission(tmp_path / "submission", "python", submitted_value=0.5)

    proof = execute_submission(
        release,
        submission,
        task_id="te_fixture",
        expected_sha256=expected,
        expected_executor_image_id=live_executor.image_id,
        expected_executor_environment_sha256=live_executor.semantic_sha256(),
    )

    assert proof.status == "fail"
    assert proof.replay_status == "result_mismatch"
    assert proof.failure_code == "result_mismatch"
    assert proof.comparison_classification == "irreproducible"


@pytest.mark.executor
def test_live_host_cross_attempt_and_orphan_are_contained_before_clean_recovery(
    tmp_path: Path, live_executor: ExecutorEnvironmentV1
) -> None:
    release = _participant_release(tmp_path / "participant")
    expected = hashlib.sha256((release / "release_manifest.json").read_bytes()).hexdigest()
    host_marker = tmp_path / "host-only-secret.txt"
    host_marker.write_text("host-secret\n", encoding="utf-8")
    hostile = tmp_path / "hostile"
    hostile.mkdir()
    csv_text = "target_id,estimate,lower,upper\neffect,0.25,0.1,0.4\n"
    program = (
        "import json, os, time\n"
        "from pathlib import Path\n"
        f"assert not Path({str(host_marker)!r}).exists()\n"
        "assert not Path('tasks/sibling/controlled-answer.json').exists()\n"
        f"Path('scratch/estimates.csv').write_text({csv_text!r})\n"
        "first = os.fork()\n"
        "if first == 0:\n"
        "    os.setsid()\n"
        "    second = os.fork()\n"
        "    if second == 0:\n"
        "        time.sleep(0.5)\n"
        "        Path('scratch/orphan.txt').write_text('escaped')\n"
        "        os._exit(0)\n"
        "    os._exit(0)\n"
        "os.waitpid(first, 0)\n"
    )
    (hostile / "analysis.py").write_text(program, encoding="utf-8")
    (hostile / "estimates.csv").write_text(csv_text)

    hostile_proof = execute_submission(
        release,
        hostile,
        task_id="te_fixture",
        expected_sha256=expected,
        expected_executor_image_id=live_executor.image_id,
        expected_executor_environment_sha256=live_executor.semantic_sha256(),
    )
    clean_proof = execute_submission(
        release,
        _executable_submission(tmp_path / "clean", "python"),
        task_id="te_fixture",
        expected_sha256=expected,
        expected_executor_image_id=live_executor.image_id,
        expected_executor_environment_sha256=live_executor.semantic_sha256(),
    )

    assert hostile_proof.status == "pass"
    assert clean_proof.status == "pass"


@pytest.mark.executor
def test_live_resource_attack_cannot_hide_behind_a_correct_result(
    tmp_path: Path, live_executor: ExecutorEnvironmentV1
) -> None:
    release = _participant_release(tmp_path / "participant")
    expected = hashlib.sha256((release / "release_manifest.json").read_bytes()).hexdigest()
    hostile = tmp_path / "resource"
    hostile.mkdir()
    csv_text = "target_id,estimate,lower,upper\neffect,0.25,0.1,0.4\n"
    program = (
        "import json, subprocess\n"
        "from pathlib import Path\n"
        f"Path('scratch/estimates.csv').write_text({csv_text!r})\n"
        "children = [subprocess.Popen(['sleep', '30']) for _ in range(256)]\n"
    )
    (hostile / "analysis.py").write_text(program, encoding="utf-8")
    (hostile / "estimates.csv").write_text(csv_text)

    proof = execute_submission(
        release,
        hostile,
        task_id="te_fixture",
        expected_sha256=expected,
        expected_executor_image_id=live_executor.image_id,
        expected_executor_environment_sha256=live_executor.semantic_sha256(),
    )
    clean = execute_submission(
        release,
        _executable_submission(tmp_path / "after-resource", "python"),
        task_id="te_fixture",
        expected_sha256=expected,
        expected_executor_image_id=live_executor.image_id,
        expected_executor_environment_sha256=live_executor.semantic_sha256(),
    )

    assert proof.status == "fail"
    assert proof.replay_status == "execution_error"
    assert proof.execution_status == "execution_error"
    assert clean.status == "pass"


@pytest.mark.executor
@pytest.mark.slow
def test_installed_verifier_alone_executes_python_and_r_from_a_foreign_directory(
    tmp_path: Path, live_executor: ExecutorEnvironmentV1
) -> None:
    package = Path(__file__).resolve().parents[1]
    verifier_dist = tmp_path / "verifier-dist"
    verifier_dist.mkdir()
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(verifier_dist), str(package)],
        check=True,
        capture_output=True,
        text=True,
    )
    installed = tmp_path / "installed"
    wheels = list(verifier_dist.glob("*.whl"))
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(installed), *map(str, wheels)],
        check=True,
        capture_output=True,
        text=True,
    )
    release = _participant_release(tmp_path / "participant")
    expected = hashlib.sha256((release / "release_manifest.json").read_bytes()).hexdigest()
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    # Skip editable-install hooks and deny application imports even when the
    # dependency directory happens to contain an older noneditable harness.
    # A separate clean-venv release check validates the installation closure.
    bootstrap = (
        "import importlib.abc, runpy, sys\n"
        "class DenyApplicationImports(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname == 'trialagentbench' or fullname.startswith(('trialagentbench.', 'ariadne', 'trialagentbench_construction', 'trialagentbench_qualification')):\n"
        "            raise ModuleNotFoundError('Application import denied: ' + fullname)\n"
        "        return None\n"
        "sys.meta_path.insert(0, DenyApplicationImports())\n"
        f"sys.path.append({sysconfig.get_path('purelib')!r})\n"
        "module = sys.argv.pop(1)\nrunpy.run_module(module, run_name='__main__')\n"
    )
    isolated_python = [sys.executable, "-S", "-c", bootstrap]
    for filename in ("test_public_contract.py", "test_terminal_value.py"):
        shutil.copyfile(Path(__file__).with_name(filename), foreign / filename)
    contracts = subprocess.run(
        [
            *isolated_python,
            "pytest",
            "-q",
            "test_public_contract.py",
            "test_terminal_value.py",
        ],
        cwd=foreign,
        env={**os.environ, "PYTHONPATH": str(installed), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        check=False,
        capture_output=True,
        text=True,
    )
    (foreign / "contract-tests.log").write_text(contracts.stdout + contracts.stderr, encoding="utf-8")
    assert contracts.returncode == 0, contracts.stdout + contracts.stderr
    practice_output = foreign / "practice.json"
    practice = subprocess.run(
        [
            *isolated_python,
            "trialagentbench_verifier",
            "practice",
            "--output",
            str(practice_output),
        ],
        cwd=foreign,
        env={**os.environ, "PYTHONPATH": str(installed)},
        check=False,
        capture_output=True,
        text=True,
    )
    assert practice.returncode == 0, practice.stderr
    practice_report = json.loads(practice_output.read_text(encoding="utf-8"))
    assert practice_report["passed_check_count"] == practice_report["scheduled_check_count"]
    assert practice_report["passed_check_count"] > 0
    for language in ("python", "r"):
        submission = _executable_submission(tmp_path / f"installed-{language}", language)
        output = foreign / f"{language}-proof.json"
        completed = subprocess.run(
            [
                *isolated_python,
                "trialagentbench_verifier",
                "execute",
                str(release),
                "--expected-sha256",
                expected,
                "--task-id",
                "te_fixture",
                "--submission",
                str(submission),
                "--expected-executor-image-id",
                live_executor.image_id,
                "--expected-executor-environment-sha256",
                live_executor.semantic_sha256(),
                "--output",
                str(output),
            ],
            cwd=foreign,
            env={**os.environ, "PYTHONPATH": str(installed)},
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        proof = json.loads(output.read_text(encoding="utf-8"))
        assert proof["status"] == "pass"
        assert proof["language"] == language


@pytest.mark.parametrize("reporting_interface", ("v1", "clarified"))
def test_batch_execution_matches_scalar_and_closes_its_shared_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reporting_interface: ReportingInterfaceV1
) -> None:
    """Keep per-item isolation while authenticating one shared participant input."""
    release = _participant_release(tmp_path / "participant")
    submission = _submission(tmp_path / "submission")
    expected = hashlib.sha256((release / "release_manifest.json").read_bytes()).hexdigest()
    _install_fake_executor(monkeypatch, expected_reporting_interface=reporting_interface)
    arguments = {
        "expected_sha256": expected,
        "executor_image": "executor:test",
        "reporting_interface": reporting_interface,
        "expected_executor_image_id": _FAKE_EXECUTOR_IMAGE_ID,
        "expected_executor_environment_sha256": _FAKE_EXECUTOR_ENVIRONMENT_SHA256,
    }
    scalar = execute_submission(release, submission, task_id="te_fixture", **arguments)
    original = execution_module.materialized_release
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

    monkeypatch.setattr(execution_module, "materialized_release", observed)
    request = ExecutionSubmissionV1(task_id="te_fixture", submission=submission)
    proofs = list(execute_submissions(release, (request, request), **arguments))
    assert proofs == [scalar, scalar]
    assert entered == closed == [release]
    stream = execute_submissions(release, (request, request), **arguments)
    assert next(stream) == scalar
    stream.close()
    assert len(entered) == len(closed) == 2


@pytest.mark.parametrize("entrypoint", ["analysis.py", "analysis.R"])
def test_standalone_stages_local_analysis_modules(tmp_path, entrypoint):
    """The collaborator replay receives the same source project as the harness."""
    source = tmp_path / "submission"
    (source / "methods").mkdir(parents=True)
    (source / entrypoint).write_text("# entrypoint\n")
    (source / "methods" / "effect.py").write_text("value = 1\n")
    (source / "methods" / "effect.R").write_text("value <- 1\n")
    (source / "generated.csv").write_text("cached_answer\n999\n")
    destination = tmp_path / "workspace"
    destination.mkdir()
    language = execution_module._stage_submission(source, destination)
    assert language == ("python" if entrypoint == "analysis.py" else "r")
    for name in (entrypoint, "methods/effect.py", "methods/effect.R"):
        assert (destination / "scratch" / name).read_bytes() == (source / name).read_bytes()
    assert not (destination / "scratch" / "generated.csv").exists()


@pytest.mark.parametrize("linked", ["analysis.py", "methods/effect.py"])
def test_standalone_rejects_linked_project_sources(tmp_path, linked):
    """Source modules cannot smuggle files from outside the submitted project."""
    source = tmp_path / "submission"
    (source / "methods").mkdir(parents=True)
    if linked != "analysis.py":
        (source / "analysis.py").write_text("# entrypoint\n")
    external = tmp_path / "external.py"
    external.write_text("secret = 1\n")
    (source / linked).symlink_to(external)
    destination = tmp_path / "workspace"
    destination.mkdir()
    with pytest.raises(ValueError, match="regular|symbolic"):
        execution_module._stage_submission(source, destination)


@pytest.mark.executor
@pytest.mark.parametrize("language", ["python", "r"])
@pytest.mark.parametrize("paired", [False, True])
def test_live_csv_execution_and_authenticated_score(tmp_path, live_executor, language, paired):
    """Public execution and score authentication share the same numerical CSV."""
    targets = ("effect", "secondary") if paired else ("effect",)
    release = _participant_release(tmp_path / "participant", target_ids=targets)
    submission = tmp_path / "submission"
    submission.mkdir()
    if language == "python":
        entrypoint = submission / "analysis.py"
        entrypoint.write_text(
            "from pathlib import Path\nimport csv\n"
            "value=len(Path('data/analysis.parquet').read_bytes())/100\n"
            "with open('scratch/estimates.csv','w',newline='') as handle:\n"
            "    w=csv.writer(handle)\n    w.writerow(['target_id','estimate','lower','upper'])\n"
            "    w.writerow(['effect',value,value-0.1,value+0.1])\n"
        )
    else:
        entrypoint = submission / "analysis.R"
        entrypoint.write_text(
            "value <- file.info('data/analysis.parquet')$size/100\n"
            "write.csv(data.frame(target_id='effect',estimate=value,lower=value-0.1,upper=value+0.1),"
            "'scratch/estimates.csv',row.names=FALSE)\n"
        )
    if paired:
        with entrypoint.open("a") as handle:
            handle.write(
                "with open('scratch/estimates.csv','a') as handle: handle.write('secondary,4,2,6\\n')\n"
                if language == "python"
                else "cat('secondary,4,2,6\\n',file='scratch/estimates.csv',append=TRUE)\n"
            )
    result_path = submission / "estimates.csv"
    result_path.write_text(
        "target_id,estimate,lower,upper\neffect,0.15,0.05,0.25\n" + ("secondary,4,2,6\n" if paired else "")
    )
    proof = execute_submission(
        release,
        submission,
        task_id="te_fixture",
        expected_sha256=hashlib.sha256((release / "release_manifest.json").read_bytes()).hexdigest(),
        expected_executor_image_id=live_executor.image_id,
        expected_executor_environment_sha256=live_executor.semantic_sha256(),
    )
    assert proof.status == "pass" and proof.result_format == "csv"
    assert proof.target_ids == targets
    assert proof.evidence_artifacts[0].relative_path == "data/analysis.parquet"
    proof_path = tmp_path / "proof.json"
    _write_json(proof_path, proof.model_dump(mode="json"))
    normalized = NormalizedSubmissionV1(
        benchmark_stream="trialeval",
        task_id="te_fixture",
        submission_kind="analysis",
        payload={
            "schema_id": "trialagentbench.trialeval_score_input/v1",
            "task_id": "te_fixture",
            "attempt_status": "complete",
            "reproducible": True,
            "submitted_result": {
                "kind": "point",
                "value": 0.15,
                "effect_measure": "risk_difference",
                "unit": "proportion",
                "interval": {
                    "lower": 0.05,
                    "upper": 0.25,
                    "confidence_level": 0.95,
                    "interval_type": "central_confidence_interval",
                },
                "conclusion": "favors_treatment",
            },
            "evidence_source_artifacts": ["data/analysis.parquet"],
        },
    )

    if paired:
        from copy import deepcopy

        payload = deepcopy(normalized.payload)
        payload["target_id"] = "effect"
        secondary = {key: deepcopy(value) for key, value in payload.items() if key not in {"schema_id", "task_id"}}
        secondary["target_id"] = "secondary"
        secondary["submitted_result"].update(
            value=4.0,
            effect_measure="restricted_mean_survival_time_difference",
            unit="days",
            interval={"lower": 2.0, "upper": 6.0, "confidence_level": 0.95},
        )
        payload["additional_inputs"] = [secondary]
        normalized = normalized.model_copy(update={"payload": payload})

    def authenticate():
        return authenticate_execution_proof(
            proof_path,
            expected_sha256=hashlib.sha256(proof_path.read_bytes()).hexdigest(),
            program_path=entrypoint,
            program_result_path=result_path,
            root=release,
            stream="trialeval",
            candidate_digest=proof.candidate_digest,
            release_digest=proof.release_digest,
            submission=normalized,
        )

    facts = authenticate()
    assert facts.program_artifacts == proof.program_artifacts
    assert facts.program_result_sha256 == hashlib.sha256(result_path.read_bytes()).hexdigest()
    result_path.write_text(
        "target_id,estimate,lower,upper\neffect,999,0.05,0.25\n" + ("secondary,4,2,6\n" if paired else "")
    )
    with pytest.raises(VerificationInputError, match="CSV numerical values"):
        authenticate()


@pytest.mark.parametrize("targets", [[], ["effect", "effect"], [""]])
def test_trialdev_execution_rejects_missing_or_ambiguous_public_targets(tmp_path, monkeypatch, targets):
    release = _participant_release(tmp_path / "participant")
    task_path = release / "tasks/te_fixture/task.json"
    task = json.loads(task_path.read_text())
    task.pop("target_ids")
    task["stream"] = "evidence_analysis"
    task["case_id"] = "tdw-fixture"
    task["initial_target_ids"] = targets
    _write_json(task_path, task)
    expected = hashlib.sha256((release / "release_manifest.json").read_bytes()).hexdigest()
    monkeypatch.setattr(execution_module, "_participant_identity", lambda root: ("trialdev", "a" * 64, expected))
    monkeypatch.setattr(execution_module, "_verify_participant_release", lambda *args, **kwargs: None)
    _install_fake_executor(monkeypatch)
    submission = _executable_submission(tmp_path / "submission", "python")
    with pytest.raises(VerificationInputError, match="numerical task contract|nonempty and unique"):
        execute_submission(
            release,
            submission,
            task_id="te_fixture",
            expected_sha256=expected,
            expected_executor_image_id=_FAKE_EXECUTOR_IMAGE_ID,
            expected_executor_environment_sha256=_FAKE_EXECUTOR_ENVIRONMENT_SHA256,
        )


def test_trialdev_initial_csv_uses_public_targets(tmp_path, monkeypatch):
    """Exercise staging and proof binding; archive admission and execution are isolated."""
    from trialagentbench_verifier import execution as owner

    release = _participant_release(tmp_path / "participant")
    task_path = release / "tasks/te_fixture/task.json"
    task = json.loads(task_path.read_text())
    task.pop("target_ids")
    task["stream"] = "evidence_analysis"
    task["initial_target_ids"] = ["effect"]
    task["case_id"] = "tdw-fixture"
    # TrialDev materials live under their world, as in released TrialDev archives.
    source = release / task["available_materials"][0]["relative_path"]
    world_data = release / "worlds/tdw-fixture/data/analysis.parquet"
    world_data.parent.mkdir(parents=True)
    world_data.write_bytes(source.read_bytes())
    task["available_materials"][0]["relative_path"] = "data/analysis.parquet"
    _write_json(task_path, task)
    expected = hashlib.sha256((release / "release_manifest.json").read_bytes()).hexdigest()
    monkeypatch.setattr(owner, "_participant_identity", lambda root: ("trialdev", "a" * 64, expected))
    monkeypatch.setattr(owner, "_verify_participant_release", lambda *args, **kwargs: None)
    _install_fake_executor(monkeypatch)
    submission = _executable_submission(tmp_path / "submission", "python")
    (submission / "analysis_result.json").write_text("unrelated JSON file")
    proof = execute_submission(
        release,
        submission,
        task_id="te_fixture",
        expected_sha256=expected,
        expected_executor_image_id="sha256:" + "d" * 64,
        expected_executor_environment_sha256=_FAKE_EXECUTOR_ENVIRONMENT_SHA256,
    )
    assert proof.status == "pass"
    assert proof.benchmark_stream == "trialdev"
    assert proof.result_format == "csv"
    assert proof.target_ids == ("effect",)
    assert proof.diagnostic_artifacts == ()
    assert proof.score_projection_sha256 == _canonical(
        {"estimates": [{"target_id": "effect", "estimate": 0.25, "lower": 0.1, "upper": 0.4}]}
    )
    from trialagentbench_verifier.contracts import NormalizedSubmissionV1
    from trialagentbench_verifier.proof import authenticate_execution_proof

    proof_path = tmp_path / "proof.json"
    proof_path.write_text(proof.model_dump_json())
    normalized = NormalizedSubmissionV1(
        benchmark_stream="trialdev",
        task_id="te_fixture",
        submission_kind="evidence",
        payload={
            "task_id": "te_fixture",
            "estimates": [{"target_id": "effect", "estimate": 0.25, "lower": 0.1, "upper": 0.4}],
        },
    )
    facts = authenticate_execution_proof(
        proof_path,
        expected_sha256=hashlib.sha256(proof_path.read_bytes()).hexdigest(),
        program_path=submission / "analysis.py",
        program_result_path=submission / "estimates.csv",
        root=release,
        stream="trialdev",
        candidate_digest="a" * 64,
        release_digest=expected,
        submission=normalized,
    )
    assert facts.program_result_sha256 == proof.program_result_sha256
    from trialagentbench_verifier import release as score_owner
    from trialagentbench_verifier.contracts import CheckResultV1

    _write_json(release / "participant/tasks/te_fixture/task.json", task)

    def capture_score(root, submission_path, normalized, **kwargs):
        assert kwargs["execution_authenticated"]
        assert normalized.payload["estimates"] == [
            {"target_id": "effect", "estimate": 0.25, "lower": 0.1, "upper": 0.4}
        ]
        return (CheckResultV1(check_id="score.control", scope_id="te_fixture", status="pass", observed=1, expected=1),)

    monkeypatch.setattr(score_owner, "_score_trialdev", capture_score)
    scored = score_owner._score_materialized_submission(
        release,
        score_owner.ScoreSubmissionV1(
            submission=submission / "estimates.csv",
            execution_proof=proof_path,
            expected_execution_proof_sha256=hashlib.sha256(proof_path.read_bytes()).hexdigest(),
            program=submission / "analysis.py",
            program_result=submission / "estimates.csv",
        ),
        stream="trialdev",
        candidate="a" * 64,
        release_digest=expected,
        report_digests=(),
        development=False,
        input_kind="directory",
        input_sha256=expected,
        expected_sha256=expected,
    )
    assert scored.status == "pass"
    assert (
        scored.score_binding.submission_artifact_sha256
        == hashlib.sha256((submission / "estimates.csv").read_bytes()).hexdigest()
    )


@pytest.mark.executor
@pytest.mark.parametrize("language", ["python", "r"])
def test_live_paired_analysis_from_participant_records(tmp_path, live_executor, language):
    """Independent arm summaries and regression replay both complete-data targets."""
    import math

    import pandas as pd

    records = pd.DataFrame(
        {
            "arm": [0] * 4 + [1] * 4,
            "time": [1.0, 3.0, 10.0, 10.0, 6.0, 10.0, 10.0, 10.0],
            "event": [1, 1, 0, 0, 1, 0, 0, 0],
        }
    )
    release = _participant_release(
        tmp_path / "participant", target_ids=("risk", "rmst"), analysis_bytes=records.to_parquet(index=False)
    )
    submission = tmp_path / "submission"
    submission.mkdir()
    if language == "python":
        program = """import pandas as pd
import math
x = pd.read_parquet('data/analysis.parquet')
rows = []
for target, column in [('risk', 'event'), ('rmst', 'time')]:
    a = x.loc[x.arm == 1, column]
    c = x.loc[x.arm == 0, column]
    estimate = a.mean() - c.mean()
    se = math.sqrt(a.var(ddof=1)/len(a) + c.var(ddof=1)/len(c))
    rows.append([target, estimate, estimate-1.96*se, estimate+1.96*se])
pd.DataFrame(rows, columns=['target_id','estimate','lower','upper']).to_csv('scratch/estimates.csv',index=False)
"""
        entrypoint = "analysis.py"
    else:
        program = """library(trialagentbenchR)
x <- read_trialagentbench_parquet('data/analysis.parquet')
rows <- lapply(c('event', 'time'), function(column) {
    fit <- lm(reformulate('arm', response=column), data=x)
    estimate <- coef(fit)[['arm']]
    se <- sqrt(vcov(fit)['arm','arm'])
    c(estimate, estimate-1.96*se, estimate+1.96*se)
})
result <- data.frame(target_id=c('risk','rmst'), do.call(rbind,rows))
names(result) <- c('target_id','estimate','lower','upper')
write.csv(result, 'scratch/estimates.csv', row.names=FALSE)
"""
        entrypoint = "analysis.R"
    (submission / entrypoint).write_text(program)
    expected = []
    for target, estimate, variance in [("risk", -0.25, 7 / 48), ("rmst", 3.0, 6.5)]:
        se = math.sqrt(variance)
        expected.append([target, estimate, estimate - 1.96 * se, estimate + 1.96 * se])
    pd.DataFrame(expected, columns=["target_id", "estimate", "lower", "upper"]).to_csv(
        submission / "estimates.csv", index=False
    )
    (submission / "report.md").write_text(
        "At day 10 the observed event-risk difference is -0.25 and the restricted mean difference is 3 days. "
        "Both intervals are marginal normal intervals from the complete follow-up data."
    )
    proof = execute_submission(
        release,
        submission,
        task_id="te_fixture",
        expected_sha256=hashlib.sha256((release / "release_manifest.json").read_bytes()).hexdigest(),
        expected_executor_image_id=live_executor.image_id,
        expected_executor_environment_sha256=live_executor.semantic_sha256(),
    )
    assert proof.status == "pass", proof
    assert proof.target_ids == ("risk", "rmst")
    assert proof.attempt_count == 2
    assert proof.replay_status in {"reproduced", "within_tolerance"}


@pytest.mark.executor
def test_live_r_design_models_use_synthetic_records(tmp_path, live_executor) -> None:
    """Installed R libraries fit clustered survival, repeated measures and dispersed counts."""
    import numpy as np
    import pandas as pd

    from trialagentbench_verifier.executor import DockerCodeSession

    data = tmp_path / "data"
    data.mkdir()
    rng = np.random.RandomState(10)
    clusters = np.repeat(np.arange(24), 8)
    event_time = rng.exponential(240, size=len(clusters))
    pd.DataFrame(
        {
            "cluster_id": clusters,
            "assigned_arm": np.where(clusters % 2, "active", "control"),
            "observed_time": np.minimum(event_time, 180),
            "event_observed": event_time <= 180,
        }
    ).to_parquet(data / "clustered.parquet", index=False)
    people = 64
    age = rng.randint(30, 80, size=people)
    bmi = rng.normal(25, 3, size=people)
    treated = np.arange(people) % 2
    visits = np.tile(np.arange(4), people)
    pd.DataFrame(
        {
            "participant_id": np.repeat(np.arange(people), 4),
            "assigned_arm": np.repeat(np.where(treated, "active", "control"), 4),
            "occurrence_time": visits,
            "age_years": np.repeat(age, 4),
            "baseline_body_mass_index": np.repeat(bmi, 4),
            "asthma_control_score": (
                np.repeat(10 + age * 0.02 + bmi * 0.1 + rng.normal(size=people), 4)
                - np.repeat(treated, 4) * visits * 0.4
                + rng.normal(size=people * 4)
            ),
        }
    ).to_parquet(data / "longitudinal.parquet", index=False)
    pd.DataFrame(
        {
            "assigned_arm": np.where(treated, "active", "control"),
            "age_years": age,
            "baseline_body_mass_index": bmi,
            "heart_failure_hospitalization_count": rng.negative_binomial(2, 0.4 + 0.1 * treated),
        }
    ).to_parquet(data / "counts.parquet", index=False)
    program = (Path(__file__).parent / "fixtures/design_models.R").read_text()
    session = DockerCodeSession(cwd=tmp_path, image=live_executor.image_id, limits=RELEASE_EXECUTION_LIMITS_V1)
    try:
        result = session.execute_result(program, language="r")
        assert result.status == "success", result.output
        scratch = session.snapshot_scratch()
        models = pd.read_csv(scratch / "design_models.csv")
        assert models.model.tolist() == ["clustered_survival", "longitudinal_random_intercept", "negative_binomial"]
        assert models.participants.gt(0).all()
        assert models.observations.ge(models.participants).all()
        assert models.version.notna().all()
    finally:
        session.close()


def test_readonly_cached_material_is_sanitized_in_private_replay_copy(tmp_path: Path) -> None:
    root = tmp_path / "release"
    root.mkdir()
    (root / "documents").mkdir()
    policy = root / "documents/data_integrity_policy.json"
    policy.write_text('{"description":"ordinary source records"}')
    policy.chmod(0o444)
    task = execution_module._Task.model_validate(
        {"task_id": "fixture", "available_materials": [{"relative_path": "documents/data_integrity_policy.json"}]}
    )
    workspace = tmp_path / "workspace"
    _stage_task_inputs(root, workspace, stream="trialeval", task=task)
    staged = workspace / "data_integrity_policy.json"
    assert staged.stat().st_ino != policy.stat().st_ino
    assert staged.stat().st_mode & 0o200
    assert not policy.stat().st_mode & 0o200
    assert json.loads(staged.read_text()) == {"description": "ordinary source records"}
