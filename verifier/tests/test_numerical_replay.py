"""CSV replay binds the project and reporting-cutoff evidence for both languages."""

from pathlib import Path

import pytest

from trialagentbench_verifier import numerical
from trialagentbench_verifier.execution_contracts import (
    CodeExecutionLimitsV1,
    CodeExecutionResultV1,
    ReplayInputArtifactV1,
)
from trialagentbench_verifier.io import sha256_file
from trialagentbench_verifier.numerical import replay_numerical_program


def _workspace(tmp_path, language="python"):
    root = tmp_path / "workspace"
    (root / "scratch" / "methods").mkdir(parents=True)
    (root / "data").mkdir()
    values = root / "data" / "values.csv"
    values.write_text("value\n1\n2\n6\n")
    (root / "data" / "future.csv").write_text("hidden_future\n999\n")
    (root / "scratch" / "estimates.csv").write_text("target_id,estimate,lower,upper\nmean,3,1,5\n")
    entrypoint = "analysis.py" if language == "python" else "analysis.R"
    if language == "python":
        (root / "scratch" / "methods" / "effect.py").write_text(
            "def mean(values):\n    return sum(values) / len(values)\n"
        )
        code = (
            "import csv\nfrom pathlib import Path\nfrom methods.effect import mean\n"
            "assert not Path('data/future.csv').exists()\n"
            "assert not Path('scratch/estimates.csv').exists()\n"
            "with open('data/values.csv') as handle:\n    value = mean([float(r['value']) for r in csv.DictReader(handle)])\n"
            "with open('scratch/estimates.csv', 'w', newline='') as handle:\n"
            "    writer=csv.writer(handle)\n    writer.writerow(['target_id','estimate','lower','upper'])\n"
            "    writer.writerow(['mean',value,1,5])\n"
        )
    else:
        (root / "scratch" / "methods" / "effect.R").write_text("effect <- function(x) mean(x)\n")
        code = (
            "source('scratch/methods/effect.R')\n"
            "stopifnot(!file.exists('data/future.csv'), !file.exists('scratch/estimates.csv'))\n"
            "value <- effect(read.csv('data/values.csv')$value)\n"
            "write.csv(data.frame(target_id='mean',estimate=value,lower=1,upper=5), 'scratch/estimates.csv',row.names=FALSE)\n"
        )
    (root / "scratch" / entrypoint).write_text(code)
    artifact = ReplayInputArtifactV1(
        relative_path="data/values.csv", sha256=sha256_file(values), byte_length=values.stat().st_size
    )
    return root, entrypoint, (artifact,)


def _run(root, entrypoint, artifacts):
    return replay_numerical_program(
        workspace=root,
        entrypoint=entrypoint,
        target_ids=("mean", "unreported"),
        evidence_artifacts=artifacts,
        executor_limits=CodeExecutionLimitsV1(
            timeout_seconds=10, memory_mb=512, cpu_count=1, process_limit=32, workspace_mb=128
        ),
    )


@pytest.mark.executor
@pytest.mark.parametrize("language", ["python", "r"])
def test_real_csv_replay_uses_modules_but_not_future_files_or_submitted_answers(tmp_path, language):
    root, entrypoint, artifacts = _workspace(tmp_path, language)
    result = _run(root, entrypoint, artifacts)
    assert result.reproducible
    assert len(result.executions) == len(result.replayed_csv_sha256s) == 2
    assert result.estimates is not None and result.estimates[0].estimate == 3
    assert result.target_ids == ("mean", "unreported")
    assert result.evidence_artifacts == artifacts
    assert len(result.program_artifacts) == 2
    assert all(row.elapsed_seconds > 0 for row in result.executions)


def test_invalid_submission_never_starts_an_executor(tmp_path, monkeypatch):
    root, entrypoint, artifacts = _workspace(tmp_path)
    (root / "scratch" / "estimates.csv").write_text("target_id,estimate,lower,upper\nmean,3,5,1\n")

    def unexpected(*args, **kwargs):
        raise AssertionError("invalid submission must not start an executor")

    monkeypatch.setattr(numerical, "DockerCodeSession", unexpected)
    result = _run(root, entrypoint, artifacts)
    assert result.status == "invalid_result"
    assert not result.executions and result.estimates is None


@pytest.mark.parametrize("initial", ["stale", "missing"])
@pytest.mark.parametrize("deterministic", [True, False])
def test_live_capture_uses_saved_script_and_requires_two_agreeing_executions(
    tmp_path, monkeypatch, initial, deterministic
):
    root, entrypoint, artifacts = _workspace(tmp_path)
    result_path = root / "scratch" / "estimates.csv"
    if initial == "missing":
        result_path.unlink()
    else:
        result_path.write_text("stale invalid CSV\n")
    before = result_path.read_bytes() if result_path.exists() else None
    calls = []

    class Session:
        def __init__(self, cwd, **kwargs):
            self.root = Path(cwd)

        def execute_result(self, code, *, language):
            assert not (self.root / "data" / "future.csv").exists()
            assert not (self.root / "scratch" / "estimates.csv").exists()
            calls.append(self.root)
            value = 3 if deterministic or len(calls) == 1 else 9
            (self.root / "scratch" / "estimates.csv").write_text(f"target_id,estimate,lower,upper\nmean,{value},1,5\n")
            return CodeExecutionResultV1(status="success", output="", elapsed_seconds=0.01)

        def snapshot_scratch(self):
            return self.root / "scratch"

        def close(self):
            pass

    monkeypatch.setattr(numerical, "DockerCodeSession", Session)
    result = replay_numerical_program(
        workspace=root,
        entrypoint=entrypoint,
        target_ids=("mean",),
        evidence_artifacts=artifacts,
        executor_limits=CodeExecutionLimitsV1(),
        capture_result=True,
    )
    assert len(calls) == 2
    assert result.reproducible is deterministic
    if deterministic:
        assert result.estimates[0].estimate == 3
        assert result.submitted_csv_sha256 == sha256_file(result_path)
        # Grading still checks the captured CSV and never rewrites the captured output.
        result_path.write_text("target_id,estimate,lower,upper\nmean,100,1,5\n")
        assert _run(root, entrypoint, artifacts).status == "result_mismatch"
    else:
        assert result.status == "result_mismatch"
        assert (result_path.read_bytes() if result_path.exists() else None) == before


def test_changed_evidence_is_an_integrity_error_not_a_model_grade(tmp_path, monkeypatch):
    root, entrypoint, artifacts = _workspace(tmp_path)
    (root / "data" / "values.csv").write_text("value\n999\n")

    def unexpected(*args, **kwargs):
        raise AssertionError("changed inputs must not start an executor")

    monkeypatch.setattr(numerical, "DockerCodeSession", unexpected)
    with pytest.raises(ValueError, match="committed snapshot"):
        _run(root, entrypoint, artifacts)


@pytest.mark.parametrize("alter_source", [False, True])
def test_mismatch_and_between_replay_source_changes_cannot_be_certified(tmp_path, monkeypatch, alter_source):
    root, entrypoint, artifacts = _workspace(tmp_path)
    calls = []

    class Session:
        def __init__(self, cwd, **kwargs):
            self.root = Path(cwd)

        def execute_result(self, code, *, language):
            calls.append(self.root)
            value = 3 if alter_source else 999
            (self.root / "scratch" / "estimates.csv").write_text(f"target_id,estimate,lower,upper\nmean,{value},1,5\n")
            if alter_source:
                (root / "scratch" / "methods" / "effect.py").write_text("# changed source after first replay\n")
            return CodeExecutionResultV1(status="success", output="", elapsed_seconds=0.01)

        def snapshot_scratch(self):
            return self.root / "scratch"

        def close(self):
            pass

    monkeypatch.setattr(numerical, "DockerCodeSession", Session)
    if alter_source:
        with pytest.raises(ValueError, match="committed project"):
            _run(root, entrypoint, artifacts)
    else:
        result = _run(root, entrypoint, artifacts)
        assert result.status == "result_mismatch" and result.estimates is None
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("offline", "outcomes", "status", "sessions"),
    [
        # Host load times out the first offline run; it reruns alone and scores normally.
        (True, ("timeout", "success", "success"), "reproduced", (False, True, False)),
        # A script that cannot finish alone still times out after its single rerun.
        (True, ("timeout", "timeout"), "execution_error", (False, True)),
        # Model outcomes and live replays are never rerun.
        (True, ("execution_error",), "execution_error", (False,)),
        (False, ("timeout",), "execution_error", (False,)),
    ],
)
def test_offline_replay_reruns_a_timeout_once_alone(tmp_path, monkeypatch, offline, outcomes, status, sessions):
    root, entrypoint, artifacts = _workspace(tmp_path)
    opened = []
    pending = list(outcomes)

    class Session:
        def __init__(self, cwd, *, exclusive, **kwargs):
            self.root = Path(cwd)
            opened.append(exclusive)

        def execute_result(self, code, *, language):
            assert not (self.root / "scratch" / "estimates.csv").exists()
            outcome = pending.pop(0)
            if outcome == "success":
                (self.root / "scratch" / "estimates.csv").write_text("target_id,estimate,lower,upper\nmean,3,1,5\n")
            return CodeExecutionResultV1(status=outcome, output="", elapsed_seconds=0.01)

        def snapshot_scratch(self):
            return self.root / "scratch"

        def close(self):
            pass

    monkeypatch.setattr(numerical, "DockerCodeSession", Session)
    monkeypatch.setattr(numerical, "offline_executor_capacity", lambda: offline)
    result = _run(root, entrypoint, artifacts)
    assert (result.status, tuple(opened), pending) == (status, sessions, [])
    # The load-induced timeout is discarded; a genuine failure keeps only its final run.
    recorded = ["success", "success"] if status == "reproduced" else [outcomes[-1]]
    assert [execution.status for execution in result.executions] == recorded


@pytest.mark.executor
def test_captured_submission_survives_later_workspace_edits(tmp_path):
    import shutil

    root, entrypoint, artifacts = _workspace(tmp_path)
    captured = tmp_path / "submission-0001"
    shutil.copytree(root / "scratch", captured)
    (root / "scratch" / entrypoint).write_text("raise RuntimeError('later project')\n")
    (root / "scratch" / "estimates.csv").write_text("invalid later answer\n")
    context = numerical.NumericalReportingContext(
        checkpoint_id="baseline", target_ids=("mean", "unreported"), evidence_artifacts=artifacts
    )
    (captured / "reporting_context.json").write_text(context.model_dump_json())
    result = numerical.replay_numerical_submission(
        workspace=root,
        submitted_project=captured,
        executor_limits=CodeExecutionLimitsV1(
            timeout_seconds=10, memory_mb=512, cpu_count=1, process_limit=32, workspace_mb=128
        ),
    )
    assert result.reproducible
    assert result.estimates[0].estimate == 3
    assert result.submitted_csv_sha256 == sha256_file(captured / "estimates.csv")
    assert result.checkpoint_id == "baseline"
    assert result.evidence_artifacts == artifacts


@pytest.mark.parametrize("entrypoints", [(), ("analysis.py", "analysis.R")])
def test_captured_replay_rejects_missing_or_ambiguous_program(tmp_path, entrypoints):
    context = numerical.NumericalReportingContext(
        checkpoint_id="study-one", target_ids=("effect",), evidence_artifacts=()
    )
    (tmp_path / "reporting_context.json").write_text(context.model_dump_json())
    for name in entrypoints:
        (tmp_path / name).write_text("# no execution allowed")
    result = numerical.replay_numerical_submission(
        workspace=tmp_path / "absent-workspace",
        submitted_project=tmp_path,
        executor_limits=CodeExecutionLimitsV1(),
    )
    assert result.status == "invalid_program"
    assert result.checkpoint_id == "study-one"
    assert result.target_ids == ("effect",)
    assert not result.executions


@pytest.mark.parametrize("column", ["estimate", "lower", "upper"])
@pytest.mark.parametrize("within", [True, False])
def test_numerical_replay_applies_declared_tolerances_to_each_field(tmp_path, monkeypatch, column, within):
    root, entrypoint, artifacts = _workspace(tmp_path)
    calls = []

    class Session:
        def __init__(self, cwd, **kwargs):
            self.root = Path(cwd)

        def execute_result(self, code, *, language):
            values = {"estimate": 3.0, "lower": 1.0, "upper": 5.0}
            threshold = max(
                numerical.TRIALEVAL_REPLAY_ABS_TOLERANCE_V1,
                numerical.TRIALEVAL_REPLAY_REL_TOLERANCE_V1 * abs(values[column]),
            )
            values[column] += threshold * (0.5 if within else 100)
            (self.root / "scratch" / "estimates.csv").write_text(
                "target_id,estimate,lower,upper\n" f"mean,{values['estimate']},{values['lower']},{values['upper']}\n"
            )
            calls.append(self.root)
            return CodeExecutionResultV1(status="success", output="", elapsed_seconds=0.01)

        def snapshot_scratch(self):
            return self.root / "scratch"

        def close(self):
            pass

    monkeypatch.setattr(numerical, "DockerCodeSession", Session)
    result = _run(root, entrypoint, artifacts)
    assert result.status == ("within_tolerance" if within else "result_mismatch")
    assert result.reproducible is within
    assert len(calls) == (2 if within else 1)
    if within:
        assert result.estimates[0].estimate == 3
        assert result.estimates[0].lower == 1
        assert result.estimates[0].upper == 5
    else:
        assert result.estimates is None


@pytest.mark.parametrize("language,condition", [("python", "r"), ("r", "python")])
def test_replay_rejects_entrypoint_outside_language_condition(tmp_path, monkeypatch, language, condition):
    root, entrypoint, artifacts = _workspace(tmp_path, language=language)

    def forbidden(*args, **kwargs):
        raise AssertionError("Wrong-language code must not execute")

    monkeypatch.setattr(numerical, "DockerCodeSession", forbidden)
    result = replay_numerical_program(
        workspace=root,
        entrypoint=entrypoint,
        target_ids=("mean",),
        evidence_artifacts=artifacts,
        executor_limits=CodeExecutionLimitsV1(),
        analysis_language=condition,
    )
    assert result.status == "invalid_program"
    assert "analysis condition" in result.error


@pytest.mark.executor
@pytest.mark.parametrize("language", ["python", "r"])
def test_real_reporting_helpers_capture_script_output_without_manual_csv(tmp_path, language):
    root, entrypoint, artifacts = _workspace(tmp_path, language)
    if language == "python":
        code = (
            "import csv\nfrom interface.reporting import targets, report\n"
            "with open('data/values.csv') as handle:\n"
            "    values = [float(row['value']) for row in csv.DictReader(handle)]\n"
            "report([dict(target_id=targets()[0], estimate=sum(values)/len(values), lower=1, upper=5)])\n"
        )
    else:
        code = (
            "source('interface/reporting.R')\n"
            "report(data.frame(target_id=targets()[1], estimate=mean(read.csv('data/values.csv')$value), lower=1, upper=5))\n"
        )
    (root / "scratch" / entrypoint).write_text(code)
    result = replay_numerical_program(
        workspace=root,
        entrypoint=entrypoint,
        target_ids=("mean",),
        evidence_artifacts=artifacts,
        executor_limits=CodeExecutionLimitsV1(
            timeout_seconds=10, memory_mb=512, cpu_count=1, process_limit=32, workspace_mb=128
        ),
        capture_result=True,
    )
    assert result.reproducible, result.error
    assert len(result.executions) == 2
    assert result.estimates[0].estimate == 3
    assert numerical.read_numerical_table(root / "scratch" / "estimates.csv", target_ids=("mean",)) == result.estimates
