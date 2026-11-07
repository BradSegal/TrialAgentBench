"""Public preflight exercises the real replay boundary without grading references."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from trialagentbench_verifier.execution_contracts import ReplayInputArtifactV1
from trialagentbench_verifier.executor import DockerCodeSession, executor_session_capacity
from trialagentbench_verifier.numerical import NumericalReportingContext

from trialagentbench.ports import CodeExecutionLimitsV1, CodeExecutionResultV1, ToolCall
from trialagentbench.tools.analysis import check_analysis_tool_v1
from trialagentbench.trialdev.letters import CONTROL_REFERENCE_NOTE


@pytest.mark.executor
@pytest.mark.parametrize("language", ["python", "r"])
@pytest.mark.parametrize(
    "defect", ["none", "root_output", "interactive_only", "stale", "future", "unknown", "ambiguous"]
)
def test_public_check_replays_the_saved_script_without_nested_capacity(tmp_path: Path, language, defect):
    workspace = tmp_path / "workspace"
    data = workspace / "data"
    data.mkdir(parents=True)
    evidence = data / "value.csv"
    evidence.write_text("value\n2\n")
    # This file exists in the host workspace but was not released at the cutoff.
    (data / "future.csv").write_text("value\n99\n")
    scratch = workspace / "scratch"
    scratch.mkdir()
    context = NumericalReportingContext(
        checkpoint_id="initial",
        target_ids=("effect",),
        evidence_artifacts=(
            ReplayInputArtifactV1(
                relative_path="data/value.csv",
                byte_length=evidence.stat().st_size,
                sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
            ),
        ),
    )
    target = "unknown" if defect == "unknown" else "effect"
    # An interactive result for the current target exists but is never what the check grades.
    (scratch / "estimates.csv").write_text("target_id,estimate,lower,upper\neffect,2,1,3\n")
    destination = "estimates.csv" if defect == "root_output" else "scratch/estimates.csv"
    source = "data/future.csv" if defect == "future" else "data/value.csv"
    if language == "python":
        (scratch / "helper.py").write_text("def interval(x):\n    return x-1, x+1\n")
        code = (
            "import csv\nfrom helper import interval\n"
            f"with open({source!r}) as f: value=float(next(csv.DictReader(f))['value'])\n"
            "lo,hi=interval(value)\n"
            f"with open({destination!r}, 'w') as f: f.write(f'target_id,estimate,lower,upper\\n{target},{{value}},{{lo}},{{hi}}\\n')\n"
        )
        if defect == "interactive_only":
            code = "print(interactive_value)\n" + code
        if defect == "stale":
            code = "pass\n"
        entrypoint = scratch / "analysis.py"
    else:
        (scratch / "helper.R").write_text("interval <- function(x) c(x-1,x+1)\n")
        code = (
            "source('scratch/helper.R')\n"
            f"value <- read.csv('{source}')$value[1]\n"
            "bounds <- interval(value)\n"
            f"write.csv(data.frame(target_id='{target}',estimate=value,lower=bounds[1],upper=bounds[2]), '{destination}', row.names=FALSE)\n"
        )
        if defect == "interactive_only":
            code = "print(interactive_value)\n" + code
        if defect == "stale":
            code = "invisible(NULL)\n"
        entrypoint = scratch / "analysis.R"
    entrypoint.write_text(code)
    if defect == "ambiguous":
        (scratch / ("analysis.R" if language == "python" else "analysis.py")).write_text("")
    before = {p.name: p.read_bytes() for p in scratch.iterdir()}
    limits = CodeExecutionLimitsV1(timeout_seconds=20, memory_mb=512)
    # A preflight cannot acquire a second slot while holding the interactive slot.
    with executor_session_capacity(1):
        session = DockerCodeSession(workspace, limits=limits)
        assert session.execute_result("interactive_value = 7", language=language).status == "success"
        replacement = None
        try:
            result, replacement = check_analysis_tool_v1(
                ToolCall(id="check", name="check_analysis", arguments=json.dumps({"language": language})),
                session=session,
                workspace=workspace,
                context=context,
                executor_limits=limits,
                executor_image=None,
                analysis_language="python_r",
            )
            report = json.loads(result.output)
            expected = {
                "none": "reproduced",
                "root_output": "missing_result",
                "interactive_only": "execution_error",
                "stale": "missing_result",
                "future": "execution_error",
                "unknown": "invalid_result",
                "ambiguous": "invalid_program",
            }[defect]
            assert report["status"] == expected
            assert report["reproducible"] is (defect == "none")
            assert result.status == ("success" if defect == "none" else "execution_error")
            assert report["checkpoint"] == "initial"
            assert report["missing_targets"] == ([] if defect == "none" else ["effect"])
            assert len(report["executions"]) == (2 if defect == "none" else 0 if defect == "ambiguous" else 1)
            assert (
                replacement.execute_result("print(interactive_value)", language=language).status == "execution_error"
            )
            assert replacement.execute_result("print(2)", language=language).status == "success"
            replacement.snapshot_scratch()
            after = {name: (scratch / name).read_bytes() for name in before}
            if defect == "none":
                # Only a reproduced check refreshes the interactive result from the saved script.
                before.pop("estimates.csv")
                assert b"effect" in after.pop("estimates.csv")
            assert after == before
        finally:
            session.close()
            if replacement is not None:
                replacement.close()


@pytest.mark.executor
def test_saved_script_runs_directly_with_the_reporting_interface(tmp_path: Path):
    """`python scratch/analysis.py` resolves `interface` as the in-session entrypoint does."""
    from trialagentbench_verifier.numerical import stage_reporting_interface

    workspace = tmp_path / "workspace"
    (workspace / "scratch").mkdir(parents=True)
    stage_reporting_interface(workspace, ("effect",))
    (workspace / "scratch" / "analysis.py").write_text(
        "from interface.reporting import targets, report\n"
        "report([dict(target_id=t, estimate=1.0, lower=0.0, upper=2.0) for t in targets()])\n"
    )
    session = DockerCodeSession(workspace, limits=CodeExecutionLimitsV1(timeout_seconds=60, memory_mb=512))
    try:
        result = session.execute_result(
            "import subprocess, sys\n"
            "run = subprocess.run([sys.executable, 'scratch/analysis.py'], capture_output=True, text=True)\n"
            "print(run.returncode, run.stderr.strip()[-300:])\n"
            "print(open('scratch/estimates.csv').read() if run.returncode == 0 else '')"
        )
    finally:
        session.close()
    assert result.output.splitlines()[:3] == [
        "0 ",
        "target_id,estimate,lower,upper",
        "effect,1.0,0.0,2.0",
    ], result.output


@pytest.mark.parametrize("interactive", ["earlier_checkpoint", "missing", "current"])
def test_check_grades_only_fresh_runs_of_the_saved_script(tmp_path, monkeypatch, interactive):
    """A stale or missing interactive scratch/estimates.csv never decides the check."""
    import subprocess
    import sys

    from trialagentbench_verifier import numerical

    from trialagentbench.execution_policy import TRIALDEV_RELEASE_BUDGET_V1
    from trialagentbench.tools import analysis as owner

    class HostSession:
        """Run the entrypoint code on the host in the staged replay workspace, as the executor does."""

        def __init__(self, cwd, **kwargs):
            self.root = Path(cwd)

        def execute_result(self, code, *, language):
            run = subprocess.run([sys.executable, "-c", code], cwd=self.root, capture_output=True, text=True)
            return CodeExecutionResultV1(
                status="success" if run.returncode == 0 else "execution_error",
                language=language,
                output=run.stdout + run.stderr,
                elapsed_seconds=0.01,
            )

        def snapshot_scratch(self):
            return self.root / "scratch"

        def close(self):
            pass

    monkeypatch.setattr(numerical, "DockerCodeSession", HostSession)
    monkeypatch.setattr(owner, "DockerCodeSession", HostSession)
    workspace = tmp_path / "workspace"
    (workspace / "scratch").mkdir(parents=True)
    # A correct saved script for the current checkpoint (the reported reproduction).
    (workspace / "scratch" / "analysis.py").write_text(
        "from interface.reporting import targets, report\n"
        "report([dict(target_id=t, estimate=0.0, lower=-1.0, upper=1.0) for t in targets()])\n"
    )
    target = "asset-1:standard:vs_control:clinical_outcome"
    earlier = {
        "earlier_checkpoint": "asset-1:clinical_outcome,-0.7,-1.1,-0.3\n",
        "current": f"{target},5,4,6\n",
    }
    if interactive in earlier:
        (workspace / "scratch" / "estimates.csv").write_text("target_id,estimate,lower,upper\n" + earlier[interactive])
    result, _ = check_analysis_tool_v1(
        ToolCall(id="check", name="check_analysis", arguments="{}"),
        session=HostSession(workspace),
        workspace=workspace,
        context=NumericalReportingContext(
            checkpoint_id="evidence/study-101-look-1", target_ids=(target,), evidence_artifacts=()
        ),
        executor_limits=TRIALDEV_RELEASE_BUDGET_V1.code_execution,
        executor_image=None,
        analysis_language="python",
    )
    report = json.loads(result.output)
    assert (result.status, report["status"], report["error"]) == ("success", "reproduced", None)
    assert len(report["executions"]) == 2 and report["missing_targets"] == []
    # The interactive file now holds the saved script's output, not the earlier rows.
    assert (workspace / "scratch" / "estimates.csv").read_text().splitlines()[1:] == [f"{target},0.0,-1.0,1.0"]


def test_readout_letter_names_files_one_clock_and_every_requested_result_compactly():
    from trialagentbench.trialdev.actions import TrialDevProgrammeStatus
    from trialagentbench.trialdev.letters import readout_letter
    from trialagentbench.trialdev.materialization import (
        TrialDevReadoutArm,
        TrialDevReadoutManifest,
        TrialDevReadoutTarget,
    )
    from trialagentbench.trialdev.resources import TrialDevStudyCost, trialdev_phase_tariff

    targets = tuple(
        TrialDevReadoutTarget(
            target_id=f"candidate-{i}:standard:vs_control:{endpoint}",
            arm_id=f"candidate-{i}",
            endpoint=endpoint,
            unit="risk difference" if endpoint == "safety_risk" else "score points",
        )
        for i in range(5)
        for endpoint in ("clinical_outcome", "safety_risk")
    )
    manifest = TrialDevReadoutManifest(
        study_id="study-201",
        outcome_object="Score",
        readout_day=200,
        follow_up_days=168,
        assessment_days=(28, 56, 168),
        surveillance_days=(28, 56, 168),
        arms=(
            TrialDevReadoutArm(arm_id="control", asset_id="control", dose=None),
            *(TrialDevReadoutArm(arm_id=f"candidate-{i}", asset_id=f"cnd-{i}", dose="standard") for i in range(5)),
        ),
        targets=targets,
        tariff=trialdev_phase_tariff("exploration"),
        sites=10,
        screened=600,
        cost=TrialDevStudyCost(accrued_usd=1.0, remaining_liability_usd=0.0),
    )
    status = TrialDevProgrammeStatus(
        programme_day=200,
        horizon_days=2190,
        budget_usd=1000,
        spent_usd=100,
        committed_followup_usd=50,
        available_usd=850,
        studies=(),
        terminal=None,
    )
    letter = readout_letter(
        manifest,
        folder="evidence/study-201-look-1",
        look=1,
        status=status,
        previous_capture_error="missing targets initial-effect",
    )
    assert len(letter) < 4000
    assert "`evidence/study-201-look-1/`" in letter and "enrollment_day" in letter and "day 168" in letter
    assert "missing targets initial-effect" in letter
    assert [line.split("`")[1] for line in letter.splitlines() if line.startswith("| `")] == [
        target.target_id for target in targets
    ]
    assert CONTROL_REFERENCE_NOTE in letter
    assert "These files are final" in letter


def test_trialdev_instructions_state_the_shared_clock_and_look_rules():
    """The tools and system prompt say how time, looks and released readouts behave."""
    from trialagentbench.trialdev.actions import trialdev_programme_tools
    from trialagentbench.trialdev.run import trialdev_programme_system_prompt

    descriptions = {tool["function"]["name"]: tool["function"]["description"] for tool in trialdev_programme_tools()}
    readout, wait = descriptions["request_readout"], descriptions["wait_until"]
    assert "shared programme clock" in readout and "never changes" in readout
    assert "one planned look, even if few or no participants have completed" in readout
    assert "shared programme clock" in wait and "deadline passes" in wait and "is kept" in wait
    for view in ("raw", "analysis_ready"):
        assert "All studies share one programme clock" in trialdev_programme_system_prompt(view)


def test_trialdev_reporting_names_the_estimand_without_prescribing_an_analysis():
    """Reference treatment, natural units and direction suffice without formula coaching."""
    from trialagentbench.trialdev.release import TrialDevInitialEvidence
    from trialagentbench.trialdev.run import _initial_reporting_instructions

    initial = TrialDevInitialEvidence.model_construct(
        context=SimpleNamespace(target_ids=("cnp-22:effect",)),
        targets=(SimpleNamespace(asset_id="cnp-22", endpoint="clinical_outcome", unit="points"),),
    )
    instructions = _initial_reporting_instructions(initial)
    assert "case-mix-standardized mean difference (natural units)" in instructions
    assert "cnp-22 minus control" in instructions
    assert instructions.endswith(CONTROL_REFERENCE_NOTE)
    assert all(term not in instructions for term in ("Treatment(", "relevel(", "categorical coding", "formula"))


@pytest.mark.executor
def test_trialdev_preflight_resume_reuses_committed_result(tmp_path, monkeypatch):
    from unittest.mock import Mock

    from trialagentbench.ports import LLMResponse
    from trialagentbench.trialdev import task_agent as owner
    from trialagentbench.util.agent_runtime import AgentTurn

    workspace = tmp_path / "workspace"
    scratch = workspace / "scratch"
    scratch.mkdir(parents=True)
    csv = "target_id,estimate,lower,upper\neffect,2,1,3\n"
    (scratch / "estimates.csv").write_text(csv)
    (scratch / "analysis.py").write_text(
        f"from pathlib import Path\nPath('scratch/estimates.csv').write_text({csv!r})\n"
    )
    arguments = dict(
        provider=Mock(),
        workdir=workspace,
        system_prompt="Analyse.",
        program_id="programme",
        scenario_id="case",
        objective_id="initial",
        max_turns_per_step=2,
        reporting_context=NumericalReportingContext(
            checkpoint_id="initial", target_ids=("effect",), evidence_artifacts=()
        ),
    )
    calls = []
    checked = []
    real_check = owner.check_analysis_tool_v1

    def check(*args, **kwargs):
        checked.append(kwargs.get("reporting_cutoff", False))
        return real_check(*args, **kwargs)

    monkeypatch.setattr(owner, "check_analysis_tool_v1", check)

    def provider_turn(**kwargs):
        calls.append(kwargs["turn"])
        call = (
            ToolCall(id="check", name="check_analysis", arguments="{}")
            if kwargs["turn"] == 1
            else ToolCall(
                id="report", name="submit_analysis_report", arguments='{"content":"Estimate 2, interval 1 to 3."}'
            )
        )
        return AgentTurn(
            response=LLMResponse(tool_calls=[call]),
            message={"role": "assistant", "content": "Proceed."},
            elapsed_seconds=0.0,
        )

    first = owner.TrialDevTaskAgent(**arguments)
    monkeypatch.setattr(first, "_provider_turn", provider_turn)
    first.begin_step(phase_id="initial", step_id="analysis")
    first.append_user_message("Analyse the evidence.")
    execute = first._execute_checkpointed_tool

    def interrupt(call):
        execute(call)
        raise OSError("interrupted after commit")

    monkeypatch.setattr(first, "_execute_checkpointed_tool", interrupt)
    try:
        with pytest.raises(OSError, match="after commit"):
            first.run_until_report()
    finally:
        first.close()
    resumed = owner.TrialDevTaskAgent(**arguments, resume=True)
    monkeypatch.setattr(resumed, "_provider_turn", provider_turn)
    try:
        report = resumed.run_until_report()
        assert report.content == "Estimate 2, interval 1 to 3."
        assert len(resumed.execution_outputs) == 1
        result = resumed.execution_outputs[0]["result"]
        assert json.loads(result["output"])["reproducible"] is True
    finally:
        resumed.close()
    assert checked == [False, True]  # One resumed check, then the new submission capture.
    assert calls == [1, 2]
    assert (scratch / "estimates.csv").read_text() == csv
