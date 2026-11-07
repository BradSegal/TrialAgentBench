"""Participant-workspace tests for TrialDev execution."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest
from trialagentbench_verifier.grading_key import TrialDevWorldStratumV1

from trialagentbench.contracts.trialdev.release import (
    TrialDevMaterialV1,
    TrialDevReleaseArtifactV1,
    TrialDevTaskV1,
)
from trialagentbench.contracts.trialdev.run import TrialDevTaskRunV1
from trialagentbench.io import sha256_file, write_json_model
from trialagentbench.ports import CodeExecutionResultV1, ToolCall
from trialagentbench.trialdev.release import TrialDevEvaluatorRelease
from trialagentbench.trialdev.run import (
    stage_participant_workspace,
)
from trialagentbench.trialdev.task_agent import TrialDevTaskAgent


def _artifact(root: Path, relative_path: str) -> TrialDevReleaseArtifactV1:
    path = root / relative_path
    return TrialDevReleaseArtifactV1(
        relative_path=relative_path,
        sha256=sha256_file(path),
        byte_length=path.stat().st_size,
    )


class _ExecutingSession:
    """Execute generated inspection code in the participant workspace."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    def execute_result(self, code: str, *, language: str = "python") -> CodeExecutionResultV1:
        assert language == "python"
        namespace: dict[str, object] = {}
        try:
            compiled = compile(code, "<participant-tool>", "exec")
            current = Path.cwd()
            try:
                import os

                os.chdir(self.workspace)
                exec(compiled, namespace)
            finally:
                os.chdir(current)
        except (OSError, SyntaxError, ValueError) as error:
            return CodeExecutionResultV1(
                status="execution_error",
                output=f"{type(error).__name__}: {error}",
                elapsed_seconds=0.0,
            )
        return CodeExecutionResultV1(status="success", output="completed", elapsed_seconds=0.0)


def test_trialdev_inspect_parquet_generates_executable_python(tmp_path: Path) -> None:
    """Exercise the built-in inspection tool at its generated-code boundary."""

    data_path = tmp_path / "data" / "observational_cohort.parquet"
    data_path.parent.mkdir()
    pd.DataFrame({"participant_id": [1, 2], "outcome": [0.2, None]}).to_parquet(data_path)
    agent = object.__new__(TrialDevTaskAgent)
    agent.session = _ExecutingSession(tmp_path)
    agent.workdir = tmp_path
    agent.verbose = False

    result = agent._dispatch_local_tool(
        ToolCall(
            id="inspect-1",
            name="inspect_parquet",
            arguments='{"path":"data/observational_cohort.parquet"}',
        )
    )

    assert result.status == "success"


def test_trialdev_workspace_contains_only_declared_analyst_materials(tmp_path: Path) -> None:
    """Keep runtime contracts and private machine state outside the agent workspace."""

    world = tmp_path / "world"
    (world / "documents").mkdir(parents=True)
    (world / "data").mkdir()
    (world / "documents" / "brief.md").write_text("# Trial brief\n", encoding="utf-8")
    (world / "data" / "cohort.csv").write_text("participant_id,outcome\n1,2\n", encoding="utf-8")
    (world / "portfolio_state.json").write_text("{}\n", encoding="utf-8")
    (world / "portfolio_policy.json").write_text("{}\n", encoding="utf-8")
    task = TrialDevTaskV1(
        task_id="td-123456789abc",
        case_id="tdw-0001",
        stream="evidence_analysis",
        title="Comparative evidence analysis",
        clinical_question="Which candidate is supported by the available evidence?",
        available_materials=(
            TrialDevMaterialV1(relative_path="documents/brief.md", description="Trial brief"),
            TrialDevMaterialV1(relative_path="data/cohort.csv", description="Analysis cohort"),
        ),
        requested_outputs=("Report estimates and uncertainty.",),
    )
    participant = SimpleNamespace(world_root=lambda selected: world)
    release = cast(TrialDevEvaluatorRelease, SimpleNamespace(participant=participant))
    workspace = tmp_path / "workspace"

    stage_participant_workspace(release=release, task=task, workspace=workspace)

    assert sorted(path.relative_to(workspace).as_posix() for path in workspace.rglob("*") if path.is_file()) == [
        "data/cohort.csv",
        "documents/brief.md",
    ]


def test_trialdev_workspace_does_not_stage_alternate_request_or_runtime_mirrors(tmp_path: Path) -> None:
    """Give one task one entry point even when the release contains both streams."""

    world = tmp_path / "world"
    (world / "documents").mkdir(parents=True)
    (world / "documents" / "evidence_analysis_request.md").write_text("Evidence analysis\n", encoding="utf-8")
    (world / "documents" / "portfolio_management_request.md").write_text("Portfolio management\n", encoding="utf-8")
    (world / "portfolio_state.json").write_text("{}\n", encoding="utf-8")
    (world / "study_options.json").write_text("[]\n", encoding="utf-8")
    task = TrialDevTaskV1(
        task_id="td-123456789abc",
        case_id="tdw-0001",
        stream="evidence_analysis",
        title="Comparative evidence analysis",
        clinical_question="Which candidate is supported by the available evidence?",
        available_materials=(
            TrialDevMaterialV1(
                relative_path="documents/evidence_analysis_request.md",
                description="Analysis request",
            ),
        ),
        requested_outputs=("Report estimates and uncertainty.",),
    )
    participant = SimpleNamespace(world_root=lambda selected: world)
    release = cast(TrialDevEvaluatorRelease, SimpleNamespace(participant=participant))
    workspace = tmp_path / "workspace"

    stage_participant_workspace(release=release, task=task, workspace=workspace)

    assert (workspace / "documents" / "evidence_analysis_request.md").is_file()
    assert not (workspace / "documents" / "portfolio_management_request.md").exists()
    assert not (workspace / "portfolio_state.json").exists()
    assert not (workspace / "study_options.json").exists()


@pytest.mark.parametrize("recovery", [False, True])
def test_unsupported_task_cannot_start_or_resume_a_menu_run(tmp_path, recovery):
    from unittest.mock import Mock

    from trialagentbench.ports import CodeExecutionLimitsV1
    from trialagentbench.trialdev import run as owner

    task = SimpleNamespace(task_id="unsupported-task")
    release = SimpleNamespace(
        participant=SimpleNamespace(tasks=(task,)),
        authored_configuration=lambda selected: None,
        analysis_configuration=lambda selected: None,
    )
    (tmp_path / task.task_id).mkdir()
    with pytest.raises(ValueError, match="requires a current numerical or authored-programme configuration"):
        if recovery:
            owner.recover_terminal_trialdev_task(release=release, task_id=task.task_id, output_root=tmp_path)
        else:
            owner.run_trialdev_task(
                release=release,
                task_id=task.task_id,
                output_root=tmp_path,
                provider=Mock(),
                max_turns_per_step=10,
                max_tokens=100,
                max_context_characters=10000,
                watchdog_seconds=60,
                tool_choice="auto",
                executor_image=None,
                executor_limits=CodeExecutionLimitsV1(),
                resume=True,
            )


def test_submission_capture_preserves_local_source_modules(tmp_path):
    scratch = tmp_path / "workspace" / "scratch"
    (scratch / "methods").mkdir(parents=True)
    (scratch / "analysis.py").write_text("from methods.effect import estimate\n")
    (scratch / "methods" / "effect.py").write_text("estimate = 3\n")
    (scratch / "methods" / "derived.csv").write_text("estimate\n3\n")
    (scratch / "estimates.csv").write_text("target_id,estimate,lower,upper\neffect,3,2,4\n")
    agent = object.__new__(TrialDevTaskAgent)
    agent.workdir = scratch.parent
    agent.report_index = 0
    agent.executed_code = []
    agent.execution_outputs = []
    report = agent._capture_report("Estimated effect: 3.")
    assert (report.parent / "methods" / "effect.py").read_text() == "estimate = 3\n"
    assert not (report.parent / "methods" / "derived.csv").exists()
    assert (report.parent / "estimates.csv").read_text() == "target_id,estimate,lower,upper\neffect,3,2,4\n"


def test_csv_submission_uses_runner_targets_without_json_reports(tmp_path):
    from trialagentbench.ports.tool_input import ToolInputError

    scratch = tmp_path / "workspace" / "scratch"
    scratch.mkdir(parents=True)
    (scratch / "analysis.py").write_text("# recompute estimates.csv from participant evidence\n")
    agent = object.__new__(TrialDevTaskAgent)
    agent.workdir = scratch.parent
    agent.report_index = 0
    agent.executed_code = []
    agent.execution_outputs = []
    from trialagentbench_verifier.execution_contracts import ReplayInputArtifactV1
    from trialagentbench_verifier.numerical import NumericalReportingContext

    data = agent.workdir / "baseline.csv"
    data.write_text("participant,value\n1,3\n")
    agent.reporting_context = NumericalReportingContext(
        checkpoint_id="baseline",
        target_ids=("effect",),
        evidence_artifacts=(
            ReplayInputArtifactV1(
                relative_path="baseline.csv", sha256=sha256_file(data), byte_length=data.stat().st_size
            ),
        ),
    )
    missing = agent._capture_report("The numerical output was omitted.")
    with pytest.raises(ToolInputError, match="Invalid scratch/estimates.csv"):
        agent._validate_public_result(missing.parent)
    (scratch / "estimates.csv").write_text("target_id,estimate,lower,upper\neffect,-0.4,-0.7,-0.1\n")
    report = agent._capture_report("Estimated effect -0.4, interval [-0.7, -0.1].")
    agent._validate_public_result(report.parent)
    assert not (report.parent / "analysis_result.json").exists()
    captured_context = (report.parent / "reporting_context.json").read_text()
    assert NumericalReportingContext.model_validate_json(captured_context) == agent.reporting_context
    agent.reporting_context = agent.reporting_context.model_copy(update={"target_ids": ("later",)})
    agent._validate_public_result(report.parent)
    assert (report.parent / "reporting_context.json").read_text() == captured_context
    (scratch / "estimates.csv").write_text("target_id,estimate,lower,upper\nother,-0.4,-0.7,-0.1\n")
    rejected = agent._capture_report("Wrong target identity.")
    with pytest.raises(ToolInputError, match="unknown numerical target"):
        agent._validate_public_result(rejected.parent)
    assert (report.parent / "estimates.csv").read_text().startswith("target_id,estimate,lower,upper\neffect,")


def test_explicit_numerical_report_precedes_later_scratch_at_cutoff(tmp_path):
    from unittest.mock import Mock

    from trialagentbench_verifier.numerical import NumericalReportingContext

    from trialagentbench.ports.tool_input import ToolInputError

    agent = object.__new__(TrialDevTaskAgent)
    agent.workdir = tmp_path / "workspace"
    scratch = agent.workdir / "scratch"
    scratch.mkdir(parents=True)
    agent.session = Mock()
    agent.report_index = 0
    agent.executed_code = []
    agent.execution_outputs = []
    agent.reporting_context = NumericalReportingContext(
        checkpoint_id="study-one", target_ids=("effect",), evidence_artifacts=()
    )
    (scratch / "analysis.py").write_text("# original source\n")
    (scratch / "estimates.csv").write_text("target_id,estimate,lower,upper\neffect,1,0,2\n")
    report = agent._capture_report("Original report")
    agent._validate_public_result(report.parent)
    agent._commit_numerical_report(report.parent)
    (scratch / "analysis.py").write_text("# later source\n")
    (scratch / "estimates.csv").write_text("target_id,estimate,lower,upper\neffect,9,8,10\n")
    replacement = agent._capture_report("Replacement report")
    agent._validate_public_result(replacement.parent)
    with pytest.raises(ToolInputError, match="already been submitted"):
        agent._commit_numerical_report(replacement.parent)
    pending = agent._prepare_cutoff("next-readout", published=None)
    assert (pending / "analysis.py").read_text() == "# original source\n"
    assert (pending / "estimates.csv").read_text() == (report.parent / "estimates.csv").read_text()
    assert (pending / "report.md").read_text() == "Original report"
    agent.session.snapshot_scratch.assert_not_called()


@pytest.mark.parametrize("accepted", [False, True])
def test_automatic_cutoff_blocks_late_reports_only_after_action_success(tmp_path, accepted):
    from unittest.mock import Mock

    from trialagentbench_verifier.numerical import NumericalReportingContext

    from trialagentbench.ports.tool_input import ToolInputError

    agent = object.__new__(TrialDevTaskAgent)
    agent.workdir = tmp_path / "workspace"
    scratch = agent.workdir / "scratch"
    scratch.mkdir(parents=True)
    agent.session = Mock()
    agent.report_index = 0
    agent.executed_code = []
    agent.execution_outputs = []
    agent.reporting_context = NumericalReportingContext(
        checkpoint_id="study-one", target_ids=("effect",), evidence_artifacts=()
    )
    pending = agent._prepare_cutoff("readout", published=None)
    agent._prepare_cutoff("different-action", published=None)
    if accepted:
        destination = pending.parent.parent / "cutoffs" / pending.name
        destination.parent.mkdir(parents=True)
        pending.rename(destination)
    (scratch / "analysis.py").write_text("# answer prepared later\n")
    (scratch / "estimates.csv").write_text("target_id,estimate,lower,upper\neffect,1,0,2\n")
    report = agent._capture_report("Late report")
    agent._validate_public_result(report.parent)
    if accepted:
        with pytest.raises(ToolInputError, match="cutoff has already closed"):
            agent._commit_numerical_report(report.parent)
        assert agent._prepare_cutoff("different-action", published=None) is None
    else:
        agent._commit_numerical_report(report.parent)


def test_authored_release_entrypoint_routes_and_recovers_without_json_reports(tmp_path, monkeypatch):
    """The public runner preserves an authored stop and verifies its result bytes."""
    from unittest.mock import Mock

    from trialagentbench_verifier.authored_results import TrialDevNumericalRecovery
    from trialagentbench_verifier.numerical import NumericalReplay

    from trialagentbench.ports import CodeExecutionLimitsV1
    from trialagentbench.trialdev import run as owner
    from trialagentbench.trialdev.value import TrialDevFrozenProgrammeValue

    task = TrialDevTaskV1(
        task_id="td-123456789abc",
        case_id="tdw-0001",
        stream="portfolio_management",
        resources={"budget_usd": 15_000_000},
        title="Develop assets",
        clinical_question="Which asset should advance?",
        initial_target_ids=("candidate:clinical_outcome",),
        available_materials=(TrialDevMaterialV1(relative_path="documents/brief.md", description="Brief"),),
        requested_outputs=("Estimate effects and make development decisions.",),
    )
    world = tmp_path / "world"
    (world / "documents").mkdir(parents=True)
    (world / "documents" / "brief.md").write_text("Clinical brief")
    configuration = SimpleNamespace(
        budget_usd=15_000_000,
        recruitment_seed=7,
        pool_manifest_sha256="a" * 64,
        preferences=Mock(),
        initial_evidence=SimpleNamespace(context=SimpleNamespace(target_ids=task.initial_target_ids)),
    )
    release = SimpleNamespace(
        root=tmp_path / "release",
        authored_configuration=lambda selected: configuration,
        analysis_configuration=lambda selected: None,
        manifest=SimpleNamespace(release_id="release_" + "b" * 64, task_count=1),
        participant=SimpleNamespace(
            tasks=(task,), world_root=lambda selected: world, release_manifest_sha256="c" * 64
        ),
    )
    loop = Mock()
    constructor = Mock(return_value=loop)
    monkeypatch.setattr(owner, "TrialDevTaskAgent", constructor)
    result = owner.TrialDevProgrammeRun(
        execution_status="completed",
        error=None,
        evaluation=owner.TrialDevProgrammeEvaluation(
            programme_complete=True,
            studies=(),
            initial_replay=NumericalReplay(
                status="missing_result",
                entrypoint="",
                checkpoint_id="initial",
                target_ids=task.initial_target_ids,
                evidence_artifacts=(),
                error="No initial analysis was committed before termination.",
            ),
            initial_scores=(
                TrialDevNumericalRecovery(
                    target_id="candidate:clinical_outcome",
                    status="missing",
                    reference_value=0,
                    normalization_scale=1,
                    normalized_absolute_error=None,
                    normalized_interval_loss=None,
                ),
            ),
            value=TrialDevFrozenProgrammeValue(
                decision="stop",
                approved=False,
                clinical_value_usd=0,
                safety_cost_usd=0,
                study_cost_usd=0,
                time_cost_usd=0,
                net_value_usd=0,
            ),
        ),
    )
    task_root = tmp_path / "tasks" / task.task_id

    def execute(**kwargs):
        assert kwargs["initial_evidence"] is configuration.initial_evidence
        write_json_model(task_root / "programme_run.json", result)
        return result

    runner = Mock(side_effect=execute)
    monkeypatch.setattr(owner, "run_trialdev_programme", runner)
    arguments = dict(
        release=release,
        task_id=task.task_id,
        provider=Mock(),
        output_root=tmp_path / "tasks",
        max_turns_per_step=10,
        max_tokens=100,
        max_context_characters=10000,
        watchdog_seconds=60,
        tool_choice="auto",
        executor_image=None,
        executor_limits=CodeExecutionLimitsV1(),
    )
    summary = owner.run_trialdev_task(**arguments)
    assert summary.execution_status == "completed" and summary.final_state is None
    assert summary.programme_relative_path == "programme_run.json"
    assert not summary.report_relative_paths and not summary.grade_relative_paths
    assert not (task_root / "workspace" / "interface" / "analysis_result.schema.json").exists()
    assert constructor.call_args.kwargs["authored_programme"].pool_root == (
        release.root / "evaluation" / "worlds" / task.case_id / "pool"
    )
    assert owner.run_trialdev_task(**arguments, resume=True) == summary
    assert (
        owner.recover_terminal_trialdev_task(release=release, task_id=task.task_id, output_root=tmp_path / "tasks")
        == summary
    )
    assert runner.call_count == constructor.call_count == 1
    from trialagentbench.trialdev import ranking as ranking_owner
    from trialagentbench.trialdev.regrade import regrade_trialdev_task_directory

    with pytest.raises(ValueError, match="requires the invocation executor"):
        regrade_trialdev_task_directory(release=release, task_root=task_root)
    original_reader = ranking_owner.read_json_model
    invocation = SimpleNamespace(
        release_id=release.manifest.release_id,
        release_manifest_sha256=release.participant.release_manifest_sha256,
        scheduled_task_ids=(task.task_id,),
        checksum="d" * 64,
        model="scripted",
        scheduled_task_streams={task.task_id: task.stream},
    )
    monkeypatch.setattr(
        ranking_owner,
        "read_json_model",
        lambda model, path: invocation if path.name == "invocation.json" else original_reader(model, path),
    )
    stratum = TrialDevWorldStratumV1(
        world_id=task.case_id, clinical_profile_id="test", candidate_count=2, scenario_ref="scenario_" + "0" * 20
    )
    monkeypatch.setattr(release, "world_stratum", lambda world_id: stratum, raising=False)
    ranking = ranking_owner.rank_trialdev_run_v1(release=release, run_root=tmp_path / "tasks", task_roots=(task_root,))
    assert ranking.items[0].budget_usd == 15_000_000
    assert ranking.items[0].data_view == "raw"
    assert all(row.stream == "portfolio_management" for row in ranking.profile_means)
    assert ranking.items[0].net_development_value_usd == 0
    assert ranking.items[0].initial_target_statuses == {"candidate:clinical_outcome": "missing"}
    assert "programme" not in ranking.items[0].model_dump()
    assert next(row for row in ranking.profile_means if row.metric == "net_development_value_usd").mean == 0
    unfinished = ranking_owner.rank_trialdev_run_v1(release=release, run_root=tmp_path / "tasks", task_roots=())
    assert next(row for row in unfinished.profile_means if row.metric == "net_development_value_usd").mean is None
    import shutil

    altered = tmp_path / "altered"
    shutil.copytree(task_root, altered)
    (altered / "private" / "extra_commitment.json").write_text("{}")
    with pytest.raises(ValueError, match="finalized artifact inventory"):
        owner.verify_trialdev_task_run_artifacts_v1(task_root=altered, run=summary)
    redirected = tmp_path / "redirected"
    shutil.copytree(task_root, redirected)
    (redirected / "private" / "extra_sources").symlink_to(world, target_is_directory=True)
    with pytest.raises(ValueError, match="regular files and directories"):
        owner.verify_trialdev_task_run_artifacts_v1(task_root=redirected, run=summary)
    (task_root / "programme_run.json").write_text("{}")
    with pytest.raises(ValueError):
        owner.run_trialdev_task(**arguments, resume=True)


@pytest.mark.parametrize("status,error", [("completed", None), ("model_noncompletion", None)])
def test_authored_result_rejects_incomplete_evaluation_as_completion(status, error):
    """Persisted result parsing enforces the same completion boundary as execution."""
    from trialagentbench.trialdev.run import TrialDevProgrammeEvaluation, TrialDevProgrammeRun
    from trialagentbench.trialdev.value import TrialDevFrozenProgrammeValue

    incurred = TrialDevFrozenProgrammeValue(
        decision="stop", clinical_value_usd=0, safety_cost_usd=0, study_cost_usd=0, time_cost_usd=1, net_value_usd=-1
    )
    evaluation = TrialDevProgrammeEvaluation(programme_complete=False, value=incurred, studies=())
    with pytest.raises(ValueError, match="disagree"):
        TrialDevProgrammeRun(execution_status=status, error=error, evaluation=evaluation)
    with pytest.raises(ValueError, match="value"):
        TrialDevProgrammeEvaluation(programme_complete=True, value=None, studies=())


def test_observational_task_does_not_load_shared_world_development_configuration(tmp_path: Path) -> None:
    """A shared world does not turn a single analysis into a development trajectory."""
    from unittest.mock import Mock

    task = TrialDevTaskV1(
        task_id="td-123456789abc",
        case_id="tdw-0001",
        stream="evidence_analysis",
        title="Analyse evidence",
        clinical_question="What are the observed associations?",
        available_materials=(TrialDevMaterialV1(relative_path="data/cohort.parquet", description="Cohort"),),
        requested_outputs=("Estimate the associations and uncertainty.",),
    )
    configuration = tmp_path / "evaluation/tasks/td-123456789abc/authored_programme.json"
    configuration.parent.mkdir(parents=True)
    configuration.write_text("not an observational analysis configuration")
    release = TrialDevEvaluatorRelease(root=tmp_path, manifest=Mock(), participant=Mock(), evaluation_manifest=Mock())
    assert release.authored_configuration(task) is None


def test_standalone_configuration_is_task_owned_and_inventory_bound(tmp_path: Path) -> None:
    from unittest.mock import Mock

    from trialagentbench_verifier.execution_contracts import ReplayInputArtifactV1
    from trialagentbench_verifier.numerical import NumericalReportingContext

    from trialagentbench.contracts.trialdev.grading_key import TrialDevEvidenceTargetV1
    from trialagentbench.io import sha256_file
    from trialagentbench.trialdev.release import (
        TrialDevAnalysisConfiguration,
        TrialDevInitialEvidence,
        TrialDevReleaseError,
    )
    from trialagentbench.trialdev.value import TrialDevSponsorPreferences

    task = TrialDevTaskV1(
        task_id="td-123456789abc",
        case_id="tdw-0001",
        stream="evidence_analysis",
        title="Analyse",
        clinical_question="Estimate the observed association.",
        available_materials=(TrialDevMaterialV1(relative_path="data/cohort.parquet", description="Cohort"),),
        requested_outputs=("Estimate and interval.",),
    )
    public = tmp_path / "participant"
    public.mkdir()
    protocol = public / "protocol.md"
    protocol.write_text("original protocol")
    evidence = ReplayInputArtifactV1(
        relative_path="protocol.md", sha256=sha256_file(protocol), byte_length=protocol.stat().st_size
    )
    configuration = TrialDevAnalysisConfiguration(
        preferences=TrialDevSponsorPreferences(outcome_unit="DAS28", benefit_unit=1),
        initial_evidence=TrialDevInitialEvidence(
            context=NumericalReportingContext(
                checkpoint_id="initial", target_ids=("candidate:clinical_outcome",), evidence_artifacts=(evidence,)
            ),
            targets=(
                TrialDevEvidenceTargetV1(
                    asset_id="candidate",
                    endpoint="clinical_outcome",
                    unit="DAS28",
                    decision_scale=1,
                    comparison_sample_size=40,
                    point_identified=True,
                    estimand="standardized_observed_association",
                    oracle_value=0,
                ),
            ),
        ),
    )
    relative = f"tasks/{task.task_id}/analysis.json"
    evaluation = tmp_path / "evaluation"
    path = evaluation / relative
    write_json_model(path, configuration)
    artifact = _artifact(tmp_path, f"evaluation/{relative}")
    release = TrialDevEvaluatorRelease(
        root=tmp_path,
        manifest=Mock(),
        participant=Mock(
            world_root=lambda _: public,
            manifest=Mock(
                artifacts=(
                    TrialDevReleaseArtifactV1(
                        relative_path=f"participant/worlds/{task.case_id}/protocol.md",
                        sha256=evidence.sha256,
                        byte_length=evidence.byte_length,
                    ),
                )
            ),
        ),
        evaluation_manifest=Mock(artifacts=(artifact,)),
    )
    assert release.analysis_configuration(task) == configuration
    original = release.participant.manifest.artifacts
    release.participant.manifest.artifacts = (original[0].model_copy(update={"sha256": "0" * 64}),)
    with pytest.raises(ValueError, match="released evidence inventory"):
        release.analysis_configuration(task)
    release.participant.manifest.artifacts = original
    assert release.analysis_configuration(task.model_copy(update={"task_id": "td-ffffffffffff"})) is None
    assert (
        release.analysis_configuration(task.model_copy(update={"stream": type(task.stream).PORTFOLIO_MANAGEMENT}))
        is None
    )
    release.evaluation_manifest.artifacts = ()
    with pytest.raises(TrialDevReleaseError, match="evaluator inventory"):
        release.analysis_configuration(task)
    release.evaluation_manifest.artifacts = (artifact,)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(TrialDevReleaseError, match="evaluator inventory"):
        release.analysis_configuration(task)
    write_json_model(path, configuration)
    moved = path.with_suffix(".source")
    path.rename(moved)
    path.symlink_to(moved)
    with pytest.raises(TrialDevReleaseError, match="evaluator inventory"):
        release.analysis_configuration(task)


@pytest.mark.parametrize("defect", [None, "unbound", "wrong_stream", "mixed", "missing_binding"])
def test_standalone_result_inventory_has_one_execution_kind(defect) -> None:
    payload = dict(
        release_id="release_" + "a" * 64,
        release_manifest_sha256="b" * 64,
        task_id="td-123456789abc",
        stream="evidence_analysis",
        execution_status="completed",
        report_relative_paths=(),
        ingress_relative_paths=(),
        grade_relative_paths=(),
        analysis_relative_path="analysis_run.json",
        artifacts=(TrialDevReleaseArtifactV1(relative_path="analysis_run.json", sha256="c" * 64, byte_length=1),),
    )
    if defect == "unbound":
        payload["artifacts"] = ()
    elif defect == "wrong_stream":
        payload["stream"] = "portfolio_management"
    elif defect == "mixed":
        payload["programme_relative_path"] = "programme_run.json"
    elif defect == "missing_binding":
        payload["report_relative_paths"] = ("report.md",)
        payload["ingress_relative_paths"] = ("ingress.json",)
    if defect is not None:
        with pytest.raises(ValueError):
            TrialDevTaskRunV1.model_validate(payload)
    else:
        result = TrialDevTaskRunV1.model_validate(payload)
        assert TrialDevTaskRunV1.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("defect", [None, "extra", "symlink"])
def test_standalone_inventory_covers_the_exact_captured_sources(tmp_path: Path, defect) -> None:
    from trialagentbench.trialdev.run import _numerical_artifact_paths, verify_trialdev_task_run_artifacts_v1

    (tmp_path / "analysis_sources").mkdir()
    (tmp_path / "analysis_run.json").write_text("{}")
    source = tmp_path / "analysis_sources/analysis.py"
    source.write_text("print(1)")
    paths = _numerical_artifact_paths(tmp_path, programme=False)
    result = TrialDevTaskRunV1(
        release_id="release_" + "a" * 64,
        release_manifest_sha256="b" * 64,
        task_id="td-123456789abc",
        stream="evidence_analysis",
        execution_status="completed",
        report_relative_paths=(),
        ingress_relative_paths=(),
        grade_relative_paths=(),
        analysis_relative_path="analysis_run.json",
        artifacts=tuple(_artifact(tmp_path, path) for path in paths),
    )
    if defect == "extra":
        (tmp_path / "analysis_sources/extra.py").write_text("print(2)")
    elif defect == "symlink":
        (tmp_path / "analysis_sources/link.py").symlink_to(source)
    if defect is None:
        verify_trialdev_task_run_artifacts_v1(task_root=tmp_path, run=result)
    else:
        with pytest.raises(ValueError):
            verify_trialdev_task_run_artifacts_v1(task_root=tmp_path, run=result)
