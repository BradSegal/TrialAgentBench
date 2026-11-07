"""Initial association grading uses the same numerical loss as acquired trials."""

import pytest
from trialagentbench_verifier.grading_key import TrialDevWorldStratumV1
from trialagentbench_verifier.numerical import NumericalEstimate, NumericalReplay

from trialagentbench.contracts.trialdev.grading_key import TrialDevEvidenceTargetV1
from trialagentbench.ports import CodeExecutionResultV1
from trialagentbench.trialdev.target_grading import grade_trialdev_observational_numbers
from trialagentbench.trialdev.value import TrialDevSponsorPreferences


@pytest.fixture(autouse=True)
def forbid_float_grading(monkeypatch):
    """Every runtime, resume and regrade scenario must bypass the frozen grader."""
    from trialagentbench.trialdev import targets

    def float_grading_called(*args, **kwargs):
        raise AssertionError("runtime grading reached frozen float implementation")

    monkeypatch.setattr(targets, "_numerical_recovery", float_grading_called)
    monkeypatch.setattr(targets, "grade_trialdev_observational_numbers", float_grading_called)
    monkeypatch.setattr(targets, "grade_trialdev_readout_numbers", float_grading_called)


def _inputs():
    targets = tuple(
        TrialDevEvidenceTargetV1(
            asset_id="candidate",
            endpoint=endpoint,
            unit=unit,
            decision_scale=1,
            comparison_sample_size=100,
            point_identified=True,
            estimand="standardized_observed_association",
            oracle_value=value,
        )
        for endpoint, unit, value in (("clinical_outcome", "DAS28", 2), ("safety_risk", "risk difference", 0.01))
    )
    replay = NumericalReplay(
        status="reproduced",
        entrypoint="analysis.py",
        checkpoint_id="initial",
        target_ids=("candidate:clinical_outcome", "candidate:safety_risk"),
        evidence_artifacts=(),
        submitted_csv_sha256="a" * 64,
        replayed_csv_sha256s=("a" * 64, "a" * 64),
        executions=(CodeExecutionResultV1(status="success", output="", elapsed_seconds=1),) * 2,
        estimates=(NumericalEstimate(target_id="candidate:clinical_outcome", estimate=3, lower=1, upper=4),),
    )
    return targets, replay, TrialDevSponsorPreferences(outcome_unit="DAS28", benefit_unit=1)


@pytest.mark.parametrize("defect", ["causal_effect", "population_linear_projection", "missing", "nonidentified"])
def test_initial_key_rejects_unsupported_target_meanings_at_ingress(defect):
    from pydantic import ValidationError
    from trialagentbench_verifier.development import _InitialTarget

    targets, _, _ = _inputs()
    payload = targets[0].model_dump(mode="json")
    assert _InitialTarget.model_validate(payload).estimand == "standardized_observed_association"
    if defect == "missing":
        del payload["estimand"]
    elif defect == "nonidentified":
        payload["point_identified"] = False
    else:
        payload["estimand"] = defect
    for contract in (TrialDevEvidenceTargetV1, _InitialTarget):
        with pytest.raises(ValidationError):
            contract.model_validate(payload)


def test_initial_association_scores_signal_and_interval_without_method_metadata():
    targets, replay, preferences = _inputs()
    clinical, safety = grade_trialdev_observational_numbers(targets=targets, replay=replay, preferences=preferences)
    assert clinical.normalized_absolute_error == 1
    assert clinical.normalized_interval_loss == 3
    assert clinical.loss == pytest.approx(2)
    assert safety.status == "missing" and safety.loss is None
    from trialagentbench_verifier.authored_results import mean_numerical_loss, numerical_target_statuses

    assert mean_numerical_loss((clinical, safety)) is None
    assert numerical_target_statuses((clinical, safety)) == {
        "candidate:clinical_outcome": "reported",
        "candidate:safety_risk": "missing",
    }
    with pytest.raises(ValueError, match="repeats an identity"):
        numerical_target_statuses((clinical, clinical))
    failed = replay.model_copy(update={"status": "invalid_result", "estimates": None})
    rows = grade_trialdev_observational_numbers(targets=targets, replay=failed, preferences=preferences)
    assert len(rows) == 2 and all(row.loss is None and row.status == "invalid_replay" for row in rows)


@pytest.mark.parametrize("defect", ["causal", "nonidentified", "units", "duplicate", "inventory"])
def test_initial_association_rejects_incompatible_scientific_keys(defect):
    targets, replay, preferences = _inputs()
    if defect == "causal":
        targets = (targets[0].model_copy(update={"estimand": "causal_effect"}), targets[1])
    elif defect == "nonidentified":
        targets = (targets[0].model_copy(update={"point_identified": False}), targets[1])
    elif defect == "units":
        preferences = preferences.model_copy(update={"outcome_unit": "different units"})
    elif defect == "duplicate":
        targets = (targets[0], targets[0])
    else:
        replay = replay.model_copy(update={"target_ids": ("unassigned",)})
    with pytest.raises(ValueError):
        grade_trialdev_observational_numbers(targets=targets, replay=replay, preferences=preferences)


def test_programme_evaluation_retains_initial_report_and_missingness(tmp_path, monkeypatch):
    from trialagentbench_test_helpers import write_minimal_trialdev_pool
    from trialagentbench_verifier.numerical import NumericalReportingContext

    from trialagentbench.io import sha256_file, write_json_model
    from trialagentbench.ports import CodeExecutionLimitsV1
    from trialagentbench.trialdev.actions import TrialDevActionSession
    from trialagentbench.trialdev.resources import TrialDevBudgetLedger
    from trialagentbench.trialdev.run import TrialDevInitialEvidence, evaluate_trialdev_programme

    targets, replay, preferences = _inputs()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    budget = tmp_path / "budget.json"
    pool = write_minimal_trialdev_pool(tmp_path / "pool")
    write_json_model(
        budget, TrialDevBudgetLedger(budget_usd=1000, pool_manifest_sha256=sha256_file(pool / "manifest.json"))
    )
    programme = TrialDevActionSession(pool, budget, workspace / "evidence", tmp_path / "commits", 7)
    context = NumericalReportingContext(checkpoint_id="initial", target_ids=replay.target_ids, evidence_artifacts=())
    initial = TrialDevInitialEvidence(context=context, targets=targets)
    kwargs = dict(
        workspace=workspace,
        programme=programme,
        preferences=preferences,
        executor_limits=CodeExecutionLimitsV1(),
        initial_evidence=initial,
    )
    missing = evaluate_trialdev_programme(**kwargs)
    # The unfinished programme incurred nothing by day 0, so it is valued as a day-0 stop.
    assert (missing.value.decision, missing.value.net_value_usd) == ("stop", 0)
    assert missing.studies == ()
    assert missing.initial_replay.status == "missing_result"
    assert len(missing.initial_scores) == 2 and all(row.loss is None for row in missing.initial_scores)
    captured = tmp_path / "analysis_sources" / "cutoffs" / "first-readout"
    write_json_model(captured / "reporting_context.json", context)
    monkeypatch.setattr("trialagentbench.trialdev.run.replay_numerical_submission", lambda **kwargs: replay)
    observed = evaluate_trialdev_programme(**kwargs)
    assert observed.initial_replay == replay
    assert observed.initial_scores[0].loss == pytest.approx(2)
    assert observed.initial_scores[1].status == "missing"
    assert observed.value == missing.value and observed.studies == ()
    write_json_model(captured / "reporting_context.json", context.model_copy(update={"target_ids": ("changed",)}))
    with pytest.raises(ValueError, match="release-bound evidence context"):
        evaluate_trialdev_programme(**kwargs)


@pytest.mark.parametrize("defect", ["missing", "extra", "reordered", "duplicate"])
def test_initial_release_rejects_mismatched_requested_and_graded_targets(defect):
    from trialagentbench_verifier.numerical import NumericalReportingContext

    from trialagentbench.trialdev.release import TrialDevInitialEvidence

    targets, replay, _ = _inputs()
    identities = replay.target_ids
    if defect == "missing":
        identities = identities[:1]
    elif defect == "extra":
        identities = (*identities, "other:clinical_outcome")
    elif defect == "reordered":
        identities = tuple(reversed(identities))
    else:
        targets = (targets[0], targets[0])
    context = NumericalReportingContext(checkpoint_id="initial", target_ids=identities, evidence_artifacts=())
    with pytest.raises(ValueError, match="exact grading target inventory"):
        TrialDevInitialEvidence(context=context, targets=targets)


@pytest.mark.parametrize("outcome", ["submitted", "turn_limit", "watchdog"])
@pytest.mark.parametrize("public_entrypoint", [True, False])
def test_standalone_runner_scores_analysis_without_programme_state(tmp_path, monkeypatch, outcome, public_entrypoint):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from trialagentbench_verifier.numerical import NumericalReportingContext

    from trialagentbench.io import read_json_model, write_json_model
    from trialagentbench.ports import CodeExecutionLimitsV1
    from trialagentbench.trialdev.release import TrialDevAnalysisConfiguration, TrialDevInitialEvidence
    from trialagentbench.trialdev.run import TrialDevAnalysisRun, run_trialdev_analysis
    from trialagentbench.trialdev.task_agent import AgentTurnLimitExceeded
    from trialagentbench.util.runtime_context import RuntimeDeadlineExceeded

    submitted = outcome == "submitted"
    targets, replay, preferences = _inputs()
    context = NumericalReportingContext(checkpoint_id="initial", target_ids=replay.target_ids, evidence_artifacts=())
    configuration = TrialDevAnalysisConfiguration(
        preferences=preferences, initial_evidence=TrialDevInitialEvidence(context=context, targets=targets)
    )
    task_root = tmp_path / "tasks" / "td-123456789abc" if public_entrypoint else tmp_path
    workspace = task_root / "workspace"
    if not public_entrypoint:
        workspace.mkdir()
    loop = Mock()
    loop.workdir = workspace
    loop.authored_programme = None
    loop.reporting_context = None
    loop.active_prompt_index = None
    capture = task_root / "analysis_sources" / "submission-test"

    def submit():
        capture.mkdir(parents=True)
        (capture / "report.md").write_text("Analysis report")
        write_json_model(capture / "reporting_context.json", context)
        return SimpleNamespace(path=capture / "report.md")

    if submitted:
        loop.run_until_report.side_effect = submit
    else:
        # A task time limit is a task outcome (F-2): it never stops the run or its siblings.
        loop.run_until_report.side_effect = (
            AgentTurnLimitExceeded("Turn budget exhausted")
            if outcome == "turn_limit"
            else RuntimeDeadlineExceeded("Task watchdog exhausted during report check_analysis")
        )
    replay_call = Mock(return_value=replay)
    monkeypatch.setattr("trialagentbench.trialdev.run.replay_numerical_submission", replay_call)
    if public_entrypoint:
        from trialagentbench.contracts.trialdev.release import TrialDevMaterialV1, TrialDevTaskV1
        from trialagentbench.trialdev.run import run_trialdev_task

        task = TrialDevTaskV1(
            task_id="td-123456789abc",
            case_id="tdw-0001",
            stream="evidence_analysis",
            title="Analyse assets",
            clinical_question="Estimate the associations.",
            available_materials=(TrialDevMaterialV1(relative_path="brief.md", description="Brief"),),
            requested_outputs=("Estimate associations with uncertainty.",),
        )
        world = tmp_path / "world"
        world.mkdir()
        (world / "brief.md").write_text("Public clinical brief")
        release = SimpleNamespace(
            authored_configuration=lambda task: None,
            analysis_configuration=lambda task: configuration,
            manifest=SimpleNamespace(release_id="release_" + "b" * 64, task_count=1),
            participant=SimpleNamespace(
                tasks=(task,),
                world_root=lambda task: world,
                release_manifest_sha256="c" * 64,
            ),
        )
        constructor = Mock(return_value=loop)
        monkeypatch.setattr("trialagentbench.trialdev.run.TrialDevTaskAgent", constructor)
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
        summary = run_trialdev_task(**arguments)
        result = read_json_model(TrialDevAnalysisRun, task_root / "analysis_run.json")
        assert summary.analysis_relative_path == "analysis_run.json"
        assert summary.programme_relative_path is None
        assert constructor.call_args.kwargs["authored_programme"] is None
        assert constructor.call_args.kwargs["reporting_context"] == context
        assert (workspace / "interface" / "targets.csv").read_text().splitlines() == ["target_id", *context.target_ids]
        assert (workspace / "brief.md").read_text() == "Public clinical brief"
        assert "`brief.md`: Brief" in loop.append_user_message.call_args.args[0]
        assert "paths relative to the task workspace" in loop.append_user_message.call_args.args[0]
        assert run_trialdev_task(**arguments, resume=True) == summary
        assert constructor.call_count == 1
        from trialagentbench.trialdev import ranking as ranking_owner
        from trialagentbench.trialdev.regrade import regrade_trialdev_task_directory

        with pytest.raises(ValueError, match="requires the invocation executor"):
            regrade_trialdev_task_directory(release=release, task_root=task_root)
        updated = regrade_trialdev_task_directory(
            release=release,
            task_root=task_root,
            executor_limits=CodeExecutionLimitsV1(),
            executor_image="test-image",
        )
        assert updated == summary
        release.root = tmp_path / "release"
        original_reader = ranking_owner.read_json_model
        invocation = SimpleNamespace(
            release_id=release.manifest.release_id,
            release_manifest_sha256=release.participant.release_manifest_sha256,
            scheduled_task_ids=(task.task_id,),
            checksum="d" * 64,
            model="scripted",
            scheduled_task_streams={task.task_id: task.stream},
        )
        stratum = TrialDevWorldStratumV1(
            world_id=task.case_id, clinical_profile_id="test", candidate_count=2, scenario_ref="scenario_" + "0" * 20
        )
        monkeypatch.setattr(release, "world_stratum", lambda world_id: stratum, raising=False)
        monkeypatch.setattr(
            ranking_owner,
            "read_json_model",
            lambda model, path: invocation if path.name == "invocation.json" else original_reader(model, path),
        )
        ranking = ranking_owner.rank_trialdev_run_v1(
            release=release,
            run_root=tmp_path / "tasks",
            task_roots=(task_root,),
        )
        assert ranking.items[0].observational_loss is None  # The second scheduled target is missing.
        assert "programme" not in ranking.items[0].model_dump()
        assert ranking.items[0].net_development_value_usd is None
        unfinished = ranking_owner.rank_trialdev_run_v1(
            release=release,
            run_root=tmp_path / "tasks",
            task_roots=(),
        )
        assert unfinished.items[0].execution_status == "incomplete_run"
        assert unfinished.items[0].observational_loss is None
        assert all(row.mean is None for row in unfinished.profile_means)
        paired_task = task.model_copy(update={"task_id": "td-abcdef123456", "data_view": "analysis_ready"})
        with monkeypatch.context() as pairing:
            pairing.setattr(release.participant, "tasks", (task, paired_task))
            pairing.setattr(invocation, "scheduled_task_ids", (task.task_id, paired_task.task_id))
            pairing.setattr(
                invocation, "scheduled_task_streams", {task.task_id: task.stream, paired_task.task_id: task.stream}
            )
            paired_ranking = ranking_owner.rank_trialdev_run_v1(
                release=release,
                run_root=tmp_path / "tasks",
                task_roots=(task_root,),
            )
        means = {(row.stream, row.data_view): row.mean for row in paired_ranking.profile_means}
        assert means == {
            ("evidence_analysis", "raw"): None,
            ("evidence_analysis", "analysis_ready"): None,
        }
    else:
        result = run_trialdev_analysis(
            loop=loop,
            prompt="Estimate the associations.",
            configuration=configuration,
            executor_limits=CodeExecutionLimitsV1(),
        )
    assert result.execution_status == ("completed" if submitted else "model_noncompletion")
    assert (result.error is None) == submitted
    if outcome == "watchdog":
        assert result.error == "Task watchdog exhausted during report check_analysis"
    assert len(result.scores) == 2
    assert result.scores[1].loss is None
    assert result.scores[0].loss == (pytest.approx(2) if submitted else None)
    assert replay_call.call_count == int(submitted) * (2 if public_entrypoint else 1)
    assert "candidate:clinical_outcome" in loop.append_user_message.call_args.args[0]
    loop.close.assert_called_once()
    assert read_json_model(TrialDevAnalysisRun, task_root / "analysis_run.json") == result
    assert not (task_root / "private/budget.json").exists()
    assert not (task_root / "programme_run.json").exists()


def test_standalone_runner_does_not_convert_operation_timeout_to_noncompletion(tmp_path):
    from unittest.mock import Mock

    from trialagentbench_verifier.numerical import NumericalReportingContext

    from trialagentbench.ports import CodeExecutionLimitsV1
    from trialagentbench.trialdev.release import TrialDevAnalysisConfiguration, TrialDevInitialEvidence
    from trialagentbench.trialdev.run import run_trialdev_analysis

    targets, replay, preferences = _inputs()
    context = NumericalReportingContext(checkpoint_id="initial", target_ids=replay.target_ids, evidence_artifacts=())
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    loop = Mock(workdir=workspace, authored_programme=None, reporting_context=None, active_prompt_index=None)
    interruption = TimeoutError("An operation timed out before the task watchdog expired")
    loop.run_until_report.side_effect = interruption
    with pytest.raises(TimeoutError) as raised:
        run_trialdev_analysis(
            loop=loop,
            prompt="Estimate the associations.",
            configuration=TrialDevAnalysisConfiguration(
                preferences=preferences, initial_evidence=TrialDevInitialEvidence(context=context, targets=targets)
            ),
            executor_limits=CodeExecutionLimitsV1(),
        )
    assert raised.value is interruption
    loop.close.assert_called_once()
    assert not (tmp_path / "analysis_run.json").exists()


@pytest.mark.parametrize("completed", [True, False])
def test_standalone_terminal_recovery_verifies_outcome_and_scores(tmp_path, completed):
    from types import SimpleNamespace

    from trialagentbench_verifier.numerical import NumericalReportingContext

    from trialagentbench.contracts.trialdev.release import TrialDevMaterialV1, TrialDevTaskV1
    from trialagentbench.contracts.trialdev.run import TrialDevTaskRunV1
    from trialagentbench.io import write_json_model
    from trialagentbench.trialdev.release import TrialDevAnalysisConfiguration, TrialDevInitialEvidence
    from trialagentbench.trialdev.run import (
        TrialDevAnalysisRun,
        _numerical_artifact_paths,
        recover_terminal_trialdev_task,
        trialdev_task_artifact_v1,
    )

    targets, replay, preferences = _inputs()
    if not completed:
        replay = NumericalReplay(
            status="missing_result",
            entrypoint="",
            checkpoint_id="initial",
            target_ids=replay.target_ids,
            evidence_artifacts=(),
            error="No report",
        )
    configuration = TrialDevAnalysisConfiguration(
        preferences=preferences,
        initial_evidence=TrialDevInitialEvidence(
            context=NumericalReportingContext(
                checkpoint_id="initial", target_ids=replay.target_ids, evidence_artifacts=()
            ),
            targets=targets,
        ),
    )
    task = TrialDevTaskV1(
        task_id="td-123456789abc",
        case_id="tdw-0001",
        stream="evidence_analysis",
        title="Analyse",
        clinical_question="Estimate.",
        available_materials=(TrialDevMaterialV1(relative_path="data.csv", description="Data"),),
        requested_outputs=("Estimate and interval.",),
    )
    root = tmp_path / task.task_id
    root.mkdir()
    report = None
    if completed:
        (root / "analysis_sources").mkdir()
        report = "analysis_sources/report.md"
        (root / report).write_text("Analysis")
    analysis = TrialDevAnalysisRun(
        execution_status="completed" if completed else "model_noncompletion",
        replay=replay,
        scores=grade_trialdev_observational_numbers(targets=targets, replay=replay, preferences=preferences),
        report_relative_path=report,
        error=None if completed else "Turn limit",
    )
    write_json_model(root / "analysis_run.json", analysis)
    release = SimpleNamespace(
        manifest=SimpleNamespace(release_id="release_" + "a" * 64),
        participant=SimpleNamespace(tasks=(task,), release_manifest_sha256="b" * 64),
        authored_configuration=lambda task: None,
        analysis_configuration=lambda task: configuration,
    )

    def seal():
        record = TrialDevTaskRunV1(
            release_id=release.manifest.release_id,
            release_manifest_sha256=release.participant.release_manifest_sha256,
            task_id=task.task_id,
            stream=task.stream,
            execution_status=analysis.execution_status,
            error=analysis.error,
            report_relative_paths=(),
            ingress_relative_paths=(),
            grade_relative_paths=(),
            analysis_relative_path="analysis_run.json",
            artifacts=tuple(
                trialdev_task_artifact_v1(root, path) for path in _numerical_artifact_paths(root, programme=False)
            ),
        )
        write_json_model(root / "run.json", record)
        return record

    expected = seal()
    assert recover_terminal_trialdev_task(release=release, task_id=task.task_id, output_root=tmp_path) == expected
    write_json_model(root / "analysis_run.json", analysis.model_copy(update={"scores": ()}))
    seal()
    with pytest.raises(ValueError, match="Analysis scores differ from the scheduled replay target census"):
        recover_terminal_trialdev_task(release=release, task_id=task.task_id, output_root=tmp_path)


@pytest.mark.parametrize("stream", ["evidence_analysis", "portfolio_management"])
def test_ranking_preserves_evaluator_failure_before_numerical_result(tmp_path, monkeypatch, stream):
    from types import SimpleNamespace

    from trialagentbench.contracts.trialdev.release import TrialDevMaterialV1, TrialDevTaskV1
    from trialagentbench.contracts.trialdev.run import TrialDevTaskRunV1
    from trialagentbench.io import write_json_model
    from trialagentbench.trialdev import ranking as owner

    task = TrialDevTaskV1(
        task_id="td-123456789abc",
        case_id="tdw-0001",
        stream=stream,
        title="Analyse or develop assets",
        clinical_question="Recover the signal.",
        available_materials=(TrialDevMaterialV1(relative_path="brief.md", description="Brief"),),
        requested_outputs=("Estimate with uncertainty.",),
    )
    release_id = "release_" + "a" * 64
    manifest_sha = "b" * 64
    root = tmp_path / task.task_id
    root.mkdir()
    write_json_model(
        root / "run.json",
        TrialDevTaskRunV1(
            release_id=release_id,
            release_manifest_sha256=manifest_sha,
            task_id=task.task_id,
            stream=stream,
            execution_status="evaluator_error",
            report_relative_paths=(),
            ingress_relative_paths=(),
            grade_relative_paths=(),
            error="Executor unavailable before numerical result publication",
        ),
    )
    invocation = SimpleNamespace(
        release_id=release_id,
        release_manifest_sha256=manifest_sha,
        scheduled_task_ids=(task.task_id,),
        scheduled_task_streams={task.task_id: task.stream},
        checksum="c" * 64,
        model="test",
    )
    stratum = TrialDevWorldStratumV1(
        world_id=task.case_id, clinical_profile_id="test", candidate_count=1, scenario_ref="scenario_" + "0" * 20
    )
    reader = owner.read_json_model
    monkeypatch.setattr(
        owner,
        "read_json_model",
        lambda model, path: invocation if path.name == "invocation.json" else reader(model, path),
    )
    release = SimpleNamespace(
        world_stratum=lambda world_id: stratum,
        root=tmp_path,
        manifest=SimpleNamespace(release_id=release_id, task_count=1),
        participant=SimpleNamespace(tasks=(task,), release_manifest_sha256=manifest_sha),
        analysis_configuration=lambda task: (
            SimpleNamespace(initial_evidence=SimpleNamespace(context=SimpleNamespace(target_ids=("effect",))))
            if stream == "evidence_analysis"
            else None
        ),
        authored_configuration=lambda task: (
            SimpleNamespace(initial_evidence=SimpleNamespace(context=SimpleNamespace(target_ids=("effect",))))
            if stream == "portfolio_management"
            else None
        ),
    )
    result = owner.rank_trialdev_run_v1(release=release, run_root=tmp_path, task_roots=(root,))
    assert result.items[0].execution_status == "evaluator_error"
    assert result.items[0].initial_target_statuses == {"effect": "evaluator_error"}
    assert result.items[0].acquired_target_statuses is None
    assert result.items[0].observational_loss is None
    assert result.items[0].net_development_value_usd is None
    assert all(
        row.scheduled_tasks == 1 and row.evaluator_errors == 1 and row.mean is None for row in result.profile_means
    )


def test_extreme_finite_numbers_keep_all_targets_and_do_not_crash_grading() -> None:
    import json

    targets, replay, preferences = _inputs()
    replay = replay.model_copy(
        update={
            "estimates": (
                NumericalEstimate(target_id="candidate:clinical_outcome", estimate=1e308, lower=-1e308, upper=1e308),
            )
        }
    )
    clinical, safety = grade_trialdev_observational_numbers(targets=targets, replay=replay, preferences=preferences)
    assert clinical.status == "reported"
    assert clinical.loss is None
    assert clinical.normalized_interval_loss is None
    assert safety.status == "missing" and safety.loss is None
    from trialagentbench_verifier.authored_results import mean_numerical_loss, numerical_target_statuses

    assert mean_numerical_loss((clinical, safety)) is None
    assert numerical_target_statuses((clinical, safety)) == {
        "candidate:clinical_outcome": "unscored",
        "candidate:safety_risk": "missing",
    }
    with pytest.raises(ValueError, match="repeats an identity"):
        numerical_target_statuses((clinical, clinical))
    json.dumps([row.model_dump(mode="json") for row in (clinical, safety)], allow_nan=False)


def test_duplicate_private_checkpoint_cannot_add_a_second_scored_report(tmp_path) -> None:
    from trialagentbench_verifier.numerical import NumericalReportingContext

    from trialagentbench.io import write_json_model
    from trialagentbench.trialdev.run import _accepted_captures

    workspace = tmp_path / "workspace"
    context = NumericalReportingContext(
        checkpoint_id="initial", target_ids=("asset:clinical_outcome",), evidence_artifacts=()
    )
    root = tmp_path / "analysis_sources" / "cutoffs"
    write_json_model(root / "first" / "reporting_context.json", context)
    assert tuple(_accepted_captures(workspace)) == ("initial",)
    write_json_model(root / "repeated" / "reporting_context.json", context)
    with pytest.raises(ValueError, match="multiple accepted reports"):
        _accepted_captures(workspace)


def test_standalone_independent_verifier_reconstructs_retained_numerical_scores(tmp_path):
    from trialagentbench_verifier.development import _analysis_values, _Run, _Task
    from trialagentbench_verifier.executor import analysis_project_artifacts
    from trialagentbench_verifier.numerical import NumericalReportingContext

    from trialagentbench.io import sha256_file, write_json_model
    from trialagentbench.trialdev.release import TrialDevAnalysisConfiguration, TrialDevInitialEvidence
    from trialagentbench.trialdev.run import TrialDevAnalysisRun

    targets, replay, preferences = _inputs()
    task_id = "td-123456789abc"
    task_root = tmp_path / "tasks" / task_id
    project = task_root / "analysis_sources" / "submission"
    project.mkdir(parents=True)
    (project / "analysis.py").write_text("# retained analysis source\n")
    (project / "estimates.csv").write_text("target_id,estimate,lower,upper\ncandidate:clinical_outcome,3,1,4\n")
    (project / "report.md").write_text("Observed association, with uncertainty.\n")
    context = NumericalReportingContext(checkpoint_id="initial", target_ids=replay.target_ids, evidence_artifacts=())
    write_json_model(project / "reporting_context.json", context)
    csv_hash = sha256_file(project / "estimates.csv")
    replay = replay.model_copy(
        update={
            "program_artifacts": analysis_project_artifacts(project / "analysis.py"),
            "submitted_csv_sha256": csv_hash,
            "replayed_csv_sha256s": (csv_hash, csv_hash),
        }
    )
    configuration = TrialDevAnalysisConfiguration(
        preferences=preferences, initial_evidence=TrialDevInitialEvidence(context=context, targets=targets)
    )
    config_path = tmp_path / "evaluation" / "tasks" / task_id / "analysis.json"
    config_path.parent.mkdir(parents=True)
    write_json_model(config_path, configuration)
    result = TrialDevAnalysisRun(
        execution_status="completed",
        replay=replay,
        scores=grade_trialdev_observational_numbers(targets=targets, replay=replay, preferences=preferences),
        report_relative_path="analysis_sources/submission/report.md",
    )
    write_json_model(task_root / "analysis_run.json", result)

    def record():
        return _Run(
            schema_id="ariadne.trialdev.task_run/v1",
            release_id="release",
            release_manifest_sha256="a" * 64,
            task_id=task_id,
            stream="evidence_analysis",
            execution_status="completed",
            report_relative_paths=(),
            ingress_relative_paths=(),
            grade_relative_paths=(),
            artifacts=tuple(
                {
                    "relative_path": p.relative_to(task_root).as_posix(),
                    "sha256": sha256_file(p),
                    "byte_length": p.stat().st_size,
                }
                for p in task_root.rglob("*")
                if p.is_file()
            ),
            final_state=None,
            error=None,
            checksum="b" * 64,
            analysis_relative_path="analysis_run.json",
        )

    task = _Task(task_id=task_id, case_id="tdw-0001", stream="evidence_analysis", available_materials=())
    assert (
        _analysis_values(
            release=tmp_path, task_root=task_root, run=record(), task=task, executor_image="sha256:" + "c" * 64
        )
        is None
    )
    # Even a newly checksummed result cannot substitute its own recovery score.
    changed = result.model_copy(
        update={"scores": (result.scores[0].model_copy(update={"normalized_absolute_error": 999}), result.scores[1])}
    )
    write_json_model(task_root / "analysis_run.json", changed)
    with pytest.raises(ValueError, match="independent recovery"):
        _analysis_values(
            release=tmp_path, task_root=task_root, run=record(), task=task, executor_image="sha256:" + "c" * 64
        )
    write_json_model(task_root / "analysis_run.json", result)
    (project / "estimates.csv").write_text("target_id,estimate,lower,upper\ncandidate:clinical_outcome,2,2,2\n")
    with pytest.raises(ValueError, match="CSV differs"):
        _analysis_values(
            release=tmp_path, task_root=task_root, run=record(), task=task, executor_image="sha256:" + "c" * 64
        )


def test_standalone_verifier_rejects_reproduction_without_captured_source(tmp_path) -> None:
    """A rebound result cannot claim credit without the program that produced it."""
    from trialagentbench_verifier.development import _verify_numerical_capture
    from trialagentbench_verifier.numerical import NumericalReportingContext

    _, replay, _ = _inputs()
    context = NumericalReportingContext(
        checkpoint_id=replay.checkpoint_id, target_ids=replay.target_ids, evidence_artifacts=replay.evidence_artifacts
    )
    with pytest.raises(ValueError, match="no captured source"):
        _verify_numerical_capture(
            task_root=tmp_path,
            project=None,
            context=context,
            replay=replay,
            artifacts={},
            executor_image="sha256:" + "c" * 64,
        )


def _cached_capture(tmp_path, monkeypatch):
    from trialagentbench_verifier.numerical import (
        NumericalReportingContext,
        ReplayInputArtifactV1,
        analysis_project_artifacts,
    )

    from trialagentbench.io import sha256_file
    from trialagentbench.ports import CodeExecutionLimitsV1
    from trialagentbench.trialdev import run

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    evidence = workspace / "cohort.csv"
    evidence.write_text("participant,outcome\n1,2\n")
    captured = tmp_path / "capture"
    captured.mkdir()
    (captured / "analysis.py").write_text("# immutable submitted program\n")
    output = captured / "estimates.csv"
    output.write_text("target_id,estimate,lower,upper\ncandidate:clinical_outcome,3,1,4\n")
    _, original, _ = _inputs()
    context = NumericalReportingContext(
        checkpoint_id="initial",
        target_ids=original.target_ids,
        evidence_artifacts=(
            ReplayInputArtifactV1(
                relative_path="cohort.csv",
                sha256=sha256_file(evidence),
                byte_length=evidence.stat().st_size,
            ),
        ),
    )
    replay = original.model_copy(
        update={
            "evidence_artifacts": context.evidence_artifacts,
            "program_artifacts": analysis_project_artifacts(captured / "analysis.py"),
            "submitted_csv_sha256": sha256_file(output),
            "replayed_csv_sha256s": (sha256_file(output), sha256_file(output)),
        }
    )
    calls = []

    def execute(**kwargs):
        calls.append(kwargs)
        return replay

    monkeypatch.setattr(run, "replay_numerical_submission", execute)
    arguments = dict(
        workspace=workspace,
        captured=captured,
        context=context,
        executor_limits=CodeExecutionLimitsV1(),
        executor_image="sha256:" + "c" * 64,
    )
    return run, arguments, calls


def test_changed_verifier_reuses_the_same_accepted_capture(tmp_path, monkeypatch):
    """Unrelated verifier edits do not invalidate an accepted scientific replay."""
    import json

    run, arguments, calls = _cached_capture(tmp_path, monkeypatch)
    first = run._replay_accepted_capture(**arguments)
    assert run._replay_accepted_capture(**arguments) == first
    assert len(calls) == 1
    checkpoint = arguments["captured"] / "verified_replay.json"
    payload = json.loads(checkpoint.read_text())
    checkpoint.write_text(json.dumps({**payload, "verifier_source_sha256": "0" * 64}))
    assert run._replay_accepted_capture(**arguments) == first
    assert len(calls) == 1
    assert json.loads(checkpoint.read_text())["verifier_source_sha256"] == "0" * 64
    assert run._replay_accepted_capture(**arguments) == first
    assert len(calls) == 1


@pytest.mark.parametrize("defect", ["program", "csv", "evidence", "image", "limits", "context"])
def test_changed_verifier_never_bypasses_committed_input_custody(tmp_path, monkeypatch, defect):
    """Even a stale cache must reject altered scientific or execution inputs."""
    import json

    run, arguments, calls = _cached_capture(tmp_path, monkeypatch)
    run._replay_accepted_capture(**arguments)
    checkpoint = arguments["captured"] / "verified_replay.json"
    payload = json.loads(checkpoint.read_text())
    checkpoint.write_text(json.dumps({**payload, "verifier_source_sha256": "0" * 64}))
    if defect in {"program", "csv"}:
        (arguments["captured"] / ("analysis.py" if defect == "program" else "estimates.csv")).write_text("changed")
    elif defect == "evidence":
        (arguments["workspace"] / "cohort.csv").write_text("changed")
    elif defect == "image":
        arguments["executor_image"] = "sha256:" + "d" * 64
    elif defect == "limits":
        arguments["executor_limits"] = arguments["executor_limits"].model_copy(update={"cpu_count": 1})
    else:
        arguments["context"] = arguments["context"].model_copy(update={"checkpoint_id": "different"})
    with pytest.raises(ValueError):
        run._replay_accepted_capture(**arguments)
    assert len(calls) == 1


def test_standalone_runner_keeps_a_captured_report_when_the_deadline_is_observed_afterwards(tmp_path, monkeypatch):
    """A report captured before the deadline was observed (while closing the step) stands."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    from trialagentbench_verifier.numerical import NumericalReportingContext

    from trialagentbench.io import write_json_model
    from trialagentbench.ports import CodeExecutionLimitsV1, RuntimeDeadlineExceeded
    from trialagentbench.trialdev.release import TrialDevAnalysisConfiguration, TrialDevInitialEvidence
    from trialagentbench.trialdev.run import run_trialdev_analysis

    targets, replay, preferences = _inputs()
    context = NumericalReportingContext(checkpoint_id="initial", target_ids=replay.target_ids, evidence_artifacts=())
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    capture = tmp_path / "analysis_sources" / "submission-test"

    def submit():
        capture.mkdir(parents=True)
        (capture / "report.md").write_text("Analysis report")
        write_json_model(capture / "reporting_context.json", context)
        return SimpleNamespace(path=capture / "report.md")

    loop = Mock(workdir=workspace, authored_programme=None, reporting_context=None, active_prompt_index=None)
    loop.run_until_report.side_effect = submit
    loop.finish_step.side_effect = RuntimeDeadlineExceeded("TrialDev task exceeded its declared wall-clock budget.")
    monkeypatch.setattr("trialagentbench.trialdev.run.replay_numerical_submission", Mock(return_value=replay))
    result = run_trialdev_analysis(
        loop=loop,
        prompt="Estimate the associations.",
        configuration=TrialDevAnalysisConfiguration(
            preferences=preferences, initial_evidence=TrialDevInitialEvidence(context=context, targets=targets)
        ),
        executor_limits=CodeExecutionLimitsV1(),
    )
    assert result.execution_status == "completed"
    assert result.error is None
    loop.close.assert_called_once()


@pytest.mark.parametrize("endpoint", ["clinical_outcome", "safety_risk"])
@pytest.mark.parametrize("benefit_unit", [0.5, 1.0, 2.0])
def test_final_trialdev_targets_elicit_median_and_interval_quantiles(endpoint, benefit_unit):
    """Exhaust a finite truth/report grid through runtime and independent regrading."""
    from itertools import product
    from statistics import mean

    from trialagentbench_verifier.development import _initial_recovery, _InitialEvidence, _NumericalPreferences
    from trialagentbench_verifier.numerical import NumericalReportingContext

    targets, replay, preferences = _inputs()
    preferences = preferences.model_copy(update={"benefit_unit": benefit_unit})
    scale = benefit_unit if endpoint == "clinical_outcome" else 0.01
    index = 0 if endpoint == "clinical_outcome" else 1
    centre = targets[index].oracle_value
    context = NumericalReportingContext(checkpoint_id="initial", target_ids=replay.target_ids, evidence_artifacts=())
    independent_preferences = _NumericalPreferences.model_validate(preferences.model_dump())

    def expected_loss(point, lower, upper):
        estimates = tuple(
            NumericalEstimate(
                target_id=identity,
                estimate=centre + point * scale if number == index else target.oracle_value,
                lower=centre + lower * scale if number == index else target.oracle_value,
                upper=centre + upper * scale if number == index else target.oracle_value,
            )
            for number, (target, identity) in enumerate(zip(targets, replay.target_ids, strict=True))
        )
        report = replay.model_copy(update={"estimates": estimates})
        losses = []
        for truth in range(-20, 21):
            assigned = list(targets)
            assigned[index] = targets[index].model_copy(update={"oracle_value": centre + truth / 20 * scale})
            scores = grade_trialdev_observational_numbers(
                targets=tuple(assigned), replay=report, preferences=preferences
            )
            independently_graded = _initial_recovery(
                replay=report,
                scores=scores,
                initial=_InitialEvidence.model_validate(
                    {"context": context.model_dump(), "targets": [target.model_dump() for target in assigned]}
                ),
                preferences=independent_preferences,
            )
            assert scores[index].loss is not None
            assert independently_graded == pytest.approx(scores[index].loss / 2)
            losses.append(scores[index].loss)
        return mean(losses)

    optimal = expected_loss(0, -0.95, 0.95)
    for point in (-0.8, -0.4, 0, 0.4, 0.8):
        assert expected_loss(point, -0.95, 0.95) >= optimal - 1e-12
    for lower, upper in product((-1.1, -1, -0.95, -0.9, -0.8, -0.5), (0.5, 0.8, 0.9, 0.95, 1, 1.1)):
        assert expected_loss(0, lower, upper) >= optimal - 1e-12, (lower, upper)
