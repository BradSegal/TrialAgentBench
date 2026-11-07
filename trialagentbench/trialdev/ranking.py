"""Scheduled-task TrialDev recovery and development value summaries."""

from __future__ import annotations

from pathlib import Path

from trialagentbench_verifier.authored_results import mean_numerical_loss, numerical_target_statuses
from trialagentbench_verifier.development import ranking_means
from trialagentbench_verifier.response import (
    TrialDevRankingItemV1 as TrialDevRankingItemV1,
)
from trialagentbench_verifier.response import (
    TrialDevRankingMeanV1 as TrialDevRankingMeanV1,
)
from trialagentbench_verifier.response import (
    TrialDevRunRankingV1 as TrialDevRunRankingV1,
)
from trialagentbench_verifier.response import TrialDevTargetStatus
from trialagentbench_verifier.scorer import current_scorer

from trialagentbench.contracts.trialdev.run import TrialDevRunInvocationV1, TrialDevTaskRunV1
from trialagentbench.io import read_json_model
from trialagentbench.trialdev.release import TrialDevEvaluatorRelease
from trialagentbench.trialdev.run import (
    TrialDevAnalysisRun,
    TrialDevProgrammeRun,
    verify_trialdev_task_run_artifacts_v1,
)
from trialagentbench.util.source_identity import require_harness_scorer_v1


def rank_trialdev_run_v1(
    *, release: TrialDevEvaluatorRelease, run_root: Path, task_roots: tuple[Path, ...]
) -> TrialDevRunRankingV1:
    """Summarize authenticated regraded tasks against the full invocation denominator.

    Parameters
    ----------
    release : TrialDevEvaluatorRelease
        Verified evaluator release.
    run_root : Path
        Directory containing the immutable invocation.
    task_roots : tuple of Path
        Task directories already reconstructed by the canonical offline grader.

    Returns
    -------
    TrialDevRunRankingV1
        Separate recovery and development values, with profile means withheld
        where evaluator faults remain. A partial schedule is labelled as such.

    Raises
    ------
    ValueError
        If release, schedule, task custody or world identities disagree.
    """

    require_harness_scorer_v1()
    invocation = read_json_model(TrialDevRunInvocationV1, run_root / "invocation.json")
    if (
        invocation.release_id != release.manifest.release_id
        or invocation.release_manifest_sha256 != release.participant.release_manifest_sha256
    ):
        raise ValueError("Ranking invocation and evaluator release differ")
    tasks = {task.task_id: task for task in release.participant.tasks}
    if not set(invocation.scheduled_task_ids) <= tasks.keys():
        raise ValueError("Ranking schedule contains a task outside the release")
    observed = {}
    for root in task_roots:
        root.relative_to(run_root)
        run = read_json_model(TrialDevTaskRunV1, root / "run.json")
        verify_trialdev_task_run_artifacts_v1(task_root=root, run=run)
        if run.task_id in observed or run.task_id not in invocation.scheduled_task_ids:
            raise ValueError("Ranking task runs are duplicated or unscheduled")
        if (
            run.release_id != invocation.release_id
            or run.release_manifest_sha256 != invocation.release_manifest_sha256
            or run.stream != invocation.scheduled_task_streams[run.task_id]
        ):
            raise ValueError("Ranking task custody differs from the invocation")
        observed[run.task_id] = (root, run)
    rows = []
    for task_id in invocation.scheduled_task_ids:
        task = tasks[task_id]
        if task.stream != invocation.scheduled_task_streams[task_id]:
            raise ValueError("Ranking task stream differs from its schedule")
        stratum = release.world_stratum(task.case_id)
        record = observed.get(task_id)
        unfinished = record is None and any((parent / task_id).exists() for parent in (run_root, run_root / "tasks"))
        status = ("incomplete_run" if unfinished else "not_started") if record is None else record[1].execution_status
        analysis_configuration = release.analysis_configuration(task)
        authored_configuration = release.authored_configuration(task)
        configuration = authored_configuration or analysis_configuration
        if configuration is None:
            raise ValueError("TrialDev ranking requires a numerical target inventory")
        initial_target_statuses: dict[str, TrialDevTargetStatus] = {
            target_id: (
                "not_started"
                if status == "not_started"
                else (
                    "incomplete_run"
                    if status == "incomplete_run"
                    else "evaluator_error" if status == "evaluator_error" else "missing"
                )
            )
            for target_id in configuration.initial_evidence.context.target_ids
        }
        acquired_target_statuses: dict[str, dict[str, TrialDevTargetStatus]] | None = None
        recovery: float | None = None
        if analysis_configuration is not None and record is not None and status != "evaluator_error":
            if record[1].analysis_relative_path is None:
                raise ValueError("Numerical analysis task requires an analysis result")
            analysis = read_json_model(TrialDevAnalysisRun, record[0] / record[1].analysis_relative_path)
            if analysis.execution_status != status or analysis.error != record[1].error:
                raise ValueError("Standalone analysis and task outcome disagree")
            recovery = mean_numerical_loss(analysis.scores)
            initial_target_statuses = numerical_target_statuses(analysis.scores)
        authored = authored_configuration is not None
        net_value = None
        if authored and record is not None and status != "evaluator_error":
            if record[1].programme_relative_path is None:
                raise ValueError("Authored task requires a programme result")
            result = read_json_model(TrialDevProgrammeRun, record[0] / record[1].programme_relative_path)
            if result.execution_status != status or result.error != record[1].error:
                raise ValueError("Authored task result and programme outcome disagree")
            net_value = result.evaluation.value.net_value_usd
            recovery = mean_numerical_loss(result.evaluation.initial_scores)
            initial_target_statuses = numerical_target_statuses(result.evaluation.initial_scores)
            acquired_target_statuses = {
                study.report.study_id: numerical_target_statuses(study.scores) for study in result.evaluation.studies
            }
        if set(initial_target_statuses) != set(configuration.initial_evidence.context.target_ids):
            raise ValueError("Initial result target census differs from the released task")
        if not authored and task.stream.value == "portfolio_management":
            raise ValueError("TrialDev ranking requires an authored-programme configuration")
        if task.stream.value == "evidence_analysis" and analysis_configuration is None:
            raise ValueError("TrialDev ranking requires a numerical-analysis configuration")
        rows.append(
            TrialDevRankingItemV1(
                task_id=task_id,
                world_id=task.case_id,
                stream=task.stream.value,
                data_view=task.data_view,
                budget_usd=None if task.resources is None else task.resources.budget_usd,
                clinical_profile_id=stratum.clinical_profile_id,
                scenario_profile_id=stratum.scenario_ref,
                asset_count=stratum.candidate_count,
                execution_status=status,
                run_checksum=None if record is None else record[1].checksum,
                observational_loss=recovery,
                initial_target_statuses=initial_target_statuses,
                acquired_target_statuses=acquired_target_statuses,
                authored_programme=authored,
                net_development_value_usd=net_value,
            )
        )
    means = ranking_means(rows)
    return TrialDevRunRankingV1(
        scorer=current_scorer(),
        release_id=invocation.release_id,
        release_manifest_sha256=invocation.release_manifest_sha256,
        invocation_checksum=str(invocation.checksum),
        model_id=invocation.model,
        scope=(
            "complete_release_task_inventory"
            if len(invocation.scheduled_task_ids) == release.manifest.task_count
            else "scheduled_panel"
        ),
        items=tuple(rows),
        profile_means=tuple(means),
    )
