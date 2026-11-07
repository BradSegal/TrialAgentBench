"""Recompute canonical TrialDev grades from authenticated program ingress."""

from __future__ import annotations

from pathlib import Path

from trialagentbench_verifier.execution_contracts import AnalysisMode
from trialagentbench_verifier.io import trialdev_task_run_paths

from trialagentbench.contracts.trialdev.release import TrialDevTaskV1
from trialagentbench.contracts.trialdev.run import TrialDevRunIndexV1, TrialDevRunInvocationV1, TrialDevTaskRunV1
from trialagentbench.io import read_json_model, write_json_model
from trialagentbench.ports import CodeExecutionLimitsV1
from trialagentbench.trialdev.actions import TrialDevActionSession
from trialagentbench.trialdev.ranking import rank_trialdev_run_v1
from trialagentbench.trialdev.release import TrialDevEvaluatorRelease
from trialagentbench.trialdev.resources import TrialDevBudgetLedger
from trialagentbench.trialdev.run import (
    TrialDevAnalysisRun,
    TrialDevProgrammeRun,
    _numerical_artifact_paths,
    evaluate_trialdev_analysis,
    evaluate_trialdev_programme,
    trialdev_task_artifact_v1,
    verify_trialdev_task_run_artifacts_v1,
)


def _task(release: TrialDevEvaluatorRelease, task_id: str) -> TrialDevTaskV1:
    matches = tuple(task for task in release.participant.tasks if task.task_id == task_id)
    if len(matches) != 1:
        raise ValueError(f"TrialDev release does not contain one task_id={task_id!r}")
    return matches[0]


def regrade_trialdev_task_directory(
    *,
    release: TrialDevEvaluatorRelease,
    task_root: Path,
    executor_limits: CodeExecutionLimitsV1 | None = None,
    analysis_language: AnalysisMode = "python_r",
    executor_image: str | None = None,
) -> TrialDevTaskRunV1:
    """Regrade one canonical task directory without a provider or construction code."""

    task_root = Path(task_root)
    run = read_json_model(TrialDevTaskRunV1, task_root / "run.json")
    verify_trialdev_task_run_artifacts_v1(task_root=task_root, run=run)
    if (
        run.release_id != release.manifest.release_id
        or run.release_manifest_sha256 != release.participant.release_manifest_sha256
    ):
        raise ValueError("TrialDev run and evaluator release identities differ")
    task = _task(release, run.task_id)
    if task.stream is not run.stream:
        raise ValueError("TrialDev task and run streams differ")
    if run.programme_relative_path is not None:
        if executor_limits is None or executor_image is None:
            raise ValueError("Authored regrading requires the invocation executor image and limits")
        configuration = release.authored_configuration(task)
        if configuration is None:
            raise ValueError("Authored run lacks a released programme configuration")
        budget_path = task_root / "private" / "budget.json"
        ledger = read_json_model(TrialDevBudgetLedger, budget_path)
        if (
            ledger.pool_manifest_sha256 != configuration.pool_manifest_sha256
            or ledger.recruitment_seed != configuration.recruitment_seed
            or ledger.budget_usd != configuration.budget_usd
        ):
            raise ValueError("Authored ledger differs from the release configuration")
        previous = read_json_model(TrialDevProgrammeRun, task_root / run.programme_relative_path)
        evaluation = evaluate_trialdev_programme(
            workspace=task_root / "workspace",
            programme=TrialDevActionSession(
                pool_root=release.root / "evaluation" / "worlds" / task.case_id / "pool",
                budget_path=budget_path,
                evidence_root=task_root / "workspace" / "evidence",
                commitment_root=task_root / "private" / "commitments",
                recruitment_seed=configuration.recruitment_seed,
            ),
            preferences=configuration.preferences,
            initial_evidence=configuration.initial_evidence,
            executor_limits=executor_limits,
            analysis_language=analysis_language,
            executor_image=executor_image,
        )
        result = TrialDevProgrammeRun(
            execution_status=run.execution_status,
            error=run.error,
            evaluation=evaluation,
        )
        if previous.execution_status != run.execution_status or previous.error != run.error:
            raise ValueError("Regrading cannot change the recorded model execution outcome")
        write_json_model(task_root / run.programme_relative_path, result)
        payload = run.model_dump(mode="python", exclude={"checksum"})
        payload["artifacts"] = tuple(
            trialdev_task_artifact_v1(task_root, relative)
            for relative in _numerical_artifact_paths(task_root, programme=True)
        )
        updated = TrialDevTaskRunV1.model_validate(payload)
        write_json_model(task_root / "run.json", updated)
        return updated
    if run.analysis_relative_path is not None:
        if executor_limits is None or executor_image is None:
            raise ValueError("Numerical regrading requires the invocation executor image and limits")
        analysis_configuration = release.analysis_configuration(task)
        if analysis_configuration is None:
            raise ValueError("Numerical analysis lacks its released configuration")
        previous_analysis = read_json_model(TrialDevAnalysisRun, task_root / run.analysis_relative_path)
        if previous_analysis.execution_status != run.execution_status or previous_analysis.error != run.error:
            raise ValueError("Regrading cannot change the recorded model execution outcome")
        if previous_analysis.report_relative_path is not None and previous_analysis.report_relative_path not in {
            artifact.relative_path for artifact in run.artifacts
        }:
            raise ValueError("Standalone report is not bound by its result inventory")
        analysis = evaluate_trialdev_analysis(
            workspace=task_root / "workspace",
            configuration=analysis_configuration,
            report_relative_path=previous_analysis.report_relative_path,
            execution_status=previous_analysis.execution_status,
            error=previous_analysis.error,
            executor_limits=executor_limits,
            analysis_language=analysis_language,
            executor_image=executor_image,
        )
        write_json_model(task_root / run.analysis_relative_path, analysis)
        payload = run.model_dump(mode="python", exclude={"checksum"})
        payload["artifacts"] = tuple(
            trialdev_task_artifact_v1(task_root, relative)
            for relative in _numerical_artifact_paths(task_root, programme=False)
        )
        updated = TrialDevTaskRunV1.model_validate(payload)
        write_json_model(task_root / "run.json", updated)
        return updated
    raise ValueError("TrialDev regrading requires a current numerical or authored-programme result")


def regrade_trialdev_run(
    *,
    release: TrialDevEvaluatorRelease,
    run_root: Path,
) -> tuple[TrialDevTaskRunV1, ...]:
    """Regrade tasks at the recorded worker bound, then publish their ordered index."""

    roots = tuple(sorted(path.parent for path in trialdev_task_run_paths(Path(run_root)).values()))
    index_path = run_root / "run_index.json"
    index = read_json_model(TrialDevRunIndexV1, index_path) if index_path.is_file() else None
    invocation = read_json_model(TrialDevRunInvocationV1, run_root / "invocation.json")
    if index is not None:
        if (
            index.invocation_checksum != invocation.checksum
            or index.release_id != invocation.release_id
            or index.model != invocation.model
            or index.scheduled_task_ids != invocation.scheduled_task_ids
        ):
            raise ValueError("TrialDev input run index and invocation differ")
        prior_runs = tuple(read_json_model(TrialDevTaskRunV1, root / "run.json") for root in roots)
        before = {run.task_id: run.checksum for run in prior_runs}
        if len(before) != len(prior_runs):
            raise ValueError("TrialDev input repeats a task run")
        if before != {row.task_id: row.checksum for row in index.results}:
            raise ValueError("TrialDev input run index and task custody differ")

    def regrade(item: tuple[int, Path]) -> tuple[int, TrialDevTaskRunV1]:
        position, root = item
        return position, regrade_trialdev_task_directory(
            release=release,
            task_root=root,
            executor_limits=invocation.executor_limits,
            analysis_language=invocation.experiment_condition.analysis_language,
            executor_image=invocation.executor_image,
        )

    from trialagentbench_verifier.executor import executor_session_capacity

    from trialagentbench.util.processes import process_map

    with executor_session_capacity(limits=invocation.executor_limits, queue=True):
        completed = process_map(regrade, enumerate(roots))
    runs = tuple(run for _, run in sorted(completed, key=lambda item: item[0]))
    if index is not None:
        updated_runs = {run.task_id: run for run in runs}
        payload = index.model_dump(mode="python", exclude={"checksum"})
        payload["results"] = tuple(
            row.model_copy(update={"checksum": updated_runs[row.task_id].checksum}) for row in index.results
        )
        write_json_model(index_path, TrialDevRunIndexV1.model_validate(payload))
    write_json_model(
        run_root / "ranking.json", rank_trialdev_run_v1(release=release, run_root=run_root, task_roots=roots)
    )
    return runs


__all__ = ["regrade_trialdev_run", "regrade_trialdev_task_directory"]
