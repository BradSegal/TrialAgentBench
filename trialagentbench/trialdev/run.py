"""Run canonical TrialDev tasks from role-separated release artifacts."""

from __future__ import annotations

import shutil
import time
from pathlib import Path, PurePosixPath
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field
from trialagentbench_verifier import numerical as numerical_owner
from trialagentbench_verifier.authored_results import (
    TrialDevAnalysisRun as TrialDevAnalysisRun,
)
from trialagentbench_verifier.authored_results import (
    TrialDevProgrammeEvaluation as TrialDevProgrammeEvaluation,
)
from trialagentbench_verifier.authored_results import (
    TrialDevProgrammeRun as TrialDevProgrammeRun,
)
from trialagentbench_verifier.authored_results import (
    TrialDevStudyEvaluation as TrialDevStudyEvaluation,
)
from trialagentbench_verifier.authored_results import (
    TrialDevStudyReport as TrialDevStudyReport,
)
from trialagentbench_verifier.execution_contracts import AnalysisMode, ReplayInputArtifactV1
from trialagentbench_verifier.executor import (
    CodeExecutionSessionTerminated,
    analysis_project_artifacts,
    verified_replay_inputs,
)
from trialagentbench_verifier.numerical import (
    NumericalReplay,
    NumericalReportingContext,
    read_numerical_table,
    replay_numerical_submission,
)

from trialagentbench.contracts.core.config import ProviderModelCapabilityV1, ToolChoiceV1
from trialagentbench.contracts.trialdev.pools import TrialDevFrozenPool
from trialagentbench.contracts.trialdev.release import (
    TrialDevReleaseArtifactV1,
    TrialDevTaskV1,
)
from trialagentbench.contracts.trialdev.run import (
    TrialDevTaskRunV1,
)
from trialagentbench.io import canonical_payload_sha256, read_json_model, sha256_file, write_json_model
from trialagentbench.ports import (
    CodeExecutionLimitsV1,
    LLMProvider,
    ModelResponseNoncompletion,
    RuntimeDeadlineExceeded,
)
from trialagentbench.trialdev.actions import TrialDevActionSession
from trialagentbench.trialdev.clinical_text import trialdev_clinical_text
from trialagentbench.trialdev.letters import CONTROL_REFERENCE_NOTE
from trialagentbench.trialdev.materialization import TrialDevReadoutCommitment, TrialDevReadoutManifest
from trialagentbench.trialdev.release import TrialDevAnalysisConfiguration, TrialDevEvaluatorRelease
from trialagentbench.trialdev.release import TrialDevInitialEvidence as TrialDevInitialEvidence
from trialagentbench.trialdev.resources import (
    TrialDevBudgetLedger,
    TrialDevTerminalDecision,
    finish_trialdev_programme,
)
from trialagentbench.trialdev.target_grading import (
    TrialDevNumericalRecovery,
    grade_trialdev_observational_numbers,
    grade_trialdev_readout_numbers,
)
from trialagentbench.trialdev.task_agent import (
    AgentExecutionTerminated,
    AgentTurnLimitExceeded,
    TrialDevTaskAgent,
)
from trialagentbench.trialdev.value import (
    TrialDevSponsorPreferences,
    grade_trialdev_finished_programme,
)

# Exceptions that end only their task, as a model noncompletion, never as a run failure.
_TASK_NONCOMPLETION = (
    AgentTurnLimitExceeded,
    ModelResponseNoncompletion,
    AgentExecutionTerminated,
    CodeExecutionSessionTerminated,
    RuntimeDeadlineExceeded,
)


def _accepted_captures(workspace: Path) -> dict[str, tuple[Path, NumericalReportingContext]]:
    """Read unique runner-owned reporting cutoffs without following redirects."""
    root = workspace.parent / "analysis_sources" / "cutoffs"
    if root.parent.is_symlink() or root.is_symlink():
        raise ValueError("accepted capture directories cannot use symbolic links")
    reports = {}
    for captured in sorted(root.glob("*")):
        context_path = captured / "reporting_context.json"
        if captured.is_symlink() or context_path.is_symlink() or not captured.is_dir():
            raise ValueError("accepted capture must be a private regular directory")
        context = read_json_model(NumericalReportingContext, context_path)
        if context.checkpoint_id in reports:
            raise ValueError("multiple accepted reports exist for one checkpoint")
        reports[context.checkpoint_id] = (captured, context)
    return reports


def select_trialdev_study_reports(
    *, workspace: Path, programme: TrialDevActionSession
) -> tuple[TrialDevStudyReport, ...]:
    """Select the latest acquired readout per study without inventing completion.

    Parameters
    ----------
    workspace
        Participant evidence workspace with runner-owned accepted captures nearby.
    programme
        Programme's private ledger and acquisition commitments.

    Returns
    -------
    tuple of TrialDevStudyReport
        Last acquired readout for each observed study, including missing reports.
        Earlier interim captures remain on disk but are not selected again.

    Raises
    ------
    ValueError
        If acquisition/report identities conflict.
    """
    ledger = read_json_model(TrialDevBudgetLedger, programme.budget_path)
    receipts = {item.readout_id: item for item in ledger.receipts if item.kind == "readout"}
    latest: dict[str, tuple[float, str]] = {}
    matched = set()
    for path in sorted(programme.commitment_root.glob("*.json")):
        if path.name.endswith((".request.json", ".study.json")):
            continue
        commitment = read_json_model(TrialDevReadoutCommitment, path)
        identity = canonical_payload_sha256(commitment.model_dump(mode="json"))
        receipt = receipts.get(identity)
        if receipt is None:
            continue  # A durable pre-publication commitment may belong to an unfinished action.
        if (
            identity in matched
            or commitment.study_id != receipt.study_id
            or commitment.readout.readout_day != receipt.programme_day
        ):
            raise ValueError("acquired readout differs from its accounting receipt")
        matched.add(identity)
        readout_root = programme.evidence_root / path.stem
        public_manifest = readout_root / "study.json"
        if readout_root.is_symlink() or public_manifest.is_symlink() or not public_manifest.is_file():
            raise ValueError("acquired readout publication is missing or unsafe")
        if read_json_model(TrialDevReadoutManifest, public_manifest) != commitment.readout:
            raise ValueError("published study differs from its acquisition commitment")
        checkpoint = readout_root.relative_to(workspace).as_posix()
        previous = latest.get(receipt.study_id)
        if previous is None or receipt.programme_day > previous[0]:
            latest[receipt.study_id] = (receipt.programme_day, checkpoint)
        elif receipt.programme_day == previous[0]:
            raise ValueError("study has ambiguous final readouts at the same cutoff")
    if matched != set(receipts):
        raise ValueError("an acquired readout has no private commitment")
    reports = {key: captured for key, (captured, _) in _accepted_captures(workspace).items()}
    designs = {item.study_id: item for item in ledger.recruitment_commitments}
    return tuple(
        TrialDevStudyReport(
            study_id=study,
            checkpoint_id=checkpoint,
            captured_relative_path=(
                None if checkpoint not in reports else reports[checkpoint].relative_to(workspace.parent).as_posix()
            ),
            programme_complete=ledger.terminal is not None,
            final_readout=ledger.terminal is not None
            or sum(item.study_id == study for item in receipts.values()) >= len(designs[study].looks),
        )
        for study, (_, checkpoint) in sorted(latest.items())
    )


class _TrialDevReplayCheckpoint(BaseModel):
    """Private completed replay bound to its immutable execution configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    context: NumericalReportingContext
    executor_image: str
    executor_limits: CodeExecutionLimitsV1
    verifier_source_sha256: str | None = Field(default=None, exclude_if=lambda value: value is None)
    analysis_language: AnalysisMode = Field(default="python_r", exclude_if=lambda value: value == "python_r")
    replay: NumericalReplay


def _replay_accepted_capture(
    *,
    workspace: Path,
    captured: Path,
    context: NumericalReportingContext,
    executor_limits: CodeExecutionLimitsV1,
    analysis_language: AnalysisMode = "python_r",
    executor_image: str | None,
) -> NumericalReplay:
    """Replay or recover the same committed analysis at any evidence cutoff."""
    replay_path = captured / "verified_replay.json"
    pinned_image = executor_image is not None and (
        executor_image.startswith("sha256:") or "@sha256:" in executor_image
    )
    replay: NumericalReplay | None = None
    if pinned_image and replay_path.exists():
        if replay_path.is_symlink():
            raise ValueError("replay checkpoint cannot use a symbolic link")
        saved = read_json_model(_TrialDevReplayCheckpoint, replay_path)
        if (
            saved.context != context
            or saved.executor_image != executor_image
            or saved.executor_limits != executor_limits
            or saved.analysis_language != analysis_language
            or saved.replay.evidence_artifacts != context.evidence_artifacts
            or saved.replay.entrypoint not in ("analysis.py", "analysis.R")
            or not saved.replay.reproducible
        ):
            raise ValueError("completed replay configuration differs from this invocation")
        previous = saved.replay
        program = captured / previous.entrypoint
        if (
            analysis_project_artifacts(program) != previous.program_artifacts
            or sha256_file(captured / "estimates.csv") != previous.submitted_csv_sha256
            or read_numerical_table(captured / "estimates.csv", target_ids=context.target_ids) != previous.estimates
        ):
            raise ValueError("captured analysis changed after verified replay")
        verified_replay_inputs(workspace, context.evidence_artifacts)
        replay = previous
    if replay is None:
        replay = replay_numerical_submission(
            workspace=workspace,
            submitted_project=captured,
            executor_limits=executor_limits,
            analysis_language=analysis_language,
            executor_image=executor_image,
        )
        # The first proof belongs to the committed attempt. A later verifier
        # may derive a fresh result, but cannot replace that original custody.
        if pinned_image and replay.reproducible and not replay_path.exists():
            write_json_model(
                replay_path,
                _TrialDevReplayCheckpoint(
                    context=context,
                    executor_image=cast(str, executor_image),
                    executor_limits=executor_limits,
                    analysis_language=analysis_language,
                    replay=replay,
                ),
            )
    if replay.checkpoint_id != context.checkpoint_id or replay.target_ids != context.target_ids:
        raise ValueError("captured reporting context changed during replay")
    return replay


def evaluate_trialdev_cutoff(
    *,
    workspace: Path,
    captured: Path,
    programme: TrialDevActionSession,
    preferences: TrialDevSponsorPreferences,
    executor_limits: CodeExecutionLimitsV1,
    analysis_language: AnalysisMode = "python_r",
    executor_image: str | None = None,
) -> tuple[NumericalReplay, tuple[TrialDevNumericalRecovery, ...]]:
    """Replay and grade one acquired-study report using its private commitment.

    Parameters
    ----------
    workspace, captured
        Participant evidence workspace and accepted captured submission directory.
    programme
        Run-owned frozen pool and acquisition commitment locations.
    preferences
        Public scientific units and normalization scales.
    executor_limits, executor_image
        Assigned isolated execution resources.

    Returns
    -------
    tuple
        The replay observations and all required numerical target scores.

    Raises
    ------
    ValueError
        If the captured checkpoint is not an acquired readout in this programme.
    """
    accepted_root = workspace.parent / "analysis_sources" / "cutoffs"
    context_path = captured / "reporting_context.json"
    if (
        captured.parent.absolute() != accepted_root.absolute()
        or any(path.is_symlink() for path in (accepted_root.parent, accepted_root, captured, context_path))
        or not captured.is_dir()
    ):
        raise ValueError("cutoff evaluation requires an accepted capture from this run")
    context = read_json_model(NumericalReportingContext, context_path)
    relative = PurePosixPath(context.checkpoint_id)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != context.checkpoint_id:
        raise ValueError("captured checkpoint is not an acquired programme readout")
    checkpoint = workspace / context.checkpoint_id
    if checkpoint.parent.resolve() != programme.evidence_root.resolve() or any(
        path.is_symlink() for path in (workspace, programme.evidence_root, checkpoint)
    ):
        raise ValueError("captured checkpoint is not an acquired programme readout")
    commitment_path = programme.commitment_root / f"{checkpoint.name}.json"
    if programme.commitment_root.is_symlink() or commitment_path.is_symlink():
        raise ValueError("acquisition commitment must be a private regular file")
    commitment = read_json_model(TrialDevReadoutCommitment, commitment_path)
    if context.target_ids != tuple(target.target_id for target in commitment.readout.targets):
        raise ValueError("captured target inventory differs from its acquisition commitment")
    replay = _replay_accepted_capture(
        workspace=workspace,
        captured=captured,
        context=context,
        executor_limits=executor_limits,
        analysis_language=analysis_language,
        executor_image=executor_image,
    )
    scores = grade_trialdev_readout_numbers(
        workspace=workspace,
        pool_root=programme.pool_root,
        readout_commitment=commitment,
        replay=replay,
        preferences=preferences,
    )
    return replay, scores


def evaluate_trialdev_programme(
    *,
    workspace: Path,
    programme: TrialDevActionSession,
    preferences: TrialDevSponsorPreferences,
    executor_limits: CodeExecutionLimitsV1,
    analysis_language: AnalysisMode = "python_r",
    executor_image: str | None = None,
    initial_evidence: TrialDevInitialEvidence | None = None,
) -> TrialDevProgrammeEvaluation:
    """Evaluate acquired evidence and an explicit terminal decision separately.

    Missing study reports have unavailable numerical loss without fabricating an
    analysis or executing a container. An unfinished programme is valued exactly
    as a stop at its last programme day: no market value, its incurred study
    costs and overhead. An explicit stop can be valued without a
    numerical report or study. Interim observations remain labelled and are not
    averaged into a primary score.

    Parameters
    ----------
    workspace, programme
        Participant workspace and its private acquisition/accounting session.
    preferences
        Task-owned clinical units and sponsor valuation.
    initial_evidence
        Original cohort context and scientific targets fixed by release construction.
    executor_limits, executor_image
        Resources for replaying accepted study analyses.

    Returns
    -------
    TrialDevProgrammeEvaluation
        Fixed-population development value and study diagnostics.

    Raises
    ------
    ValueError
        If acquired evidence, accepted captures or private accounting disagree.
    """
    initial_replay = None
    initial_scores: tuple[TrialDevNumericalRecovery, ...] = ()
    if initial_evidence is not None:
        context = initial_evidence.context
        match = _accepted_captures(workspace).get(context.checkpoint_id)
        if match is None:
            initial_replay = NumericalReplay(
                status="missing_result",
                entrypoint="",
                checkpoint_id=context.checkpoint_id,
                target_ids=context.target_ids,
                evidence_artifacts=context.evidence_artifacts,
                error="No initial analysis was committed before subsequent evidence or termination.",
            )
        else:
            captured, captured_context = match
            if captured_context != context:
                raise ValueError("initial report differs from its release-bound evidence context")
            initial_replay = _replay_accepted_capture(
                workspace=workspace,
                captured=captured,
                context=context,
                executor_limits=executor_limits,
                analysis_language=analysis_language,
                executor_image=executor_image,
            )
        initial_scores = grade_trialdev_observational_numbers(
            targets=initial_evidence.targets,
            replay=initial_replay,
            preferences=preferences,
        )
    ledger = read_json_model(TrialDevBudgetLedger, programme.budget_path)
    # Value an unfinished programme as the stop action would at its last day, without persisting it,
    # so abandoning a failing programme cannot raise its value.
    valued = (
        ledger
        if ledger.terminal is not None
        else finish_trialdev_programme(
            ledger, TrialDevTerminalDecision(decision="stop", programme_day=ledger.current_day)
        )
    )
    value = grade_trialdev_finished_programme(pool_root=programme.pool_root, ledger=valued, preferences=preferences)
    studies = []
    for report in select_trialdev_study_reports(workspace=workspace, programme=programme):
        if report.captured_relative_path is not None:
            replay, scores = evaluate_trialdev_cutoff(
                workspace=workspace,
                captured=workspace.parent / report.captured_relative_path,
                programme=programme,
                preferences=preferences,
                executor_limits=executor_limits,
                analysis_language=analysis_language,
                executor_image=executor_image,
            )
        else:
            checkpoint = workspace / report.checkpoint_id
            commitment = read_json_model(
                TrialDevReadoutCommitment, programme.commitment_root / f"{checkpoint.name}.json"
            )
            manifest = checkpoint / "study.json"
            replay = NumericalReplay(
                status="missing_result",
                entrypoint="",
                checkpoint_id=report.checkpoint_id,
                target_ids=tuple(target.target_id for target in commitment.readout.targets),
                evidence_artifacts=(
                    ReplayInputArtifactV1(
                        relative_path=manifest.relative_to(workspace).as_posix(),
                        sha256=sha256_file(manifest),
                        byte_length=manifest.stat().st_size,
                    ),
                ),
                error="No accepted analysis was captured for this acquired readout.",
            )
            scores = grade_trialdev_readout_numbers(
                workspace=workspace,
                pool_root=programme.pool_root,
                readout_commitment=commitment,
                replay=replay,
                preferences=preferences,
            )
        studies.append(TrialDevStudyEvaluation(report=report, replay=replay, scores=scores))
    return TrialDevProgrammeEvaluation(
        programme_complete=ledger.terminal is not None,
        value=value,
        studies=tuple(studies),
        initial_replay=initial_replay,
        initial_scores=initial_scores,
    )


def _initial_reporting_instructions(initial: TrialDevInitialEvidence) -> str:
    """Name each requested initial result with its measure and unit, in reporting order; never reference values."""
    measures = {
        "clinical_outcome": "case-mix-standardized mean difference (natural units)",
        "safety_risk": "case-mix-standardized risk difference (proportions)",
    }
    rows = "\n".join(
        f"| `{identity}` | {target.asset_id} minus control, {measures[target.endpoint]} | {target.unit} |"
        for identity, target in zip(initial.context.target_ids, initial.targets, strict=True)
    )
    return (
        "\n\nFirst results to report, in this order (interface/targets.csv), each with a 95% confidence interval, "
        "from the initial observational cohort as the evidence protocol defines them:\n\n"
        "| Target | Measure | Unit |\n| --- | --- | --- |\n" + rows + "\n\n" + CONTROL_REFERENCE_NOTE
    )


def run_trialdev_analysis(
    *,
    loop: TrialDevTaskAgent,
    prompt: str,
    configuration: TrialDevAnalysisConfiguration,
    executor_limits: CodeExecutionLimitsV1,
    analysis_language: AnalysisMode = "python_r",
    executor_image: str | None = None,
) -> TrialDevAnalysisRun:
    """Run and grade one observed-association analysis without a development ledger.

    Parameters
    ----------
    loop
        Agent with its staged initial evidence and writable analysis workspace.
    prompt
        Public clinical question.
    configuration
        Release-bound initial inputs, target references and normalization.
    executor_limits, executor_image
        Assigned isolated replay resources.

    Returns
    -------
    TrialDevAnalysisRun
        Completion status, reproducibility and recovery for every assigned target.

    Raises
    ------
    ValueError
        If initial evidence or captured context differs from the release.
    """
    context = configuration.initial_evidence.context
    capture = None
    error = None
    try:
        if loop.authored_programme is not None:
            raise ValueError("Standalone analysis cannot execute a development programme")
        verified_replay_inputs(loop.workdir, context.evidence_artifacts)
        if loop.reporting_context is not None and loop.reporting_context != context:
            raise ValueError("Standalone continuation differs from its released reporting context")
        loop.reporting_context = context
        numerical_owner.stage_reporting_interface(loop.workdir, context.target_ids)
        if loop.active_prompt_index is None:
            loop.begin_step(phase_id="evidence_analysis", step_id=context.checkpoint_id)
            loop.append_user_message(prompt + _initial_reporting_instructions(configuration.initial_evidence))
        capture = loop.run_until_report()
        loop.finish_step(status="completed")
    except _TASK_NONCOMPLETION as exception:
        error = str(exception)
    finally:
        loop.close()
    if capture is not None:
        # A limit observed after the report was captured (for example while closing the step) leaves it standing.
        error = None
    result = evaluate_trialdev_analysis(
        workspace=loop.workdir,
        configuration=configuration,
        report_relative_path=None if capture is None else capture.path.relative_to(loop.workdir.parent).as_posix(),
        execution_status="completed" if capture is not None else "model_noncompletion",
        error=error,
        executor_limits=executor_limits,
        analysis_language=analysis_language,
        executor_image=executor_image,
    )
    write_json_model(loop.workdir.parent / "analysis_run.json", result)
    return result


def evaluate_trialdev_analysis(
    *,
    workspace: Path,
    configuration: TrialDevAnalysisConfiguration,
    report_relative_path: str | None,
    execution_status: Literal["completed", "model_noncompletion"],
    error: str | None,
    executor_limits: CodeExecutionLimitsV1,
    analysis_language: AnalysisMode = "python_r",
    executor_image: str | None = None,
) -> TrialDevAnalysisRun:
    """Replay a captured standalone analysis and grade every assigned target.

    Parameters
    ----------
    workspace
        Task workspace containing the released participant inputs.
    configuration
        Release-bound evidence inventory, targets and normalization.
    report_relative_path
        Captured report beneath the task directory, or no submitted analysis.
    execution_status, error
        Preserved model execution outcome.
    executor_limits, executor_image
        Isolated replay resources from the run invocation.

    Returns
    -------
    TrialDevAnalysisRun
        Recomputed reproducibility and target recovery.

    Raises
    ------
    ValueError
        If the report, execution outcome or reporting context disagree.
    """
    context = configuration.initial_evidence.context
    if (report_relative_path is not None) != (execution_status == "completed"):
        raise ValueError("Standalone report presence differs from its execution outcome")
    if report_relative_path is not None:
        relative = PurePosixPath(report_relative_path)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != report_relative_path:
            raise ValueError("Standalone report must use a canonical task-relative path")
    if report_relative_path is None:
        replay = NumericalReplay(
            status="missing_result",
            entrypoint="",
            checkpoint_id=context.checkpoint_id,
            target_ids=context.target_ids,
            evidence_artifacts=context.evidence_artifacts,
            error="No standalone numerical analysis was submitted.",
        )
    else:
        captured_context = read_json_model(
            NumericalReportingContext, (workspace.parent / report_relative_path).parent / "reporting_context.json"
        )
        if captured_context != context:
            raise ValueError("Standalone report differs from its release-bound initial context")
        replay = replay_numerical_submission(
            workspace=workspace,
            submitted_project=(workspace.parent / report_relative_path).parent,
            executor_limits=executor_limits,
            analysis_language=analysis_language,
            executor_image=executor_image,
        )
    result = TrialDevAnalysisRun(
        execution_status=execution_status,
        replay=replay,
        scores=grade_trialdev_observational_numbers(
            targets=configuration.initial_evidence.targets,
            replay=replay,
            preferences=configuration.preferences,
        ),
        report_relative_path=report_relative_path,
        error=error,
    )
    return result


def run_trialdev_programme(
    *,
    loop: TrialDevTaskAgent,
    prompt: str,
    preferences: TrialDevSponsorPreferences,
    executor_limits: CodeExecutionLimitsV1,
    analysis_language: AnalysisMode = "python_r",
    executor_image: str | None = None,
    initial_evidence: TrialDevInitialEvidence | None = None,
) -> TrialDevProgrammeRun:
    """Execute and grade a prepared authored task without requiring a final report.

    Parameters
    ----------
    loop
        Configured participant agent with its private authored programme session.
    prompt
        Public task instructions and available initial evidence.
    preferences
        Task-owned sponsor preferences for numerical and development valuation.
    initial_evidence
        Fixed initial observational reporting context and grading references.
    executor_limits, executor_image
        Resources assigned to post-commit analysis replay.

    Returns
    -------
    TrialDevProgrammeRun
        Persisted execution status and separate development/analysis evaluation.

    Raises
    ------
    ValueError
        If the agent lacks an authored session or execution/accounting disagree.
        Provider interruptions and integrity failures propagate; they are not
        converted into model noncompletion or voluntary stopping.
    """
    programme = loop.authored_programme
    if programme is None:
        raise ValueError("authored execution requires a configured programme session")
    error = None
    try:
        ledger = read_json_model(TrialDevBudgetLedger, programme.budget_path)
        if ledger.pool_manifest_sha256 is None or ledger.pool_manifest_sha256 != sha256_file(
            programme.pool_root / "manifest.json"
        ):
            raise ValueError("programme must be bound to its released frozen pool before execution")
        manifest = read_json_model(TrialDevFrozenPool, programme.pool_root / "manifest.json")
        if any(target.outcome_unit != preferences.outcome_unit for target in manifest.targets):
            raise ValueError("sponsor outcome units differ from the released programme")
        if initial_evidence is not None:
            verified_replay_inputs(loop.workdir, initial_evidence.context.evidence_artifacts)
        if initial_evidence is not None and loop.reporting_context is None and not ledger.receipts:
            loop.reporting_context = initial_evidence.context
        if loop.reporting_context is not None:
            numerical_owner.stage_reporting_interface(loop.workdir, loop.reporting_context.target_ids)
        if loop.active_prompt_index is None:
            loop.begin_step(phase_id="development", step_id="programme")
            loop.append_user_message(
                prompt + ("" if initial_evidence is None else _initial_reporting_instructions(initial_evidence))
            )
        loop.run_programme()
    except _TASK_NONCOMPLETION as exception:
        error = str(exception)
    finally:
        loop.close()
    evaluation = evaluate_trialdev_programme(
        workspace=loop.workdir,
        programme=programme,
        preferences=preferences,
        executor_limits=executor_limits,
        analysis_language=analysis_language,
        executor_image=executor_image,
        initial_evidence=initial_evidence,
    )
    if evaluation.programme_complete:
        # A deadline, turn limit or termination observed after the stop or advancement was committed to the
        # ledger (for example while returning its tool result) leaves the validated decision standing.
        error = None
    elif error is None:
        raise ValueError("programme execution outcome differs from its terminal accounting")
    result = TrialDevProgrammeRun(
        execution_status="completed" if error is None else "model_noncompletion",
        error=error,
        evaluation=evaluation,
    )
    write_json_model(loop.workdir.parent / "programme_run.json", result)
    return result


def stage_participant_workspace(
    *,
    release: TrialDevEvaluatorRelease,
    task: TrialDevTaskV1,
    workspace: Path,
) -> None:
    """Copy only the analyst materials declared by the selected task."""

    source_root = release.participant.world_root(task)
    workspace.mkdir()
    for material in task.available_materials:
        relative = PurePosixPath(material.relative_path)
        source = source_root.joinpath(*relative.parts)
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"TrialDev participant material is unavailable: {material.relative_path}")
        destination = workspace.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def trialdev_task_artifact_v1(task_root: Path, relative_path: str) -> TrialDevReleaseArtifactV1:
    """Return the immutable byte identity of one regular task artifact."""

    path = task_root / relative_path
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"TrialDev checkpoint artifact is missing: {path}")
    return TrialDevReleaseArtifactV1(
        relative_path=relative_path,
        sha256=sha256_file(path),
        byte_length=path.stat().st_size,
    )


def _numerical_artifact_paths(task_root: Path, *, programme: bool) -> tuple[str, ...]:
    """Enumerate the private scientific record without following filesystem redirects."""
    paths = {"programme_run.json", "private/budget.json"} if programme else {"analysis_run.json"}
    for pattern in ("events*.jsonl", "conversation*.json", "provider_responses*.jsonl"):
        for path in task_root.glob(pattern):
            if path.is_symlink() or not path.is_file():
                raise ValueError("Numerical run logs require regular files")
            paths.add(path.relative_to(task_root).as_posix())
    directories: tuple[Path, ...] = (task_root / "private", task_root / "analysis_sources")
    if programme:
        directories += (task_root / "workspace" / "evidence", task_root / "workspace" / "analysis_ready")
    for directory in directories:
        if directory.is_symlink():
            raise ValueError("Numerical scientific record cannot contain symbolic links")
        for path in directory.rglob("*"):
            if path.is_symlink() or not (path.is_file() or path.is_dir()):
                raise ValueError("Numerical scientific record requires regular files and directories")
            if path.is_file():
                paths.add(path.relative_to(task_root).as_posix())
    return tuple(sorted(paths))


def verify_trialdev_task_run_artifacts_v1(*, task_root: Path, run: TrialDevTaskRunV1) -> None:
    """Reject a finalized task whose content-addressed artifacts changed."""

    _verify_task_artifacts(task_root=task_root, artifacts=run.artifacts, owner="task")
    if run.programme_relative_path is not None or run.analysis_relative_path is not None:
        actual = set(_numerical_artifact_paths(task_root, programme=run.programme_relative_path is not None))
        declared = {item.relative_path for item in run.artifacts}
        if actual != declared:
            raise ValueError("Numerical scientific record differs from its finalized artifact inventory")


def _verify_task_artifacts(
    *,
    task_root: Path,
    artifacts: tuple[TrialDevReleaseArtifactV1, ...],
    owner: Literal["checkpoint", "task"],
) -> None:
    """Verify confined immutable task artifacts for runtime consumers."""

    root = Path(task_root)
    resolved_root = root.resolve(strict=True)
    for artifact in artifacts:
        relative = PurePosixPath(artifact.relative_path)
        path = root.joinpath(*relative.parts)
        current = root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError(f"TrialDev {owner} artifact path contains a symlink: {artifact.relative_path}")
        if (
            not path.is_file()
            or path.is_symlink()
            or not path.resolve(strict=True).is_relative_to(resolved_root)
            or path.stat().st_size != artifact.byte_length
        ):
            raise ValueError(f"TrialDev {owner} artifact changed: {artifact.relative_path}")


def _recover_numerical_task(
    *,
    release: TrialDevEvaluatorRelease,
    task: TrialDevTaskV1,
    task_root: Path,
) -> TrialDevTaskRunV1 | None:
    """Authenticate a numerical terminal result before constructing a provider."""
    path = task_root / "run.json"
    if path.is_symlink():
        raise ValueError("Authored run record cannot be a symbolic link")
    if not path.exists():
        return None
    run = read_json_model(TrialDevTaskRunV1, path)
    if (
        run.release_id != release.manifest.release_id
        or run.release_manifest_sha256 != release.participant.release_manifest_sha256
        or run.task_id != task.task_id
        or run.stream != task.stream
        or (run.programme_relative_path is None and run.analysis_relative_path is None)
    ):
        raise ValueError("Numerical run identity differs from the requested release task")
    verify_trialdev_task_run_artifacts_v1(task_root=task_root, run=run)
    if run.analysis_relative_path is not None:
        analysis = read_json_model(TrialDevAnalysisRun, task_root / run.analysis_relative_path)
        configuration = release.analysis_configuration(task)
        if configuration is None:
            raise ValueError("Standalone result has no release-bound numerical configuration")
        context = configuration.initial_evidence.context
        if (
            analysis.execution_status != run.execution_status
            or analysis.error != run.error
            or analysis.replay.checkpoint_id != context.checkpoint_id
            or analysis.replay.target_ids != context.target_ids
            or analysis.replay.evidence_artifacts != context.evidence_artifacts
            or (analysis.report_relative_path is not None) != (run.execution_status == "completed")
            or (
                analysis.report_relative_path is not None
                and analysis.report_relative_path not in {item.relative_path for item in run.artifacts}
            )
            or analysis.scores
            != grade_trialdev_observational_numbers(
                targets=configuration.initial_evidence.targets,
                replay=analysis.replay,
                preferences=configuration.preferences,
            )
        ):
            raise ValueError("Standalone analysis outcome differs from its release-bound task result")
        return run
    assert run.programme_relative_path is not None
    result = read_json_model(TrialDevProgrammeRun, task_root / run.programme_relative_path)
    if (
        result.execution_status != run.execution_status
        or result.error != run.error
        or result.evaluation.programme_complete != (run.execution_status == "completed")
    ):
        raise ValueError("Authored programme outcome differs from the task result")
    return run


def recover_terminal_trialdev_task(
    *,
    release: TrialDevEvaluatorRelease,
    task_id: str,
    output_root: Path,
) -> TrialDevTaskRunV1 | None:
    """Return a verified terminal task without constructing runtime providers.

    Parameters
    ----------
    release:
        Loaded evaluator release that owns the task and checkpoint.
    task_id:
        Exact participant task identifier.
    output_root:
        Directory containing per-task run directories.

    Returns
    -------
    TrialDevTaskRunV1 | None
        The verified terminal run, or ``None`` when execution was interrupted
        before a terminal result was written.

    Raises
    ------
    ValueError
        If a present checkpoint, workspace, or terminal run is inconsistent.
    """

    matches = tuple(task for task in release.participant.tasks if task.task_id == task_id)
    if len(matches) != 1:
        raise ValueError(f"TrialDev release does not contain one task_id={task_id!r}")
    task = matches[0]
    task_root = Path(output_root) / task.task_id
    if not task_root.exists():
        return None
    if release.authored_configuration(task) is None and release.analysis_configuration(task) is None:
        raise ValueError("TrialDev recovery requires a current numerical or authored-programme configuration")
    return _recover_numerical_task(release=release, task=task, task_root=task_root)


def trialdev_programme_system_prompt(data_view: Literal["raw", "analysis_ready"]) -> str:
    """Return the portfolio-management system prompt; the committee brief holds the rules."""
    return (
        "Progression and approval are simplified benchmark rules, not regulatory practice. "
        "Candidates are distinguished only by the evidence the programme acquires. "
        "You lead clinical development of a sponsor's candidate treatments. Analyse the evidence, run studies with "
        "the programme tools, and end by stopping development or advancing one compound and dose. The committee "
        "brief sets the portfolio, study rules, costs, budget, time horizon and how the sponsor values an approved "
        "treatment. All studies share one programme clock; requesting a readout or waiting moves it forward for "
        "every study. A look's deadline_day is a programme day counted from programme start, not a duration after "
        "the study starts; recruitment begins after the study's start-up and follow-up runs from each "
        "participant's enrolment. A readout uses its planned look even when few or no participants have "
        "completed, and its files never change afterwards. Each readout and your final decision first rerun your saved analysis and keep its results as "
        "your report for that point; time and money already spent stay spent. If that rerun fails or does not report "
        "exactly the targets now in interface/targets.csv, the readout or decision is refused and nothing is "
        "committed, unless you set accept_incomplete=true, which keeps the incomplete result. Each rerun also "
        "restarts the interactive Python/R session; saved files remain."
        + (
            " Each readout is also provided as consolidated tables under analysis_ready/."
            if data_view == "analysis_ready"
            else ""
        )
    )


def run_trialdev_task(
    *,
    release: TrialDevEvaluatorRelease,
    task_id: str,
    provider: LLMProvider,
    output_root: Path,
    max_turns_per_step: int,
    max_tokens: int,
    max_context_characters: int,
    provider_capability: ProviderModelCapabilityV1 | None = None,
    watchdog_seconds: int,
    tool_choice: ToolChoiceV1,
    executor_image: str | None,
    executor_limits: CodeExecutionLimitsV1,
    analysis_language: AnalysisMode = "python_r",
    resume: bool = False,
    verbose: bool = False,
    provider_unavailability_policy: Literal["charge_all_v1", "defer_unanswered_v1"] = "charge_all_v1",
) -> TrialDevTaskRunV1:
    """Run one evidence or sequential task with raw-report and deterministic grade custody."""

    matches = tuple(task for task in release.participant.tasks if task.task_id == task_id)
    if len(matches) != 1:
        raise ValueError(f"TrialDev release does not contain one task_id={task_id!r}")
    task = matches[0]
    task_root = Path(output_root) / task.task_id
    workspace = task_root / "workspace"
    configuration = release.authored_configuration(task)
    analysis_configuration = release.analysis_configuration(task)
    if configuration is None and analysis_configuration is None:
        raise ValueError("TrialDev execution requires a current numerical or authored-programme configuration")
    resuming = task_root.exists()
    if resuming:
        if not resume:
            raise FileExistsError(f"TrialDev task output already exists: {task_root}")
        recovered = _recover_numerical_task(release=release, task=task, task_root=task_root)
        if recovered is not None:
            return recovered
        if not workspace.is_dir() or workspace.is_symlink():
            raise ValueError("Numerical continuation workspace is missing or invalid")
    else:
        task_root.mkdir(parents=True)
        stage_participant_workspace(release=release, task=task, workspace=workspace)
    programme = None
    if configuration is not None:
        budget_path = task_root / "private" / "budget.json"
        if not resuming:
            write_json_model(
                budget_path,
                TrialDevBudgetLedger(
                    budget_usd=configuration.budget_usd,
                    pool_manifest_sha256=configuration.pool_manifest_sha256,
                    recruitment_seed=configuration.recruitment_seed,
                ),
            )
        else:
            ledger = read_json_model(TrialDevBudgetLedger, budget_path)
            if (
                ledger.pool_manifest_sha256 != configuration.pool_manifest_sha256
                or ledger.recruitment_seed != configuration.recruitment_seed
                or ledger.budget_usd != configuration.budget_usd
            ):
                raise ValueError("Authored continuation ledger differs from the released programme")
        programme = TrialDevActionSession(
            pool_root=release.root / "evaluation" / "worlds" / task.case_id / "pool",
            budget_path=budget_path,
            evidence_root=workspace / "evidence",
            commitment_root=task_root / "private" / "commitments",
            recruitment_seed=configuration.recruitment_seed,
        )
    loop = TrialDevTaskAgent(
        provider=provider,
        workdir=workspace,
        system_prompt=(
            trialdev_programme_system_prompt(task.data_view)
            if configuration is not None
            else "You are the statistical lead for this evidence analysis. Estimate the requested results and "
            "submit a concise report."
        ),
        max_tokens=max_tokens,
        max_turns_per_step=max_turns_per_step,
        tool_choice=tool_choice,
        verbose=verbose,
        conversation_log_path=task_root / "conversation.json",
        event_log_path=task_root / "events.jsonl",
        provider_log_path=task_root / "provider_responses.jsonl",
        program_id=task.task_id,
        scenario_id=task.case_id,
        objective_id=task.stream.value,
        executor_image=executor_image,
        executor_limits=executor_limits,
        analysis_language=analysis_language,
        deadline_monotonic=time.monotonic() + watchdog_seconds,
        agent_time_budget_seconds=watchdog_seconds,
        provider_unavailability_policy=provider_unavailability_policy,
        max_context_chars=max_context_characters,
        provider_capability=provider_capability,
        authored_programme=programme,
        reporting_context=(
            None if analysis_configuration is None else analysis_configuration.initial_evidence.context
        ),
        resume=resuming,
        data_view=task.data_view,
    )
    materials = "\n".join(f"- `{item.relative_path}`: {item.description}" for item in task.available_materials)
    prompt = (
        trialdev_clinical_text(task.clinical_question)
        + "\n\nAvailable materials (paths relative to the task workspace):\n"
        + materials
    )
    result: TrialDevProgrammeRun | TrialDevAnalysisRun
    if configuration is not None:
        result = run_trialdev_programme(
            loop=loop,
            prompt=prompt,
            preferences=configuration.preferences,
            executor_limits=executor_limits,
            analysis_language=analysis_language,
            executor_image=executor_image,
            initial_evidence=configuration.initial_evidence,
        )
    else:
        assert analysis_configuration is not None
        result = run_trialdev_analysis(
            loop=loop,
            prompt=prompt,
            configuration=analysis_configuration,
            executor_limits=executor_limits,
            analysis_language=analysis_language,
            executor_image=executor_image,
        )
    summary = TrialDevTaskRunV1(
        release_id=release.manifest.release_id,
        release_manifest_sha256=release.participant.release_manifest_sha256,
        task_id=task.task_id,
        stream=task.stream,
        execution_status=result.execution_status,
        report_relative_paths=(),
        ingress_relative_paths=(),
        grade_relative_paths=(),
        programme_relative_path="programme_run.json" if configuration is not None else None,
        analysis_relative_path="analysis_run.json" if analysis_configuration is not None else None,
        error=result.error,
        artifacts=tuple(
            trialdev_task_artifact_v1(task_root, relative)
            for relative in _numerical_artifact_paths(task_root, programme=configuration is not None)
        ),
    )
    write_json_model(task_root / "run.json", summary)
    return summary


__all__ = [
    "recover_terminal_trialdev_task",
    "run_trialdev_task",
    "trialdev_task_artifact_v1",
    "verify_trialdev_task_run_artifacts_v1",
]
