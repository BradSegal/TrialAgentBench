"""Independent whole-run TrialDev ranking from authenticated exported custody."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from trialagentbench_verifier.archive import opened_release
from trialagentbench_verifier.authored_readouts import _Readout, verify_trialdev_readout_records
from trialagentbench_verifier.authored_results import (
    TrialDevAnalysisRun,
    TrialDevNumericalRecovery,
    TrialDevProgrammeRun,
    numerical_target_statuses,
)
from trialagentbench_verifier.authored_scoring import verify_numerical_recovery, verify_trialdev_prospective_scores
from trialagentbench_verifier.contracts import CheckResultV1, VerificationReportV1
from trialagentbench_verifier.executor import analysis_project_artifacts
from trialagentbench_verifier.grading_key import read_trialdev_world_strata
from trialagentbench_verifier.io import (
    read_json_object,
    release_archive,
    release_asset_scope,
    sha256_file,
    trialdev_task_run_paths,
)
from trialagentbench_verifier.numerical import NumericalReplay, NumericalReportingContext, read_numerical_table
from trialagentbench_verifier.numerical_scoring import bounded_recovery
from trialagentbench_verifier.release import _release_identity, _report
from trialagentbench_verifier.release_set import _portable_relative_path
from trialagentbench_verifier.resource_accounting import (
    _Terminal,
    reconstruct_trialdev_terminal_value,
    verify_trialdev_terminal_value,
)
from trialagentbench_verifier.response import (
    TrialDevRankingItemV1,
    TrialDevRankingMeanV1,
    TrialDevRunRankingV1,
    TrialDevTargetStatus,
)
from trialagentbench_verifier.scorer import current_scorer
from trialagentbench_verifier.units import compatible_units


class _Projection(BaseModel):
    """Read needed fields from a separately authenticated complete wire record."""

    model_config = ConfigDict(extra="ignore", frozen=True, allow_inf_nan=False)


class _InitialTarget(BaseModel):
    """Authenticated initial numerical target carried by the released key."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    asset_id: str
    endpoint: Literal["clinical_outcome", "safety_risk"]
    unit: str = Field(min_length=1)
    decision_scale: float = Field(gt=0)
    comparison_sample_size: int = Field(ge=2)
    point_identified: Literal[True]
    oracle_value: float
    estimand: Literal["standardized_observed_association"]
    oracle_monte_carlo_standard_error: float = Field(default=0.0, ge=0)


class _Invocation(_Projection):
    schema_id: Literal["ariadne.trialdev.run_invocation/v1"]
    release_id: str
    release_manifest_sha256: str
    scheduled_task_ids: tuple[str, ...] = Field(min_length=1)
    scheduled_task_streams: dict[str, Literal["evidence_analysis", "portfolio_management"]]
    model: str
    executor_image: str
    checksum: str


class _Artifact(_Projection):
    relative_path: str
    sha256: str
    byte_length: int = Field(ge=0)


class _Run(_Projection):
    schema_id: Literal["ariadne.trialdev.task_run/v1"]
    release_id: str
    release_manifest_sha256: str
    task_id: str
    stream: Literal["evidence_analysis", "portfolio_management"]
    execution_status: Literal["completed", "model_noncompletion", "evaluator_error"]
    report_relative_paths: tuple[()]
    ingress_relative_paths: tuple[()]
    grade_relative_paths: tuple[()]
    artifacts: tuple[_Artifact, ...]
    final_state: None
    analysis_relative_path: str | None = None
    programme_relative_path: str | None = None
    error: str | None
    checksum: str


class _IndexedResult(_Projection):
    task_id: str
    stream: Literal["evidence_analysis", "portfolio_management"]
    execution_status: Literal["completed", "model_noncompletion", "evaluator_error"]
    checksum: str


class _Index(_Projection):
    schema_id: Literal["ariadne.trialdev.run_index/v1"]
    release_id: str
    scheduled_task_ids: tuple[str, ...]
    model: str
    invocation_checksum: str
    results: tuple[_IndexedResult, ...]


class _Material(_Projection):
    relative_path: str


class _Resources(_Projection):
    budget_usd: float = Field(gt=0)


class _Task(_Projection):
    task_id: str = Field(pattern=r"^td-[0-9a-f]{12}$")
    case_id: str = Field(pattern=r"^tdw-[0-9a-f]{4,16}$")
    stream: Literal["evidence_analysis", "portfolio_management"]
    available_materials: tuple[_Material, ...]
    data_view: Literal["raw", "analysis_ready"] = "raw"
    resources: _Resources | None = None


_T = TypeVar("_T", bound=BaseModel)


def _bound(model: type[_T], path: Path) -> _T:
    return model.model_validate(read_json_object(path))


def _member(root: Path, relative: str) -> Path:
    _portable_relative_path(relative)
    path = root / relative
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("TrialDev custody member is absent or unsafe")
    return path


def _agrees(left: JsonValue, right: JsonValue) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_agrees(value, right[key]) for key, value in left.items())
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_agrees(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, float) and isinstance(right, float):
        return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10)
    return type(left) is type(right) and left == right


def _checked_artifacts(task_root: Path, run: _Run) -> dict[str, _Artifact]:
    artifacts = {row.relative_path: row for row in run.artifacts}
    if len(artifacts) != len(run.artifacts):
        raise ValueError("TrialDev task repeats an artifact")
    for relative, identity in artifacts.items():
        path = _member(task_root, relative)
        if path.stat().st_size != identity.byte_length:
            raise ValueError("TrialDev task custody artifact changed")
    return artifacts


class _NumericalPreferences(_Projection):
    outcome_unit: str
    benefit_unit: float = Field(gt=0)


class _InitialEvidence(_Projection):
    context: NumericalReportingContext
    targets: tuple[_InitialTarget, ...] = Field(min_length=1)


class _AnalysisConfiguration(_Projection):
    schema_id: Literal["trialdev.numerical-analysis/v1"]
    preferences: _NumericalPreferences
    initial_evidence: _InitialEvidence


def _verify_numerical_capture(
    *,
    task_root: Path,
    project: Path | None,
    context: NumericalReportingContext,
    replay: NumericalReplay,
    artifacts: dict[str, _Artifact],
    executor_image: str,
) -> None:
    """Bind a retained numerical replay to its exact captured source and CSV."""
    if (
        replay.checkpoint_id != context.checkpoint_id
        or replay.target_ids != context.target_ids
        or replay.evidence_artifacts != context.evidence_artifacts
    ):
        raise ValueError("Numerical replay differs from its reporting context")
    if project is None and replay.reproducible:
        raise ValueError("Reproduced numerical analysis has no captured source")
    if project is not None:
        captured = NumericalReportingContext.model_validate_json((project / "reporting_context.json").read_bytes())
        if captured != context:
            raise ValueError("Captured analysis context differs from the release")
        for path in (project / "reporting_context.json",):
            if path.relative_to(task_root).as_posix() not in artifacts:
                raise ValueError("Captured numerical analysis omits committed report or context")
        if replay.reproducible:
            if re.fullmatch(r"sha256:[0-9a-f]{64}", executor_image) is None:
                raise ValueError("Numerical analysis requires an immutable executor identity")
            if replay.entrypoint not in {"analysis.py", "analysis.R"}:
                raise ValueError("Numerical analysis has an unsupported entrypoint")
            language = "python" if replay.entrypoint == "analysis.py" else "r"
            if any(execution.language != language for execution in replay.executions):
                raise ValueError("Numerical analysis execution language differs from its source")
            if replay.program_artifacts != analysis_project_artifacts(_member(project, replay.entrypoint)):
                raise ValueError("Numerical analysis source differs from committed replay")
            csv_path = project / "estimates.csv"
            if sha256_file(csv_path) != replay.submitted_csv_sha256:
                raise ValueError("Numerical analysis CSV differs from committed replay")
            if read_numerical_table(csv_path, target_ids=context.target_ids) != replay.estimates:
                raise ValueError("Numerical analysis estimates differ from its CSV")
            for relative in (
                *[Path(row.relative_path).relative_to("scratch") for row in replay.program_artifacts],
                Path("estimates.csv"),
            ):
                if (project / relative).relative_to(task_root).as_posix() not in artifacts:
                    raise ValueError("Numerical analysis omitted source or CSV custody")


def _initial_recovery(
    *,
    replay: NumericalReplay,
    scores: tuple[TrialDevNumericalRecovery, ...],
    initial: _InitialEvidence,
    preferences: _NumericalPreferences,
) -> float | None:
    """Reconstruct initial association loss for either task stream."""
    identities = tuple(f"{target.asset_id}:{target.endpoint}" for target in initial.targets)
    if (
        identities != replay.target_ids
        or set(identities) != {row.target_id for row in scores}
        or len(scores) != len(identities)
    ):
        raise ValueError("Initial numerical score census differs from the release")
    estimates = {row.target_id: row for row in replay.estimates or ()}
    by_target = {row.target_id: row for row in scores}
    losses: list[float | None] = []
    for identity, target in zip(replay.target_ids, initial.targets, strict=True):
        clinical = target.endpoint == "clinical_outcome"
        unit = preferences.outcome_unit if clinical else "risk difference"
        if target.estimand != "standardized_observed_association" or not target.point_identified:
            raise ValueError("Initial numerical reference must be an identified observed association")
        if not compatible_units(target.unit, unit):
            raise ValueError("Initial numerical reference has incompatible units")
        scale = preferences.benefit_unit if clinical else 0.01
        estimate = estimates.get(identity)
        verify_numerical_recovery(
            score=by_target[identity],
            estimate=estimate,
            reference=target.oracle_value,
            scale=scale,
            confidence=0.95,
            reproducible=replay.reproducible,
        )
        score = by_target[identity]
        point, interval = score.normalized_absolute_error, score.normalized_interval_loss
        losses.append(None if point is None or interval is None else point / 2 + interval / 2)
    return None if any(value is None for value in losses) else mean(value for value in losses if value is not None)


def _analysis_values(*, release: Path, task_root: Path, run: _Run, task: _Task, executor_image: str) -> float | None:
    """Recompute observed-association recovery from release keys and captured CSV."""
    artifacts = _checked_artifacts(task_root, run)
    if run.analysis_relative_path is None or run.analysis_relative_path not in artifacts:
        raise ValueError("Numerical analysis result lacks committed custody")
    result = TrialDevAnalysisRun.model_validate_json(_member(task_root, run.analysis_relative_path).read_bytes())
    if (result.execution_status, result.error) != (run.execution_status, run.error):
        raise ValueError("Numerical analysis and task execution disagree")
    configuration = _AnalysisConfiguration.model_validate_json(
        _member(release / "evaluation/tasks" / task.task_id, "analysis.json").read_bytes()
    )
    initial = configuration.initial_evidence
    context, replay = initial.context, result.replay
    identities = tuple(f"{target.asset_id}:{target.endpoint}" for target in initial.targets)
    if identities != context.target_ids or identities != replay.target_ids or len(set(identities)) != len(identities):
        raise ValueError("Numerical analysis changed the released target inventory")
    if replay.checkpoint_id != context.checkpoint_id or replay.evidence_artifacts != context.evidence_artifacts:
        raise ValueError("Numerical analysis changed its initial evidence context")
    _verify_numerical_capture(
        task_root=task_root,
        project=(
            None if result.report_relative_path is None else _member(task_root, result.report_relative_path).parent
        ),
        context=context,
        replay=replay,
        artifacts=artifacts,
        executor_image=executor_image,
    )
    return _initial_recovery(
        replay=replay, scores=result.scores, initial=initial, preferences=configuration.preferences
    )


class _ProgrammeLedger(_Projection):
    terminal: _Terminal | None


class _AuthoredConfiguration(_Projection):
    schema_id: Literal["trialdev.authored-programme/v1"]
    pool_manifest_sha256: str
    budget_usd: float
    preferences: _NumericalPreferences
    initial_evidence: _InitialEvidence


def _authored_values(
    *, release: Path, task_root: Path, run: _Run, task: _Task, executor_image: str
) -> tuple[float | None, float]:
    """Independently reconstruct authored diagnostics and development dollar value."""
    artifacts = _checked_artifacts(task_root, run)
    if run.programme_relative_path is None or run.programme_relative_path not in artifacts:
        raise ValueError("Authored programme result lacks committed custody")
    result = TrialDevProgrammeRun.model_validate_json(_member(task_root, run.programme_relative_path).read_bytes())
    if (result.execution_status, result.error) != (run.execution_status, run.error):
        raise ValueError("Authored programme and task execution disagree")
    config_path = _member(release / "evaluation/tasks" / task.task_id, "authored_programme.json")
    configuration = _AuthoredConfiguration.model_validate_json(config_path.read_bytes())
    pool = release / "evaluation/worlds" / task.case_id / "pool"
    if (
        task.resources is None
        or task.resources.budget_usd != configuration.budget_usd
        or sha256_file(pool / "manifest.json") != configuration.pool_manifest_sha256
    ):
        raise ValueError("Authored resources or pool differ from the released task")
    evaluation = result.evaluation
    if evaluation.initial_replay is None:
        raise ValueError("Authored programme omits the initial analysis denominator")
    captures = {}
    for context_path in (task_root / "analysis_sources/cutoffs").glob("*/reporting_context.json"):
        project = _member(task_root, context_path.relative_to(task_root).as_posix()).parent
        context = NumericalReportingContext.model_validate_json(context_path.read_bytes())
        if context.checkpoint_id in captures:
            raise ValueError("Authored programme repeats an accepted reporting cutoff")
        captures[context.checkpoint_id] = project
    initial = configuration.initial_evidence
    _verify_numerical_capture(
        task_root=task_root,
        project=captures.get(initial.context.checkpoint_id),
        context=initial.context,
        replay=evaluation.initial_replay,
        artifacts=artifacts,
        executor_image=executor_image,
    )
    recovery = _initial_recovery(
        replay=evaluation.initial_replay,
        scores=evaluation.initial_scores,
        initial=initial,
        preferences=configuration.preferences,
    )
    workspace = task_root / "workspace"
    latest: dict[str, tuple[float, str]] = {}
    for manifest in (workspace / "evidence").glob("*/study.json"):
        readout = _Readout.model_validate_json(manifest.read_bytes())
        checkpoint_id = manifest.parent.relative_to(workspace).as_posix()
        previous = latest.get(readout.study_id)
        if previous is not None and previous[0] == readout.readout_day:
            raise ValueError("Study has ambiguous acquired readouts at the same cutoff")
        if previous is None or previous[0] < readout.readout_day:
            latest[readout.study_id] = (readout.readout_day, checkpoint_id)
        verify_trialdev_readout_records(
            pool_root=pool,
            commitment_path=_member(task_root, "private/commitments/" + manifest.parent.name + ".json"),
            readout_root=manifest.parent,
        )
    if {study.report.study_id: study.report.checkpoint_id for study in evaluation.studies} != {
        study_id: checkpoint for study_id, (_, checkpoint) in latest.items()
    }:
        raise ValueError("Authored result omits or replaces a latest acquired study readout")
    for study in evaluation.studies:
        if study.report.programme_complete != evaluation.programme_complete:
            raise ValueError("Study reporting status differs from programme completion")
        study_project = captures.get(study.report.checkpoint_id)
        relative = None if study_project is None else study_project.relative_to(task_root).as_posix()
        if relative != study.report.captured_relative_path:
            raise ValueError("Study result differs from its accepted reporting cutoff")
        context = NumericalReportingContext(
            checkpoint_id=study.report.checkpoint_id,
            target_ids=study.replay.target_ids,
            evidence_artifacts=study.replay.evidence_artifacts,
        )
        _verify_numerical_capture(
            task_root=task_root,
            project=study_project,
            context=context,
            replay=study.replay,
            artifacts=artifacts,
            executor_image=executor_image,
        )
    verify_trialdev_prospective_scores(
        pool_root=pool, configuration_path=config_path, workspace=workspace, evaluation=evaluation
    )
    ledger_path = _member(task_root, "private/budget.json")
    if "private/budget.json" not in artifacts:
        raise ValueError("Authored programme omits its retained resource ledger")
    ledger = _ProgrammeLedger.model_validate_json(ledger_path.read_bytes())
    if (ledger.terminal is not None) != evaluation.programme_complete:
        raise ValueError("Authored completion differs from the retained terminal decision")
    value = reconstruct_trialdev_terminal_value(
        ledger_path=ledger_path, configuration_path=config_path, pool_path=pool / "manifest.json"
    )
    verify_trialdev_terminal_value(value, evaluation.value)
    return recovery, value.net_value_usd


def ranking_means(rows: list[TrialDevRankingItemV1]) -> tuple[TrialDevRankingMeanV1, ...]:
    """Retain separate task streams and presentations while averaging linked worlds.

    Parameters
    ----------
    rows
        Scheduled ranking items, including unscored and failed tasks.

    Returns
    -------
    tuple of TrialDevRankingMeanV1
        One equal-world summary per stream, presentation, profile and metric.
        Observational recovery also carries bounded recovery, the transform
        of the complete-panel mean loss with the same world weights. Development
        value scores an unfinished programme at its incurred costs; every summary
        still counts model noncompletion.
    """
    groups: dict[tuple[str, str, str, str, int, str], list[TrialDevRankingItemV1]] = defaultdict(list)
    for row in rows:
        groups[
            row.stream,
            row.data_view,
            row.clinical_profile_id,
            row.scenario_profile_id,
            row.asset_count,
            "observational_loss",
        ].append(row)
        if row.authored_programme:
            groups[
                row.stream,
                row.data_view,
                row.clinical_profile_id,
                row.scenario_profile_id,
                row.asset_count,
                "net_development_value_usd",
            ].append(row)
    means = []
    for (stream, data_view, clinical, scenario, asset_count, metric), members in sorted(groups.items()):
        errors = sum(row.execution_status == "evaluator_error" for row in members)
        values = [
            (row.net_development_value_usd if metric == "net_development_value_usd" else row.observational_loss)
            for row in members
        ]
        missing = any(value is None for value in values)
        incomplete = any(row.execution_status == "incomplete_run" for row in members)
        world_values: dict[str, list[float]] = defaultdict(list)
        for member, value in zip(members, values, strict=True):
            if value is not None:
                world_values[member.world_id].append(value)
        estimate = None if errors or incomplete or missing else mean(mean(group) for group in world_values.values())
        means.append(
            TrialDevRankingMeanV1(
                stream=stream,
                data_view=data_view,
                clinical_profile_id=clinical,
                scenario_profile_id=scenario,
                asset_count=asset_count,
                metric=metric,
                scheduled_tasks=len(members),
                scored_tasks=sum(value is not None for value in values),
                scheduled_worlds=len({row.world_id for row in members}),
                unscored_task_counts=dict(
                    Counter(
                        "unscored" if row.execution_status == "completed" else row.execution_status
                        for row in members
                        if (
                            row.net_development_value_usd
                            if metric == "net_development_value_usd"
                            else row.observational_loss
                        )
                        is None
                    )
                ),
                target_status_counts=(
                    dict(Counter(status for row in members for status in row.initial_target_statuses.values()))
                    if metric == "observational_loss"
                    else None
                ),
                evaluator_errors=errors,
                model_noncompletion=sum(row.execution_status == "model_noncompletion" for row in members),
                scored_worlds=len(world_values),
                observed_mean=mean(mean(values) for values in world_values.values()) if world_values else None,
                bounded_recovery=bounded_recovery(estimate) if metric == "observational_loss" else None,
                mean=estimate,
            )
        )
    return tuple(means)


def verify_trialdev_ranking(
    release: Path,
    result_export: Path,
    *,
    expected_release_sha256: str,
    expected_result_export_sha256: str | None,
    expected_invocation_sha256: str,
    development: bool = False,
) -> VerificationReportV1:
    """Recompute the primary TrialDev ranking from one complete custody export.

    The invocation digest must be retained prospectively. The result archive and
    its contained execution proofs require evaluator custody. This performs no
    fresh program execution and makes no current-authorization claim.
    """
    with (
        release_archive(release, expected_sha256=expected_release_sha256) as released_archive,
        release_archive(result_export, expected_sha256=expected_result_export_sha256) as archived,
    ):
        with opened_release(archived) as exported:
            from trialagentbench_verifier.campaign import _ResultExportManifestV1

            manifest = _ResultExportManifestV1.model_validate(read_json_object(exported / "manifest.json"))
            run_sources = [source for source in manifest.sources if source.kind == "run"]
            if len(run_sources) != 1:
                raise ValueError("TrialDev ranking requires exactly one original invocation")
            source_invocation_path = exported / run_sources[0].archive_prefix / "invocation.json"
            if sha256_file(source_invocation_path) != expected_invocation_sha256:
                raise ValueError("TrialDev schedule differs from the prospectively trusted invocation")
            source_invocation = _bound(_Invocation, source_invocation_path)
            with opened_release(
                released_archive,
                expected_sha256=expected_release_sha256,
                task_ids=source_invocation.scheduled_task_ids,
            ) as root:
                stream, candidate, digest = _release_identity(root)
                report_digests: tuple[str, ...] = ()
                if stream != "trialdev" or manifest.release_manifest_sha256 != digest:
                    raise ValueError("TrialDev ranking inputs identify different released bytes")
                release_id = str(read_json_object(root / "release_manifest.json")["release_id"])
                if manifest.release_id != release_id:
                    raise ValueError("TrialDev result export identifies a different release")
                grade_sources = [source for source in manifest.sources if source.kind == "grade"]
                if len(grade_sources) != 1:
                    raise ValueError("TrialDev ranking requires exactly one regraded invocation")
                graded = exported / grade_sources[0].archive_prefix
                original = exported / run_sources[0].archive_prefix
                if sha256_file(graded / "invocation.json") != expected_invocation_sha256:
                    raise ValueError("TrialDev schedule differs from the prospectively trusted invocation")
                invocation = source_invocation
                if invocation.release_id != release_id or invocation.release_manifest_sha256 != digest:
                    raise ValueError("TrialDev invocation identifies a different release")
                scheduled = invocation.scheduled_task_ids
                if len(set(scheduled)) != len(scheduled) or set(invocation.scheduled_task_streams) != set(scheduled):
                    raise ValueError("TrialDev invocation has an invalid scheduled denominator")
                observed = trialdev_task_run_paths(graded)
                originals = trialdev_task_run_paths(original)
                if not observed.keys() <= set(scheduled):
                    raise ValueError("TrialDev export contains unscheduled task results")
                if originals.keys() != observed.keys():
                    raise ValueError("TrialDev regrade omitted or added a committed primary task")
                for source_root in (original, graded):
                    index_path = source_root / "run_index.json"
                    if not index_path.is_file():
                        continue
                    index = _bound(_Index, index_path)
                    if (index.release_id, index.scheduled_task_ids, index.model, index.invocation_checksum) != (
                        release_id,
                        scheduled,
                        invocation.model,
                        invocation.checksum,
                    ) or tuple(row.task_id for row in index.results) != scheduled:
                        raise ValueError("TrialDev run index changes its invocation denominator")
                    if {row.task_id for row in index.results} != observed.keys():
                        raise ValueError("TrialDev run index and task custody census differ")
                    for item in index.results:
                        indexed_run = _bound(_Run, (originals if source_root == original else observed)[item.task_id])
                        if (item.stream, item.execution_status, item.checksum) != (
                            indexed_run.stream,
                            indexed_run.execution_status,
                            indexed_run.checksum,
                        ):
                            raise ValueError("TrialDev run index changes the committed status or task checksum")
                strata = read_trialdev_world_strata(root)
                rows = []
                for task_id in scheduled:
                    task = _Task.model_validate(
                        read_json_object(_member(root / "participant/tasks", f"{task_id}/task.json"))
                    )
                    if task.task_id != task_id or task.stream != invocation.scheduled_task_streams[task_id]:
                        raise ValueError("TrialDev task differs from the prospective schedule")
                    stratum = strata.stratum(task.case_id)
                    authored = (root / "evaluation/tasks" / task_id / "authored_programme.json").is_file()
                    if task.stream == "portfolio_management" and not authored:
                        raise ValueError("TrialDev ranking requires an authored-programme configuration")
                    if (
                        task.stream == "evidence_analysis"
                        and not (root / "evaluation/tasks" / task_id / "analysis.json").is_file()
                    ):
                        raise ValueError("TrialDev ranking requires a numerical-analysis configuration")
                    run = None if task_id not in observed else _bound(_Run, observed[task_id])
                    unfinished = run is None and any(
                        (parent / task_id).exists() for parent in (original, original / "tasks")
                    )
                    status = (
                        ("incomplete_run" if unfinished else "not_started") if run is None else run.execution_status
                    )
                    config_type = _AuthoredConfiguration if authored else _AnalysisConfiguration
                    config_name = "authored_programme.json" if authored else "analysis.json"
                    configuration = config_type.model_validate_json(
                        _member(root / "evaluation/tasks" / task_id, config_name).read_bytes()
                    )
                    released_ids = tuple(
                        f"{target.asset_id}:{target.endpoint}" for target in configuration.initial_evidence.targets
                    )
                    if released_ids != configuration.initial_evidence.context.target_ids or len(
                        set(released_ids)
                    ) != len(released_ids):
                        raise ValueError("Released initial context differs from its target inventory")
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
                    net_value = None
                    if run is not None:
                        if (run.task_id, run.stream, run.release_id, run.release_manifest_sha256) != (
                            task_id,
                            task.stream,
                            release_id,
                            digest,
                        ):
                            raise ValueError("TrialDev run differs from its released task")
                        raw = _bound(_Run, originals[task_id])
                        if (raw.task_id, raw.stream, raw.release_id, raw.release_manifest_sha256) != (
                            task_id,
                            task.stream,
                            release_id,
                            digest,
                        ):
                            raise ValueError("TrialDev committed primary identifies a different task or release")
                        if (
                            raw.programme_relative_path != run.programme_relative_path
                            or raw.analysis_relative_path != run.analysis_relative_path
                            or raw.report_relative_paths != run.report_relative_paths
                            or raw.ingress_relative_paths != run.ingress_relative_paths
                            or raw.execution_status != run.execution_status
                            or raw.error != run.error
                        ):
                            raise ValueError("TrialDev regrade substituted the committed primary ingress")
                        raw_artifacts = _checked_artifacts(originals[task_id].parent, raw)
                        # These results are independently reconstructed below. Fresh
                        # replay can change execution timings without changing any
                        # participant submission, accepted capture or acquired evidence.
                        derived_results = {run.programme_relative_path, run.analysis_relative_path}
                        if {
                            key: value
                            for key, value in raw_artifacts.items()
                            if key not in raw.grade_relative_paths and key not in derived_results
                        } != {
                            row.relative_path: row
                            for row in run.artifacts
                            if row.relative_path not in run.grade_relative_paths
                            and row.relative_path not in derived_results
                        }:
                            raise ValueError("TrialDev regrade changed committed primary response or replay custody")
                        if authored and status != "evaluator_error":
                            recovery, net_value = _authored_values(
                                release=root,
                                task_root=observed[task_id].parent,
                                run=run,
                                task=task,
                                executor_image=invocation.executor_image,
                            )
                            verified_programme = TrialDevProgrammeRun.model_validate_json(
                                _member(observed[task_id].parent, str(run.programme_relative_path)).read_bytes()
                            )
                            initial_target_statuses = numerical_target_statuses(
                                verified_programme.evaluation.initial_scores
                            )
                            acquired_target_statuses = {
                                study.report.study_id: numerical_target_statuses(study.scores)
                                for study in verified_programme.evaluation.studies
                            }
                        elif not authored and status != "evaluator_error":
                            recovery = _analysis_values(
                                release=root,
                                task_root=observed[task_id].parent,
                                run=run,
                                executor_image=invocation.executor_image,
                                task=task,
                            )
                            verified_analysis = TrialDevAnalysisRun.model_validate_json(
                                _member(observed[task_id].parent, str(run.analysis_relative_path)).read_bytes()
                            )
                            initial_target_statuses = numerical_target_statuses(verified_analysis.scores)
                    if set(initial_target_statuses) != set(configuration.initial_evidence.context.target_ids):
                        raise ValueError("Initial result target census differs from the released task")
                    rows.append(
                        TrialDevRankingItemV1(
                            task_id=task_id,
                            world_id=task.case_id,
                            stream=task.stream,
                            data_view=task.data_view,
                            budget_usd=None if task.resources is None else task.resources.budget_usd,
                            clinical_profile_id=stratum.clinical_profile_id,
                            scenario_profile_id=stratum.scenario_ref,
                            asset_count=stratum.candidate_count,
                            execution_status=status,
                            run_checksum=None if run is None else run.checksum,
                            observational_loss=None if status == "evaluator_error" else recovery,
                            initial_target_statuses=initial_target_statuses,
                            acquired_target_statuses=acquired_target_statuses,
                            authored_programme=authored,
                            net_development_value_usd=net_value,
                        )
                    )
                means = ranking_means(rows)
                scope = release_asset_scope(root)
                released_task_count = (
                    len(scope.catalog.tasks)
                    if scope is not None
                    else len(tuple((root / "participant/tasks").glob("*/task.json")))
                )
                computed = TrialDevRunRankingV1(
                    scorer=current_scorer(),
                    release_id=release_id,
                    release_manifest_sha256=digest,
                    invocation_checksum=invocation.checksum,
                    model_id=invocation.model,
                    scope=(
                        "complete_release_task_inventory"
                        if len(scheduled) == released_task_count
                        else "scheduled_panel"
                    ),
                    items=tuple(rows),
                    profile_means=tuple(means),
                )
                reported = TrialDevRunRankingV1.model_validate(read_json_object(graded / "ranking.json"))
                matched = _agrees(reported.model_dump(mode="json"), computed.model_dump(mode="json"))
                return _report(
                    stream="trialdev",
                    operation="analyse-trialdev",
                    candidate_digest=candidate,
                    release_digest=digest,
                    input_kind="archive",
                    input_sha256=sha256_file(archived),
                    expected_input_sha256=expected_result_export_sha256,
                    report_digests=report_digests,
                    development=development,
                    checks=(
                        CheckResultV1(
                            check_id="score.trialdev.scheduled_ranking",
                            scope_id=invocation.checksum,
                            status="pass" if matched else "fail",
                            observed=computed.model_dump(mode="json"),
                            expected=reported.model_dump(mode="json"),
                            artifact_paths=("ranking.json", "invocation.json"),
                            failure_code=None if matched else "trialdev_primary_ranking_mismatch",
                        ),
                    ),
                )
