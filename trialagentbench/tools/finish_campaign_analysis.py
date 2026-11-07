"""Compute campaign comparisons directly from completed native grades."""

from __future__ import annotations

import csv
import json
import tempfile
from html import escape
from itertools import combinations
from pathlib import Path
from typing import Literal
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict
from trialagentbench_verifier.campaign import (
    TrialEvalCampaignAttemptV1,
    TrialEvalCampaignPlanV1,
    TrialEvalCampaignUnavailableV1,
    UnsupportedTrialEvalPanelError,
    analyse_trial_eval_campaign_v1,
    load_trial_eval_campaign_grade_source_v1,
    reproduced_answer_v1,
    trial_eval_answer_rows_v1,
)
from trialagentbench_verifier.campaign_contracts import BudgetCensoringV1, CensoredRunV1, LeaderboardRowV1
from trialagentbench_verifier.campaign_inputs import DevelopmentCampaignInputsV1
from trialagentbench_verifier.development_campaign import (
    TrialDevCampaignExportV1,
    analyse_trialdev_campaign,
    render_trialdev_comparisons,
)
from trialagentbench_verifier.io import sha256_file
from trialagentbench_verifier.public_results import (
    CampaignLeaderboardV1,
    PublicTrialDevCampaignV1,
    PublicTrialEvalCampaignV1,
    public_trialdev_campaign,
    public_trialeval_campaign,
    trialeval_leaderboard_rows,
)
from trialagentbench_verifier.release_set import ReleaseSetManifestV1, role_filename
from trialagentbench_verifier.replay_scoring import (
    AnswerReplaysV1,
    ReplayOutcomeV1,
    TrialEvalReplayEvidenceV1,
    answer_replays_path,
    answers_csv,
    load_replay_evidence_v1,
    read_answer_replays_v1,
    replay_once_v1,
    require_saved_source_v1,
    rerun_timeout_v1,
    saved_source_sha256,
)
from trialagentbench_verifier.review import write_review
from trialagentbench_verifier.scorer import current_scorer

from trialagentbench.contracts.campaign import CampaignPlanV1
from trialagentbench.contracts.core.runs import RunCoverageV1
from trialagentbench.contracts.release.archive_safety import materialized_release_archive
from trialagentbench.io import read_json_model
from trialagentbench.tools.export_results import export_results_v1
from trialagentbench.tools.finish_run import _write_once
from trialagentbench.util.campaign_control import CAMPAIGN_BUDGET_SCOPE, exhausted_budgets


def campaign_censoring(root: Path, plan: CampaignPlanV1) -> BudgetCensoringV1 | None:
    """Classify every pending scheduled task, refusing any not explained by spending-limit exhaustion.

    A pending task is budget-censored only when the campaign ledger records the
    permanent exhaustion of its model's or the campaign's spending limit. Such
    work can never run, because budgets never refill. Any other pending task
    (a crash or interruption) can resume, so finishing is refused.

    Returns
    -------
    BudgetCensoringV1 or None
        Censoring of every planned job, or None when no task is censored.

    Raises
    ------
    ValueError
        If a run has pending tasks without recorded budget exhaustion.
    """
    exhausted = exhausted_budgets(root / "requests.sqlite", plan.configuration)
    # Identities a converged release left unresolved for provider or infrastructure reasons are declared
    # in operator-exclusions.json: reported and unscored like budget censoring, never charged to the model.
    exclusions_path = root / "operator-exclusions.json"
    excluded: dict[str, list[str]] = (
        json.loads(exclusions_path.read_text())["tasks"] if exclusions_path.is_file() else {}
    )
    models = {model.name: model.model for model in plan.configuration.models}
    runs = []
    for run in plan.runs:
        coverage = read_json_model(RunCoverageV1, root / run.relative_run_root / "coverage.json")
        pending = tuple(task for task in coverage.unit_ids if task not in coverage.completed_unit_ids)
        if pending and set(pending) == set(excluded.get(run.job_id, ())):
            pass
        elif pending and CAMPAIGN_BUDGET_SCOPE not in exhausted and run.model_name not in exhausted:
            raise ValueError(
                f"Finish requires all scheduled terminal checkpoints; {run.job_id} has {len(pending)} pending "
                "tasks without an exhausted spending limit. Resume pending work first."
            )
        runs.append(
            CensoredRunV1(
                job_id=run.job_id,
                model_name=run.model_name,
                model_id=models[run.model_name],
                benchmark=run.benchmark,
                repetition=run.repetition,
                # Schedules follow release construction order; shared records list tasks by opaque ID.
                scheduled_task_ids=tuple(sorted(coverage.unit_ids)),
                censored_task_ids=tuple(sorted(pending)),
            )
        )
    if not any(run.censored_task_ids for run in runs):
        return None
    operator_excluded = {run.model_name for run in plan.runs if excluded.get(run.job_id)}
    return BudgetCensoringV1(
        campaign_plan_sha256=str(plan.checksum),
        exhausted_models=tuple(
            sorted({name for name in exhausted if name != CAMPAIGN_BUDGET_SCOPE} | operator_excluded)
        ),
        campaign_budget_exhausted=CAMPAIGN_BUDGET_SCOPE in exhausted,
        runs=tuple(runs),
    )


def write_campaign_index(root: Path, plan: CampaignPlanV1) -> Path:
    """Link canonical campaign outputs without recalculating scientific results.

    The portable page is evaluator-side: it links submitted work and verification
    records which may disclose grading references. Only the explicitly marked
    public projections are intended for unrestricted sharing.
    """

    def link(path: Path, label: str) -> str:
        if not path.is_file():
            return ""
        relative = path.relative_to(root).as_posix()
        return f'<li><a href="{quote(relative)}">{escape(label)}</a></li>'

    censoring_path = root / "comparisons" / "censoring.json"
    censoring = (
        BudgetCensoringV1.model_validate_json(censoring_path.read_bytes()) if censoring_path.is_file() else None
    )
    censored = {} if censoring is None else {run.job_id: len(run.censored_task_ids) for run in censoring.runs}
    rows: list[str] = []
    for run in plan.runs:
        native = root / run.relative_run_root
        coverage = read_json_model(RunCoverageV1, native / "coverage.json")
        result = root / "results" / run.job_id
        links = [
            link(native / "coverage.json", "Scheduled and terminal task identities"),
            link(native / "run_index.json", "Submitted work and task locations"),
            link(native / "provider_telemetry_summary.json", "Model usage"),
            link(result / "grade" / "RESULTS_SUMMARY.md", "Numerical recovery, completion and failure origins"),
            link(result / "grade" / "summary.json", "Canonical task grades"),
            link(result / "grade" / "ranking.json", "Development results and completion"),
        ]
        links.extend(
            link(path, f"Verification: {path.stem}") for path in sorted((result / "verification").glob("*.json"))
        )
        rows.append(
            f"<section><h2>{escape(run.job_id)}</h2><p>{escape(run.benchmark)} · "
            f"repetition {run.repetition} · terminal checkpoints "
            f"{len(coverage.completed_unit_ids)}/{len(coverage.unit_ids)}"
            f"{f' · budget-censored {censored[run.job_id]}' if censored.get(run.job_id) else ''}</p>"
            "<p>Terminal checkpoints include unsuccessful attempts; they do not establish "
            "reproduced analyses or scientific success. Consult the native grades below.</p>"
            f'<ul>{"".join(links)}</ul></section>'
        )
    comparisons: list[str] = []
    for benchmark in plan.configuration.panel.benchmarks:
        directory = root / "comparisons" / benchmark
        links = [
            link(directory / "review" / "index.html", "Scientific comparison report"),
            link(
                directory / "review" / "model_contrasts.csv",
                "Paired comparisons — standard errors and detectable differences",
            ),
            link(
                directory / "review" / "comparisons.csv",
                "Paired comparisons — standard errors and detectable differences",
            ),
            link(directory / "analysis.json", "Analysis and unavailable-comparison reasons"),
            link(directory / "public-results.json", "Public summary — shareable projection"),
            link(directory / "development-results.json", "Development results — unqualified rehearsal"),
        ]
        links.extend(link(path, path.stem.replace("_", " ")) for path in sorted(directory.glob("*.csv")))
        comparisons.append(f'<h3>{escape(benchmark)}</h3><ul>{"".join(links)}</ul>')
    comparisons.append(
        "<h3>Leaderboard</h3><ul>"
        + link(root / "comparisons" / "leaderboard.json", "Leaderboard — shareable, world-clustered intervals")
        + link(root / "comparisons" / "leaderboard.csv", "Leaderboard table")
        + "</ul>"
        + "<p>Leaderboard standard errors describe each model. Detectable differences belong to the paired "
        "comparisons linked above; TrialEval headlines use the C5 decision metrics.</p>"
    )
    if censoring is not None:
        comparisons.append(
            "<h3>Budget censoring</h3><p>Exhausted spending limits left these tasks unrun. They are unscored, "
            "not participant failures; comparisons and the leaderboard use only tasks every model completed.</p>"
            + "".join(f'<p role="alert"><strong>Warning:</strong> {escape(text)}</p>' for text in censoring.warnings)
            + "<ul>"
            + "".join(
                f"<li>{escape(row.benchmark)} · {escape(row.model_id)}: {row.budget_censored_tasks}/"
                f"{row.scheduled_tasks} censored ({row.censoring_rate:.0%})</li>"
                for row in censoring.models
            )
            + link(censoring_path, "Censoring record")
            + "</ul>"
        )
    diagnostics = "".join(
        link(path, f"{directory}: {path.stem.replace('_', ' ')}")
        for directory in ("behaviour-summary", "resources")
        for path in sorted((root / directory).glob("*"))
        if path.suffix in {".html", ".md", ".csv", ".svg"}
    )
    html = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Campaign results · TrialAgentBench</title><style>"
        "body{font:17px/1.6 system-ui,sans-serif;color:#183c38;background:#faf9f5;max-width:70rem;"
        "margin:auto;padding:2rem}a{color:#165c70;overflow-wrap:anywhere}h1,h2,h3{line-height:1.2}"
        "section{border-top:1px solid #b6c9c3;margin-top:2rem;padding-top:1rem}"
        "li{margin:.4rem 0}code{overflow-wrap:anywhere}a:focus-visible{outline:3px solid #b75e24}"
        "</style><main><h1>Campaign results</h1>"
        "<p>Inspect the submitted work, its independent replay and the resulting scientific outcomes.</p>"
        f"<p>Analysis language: <strong>{escape(plan.configuration.panel.analysis_language)}</strong>. "
        f"Native runs: {len(plan.runs)}. Input mode: {escape(plan.configuration.input_mode)}.</p>"
        "<p>This evaluator report links records that may contain reference answers. For public sharing, "
        "use the designated public summary files. A diagnostic subset may have individual grades "
        "without supporting a benchmark ranking.</p><ul>"
        + link(root / "configuration.json", "Exact evaluation configuration")
        + link(root / "campaign.json", "Frozen campaign plan")
        + "</ul><section><h2>Scientific comparisons</h2>"
        + "".join(comparisons)
        + "</section><section><h2>Behavior and resources</h2><ul>"
        + diagnostics
        + "</ul></section>"
        + "".join(rows)
        + "</main></html>\n"
    )
    destination = root / "index.html"
    _write_once(destination, html)
    return destination


class _DevelopmentResults(BaseModel):
    """Keep rehearsal aggregates incompatible with qualified public result schemas."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_id: Literal["trialagentbench.development_campaign_results/v1"] = (
        "trialagentbench.development_campaign_results/v1"
    )
    status: Literal["unqualified"] = "unqualified"
    input_manifest_sha256: str
    result: PublicTrialEvalCampaignV1 | PublicTrialDevCampaignV1 | TrialEvalCampaignUnavailableV1


def _write_aggregate(
    destination: Path,
    result: PublicTrialEvalCampaignV1 | PublicTrialDevCampaignV1 | TrialEvalCampaignUnavailableV1,
    plan: CampaignPlanV1,
) -> None:
    if plan.configuration.input_mode == "development":
        (destination / "development-results.json").write_text(
            _DevelopmentResults(
                input_manifest_sha256=plan.configuration.expected_release_manifest_sha256, result=result
            ).model_dump_json(indent=2)
            + "\n",
            encoding="utf-8",
        )
    else:
        (destination / "public-results.json").write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _write_leaderboard(
    destination: Path,
    plan: CampaignPlanV1,
    rows: tuple[LeaderboardRowV1, ...],
    censoring: BudgetCensoringV1 | None,
) -> None:
    """Write the shareable leaderboard as JSON and as a flat CSV table."""
    analysis = plan.configuration.analysis
    split = plan.configuration.panel.split
    leaderboard = CampaignLeaderboardV1(
        scorer=current_scorer(),
        campaign_plan_sha256=str(plan.checksum),
        release_set_manifest_sha256=plan.configuration.expected_release_manifest_sha256,
        split=split,
        confidence_level=analysis.confidence_level,
        bootstrap_resamples=analysis.bootstrap_repetitions,
        bootstrap_seed=analysis.seed,
        rows=rows,
        # A withheld leaderboard keeps each model's censored count in its rows, not task names.
        budget_censoring=None if split == "withheld" else censoring,
        warnings=() if censoring is None else censoring.warnings,
    )
    (destination / "leaderboard.json").write_text(leaderboard.model_dump_json(indent=2) + "\n", encoding="utf-8")
    with (destination / "leaderboard.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LeaderboardRowV1.model_fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            row.model_dump(mode="json") | {"scorer": row.scorer.model_dump_json()} for row in leaderboard.rows
        )


def combined_trialeval_plan(root: Path, plan: CampaignPlanV1) -> TrialEvalCampaignPlanV1:
    """Derive the combined plan solely from prospectively frozen per-run plans.

    Repetition one is the primary by design. Later repetitions remain linked
    stochastic duplicates in the existing variability analysis; no outcome can
    select a primary. A retry run's attempts are retries of the same task's
    primary in the job it repeats. TrialDev instead averages repetitions within
    each world.
    """
    runs = tuple(run for run in plan.runs if run.benchmark == "trialeval")
    sources = {
        run.job_id: read_json_model(TrialEvalCampaignPlanV1, root / str(run.analysis_plan_path)) for run in runs
    }
    if not sources:
        raise ValueError("No TrialEval runs were scheduled")
    first = sources[runs[0].job_id]
    identity = (
        first.release_id,
        first.release_manifest_sha256,
        first.participant_manifest_checksum,
        first.prospective_analysis_plan_sha256,
    )
    attempts: list[TrialEvalCampaignAttemptV1] = []
    for run in runs:
        source = sources[run.job_id]
        if sha256_file(root / str(run.analysis_plan_path)) != run.analysis_plan_sha256:
            raise ValueError("Native TrialEval plan differs from its frozen identity")
        # A retry run covers only its failed tasks, so its cells are a subset of the panel's.
        if (
            source.release_id,
            source.release_manifest_sha256,
            source.participant_manifest_checksum,
            source.prospective_analysis_plan_sha256,
        ) != identity or not (
            set(source.cells) <= set(first.cells) if run.retry_of_job_id else source.cells == first.cells
        ):
            raise ValueError("Combined TrialEval plans differ in scientific condition or task cells")
        if run.retry_of_job_id is None:
            attempts.extend(
                row.model_copy(
                    update={
                        "attempt_role": "primary" if run.repetition == 1 else "stochastic_duplicate",
                        "scheduled_unit_id": row.attempt_id,
                    }
                )
                for row in source.attempts
            )
        else:
            originals = {row.task_id: row.attempt_id for row in sources[run.retry_of_job_id].attempts}
            attempts.extend(
                row.model_copy(
                    update={
                        "attempt_role": "retry",
                        "retry_of_attempt_id": originals[row.task_id],
                        "scheduled_unit_id": originals[row.task_id],
                    }
                )
                for row in source.attempts
            )
    payload = first.model_dump(exclude={"checksum", "attempts", "model_comparisons"})
    return TrialEvalCampaignPlanV1.model_validate(
        {
            **payload,
            "attempts": attempts,
            "model_comparisons": tuple(combinations(sorted(row.model for row in plan.configuration.models), 2)),
        }
    )


REPLAY_STORE = "trialeval-replays"
"""Campaign directory of answer replay outcomes, keyed by bank, task and saved source (a compute cache)."""


def replay_trialeval_answers(
    root: Path,
    plan: CampaignPlanV1,
    grade_roots: tuple[Path, ...],
    *,
    bank: Path,
    bank_manifest_sha256: str,
    release_id: str,
    release_manifest_sha256: str,
    workers: int | None = None,
) -> TrialEvalReplayEvidenceV1:
    """Replay every reproduced answer to a banked task not already in the campaign's replay store.

    Each replay is one job in the offline grading pool under the queued executor capacity, so one slow
    script occupies one slot rather than serialising its task's replays. An answer's outcomes are stored
    under ``REPLAY_STORE`` as soon as all of its replays finish; re-finishing, or finishing a campaign grown
    with a new archive, replays only answers whose (task, saved source, bank) is not stored yet.

    ``workers`` sets the number of simultaneous replay containers (default: the campaign's executor
    capacity); it changes throughput only, since a replay that times out under load is rerun exclusively
    once the parallel replays have finished.

    Returns
    -------
    TrialEvalReplayEvidenceV1
        The bank, loaded once, and every stored answer's outcomes against it.
    """
    from trialagentbench_verifier.execution_contracts import ExecutorEnvironmentV1
    from trialagentbench_verifier.executor import executor_session_capacity

    from trialagentbench.contracts.core.runs import TrialEvalRunConfigV1
    from trialagentbench.util.processes import process_imap

    evidence = load_replay_evidence_v1(
        bank, expected_manifest_sha256=bank_manifest_sha256, expected_release_id=release_id, replays=None
    )
    store = root / REPLAY_STORE
    jobs: dict[Path, dict[str, object]] = {}
    for grade in grade_roots:
        executor = read_json_model(ExecutorEnvironmentV1, grade / "replay_executor.json")
        interface = read_json_model(
            TrialEvalRunConfigV1, grade / "run_config.json"
        ).experiment_condition.reporting_interface
        source = load_trial_eval_campaign_grade_source_v1(
            "",
            expected_release_id=release_id,
            expected_release_manifest_sha256=release_manifest_sha256,
            expected_archive_sha256=None,
            grade_roots=(grade,),
        )
        for row in source.rows:
            if row.task_id not in evidence.bank.records or reproduced_answer_v1(row) is None:
                continue
            source_sha256 = saved_source_sha256(row.replay_program_artifacts)
            path = answer_replays_path(
                store, bank_manifest_sha256=bank_manifest_sha256, task_id=row.task_id, source_sha256=source_sha256
            )
            if path.exists() or path in jobs:
                continue
            (entrypoint,) = (
                artifact.relative_path.removeprefix("scratch/")
                for artifact in row.replay_program_artifacts
                if artifact.relative_path in {"scratch/analysis.py", "scratch/analysis.R"}
            )
            workspace = grade / "logs" / f"{row.task_id}_workspace" / "item"
            require_saved_source_v1(
                workspace,
                entrypoint,  # type: ignore[arg-type]
                task_id=row.task_id,
                source_sha256=source_sha256,
            )
            jobs[path] = {
                "task_id": row.task_id,
                "primary_target_id": row.target.binding.estimand_id,
                "workspace": workspace,
                "entrypoint": entrypoint,
                "image": executor.image_id,
                "limits": executor.limits,
                "interface": interface,
            }

    def replay(unit: tuple[Path, int]) -> ReplayOutcomeV1:
        path, index = unit
        return replay_once_v1(bank=evidence.bank, replay_index=index, **jobs[path])  # type: ignore[arg-type]

    def save(path: Path, outcomes: list[ReplayOutcomeV1]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        answer = AnswerReplaysV1(
            task_id=str(jobs[path]["task_id"]),
            source_sha256=path.stem,
            bank_manifest_sha256=bank_manifest_sha256,
            replays=tuple(outcomes),
        )
        _write_once(path, answer.model_dump_json(indent=1) + "\n")

    if jobs:
        units = [
            (path, record.replay_index)
            for path, arguments in sorted(jobs.items())
            for record in evidence.bank.records[str(arguments["task_id"])]
        ]
        capacity = workers or plan.configuration.operations.executor_capacity
        print(f"Replaying {len(jobs)} TrialEval answers ({len(units)} replays, {capacity} at once)", flush=True)
        outcomes: list[ReplayOutcomeV1] = []
        timed_out: dict[Path, list[ReplayOutcomeV1]] = {}
        with executor_session_capacity(capacity, queue=True):
            for (path, _), outcome in zip(units, process_imap(replay, units, workers=capacity), strict=True):
                outcomes.append(outcome)
                if len(outcomes) == len(evidence.bank.records[str(jobs[path]["task_id"])]):
                    if any(item.status == "timeout" for item in outcomes):
                        timed_out[path] = outcomes
                    else:
                        save(path, outcomes)
                    outcomes = []
            # A timeout under parallel load is rerun once with the host to itself. Doing so after the
            # parallel replays drains the pool once instead of once per timeout.
            reruns = sum(item.status == "timeout" for items in timed_out.values() for item in items)
            if reruns:
                print(f"Rerunning {reruns} timed-out replays one at a time", flush=True)
            for path, items in sorted(timed_out.items()):
                save(
                    path,
                    [
                        (
                            rerun_timeout_v1(bank=evidence.bank, replay_index=item.replay_index, **jobs[path])
                            if item.status == "timeout"
                            else item
                        )
                        for item in items
                    ],
                )
    return TrialEvalReplayEvidenceV1(
        bank=evidence.bank,
        bank_manifest_sha256=bank_manifest_sha256,
        answers=read_answer_replays_v1(store, bank_manifest_sha256=bank_manifest_sha256),
    )


def finish_comparisons(
    root: Path,
    plan: CampaignPlanV1,
    manifest: ReleaseSetManifestV1 | DevelopmentCampaignInputsV1,
    *,
    replay_bank: tuple[Path, str] | None = None,
    replay_workers: int | None = None,
    oracle_values: tuple[Path, str] | None = None,
) -> None:
    """Reproduce all scheduled model comparisons without provider calls or fresh simulations.

    Budget-censored tasks are excluded from every model's comparison and
    leaderboard panel; their counts, rates and warnings are retained. With
    ``replay_bank`` (directory and expected manifest SHA-256), TrialEval answers
    are scored by replay at the true effect; without it, at the truth on their
    original datasets; ``replay_workers`` sets replay concurrency.
    ``trialeval/trialeval_answers.csv`` lists every observed answer's score and the
    fixed-policy anchors, whether or not the paired analysis is available.
    """
    configuration = plan.configuration
    censoring = campaign_censoring(root, plan)
    anchors = {row.filename: row.sha256 for row in manifest.artifacts}
    evaluator_role = "evaluator" if configuration.panel.split == "public" else "withheld-evaluator"
    leaderboard: list[LeaderboardRowV1] = []
    output = root / "comparisons"
    with tempfile.TemporaryDirectory(prefix=".trialagentbench-comparisons-", dir=root) as temporary:
        working = Path(temporary)
        if "trialdev" in configuration.panel.benchmarks:
            evaluator_name = role_filename(anchors, "trialdev", evaluator_role)
            exports = []
            for run in plan.runs:
                if run.benchmark == "trialdev":
                    archives = tuple((root / "results" / run.job_id / "export").glob("*.zip"))
                    if len(archives) != 1:
                        raise ValueError("Each TrialDev job must retain exactly one full custody export")
                    exports.append(TrialDevCampaignExportV1(job_id=run.job_id, path=archives[0], sha256=None))
            report = analyse_trialdev_campaign(
                configuration.release_directory / evaluator_name,
                expected_release_sha256=anchors[evaluator_name],
                plan=plan,
                expected_plan_checksum=str(plan.checksum),
                exports=tuple(exports),
                oracle_values=None if oracle_values is None else oracle_values[0],
                expected_oracle_values_sha256=None if oracle_values is None else oracle_values[1],
                censoring=censoring,
                grade_roots={
                    run.job_id: root / "results" / run.job_id / "grade"
                    for run in plan.runs
                    if run.benchmark == "trialdev"
                },
            )
            # Portable relative locations remain resolvable from the copied export index.
            from pydantic import TypeAdapter

            index = tuple(
                row.model_copy(update={"path": Path("../..") / row.path.relative_to(root)}) for row in exports
            )
            report = report.model_copy(update={"exports": index})
            leaderboard.extend(report.leaderboard)
            render_trialdev_comparisons(report, working / "trialdev")
            _write_aggregate(
                working / "trialdev",
                public_trialdev_campaign(report, policy=configuration.analysis, split=configuration.panel.split),
                plan,
            )
            (working / "trialdev" / "exports.json").write_bytes(
                TypeAdapter(tuple[TrialDevCampaignExportV1, ...]).dump_json(index, indent=2)
            )
        unrun = set() if censoring is None else {row.job_id for row in censoring.runs if row.wholly_censored}
        selected = tuple(run for run in plan.runs if run.benchmark == "trialeval")
        graded = tuple(run for run in selected if run.job_id not in unrun)
        # With every TrialEval item censored there is no result to export; censoring.json reports it.
        if "trialeval" in configuration.panel.benchmarks and not (unrun and not graded):
            evaluator_name = role_filename(anchors, "trialeval", evaluator_role)
            combined = combined_trialeval_plan(root, plan)
            from trialagentbench.tools.primary_selection import select_trialeval_primary_sources

            combined, grade_roots, native_roots = select_trialeval_primary_sources(
                root,
                combined,
                graded,
                views=working / "selected-grade-sources",
                evaluator=configuration.release_directory / evaluator_name,
                evaluator_sha256=anchors[evaluator_name],
            )
            censored_tasks = {} if censoring is None else censoring.censored_by_job("trialeval")
            censored_attempts = frozenset(
                attempt.attempt_id
                for run in selected
                for attempt in read_json_model(TrialEvalCampaignPlanV1, root / str(run.analysis_plan_path)).attempts
                if attempt.task_id in censored_tasks.get(run.job_id, frozenset())
            )
            destination = working / "trialeval"
            destination.mkdir()
            (destination / "plan.json").write_text(combined.model_dump_json(indent=2) + "\n", encoding="utf-8")
            if not (output / "trialeval" / "export").exists():
                with materialized_release_archive(
                    configuration.release_directory / evaluator_name,
                    expected_sha256=anchors[evaluator_name],
                    in_place=True,
                ) as release:
                    export_results_v1(
                        release_root=release.root,
                        output_dir=destination / "export",
                        run_roots=native_roots,
                        grade_roots=grade_roots,
                        verification_roots=(),
                        profile="analysis_reproduction",
                    )
            replay = (
                None
                if replay_bank is None
                else replay_trialeval_answers(
                    root,
                    plan,
                    grade_roots,
                    bank=replay_bank[0],
                    bank_manifest_sha256=replay_bank[1],
                    workers=replay_workers,
                    release_id=combined.release_id,
                    release_manifest_sha256=combined.release_manifest_sha256,
                )
            )
            # Every observed answer is scored even when the paired analysis below is unavailable
            # (a C5-only campaign) or restricted to the tasks every model completed.
            answers = trial_eval_answer_rows_v1(
                expected_release_id=combined.release_id,
                expected_release_manifest_sha256=combined.release_manifest_sha256,
                plan=combined,
                grade_roots=grade_roots,
                replay=replay,
            )
            (destination / "trialeval_answers.csv").write_text(answers_csv(answers), encoding="utf-8")
            public_result: PublicTrialEvalCampaignV1 | TrialEvalCampaignUnavailableV1
            try:
                analysis = analyse_trial_eval_campaign_v1(
                    "",
                    expected_release_id=combined.release_id,
                    expected_release_manifest_sha256=combined.release_manifest_sha256,
                    expected_archive_sha256=None,
                    grade_roots=grade_roots,
                    plan=combined,
                    expected_campaign_plan_sha256=combined.checksum,
                    bootstrap_resamples=configuration.analysis.bootstrap_repetitions,
                    confidence_level=configuration.analysis.confidence_level,
                    bootstrap_seed=configuration.analysis.seed,
                    budget_censored_attempt_ids=censored_attempts,
                    replay=replay,
                )
            except UnsupportedTrialEvalPanelError as error:
                unavailable = TrialEvalCampaignUnavailableV1(
                    scorer=current_scorer(),
                    reason=str(error),
                    campaign_plan_sha256=combined.checksum,
                )
                (destination / "analysis.json").write_text(
                    unavailable.model_dump_json(indent=2) + "\n", encoding="utf-8"
                )
                public_result = unavailable
            else:
                write_review(analysis, destination / "review", evidence_kind="unclassified")
                (destination / "analysis.json").write_text(analysis.model_dump_json(indent=2) + "\n", encoding="utf-8")
                public_result = public_trialeval_campaign(analysis, split=configuration.panel.split)
                leaderboard.extend(trialeval_leaderboard_rows(analysis, censoring=censoring))
            _write_aggregate(destination, public_result, plan)
        if configuration.input_mode == "qualified" and leaderboard:
            _write_leaderboard(working, plan, tuple(leaderboard), censoring)
        if censoring is not None:
            (working / "censoring.json").write_text(censoring.model_dump_json(indent=2) + "\n", encoding="utf-8")
            for warning in censoring.warnings:
                print(f"WARNING: {warning}", flush=True)
        export = working / "trialeval" / "export"
        if export.exists():
            destination = output / "trialeval" / "export"
            destination.parent.mkdir(parents=True, exist_ok=True)
            export.rename(destination)
        for source in sorted(working.rglob("*")):
            if source.is_file():
                target = output / source.relative_to(working)
                target.parent.mkdir(parents=True, exist_ok=True)
                _write_once(target, source.read_bytes())
