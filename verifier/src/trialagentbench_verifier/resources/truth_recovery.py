r"""Compute TrialEval truth recovery and TrialDev share of oracle value from a campaign's recorded outputs.

The command reads the per-answer scores the verifier already writes (``trialeval_answers.csv``), the release's grading
key and, for TrialDev, each job's verified ``ranking.json`` with the sealed oracle values. It aggregates those scores
and scores nothing itself, so it sits under ``resources/``, outside the scorer identity: running it, or changing it,
leaves every retained grade valid.

Run on a finished campaign, from a download of the release::

    python -m trialagentbench_verifier.resources.truth_recovery --campaign results/comparison \\
      --evaluator-catalog benchmark/full/trialeval-evaluator.assets.json --evaluator-catalog-sha256 "$CATALOG_SHA256" \\
      --replay-bank replay-bank --replay-bank-sha256 "$REPLAY_BANK_SHA256" --fixed-analyses benchmark/fixed-analyses \\
      --oracle-values benchmark/oracle-values/trialdev_oracle_values.json \\
      --expected-oracle-values-sha256 "$ORACLE_SHA256" --output results/comparison/truth-recovery

The campaign plan gives the schedule: a scheduled task without a usable answer is a missing answer, and any other task
is outside the model's panel. The fixed-analysis tables give the process-aware analysis and the counts against always
robust. To pool campaigns, pass ``--answers`` and ``--schedule`` (a CSV with the columns ``model,task_id``) instead of
``--campaign``.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from trialagentbench_verifier.assets import materialize_asset_selection
from trialagentbench_verifier.campaign import _strata
from trialagentbench_verifier.campaign_contracts import CampaignPlanV1
from trialagentbench_verifier.development_campaign import (
    TrialDevOracleValuesV1,
    bind_trialdev_oracle_values,
    load_trialdev_oracle_values,
)
from trialagentbench_verifier.grading_key import read_trialeval_grading_key
from trialagentbench_verifier.replay_bank import (
    ReplayBankRecordV1,
    ReplayBankReportV1,
    ReplayBankV1,
    load_replay_bank,
)
from trialagentbench_verifier.replay_scoring import AnswerReplaysV1, _accuracy, _outcome_report
from trialagentbench_verifier.response import TrialDevRunRankingV1
from trialagentbench_verifier.trialeval import TrialEvalGradingKeyRecordV1

_Array = npt.NDArray[np.float64]

MAIN_CONTEXT = "C5"
"""The main selection's evidence context; rows are read at it and at the declared economics."""
REFERENCE = "policy:scripted_reference"
ROBUST = "policy:always_robust"
PIPELINE = "pipeline"
FIXED_ANALYSES = ("policy:follow_plan", ROBUST, REFERENCE, PIPELINE, "policy:always_naive")
"""The fixed analyses reported beside the models. They stay in the task-adjusted fit and out of the worst-answer pool."""
SERIOUS_MISS_SE = 0.5
"""A serious miss: a task on which an answer's error is at least this far beyond the process-aware analysis's."""
TIE_BAND_SE = 0.02
"""Paired win, tie and loss band: a task within this many standard errors of the comparator's error is a tie."""
MC_FACTOR = 2.0
"""A Monte Carlo win or loss lies beyond this many of the cell's paired Monte Carlo standard errors."""
MC_RESOLUTION_FLOOR = 1e-9
"""Smallest resolvable paired difference: below it two answers are numerically the same analysis."""
MATCH_TOLERANCE = 1e-9
"""Largest gap between a recomputed and a scored error (or answered share) that identifies a replay file."""
BANK_ROUTES = {
    "policy:follow_plan": "policy:follow_plan",
    ROBUST: ROBUST,
    REFERENCE: None,
    "policy:always_naive": "policy:always_naive",
}
"""Fixed analyses whose fresh-trial reports a full replay bank holds: answerer to route (None: the reference route)."""
FIXED_ANALYSIS_ROWS = "fixed_analysis_rows.csv"
FIXED_ANALYSIS_TRIALS = "fixed_analysis_trials.csv"
BOOTSTRAP = 2000
SEED = 20261105

MissingRule = Literal["worst_or_no_effect", "no_effect", "worst_answer"]
"""What a missing answer, or a fresh trial without a rerun, takes on a task. The first is primary."""
Stratum = tuple[str, str, int]
TaskDesign = Mapping[str, tuple[str, str, str]]
"""Task to its (cell, clinical question, base world) in the release's allocation (grading key)."""


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class TrialEvalCell(_Frozen):
    """One answerer's outcome on one task of its panel."""

    answerer: str
    task_id: str
    status: Literal["answered", "missing"]
    answered_share: float = Field(ge=0, le=1)
    interval_coverage: float | None = Field(default=None, ge=0, le=1)
    own_error: float | None = Field(default=None, ge=0, description="Mean error over the fresh trials it reran on.")
    replay_store: str | None = Field(
        default=None, description="The answer's own replay store, when its row names one (pooled campaigns)."
    )


class TruthRecoveryRowV1(_Frozen):
    """One answerer's truth recovery; errors are in standard errors, and ranks are among models only."""

    answerer: str
    kind: Literal["model", "fixed_analysis"]
    panel_tasks: int = Field(ge=1)
    missing_answer_tasks: int = Field(ge=0)
    error: float = Field(description="Task-adjusted mean error (two-way task + answerer fit).")
    error_low: float
    error_high: float
    rank: int | None
    rank_low: int | None
    rank_high: int | None
    serious_miss_share: float = Field(ge=0, le=1)
    serious_misses: int = Field(ge=0)
    interval_coverage: float | None = Field(ge=0, le=1)
    rerun_share: float = Field(ge=0, le=1)
    wins: int | None = Field(description="Tasks with lower error than always corrected by more than the tie band.")
    ties: int | None
    losses: int | None
    mc_wins: int | None = Field(
        default=None, description="Tasks with lower error than always corrected beyond twice the paired rerun noise."
    )
    mc_ties: int | None = None
    mc_losses: int | None = None


class Programme(_Frozen):
    """One TrialDev programme in a model's panel, with the fixed developers' values on the same programme."""

    task_id: str
    stratum: Stratum
    status: Literal["answered", "missing"]
    net_value_usd: float
    oracle_value_usd: float
    rule_based_value_usd: float


class OracleShareRowV1(_Frozen):
    """One model's share of oracle value on its own programmes, with the rule-based developer's on the same ones."""

    model: str
    programmes: int = Field(ge=1)
    oracle_share: float
    oracle_share_low: float | None
    oracle_share_high: float | None
    rule_based_share: float


def cell_from_row(answerer: str, row: Mapping[str, str]) -> TrialEvalCell:
    """Return one ``trialeval_answers.csv`` row's cell: its error over the fresh trials it reran on, and their share.

    Parameters
    ----------
    answerer
        Model or fixed-analysis key.
    row
        The answer's row.

    Returns
    -------
    TrialEvalCell
        A failed answer (no answer, not reproduced, another question) is a missing cell.

    Raises
    ------
    ValueError
        If the row was not scored on fresh trials, or reports reruns without an error.

    Examples
    --------
    >>> cell_from_row("m", {"task_id": "t", "scoring": "failure"}).status
    'missing'
    """
    task = row["task_id"]
    if row["scoring"] == "failure":
        return TrialEvalCell(answerer=answerer, task_id=task, status="missing", answered_share=0.0)
    if row["scoring"] != "replay":
        raise ValueError(f"{answerer} {task}: scoring {row['scoring']!r} has no fresh trials; finish with a bank")
    share = float(row["answered_share"])
    if share > 0 and not row["normalised_absolute_error"]:
        raise ValueError(f"{answerer} {task}: answered fresh trials without an error")
    return TrialEvalCell(
        answerer=answerer,
        task_id=task,
        status="answered",
        answered_share=share,
        interval_coverage=float(row["interval_coverage"]) if share > 0 else None,
        own_error=float(row["normalised_absolute_error"]) if share > 0 else None,
        replay_store=row.get("replay_store") or None,
    )


def _declared_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row["context"] == MAIN_CONTEXT and float(row["value_multiplier"]) == float(row["cost_multiplier"]) == 1
        ]


def read_cells(
    answers: Iterable[Path], schedule: Mapping[str, frozenset[str]], pipeline: Path | None = None
) -> dict[str, dict[str, TrialEvalCell]]:
    """Return every model's panel and each fixed analysis's cells from answer CSVs.

    Parameters
    ----------
    answers
        ``trialeval_answers.csv`` files. A model and task must appear once; a fixed analysis may repeat across files,
        and its repeats must agree.
    schedule
        Each model's scheduled tasks. A scheduled task without a row is a missing answer.
    pipeline
        Optional answer CSV of the test-then-correct pipeline, which the replay bank does not hold.

    Returns
    -------
    dict
        Answerer to task to cell.

    Raises
    ------
    ValueError
        If a row names an unscheduled model or task, a model's task repeats, or a fixed analysis disagrees with itself.
    """
    cells: dict[str, dict[str, TrialEvalCell]] = defaultdict(dict)
    for path in answers:
        for row in _declared_rows(path):
            model, task = row["model"], row["task_id"]
            if model in FIXED_ANALYSES:
                cell = cell_from_row(model, row)
                if cells[model].setdefault(task, cell) != cell:
                    raise ValueError(f"{model} {task}: the fixed analysis differs between answer files")
            elif model != "policy:prior_only":
                if task not in schedule.get(model, ()):
                    raise ValueError(f"{model} {task}: an answer outside the schedule")
                if task in cells[model]:
                    raise ValueError(f"{model} {task}: more than one answer; supply the selected one")
                cells[model][task] = cell_from_row(model, row)
    if pipeline is not None:
        for row in _declared_rows(pipeline):
            cells[PIPELINE][row["task_id"]] = cell_from_row(PIPELINE, row)
    for model, tasks in schedule.items():
        for task in tasks - set(cells[model]):
            cells[model][task] = TrialEvalCell(answerer=model, task_id=task, status="missing", answered_share=0.0)
    return dict(cells)


def _bank_reports(replays: Sequence[ReplayBankRecordV1], route: str | None) -> list[ReplayBankReportV1 | None]:
    return [replay.routes[route or replay.reference_route_id] for replay in replays]


def bank_cell(
    answerer: str, record: TrialEvalGradingKeyRecordV1, replays: Sequence[ReplayBankRecordV1], route: str | None
) -> TrialEvalCell:
    """Return a bank route's cell on one task.

    Parameters
    ----------
    answerer
        The answerer the route belongs to.
    record
        The task's grading-key record.
    replays
        The task's bank records.
    route
        The route; ``None`` is the first reference route.

    Returns
    -------
    TrialEvalCell
        The route's error, coverage and rerun share over the fresh trials.
    """
    log = record.target.binding.effect_measure == "rate_ratio"
    answered = [
        (replay.truth, tuple(math.exp(v) if log else v for v in (report.estimate, report.lower, report.upper)))
        for replay, report in zip(replays, _bank_reports(replays, route), strict=True)
        if report is not None
    ]
    error, coverage, _ = _accuracy(record.target, answered)  # type: ignore[arg-type]
    return TrialEvalCell(
        answerer=answerer,
        task_id=record.task_id,
        status="answered",
        answered_share=len(answered) / len(replays),
        interval_coverage=coverage,
        own_error=error,
    )


def reference_cell(record: TrialEvalGradingKeyRecordV1, replays: Sequence[ReplayBankRecordV1]) -> TrialEvalCell:
    """Return the process-aware analysis's cell on one task from a replay bank, public or full.

    Parameters
    ----------
    record
        The task's grading-key record.
    replays
        The task's bank records.

    Returns
    -------
    TrialEvalCell
        The first reference route's error, coverage and rerun share over the fresh trials.
    """
    return bank_cell(REFERENCE, record, replays, None)


def bank_trial_errors(
    record: TrialEvalGradingKeyRecordV1, replays: Sequence[ReplayBankRecordV1], route: str | None
) -> list[float | None]:
    """Return a bank route's error on each fresh trial, ``None`` where it reported nothing.

    Parameters
    ----------
    record
        The task's grading-key record.
    replays
        The task's bank records.
    route
        The route; ``None`` is the first reference route.

    Returns
    -------
    list
        Errors in replay order, in standard errors.
    """
    scale = record.target.reference_sampling_standard_deviation
    return [
        None if report is None else abs(report.estimate - replay.truth) / scale
        for report, replay in zip(_bank_reports(replays, route), replays, strict=True)
    ]


def read_fixed_trials(path: Path) -> dict[str, dict[str, list[float | None]]]:
    """Read the release's fixed-analysis errors on each fresh trial.

    Parameters
    ----------
    path
        ``fixed_analysis_trials.csv``.

    Returns
    -------
    dict
        Answerer to task to errors in replay order.

    Raises
    ------
    ValueError
        If a task's replay indexes are not consecutive from 1.
    """
    trials: dict[str, dict[str, list[float | None]]] = defaultdict(lambda: defaultdict(list))
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            errors = trials[row["model"]][row["task_id"]]
            if int(row["replay_index"]) != len(errors) + 1:
                raise ValueError(f"{row['model']} {row['task_id']}: replay {row['replay_index']} is out of order")
            errors.append(float(row["error"]) if row["error"] else None)
    return {answerer: dict(tasks) for answerer, tasks in trials.items()}


def no_effect_error(record: TrialEvalGradingKeyRecordV1) -> float:
    """Return the error of guessing no effect on a task: the distance of the truth from no effect, in standard errors.

    Parameters
    ----------
    record
        The task's grading-key record.

    Returns
    -------
    float
        ``|truth| / standard error`` on the comparison scale, which is the log scale for a rate ratio.

    Raises
    ------
    ValueError
        If a ratio truth is not positive.
    """
    target = record.target
    log = target.binding.effect_measure == "rate_ratio"
    if log and target.value <= 0:
        raise ValueError(f"{record.task_id}: a ratio truth must be positive")
    return abs(math.log(target.value) if log else target.value) / target.reference_sampling_standard_deviation


def worst_answer_errors(cells: Mapping[str, Mapping[str, TrialEvalCell]], tasks: Sequence[str]) -> dict[str, float]:
    """Return the worst mean error any model made on each task.

    A model's mean error on a task is over the fresh trials it reran on. The fixed analyses are left out.

    Parameters
    ----------
    cells
        Answerer to task to cell.
    tasks
        The tasks.

    Returns
    -------
    dict
        Task to its worst answer's error.

    Raises
    ------
    ValueError
        If no model answered a task.
    """
    worst: dict[str, float] = {}
    models = [own for answerer, own in cells.items() if answerer not in FIXED_ANALYSES]
    for task in tasks:
        errors = [cell.own_error for own in models if (cell := own.get(task)) and cell.own_error is not None]
        if not errors:
            raise ValueError(f"{task}: no model answer to take the worst error from")
        worst[task] = max(errors)
    return worst


def imputed_errors(
    cells: Mapping[str, Mapping[str, TrialEvalCell]],
    no_effect: Mapping[str, float],
    rule: MissingRule = "worst_or_no_effect",
) -> dict[str, float]:
    """Return what a missing answer takes on each task under a missing-answer rule.

    Parameters
    ----------
    cells
        Answerer to task to cell.
    no_effect
        Each task's ``no_effect_error``.
    rule
        ``worst_or_no_effect`` (primary) takes the larger of the two; the others are its sensitivities.

    Returns
    -------
    dict
        Task to imputed error.

    Examples
    --------
    >>> cells = {"a": {"t": TrialEvalCell(answerer="a", task_id="t", status="answered", answered_share=1.0,
    ...                                   own_error=1.5)}}
    >>> [imputed_errors(cells, {"t": 2.0}, rule)["t"] for rule in ("worst_or_no_effect", "no_effect", "worst_answer")]
    [2.0, 2.0, 1.5]
    """
    if rule == "no_effect":
        return dict(no_effect)
    worst = worst_answer_errors(cells, sorted(no_effect))
    if rule == "worst_answer":
        return worst
    return {task: max(worst[task], no_effect[task]) for task in no_effect}


def cell_error(cell: TrialEvalCell, imputed: float) -> float:
    """Return a cell's error over all its fresh trials, a trial without a rerun (or a missing answer) at ``imputed``.

    Parameters
    ----------
    cell
        The cell.
    imputed
        The task's imputed error.

    Returns
    -------
    float
        ``answered share x own error + (1 - answered share) x imputed``.

    Examples
    --------
    >>> cell_error(TrialEvalCell(answerer="m", task_id="t", status="answered", answered_share=0.5, own_error=0.4),
    ...            imputed=3.0)
    1.7
    """
    if cell.own_error is None:
        return imputed
    return cell.answered_share * cell.own_error + (1 - cell.answered_share) * imputed


def two_way_fit(values: _Array, iterations: int = 500) -> _Array:
    """Return each row's effect in a two-way additive (row + column) fit, on the average column's scale.

    Least squares on the observed cells by alternating means. A row observed on every column gets its plain mean;
    a row observed on part of the columns is adjusted for how hard its columns are.

    Parameters
    ----------
    values
        Rows (answerers) by columns (tasks); NaN where unobserved.
    iterations
        Maximum alternating passes.

    Returns
    -------
    numpy.ndarray
        Row effect plus the mean column effect: the row's predicted mean over all columns.

    Raises
    ------
    ValueError
        If a row or column is entirely unobserved.

    Examples
    --------
    >>> y = np.array([[1.0, 3.0], [2.0, np.nan]])
    >>> [round(float(v), 6) for v in two_way_fit(y)]
    [2.0, 3.0]
    """
    observed = ~np.isnan(values)
    if not observed.any(axis=0).all() or not observed.any(axis=1).all():
        raise ValueError("every row and column needs an observed cell")
    known = np.where(observed, values, 0.0)
    rows, columns = observed.sum(axis=1), observed.sum(axis=0)
    row_effect = np.zeros(values.shape[0])
    column_effect = known.sum(axis=0) / columns
    for _ in range(iterations):
        updated = ((known - column_effect) * observed).sum(axis=1) / rows
        column_effect = ((known - updated[:, None]) * observed).sum(axis=0) / columns
        if np.max(np.abs(updated - row_effect)) < 1e-12:
            row_effect = updated
            break
        row_effect = updated
    result: _Array = row_effect + column_effect.mean()
    return result


def world_resamples(
    tasks: Sequence[str], design: TaskDesign, observed: _Array, *, bootstrap: int, seed: int
) -> tuple[list[npt.NDArray[np.int64]], float]:
    """Return task-column resamples drawn by base world and their deviation scale.

    Base worlds are drawn with replacement within each of the verifier's strata (``campaign._strata``): cells of a
    clinical question that share base worlds move together, and cells with disjoint worlds are resampled separately.
    Each draw keeps every task of a drawn world. A resample that leaves an answerer with no task is redrawn.

    Parameters
    ----------
    tasks
        The task columns.
    design
        Each task's cell, clinical question and base world.
    observed
        The error matrix.
    bootstrap
        Resamples.
    seed
        Seed of the draws.

    Returns
    -------
    tuple
        The column draws, and sqrt(n / (n - 1)) for the n worlds per stratum.

    Raises
    ------
    ValueError
        If strata differ in size (one deviation scale would then misstate them) or hold fewer than two worlds.
    """
    cell_question = {design[task][0]: design[task][1] for task in tasks}
    by_cell_world = {(design[task][0], design[task][2]): 0.0 for task in tasks}
    strata = _strata(cell_question, by_cell_world)
    sizes = {len(worlds) for _, worlds in strata}
    if len(sizes) != 1 or min(sizes) < 2:
        raise ValueError(f"world strata must share one size of two or more worlds, not {sorted(sizes)}")
    index = {(design[task][0], design[task][2]): column for column, task in enumerate(tasks)}
    rng = np.random.default_rng(seed)
    columns: list[npt.NDArray[np.int64]] = []
    while len(columns) < bootstrap:
        draw = np.array(
            [
                index[cell, worlds[pick]]
                for cells, worlds in strata
                for pick in rng.integers(0, len(worlds), len(worlds))
                for cell in cells
                if (cell, worlds[pick]) in index
            ]
        )
        if not np.isnan(observed[:, draw]).all(axis=1).any():
            columns.append(draw)
    n = sizes.pop()
    return columns, math.sqrt(n / (n - 1))


def truth_recovery_table(
    cells: Mapping[str, Mapping[str, TrialEvalCell]],
    design: TaskDesign,
    imputed: Mapping[str, float],
    *,
    vectors: Mapping[str, Mapping[str, Sequence[float | None]]] | None = None,
    bootstrap: int = BOOTSTRAP,
    seed: int = SEED,
) -> tuple[TruthRecoveryRowV1, ...]:
    """Return every model's and fixed analysis's truth recovery.

    Parameters
    ----------
    cells
        Answerer to task to cell; must hold the process-aware analysis on every task.
    design
        Each task's cell, clinical question and base world.
    imputed
        Each task's imputed error (``imputed_errors``).
    vectors
        Fresh-trial errors (``trial_vectors``); with them each row also counts wins, ties and losses against always
        robust beyond twice the cell's paired rerun noise.
    bootstrap, seed
        Base-world bootstrap of the error's 95% interval and the rank range.

    Returns
    -------
    tuple of TruthRecoveryRowV1
        Models in name order, then the fixed analyses present. Wins, ties and losses are empty without always corrected.

    Raises
    ------
    ValueError
        If the process-aware analysis is absent, an answerer has a task outside its panel, or a cell's fresh-trial
        errors do not average to its error.
    """
    if REFERENCE not in cells:
        raise ValueError("the process-aware analysis is absent: supply its rows or the replay bank")
    tasks = sorted(cells[REFERENCE])
    models = sorted(answerer for answerer in cells if answerer not in FIXED_ANALYSES)
    answerers = [*models, *(answerer for answerer in FIXED_ANALYSES if answerer in cells)]
    if any(set(cells[answerer]) - set(tasks) for answerer in answerers):
        raise ValueError("an answerer has a task outside the process-aware analysis's")
    values = np.array(
        [[cell_error(cells[a][t], imputed[t]) if t in cells[a] else np.nan for t in tasks] for a in answerers]
    )
    noise = np.full(values.shape, np.nan)
    robust_trials = (vectors or {}).get(ROBUST, {})
    for i, answerer in enumerate(answerers):
        for j, task in enumerate(tasks):
            own_trials = (vectors or {}).get(answerer, {}).get(task)
            if own_trials is None or task not in robust_trials or answerer == ROBUST:
                continue
            trials = [imputed[task] if error is None else error for error in own_trials]
            if abs(statistics.fmean(trials) - values[i, j]) > MATCH_TOLERANCE:
                raise ValueError(f"{answerer} {task}: fresh-trial errors do not average to the cell's error")
            other = [imputed[task] if error is None else error for error in robust_trials[task]]
            noise[i, j] = MC_FACTOR * paired_noise(trials, other)
    columns, scale = world_resamples(tasks, design, values, bootstrap=bootstrap, seed=seed)
    estimate = two_way_fit(values)
    draws = estimate + scale * (np.array([two_way_fit(values[:, draw]) for draw in columns]) - estimate)
    point_rank = np.argsort(np.argsort(estimate[: len(models)])) + 1
    ranks = np.argsort(np.argsort(draws[:, : len(models)], axis=1), axis=1) + 1
    reference = values[answerers.index(REFERENCE)]
    robust = values[answerers.index(ROBUST)] if ROBUST in cells else None
    rows = []
    for i, answerer in enumerate(answerers):
        row, own = values[i], cells[answerer]
        observed = ~np.isnan(row)
        excess = row[observed] - reference[observed]
        answered = [cell for cell in own.values() if cell.interval_coverage is not None]
        weight = sum(cell.answered_share for cell in answered)
        low, high = np.percentile(draws[:, i], [2.5, 97.5])
        is_model = i < len(models)
        difference = bound = None
        if robust is not None and answerer != ROBUST:
            shared = observed & ~np.isnan(robust)
            difference = row[shared] - robust[shared]
            if not np.isnan(noise[i][shared]).any():
                bound = np.maximum(noise[i][shared], MC_RESOLUTION_FLOOR)
        rows.append(
            TruthRecoveryRowV1(
                answerer=answerer,
                kind="model" if is_model else "fixed_analysis",
                panel_tasks=int(observed.sum()),
                missing_answer_tasks=sum(cell.status == "missing" for cell in own.values()),
                error=float(estimate[i]),
                error_low=float(low),
                error_high=float(high),
                rank=int(point_rank[i]) if is_model else None,
                rank_low=int(np.percentile(ranks[:, i], 2.5)) if is_model else None,
                rank_high=int(np.percentile(ranks[:, i], 97.5)) if is_model else None,
                serious_miss_share=float(np.mean(excess >= SERIOUS_MISS_SE)),
                serious_misses=int(np.sum(excess >= SERIOUS_MISS_SE)),
                interval_coverage=(
                    sum(cell.answered_share * (cell.interval_coverage or 0.0) for cell in answered) / weight
                    if weight
                    else None
                ),
                rerun_share=float(np.mean([cell.answered_share for cell in own.values()])),
                wins=None if difference is None else int((difference < -TIE_BAND_SE).sum()),
                ties=None if difference is None else int((np.abs(difference) <= TIE_BAND_SE).sum()),
                losses=None if difference is None else int((difference > TIE_BAND_SE).sum()),
                mc_wins=None if difference is None or bound is None else int((difference < -bound).sum()),
                mc_ties=None if difference is None or bound is None else int((np.abs(difference) <= bound).sum()),
                mc_losses=None if difference is None or bound is None else int((difference > bound).sum()),
            )
        )
    return tuple(rows)


def trial_vectors(
    cells: Mapping[str, Mapping[str, TrialEvalCell]],
    key: Mapping[str, TrialEvalGradingKeyRecordV1],
    bank: ReplayBankV1,
    bank_sha256: str,
    stores: Sequence[Path],
    fixed_trials: Mapping[str, Mapping[str, list[float | None]]] | None = None,
) -> dict[str, dict[str, list[float | None]]]:
    """Return every answer's own error on each fresh trial, ``None`` where it did not rerun.

    A fixed analysis's errors come from the release's table, or else from a full bank's routes. A model's come from the replay file, in the store its
    row names or else among the stores, whose recomputed error and answered share reproduce its scored row.

    Parameters
    ----------
    cells
        Answerer to task to cell.
    key
        Grading-key records by task.
    bank
        The replay bank.
    bank_sha256
        SHA-256 of the bank's manifest, which keys the stores.
    stores
        Campaign replay stores (``trialeval-replays``).
    fixed_trials
        The release's fixed-analysis errors (``read_fixed_trials``).

    Returns
    -------
    dict
        Answerer to task to errors in replay order, for models and the bank's fixed analyses.

    Raises
    ------
    ValueError
        If neither the table nor the bank holds the fixed analyses, or no replay file, or files that disagree, reproduce
        a scored row.
    """
    if fixed_trials is None and bank.manifest.variant != "full":
        raise ValueError("rerun noise against always corrected needs the fixed-analysis table or the full replay bank")
    vectors: dict[str, dict[str, list[float | None]]] = defaultdict(dict)
    stored: dict[tuple[str, tuple[Path, ...]], list[list[float | None]]] = {}
    for answerer, own in cells.items():
        for task, cell in own.items():
            replays = bank.records[task]
            if answerer in BANK_ROUTES:
                vectors[answerer][task] = (
                    list(fixed_trials[answerer][task])
                    if fixed_trials is not None
                    else bank_trial_errors(key[task], replays, BANK_ROUTES[answerer])
                )
            elif answerer != PIPELINE:
                where = (Path(cell.replay_store),) if cell.replay_store else tuple(stores)
                if (task, where) not in stored:
                    stored[task, where] = _stored_errors(key[task], replays, bank_sha256, where)
                vectors[answerer][task] = _model_vector(cell, stored[task, where], len(replays))
    return dict(vectors)


def _stored_errors(
    record: TrialEvalGradingKeyRecordV1,
    replays: Sequence[ReplayBankRecordV1],
    bank_sha256: str,
    stores: Sequence[Path],
) -> list[list[float | None]]:
    scale = record.target.reference_sampling_standard_deviation
    found = []
    for store in stores:
        for path in sorted((store / bank_sha256 / record.task_id).glob("*.json")):
            outcomes = AnswerReplaysV1.model_validate_json(path.read_bytes()).replays
            reports = (_outcome_report(outcome, record.target) for outcome in outcomes)
            found.append(
                [
                    None if report is None else abs(report.estimate - replay.truth) / scale
                    for report, replay in zip(reports, replays, strict=True)
                ]
            )
    return found


def _model_vector(cell: TrialEvalCell, stored: Sequence[list[float | None]], trials: int) -> list[float | None]:
    if cell.own_error is None:
        return [None] * trials
    found = []
    for errors in stored:
        answered = [error for error in errors if error is not None]
        if (
            answered
            and abs(len(answered) / trials - cell.answered_share) <= MATCH_TOLERANCE
            and abs(statistics.fmean(answered) - cell.own_error) <= MATCH_TOLERANCE
        ):
            found.append(errors)
    if not found or any(
        [e is None for e in vector] != [e is None for e in found[0]]
        or any(abs((a or 0.0) - (b or 0.0)) > MATCH_TOLERANCE for a, b in zip(vector, found[0], strict=True))
        for vector in found
    ):
        raise ValueError(f"{cell.answerer} {cell.task_id}: {len(found)} replay files reproduce the scored row")
    return found[0]


def paired_noise(own: Sequence[float], other: Sequence[float]) -> float:
    """Return the Monte Carlo standard error of a paired difference of mean errors over the same fresh trials.

    Parameters
    ----------
    own, other
        Fresh-trial errors in replay order.

    Returns
    -------
    float
        The standard deviation of the differences over the square root of their number.

    Examples
    --------
    >>> round(paired_noise([1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]), 6)
    0.288675
    """
    differences = [a - b for a, b in zip(own, other, strict=True)]
    return statistics.stdev(differences) / math.sqrt(len(differences))


def read_grading_records(root: Path) -> dict[str, TrialEvalGradingKeyRecordV1]:
    """Return an extracted evaluator release's verified grading-key records by task.

    Parameters
    ----------
    root
        Extracted TrialEval evaluator release containing ``grading_key/``.

    Returns
    -------
    dict
        Task to record.
    """
    _, records = read_trialeval_grading_key(root)
    return {record.task_id: record for record in records}


def campaign_truth_recovery(
    answers: Iterable[Path],
    schedule: Mapping[str, frozenset[str]],
    key: Mapping[str, TrialEvalGradingKeyRecordV1],
    *,
    pipeline: Path | None = None,
    replay_bank: tuple[Path, str] | None = None,
    stores: Sequence[Path] = (),
    fixed_analyses: Path | None = None,
    rule: MissingRule = "worst_or_no_effect",
) -> tuple[TruthRecoveryRowV1, ...]:
    """Return truth recovery from a campaign's answer files.

    Parameters
    ----------
    answers
        ``trialeval_answers.csv`` files.
    schedule
        Each model's scheduled tasks.
    key
        Grading-key records by task.
    pipeline
        Optional answer CSV of the test-then-correct pipeline.
    replay_bank
        Bank directory and manifest SHA-256, the process-aware analysis's source when the rows carry none.
    stores
        Campaign replay stores. With the fixed-analysis tables or a full bank they add the counts beyond twice the
        paired rerun noise.
    fixed_analyses
        The release's ``fixed-analyses`` directory.
    rule
        The missing-answer rule.

    Returns
    -------
    tuple of TruthRecoveryRowV1
        Models in name order, then the fixed analyses present.

    Raises
    ------
    ValueError
        If the process-aware analysis has no source.
    """
    tables = () if fixed_analyses is None else (fixed_analyses / FIXED_ANALYSIS_ROWS,)
    cells = read_cells((*answers, *tables), schedule, pipeline)
    if fixed_analyses is not None:
        scheduled_tasks = set().union(*schedule.values())
        cells = {
            answerer: {
                task: cell for task, cell in own.items() if answerer not in BANK_ROUTES or task in scheduled_tasks
            }
            for answerer, own in cells.items()
        }
    bank = None if replay_bank is None else load_replay_bank(replay_bank[0], expected_manifest_sha256=replay_bank[1])
    if REFERENCE not in cells and bank is not None:
        scheduled = sorted(set().union(*schedule.values()))
        cells[REFERENCE] = {task: reference_cell(key[task], bank.records[task]) for task in scheduled}
    if REFERENCE not in cells:
        raise ValueError("the process-aware analysis is absent: supply its rows or a replay bank")
    tasks = sorted(cells[REFERENCE])
    no_effect = {task: no_effect_error(key[task]) for task in tasks}
    design = {task: (key[task].cell_ref, key[task].series_ref, key[task].base_world_id) for task in tasks}
    vectors = None
    if stores:
        if bank is None or replay_bank is None:
            raise ValueError("replay stores need the full replay bank they were run against")
        fixed_trials = None if fixed_analyses is None else read_fixed_trials(fixed_analyses / FIXED_ANALYSIS_TRIALS)
        vectors = trial_vectors(cells, key, bank, replay_bank[1], stores, fixed_trials)
    return truth_recovery_table(cells, design, imputed_errors(cells, no_effect, rule), vectors=vectors)


def read_programmes(rankings: Iterable[Path], oracle: TrialDevOracleValuesV1) -> dict[str, list[Programme]]:
    """Return each model's main-selection TrialDev programmes from its verified rankings.

    An attempt that ended without an outcome of the model's own stops at once, at zero value.

    Parameters
    ----------
    rankings
        Each job's ``ranking.json``.
    oracle
        The sealed oracle and rule-based developer values.

    Returns
    -------
    dict
        Model to programmes.

    Raises
    ------
    ValueError
        If a ranking belongs to another release or scorer, a model's programme repeats, a programme lacks its oracle
        value, or a scored programme lacks a value.
    """
    rule_based = {row.task_id: row.reference_value_usd for row in oracle.values}
    panels: dict[str, dict[str, Programme]] = defaultdict(dict)
    for path in rankings:
        ranking = TrialDevRunRankingV1.model_validate_json(path.read_bytes())
        main = tuple(
            row
            for row in ranking.items
            if row.authored_programme and row.stream == "portfolio_management" and row.data_view == "raw"
        )
        ranking = bind_trialdev_oracle_values(ranking.model_copy(update={"items": main}), oracle)
        for row in ranking.items:
            missing = row.execution_status in {"not_started", "incomplete_run"}
            if row.task_id in panels[ranking.model_id]:
                raise ValueError(f"{ranking.model_id} {row.task_id}: more than one outcome; supply the selected one")
            if row.oracle_development_value_usd is None or (row.net_development_value_usd is None and not missing):
                raise ValueError(f"{ranking.model_id} {row.task_id}: the programme has no value")
            panels[ranking.model_id][row.task_id] = Programme(
                task_id=row.task_id,
                stratum=(row.clinical_profile_id, row.scenario_profile_id, row.asset_count),
                status="missing" if missing else "answered",
                net_value_usd=0.0 if row.net_development_value_usd is None else row.net_development_value_usd,
                oracle_value_usd=row.oracle_development_value_usd,
                rule_based_value_usd=rule_based[row.task_id],
            )
    return {model: list(panel.values()) for model, panel in panels.items()}


def stratum_weights(programmes: Sequence[Programme]) -> list[float]:
    """Return equal-stratum weights: each stratum counts equally, each programme equally within it.

    Parameters
    ----------
    programmes
        A model's panel.

    Returns
    -------
    list of float
        One weight per programme.
    """
    sizes: dict[Stratum, int] = defaultdict(int)
    for programme in programmes:
        sizes[programme.stratum] += 1
    return [1 / (len(sizes) * sizes[programme.stratum]) for programme in programmes]


def share_interval(
    programmes: Sequence[Programme], *, bootstrap: int = BOOTSTRAP, seed: int = SEED
) -> tuple[float | None, float | None]:
    """Return a percentile 95% interval of the share of oracle value.

    Worlds are resampled within each stratum, and each stratum's bootstrap deviation of its mean value (the model's
    and the oracle's) is scaled by sqrt(n / (n - 1)), n its worlds, before the ratio of equal-stratum means is taken.

    Parameters
    ----------
    programmes
        A model's panel.
    bootstrap, seed
        Resamples and seed.

    Returns
    -------
    tuple
        The interval, or ``(None, None)`` when a stratum has fewer than two worlds.

    Raises
    ------
    ValueError
        If a resample's oracle value is not positive.
    """
    groups: dict[Stratum, list[Programme]] = defaultdict(list)
    for programme in programmes:
        groups[programme.stratum].append(programme)
    if min(map(len, groups.values())) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    agent_total = np.zeros(bootstrap)
    oracle_total = np.zeros(bootstrap)
    for group in groups.values():
        n = len(group)
        agent = np.array([p.net_value_usd for p in group])
        oracle = np.array([p.oracle_value_usd for p in group])
        picks = rng.integers(0, n, size=(bootstrap, n))
        scale = math.sqrt(n / (n - 1))
        agent_total += agent.mean() + scale * (agent[picks].mean(axis=1) - agent.mean())
        oracle_total += oracle.mean() + scale * (oracle[picks].mean(axis=1) - oracle.mean())
    if (oracle_total <= 0).any():
        raise ValueError("a resample's oracle value is not positive")
    low, high = np.percentile(agent_total / oracle_total, [2.5, 97.5])
    return float(low), float(high)


def oracle_share_table(panels: Mapping[str, Sequence[Programme]]) -> tuple[OracleShareRowV1, ...]:
    """Return every model's share of oracle value on the programmes it ran, on which it is ranked.

    Parameters
    ----------
    panels
        Model to programmes.

    Returns
    -------
    tuple of OracleShareRowV1
        Models in name order; a panel whose oracle value is not positive has no share and no row.
    """
    rows = []
    for model, programmes in sorted(panels.items()):
        weights = stratum_weights(programmes)
        oracle = sum(w * p.oracle_value_usd for w, p in zip(weights, programmes, strict=True))
        if oracle <= 0:  # nothing to capture, as on a panel of a programme or two: not a measurement
            continue
        low, high = share_interval(programmes)
        rows.append(
            OracleShareRowV1(
                model=model,
                programmes=len(programmes),
                oracle_share=sum(w * p.net_value_usd for w, p in zip(weights, programmes, strict=True)) / oracle,
                oracle_share_low=low,
                oracle_share_high=high,
                rule_based_share=sum(w * p.rule_based_value_usd for w, p in zip(weights, programmes, strict=True))
                / oracle,
            )
        )
    return tuple(rows)


def _write(path: Path, rows: Sequence[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(type(rows[0]).model_fields))
        writer.writeheader()
        writer.writerows(row.model_dump() for row in rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute TrialEval truth recovery and TrialDev share of oracle value."
    )
    parser.add_argument("--campaign", type=Path, help="Finished campaign directory.")
    parser.add_argument("--answers", type=Path, nargs="+", help="trialeval_answers.csv files, to pool campaigns.")
    parser.add_argument("--schedule", type=Path, help="CSV of every scheduled model,task_id, to pool campaigns.")
    parser.add_argument("--evaluator-catalog", type=Path, help="Evaluator catalog beside its shards.")
    parser.add_argument("--evaluator-catalog-sha256", help="SHA-256 of the evaluator catalog.")
    parser.add_argument("--evaluator-release", type=Path, help="Extracted evaluator release, instead of the catalog.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for the output CSVs.")
    parser.add_argument("--pipeline-answers", type=Path, help="Answer CSV of the test-then-correct pipeline.")
    parser.add_argument("--replay-bank", type=Path, help="Replay bank, when the rows carry no process-aware analysis.")
    parser.add_argument("--replay-bank-sha256", help="Published SHA-256 of the bank's manifest.json.")
    parser.add_argument(
        "--replays", type=Path, nargs="*", default=(), help="Replay stores, for the counts beyond twice rerun noise."
    )
    parser.add_argument(
        "--missing-rule", choices=("worst_or_no_effect", "no_effect", "worst_answer"), default="worst_or_no_effect"
    )
    parser.add_argument("--fixed-analyses", type=Path, help="The release's fixed-analyses directory.")
    parser.add_argument("--trialdev-rankings", type=Path, nargs="*", default=(), help="Each job's ranking.json.")
    parser.add_argument("--oracle-values", type=Path, help="The release's TrialDev oracle values.")
    parser.add_argument("--expected-oracle-values-sha256", help="Retained SHA-256 of the oracle values.")
    return parser


def _campaign_inputs(args: argparse.Namespace) -> tuple[list[Path], dict[str, set[str]], list[Path]]:
    if (args.campaign is None) == (args.answers is None or args.schedule is None):
        raise ValueError("pass --campaign, or both --answers and --schedule")
    if args.campaign is None:
        schedule: dict[str, set[str]] = defaultdict(set)
        with args.schedule.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                schedule[row["model"]].add(row["task_id"])
        return args.answers, schedule, list(args.replays)
    plan = CampaignPlanV1.model_validate_json((args.campaign / "campaign.json").read_bytes())
    tasks = set(plan.configuration.panel.trialeval_tasks)
    if not tasks:
        raise ValueError("the campaign's panel names no TrialEval tasks; pass --answers and --schedule")
    answers = [args.campaign / "comparisons" / "trialeval" / "trialeval_answers.csv"]
    stores = list(args.replays)
    replays = args.campaign / "trialeval-replays"
    if not stores and args.replay_bank is not None and args.fixed_analyses is not None and replays.is_dir():
        stores = [replays]
    return answers, {model.model: set(tasks) for model in plan.configuration.models}, stores


def _grading_records(args: argparse.Namespace, task: str) -> dict[str, TrialEvalGradingKeyRecordV1]:
    if args.evaluator_release is not None:
        return read_grading_records(args.evaluator_release)
    if args.evaluator_catalog is None or args.evaluator_catalog_sha256 is None:
        raise ValueError("pass --evaluator-catalog with its SHA-256, or --evaluator-release")
    with tempfile.TemporaryDirectory() as directory:
        materialize_asset_selection(
            args.evaluator_catalog,
            expected_sha256=args.evaluator_catalog_sha256,
            output=Path(directory) / "evaluator",
            task_ids=(task,),
        )
        return read_grading_records(Path(directory) / "evaluator")


def main(argv: Sequence[str] | None = None) -> None:
    """Write ``truth_recovery.csv`` and, with TrialDev inputs, ``oracle_share.csv``.

    Parameters
    ----------
    argv
        Command-line arguments; ``sys.argv`` by default.

    Raises
    ------
    ValueError
        If the inputs are incomplete, or an input breaks a rule of the functions above.
    """
    args = _parser().parse_args(argv)
    answers, schedule, stores = _campaign_inputs(args)
    key = _grading_records(args, next((task for tasks in schedule.values() for task in tasks), ""))
    table = campaign_truth_recovery(
        answers,
        {model: frozenset(task for task in tasks if task in key) for model, tasks in schedule.items()},
        key,
        pipeline=args.pipeline_answers,
        replay_bank=None if args.replay_bank is None else (args.replay_bank, args.replay_bank_sha256),
        stores=stores,
        fixed_analyses=args.fixed_analyses,
        rule=args.missing_rule,
    )
    _write(args.output / "truth_recovery.csv", table)
    rankings = args.trialdev_rankings
    if args.campaign is not None and args.oracle_values is not None and not rankings:
        rankings = sorted(args.campaign.glob("results/*/grade/ranking.json"))
    if rankings:
        oracle = load_trialdev_oracle_values(args.oracle_values, expected_sha256=args.expected_oracle_values_sha256)
        _write(args.output / "oracle_share.csv", oracle_share_table(read_programmes(rankings, oracle)))


if __name__ == "__main__":
    main()
