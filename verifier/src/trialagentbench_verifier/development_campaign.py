"""Paired finite-world TrialDev comparisons from independently verified rankings.

Worlds, rather than views, repetitions or patients, are the resampling units.
The two metrics retain their own scales; this module creates no overall score.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from html import escape
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Literal, Self
from zipfile import ZipFile

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.stats import norm

from trialagentbench_verifier.archive import opened_release
from trialagentbench_verifier.assets import read_asset_catalog
from trialagentbench_verifier.campaign import holm_adjusted_p_values_v1
from trialagentbench_verifier.campaign_contracts import (
    BudgetCensoringV1,
    CampaignAnalysisV1,
    CampaignPlanV1,
    Digest,
    LeaderboardRowV1,
)
from trialagentbench_verifier.development import _Task, ranking_means, verify_trialdev_ranking
from trialagentbench_verifier.grading_key import (
    TRIALDEV_WORLD_STRATA,
    TrialDevWorldStrataV1,
    read_trialdev_world_strata,
)
from trialagentbench_verifier.io import release_archive, sha256_file
from trialagentbench_verifier.numerical_scoring import bounded_recovery_summary
from trialagentbench_verifier.response import TrialDevRankingMeanV1, TrialDevRunRankingV1
from trialagentbench_verifier.scorer import CurrentScorer, current_scorer, require_same_scorer

Metric = Literal["observational_loss", "net_development_value_usd"]
Stratum = tuple[str, str, int]

MDD_MULTIPLIER = float(norm.ppf(1 - 0.05 / 2) + norm.ppf(0.80))
"""Multiplier for the unadjusted normal-approximation MDD at alpha 0.05 and power 0.80."""


class TrialDevOracleValueV1(BaseModel):
    """Construction policy values on one released world and task budget.

    Parameters
    ----------
    task_id, world_id
        Opaque released task and world identities.
    net_value_usd, reference_value_usd
        Released-seed oracle and scripted-reference net programme values in dollars.

    Raises
    ------
    pydantic.ValidationError
        If identities are blank, values are nonfinite or unknown fields are supplied.

    Examples
    --------
    >>> value = TrialDevOracleValueV1(task_id="task", world_id="world", net_value_usd=10, reference_value_usd=5)
    >>> value.net_value_usd
    10.0
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    task_id: str = Field(min_length=1)
    world_id: str = Field(min_length=1)
    net_value_usd: float
    reference_value_usd: float


class TrialDevOracleValuesV1(BaseModel):
    """Pinned construction-policy outputs bound to the unchanged release manifest.

    Parameters
    ----------
    release_id, release_manifest_sha256
        Identity of the release whose tasks the policy values describe.
    policy_source_sha256, source_artifacts
        Construction policy source digest and retained policy-output/fingerprint digests.
    values
        One oracle/reference value pair per unique authored programme task.

    Raises
    ------
    pydantic.ValidationError
        If digests or values are invalid, tasks repeat or unknown fields are supplied.

    Examples
    --------
    Load the sealed file through ``load_trialdev_oracle_values`` with its externally retained SHA-256.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_id: Literal["trialagentbench.trialdev_oracle_values/v1"] = "trialagentbench.trialdev_oracle_values/v1"
    release_id: str
    release_manifest_sha256: Digest
    policy_source_sha256: Digest
    source_artifacts: dict[str, Digest] = Field(min_length=1)
    values: tuple[TrialDevOracleValueV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_tasks(self) -> Self:
        if len({row.task_id for row in self.values}) != len(self.values):
            raise ValueError("oracle values repeat a released task")
        return self


def load_trialdev_oracle_values(path: Path, *, expected_sha256: str) -> TrialDevOracleValuesV1:
    """Read construction oracle values after checking their externally pinned digest.

    Parameters
    ----------
    path
        The release-bound construction-policy value file.
    expected_sha256
        File SHA-256 retained independently in the scoring inputs.

    Returns
    -------
    TrialDevOracleValuesV1
        Validated release identity, source provenance and per-task policy values.

    Raises
    ------
    ValueError
        If file bytes differ from the pinned digest or violate the schema.
    OSError
        If the input cannot be read.

    Examples
    --------
    Use ``load_trialdev_oracle_values(path, expected_sha256=retained_digest)`` before binding rankings.
    """
    if sha256_file(path) != expected_sha256:
        raise ValueError("TrialDev oracle values differ from their pinned digest")
    return TrialDevOracleValuesV1.model_validate_json(path.read_bytes())


def bind_trialdev_oracle_values(ranking: TrialDevRunRankingV1, oracle: TrialDevOracleValuesV1) -> TrialDevRunRankingV1:
    """Bind authenticated oracle denominators to every scheduled authored programme.

    Parameters
    ----------
    ranking
        Independently verified task outcomes for one frozen invocation.
    oracle
        Construction values loaded with the externally pinned file digest.

    Returns
    -------
    TrialDevRunRankingV1
        A derived ranking with oracle values attached; the original ranking is unchanged.

    Raises
    ------
    ValueError
        If release identities differ or a scheduled programme lacks its matching world/task value.

    Examples
    --------
    Pass ``bind_trialdev_oracle_values(ranking, oracle)`` to the world-level leaderboard aggregator.
    """
    if (ranking.release_id, ranking.release_manifest_sha256) != (oracle.release_id, oracle.release_manifest_sha256):
        raise ValueError("Oracle values belong to a different release")
    values = {row.task_id: row for row in oracle.values}
    items = []
    for row in ranking.items:
        if not row.authored_programme:
            items.append(row)
            continue
        if row.task_id not in values or values[row.task_id].world_id != row.world_id:
            raise ValueError("Oracle values lack a scheduled programme or identify a different world")
        items.append(row.model_copy(update={"oracle_development_value_usd": values[row.task_id].net_value_usd}))
    return ranking.model_copy(update={"items": tuple(items)})


def _capture_bootstrap(
    groups: dict[Stratum, list[float]], denominators: dict[Stratum, list[float]], policy: CampaignAnalysisV1
) -> tuple[float | None, float | None, float | None, float | None, str | None]:
    """Bootstrap the ratio of equally weighted stratum means using the same world draws in both sums.

    Nonpositive oracle values contribute zero to the denominator; their programme outcomes remain in
    the numerator. A zero aggregate denominator withholds the percentage. Zero bootstrap denominators
    withhold uncertainty, without dropping draws or assigning arbitrary percentages.
    """
    numerator = mean(mean(values) for values in groups.values())
    denominator = mean(mean(values) for values in denominators.values())
    if denominator <= 0:
        return None, None, None, None, "zero_oracle_value"
    estimate = 100 * numerator / denominator
    rng = np.random.default_rng(policy.seed)
    top = np.zeros(policy.bootstrap_repetitions)
    bottom = np.zeros(policy.bootstrap_repetitions)
    for stratum in sorted(groups):
        values, oracles = np.asarray(groups[stratum]), np.asarray(denominators[stratum])
        if values.shape != oracles.shape:
            raise ValueError("Oracle values lost world pairing")
        indices = rng.integers(0, len(values), size=(policy.bootstrap_repetitions, len(values)))
        top += values[indices].mean(axis=1) / len(groups)
        bottom += oracles[indices].mean(axis=1) / len(groups)
    if np.any(bottom <= 0):
        return estimate, None, None, None, "zero_oracle_bootstrap_denominator"
    draws = 100 * top / bottom
    tail = (1 - policy.confidence_level) / 2
    low, high = (float(value) for value in np.quantile(draws, [tail, 1 - tail]))
    return estimate, low, high, float(np.std(draws, ddof=1)), None


def _world_bootstrap(
    groups: dict[Stratum, list[float]], policy: CampaignAnalysisV1
) -> tuple[NDArray[np.float64], float, float, float]:
    """Resample worlds within equally weighted strata, giving each stratum mean its unbiased variance.

    Resampling n worlds gives a stratum mean the variance (n - 1) / n * s**2 / n,
    half the unbiased s**2 / n at the two worlds a withheld stratum may hold, so
    percentile intervals would under-cover. Each draw's deviation from the
    stratum's observed mean is therefore scaled by sqrt(n / (n - 1)) before
    strata are combined. A one-world stratum, which callers never bootstrap
    because inference needs ``policy.minimum_worlds_per_stratum`` worlds, keeps
    its observed value with no spread, as resampling a single world gives.

    Returns
    -------
    tuple
        Bootstrap draws of the equal-stratum mean, the percentile interval at
        the policy's confidence level, and the draws' standard deviation, the
        bootstrap standard error.
    """
    rng = np.random.default_rng(policy.seed)
    draws = np.zeros(policy.bootstrap_repetitions)
    for stratum in sorted(groups):
        vector = np.asarray(groups[stratum], dtype=float)
        size = len(vector)
        resampled = vector[rng.integers(0, size, size=(policy.bootstrap_repetitions, size))].mean(axis=1)
        if size >= 2:
            observed = vector.mean()
            resampled = observed + np.sqrt(size / (size - 1)) * (resampled - observed)
        draws += resampled / len(groups)
    tail = (1 - policy.confidence_level) / 2
    low, high = (float(value) for value in np.quantile(draws, [tail, 1 - tail]))
    return draws, low, high, float(np.std(draws, ddof=1))


class TrialDevComparisonV1(BaseModel):
    """One prespecified paired difference, with explicit unsupported inference.

    The common-world estimate is a descriptive sensitivity on worlds where both
    models have the metric for every linked task and repetition. It weights
    retained strata equally; its counts disclose selection and it never supplies
    a panel ranking. Evaluator faults withhold both comparisons.

    The interval, p-value, standard error and minimal detectable difference
    come from one paired world bootstrap whose stratum means carry their
    unbiased small-sample variance; all are withheld with the inference.
    """

    scorer: CurrentScorer

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    metric: Metric
    stream: Literal["evidence_analysis", "portfolio_management"]
    data_view: Literal["raw", "analysis_ready"]
    model_left: str
    model_right: str
    worlds: int
    strata: int
    estimate: float | None
    confidence_low: float | None
    confidence_high: float | None
    p_value: float | None
    adjusted_p_value: float | None = None
    standard_error: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Paired world-bootstrap standard error of the left-minus-right estimate, in the metric's unit: "
            "numerical loss, or US dollars for net development value."
        ),
    )
    minimal_detectable_difference: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Smallest true paired difference detected at two-sided alpha 0.05 with 80% power, "
            "(z_0.975 + z_0.80) times the standard error, in the metric's unit: "
            "numerical loss, or US dollars for net development value."
        ),
    )
    unavailable_reason: str | None
    common_world_estimate: float | None = None
    common_worlds: int = 0
    common_strata: int = 0

    @model_validator(mode="after")
    def validate_estimate(self) -> Self:
        """Keep paired uncertainty together and withhold it with the estimate."""
        if (self.standard_error is None) != (self.minimal_detectable_difference is None):
            raise ValueError("Paired uncertainty requires both a standard error and minimal detectable difference")
        if self.estimate is None and self.standard_error is not None:
            raise ValueError("A withheld estimate cannot carry a standard error or minimal detectable difference")
        return self


def compare_trialdev_rankings(
    rankings: tuple[TrialDevRunRankingV1, ...],
    *,
    required_strata: tuple[Stratum, ...],
    policy: CampaignAnalysisV1,
) -> tuple[TrialDevComparisonV1, ...]:
    """Compare authenticated complete common panels, keeping streams and data views separate.

    Callers must independently verify the rankings and bind their complete job
    census to the prospective campaign. This numerical owner does not authenticate
    files. Missing tasks and changed panels are errors, not zero-valued results.
    Evaluator faults withhold the affected comparison, including its estimate.
    Authored development values remain in dollars. An unfinished programme
    keeps its world, valued at the costs it incurred as a stop at that point,
    so abandoning a failing programme cannot raise a score.
    """
    require_same_scorer(ranking.scorer for ranking in rankings)
    if not rankings or not required_strata or len(set(required_strata)) != len(required_strata):
        raise ValueError("Rankings and unique required strata must be nonempty")
    by_model: dict[str, list[TrialDevRunRankingV1]] = defaultdict(list)
    panel = None
    identity = (rankings[0].release_id, rankings[0].release_manifest_sha256)
    for ranking in rankings:
        if (ranking.release_id, ranking.release_manifest_sha256) != identity:
            raise ValueError("Campaign rankings identify different releases")
        tasks = tuple(
            sorted(
                (
                    row.task_id,
                    row.world_id,
                    row.stream,
                    row.clinical_profile_id,
                    row.scenario_profile_id,
                    row.asset_count,
                    row.data_view,
                    row.budget_usd,
                    row.authored_programme,
                    tuple(sorted(row.initial_target_statuses)),
                )
                for row in ranking.items
            )
        )
        if not tasks or len({row.task_id for row in ranking.items}) != len(tasks):
            raise ValueError("Campaign ranking repeats or omits its task inventory")
        if panel is not None and panel != tasks:
            raise ValueError("Campaign rankings must retain the same scheduled task panel")
        panel = tasks
        if any(row.execution_status in {"not_started", "incomplete_run"} for row in ranking.items):
            raise ValueError("Campaign comparison requires every scheduled terminal primary")
        by_model[ranking.model_id].append(ranking)
    if len({len(rows) for rows in by_model.values()}) != 1:
        raise ValueError("Campaign models must retain the same scheduled repetitions")
    authored = {row.authored_programme for row in rankings[0].items if row.stream == "portfolio_management"}
    if False in authored:
        raise ValueError("Campaign requires authored development values")
    results: list[TrialDevComparisonV1] = []
    conditions = sorted({(row.stream, row.data_view) for row in rankings[0].items})
    for stream, data_view, metric in (
        (stream, view, metric)
        for stream, view in conditions
        for metric in ("observational_loss", "net_development_value_usd")
        if metric == "observational_loss" or stream == "portfolio_management"
    ):
        faulty_models: set[str] = set()
        incomplete_models: set[str] = set()
        values: dict[str, dict[tuple[Stratum, str], list[float | None]]] = {}
        for model, runs in sorted(by_model.items()):
            world_values: dict[tuple[Stratum, str], list[float | None]] = defaultdict(list)
            for run in runs:
                for row in run.items:
                    if row.stream != stream or row.data_view != data_view:
                        continue
                    stratum = (row.clinical_profile_id, row.scenario_profile_id, row.asset_count)
                    if stratum not in required_strata:
                        raise ValueError("Observed world falls outside the required release strata")
                    value = row.observational_loss if metric == "observational_loss" else row.net_development_value_usd
                    if row.execution_status == "evaluator_error":
                        faulty_models.add(model)
                        value = None
                    elif row.stream == "evidence_analysis" and row.execution_status == "model_noncompletion":
                        if value is not None:
                            raise ValueError("An uncompleted evidence analysis cannot carry numerical loss")
                        incomplete_models.add(model)
                    elif value is None:
                        if metric == "net_development_value_usd":
                            raise ValueError("Authored programme lacks its development value")
                        incomplete_models.add(model)
                    world_values[stratum, row.world_id].append(value)
            values[model] = world_values
        metric_results = []
        for left, right in combinations(sorted(values), 2):
            left_values, right_values = values[left], values[right]
            if left_values.keys() != right_values.keys():
                raise ValueError("Model comparisons lost world pairing")
            groups: dict[Stratum, list[float]] = defaultdict(list)
            fault = left in faulty_models or right in faulty_models
            incomplete = left in incomplete_models or right in incomplete_models
            for key in sorted(left_values):
                a, b = left_values[key], right_values[key]
                if len(a) != len(b):
                    raise ValueError("Model comparisons lost linked resource conditions or repetitions")
                if not any(value is None for value in (*a, *b)):
                    groups[key[0]].append(
                        mean(value for value in a if value is not None)
                        - mean(value for value in b if value is not None)
                    )
            reason = (
                "evaluator_fault"
                if fault
                else (
                    "model_noncompletion"
                    if incomplete
                    else policy.panel_unavailable_reason(
                        tuple(map(len, groups.values())), required_strata=len(required_strata)
                    )
                )
            )
            estimate = None if reason else mean(mean(group) for group in groups.values())
            low = high = p_value = error = None
            if reason is None:
                draws, low, high, error = _world_bootstrap(groups, policy)
                assert estimate is not None
                p_value = (1 + float(np.sum(np.abs(draws - estimate) >= abs(estimate)))) / (len(draws) + 1)
            metric_results.append(
                TrialDevComparisonV1(
                    scorer=current_scorer(),
                    metric=metric,
                    stream=stream,
                    data_view=data_view,
                    model_left=left,
                    model_right=right,
                    worlds=len(left_values),
                    strata=len(required_strata),
                    estimate=estimate,
                    confidence_low=low,
                    confidence_high=high,
                    p_value=p_value,
                    standard_error=error,
                    minimal_detectable_difference=None if error is None else MDD_MULTIPLIER * error,
                    unavailable_reason=reason,
                    common_world_estimate=(
                        mean(mean(group) for group in groups.values()) if groups and not fault else None
                    ),
                    common_worlds=sum(len(group) for group in groups.values()) if not fault else 0,
                    common_strata=len(groups) if not fault else 0,
                )
            )
        # Unavailable planned comparisons still occupy the prespecified family.
        if not metric_results:
            continue
        adjusted = holm_adjusted_p_values_v1(
            tuple(row.p_value if row.p_value is not None else 1.0 for row in metric_results)
        )
        results.extend(
            row.model_copy(update={"adjusted_p_value": value if row.p_value is not None else None})
            for row, value in zip(metric_results, adjusted, strict=True)
        )
    return tuple(results)


def trialdev_leaderboard_rows(
    rankings: tuple[TrialDevRunRankingV1, ...],
    *,
    required_strata: tuple[Stratum, ...],
    policy: CampaignAnalysisV1,
) -> tuple[LeaderboardRowV1, ...]:
    """Estimate each model's headline per stream and data view with world-clustered intervals and standard errors.

    Parameters
    ----------
    rankings
        Independently verified rankings for every scheduled job.
    required_strata
        Clinical/scenario/asset strata that weight the estimate equally.
    policy
        Prospective bootstrap repetitions, seed, confidence and minimum worlds.

    Returns
    -------
    tuple of LeaderboardRowV1
        Bounded recovery for every task stream, and development value for
        authored programmes (unfinished ones at their incurred costs) when every
        scheduled programme has a value.
        Linked conditions and repetitions are averaged within each world first.
        Loss is aggregated before transforming to recovery. Recovery limits
        reverse the primary loss limits; its standard error uses the delta method.
        The primary world bootstrap requires every stratum's minimum worlds.
    """
    require_same_scorer(ranking.scorer for ranking in rankings)
    rows: list[LeaderboardRowV1] = []
    for model in sorted({ranking.model_id for ranking in rankings}):
        items = [row for ranking in rankings if ranking.model_id == model for row in ranking.items]
        for stream, data_view in sorted({(row.stream, row.data_view) for row in items}):
            selected = [row for row in items if (row.stream, row.data_view) == (stream, data_view)]
            metrics: tuple[
                Literal["bounded_recovery", "net_development_value_usd", "oracle_value_capture_pct"], ...
            ] = (
                ("bounded_recovery", "net_development_value_usd")
                if stream == "portfolio_management"
                else ("bounded_recovery",)
            )
            if stream == "portfolio_management" and any(
                row.oracle_development_value_usd is not None for row in selected
            ):
                metrics = ("oracle_value_capture_pct", *metrics)
            for metric in metrics:
                values: dict[tuple[Stratum, str], list[float | None]] = defaultdict(list)
                oracle_worlds: dict[tuple[Stratum, str], list[float]] = defaultdict(list)
                for row in selected:
                    stratum = (row.clinical_profile_id, row.scenario_profile_id, row.asset_count)
                    if stratum not in required_strata:
                        raise ValueError("Observed world falls outside the required release strata")
                    if metric == "oracle_value_capture_pct":
                        if row.oracle_development_value_usd is None:
                            raise ValueError("Oracle percentage requires a value for every scheduled programme")
                        oracle_worlds[stratum, row.world_id].append(row.oracle_development_value_usd)
                    values[stratum, row.world_id].append(
                        None
                        if row.execution_status in {"not_started", "incomplete_run", "evaluator_error"}
                        else (
                            row.observational_loss if metric == "bounded_recovery" else row.net_development_value_usd
                        )
                    )
                scored = sum(
                    (row.observational_loss if metric == "bounded_recovery" else row.net_development_value_usd)
                    is not None
                    and row.execution_status not in {"not_started", "incomplete_run", "evaluator_error"}
                    for row in selected
                )
                estimate = low = high = error = None
                groups: dict[Stratum, list[float]] = defaultdict(list)
                for (stratum, _), group in sorted(values.items()):
                    if all(value is not None for value in group):
                        groups[stratum].append(mean(value for value in group if value is not None))
                missing = tuple(sorted(set(required_strata) - set(groups)))
                reason = None
                if any(row.execution_status == "evaluator_error" for row in selected):
                    reason = "evaluator_fault"
                elif any(row.execution_status in {"not_started", "incomplete_run"} for row in selected):
                    reason = "missing_scheduled_outcome"
                elif any(value is None for group in values.values() for value in group):
                    reason = "missing_metric"
                else:
                    reason = policy.panel_unavailable_reason(
                        tuple(map(len, groups.values())), required_strata=len(required_strata)
                    )
                if reason is None:
                    estimate = mean(mean(group) for group in groups.values())
                    if metric == "oracle_value_capture_pct":
                        denominators: dict[Stratum, list[float]] = defaultdict(list)
                        for (stratum, _), oracle_group in sorted(oracle_worlds.items()):
                            denominators[stratum].append(max(0.0, mean(oracle_group)))
                        estimate, low, high, error, reason = _capture_bootstrap(groups, denominators, policy)
                    else:
                        _, low, high, error = _world_bootstrap(groups, policy)
                        if metric == "bounded_recovery":
                            estimate, low, high, error = bounded_recovery_summary(
                                estimate, max(0.0, low), max(0.0, high), error
                            )
                rows.append(
                    LeaderboardRowV1(
                        scorer=current_scorer(),
                        benchmark="trialdev",
                        track=f"{stream}/{data_view}",
                        model_id=model,
                        metric=metric,
                        direction="higher_is_better",
                        estimate=estimate,
                        interval_low=low,
                        interval_high=high,
                        standard_error=error,
                        scheduled_tasks=len(selected),
                        scored_tasks=scored,
                        independent_worlds=len({world for _, world in values}),
                        population="partial_release_strata" if missing else "full_release_strata",
                        missing_strata=missing,
                        minimum_worlds_per_stratum=policy.minimum_worlds_per_stratum,
                        nonpositive_oracle_worlds=sum(mean(group) <= 0 for group in oracle_worlds.values()),
                        unavailable_reason=reason if estimate is None else None,
                        uncertainty_unavailable_reason=reason,
                        warnings=(
                            (
                                f"{'Headline' if estimate is None else 'Uncertainty'} unavailable: {reason}; "
                                f"{len(groups)}/{len(required_strata)} required strata, "
                                f"{scored}/{len(selected)} scheduled outcomes scored; requires at least "
                                f"{policy.minimum_worlds_per_stratum} independent worlds per stratum.",
                            )
                            if reason
                            else ()
                        ),
                    )
                )
    return tuple(rows)


class TrialDevCampaignExportV1(BaseModel):
    """One native job export, optionally accompanied by an externally supplied digest."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    job_id: str
    path: Path
    sha256: Digest | None = Field(default=None, exclude_if=lambda value: value is None)


class TrialDevCampaignProfileV1(TrialDevRankingMeanV1):
    """Model profile summary; task counts include scheduled repetitions.

    Linked resource conditions and repetitions are averaged within worlds before
    averaging worlds. Observed means retain their explicit scored denominator.
    """

    model_id: str
    observed_mean: float | None = None
    scored_worlds: int = Field(default=0, ge=0)


def summarize_trialdev_campaign_profiles(
    rankings: tuple[TrialDevRunRankingV1, ...],
) -> tuple[TrialDevCampaignProfileV1, ...]:
    """Summarize authenticated campaign rows using the existing equal-world owner."""
    return tuple(
        TrialDevCampaignProfileV1(model_id=model, **summary.model_dump())
        for model in sorted({ranking.model_id for ranking in rankings})
        for summary in ranking_means(
            [item for ranking in rankings if ranking.model_id == model for item in ranking.items]
        )
    )


class TrialDevCampaignAnalysisV1(BaseModel):
    """Release- and plan-bound comparisons, without a combined benchmark ranking.

    With budget censoring, comparisons and the leaderboard cover the paired
    panel of tasks every model completed; profile summaries cover each model's
    own completed tasks. Censored tasks are reported, never scored.
    """

    scorer: CurrentScorer

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_id: Literal["trialagentbench.trialdev_campaign_analysis/v2"] = (
        "trialagentbench.trialdev_campaign_analysis/v2"
    )
    campaign_plan_sha256: Digest
    release_sha256: Digest
    exports: tuple[TrialDevCampaignExportV1, ...]
    comparisons: tuple[TrialDevComparisonV1, ...]
    profile_summaries: tuple[TrialDevCampaignProfileV1, ...] = ()
    leaderboard: tuple[LeaderboardRowV1, ...] = ()
    budget_censoring: BudgetCensoringV1 | None = Field(default=None, exclude_if=lambda value: value is None)
    oracle_values_sha256: Digest | None = Field(default=None, exclude_if=lambda value: value is None)


def _retain_tasks(ranking: TrialDevRunRankingV1, task_ids: frozenset[str]) -> TrialDevRunRankingV1:
    return ranking.model_copy(update={"items": tuple(row for row in ranking.items if row.task_id in task_ids)})


def analyse_trialdev_campaign(
    release: Path,
    *,
    expected_release_sha256: str,
    plan: CampaignPlanV1,
    expected_plan_checksum: str,
    exports: tuple[TrialDevCampaignExportV1, ...],
    censoring: BudgetCensoringV1 | None = None,
    grade_roots: dict[str, Path] | None = None,
    oracle_values: Path | None = None,
    expected_oracle_values_sha256: str | None = None,
) -> TrialDevCampaignAnalysisV1:
    """Authenticate every planned TrialDev job before computing paired differences.

    Parameters
    ----------
    release
        Authenticated TrialDev evaluator archive or selective asset catalog.
    expected_release_sha256
        Externally retained release digest.
    plan
        Frozen campaign plan.
    expected_plan_checksum
        Externally retained campaign plan checksum.
    exports
        One full custody export per planned TrialDev job.
    censoring
        Budget censoring recorded at finish. Every verified pending task must
        be declared censored, and every censored task must still be pending.

    Returns
    -------
    TrialDevCampaignAnalysisV1
        Paired-panel comparisons and leaderboard, completed-only profiles.

    Raises
    ------
    ValueError
        If custody, the panel or the censoring record disagrees with the
        verified rankings, including pending tasks without budget censoring.
    """
    if (oracle_values is None) != (expected_oracle_values_sha256 is None):
        raise ValueError("Oracle values require both the input file and its pinned SHA-256")
    oracle = (
        None
        if oracle_values is None
        else load_trialdev_oracle_values(oracle_values, expected_sha256=str(expected_oracle_values_sha256))
    )
    if plan.checksum != expected_plan_checksum:
        raise ValueError("Campaign plan differs from the prospectively retained identity")
    if censoring is not None and censoring.campaign_plan_sha256 != expected_plan_checksum:
        raise ValueError("Budget censoring belongs to a different campaign plan")
    jobs = {run.job_id: run for run in plan.runs if run.benchmark == "trialdev"}
    if censoring is not None and set(censoring.censored_by_job("trialdev")) != set(jobs):
        raise ValueError("Budget censoring must cover every planned TrialDev job")
    if not jobs or len(exports) != len(jobs) or {row.job_id for row in exports} != set(jobs):
        raise ValueError("TrialDev exports must cover every planned job exactly once")
    models = {row.name: row.model for row in plan.configuration.models}
    rankings: dict[str, TrialDevRunRankingV1] = {}
    with release_archive(release, expected_sha256=expected_release_sha256) as archive:
        if archive.name.endswith(".assets.json"):
            catalog = read_asset_catalog(archive, expected_sha256=expected_release_sha256)
            selectors = set(plan.configuration.panel.trialdev_tasks)
            selected = tuple(
                task.task_id
                for task in catalog.tasks
                if (
                    bool(
                        selectors.intersection(
                            (task.task_id, task.case_id, task.stream, f"{task.case_id}:{task.stream}")
                        )
                    )
                    if selectors
                    else task.main
                )
            )
            if not selected:
                raise ValueError("Campaign selectors match no task in the authenticated release")
            with opened_release(archive, expected_sha256=expected_release_sha256, task_ids=selected) as root:
                tasks = tuple(
                    _Task.model_validate_json(path.read_bytes())
                    for path in sorted((root / "participant/tasks").glob("*/task.json"))
                )
                strata = read_trialdev_world_strata(root)
                if {row.world_id for row in strata.worlds} != {task.case_id for task in catalog.tasks}:
                    raise ValueError("Selective campaign comparison requires complete shared world metadata")
        else:
            with ZipFile(archive) as files:
                tasks = tuple(
                    _Task.model_validate_json(files.read(name))
                    for name in files.namelist()
                    if name.startswith("participant/tasks/") and name.endswith("/task.json")
                )
                strata = TrialDevWorldStrataV1.model_validate_json(files.read(TRIALDEV_WORLD_STRATA))
        required = tuple(
            sorted({(row.clinical_profile_id, row.scenario_ref, row.candidate_count) for row in strata.worlds})
        )
        selectors = set(plan.configuration.panel.trialdev_tasks)
        selector_tasks: dict[str, set[str]] = defaultdict(set)
        for task in tasks:
            for selector in (task.task_id, task.case_id, task.stream, f"{task.case_id}:{task.stream}"):
                selector_tasks[selector].add(task.task_id)
        if selectors - selector_tasks.keys():
            raise ValueError("Campaign selectors match no task in the authenticated release")
        scheduled = (
            {task_id for selector in selectors for task_id in selector_tasks[selector]}
            if selectors
            else {task.task_id for task in tasks if task.data_view == "raw"}
        )
        for source in sorted(exports, key=lambda row: row.job_id):
            job = jobs[source.job_id]
            if grade_roots is not None:
                ranking = TrialDevRunRankingV1.model_validate_json(
                    (grade_roots[job.job_id] / "ranking.json").read_bytes()
                )
            else:
                report = verify_trialdev_ranking(
                    archive,
                    source.path,
                    expected_release_sha256=expected_release_sha256,
                    expected_result_export_sha256=source.sha256,
                    expected_invocation_sha256=job.invocation_sha256,
                    development=plan.configuration.input_mode == "development",
                )
                if report.status != "pass":
                    raise ValueError("Independent TrialDev ranking verification failed")
                ranking = TrialDevRunRankingV1.model_validate(report.checks[0].observed)
            if ranking.model_id != models[job.model_name] or {row.task_id for row in ranking.items} != scheduled:
                raise ValueError("Verified ranking differs from the frozen model or task panel")
            if censoring is not None and set(
                next(run for run in censoring.runs if run.job_id == job.job_id).scheduled_task_ids
            ) != set(scheduled):
                raise ValueError("Budget censoring differs from the frozen task panel")
            if oracle is not None:
                ranking = bind_trialdev_oracle_values(ranking, oracle)
            rankings[job.job_id] = ranking
    comparisons, profiles, leaderboard = paired_trialdev_results(
        rankings, censoring=censoring, required_strata=required, policy=plan.configuration.analysis, oracle=oracle
    )
    return TrialDevCampaignAnalysisV1(
        scorer=current_scorer(),
        campaign_plan_sha256=expected_plan_checksum,
        release_sha256=expected_release_sha256,
        exports=exports,
        comparisons=comparisons,
        profile_summaries=profiles,
        leaderboard=leaderboard,
        budget_censoring=censoring,
        oracle_values_sha256=expected_oracle_values_sha256,
    )


def paired_trialdev_results(
    rankings: dict[str, TrialDevRunRankingV1],
    *,
    censoring: BudgetCensoringV1 | None,
    required_strata: tuple[Stratum, ...],
    policy: CampaignAnalysisV1,
    oracle: TrialDevOracleValuesV1 | None = None,
) -> tuple[tuple[TrialDevComparisonV1, ...], tuple[TrialDevCampaignProfileV1, ...], tuple[LeaderboardRowV1, ...]]:
    """Compare models on the paired panel and summarise each model over its completed tasks.

    Parameters
    ----------
    rankings
        Independently verified rankings keyed by campaign job identity.
    censoring
        Recorded budget censoring, or None when every task must be terminal.
    required_strata
        Release strata that weight worlds equally.
    policy
        Prospective comparison policy.

    Returns
    -------
    tuple
        Paired comparisons, completed-only profile summaries and paired
        leaderboard rows annotated with each model's censored count.

    Raises
    ------
    ValueError
        If a pending task is not declared budget-censored, or a declared
        censored task is not pending.
    """
    require_same_scorer(ranking.scorer for ranking in rankings.values())
    censored = {} if censoring is None else censoring.censored_by_job("trialdev")
    for job_id, ranking in rankings.items():
        pending = {row.task_id for row in ranking.items if row.execution_status in {"not_started", "incomplete_run"}}
        if pending != censored.get(job_id, frozenset()):
            raise ValueError(
                "Campaign comparison requires every scheduled terminal primary; "
                "only recorded budget censoring may leave a task pending"
            )
    scheduled = frozenset(row.task_id for ranking in rankings.values() for row in ranking.items)
    paired = scheduled if censoring is None else censoring.paired_task_ids("trialdev")
    compared = tuple(_retain_tasks(ranking, paired) for ranking in rankings.values())
    completed = tuple(
        _retain_tasks(ranking, scheduled - censored.get(job_id, frozenset())) for job_id, ranking in rankings.items()
    )
    comparisons = compare_trialdev_rankings(compared, required_strata=required_strata, policy=policy) if paired else ()
    leaderboard = trialdev_leaderboard_rows(compared, required_strata=required_strata, policy=policy) if paired else ()
    if censoring is not None:
        leaderboard = tuple(
            row.model_copy(
                update={
                    "budget_censored_tasks": censoring.censored_count("trialdev", row.model_id),
                    "population": "paired_completed_panel",
                    "warnings": (
                        *row.warnings,
                        "Population: paired completed tasks; budget-censored tasks are excluded.",
                    ),
                }
            )
            for row in leaderboard
        )
    if oracle is not None and compared:
        reference_values = {row.task_id: row.reference_value_usd for row in oracle.values}
        reference = compared[0].model_copy(
            update={
                "model_id": "policy:reference",
                "items": tuple(
                    row.model_copy(
                        update={
                            "execution_status": "completed",
                            "run_checksum": None,
                            "observational_loss": None,
                            "net_development_value_usd": reference_values[row.task_id],
                        }
                    )
                    for row in compared[0].items
                    if row.authored_programme
                ),
            }
        )
        if reference.items:
            leaderboard += tuple(
                row
                for row in trialdev_leaderboard_rows((reference,), required_strata=required_strata, policy=policy)
                if row.metric != "bounded_recovery"
            )
    profiles = summarize_trialdev_campaign_profiles(tuple(row for row in completed if row.items))
    return comparisons, profiles, leaderboard


def render_trialdev_comparisons(report: TrialDevCampaignAnalysisV1, output: Path) -> None:
    """Write a compact comparison table and separate-scale accessible vector plot."""
    require_same_scorer((report.scorer, *(row.scorer for row in report.comparisons)))
    output.mkdir(parents=True, exist_ok=False)
    (output / "analysis.json").write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    with (output / "leaderboard.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LeaderboardRowV1.model_fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            row.model_dump(mode="json") | {"scorer": row.scorer.model_dump_json()} for row in report.leaderboard
        )
    with (output / "comparisons.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TrialDevComparisonV1.model_fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            row.model_dump(mode="json") | {"scorer": row.scorer.model_dump_json()} for row in report.comparisons
        )
    with (output / "profiles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["scorer", *TrialDevCampaignProfileV1.model_fields], lineterminator="\n"
        )
        writer.writeheader()
        for profile in report.profile_summaries:
            values = profile.model_dump()
            values["scorer"] = report.scorer.model_dump_json()
            values["unscored_task_counts"] = json.dumps(profile.unscored_task_counts, sort_keys=True)
            values["target_status_counts"] = (
                ""
                if profile.target_status_counts is None
                else json.dumps(profile.target_status_counts, sort_keys=True)
            )
            writer.writerow(values)
    for metric in sorted({row.metric for row in report.comparisons}):
        rows = [row for row in report.comparisons if row.metric == metric]
        height = 130 + 60 * max(1, len(rows))
        extent = (
            max(
                (
                    abs(value)
                    for row in rows
                    for value in (row.estimate, row.confidence_low, row.confidence_high)
                    if value is not None
                ),
                default=1,
            )
            or 1
        )

        def x(value: float, scale: float = extent) -> float:
            return 730 + 190 * value / scale

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}" role="img" aria-labelledby="title desc">',
            f"<metadata>{escape(report.scorer.model_dump_json())}</metadata>",
            f'<title id="title">TrialDev {metric.replace("_", " ")}: paired model differences</title>',
            '<desc id="desc">Left model minus right model. Equal clinical, scenario and asset-count strata. Intervals resample independent worlds; missing support is labelled. No forced ordering.</desc>',
            '<rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" fill="#172b3a">',
            f'<text x="24" y="30" font-size="20">{metric.replace("_", " ").capitalize()} · paired differences</text>',
            '<text x="24" y="55" font-size="13">World-bootstrap intervals; Holm-adjusted p-values, standard errors '
            "and minimal detectable differences are in the table.</text>",
            f'<path d="M730 75 V{height-45}" stroke="#aab7bf"/>',
        ]
        if not rows:
            parts.append('<text x="24" y="100">No model pair scheduled.</text>')
        for index, row in enumerate(rows):
            y = 92 + index * 60
            parts.append(
                f'<text x="24" y="{y}" font-size="13">{escape(row.model_left)} − {escape(row.model_right)}</text>'
            )
            parts.append(
                f'<text x="24" y="{y+16}" font-size="11">{escape(row.stream)} · {escape(row.data_view)}</text>'
            )
            if row.estimate is not None:
                parts.append(f'<circle cx="{x(row.estimate):.2f}" cy="{y-4}" r="4" fill="#237c86"/>')
                parts.append(f'<text x="970" y="{y}" font-size="13">{row.estimate:.4g}</text>')
            if row.confidence_low is not None and row.confidence_high is not None:
                parts.append(
                    f'<path d="M{x(row.confidence_low):.2f} {y-4} H{x(row.confidence_high):.2f}" stroke="#237c86" stroke-width="2"/>'
                )
            if row.unavailable_reason:
                parts.append(
                    f'<text x="24" y="{y+30}" font-size="12">Inference unavailable: {escape(row.unavailable_reason.replace("_", " "))}</text>'
                )
        parts.extend([f'<text x="720" y="{height-20}" font-size="12">0</text>', "</g></svg>"])
        (output / f"{metric}.svg").write_text("\n".join(parts) + "\n", encoding="utf-8")
