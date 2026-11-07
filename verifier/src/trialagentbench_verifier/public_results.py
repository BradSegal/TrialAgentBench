"""Public aggregate campaign results projected from standalone analysis owners."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_verifier.campaign import (
    AnalysisStatus,
    TrialEvalBetweenModelContrastV1,
    TrialEvalCampaignAnalysisV1,
    TrialEvalDifficultySummaryV1,
    TrialEvalLossRankingV1,
    TrialEvalMetricSummaryV1,
    TrialEvalTargetSummaryV1,
)
from trialagentbench_verifier.campaign_contracts import BudgetCensoringV1, CampaignAnalysisV1, LeaderboardRowV1
from trialagentbench_verifier.development_campaign import (
    TrialDevCampaignAnalysisV1,
    TrialDevCampaignProfileV1,
    TrialDevComparisonV1,
)
from trialagentbench_verifier.scorer import CurrentScorer, current_scorer, require_same_scorer

Split = Literal["public", "withheld"]


def _omitted_when_public(value: Split) -> bool:
    return value == "public"


class PublicTrialEvalCampaignV1(BaseModel):
    """Aggregate numerical recovery with private per-task targets excluded.

    ``plan_execution_summaries`` report, per model and plan-given view (C1, C3), the share of answers that
    executed the locked plan within tolerance; it is scored apart from the numerical loss.

    A withheld campaign carries only per-model, per-view aggregates. Its rankings list
    cells, and its cell, target and family breakdowns, set against the public split,
    would locate the withheld panel's hidden cell structure; its panel rankings are in
    the leaderboard. ``split`` is omitted from public-split results.
    """

    scorer: CurrentScorer

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_id: Literal["trialagentbench.public_trialeval_campaign/v4"] = "trialagentbench.public_trialeval_campaign/v4"
    source_analysis_checksum: str
    release_id: str
    campaign_plan_checksum: str
    source_archive_sha256: str | None = Field(default=None, exclude_if=lambda value: value is None)
    scientific_analysis_status: AnalysisStatus
    unavailable_reason: str | None = Field(default=None, exclude_if=lambda value: value is None)
    confidence_level: float = Field(gt=0, lt=1)
    bootstrap_resamples: int = Field(ge=100)
    bootstrap_seed: int = Field(ge=0)
    loss_rankings: tuple[TrialEvalLossRankingV1, ...]
    cell_loss_summaries: tuple[TrialEvalLossRankingV1, ...]
    difficulty_summaries: tuple[TrialEvalDifficultySummaryV1, ...]
    target_summaries: tuple[TrialEvalTargetSummaryV1, ...]
    between_model_contrasts: tuple[TrialEvalBetweenModelContrastV1, ...]
    plan_execution_summaries: tuple[TrialEvalMetricSummaryV1, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    split: Split = Field(default="public", exclude_if=_omitted_when_public)

    @model_validator(mode="after")
    def withhold_cell_structure(self) -> Self:
        """Reject withheld results that list cells or break the panel down below a model and view."""
        if self.split == "withheld" and (
            self.loss_rankings
            or self.cell_loss_summaries
            or self.difficulty_summaries
            or self.target_summaries
            or any(row.common_cell_refs for row in self.between_model_contrasts)
        ):
            raise ValueError("Withheld public results cannot list cells or carry cell, target or family breakdowns")
        return self


def public_trialeval_campaign(report: TrialEvalCampaignAnalysisV1, *, split: Split) -> PublicTrialEvalCampaignV1:
    """Preserve existing aggregate estimates without exporting oracle targets.

    For a withheld campaign, rows are also sorted by model, view and metric, so their
    order owes nothing to the analysis's cell iteration.
    """
    require_same_scorer((report.scorer, *(row.scorer for row in report.between_model_contrasts)))
    contrasts = tuple(
        row
        for row in report.between_model_contrasts
        if row.metric.metric_id.endswith("_signal_loss") or is_trialeval_headline_contrast(row)
    )
    plan_execution = tuple(row for row in report.metric_summaries if row.metric.metric_id == "plan_execution_accuracy")
    withheld = split == "withheld"
    if withheld:
        contrasts = tuple(
            sorted(
                (row.model_copy(update={"common_cell_refs": ()}) for row in contrasts),
                key=lambda row: (row.minuend_model_id, row.subtrahend_model_id, row.context, row.metric.metric_id),
            )
        )
        plan_execution = tuple(sorted(plan_execution, key=lambda row: (row.model_id, row.context)))
    return PublicTrialEvalCampaignV1(
        scorer=current_scorer(),
        source_analysis_checksum=report.checksum,
        release_id=report.release_id,
        campaign_plan_checksum=report.campaign_plan_checksum,
        source_archive_sha256=report.source_archive_sha256,
        scientific_analysis_status=report.scientific_analysis_status,
        unavailable_reason=report.unavailable_reason,
        confidence_level=report.confidence_level,
        bootstrap_resamples=report.bootstrap_resamples,
        bootstrap_seed=report.bootstrap_seed,
        loss_rankings=() if withheld else report.loss_rankings,
        cell_loss_summaries=() if withheld else report.cell_loss_summaries,
        difficulty_summaries=() if withheld else report.difficulty_summaries,
        target_summaries=() if withheld else report.target_summaries,
        between_model_contrasts=contrasts,
        plan_execution_summaries=plan_execution,
        split=split,
    )


class CampaignLeaderboardV1(BaseModel):
    """Shareable per-model headlines for one prospectively planned campaign.

    Rows cover the paired panel of tasks no model lost to budget censoring.
    ``budget_censoring`` reports every model's censoring, and ``warnings``
    flags materially different censoring between compared models. A withheld
    leaderboard omits ``budget_censoring``, which names tasks; its rows keep
    each model's censored count.
    """

    scorer: CurrentScorer

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_id: Literal["trialagentbench.campaign_leaderboard/v3"] = "trialagentbench.campaign_leaderboard/v3"
    campaign_plan_sha256: str
    release_set_manifest_sha256: str
    split: Split
    confidence_level: float = Field(gt=0, lt=1)
    bootstrap_resamples: int = Field(ge=100)
    bootstrap_seed: int = Field(ge=0)
    rows: tuple[LeaderboardRowV1, ...]
    budget_censoring: BudgetCensoringV1 | None = None
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def withhold_task_identities(self) -> Self:
        """Reject a withheld leaderboard that names tasks."""
        require_same_scorer((self.scorer, *(row.scorer for row in self.rows)))
        if self.split == "withheld" and self.budget_censoring is not None:
            raise ValueError("A withheld leaderboard cannot name budget-censored tasks")
        return self


TRIALEVAL_HEADLINE_CONTEXT = "C5"
"""The main TrialEval selection: raw records with recoverable defects, no locked plan."""
_DECISION_METRICS: tuple[
    tuple[str, Literal["decision_regret_musd", "consequential_error_rate", "material_decision_error_rate"]], ...
] = (
    ("decision_regret_musd", "decision_regret_musd"),
    ("consequential_error", "consequential_error_rate"),
    ("material_decision_error_rate", "material_decision_error_rate"),
)


def is_trialeval_headline_contrast(row: TrialEvalBetweenModelContrastV1) -> bool:
    """Select paired decision metrics on the same context as the per-model headline.

    Parameters
    ----------
    row
        Paired metric contrast from the campaign analysis.

    Returns
    -------
    bool
        Whether the contrast is a headline decision metric on the headline context.
    """
    return row.context == TRIALEVAL_HEADLINE_CONTEXT and any(
        row.metric.metric_id == metric for metric, _ in _DECISION_METRICS
    )


def trialeval_leaderboard_rows(
    report: TrialEvalCampaignAnalysisV1, *, censoring: BudgetCensoringV1 | None = None
) -> tuple[LeaderboardRowV1, ...]:
    """Project panel-level TrialEval rankings, which already resample independent worlds.

    The headline prices each answer by replay at the true effect: its saved script's regret against the
    scripted reference route over its task's bank replays (US$ millions, lower is better, below zero beats the
    reference), as the cell-balanced mean, with the share of material decision errors, both on the main C5 view.
    A missing or unreproduced answer drives the prior-only decision, so both are defined for every scheduled
    answer. Mean numerical loss is the secondary row.

    Parameters
    ----------
    report
        Analysis of the paired TrialEval panel.
    censoring
        Budget censoring whose per-model counts annotate each row.

    Returns
    -------
    tuple of LeaderboardRowV1
        Decision regret, material-decision-error rate and mean loss for each model and target kind.
    """
    require_same_scorer((report.scorer, *(row.scorer for row in report.between_model_contrasts)))
    rows = []
    summaries = {
        (row.model_id, row.metric.metric_id): row
        for row in getattr(report, "metric_summaries", ())
        if row.context == TRIALEVAL_HEADLINE_CONTEXT
    }
    for ranking in report.loss_rankings:
        common = {
            "benchmark": "trialeval",
            "track": ranking.target_kind,
            "model_id": ranking.model_id,
            "scheduled_tasks": ranking.scheduled_attempts,
            "scored_tasks": ranking.scored_attempts,
            "independent_worlds": ranking.independent_worlds,
            "budget_censored_tasks": (
                0 if censoring is None else censoring.censored_count("trialeval", ranking.model_id)
            ),
            "population": "partial_release_strata" if report.unavailable_reason else "full_release_strata",
            "minimum_worlds_per_stratum": CampaignAnalysisV1().minimum_worlds_per_stratum,
            "unavailable_reason": report.unavailable_reason,
            "uncertainty_unavailable_reason": report.unavailable_reason,
            "warnings": (
                (f"Diagnostic panel only: {report.unavailable_reason}; comparisons and uncertainty withheld.",)
                if report.unavailable_reason
                else ()
            ),
        }
        for metric_id, label in _DECISION_METRICS:
            summary = summaries.get((ranking.model_id, metric_id))
            if summary is None:
                if report.unavailable_reason:
                    rows.append(
                        LeaderboardRowV1(
                            scorer=current_scorer(), **common, metric=label, direction="lower_is_better", estimate=None
                        )
                    )
                continue
            rows.append(
                LeaderboardRowV1(
                    scorer=current_scorer(),
                    **common
                    | {"scheduled_tasks": summary.scheduled_attempts, "scored_tasks": summary.observed_attempts},
                    metric=label,
                    direction="lower_is_better",
                    estimate=summary.equal_cell_effect_mean,
                    interval_low=summary.interval_low,
                    interval_high=summary.interval_high,
                    standard_error=summary.standard_error,
                )
            )
        rows.append(
            LeaderboardRowV1(
                scorer=current_scorer(),
                **common,
                metric="mean_loss",
                direction="lower_is_better",
                estimate=ranking.mean_loss,
                interval_low=ranking.interval_low,
                interval_high=ranking.interval_high,
                standard_error=ranking.standard_error,
            )
        )
    return tuple(rows)


class PublicTrialDevCampaignV1(BaseModel):
    """Paired recovery and development-value comparisons without local custody paths.

    A withheld campaign omits ``budget_censoring``, which names tasks. ``split`` is
    omitted from public-split results.
    """

    scorer: CurrentScorer

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_id: Literal["trialagentbench.public_trialdev_campaign/v4"] = "trialagentbench.public_trialdev_campaign/v4"
    campaign_plan_sha256: str
    release_sha256: str
    oracle_values_sha256: str | None = Field(default=None, exclude_if=lambda value: value is None)
    leaderboard: tuple[LeaderboardRowV1, ...] = ()
    export_sha256: tuple[str, ...] = Field(default=(), exclude_if=lambda value: not value)
    analysis_policy: CampaignAnalysisV1
    comparisons: tuple[TrialDevComparisonV1, ...]
    profile_summaries: tuple[TrialDevCampaignProfileV1, ...] = ()
    budget_censoring: BudgetCensoringV1 | None = None
    split: Split = Field(default="public", exclude_if=_omitted_when_public)

    @model_validator(mode="after")
    def withhold_task_identities(self) -> Self:
        """Reject withheld results that name tasks."""
        if self.split == "withheld" and self.budget_censoring is not None:
            raise ValueError("Withheld public results cannot name budget-censored tasks")
        return self


def public_trialdev_campaign(
    report: TrialDevCampaignAnalysisV1, *, policy: CampaignAnalysisV1, split: Split
) -> PublicTrialDevCampaignV1:
    """Retain the prespecified inference policy and exact standalone comparisons."""
    require_same_scorer((report.scorer, *(row.scorer for row in report.comparisons)))
    return PublicTrialDevCampaignV1(
        scorer=current_scorer(),
        campaign_plan_sha256=report.campaign_plan_sha256,
        release_sha256=report.release_sha256,
        oracle_values_sha256=report.oracle_values_sha256,
        leaderboard=report.leaderboard,
        export_sha256=tuple(row.sha256 for row in report.exports if row.sha256 is not None),
        analysis_policy=policy,
        comparisons=report.comparisons,
        profile_summaries=report.profile_summaries,
        budget_censoring=None if split == "withheld" else report.budget_censoring,
        split=split,
    )
