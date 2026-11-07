"""Shared public wire values for executable responses; no grading rules."""

from __future__ import annotations

import math
from enum import Enum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_verifier.scorer import CurrentScorer

TRIALDEV_ASSET_ID_PATTERN = r"^[a-z][a-z0-9_]*(?:[.-][a-z0-9_]+)*$"


class StrictSubmissionModel(BaseModel):
    """Reject undeclared fields and non-finite participant-controlled values."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class _GradeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class TrialDevFrozenProgrammeValue(_GradeModel):
    """Components of fixed-population terminal value, shared with the harness."""

    decision: Literal["stop", "advance"]
    approved: bool = False
    clinical_value_usd: float
    safety_cost_usd: float
    study_cost_usd: float = Field(ge=0)
    time_cost_usd: float = Field(ge=0)
    net_value_usd: float

    @model_validator(mode="after")
    def reconcile_value(self) -> Self:
        """Reject contradictory totals without authenticating their source history."""
        if self.decision == "stop" and self.approved:
            raise ValueError("Stopping cannot obtain regulatory approval")
        if not self.approved and (self.clinical_value_usd != 0 or self.safety_cost_usd != 0):
            raise ValueError("Only an approved advancement retains an asset's clinical or safety payoff")
        expected = self.clinical_value_usd - self.safety_cost_usd - self.study_cost_usd - self.time_cost_usd
        if not math.isfinite(expected) or not math.isclose(expected, self.net_value_usd, rel_tol=0, abs_tol=1e-6):
            raise ValueError("Terminal net value differs from its clinical, safety, study and time components")
        return self


class TrialDevAction(str, Enum):
    """Actions exposed by the sequential development environment."""

    NOMINATE = "nominate"
    RUN_TRIAL = "run_trial"
    CONTINUE = "continue"
    STOP = "stop"
    COMPLETE = "complete"
    DEFER = "defer"
    BACKTRACK = "backtrack"
    SWITCH = "switch"

    @property
    def requires_candidate(self) -> bool:
        """Return whether the action explicitly selects a candidate."""

        return self in {self.NOMINATE, self.RUN_TRIAL, self.CONTINUE, self.COMPLETE, self.SWITCH}

    @property
    def requires_study(self) -> bool:
        """Return whether the action purchases a prospective study."""

        return self in {self.RUN_TRIAL, self.CONTINUE}


class TrialDevActionArgumentsV1(BaseModel):
    """Public argument shapes, independent of scientific or portfolio state."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "allOf": [
                {
                    "if": {"properties": {"action": {"enum": [value for value in actions]}}, "required": ["action"]},
                    "then": {"properties": {field: {"type": "string"}}, "required": [field]},
                    "else": {"properties": {field: {"type": "null"}}},
                }
                for field, actions in (
                    ("asset_id", [action.value for action in TrialDevAction if action.requires_candidate]),
                    ("study_id", [action.value for action in TrialDevAction if action.requires_study]),
                )
            ]
        },
    )

    action: TrialDevAction
    asset_id: str | None = Field(default=None, pattern=TRIALDEV_ASSET_ID_PATTERN)
    study_id: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]*$")

    @model_validator(mode="after")
    def _argument_shapes(self) -> Self:
        if self.action.requires_candidate != (self.asset_id is not None):
            raise ValueError("Action candidate argument must match the public action contract")
        if self.action.requires_study != (self.study_id is not None):
            raise ValueError("TrialDev run_trial and continue decisions require exactly one study ID")
        return self


TRIALDEV_SAMPLING_INTERVAL_DESCRIPTION_V1 = "central two-sided 95% frequentist confidence interval"


class TrialDevEndpointEstimateV1(_GradeModel):
    """One candidate-control estimate extracted from an analyst report."""

    asset_id: str = Field(pattern=TRIALDEV_ASSET_ID_PATTERN)
    endpoint: Literal["clinical_outcome", "safety_risk"]
    unit: str = Field(min_length=1)
    estimate: float
    standard_error: float | None = Field(default=None, gt=0)
    interval_95: tuple[float, float] | None = Field(
        default=None,
        description=TRIALDEV_SAMPLING_INTERVAL_DESCRIPTION_V1,
    )
    sensitivity_interval: tuple[float, float] | None = None

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        """Require ordered intervals containing their corresponding estimates."""

        if self.interval_95 is not None and not self.interval_95[0] <= self.estimate <= self.interval_95[1]:
            raise ValueError("TrialDev confidence interval must contain its estimate")
        if self.sensitivity_interval is not None and self.interval_95 is None:
            raise ValueError("TrialDev sensitivity intervals require a sampling confidence interval")
        if self.sensitivity_interval is not None and not (
            self.sensitivity_interval[0] <= self.estimate <= self.sensitivity_interval[1]
        ):
            raise ValueError("TrialDev sensitivity interval must contain its estimate")
        return self


class TrialDevCandidateDiagnosticsV1(_GradeModel):
    """Overlap and balance diagnostics for one candidate comparison."""

    model_config = ConfigDict(
        json_schema_extra={
            "dependentRequired": {
                "propensity_minimum": ["propensity_maximum"],
                "propensity_maximum": ["propensity_minimum"],
                "effective_sample_size_candidate": ["effective_sample_size_control"],
                "effective_sample_size_control": ["effective_sample_size_candidate"],
            }
        }
    )

    asset_id: str = Field(pattern=TRIALDEV_ASSET_ID_PATTERN)
    propensity_minimum: float | None = Field(default=None, gt=0, lt=1)
    propensity_maximum: float | None = Field(default=None, gt=0, lt=1)
    effective_sample_size: float | None = Field(
        default=None, gt=0, description="Sum of candidate-arm and control-arm Kish effective sample sizes."
    )
    effective_sample_size_ratio: float | None = Field(
        default=None,
        gt=0,
        le=1,
        description="Sum of arm-specific Kish effective sample sizes divided by the comparison participant count.",
    )
    effective_sample_size_candidate: float | None = Field(default=None, gt=0)
    effective_sample_size_control: float | None = Field(default=None, gt=0)
    standardized_mean_differences_after: tuple[float, ...] = Field(
        default=(),
        description="Balance after the declared weighting scheme; raw balance for unweighted diagnostics.",
    )
    standardized_mean_difference_after: float | None = None
    maximum_absolute_standardized_mean_difference_after: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_arm_specific_effective_sample_size(self) -> Self:
        """Require ordered propensity limits and complete arm-specific ESS."""

        propensity = (self.propensity_minimum is not None, self.propensity_maximum is not None)
        if propensity[0] != propensity[1]:
            raise ValueError("TrialDev propensity support requires both minimum and maximum values")
        if (
            self.propensity_minimum is not None
            and self.propensity_maximum is not None
            and self.propensity_minimum > self.propensity_maximum
        ):
            raise ValueError("TrialDev propensity minimum cannot exceed its maximum")
        paired = (self.effective_sample_size_candidate is not None, self.effective_sample_size_control is not None)
        if paired[0] != paired[1]:
            raise ValueError("TrialDev arm-specific effective sample sizes require both candidate and control values")
        return self


class _RankingModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


TrialDevTargetStatus = Literal[
    "reported", "missing", "invalid_replay", "unscored", "not_started", "incomplete_run", "evaluator_error"
]


class TrialDevRankingItemV1(_RankingModel):
    """One scheduled task, retaining its initial and acquired target inventories.

    Initial targets exist before execution. Acquired targets describe only the
    latest verified readout of each chosen study. None means no verified acquired
    inventory; an empty mapping means verified development with no readouts.
    The stratum (clinical profile, opaque scenario reference, candidate count) comes
    from the release's grading key.
    """

    task_id: str
    world_id: str
    stream: Literal["evidence_analysis", "portfolio_management"]
    data_view: Literal["raw", "analysis_ready"] = "raw"
    clinical_profile_id: str
    scenario_profile_id: str
    asset_count: int = Field(default=3, ge=1)
    execution_status: Literal["completed", "model_noncompletion", "evaluator_error", "incomplete_run", "not_started"]
    run_checksum: str | None
    observational_loss: float | None = Field(ge=0)
    initial_target_statuses: dict[str, TrialDevTargetStatus] = Field(min_length=1)
    acquired_target_statuses: dict[str, dict[str, TrialDevTargetStatus]] | None = None
    authored_programme: bool = False
    net_development_value_usd: float | None = None
    oracle_development_value_usd: float | None = Field(default=None, exclude_if=lambda value: value is None)
    budget_usd: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_target_reporting(self) -> Self:
        """Reject invalid identities and a task loss with incomplete targets."""
        if any(not identity.strip() for identity in self.initial_target_statuses):
            raise ValueError("Initial target identities must be nonblank")
        if self.observational_loss is not None and any(
            status != "reported" for status in self.initial_target_statuses.values()
        ):
            raise ValueError("Observational loss requires every initial target to be scored")
        if self.acquired_target_statuses is not None and (
            not self.authored_programme
            or any(
                not study.strip() or not targets or any(not target.strip() for target in targets)
                for study, targets in self.acquired_target_statuses.items()
            )
        ):
            raise ValueError("Acquired target inventories require authored studies with named targets")
        return self


class TrialDevRankingMeanV1(_RankingModel):
    """Equal-world mean within one stream, presentation and scientific profile.

    Target counts describe initial analyses, not the adaptively chosen studies.
    They report completion without weighting tasks by their number of targets.
    ``model_noncompletion`` counts tasks the model did not finish, scored or not:
    an unfinished programme is valued at its incurred costs.
    """

    stream: Literal["evidence_analysis", "portfolio_management"] = "evidence_analysis"
    data_view: Literal["raw", "analysis_ready"] = "raw"

    clinical_profile_id: str
    scenario_profile_id: str
    asset_count: int = Field(default=3, ge=1)
    metric: Literal["observational_loss", "net_development_value_usd"]
    scheduled_tasks: int = Field(ge=1)
    scored_tasks: int = Field(ge=0)
    scheduled_worlds: int = Field(ge=1)
    unscored_task_counts: dict[
        Literal["model_noncompletion", "evaluator_error", "incomplete_run", "not_started", "unscored"], int
    ]
    target_status_counts: dict[TrialDevTargetStatus, int] | None
    evaluator_errors: int = Field(ge=0)
    mean: float | None
    observed_mean: float | None = Field(default=None, exclude_if=lambda value: value is None)
    scored_worlds: int = Field(default=0, ge=0, exclude_if=lambda value: value == 0)
    bounded_recovery: float | None = Field(default=None, ge=0, le=1, exclude_if=lambda value: value is None)
    model_noncompletion: int = Field(default=0, ge=0, exclude_if=lambda value: value == 0)

    @model_validator(mode="after")
    def validate_completion(self) -> Self:
        """Only a fully scored panel without evaluator faults can carry a mean."""
        if (self.metric == "observational_loss") != (self.target_status_counts is not None):
            raise ValueError("Numerical target counts belong only to observational recovery")
        if self.target_status_counts is not None and (
            not self.target_status_counts or any(count <= 0 for count in self.target_status_counts.values())
        ):
            raise ValueError("Numerical target counts must retain a positive scheduled census")
        if max(self.scored_tasks, self.evaluator_errors, self.model_noncompletion) > self.scheduled_tasks:
            raise ValueError("Completion counts exceed the scheduled panel")
        if (
            any(count <= 0 for count in self.unscored_task_counts.values())
            or sum(self.unscored_task_counts.values()) != self.scheduled_tasks - self.scored_tasks
        ):
            raise ValueError("Unscored task counts must account for every missing metric")
        if self.scored_worlds > min(self.scheduled_worlds, self.scored_tasks):
            raise ValueError("Observed worlds exceed their scheduled or scored denominator")
        if self.observed_mean is not None and (not self.scored_tasks or not self.scored_worlds):
            raise ValueError("Observed-case mean requires scored tasks and worlds")
        if self.metric == "observational_loss" and self.observed_mean is not None and self.observed_mean < 0:
            raise ValueError("Observed numerical loss cannot be negative")
        if self.mean is not None and (self.scored_tasks != self.scheduled_tasks or self.evaluator_errors):
            raise ValueError("Scientific mean requires a fully scored panel without evaluator faults")
        if self.metric == "observational_loss" and self.mean is not None and self.mean < 0:
            raise ValueError("Mean numerical loss cannot be negative")
        if self.bounded_recovery is not None and self.metric != "observational_loss":
            raise ValueError("Bounded recovery belongs only to observational recovery")
        return self


class TrialDevRunRankingV1(_RankingModel):
    """Reproducible separate recovery/value results for one frozen invocation."""

    scorer: CurrentScorer

    schema_id: Literal["trialagentbench.trialdev_run_ranking/v6"] = "trialagentbench.trialdev_run_ranking/v6"
    release_id: str
    release_manifest_sha256: str
    invocation_checksum: str
    model_id: str
    scope: Literal["complete_release_task_inventory", "scheduled_panel"]
    items: tuple[TrialDevRankingItemV1, ...]
    profile_means: tuple[TrialDevRankingMeanV1, ...]
