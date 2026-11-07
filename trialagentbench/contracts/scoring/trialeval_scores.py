"""Typed TrialEval score artifacts and analysis rows."""

from __future__ import annotations

from statistics import mean
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator
from trialagentbench_verifier.numerical import NumericalReplay
from trialagentbench_verifier.scorer import CurrentScorer

from trialagentbench.contracts.core.config import (
    ProcedureAssistanceV1,
    ReasoningEffortV1,
    TrialEvalAgentStatusV1,
)
from trialagentbench.contracts.trace.observable import BenchmarkRuntimeFailureTypeV1
from trialagentbench.grading.outcome import (
    TrialEvalComponentOutcomeV1,
    TrialEvalOracleTargetV1,
    TrialEvalOutcomeGradeV1,
    TrialEvalPlanExecutionV1,
)

TrialEvalObservedAttemptOutcomeV1: TypeAlias = Literal[
    "completed",
    "provider_stop",
    "provider_unavailable",
    "model_noncompletion",
    "malformed_output",
    "replay_failure",
    "budget_exhausted",
    "evaluator_error",
    "pre_run_identity_exclusion",
]


def trialeval_infrastructure_outcome_v1(
    failure_type: BenchmarkRuntimeFailureTypeV1 | None,
) -> TrialEvalObservedAttemptOutcomeV1 | None:
    """Identify infrastructure eligibility independently of numerical replay."""

    # A receipt-bound transport outage is unfinished even when a partial script
    # already left replayable estimates. Its one replacement resolves that primary.
    if failure_type == "provider_unavailable":
        return "provider_unavailable"
    if failure_type in {"cancelled", "evaluator_error"}:
        return "evaluator_error"
    if failure_type == "pre_run_identity_error":
        return "pre_run_identity_exclusion"
    return None


def trialeval_observed_attempt_outcome_v1(
    *,
    agent_status: TrialEvalAgentStatusV1,
    failure_type: BenchmarkRuntimeFailureTypeV1 | None,
    outcome: TrialEvalOutcomeGradeV1,
) -> TrialEvalObservedAttemptOutcomeV1:
    """Derive one campaign outcome from canonical terminal and grade records."""

    infrastructure = trialeval_infrastructure_outcome_v1(failure_type)
    if infrastructure is not None:
        return infrastructure
    if failure_type == "participant_session_terminated":
        return "model_noncompletion"
    if outcome.completion.status == "satisfied":
        if outcome.reproducibility.status != "satisfied":
            return "replay_failure"
        return "completed"
    if agent_status == "max_turns_reached" or failure_type == "turn_limit_no_submission":
        return "model_noncompletion"
    if failure_type == "provider_error":
        return "provider_stop"
    if failure_type == "timeout":
        return "budget_exhausted"
    return "malformed_output"


class TrialEvalTargetScoreV1(BaseModel):
    """One requested numerical result with its own reference and diagnostics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: TrialEvalOutcomeGradeV1
    target: TrialEvalOracleTargetV1
    oracle_worlds: int = Field(ge=1)
    oracle_monte_carlo_standard_error: float = Field(ge=0.0)
    maximum_oracle_monte_carlo_standard_error: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _target_matches_grade(self) -> TrialEvalTargetScoreV1:
        if self.target.kind != self.outcome.target_kind:
            raise ValueError("TrialEval result kind differs from its target")
        if self.oracle_monte_carlo_standard_error > self.maximum_oracle_monte_carlo_standard_error:
            raise ValueError("TrialEval reference exceeds its precision limit")
        point = self.outcome.point_performance
        if point is not None and (
            point.oracle_value != self.target.value
            or point.reference_sampling_standard_deviation != self.target.reference_sampling_standard_deviation
        ):
            raise ValueError("TrialEval point performance disagrees with its oracle target.")
        return self


class _TrialEvalTaskScoresV1(TrialEvalTargetScoreV1):
    """Retain the primary result and average all required numerical results equally.

    ``plan_execution`` scores the plan as executed of a plan-given task apart: it does not enter the loss
    or the decision.
    """

    scorer: CurrentScorer

    additional_results: tuple[TrialEvalTargetScoreV1, ...] = Field(default=(), exclude_if=lambda value: not value)
    replay: NumericalReplay
    decision: TrialEvalComponentOutcomeV1 | None = Field(default=None, exclude_if=lambda value: value is None)
    plan_execution: TrialEvalPlanExecutionV1 | None = Field(default=None, exclude_if=lambda value: value is None)

    @property
    def target_results(self) -> tuple[TrialEvalTargetScoreV1, ...]:
        """Return primary and complementary results in their reporting order."""
        return (self, *self.additional_results)

    @property
    def signal_loss(self) -> float | None:
        """Average target losses only when every required result has a loss."""
        values = tuple(result.outcome.signal_loss for result in self.target_results)
        return None if any(value is None for value in values) else mean(value for value in values if value is not None)

    @model_validator(mode="after")
    def _coherent_result_inventory(self) -> _TrialEvalTaskScoresV1:
        identities = tuple(result.target.binding.estimand_id for result in self.target_results)
        if len(set(identities)) != len(identities):
            raise ValueError("TrialEval scores repeat a requested target")
        plan = () if self.plan_execution is None else (self.plan_execution.plan.target_id,)
        if (self.additional_results or plan) and (*identities, *plan) != self.replay.target_ids:
            raise ValueError("TrialEval score targets differ from the complete replay inventory")
        if any(result.outcome.context != self.outcome.context for result in self.additional_results):
            raise ValueError("TrialEval target scores have different task contexts")
        if self.decision is not None and self.decision.component_id != "decision":
            raise ValueError("TrialEval task decision must be the decision component")
        return self


class TrialEvalItemScoresV1(_TrialEvalTaskScoresV1):
    """Schema-bearing score wrapper persisted with one TrialEval item."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["ariadne.trialeval.item_scores/v1"] = "ariadne.trialeval.item_scores/v1"
    schema_version: Literal[1] = 1
    item_id: str
    task_id: str | None = None
    release_id: str = Field(min_length=1)
    trial_id: str = Field(pattern=r"^trial_[0-9a-f]{20}$")
    base_world_id: str = Field(pattern=r"^world_[0-9a-f]{20}$")
    trial_name: str | None = None
    cell_ref: str = Field(pattern=r"^cell_[0-9a-f]{20}$")
    effect_state: Literal["favourable", "near_null", "unfavourable"]
    design_tier: str = ""
    design_subtype: str = ""
    context_tier: str = ""
    effect_measure: str = Field(min_length=1)
    result_unit: str = Field(min_length=1)
    estimand_mode: str = ""
    model: str
    turns_used: int = Field(ge=0)
    agent_status: TrialEvalAgentStatusV1

    @model_validator(mode="after")
    def _score_matches_item(self) -> TrialEvalItemScoresV1:
        if self.outcome.context != self.context_tier:
            raise ValueError("TrialEval outcome context must match its score wrapper.")
        # The primary outcome is reproducible only when its own answer was usable: reported (a complementary or
        # plan row alone leaves it missing) and gradable (a nonpositive ratio is retained but not eligible). The
        # outcome's completion records that binding, so the replay must agree with it, not with mere presence.
        primary_reported = any(row.target_id == self.target.binding.estimand_id for row in self.replay.estimates or ())
        usable_primary = self.outcome.completion.status == "satisfied"
        if usable_primary != (self.outcome.reproducibility.status == "satisfied") or (
            usable_primary and not (self.replay.reproducible and primary_reported)
        ):
            raise ValueError("TrialEval replay and reproducibility component disagree.")
        binding = self.target.binding
        if (
            self.target.kind != self.outcome.target_kind
            or binding.effect_measure != self.effect_measure
            or binding.result_unit != self.result_unit
            or self.oracle_monte_carlo_standard_error > self.maximum_oracle_monte_carlo_standard_error
        ):
            raise ValueError("TrialEval oracle target provenance disagrees with its score wrapper.")
        return self


class TrialEvalGradedItemRowV1(_TrialEvalTaskScoresV1):
    """One canonical graded item used by trace and downstream analysis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["ariadne.trialeval_graded_item_row/v1"] = "ariadne.trialeval_graded_item_row/v1"
    release_id: str = Field(min_length=1)
    participant_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    run_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_id: str = Field(min_length=1)
    trial_id: str = Field(pattern=r"^trial_[0-9a-f]{20}$")
    base_world_id: str = Field(pattern=r"^world_[0-9a-f]{20}$")
    condition_id: str = Field(min_length=1)
    request_replicate_id: str = Field(min_length=1)
    reasoning_effort: ReasoningEffortV1 | None = None
    provider_capability_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    procedure_assistance: ProcedureAssistanceV1
    analysis_specification: Literal["protocol_only", "locked_sap"]
    trial_name: str = ""
    cell_ref: str = Field(pattern=r"^cell_[0-9a-f]{20}$")
    effect_state: Literal["favourable", "near_null", "unfavourable"]
    design_tier: str = Field(min_length=1)
    design_subtype: str = Field(min_length=1)
    context_tier: str = Field(min_length=1)
    effect_measure: str = Field(min_length=1)
    result_unit: str = Field(min_length=1)
    estimand_mode: str = ""
    agent_status: TrialEvalAgentStatusV1
    terminal_failure_type: BenchmarkRuntimeFailureTypeV1 | None = None
    observed_attempt_outcome: TrialEvalObservedAttemptOutcomeV1
    turns_used: int = Field(ge=0)
    source_json_path: str = Field(min_length=1)
    source_json_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _reasoning_identity_is_complete(self) -> TrialEvalGradedItemRowV1:
        if self.reasoning_effort is not None and self.provider_capability_sha256 is None:
            raise ValueError("A requested reasoning effort requires a capability identity.")
        return self

    @model_validator(mode="after")
    def _outcome_matches_context(self) -> TrialEvalGradedItemRowV1:
        if self.outcome.context != self.context_tier:
            raise ValueError("TrialEval outcome context must match the analysis row.")
        expected_outcome = trialeval_observed_attempt_outcome_v1(
            agent_status=self.agent_status,
            failure_type=self.terminal_failure_type,
            outcome=self.outcome,
        )
        if self.observed_attempt_outcome != expected_outcome:
            raise ValueError("TrialEval observed attempt outcome differs from canonical terminal evidence.")
        if (
            self.target.kind != self.outcome.target_kind
            or self.target.binding.effect_measure != self.effect_measure
            or self.target.binding.result_unit != self.result_unit
            or self.oracle_monte_carlo_standard_error > self.maximum_oracle_monte_carlo_standard_error
        ):
            raise ValueError("TrialEval analysis row oracle provenance disagrees with its outcome.")
        return self


__all__ = [
    "trialeval_infrastructure_outcome_v1",
    "TrialEvalGradedItemRowV1",
    "TrialEvalItemScoresV1",
    "TrialEvalTargetScoreV1",
    "TrialEvalObservedAttemptOutcomeV1",
    "trialeval_observed_attempt_outcome_v1",
]
