"""Authenticated, denominator-complete analysis for controlled TrialEval campaigns."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath
from statistics import mean
from typing import Literal, NamedTuple, Self, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, model_validator
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from trialagentbench_verifier.campaign_contracts import CampaignAnalysisV1
from trialagentbench_verifier.decision_consequence import (
    DecisionConsequenceV1,
    DecisionReportV1,
    decision_consequence_v1,
)
from trialagentbench_verifier.execution_contracts import ReplayInputArtifactV1
from trialagentbench_verifier.io import (
    canonical_sha256,
    read_json_object,
)
from trialagentbench_verifier.layer_priors import load_layer_prior
from trialagentbench_verifier.numerical import NumericalReplay
from trialagentbench_verifier.numerical_scoring import bounded_recovery_summary
from trialagentbench_verifier.replay_scoring import (
    ANCHOR_POLICIES,
    ECONOMICS_BAND,
    FailureCause,
    ReplayScoreV1,
    TrialEvalAnswerRowV1,
    TrialEvalReplayEvidenceV1,
    anchor_scores_v1,
    economics_v1,
    saved_source_sha256,
    score_replays_v1,
    score_single_dataset_v1,
)
from trialagentbench_verifier.run_contracts import ProviderUnavailabilityReceiptV1
from trialagentbench_verifier.scorer import (
    CurrentScorer,
    ScorerIdentityV1,
    current_scorer,
    require_current_scorer,
    require_same_scorer,
)
from trialagentbench_verifier.trialeval import (
    TrialEvalGradingPlanV1,
    TrialEvalScoreInputV1,
    TrialEvalTargetV1,
    VerifiedTrialEvalPlanExecutionV1,
    VerifiedTrialEvalTargetOutcomeV1,
    _estimand_core_matches,
    _metrics,
    _PointResult,
    _Result,
    _signal_loss,
    trialeval_plan_execution_v1,
)

EffectState = Literal["favourable", "near_null", "unfavourable"]
Context = Literal["C1", "C2", "C3", "C4", "C5"]
AttemptRole = Literal["primary", "stochastic_duplicate", "retry"]
AttemptOutcome = Literal[
    "completed",
    "not_started",
    "provider_stop",
    "provider_unavailable",
    "model_noncompletion",
    "malformed_output",
    "replay_failure",
    "budget_exhausted",
    "evaluator_error",
    "pre_run_identity_exclusion",
]
MetricDirection = Literal["higher_is_better", "lower_is_better"]
MetricDomain = Literal[
    "completion",
    "clean_replay",
    "numerical_performance",
    "uncertainty",
    "canonical_diagnostic",
    "decision_consequence",
    "plan_execution",
]
# Infrastructure outcomes are the only ones a primary may be retried after, exactly once. Model
# noncompletion, malformed output, provider_stop and budget exhaustion are scored outcomes.
# Only the receipt-bound, prospectively declared provider_unavailable outcome is recoverable.
INFRASTRUCTURE_ATTEMPT_OUTCOMES_V1: frozenset[AttemptOutcome] = frozenset(
    {"evaluator_error", "pre_run_identity_exclusion", "provider_unavailable"}
)
AnalysisStatus = Literal[
    "complete", "unavailable_due_unscored_primary_attempts", "unavailable_due_insufficient_independent_worlds"
]
EstimateStatus = Literal["estimated", "unavailable_due_missing_metric"]

EFFECT_STATES_V1: tuple[EffectState, ...] = ("favourable", "near_null", "unfavourable")
CONTEXTS_V1: tuple[Context, ...] = ("C1", "C2", "C3", "C4", "C5")
CELL_KEY_PATTERN = r"^cell_[0-9a-f]{20}$"
SERIES_KEY_PATTERN = r"^series_[0-9a-f]{20}$"
PLAN_CONTEXTS_V1: frozenset[Context] = frozenset({"C1", "C3"})
C_CONTRASTS_V1: tuple[tuple[str, Context, Context], ...] = (
    ("specification_ready", "C1", "C2"),
    ("specification_raw", "C3", "C4"),
    ("reconstruction_sap", "C1", "C3"),
    ("reconstruction_protocol", "C2", "C4"),
    ("transfer_repair", "C4", "C5"),
)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class _ResultExportMemberV1(_Model):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class _ResultExportSourceV1(_Model):
    kind: Literal["run", "grade", "verification", "analysis"]
    label: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    archive_prefix: str = Field(min_length=1)
    file_count: int = Field(ge=1)
    custody_file_count: int | None = Field(default=None, ge=1)
    custody_byte_length: int | None = Field(default=None, ge=1)
    custody_tree_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_custody(self) -> Self:
        values = (self.custody_file_count, self.custody_byte_length, self.custody_tree_sha256)
        if any(value is None for value in values) != all(value is None for value in values):
            raise ValueError("result-export custody identity must be complete when present")
        if self.custody_file_count is not None and self.file_count > self.custody_file_count:
            raise ValueError("included result files cannot exceed the external custody tree")
        return self


class _ResultExportManifestV1(_Model):
    schema_id: Literal["ariadne.result_export_bundle/v1"]
    release_id: str = Field(min_length=1)
    release_stage: Literal["benchmark_publication"]
    release_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    release_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    export_profile: Literal["full_custody", "analysis_reproduction"]
    sources: tuple[_ResultExportSourceV1, ...] = Field(min_length=2)
    members: tuple[_ResultExportMemberV1, ...] = Field(min_length=1)
    member_count: int = Field(ge=1)
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if self.member_count != len(self.members):
            raise ValueError("member_count differs from result member inventory")
        if not {"run", "grade"}.issubset({source.kind for source in self.sources}):
            raise ValueError("result export omits a required source kind")

        if tuple(member.path for member in self.members) != tuple(sorted({member.path for member in self.members})):
            raise ValueError("result-export member paths are not sorted and unique")
        expected = canonical_sha256(self.model_dump(mode="json", exclude={"checksum"}))
        if self.checksum != expected:
            raise ValueError("result-export manifest checksum mismatch")
        return self


class _GradeManifestV1(_Model):
    scorer: CurrentScorer
    schema_id: Literal["trialagentbench_grade_manifest_v1"]
    schema_version: Literal[1]
    suite: Literal["trialeval"]
    harness_version: str = Field(min_length=1)
    harness_git_sha: str | None = Field(min_length=1)
    harness_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    timestamp_utc: datetime
    input_run_dir: str = Field(min_length=1)
    output_run_dir: str = Field(min_length=1)
    input_run_sha256: str | None = Field(
        default=None, exclude_if=lambda value: value is None, pattern=r"^[0-9a-f]{64}$"
    )
    evaluator_release_sha256: str | None = Field(
        default=None, exclude_if=lambda value: value is None, pattern=r"^[0-9a-f]{64}$"
    )
    release_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    score_profile_id: Literal["oracle_outcome_components_v1"]
    scorer_surface_sha256: str | None = Field(
        default=None, exclude_if=lambda value: value is None, pattern=r"^[0-9a-f]{64}$"
    )
    suite_task_count: int = Field(ge=1)
    graded_task_count: int = Field(ge=0)
    notes: tuple[str, ...] = ()
    checksum: str | None = Field(default=None, exclude_if=lambda value: value is None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_denominator(self) -> Self:
        if self.graded_task_count > self.suite_task_count:
            raise ValueError("graded task count exceeds its scheduled denominator")
        if self.checksum is not None and self.checksum != canonical_sha256(
            self.model_dump(mode="json", exclude={"checksum"})
        ):
            raise ValueError("grade manifest checksum mismatch")
        return self


class _ComponentV1(_Model):
    component_id: str = Field(min_length=1)
    status: Literal["satisfied", "not_satisfied", "observed", "not_evaluable", "not_applicable"]
    failure_origin: str | None = None


class _PointPerformanceV1(_Model):
    normalized_absolute_error: float | None = Field(default=None, ge=0)
    reference_sampling_standard_deviation: float = Field(gt=0)


class _IntervalPerformanceV1(_Model):
    covers_target: bool
    native_width: float | None = Field(default=None, ge=0)
    proper_interval_score: float | None = Field(default=None, ge=0)
    native_interval_score: float | None = Field(default=None, ge=0)


class _CampaignOutcomeV1(_Model):
    context: Context
    target_kind: Literal["point"]
    completion: _ComponentV1
    question_alignment: _ComponentV1
    evidence_eligibility: _ComponentV1
    reproducibility: _ComponentV1
    estimand_performance: _ComponentV1
    uncertainty: _ComponentV1
    point_performance: _PointPerformanceV1 | None = None
    interval_performance: _IntervalPerformanceV1 | None = None
    verification_metrics: dict[str, float | bool | str | None] | None = None
    signal_loss: float | None = Field(default=None, ge=0.0)
    target_outcomes: tuple[VerifiedTrialEvalTargetOutcomeV1, ...] = ()
    decision: _ComponentV1 | None = Field(default=None, exclude_if=lambda value: value is None)


class TrialEvalOraclePrecisionV1(_Model):
    """Monte Carlo precision for one target, separate from its sampling scale."""

    oracle_worlds: int = Field(ge=1)
    oracle_monte_carlo_standard_error: float = Field(ge=0.0)
    maximum_oracle_monte_carlo_standard_error: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_precision(self) -> Self:
        """Reject reference uncertainty above the declared precision limit."""
        if self.oracle_monte_carlo_standard_error > self.maximum_oracle_monte_carlo_standard_error:
            raise ValueError("TrialEval reference exceeds its precision limit")
        return self


class TrialEvalCampaignRowV1(_Model):
    """Independent campaign projection of one authenticated graded item."""

    scorer: CurrentScorer

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
    procedure_assistance: str = Field(min_length=1)
    analysis_specification: Literal["protocol_only", "locked_sap"]
    cell_ref: str = Field(pattern=CELL_KEY_PATTERN)
    effect_state: EffectState
    context_tier: Context
    effect_measure: str = Field(min_length=1)
    result_unit: str = Field(min_length=1)
    agent_status: str = Field(min_length=1)
    terminal_failure_type: str | None = None
    observed_attempt_outcome: AttemptOutcome
    outcome: _CampaignOutcomeV1
    target: TrialEvalTargetV1
    additional_targets: tuple[TrialEvalTargetV1, ...] = ()
    additional_oracle_precision: tuple[TrialEvalOraclePrecisionV1, ...] = ()
    oracle_worlds: int = Field(ge=1)
    oracle_monte_carlo_standard_error: float = Field(ge=0)
    maximum_oracle_monte_carlo_standard_error: float = Field(gt=0)
    source_json_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    score_input_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    replay_program_artifacts: tuple[ReplayInputArtifactV1, ...] = ()
    replay_program_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    replay_submitted_result_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    replayed_result_sha256s: tuple[str, ...] = ()
    plan_execution: VerifiedTrialEvalPlanExecutionV1 | None = None
    primary_answer: tuple[float, float, float] | None = Field(
        default=None, description="The submitted primary estimate and interval (native scale), when reported."
    )

    @property
    def targets(self) -> tuple[TrialEvalTargetV1, ...]:
        """Return every numerical reference defining this task."""
        return (self.target, *self.additional_targets)

    @model_validator(mode="after")
    def validate_attempt_outcome(self) -> Self:
        if len(self.additional_targets) != len(self.additional_oracle_precision):
            raise ValueError("Each additional target requires its own oracle precision")
        if self.additional_targets or self.outcome.target_outcomes:
            if tuple(row.target_id for row in self.outcome.target_outcomes) != tuple(
                target.binding.estimand_id for target in self.targets
            ):
                raise ValueError("Campaign diagnostics differ from the ordered target inventory")
        expected = _attempt_outcome(
            agent_status=self.agent_status,
            failure_type=self.terminal_failure_type,
            completion=self.outcome.completion,
            reproducibility=self.outcome.reproducibility,
        )
        if self.observed_attempt_outcome != expected:
            raise ValueError("attempt outcome differs from canonical terminal evidence")
        return self


_TARGET_ADAPTER: TypeAdapter[TrialEvalTargetV1] = TypeAdapter(TrialEvalTargetV1)
# The score components every required target carries; the item-level decision is graded separately.
_TARGET_COMPONENTS = (
    "completion",
    "question_alignment",
    "evidence_eligibility",
    "reproducibility",
    "estimand_performance",
    "uncertainty",
)


def _object(value: JsonValue, *, label: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _string(value: JsonValue | None, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _component_from_grade(payload: dict[str, JsonValue], name: str) -> _ComponentV1:
    component = _ComponentV1.model_validate(_object(payload.get(name), label=f"outcome.{name}"))
    if component.component_id != name:
        raise ValueError(f"outcome.{name} has the wrong component identity")
    return component


def _attempt_outcome(
    *,
    agent_status: str,
    failure_type: str | None,
    completion: _ComponentV1,
    reproducibility: _ComponentV1,
) -> AttemptOutcome:
    # A partial executable answer cannot erase a proven interruption before the
    # agent's final submission. Receipt/policy validation occurs at source binding.
    if failure_type == "provider_unavailable":
        return "provider_unavailable"
    if failure_type in {"cancelled", "evaluator_error"}:
        return "evaluator_error"
    if failure_type == "pre_run_identity_error":
        return "pre_run_identity_exclusion"
    if failure_type == "participant_session_terminated":
        return "model_noncompletion"
    if completion.status == "satisfied":
        return "completed" if reproducibility.status == "satisfied" else "replay_failure"
    if agent_status == "max_turns_reached" or failure_type == "turn_limit_no_submission":
        return "model_noncompletion"
    if failure_type == "provider_error":
        return "provider_stop"
    if failure_type == "timeout":
        return "budget_exhausted"
    return "malformed_output"


def _independent_outcome(
    producer: dict[str, JsonValue],
    *,
    target: TrialEvalTargetV1,
    submitted: _Result | None,
) -> _CampaignOutcomeV1:
    context = _string(producer.get("context"), label="outcome.context")
    target_kind = _string(producer.get("target_kind"), label="outcome.target_kind")
    if context not in CONTEXTS_V1 or target_kind != target.kind:
        raise ValueError("graded outcome differs from its context or oracle target kind")
    components = {name: _component_from_grade(producer, name) for name in _TARGET_COMPONENTS}
    verification_metrics: dict[str, float | bool | str | None] | None = None
    point: _PointPerformanceV1 | None = None
    interval: _IntervalPerformanceV1 | None = None
    if submitted is not None:
        verification_metrics = _metrics(target, submitted)
        point = _PointPerformanceV1.model_validate(
            {
                "normalized_absolute_error": verification_metrics["normalized_absolute_error"],
                "reference_sampling_standard_deviation": target.reference_sampling_standard_deviation,
            }
        )
        if submitted.interval is not None:
            interval = _IntervalPerformanceV1.model_validate(
                {
                    "covers_target": verification_metrics["interval_covers_oracle"],
                    "native_width": verification_metrics["native_interval_width"],
                    "proper_interval_score": verification_metrics["proper_interval_score"],
                    "native_interval_score": verification_metrics["native_interval_score"],
                }
            )
    return _CampaignOutcomeV1(
        context=cast(Context, context),
        target_kind=cast(Literal["point"], target_kind),
        **components,
        point_performance=point,
        interval_performance=interval,
        verification_metrics=verification_metrics,
        signal_loss=(
            _signal_loss(target, submitted)
            if submitted is not None
            and all(components[name].status in {"satisfied", "observed"} for name in _TARGET_COMPONENTS)
            else None
        ),
    )


def _campaign_row(
    payload: dict[str, JsonValue],
    *,
    source_sha256: str,
    score_input_sha256: str,
    score_input: TrialEvalScoreInputV1,
) -> TrialEvalCampaignRowV1:
    scorer = require_current_scorer(
        ScorerIdentityV1.model_validate(_object(payload.get("scores"), label="scores").get("scorer"))
    )
    required_item = {"schema_id", "schema_version", "item_id", "timestamp_utc", "run_config", "agent_output", "scores"}
    if set(payload) != required_item:
        raise ValueError("TrialEval item result has an unexpected top-level schema")
    item_id = _string(payload.get("item_id"), label="item_id")
    run = _object(payload.get("run_config"), label="run_config")
    agent = _object(payload.get("agent_output"), label="agent_output")
    scores = _object(payload.get("scores"), label="scores")
    run_identity = _string(run.get("run_identity_sha256"), label="run_config.run_identity_sha256")
    identity_payload = {
        key: value
        for key, value in run.items()
        if key not in {"timestamp_utc", "participant_archive", "workers", "run_identity_sha256"}
    }
    if run_identity != canonical_sha256(identity_payload):
        raise ValueError("TrialEval run config has an invalid authoritative identity")
    run_payload = {key: value for key, value in run.items() if value is not None}
    run_id = f"trialeval:{canonical_sha256(cast(JsonValue, run_payload))}"
    if _string(scores.get("item_id"), label="scores.item_id") != item_id:
        raise ValueError("TrialEval score wrapper differs from its item")
    task_id = scores.get("task_id")
    if task_id is not None and task_id != item_id:
        raise ValueError("TrialEval score wrapper has a different task identity")
    model = _string(run.get("model"), label="run_config.model")
    if scores.get("model") != model or scores.get("agent_status") != agent.get("status"):
        raise ValueError("TrialEval score wrapper differs from terminal run evidence")
    condition = _object(run.get("experiment_condition"), label="run_config.experiment_condition")
    factors = _object(run.get("task_evidence_factors"), label="run_config.task_evidence_factors")
    task_factors = _object(factors.get(item_id), label="run_config.task_evidence_factors[item]")
    provenance = _object(agent.get("condition_provenance"), label="agent_output.condition_provenance")
    if (
        provenance.get("procedure_assistance") != condition.get("procedure_assistance")
        or provenance.get("analysis_specification") != task_factors.get("analysis_specification")
        or provenance.get("max_turns") != condition.get("maximum_turns_per_step")
        or provenance.get("prompt_set_sha256") != run.get("prompt_set_sha256")
    ):
        raise ValueError("TrialEval condition provenance differs from its run configuration")
    target = _TARGET_ADAPTER.validate_python(scores.get("target"))
    replay = NumericalReplay.model_validate(scores.get("replay"))
    extra_payload = scores.get("additional_results", [])
    if not isinstance(extra_payload, list):
        raise ValueError("Complementary target scores must be a list")
    extra_scores = tuple(_object(item, label="additional target score") for item in extra_payload)
    targets = (target, *(_TARGET_ADAPTER.validate_python(item.get("target")) for item in extra_scores))
    expected_ids = tuple(item.binding.estimand_id for item in targets)
    # A plan-given task also reports the plan as executed, graded apart.
    plan_payload = scores.get("plan_execution")
    plan = (
        None
        if plan_payload is None
        else TrialEvalGradingPlanV1.model_validate(_object(plan_payload, label="plan").get("plan"))
    )
    if (plan is None) != (score_input.plan_input is None):
        raise ValueError("campaign plan execution differs from its portable score input")
    reported_ids = (*expected_ids, *(() if plan is None else (plan.target_id,)))
    if len(set(reported_ids)) != len(reported_ids) or replay.target_ids != reported_ids:
        raise ValueError("campaign replay must identify the complete assigned target inventory")
    inputs = score_input.numerical_inputs
    input_ids = tuple(item.target_id or target.binding.estimand_id for item in inputs)
    if input_ids != expected_ids:
        raise ValueError("Campaign score inputs differ from required targets")
    entrypoint = tuple(row for row in replay.program_artifacts if row.relative_path == f"scratch/{replay.entrypoint}")
    if len(entrypoint) > 1 or (replay.reproducible and len(entrypoint) != 1):
        raise ValueError("campaign replay does not bind one source entrypoint")
    program_sha256 = None if not entrypoint else entrypoint[0].sha256
    submitted_result_sha256 = replay.submitted_csv_sha256
    replayed_hashes = replay.replayed_csv_sha256s
    estimates = {item.target_id: item for item in replay.estimates or ()}
    for identity, item in zip(reported_ids, score_input.program_inputs, strict=True):
        if (
            item.submitted_result is not None
            and tuple(sorted(row.relative_path for row in replay.evidence_artifacts)) != item.evidence_source_artifacts
        ):
            raise ValueError("campaign score input differs from replay evidence")
        submitted = item.submitted_result
        estimate = estimates.get(identity)
        if estimate is None:
            if submitted is not None:
                raise ValueError("campaign score input has no certified CSV result")
        elif item.attempt_status != "complete":
            # A retained CSV row the grader scored as missing (for example a nonpositive ratio) carries no answer.
            if submitted is not None:
                raise ValueError("campaign score input commits an answer it scored as missing")
        elif (
            not isinstance(submitted, _PointResult)
            or submitted.interval is None
            or (submitted.value, submitted.interval.lower, submitted.interval.upper)
            != (estimate.estimate, estimate.lower, estimate.upper)
        ):
            raise ValueError("campaign score input differs from certified CSV numbers")
    outcomes = tuple(
        _independent_outcome(
            _object(item.get("outcome"), label="target outcome"), target=reference, submitted=source.submitted_result
        )
        for item, reference, source in zip((scores, *extra_scores), targets, inputs, strict=True)
    )
    outcome = outcomes[0]
    if extra_scores:
        target_outcomes = tuple(
            VerifiedTrialEvalTargetOutcomeV1.model_validate(
                {
                    "target_id": identity,
                    "context": value.context,
                    "components": [getattr(value, name).model_dump() for name in _TARGET_COMPONENTS],
                    "metrics": value.verification_metrics,
                    "signal_loss": value.signal_loss,
                }
            )
            for identity, value in zip(expected_ids, outcomes, strict=True)
        )
        updates = {
            name: next(
                (
                    getattr(value, name)
                    for value in outcomes
                    if getattr(value, name).status in {"not_satisfied", "not_evaluable"}
                ),
                getattr(outcome, name),
            )
            for name in _TARGET_COMPONENTS
        }
        losses = tuple(value.signal_loss for value in outcomes)
        outcome = outcome.model_copy(
            update={
                **updates,
                "target_outcomes": target_outcomes,
                "signal_loss": (
                    None
                    if any(value is None for value in losses)
                    else mean(value for value in losses if value is not None)
                ),
            }
        )
    if scores.get("decision") is not None:
        outcome = outcome.model_copy(update={"decision": _component_from_grade(scores, "decision")})
    plan_execution = (
        None
        if plan is None
        else trialeval_plan_execution_v1(plan=plan, target=target, plan_input=score_input.plan_input)
    )
    if plan_execution is not None and _object(plan_payload, label="plan").get("accurate") != plan_execution.accurate:
        raise ValueError("graded plan execution differs from its independent recomputation")
    failure = agent.get("failure_type")
    if failure is not None and not isinstance(failure, str):
        raise ValueError("agent failure type must be a string or null")
    receipt_payload = agent.get("provider_unavailability")
    if (failure == "provider_unavailable") != (receipt_payload is not None):
        raise ValueError(
            "Provider unavailability requires its retained request receipt, and no other outcome may carry one"
        )
    if receipt_payload is not None:
        receipt = ProviderUnavailabilityReceiptV1.model_validate(receipt_payload)
        receipt.require_run_binding(
            item_id=item_id,
            model=model,
            policy=_string(
                _object(run.get("retry"), label="run_config.retry").get("unavailability_policy"),
                label="unavailability_policy",
            ),
        )
        if agent.get("status") != "failed" or agent.get("report") is not None:
            raise ValueError("Provider unavailability cannot replace a returned model result")
    return TrialEvalCampaignRowV1(
        scorer=scorer,
        release_id=_string(scores.get("release_id"), label="scores.release_id"),
        participant_manifest_checksum=_string(
            run.get("participant_manifest_checksum"), label="run_config.participant_manifest_checksum"
        ),
        model_id=model,
        run_id=run_id,
        run_identity_sha256=run_identity,
        task_id=item_id,
        trial_id=_string(scores.get("trial_id"), label="scores.trial_id"),
        base_world_id=_string(scores.get("base_world_id"), label="scores.base_world_id"),
        condition_id=_string(condition.get("condition_id"), label="condition_id"),
        request_replicate_id=_string(condition.get("request_replicate_id"), label="request_replicate_id"),
        procedure_assistance=_string(condition.get("procedure_assistance"), label="procedure_assistance"),
        analysis_specification=cast(
            Literal["protocol_only", "locked_sap"],
            _string(task_factors.get("analysis_specification"), label="analysis_specification"),
        ),
        cell_ref=_string(scores.get("cell_ref"), label="cell_ref"),
        effect_state=cast(EffectState, _string(scores.get("effect_state"), label="effect_state")),
        context_tier=cast(Context, _string(scores.get("context_tier"), label="context_tier")),
        effect_measure=_string(scores.get("effect_measure"), label="effect_measure"),
        result_unit=_string(scores.get("result_unit"), label="result_unit"),
        agent_status=_string(agent.get("status"), label="agent_output.status"),
        terminal_failure_type=cast(str | None, failure),
        observed_attempt_outcome=_attempt_outcome(
            agent_status=_string(agent.get("status"), label="agent_output.status"),
            failure_type=cast(str | None, failure),
            completion=outcome.completion,
            reproducibility=outcome.reproducibility,
        ),
        outcome=outcome,
        target=target,
        additional_targets=tuple(targets[1:]),
        additional_oracle_precision=tuple(
            TrialEvalOraclePrecisionV1.model_validate(
                {name: item.get(name) for name in TrialEvalOraclePrecisionV1.model_fields}
            )
            for item in extra_scores
        ),
        oracle_worlds=cast(int, scores.get("oracle_worlds")),
        oracle_monte_carlo_standard_error=cast(float, scores.get("oracle_monte_carlo_standard_error")),
        maximum_oracle_monte_carlo_standard_error=cast(float, scores.get("maximum_oracle_monte_carlo_standard_error")),
        source_json_sha256=source_sha256,
        score_input_sha256=score_input_sha256,
        replay_program_artifacts=replay.program_artifacts,
        replay_program_sha256=program_sha256,
        replay_submitted_result_sha256=submitted_result_sha256,
        replayed_result_sha256s=replayed_hashes,
        plan_execution=plan_execution,
        primary_answer=(
            (inputs[0].submitted_result.value, interval.lower, interval.upper)
            if isinstance(inputs[0].submitted_result, _PointResult)
            and (interval := inputs[0].submitted_result.interval) is not None
            else None
        ),
    )


class UnsupportedTrialEvalPanelError(ValueError):
    """A valid diagnostic task subset cannot support the prespecified crossed-panel analysis."""


class TrialEvalCampaignUnavailableV1(_Model):
    """Authenticate individual grades while withholding unsupported panel inference."""

    scorer: CurrentScorer

    schema_id: Literal["trialagentbench.trialeval_campaign_unavailable/v1"] = (
        "trialagentbench.trialeval_campaign_unavailable/v1"
    )
    status: Literal["unavailable"] = "unavailable"
    reason: str = Field(min_length=1)
    result_export_sha256: str | None = Field(
        default=None, exclude_if=lambda value: value is None, pattern=r"^[0-9a-f]{64}$"
    )
    campaign_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class TrialEvalCampaignCellV1(_Model):
    """One TrialEval scenario and its clinical question, identified by opaque grading keys."""

    cell_ref: str = Field(pattern=CELL_KEY_PATTERN)
    series_ref: str = Field(pattern=SERIES_KEY_PATTERN)


class TrialEvalCampaignInventoryV1(_Model):
    """Derived campaign counts for the admitted inventory."""

    series_count: int = Field(ge=1)
    regime_cell_count: int = Field(ge=1)
    context_count: int = Field(default=5, ge=1, le=5)
    base_worlds_per_series: int | None = Field(ge=1)
    independent_base_worlds: int = Field(ge=1)
    primary_attempts_per_model: int = Field(ge=1)
    stochastic_duplicates_per_series: int | None = Field(ge=0)
    stochastic_duplicate_attempts_per_model: int = Field(ge=0)
    model_count: int = Field(ge=1)
    primary_attempts: int = Field(ge=1)
    stochastic_duplicate_attempts: int = Field(ge=0)
    total_scheduled_attempts: int = Field(ge=1)


def derive_trial_eval_campaign_inventory_v1(
    cells: tuple[TrialEvalCampaignCellV1, ...],
    *,
    model_count: int = 1,
    contexts: tuple[Context, ...] = CONTEXTS_V1,
    base_worlds_per_series: int = 21,
    stochastic_duplicates_per_series: int = 3,
) -> TrialEvalCampaignInventoryV1:
    """Derive all campaign denominators from admitted cells and model count."""

    if not cells or len({row.cell_ref for row in cells}) != len(cells):
        raise ValueError("TrialEval campaign cells must be nonempty and unique")
    if model_count < 1 or base_worlds_per_series < 1:
        raise ValueError("TrialEval campaign requires models and positive base-world allocation")
    if stochastic_duplicates_per_series < 0:
        raise ValueError("TrialEval stochastic-duplicate allocation cannot be negative")
    if not contexts or len(set(contexts)) != len(contexts) or not set(contexts) <= set(CONTEXTS_V1):
        raise ValueError("TrialEval campaign contexts must be nonempty, unique and supported")
    series_count = len({row.series_ref for row in cells})
    cell_count = len(cells)
    worlds = series_count * base_worlds_per_series
    primary_per_model = cell_count * base_worlds_per_series * len(contexts)
    duplicates_per_model = cell_count * stochastic_duplicates_per_series * len(contexts)
    return TrialEvalCampaignInventoryV1(
        series_count=series_count,
        context_count=len(contexts),
        regime_cell_count=cell_count,
        base_worlds_per_series=base_worlds_per_series,
        independent_base_worlds=worlds,
        primary_attempts_per_model=primary_per_model,
        stochastic_duplicates_per_series=stochastic_duplicates_per_series,
        stochastic_duplicate_attempts_per_model=duplicates_per_model,
        model_count=model_count,
        primary_attempts=primary_per_model * model_count,
        stochastic_duplicate_attempts=duplicates_per_model * model_count,
        total_scheduled_attempts=(primary_per_model + duplicates_per_model) * model_count,
    )


class TrialEvalAuthenticatedGradeSourceV1(_Model):
    """Rows authenticated by one portable result-export member inventory."""

    scorer: CurrentScorer

    release_id: str = Field(min_length=1)
    release_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    export_manifest_checksum: str | None = Field(
        default=None, exclude_if=lambda value: value is None, pattern=r"^[0-9a-f]{64}$"
    )
    archive_sha256: str | None = Field(default=None, exclude_if=lambda value: value is None, pattern=r"^[0-9a-f]{64}$")
    member_sha256s: tuple[str, ...] = Field(min_length=1)
    rows: tuple[TrialEvalCampaignRowV1, ...]


def load_trial_eval_campaign_grade_source_v1(
    archive_path: str,
    *,
    expected_release_id: str,
    expected_release_manifest_sha256: str,
    expected_archive_sha256: str | None,
    grade_roots: tuple[Path, ...] = (),
) -> TrialEvalAuthenticatedGradeSourceV1:
    """Authenticate archived inputs or load native grades, then recompute score arithmetic."""
    import hashlib
    import json
    from zipfile import ZipFile

    rows: list[TrialEvalCampaignRowV1] = []
    manifest_checksum = None

    def append(item_bytes: bytes, input_bytes: bytes) -> None:
        payload = json.loads(item_bytes)
        score_input = TrialEvalScoreInputV1.model_validate_json(input_bytes)
        row = _campaign_row(
            payload,
            source_sha256=hashlib.sha256(item_bytes).hexdigest(),
            score_input_sha256=hashlib.sha256(input_bytes).hexdigest(),
            score_input=score_input,
        )
        if row.task_id != score_input.task_id:
            raise ValueError("TrialEval portable input identifies a different task")
        if payload["scores"]["release_id"] != expected_release_id:
            raise ValueError("TrialEval graded item identifies a different release")
        rows.append(row)

    def validate_grade(payload: dict[str, JsonValue]) -> None:
        grade = _GradeManifestV1.model_validate(payload)
        if grade.suite != "trialeval" or grade.release_manifest_sha256 != expected_release_manifest_sha256:
            raise ValueError("TrialEval grade manifest identifies different released bytes")

    if grade_roots:
        for root in grade_roots:
            validate_grade(read_json_object(root / "GRADE_MANIFEST.json"))
            for item in sorted((root / "items").glob("*.json")):
                append(item.read_bytes(), (root / "verification_inputs" / item.name).read_bytes())
    else:
        from trialagentbench_verifier.release_set import verify_result_export

        if expected_archive_sha256 is None:
            raise ValueError("TrialEval archive requires an externally trusted SHA-256")
        manifest = verify_result_export(Path(archive_path), expected_sha256=expected_archive_sha256)
        with ZipFile(archive_path) as archive:
            manifest_checksum = manifest.checksum
            if (manifest.release_id, manifest.release_manifest_sha256) != (
                expected_release_id,
                expected_release_manifest_sha256,
            ):
                raise ValueError("TrialEval result export identifies different released bytes")
            for source in manifest.sources:
                if source.kind != "grade":
                    continue
                prefix = source.archive_prefix
                validate_grade(json.loads(archive.read(f"{prefix}/GRADE_MANIFEST.json")))
                for name in sorted(archive.namelist()):
                    if name.startswith(f"{prefix}/items/") and name.endswith(".json"):
                        append(
                            archive.read(name),
                            archive.read(f"{prefix}/verification_inputs/{PurePosixPath(name).name}"),
                        )
    if not rows:
        raise ValueError("TrialEval result contains no canonical graded item rows")
    hashes = tuple(row.source_json_sha256 for row in rows)
    if len(hashes) != len(set(hashes)):
        raise ValueError("TrialEval result reuses one graded item")
    return TrialEvalAuthenticatedGradeSourceV1(
        scorer=current_scorer(),
        release_id=expected_release_id,
        release_manifest_sha256=expected_release_manifest_sha256,
        export_manifest_checksum=manifest_checksum,
        archive_sha256=expected_archive_sha256,
        member_sha256s=hashes,
        rows=tuple(rows),
    )


class TrialEvalCampaignAttemptV1(_Model):
    """One prospectively scheduled attempt with no caller-authored outcome."""

    attempt_id: str = Field(min_length=1)
    scheduled_unit_id: str = Field(min_length=1)
    attempt_role: AttemptRole
    retry_of_attempt_id: str | None = None
    model_id: str = Field(min_length=1)
    run_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_id: str = Field(min_length=1)
    trial_id: str = Field(pattern=r"^trial_[0-9a-f]{20}$")
    base_world_id: str = Field(pattern=r"^world_[0-9a-f]{20}$")
    series_ref: str = Field(pattern=SERIES_KEY_PATTERN)
    cell_ref: str = Field(pattern=CELL_KEY_PATTERN)
    effect_state: EffectState
    context: Context
    condition_id: str = Field(min_length=1)
    request_replicate_id: str = Field(min_length=1)
    procedure_assistance: Literal["output_contract_only", "unordered_checklist"]
    analysis_specification: Literal["protocol_only", "locked_sap"]
    design_family: str = Field(min_length=1)
    reconstruction_family: str | None = Field(pattern=r"^[a-z][a-z0-9_]+$")
    target_ids: tuple[str, ...] = Field(min_length=1)
    target_kind: Literal["point"]
    effect_measure: str = Field(min_length=1)
    result_unit: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_identity(self) -> TrialEvalCampaignAttemptV1:
        if (self.context == "C5") != (self.reconstruction_family is not None):
            raise ValueError("Reconstruction family is required exactly for C5 attempts")
        if len(set(self.target_ids)) != len(self.target_ids) or any(not target for target in self.target_ids):
            raise ValueError("TrialEval attempt requires unique nonempty target identities")
        if (self.attempt_role == "retry") is (self.retry_of_attempt_id is None):
            raise ValueError("TrialEval retry identity does not match its attempt role")
        return self


class TrialEvalCampaignPlanV1(_Model):
    """Frozen analysis and exact primary, duplicate, and conditional retry schedule."""

    schema_id: Literal["trialagentbench.trialeval_campaign_plan/v4"] = "trialagentbench.trialeval_campaign_plan/v4"
    release_id: str = Field(min_length=1)
    release_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    participant_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    prospective_analysis_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cells: tuple[TrialEvalCampaignCellV1, ...] = Field(min_length=1)
    nested_series_refs: tuple[str, ...] = Field(
        default=(),
        description=(
            "Series whose cells the release presents on disjoint base worlds (the withheld split); every other "
            "series crosses its cells with every world."
        ),
    )
    base_worlds_per_series: int | None = Field(default=None, ge=1)
    stochastic_duplicates_per_series: int | None = Field(default=None, ge=0)
    model_comparisons: tuple[tuple[str, str], ...] = ()
    assumption_comparisons: tuple[tuple[str, str], ...] = ()
    attempts: tuple[TrialEvalCampaignAttemptV1, ...] = Field(min_length=1)
    checksum: str = ""

    @property
    def contexts(self) -> tuple[Context, ...]:
        """Return the presentation panel fixed by scheduled primary attempts."""
        selected = {row.context for row in self.attempts if row.attempt_role == "primary"}
        return tuple(context for context in CONTEXTS_V1 if context in selected)

    @model_validator(mode="after")
    def bind_plan(self) -> TrialEvalCampaignPlanV1:
        pairs = tuple(tuple(sorted(pair)) for pair in self.model_comparisons)
        if any(left == right for left, right in pairs) or len(pairs) != len(set(pairs)):
            raise ValueError("TrialEval model comparisons must be unique pairs")
        if pairs != tuple(sorted(pairs)):
            raise ValueError("TrialEval model comparisons must be sorted and canonical")
        cells = {cell.cell_ref: cell.series_ref for cell in self.cells}
        if any(cells.get(row.cell_ref) != row.series_ref for row in self.attempts):
            raise ValueError("TrialEval attempts must name an admitted cell of their series")
        if self.nested_series_refs != tuple(sorted(set(self.nested_series_refs))) or not set(
            self.nested_series_refs
        ) <= set(cells.values()):
            raise ValueError("TrialEval nested series must be sorted, unique and among the plan's series")
        if len(self.assumption_comparisons) != len({tuple(sorted(pair)) for pair in self.assumption_comparisons}):
            raise ValueError("TrialEval assumption comparisons repeat a pair")
        for left, right in self.assumption_comparisons:
            if left == right or left not in cells or right not in cells or cells[left] != cells[right]:
                raise ValueError("TrialEval assumption comparisons require distinct cells in one matched series")
        attempt_ids = [row.attempt_id for row in self.attempts]
        if len(attempt_ids) != len(set(attempt_ids)):
            raise ValueError("TrialEval campaign attempt identifiers must be unique")
        retry_origins = [row.retry_of_attempt_id for row in self.attempts if row.attempt_role == "retry"]
        if len(retry_origins) != len(set(retry_origins)):
            raise ValueError("TrialEval campaign permits at most one conditional retry per scheduled unit")
        payload = self.model_dump(mode="json", exclude={"checksum"})
        expected = canonical_sha256(payload)
        if self.checksum and self.checksum != expected:
            raise ValueError("TrialEval campaign plan checksum mismatch")
        object.__setattr__(self, "checksum", expected)
        return self


class TrialEvalAttemptOutcomeCountsV1(_Model):
    """Operational outcomes without collapsed failure origins."""

    completed: int = Field(ge=0)
    not_started: int = Field(ge=0)
    provider_stop: int = Field(ge=0)
    provider_unavailable: int = Field(default=0, ge=0, exclude_if=lambda value: value == 0)
    model_noncompletion: int = Field(ge=0)
    malformed_output: int = Field(ge=0)
    replay_failure: int = Field(ge=0)
    budget_exhausted: int = Field(ge=0)
    evaluator_error: int = Field(ge=0)
    pre_run_identity_exclusion: int = Field(ge=0)


class TrialEvalAttemptLedgerSummaryV1(_Model):
    """Complete operational and scientific attempt denominators."""

    scheduled_primary_attempts: int = Field(ge=0)
    scheduled_stochastic_duplicates: int = Field(ge=0)
    retry_attempts: int = Field(ge=0)
    total_attempts: int = Field(ge=1)
    recovered_primary_attempt_ids: tuple[str, ...]
    primary_scientifically_unscored_attempt_ids: tuple[str, ...]
    outcomes: TrialEvalAttemptOutcomeCountsV1


def failure_cause_v1(row: TrialEvalCampaignRowV1) -> FailureCause | None:
    """Return why a graded row's primary answer cannot be replay-scored, or None when it can.

    Parameters
    ----------
    row
        One authenticated campaign row.

    Returns
    -------
    FailureCause or None
        ``no_answer`` without a primary estimate and interval, ``unreproduced`` when the execution proof did
        not reproduce it, ``wrong_question`` when it answers another question; None for a scorable answer.
    """
    if row.primary_answer is None:
        return "no_answer"
    if row.outcome.reproducibility.status != "satisfied" or not row.replay_program_artifacts:
        return "unreproduced"
    estimand = (
        next(item for item in row.outcome.target_outcomes[0].components if item.component_id == "estimand_performance")
        if row.outcome.target_outcomes
        else row.outcome.estimand_performance
    )
    return None if estimand.status == "observed" else "wrong_question"


def reproduced_answer_v1(row: TrialEvalCampaignRowV1) -> tuple[float, float, float] | None:
    """Return a graded row's primary answer when its saved script reproduced it on the original dataset.

    Parameters
    ----------
    row
        One authenticated campaign row.

    Returns
    -------
    tuple of float or None
        The native-scale estimate and interval; None for a failure (``failure_cause_v1``).
    """
    return None if failure_cause_v1(row) is not None else row.primary_answer


def _answer_scores(
    row: TrialEvalCampaignRowV1, replay: TrialEvalReplayEvidenceV1 | None
) -> dict[str, ReplayScoreV1] | None:
    """Score one row at every economics band point; None when its target has no decision consequence."""
    if row.target.consequence is None:
        return None
    answer = reproduced_answer_v1(row)
    records = None if replay is None else replay.bank.records.get(row.task_id)
    if not records:
        return {
            name: score_single_dataset_v1(row.target, answer, value=value, cost=cost)
            for name, (value, cost) in ECONOMICS_BAND.items()
        }
    outcomes = None
    if answer is not None:
        assert replay is not None
        source = saved_source_sha256(row.replay_program_artifacts)
        outcomes = replay.replays_for(row.task_id, source)
        if outcomes is None:
            raise ValueError(f"reproduced answer to {row.task_id} ({source}) has no replay outcomes")
    return {
        name: score_replays_v1(row.target, records, outcomes, value=value, cost=cost)
        for name, (value, cost) in ECONOMICS_BAND.items()
    }


class _BoundAttempt:
    def __init__(
        self,
        plan: TrialEvalCampaignAttemptV1,
        row: TrialEvalCampaignRowV1 | None,
        replay: TrialEvalReplayEvidenceV1 | None = None,
    ):
        self.plan = plan
        self.row = row
        self.scores = None if row is None else _answer_scores(row, replay)

    @property
    def outcome(self) -> AttemptOutcome:
        """Return the grade-derived outcome or an explicit missing-attempt label."""

        return "not_started" if self.row is None else self.row.observed_attempt_outcome


# Request replicate of an authorized once-only infrastructure replacement, which stands in for the
# primary attempt an infrastructure fault prevented (never for a model outcome).
_INFRASTRUCTURE_REPLACEMENT_PREFIX = "infra-"


def _attempt_identity(
    row: TrialEvalCampaignRowV1 | TrialEvalCampaignAttemptV1, *, replicate: bool = True
) -> tuple[str, ...]:
    # The run identity and condition label name the operational cohort (batch) that produced a row. A
    # converged campaign draws one attempt per task from several cohorts of one scientific condition, so
    # binding uses the scientific factors only; the plan's condition label still governs comparisons.
    return (
        row.model_id,
        row.task_id,
        row.trial_id,
        row.base_world_id,
        row.cell_ref,
        row.effect_state,
        row.context_tier if isinstance(row, TrialEvalCampaignRowV1) else row.context,
        *((row.request_replicate_id,) if replicate else ()),
        row.procedure_assistance,
        row.analysis_specification,
    )


def _bind_attempts(
    plan: TrialEvalCampaignPlanV1,
    source: TrialEvalAuthenticatedGradeSourceV1,
    replay: TrialEvalReplayEvidenceV1 | None = None,
) -> tuple[_BoundAttempt, ...]:
    require_same_scorer((source.scorer, *(row.scorer for row in source.rows)))
    if source.release_id != plan.release_id or source.release_manifest_sha256 != plan.release_manifest_sha256:
        raise ValueError("TrialEval campaign plan identifies different exported release bytes")
    validated_rows = tuple(
        TrialEvalCampaignRowV1.model_validate(row.model_dump(mode="python"), extra="ignore") for row in source.rows
    )
    by_identity = {_attempt_identity(row): row for row in validated_rows}
    if len(by_identity) != len(validated_rows):
        raise ValueError("TrialEval authenticated grade export repeats one attempt identity")
    replacements = {
        _attempt_identity(row, replicate=False): row
        for row in validated_rows
        if row.request_replicate_id.startswith(_INFRASTRUCTURE_REPLACEMENT_PREFIX)
    }
    bound = []
    used_rows: set[str] = set()
    for attempt in plan.attempts:
        row = by_identity.get(_attempt_identity(attempt))
        if row is None and attempt.attempt_role == "primary":
            row = replacements.get(_attempt_identity(attempt, replicate=False))
        if row is not None:
            if (
                row.release_id != plan.release_id
                or row.participant_manifest_checksum != plan.participant_manifest_checksum
                or tuple(target.binding.estimand_id for target in row.targets) != attempt.target_ids
                or row.target.kind != attempt.target_kind
                or row.effect_measure != attempt.effect_measure
                or row.result_unit != attempt.result_unit
            ):
                raise ValueError("TrialEval canonical grade row differs from campaign release identity")
            if row.source_json_sha256 in used_rows:
                raise ValueError("TrialEval campaign reuses one canonical grade row")
            used_rows.add(row.source_json_sha256)
        bound.append(_BoundAttempt(attempt, row, replay))
    if used_rows != {row.source_json_sha256 for row in validated_rows}:
        raise ValueError("TrialEval authenticated grade export contains unplanned or omitted rows")
    return tuple(bound)


def _validate_retry_ledger(bound: tuple[_BoundAttempt, ...]) -> dict[str, _BoundAttempt]:
    by_id = {row.plan.attempt_id: row for row in bound}
    retry_origins = [row.plan.retry_of_attempt_id for row in bound if row.plan.attempt_role == "retry"]
    if len(retry_origins) != len(set(retry_origins)):
        raise ValueError("TrialEval campaign permits at most one conditional retry per scheduled unit")
    scheduled = [row.plan for row in bound if row.plan.attempt_role != "retry"]
    if len({row.scheduled_unit_id for row in scheduled}) != len(scheduled):
        raise ValueError("TrialEval scheduled unit identifiers must be unique")
    for item in bound:
        row = item.plan
        if row.attempt_role != "retry":
            continue
        original = by_id.get(row.retry_of_attempt_id or "")
        if original is None or original.plan.attempt_role != "primary":
            raise ValueError("TrialEval retry must bind to a primary attempt")
        identity = (
            "scheduled_unit_id",
            "model_id",
            "task_id",
            "target_ids",
            "design_family",
            "reconstruction_family",
            "trial_id",
            "base_world_id",
            "series_ref",
            "cell_ref",
            "effect_state",
            "context",
            "condition_id",
            "procedure_assistance",
            "analysis_specification",
            "target_kind",
            "effect_measure",
            "result_unit",
        )
        if any(getattr(row, field) != getattr(original.plan, field) for field in identity):
            raise ValueError("TrialEval retry factors differ from its original attempt")
        if item.row is not None and original.row is not None and item.row.targets != original.row.targets:
            raise ValueError("TrialEval retry target differs from its original attempt")
    return by_id


def _recovering_retries(bound: tuple[_BoundAttempt, ...]) -> dict[str, _BoundAttempt]:
    """Map each infrastructure-failed primary to its retry when that retry ran without an infrastructure fault.

    A retry is admitted only after its primary's evaluator error, pre-run
    identity exclusion or receipt-bound prospective provider unavailability. Outcomes are checked once either attempt has a graded
    row, so a prospective plan without outcomes still validates.
    """
    by_id = _validate_retry_ledger(bound)
    recovered = {}
    for item in bound:
        if item.plan.attempt_role != "retry":
            continue
        original = by_id[str(item.plan.retry_of_attempt_id)]
        if (original.row is not None or item.row is not None) and (
            original.outcome not in INFRASTRUCTURE_ATTEMPT_OUTCOMES_V1
        ):
            raise ValueError("TrialEval retry is permitted only after an infrastructure failure of its primary")
        if item.row is not None and item.outcome not in INFRASTRUCTURE_ATTEMPT_OUTCOMES_V1:
            recovered[original.plan.attempt_id] = item
    return recovered


def _scientific_primaries(bound: tuple[_BoundAttempt, ...]) -> tuple[_BoundAttempt, ...]:
    """Return each scheduled primary, or the retry that recovered it from an infrastructure fault."""
    recovered = _recovering_retries(bound)
    return tuple(recovered.get(item.plan.attempt_id, item) for item in bound if item.plan.attempt_role == "primary")


def summarize_trial_eval_attempt_ledger_v1(bound: tuple[_BoundAttempt, ...]) -> TrialEvalAttemptLedgerSummaryV1:
    """Summarize all source-bound attempts and recoverable operational failures."""

    recovered = _recovering_retries(bound)
    executed = tuple(item for item in bound if item.plan.attempt_role != "retry" or item.row is not None)
    outcomes = Counter(item.outcome for item in executed)
    unresolved = tuple(
        sorted(
            item.plan.attempt_id
            for item in bound
            if item.plan.attempt_role == "primary"
            and item.outcome in INFRASTRUCTURE_ATTEMPT_OUTCOMES_V1
            and item.plan.attempt_id not in recovered
        )
    )
    return TrialEvalAttemptLedgerSummaryV1(
        scheduled_primary_attempts=sum(item.plan.attempt_role == "primary" for item in bound),
        scheduled_stochastic_duplicates=sum(item.plan.attempt_role == "stochastic_duplicate" for item in bound),
        retry_attempts=sum(item.plan.attempt_role == "retry" for item in executed),
        total_attempts=len(executed),
        recovered_primary_attempt_ids=tuple(sorted(recovered)),
        primary_scientifically_unscored_attempt_ids=unresolved,
        outcomes=TrialEvalAttemptOutcomeCountsV1(
            completed=outcomes["completed"],
            not_started=outcomes["not_started"],
            provider_stop=outcomes["provider_stop"],
            provider_unavailable=outcomes["provider_unavailable"],
            model_noncompletion=outcomes["model_noncompletion"],
            malformed_output=outcomes["malformed_output"],
            replay_failure=outcomes["replay_failure"],
            budget_exhausted=outcomes["budget_exhausted"],
            evaluator_error=outcomes["evaluator_error"],
            pre_run_identity_exclusion=outcomes["pre_run_identity_exclusion"],
        ),
    )


class TrialEvalMetricDefinitionV1(_Model):
    """One homogeneous observable metric and its applicability stratum."""

    metric_id: str = Field(min_length=1)
    domain: MetricDomain
    direction: MetricDirection
    target_kind: Literal["all", "point"]
    effect_measure: str | None = None
    result_unit: str | None = None


class TrialEvalMetricObservationV1(_Model):
    """One canonical metric value derived from an authoritative grade row."""

    attempt_id: str
    model_id: str
    task_id: str
    trial_id: str
    base_world_id: str
    cell_ref: str
    effect_state: EffectState
    context: Context
    metric: TrialEvalMetricDefinitionV1
    value: float


def _binary_status(status: str) -> float | None:
    if status in {"satisfied", "observed"}:
        return 1.0
    if status == "not_satisfied":
        return 0.0
    if status in {"not_evaluable", "not_applicable"}:
        return None
    raise ValueError(f"unknown TrialEval component status: {status!r}")


def _metric(
    metric_id: str,
    domain: MetricDomain,
    direction: MetricDirection,
    row: TrialEvalCampaignRowV1,
    *,
    target_kind: Literal["all", "point"] = "all",
    native_scale: bool = False,
) -> TrialEvalMetricDefinitionV1:
    return TrialEvalMetricDefinitionV1(
        metric_id=metric_id,
        domain=domain,
        direction=direction,
        target_kind=target_kind,
        effect_measure=row.effect_measure if native_scale else None,
        result_unit=row.result_unit if native_scale else None,
    )


def _decision_consequence(row: TrialEvalCampaignRowV1) -> DecisionConsequenceV1 | None:
    target = row.target
    metrics = row.outcome.verification_metrics or {}
    error, width = metrics.get("comparison_signed_error"), metrics.get("interval_width")
    answer = None
    if isinstance(error, float) and isinstance(width, float):
        truth = math.log(target.value) if target.binding.effect_measure == "rate_ratio" else target.value
        estimate = truth + error
        # The decision likelihood uses the estimate and interval width separately;
        # recentering a wide interval on a large estimate can overflow its bounds.
        answer = DecisionReportV1(estimate=estimate, lower=-width / 2, upper=width / 2)
    return trialeval_target_consequence_v1(target, answer)


def trialeval_target_consequence_v1(
    target: TrialEvalTargetV1,
    answer: DecisionReportV1 | None,
    *,
    value: float = 1.0,
    cost: float = 1.0,
) -> DecisionConsequenceV1 | None:
    """Price a graded answer by its reference-posterior regret; ``None`` without a reference.

    Reference-posterior regret is retained in grade records; replay regret is the
    reported headline.

    ``answer`` uses the comparison scale (log scale for rate ratios). ``value``
    and ``cost`` scale the trial's declared economics for the disclosed sensitivity band.

    Parameters
    ----------
    target
        Authenticated grading-key target, including its reference analyses and economics.
    answer
        Reproduced estimate and interval, or None for the prior-only decision.
    value, cost
        Multipliers for the declared confirmation economics.

    Returns
    -------
    DecisionConsequenceV1 or None
        The answer's stop-or-confirm action and expected regret under the posterior;
        None when the target has no consequence reference.
    """
    reference = target.consequence
    if reference is None:
        return None
    economics = economics_v1(target, value=value, cost=cost)

    def comparison(value: float) -> float:
        return math.log(value) if target.binding.effect_measure == "rate_ratio" else value

    credited = tuple(
        DecisionReportV1(estimate=comparison(estimate), lower=comparison(lower), upper=comparison(upper))
        for estimate, lower, upper in reference.credited
    )
    return decision_consequence_v1(
        answer,
        credited,
        truth=comparison(target.value),
        scale=target.reference_sampling_standard_deviation,
        participants=reference.participants,
        favorable_direction=target.binding.favorable_direction,
        economics=economics,
        prior=load_layer_prior(target.binding.estimand_id),
    )


def _row_metrics(item: _BoundAttempt) -> tuple[TrialEvalMetricObservationV1, ...]:
    row = item.row
    if row is None:
        missing_metrics = (
            TrialEvalMetricDefinitionV1(
                metric_id="completion",
                domain="completion",
                direction="higher_is_better",
                target_kind="all",
            ),
            TrialEvalMetricDefinitionV1(
                metric_id="clean_replay",
                domain="clean_replay",
                direction="higher_is_better",
                target_kind="all",
            ),
        )
        return tuple(
            TrialEvalMetricObservationV1(
                attempt_id=item.plan.attempt_id,
                model_id=item.plan.model_id,
                task_id=item.plan.task_id,
                trial_id=item.plan.trial_id,
                base_world_id=item.plan.base_world_id,
                cell_ref=item.plan.cell_ref,
                effect_state=item.plan.effect_state,
                context=item.plan.context,
                metric=metric,
                value=0.0,
            )
            for metric in missing_metrics
        )
    outcome = row.outcome
    point, interval = outcome.point_performance, outcome.interval_performance
    loss = outcome.signal_loss
    values: list[tuple[TrialEvalMetricDefinitionV1, float | None]] = [
        (
            _metric(
                f"{item.plan.target_kind}_signal_loss",
                "numerical_performance",
                "lower_is_better",
                row,
                target_kind=item.plan.target_kind,
            ),
            loss,
        ),
        (_metric("completion", "completion", "higher_is_better", row), _binary_status(outcome.completion.status)),
        (
            _metric("clean_replay", "clean_replay", "higher_is_better", row),
            _binary_status(outcome.reproducibility.status),
        ),
        (
            _metric("question_alignment", "canonical_diagnostic", "higher_is_better", row),
            _binary_status(outcome.question_alignment.status),
        ),
        (
            _metric("evidence_eligibility", "canonical_diagnostic", "higher_is_better", row),
            _binary_status(outcome.evidence_eligibility.status),
        ),
    ]
    if (point := outcome.point_performance) is not None:
        values.append(
            (
                _metric(
                    "primary_point_normalized_absolute_error",
                    "numerical_performance",
                    "lower_is_better",
                    row,
                    target_kind="point",
                ),
                point.normalized_absolute_error,
            )
        )
    if (interval := outcome.interval_performance) is not None:
        values.extend(
            (
                (
                    _metric(
                        "primary_point_interval_coverage", "uncertainty", "higher_is_better", row, target_kind="point"
                    ),
                    float(interval.covers_target),
                ),
                (
                    _metric(
                        "primary_point_interval_width",
                        "uncertainty",
                        "lower_is_better",
                        row,
                        target_kind="point",
                        native_scale=True,
                    ),
                    interval.native_width,
                ),
                (
                    _metric(
                        "primary_point_native_interval_score",
                        "uncertainty",
                        "lower_is_better",
                        row,
                        target_kind="point",
                        native_scale=True,
                    ),
                    interval.native_interval_score,
                ),
                (
                    _metric(
                        "primary_point_proper_interval_score",
                        "uncertainty",
                        "lower_is_better",
                        row,
                        target_kind="point",
                    ),
                    (
                        interval.proper_interval_score / point.reference_sampling_standard_deviation
                        if point is not None and interval.proper_interval_score is not None
                        else None
                    ),
                ),
            )
        )
    if item.scores is not None:
        declared = item.scores["declared"]
        values.extend(
            (
                (
                    _metric(
                        "decision_regret_musd", "decision_consequence", "lower_is_better", row, target_kind="point"
                    ),
                    declared.replay_regret_musd,
                ),
                *(
                    (
                        _metric(
                            f"decision_regret_musd_{name}",
                            "decision_consequence",
                            "lower_is_better",
                            row,
                            target_kind="point",
                        ),
                        score.replay_regret_musd,
                    )
                    for name, score in item.scores.items()
                    if name != "declared"
                ),
                (
                    _metric(
                        "consequential_error", "decision_consequence", "lower_is_better", row, target_kind="point"
                    ),
                    declared.consequential_share,
                ),
                (
                    _metric("replay_answered_share", "decision_consequence", "higher_is_better", row),
                    declared.answered_share,
                ),
                (
                    _metric(
                        "material_decision_error_rate",
                        "decision_consequence",
                        "lower_is_better",
                        row,
                        target_kind="point",
                    ),
                    declared.material_decision_error_rate,
                ),
            )
        )
    if row.plan_execution is not None:
        # Execution of a given plan, reported apart from the loss and the decision.
        values.append(
            (
                _metric("plan_execution_accuracy", "plan_execution", "higher_is_better", row),
                float(row.plan_execution.accurate),
            )
        )
    return tuple(
        TrialEvalMetricObservationV1(
            attempt_id=item.plan.attempt_id,
            model_id=item.plan.model_id,
            task_id=item.plan.task_id,
            trial_id=item.plan.trial_id,
            base_world_id=item.plan.base_world_id,
            cell_ref=item.plan.cell_ref,
            effect_state=item.plan.effect_state,
            context=item.plan.context,
            metric=metric,
            value=value,
        )
        for metric, value in values
        if value is not None
    )


class TrialEvalOracleProvenanceV1(TrialEvalOraclePrecisionV1):
    """Target-bound reference precision and independently reconstructed diagnostics.

    Native metrics retain the target's unit and are not pooled across targets.
    These private diagnostics do not change the task loss or scientific weights.
    """

    attempt_id: str
    model_id: str
    task_id: str
    trial_id: str
    base_world_id: str
    target: TrialEvalTargetV1
    metrics: dict[str, float | bool | str | None] | None = None
    signal_loss: float | None = Field(default=None, ge=0)


class TrialEvalMetricSummaryV1(_Model):
    """Equal-cell summary for one model, C view, and homogeneous metric."""

    model_id: str
    context: Context
    metric: TrialEvalMetricDefinitionV1
    status: EstimateStatus
    scheduled_attempts: int = Field(ge=1)
    observed_attempts: int = Field(ge=0)
    raw_attempt_mean: float | None
    equal_cell_effect_mean: float | None
    interval_low: float | None
    interval_high: float | None
    standard_error: float | None = Field(default=None, ge=0)
    confidence_level: float = Field(gt=0.0, lt=1.0)
    bootstrap_resamples: int = Field(ge=100)
    bootstrap_seed: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_estimate_state(self) -> TrialEvalMetricSummaryV1:
        values = (self.raw_attempt_mean, self.equal_cell_effect_mean)
        if (self.interval_low is None) != (self.interval_high is None):
            raise ValueError("TrialEval uncertainty interval requires both endpoints")
        if self.observed_attempts > self.scheduled_attempts:
            raise ValueError("TrialEval metric observations exceed the scheduled denominator")
        if (self.status == "estimated") != all(value is not None for value in values):
            raise ValueError("TrialEval metric estimate fields differ from its status")
        if self.status != "estimated" and self.standard_error is not None:
            raise ValueError("TrialEval unavailable metric cannot carry a standard error")
        if self.status == "estimated" and self.observed_attempts != self.scheduled_attempts:
            raise ValueError("TrialEval estimated metric does not have complete scheduled observations")
        return self


class TrialEvalCContrastEstimateV1(_Model):
    """One base-world-paired C contrast for a homogeneous metric."""

    contrast_id: str
    minuend_context: Context
    subtrahend_context: Context
    estimate: float
    interval_low: float | None
    interval_high: float | None
    standard_error: float | None = Field(default=None, ge=0)
    raw_p_value: float | None = Field(ge=0.0, le=1.0)
    holm_adjusted_p_value: float | None = Field(ge=0.0, le=1.0)


class TrialEvalCContrastFamilyV1(_Model):
    """Five C contrasts and Holm adjustment within one homogeneous metric family."""

    model_id: str
    metric: TrialEvalMetricDefinitionV1
    status: EstimateStatus
    scheduled_cell_effect_world_pairs: int = Field(ge=1)
    observed_complete_context_pairs: int = Field(ge=0)
    confidence_level: float = Field(gt=0.0, lt=1.0)
    bootstrap_resamples: int = Field(ge=100)
    bootstrap_seed: int = Field(ge=0)
    multiplicity_method: Literal["holm_across_five_C_contrasts_within_homogeneous_metric"] = (
        "holm_across_five_C_contrasts_within_homogeneous_metric"
    )
    estimates: tuple[TrialEvalCContrastEstimateV1, ...]

    @model_validator(mode="after")
    def validate_estimate_state(self) -> TrialEvalCContrastFamilyV1:
        if self.observed_complete_context_pairs > self.scheduled_cell_effect_world_pairs:
            raise ValueError("TrialEval C-pair observations exceed the scheduled denominator")
        if (self.status == "estimated") != (len(self.estimates) == len(C_CONTRASTS_V1)):
            raise ValueError("TrialEval C-contrast estimates differ from their family status")
        if self.status == "estimated" and (
            self.observed_complete_context_pairs != self.scheduled_cell_effect_world_pairs
        ):
            raise ValueError("TrialEval estimated C family does not have complete scheduled pairs")
        return self


class TrialEvalDuplicateVariabilityV1(_Model):
    """Within-agent absolute difference from planned stochastic duplicate runs."""

    model_id: str
    metric: TrialEvalMetricDefinitionV1
    scheduled_pairs: int = Field(ge=1)
    observed_pairs: int = Field(ge=0)
    mean_absolute_difference: float | None
    disagreement_rate: float | None

    @model_validator(mode="after")
    def validate_pair_state(self) -> TrialEvalDuplicateVariabilityV1:
        if self.observed_pairs > self.scheduled_pairs:
            raise ValueError("TrialEval duplicate observations exceed the scheduled denominator")
        values = (self.mean_absolute_difference, self.disagreement_rate)
        if (self.observed_pairs > 0) != all(value is not None for value in values):
            raise ValueError("TrialEval duplicate estimates differ from the observed denominator")
        return self


class TrialEvalBetweenModelContrastV1(_Model):
    """Identical-world paired difference for one prespecified model pair and metric."""

    scorer: CurrentScorer

    minuend_model_id: str
    subtrahend_model_id: str
    context: Context | Literal["all"]
    metric: TrialEvalMetricDefinitionV1
    status: EstimateStatus
    scheduled_pairs: int = Field(ge=1)
    observed_pairs: int = Field(ge=0)
    common_world_estimate: float | None = None
    common_cell_refs: tuple[str, ...] = ()
    estimate: float | None
    interval_low: float | None
    interval_high: float | None
    standard_error: float | None = Field(default=None, ge=0)
    minimal_detectable_difference: float | None = Field(
        default=None,
        ge=0,
        description="Unadjusted normal-approximation MDD at two-sided alpha 0.05 and 80% power.",
    )
    p_value: float | None = Field(default=None, ge=0, le=1)
    adjusted_p_value: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_estimate_state(self) -> TrialEvalBetweenModelContrastV1:
        if self.observed_pairs > self.scheduled_pairs:
            raise ValueError("TrialEval between-model observations exceed the scheduled denominator")
        values = (self.estimate,)
        if (self.interval_low is None) != (self.interval_high is None):
            raise ValueError("TrialEval contrast interval requires both endpoints")
        if (self.status == "estimated") != all(value is not None for value in values):
            raise ValueError("TrialEval between-model estimate fields differ from its status")
        if self.estimate is None and (
            self.standard_error is not None or self.minimal_detectable_difference is not None
        ):
            raise ValueError("TrialEval unavailable model contrast cannot carry an error or detectable difference")
        if (self.standard_error is None) != (self.minimal_detectable_difference is None):
            raise ValueError("TrialEval model contrast reports its standard error and detectable difference together")
        if self.status == "estimated" and self.observed_pairs != self.scheduled_pairs:
            raise ValueError("TrialEval estimated model contrast does not have complete scheduled pairs")
        return self


def holm_adjusted_p_values_v1(p_values: tuple[float, ...]) -> tuple[float, ...]:
    """Return monotone Holm adjusted p-values in original order."""

    if not p_values or any(not 0.0 <= value <= 1.0 for value in p_values):
        raise ValueError("Holm adjustment requires a nonempty family of valid p-values")
    return tuple(float(value) for value in multipletests(p_values, method="holm")[1])


def _cell_mean(values: dict[tuple[str, EffectState, str], float]) -> float:
    groups: dict[str, list[float]] = defaultdict(list)
    for (cell, _, _), value in values.items():
        groups[cell].append(value)
    return mean(mean(group) for group in groups.values())


class _Bootstrap(NamedTuple):
    """Equal-cell estimate with its percentile interval, bootstrap p-value and standard error."""

    estimate: float
    low: float | None
    high: float | None
    p_value: float | None
    standard_error: float | None


def minimal_detectable_difference_v1(standard_error: float, *, alpha: float, power: float = 0.8) -> float:
    """Return the unadjusted normal-approximation detectable difference.

    This scale does not guarantee the requested power under the bootstrap test
    or the Holm correction across model pairs.

    Parameters
    ----------
    standard_error
        Standard error of the paired difference.
    alpha
        Two-sided significance level.
    power
        Probability of detecting a true difference of the returned size.

    Returns
    -------
    float
        ``(z_{1 - alpha / 2} + z_{power}) * standard_error``.
    """
    if standard_error < 0 or not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("minimal detectable difference requires a nonnegative SE, alpha and power in (0, 1)")
    return float((norm.isf(alpha / 2) + norm.ppf(power)) * standard_error)


def _strata(
    cell_series: dict[str, str], by_cell_world: dict[tuple[str, str], float]
) -> tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]:
    """Return the resampling strata: each one's cells and the base worlds resampled jointly across them.

    Scenarios of one clinical question that share base worlds form one stratum, so their paired views
    move together. Scenarios using disjoint worlds resample their worlds separately.
    """
    strata = []
    for series in sorted(set(cell_series.values())):
        worlds = {
            cell: frozenset(world for owner, world in by_cell_world if owner == cell)
            for cell, owner in sorted(cell_series.items())
            if owner == series
        }
        if any(not cell_worlds for cell_worlds in worlds.values()):
            raise ValueError("TrialEval metric has no base world for a selected cell")
        if len(set(worlds.values())) == 1:
            strata.append((tuple(worlds), tuple(sorted(next(iter(worlds.values()))))))
        elif sum(map(len, worlds.values())) == len(frozenset().union(*worlds.values())):
            strata.extend(((cell,), tuple(sorted(cell_worlds))) for cell, cell_worlds in worlds.items())
        else:
            raise ValueError("TrialEval metric base worlds differ across cells within a series")
    return tuple(strata)


def _bootstrap(
    values: dict[tuple[str, EffectState, str], float],
    *,
    cells: tuple[TrialEvalCampaignCellV1, ...],
    resamples: int,
    confidence_level: float,
    seed: int,
    support: tuple[float | None, float | None] = (None, None),
) -> _Bootstrap:
    """Bootstrap the equal-scenario mean over base worlds, grouped by clinical question or scenario.

    Each stratum's resampled contribution is rescaled about its observed value by ``sqrt(n / (n - 1))``
    (``n`` its worlds), so its bootstrap variance is the unbiased ``s**2 / n`` rather than the plug-in
    ``(n - 1) / n`` fraction of it; with four worlds per group the plug-in intervals are about 13% too
    narrow. A stratum with one world gives no interval. The rescaling can carry a draw past the metric's
    ``support`` (a rate below zero); reported endpoints are clipped to it, the standard error is not.
    """
    if resamples < 100 or not 0 < confidence_level < 1:
        raise ValueError("TrialEval bootstrap configuration is invalid")
    cell_series = {row.cell_ref: row.series_ref for row in cells}
    estimate = _cell_mean(values)
    rng = np.random.default_rng(seed)
    draws = np.zeros(resamples)
    # Resample in bounded units so finite mean losses do not overflow their sum.
    magnitude = max(abs(value) for value in values.values()) or 1.0
    cell_sizes = Counter(cell for cell, _, _ in values)
    by_cell_world = {(cell, world): value for (cell, _, world), value in values.items()}
    # The effect prior is continuous, so base worlds are exchangeable within a clinical question.
    for stratum_cells, worlds in _strata(cell_series, by_cell_world):
        if len(worlds) < 2:
            return _Bootstrap(estimate, None, None, None, None)
        sampled = rng.integers(0, len(worlds), size=(resamples, len(worlds)))
        contribution = np.zeros(resamples)
        observed = 0.0
        for cell in stratum_cells:
            if not all((cell, world) in by_cell_world for world in worlds):
                raise ValueError("TrialEval metric base worlds differ across cells within a series")
            vector = np.asarray([by_cell_world[cell, world] / magnitude for world in worlds])
            weight = len(worlds) / cell_sizes[cell] / len(cell_sizes)
            contribution += vector[sampled].mean(axis=1) * weight
            observed += float(vector.mean()) * weight
        draws += observed + math.sqrt(len(worlds) / (len(worlds) - 1)) * (contribution - observed)
    tail = (1 - confidence_level) / 2
    low, high = (float(value) * magnitude for value in np.quantile(draws, [tail, 1 - tail]))
    if support[0] is not None:
        low, high = max(low, support[0]), max(high, support[0])
    if support[1] is not None:
        low, high = min(low, support[1]), min(high, support[1])
    centered = np.abs(draws - estimate / magnitude)
    p_value = (1 + float(np.sum(centered >= abs(estimate / magnitude)))) / (resamples + 1)
    return _Bootstrap(estimate, low, high, p_value, float(np.std(draws, ddof=1) * magnitude))


_RATE_METRICS = frozenset(
    {
        "completion",
        "clean_replay",
        "question_alignment",
        "evidence_eligibility",
        "primary_point_interval_coverage",
        "consequential_error",
        "material_decision_error_rate",
        "replay_answered_share",
        "plan_execution_accuracy",
    }
)
_NONNEGATIVE_METRICS = frozenset(
    {
        "primary_point_normalized_absolute_error",
        "primary_point_interval_width",
        "primary_point_native_interval_score",
        "primary_point_proper_interval_score",
    }
)
# Replay regret (decision_regret_musd*) is signed: an answer can beat the scripted reference at the truth.


def _metric_support(metric: TrialEvalMetricDefinitionV1) -> tuple[float | None, float | None]:
    """Return the logical range of a metric's mean, fixed by the metric's definition, never by its sample."""
    if metric.metric_id in _RATE_METRICS:
        return 0.0, 1.0
    if metric.metric_id in _NONNEGATIVE_METRICS or metric.metric_id.endswith("_signal_loss"):
        return 0.0, None
    return None, None


def _validate_schedule(
    bound: tuple[_BoundAttempt, ...],
    *,
    plan: TrialEvalCampaignPlanV1,
    inventory: TrialEvalCampaignInventoryV1,
) -> None:
    cells = {row.cell_ref: row.series_ref for row in plan.cells}
    scheduled = [item.plan for item in bound if item.plan.attempt_role != "retry"]
    scientific_primaries = _scientific_primaries(bound)
    models = sorted({row.model_id for row in scheduled})
    if len(models) != inventory.model_count:
        raise ValueError("TrialEval schedule model count differs from its inventory")
    for model in models:
        primary = [row for row in scheduled if row.model_id == model and row.attempt_role == "primary"]
        duplicate = [row for row in scheduled if row.model_id == model and row.attempt_role == "stochastic_duplicate"]
        primary_worlds: dict[str, set[str]] = {}
        world_cells: dict[tuple[str, str], set[str]] = {}
        for series in sorted(set(cells.values())):
            series_cells = {cell for cell, owner in cells.items() if owner == series}
            worlds = {row.base_world_id for row in primary if row.series_ref == series}
            if not worlds:
                raise UnsupportedTrialEvalPanelError("Selected TrialEval tasks omit a required series")
            if inventory.base_worlds_per_series is not None and len(worlds) != inventory.base_worlds_per_series:
                raise ValueError("TrialEval primary schedule has the wrong independent-world count")
            primary_worlds[series] = worlds
            cell_worlds = {
                cell: {row.base_world_id for row in primary if row.cell_ref == cell} for cell in series_cells
            }
            # The release fixes whether scenarios of one clinical question share worlds or use disjoint worlds.
            # Dropping views cannot change this allocation.
            nested = series in plan.nested_series_refs
            if nested and not all(cell_worlds.values()):
                raise UnsupportedTrialEvalPanelError("Selected TrialEval tasks omit a cell of a nested series")
            if nested and sum(map(len, cell_worlds.values())) != len(worlds):
                raise ValueError("TrialEval nested series presents a base world in more than one cell")
            for world in worlds:
                world_cells[series, world] = (
                    {cell for cell in series_cells if world in cell_worlds[cell]} if nested else series_cells
                )
            expected = {
                (cell, world, context)
                for world in worlds
                for cell in world_cells[series, world]
                for context in plan.contexts
            }
            observed = [(row.cell_ref, row.base_world_id, row.context) for row in primary if row.series_ref == series]
            if len(observed) != len(expected) or set(observed) != expected:
                if len(observed) == len(set(observed)) and set(observed) < expected:
                    raise UnsupportedTrialEvalPanelError(
                        "Selected TrialEval tasks omit linked cell/world/context views"
                    )
                raise ValueError("TrialEval primary schedule does not exactly cross cell, world, and C")
            for world in worlds:
                siblings = [
                    item
                    for item in bound
                    if item.plan.model_id == model
                    and item.plan.attempt_role == "primary"
                    and item.plan.series_ref == series
                    and item.plan.base_world_id == world
                ]
                if len({item.plan.effect_state for item in siblings}) != 1:
                    raise ValueError("TrialEval world siblings disagree on their effect label")
                if len({item.plan.request_replicate_id for item in siblings}) != 1:
                    raise ValueError("TrialEval primary world siblings do not share one request replicate")
                if len({item.plan.task_id for item in siblings}) != len(siblings):
                    raise ValueError("TrialEval C/A siblings must be distinct isolated tasks")
                scientific_siblings = [
                    item
                    for item in scientific_primaries
                    if item.plan.model_id == model
                    and item.plan.series_ref == series
                    and item.plan.base_world_id == world
                ]
                for cell in world_cells[series, world]:
                    planned_targets = {
                        (
                            item.plan.target_ids,
                            item.plan.target_kind,
                            item.plan.effect_measure,
                            item.plan.result_unit,
                        )
                        for item in siblings
                        if item.plan.cell_ref == cell
                    }
                    if len(planned_targets) != 1:
                        raise ValueError("TrialEval matched C siblings differ in scheduled target applicability")
                    cell_targets = {
                        tuple(target.model_dump_json() for target in item.row.targets)
                        for item in scientific_siblings
                        if item.plan.cell_ref == cell and item.row is not None
                    }
                    if len(cell_targets) > 1:
                        raise ValueError("TrialEval matched C siblings do not share one oracle target")
        groups: dict[tuple[str, str, str], list[TrialEvalCampaignAttemptV1]] = defaultdict(list)
        for row in duplicate:
            if row.base_world_id not in primary_worlds.get(row.series_ref, set()):
                raise ValueError("TrialEval stochastic duplicate is not tied to a primary world")
            groups[(row.series_ref, row.base_world_id, row.request_replicate_id)].append(row)
        for series in sorted(set(cells.values())):
            selected = [rows for (owner, _, _), rows in groups.items() if owner == series]
            if (
                inventory.stochastic_duplicates_per_series is not None
                and len(selected) != inventory.stochastic_duplicates_per_series
            ):
                raise ValueError("TrialEval stochastic-duplicate allocation differs from plan")
            for rows in selected:
                members = {
                    (cell, context) for cell in world_cells[series, rows[0].base_world_id] for context in plan.contexts
                }
                if {(row.cell_ref, row.context) for row in rows} != members or len(rows) != len(members):
                    raise ValueError("TrialEval stochastic duplicate is incomplete across cells and C")
            if any(
                rows[0].request_replicate_id
                in {
                    candidate.request_replicate_id
                    for candidate in primary
                    if candidate.series_ref == series and candidate.base_world_id == rows[0].base_world_id
                }
                for rows in selected
            ):
                raise ValueError("TrialEval stochastic duplicate reuses a primary request replicate")
    declared_models = set(models)
    if any(set(pair) - declared_models for pair in plan.model_comparisons):
        raise ValueError("TrialEval model comparison names a model outside the schedule")
    scheduled_primaries = [item for item in bound if item.plan.attempt_role == "primary"]
    scheduled_by_model = {
        model: {
            (
                item.plan.cell_ref,
                item.plan.effect_state,
                item.plan.base_world_id,
                item.plan.context,
            ): item
            for item in scheduled_primaries
            if item.plan.model_id == model
        }
        for model in models
    }
    scientific_by_model = {
        model: {
            (
                item.plan.cell_ref,
                item.plan.effect_state,
                item.plan.base_world_id,
                item.plan.context,
            ): item
            for item in scientific_primaries
            if item.plan.model_id == model
        }
        for model in models
    }
    for minuend, subtrahend in plan.model_comparisons:
        left = scheduled_by_model[minuend]
        right = scheduled_by_model[subtrahend]
        if left.keys() != right.keys():
            raise ValueError("TrialEval model comparison lacks identical base-world allocation")
        for key in left:
            a = left[key]
            b = right[key]
            packet_a = (
                a.plan.task_id,
                a.plan.trial_id,
                a.plan.condition_id,
                a.plan.procedure_assistance,
                a.plan.analysis_specification,
                a.plan.target_kind,
                a.plan.target_ids,
                a.plan.design_family,
                a.plan.reconstruction_family,
                a.plan.effect_measure,
                a.plan.result_unit,
            )
            packet_b = (
                b.plan.task_id,
                b.plan.trial_id,
                b.plan.condition_id,
                b.plan.procedure_assistance,
                b.plan.analysis_specification,
                b.plan.target_kind,
                b.plan.target_ids,
                b.plan.design_family,
                b.plan.reconstruction_family,
                b.plan.effect_measure,
                b.plan.result_unit,
            )
            if packet_a != packet_b:
                raise ValueError("TrialEval model comparison differs in task, trial, or condition identity")
            scientific_a = scientific_by_model[minuend][key]
            scientific_b = scientific_by_model[subtrahend][key]
            if (
                scientific_a.row is not None
                and scientific_b.row is not None
                and scientific_a.row.targets != scientific_b.row.targets
            ):
                raise ValueError("TrialEval model comparison differs in its evaluator target")


def _observations(bound: tuple[_BoundAttempt, ...]) -> tuple[TrialEvalMetricObservationV1, ...]:
    return tuple(value for item in _scientific_primaries(bound) for value in _row_metrics(item))


def _metric_definition_order(metric: TrialEvalMetricDefinitionV1) -> tuple[bool, str]:
    # Append the additive metric without shifting existing metric indices or bootstrap seeds.
    return metric.metric_id == "material_decision_error_rate", metric.model_dump_json()


def _definitions(observations: tuple[TrialEvalMetricObservationV1, ...]) -> tuple[TrialEvalMetricDefinitionV1, ...]:
    return tuple(sorted({row.metric for row in observations}, key=_metric_definition_order))


def _applicable_attempts(
    primaries: tuple[_BoundAttempt, ...],
    metric: TrialEvalMetricDefinitionV1,
    *,
    model: str,
    context: Context | Literal["all"],
) -> tuple[_BoundAttempt, ...]:
    selected = []
    for item in primaries:
        if item.plan.model_id != model or (context != "all" and item.plan.context != context):
            continue
        if metric.target_kind != "all" and item.plan.target_kind != metric.target_kind:
            continue
        if metric.domain == "plan_execution" and item.plan.context not in PLAN_CONTEXTS_V1:
            continue
        if metric.effect_measure is not None and item.plan.effect_measure != metric.effect_measure:
            continue
        if metric.result_unit is not None and item.plan.result_unit != metric.result_unit:
            continue
        selected.append(item)
    return tuple(selected)


def _metric_index(observations: tuple[TrialEvalMetricObservationV1, ...]) -> dict[tuple[str, str], float]:
    return {(row.attempt_id, row.metric.model_dump_json()): row.value for row in observations}


def _summaries(
    bound: tuple[_BoundAttempt, ...], plan: TrialEvalCampaignPlanV1, *, resamples: int, confidence: float, seed: int
) -> tuple[TrialEvalMetricSummaryV1, ...]:
    primaries = _scientific_primaries(bound)
    observations = _observations(bound)
    definitions = _definitions(observations)
    index = _metric_index(observations)
    models = sorted({item.plan.model_id for item in primaries})
    output = []
    for model_index, model in enumerate(models):
        for metric_index, metric in enumerate(definitions):
            for context_index, context in enumerate(plan.contexts):
                applicable = _applicable_attempts(primaries, metric, model=model, context=context)
                if not applicable:
                    continue
                observed = [item for item in applicable if (item.plan.attempt_id, metric.model_dump_json()) in index]
                complete = len(observed) == len(applicable)
                values = {
                    (item.plan.cell_ref, item.plan.effect_state, item.plan.base_world_id): index[
                        (item.plan.attempt_id, metric.model_dump_json())
                    ]
                    for item in observed
                }
                estimate: float | None
                low: float | None
                high: float | None
                raw: float | None
                standard_error: float | None = None
                if complete:
                    estimate, low, high, _, standard_error = _bootstrap(
                        values,
                        cells=tuple(
                            cell for cell in plan.cells if cell.cell_ref in {item.plan.cell_ref for item in applicable}
                        ),
                        resamples=resamples,
                        confidence_level=confidence,
                        seed=seed + model_index * 1000 + metric_index * 10 + context_index,
                        support=_metric_support(metric),
                    )
                    raw = mean(values.values())
                else:
                    estimate = low = high = raw = None
                output.append(
                    TrialEvalMetricSummaryV1(
                        model_id=model,
                        context=context,
                        metric=metric,
                        status="estimated" if complete else "unavailable_due_missing_metric",
                        scheduled_attempts=len(applicable),
                        observed_attempts=len(observed),
                        raw_attempt_mean=raw,
                        equal_cell_effect_mean=estimate,
                        interval_low=low,
                        interval_high=high,
                        standard_error=standard_error,
                        confidence_level=confidence,
                        bootstrap_resamples=resamples,
                        bootstrap_seed=seed + model_index * 1000 + metric_index * 10 + context_index,
                    )
                )
    return tuple(output)


def _c_families(
    bound: tuple[_BoundAttempt, ...], plan: TrialEvalCampaignPlanV1, *, resamples: int, confidence: float, seed: int
) -> tuple[TrialEvalCContrastFamilyV1, ...]:
    if plan.contexts != CONTEXTS_V1:
        return ()
    primaries = _scientific_primaries(bound)
    observations = _observations(bound)
    definitions = _definitions(observations)
    index = _metric_index(observations)
    models = sorted({item.plan.model_id for item in primaries})
    output = []
    for model_index, model in enumerate(models):
        for metric_index, metric in enumerate(definitions):
            if metric.domain == "plan_execution":
                continue  # defined only where a plan is given, so it has no five-context contrast
            applicable = tuple(
                item
                for context in CONTEXTS_V1
                for item in _applicable_attempts(primaries, metric, model=model, context=context)
            )
            if not applicable:
                continue
            expected_keys = {
                (item.plan.cell_ref, item.plan.effect_state, item.plan.base_world_id) for item in applicable
            }
            by_key: dict[tuple[str, EffectState, str], dict[Context, float]] = defaultdict(dict)
            for item in applicable:
                key = (item.plan.cell_ref, item.plan.effect_state, item.plan.base_world_id)
                value = index.get((item.plan.attempt_id, metric.model_dump_json()))
                if value is not None:
                    by_key[key][item.plan.context] = value
            complete = set(by_key) == expected_keys and all(
                set(values) == set(CONTEXTS_V1) for values in by_key.values()
            )
            if not complete:
                output.append(
                    TrialEvalCContrastFamilyV1(
                        model_id=model,
                        metric=metric,
                        status="unavailable_due_missing_metric",
                        scheduled_cell_effect_world_pairs=len(expected_keys),
                        observed_complete_context_pairs=sum(
                            set(values) == set(CONTEXTS_V1) for values in by_key.values()
                        ),
                        confidence_level=confidence,
                        bootstrap_resamples=resamples,
                        bootstrap_seed=seed + model_index * 1000 + metric_index * 10,
                        estimates=(),
                    )
                )
                continue
            provisional = []
            for contrast_index, (contrast_id, minuend, subtrahend) in enumerate(C_CONTRASTS_V1):
                paired = {key: rows[minuend] - rows[subtrahend] for key, rows in by_key.items()}
                result = _bootstrap(
                    paired,
                    cells=tuple(cell for cell in plan.cells if cell.cell_ref in {key[0] for key in paired}),
                    resamples=resamples,
                    confidence_level=confidence,
                    seed=seed + model_index * 1000 + metric_index * 10 + contrast_index,
                )
                provisional.append((contrast_id, minuend, subtrahend, result))
            p_values = tuple(row[3].p_value for row in provisional if row[3].p_value is not None)
            adjusted = holm_adjusted_p_values_v1(p_values) if len(p_values) == 5 else (None,) * 5
            output.append(
                TrialEvalCContrastFamilyV1(
                    model_id=model,
                    metric=metric,
                    status="estimated",
                    scheduled_cell_effect_world_pairs=len(expected_keys),
                    observed_complete_context_pairs=len(by_key),
                    confidence_level=confidence,
                    bootstrap_resamples=resamples,
                    bootstrap_seed=seed + model_index * 1000 + metric_index * 10,
                    estimates=tuple(
                        TrialEvalCContrastEstimateV1(
                            contrast_id=row[0],
                            minuend_context=row[1],
                            subtrahend_context=row[2],
                            estimate=row[3].estimate,
                            interval_low=row[3].low,
                            interval_high=row[3].high,
                            standard_error=row[3].standard_error,
                            raw_p_value=row[3].p_value,
                            holm_adjusted_p_value=adjusted[index_],
                        )
                        for index_, row in enumerate(provisional)
                    ),
                )
            )
    return tuple(output)


def _duplicate_variability(bound: tuple[_BoundAttempt, ...]) -> tuple[TrialEvalDuplicateVariabilityV1, ...]:
    primaries = _scientific_primaries(bound)
    primary_by_key = {
        (
            item.plan.model_id,
            item.plan.cell_ref,
            item.plan.effect_state,
            item.plan.base_world_id,
            item.plan.context,
        ): item
        for item in primaries
    }
    duplicates = [item for item in bound if item.plan.attempt_role == "stochastic_duplicate"]
    all_observations = tuple(value for item in (*primaries, *duplicates) for value in _row_metrics(item))
    index = _metric_index(all_observations)
    definitions = _definitions(all_observations)
    output = []
    for model in sorted({item.plan.model_id for item in duplicates}):
        for metric in definitions:
            pairs = []
            scheduled = 0
            for duplicate in duplicates:
                if duplicate.plan.model_id != model:
                    continue
                primary = primary_by_key[
                    (
                        model,
                        duplicate.plan.cell_ref,
                        duplicate.plan.effect_state,
                        duplicate.plan.base_world_id,
                        duplicate.plan.context,
                    )
                ]
                if metric.target_kind != "all" and primary.plan.target_kind != metric.target_kind:
                    continue
                if metric.effect_measure is not None and primary.plan.effect_measure != metric.effect_measure:
                    continue
                if metric.result_unit is not None and primary.plan.result_unit != metric.result_unit:
                    continue
                if metric.domain == "plan_execution" and duplicate.plan.context not in PLAN_CONTEXTS_V1:
                    continue
                scheduled += 1
                if duplicate.row is None or primary.row is None:
                    continue
                left = index.get((duplicate.plan.attempt_id, metric.model_dump_json()))
                right = index.get((primary.plan.attempt_id, metric.model_dump_json()))
                if left is not None and right is not None:
                    pairs.append(abs(left - right))
            if scheduled:
                output.append(
                    TrialEvalDuplicateVariabilityV1(
                        model_id=model,
                        metric=metric,
                        scheduled_pairs=scheduled,
                        observed_pairs=len(pairs),
                        mean_absolute_difference=mean(pairs) if pairs else None,
                        disagreement_rate=float(np.mean([value > 0 for value in pairs])) if pairs else None,
                    )
                )
    return tuple(output)


def _between_models(
    bound: tuple[_BoundAttempt, ...], plan: TrialEvalCampaignPlanV1, *, resamples: int, confidence: float, seed: int
) -> tuple[TrialEvalBetweenModelContrastV1, ...]:
    primaries = _scientific_primaries(bound)
    observations = _observations(bound)
    definitions = tuple(
        sorted(
            set(_definitions(observations))
            | {
                TrialEvalMetricDefinitionV1(
                    metric_id=f"{item.plan.target_kind}_signal_loss",
                    domain="numerical_performance",
                    direction="lower_is_better",
                    target_kind=item.plan.target_kind,
                )
                for item in primaries
            },
            key=_metric_definition_order,
        )
    )
    index = _metric_index(observations)
    by_model_key = {
        (
            item.plan.model_id,
            item.plan.cell_ref,
            item.plan.effect_state,
            item.plan.base_world_id,
            item.plan.context,
        ): item
        for item in primaries
    }
    output = []
    for pair_index, (minuend, subtrahend) in enumerate(plan.model_comparisons):
        for metric_index, metric in enumerate(definitions):
            contexts: tuple[Context | Literal["all"], ...] = (
                (*plan.contexts, "all") if metric.metric_id.endswith("_signal_loss") else plan.contexts
            )
            for context_index, context in enumerate(contexts):
                left = _applicable_attempts(primaries, metric, model=minuend, context=context)
                right = _applicable_attempts(primaries, metric, model=subtrahend, context=context)
                left_keys = {(item.plan.cell_ref, item.plan.effect_state, item.plan.base_world_id) for item in left}
                right_keys = {(item.plan.cell_ref, item.plan.effect_state, item.plan.base_world_id) for item in right}
                if left_keys != right_keys:
                    raise ValueError("TrialEval model comparison lacks identical base-world allocation")
                if not left_keys:
                    continue
                differences = {}
                for key in left_keys:
                    paired_contexts = plan.contexts if context == "all" else (context,)
                    deltas = []
                    for paired_context in paired_contexts:
                        a = by_model_key[(minuend, *key, paired_context)]
                        b = by_model_key[(subtrahend, *key, paired_context)]
                        av = index.get((a.plan.attempt_id, metric.model_dump_json()))
                        bv = index.get((b.plan.attempt_id, metric.model_dump_json()))
                        if av is not None and bv is not None:
                            deltas.append(av - bv)
                    if len(deltas) == len(paired_contexts):
                        differences[key] = mean(deltas)
                complete = len(differences) == len(left_keys)
                estimate: float | None
                low: float | None
                high: float | None
                standard_error: float | None = None
                if complete:
                    estimate, low, high, p_value, standard_error = _bootstrap(
                        differences,
                        cells=tuple(cell for cell in plan.cells if cell.cell_ref in {key[0] for key in differences}),
                        resamples=resamples,
                        confidence_level=confidence,
                        seed=seed + pair_index * 1000 + metric_index * 10 + context_index,
                    )
                else:
                    estimate = low = high = p_value = None
                output.append(
                    TrialEvalBetweenModelContrastV1(
                        scorer=current_scorer(),
                        minuend_model_id=minuend,
                        subtrahend_model_id=subtrahend,
                        context=context,
                        metric=metric,
                        status="estimated" if complete else "unavailable_due_missing_metric",
                        scheduled_pairs=len(left_keys),
                        observed_pairs=len(differences),
                        common_world_estimate=_cell_mean(differences) if differences else None,
                        common_cell_refs=tuple(sorted({key[0] for key in differences})),
                        estimate=estimate,
                        interval_low=low,
                        interval_high=high,
                        standard_error=standard_error,
                        minimal_detectable_difference=(
                            None
                            if standard_error is None
                            else minimal_detectable_difference_v1(standard_error, alpha=0.05)
                        ),
                        p_value=p_value,
                    )
                )
    families: dict[tuple[str, str], list[int]] = defaultdict(list)
    for position, row in enumerate(output):
        families[row.metric.model_dump_json(), row.context].append(position)
    for indices in families.values():
        adjusted = holm_adjusted_p_values_v1(
            tuple(
                row.p_value if row.p_value is not None else 1.0 for row in (output[position] for position in indices)
            )
        )
        for position, value in zip(indices, adjusted, strict=True):
            output[position] = output[position].model_copy(
                update={"adjusted_p_value": value if output[position].p_value is not None else None}
            )
    return tuple(output)


class TrialEvalAssumptionContrastV1(_Model):
    """A prospectively declared, matched-world A contrast averaged over C."""

    model_id: str
    minuend_cell: str
    subtrahend_cell: str
    target_kind: Literal["point"]
    scheduled_world_pairs: int = Field(ge=1)
    observed_world_pairs: int | None = Field(
        default=None,
        ge=0,
        exclude_if=lambda value: value is None,
        description="Complete observed pairs when fewer than scheduled; omitted for a complete contrast.",
    )
    estimate: float | None
    interval_low: float | None
    interval_high: float | None
    standard_error: float | None = Field(default=None, ge=0)
    raw_p_value: float | None = Field(ge=0, le=1)
    holm_adjusted_p_value: float | None = Field(ge=0, le=1)


def _assumption_contrasts(
    bound: tuple[_BoundAttempt, ...], plan: TrialEvalCampaignPlanV1, *, resamples: int, confidence: float, seed: int
) -> tuple[TrialEvalAssumptionContrastV1, ...]:
    primaries = _scientific_primaries(bound)
    index = _metric_index(_observations(bound))
    output: list[TrialEvalAssumptionContrastV1] = []
    for model in sorted({item.plan.model_id for item in primaries}):
        contrasts = []
        for left_cell, right_cell in plan.assumption_comparisons:
            left = {
                (item.plan.effect_state, item.plan.base_world_id, item.plan.context): item
                for item in primaries
                if item.plan.model_id == model and item.plan.cell_ref == left_cell
            }
            right = {
                (item.plan.effect_state, item.plan.base_world_id, item.plan.context): item
                for item in primaries
                if item.plan.model_id == model and item.plan.cell_ref == right_cell
            }
            if not left or left.keys() != right.keys():
                raise ValueError("TrialEval A contrast requires exactly matched worlds and contexts")
            kinds = {item.plan.target_kind for item in (*left.values(), *right.values())}
            if len(kinds) != 1:
                raise ValueError("TrialEval A contrast cannot pool point and identified-set recovery")
            kind = next(iter(kinds))
            metric = TrialEvalMetricDefinitionV1(
                metric_id=f"{kind}_signal_loss",
                domain="numerical_performance",
                direction="lower_is_better",
                target_kind=kind,
            )
            grouped: dict[tuple[str, EffectState, str], list[float]] = defaultdict(list)
            for (effect, world, context), a in left.items():
                b = right[effect, world, context]
                if (
                    a.row is not None
                    and b.row is not None
                    and (
                        len(a.row.targets) != len(b.row.targets)
                        or any(
                            not _estimand_core_matches(left.binding, right.binding)
                            for left, right in zip(a.row.targets, b.row.targets, strict=True)
                        )
                    )
                ):
                    raise ValueError("TrialEval A contrast differs in its clinical target")
                if (a.plan.effect_measure, a.plan.result_unit) != (b.plan.effect_measure, b.plan.result_unit):
                    raise ValueError("TrialEval A contrast differs in its endpoint scale")
                av = index.get((a.plan.attempt_id, metric.model_dump_json()))
                bv = index.get((b.plan.attempt_id, metric.model_dump_json()))
                if av is not None and bv is not None:
                    grouped[left_cell, effect, world].append(av - bv)
            scheduled_worlds = len({world for _, world, _ in left})
            values = {key: mean(rows) for key, rows in grouped.items() if len(rows) == len(plan.contexts)}
            result = (
                _bootstrap(
                    values,
                    cells=tuple(cell for cell in plan.cells if cell.cell_ref == left_cell),
                    resamples=resamples,
                    confidence_level=confidence,
                    seed=seed,
                )
                if len(values) == scheduled_worlds
                else None
            )
            contrasts.append(
                TrialEvalAssumptionContrastV1(
                    model_id=model,
                    minuend_cell=left_cell,
                    subtrahend_cell=right_cell,
                    target_kind=kind,
                    scheduled_world_pairs=scheduled_worlds,
                    observed_world_pairs=len(values) if len(values) < scheduled_worlds else None,
                    estimate=None if result is None else result.estimate,
                    interval_low=None if result is None else result.low,
                    interval_high=None if result is None else result.high,
                    standard_error=None if result is None else result.standard_error,
                    raw_p_value=None if result is None else result.p_value,
                    holm_adjusted_p_value=None,
                )
            )
        for kind in ("point",):
            family = [row for row in contrasts if row.target_kind == kind]
            p_values = tuple(row.raw_p_value if row.raw_p_value is not None else 1.0 for row in family)
            adjusted = holm_adjusted_p_values_v1(p_values) if p_values else ()
            output.extend(
                row.model_copy(update={"holm_adjusted_p_value": adjusted[i] if row.raw_p_value is not None else None})
                for i, row in enumerate(family)
            )
    return tuple(output)


class TrialEvalLossRankingV1(_Model):
    """Equal-cell mean loss and its strictly monotone bounded companion.

    ``bounded_recovery`` and ``display_recovery`` are ``1 / (1 + mean_loss)``.
    Both are withheld with mean loss for incomplete panels or insufficient independent-world support. Recovery intervals
    reverse the transformed loss limits; its standard error uses the delta method.
    """

    model_id: str
    unavailable_reason: str | None = Field(default=None, exclude_if=lambda value: value is None)
    target_kind: Literal["point"]
    scheduled_attempts: int = Field(ge=1)
    scored_attempts: int = Field(ge=0)
    scheduled_targets: int = Field(ge=1)
    scored_targets: int = Field(ge=0)
    unscored_targets: int = Field(ge=0)
    target_failure_counts: dict[str, int]
    observed_mean_loss: float | None = Field(default=None, ge=0)
    scored_worlds: int = Field(default=0, ge=0)
    scored_cell_refs: tuple[str, ...] = ()
    independent_worlds: int = Field(ge=1)
    cell_refs: tuple[str, ...]
    mean_loss: float | None = Field(ge=0)
    display_recovery: float | None = Field(ge=0, le=1)
    interval_low: float | None
    interval_high: float | None
    standard_error: float | None = Field(default=None, ge=0)
    bounded_recovery: float | None = Field(ge=0, le=1)
    bounded_recovery_low: float | None = Field(default=None, ge=0, le=1)
    bounded_recovery_high: float | None = Field(default=None, ge=0, le=1)
    bounded_recovery_standard_error: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_completion(self) -> Self:
        """Withhold scientific rankings for incomplete scheduled panels."""
        if (
            any(count < 1 for count in self.target_failure_counts.values())
            or sum(self.target_failure_counts.values()) != self.unscored_targets
        ):
            raise ValueError("Target failure counts do not reconcile")
        if self.scored_targets + self.unscored_targets != self.scheduled_targets:
            raise ValueError("Target completion counts do not reconcile")
        if self.scored_attempts > self.scheduled_attempts:
            raise ValueError("Scored attempts exceed the scheduled panel")
        if self.scored_worlds > min(self.independent_worlds, self.scored_attempts):
            raise ValueError("Scored worlds exceed the scheduled or scored denominator")
        if not set(self.scored_cell_refs) <= set(self.cell_refs):
            raise ValueError("Scored cells fall outside the scheduled panel")
        if self.observed_mean_loss is not None and (not self.scored_worlds or not self.scored_cell_refs):
            raise ValueError("Observed loss requires scored worlds and cells")
        complete = self.scored_attempts == self.scheduled_attempts
        if (complete and self.unavailable_reason is None) != (self.mean_loss is not None):
            raise ValueError("Mean loss requires every scheduled attempt")
        expected = bounded_recovery_summary(self.mean_loss, self.interval_low, self.interval_high, self.standard_error)
        if (
            self.display_recovery != expected[0]
            or (
                self.bounded_recovery,
                self.bounded_recovery_low,
                self.bounded_recovery_high,
                self.bounded_recovery_standard_error,
            )
            != expected
        ):
            raise ValueError("Bounded recovery must transform the aggregate loss and its uncertainty")
        if not complete and (self.interval_low is not None or self.interval_high is not None):
            raise ValueError("Incomplete panels cannot have scientific comparison intervals")
        return self


def _loss_rankings(
    bound: tuple[_BoundAttempt, ...],
    plan: TrialEvalCampaignPlanV1,
    *,
    resamples: int,
    confidence: float,
    seed: int,
    cell_id: str | None = None,
    descriptive: bool = False,
) -> tuple[TrialEvalLossRankingV1, ...]:
    """Summarize the panel or one planned cell; cell summaries are descriptive."""
    if cell_id is not None:
        bound = tuple(item for item in bound if item.plan.cell_ref == cell_id)
    primaries = _scientific_primaries(bound)
    index = _metric_index(_observations(bound))
    rankings = []
    for model in sorted({item.plan.model_id for item in primaries}):
        for kind in sorted({item.plan.target_kind for item in primaries if item.plan.model_id == model}):
            metric = TrialEvalMetricDefinitionV1(
                metric_id=f"{kind}_signal_loss",
                domain="numerical_performance",
                direction="lower_is_better",
                target_kind=kind,
            )
            applicable = _applicable_attempts(primaries, metric, model=model, context="all")
            grouped: dict[tuple[str, EffectState, str], list[float]] = defaultdict(list)
            scheduled_targets = sum(len(item.plan.target_ids) for item in applicable)
            scored_targets = 0
            target_failures: dict[str, int] = defaultdict(int)
            for item in applicable:
                if item.row is None:
                    target_failures["not_started"] += len(item.plan.target_ids)
                    continue
                outcomes = item.row.outcome.target_outcomes or (item.row.outcome,)
                for outcome in outcomes:
                    if outcome.signal_loss is not None:
                        scored_targets += 1
                        continue
                    components = (
                        outcome.components
                        if isinstance(outcome, VerifiedTrialEvalTargetOutcomeV1)
                        else (
                            outcome.completion,
                            outcome.question_alignment,
                            outcome.evidence_eligibility,
                            outcome.reproducibility,
                            outcome.estimand_performance,
                            outcome.uncertainty,
                        )
                    )
                    reason = next(
                        (
                            component.failure_origin
                            for component in components
                            if component.failure_origin is not None
                            and component.status in {"not_satisfied", "not_evaluable"}
                        ),
                        "unscored",
                    )
                    target_failures[reason] += 1
            scored = 0
            for item in applicable:
                value = index.get((item.plan.attempt_id, metric.model_dump_json()))
                key = (item.plan.cell_ref, item.plan.effect_state, item.plan.base_world_id)
                if value is not None:
                    scored += 1
                    grouped[key].append(value)
            cells = tuple(sorted({item.plan.cell_ref for item in applicable}))
            values = {key: mean(rows) for key, rows in grouped.items()}
            estimate = low = high = standard_error = None
            if scored == len(applicable) and (cell_id is not None or descriptive):
                estimate = _cell_mean(values)
            elif scored == len(applicable):
                estimate, low, high, _, standard_error = _bootstrap(
                    values,
                    cells=tuple(cell for cell in plan.cells if cell.cell_ref in cells),
                    resamples=resamples,
                    confidence_level=confidence,
                    seed=seed,
                    support=(0.0, None),
                )
            recovery_estimate, recovery_low, recovery_high, recovery_standard_error = bounded_recovery_summary(
                estimate, low, high, standard_error
            )
            rankings.append(
                TrialEvalLossRankingV1(
                    model_id=model,
                    target_kind=kind,
                    scheduled_attempts=len(applicable),
                    scored_attempts=scored,
                    scheduled_targets=scheduled_targets,
                    scored_targets=scored_targets,
                    unscored_targets=scheduled_targets - scored_targets,
                    target_failure_counts=dict(sorted(target_failures.items())),
                    observed_mean_loss=_cell_mean(values) if values else None,
                    scored_worlds=len({world for _, _, world in values}),
                    scored_cell_refs=tuple(sorted({cell for cell, _, _ in values})),
                    independent_worlds=len({item.plan.base_world_id for item in applicable}),
                    cell_refs=cells,
                    mean_loss=estimate,
                    display_recovery=recovery_estimate,
                    interval_low=low,
                    interval_high=high,
                    standard_error=standard_error,
                    bounded_recovery=recovery_estimate,
                    bounded_recovery_low=recovery_low,
                    bounded_recovery_high=recovery_high,
                    bounded_recovery_standard_error=recovery_standard_error,
                )
            )
    return tuple(
        sorted(
            rankings,
            key=lambda row: (
                row.target_kind,
                row.mean_loss is None,
                row.mean_loss if row.mean_loss is not None else math.inf,
                row.model_id,
            ),
        )
    )


class TrialEvalDifficultySummaryV1(TrialEvalLossRankingV1):
    """Descriptive task recovery within a planned difficulty family."""

    axis: Literal["design_family", "reconstruction_family"]
    family: str = Field(min_length=1)


def _difficulty_summaries(
    bound: tuple[_BoundAttempt, ...], plan: TrialEvalCampaignPlanV1
) -> tuple[TrialEvalDifficultySummaryV1, ...]:
    """Keep absent outcomes in family denominators without resampling or reranking."""
    summaries = []
    for axis in ("design_family", "reconstruction_family"):
        families = sorted({getattr(item.plan, axis) for item in bound} - {None})
        for family in families:
            selected = tuple(item for item in bound if getattr(item.plan, axis) == family)
            for summary in _loss_rankings(selected, plan, resamples=100, confidence=0.95, seed=0, descriptive=True):
                summaries.append(TrialEvalDifficultySummaryV1(axis=axis, family=family, **summary.model_dump()))
    return tuple(summaries)


class TrialEvalTargetSummaryV1(_Model):
    """Descriptive recovery for one target within one planned clinical cell.

    Counts retain missing primaries. Means average observed primary views within
    each world, then worlds equally within the cell. No cross-target ranking is made.
    """

    model_id: str
    cell_ref: str
    target_id: str
    effect_measure: str | None
    result_unit: str | None
    scheduled_attempts: int = Field(ge=1)
    scored_attempts: int = Field(ge=0)
    observed_mean_loss: float | None = Field(ge=0)
    metric_means: dict[str, float]
    metric_observations: dict[str, int]

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        """Keep every displayed mean attached to its observed denominator."""
        if self.scored_attempts > self.scheduled_attempts:
            raise ValueError("Target observations exceed their scheduled denominator")
        if (self.observed_mean_loss is not None) != bool(self.scored_attempts):
            raise ValueError("Observed target loss requires scored attempts")
        if self.metric_means.keys() != self.metric_observations.keys() or any(
            count < 1 or count > self.scored_attempts for count in self.metric_observations.values()
        ):
            raise ValueError("Target metric means require their scored observation counts")
        if (self.effect_measure is None) != (self.result_unit is None):
            raise ValueError("Target metric scale requires both measure and unit")
        return self


def _target_summaries(
    bound: tuple[_BoundAttempt, ...], oracle: tuple[TrialEvalOracleProvenanceV1, ...]
) -> tuple[TrialEvalTargetSummaryV1, ...]:
    records = {(row.attempt_id, row.target.binding.estimand_id): row for row in oracle}
    groups: dict[tuple[str, str, str], list[_BoundAttempt]] = defaultdict(list)
    for item in _scientific_primaries(bound):
        for target_id in item.plan.target_ids:
            groups[item.plan.model_id, item.plan.cell_ref, target_id].append(item)
    summaries = []
    for (model, cell, target_id), items in sorted(groups.items()):
        scales = set()
        values: dict[str, dict[tuple[str, EffectState, str], list[float]]] = defaultdict(lambda: defaultdict(list))
        counts: dict[str, int] = defaultdict(int)
        for item in items:
            record = records.get((item.plan.attempt_id, target_id))
            if record is None:
                continue
            scales.add((record.target.binding.effect_measure, record.target.binding.result_unit))
            if record.signal_loss is None:
                continue
            key = (cell, item.plan.effect_state, item.plan.base_world_id)
            values["loss"][key].append(record.signal_loss)
            counts["loss"] += 1
            for metric in (
                "native_absolute_error",
                "normalized_absolute_error",
                "native_interval_width",
                "native_interval_score",
                "interval_covers_oracle",
            ):
                value = None if record.metrics is None else record.metrics.get(metric)
                if value is not None:
                    if not isinstance(value, (float, int)):
                        raise ValueError("Target diagnostic must be numerical")
                    values[metric][key].append(float(value))
                    counts[metric] += 1
        if len(scales) > 1:
            raise ValueError("Target summaries cannot mix effect measures or units")
        measure, unit = next(iter(scales)) if scales else (None, None)
        means = {
            metric: _cell_mean({key: mean(samples) for key, samples in worlds.items()})
            for metric, worlds in values.items()
        }
        summaries.append(
            TrialEvalTargetSummaryV1(
                model_id=model,
                cell_ref=cell,
                target_id=target_id,
                effect_measure=measure,
                result_unit=unit,
                scheduled_attempts=len(items),
                scored_attempts=counts.get("loss", 0),
                observed_mean_loss=means.pop("loss", None),
                metric_means=means,
                metric_observations={key: count for key, count in counts.items() if key != "loss"},
            )
        )
    return tuple(summaries)


class TrialEvalCampaignAnalysisV1(_Model):
    """Release-bound TrialEval campaign inference and provenance."""

    scorer: CurrentScorer

    schema_id: Literal["trialagentbench.trialeval_campaign_analysis/v4"] = (
        "trialagentbench.trialeval_campaign_analysis/v4"
    )
    release_id: str
    release_manifest_sha256: str
    export_manifest_checksum: str | None = Field(default=None, exclude_if=lambda value: value is None)
    prospective_analysis_plan_sha256: str
    campaign_plan_checksum: str
    source_archive_sha256: str | None = Field(default=None, exclude_if=lambda value: value is None)
    inventory: TrialEvalCampaignInventoryV1
    ledger: TrialEvalAttemptLedgerSummaryV1
    scientific_analysis_status: AnalysisStatus
    unavailable_reason: str | None = Field(default=None, exclude_if=lambda value: value is None)
    oracle_provenance: tuple[TrialEvalOracleProvenanceV1, ...]
    metric_summaries: tuple[TrialEvalMetricSummaryV1, ...]
    c_contrast_families: tuple[TrialEvalCContrastFamilyV1, ...]
    duplicate_variability: tuple[TrialEvalDuplicateVariabilityV1, ...]
    between_model_contrasts: tuple[TrialEvalBetweenModelContrastV1, ...]
    loss_rankings: tuple[TrialEvalLossRankingV1, ...] = ()
    cell_loss_summaries: tuple[TrialEvalLossRankingV1, ...] = ()
    difficulty_summaries: tuple[TrialEvalDifficultySummaryV1, ...] = ()
    target_summaries: tuple[TrialEvalTargetSummaryV1, ...] = ()
    assumption_contrasts: tuple[TrialEvalAssumptionContrastV1, ...] = ()
    confidence_level: float = Field(gt=0.0, lt=1.0)
    bootstrap_resamples: int = Field(ge=100)
    bootstrap_seed: int = Field(ge=0)
    scope: Literal["separate_recovery_tracks_with_declared_paired_C_A_and_model_comparisons"] = (
        "separate_recovery_tracks_with_declared_paired_C_A_and_model_comparisons"
    )
    checksum: str = ""

    @model_validator(mode="after")
    def bind_analysis(self) -> TrialEvalCampaignAnalysisV1:
        inferential_sections = (
            self.metric_summaries,
            self.c_contrast_families,
            self.duplicate_variability,
            self.between_model_contrasts,
            self.cell_loss_summaries,
            self.difficulty_summaries,
            self.target_summaries,
            self.assumption_contrasts,
        )
        if self.scientific_analysis_status == "complete":
            if not self.metric_summaries or (self.inventory.context_count == 5 and not self.c_contrast_families):
                raise ValueError("complete TrialEval campaign analysis lacks prespecified inference")
        elif any(inferential_sections):
            raise ValueError("unscored TrialEval campaign analysis must not report confirmatory inference")
        if self.scientific_analysis_status != "complete" and any(
            row.mean_loss is not None or row.interval_low is not None or row.standard_error is not None
            for row in self.loss_rankings
        ):
            raise ValueError("Unavailable TrialEval analysis cannot carry a ranking estimate or uncertainty")
        if self.inventory.context_count < 5 and self.c_contrast_families:
            raise ValueError("TrialEval five-context contrasts require the complete presentation panel")
        if any(row.confidence_level != self.confidence_level for row in self.metric_summaries):
            raise ValueError("TrialEval metric summaries differ from the campaign confidence level")
        if any(row.confidence_level != self.confidence_level for row in self.c_contrast_families):
            raise ValueError("TrialEval C families differ from the campaign confidence level")
        if any(row.bootstrap_resamples != self.bootstrap_resamples for row in self.metric_summaries):
            raise ValueError("TrialEval metric summaries differ from the campaign bootstrap count")
        if any(row.bootstrap_resamples != self.bootstrap_resamples for row in self.c_contrast_families):
            raise ValueError("TrialEval C families differ from the campaign bootstrap count")
        expected = canonical_sha256(self.model_dump(mode="json", exclude={"checksum"}))
        if self.checksum and self.checksum != expected:
            raise ValueError("TrialEval campaign analysis checksum mismatch")
        object.__setattr__(self, "checksum", expected)
        return self


def _observed_plan_inventory(plan: TrialEvalCampaignPlanV1) -> TrialEvalCampaignInventoryV1:
    """Count the actual frozen schedule without substituting default allocations."""
    models = {row.model_id for row in plan.attempts if row.attempt_role != "retry"}
    scheduled = tuple(row for row in plan.attempts if row.attempt_role != "retry")
    primary_counts = Counter(row.model_id for row in scheduled if row.attempt_role == "primary")
    duplicate_counts = Counter(row.model_id for row in scheduled if row.attempt_role == "stochastic_duplicate")
    if not models or not primary_counts:
        raise ValueError("TrialEval campaign requires scheduled primary attempts")
    if len({cell.cell_ref for cell in plan.cells}) != len(plan.cells):
        raise ValueError("TrialEval campaign cells must be unique")
    if len(set(primary_counts.values())) != 1 or len({duplicate_counts[model] for model in models}) != 1:
        raise ValueError("TrialEval compared models have different scheduled attempt counts")
    first_model = sorted(models)[0]
    independent = {
        row.base_world_id for row in scheduled if row.model_id == first_model and row.attempt_role == "primary"
    }
    return TrialEvalCampaignInventoryV1(
        series_count=len({cell.series_ref for cell in plan.cells}),
        context_count=len(plan.contexts),
        regime_cell_count=len(plan.cells),
        base_worlds_per_series=plan.base_worlds_per_series,
        independent_base_worlds=len(independent),
        primary_attempts_per_model=primary_counts[first_model],
        stochastic_duplicates_per_series=plan.stochastic_duplicates_per_series,
        stochastic_duplicate_attempts_per_model=duplicate_counts[first_model],
        model_count=len(models),
        primary_attempts=sum(primary_counts.values()),
        stochastic_duplicate_attempts=sum(duplicate_counts.values()),
        total_scheduled_attempts=len(scheduled),
    )


def validate_trial_eval_campaign_plan_v1(plan: TrialEvalCampaignPlanV1) -> TrialEvalCampaignInventoryV1:
    """Check full crossed support before any outcomes or paid provider requests."""
    inventory = _observed_plan_inventory(plan)
    _validate_schedule(tuple(_BoundAttempt(row, None) for row in plan.attempts), plan=plan, inventory=inventory)
    return inventory


def _analyse_trial_eval_campaign_source_v1(
    source: TrialEvalAuthenticatedGradeSourceV1,
    *,
    plan: TrialEvalCampaignPlanV1,
    expected_campaign_plan_sha256: str,
    bootstrap_resamples: int = 2_000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 65_009,
    replay: TrialEvalReplayEvidenceV1 | None = None,
) -> TrialEvalCampaignAnalysisV1:
    """Analyse one authenticated campaign under its frozen plan."""

    if (
        len(expected_campaign_plan_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_campaign_plan_sha256)
        or len(plan.checksum) != 64
        or any(character not in "0123456789abcdef" for character in plan.checksum)
        or plan.checksum != expected_campaign_plan_sha256
    ):
        raise ValueError("TrialEval campaign plan differs from its prospective external identity")
    bound = _bind_attempts(plan, source, replay)
    inventory = _observed_plan_inventory(plan)
    ledger = summarize_trial_eval_attempt_ledger_v1(bound)
    if ledger.scheduled_primary_attempts != inventory.primary_attempts:
        raise ValueError("TrialEval primary ledger differs from the derived inventory")
    if ledger.scheduled_stochastic_duplicates != inventory.stochastic_duplicate_attempts:
        raise ValueError("TrialEval duplicate ledger differs from the derived inventory")
    _validate_schedule(bound, plan=plan, inventory=inventory)
    primaries = _scientific_primaries(bound)
    oracle = tuple(
        TrialEvalOracleProvenanceV1(
            attempt_id=item.plan.attempt_id,
            model_id=item.plan.model_id,
            task_id=item.plan.task_id,
            trial_id=item.plan.trial_id,
            base_world_id=item.plan.base_world_id,
            target=target,
            metrics=(
                diagnostic.metrics
                if isinstance(diagnostic, VerifiedTrialEvalTargetOutcomeV1)
                else diagnostic.verification_metrics
            ),
            signal_loss=diagnostic.signal_loss,
            **precision.model_dump(),
        )
        for item in primaries
        if item.row is not None
        for target, precision, diagnostic in zip(
            item.row.targets,
            (
                TrialEvalOraclePrecisionV1(
                    oracle_worlds=item.row.oracle_worlds,
                    oracle_monte_carlo_standard_error=item.row.oracle_monte_carlo_standard_error,
                    maximum_oracle_monte_carlo_standard_error=item.row.maximum_oracle_monte_carlo_standard_error,
                ),
                *item.row.additional_oracle_precision,
            ),
            cast(
                tuple[VerifiedTrialEvalTargetOutcomeV1 | _CampaignOutcomeV1, ...],
                item.row.outcome.target_outcomes or (item.row.outcome,),
            ),
            strict=True,
        )
    )
    policy = CampaignAnalysisV1()
    panel_reason = policy.panel_unavailable_reason(
        tuple(
            len({item.plan.base_world_id for item in primaries if item.plan.cell_ref == cell.cell_ref})
            for cell in plan.cells
        ),
        required_strata=len(plan.cells),
    )
    reason = "unscored_primary_attempts" if ledger.primary_scientifically_unscored_attempt_ids else panel_reason
    unavailable = reason is not None
    rankings = (
        ()
        if ledger.primary_scientifically_unscored_attempt_ids
        else _loss_rankings(
            bound,
            plan,
            resamples=bootstrap_resamples,
            confidence=confidence_level,
            seed=bootstrap_seed,
            descriptive=unavailable,
        )
    )
    if unavailable:
        rankings = tuple(
            row.model_copy(
                update={
                    "mean_loss": None,
                    "bounded_recovery": None,
                    "display_recovery": None,
                    "unavailable_reason": reason,
                }
            )
            for row in rankings
        )
    return TrialEvalCampaignAnalysisV1(
        scorer=current_scorer(),
        release_id=plan.release_id,
        release_manifest_sha256=plan.release_manifest_sha256,
        export_manifest_checksum=source.export_manifest_checksum,
        prospective_analysis_plan_sha256=plan.prospective_analysis_plan_sha256,
        campaign_plan_checksum=plan.checksum,
        source_archive_sha256=source.archive_sha256,
        inventory=inventory,
        ledger=ledger,
        scientific_analysis_status=(
            "unavailable_due_unscored_primary_attempts"
            if ledger.primary_scientifically_unscored_attempt_ids
            else "unavailable_due_insufficient_independent_worlds" if panel_reason else "complete"
        ),
        unavailable_reason=reason,
        oracle_provenance=oracle,
        target_summaries=() if unavailable else _target_summaries(bound, oracle),
        difficulty_summaries=() if unavailable else _difficulty_summaries(bound, plan),
        assumption_contrasts=(
            ()
            if unavailable
            else _assumption_contrasts(
                bound, plan, resamples=bootstrap_resamples, confidence=confidence_level, seed=bootstrap_seed
            )
        ),
        loss_rankings=rankings,
        cell_loss_summaries=(
            ()
            if unavailable
            else tuple(
                summary
                for cell in plan.cells
                for summary in _loss_rankings(
                    bound,
                    plan,
                    resamples=bootstrap_resamples,
                    confidence=confidence_level,
                    seed=bootstrap_seed,
                    cell_id=cell.cell_ref,
                )
            )
        ),
        metric_summaries=(
            ()
            if unavailable
            else _summaries(
                bound, plan, resamples=bootstrap_resamples, confidence=confidence_level, seed=bootstrap_seed
            )
        ),
        c_contrast_families=(
            ()
            if unavailable
            else _c_families(
                bound, plan, resamples=bootstrap_resamples, confidence=confidence_level, seed=bootstrap_seed
            )
        ),
        duplicate_variability=(() if unavailable else _duplicate_variability(bound)),
        between_model_contrasts=(
            ()
            if unavailable
            else _between_models(
                bound, plan, resamples=bootstrap_resamples, confidence=confidence_level, seed=bootstrap_seed
            )
        ),
        confidence_level=confidence_level,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )


def _answer_rows(
    primaries: tuple[_BoundAttempt, ...], replay: TrialEvalReplayEvidenceV1 | None
) -> tuple[TrialEvalAnswerRowV1, ...]:
    """Return each scored primary answer's rows at every economics band point, then each banked task's anchors.

    Replays run once; only their valuation repeats per band point, with the consequential bar scaled by the
    value multiplier. Rows at the declared economics (1, 1) are the headline. Fixed-policy anchors need the
    operator's full bank; the primary-reference public bank leaves these anchors to the operator.
    """
    rows: list[TrialEvalAnswerRowV1] = []
    targets: dict[str, tuple[TrialEvalTargetV1, dict[str, str]]] = {}
    for item in primaries:
        row = item.row
        if row is None or item.scores is None:
            continue
        identity = {"task_id": row.task_id, "cell": row.cell_ref, "context": row.context_tier, "law_id": row.trial_id}
        consequence = _decision_consequence(row)
        cause = failure_cause_v1(row)
        rows.extend(
            TrialEvalAnswerRowV1(
                model=row.model_id,
                **identity,
                **item.scores[name].model_dump(),
                failure_cause=cause,
                value_multiplier=value,
                cost_multiplier=cost,
                reference_posterior_regret_musd=(
                    None if consequence is None or name != "declared" else consequence.regret_musd
                ),
            )
            for name, (value, cost) in ECONOMICS_BAND.items()
        )
        targets[row.task_id] = (row.target, identity)
    anchors: list[TrialEvalAnswerRowV1] = []
    for task_id, (target, identity) in sorted(targets.items()):
        if replay is None or replay.bank.manifest.variant == "public":
            break
        records = replay.bank.records.get(task_id)
        if not records:
            continue
        for value, cost in ECONOMICS_BAND.values():
            scores = anchor_scores_v1(target, records, value=value, cost=cost)
            anchors.extend(
                TrialEvalAnswerRowV1(
                    model=f"policy:{name}",
                    **identity,
                    **scores[name].model_dump(),
                    failure_cause="no_answer" if scores[name].scoring == "failure" else None,
                    value_multiplier=value,
                    cost_multiplier=cost,
                )
                for name in ANCHOR_POLICIES
            )
    return (*sorted(rows, key=lambda row: (row.model, row.task_id)), *anchors)


def trial_eval_answer_rows_v1(
    archive_path: str = "",
    *,
    expected_release_id: str,
    expected_release_manifest_sha256: str,
    expected_archive_sha256: str | None = None,
    plan: TrialEvalCampaignPlanV1,
    grade_roots: tuple[Path, ...] = (),
    replay: TrialEvalReplayEvidenceV1 | None,
) -> tuple[TrialEvalAnswerRowV1, ...]:
    """Score every graded primary answer of a campaign, then each banked task's fixed-policy anchors.

    The rows of ``trialeval_answers.csv``. Unlike the paired campaign analysis, this needs no complete
    cell/world/context panel (a C5-only campaign is scored) and keeps every observed answer, including those
    of a model whose other tasks were budget-censored; unrun attempts have no row.

    Parameters
    ----------
    archive_path
        Portable result export, read when ``grade_roots`` is empty.
    expected_release_id, expected_release_manifest_sha256
        The graded release.
    expected_archive_sha256
        Trusted SHA-256 of the result export.
    plan
        The campaign's combined TrialEval plan.
    grade_roots
        The graded native runs, read in place of an export.
    replay
        The replay bank and stored outcomes; None scores every answer at the truth on its original dataset.

    Returns
    -------
    tuple of TrialEvalAnswerRowV1
        Answer rows at every economics band point by model and task, then anchors by task.

    Raises
    ------
    ValueError
        If the grades differ from the plan or release, or a reproduced answer to a banked task has no stored
        replay outcomes.
    """
    source = load_trial_eval_campaign_grade_source_v1(
        archive_path,
        expected_release_id=expected_release_id,
        expected_release_manifest_sha256=expected_release_manifest_sha256,
        expected_archive_sha256=expected_archive_sha256,
        grade_roots=grade_roots,
    )
    return _answer_rows(_scientific_primaries(_bind_attempts(plan, source, replay)), replay)


def _paired_trial_eval_panel(
    plan: TrialEvalCampaignPlanV1,
    source: TrialEvalAuthenticatedGradeSourceV1,
    censored_attempt_ids: frozenset[str],
) -> tuple[TrialEvalCampaignPlanV1, TrialEvalAuthenticatedGradeSourceV1]:
    """Drop every task any model lost to budget censoring, keeping the paired panel.

    Censored attempts must be scheduled and ungraded. Primary/retry censoring
    excludes the task for every model. An unrun stochastic duplicate remains
    in the variability denominator and cannot remove its completed primary.
    """
    planned = {row.attempt_id: row for row in plan.attempts}
    if not censored_attempt_ids <= planned.keys():
        raise ValueError("Budget-censored TrialEval attempts are not scheduled")
    censored = {_attempt_identity(planned[attempt]) for attempt in censored_attempt_ids}
    if any(_attempt_identity(row) in censored for row in source.rows):
        raise ValueError("A budget-censored TrialEval attempt has a graded result")
    excluded = {
        planned[attempt].task_id
        for attempt in censored_attempt_ids
        if planned[attempt].attempt_role != "stochastic_duplicate"
    }
    if not excluded:
        return plan, source
    attempts = tuple(row for row in plan.attempts if row.task_id not in excluded)
    if not attempts:
        raise UnsupportedTrialEvalPanelError("No scheduled TrialEval task was completed by every model")
    paired = TrialEvalCampaignPlanV1.model_validate(
        {
            **plan.model_dump(mode="python", exclude={"checksum", "attempts"}),
            "attempts": attempts,
            # The retained allocation can differ by clinical question after censoring.
            # Its exact attempts, rather than the original uniform counts, govern it.
            "base_worlds_per_series": None,
            "stochastic_duplicates_per_series": None,
        }
    )
    rows = tuple(row for row in source.rows if row.task_id not in excluded)
    return paired, source.model_copy(update={"rows": rows})


def analyse_trial_eval_campaign_v1(
    archive_path: str,
    *,
    expected_release_id: str,
    expected_release_manifest_sha256: str,
    expected_archive_sha256: str | None,
    plan: TrialEvalCampaignPlanV1,
    expected_campaign_plan_sha256: str,
    bootstrap_resamples: int = 2_000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 65_009,
    budget_censored_attempt_ids: frozenset[str] = frozenset(),
    replay: TrialEvalReplayEvidenceV1 | None = None,
    grade_roots: tuple[Path, ...] = (),
) -> TrialEvalCampaignAnalysisV1:
    """Authenticate and analyse one evaluator-custody result export.

    Budget-censored attempts restrict the analysis to the paired panel of
    tasks every scheduled attempt completed; the reported plan checksum then
    identifies that derived paired plan. ``replay`` holds the replay bank and the replay
    outcomes of every reproduced answer to a banked task; answers are then scored by replay at
    the true effect, and tasks without bank records (or every task, without ``replay``) at the
    truth on their original dataset.
    """

    if budget_censored_attempt_ids:
        if plan.checksum != expected_campaign_plan_sha256:
            raise ValueError("TrialEval campaign plan differs from its prospective external identity")
        censored_tasks = {
            row.task_id
            for row in plan.attempts
            if row.attempt_id in budget_censored_attempt_ids and row.attempt_role != "stochastic_duplicate"
        }
        if all(row.task_id in censored_tasks for row in plan.attempts):
            # A wholly censored panel has no graded rows to authenticate or analyse.
            raise UnsupportedTrialEvalPanelError("No scheduled TrialEval task was completed by every model")
    source = load_trial_eval_campaign_grade_source_v1(
        archive_path,
        expected_release_id=expected_release_id,
        expected_release_manifest_sha256=expected_release_manifest_sha256,
        expected_archive_sha256=expected_archive_sha256,
        grade_roots=grade_roots,
    )
    if budget_censored_attempt_ids:
        plan, source = _paired_trial_eval_panel(plan, source, budget_censored_attempt_ids)
        expected_campaign_plan_sha256 = plan.checksum
    return _analyse_trial_eval_campaign_source_v1(
        source,
        plan=plan,
        expected_campaign_plan_sha256=expected_campaign_plan_sha256,
        bootstrap_resamples=bootstrap_resamples,
        confidence_level=confidence_level,
        bootstrap_seed=bootstrap_seed,
        replay=replay,
    )


def _recovery_text(ranking: TrialEvalLossRankingV1) -> str:
    if ranking.bounded_recovery is None:
        return "unavailable"
    if ranking.bounded_recovery_low is None or ranking.bounded_recovery_high is None:
        return f"{ranking.bounded_recovery:.4g}"
    return (
        f"{ranking.bounded_recovery:.4g} "
        f"({ranking.bounded_recovery_low:.4g} to {ranking.bounded_recovery_high:.4g})"
    )


def render_trial_eval_campaign_analysis_v1(report: TrialEvalCampaignAnalysisV1) -> str:
    """Render the typed campaign analysis without changing its claim scope."""

    require_same_scorer((report.scorer, *(row.scorer for row in report.between_model_contrasts)))
    confidence = 100 * report.confidence_level
    outcomes = report.ledger.outcomes
    lines = [
        "# TrialEval controlled campaign analysis",
        "",
        f"Scorer: `{report.scorer.version}`; sources SHA-256: `{report.scorer.sources_sha256}`.",
        "",
        f"Release: `{report.release_id}`.",
        f"Scientific analysis status: **{report.scientific_analysis_status}**.",
        f"Models: {report.inventory.model_count}; independent base worlds per model: {report.inventory.independent_base_worlds}.",
        f"Primary attempts: {report.inventory.primary_attempts}; stochastic duplicates: {report.inventory.stochastic_duplicate_attempts}; retries: {report.ledger.retry_attempts}.",
        (
            f"Outcomes — completed {outcomes.completed}; not started {outcomes.not_started}; provider stops "
            f"{outcomes.provider_stop}; provider unavailable {outcomes.provider_unavailable}; model noncompletion {outcomes.model_noncompletion}; malformed "
            f"{outcomes.malformed_output}; replay failures {outcomes.replay_failure}; budget exhaustion "
            f"{outcomes.budget_exhausted}; evaluator errors {outcomes.evaluator_error}; pre-run exclusions "
            f"{outcomes.pre_run_identity_exclusion}."
        ),
        (
            "Retry policy: a primary that ended in an evaluator error, pre-run identity exclusion, or receipt-bound "
            "provider unavailability under the prospective defer_unanswered_v1 condition is retried once; "
            "a retry free of infrastructure faults replaces it. Model outcomes are never retried. "
            f"Recovered primaries: {len(report.ledger.recovered_primary_attempt_ids)}; "
            f"unresolved: {len(report.ledger.primary_scientifically_unscored_attempt_ids)}."
        ),
    ]
    if report.scientific_analysis_status != "complete":
        lines.extend(
            (
                "",
                f"Official comparisons are withheld, including uncertainty: {report.unavailable_reason}. "
                "Per-task grades remain available; this panel does not support a scientific comparison.",
            )
        )
        return "\n".join(lines) + "\n"
    lines.extend(
        (
            "",
            "Observed-case means describe the scored subset and do not rank incomplete panels. Lower loss is better. "
            "Bounded recovery, 1/(1 + mean loss), transforms the complete-panel loss after aggregation; "
            "higher is better, its interval transforms the primary world-bootstrap limits in reverse order, "
            "and its standard error uses the delta method.",
            "",
            "| Model | Target | Scheduled | Scored | Worlds scored/scheduled | Targets scored/scheduled | Observed mean loss | Full-panel mean loss | Bounded recovery | Target failures |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        )
    )
    for ranking in report.loss_rankings:
        failures = ", ".join(f"{origin}: {count}" for origin, count in ranking.target_failure_counts.items()) or "none"
        lines.append(
            f"| {ranking.model_id} | {ranking.target_kind} | {ranking.scheduled_attempts} | {ranking.scored_attempts} | "
            f"{ranking.scored_worlds}/{ranking.independent_worlds} | "
            f"{ranking.scored_targets}/{ranking.scheduled_targets} | "
            f"{ranking.observed_mean_loss if ranking.observed_mean_loss is not None else 'unavailable'} | "
            f"{ranking.mean_loss if ranking.mean_loss is not None else 'unavailable'} | "
            f"{_recovery_text(ranking)} | {failures} |"
        )
    if report.difficulty_summaries:
        lines.extend(
            (
                "",
                "Difficulty summaries are descriptive; family differences also reflect their clinical case mix.",
                "",
                "| Model | Axis | Family | Tasks scored/scheduled | Observed mean loss | Complete-family mean loss |",
                "|---|---|---|---:|---:|---:|",
            )
        )
        for summary in report.difficulty_summaries:
            lines.append(
                f"| {summary.model_id} | {summary.axis} | {summary.family} | "
                f"{summary.scored_attempts}/{summary.scheduled_attempts} | "
                f"{summary.observed_mean_loss if summary.observed_mean_loss is not None else 'unavailable'} | "
                f"{summary.mean_loss if summary.mean_loss is not None else 'unavailable'} |"
            )
    if report.cell_loss_summaries:
        lines.extend(
            (
                "",
                "Clinical scenario summaries are descriptive. Empty planned cells remain visible; target counts do not change task weights.",
                "",
                "| Model | Cell | Tasks scored/scheduled | Worlds scored/scheduled | Targets scored/scheduled | Observed mean loss | Complete-cell mean loss |",
                "|---|---|---:|---:|---:|---:|---:|",
            )
        )
        for cell_summary in report.cell_loss_summaries:
            lines.append(
                f"| {cell_summary.model_id} | {cell_summary.cell_refs[0]} | "
                f"{cell_summary.scored_attempts}/{cell_summary.scheduled_attempts} | "
                f"{cell_summary.scored_worlds}/{cell_summary.independent_worlds} | "
                f"{cell_summary.scored_targets}/{cell_summary.scheduled_targets} | "
                f"{cell_summary.observed_mean_loss if cell_summary.observed_mean_loss is not None else 'unavailable'} | "
                f"{cell_summary.mean_loss if cell_summary.mean_loss is not None else 'unavailable'} |"
            )
    lines.extend(
        (
            "",
            "Primary-target diagnostics describe the first requested estimand. Task loss averages every requested target.",
            "",
            f"| Model | Metric | C | Scheduled | Observed | Mean | {confidence:g}% interval |",
            "|---|---|---|---:|---:|---:|---:|",
        )
    )
    for row in report.metric_summaries:
        mean = "unavailable" if row.equal_cell_effect_mean is None else f"{row.equal_cell_effect_mean:.4f}"
        interval = "unavailable" if row.interval_low is None else f"[{row.interval_low:.4f}, {row.interval_high:.4f}]"
        lines.append(
            f"| {row.model_id} | {row.metric.metric_id} | {row.context} | {row.scheduled_attempts} | {row.observed_attempts} | {mean} | {interval} |"
        )
    if report.target_summaries:
        lines.extend(
            (
                "",
                "Target summaries describe scored results within each clinical cell. Each diagnostic shows its observed count; absent results remain unavailable.",
                "",
                "| Model | Cell | Target | Unit | Scored/scheduled | Observed target loss | Native absolute error | Native interval width | Coverage |",
                "|---|---|---|---|---:|---:|---:|---:|---:|",
            )
        )
        for target_summary in report.target_summaries:
            diagnostics = [
                (
                    f"{target_summary.metric_means[name]:.4g} (n={target_summary.metric_observations[name]})"
                    if name in target_summary.metric_means
                    else "unavailable"
                )
                for name in ("native_absolute_error", "native_interval_width", "interval_covers_oracle")
            ]
            lines.append(
                f"| {target_summary.model_id} | {target_summary.cell_ref} | {target_summary.target_id} | "
                f"{target_summary.result_unit or 'unavailable'} | {target_summary.scored_attempts}/{target_summary.scheduled_attempts} | "
                f"{target_summary.observed_mean_loss if target_summary.observed_mean_loss is not None else 'unavailable'} | "
                + " | ".join(diagnostics)
                + " |"
            )
    headline = tuple(
        row
        for row in report.between_model_contrasts
        if row.metric.metric_id in {"decision_regret_musd", "consequential_error", "material_decision_error_rate"}
        and row.context == "C5"
    )
    if headline:
        lines.extend(
            (
                "",
                "Paired model differences on the headline decision metrics (C5): the same worlds for both models, "
                f"{confidence:g}% world-resampling intervals, Holm across model pairs within each metric. The minimal "
                "detectable difference is the unadjusted normal approximation at 80% power.",
                "",
                "| Models (first minus second) | Metric | Difference | Interval | SE | MDD | Holm p |",
                "|---|---|---:|---|---:|---:|---:|",
            )
        )
        for comparison in headline:
            interval = (
                f"{comparison.interval_low:.4g} to {comparison.interval_high:.4g}"
                if comparison.interval_low is not None and comparison.interval_high is not None
                else "unavailable"
            )
            cells = (
                comparison.estimate,
                comparison.standard_error,
                comparison.minimal_detectable_difference,
                comparison.adjusted_p_value,
            )
            difference, error, detectable, adjusted = ("unavailable" if value is None else value for value in cells)
            lines.append(
                f"| {comparison.minuend_model_id} − {comparison.subtrahend_model_id} | {comparison.metric.metric_id} | "
                f"{difference} | {interval} | {error} | {detectable} | {adjusted} |"
            )
    comparisons = tuple(row for row in report.between_model_contrasts if row.metric.metric_id.endswith("_signal_loss"))
    if comparisons:
        lines.extend(
            (
                "",
                "Common-world differences compare the same completed worlds for both models, retaining all linked contexts. "
                "They average worlds within cells and then cells equally; incomplete subsets are descriptive, not rankings.",
                "",
                "| Models (first minus second) | C | Worlds observed/scheduled | Retained cells | Common-world loss difference | Full-panel loss difference |",
                "|---|---|---:|---|---:|---:|",
            )
        )
        for comparison in comparisons:
            lines.append(
                f"| {comparison.minuend_model_id} − {comparison.subtrahend_model_id} | {comparison.context} | "
                f"{comparison.observed_pairs}/{comparison.scheduled_pairs} | "
                f"{', '.join(comparison.common_cell_refs) or 'none'} | "
                f"{comparison.common_world_estimate if comparison.common_world_estimate is not None else 'unavailable'} | "
                f"{comparison.estimate if comparison.estimate is not None else 'unavailable'} |"
            )
    lines.extend(
        (
            "",
            f"C-contrast intervals use {confidence:g}% world resampling; Holm correction is within each homogeneous five-contrast metric family.",
            "C4 and C5 share raw records and the clinical question; their contrast measures the added transfer-repair burden in C5.",
            "Declared A comparisons retain matched worlds and form a separate Holm family within each model and recovery track. D and TrialDev scenario profiles are categorical.",
            "Intervals and tests are unavailable when an applicable series/effect has fewer than two independent worlds; paired means remain descriptive.",
            f"Scope: `{report.scope}`.",
            "",
        )
    )
    return "\n".join(lines)


__all__ = [
    "CONTEXTS_V1",
    "C_CONTRASTS_V1",
    "EFFECT_STATES_V1",
    "INFRASTRUCTURE_ATTEMPT_OUTCOMES_V1",
    "TrialEvalAuthenticatedGradeSourceV1",
    "TrialEvalAttemptLedgerSummaryV1",
    "TrialEvalAttemptOutcomeCountsV1",
    "TrialEvalBetweenModelContrastV1",
    "TrialEvalCContrastEstimateV1",
    "TrialEvalCContrastFamilyV1",
    "TrialEvalCampaignAnalysisV1",
    "TrialEvalCampaignAttemptV1",
    "TrialEvalCampaignCellV1",
    "TrialEvalCampaignInventoryV1",
    "TrialEvalCampaignPlanV1",
    "TrialEvalDuplicateVariabilityV1",
    "TrialEvalMetricDefinitionV1",
    "TrialEvalMetricSummaryV1",
    "analyse_trial_eval_campaign_v1",
    "trial_eval_answer_rows_v1",
    "derive_trial_eval_campaign_inventory_v1",
    "validate_trial_eval_campaign_plan_v1",
    "holm_adjusted_p_values_v1",
    "load_trial_eval_campaign_grade_source_v1",
    "render_trial_eval_campaign_analysis_v1",
    "summarize_trial_eval_attempt_ledger_v1",
    "trialeval_target_consequence_v1",
]
