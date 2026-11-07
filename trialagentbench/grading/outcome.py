"""Estimator-neutral component grading for TrialEval outcomes."""

from __future__ import annotations

import math
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from trialagentbench_verifier.numerical_scoring import finite_diagnostic, interval_score, numerical_loss
from trialagentbench_verifier.trialeval import TrialEvalDataIntegrityKeyV1 as TrialEvalDataIntegrityKeyV1
from trialagentbench_verifier.trialeval import (
    TrialEvalDecisionRuleV1,
    TrialEvalGradingKeyRecordV1,
    TrialEvalGradingPlanV1,
    trialeval_decision_targets_v1,
    trialeval_decisions_v1,
)
from trialagentbench_verifier.trialeval import TrialEvalScoreInputV1 as TrialEvalScoreInputV1

TrialEvalContextV1 = Literal["C1", "C2", "C3", "C4", "C5"]
TrialEvalEffectMeasureV1 = Literal[
    "risk_difference",
    "restricted_mean_survival_time_difference",
    "mean_difference",
    "rate_ratio",
]
TrialEvalDeclaredEffectMeasureV1 = Literal[
    "bounds_interval",
    "cif_difference_tau",
    "diagnostic_summary",
    "hazard_ratio",
    "log_hr",
    "log_time_ratio",
    "mean_difference",
    "milestone_risk_difference_tau",
    "not_applicable",
    "odds_ratio",
    "piecewise_log_hr_vector",
    "rate_ratio",
    "restricted_mean_survival_time_difference",
    "risk_difference",
    "risk_difference_tau",
    "risk_ratio",
    "rmst_difference_tau",
    "standardized_risk_difference",
    "standardized_risk_difference_tau_reference",
    "survival_probability",
    "time_varying_log_hr",
    "weighted_logrank_test",
]
TrialEvalAttemptStatusV1 = Literal["complete", "incomplete", "missing", "missing_required_output"]
TrialEvalConclusionV1 = Literal[
    "favors_treatment",
    "favors_comparator",
    "no_clear_difference",
    "not_identified",
]
TrialEvalFailureOriginV1 = Literal[
    "confidence_level_mismatch",
    "decision_mismatch",
    "interval_type_mismatch",
    "outer_region_noncoverage",
    "uninformative_identified_set",
    "identified_set_definition_mismatch",
    "incomplete_submission",
    "irreproducible_primary_result",
    "invalid_sampling_uncertainty",
    "missing_sampling_uncertainty",
    "missing_required_output",
    "missing_submission",
    "prohibited_evidence",
    "result_type_mismatch",
    "unsupported_nonidentification",
    "wrong_question",
]
TrialEvalComponentStatusV1 = Literal[
    "satisfied",
    "not_satisfied",
    "observed",
    "not_evaluable",
    "not_applicable",
]
TrialEvalComponentIdV1 = Literal[
    "completion",
    "question_alignment",
    "evidence_eligibility",
    "reproducibility",
    "estimand_performance",
    "uncertainty",
    "decision",
]

_PLAN_CONTEXTS = frozenset({"C1", "C3"})
_RATIO_EFFECT_MEASURES = frozenset({"hazard_ratio", "odds_ratio", "rate_ratio", "risk_ratio"})
_RELEASE_EFFECT_MEASURES = frozenset(
    {"mean_difference", "rate_ratio", "restricted_mean_survival_time_difference", "risk_difference"}
)


class _OutcomeModelV1(BaseModel):
    """Strict immutable base for outcome-grading contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class TrialEvalQuestionBindingV1(_OutcomeModelV1):
    """Participant-visible clinical question, excluding estimator identity."""

    schema_id: Literal["trialagentbench.trialeval_question_binding/v1"] = (
        "trialagentbench.trialeval_question_binding/v1"
    )
    analysis_population_id: str = Field(min_length=1)
    estimand_id: str = Field(min_length=1)
    intercurrent_event_strategy_ids: tuple[str, ...] = ()
    assessment_horizon_days: float | None = Field(default=None, gt=0.0)
    treatment_id: str = Field(min_length=1)
    comparator_id: str = Field(min_length=1)
    endpoint_id: str = Field(min_length=1)
    effect_measure: TrialEvalDeclaredEffectMeasureV1
    analysis_unit: Literal["participant", "cluster"]
    population_basis: Literal["finite_population", "superpopulation"]
    identification_status: Literal[
        "point_identified",
        "identified_under_assumptions",
        "partially_identified",
    ]
    identification_assumption_ids: tuple[str, ...] = ()
    result_unit: str = Field(min_length=1)
    uncertainty_type: Literal[
        "central_confidence_interval",
        "repeated_confidence_interval",
        "simultaneous_outer_confidence_region",
    ]
    confidence_level: float = Field(gt=0.0, lt=1.0)
    sidedness: Literal["two_sided_central", "simultaneous_outer"]
    null_value: float
    favorable_direction: Literal["higher", "lower"]

    @field_validator("intercurrent_event_strategy_ids", "identification_assumption_ids", mode="before")
    @classmethod
    def _restore_string_tuple(cls, value: object) -> object:
        """Restore immutable tuples after ordinary JSON decoding."""

        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _canonical_question(self) -> Self:
        if self.treatment_id == self.comparator_id:
            raise ValueError("treatment and comparator must differ")
        for name, values in (
            ("intercurrent_event_strategy_ids", self.intercurrent_event_strategy_ids),
            ("identification_assumption_ids", self.identification_assumption_ids),
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
        assumption_dependent = self.identification_status in {
            "identified_under_assumptions",
            "partially_identified",
        }
        if assumption_dependent != bool(self.identification_assumption_ids):
            raise ValueError("assumption-dependent questions require named identification assumptions")
        if (self.identification_status == "partially_identified") != (
            self.uncertainty_type == "simultaneous_outer_confidence_region"
        ):
            raise ValueError("partially identified questions require a simultaneous outer confidence region")
        expected_sidedness = (
            "simultaneous_outer" if self.identification_status == "partially_identified" else "two_sided_central"
        )
        if self.sidedness != expected_sidedness:
            raise ValueError("question sidedness differs from its requested uncertainty object")
        if self.effect_measure in _RATIO_EFFECT_MEASURES and self.result_unit != "ratio":
            raise ValueError("ratio-scale questions require result unit ratio")
        expected_null = 1.0 if self.effect_measure in _RATIO_EFFECT_MEASURES else 0.0
        if self.null_value != expected_null:
            raise ValueError(f"{self.effect_measure} questions require null value {expected_null}")
        return self


class TrialEvalConfidenceIntervalV1(_OutcomeModelV1):
    """Ordered confidence interval for one scalar quantity."""

    lower: float
    upper: float
    confidence_level: float = Field(gt=0.0, lt=1.0)
    interval_type: Literal["central_confidence_interval", "repeated_confidence_interval"] = (
        "central_confidence_interval"
    )
    sidedness: Literal["two_sided_central"] = "two_sided_central"

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.lower > self.upper:
            raise ValueError("confidence interval lower endpoint exceeds its upper endpoint")
        return self


class TrialEvalConsequenceEconomicsV1(_OutcomeModelV1):
    """Declared confirmation economics of the task's trial (US$ millions)."""

    cost_type: str = Field(min_length=1)
    value_of_success_musd: float = Field(gt=0)
    replicate_cost_musd: float = Field(gt=0)
    fixed_trial_cost_musd: float = Field(ge=0)


class TrialEvalConsequenceReferenceV1(_OutcomeModelV1):
    """Reference analyses of a task's data, its size and its economics, for pricing an answer's decision.

    Each reference analysis is held as (estimate, lower, upper) on the native scale.
    """

    participants: int = Field(ge=1)
    credited: tuple[tuple[float, float, float], ...] = Field(min_length=1)
    economics: TrialEvalConsequenceEconomicsV1

    @field_validator("credited", mode="before")
    @classmethod
    def _restore_credited(cls, value: object) -> object:
        """Restore immutable triples after JSON decoding."""

        return tuple(tuple(row) for row in value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if any(not lower <= estimate <= upper for estimate, lower, upper in self.credited):
            raise ValueError("credited analyses require an estimate inside its interval")
        return self


class TrialEvalPointOracleTargetV1(_OutcomeModelV1):
    """Qualified point oracle and pre-agent normalization scale.

    ``point_scored`` is false where the family's analysis plan reports the point estimate without scoring
    it (TE-S09); the numerical loss then keeps only its interval half.
    """

    kind: Literal["point"] = "point"
    binding: TrialEvalQuestionBindingV1
    value: float
    confidence_level: float = Field(gt=0.0, lt=1.0)
    reference_sampling_standard_deviation: float = Field(gt=0.0)
    reference_sampling_basis_id: Literal["finite_world_oracle_contrast"]
    point_scored: bool = Field(default=True, exclude_if=lambda value: value)
    consequence: TrialEvalConsequenceReferenceV1 | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def _point_identification(self) -> Self:
        if self.binding.effect_measure not in _RELEASE_EFFECT_MEASURES:
            raise ValueError("point oracle uses an unsupported release effect measure")
        if self.binding.identification_status == "partially_identified":
            raise ValueError("partially identified questions require an identified-set target")
        if self.binding.effect_measure == "rate_ratio" and self.value <= 0.0:
            raise ValueError("rate-ratio oracle values must be positive")
        if self.confidence_level != self.binding.confidence_level:
            raise ValueError("point target confidence level differs from its question")
        return self


TrialEvalOracleTargetV1 = TrialEvalPointOracleTargetV1


class TrialEvalGradingReferenceV1(_OutcomeModelV1):
    """One target with its own normalization and reference precision evidence."""

    target: TrialEvalOracleTargetV1
    oracle_worlds: int = Field(ge=1)
    oracle_monte_carlo_standard_error: float = Field(ge=0.0)
    maximum_oracle_monte_carlo_standard_error: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _qualified_reference(self) -> Self:
        finite_population = self.target.binding.population_basis == "finite_population"
        if finite_population != (self.oracle_worlds == 1):
            raise ValueError("finite-population targets require one exact attained world")
        if finite_population and self.oracle_monte_carlo_standard_error != 0.0:
            raise ValueError("exact finite-population targets cannot have Monte Carlo error")
        if self.oracle_monte_carlo_standard_error > self.maximum_oracle_monte_carlo_standard_error:
            raise ValueError("oracle Monte Carlo standard error exceeds its qualified maximum")
        return self


class TrialEvalOracleReferenceV1(TrialEvalGradingReferenceV1):
    """A graded target with the identities of its reference analyses (construction-private)."""

    reference_analysis_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator("reference_analysis_ids", mode="before")
    @classmethod
    def _restore_reference_tuple(cls, value: object) -> object:
        """Restore the immutable analysis inventory after JSON decoding."""

        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _sorted_reference_analyses(self) -> Self:
        if self.reference_analysis_ids != tuple(sorted(set(self.reference_analysis_ids))):
            raise ValueError("reference analysis IDs must be sorted and unique")
        return self


TRIALEVAL_PLAN_EXECUTION_TOLERANCE_SD_V1 = 0.05
"""Plan execution is accurate when the point and both interval endpoints lie within this many grading
standard deviations of the plan route's own result (comparison scale). Every plan route is deterministic, so
the allowance covers solver and reporting precision only."""


class TrialEvalPlanReplayTargetV1(_OutcomeModelV1):
    """The plan analysis with the identity of its route, as construction retains it privately."""

    target_id: Literal["prespecified_primary_analysis"] = "prespecified_primary_analysis"
    route_id: str = Field(min_length=1)
    estimate: float
    lower: float
    upper: float
    tolerance_sd: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.lower > self.upper:
            raise ValueError("plan replay interval lower endpoint exceeds its upper endpoint")
        return self


class TrialEvalEvaluatorRecordV1(TrialEvalOracleReferenceV1):
    """The construction-private task record from which the grading key is derived."""

    schema_id: Literal["trialagentbench.trialeval_evaluator_record/v1"] = (
        "trialagentbench.trialeval_evaluator_record/v1"
    )
    release_id: str = Field(min_length=1)
    task_id: str = Field(pattern=r"^te_[0-9a-f]{20}$")
    trial_id: str = Field(pattern=r"^trial_[0-9a-f]{20}$")
    base_world_id: str = Field(pattern=r"^world_[0-9a-f]{20}$")
    context: TrialEvalContextV1
    participant_data_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    participant_artifact_paths: tuple[str, ...] = Field(min_length=1)
    plan_analysis_method_ids: tuple[str, ...] = ()
    additional_references: tuple[TrialEvalOracleReferenceV1, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    decision: TrialEvalDecisionRuleV1 | None = Field(default=None, exclude_if=lambda value: value is None)
    plan_replay: TrialEvalPlanReplayTargetV1 | None = Field(default=None, exclude_if=lambda value: value is None)

    @field_validator("participant_artifact_paths", "additional_references", mode="before")
    @classmethod
    def _restore_record_tuple(cls, value: object) -> object:
        """Restore immutable inventories after JSON decoding."""
        return tuple(value) if isinstance(value, list) else value

    @property
    def references(self) -> tuple[TrialEvalOracleReferenceV1, ...]:
        """Return the primary reference followed by required complementary results."""
        return (self, *self.additional_references)

    @property
    def target_ids(self) -> tuple[str, ...]:
        """Return the scored results in the assigned reporting order."""
        return tuple(reference.target.binding.estimand_id for reference in self.references)

    @property
    def reported_target_ids(self) -> tuple[str, ...]:
        """Return every reported result: the scored results, then the plan as executed where a plan is given."""
        return (*self.target_ids, *(() if self.plan_replay is None else (self.plan_replay.target_id,)))

    @model_validator(mode="after")
    def _complete_evaluator_record(self) -> Self:
        reported = self.reported_target_ids
        if len(set(reported)) != len(reported):
            raise ValueError("task oracle references repeat an estimand identifier")
        if self.participant_artifact_paths != tuple(sorted(set(self.participant_artifact_paths))):
            raise ValueError("participant artifact paths must be sorted and unique")
        if self.plan_analysis_method_ids != tuple(sorted(set(self.plan_analysis_method_ids))):
            raise ValueError("visible-plan analysis method IDs must be sorted and unique")
        planned = self.context in _PLAN_CONTEXTS
        if planned != bool(self.plan_analysis_method_ids):
            raise ValueError("exactly C1 and C3 evaluator records require visible-plan methods")
        if self.plan_replay is not None and not planned:
            raise ValueError("only C1 and C3 evaluator records can carry a plan replay")
        if self.decision is not None:
            tested = trialeval_decision_targets_v1(self.decision.kind, len(self.references))
            if any(len(row) != tested for row in self.decision.reference_decisions):
                raise ValueError("reference decisions do not test the declared targets")
        return self


class TrialEvalPointEstimateV1(_OutcomeModelV1):
    """Submitted point estimate with optional sampling uncertainty."""

    kind: Literal["point"] = "point"
    value: float
    effect_measure: TrialEvalEffectMeasureV1
    unit: str = Field(min_length=1)
    interval: TrialEvalConfidenceIntervalV1 | None = None
    conclusion: TrialEvalConclusionV1


TrialEvalSubmittedResultV1 = TrialEvalPointEstimateV1


class TrialEvalOutcomeGradingInputV1(_OutcomeModelV1):
    """Complete deterministic observations needed for component grading."""

    schema_id: Literal["trialagentbench.trialeval_outcome_grading_input/v1"] = (
        "trialagentbench.trialeval_outcome_grading_input/v1"
    )
    context: TrialEvalContextV1
    target: TrialEvalOracleTargetV1
    attempt_status: TrialEvalAttemptStatusV1
    submitted_binding: TrialEvalQuestionBindingV1 | None = None
    submitted_result: TrialEvalSubmittedResultV1 | None = None
    evidence_eligible: bool | None = None
    reproducible: bool | None = None
    uncertainty_failure: Literal["invalid_sampling_uncertainty"] | None = None

    @model_validator(mode="after")
    def _coherent_attempt(self) -> Self:
        observations = (
            self.submitted_binding,
            self.submitted_result,
            self.evidence_eligible,
            self.reproducible,
            self.uncertainty_failure,
        )
        if self.attempt_status == "missing" and any(value is not None for value in observations):
            raise ValueError("missing submissions cannot contain grading observations")
        if self.attempt_status == "complete" and any(
            value is None for value in (self.submitted_result, self.evidence_eligible, self.reproducible)
        ):
            raise ValueError("complete submissions require a result, evidence observation, and reproducibility")
        if self.uncertainty_failure is not None and self.submitted_result is None:
            raise ValueError("an uncertainty failure requires a retained submitted result")
        return self


class TrialEvalComponentOutcomeV1(_OutcomeModelV1):
    """One independent component disposition with a precise failure origin."""

    component_id: TrialEvalComponentIdV1
    status: TrialEvalComponentStatusV1
    failure_origin: TrialEvalFailureOriginV1 | None = None

    @model_validator(mode="after")
    def _failure_is_explicit(self) -> Self:
        requires_origin = self.status in {"not_satisfied", "not_evaluable"}
        if requires_origin != (self.failure_origin is not None):
            raise ValueError("unsatisfied and non-evaluable components require exactly one failure origin")
        return self


class TrialEvalPointPerformanceV1(_OutcomeModelV1):
    """Point errors; null derived diagnostics exceed floating-point range."""

    submitted_value: float
    oracle_value: float
    native_signed_error: float | None
    native_absolute_error: float | None = Field(ge=0.0)
    native_squared_error: float | None = Field(ge=0.0)
    comparison_scale: Literal["native", "log"]
    comparison_signed_error: float | None
    comparison_absolute_error: float | None = Field(ge=0.0)
    comparison_squared_error: float | None = Field(ge=0.0)
    reference_sampling_standard_deviation: float = Field(gt=0.0)
    normalized_absolute_error: float | None = Field(ge=0.0)
    normalized_squared_error: float | None = Field(ge=0.0)


class TrialEvalIntervalPerformanceV1(_OutcomeModelV1):
    """Interval diagnostics; null losses or widths exceed floating-point range."""

    target_value: float
    native_lower: float
    native_upper: float
    confidence_level: float = Field(gt=0.0, lt=1.0)
    covers_target: bool
    native_width: float | None = Field(ge=0.0)
    comparison_scale: Literal["native", "log"]
    comparison_lower: float
    comparison_upper: float
    comparison_width: float | None = Field(ge=0.0)
    proper_interval_score: float | None = Field(ge=0.0)
    native_interval_score: float | None = Field(ge=0.0)


class TrialEvalOutcomeGradeV1(_OutcomeModelV1):
    """Denominator-retaining TrialEval component outcomes without an overall grade."""

    schema_id: Literal["trialagentbench.trialeval_outcome_grade/v2"] = "trialagentbench.trialeval_outcome_grade/v2"
    context: TrialEvalContextV1
    target_kind: Literal["point"]
    included_in_denominator: Literal[True] = True
    completion: TrialEvalComponentOutcomeV1
    question_alignment: TrialEvalComponentOutcomeV1
    evidence_eligibility: TrialEvalComponentOutcomeV1
    reproducibility: TrialEvalComponentOutcomeV1
    estimand_performance: TrialEvalComponentOutcomeV1
    uncertainty: TrialEvalComponentOutcomeV1
    point_performance: TrialEvalPointPerformanceV1 | None = None
    interval_performance: TrialEvalIntervalPerformanceV1 | None = None
    failure_origins: tuple[TrialEvalFailureOriginV1, ...] = ()
    signal_loss: float | None = Field(default=None, ge=0.0)

    @field_validator("failure_origins", mode="before")
    @classmethod
    def _restore_failure_tuple(cls, value: object) -> object:
        """Restore immutable failure origins after JSON decoding."""

        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _component_metrics_agree(self) -> Self:
        point_present = self.point_performance is not None
        if (self.estimand_performance.status == "observed") != point_present:
            raise ValueError("observed estimand performance requires a point metric record")
        uncertainty_present = self.interval_performance is not None
        if (self.uncertainty.status == "observed") != uncertainty_present:
            raise ValueError("observed uncertainty requires exactly one matching uncertainty record")
        components = (
            self.completion,
            self.question_alignment,
            self.evidence_eligibility,
            self.reproducibility,
            self.estimand_performance,
            self.uncertainty,
        )
        expected_origins = tuple(
            sorted({component.failure_origin for component in components if component.failure_origin is not None})
        )
        if self.failure_origins != expected_origins:
            raise ValueError("failure_origins must be the sorted unique component failure origins")
        return self


class TrialEvalPlanExecutionV1(_OutcomeModelV1):
    """Execution accuracy of the prespecified primary analysis, scored apart from the task.

    Each deviation is the absolute difference between the reported plan-as-executed result and the plan
    route's own result, on the comparison scale in grading standard deviations. Deviations are absent when
    the plan as executed was not reported or not reproduced, which is inaccurate.
    """

    plan: TrialEvalGradingPlanV1
    point_deviation: float | None = Field(default=None, ge=0.0)
    lower_deviation: float | None = Field(default=None, ge=0.0)
    upper_deviation: float | None = Field(default=None, ge=0.0)
    accurate: bool

    @model_validator(mode="after")
    def _accuracy_follows_deviations(self) -> Self:
        deviations = (self.point_deviation, self.lower_deviation, self.upper_deviation)
        within = all(value is not None and value <= self.plan.tolerance_sd for value in deviations)
        if self.accurate != within:
            raise ValueError("plan execution accuracy differs from its deviations and tolerance")
        return self


def _component(
    component_id: TrialEvalComponentIdV1,
    status: TrialEvalComponentStatusV1,
    failure_origin: TrialEvalFailureOriginV1 | None = None,
) -> TrialEvalComponentOutcomeV1:
    return TrialEvalComponentOutcomeV1(
        component_id=component_id,
        status=status,
        failure_origin=failure_origin,
    )


def _missing_origin(attempt_status: TrialEvalAttemptStatusV1) -> TrialEvalFailureOriginV1:
    if attempt_status == "missing":
        return "missing_submission"
    if attempt_status == "missing_required_output":
        return "missing_required_output"
    return "incomplete_submission"


def _binary_component(
    component_id: TrialEvalComponentIdV1,
    value: bool | None,
    *,
    attempt_status: TrialEvalAttemptStatusV1,
    false_origin: TrialEvalFailureOriginV1,
) -> TrialEvalComponentOutcomeV1:
    if value is None:
        return _component(component_id, "not_evaluable", _missing_origin(attempt_status))
    if value:
        return _component(component_id, "satisfied")
    return _component(component_id, "not_satisfied", false_origin)


def _comparison_value(value: float, effect_measure: TrialEvalDeclaredEffectMeasureV1) -> float:
    if effect_measure != "rate_ratio":
        return value
    if value <= 0.0:
        raise ValueError("rate-ratio estimates, intervals, and sets must be positive")
    return math.log(value)


def _point_performance(
    target: TrialEvalPointOracleTargetV1,
    result: TrialEvalPointEstimateV1,
) -> TrialEvalPointPerformanceV1:
    native_error = result.value - target.value
    submitted_comparison = _comparison_value(result.value, target.binding.effect_measure)
    oracle_comparison = _comparison_value(target.value, target.binding.effect_measure)
    comparison_error = submitted_comparison - oracle_comparison
    reference_deviation = target.reference_sampling_standard_deviation
    return TrialEvalPointPerformanceV1(
        submitted_value=result.value,
        oracle_value=target.value,
        native_signed_error=finite_diagnostic(native_error),
        native_absolute_error=finite_diagnostic(abs(native_error)),
        native_squared_error=finite_diagnostic(native_error * native_error),
        comparison_scale="log" if target.binding.effect_measure == "rate_ratio" else "native",
        comparison_signed_error=finite_diagnostic(comparison_error),
        comparison_absolute_error=finite_diagnostic(abs(comparison_error)),
        comparison_squared_error=finite_diagnostic(comparison_error * comparison_error),
        reference_sampling_standard_deviation=reference_deviation,
        normalized_absolute_error=finite_diagnostic(abs(comparison_error) / reference_deviation),
        normalized_squared_error=finite_diagnostic(
            (comparison_error / reference_deviation) * (comparison_error / reference_deviation)
        ),
    )


def _interval_performance(
    interval: TrialEvalConfidenceIntervalV1,
    target_value: float,
    effect_measure: TrialEvalDeclaredEffectMeasureV1,
) -> TrialEvalIntervalPerformanceV1:
    comparison_lower = _comparison_value(interval.lower, effect_measure)
    comparison_upper = _comparison_value(interval.upper, effect_measure)
    comparison_target = _comparison_value(target_value, effect_measure)
    return TrialEvalIntervalPerformanceV1(
        target_value=target_value,
        native_lower=interval.lower,
        native_upper=interval.upper,
        confidence_level=interval.confidence_level,
        covers_target=interval.lower <= target_value <= interval.upper,
        native_width=finite_diagnostic(interval.upper - interval.lower),
        comparison_scale="log" if effect_measure == "rate_ratio" else "native",
        comparison_lower=comparison_lower,
        comparison_upper=comparison_upper,
        comparison_width=finite_diagnostic(comparison_upper - comparison_lower),
        proper_interval_score=finite_diagnostic(
            interval_score(
                comparison_lower,
                comparison_upper,
                comparison_target,
                interval.confidence_level,
            )
        ),
        native_interval_score=finite_diagnostic(
            interval_score(interval.lower, interval.upper, target_value, interval.confidence_level)
        ),
    )


def _numerical_failure(
    grading_input: TrialEvalOutcomeGradingInputV1,
) -> TrialEvalFailureOriginV1 | None:
    if grading_input.attempt_status != "complete":
        return _missing_origin(grading_input.attempt_status)
    submitted_binding = grading_input.submitted_binding
    if submitted_binding is not None and not _estimand_core_matches(submitted_binding, grading_input.target.binding):
        return "wrong_question"
    if not grading_input.reproducible:
        return "irreproducible_primary_result"
    target = grading_input.target
    result = grading_input.submitted_result
    assert isinstance(result, TrialEvalPointEstimateV1)
    if not _result_scale_matches(result, target.binding):
        return "wrong_question"
    return None


def _question_outcome(grading_input: TrialEvalOutcomeGradingInputV1) -> TrialEvalComponentOutcomeV1:
    result = grading_input.submitted_result
    if result is None:
        return _component("question_alignment", "not_evaluable", _missing_origin(grading_input.attempt_status))
    if grading_input.submitted_binding is not None and not _question_matches(
        grading_input.submitted_binding, grading_input.target.binding
    ):
        return _component("question_alignment", "not_satisfied", "wrong_question")
    if _result_scale_matches(result, grading_input.target.binding):
        return _component("question_alignment", "observed")
    return _component("question_alignment", "not_satisfied", "wrong_question")


def _result_scale_matches(
    result: TrialEvalSubmittedResultV1,
    target: TrialEvalQuestionBindingV1,
) -> bool:
    """Check the numerical result scale that the executable program declares."""

    return result.effect_measure == target.effect_measure and result.unit == target.result_unit


def _question_matches(
    submitted: TrialEvalQuestionBindingV1,
    target: TrialEvalQuestionBindingV1,
) -> bool:
    """Compare every declared part of the clinical question."""

    return submitted == target


def _estimand_core_matches(
    submitted: TrialEvalQuestionBindingV1,
    target: TrialEvalQuestionBindingV1,
) -> bool:
    """Compare the clinical quantity without gating on identification prose."""

    excluded = {"identification_status", "identification_assumption_ids"}
    return submitted.model_dump(mode="python", exclude=excluded) == target.model_dump(
        mode="python",
        exclude=excluded,
    )


def _expected_conclusion(
    result: TrialEvalPointEstimateV1,
    binding: TrialEvalQuestionBindingV1,
) -> TrialEvalConclusionV1:
    assert result.interval is not None
    lower, upper = result.interval.lower, result.interval.upper
    if binding.favorable_direction == "higher":
        if lower > binding.null_value:
            return "favors_treatment"
        if upper < binding.null_value:
            return "favors_comparator"
    else:
        if upper < binding.null_value:
            return "favors_treatment"
        if lower > binding.null_value:
            return "favors_comparator"
    return "no_clear_difference"


def _sampling_uncertainty_failure(grading_input: TrialEvalOutcomeGradingInputV1) -> TrialEvalFailureOriginV1 | None:
    """Validate the requested interval without adjudicating its statistical method."""
    if grading_input.uncertainty_failure is not None:
        return grading_input.uncertainty_failure
    result = grading_input.submitted_result
    assert result is not None
    interval = result.interval
    if interval is None:
        return "missing_sampling_uncertainty"
    if interval.confidence_level != grading_input.target.confidence_level:
        return "confidence_level_mismatch"
    if interval.interval_type != grading_input.target.binding.uncertainty_type:
        return "interval_type_mismatch"
    return None


def grade_trialeval_outcome_v1(
    grading_input: TrialEvalOutcomeGradingInputV1,
) -> TrialEvalOutcomeGradeV1:
    """Grade one TrialEval attempt as independent estimator-neutral components.

    Parameters
    ----------
    grading_input
        Frozen oracle target and deterministic observations from the submitted
        report, program, and typed result.

    Returns
    -------
    TrialEvalOutcomeGradeV1
        Denominator-retaining component outcomes and applicable numerical
        metrics. The result deliberately has no overall pass or route gate.

    Raises
    ------
    ValueError
        If rate-ratio values are nonpositive and therefore undefined on the
        required log comparison scale.
    """

    completion = (
        _component("completion", "satisfied")
        if grading_input.attempt_status == "complete"
        else _component("completion", "not_satisfied", _missing_origin(grading_input.attempt_status))
    )
    question_alignment = _question_outcome(grading_input)
    evidence_eligibility = _binary_component(
        "evidence_eligibility",
        grading_input.evidence_eligible,
        attempt_status=grading_input.attempt_status,
        false_origin="prohibited_evidence",
    )
    reproducibility = _binary_component(
        "reproducibility",
        grading_input.reproducible,
        attempt_status=grading_input.attempt_status,
        false_origin="irreproducible_primary_result",
    )

    numerical_failure = _numerical_failure(grading_input)
    point_performance = None
    interval_performance = None
    if numerical_failure is not None:
        estimand_performance = _component("estimand_performance", "not_evaluable", numerical_failure)
        uncertainty = _component("uncertainty", "not_evaluable", numerical_failure)
    else:
        assert grading_input.submitted_result is not None
        point_performance = _point_performance(grading_input.target, grading_input.submitted_result)
        estimand_performance = _component("estimand_performance", "observed")
        submitted_interval = grading_input.submitted_result.interval
        uncertainty_failure = _sampling_uncertainty_failure(grading_input)
        if uncertainty_failure is not None:
            uncertainty = _component("uncertainty", "not_evaluable", uncertainty_failure)
        else:
            assert submitted_interval is not None
            interval_performance = _interval_performance(
                submitted_interval,
                grading_input.target.value,
                grading_input.target.binding.effect_measure,
            )
            uncertainty = _component("uncertainty", "observed")
    components = (
        completion,
        question_alignment,
        evidence_eligibility,
        reproducibility,
        estimand_performance,
        uncertainty,
    )
    failure_origins = tuple(
        sorted({component.failure_origin for component in components if component.failure_origin is not None})
    )
    outcome = TrialEvalOutcomeGradeV1(
        context=grading_input.context,
        target_kind=grading_input.target.kind,
        completion=completion,
        question_alignment=question_alignment,
        evidence_eligibility=evidence_eligibility,
        reproducibility=reproducibility,
        estimand_performance=estimand_performance,
        uncertainty=uncertainty,
        point_performance=point_performance,
        interval_performance=interval_performance,
        failure_origins=failure_origins,
    )
    return outcome.model_copy(
        update={"signal_loss": trialeval_signal_loss_v1(target=grading_input.target, outcome=outcome)}
    )


def grade_trialeval_decision_v1(
    record: TrialEvalGradingKeyRecordV1,
    grading_inputs: tuple[TrialEvalOutcomeGradingInputV1, ...],
    outcomes: tuple[TrialEvalOutcomeGradeV1, ...],
) -> TrialEvalComponentOutcomeV1 | None:
    """Score a task's declared decision from its submitted intervals.

    Parameters
    ----------
    record
        Grading-key record whose ``decision`` declares the rule and the reference decisions.
    grading_inputs, outcomes
        Grading inputs and component grades of every target, in the record's reporting order.

    Returns
    -------
    TrialEvalComponentOutcomeV1 or None
        ``satisfied`` when the participant's decision is a reference decision, ``not_satisfied``
        otherwise, ``not_evaluable`` when a tested interval was not observed, and None without a rule.

    Raises
    ------
    ValueError
        If the inputs do not cover the record's targets in order.
    """
    rule = record.decision
    if rule is None:
        return None
    if tuple(item.target.binding.estimand_id for item in grading_inputs) != record.target_ids or len(outcomes) != len(
        grading_inputs
    ):
        raise ValueError("decision grading requires every target in reporting order")
    intervals = []
    for grading_input, outcome in zip(grading_inputs, outcomes, strict=True):
        if outcome.interval_performance is None:
            return _component("decision", "not_evaluable", outcome.uncertainty.failure_origin)
        assert grading_input.submitted_result is not None and grading_input.submitted_result.interval is not None
        intervals.append(
            (grading_input.submitted_result.interval.lower, grading_input.submitted_result.interval.upper)
        )
    decisions = trialeval_decisions_v1(
        kind=rule.kind,
        margin=rule.margin,
        questions=tuple(
            (reference.target.binding.null_value, reference.target.binding.favorable_direction)
            for reference in record.references
        ),
        intervals=tuple(intervals),
    )
    if decisions in rule.reference_decisions:
        return _component("decision", "satisfied")
    return _component("decision", "not_satisfied", "decision_mismatch")


def grade_trialeval_plan_execution_v1(
    record: TrialEvalGradingKeyRecordV1,
    grading_input: TrialEvalOutcomeGradingInputV1,
) -> TrialEvalPlanExecutionV1 | None:
    """Score the reported plan as executed against the plan route's result on the same data.

    Parameters
    ----------
    record
        Grading-key record whose ``plan_replay`` holds the plan analysis result and tolerance.
    grading_input
        The plan-as-executed row bound with the primary question's scale.

    Returns
    -------
    TrialEvalPlanExecutionV1 or None
        Execution accuracy, kept apart from the numerical loss, the decision and its consequence; None
        when no plan is given.
    """
    plan = record.plan_replay
    if plan is None:
        return None
    result = grading_input.submitted_result
    deviations: tuple[float | None, ...] = (None, None, None)
    if (
        grading_input.attempt_status == "complete"
        and grading_input.reproducible
        and result is not None
        and result.interval is not None
    ):
        measure = record.target.binding.effect_measure
        scale = record.target.reference_sampling_standard_deviation
        deviations = tuple(
            finite_diagnostic(abs(_comparison_value(reported, measure) - _comparison_value(planned, measure)) / scale)
            for reported, planned in (
                (result.value, plan.estimate),
                (result.interval.lower, plan.lower),
                (result.interval.upper, plan.upper),
            )
        )
    point, lower, upper = deviations
    return TrialEvalPlanExecutionV1(
        plan=plan,
        point_deviation=point,
        lower_deviation=lower,
        upper_deviation=upper,
        accurate=all(value is not None and value <= plan.tolerance_sd for value in deviations),
    )


def trialeval_signal_loss_v1(*, target: TrialEvalOracleTargetV1, outcome: TrialEvalOutcomeGradeV1) -> float | None:
    """Return numerical loss, withholding failed or unreproduced answers.

    Parameters
    ----------
    target : TrialEvalOracleTargetV1
        Qualified evaluator target and fixed comparison scale.
    outcome : TrialEvalOutcomeGradeV1
        Component grade from the same task and target.

    Returns
    -------
    float
        Equal-weight normalised point and interval loss (the interval half alone for a target whose
        point is not scored), or an unavailable value.

    Raises
    ------
    ValueError
        If target identities or scientific values differ.
    """

    if target.kind != outcome.target_kind:
        raise ValueError("Recovery target and outcome kinds differ")
    eligible = (outcome.completion, outcome.question_alignment, outcome.evidence_eligibility, outcome.reproducibility)
    if any(component.status not in {"observed", "satisfied"} for component in eligible):
        return None
    point, interval = outcome.point_performance, outcome.interval_performance
    if point is None or interval is None:
        return None
    if point.oracle_value != target.value or interval.target_value != target.value:
        raise ValueError("Recovery grade identifies a different point target")
    return numerical_loss(
        estimate=(
            _comparison_value(point.submitted_value, target.binding.effect_measure) if target.point_scored else None
        ),
        lower=interval.comparison_lower,
        upper=interval.comparison_upper,
        target=_comparison_value(target.value, target.binding.effect_measure),
        scale=target.reference_sampling_standard_deviation,
        confidence_level=interval.confidence_level,
    )


__all__ = [
    "TRIALEVAL_PLAN_EXECUTION_TOLERANCE_SD_V1",
    "trialeval_signal_loss_v1",
    "TrialEvalComponentOutcomeV1",
    "TrialEvalConfidenceIntervalV1",
    "TrialEvalDataIntegrityKeyV1",
    "TrialEvalEvaluatorRecordV1",
    "TrialEvalGradingReferenceV1",
    "TrialEvalOracleReferenceV1",
    "TrialEvalIntervalPerformanceV1",
    "TrialEvalOutcomeGradeV1",
    "TrialEvalOutcomeGradingInputV1",
    "TrialEvalPlanExecutionV1",
    "TrialEvalPlanReplayTargetV1",
    "TrialEvalScoreInputV1",
    "TrialEvalPointEstimateV1",
    "TrialEvalPointOracleTargetV1",
    "TrialEvalPointPerformanceV1",
    "TrialEvalQuestionBindingV1",
    "grade_trialeval_decision_v1",
    "grade_trialeval_outcome_v1",
    "grade_trialeval_plan_execution_v1",
]
