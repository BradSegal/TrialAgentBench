"""Independent estimator-neutral verification for TrialEval outcomes."""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from pathlib import Path
from statistics import mean
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from trialagentbench_verifier.numerical_scoring import NUMERICAL_LOSS_CONTEXT, round_numerical_loss

Context = Literal["C1", "C2", "C3", "C4", "C5"]
DeclaredEffectMeasure = Literal[
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
ResultEffectMeasure = Literal[
    "mean_difference",
    "rate_ratio",
    "restricted_mean_survival_time_difference",
    "risk_difference",
]
AttemptStatus = Literal["complete", "incomplete", "missing", "missing_required_output"]
Conclusion = Literal["favors_treatment", "favors_comparator", "no_clear_difference", "not_identified"]
ComponentStatus = Literal["satisfied", "not_satisfied", "observed", "not_evaluable", "not_applicable"]
ComponentId = Literal[
    "completion",
    "question_alignment",
    "evidence_eligibility",
    "reproducibility",
    "estimand_performance",
    "uncertainty",
    "decision",
]
DecisionKind = Literal["non_inferiority", "superiority", "fixed_sequence"]

_PLAN_CONTEXTS = frozenset({"C1", "C3"})
_RATIO_EFFECT_MEASURES = frozenset({"hazard_ratio", "odds_ratio", "rate_ratio", "risk_ratio"})
_ORACLE_EFFECT_MEASURES = frozenset(
    {"mean_difference", "rate_ratio", "restricted_mean_survival_time_difference", "risk_difference"}
)


class TrialEvalOutcomeVerificationError(ValueError):
    """Raised when TrialEval outcome bytes violate the independent contract."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class _Question(_Model):
    schema_id: Literal["trialagentbench.trialeval_question_binding/v1"]
    analysis_population_id: str = Field(min_length=1)
    estimand_id: str = Field(min_length=1)
    intercurrent_event_strategy_ids: tuple[str, ...] = ()
    assessment_horizon_days: float | None = Field(default=None, gt=0)
    treatment_id: str = Field(min_length=1)
    comparator_id: str = Field(min_length=1)
    endpoint_id: str = Field(min_length=1)
    effect_measure: DeclaredEffectMeasure
    analysis_unit: Literal["participant", "cluster"]
    population_basis: Literal["finite_population", "superpopulation"]
    identification_status: Literal["point_identified", "identified_under_assumptions", "partially_identified"]
    identification_assumption_ids: tuple[str, ...] = ()
    result_unit: str = Field(min_length=1)
    uncertainty_type: Literal[
        "central_confidence_interval",
        "repeated_confidence_interval",
        "simultaneous_outer_confidence_region",
    ]
    confidence_level: float = Field(gt=0, lt=1)
    sidedness: Literal["two_sided_central", "simultaneous_outer"]
    null_value: float
    favorable_direction: Literal["higher", "lower"]

    @field_validator("intercurrent_event_strategy_ids", "identification_assumption_ids", mode="before")
    @classmethod
    def _tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if self.treatment_id == self.comparator_id:
            raise ValueError("treatment and comparator must differ")
        if self.intercurrent_event_strategy_ids != tuple(sorted(set(self.intercurrent_event_strategy_ids))):
            raise ValueError("intercurrent-event strategies must be sorted and unique")
        if self.identification_assumption_ids != tuple(sorted(set(self.identification_assumption_ids))):
            raise ValueError("identification assumptions must be sorted and unique")
        assumption_dependent = self.identification_status != "point_identified"
        if assumption_dependent != bool(self.identification_assumption_ids):
            raise ValueError("assumption-dependent questions require named assumptions")
        if (self.identification_status == "partially_identified") != (
            self.uncertainty_type == "simultaneous_outer_confidence_region"
        ):
            raise ValueError("partially identified questions require a simultaneous outer confidence region")
        expected_sidedness = (
            "simultaneous_outer" if self.identification_status == "partially_identified" else "two_sided_central"
        )
        if self.sidedness != expected_sidedness:
            raise ValueError("question sidedness differs from its requested uncertainty object")
        expected_null = 1.0 if self.effect_measure in _RATIO_EFFECT_MEASURES else 0.0
        if self.null_value != expected_null:
            raise ValueError("question null differs from its effect measure")
        if self.effect_measure in _RATIO_EFFECT_MEASURES and self.result_unit != "ratio":
            raise ValueError("ratio-scale questions require ratio units")
        return self


class _ConsequenceEconomics(_Model):
    cost_type: str = Field(min_length=1)
    value_of_success_musd: float = Field(gt=0)
    replicate_cost_musd: float = Field(gt=0)
    fixed_trial_cost_musd: float = Field(ge=0)


class _ConsequenceReference(_Model):
    participants: int = Field(ge=1)
    credited: tuple[tuple[float, float, float], ...] = Field(min_length=1)
    economics: _ConsequenceEconomics

    @field_validator("credited", mode="before")
    @classmethod
    def _tuples(cls, value: object) -> object:
        return tuple(tuple(row) for row in value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if any(not lower <= estimate <= upper for estimate, lower, upper in self.credited):
            raise ValueError("credited analyses require an estimate inside its interval")
        return self


class _PointTarget(_Model):
    kind: Literal["point"]
    binding: _Question
    value: float
    confidence_level: float = Field(gt=0, lt=1)
    reference_sampling_standard_deviation: float = Field(gt=0)
    reference_sampling_basis_id: Literal["finite_world_oracle_contrast"]
    point_scored: bool = True
    consequence: _ConsequenceReference | None = None

    @model_validator(mode="after")
    def _supported_oracle_measure(self) -> Self:
        if self.binding.effect_measure not in _ORACLE_EFFECT_MEASURES:
            raise ValueError("point oracle uses an unsupported release effect measure")
        if self.binding.identification_status == "partially_identified":
            raise ValueError("partially identified questions require an identified-set target")
        if self.binding.effect_measure == "rate_ratio" and self.value <= 0:
            raise ValueError("rate-ratio oracle values must be positive")
        if self.confidence_level != self.binding.confidence_level:
            raise ValueError("point target confidence level differs from its question")
        return self


TrialEvalTargetV1 = _PointTarget


class _Reference(_Model):
    """One scored target with its grading scale and the precision of its truth."""

    target: TrialEvalTargetV1
    oracle_worlds: int = Field(ge=1)
    oracle_monte_carlo_standard_error: float = Field(ge=0)
    maximum_oracle_monte_carlo_standard_error: float = Field(gt=0)

    @model_validator(mode="after")
    def _qualified_reference(self) -> Self:
        finite = self.target.binding.population_basis == "finite_population"
        if finite != (self.oracle_worlds == 1):
            raise ValueError("finite-population targets require one exact attained world")
        if finite and self.oracle_monte_carlo_standard_error != 0:
            raise ValueError("finite-population targets cannot have Monte Carlo error")
        if self.oracle_monte_carlo_standard_error > self.maximum_oracle_monte_carlo_standard_error:
            raise ValueError("oracle Monte Carlo standard error exceeds its qualified maximum")
        return self


def trialeval_decision_targets_v1(kind: DecisionKind, target_count: int) -> int:
    """Return how many of a task's targets, in reporting order, a declared decision tests.

    Parameters
    ----------
    kind
        Declared decision rule.
    target_count
        Number of the task's targets.

    Returns
    -------
    int
        Every target for a fixed sequence; the primary target otherwise.

    Examples
    --------
    >>> trialeval_decision_targets_v1("fixed_sequence", 2), trialeval_decision_targets_v1("superiority", 1)
    (2, 1)
    """
    return target_count if kind == "fixed_sequence" else 1


def trialeval_decisions_v1(
    *,
    kind: DecisionKind,
    margin: float,
    questions: tuple[tuple[float, Literal["higher", "lower"]], ...],
    intervals: tuple[tuple[float, float], ...],
) -> tuple[bool, ...]:
    """Return the rejections of a declared non-inferiority, superiority or fixed-sequence decision.

    Parameters
    ----------
    kind
        ``non_inferiority`` tests the first target against its null shifted by ``margin`` in the harmful
        direction. ``superiority`` tests the first target at its null (efficacy is concluded when the
        interval excludes the null on the favourable side). ``fixed_sequence`` tests every target in order
        at its own null, each only when every earlier hypothesis was rejected.
    margin
        Non-inferiority margin on the first target's result scale; zero for superiority and a fixed sequence.
    questions
        Null value and favourable direction of every tested target, in reporting order.
    intervals
        Two-sided interval of every tested target at the declared confidence level.

    Returns
    -------
    tuple of bool
        One rejection per tested target.

    Raises
    ------
    ValueError
        If the rule, questions and intervals disagree in shape.

    Examples
    --------
    >>> trialeval_decisions_v1(
    ...     kind="non_inferiority", margin=0.02, questions=((0.0, "lower"),), intervals=((-0.03, 0.01),)
    ... )
    (True,)
    >>> trialeval_decisions_v1(
    ...     kind="fixed_sequence",
    ...     margin=0.0,
    ...     questions=((0.0, "higher"), (0.0, "lower")),
    ...     intervals=((-1.0, 5.0), (-0.05, -0.01)),
    ... )
    (False, False)
    >>> trialeval_decisions_v1(kind="superiority", margin=0.0, questions=((0.0, "lower"),), intervals=((-0.1, -0.02),))
    (True,)
    """
    tested = trialeval_decision_targets_v1(kind, len(questions))
    if (
        len(questions) != len(intervals)
        or len(questions) < tested
        or (kind == "fixed_sequence" and tested < 2)
        or (kind != "non_inferiority" and margin != 0.0)
        or margin < 0.0
    ):
        raise ValueError("decision rule, questions and intervals disagree")
    rejections: list[bool] = []
    open_sequence = True
    for (null_value, favourable), (lower, upper) in zip(questions[:tested], intervals[:tested], strict=True):
        if favourable == "lower":
            rejected = upper < null_value + margin
        else:
            rejected = lower > null_value - margin
        open_sequence = open_sequence and rejected
        rejections.append(open_sequence)
    return tuple(rejections)


class TrialEvalDecisionRuleV1(_Model):
    """Declared hypothesis-test decision of a task and the reference decisions.

    ``non_inferiority`` tests the primary target against its null shifted by ``margin`` in the harmful
    direction; ``superiority`` tests the primary target at its null (the efficacy decision);
    ``fixed_sequence`` tests every target in reporting order, each only after every earlier rejection. A
    participant's decision follows from the submitted intervals and is scored against the decisions of the
    reference analyses on the same data.
    """

    kind: DecisionKind
    margin: float = Field(ge=0)
    reference_decisions: tuple[tuple[bool, ...], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _coherent_rule(self) -> Self:
        if self.reference_decisions != tuple(sorted(set(self.reference_decisions))):
            raise ValueError("reference decisions must be sorted and unique")
        if self.kind != "non_inferiority" and self.margin != 0:
            raise ValueError("superiority and fixed-sequence decisions test each target at its own null")
        if self.kind == "non_inferiority" and (
            self.margin <= 0 or {len(row) for row in self.reference_decisions} != {1}
        ):
            raise ValueError("a non-inferiority decision tests the primary target against a positive margin")
        return self


class TrialEvalGradingPlanV1(_Model):
    """The locked plan's primary analysis on the task's observed data (C1 and C3 only).

    The participant reports this analysis executed exactly as planned under ``target_id``, beside the
    analysis it would act on. It is graded apart, for execution accuracy: it never enters the numerical loss,
    the decision or its consequence.
    """

    target_id: Literal["prespecified_primary_analysis"]
    estimate: float
    lower: float
    upper: float
    tolerance_sd: float = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.lower > self.upper:
            raise ValueError("plan replay interval is not ordered")
        return self


class TrialEvalDataIntegrityKeyV1(_Model):
    """The resolution a C5 task's declared data-quality incident admits, by ordinary record keys."""

    affected_domain: str = Field(pattern=r"^data/raw/[A-Za-z0-9_.-]+\.parquet$")
    compound_key_fields: tuple[str, ...] = Field(min_length=1)
    resolution_action: str = Field(pattern=r"^[a-z][a-z0-9_]+$")
    resolution_status: Literal["clean", "repaired", "qualified_exclusion"]
    observed_issue_group_count: int = Field(ge=0)
    observed_affected_row_count: int = Field(ge=0)
    selected_affected_row_count: int = Field(ge=0)
    selected_observed_compound_keys: tuple[str, ...] = ()
    nonmaterial_to_primary_analysis: bool
    resolved_domain_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _coherent_keys(self) -> Self:
        if len(set(self.compound_key_fields)) != len(self.compound_key_fields):
            raise ValueError("compound-key fields repeat")
        keys = self.selected_observed_compound_keys
        if len(keys) != self.selected_affected_row_count or len(set(keys)) != len(keys):
            raise ValueError("selected compound keys differ from the selected affected-row count")
        return self


class TrialEvalGradingKeyRecordV1(_Reference):
    """Everything local scoring needs for one task, grouped by opaque scenario, clinical-question and world keys."""

    schema_id: Literal["trialagentbench.trialeval_grading_key_record/v1"]
    release_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    trial_id: str = Field(min_length=1)
    base_world_id: str = Field(pattern=r"^world_[0-9a-f]{20}$")
    cell_ref: str = Field(pattern=r"^cell_[0-9a-f]{20}$")
    series_ref: str = Field(pattern=r"^series_[0-9a-f]{20}$")
    context: Context
    effect_state: Literal["favourable", "near_null", "unfavourable"]
    design_tier: Literal["D1", "D2", "D3", "D4"]
    design_subtype: str = Field(min_length=1)
    design_family: str = Field(pattern=r"^[a-z][a-z0-9_]+$")
    participant_data_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    participant_artifact_paths: tuple[str, ...] = Field(min_length=1)
    additional_references: tuple[_Reference, ...] = ()
    decision: TrialEvalDecisionRuleV1 | None = None
    plan_replay: TrialEvalGradingPlanV1 | None = None
    data_integrity: TrialEvalDataIntegrityKeyV1 | None = None

    @property
    def references(self) -> tuple[_Reference, ...]:
        """Return the primary reference followed by required complementary results."""
        return (self, *self.additional_references)

    @property
    def target_ids(self) -> tuple[str, ...]:
        """Return the scored results in the assigned reporting order."""
        return tuple(reference.target.binding.estimand_id for reference in self.references)

    @property
    def reported_target_ids(self) -> tuple[str, ...]:
        """Return the scored targets, then the plan as executed where a plan is given."""
        return (*self.target_ids, *(() if self.plan_replay is None else (self.plan_replay.target_id,)))

    @model_validator(mode="after")
    def _plan_lane(self) -> Self:
        if self.participant_artifact_paths != tuple(sorted(set(self.participant_artifact_paths))):
            raise ValueError("participant artifact paths are not sorted and unique")
        if self.plan_replay is not None and self.context not in _PLAN_CONTEXTS:
            raise ValueError("a plan replay is present outside the plan lane")
        if (self.context == "C5") != (self.data_integrity is not None):
            raise ValueError("a data-integrity key belongs exactly to C5 tasks")
        if len(set(self.reported_target_ids)) != len(self.reported_target_ids):
            raise ValueError("evaluator targets repeat a result identifier")
        if self.decision is not None:
            tested = trialeval_decision_targets_v1(self.decision.kind, len(self.references))
            if any(len(row) != tested for row in self.decision.reference_decisions):
                raise ValueError("reference decisions do not test the declared targets")
        return self


class _Interval(_Model):
    lower: float
    upper: float
    confidence_level: float = Field(gt=0, lt=1)
    interval_type: Literal["central_confidence_interval", "repeated_confidence_interval"] = (
        "central_confidence_interval"
    )
    sidedness: Literal["two_sided_central"] = "two_sided_central"

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.lower > self.upper:
            raise ValueError("confidence interval is not ordered")
        return self


class _PointResult(_Model):
    kind: Literal["point"]
    value: float
    effect_measure: ResultEffectMeasure
    unit: str = Field(min_length=1)
    interval: _Interval | None = None
    conclusion: Conclusion


_Result = _PointResult


class TrialEvalTargetScoreInputV1(_Model):
    """Numerical observations for one declared result in a task."""

    target_id: str | None = Field(default=None, exclude_if=lambda value: value is None)
    attempt_status: AttemptStatus
    submitted_binding: _Question | None = None
    submitted_result: _Result | None = None
    evidence_source_artifacts: tuple[str, ...] = ()
    reproducible: bool | None = None
    uncertainty_failure: Literal["invalid_sampling_uncertainty"] | None = None

    @field_validator("evidence_source_artifacts", mode="before")
    @classmethod
    def _restore_evidence_paths(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _complete_attempt(self) -> Self:
        observations = (
            self.submitted_binding,
            self.submitted_result,
            self.reproducible,
            self.uncertainty_failure,
        )
        if self.attempt_status == "missing" and (
            any(item is not None for item in observations) or self.evidence_source_artifacts
        ):
            raise ValueError("missing attempts cannot carry observations")
        if self.attempt_status == "complete" and any(
            item is None for item in (self.submitted_result, self.reproducible)
        ):
            raise ValueError("complete attempts require a result and reproducibility")
        if self.uncertainty_failure is not None and self.submitted_result is None:
            raise ValueError("an uncertainty failure requires a retained result")
        if self.evidence_source_artifacts != tuple(sorted(set(self.evidence_source_artifacts))):
            raise ValueError("evidence source artifacts are not sorted and unique")
        return self


class TrialEvalScoreInputV1(TrialEvalTargetScoreInputV1):
    """Portable primary and complementary results from one committed analysis."""

    schema_id: Literal["trialagentbench.trialeval_score_input/v1"] = "trialagentbench.trialeval_score_input/v1"
    task_id: str = Field(min_length=1)
    additional_inputs: tuple[TrialEvalTargetScoreInputV1, ...] = Field(default=(), exclude_if=lambda value: not value)
    plan_input: TrialEvalTargetScoreInputV1 | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def _unique_results(self) -> Self:
        identities = tuple(item.target_id for item in self.additional_inputs)
        if identities and self.target_id is None:
            raise ValueError("Paired inputs require the primary target identity")
        if None in identities or len(set(identities)) != len(identities) or self.target_id in identities:
            raise ValueError("Complementary inputs require distinct target identities")
        if self.plan_input is not None and self.plan_input.target_id != "prespecified_primary_analysis":
            raise ValueError("The plan-as-executed input requires its own target identity")
        return self

    @property
    def numerical_inputs(self) -> tuple[TrialEvalTargetScoreInputV1, ...]:
        """Return every result committed by the analysis program."""
        return (self, *self.additional_inputs)

    @property
    def program_inputs(self) -> tuple[TrialEvalTargetScoreInputV1, ...]:
        """Return every program result in CSV order: requested targets, then the plan as executed."""
        return (*self.numerical_inputs, *(() if self.plan_input is None else (self.plan_input,)))


class VerifiedTrialEvalComponentV1(_Model):
    """One independently reconstructed score component."""

    component_id: ComponentId
    status: ComponentStatus
    failure_origin: str | None = None


class VerifiedTrialEvalTargetOutcomeV1(_Model):
    """Independent components and numerical diagnostics for one required target."""

    target_id: str
    context: Context
    components: tuple[VerifiedTrialEvalComponentV1, ...] = Field(min_length=6, max_length=6)
    metrics: dict[str, float | bool | str | None] | None = None
    signal_loss: float | None = Field(default=None, ge=0.0)


class VerifiedTrialEvalPlanExecutionV1(_Model):
    """Independent execution accuracy of the reported plan as executed.

    Deviations are absolute differences from the plan route's result on the comparison scale in grading
    standard deviations, absent when the plan as executed was not reported or not reproduced.
    """

    plan: TrialEvalGradingPlanV1
    point_deviation: float | None = Field(default=None, ge=0)
    lower_deviation: float | None = Field(default=None, ge=0)
    upper_deviation: float | None = Field(default=None, ge=0)
    accurate: bool


def trialeval_plan_execution_v1(
    *,
    plan: TrialEvalGradingPlanV1,
    target: _PointTarget,
    plan_input: TrialEvalTargetScoreInputV1 | None,
) -> VerifiedTrialEvalPlanExecutionV1:
    """Recompute plan execution accuracy from the reported numbers and the plan route's result.

    Parameters
    ----------
    plan
        The plan route's result on the task's data and its tolerance in grading standard deviations.
    target
        The primary target, whose scale and grading standard deviation the plan shares.
    plan_input
        The reported plan as executed, or None when it was not reported.

    Returns
    -------
    VerifiedTrialEvalPlanExecutionV1
        Accurate when the point and both interval endpoints lie within the tolerance.
    """
    result = None if plan_input is None else plan_input.submitted_result
    deviations: tuple[float | None, ...] = (None, None, None)
    if (
        plan_input is not None
        and plan_input.attempt_status == "complete"
        and plan_input.reproducible
        and result is not None
        and result.interval is not None
    ):
        measure = target.binding.effect_measure
        with localcontext() as context:
            context.prec = 80
            scale = Decimal.from_float(target.reference_sampling_standard_deviation)
            deviations = tuple(
                _finite_decimal(
                    abs(
                        Decimal.from_float(_comparison(reported, measure))
                        - Decimal.from_float(_comparison(planned, measure))
                    )
                    / scale
                )
                for reported, planned in (
                    (result.value, plan.estimate),
                    (result.interval.lower, plan.lower),
                    (result.interval.upper, plan.upper),
                )
            )
    point, lower, upper = deviations
    return VerifiedTrialEvalPlanExecutionV1(
        plan=plan,
        point_deviation=point,
        lower_deviation=lower,
        upper_deviation=upper,
        accurate=all(value is not None and value <= plan.tolerance_sd for value in deviations),
    )


class VerifiedTrialEvalOutcomeV1(_Model):
    """Task loss, completion and retained primary and complementary diagnostics."""

    schema_id: Literal["trialagentbench.trialeval_outcome_verification/v2"] = (
        "trialagentbench.trialeval_outcome_verification/v2"
    )
    task_id: str
    context: Context
    included_in_denominator: Literal[True] = True
    components: tuple[VerifiedTrialEvalComponentV1, ...] = Field(min_length=6, max_length=6)
    metrics: dict[str, float | bool | str | None] | None = None
    signal_loss: float | None = Field(default=None, ge=0.0)
    target_outcomes: tuple[VerifiedTrialEvalTargetOutcomeV1, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    decision: VerifiedTrialEvalComponentV1 | None = Field(default=None, exclude_if=lambda value: value is None)
    plan_execution: VerifiedTrialEvalPlanExecutionV1 | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


def _component(
    component_id: ComponentId,
    status: ComponentStatus,
    failure: str | None = None,
) -> VerifiedTrialEvalComponentV1:
    return VerifiedTrialEvalComponentV1(component_id=component_id, status=status, failure_origin=failure)


def _missing_origin(status: AttemptStatus) -> str:
    return {
        "missing": "missing_submission",
        "missing_required_output": "missing_required_output",
        "incomplete": "incomplete_submission",
        "complete": "incomplete_submission",
    }[status]


def _comparison(value: float, measure: DeclaredEffectMeasure) -> float:
    if measure != "rate_ratio":
        return value
    if value <= 0:
        raise TrialEvalOutcomeVerificationError("rate-ratio values must be positive")
    return math.log(value)


def _expected_conclusion(result: _PointResult, binding: _Question) -> Conclusion:
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


def _finite_decimal(value: Decimal) -> float | None:
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _interval_loss(lower: Decimal, upper: Decimal, target: Decimal, confidence: Decimal) -> Decimal:
    return upper - lower + 2 * max(lower - target, target - upper, Decimal(0)) / (1 - confidence)


def _metrics(target: _PointTarget, result: _PointResult) -> dict[str, float | bool | str | None]:
    """Recompute diagnostics without importing the producer's scoring equations."""
    measure = target.binding.effect_measure
    with localcontext() as context:
        context.prec = 80
        truth = Decimal.from_float(target.value)
        estimate = Decimal.from_float(result.value)
        comparison_truth = Decimal.from_float(_comparison(target.value, measure))
        comparison_estimate = Decimal.from_float(_comparison(result.value, measure))
        error = comparison_estimate - comparison_truth
        normalized = error / Decimal.from_float(target.reference_sampling_standard_deviation)
        metrics: dict[str, float | bool | str | None] = {
            "comparison_scale": "log" if measure == "rate_ratio" else "native",
            "native_signed_error": _finite_decimal(estimate - truth),
            "native_absolute_error": _finite_decimal(abs(estimate - truth)),
            "comparison_signed_error": _finite_decimal(error),
            "comparison_absolute_error": _finite_decimal(abs(error)),
            "normalized_absolute_error": _finite_decimal(abs(normalized)),
            "normalized_squared_error": _finite_decimal(normalized * normalized),
        }
        if result.interval is not None:
            lower = Decimal.from_float(_comparison(result.interval.lower, measure))
            upper = Decimal.from_float(_comparison(result.interval.upper, measure))
            native_lower = Decimal.from_float(result.interval.lower)
            native_upper = Decimal.from_float(result.interval.upper)
            confidence = Decimal.from_float(result.interval.confidence_level)
            metrics.update(
                {
                    "interval_covers_oracle": result.interval.lower <= target.value <= result.interval.upper,
                    "interval_width": _finite_decimal(upper - lower),
                    "native_interval_width": _finite_decimal(native_upper - native_lower),
                    "proper_interval_score": _finite_decimal(
                        _interval_loss(lower, upper, comparison_truth, confidence)
                    ),
                    "native_interval_score": _finite_decimal(
                        _interval_loss(native_lower, native_upper, truth, confidence)
                    ),
                }
            )
    return metrics


def _load_record(root: Path, task_id: str) -> TrialEvalGradingKeyRecordV1:
    from trialagentbench_verifier.grading_key import read_trialeval_grading_key

    matches = tuple(record for record in read_trialeval_grading_key(root)[1] if record.task_id == task_id)
    if len(matches) != 1:
        raise TrialEvalOutcomeVerificationError("grading key does not contain the task exactly once")
    return matches[0]


def _question_matches(submitted: _Question, target: _Question) -> bool:
    """Compare every declared part of the clinical question."""

    return submitted == target


def _estimand_core_matches(submitted: _Question, target: _Question) -> bool:
    """Compare the clinical quantity without gating on identification prose."""

    excluded = {"identification_status", "identification_assumption_ids"}
    return submitted.model_dump(mode="python", exclude=excluded) == target.model_dump(
        mode="python",
        exclude=excluded,
    )


def _result_scale_matches(result: _PointResult, target: _Question) -> bool:
    """Check the scale and unit declared by the numerical result."""

    return result.effect_measure == target.effect_measure and result.unit == target.result_unit


def verify_trialeval_outcome_v1(
    *,
    release_root: Path,
    score_input: TrialEvalScoreInputV1,
    authenticated_execution_reproduced: Literal[True] | None = True,
) -> VerifiedTrialEvalOutcomeV1:
    """Recompute estimator-neutral TrialEval components from portable bytes."""

    record = _load_record(Path(release_root), score_input.task_id)
    primary_id = record.target.binding.estimand_id
    if score_input.target_id not in (None, primary_id):
        raise TrialEvalOutcomeVerificationError("Primary input identifies a different target")
    expected = tuple(reference.target.binding.estimand_id for reference in record.references)
    if len(set(expected)) != len(expected):
        raise TrialEvalOutcomeVerificationError("Evaluator references repeat a target")
    supplied = {primary_id: score_input, **{item.target_id: item for item in score_input.additional_inputs}}
    if primary_id in {item.target_id for item in score_input.additional_inputs} or not set(supplied) <= set(expected):
        raise TrialEvalOutcomeVerificationError("Score inputs contain duplicate or unassigned targets")
    if score_input.plan_input is not None and record.plan_replay is None:
        raise TrialEvalOutcomeVerificationError("A plan-as-executed input requires a plan-given task")
    results = tuple(
        _verify_target(
            record=record,
            target=reference.target,
            score_input=supplied.get(
                reference.target.binding.estimand_id,
                TrialEvalTargetScoreInputV1(attempt_status="missing_required_output"),
            ),
            authenticated_execution_reproduced=authenticated_execution_reproduced,
        )
        for reference in record.references
    )
    losses = tuple(result.signal_loss for result in results)
    components = tuple(
        next((item for item in group if item.status in {"not_satisfied", "not_evaluable"}), group[0])
        for group in zip(*(result.components for result in results), strict=True)
    )
    return VerifiedTrialEvalOutcomeV1(
        task_id=score_input.task_id,
        context=record.context,
        components=components,
        metrics=results[0].metrics,
        signal_loss=(
            None if any(value is None for value in losses) else mean(value for value in losses if value is not None)
        ),
        target_outcomes=results if len(results) > 1 else (),
        decision=(
            None
            if record.decision is None
            else _verify_decision(record=record, results=results, supplied=supplied, rule=record.decision)
        ),
        plan_execution=(
            None
            if record.plan_replay is None
            else trialeval_plan_execution_v1(
                plan=record.plan_replay, target=record.target, plan_input=score_input.plan_input
            )
        ),
    )


def _verify_decision(
    *,
    record: TrialEvalGradingKeyRecordV1,
    results: tuple[VerifiedTrialEvalTargetOutcomeV1, ...],
    supplied: dict[str | None, TrialEvalTargetScoreInputV1],
    rule: TrialEvalDecisionRuleV1,
) -> VerifiedTrialEvalComponentV1:
    """Apply the declared decision rule to the submitted intervals and compare it with the reference."""
    intervals = []
    for reference, result in zip(record.references, results, strict=True):
        uncertainty = next(item for item in result.components if item.component_id == "uncertainty")
        if uncertainty.status != "observed":
            return _component("decision", "not_evaluable", uncertainty.failure_origin)
        submitted = supplied[reference.target.binding.estimand_id].submitted_result
        assert submitted is not None and submitted.interval is not None
        intervals.append((submitted.interval.lower, submitted.interval.upper))
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


def _verify_target(
    *,
    record: TrialEvalGradingKeyRecordV1,
    target: TrialEvalTargetV1,
    score_input: TrialEvalTargetScoreInputV1,
    authenticated_execution_reproduced: Literal[True] | None,
) -> VerifiedTrialEvalTargetOutcomeV1:
    """Apply the same independent numerical calculation to each assigned target."""
    evidence_eligible = bool(score_input.evidence_source_artifacts) and set(
        score_input.evidence_source_artifacts
    ) <= set(record.participant_artifact_paths)
    missing = score_input.attempt_status != "complete"
    missing_origin = _missing_origin(score_input.attempt_status)
    completion = _component(
        "completion", "not_satisfied" if missing else "satisfied", missing_origin if missing else None
    )
    result = score_input.submitted_result
    if result is None:
        question = _component("question_alignment", "not_evaluable", missing_origin)
    elif score_input.submitted_binding is not None and not _question_matches(
        score_input.submitted_binding, target.binding
    ):
        question = _component("question_alignment", "not_satisfied", "wrong_question")
    elif _result_scale_matches(result, target.binding):
        question = _component("question_alignment", "observed")
    else:
        question = _component("question_alignment", "not_satisfied", "wrong_question")
    evidence = (
        _component("evidence_eligibility", "not_evaluable", missing_origin)
        if missing
        else _component(
            "evidence_eligibility",
            "satisfied" if evidence_eligible else "not_satisfied",
            None if evidence_eligible else "prohibited_evidence",
        )
    )
    reproducibility = (
        _component("reproducibility", "not_evaluable", missing_origin)
        if missing
        else (
            _component("reproducibility", "not_evaluable", "execution_proof_required")
            if authenticated_execution_reproduced is None
            else _component("reproducibility", "satisfied")
        )
    )
    numerical_failure: str | None = None
    if missing:
        numerical_failure = missing_origin
    elif score_input.submitted_binding is not None and not _estimand_core_matches(
        score_input.submitted_binding, target.binding
    ):
        numerical_failure = "wrong_question"
    if (
        numerical_failure is None
        and isinstance(result, _PointResult)
        and not _result_scale_matches(result, target.binding)
    ):
        numerical_failure = "wrong_question"
    metrics = None if numerical_failure is not None or result is None else _metrics(target, result)
    estimand = _component(
        "estimand_performance",
        "not_evaluable" if numerical_failure else "observed",
        numerical_failure,
    )
    uncertainty_failure = numerical_failure or score_input.uncertainty_failure
    if uncertainty_failure is None and isinstance(target, _PointTarget):
        assert isinstance(result, _PointResult)
        if result.interval is None:
            uncertainty_failure = "missing_sampling_uncertainty"
        elif result.interval.confidence_level != target.confidence_level:
            uncertainty_failure = "confidence_level_mismatch"
        elif result.interval.interval_type != target.binding.uncertainty_type:
            uncertainty_failure = "interval_type_mismatch"
    uncertainty_status: ComponentStatus = "not_evaluable" if uncertainty_failure else "observed"
    uncertainty = _component("uncertainty", uncertainty_status, uncertainty_failure)
    components = (completion, question, evidence, reproducibility, estimand, uncertainty)
    recovery = None
    if all(component.status in {"satisfied", "observed"} for component in components):
        assert result is not None
        recovery = _signal_loss(target, result)
    return VerifiedTrialEvalTargetOutcomeV1(
        target_id=target.binding.estimand_id,
        context=record.context,
        components=components,
        metrics=metrics,
        signal_loss=recovery,
    )


def _signal_loss(target: TrialEvalTargetV1, result: _Result) -> float | None:
    """Independently recompute mean point/interval loss before any display transform.

    A target whose point is reported but not scored (``point_scored`` false, TE-S09) keeps only the
    interval half of the loss.
    """
    if result.interval is None:
        return None
    measure = target.binding.effect_measure
    with localcontext(NUMERICAL_LOSS_CONTEXT):
        estimate, lower, upper, truth = (
            Decimal.from_float(_comparison(value, measure))
            for value in (result.value, result.interval.lower, result.interval.upper, target.value)
        )
        scale = Decimal.from_float(target.reference_sampling_standard_deviation)
        confidence = Decimal.from_float(result.interval.confidence_level)
        point = abs(estimate - truth) if target.point_scored else Decimal(0)
        loss = (point + _interval_loss(lower, upper, truth, confidence)) / (2 * scale)
    return round_numerical_loss(loss)


__all__ = [
    "TrialEvalDataIntegrityKeyV1",
    "TrialEvalDecisionRuleV1",
    "TrialEvalGradingKeyRecordV1",
    "TrialEvalGradingPlanV1",
    "TrialEvalOutcomeVerificationError",
    "TrialEvalScoreInputV1",
    "VerifiedTrialEvalOutcomeV1",
    "VerifiedTrialEvalPlanExecutionV1",
    "trialeval_decision_targets_v1",
    "trialeval_decisions_v1",
    "trialeval_plan_execution_v1",
    "verify_trialeval_outcome_v1",
]
