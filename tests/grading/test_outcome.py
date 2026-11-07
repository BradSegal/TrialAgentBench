"""Estimator-neutral TrialEval outcome grading."""

from __future__ import annotations

import math

import pytest

from trialagentbench.grading.outcome import (
    TrialEvalConfidenceIntervalV1,
    TrialEvalOutcomeGradingInputV1,
    TrialEvalPointEstimateV1,
    TrialEvalPointOracleTargetV1,
    TrialEvalQuestionBindingV1,
    grade_trialeval_outcome_v1,
)


def _binding(*, set_target: bool = False, repeated: bool = False) -> TrialEvalQuestionBindingV1:
    return TrialEvalQuestionBindingV1(
        analysis_population_id="intention_to_treat",
        estimand_id="primary_effect",
        intercurrent_event_strategy_ids=("treatment_policy",),
        assessment_horizon_days=365.0,
        treatment_id="active",
        comparator_id="control",
        endpoint_id="primary_endpoint",
        effect_measure="risk_difference" if set_target else "mean_difference",
        analysis_unit="participant",
        population_basis="superpopulation",
        identification_status="partially_identified" if set_target else "point_identified",
        identification_assumption_ids=("bounded_deviation",) if set_target else (),
        result_unit="proportion" if set_target else "points",
        uncertainty_type=(
            "simultaneous_outer_confidence_region"
            if set_target
            else "repeated_confidence_interval" if repeated else "central_confidence_interval"
        ),
        confidence_level=0.95,
        sidedness="simultaneous_outer" if set_target else "two_sided_central",
        null_value=0.0,
        favorable_direction="higher",
    )


def _point_target(*, repeated: bool = False) -> TrialEvalPointOracleTargetV1:
    return TrialEvalPointOracleTargetV1(
        binding=_binding(repeated=repeated),
        value=10.0,
        confidence_level=0.95,
        reference_sampling_standard_deviation=2.0,
        reference_sampling_basis_id="finite_world_oracle_contrast",
    )


def _grade_point(result: TrialEvalPointEstimateV1, *, target: TrialEvalPointOracleTargetV1 | None = None):
    resolved = target or _point_target()
    return grade_trialeval_outcome_v1(
        TrialEvalOutcomeGradingInputV1(
            context="C2",
            target=resolved,
            attempt_status="complete",
            submitted_result=result,
            evidence_eligible=True,
            reproducible=True,
        )
    )


def _point(*, interval: TrialEvalConfidenceIntervalV1 | None) -> TrialEvalPointEstimateV1:
    return TrialEvalPointEstimateV1(
        value=12.0,
        effect_measure="mean_difference",
        unit="points",
        interval=interval,
        conclusion="favors_treatment",
    )


def test_minimal_result_is_task_bound_without_fabricated_alignment() -> None:
    grade = _grade_point(_point(interval=TrialEvalConfidenceIntervalV1(lower=8.0, upper=14.0, confidence_level=0.95)))

    assert grade.question_alignment.status == "observed"
    assert grade.estimand_performance.status == "observed"
    assert grade.uncertainty.status == "observed"
    assert grade.point_performance is not None
    assert grade.point_performance.normalized_absolute_error == 1.0


def test_current_state_point_without_interval_keeps_point_only() -> None:
    grade = _grade_point(_point(interval=None))

    assert grade.estimand_performance.status == "observed"
    assert grade.point_performance is not None
    assert grade.uncertainty.failure_origin == "missing_sampling_uncertainty"
    assert grade.interval_performance is None


def test_current_state_wrong_confidence_level_keeps_point_only() -> None:
    grade = _grade_point(_point(interval=TrialEvalConfidenceIntervalV1(lower=8.0, upper=14.0, confidence_level=0.90)))

    assert grade.estimand_performance.status == "observed"
    assert grade.uncertainty.failure_origin == "confidence_level_mismatch"


def test_current_state_wrong_interval_type_keeps_point_only() -> None:
    grade = _grade_point(
        _point(interval=TrialEvalConfidenceIntervalV1(lower=8.0, upper=14.0, confidence_level=0.95)),
        target=_point_target(repeated=True),
    )

    assert grade.estimand_performance.status == "observed"
    assert grade.uncertainty.failure_origin == "interval_type_mismatch"


def test_invalid_optional_uncertainty_failure_retains_point_metrics() -> None:
    result = _point(interval=None)
    grade = grade_trialeval_outcome_v1(
        TrialEvalOutcomeGradingInputV1(
            context="C2",
            target=_point_target(),
            attempt_status="complete",
            submitted_result=result,
            evidence_eligible=True,
            reproducible=True,
            uncertainty_failure="invalid_sampling_uncertainty",
        )
    )

    assert grade.estimand_performance.status == "observed"
    assert grade.uncertainty.failure_origin == "invalid_sampling_uncertainty"


def test_wrong_result_scale_is_a_real_question_mismatch() -> None:
    result = _point(interval=TrialEvalConfidenceIntervalV1(lower=8.0, upper=14.0, confidence_level=0.95))
    grade = _grade_point(result.model_copy(update={"unit": "days"}))

    assert grade.question_alignment.status == "not_satisfied"
    assert grade.estimand_performance.failure_origin == "wrong_question"


def test_rate_ratio_metrics_use_log_comparison_scale() -> None:
    binding = _binding().model_copy(update={"effect_measure": "rate_ratio", "result_unit": "ratio", "null_value": 1.0})
    target = _point_target().model_copy(
        update={"binding": binding, "value": 2.0, "reference_sampling_standard_deviation": 0.5}
    )
    result = TrialEvalPointEstimateV1(
        value=4.0,
        effect_measure="rate_ratio",
        unit="ratio",
        interval=TrialEvalConfidenceIntervalV1(lower=1.0, upper=8.0, confidence_level=0.95),
        conclusion="no_clear_difference",
    )

    grade = _grade_point(result, target=target)

    assert grade.point_performance is not None
    assert grade.point_performance.comparison_signed_error == pytest.approx(math.log(2.0))
    assert grade.interval_performance is not None
    assert grade.interval_performance.comparison_width == pytest.approx(math.log(8.0))
    assert grade.interval_performance.native_interval_score == pytest.approx(7.0)
    expected = math.log(2.0) + math.log(8.0)
    assert grade.signal_loss == pytest.approx(expected)
    from pydantic import TypeAdapter
    from trialagentbench_verifier.trialeval import TrialEvalTargetV1, _Result, _signal_loss

    assert _signal_loss(
        TypeAdapter(TrialEvalTargetV1).validate_json(target.model_dump_json()),
        TypeAdapter(_Result).validate_json(result.model_dump_json()),
    ) == pytest.approx(expected)


def test_independent_losses_agree_at_an_output_rounding_boundary() -> None:
    """Binary arithmetic before final rounding can cross a half-even boundary."""
    from pydantic import TypeAdapter
    from trialagentbench_verifier.trialeval import TrialEvalTargetV1, _Result, _signal_loss

    target = _point_target().model_copy(
        update={"value": 0.24908681922709874, "reference_sampling_standard_deviation": 1.0}
    )
    result = TrialEvalPointEstimateV1(
        value=9.059075834998605,
        effect_measure="mean_difference",
        unit="points",
        interval=TrialEvalConfidenceIntervalV1(
            lower=-4.5865682384008455, upper=6.404661812657647, confidence_level=0.95
        ),
        conclusion="no_clear_difference",
    )
    independent = _signal_loss(
        TypeAdapter(TrialEvalTargetV1).validate_json(target.model_dump_json()),
        TypeAdapter(_Result).validate_json(result.model_dump_json()),
    )
    assert _grade_point(result, target=target).signal_loss == independent == 9.90060953341


@pytest.mark.parametrize("context", ["C1", "C3", "C5"])
def test_numerical_grading_does_not_adjudicate_plan_adherence(context: str) -> None:
    grade = grade_trialeval_outcome_v1(
        TrialEvalOutcomeGradingInputV1(
            context=context,
            target=_point_target(),
            attempt_status="complete",
            submitted_result=_point(
                interval=TrialEvalConfidenceIntervalV1(lower=8.0, upper=14.0, confidence_level=0.95)
            ),
            evidence_eligible=True,
            reproducible=True,
        )
    )

    assert "plan_adherence" not in grade.model_dump()
    assert not grade.failure_origins
    assert grade.signal_loss > 0


@pytest.mark.parametrize("lower,upper", [(8.0, 14.0), (-1000.0, 1000.0), (11.99, 12.01)])
def test_recovery_penalizes_vacuous_and_narrow_wrong_intervals(lower: float, upper: float) -> None:
    """Evaluate the fixed location/uncertainty equation against hand arithmetic."""

    result = _point(interval=TrialEvalConfidenceIntervalV1(lower=lower, upper=upper, confidence_level=0.95))
    grade = _grade_point(result)
    loss = (abs(result.value - 10.0) + upper - lower + 40 * max(0, lower - 10.0, 10.0 - upper)) / 4.0
    assert grade.signal_loss == pytest.approx(loss)
    if lower == 8.0:
        assert grade.signal_loss == pytest.approx(2)
    else:
        assert grade.signal_loss > 2


@pytest.mark.parametrize("failed_boundary", ["evidence_eligible", "reproducible"])
def test_recovery_is_zero_for_ineligible_exact_numbers(failed_boundary: str) -> None:
    """Exact oracle numbers cannot bypass the evidence and execution boundaries."""

    request = TrialEvalOutcomeGradingInputV1(
        context="C2",
        target=_point_target(),
        attempt_status="complete",
        submitted_result=_point(interval=TrialEvalConfidenceIntervalV1(lower=8, upper=14, confidence_level=0.95)),
        evidence_eligible=True,
        reproducible=True,
    ).model_copy(update={failed_boundary: False})
    assert grade_trialeval_outcome_v1(request).signal_loss is None


@pytest.mark.parametrize("kind", ["identified_set", "non_identification"])
def test_unsupported_answer_kinds_are_rejected(kind: str) -> None:
    """Every released task requests an estimate and an interval."""
    from pydantic import ValidationError

    from trialagentbench.grading.outcome import TrialEvalSubmittedResultV1

    with pytest.raises(ValidationError):
        TrialEvalSubmittedResultV1.model_validate({"kind": kind, "lower": 0, "upper": 1})


def _decision_record(kind: str, margin: float, reference_decisions: tuple[tuple[bool, ...], ...]):
    from trialagentbench_verifier.trialeval import TrialEvalDecisionRuleV1

    from trialagentbench.grading.outcome import (
        TrialEvalEvaluatorRecordV1,
        TrialEvalOracleReferenceV1,
    )

    def reference(estimand_id: str) -> TrialEvalOracleReferenceV1:
        target = _point_target()
        return TrialEvalOracleReferenceV1(
            target=target.model_copy(
                update={"binding": target.binding.model_copy(update={"estimand_id": estimand_id})}
            ),
            oracle_worlds=50,
            oracle_monte_carlo_standard_error=0.01,
            maximum_oracle_monte_carlo_standard_error=0.1,
            reference_analysis_ids=("observed:route",),
        )

    targets = ("primary_effect", "key_secondary") if kind == "fixed_sequence" else ("primary_effect",)
    return TrialEvalEvaluatorRecordV1(
        release_id="release",
        task_id="te_" + "0" * 20,
        trial_id="trial_" + "0" * 20,
        base_world_id="world_" + "0" * 20,
        context="C2",
        participant_data_sha256="0" * 64,
        participant_artifact_paths=("data/analysis.parquet",),
        **reference(targets[0]).model_dump(),
        additional_references=tuple(reference(item) for item in targets[1:]),
        decision=TrialEvalDecisionRuleV1(kind=kind, margin=margin, reference_decisions=reference_decisions),
    )


def _decision_inputs(record, intervals):
    inputs = tuple(
        TrialEvalOutcomeGradingInputV1(
            context="C2",
            target=reference.target,
            attempt_status="complete",
            submitted_result=_point(
                interval=(
                    None
                    if interval is None
                    else TrialEvalConfidenceIntervalV1(lower=interval[0], upper=interval[1], confidence_level=0.95)
                )
            ),
            evidence_eligible=True,
            reproducible=True,
        )
        for reference, interval in zip(record.references, intervals, strict=True)
    )
    return inputs, tuple(grade_trialeval_outcome_v1(item) for item in inputs)


@pytest.mark.parametrize(
    ("intervals", "status"),
    [(((1.0, 5.0), (0.5, 2.0)), "satisfied"), (((-1.0, 5.0), (0.5, 2.0)), "not_satisfied")],
)
def test_fixed_sequence_decision_is_scored_against_reference_decisions(intervals, status) -> None:
    from trialagentbench.grading.outcome import grade_trialeval_decision_v1

    record = _decision_record("fixed_sequence", 0.0, ((True, True),))
    decision = grade_trialeval_decision_v1(record, *_decision_inputs(record, intervals))
    assert decision is not None and decision.status == status
    assert decision.failure_origin == (None if status == "satisfied" else "decision_mismatch")


def test_non_inferiority_decision_uses_the_margin_and_needs_an_interval() -> None:
    from trialagentbench.grading.outcome import grade_trialeval_decision_v1

    record = _decision_record("non_inferiority", 0.5, ((True,),))
    # Higher is favourable: non-inferiority holds when the lower limit exceeds -0.5.
    decision = grade_trialeval_decision_v1(record, *_decision_inputs(record, ((-0.4, 3.0),)))
    assert decision is not None and decision.status == "satisfied"
    missing = grade_trialeval_decision_v1(record, *_decision_inputs(record, (None,)))
    assert missing is not None and missing.status == "not_evaluable"
    assert missing.failure_origin == "missing_sampling_uncertainty"
    with pytest.raises(ValueError, match="positive margin"):
        _decision_record("non_inferiority", 0.0, ((True,),))
    with pytest.raises(ValueError, match="do not test the declared targets"):
        _decision_record("fixed_sequence", 0.0, ((True,),))


def test_unscored_point_is_reported_but_carries_no_numerical_loss() -> None:
    """An unscored point is reported but scored only through its interval and decision."""
    unscored = TrialEvalPointOracleTargetV1.model_validate(
        _point_target(repeated=True).model_dump() | {"point_scored": False}
    )
    interval = TrialEvalConfidenceIntervalV1(
        lower=8.0, upper=13.0, confidence_level=0.95, interval_type="repeated_confidence_interval"
    )
    near, far = (
        _grade_point(_point(interval=interval).model_copy(update={"value": value}), target=unscored)
        for value in (10.1, 12.9)
    )
    assert near.signal_loss == far.signal_loss == pytest.approx(5.0 / 2.0 / 2.0)
    assert far.estimand_performance.status == "observed" and far.point_performance is not None
    assert far.point_performance.native_absolute_error == pytest.approx(2.9)
    scored = _grade_point(
        _point(interval=interval).model_copy(update={"value": 12.9}), target=_point_target(repeated=True)
    )
    assert scored.signal_loss == pytest.approx((2.9 + 5.0) / 2.0 / 2.0)
    assert "point_scored" not in _point_target().model_dump(mode="json")
    assert unscored.model_dump(mode="json")["point_scored"] is False


@pytest.mark.parametrize(("interval", "status"), [((0.5, 3.0), "satisfied"), ((-0.5, 3.0), "not_satisfied")])
def test_superiority_decision_is_the_efficacy_conclusion_of_the_interval(interval, status) -> None:
    """Efficacy is concluded when the whole interval lies on the favourable side of the null."""
    from trialagentbench.grading.outcome import grade_trialeval_decision_v1

    record = _decision_record("superiority", 0.0, ((True,),))
    decision = grade_trialeval_decision_v1(record, *_decision_inputs(record, (interval,)))
    assert decision is not None and decision.status == status
    with pytest.raises(ValueError, match="test each target at its own null"):
        _decision_record("superiority", 0.1, ((True,),))
    with pytest.raises(ValueError, match="do not test the declared targets"):
        _decision_record("superiority", 0.0, ((True, True),))


@pytest.mark.parametrize(
    "measure,unit,spread",
    [
        ("mean_difference", "points", 1.0),
        ("risk_difference", "proportion", 0.1),
        ("rate_ratio", "ratio", 0.5),
        ("restricted_mean_survival_time_difference", "days", 30.0),
    ],
)
@pytest.mark.parametrize("point_scored", [True, False])
def test_final_target_loss_is_minimized_by_median_and_central_quantiles(measure, unit, spread, point_scored):
    """Exhaust all reports on a finite grid through both final scoring implementations.

    Absolute error elicits a median, and interval score elicits the central
    quantiles. Ratio reports elicit these on the disclosed log comparison scale.
    The 41 equiprobable truths have unique median zero and 2.5/97.5% quantiles
    -0.95/+0.95. Scored and interval-only targets use the same quantiles.
    """
    from itertools import product
    from statistics import mean

    from pydantic import TypeAdapter
    from trialagentbench_verifier.trialeval import TrialEvalTargetV1, _Result, _signal_loss

    binding = TrialEvalQuestionBindingV1.model_validate(
        _binding().model_dump()
        | {"effect_measure": measure, "result_unit": unit, "null_value": float(measure == "rate_ratio")}
    )

    def native(value):
        return math.exp(value * spread) if measure == "rate_ratio" else value * spread

    targets = tuple(
        TrialEvalPointOracleTargetV1(
            binding=binding,
            value=native(index / 20),
            confidence_level=0.95,
            reference_sampling_standard_deviation=spread,
            reference_sampling_basis_id="finite_world_oracle_contrast",
            point_scored=point_scored,
        )
        for index in range(-20, 21)
    )
    verifier_targets = tuple(
        TypeAdapter(TrialEvalTargetV1).validate_json(target.model_dump_json()) for target in targets
    )

    def expected_loss(point, lower, upper):
        result = TrialEvalPointEstimateV1(
            value=native(point),
            effect_measure=measure,
            unit=unit,
            interval=TrialEvalConfidenceIntervalV1(lower=native(lower), upper=native(upper), confidence_level=0.95),
            conclusion="no_clear_difference",
        )
        independent_result = TypeAdapter(_Result).validate_json(result.model_dump_json())
        losses = []
        for target, verifier_target in zip(targets, verifier_targets, strict=True):
            loss = _grade_point(result, target=target).signal_loss
            assert loss is not None
            assert loss == pytest.approx(_signal_loss(verifier_target, independent_result))
            losses.append(loss)
        return mean(losses)

    optimal = expected_loss(0, -0.95, 0.95)
    # Point and interval terms are separable, so these grids exhaust their joint product.
    for point in (-0.8, -0.4, 0, 0.4, 0.8):
        candidate = expected_loss(point, -0.95, 0.95)
        assert candidate >= optimal - 1e-12
        if point_scored and point != 0:
            assert candidate > optimal
        elif not point_scored:
            assert candidate == pytest.approx(optimal)
    for lower, upper in product((-1.1, -1, -0.95, -0.9, -0.8, -0.5), (0.5, 0.8, 0.9, 0.95, 1, 1.1)):
        candidate = expected_loss(0, lower, upper)
        assert candidate >= optimal - 1e-12, (lower, upper)
        if (lower, upper) != (-0.95, 0.95):
            assert candidate > optimal


@pytest.mark.parametrize("field", ["value", "lower", "upper"])
@pytest.mark.parametrize("nonpositive", [0.0, -1.0])
def test_final_ratio_scoring_refuses_nonpositive_values(field, nonpositive):
    """Neither scorer silently logs, clips or assigns credit to a nonpositive ratio."""
    from pydantic import TypeAdapter
    from trialagentbench_verifier.trialeval import (
        TrialEvalOutcomeVerificationError,
        TrialEvalTargetV1,
        _Result,
        _signal_loss,
    )

    binding = TrialEvalQuestionBindingV1.model_validate(
        _binding().model_dump() | {"effect_measure": "rate_ratio", "result_unit": "ratio", "null_value": 1.0}
    )
    target = TrialEvalPointOracleTargetV1.model_validate(
        _point_target().model_dump() | {"binding": binding, "value": 1.0}
    )
    endpoints = {"lower": 0.5, "upper": 2.0}
    endpoints[field] = nonpositive
    if field == "upper":
        endpoints["lower"] = nonpositive - 1
    result = TrialEvalPointEstimateV1(
        value=nonpositive if field == "value" else 1.0,
        effect_measure="rate_ratio",
        unit="ratio",
        interval=TrialEvalConfidenceIntervalV1(
            lower=endpoints["lower"], upper=endpoints["upper"], confidence_level=0.95
        ),
        conclusion="no_clear_difference",
    )
    with pytest.raises(ValueError, match="positive"):
        _grade_point(result, target=target)
    with pytest.raises(TrialEvalOutcomeVerificationError, match="positive"):
        _signal_loss(
            TypeAdapter(TrialEvalTargetV1).validate_json(target.model_dump_json()),
            TypeAdapter(_Result).validate_json(result.model_dump_json()),
        )
