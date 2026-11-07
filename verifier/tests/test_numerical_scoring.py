"""Numerical recovery uses consistent units and penalizes missed reference values."""

import pytest

from trialagentbench_verifier.numerical_scoring import interval_score, numerical_loss


@pytest.mark.parametrize(
    "value,expected",
    [("1.234567890125", 1.23456789012), ("1.234567890135", 1.23456789014), ("1e-200", 1e-200), ("1e308", 1e308)],
)
def test_target_loss_precision_and_half_even_rule(value, expected):
    from decimal import ROUND_UP, Decimal, localcontext

    from trialagentbench_verifier.numerical_scoring import round_numerical_loss

    with localcontext() as context:
        context.prec = 3
        context.rounding = ROUND_UP
        assert round_numerical_loss(Decimal(value)) == expected
        assert round_numerical_loss(float(value)) == expected


def test_recovery_is_invariant_to_units_and_direction():
    original = numerical_loss(estimate=1.2, lower=0.5, upper=1.5, target=1, scale=2, confidence_level=0.95)
    assert numerical_loss(
        estimate=120, lower=50, upper=150, target=100, scale=200, confidence_level=0.95
    ) == pytest.approx(original)
    assert numerical_loss(
        estimate=-1.2, lower=-1.5, upper=-0.5, target=-1, scale=2, confidence_level=0.95
    ) == pytest.approx(original)


def test_interval_noncoverage_and_point_error_both_increase_loss():
    assert interval_score(0, 2, 1, 0.95) == 2
    assert interval_score(2, 3, 1, 0.95) == pytest.approx(41)
    assert numerical_loss(estimate=1, lower=0, upper=2, target=1, scale=1, confidence_level=0.95) == 1
    assert numerical_loss(estimate=2, lower=2, upper=3, target=1, scale=1, confidence_level=0.95) == pytest.approx(21)
    # The estimate is not required to lie inside the reported interval.
    assert numerical_loss(estimate=1, lower=2, upper=3, target=1, scale=1, confidence_level=0.95) == pytest.approx(
        20.5
    )


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf")])
def test_invalid_normalization_is_an_evaluator_error(scale):
    with pytest.raises(ValueError, match="scale"):
        numerical_loss(estimate=0, lower=-1, upper=1, target=0, scale=scale, confidence_level=0.95)


@pytest.mark.parametrize(
    "estimate,lower,upper,target,scale",
    [
        (1e308, 0, 1e308, 0, 1e308),
        (1e308, -1e308, 1e308, 0, 0.01),
        (-1e308, -1e308, 0, 1e308, 1e308),
        (1e200, -1, 1, 0, 1),
        (1e308, -1e308, 1e308, 0, 5e-324),
    ],
)
def test_extreme_finite_answers_match_decimal_reference(
    estimate: float, lower: float, upper: float, target: float, scale: float
) -> None:
    import math
    from decimal import Decimal, localcontext

    with localcontext() as context:
        context.prec = 80
        e, lo, hi, t, s, c = map(Decimal.from_float, (estimate, lower, upper, target, scale, 0.95))
        width = hi - lo
        penalty = 2 * max(lo - t, t - hi, Decimal(0)) / (1 - c)
        expected = float((abs(e - t) + width + penalty) / (2 * s))
    actual = numerical_loss(
        estimate=estimate, lower=lower, upper=upper, target=target, scale=scale, confidence_level=0.95
    )
    if math.isfinite(expected):
        assert actual == pytest.approx(expected, rel=1e-12, abs=5e-324)
    else:
        assert actual is None


def test_mean_loss_preserves_interval_incentive_and_explicit_missingness():
    """Aggregation must not reproduce the bounded-score undercoverage incentive."""
    from math import log

    from trialagentbench_verifier.authored_results import TrialDevNumericalRecovery, mean_numerical_loss

    def expected_loss(width):
        # Exact integration for T ~ Uniform[-1, 1], estimate=0, scale=1.
        return 0.25 + width + 10 * (1 - width) ** 2

    def expected_old_credit(width):
        return 2 * log((2 + 3 * width) / (2 + 2 * width)) + 2 / 41 * log((43 - 38 * width) / (2 + 3 * width))

    assert expected_old_credit(0.8837) > expected_old_credit(0.95)
    assert expected_loss(0.95) < expected_loss(0.8837)
    assert min(range(10001), key=lambda i: expected_loss(i / 10000)) == 9500
    scores = tuple(
        TrialDevNumericalRecovery(
            target_id=str(index),
            status="reported",
            reference_value=0,
            normalization_scale=1,
            normalized_absolute_error=point,
            normalized_interval_loss=interval,
        )
        for index, (point, interval) in enumerate(((0, 0.2), (2, 8)))
    )
    assert mean_numerical_loss(scores) == pytest.approx(2.55)
    assert 1 / (1 + mean_numerical_loss(scores)) != pytest.approx(sum(1 / (1 + score.loss) for score in scores) / 2)
    missing = scores[1].model_copy(
        update={
            "status": "missing",
            "normalized_absolute_error": None,
            "normalized_interval_loss": None,
        }
    )
    assert mean_numerical_loss((scores[0], missing)) is None
    assert mean_numerical_loss(()) is None


def test_bounded_companion_preserves_loss_ordering_on_a_fixed_panel():
    """Recovery reverses aggregate loss ordering only on the panel being compared."""
    from statistics import mean

    from trialagentbench_verifier.numerical_scoring import bounded_recovery

    truths = (-1, -0.95, -0.4, 0, 0.2, 0.95, 1)
    losses = [
        mean(
            numerical_loss(estimate=0, lower=-width, upper=width, target=truth, scale=1, confidence_level=0.95)
            for truth in truths
        )
        for width in (0.5, 0.8837, 0.93, 0.95, 1)
    ]
    ordered = sorted(losses)
    assert [bounded_recovery(loss) for loss in ordered] == sorted(
        (bounded_recovery(loss) for loss in losses), reverse=True
    )


def test_expected_recovery_can_reward_worse_expected_primary_loss():
    """A display transform is not a proper score over independent truth draws."""
    from scipy.integrate import quad

    from trialagentbench_verifier.numerical_scoring import bounded_recovery

    def expectations(width):
        def loss(truth):
            return numerical_loss(estimate=0, lower=-width, upper=width, target=truth, scale=1, confidence_level=0.95)

        # For independent Uniform[-1, 1] truths and symmetric reports, |truth| is Uniform[0, 1].
        primary = quad(loss, 0, 1, points=[width], epsabs=1e-9)[0]
        companion = quad(
            lambda first: quad(
                lambda second: bounded_recovery((loss(first) + loss(second)) / 2),
                0,
                1,
                points=[width],
                epsabs=1e-9,
            )[0],
            0,
            1,
            points=[width],
            epsabs=1e-9,
        )[0]
        return primary, companion

    honest_loss, honest_companion = expectations(0.95)
    distorted_loss, distorted_companion = expectations(0.93)
    assert honest_loss == pytest.approx(1.225)
    assert distorted_loss == pytest.approx(1.229)
    assert distorted_loss > honest_loss
    assert honest_companion == pytest.approx(0.4514658663)
    assert distorted_companion == pytest.approx(0.4520224363)
    assert distorted_companion > honest_companion


def test_bounded_summary_preserves_missingness_and_singleton_uncertainty():
    from trialagentbench_verifier.numerical_scoring import bounded_recovery_summary

    assert bounded_recovery_summary(None, None, None, None) == (None, None, None, None)
    assert bounded_recovery_summary(1.0, None, None, None) == (0.5, None, None, None)
    assert bounded_recovery_summary(1.0, 0.0, 3.0, 0.4) == (0.5, 0.25, 1.0, 0.1)


@pytest.mark.parametrize(
    "loss,low,high,error",
    [
        (-1.0, None, None, None),
        (float("inf"), None, None, None),
        (1.0, 0.0, None, None),
        (None, 0.0, 1.0, None),
        (None, None, None, 0.1),
        (1.0, 2.0, 0.0, None),
        (1.0, -0.1, 2.0, None),
        (1.0, 0.0, 2.0, -0.1),
        (1.0, 0.0, 2.0, float("nan")),
    ],
)
def test_bounded_summary_rejects_invalid_primary_statistics(loss, low, high, error):
    from trialagentbench_verifier.numerical_scoring import bounded_recovery_summary

    with pytest.raises(ValueError):
        bounded_recovery_summary(loss, low, high, error)
