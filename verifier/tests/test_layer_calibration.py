"""Independent conditional-density fixtures for the calibrated decision owner."""

import hashlib
import math
from decimal import Inexact, localcontext

import numpy as np
import pytest
from scipy.stats import norm

from trialagentbench_verifier import layer_calibration, layer_priors
from trialagentbench_verifier.decision_consequence import (
    CONFIRMATORY_SIZE_MULTIPLES,
    DecisionEconomicsV1,
    DecisionReportV1,
    DiscreteEffectPriorV1,
    ReportCalibrationV1,
    ReportDensityComponentV1,
    _atom_log_posterior,
    _report_posterior,
    _value_curve,
    optimal_action_v1,
)
from trialagentbench_verifier.layer_calibration import (
    LayerCalibrationFileV1,
    QuestionReportCalibrationV1,
    load_layer_calibration_file,
)
from trialagentbench_verifier.layer_priors import LAYER_PRIOR_SHA256


def _normal_joint() -> ReportCalibrationV1:
    # Effect marginal deliberately differs from the public prior. The joint is
    # theta ~ N(2,4), estimate|theta ~ N(theta,1), logSE ~ N(0,1).
    return ReportCalibrationV1(
        centre=(0, 0, 0),
        scale=(1, 1, 1),
        components=(ReportDensityComponentV1(weight=1, mean=(2, 2, 0), covariance=((4, 4, 0), (4, 5, 0), (0, 0, 1))),),
    )


@pytest.mark.parametrize("direction", ["higher", "lower"])
def test_joint_marginal_cancels_and_recovers_known_normal_likelihood(direction):
    """A mismatched fitted effect marginal must not overwrite the public masses."""
    prior = DiscreteEffectPriorV1(support=(-2, -1, 0, 1, 3), weights=(0.1, 0.2, 0.3, 0.3, 0.1))
    report = DecisionReportV1(estimate=0.7, lower=0.7 - norm.isf(0.025), upper=0.7 + norm.isf(0.025))
    baseline = _report_posterior(report, 1, direction, prior)
    calibrated = _report_posterior(
        report, 1, direction, prior.model_copy(update={"report_calibration": _normal_joint()})
    )
    np.testing.assert_allclose(calibrated[0], baseline[0], atol=1e-14, rtol=1e-12)
    np.testing.assert_array_equal(calibrated[1], baseline[1])


def test_missing_fit_preserves_exact_atom_prior():
    """Failure retains prior-only scoring even with a simulation likelihood installed."""
    prior = DiscreteEffectPriorV1(support=(-2, 1, 3), weights=(0.2, 0.5, 0.3), report_calibration=_normal_joint())
    weights, _, _ = _report_posterior(None, 2, "higher", prior)
    np.testing.assert_array_equal(weights, prior.weights)


def test_coordinate_standardization_preserves_the_physical_likelihood():
    """Fitted coordinate centers and scales must cancel out of posterior odds."""
    density = _normal_joint()
    centre = np.asarray((5.0, -3.0, 0.2))
    scale = np.asarray((2.0, 5.0, 0.6))
    component = density.components[0]
    transformed = ReportCalibrationV1(
        centre=tuple(centre),
        scale=tuple(scale),
        components=(
            ReportDensityComponentV1(
                weight=1,
                mean=tuple((np.asarray(component.mean) - centre) / scale),
                covariance=tuple(map(tuple, np.asarray(component.covariance) / np.outer(scale, scale))),
            ),
        ),
    )
    prior = DiscreteEffectPriorV1(support=(-2, 0, 1, 3), weights=(0.1, 0.3, 0.4, 0.2))
    report = DecisionReportV1(estimate=1, lower=1 - norm.isf(0.025), upper=1 + norm.isf(0.025))
    expected = _report_posterior(report, 1, "higher", prior)[0]
    observed = _report_posterior(report, 1, "higher", prior.model_copy(update={"report_calibration": transformed}))[0]
    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-14)


def test_success_event_likelihood_is_included_but_failure_still_uses_prior():
    """A successful report includes its effect-dependent chance of fitting."""
    density = _normal_joint().model_copy(update={"success_probability": (0.25, 1.0)})
    prior = DiscreteEffectPriorV1(support=(-1, 1), weights=(0.5, 0.5), report_calibration=density)
    report = DecisionReportV1(estimate=0, lower=-norm.isf(0.025), upper=norm.isf(0.025))
    np.testing.assert_allclose(_report_posterior(report, 1, "higher", prior)[0], (0.2, 0.8), atol=1e-14)
    np.testing.assert_array_equal(_report_posterior(None, 1, "higher", prior)[0], (0.5, 0.5))
    with pytest.raises(ValueError, match="align"):
        DiscreteEffectPriorV1(support=(-1, 0, 1), weights=(0.2, 0.3, 0.5), report_calibration=density)
    certain = prior.model_copy(
        update={"report_calibration": density.model_copy(update={"success_probability": (0.0, 1.0)})}
    )
    np.testing.assert_array_equal(_report_posterior(report, 1, "higher", certain)[0], (0, 1))


@pytest.mark.parametrize("estimate", [-1.0, 1.0])
@pytest.mark.parametrize("weight", [0.75, 1.0])
def test_interval_width_only_changes_the_normal_regulariser_toward_the_prior(estimate, weight):
    """Width supplies no calibrated evidence, even when its fitted effect correlation is strong."""
    calibration = ReportCalibrationV1(
        centre=(0, 0, 0),
        scale=(1, 1, 1),
        components=(
            ReportDensityComponentV1(weight=1, mean=(0, 0, 0), covariance=((1, 0, 0.8), (0, 100, 0), (0.8, 0, 1))),
        ),
    )
    prior = DiscreteEffectPriorV1(
        support=(-1, 1), weights=(0.5, 0.5), report_calibration=calibration, report_likelihood_weight=weight
    )
    previous = None
    for error in (0.5, 1.0, 2.0, 10.0, 1e6):
        half = norm.isf(0.025) * error
        report = DecisionReportV1(estimate=estimate, lower=-half, upper=half)
        probability = _report_posterior(report, 1, "higher", prior)[0][1]
        log_odds = 2 * (1 - weight) * estimate / error**2
        expected = 1 / (1 + math.exp(-log_odds))
        assert probability == pytest.approx(expected, abs=1e-14)
        assert (probability - 0.5) * estimate >= -1e-14
        if previous is not None:
            assert abs(probability - 0.5) <= abs(previous - 0.5) + 1e-14
        previous = probability


def test_log_se_parameters_do_not_change_the_exact_estimate_marginal():
    """Different fitted width coordinates with the same estimate marginal price identically."""
    base = _normal_joint()
    changed = base.model_copy(
        update={
            "centre": (0.0, 0.0, 100.0),
            "scale": (1.0, 1.0, 0.01),
            "components": (
                ReportDensityComponentV1(
                    weight=1, mean=(2, 2, -50), covariance=((4, 4, 0.3), (4, 5, -0.2), (0.3, -0.2, 10))
                ),
            ),
        }
    )
    prior = DiscreteEffectPriorV1(support=(-2, -1, 0, 1, 3), weights=(0.1, 0.2, 0.3, 0.3, 0.1))
    for estimate in (0.7, 1e8, 1e100):
        report = DecisionReportV1(estimate=estimate, lower=-10, upper=10)
        expected = _report_posterior(report, 1, "higher", prior.model_copy(update={"report_calibration": base}))[0]
        actual = _report_posterior(report, 1, "higher", prior.model_copy(update={"report_calibration": changed}))[0]
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("entry", layer_priors._public_file().priors, ids=lambda entry: entry.estimand_id)
def test_public_width_changes_obey_the_normal_regulariser_odds_bound(entry):
    """Widening a fixed estimate supplies only the declared normal-likelihood tilt."""
    prior = layer_priors.load_layer_prior(entry.estimand_id)
    calibration = prior.report_calibration
    assert calibration is not None
    support = np.asarray(prior.support)
    positive = support > 0
    prior_probability = float(np.asarray(prior.weights)[positive].sum())
    base_error = math.exp(calibration.centre[2])
    for estimate in (
        calibration.centre[1] - calibration.scale[1],
        calibration.centre[1],
        calibration.centre[1] + calibration.scale[1],
    ):
        probabilities = []
        for multiple in (1.0, 1.25, 1.5, 2.0):
            half = norm.isf(0.025) * base_error * multiple
            report = DecisionReportV1(estimate=estimate, lower=-half, upper=half)
            posterior = _report_posterior(report, 1, "higher", prior)[0]
            probabilities.append(float(posterior[positive].sum()))
        base = probabilities[0]
        for multiple, probability in zip((1.25, 1.5, 2.0), probabilities[1:], strict=True):
            # Reweighting by exp(tilt) bounds the odds between any two atom groups.
            tilt = (
                (1 - prior.report_likelihood_weight)
                * (support - estimate) ** 2
                / 2
                * (1 / base_error**2 - 1 / (base_error * multiple) ** 2)
            )
            lower = float(tilt[positive].min() - tilt[~positive].max())
            upper = float(tilt[positive].max() - tilt[~positive].min())
            log_odds_change = math.log(probability / (1 - probability)) - math.log(base / (1 - base))
            assert lower - 1e-10 <= log_odds_change <= upper + 1e-10
            assert (probability - prior_probability) * (base - prior_probability) >= -1e-12


def test_sha_and_strict_public_schema(tmp_path):
    """Unknown selectors, bad integrity and nonpositive covariances fail loudly."""
    public = LayerCalibrationFileV1(
        schema_id="trialagentbench.trialeval_report_calibration/v1",
        prior_sha256=LAYER_PRIOR_SHA256,
        coordinates=("comparison_effect", "reported_estimate", "log_interval_se"),
        method="conditional_gaussian_mixture_divided_by_effect_marginal",
        calibrations=(QuestionReportCalibrationV1(estimand_id="public_question", density=_normal_joint()),),
    )
    content = public.model_dump_json().encode()
    path = tmp_path / "calibration.json"
    path.write_bytes(content)
    assert load_layer_calibration_file(path, expected_sha256=hashlib.sha256(content).hexdigest()) == public
    with pytest.raises(ValueError, match="SHA-256"):
        load_layer_calibration_file(path, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="Extra inputs"):
        LayerCalibrationFileV1.model_validate(public.model_dump() | {"tier": "A1"})
    with pytest.raises(ValueError, match="positive definite"):
        ReportDensityComponentV1(weight=1, mean=(0, 0, 0), covariance=((1, 2, 0), (2, 1, 0), (0, 0, 1)))


def test_calibrated_zero_width_and_extreme_reports():
    """Exact and arbitrarily large finite reports have deterministic finite decisions."""
    prior = DiscreteEffectPriorV1(support=(-1, 1), weights=(0.5, 0.5), report_calibration=_normal_joint())
    economics = DecisionEconomicsV1(value_of_success_musd=100, replicate_cost_musd=1, fixed_trial_cost_musd=1)
    action = optimal_action_v1(
        DecisionReportV1(estimate=0, lower=0, upper=0),
        scale=1,
        favorable_direction="higher",
        economics=economics,
        prior=prior,
    )
    assert math.isfinite(action.expected_value_musd)
    for estimate in (1e100, -1e100, 1e200, -1e200, 1e308, -1e308, np.finfo(float).max, -np.finfo(float).max):
        action = optimal_action_v1(
            DecisionReportV1(estimate=estimate, lower=estimate, upper=estimate),
            scale=1,
            favorable_direction="higher",
            economics=economics,
            prior=prior,
        )
        assert math.isfinite(action.expected_value_musd)
        np.testing.assert_array_equal(
            _report_posterior(DecisionReportV1(estimate=estimate, lower=-1, upper=1), 1, "higher", prior)[0],
            (0, 1) if estimate > 0 else (1, 0),
        )


@pytest.mark.parametrize("weight", [0.5, 0.75, 1.0])
def test_extreme_report_common_density_cancels_without_losing_prior_odds(weight):
    """An effect-independent report must retain the prior even at float64's limits."""
    density = ReportCalibrationV1(
        centre=(0, 0, 0),
        scale=(1, 1, 1),
        components=(ReportDensityComponentV1(weight=1, mean=(0, 0, 0), covariance=((1, 0, 0), (0, 1, 0), (0, 0, 1))),),
    )
    prior = DiscreteEffectPriorV1(
        support=(-1, 0, 1), weights=(0.2, 0.5, 0.3), report_calibration=density, report_likelihood_weight=weight
    )
    report = DecisionReportV1(estimate=np.finfo(float).max, lower=-1e308, upper=1e308)
    np.testing.assert_allclose(_report_posterior(report, 1, "higher", prior)[0], prior.weights, atol=1e-14)


@pytest.mark.parametrize("estimate", [1e6, 1e7, 3e7, 5e7, 6e7, 7e7, 1e8])
@pytest.mark.parametrize("fixed_cost", [1.0, 17.5])
def test_finite_common_density_preserves_prior_odds_and_the_confirm_decision(estimate, fixed_cost):
    """Effect-independent finite likelihoods cancel before prior odds are rounded away."""
    density = ReportCalibrationV1(
        centre=(0, 0, 0),
        scale=(1, 1, 1),
        components=(ReportDensityComponentV1(weight=1, mean=(0, 0, 0), covariance=((1, 0, 0), (0, 1, 0), (0, 0, 1))),),
    )
    prior = DiscreteEffectPriorV1(
        support=(-1, 0, 1), weights=(0.2, 0.5, 0.3), report_calibration=density, report_likelihood_weight=0.75
    )
    report = DecisionReportV1(estimate=estimate, lower=-1e100, upper=1e100)
    economics = DecisionEconomicsV1(value_of_success_musd=100, replicate_cost_musd=1, fixed_trial_cost_musd=fixed_cost)
    posterior, means, variances = _report_posterior(report, 1, "higher", prior)
    np.testing.assert_allclose(posterior, prior.weights, atol=1e-14, rtol=0)
    sizes = np.asarray(CONFIRMATORY_SIZE_MULTIPLES)
    powers = norm.cdf(np.sqrt(sizes[:, None]) * np.asarray(prior.support)[None, :] - norm.isf(0.025))
    expected = 100 * (powers @ np.asarray(prior.weights)) - sizes - fixed_cost
    np.testing.assert_allclose(
        _value_curve(posterior, means, variances, scale=1, economics=economics), expected, atol=1e-12, rtol=0
    )
    best = int(np.argmax(expected))
    assert expected[best] > 0
    observed = optimal_action_v1(report, scale=1, favorable_direction="higher", economics=economics, prior=prior)
    assert observed.confirm
    assert observed.size_multiple == sizes[best]
    assert observed.expected_value_musd == pytest.approx(expected[best], abs=1e-12, rel=0)


@pytest.mark.parametrize("centre", [1e8, float(np.nextafter(1e8, np.inf)), float(np.nextafter(1e8, -np.inf)), 1e100])
@pytest.mark.parametrize("effect_variance", [1.0, 1.3])
def test_shifted_effect_marginal_cancels_at_decimal_precision(centre, effect_variance):
    """Independent report coordinates retain prior odds despite a huge effect-centre shift."""
    density = ReportCalibrationV1(
        centre=(centre, 0, 0),
        scale=(1, 1, 1),
        components=(
            ReportDensityComponentV1(
                weight=1, mean=(0, 0, 0), covariance=((effect_variance, 0, 0), (0, 1, 0), (0, 0, 1))
            ),
        ),
    )
    prior = DiscreteEffectPriorV1(
        support=(-1, 0, 1), weights=(0.2, 0.5, 0.3), report_calibration=density, report_likelihood_weight=0.75
    )
    report = DecisionReportV1(estimate=0, lower=-1e100, upper=1e100)
    economics = DecisionEconomicsV1(value_of_success_musd=100, replicate_cost_musd=1, fixed_trial_cost_musd=17.5)
    posterior, means, variances = _report_posterior(report, 1, "higher", prior)
    np.testing.assert_allclose(posterior, (0.2, 0.5, 0.3), atol=1e-14, rtol=0)
    sizes = np.asarray(CONFIRMATORY_SIZE_MULTIPLES)
    powers = norm.cdf(np.sqrt(sizes[:, None]) * np.asarray(prior.support)[None, :] - norm.isf(0.025))
    expected = 100 * (powers @ np.asarray(prior.weights)) - sizes - 17.5
    np.testing.assert_allclose(
        _value_curve(posterior, means, variances, scale=1, economics=economics), expected, atol=1e-12, rtol=0
    )
    observed = optimal_action_v1(report, scale=1, favorable_direction="higher", economics=economics, prior=prior)
    assert observed.confirm
    assert observed.size_multiple == sizes[int(expected.argmax())] == 10.007111322746532
    assert observed.expected_value_musd == pytest.approx(expected.max(), abs=1e-12, rel=0)


@pytest.mark.parametrize("estimate", [1e100, 1e308])
def test_opposing_extreme_log_likelihoods_cancel_before_conversion_to_float(estimate):
    """Opposite report slopes leave finite posterior odds, rather than a spurious point mass."""
    density = ReportCalibrationV1(
        centre=(0, 0, 0),
        scale=(1, 1, 1),
        components=(
            ReportDensityComponentV1(weight=1, mean=(0, 0, 0), covariance=((1, -1, 0), (-1, 2, 0), (0, 0, 1))),
        ),
    )
    prior = DiscreteEffectPriorV1(
        support=(-2, 0, 1), weights=(0.2, 0.5, 0.3), report_calibration=density, report_likelihood_weight=0.5
    )
    report = DecisionReportV1(estimate=estimate, lower=-norm.isf(0.025), upper=norm.isf(0.025))
    expected = np.asarray(prior.weights) * np.exp(-np.square(prior.support) / 2)
    expected /= expected.sum()
    np.testing.assert_allclose(_report_posterior(report, 1, "higher", prior)[0], expected, atol=1e-14)


def test_zero_width_conditions_on_the_nearest_atom_with_positive_fit_support():
    density = _normal_joint().model_copy(update={"success_probability": (1.0, 0.0)})
    prior = DiscreteEffectPriorV1(
        support=(-1, 1), weights=(0.5, 0.5), report_calibration=density, report_likelihood_weight=0.75
    )
    report = DecisionReportV1(estimate=1, lower=1, upper=1)
    np.testing.assert_array_equal(_report_posterior(report, 1, "higher", prior)[0], (1, 0))


def test_extreme_mixture_uses_the_dominant_report_component_and_divides_out_its_effect_marginal():
    density = ReportCalibrationV1(
        centre=(0, 0, 0),
        scale=(1, 1, 1),
        components=(
            ReportDensityComponentV1(weight=0.3, mean=(-1, 0, 0), covariance=((1, 0, 0), (0, 1, 0), (0, 0, 1))),
            ReportDensityComponentV1(weight=0.7, mean=(1, 0, 0), covariance=((4, 0, 0), (0, 4, 0), (0, 0, 1))),
        ),
    )
    prior = DiscreteEffectPriorV1(support=(-2, 0, 1), weights=(0.2, 0.5, 0.3), report_calibration=density)
    effect = np.asarray(prior.support)
    broad = 0.7 * norm.pdf(effect, loc=1, scale=2)
    expected = np.asarray(prior.weights) * broad / (0.3 * norm.pdf(effect, loc=-1, scale=1) + broad)
    expected /= expected.sum()
    report = DecisionReportV1(estimate=1e308, lower=-1, upper=1)
    np.testing.assert_allclose(_report_posterior(report, 1, "higher", prior)[0], expected, atol=1e-14)


def test_extreme_scoring_is_independent_of_the_callers_decimal_context():
    prior = DiscreteEffectPriorV1(support=(-1, 1), weights=(0.5, 0.5), report_calibration=_normal_joint())
    report = DecisionReportV1(estimate=2e306, lower=-1, upper=1)
    with localcontext() as context:
        context.prec = 2
        context.Emax, context.Emin = 9, -9
        context.traps[Inexact] = True
        np.testing.assert_array_equal(_report_posterior(report, 1, "higher", prior)[0], (0, 1))


@pytest.mark.parametrize("width", [1e200, 1e308])
def test_explicit_normal_likelihood_wide_interval_retains_its_prior_limit(width):
    prior = DiscreteEffectPriorV1(support=(-2, 0, 1), weights=(0.2, 0.5, 0.3))
    report = DecisionReportV1(estimate=0, lower=-width, upper=width)
    np.testing.assert_allclose(_report_posterior(report, 1, "higher", prior)[0], prior.weights, atol=1e-14, rtol=0)


@pytest.mark.parametrize("weight", [0.0, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("direction", ["higher", "lower"])
def test_regularised_likelihood_ratio_has_analytic_posterior(weight, direction):
    """Geometric likelihood interpolation must reproduce its known atom probabilities."""
    prior = DiscreteEffectPriorV1(support=(-2, -1, 0, 1, 3), weights=(0.1, 0.2, 0.3, 0.3, 0.1))
    regularised = prior.model_copy(update={"report_calibration": _normal_joint(), "report_likelihood_weight": weight})
    half = 0.5 * norm.isf(0.025)
    report = DecisionReportV1(estimate=0.7, lower=0.7 - half, upper=0.7 + half)
    actual = _report_posterior(report, 1, direction, regularised)[0]
    precision = weight + (1 - weight) / 0.5**2
    log_expected = np.log(prior.weights) - 0.5 * precision * (np.asarray(prior.support) - report.estimate) ** 2
    expected = np.exp(log_expected - log_expected.max())
    expected /= expected.sum()
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-12)
    if weight == 0:
        np.testing.assert_array_equal(actual, _report_posterior(report, 1, direction, prior)[0])


def test_zero_correction_and_missing_fit_keep_corrected_decisions():
    """Zero weight restores the corrected layer even for extreme or zero-width reports."""
    prior = DiscreteEffectPriorV1(support=(-1, 0, 1), weights=(0.2, 0.3, 0.5))
    regularised = prior.model_copy(update={"report_calibration": _normal_joint(), "report_likelihood_weight": 0.0})
    for report in (
        None,
        DecisionReportV1(estimate=0, lower=0, upper=0),
        DecisionReportV1(estimate=1e300, lower=1e300, upper=1e300),
    ):
        np.testing.assert_array_equal(
            _report_posterior(report, 1, "higher", regularised)[0],
            _report_posterior(report, 1, "higher", prior)[0],
        )
    for weight in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            DiscreteEffectPriorV1.model_validate(prior.model_dump() | {"report_likelihood_weight": weight})


def test_public_regularisation_is_one_common_value_without_question_overrides():
    """One public scalar controls every question; per-question scalar inputs fail loudly."""
    public = LayerCalibrationFileV1(
        schema_id="trialagentbench.trialeval_report_calibration/v1",
        prior_sha256=LAYER_PRIOR_SHA256,
        coordinates=("comparison_effect", "reported_estimate", "log_interval_se"),
        method="conditional_gaussian_mixture_divided_by_effect_marginal",
        likelihood_ratio_weight=0.25,
        calibrations=tuple(
            QuestionReportCalibrationV1(estimand_id=q, density=_normal_joint()) for q in ("one", "two")
        ),
    )
    prior = DiscreteEffectPriorV1(support=(-1, 1), weights=(0.5, 0.5))
    for q in ("one", "two"):
        assert public.apply(q, prior).report_likelihood_weight == 0.25
    assert public.model_dump_json().count('"likelihood_ratio_weight"') == 1
    changed = public.model_dump()
    changed["calibrations"][0]["density"]["likelihood_ratio_weight"] = 0.5
    with pytest.raises(ValueError, match="Extra inputs"):
        LayerCalibrationFileV1.model_validate(changed)
    with pytest.raises(ValueError, match="no density"):
        public.apply("missing", prior)


@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
def test_log_score_remains_finite_when_atom_probabilities_underflow(weight):
    """Training must compare real log densities rather than numerical probability zeros."""
    prior = DiscreteEffectPriorV1(
        support=(-1000, 0, 1),
        weights=(0.2, 0.3, 0.5),
        report_calibration=_normal_joint(),
        report_likelihood_weight=weight,
    )
    half = 0.5 * norm.isf(0.025)
    report = DecisionReportV1(estimate=1, lower=1 - half, upper=1 + half)
    logs = _atom_log_posterior(report, prior)
    assert np.isfinite(logs).all()
    assert logs[0] < -100_000
    assert _report_posterior(report, 1, "higher", prior)[0][0] == 0
    np.testing.assert_allclose(np.exp(logs), _report_posterior(report, 1, "higher", prior)[0], atol=1e-14)


@pytest.fixture
def uncached_public_calibration():
    loader = layer_calibration.load_public_layer_calibration
    loader.cache_clear()
    layer_priors._scoring_priors.cache_clear()
    yield
    loader.cache_clear()
    layer_priors._scoring_priors.cache_clear()


def test_production_calibration_matches_the_frozen_policy_for_every_question():
    content = (
        layer_calibration.files("trialagentbench_verifier")
        .joinpath(layer_calibration.LAYER_CALIBRATION_RESOURCE)
        .read_bytes()
    )
    assert hashlib.sha256(content).hexdigest() == layer_calibration.LAYER_CALIBRATION_SHA256
    public = LayerCalibrationFileV1.model_validate_json(content)
    assert public.likelihood_ratio_weight == 0.75
    assert content.count(b'"likelihood_ratio_weight"') == 1
    economics = DecisionEconomicsV1(value_of_success_musd=1000, replicate_cost_musd=30, fixed_trial_cost_musd=6)
    for entry in layer_priors._public_file().priors:
        expected = public.apply(entry.estimand_id, entry.discrete_prior)
        actual = layer_priors.load_layer_prior(entry.estimand_id)
        assert actual == expected
        density = actual.report_calibration
        assert density is not None
        half = norm.isf(0.025) * math.exp(density.centre[2])
        estimate = density.centre[1]
        for report in (None, DecisionReportV1(estimate=estimate, lower=estimate - half, upper=estimate + half)):
            for direction in ("higher", "lower"):
                np.testing.assert_array_equal(
                    _report_posterior(report, 1, direction, actual)[0],
                    _report_posterior(report, 1, direction, expected)[0],
                )
                assert optimal_action_v1(
                    report, scale=1, favorable_direction=direction, economics=economics, prior=actual
                ) == optimal_action_v1(
                    report, scale=1, favorable_direction=direction, economics=economics, prior=expected
                )


@pytest.mark.parametrize("damage", ["missing", "changed", "schema"])
def test_production_fails_on_missing_or_damaged_calibration(
    tmp_path, monkeypatch, uncached_public_calibration, damage
):
    resource = tmp_path / layer_calibration.LAYER_CALIBRATION_RESOURCE
    resource.parent.mkdir(parents=True)
    if damage != "missing":
        resource.write_bytes(b"{}")
    if damage == "schema":
        monkeypatch.setattr(layer_calibration, "LAYER_CALIBRATION_SHA256", hashlib.sha256(b"{}").hexdigest())
    monkeypatch.setattr(layer_calibration, "files", lambda _: tmp_path)
    error = FileNotFoundError if damage == "missing" else ValueError
    with pytest.raises(error):
        layer_priors.load_layer_prior(layer_priors._public_file().priors[0].estimand_id)


def test_production_requires_exact_question_coverage(monkeypatch, uncached_public_calibration):
    public = layer_calibration.load_public_layer_calibration()
    missing = public.model_copy(update={"calibrations": public.calibrations[:-1]})
    monkeypatch.setattr(layer_calibration, "load_public_layer_calibration", lambda: missing)
    with pytest.raises(ValueError, match="exactly the public prior questions"):
        layer_priors.load_layer_prior(public.calibrations[0].estimand_id)
