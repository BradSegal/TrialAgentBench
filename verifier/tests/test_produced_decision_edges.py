"""Decision limits exercised on targets from the produced development release."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import logsumexp, ndtr
from scipy.stats import multivariate_normal, norm

from trialagentbench_verifier.campaign import _decision_consequence, trialeval_target_consequence_v1
from trialagentbench_verifier.decision_consequence import (
    CONFIRMATORY_SIZE_MULTIPLES,
    DecisionActionV1,
    DecisionReportV1,
    realised_value_v1,
)
from trialagentbench_verifier.layer_priors import load_layer_prior
from trialagentbench_verifier.replay_scoring import economics_v1
from trialagentbench_verifier.trialeval import _PointTarget

_CASES = json.loads((Path(__file__).parent / "fixtures/produced_decision_targets.json").read_text())


def _independent_calibrated_values(target, report):
    """Integrate declared action utilities under a directly evaluated SciPy joint density."""
    prior = load_layer_prior(target.binding.estimand_id)
    calibration = prior.report_calibration
    assert calibration is not None
    support = np.asarray(prior.support)
    error = report.upper / (2 * norm.isf(0.025)) - report.lower / (2 * norm.isf(0.025))
    coordinates = (
        np.column_stack((support, np.full(len(support), report.estimate))) - calibration.centre[:2]
    ) / calibration.scale[:2]
    joint, marginal = [], []
    for component in calibration.components:
        joint.append(
            math.log(component.weight)
            + multivariate_normal.logpdf(
                coordinates, mean=component.mean[:2], cov=np.asarray(component.covariance)[:2, :2]
            )
        )
        marginal.append(
            math.log(component.weight)
            + norm.logpdf(coordinates[:, 0], loc=component.mean[0], scale=math.sqrt(component.covariance[0][0]))
        )
    log_likelihood = logsumexp(joint, axis=0) - logsumexp(marginal, axis=0)
    if calibration.success_probability is not None:
        log_likelihood += np.log(calibration.success_probability)
    weight = prior.report_likelihood_weight
    logs = (
        np.log(prior.weights)
        + weight * log_likelihood
        + (1 - weight) * norm.logpdf(report.estimate, loc=support, scale=error)
    )
    posterior = np.exp(logs - logsumexp(logs))
    sign = 1 if target.binding.favorable_direction == "higher" else -1
    multiples = np.asarray((0, *CONFIRMATORY_SIZE_MULTIPLES))
    powers = ndtr(
        sign * support[:, None] * np.sqrt(multiples) / target.reference_sampling_standard_deviation - norm.isf(0.025)
    )
    economics = economics_v1(target)
    values = (
        economics.value_of_success_musd * (posterior @ powers)
        - economics.fixed_trial_cost_musd
        - economics.replicate_cost_musd * multiples
    )
    values[0] = 0
    return posterior, values


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case["target"]["binding"]["effect_measure"])
def test_zero_width_has_the_same_decision_as_its_point_mass_limit(case):
    """A finite exact report carries information rather than selecting the prior."""
    target = _PointTarget.model_validate(case["target"])
    direction = 1 if target.binding.favorable_direction == "higher" else -1
    estimate = direction * 10 * target.reference_sampling_standard_deviation
    exact = DecisionReportV1(estimate=estimate, lower=estimate, upper=estimate)
    narrow = DecisionReportV1(
        estimate=estimate,
        lower=math.nextafter(estimate, -math.inf),
        upper=math.nextafter(estimate, math.inf),
    )
    observed = trialeval_target_consequence_v1(target, exact)
    limit = trialeval_target_consequence_v1(target, narrow)
    assert observed is not None and limit is not None
    assert observed.agent.confirm == limit.agent.confirm
    assert observed.agent.size_multiple == limit.agent.size_multiple
    assert observed.regret_musd == pytest.approx(limit.regret_musd, abs=1e-10)


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case["target"]["binding"]["effect_measure"])
@pytest.mark.parametrize("benefit", [1e200, -1e200, 1e308, -1e308])
def test_extreme_finite_estimate_has_a_finite_decision(case, benefit):
    """An extreme report reaches the finite prior's endpoint, without overflow."""
    target = _PointTarget.model_validate(case["target"])
    direction = 1 if target.binding.favorable_direction == "higher" else -1
    report = DecisionReportV1(estimate=direction * benefit, lower=-1.0, upper=1.0)
    result = trialeval_target_consequence_v1(target, report)
    assert result is not None
    prior = load_layer_prior(target.binding.estimand_id)
    endpoint = (max if benefit > 0 else min)(direction * atom for atom in prior.support)
    values = [(0.0, 0.0)] + [
        (
            realised_value_v1(
                DecisionActionV1(confirm=True, size_multiple=size, expected_value_musd=0),
                true_benefit=endpoint,
                scale=target.reference_sampling_standard_deviation,
                economics=economics_v1(target),
            ),
            size,
        )
        for size in CONFIRMATORY_SIZE_MULTIPLES
    ]
    expected_value, size = max(values, key=lambda row: row[0])
    assert result.agent.size_multiple == size
    assert result.agent.expected_value_musd == pytest.approx(expected_value, abs=1e-10)
    assert math.isfinite(result.regret_musd)
    assert math.isfinite(result.realised_regret_musd)
    assert result.model_dump_json() == trialeval_target_consequence_v1(target, report).model_dump_json()


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case["target"]["binding"]["effect_measure"])
@pytest.mark.parametrize("width", [1e200, 1e308])
def test_extremely_wide_interval_uses_only_the_estimate_marginal(case, width):
    """Wide reports retain the estimate likelihood while the normal regulariser becomes flat."""
    target = _PointTarget.model_validate(case["target"])
    report = DecisionReportV1(estimate=0.0, lower=-width, upper=width)
    observed = trialeval_target_consequence_v1(target, report)
    assert observed is not None
    posterior, values = _independent_calibrated_values(target, report)
    best = int(np.argmax(values))
    assert not np.allclose(posterior, load_layer_prior(target.binding.estimand_id).weights)
    assert observed.agent.confirm == (best > 0)
    assert observed.agent.size_multiple == (0 if best == 0 else CONFIRMATORY_SIZE_MULTIPLES[best - 1])
    assert observed.agent.expected_value_musd == pytest.approx(values[best], rel=1e-8, abs=1e-8)
    reference = target.consequence
    assert reference is not None
    comparison = math.log if target.binding.effect_measure == "rate_ratio" else float
    credited = [
        DecisionReportV1(estimate=comparison(estimate), lower=comparison(lower), upper=comparison(upper))
        for estimate, lower, upper in reference.credited
    ]
    regrets = []
    for credit in credited:
        _, reference_values = _independent_calibrated_values(target, credit)
        regrets.append(max(0, reference_values.max() - reference_values[best]))
    assert observed.regret_musd == pytest.approx(min(regrets), rel=1e-8, abs=1e-8)


def test_decision_report_rejects_reversed_bounds():
    """A reversed interval is invalid rather than an implicit missing answer."""
    with pytest.raises(ValueError, match="lower.*upper"):
        DecisionReportV1(estimate=0, lower=1, upper=-1)


def test_campaign_does_not_overflow_when_reconstructing_a_finite_report():
    """A very large estimate and finite width need no synthetic infinite bound."""
    target = _PointTarget.model_validate(_CASES[0]["target"])
    row = SimpleNamespace(
        target=target,
        outcome=SimpleNamespace(verification_metrics={"comparison_signed_error": 1.7e308, "interval_width": 1e308}),
    )
    result = _decision_consequence(row)
    assert result is not None
    assert math.isfinite(result.regret_musd)
