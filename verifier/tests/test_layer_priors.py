"""Check finite-prior posterior/action quadrature, infinite limits and scoring-input integrity."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.integrate import quad
from scipy.special import roots_legendre
from scipy.stats import norm

from trialagentbench_verifier import layer_priors
from trialagentbench_verifier.decision_consequence import (
    CONFIRMATORY_SIZE_MULTIPLES,
    DecisionEconomicsV1,
    DecisionReportV1,
    DiscreteEffectPriorV1,
    UntruncatedSNRPriorV1,
    _report_posterior,
    _value_curve,
    optimal_action_v1,
)
from trialagentbench_verifier.layer_calibration import load_public_layer_calibration


@pytest.mark.parametrize("direction", ["higher", "lower"])
@pytest.mark.parametrize("reported", [True, False])
def test_truncated_mixture_posterior_and_all_actions_match_independent_quadrature(direction, reported):
    lower, upper = -0.8, 1.7
    abscissae, masses = roots_legendre(1001)
    support = lower + (upper - lower) * (abscissae + 1) / 2

    def density(effect):
        return 0.4 * norm.pdf(effect, scale=0.6) + 0.6 * norm.pdf(effect, scale=1.2)

    weights = masses * density(support)
    weights /= weights.sum()
    prior = DiscreteEffectPriorV1(support=tuple(support), weights=tuple(weights))
    estimate, error = 0.3, 0.45
    report = (
        DecisionReportV1(
            estimate=estimate, lower=estimate - norm.isf(0.025) * error, upper=estimate + norm.isf(0.025) * error
        )
        if reported
        else None
    )

    def posterior_density(effect):
        return density(effect) * (norm.pdf(estimate, loc=effect, scale=error) if reported else 1)

    normaliser = quad(posterior_density, lower, upper, epsabs=1e-12)[0]
    mean = quad(lambda effect: effect * posterior_density(effect), lower, upper, epsabs=1e-12)[0] / normaliser
    second = quad(lambda effect: effect**2 * posterior_density(effect), lower, upper, epsabs=1e-12)[0] / normaliser
    sign = 1 if direction == "higher" else -1
    scale = 0.7
    economics = DecisionEconomicsV1(value_of_success_musd=1000, replicate_cost_musd=30, fixed_trial_cost_musd=6)
    posterior = _report_posterior(report, scale, direction, prior)
    mass, means, variances = posterior
    assert float(mass @ means) == pytest.approx(sign * mean, abs=1e-11)
    assert float(mass @ (means**2 + variances) - (mass @ means) ** 2) == pytest.approx(second - mean**2, abs=1e-11)
    expected = np.asarray(
        [
            1000
            * quad(
                lambda effect, size=size: norm.cdf(sign * effect * math.sqrt(size) / scale - norm.isf(0.025))
                * posterior_density(effect),
                lower,
                upper,
                epsabs=1e-12,
            )[0]
            / normaliser
            - 30 * size
            - 6
            for size in CONFIRMATORY_SIZE_MULTIPLES
        ]
    )
    np.testing.assert_allclose(_value_curve(*posterior, scale=scale, economics=economics), expected, atol=2e-9, rtol=0)
    index = int(np.argmax(np.r_[0, expected]))
    action = optimal_action_v1(report, scale=scale, favorable_direction=direction, economics=economics, prior=prior)
    assert action.confirm == (index != 0)
    assert action.size_multiple == (0 if index == 0 else CONFIRMATORY_SIZE_MULTIPLES[index - 1])
    assert action.expected_value_musd == pytest.approx(0 if index == 0 else expected[index - 1], abs=2e-9)


def test_infinite_limits_preserve_all_160_old_actions_and_value_curves():
    reference = json.loads((Path(__file__).parent / "fixtures/untruncated_layer_reference.json").read_text())
    prior = UntruncatedSNRPriorV1(lower=-math.inf, upper=math.inf)
    assert len(reference["cases"]) == 160
    for row in reference["cases"]:
        economics = DecisionEconomicsV1.model_validate(row["economics"])
        report = DecisionReportV1.model_validate(row["report"]) if row["report"] is not None else None
        posterior = _report_posterior(report, row["scale"], row["favorable_direction"], prior)
        values = _value_curve(*posterior, scale=row["scale"], economics=economics)
        # Portable numerical tolerance; an in-process old-source comparison separately checks exact bits.
        np.testing.assert_allclose(values, row["expected_confirmatory_value_curve"], atol=2e-11, rtol=0)
        action = optimal_action_v1(
            report,
            scale=row["scale"],
            favorable_direction=row["favorable_direction"],
            economics=economics,
            prior=prior,
        )
        assert action.confirm == row["expected_action"]["confirm"]
        assert action.size_multiple == row["expected_action"]["size_multiple"]
    with pytest.raises(ValidationError, match="infinity"):
        UntruncatedSNRPriorV1(lower=-1, upper=2)


@pytest.mark.parametrize("estimate", [-1e308, 1e308])
@pytest.mark.parametrize("half_width", [0, 1e-100, 1e308])
def test_finite_prior_remains_normalised_for_extreme_or_zero_width_reports(estimate, half_width):
    prior = DiscreteEffectPriorV1(support=(-1, -0.2, 0.4, 2), weights=(0.2, 0.3, 0.3, 0.2))
    report = DecisionReportV1(estimate=estimate, lower=-half_width, upper=half_width)
    weights, _, _ = _report_posterior(report, 1, "higher", prior)
    assert np.isfinite(weights).all()
    assert weights.sum() == pytest.approx(1)
    if half_width <= 1e-100:
        assert weights[-1 if estimate > 0 else 0] == 1


def test_public_file_integrity_schema_and_question_lookup(tmp_path):
    body = layer_priors.files("trialagentbench_verifier").joinpath(layer_priors.LAYER_PRIOR_RESOURCE).read_bytes()
    path = tmp_path / "priors.json"
    path.write_bytes(body)
    parsed = layer_priors.load_layer_prior_file(path, expected_sha256=layer_priors.LAYER_PRIOR_SHA256)
    assert len(parsed.priors) == 17
    for entry, released in zip(parsed.priors, json.loads(body)["priors"], strict=True):
        assert layer_priors.load_layer_prior(entry.estimand_id) == load_public_layer_calibration().apply(
            entry.estimand_id, entry.discrete_prior
        )
        assert entry.discrete_prior.model_dump(mode="json") == released["discrete_prior"]
    path.write_bytes(body + b" ")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        layer_priors.load_layer_prior_file(path, expected_sha256=layer_priors.LAYER_PRIOR_SHA256)
    malformed = json.loads(body)
    malformed["priors"][0]["discrete_prior"]["weights"][0] = 0
    damaged = json.dumps(malformed).encode()
    path.write_bytes(damaged)
    with pytest.raises(ValidationError, match="positive"):
        layer_priors.load_layer_prior_file(path, expected_sha256=hashlib.sha256(damaged).hexdigest())
    with pytest.raises(ValueError, match="no layer prior"):
        layer_priors.load_layer_prior("unregistered-question")


def test_production_loader_fails_when_its_required_resource_is_missing(tmp_path, monkeypatch):
    layer_priors._public_file.cache_clear()
    layer_priors._scoring_priors.cache_clear()
    monkeypatch.setattr(layer_priors, "files", lambda _: tmp_path)
    with pytest.raises(FileNotFoundError):
        layer_priors.load_layer_prior("marginal_all_cause_death_risk_difference")
    layer_priors._public_file.cache_clear()
    layer_priors._scoring_priors.cache_clear()
