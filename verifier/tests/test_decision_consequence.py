"""The decision layer is proper: no distortion of a reported analysis gains in expectation.

Properness holds for any declared economics, so it is checked at the economics of several TrialEval trial types and
at every point of the disclosed band (value and costs halved and doubled), for biased, overconfident and over-wide
reports and for an agent that applies the disclosed prior itself before reporting.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import numpy.typing as npt
import pytest

from trialagentbench_verifier.decision_consequence import (
    COCHRANE_SNR_MIXTURE_STANDARD_DEVIATIONS,
    COCHRANE_SNR_MIXTURE_WEIGHTS,
    DecisionEconomicsV1,
    DecisionReportV1,
    UntruncatedSNRPriorV1,
    _posterior,
    decision_consequence_v1,
    optimal_action_v1,
    truth_centred_consequence_v1,
)

_Array = npt.NDArray[np.float64]
SCALE, PARTICIPANTS, WORLDS = 0.1, 400, 3_000
ECONOMICS = DecisionEconomicsV1(value_of_success_musd=1_000.0, replicate_cost_musd=30.0, fixed_trial_cost_musd=6.0)
SERIES_ECONOMICS = {
    "oncology, 492 participants": DecisionEconomicsV1(
        value_of_success_musd=1_391.52, replicate_cost_musd=29.6, fixed_trial_cost_musd=5.9
    ),
    "cardiovascular, 676 participants": DecisionEconomicsV1(
        value_of_success_musd=1_320.16, replicate_cost_musd=14.138, fixed_trial_cost_musd=5.9
    ),
    "anticoagulant, 605 participants": DecisionEconomicsV1(
        value_of_success_musd=588.72, replicate_cost_musd=12.6531, fixed_trial_cost_musd=5.9
    ),
    "public health, 7,680 participants": DecisionEconomicsV1(
        value_of_success_musd=1_115.0, replicate_cost_musd=2.304, fixed_trial_cost_musd=1.0
    ),
}
"""Declared economics (US$M) of TrialEval trial types at planned sizes, as the release's evaluator records state them:
the highest-cost, a mid-cost, the lowest-value and the routine-data public-health economics."""
BAND = ((1.0, 1.0), (0.5, 1.0), (2.0, 1.0), (1.0, 0.5), (1.0, 2.0))
"""Value and cost multipliers of the declared economics and its sensitivity band."""
Distortion = tuple[float, float] | Literal["shrink"]
DISTORTIONS: tuple[Distortion, ...] = (
    (1.0, 1.0),
    (-1.0, 1.0),
    (2.0, 1.0),
    (0.0, 0.5),
    (0.0, 0.7),
    (0.0, 1.5),
    "shrink",
)
"""Shift of the estimate in SE and multiple of the interval's width, or the agent's own shrinkage towards the prior."""


def _worlds(seed: int, count: int = WORLDS) -> tuple[_Array, _Array]:
    rng = np.random.default_rng(seed)
    component = rng.choice(len(COCHRANE_SNR_MIXTURE_WEIGHTS), size=count, p=COCHRANE_SNR_MIXTURE_WEIGHTS)
    truth = rng.normal(0.0, np.asarray(COCHRANE_SNR_MIXTURE_STANDARD_DEVIATIONS)[component] * SCALE)
    return truth, truth + SCALE * rng.normal(size=count)


def _report(observed: float, distortion: Distortion, sign: float = 1.0) -> DecisionReportV1:
    """Report the trial's estimate distorted; ``shrink`` reports its posterior mean and SD under the disclosed prior."""
    if distortion == "shrink":
        weights, means, variances = _posterior(observed, SCALE, SCALE)
        mean = float(weights @ means)
        spread = math.sqrt(float(weights @ (variances + means**2)) - mean**2)
        return DecisionReportV1(estimate=mean, lower=mean - 1.96 * spread, upper=mean + 1.96 * spread)
    shift, width = distortion
    reported = observed + sign * shift * SCALE
    return DecisionReportV1(
        estimate=reported, lower=reported - 1.96 * SCALE * width, upper=reported + 1.96 * SCALE * width
    )


def _regrets(
    distortion: Distortion,
    direction: str = "higher",
    *,
    missing: bool = False,
    economics: DecisionEconomicsV1 = ECONOMICS,
    worlds: int = WORLDS,
) -> tuple[_Array, _Array]:
    """Return each world's posterior-expected and realised regret of the distorted report."""
    truth, estimate = _worlds(7, worlds)
    sign = 1.0 if direction == "higher" else -1.0
    posterior, realised = [], []
    for value, observed in zip(truth * sign, estimate * sign, strict=True):
        valid = DecisionReportV1(estimate=observed, lower=observed - 1.96 * SCALE, upper=observed + 1.96 * SCALE)
        consequence = decision_consequence_v1(
            None if missing else _report(observed, distortion, sign),
            (valid,),
            truth=value,
            scale=SCALE,
            participants=PARTICIPANTS,
            favorable_direction=direction,  # type: ignore[arg-type]
            economics=economics,
            prior=UntruncatedSNRPriorV1(),
        )
        posterior.append(consequence.regret_musd)
        realised.append(consequence.realised_regret_musd)
    return np.asarray(posterior), np.asarray(realised)


def _mean_regret(
    shift: float, width: float, direction: str = "higher", *, missing: bool = False, realised: bool = False
) -> float:
    regrets = _regrets((shift, width), direction, missing=missing)
    return float(np.mean(regrets[1] if realised else regrets[0]))


@pytest.mark.parametrize("distortion", DISTORTIONS)
def test_no_distortion_of_a_report_gains_in_expectation(distortion: Distortion) -> None:
    posterior, realised = _regrets(distortion)
    assert posterior.mean() > 0.0 and realised.mean() > 0.0


@pytest.mark.parametrize(("value", "cost"), BAND)
@pytest.mark.parametrize("economics", SERIES_ECONOMICS.values(), ids=SERIES_ECONOMICS.keys())
def test_no_distortion_gains_at_any_series_economics_or_band_point(
    economics: DecisionEconomicsV1, value: float, cost: float
) -> None:
    """Every distortion loses expected value; realised regret agrees with its posterior expectation.

    Posterior-expected regret is the expectation of realised regret given the data, so its mean over prior-drawn
    worlds is the distortion's expected loss of value; it must be positive. The realised mean must agree with it
    within Monte Carlo error and must not show a gain.
    """
    for distortion in DISTORTIONS:
        posterior, realised = _regrets(distortion, economics=economics.scaled(value=value, cost=cost), worlds=300)
        gap = realised - posterior
        assert posterior.mean() > 0.0, distortion
        assert realised.mean() > -3.0 * realised.std(ddof=1) / math.sqrt(len(realised)), distortion
        assert abs(gap.mean()) <= 4.0 * gap.std(ddof=1) / math.sqrt(len(gap)), distortion


def test_pedantic_differences_cost_far_less_than_consequential_ones() -> None:
    assert abs(_mean_regret(0.1, 1.0)) < 0.1 * _mean_regret(1.0, 1.0)


def test_lower_is_better_targets_are_oriented_and_missing_answers_use_the_prior() -> None:
    assert _mean_regret(1.0, 1.0, "lower") > 0.0
    assert _mean_regret(0.0, 1.0, missing=True) > _mean_regret(1.0, 1.0)


def test_posterior_regret_is_the_expected_regret_at_the_truth_given_the_data() -> None:
    """Rao-Blackwellisation: averaging realised regret over the credited posterior recovers the headline."""
    valid = DecisionReportV1(estimate=0.21, lower=0.21 - 1.96 * SCALE, upper=0.21 + 1.96 * SCALE)
    answer = DecisionReportV1(estimate=0.02, lower=0.02 - 1.96 * SCALE, upper=0.02 + 1.96 * SCALE)
    weights, means, variances = _posterior(valid.estimate, SCALE, SCALE)
    rng = np.random.default_rng(3)
    component = rng.choice(len(weights), size=20_000, p=weights)
    truths = rng.normal(means[component], np.sqrt(variances[component]))
    common = {"scale": SCALE, "participants": PARTICIPANTS, "favorable_direction": "higher", "economics": ECONOMICS}
    headline = decision_consequence_v1(answer, (valid,), truth=0.0, **common, prior=UntruncatedSNRPriorV1()).regret_musd  # type: ignore[arg-type]
    realised = np.mean(
        [
            decision_consequence_v1(answer, (valid,), truth=float(truth), **common, prior=UntruncatedSNRPriorV1()).realised_regret_musd  # type: ignore[arg-type]
            for truth in truths
        ]
    )
    assert headline > 10.0
    assert realised == pytest.approx(headline, rel=0.05)


def test_any_credited_analysis_costs_nothing_and_strong_evidence_confirms() -> None:
    first = DecisionReportV1(estimate=0.30, lower=0.10, upper=0.50)
    second = DecisionReportV1(estimate=0.20, lower=0.00, upper=0.40)
    consequence = decision_consequence_v1(
        second,
        (first, second),
        truth=0.25,
        scale=SCALE,
        participants=PARTICIPANTS,
        favorable_direction="higher",
        economics=ECONOMICS,
        prior=UntruncatedSNRPriorV1(),
    )
    assert consequence.regret_musd == 0.0 and not consequence.consequential
    action = optimal_action_v1(
        first, scale=SCALE, favorable_direction="higher", economics=ECONOMICS, prior=UntruncatedSNRPriorV1()
    )
    assert action.confirm and action.size_multiple > 0
    harmful = optimal_action_v1(
        DecisionReportV1(estimate=-0.3, lower=-0.5, upper=-0.1),
        scale=SCALE,
        favorable_direction="higher",
        economics=ECONOMICS,
        prior=UntruncatedSNRPriorV1(),
    )
    assert not harmful.confirm
    assert not optimal_action_v1(
        DecisionReportV1(estimate=0.05, lower=-0.15, upper=0.25),
        scale=SCALE,
        favorable_direction="higher",
        economics=ECONOMICS.scaled(value=0.1),
        prior=UntruncatedSNRPriorV1(),
    ).confirm


def test_confirmatory_trials_are_reported_in_participants_and_power() -> None:
    """The decision depends on information and economics; headcount only converts multiples to participants."""
    report = DecisionReportV1(estimate=0.30, lower=0.10, upper=0.50)
    small, large = (
        decision_consequence_v1(
            report,
            (report,),
            truth=0.25,
            scale=SCALE,
            participants=count,
            favorable_direction="higher",
            economics=ECONOMICS,
            prior=UntruncatedSNRPriorV1(),
        )
        for count in (300, 150_000)
    )
    assert small.agent == large.agent and small.agent.confirm
    assert large.agent_confirmatory_participants == small.agent.size_multiple * 150_000
    assert small.reference_confirmatory_participants == small.agent_confirmatory_participants
    assert small.agent_power_at_truth is not None and 0.5 < small.agent_power_at_truth < 1.0


def test_dearer_confirmation_shrinks_or_stops_the_confirmatory_trial() -> None:
    report = DecisionReportV1(estimate=0.15, lower=-0.05, upper=0.35)
    cheap, dear = (
        optimal_action_v1(
            report,
            scale=SCALE,
            favorable_direction="higher",
            economics=ECONOMICS.scaled(cost=cost),
            prior=UntruncatedSNRPriorV1(),
        )
        for cost in (0.5, 4.0)
    )
    assert cheap.confirm and (not dear.confirm or dear.size_multiple < cheap.size_multiple)


@pytest.mark.parametrize("direction", ["higher", "lower"])
@pytest.mark.parametrize("value,cost", BAND)
@pytest.mark.parametrize("economics", SERIES_ECONOMICS.values(), ids=SERIES_ECONOMICS.keys())
def test_final_action_exhausts_disclosed_sizes_under_independently_conditioned_prior(
    direction, value, cost, economics
):
    """Check every action's posterior value, including stop, without regret's zero clamp.

    Bayes' rule applied directly to the disclosed Gaussian mixture gives an
    independent analytic expectation of confirmatory success. Exhausting all 48
    sizes proves that no shifted, narrowed, widened or missing report can select
    an action with higher conditional expected value on this likelihood grid.
    """
    from scipy.stats import norm

    from trialagentbench_verifier.decision_consequence import CONFIRMATORY_SIZE_MULTIPLES

    economics = economics.scaled(value=value, cost=cost)
    prior_sd = SCALE * np.asarray(COCHRANE_SNR_MIXTURE_STANDARD_DEVIATIONS)
    prior_weights = np.asarray(COCHRANE_SNR_MIXTURE_WEIGHTS)
    sign = 1 if direction == "higher" else -1
    for observed in (-0.3, 0, 0.1, 0.2, 0.4):
        for standard_error in (0, 0.05, 0.1, 0.2, math.inf):
            if math.isinf(standard_error):
                report = None
                weights, means, variances = prior_weights, np.zeros(4), prior_sd**2
            else:
                estimate = sign * observed
                report = DecisionReportV1(
                    estimate=estimate,
                    lower=estimate - norm.isf(0.025) * standard_error,
                    upper=estimate + norm.isf(0.025) * standard_error,
                )
                if standard_error == 0:
                    weights, means, variances = prior_weights, np.full(4, observed), np.zeros(4)
                else:
                    evidence_variance = prior_sd**2 + standard_error**2
                    weights = prior_weights * norm.pdf(observed, scale=np.sqrt(evidence_variance))
                    weights /= weights.sum()
                    means = observed * prior_sd**2 / evidence_variance
                    variances = prior_sd**2 * standard_error**2 / evidence_variance
            values = {0.0: 0.0}
            for multiple in CONFIRMATORY_SIZE_MULTIPLES:
                future_se = SCALE / math.sqrt(multiple)
                probability = weights @ norm.sf(
                    (norm.isf(0.025) * future_se - means) / np.sqrt(future_se**2 + variances)
                )
                values[multiple] = float(
                    economics.value_of_success_musd * probability
                    - economics.fixed_trial_cost_musd
                    - multiple * economics.replicate_cost_musd
                )
            action = optimal_action_v1(
                report, scale=SCALE, favorable_direction=direction, economics=economics, prior=UntruncatedSNRPriorV1()
            )
            assert action.expected_value_musd == pytest.approx(max(values.values()), abs=1e-10)
            assert values[action.size_multiple] == pytest.approx(max(values.values()), abs=1e-10)
            for shift, width in ((-0.2, 1), (0.2, 1), (0, 0.25), (0, 2)):
                distorted = DecisionReportV1(
                    estimate=sign * (observed + shift),
                    lower=sign * (observed + shift) - width * SCALE,
                    upper=sign * (observed + shift) + width * SCALE,
                )
                selected = optimal_action_v1(
                    distorted,
                    scale=SCALE,
                    favorable_direction=direction,
                    economics=economics,
                    prior=UntruncatedSNRPriorV1(),
                )
                assert values[selected.size_multiple] <= values[action.size_multiple] + 1e-10


def test_truth_centred_regret_does_not_reward_reproducing_a_credited_realisation() -> None:
    """Matching a credited estimate costs nothing in the headline but not when that estimate misses the truth."""
    credited = DecisionReportV1(estimate=-0.10, lower=-0.30, upper=0.10)
    at_truth = DecisionReportV1(estimate=0.30, lower=0.10, upper=0.50)
    common = {"scale": SCALE, "favorable_direction": "higher", "economics": ECONOMICS}
    copied = truth_centred_consequence_v1(credited, (credited,), truth=0.30, **common, prior=UntruncatedSNRPriorV1())  # type: ignore[arg-type]
    accurate = truth_centred_consequence_v1(at_truth, (credited,), truth=0.30, **common, prior=UntruncatedSNRPriorV1())  # type: ignore[arg-type]
    headline = decision_consequence_v1(credited, (credited,), truth=0.30, participants=PARTICIPANTS, **common, prior=UntruncatedSNRPriorV1())  # type: ignore[arg-type]
    assert headline.regret_musd == 0.0
    assert accurate.regret_musd == 0.0
    assert copied.regret_musd > 10.0 and copied.go_decision_differs and copied.consequential


def test_truth_centred_regret_is_non_negative_and_minimised_over_credited_precisions() -> None:
    narrow = DecisionReportV1(estimate=0.25, lower=0.15, upper=0.35)
    wide = DecisionReportV1(estimate=0.25, lower=-0.15, upper=0.65)
    common = {"truth": 0.12, "scale": SCALE, "favorable_direction": "higher", "economics": ECONOMICS}
    answers = [None, *(DecisionReportV1(estimate=e, lower=e - 0.2, upper=e + 0.2) for e in np.linspace(-0.4, 0.6, 21))]
    for answer in answers:
        both = truth_centred_consequence_v1(answer, (narrow, wide), **common, prior=UntruncatedSNRPriorV1()).regret_musd  # type: ignore[arg-type]
        alone = [
            truth_centred_consequence_v1(answer, (report,), **common, prior=UntruncatedSNRPriorV1()).regret_musd  # type: ignore[arg-type]
            for report in (narrow, wide)
        ]
        assert both >= 0.0 and both == pytest.approx(min(alone))
    with pytest.raises(ValueError):
        truth_centred_consequence_v1(narrow, (), **common, prior=UntruncatedSNRPriorV1())  # type: ignore[arg-type]
