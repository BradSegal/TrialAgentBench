"""Price a TrialEval answer by the development decision it would drive.

The analysis an agent reports, its estimate and 95% interval, feeds one published downstream decision: stop,
or run a confirmatory trial of a chosen size. The decision layer takes the action with the largest expected
value under the public layer prior, the SHA-checked question-specific simulation likelihood and the trial's
declared economics. TrialEval's layer prior is the discretised push-forward of its disclosed, truncated
mean-z prior. The calibrated likelihood uses the estimate marginal given each effect atom.
Its fitted effect marginal is divided out, preserving the public prior masses. One common training-selected
weight geometrically regularises this likelihood toward the normal report likelihood. Independent native
worlds assess the resulting decisions under the simulation process. The confirmatory size is the published
action-grid choice with the highest expected value after observing the report.

An answer's consequence is its regret against the reference analyses of the same data: the expected
value, under a reference analysis' posterior, of the action that analysis drives minus that of the action the
answer drives, taking the closest reference analysis (grading plurality: any reference analysis costs nothing).
This expected regret under the posterior is the expectation of the regret at the true effect given the data; it is never
negative and does not depend on the drawn effect, so it ranks answers with far less noise. The regret at the
task's true effect is reported alongside. A missing answer drives the action the prior alone drives. Values are
in millions of US dollars; each grade also reports both confirmatory trials in participants and in power at the
true effect.

The economics are declared per analysed trial from published sources (construction ``trialeval.economics``):
a confirmatory success is worth ``value_of_success_musd``; a confirmatory trial costs ``fixed_trial_cost_musd``
plus ``replicate_cost_musd`` for each multiple of the analysed trial's participants; success is a one-sided 2.5%
test. A confirmatory trial is sized as a multiple of the analysed trial, so its standard error is the target's
normaliser over the square root of that multiple.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext
from functools import lru_cache
from typing import Literal, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.special import logsumexp
from scipy.stats import norm

_Array = npt.NDArray[np.float64]

COCHRANE_SNR_MIXTURE_WEIGHTS: tuple[float, ...] = (0.32, 0.31, 0.30, 0.07)
"""Disclosed effect-prior mixture weights (van Zwet, Schwab and Senn, Stat Med 2021;40:6107-6117)."""
COCHRANE_SNR_MIXTURE_STANDARD_DEVIATIONS: tuple[float, ...] = (0.61, 1.42, 2.16, 5.64)
"""Standard deviations of the signal-to-noise ratio in each mixture component."""
CONFIRMATORY_SIZE_MULTIPLES: tuple[float, ...] = tuple(float(value) for value in np.geomspace(0.5, 25.0, 48))
"""Confirmatory trial sizes, as multiples of the analysed trial's participants."""
_SUCCESS_CRITICAL_Z = float(norm.isf(0.025))
_Z95 = float(norm.isf(0.025))


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class DecisionEconomicsV1(_Model):
    """Declared economics of one analysed trial's confirmatory decision (US$ millions)."""

    value_of_success_musd: float = Field(gt=0)
    replicate_cost_musd: float = Field(gt=0)
    """Cost of a confirmatory trial the size of the analysed trial, beyond its fixed cost."""
    fixed_trial_cost_musd: float = Field(ge=0)

    def scaled(self, *, value: float = 1.0, cost: float = 1.0) -> DecisionEconomicsV1:
        """Return the economics with the value and the costs multiplied (the disclosed sensitivity band)."""
        return DecisionEconomicsV1(
            value_of_success_musd=self.value_of_success_musd * value,
            replicate_cost_musd=self.replicate_cost_musd * cost,
            fixed_trial_cost_musd=self.fixed_trial_cost_musd * cost,
        )


class DecisionReportV1(_Model):
    """A reported analysis on the comparison scale (log scale for ratios): estimate and 95% interval."""

    estimate: float
    lower: float
    upper: float

    @model_validator(mode="after")
    def ordered_interval(self) -> Self:
        """Reject reversed bounds before interpreting the normal likelihood."""
        if self.lower > self.upper:
            raise ValueError("decision report lower limit exceeds upper limit")
        return self


class ReportDensityComponentV1(_Model):
    """One Gaussian component of a simulation-fitted effect/report joint density."""

    weight: float = Field(gt=0, le=1)
    mean: tuple[float, float, float]
    covariance: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]

    @model_validator(mode="after")
    def positive_covariance(self) -> Self:
        """Require symmetric positive-definite covariance before scoring."""
        matrix = np.asarray(self.covariance)
        if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0) or np.linalg.eigvalsh(matrix).min() <= 0:
            raise ValueError("report-density covariance must be symmetric positive definite")
        return self


class ReportCalibrationV1(_Model):
    """Conditional likelihood of the reported estimate given an effect atom.

    Components retain the frozen three-coordinate simulation fit. Scoring integrates
    out log interval SE exactly, then divides out the effect marginal to preserve
    the separately declared atom prior. Interval precision enters only through
    the normal-likelihood regulariser.
    No economics, action, grading scale or private cell selector enters this model.
    """

    centre: tuple[float, float, float]
    scale: tuple[float, float, float]
    components: tuple[ReportDensityComponentV1, ...] = Field(min_length=1)
    success_probability: tuple[float, ...] | None = Field(
        default=None, min_length=2, description="P(successful fit | effect) on the public effect atoms."
    )

    @model_validator(mode="after")
    def normalized_density(self) -> Self:
        """Require positive coordinate scales and a probability mixture."""
        if any(value <= 0 for value in self.scale) or not math.isclose(
            sum(component.weight for component in self.components), 1, rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError("report calibration requires positive scales and normalized mixture weights")
        if self.success_probability is not None and (
            any(not 0 <= value <= 1 for value in self.success_probability)
            or not any(value > 0 for value in self.success_probability)
        ):
            raise ValueError("calibrated fit probabilities must lie in [0, 1] with positive fit support")
        return self


class DiscreteEffectPriorV1(_Model):
    """The public layer prior: comparison-effect atoms with positive probability weights."""

    support: tuple[float, ...] = Field(min_length=2)
    weights: tuple[float, ...] = Field(min_length=2)
    report_calibration: ReportCalibrationV1 | None = Field(default=None, exclude_if=lambda value: value is None)
    report_likelihood_weight: float = Field(
        default=1.0,
        ge=0,
        le=1,
        exclude_if=lambda value: value == 1,
        description="Common scoring-file weight of the simulation log likelihood ratio to the normal layer.",
    )

    @model_validator(mode="after")
    def validate_measure(self) -> Self:
        """Require matching finite atoms and normalised positive weights."""
        if (
            len(self.support) != len(self.weights)
            or any(weight <= 0 for weight in self.weights)
            or not math.isclose(sum(self.weights), 1.0, rel_tol=0, abs_tol=1e-12)
        ):
            raise ValueError("discrete prior requires aligned support and positive, normalised weights")
        if (
            self.report_calibration is not None
            and self.report_calibration.success_probability is not None
            and len(self.report_calibration.success_probability) != len(self.support)
        ):
            raise ValueError("calibrated fit probabilities must align with the public effect atoms")
        return self


class UntruncatedSNRPriorV1(_Model):
    """Continuous infinite-limit reference model, supplied explicitly for analytic fixtures."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=True)
    lower: float = -math.inf
    upper: float = math.inf

    @model_validator(mode="after")
    def infinite_limits(self) -> Self:
        """Keep this reference branch restricted to the old unbounded mixture."""
        if self.lower != -math.inf or self.upper != math.inf:
            raise ValueError("the continuous reference requires truncation limits at negative/positive infinity")
        return self


DecisionPriorV1 = DiscreteEffectPriorV1 | UntruncatedSNRPriorV1


class DecisionActionV1(_Model):
    """The optimal downstream action: stop (multiple 0) or confirm at a multiple of the analysed trial."""

    confirm: bool
    size_multiple: float = Field(ge=0)
    expected_value_musd: float


class DecisionConsequenceV1(_Model):
    """Consequence of one answer against its closest reference analysis of the same data."""

    agent: DecisionActionV1
    reference: DecisionActionV1
    regret_musd: float = Field(ge=0)
    """Expected regret under the posterior under the closest reference analysis (the headline)."""
    realised_regret_musd: float
    """Regret at the task's true effect against the closest reference analysis there."""
    go_decision_differs: bool
    agent_confirmatory_participants: float = Field(ge=0)
    reference_confirmatory_participants: float = Field(ge=0)
    agent_power_at_truth: float | None = None
    reference_power_at_truth: float | None = None
    consequential: bool


def _benefit(value: float, favorable_direction: Literal["higher", "lower"]) -> float:
    return value if favorable_direction == "higher" else -value


def _posterior(estimate: float, standard_error: float, scale: float) -> tuple[_Array, _Array, _Array]:
    """Return the mixture posterior (weights, means, variances) of the benefit given one normal report."""
    weights = np.asarray(COCHRANE_SNR_MIXTURE_WEIGHTS, dtype=float)
    deviations = np.asarray(COCHRANE_SNR_MIXTURE_STANDARD_DEVIATIONS, dtype=float)
    if math.isinf(standard_error):
        return weights, np.zeros_like(weights), (deviations * scale) ** 2
    if standard_error == 0:
        return weights, np.full_like(weights, estimate), np.zeros_like(weights)
    log_prior_sd = np.log(deviations) + math.log(scale)
    log_se = math.log(standard_error)
    log_total_sd = np.logaddexp(2 * log_prior_sd, 2 * log_se) / 2
    # Subtract the broadest component's quadratic term before exponentiating.
    # Squaring a large finite report, or subtracting two infinite log weights,
    # otherwise crashes grading or turns a valid likelihood into NaN.
    penalty = np.zeros_like(weights)
    if estimate != 0:
        with np.errstate(over="ignore", under="ignore"):
            penalty[:-1] = np.exp(
                2 * math.log(abs(estimate))
                + 2 * math.log(scale)
                + np.log(deviations[-1] ** 2 - deviations[:-1] ** 2)
                - 2 * log_total_sd[:-1]
                - 2 * log_total_sd[-1]
            )
    log_weight = np.log(weights) - log_total_sd - penalty / 2
    posterior_weight = np.exp(log_weight - log_weight.max())
    posterior_weight /= posterior_weight.sum()
    ratio = np.exp(log_prior_sd - log_total_sd)
    posterior_sd = np.exp(log_prior_sd + log_se - log_total_sd)
    return posterior_weight, ratio * estimate * ratio, posterior_sd**2


@lru_cache(maxsize=128)
def _prior_arrays(
    prior: DiscreteEffectPriorV1, favorable_direction: Literal["higher", "lower"]
) -> tuple[_Array, _Array, _Array, _Array]:
    weights = np.asarray(prior.weights, dtype=float)
    weights /= weights.sum()
    means = np.asarray(prior.support, dtype=float) * (1 if favorable_direction == "higher" else -1)
    variances = np.zeros_like(means)
    logs = np.log(weights)
    for array in (weights, means, variances, logs):
        array.setflags(write=False)
    return weights, means, variances, logs


@lru_cache(maxsize=8192)
def _report_posterior(
    report: DecisionReportV1 | None,
    scale: float,
    favorable_direction: Literal["higher", "lower"],
    prior: DecisionPriorV1,
) -> tuple[_Array, _Array, _Array]:
    """Return the posterior of the benefit given a report (``None`` uses the prior alone)."""
    if isinstance(prior, UntruncatedSNRPriorV1):
        if report is None:
            return _posterior(0.0, math.inf, scale)
        standard_error = report.upper / (2 * _Z95) - report.lower / (2 * _Z95)
        return _posterior(_benefit(report.estimate, favorable_direction), standard_error, scale)
    weights, means, variances, logs = _prior_arrays(prior, favorable_direction)
    if report is None:
        return weights, means, variances
    if prior.report_calibration is not None and prior.report_likelihood_weight > 0:
        physical = np.asarray(prior.support)
        posterior = _calibrated_weights(
            report, physical, logs, prior.report_calibration, prior.report_likelihood_weight
        )
        return posterior, means, variances
    estimate = _benefit(report.estimate, favorable_direction)
    error = report.upper / (2 * _Z95) - report.lower / (2 * _Z95)
    likelihood = _normal_atom_log_likelihood(estimate, error, means)
    if error == 0:
        posterior = weights * np.exp(likelihood)
    else:
        log_weight = logs + likelihood
        posterior = np.exp(log_weight - log_weight.max())
    return posterior / posterior.sum(), means, variances


def _normal_atom_log_likelihood(estimate: float, error: float, means: _Array) -> _Array:
    """Compare normal likelihoods to the nearest atom without subtracting huge squares."""
    if estimate >= means.max():
        closest = float(means.max())
    elif estimate <= means.min():
        closest = float(means.min())
    else:
        closest = float(means[np.argmin(abs(means - estimate))])
    if error == 0:
        selected = (
            means == closest
            if estimate < means.min() or estimate > means.max()
            else (abs(means - estimate) == np.min(abs(means - estimate)))
        )
        return np.where(selected, 0.0, -np.inf)
    with np.errstate(over="ignore", divide="ignore"):
        return np.asarray(
            -np.exp(
                np.log(abs(means - closest)) + np.log(abs(means / 2 + closest / 2 - estimate)) - 2 * math.log(error)
            ),
            dtype=np.float64,
        )


@lru_cache(maxsize=128)
def _density_arrays(calibration: ReportCalibrationV1) -> tuple[_Array, _Array, _Array, _Array]:
    means = np.asarray([component.mean[:2] for component in calibration.components])
    covariance = np.asarray([component.covariance for component in calibration.components])[:, :2, :2]
    inverse = np.linalg.inv(covariance)
    log_weight = np.log([component.weight for component in calibration.components])
    log_joint = log_weight - np.linalg.slogdet(covariance)[1] / 2
    log_effect = log_weight - np.log(covariance[:, 0, 0]) / 2
    return means, inverse, log_joint, log_effect


def _calibrated_weights(
    report: DecisionReportV1,
    support: _Array,
    log_prior: _Array,
    calibration: ReportCalibrationV1,
    weight: float,
) -> _Array:
    """Apply the simulation conditional likelihood to the exact public atom masses."""
    log_weight = _calibrated_log_weights(report, support, log_prior, calibration, weight)
    posterior: _Array = np.asarray(np.exp(log_weight - log_weight.max()), dtype=np.float64)
    posterior /= float(posterior.sum())
    return posterior


def _atom_log_posterior(report: DecisionReportV1 | None, prior: DiscreteEffectPriorV1) -> _Array:
    """Return stable atom log probabilities for calibration's held-out proper log score."""
    log_weight = np.log(prior.weights)
    if report is not None:
        support = np.asarray(prior.support)
        if prior.report_calibration is not None and prior.report_likelihood_weight > 0:
            log_weight = _calibrated_log_weights(
                report, support, log_weight, prior.report_calibration, prior.report_likelihood_weight
            )
        else:
            error = report.upper / (2 * _Z95) - report.lower / (2 * _Z95)
            log_weight += _normal_atom_log_likelihood(report.estimate, error, support)
    return np.asarray(log_weight - logsumexp(log_weight), dtype=np.float64)


def _calibrated_log_weights(
    report: DecisionReportV1,
    support: _Array,
    log_prior: _Array,
    calibration: ReportCalibrationV1,
    weight: float,
) -> _Array:
    """Compute unnormalised log atom masses without exponentiating tiny probabilities."""
    error = report.upper / (2 * _Z95) - report.lower / (2 * _Z95)
    centre, scale = np.asarray(calibration.centre[:2]), np.asarray(calibration.scale[:2])
    means, inverse, log_joint, log_effect = _density_arrays(calibration)
    with np.errstate(over="ignore", invalid="ignore"):
        x = (support - centre[0]) / scale[0]
        raw = np.asarray((report.estimate,))
        y = (raw - centre[1:]) / scale[1:]
        delta = np.empty((len(means), len(support), 2))
        delta[:, :, 0] = x[None, :] - means[:, None, 0]
        delta[:, :, 1:] = y[None, None, :] - means[:, None, 1:]
        quadratic = np.einsum("kni,kij,knj->kn", delta, inverse, delta)
        # Bound coordinate subtraction/division as well as the quadratic; using
        # abs(delta) alone would miss cancellation against the fitted centres.
        extent = np.empty_like(delta)
        extent[:, :, 0] = (abs(support)[None, :] + abs(centre[0])) / scale[0] + abs(means[:, None, 0])
        extent[:, :, 1:] = (abs(raw)[None, None, :] + abs(centre[1:])) / scale[1:] + abs(means[:, None, 1:])
        epsilon = np.finfo(float).eps
        coordinate_error = (4 * epsilon / (1 - 4 * epsilon)) * extent
        absolute_quadratic = np.einsum("kni,kij,knj->kn", abs(delta), abs(inverse), abs(delta))
        quadratic_error = (64 * epsilon / (1 - 64 * epsilon)) * absolute_quadratic
        quadratic_error += 2 * np.einsum("kni,kij,knj->kn", abs(delta), abs(inverse), coordinate_error)
        quadratic_error += np.einsum("kni,kij,knj->kn", coordinate_error, abs(inverse), coordinate_error)
    marginal_variance = np.asarray([component.covariance[0][0] for component in calibration.components])
    marginal = logsumexp(
        log_effect[:, None] - (x[None, :] - means[:, None, 0]) ** 2 / (2 * marginal_variance[:, None]), axis=0
    )
    # Budget 1e-10 absolute log-odds error, rather than merely requiring finite
    # quadratics. gamma_64 conservatively allows for coordinate arithmetic, the
    # four-term quadratic, log-mixture/marginal subtraction and prior addition.
    # The sum of absolute products also catches cancellation inside a quadratic.
    # A 1e-10 log-mass error changes posterior odds by at most exp(2e-10), and
    # any expected value by at most expm1(2e-10) times its utility range.
    # Above that budget, cancel report-only terms in extended precision before
    # combining with the prior; even finite values far below 2**52 need this.
    roundoff = float(np.max(quadratic_error)) + (64 * epsilon / (1 - 64 * epsilon)) * (
        float(np.max(abs(log_joint))) + float(np.max(abs(log_prior))) + float(np.max(abs(marginal)))
    )
    if not np.isfinite(quadratic).all() or not math.isfinite(roundoff) or roundoff > 1e-10:
        return _calibrated_tail_log_weights(report, support, log_prior, calibration, weight)
    log_weight = log_prior + logsumexp(log_joint[:, None] - quadratic / 2, axis=0) - marginal
    if calibration.success_probability is not None:
        with np.errstate(divide="ignore"):
            log_weight += np.log(calibration.success_probability)
    if weight < 1:
        normal = _normal_atom_log_likelihood(report.estimate, error, support)
        log_weight = log_prior + weight * (log_weight - log_prior) + (1 - weight) * normal
    if np.isnan(log_weight).any() or np.isposinf(log_weight).any() or not np.isfinite(log_weight).any():
        return _calibrated_tail_log_weights(report, support, log_prior, calibration, weight)
    return np.asarray(log_weight, dtype=np.float64)


def _calibrated_tail_log_weights(
    report: DecisionReportV1,
    support: _Array,
    log_prior: _Array,
    calibration: ReportCalibrationV1,
    weight: float,
) -> _Array:
    """Center log posterior odds before converting extreme quadratics to float64.

    Decimal precision covers the quadratic's integer digits plus 80 guard digits,
    so report-only terms cancel without losing atom-dependent terms or prior odds.
    This evaluates the same density as the ordinary path; no report is clipped.
    """
    error = report.upper / (2 * _Z95) - report.lower / (2 * _Z95)
    means, inverse, _, _ = _density_arrays(calibration)
    raw_bound = max(
        (
            0.0,
            *(math.log10(abs(value)) for value in (report.estimate, *support, *calibration.centre[:2]) if value),
        )
    )
    coordinate_bound = max(
        raw_bound - min(map(math.log10, calibration.scale[:2])) + 1,
        max((0.0, *(math.log10(abs(value)) for value in means.flat if value))) + 1,
    )
    quadratic_bound = 2 * coordinate_bound + max(0.0, math.log10(float(np.max(abs(inverse))))) + 2
    normal_bound = 0.0 if error == 0 else 2 * (raw_bound + 1 - math.log10(error))
    probability = calibration.success_probability or (1.0,) * len(support)
    eligible = [index for index, value in enumerate(probability) if value > 0]
    precision_digits = max(80, math.ceil(max(quadratic_bound, normal_bound)) + 80)
    with localcontext(Context(prec=precision_digits, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999)):
        decimal = Decimal.from_float
        atoms = [decimal(float(value)) for value in support]
        estimate = decimal(report.estimate)
        reference = min((atoms[index] for index in eligible), key=lambda atom: abs(atom - estimate))
        distance = abs(reference - estimate)
        coefficient = decimal(weight)
        # Expand the quadratic into a report-only intercept and an atom polynomial.
        # Keeping these in extended precision is necessary even when the eventual
        # posterior is concentrated at one atom: two huge terms can cancel.
        components: list[tuple[Decimal, Decimal, Decimal, Decimal]] = []
        effect_components: list[tuple[Decimal, Decimal, Decimal]] = []
        for component in calibration.components:
            mean = component.mean
            covariance = [[decimal(value) for value in row[:2]] for row in component.covariance[:2]]
            # Marginalise log SE before inversion, and reconstruct both exponents
            # and normalisers at the same precision before cancellation.
            determinant = covariance[0][0] * covariance[1][1] - covariance[0][1] * covariance[1][0]
            precision = [
                [covariance[1][1] / determinant, -covariance[0][1] / determinant],
                [-covariance[1][0] / determinant, covariance[0][0] / determinant],
            ]
            log_mass = decimal(component.weight).ln() - determinant.ln() / 2
            effect_components.append(
                (
                    decimal(component.weight).ln() - covariance[0][0].ln() / 2,
                    -1 / (2 * covariance[0][0]),
                    decimal(mean[0]),
                )
            )
            dy = (estimate - decimal(calibration.centre[1])) / decimal(calibration.scale[1]) - decimal(mean[1])
            intercept = log_mass - precision[1][1] * dy**2 / 2
            linear = -(precision[0][1] + precision[1][0]) * dy / 2
            components.append((intercept, linear, -precision[0][0] / 2, decimal(mean[0])))
        logs: list[Decimal | None] = []
        for index, atom in enumerate(atoms):
            if probability[index] == 0 or (weight < 1 and error == 0 and abs(atom - estimate) != distance):
                logs.append(None)
                continue
            x = (atom - decimal(calibration.centre[0])) / decimal(calibration.scale[0])
            joint = [
                intercept + (x - mean) * linear + (x - mean) ** 2 * square
                for intercept, linear, square, mean in components
            ]
            maximum = max(joint)
            mixture = maximum + decimal(float(logsumexp([float(value - maximum) for value in joint])))
            effects = [mass + (x - mean) ** 2 * square for mass, square, mean in effect_components]
            effect_maximum = max(effects)
            marginal = effect_maximum + decimal(float(logsumexp([float(value - effect_maximum) for value in effects])))
            density = mixture - marginal + decimal(math.log(probability[index]))
            normal = Decimal(0)
            if weight < 1 and error > 0:
                normal = -(atom - reference) * (atom / 2 + reference / 2 - estimate) / decimal(error) ** 2
            logs.append(decimal(float(log_prior[index])) + coefficient * density + (1 - coefficient) * normal)
        maximum = max(value for value in logs if value is not None)
        return np.asarray(
            [float(value - maximum) if value is not None else -np.inf for value in logs], dtype=np.float64
        )


@lru_cache(maxsize=128)
def _fixed_atom_powers(means: bytes, scale: float) -> _Array:
    """Cache all confirmatory powers for fixed, zero-variance effect atoms."""
    support = np.frombuffer(means, dtype=np.float64)
    slope = np.sqrt(np.asarray(CONFIRMATORY_SIZE_MULTIPLES, dtype=float))[:, None] / scale
    with np.errstate(over="ignore"):
        power = np.asarray(norm.cdf(slope * support[None, :] - _SUCCESS_CRITICAL_Z), dtype=np.float64)
    power.setflags(write=False)
    return power


def _value_curve(
    weights: _Array,
    means: _Array,
    variances: _Array,
    *,
    scale: float,
    economics: DecisionEconomicsV1,
) -> _Array:
    """Return the expected value of confirming at every declared size multiple."""
    multiples = np.asarray(CONFIRMATORY_SIZE_MULTIPLES, dtype=float)[:, None]
    slope = np.sqrt(multiples) / scale
    if len(means) > 4 and not variances.any():
        power = _fixed_atom_powers(means.tobytes(), scale)
    else:
        with np.errstate(over="ignore"):
            power = norm.cdf(
                (slope * means[None, :] - _SUCCESS_CRITICAL_Z) / np.sqrt(1.0 + slope**2 * variances[None, :])
            )
    cost = multiples[:, 0] * economics.replicate_cost_musd + economics.fixed_trial_cost_musd
    values: _Array = economics.value_of_success_musd * (power @ weights) - cost
    return values


def optimal_action_v1(
    report: DecisionReportV1 | None,
    *,
    scale: float,
    favorable_direction: Literal["higher", "lower"],
    economics: DecisionEconomicsV1,
    prior: DecisionPriorV1,
) -> DecisionActionV1:
    """Return the expected-value-maximising action given a report (``None`` uses the prior alone).

    Parameters
    ----------
    report
        Reported estimate and 95% interval on the comparison scale.
    scale
        The target's grading normaliser: the comparison's sampling SD at the analysed trial's size.
    favorable_direction
        Direction of benefit on the comparison scale.
    economics
        Declared development economics.
    prior
        Explicit layer prior. Production TrialEval callers load the SHA-checked public scoring file.

    Returns
    -------
    DecisionActionV1
        Stop, or confirm at the size multiple of largest expected value.
    """
    curve = _value_curve(
        *_report_posterior(report, scale, favorable_direction, prior), scale=scale, economics=economics
    )
    best = int(np.argmax(curve))
    if curve[best] <= 0.0:
        return DecisionActionV1(confirm=False, size_multiple=0.0, expected_value_musd=0.0)
    return DecisionActionV1(
        confirm=True, size_multiple=CONFIRMATORY_SIZE_MULTIPLES[best], expected_value_musd=float(curve[best])
    )


def _power(action: DecisionActionV1, true_benefit: float, scale: float) -> float:
    return float(norm.cdf(true_benefit * math.sqrt(action.size_multiple) / scale - _SUCCESS_CRITICAL_Z))


def realised_value_v1(
    action: DecisionActionV1,
    *,
    true_benefit: float,
    scale: float,
    economics: DecisionEconomicsV1,
) -> float:
    """Return an action's expected value (US$ millions) when the true benefit is known."""
    if not action.confirm:
        return 0.0
    cost = action.size_multiple * economics.replicate_cost_musd
    return (
        economics.value_of_success_musd * _power(action, true_benefit, scale) - cost - economics.fixed_trial_cost_musd
    )


def _posterior_value(
    action: DecisionActionV1, posterior: tuple[_Array, _Array, _Array], *, scale: float, economics: DecisionEconomicsV1
) -> float:
    """Return an action's expected value under a posterior of the benefit."""
    if not action.confirm:
        return 0.0
    weights, means, variances = posterior
    slope = math.sqrt(action.size_multiple) / scale
    if len(means) > 4 and not variances.any() and action.size_multiple in CONFIRMATORY_SIZE_MULTIPLES:
        power = (
            _fixed_atom_powers(means.tobytes(), scale)[CONFIRMATORY_SIZE_MULTIPLES.index(action.size_multiple)]
            @ weights
        )
    else:
        power = norm.cdf((slope * means - _SUCCESS_CRITICAL_Z) / np.sqrt(1.0 + slope**2 * variances)) @ weights
    cost = action.size_multiple * economics.replicate_cost_musd + economics.fixed_trial_cost_musd
    return float(economics.value_of_success_musd * power - cost)


def decision_consequence_v1(
    answer: DecisionReportV1 | None,
    credited: tuple[DecisionReportV1, ...],
    *,
    truth: float,
    scale: float,
    participants: int,
    favorable_direction: Literal["higher", "lower"],
    economics: DecisionEconomicsV1,
    prior: DecisionPriorV1,
    consequential_regret_musd: float = 5.0,
) -> DecisionConsequenceV1:
    """Return an answer's decision consequence against the closest reference analysis of the same data.

    Parameters
    ----------
    answer
        The agent's report on the comparison scale, or ``None`` when no valid answer was submitted.
    credited
        Every reference analysis of the same data, on the comparison scale.
    truth
        The task's true effect on the comparison scale.
    scale, favorable_direction, economics, prior
        As for ``optimal_action_v1``.
    participants
        Participants in the analysed trial, to report confirmatory participants in natural units.
    consequential_regret_musd
        Expected regret under the posterior at which a changed stop-or-confirm decision counts as a material decision error.

    Returns
    -------
    DecisionConsequenceV1
        Expected regret under the posterior and realised regret, natural-unit decisions and the material-decision-error flag.

    Raises
    ------
    ValueError
        If no reference analysis is supplied.
    """
    if not credited:
        raise ValueError("decision consequence requires at least one credited analysis")
    benefit = _benefit(truth, favorable_direction)
    agent = optimal_action_v1(
        answer, scale=scale, favorable_direction=favorable_direction, economics=economics, prior=prior
    )
    agent_value = realised_value_v1(agent, true_benefit=benefit, scale=scale, economics=economics)
    expected, realised = [], []
    for report in credited:
        posterior = _report_posterior(report, scale, favorable_direction, prior)
        action = optimal_action_v1(
            report, scale=scale, favorable_direction=favorable_direction, economics=economics, prior=prior
        )
        regret = _posterior_value(action, posterior, scale=scale, economics=economics) - _posterior_value(
            agent, posterior, scale=scale, economics=economics
        )
        expected.append((max(0.0, regret), action))
        value = realised_value_v1(action, true_benefit=benefit, scale=scale, economics=economics)
        realised.append((abs(value - agent_value), value - agent_value))
    regret, reference = min(expected, key=lambda item: item[0])
    differs = all(action.confirm != agent.confirm for _, action in expected)
    return DecisionConsequenceV1(
        agent=agent,
        reference=reference,
        regret_musd=regret,
        realised_regret_musd=min(realised)[1],
        go_decision_differs=differs,
        agent_confirmatory_participants=agent.size_multiple * participants,
        reference_confirmatory_participants=reference.size_multiple * participants,
        agent_power_at_truth=_power(agent, benefit, scale) if agent.confirm else None,
        reference_power_at_truth=_power(reference, benefit, scale) if reference.confirm else None,
        consequential=differs and regret >= consequential_regret_musd,
    )


class TruthCentredConsequenceV1(_Model):
    """An answer's regret under the posterior an analyst of reference precision would hold at the true effect."""

    regret_musd: float = Field(ge=0)
    """Truth-centred posterior regret, minimised over the reference analyses' precisions (US$ millions)."""
    go_decision_differs: bool
    """The answer's stop-or-confirm differs from that of every truth-centred reference action."""
    consequential: bool


def truth_centred_consequence_v1(
    answer: DecisionReportV1 | None,
    credited: tuple[DecisionReportV1, ...],
    *,
    truth: float,
    scale: float,
    favorable_direction: Literal["higher", "lower"],
    economics: DecisionEconomicsV1,
    prior: DecisionPriorV1,
    consequential_regret_musd: float = 5.0,
) -> TruthCentredConsequenceV1:
    """Return an answer's truth-centred posterior regret, a companion to the headline regret.

    The headline forms each reference posterior from a reference analysis' realised estimate, so an answer that
    reproduces a reference method costs nothing whatever that estimate's sampling error. This companion keeps the
    layer, prior, economics and action grid but centres each reference analysis' report on the task's population
    target, keeping that analysis' interval width: the posterior of an analyst with reference precision and an
    average draw. The regret is minimised over the reference analyses, as the headline is.

    Parameters
    ----------
    answer
        The agent's report on the comparison scale, or ``None`` when no valid answer was submitted.
    credited
        Every reference analysis of the same data; only their interval widths are used.
    truth
        The task's population target on the comparison scale.
    scale, favorable_direction, economics, prior
        As for ``optimal_action_v1``.
    consequential_regret_musd
        Regret at which a changed stop-or-confirm decision counts as consequential.

    Returns
    -------
    TruthCentredConsequenceV1
        The regret, whether the stop-or-confirm decision differs, and the consequential flag.

    Raises
    ------
    ValueError
        If no reference analysis is supplied.

    Examples
    --------
    >>> economics = DecisionEconomicsV1(value_of_success_musd=1000, replicate_cost_musd=10, fixed_trial_cost_musd=1)
    >>> credited = (DecisionReportV1(estimate=0.3, lower=0.1, upper=0.5),)
    >>> at_truth = DecisionReportV1(estimate=0.25, lower=0.05, upper=0.45)
    >>> truth_centred_consequence_v1(
    ...     at_truth, credited, truth=0.25, scale=0.1, favorable_direction="higher", economics=economics,
    ...     prior=UntruncatedSNRPriorV1(),
    ... ).regret_musd
    0.0
    """
    if not credited:
        raise ValueError("truth-centred regret requires at least one credited analysis")
    agent = optimal_action_v1(
        answer, scale=scale, favorable_direction=favorable_direction, economics=economics, prior=prior
    )
    priced = []
    for report in credited:
        half = (report.upper - report.lower) / 2
        centred = DecisionReportV1(estimate=truth, lower=truth - half, upper=truth + half)
        posterior = _report_posterior(centred, scale, favorable_direction, prior)
        action = optimal_action_v1(
            centred, scale=scale, favorable_direction=favorable_direction, economics=economics, prior=prior
        )
        regret = _posterior_value(action, posterior, scale=scale, economics=economics) - _posterior_value(
            agent, posterior, scale=scale, economics=economics
        )
        priced.append((max(0.0, regret), action))
    regret = min(item[0] for item in priced)
    differs = all(action.confirm != agent.confirm for _, action in priced)
    return TruthCentredConsequenceV1(
        regret_musd=regret,
        go_decision_differs=differs,
        consequential=differs and regret >= consequential_regret_musd,
    )


__all__ = [
    "COCHRANE_SNR_MIXTURE_STANDARD_DEVIATIONS",
    "COCHRANE_SNR_MIXTURE_WEIGHTS",
    "CONFIRMATORY_SIZE_MULTIPLES",
    "DecisionActionV1",
    "DecisionConsequenceV1",
    "DecisionEconomicsV1",
    "DecisionPriorV1",
    "DecisionReportV1",
    "DiscreteEffectPriorV1",
    "ReportCalibrationV1",
    "ReportDensityComponentV1",
    "UntruncatedSNRPriorV1",
    "TruthCentredConsequenceV1",
    "decision_consequence_v1",
    "optimal_action_v1",
    "realised_value_v1",
    "truth_centred_consequence_v1",
]
