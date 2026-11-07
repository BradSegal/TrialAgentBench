"""Shared numerical recovery loss in runner-declared scientific units."""

import math
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext

NUMERICAL_LOSS_PRECISION = 12
NUMERICAL_LOSS_ROUNDING = ROUND_HALF_EVEN
NUMERICAL_LOSS_CONTEXT = Context(prec=80, rounding=NUMERICAL_LOSS_ROUNDING)


def round_numerical_loss(value: float | Decimal) -> float | None:
    """Report target loss to 12 significant decimal digits, with ties to even.

    Parameters
    ----------
    value
        Independently computed loss, before aggregation or display transforms.

    Returns
    -------
    float or None
        Rounded loss, or None when outside the finite floating-point range.
        Significant digits preserve the scale of very small and large losses.
    """
    with localcontext(NUMERICAL_LOSS_CONTEXT) as context:
        context.prec = NUMERICAL_LOSS_PRECISION
        rounded = context.create_decimal(str(value))
    return finite_diagnostic(float(rounded))


def interval_score(lower: float, upper: float, target: float, confidence_level: float) -> float:
    """Return interval width plus the confidence-weighted noncoverage penalty.

    Parameters
    ----------
    lower, upper, target
        Finite interval endpoints and reference value in the same units.
    confidence_level
        Assigned confidence level strictly between zero and one.

    Returns
    -------
    float
        Untransformed interval loss in the supplied units.

    Raises
    ------
    ValueError
        If inputs are nonfinite, reversed or have invalid confidence.
    """
    if not all(math.isfinite(value) for value in (lower, upper, target, confidence_level)):
        raise ValueError("numerical scoring requires finite inputs")
    if lower > upper or not 0 < confidence_level < 1:
        raise ValueError("numerical scoring requires ordered limits and valid confidence")
    alpha = 1.0 - confidence_level
    penalty = 0.0
    if target < lower:
        penalty = 2.0 * (lower - target) / alpha
    elif target > upper:
        penalty = 2.0 * (target - upper) / alpha
    return upper - lower + penalty


def numerical_loss(
    *, estimate: float | None, lower: float, upper: float, target: float, scale: float, confidence_level: float
) -> float | None:
    """Return mean normalised point and interval loss in fixed scientific units.

    Parameters
    ----------
    estimate
        Finite submitted point in the declared scoring units, or None when the point is reported but not
        scored; the loss is then the interval half alone.
    lower, upper, target
        Finite submitted interval and reference in the declared scoring units.
    scale
        Positive task-owned normalization scale.
    confidence_level
        Task-assigned interval confidence level.

    Returns
    -------
    float or None
        Loss rounded by :func:`round_numerical_loss`, or None outside floating-point range.
        Missing answers are handled separately and never receive a finite penalty.

    Raises
    ------
    ValueError
        If the estimate, scale or interval inputs are invalid.
    """
    if (estimate is not None and not math.isfinite(estimate)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("numerical loss requires a finite estimate and positive finite scale")
    interval_score(lower, upper, target, confidence_level)
    with localcontext(NUMERICAL_LOSS_CONTEXT):
        lo, hi, truth, denominator, confidence = map(
            Decimal.from_float, (lower, upper, target, scale, confidence_level)
        )
        point = Decimal(0) if estimate is None else abs(Decimal.from_float(estimate) - truth)
        uncertainty_decimal = hi - lo + 2 * max(lo - truth, truth - hi, Decimal(0)) / (1 - confidence)
        exact = (point + uncertainty_decimal) / (2 * denominator)
    return round_numerical_loss(exact)


def bounded_recovery(loss: float | None) -> float | None:
    """Transform aggregate numerical loss without changing its ranking.

    Parameters
    ----------
    loss
        Nonnegative aggregate loss, computed with the primary metric's weights
        and completeness rule. None denotes an unavailable primary estimate.

    Returns
    -------
    float or None
        ``1 / (1 + loss)`` in ``(0, 1]``, or None when loss is unavailable.
        Aggregate losses before transforming; never average transformed losses.

    Raises
    ------
    ValueError
        If the loss is negative or not finite.

    Examples
    --------
    >>> bounded_recovery(1.0), bounded_recovery(None)
    (0.5, None)
    """
    if loss is None:
        return None
    if not math.isfinite(loss) or loss < 0:
        raise ValueError("bounded recovery requires a finite nonnegative loss")
    return 1.0 / (1.0 + loss)


def bounded_recovery_summary(
    loss: float | None, low: float | None, high: float | None, standard_error: float | None
) -> tuple[float | None, float | None, float | None, float | None]:
    """Transform a primary loss estimate and its uncertainty together.

    Parameters
    ----------
    loss, low, high, standard_error
        Primary aggregate estimate, confidence limits and standard error, or
        None where the primary analysis withholds them. Limits must respect
        the nonnegative support of loss.

    Returns
    -------
    tuple
        Recovery estimate, reversed transformed confidence limits, and
        delta-method standard error ``SE(loss) / (1 + loss)**2``. These retain
        the primary analysis's weights, resampling units and availability.

    Raises
    ------
    ValueError
        If uncertainty is inconsistent with the primary estimate.

    Examples
    --------
    >>> bounded_recovery_summary(1.0, 0.0, 3.0, 0.4)
    (0.5, 0.25, 1.0, 0.1)
    """
    if (low is None) != (high is None):
        raise ValueError("Loss intervals require both limits")
    if loss is None and (low is not None or standard_error is not None):
        raise ValueError("Unavailable loss cannot carry uncertainty")
    if low is not None and high is not None and low > high:
        raise ValueError("Loss intervals require ordered limits")
    if standard_error is not None and (not math.isfinite(standard_error) or standard_error < 0):
        raise ValueError("Loss standard error must be finite and nonnegative")
    estimate = bounded_recovery(loss)
    return (
        estimate,
        bounded_recovery(high),
        bounded_recovery(low),
        None if standard_error is None or estimate is None else standard_error * estimate**2,
    )


def finite_diagnostic(value: float) -> float | None:
    """Represent an overflowing derived diagnostic as null, without clipping it.

    Parameters
    ----------
    value
        A derived error, width or loss from validated finite scoring inputs.

    Returns
    -------
    float or None
        The original finite value, or None when it exceeds floating-point range.
        Raw submissions and the aggregate scientific loss remain separately available.
    """
    return value if math.isfinite(value) else None
