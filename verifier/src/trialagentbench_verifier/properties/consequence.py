"""A3 consequence and A2 regret falsification using independent fitted routes."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.stats import beta, norm, t

from trialagentbench_verifier.properties.contracts import CheckResult, EconomicsPoint, Gates, StatisticalPart
from trialagentbench_verifier.properties.statistics import FitRecord, TestEvidence, mean_test


class _State(Enum):
    FALSIFIED = "falsified"
    NOT_FALSIFIED = "not_falsified"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class _Decision:
    state: _State
    p: float | None

    @property
    def truth(self) -> bool | None:
        return None if self.state is _State.UNAVAILABLE else self.state is _State.NOT_FALSIFIED


def _from_p(p: float | None, alpha: float) -> _Decision:
    return _Decision(_State.UNAVAILABLE if p is None else _State.FALSIFIED if p < alpha else _State.NOT_FALSIFIED, p)


def _truth(value: bool | None) -> _Decision:
    return _from_p(None if value is None else float(value), 0.5)


def _all(parts: Sequence[_Decision], *, alpha: float = 0.05, count: int | None = None) -> _Decision:
    """Compose the declared A3 consequence, conjunctions in falsification semantics.

    Parameters
    ----------
    parts : sequence of _Decision
        Available falsification p-values or unavailable components. Discrete
        demonstrations use zero for false and one for true.
    alpha : float, optional
        Rejection threshold after the declared Bonferroni correction.
    count : int or None, optional
        Full declared multiplicity, including unavailable components. Defaults
        to the number of parts; use one when combining an already corrected
        test with an independent discrete falsifier.

    Returns
    -------
    _Decision
        A significant available component decides falsification after correction.
        Otherwise all components must be available to return non-falsification.
        Unavailable components never reduce the multiplier. The companion OR
        accepts any non-falsified branch (conservative p=1 if another is missing),
        and NOT swaps the two available states. These are Kleene operations;
        non-rejection is distinct from demonstrating the scientific property.
    """
    available = [part.p for part in parts if part.p is not None]
    p = min(1.0, (len(parts) if count is None else count) * min(available)) if available else None
    decision = _from_p(p, alpha)
    if decision.state is _State.FALSIFIED:
        return decision
    return decision if all(part.state is not _State.UNAVAILABLE for part in parts) else _truth(None)


def _any(parts: Sequence[_Decision]) -> _Decision:
    """Apply the declared disjunction, retaining a deciding available non-rejection."""
    if any(part.state is _State.NOT_FALSIFIED for part in parts):
        return _Decision(_State.NOT_FALSIFIED, max(part.p if part.p is not None else 1.0 for part in parts))
    if any(part.state is _State.UNAVAILABLE for part in parts):
        return _truth(None)
    return _Decision(_State.FALSIFIED, max(part.p for part in parts if part.p is not None))


def _not(part: _Decision) -> _Decision:
    """Negate a declared discrete component without collapsing unavailability."""
    if part.state is _State.UNAVAILABLE:
        return part
    return _Decision(
        _State.NOT_FALSIFIED if part.state is _State.FALSIFIED else _State.FALSIFIED,
        None if part.p is None else 1 - part.p,
    )


def cp_lower(count: int, n: int, confidence: float) -> float:
    """Return the construction's two-sided Clopper–Pearson lower endpoint."""
    return 0.0 if count == 0 else float(beta.ppf((1 - confidence) / 2, count, n - count + 1))


def paired_widths(excluded: Sequence[FitRecord], credited: Sequence[FitRecord]) -> list[float]:
    """Pair interval widths by effect and replay, retaining every credited route."""
    others = {(r.effect_index, r.replication): r for r in credited}
    values = []
    for left in excluded:
        right = others[left.effect_index, left.replication]
        a, b = left.fit, right.fit
        if (
            a.lower is not None
            and a.upper is not None
            and b.lower is not None
            and b.upper is not None
            and b.upper > b.lower
        ):
            values.append((a.upper - a.lower) / (b.upper - b.lower))
    return values


def shift_parts(
    records: Sequence[FitRecord], gates: Gates
) -> tuple[list[StatisticalPart], list[TestEvidence], bool | None]:
    """Retain each task's shift p and the source's point/envelope conditions."""
    parts, evidence = [], []
    conditions = []
    signs = set()
    for effect in range(4):
        subset = [r for r in records if r.effect_index == effect]
        errors = [r.error for r in subset if r.error is not None]
        mean, se, p = mean_test(errors, gates.consequence.TRIALEVAL_CONSEQUENTIAL_SHIFT_SD, below=True, absolute=True)
        route = subset[0].route_id
        parts.append(
            StatisticalPart(
                name="shift",
                statistic=mean,
                p=p,
                n=len(subset),
                n_ok=len(errors),
                route_id=route,
                task_key=subset[0].task_key,
                effect_index=effect,
                branch="shift",
            )
        )
        evidence.append(
            TestEvidence(
                cell=subset[0].cell,
                check_id="L1-CONSEQ-A3",
                name="shift",
                declaration_field="consequence.TRIALEVAL_CONSEQUENTIAL_SHIFT_SD",
                declared=gates.consequence.TRIALEVAL_CONSEQUENTIAL_SHIFT_SD,
                estimate=mean,
                se=se,
                p=p,
                n=len(subset),
                n_ok=len(errors),
                route_id=route,
                task_key=subset[0].task_key,
            )
        )
        inner = (
            None
            if mean is None or se is None
            else abs(mean) - float(t.ppf((1 + gates.recovery_criteria.confidence_level) / 2, len(errors) - 1)) * se
        )
        conditions.append(
            _all(
                (
                    _truth(None if mean is None else abs(mean) >= gates.consequence.TRIALEVAL_CONSEQUENTIAL_SHIFT_SD),
                    _truth(
                        None if inner is None else inner >= gates.consequence.TRIALEVAL_CONSEQUENTIAL_SHIFT_BOUND_SD
                    ),
                    _truth(None if inner is None else inner > 0),
                )
            )
        )
        if mean is not None:
            signs.add(math.copysign(1, mean))
    return parts, evidence, _all((*conditions, _truth(len(signs) <= 1))).truth


def interval_branch(
    excluded: Sequence[FitRecord], credited: Sequence[FitRecord], gates: Gates
) -> tuple[float | None, bool | None]:
    """Test the declared median-width/coverage-lower-bound conjunction."""
    if not credited:
        # trialeval_recovery.py::_consequence (:1234–1246) excludes the
        # shortcut itself from benign comparators. _consequential maps an
        # empty widths list to interval=False, a declared structural false.
        return None, False
    medians = []
    conditions = []
    for route in sorted({r.route_id for r in credited}):
        ratios = paired_widths(excluded, [r for r in credited if r.route_id == route])
        if ratios:
            medians.append(float(np.median(ratios)))
        conditions.append(
            _truth(None if not ratios else medians[-1] <= gates.consequence.TRIALEVAL_CONSEQUENTIAL_WIDTH_RATIO)
        )
    width = max(medians, default=None)
    lower = cp_lower(sum(r.covers for r in excluded), len(excluded), 1 - gates.recovery_criteria.coverage_alpha)
    conditions.append(_truth(lower < gates.recovery_criteria.coverage_lower_bound))
    decision = _all(conditions)
    return (None, None) if decision.state is _State.UNAVAILABLE else (width, decision.truth)


def demonstrated_recovery(records: Sequence[FitRecord], gates: Gates) -> bool | None:
    """Demonstrate the excluded route's pooled/task bias, coverage and support conditions."""
    c = gates.recovery_criteria
    # Independent source demonstration from trialeval_recovery.py::_route_recovery
    # (:1022–1050): a pooled t envelope, four POINT biases, and CP lower bounds.
    # This is separate from the seven unchanged L1-REC falsification tests.
    scored = {r.point_scored for r in records}
    if len(scored) != 1:
        raise ValueError("point_scored differs within one cell")
    conditions = []
    if records[0].point_scored:
        errors = [r.error for r in records if r.error is not None]
        mean, se, _ = mean_test(errors, c.bias_margin_sd, below=False, absolute=True)
        conditions.append(
            _truth(
                None
                if mean is None or se is None
                else abs(mean) + float(t.ppf((1 + c.confidence_level) / 2, len(errors) - 1)) * se <= c.bias_margin_sd
            )
        )
        for effect in range(4):
            values = [r.error for r in records if r.effect_index == effect and r.error is not None]
            mean, _, _ = mean_test(values, c.tertile_bias_margin_sd, below=False, absolute=True)
            conditions.append(_truth(None if mean is None else abs(mean) <= c.tertile_bias_margin_sd))
    conditions.extend(
        (
            _truth(
                cp_lower(sum(r.covers for r in records), len(records), 1 - c.coverage_alpha) >= c.coverage_lower_bound
            ),
            _truth(
                cp_lower(sum(r.fit.status == "ok" for r in records), len(records), c.confidence_level)
                >= c.support_lower_bound
            ),
        )
    )
    return _all(conditions).truth


def mse_upper(
    excluded: Sequence[FitRecord],
    credited: Sequence[FitRecord],
    confidence: float,
    *,
    evidence: list[TestEvidence] | None = None,
) -> tuple[float | None, float | None]:
    """Return the smallest paired log-MSE ratio upper bound and its delta-method SE."""
    # Independent implementation of trialeval_recovery.py::_loses_on_squared_error;
    # The falsification test uses the upper bound, while the source demonstrates
    # loss using the lower bound. Pairing precedes sample covariance (ddof=1).
    bounds = []
    comparisons = []

    def retain(
        route: str, count: int, estimate: float | None, se: float | None, upper: float | None, message: str
    ) -> None:
        if evidence is not None:
            evidence.append(
                TestEvidence(
                    cell=excluded[0].cell,
                    check_id="L1-CONSEQ-A3",
                    name="mse_comparison",
                    parameter=route,
                    route_id=excluded[0].route_id,
                    declaration_field="consequence.TRIALEVAL_NO_INVERSION_CONFIDENCE",
                    declared=0.0,
                    estimate=estimate,
                    se=se,
                    p=None if upper is None else float(upper >= 0),
                    n=len(excluded),
                    n_ok=count,
                    message=message,
                )
            )

    for route in sorted({r.route_id for r in credited}):
        others = {(r.effect_index, r.replication): r for r in credited if r.route_id == route}
        pairs = []
        for record in excluded:
            left = record.error
            right = others[record.effect_index, record.replication].error
            if left is not None and right is not None:
                pairs.append((left**2, right**2))
        if len(pairs) < 2:
            comparisons.append(_truth(None))
            retain(route, len(pairs), None, None, None, "fewer than two paired estimates")
            continue
        a, b = np.asarray(pairs).T
        ma, mb = float(a.mean()), float(b.mean())
        if ma <= 0 or mb <= 0:
            comparisons.append(_truth(None))
            retain(route, len(pairs), None, None, None, "nonpositive mean squared error")
            continue
        covariance = np.cov(a, b, ddof=1)
        variance = float(
            (covariance[0, 0] / ma**2 + covariance[1, 1] / mb**2 - 2 * covariance[0, 1] / (ma * mb)) / len(pairs)
        )
        se = math.sqrt(max(0.0, variance))
        estimate = math.log(ma / mb)
        upper = estimate + float(norm.ppf(confidence)) * se
        bounds.append((upper, se))
        comparisons.append(_truth(upper >= 0))
        retain(
            route, len(pairs), estimate, se, upper, f"one-sided upper bound {upper:.17g}; confidence {confidence:.17g}"
        )
    if bounds and _all(comparisons).state is not _State.UNAVAILABLE:
        return min(bounds)
    return None, None


def consequence(
    cell: str,
    excluded: Sequence[FitRecord],
    credited: Sequence[FitRecord],
    benign: Sequence[FitRecord],
    benign_credited: Sequence[FitRecord],
    gates: Gates,
) -> tuple[CheckResult, tuple[TestEvidence, ...]]:
    """Evaluate the design's shift/interval, no-inversion and benign composites."""
    parts, evidence, shifted = shift_parts(excluded, gates)
    alpha = gates.recovery_criteria.coverage_alpha
    shift = _all([_from_p(part.p, alpha) for part in parts], alpha=alpha, count=4)
    errors_by_effect = [[r.error for r in excluded if r.effect_index == i and r.error is not None] for i in range(4)]
    summaries = [mean_test(e, 0, below=False, absolute=True) for e in errors_by_effect]
    critical = float(norm.ppf((1 + gates.recovery_criteria.confidence_level) / 2))
    confident_signs = {
        math.copysign(1, mean)
        for mean, se, _ in summaries
        if mean is not None and se is not None and abs(mean) > critical * se
    }
    # A pair of confidently opposite tasks is independent evidence even when
    # the other task summaries are unavailable; the task p is already corrected.
    shift = _all((shift, _truth(len(confident_signs) <= 1)), alpha=alpha, count=1)
    width, interval = interval_branch(excluded, credited, gates)
    recovered = demonstrated_recovery(excluded, gates)
    upper, mse_se = mse_upper(
        excluded, credited, gates.consequence.TRIALEVAL_NO_INVERSION_CONFIDENCE, evidence=evidence
    )
    benign_parts, _, benign_shift = shift_parts(benign, gates)
    _, benign_interval = interval_branch(benign, benign_credited, gates)
    demonstrated_shift = _all(
        (_truth(benign_shift), *(_truth(None if part.p is None else 1 - part.p < alpha) for part in benign_parts))
    )
    recovery_decision = _truth(recovered)
    mse_loss = _all((_truth(shifted), _truth(None if upper is None else upper < 0)))
    not_inverted = _not(_any((recovery_decision, mse_loss)))
    benign_decision = _not(_any((demonstrated_shift, _truth(benign_interval))))
    branch_values = [
        (
            "interval",
            "interval",
            width,
            _truth(interval).p,
            gates.consequence.TRIALEVAL_CONSEQUENTIAL_WIDTH_RATIO,
            "consequence.TRIALEVAL_CONSEQUENTIAL_WIDTH_RATIO",
        ),
        (
            "recovery",
            "not_inverted",
            None if recovered is None else float(recovered),
            _not(recovery_decision).p,
            None,
            "recovery_criteria",
        ),
        (
            "mse",
            "not_inverted",
            upper,
            _not(mse_loss).p,
            0.0,
            "consequence.TRIALEVAL_NO_INVERSION_CONFIDENCE",
        ),
        (
            "benign",
            "benign",
            None,
            benign_decision.p,
            None,
            "assumption_holds_tiers",
        ),
    ]
    n_ok = sum(r.fit.status == "ok" for r in excluded)
    for name, branch, statistic, p, declared, field in branch_values:
        parts.append(
            StatisticalPart(
                name=name,
                branch=branch,
                statistic=statistic,
                p=p,
                n=len(excluded),
                n_ok=n_ok,
                route_id=excluded[0].route_id,
            )
        )
        evidence.append(
            TestEvidence(
                cell=cell,
                check_id="L1-CONSEQ-A3",
                name=name,
                declaration_field=field,
                declared=declared,
                estimate=statistic,
                se=mse_se if name == "mse" else None,
                p=p,
                n=len(excluded),
                n_ok=n_ok,
                route_id=excluded[0].route_id,
                message=(
                    "Declared structural false: benign twin has no remaining credited comparator; trialeval_recovery.py::_consequence:1234–1246 and _consequential."
                    if name == "benign" and not benign_credited
                    else ""
                ),
            )
        )
    consequential = _any((shift, _truth(interval)))
    cell_decision = _all((consequential, not_inverted, benign_decision), alpha=alpha, count=3)
    for name, p in (
        ("p_shift", shift.p),
        ("p_interval", _truth(interval).p),
        ("p_consequential", consequential.p),
        ("p_not_inverted", not_inverted.p),
        ("p_benign", benign_decision.p),
        ("cell_p", cell_decision.p),
    ):
        evidence.append(
            TestEvidence(
                cell=cell,
                check_id="L1-CONSEQ-A3",
                name=name,
                declaration_field="L1-CONSEQ-A3 falsification composite",
                declared=None,
                estimate=None,
                se=None,
                p=p,
                n=len(excluded),
                n_ok=n_ok,
                route_id=excluded[0].route_id,
            )
        )
    return CheckResult(
        check_id="L1-CONSEQ-A3",
        scope="cell",
        scope_id=cell,
        family="F-CONSEQ",
        parts=tuple(parts),
        route_p={},
        cell_p=cell_decision.p,
        holm_p=None,
        status="unavailable" if cell_decision.state is _State.UNAVAILABLE else "pass",
    ), tuple(evidence)


def regret_test(
    cell: str, route: str, regrets: Sequence[float], n_ok: int, point: EconomicsPoint, gates: Gates
) -> tuple[CheckResult, TestEvidence]:
    """Falsify the declared A2 regret margin at one economics band point."""
    margin = gates.over_correction_margin * gates.consequence.TRIALEVAL_CONSEQUENTIAL_REGRET_MUSD * point.value
    mean, se, p = mean_test(regrets, margin, below=True)
    part = StatisticalPart(
        name="regret", statistic=mean, p=p, n=len(regrets), n_ok=n_ok, route_id=route, economics_point=point
    )
    result = CheckResult(
        check_id="L1-REGRET-A2",
        scope="cell",
        scope_id=cell,
        family="F-REGRET",
        parts=(part,),
        route_p={},
        cell_p=p,
        holm_p=None,
        status="unavailable" if p is None else "pass",
    )
    evidence = TestEvidence(
        cell=cell,
        check_id="L1-REGRET-A2",
        name="regret",
        declaration_field="over_correction_margin * consequence.TRIALEVAL_CONSEQUENTIAL_REGRET_MUSD * economics_band.value",
        declared=margin,
        estimate=mean,
        se=se,
        p=p,
        n=len(regrets),
        n_ok=n_ok,
        route_id=route,
        economics_point=point,
    )
    return result, evidence
