"""Declared falsification criteria and split-wide family adjustment."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.stats import binom, t

from trialagentbench_verifier.properties.contracts import (
    CheckResult,
    EconomicsPoint,
    RecoveryCriteria,
    REstimate,
    StatisticalPart,
)


class FitRecord(BaseModel):
    """One independent R fit with its authenticated comparison-scale target."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    cell: str
    task_key: str
    task_id: str
    effect_index: int = Field(ge=0, le=3)
    replication: int = Field(ge=1)
    route_id: str
    truth: float
    grading_sd: float = Field(gt=0)
    point_scored: bool
    fit: REstimate

    @property
    def error(self) -> float | None:
        """Return the fitted error in the task's grading SD units."""
        return None if self.fit.estimate is None else (self.fit.estimate - self.truth) / self.grading_sd

    @property
    def covers(self) -> bool:
        """Count unsuccessful fits as noncovering."""
        return (
            self.fit.lower is not None
            and self.fit.upper is not None
            and self.fit.lower <= self.truth <= self.fit.upper
        )


class TestEvidence(BaseModel):
    """Numerical evidence and the exact declaration field for a test part."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    cell: str
    check_id: str
    name: str
    declaration_field: str
    declared: float | None
    estimate: float | None
    se: float | None
    p: float | None = Field(ge=0, le=1)
    route_id: str | None = None
    task_key: str | None = None
    parameter: str | None = None
    arm_id: str | None = None
    stratum: str | None = None
    economics_point: EconomicsPoint | None = None
    n: int = Field(ge=0)
    n_ok: int | None = Field(default=None, ge=0)
    numerator: int | None = Field(default=None, ge=0)
    message: str = ""

    @model_validator(mode="after")
    def valid_counts(self) -> Self:
        """Reject impossible success counts."""
        if self.n_ok is not None and self.n_ok > self.n:
            raise ValueError("n_ok exceeds n")
        return self


def mean_test(
    values: Sequence[float], bound: float, *, below: bool, absolute: bool = False
) -> tuple[float | None, float | None, float | None]:
    """Test a mean against a declared inequality, retaining degenerate samples."""
    if len(values) < 2:
        return None, None, None
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    distance = (abs(mean) if absolute else mean) - bound
    if se == 0:
        p = 0.5 if distance == 0 else float(distance > 0 if below else distance < 0)
    else:
        statistic = distance / se
        p = float(t.cdf(statistic, len(values) - 1) if below else t.sf(statistic, len(values) - 1))
    return mean, se, p


def holm(values: Sequence[float | None]) -> tuple[float | None, ...]:
    """Adjust available hypotheses while retaining unavailable family positions."""
    ordered = sorted((p, i) for i, p in enumerate(values) if p is not None)
    adjusted: list[float | None] = [None] * len(values)
    running = 0.0
    # Unavailable declared hypotheses retain their place in the family census.
    for rank, (p, index) in enumerate(ordered):
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Holm requires finite probabilities")
        running = max(running, min(1.0, (len(values) - rank) * p))
        adjusted[index] = running
    return tuple(adjusted)


def recovery(
    cell: str, fits: Sequence[FitRecord], criteria: RecoveryCriteria
) -> tuple[CheckResult, tuple[TestEvidence, ...]]:
    """Evaluate all seven recovery parts per credited route and both Bonferroni levels."""
    groups: dict[str, list[FitRecord]] = defaultdict(list)
    for fit in fits:
        if fit.cell != cell:
            raise ValueError("recovery mixes cells")
        groups[fit.route_id].append(fit)
    if not groups:
        raise ValueError("recovery needs the declared routes, including failed fits")
    parts: list[StatisticalPart] = []
    evidence: list[TestEvidence] = []
    route_p: dict[str, float | None] = {}
    for route, records in sorted(groups.items()):
        identities = [(r.task_key, r.replication) for r in records]
        tasks = sorted({(r.effect_index, r.task_key) for r in records})
        if len(set(identities)) != len(records) or len(tasks) != 4 or {i for i, _ in tasks} != set(range(4)):
            raise ValueError("recovery requires four distinct effect tasks and unique replications")
        route_parts: list[StatisticalPart] = []
        subsets = [("bias", records, criteria.bias_margin_sd, None, None)] + [
            ("effect_bias", [r for r in records if r.task_key == task], criteria.tertile_bias_margin_sd, task, effect)
            for effect, task in tasks
        ]
        for name, subset, margin, task, effect in subsets:
            errors = [r.error for r in subset if r.error is not None]
            estimate, se, p = mean_test(errors, margin, below=False, absolute=True)
            route_parts.append(
                StatisticalPart(
                    name=name,
                    statistic=estimate,
                    p=p,
                    n=len(subset),
                    n_ok=len(errors),
                    route_id=route,
                    task_key=task,
                    effect_index=effect,
                )
            )
            evidence.append(
                TestEvidence(
                    cell=cell,
                    check_id="L1-REC",
                    name=name,
                    declaration_field="recovery_criteria."
                    + ("bias_margin_sd" if task is None else "tertile_bias_margin_sd"),
                    declared=margin,
                    estimate=estimate,
                    se=se,
                    p=p,
                    route_id=route,
                    task_key=task,
                    n=len(subset),
                    n_ok=len(errors),
                    message="fewer than two fitted errors" if p is None else "",
                )
            )
        n_ok = sum(r.fit.status == "ok" for r in records)
        for name, count, bound in (
            ("coverage", sum(r.covers for r in records), criteria.coverage_lower_bound),
            ("support", n_ok, criteria.support_lower_bound),
        ):
            p = float(binom.cdf(count, len(records), bound))
            estimate = count / len(records)
            route_parts.append(
                StatisticalPart(name=name, statistic=estimate, p=p, n=len(records), n_ok=n_ok, route_id=route)
            )
            evidence.append(
                TestEvidence(
                    cell=cell,
                    check_id="L1-REC",
                    name=name,
                    declaration_field=f"recovery_criteria.{name}_lower_bound",
                    declared=bound,
                    estimate=estimate,
                    se=math.sqrt(bound * (1 - bound) / len(records)),
                    p=p,
                    route_id=route,
                    n=len(records),
                    n_ok=n_ok,
                    numerator=count,
                )
            )
        available = [part.p for part in route_parts if part.p is not None]
        route_p[route] = min(1.0, len(route_parts) * min(available)) if available else None
        parts.extend(route_parts)
    available_routes = [p for p in route_p.values() if p is not None]
    cell_p = min(1.0, len(route_p) * min(available_routes)) if available_routes else None
    return CheckResult(
        check_id="L1-REC",
        scope="cell",
        scope_id=cell,
        family="F-REC",
        parts=tuple(parts),
        route_p=route_p,
        cell_p=cell_p,
        holm_p=None,
        status="unavailable" if any(part.p is None for part in parts) else "pass",
    ), tuple(evidence)


def adjust_families(results: Sequence[CheckResult], alpha: float) -> tuple[CheckResult, ...]:
    """Apply Holm to the declared family units and stop only rejected cells."""
    families: dict[str, list[tuple[int, int | None, float | None]]] = defaultdict(list)
    for i, result in enumerate(results):
        if result.family is None:
            continue
        if result.check_id in {"L1-MECH", "L1-OBS", "L1-ICE", "L1-CENS", "L1-CAL"}:
            families[result.family].extend((i, j, part.p) for j, part in enumerate(result.parts))
        else:
            families[result.family].append((i, None, result.cell_p))
    updated = list(results)
    for units in families.values():
        for (i, j, _), p in zip(units, holm([unit[2] for unit in units]), strict=True):
            result = updated[i]
            if j is None:
                updated[i] = result.model_copy(
                    update={"holm_p": p, "status": "fail" if p is not None and p < alpha else result.status}
                )
            else:
                parts = list(result.parts)
                parts[j] = parts[j].model_copy(update={"holm_p": p})
                updated[i] = result.model_copy(
                    update={"parts": tuple(parts), "status": "fail" if p is not None and p < alpha else result.status}
                )
    return tuple(updated)
