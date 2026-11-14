"""Nominal contact windows and independently calibrated monitoring plans."""

from __future__ import annotations

import io
import re
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import ndtr

from trialagentbench_verifier.properties.contracts import (
    CellDeclaration,
    CellDeclarations,
    SequentialPlan,
    SeriesDeclaration,
)
from trialagentbench_verifier.properties.deterministic import ExactChecks
from trialagentbench_verifier.properties.inputs import Dataset
from trialagentbench_verifier.properties.repair import Repair


@lru_cache(maxsize=128)
def sequential_alpha(fractions: tuple[float, ...], boundaries: tuple[float, ...]) -> float:
    """Integrate one-sided Brownian crossing probability without Monte Carlo.

    Canonical information-time correlation is sqrt(t_i/t_j), i < j. Conditioning
    successive looks on the preceding normal value gives deterministic quadrature.
    """

    def survival(index: int, previous: float) -> float:
        rho = 0.0 if index == 0 else np.sqrt(fractions[index - 1] / fractions[index])
        sd = np.sqrt(1.0 - rho * rho)
        upper = (boundaries[index] - rho * previous) / sd
        if index == len(fractions) - 1:
            return float(ndtr(upper))
        value, _ = quad(
            lambda z: np.exp(-z * z / 2) / np.sqrt(2 * np.pi) * survival(index + 1, rho * previous + sd * z),
            -np.inf,
            upper,
            epsabs=1e-9,
            epsrel=1e-9,
        )
        return float(value)

    return 1.0 - survival(0, 0.0)


def timing_checks(
    checks: ExactChecks,
    dataset: Dataset,
    analysis: pd.DataFrame,
    repair: Repair,
    cell: CellDeclaration,
    series: SeriesDeclaration,
    declarations: CellDeclarations,
    *,
    sequential_plan: SequentialPlan | None = None,
) -> None:
    """Check declared windows, calibrated S09 plans and safe realized prefixes."""
    evidence = dict(task=dataset.variant.task_id, kind=dataset.kind, index=dataset.index)
    protocol = dataset.documents.get("protocol.md", "")
    match = re.search(r"(?:planned|nominal) contact days ([\d., ]+)\.", protocol, re.IGNORECASE)
    schedule = tuple(float(x.strip()) for x in match[1].split(",")) if match else ()
    checks.test(
        "L1-DESIGNTIME",
        cell.regime_cell_id,
        schedule == series.contact_times_days,
        field="protocol.contact_times_days",
        actual=list(schedule),
        declared=list(series.contact_times_days),
        **evidence,
    )
    contacts = repair.tables.get("data/raw/exposure_records.parquet")
    times = (
        contacts.occurrence_time
        if contacts is not None
        else analysis.occurrence_time if series.endpoint_family == "longitudinal" else None
    )
    if times is not None:
        nominal = series.contact_times_days
        # trialeval/temporal_cases.py:78-95. Endpoints are fixed;
        # interior half-widths are one quarter of the smaller adjacent gap.
        irregular = (
            series.primary_endpoint_process is not None
            and series.primary_endpoint_process.temporal_structure == "irregular_mean_reversion"
        )
        widths = [
            min(t - nominal[i - 1], nominal[i + 1] - t) / 4 if irregular and 0 < i < len(nominal) - 1 else 0.0
            for i, t in enumerate(nominal)
        ]
        values = times.to_numpy(dtype=float)
        memberships = np.array([(values >= t - w) & (values <= t + w) for t, w in zip(nominal, widths, strict=True)])
        inside = memberships.any(axis=0)
        represented = [bool(memberships[i].any()) for i in range(len(nominal))]
        checks.test(
            "L1-DESIGNTIME",
            cell.regime_cell_id,
            bool(inside.all() and memberships.any(axis=1).all()),
            field="released_contact_times",
            outside=values[~inside][:5].tolist(),
            missing_windows=[t for t, seen in zip(nominal, represented, strict=True) if not seen],
            **evidence,
        )
    if "switch_time" in analysis:
        switch_plan = declarations.process_profiles[series.evaluation_series_id].switch_times_days
        checks.test(
            "L1-DESIGNTIME",
            cell.regime_cell_id,
            set(analysis.switch_time) == set(switch_plan),
            field="switch_times_days",
            actual=sorted(set(analysis.switch_time)),
            declared=list(switch_plan),
            **evidence,
        )
    if not series.monitoring_times_days:
        return
    if sequential_plan is None:
        checks.not_bound["L1-DESIGNTIME", cell.regime_cell_id].append(
            {**evidence, "field": "monitoring_plan", "reason": "Post-override S09 declaration not bound"}
        )
        return
    plan = pd.read_csv(io.StringIO(dataset.documents["monitoring_plan.csv"]), float_precision="round_trip")
    results = dataset.tables["data/monitoring_results.parquet"]
    serialized = plan.to_csv(index=False, float_format="%.17g")
    first = checks.plans.setdefault(series.evaluation_series_id, serialized)
    checks.test("L1-DESIGNTIME", cell.regime_cell_id, serialized == first, field="one_monitoring_plan", **evidence)
    fractions = tuple(plan.information_fraction)
    expected_fractions = sequential_plan.interim_information_fractions
    # simulation.py:1797-1809: the forecast supersedes process_profiles.
    valid = (
        len(plan) == 3
        and fractions == expected_fractions
        and tuple(plan.information_time_days) == sequential_plan.monitoring_times_days == series.monitoring_times_days
    )
    checks.test(
        "L1-DESIGNTIME",
        cell.regime_cell_id,
        valid,
        field="monitoring_plan",
        declared_fractions=list(expected_fractions),
        actual_fractions=list(fractions),
        **evidence,
    )
    boundaries = tuple(float(v) for v in plan.efficacy_z_boundary)
    ordered = (
        len(fractions) == 3
        and all(0 < t <= 1 for t in fractions)
        and all(a < b for a, b in zip(fractions, fractions[1:], strict=False))
    )
    finite = bool(np.isfinite(boundaries).all())
    alpha = sequential_alpha(fractions, boundaries) if ordered and finite else None
    constants = np.array(boundaries) * np.sqrt(fractions)
    # Exported boundaries are rounded to the declared six decimal places.
    rounding = 10.0 ** (-declarations.release_numeric_decimal_places)
    calibrated = alpha is not None and abs(alpha - 0.05) <= 1e-4 and np.ptp(constants) <= rounding
    checks.test(
        "L1-DESIGNTIME",
        cell.regime_cell_id,
        bool(
            calibrated
            and plan.familywise_alpha.eq(0.05).all()
            and plan.alpha_spending_family.eq("obrien_fleming").all()
        ),
        field="monitoring_familywise_alpha",
        actual_alpha=alpha,
        constants=constants.tolist(),
        **evidence,
    )
    n = len(results)
    checks.test(
        "L1-DESIGNTIME",
        cell.regime_cell_id,
        1 <= n <= len(plan)
        and tuple(results.look_id) == tuple(plan.look_id.iloc[:n])
        and tuple(results.information_time) == tuple(plan.information_time_days.iloc[:n]),
        field="realized_prefix",
        n_looks=n,
        **evidence,
    )
    for i, row in enumerate(results.itertuples(index=False)):
        if i >= len(plan):
            checks.test(
                "L1-DESIGNTIME",
                cell.regime_cell_id,
                False,
                field="look_beyond_plan",
                look=str(row.look_id),
                **evidence,
            )
            continue
        expected = plan.iloc[i]
        checks.test(
            "L1-DESIGNTIME",
            cell.regime_cell_id,
            float(str(row.information_fraction)) == expected.information_fraction
            and float(str(row.efficacy_z_boundary)) == expected.efficacy_z_boundary,
            field="realized_information_and_boundary",
            look=row.look_id,
            **evidence,
        )
        crossed = float(str(row.z_statistic)) >= expected.efficacy_z_boundary
        terminal = i == n - 1
        decision_ok = (
            (not crossed and row.decision == "continue")
            if not terminal
            else (
                (crossed and "efficacy" in str(row.decision))
                or (
                    not crossed
                    and n == len(plan)
                    and row.decision in {"final", "complete", "no_rejection", "final_analysis"}
                )
            )
        )
        checks.test(
            "L1-DESIGNTIME",
            cell.regime_cell_id,
            bool(decision_ok),
            field="stopping_indicator",
            look=row.look_id,
            decision=row.decision,
            z=row.z_statistic,
            boundary=float(expected.efficacy_z_boundary),
            **evidence,
        )
