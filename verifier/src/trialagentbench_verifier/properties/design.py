"""Released design evidence, planned schedules and diagnostic procedures."""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

from trialagentbench_verifier.properties.contracts import CellDeclaration, CellDeclarations, SeriesDeclaration
from trialagentbench_verifier.properties.deterministic import ExactChecks
from trialagentbench_verifier.properties.diagnostics import diagnostic_checks as diagnostic_checks
from trialagentbench_verifier.properties.inputs import Dataset
from trialagentbench_verifier.properties.instantiation import EstimandInstantiation, estimand_attributes
from trialagentbench_verifier.properties.schedules import timing_checks as timing_checks


def design_checks(
    checks: ExactChecks,
    dataset: Dataset,
    analysis: pd.DataFrame,
    cell: CellDeclaration,
    series: SeriesDeclaration,
    declarations: CellDeclarations,
    *,
    estimand_instantiation: EstimandInstantiation | None = None,
) -> None:
    """Check every mapped D-tier evidence item and every declared route obligation."""
    profile = next(d for d in declarations.design_profiles if d.design_profile_id == series.design_profile_id)
    definition = next(d for d in declarations.design_definitions if d.tier == profile.design_tier)
    with Path(__file__).with_name("design_evidence_map.csv").open() as stream:
        rows = [r for r in csv.DictReader(stream) if r["design_tier"] == profile.design_tier]
    if {r["required_public_evidence"] for r in rows} != set(definition.required_public_evidence):
        raise ValueError("Design evidence map does not cover the declaration exactly")
    protocol = dataset.documents.get("protocol.md", "")
    sap = dataset.documents.get("statistical_analysis_plan.md", "")
    population = series.target_population.replace("_", " ")
    actual = estimand_attributes(protocol)
    evidence = dict(task=dataset.variant.task_id, kind=dataset.kind, index=dataset.index)
    attribute_matches = {}
    for name in ("population", "treatment_conditions", "variable", "intercurrent_events", "summary_measure"):
        if estimand_instantiation is None:
            checks.not_bound["L1-DESIGN", cell.regime_cell_id].append(
                {**evidence, "field": f"estimand.{name}", "reason": "Instantiated declaration absent"}
            )
            attribute_matches[name] = False
            continue
        expected = getattr(estimand_instantiation, name)
        matches = (
            actual.get("estimand_id") == estimand_instantiation.estimand_id == series.estimand_id
            and actual.get(name) == expected
        )
        attribute_matches[name] = matches
        checks.test(
            "L1-DESIGN",
            cell.regime_cell_id,
            matches,
            field=f"estimand.{name}",
            declared=expected,
            actual=actual.get(name),
            **evidence,
        )
    estimand_ok = all(attribute_matches.values())
    intercurrent_ok = attribute_matches["intercurrent_events"]
    # D4 requires an executable analysis instruction, not a planning heading.
    # release_projection/documents.py:414-435 and designs/common.py:934-990.
    if series.design_profile_id == "TE-DP05":
        correction = (
            "participant-weighted risk sets" in sap
            and "delete all clusters" in sap
            and "refit" in sap
            and "Student-t" in sap
        )
    elif series.design_profile_id == "TE-DP06":
        correction = (
            "weighted least squares" in sap
            and "entry period" in sap
            and "delete one complete randomized cluster" in sap
            and "refit" in sap
        )
    else:
        plan_text = dataset.documents.get("monitoring_plan.csv", "")
        correction = all(
            token in plan_text for token in ("efficacy_z_boundary", "information_fraction", "obrien_fleming")
        )
    predicates = {
        "allocation": {"participant_id", "assigned_arm"} <= set(analysis) and "Treatment groups:" in protocol,
        "population": f"Target population: {population}." in protocol and "participant_id" in analysis,
        "endpoint": "Endpoint definition:" in protocol
        and any(f in analysis for f in (series.endpoint, "observed_time", "routine_endpoint")),
        "treatment_change": "Treatment discontinuation:" in protocol
        and ("treatment_discontinuation_time" in analysis or "data/raw/exposure_records.parquet" in dataset.tables),
        "intercurrent": intercurrent_ok,
        "timing": any(
            f in analysis
            for f in (
                "entry_time",
                "observed_time",
                "occurrence_time",
                "assessment_time",
                "treatment_discontinuation_time",
            )
        ),
        "estimand": estimand_ok,
        "baseline": all(f.field_id in analysis for f in series.clinical_profile.baseline_fields),
        "validation": cell.observation_kind != "validation_subsample"
        or {"validation_selected", "adjudicated_endpoint", "validation_stratum"} <= set(analysis),
        "structure": "cluster_id" in analysis or "data/monitoring_results.parquet" in dataset.tables,
        "period": "monitoring_plan.csv" in dataset.documents
        or "calendar_period" in analysis
        or "cluster_id" in analysis,
        "correction": correction,
    }
    evidence = dict(task=dataset.variant.task_id, kind=dataset.kind, index=dataset.index)
    for row in rows:
        if row["predicate"] == "correction" and not correction and dataset.variant.context not in {"C1", "C3"}:
            checks.not_bound["L1-DESIGN", cell.regime_cell_id].append(
                {
                    **evidence,
                    "field": row["required_public_evidence"],
                    "reason": "No released correction procedure in unlocked view",
                }
            )
            continue
        checks.test(
            "L1-DESIGN",
            cell.regime_cell_id,
            bool(predicates[row["predicate"]]),
            field=row["required_public_evidence"],
            artefacts=row["artefacts"],
            **evidence,
        )
    for route in cell.reference_analyses:
        for obligation in route.design_obligations:
            if obligation == "at_least_20_clusters_per_arm":
                counts = (
                    analysis.groupby("assigned_arm").cluster_id.nunique()
                    if "cluster_id" in analysis
                    else pd.Series(dtype=int)
                )
                ok = bool(len(counts) == 2 and counts.ge(20).all())
            elif obligation == "cluster_randomization":
                ok = "cluster_id" in analysis and bool(
                    analysis.groupby("cluster_id")[
                        "randomization_group" if "switch_time" in analysis else "assigned_arm"
                    ]
                    .nunique()
                    .eq(1)
                    .all()
                )
            elif obligation == "declared_rollout_timing":
                ok = {"entry_time", "switch_time"} <= set(analysis) and bool(
                    analysis.assigned_arm.eq("active").eq(analysis.entry_time.ge(analysis.switch_time)).all()
                )
            elif obligation == "calendar_period_adjustment":
                profile_process = declarations.process_profiles[series.evaluation_series_id]
                ok = "calendar_period" in analysis and bool(
                    analysis.calendar_period.eq(
                        np.floor(analysis.entry_time / profile_process.stepped_wedge_period_days)
                    ).all()
                )
            elif obligation in {"global_type_i_error_control", "realized_look_inference"}:
                plan = pd.read_csv(
                    io.StringIO(dataset.documents.get("monitoring_plan.csv", "")), float_precision="round_trip"
                )
                ok = bool(
                    plan.familywise_alpha.eq(0.05).all() and plan.alpha_spending_family.eq("obrien_fleming").all()
                )
                if obligation == "realized_look_inference":
                    ok &= (
                        "stagewise" in route.procedure.fitting_specification.lower()
                        and "data/monitoring_results.parquet" in dataset.tables
                    )
            elif obligation == "fixed_quantile_knots":
                field_name = next(
                    f.field_id for f in series.clinical_profile.baseline_fields if f.source_field_id == "bmi"
                )
                # common.py:1015-1027 declares empirical .05/.35/.65/.95 knots.
                # The auxiliary normal-quantile diagnostic is a different procedure.
                text = "\n".join(dataset.documents.values())
                if not re.search(r"empirical.*?(?:knots|quantiles)|(?:knots|quantiles).*?all randomized", text, re.I):
                    checks.not_bound["L1-DESIGN", cell.regime_cell_id].append(
                        {
                            **evidence,
                            "field": "design_obligations",
                            "obligation": obligation,
                            "reason": "Empirical fixed-knot procedure absent from released documents",
                        }
                    )
                    continue
                knots = analysis[field_name].quantile([0.05, 0.35, 0.65, 0.95]).to_numpy()
                ok = bool(
                    np.isfinite(knots).all()
                    and (np.diff(knots) > 0).all()
                    and all(value in text for value in ("0.05", "0.35", "0.65", "0.95"))
                )
            elif obligation == "prespecified_bmi_subgroup_empirical_reference_population":
                ok = (
                    "reference_population_member" in analysis
                    and bool(analysis.reference_population_member.any())
                    and "BMI threshold" in protocol
                )
            elif obligation == "participant_average_weighting":
                if not sap and dataset.variant.context not in {"C1", "C3"}:
                    checks.not_bound["L1-DESIGN", cell.regime_cell_id].append(
                        {
                            **evidence,
                            "field": "design_obligations",
                            "obligation": obligation,
                            "reason": "Weighting procedure absent from released documents",
                        }
                    )
                    continue
                ok = "cluster_id" in analysis and "participant-weighted risk sets" in sap
            else:
                raise ValueError(f"Unimplemented design obligation: {obligation}")
            checks.test(
                "L1-DESIGN",
                cell.regime_cell_id,
                bool(ok),
                field="design_obligations",
                obligation=obligation,
                route=route.route_id,
                **evidence,
            )
