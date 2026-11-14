"""Prespecified diagnostic descriptions and their released-column prerequisites."""

from __future__ import annotations

import re

import pandas as pd

from trialagentbench_verifier.properties.contracts import CellDeclaration, CellDeclarations, SeriesDeclaration
from trialagentbench_verifier.properties.deterministic import ExactChecks
from trialagentbench_verifier.properties.inputs import Dataset


def diagnostic_sources(dataset: Dataset, series: SeriesDeclaration, public_key: str) -> dict[str, set[str]]:
    """Name actual table/column dependencies, independently of SAP prose.

    These are the operations rendered by release construction,
    release_projection/documents.py:490-577 and its participant diagnostic
    dictionary. Raw outcomes must join to their participant roster; finding a
    column in an unrelated table does not make the procedure computable.
    """
    baseline = {f.field_id for f in series.clinical_profile.baseline_fields if f.value_type != "categorical"}
    ready = dataset.variant.context == "C1"
    main = "data/analysis.parquet" if ready else "data/raw/participant_records.parquet"
    sources = {main: {"participant_id", "assigned_arm"}}
    needed = sources[main]
    if public_key in {
        "model_form_public",
        "censoring_followup_public",
        "secular_trend_public",
        "proportional_hazards_public",
    }:
        if public_key in {"model_form_public", "censoring_followup_public", "secular_trend_public"}:
            needed.update(baseline)
        if ready:
            needed.update({"observed_time", "event_observed"})
        else:
            sources["data/raw/event_history.parquet"] = {"participant_id", "event_id", "event_kind", "occurrence_time"}
        if public_key == "secular_trend_public":
            needed.update({"cluster_id", "entry_time", "switch_time"})
    elif public_key in {
        "dispersion_public",
        "missingness_pattern_public",
        "model_residuals_public",
        "endpoint_ascertainment_public",
    }:
        outcome = needed if ready else sources.setdefault("data/raw/endpoint_observations.parquet", {"participant_id"})
        if public_key == "endpoint_ascertainment_public" and series.endpoint_family == "time_to_event":
            if ready:
                outcome.update(
                    {"routine_endpoint", "validation_stratum", "validation_selected", "adjudicated_endpoint"}
                )
            else:
                sources.pop("data/raw/endpoint_observations.parquet")
                needed.add("validation_stratum")
                sources["data/raw/event_history.parquet"] = {"participant_id", "event_id", "event_by_contact"}
        else:
            outcome.add(series.endpoint)
            if public_key == "endpoint_ascertainment_public":
                outcome.update({"validation_stratum", "validation_selected", "adjudicated_endpoint"})
            if public_key in {"missingness_pattern_public", "model_residuals_public"}:
                outcome.update({"occurrence_time", "observed"})
                if public_key == "model_residuals_public":
                    needed.update(baseline)
    else:
        raise ValueError(f"No released-column contract for diagnostic {public_key}")
    return sources


def diagnostic_checks(
    checks: ExactChecks,
    dataset: Dataset,
    analysis: pd.DataFrame,
    cell: CellDeclaration,
    declarations: CellDeclarations,
) -> None:
    """Require exactly the default route's diagnostics and real source columns.

    Release construction documents.py:545-577,650 renders only the series
    prespecified route (catalog.py:66-93), including a negative-control default.
    Severity bands are descriptions, never additional conformance gates.
    """
    if dataset.variant.context not in {"C1", "C3"}:
        return
    series = next(s for s in declarations.series if s.evaluation_series_id == cell.evaluation_series_id)
    routes = [
        r
        for c in declarations.cells
        for r in (*c.reference_analyses, *([c.negative_control] if c.negative_control else []))
    ]
    route = next((r for r in routes if r.route_id == series.default_route_id), None)
    evidence = dict(task=dataset.variant.task_id, kind=dataset.kind, index=dataset.index)
    if route is None:
        checks.not_bound["L1-DIAG", cell.regime_cell_id].append(
            {**evidence, "field": "default_route", "reason": "Default diagnostic requirements not declared"}
        )
        return
    expected = set(route.diagnostic_requirements)
    sap = dataset.documents.get("statistical_analysis_plan.md", "")
    paragraphs = [part.split("\n\n", 1)[0] for part in sap.split("Prespecified diagnostic:")[1:]]
    registry = {entry.public_key: entry for entry in declarations.diagnostic_registry.values()}

    def normalize(text: str) -> str:
        return re.sub(r"[-_]", " ", text.lower())

    described: list[str] = []
    unidentified = []
    for paragraph in paragraphs:
        keys = [
            key
            for key, entry in registry.items()
            if f"report the result on the {normalize(entry.diagnostic_scale)} scale" in normalize(paragraph)
        ]
        if len(keys) == 1:
            described.extend(keys)
        else:
            unidentified.append(paragraph[:160])
    checks.test(
        "L1-DIAG",
        cell.regime_cell_id,
        set(described) == expected and len(described) == len(expected) and not unidentified,
        field="default_route_diagnostic_keys",
        expected=sorted(expected),
        actual=described,
        unidentified=unidentified,
        **evidence,
    )
    for public_key in sorted(expected):
        entry = registry[public_key]
        metrics = set(entry.metric_units)
        if entry.severity_thresholds:
            metrics.add(entry.severity_thresholds.metric_name)
            metrics.update(entry.severity_thresholds.decision_metric_names.model_dump().values())
        section = "\n".join(
            p
            for p in paragraphs
            if f"report the result on the {normalize(entry.diagnostic_scale)} scale" in normalize(p)
        )
        missing = [metric for metric in sorted(metrics) if not re.search(rf"\b{re.escape(metric)}\b", section)]
        dependencies = diagnostic_sources(dataset, series, public_key)
        missing_fields = {
            table: sorted(columns - {str(c) for c in dataset.tables[table]} if table in dataset.tables else columns)
            for table, columns in dependencies.items()
            if table not in dataset.tables or not columns <= {str(c) for c in dataset.tables[table]}
        }
        checks.test(
            "L1-DIAG",
            cell.regime_cell_id,
            bool(section) and not missing and not missing_fields,
            field=public_key,
            missing_metrics=missing,
            missing_fields=missing_fields,
            required_sources={t: sorted(c) for t, c in dependencies.items()},
            **evidence,
        )
