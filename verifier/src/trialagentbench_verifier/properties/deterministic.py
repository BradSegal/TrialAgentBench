"""Exact declaration-to-release checks, independent of construction and scoring."""

from __future__ import annotations

import json
import math
import re
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from trialagentbench_verifier.properties.contracts import (
    CellDeclaration,
    CellDeclarations,
    CheckId,
    CheckResult,
    DeterministicResult,
    PrivateTaskAssignment,
    SeriesDeclaration,
    TaskVariant,
)
from trialagentbench_verifier.properties.inputs import Dataset
from trialagentbench_verifier.properties.instantiation import ClusterSizeLaw, completed_cohort_size
from trialagentbench_verifier.properties.repair import C5_ACTIONS, Repair


@dataclass
class ExactChecks:
    """Accumulate exact decisions with every failed assertion's evidence."""

    counts: dict[tuple[CheckId, str], int] = field(default_factory=lambda: defaultdict(int))
    failures: dict[tuple[CheckId, str], list[dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    unavailable: dict[tuple[CheckId, str], list[dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    not_bound: dict[tuple[CheckId, str], list[dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    plans: dict[str, str] = field(default_factory=dict)
    pairings: list[dict[str, Any]] = field(default_factory=list)

    def test(self, check: CheckId, cell: str, condition: bool, **evidence: Any) -> None:
        """Record one exact assertion, keeping its evidence only on failure."""
        self.counts[check, cell] += 1
        if not condition:
            self.failures[check, cell].append(evidence)

    def results(self) -> tuple[list[DeterministicResult], list[CheckResult]]:
        """Return exact-check counts and cell verdicts, preserving unavailable decisions."""
        exact, verdicts = [], []
        for check, cell in sorted(self.counts.keys() | self.unavailable.keys() | self.not_bound.keys()):
            failures = self.failures[check, cell]
            exact.append(
                DeterministicResult(
                    check_id=check,
                    cell_ref=cell,
                    n_checked=self.counts[check, cell],
                    n_failed=len(failures),
                    failures=tuple(failures),
                )
            )
            verdicts.append(
                CheckResult(
                    check_id=check,
                    scope="cell",
                    scope_id=cell,
                    family=None,
                    parts=(),
                    route_p={},
                    cell_p=None,
                    holm_p=None,
                    status=(
                        "fail"
                        if failures
                        else (
                            "unavailable"
                            if self.unavailable[check, cell]
                            or self.not_bound[check, cell]
                            or not self.counts[check, cell]
                            else "pass"
                        )
                    ),
                )
            )
        return exact, verdicts


def identity_checks(
    checks: ExactChecks,
    declarations: CellDeclarations,
    variants: tuple[TaskVariant, ...],
    *,
    task_assignments: tuple[PrivateTaskAssignment, ...] = (),
    private_cells: tuple[CellDeclaration, ...] = (),
    released_documents: dict[str, dict[str, str]] | None = None,
) -> None:
    """Check distinct-cell identity, split membership, tiers and view counts.

    Parameters
    ----------
    checks, declarations, variants
        Result accumulator, validated declarations and all authenticated tasks.
    task_assignments, private_cells, released_documents
        Independent private catalog assignments, complete inventory and opened documents.
    """
    mapping: dict[str, list[str]] = defaultdict(list)
    for row in declarations.crosswalk:
        mapping[row.cell_ref].append(row.regime_cell_id)
    by_regime = {c.regime_cell_id: c for c in declarations.cells}
    actual = {v.cell_ref for v in variants}
    for ref in sorted(set(mapping) | actual):
        regimes = mapping[ref]
        label = regimes[0] if len(regimes) == 1 else ref
        checks.test(
            "L1-ID",
            label,
            len(regimes) == 1 and regimes[0] in by_regime and ref in actual,
            field="crosswalk",
            cell_ref=ref,
            regime_ids=regimes,
        )
    split_checks(checks, declarations, variants, task_assignments, private_cells)
    observed_tiers: dict[str, set[str]] = defaultdict(set)
    for v in variants:
        regimes = mapping[v.cell_ref]
        if len(regimes) != 1 or regimes[0] not in by_regime:
            continue
        cell = by_regime[regimes[0]]
        for name, expected_value in (
            ("default_response", cell.default_response),
            ("assumption_tier", cell.assumption_tier),
            ("evaluation_series_id", cell.evaluation_series_id),
        ):
            # Historical public keys redact these labels. Never infer them from
            # the same declaration being tested (grading_key.py:210-235).
            values = [("key", v.key_record[name])] if name in v.key_record else []
            label = name.replace("_", " ")
            for document, text in (released_documents or {}).get(v.task_id, {}).items():
                for match in re.finditer(rf"^{re.escape(label)}: (.+)$", text, re.M | re.I):
                    values.append((document, match[1].strip()))
            for source, value in values:
                checks.test(
                    "L1-ID",
                    cell.regime_cell_id,
                    value == expected_value,
                    task=v.task_id,
                    source=source,
                    field=name,
                    expected=expected_value,
                    actual=value,
                )
                if name == "assumption_tier":
                    observed_tiers[cell.evaluation_series_id].add(str(value))
            if not values:
                checks.not_bound["L1-ID", cell.regime_cell_id].append(
                    {"task": v.task_id, "field": name, "reason": "Label absent from released key and documents"}
                )
    for series in declarations.series:
        series_refs = {
            str(v.key_record.get("series_ref"))
            for v in variants
            if mapping[v.cell_ref]
            and by_regime[mapping[v.cell_ref][0]].evaluation_series_id == series.evaluation_series_id
        }
        checks.test(
            "L1-ID",
            series.evaluation_series_id,
            len(series_refs) == 1,
            field="series_ref",
            observed=sorted(series_refs),
        )
        expected = {tier for tier in series.assumption_tiers if f"{series.evaluation_series_id}-{tier}" in by_regime}
        found = observed_tiers[series.evaluation_series_id]
        if found:
            checks.test(
                "L1-ID",
                series.evaluation_series_id,
                found == expected,
                field="series.assumption_tiers",
                expected=sorted(expected),
                actual=sorted(found),
            )
        else:
            checks.not_bound["L1-ID", series.evaluation_series_id].append(
                {"field": "series.assumption_tiers", "reason": "No released tier labels"}
            )
    for cell in declarations.cells:
        refs = {ref for ref, values in mapping.items() if cell.regime_cell_id in values}
        selected = [v for v in variants if v.cell_ref in refs]
        for context in cell.context_ids:
            tasks = [v for v in selected if v.context == context]
            checks.test(
                "L1-COUNT",
                cell.regime_cell_id,
                len(tasks) == cell.replicates_per_regime_context and len({v.task_key for v in tasks}) == len(tasks),
                context=context,
                expected=cell.replicates_per_regime_context,
                actual=len(tasks),
            )
    rotation_checks(checks, declarations, variants, private_cells)


def rotation_checks(
    checks: ExactChecks,
    declarations: CellDeclarations,
    variants: tuple[TaskVariant, ...],
    private_cells: tuple[CellDeclaration, ...],
) -> None:
    """Check C5 condition multisets without assuming an unexported task rank.

    Presentation ranks are not released, so compare condition multisets
    across inventory-ordered series and rank blocks.
    Pair parity uses only authenticated released base_world_id identities.
    """
    mapping = {r.cell_ref: r.regime_cell_id for r in declarations.crosswalk}
    by_cell = {c.regime_cell_id: c for c in declarations.cells}
    series_order = tuple(dict.fromkeys(c.evaluation_series_id for c in private_cells))
    views: dict[str, list[tuple[TaskVariant, str]]] = defaultdict(list)
    for variant in variants:
        if variant.context != "C5" or variant.cell_ref not in mapping:
            continue
        key: Any = variant.key_record
        integrity = key.get("data_integrity") or {}
        action = integrity.get("resolution_action")
        condition = integrity.get("condition_id") or next(
            (c for c, a in C5_ACTIONS.items() if a == action), "undeclared"
        )
        views[mapping[variant.cell_ref]].append((variant, condition))
    for cell_id, cell in by_cell.items():
        if not series_order or cell.evaluation_series_id not in series_order:
            checks.not_bound["L1-CTX", cell_id].append(
                {"field": "condition_rotation_multiset", "reason": "Construction inventory order is not bound"}
            )
            continue
        position = series_order.index(cell.evaluation_series_id)
        expected = Counter(
            tuple(C5_ACTIONS)[(position + (rank - 1) * len(series_order)) % len(C5_ACTIONS)]
            for rank in range(1, cell.replicates_per_regime_context + 1)
        )
        actual = Counter(condition for _, condition in views[cell_id])
        checks.test(
            "L1-CTX",
            cell_id,
            actual == expected,
            field="condition_rotation_multiset",
            expected=dict(expected),
            actual=dict(actual),
            tasks=[v.task_id for v, _ in views[cell_id]],
        )
    for series_id in dict.fromkeys(c.evaluation_series_id for c in declarations.cells):
        cells = {c.regime_cell_id for c in declarations.cells if c.evaluation_series_id == series_id}
        if len(cells) < 2:
            continue
        worlds: dict[str, list[tuple[str, TaskVariant, str]]] = defaultdict(list)
        for cell_id in cells:
            for variant, condition in views[cell_id]:
                world = variant.key_record.get("base_world_id")
                if not isinstance(world, str) or not world:
                    checks.not_bound["L1-CTX", series_id].append(
                        {
                            "task": variant.task_id,
                            "field": "twin_condition",
                            "reason": "No released shared-world identity",
                        }
                    )
                else:
                    worlds[world].append((cell_id, variant, condition))
        for world, group in sorted(worlds.items()):
            evidence = {
                "field": "twin_condition",
                "base_world_id": world,
                "pairing_basis": "authenticated released grading key base_world_id",
                "tasks": [v.task_id for _, v, _ in group],
            }
            if len(group) != len(cells) or {c for c, _, _ in group} != cells:
                checks.not_bound["L1-CTX", series_id].append(
                    {**evidence, "reason": "Released world does not identify one task in each twin cell"}
                )
                continue
            conditions = [condition for _, _, condition in group]
            checks.pairings.append({"series_id": series_id, **evidence, "conditions": conditions})
            checks.test("L1-CTX", series_id, len(set(conditions)) == 1, actual=conditions, **evidence)


def split_checks(
    checks: ExactChecks,
    declarations: CellDeclarations,
    variants: tuple[TaskVariant, ...],
    assignments: tuple[PrivateTaskAssignment, ...],
    private_cells: tuple[CellDeclaration, ...],
) -> None:
    """Tie every released task and public cell reference to the private catalog.

    Independently authenticated task assignments supply the private binding.
    Neither private labels nor split membership are inferred from the public map.
    """
    inventory = {c.regime_cell_id: c for c in private_cells}
    if not assignments or len(inventory) != 23 or len(private_cells) != 23:
        checks.not_bound["L1-SPLIT", declarations.split].append(
            {
                "field": "private_task_assignments",
                "reason": "Independent task assignment or 23-cell private inventory not bound",
            }
        )
        return
    by_task: dict[str, list[PrivateTaskAssignment]] = defaultdict(list)
    for row in assignments:
        by_task[row.task_id].append(row)
    reverse: dict[str, list[str]] = defaultdict(list)
    for crosswalk_row in declarations.crosswalk:
        reverse[crosswalk_row.regime_cell_id].append(crosswalk_row.cell_ref)
    for variant in variants:
        rows = by_task[variant.task_id]
        evidence = dict(task=variant.task_id, context=variant.context)
        checks.test(
            "L1-SPLIT",
            declarations.split,
            len(rows) == 1,
            field="private_assignment_unique",
            count=len(rows),
            **evidence,
        )
        if len(rows) != 1:
            for name in ("private_assignment_membership", "private_assignment_cell_ref"):
                checks.not_bound["L1-SPLIT", declarations.split].append(
                    {**evidence, "field": name, "reason": "Private task assignment is absent or ambiguous"}
                )
            continue
        label = rows[0].scenario_label
        cell = inventory.get(label)
        allowed = cell is not None and cell.withheld_only == (declarations.split == "withheld")
        checks.test(
            "L1-SPLIT",
            declarations.split,
            allowed,
            field="private_assignment_membership",
            scenario_label=label,
            **evidence,
        )
        refs = reverse[label]
        checks.test(
            "L1-SPLIT",
            declarations.split,
            len(refs) == 1 and variant.cell_ref == refs[0],
            field="private_assignment_cell_ref",
            actual=variant.cell_ref,
            expected=refs,
            **evidence,
        )


def binding_checks(
    checks: ExactChecks,
    dataset: Dataset,
    cell: CellDeclaration,
    series: SeriesDeclaration,
    declarations: CellDeclarations,
) -> None:
    """Compare the named declaration fields with keys and literal protocol lines."""
    key: Any = dataset.variant.key_record
    binding = key["target"]["binding"]
    fields = {
        "estimand_id": "estimand_id",
        "primary_effect_scale": "effect_measure",
        "result_unit": "result_unit",
        "favorable_direction": "favorable_direction",
        "assessment_horizon_days": "assessment_horizon_days",
        "primary_endpoint_id": "endpoint_id",
        "target_population": "analysis_population_id",
    }
    for source, destination in fields.items():
        checks.test(
            "L1-BIND",
            cell.regime_cell_id,
            getattr(series, source) == binding.get(destination),
            task=dataset.variant.task_id,
            field=source,
            declared=getattr(series, source),
            actual=binding.get(destination),
        )
    design = next(d for d in declarations.design_profiles if d.design_profile_id == series.design_profile_id)
    for source, destination in (("design_tier", "design_tier"), ("analysis_design_family", "design_family")):
        checks.test(
            "L1-BIND",
            cell.regime_cell_id,
            getattr(design, source) == key.get(destination),
            task=dataset.variant.task_id,
            field=source,
            declared=getattr(design, source),
            actual=key.get(destination),
        )
    # release_projection/documents.py:335-394 has no literal
    # counterpart for these fields, so the assertion is not observable.
    for name in ("primary_estimand_id", "contrast_direction", "public_question"):
        checks.not_bound["L1-BIND", cell.regime_cell_id].append(
            {"task": dataset.variant.task_id, "field": name, "reason": "No literal released binding"}
        )


def context_checks(
    checks: ExactChecks, dataset: Dataset, cell: CellDeclaration, declarations: CellDeclarations, repair: Repair
) -> None:
    """Verify evidence flags and C5 issue coordinates on opened source records."""
    context = next(c for c in declarations.context_definitions if c.tier == dataset.variant.context)
    key: Any = dataset.variant.key_record
    integrity = key.get("data_integrity")
    tables = dataset.tables
    observed = {
        "analysis_ready_tables": "data/analysis.parquet" in tables,
        "locked_sap": bool(dataset.documents.get("statistical_analysis_plan.md", "").strip()),
        "raw_participant_evidence": {
            "data/raw/participant_records.parquet",
            "data/raw/event_history.parquet",
            "data/raw/exposure_records.parquet",
        }
        <= tables.keys(),
    }
    for field_name, value in observed.items():
        checks.test(
            "L1-CTX",
            cell.regime_cell_id,
            value == getattr(context, field_name),
            task=dataset.variant.task_id,
            kind=dataset.kind,
            index=dataset.index,
            field=field_name,
            declared=getattr(context, field_name),
            actual=value,
        )
    evidence = dict(task=dataset.variant.task_id, kind=dataset.kind, index=dataset.index)
    if context.controlled_recoverable_defect:
        action = integrity.get("resolution_action") if integrity else None
        condition = next((c for c, a in C5_ACTIONS.items() if a == action), None)
        # The historical key redacts condition_id but its action uniquely names
        # the condition (data_integrity.py:50-57, 563-690).
        declared_condition = integrity.get("condition_id", condition) if integrity else None
        checks.test(
            "L1-CTX",
            cell.regime_cell_id,
            bool(condition and condition == declared_condition),
            field="declared_integrity_condition",
            declared_condition=declared_condition,
            action=action,
            **evidence,
        )
        domain = integrity.get("affected_domain") if integrity else None
        observed_action = repair.actions.get(domain) if domain else None
        observed_condition = next((c for c, a in C5_ACTIONS.items() if a == observed_action), None)
        has_defect = any(repair.affected_rows.values())
        checks.test(
            "L1-CTX",
            cell.regime_cell_id,
            condition is not None and has_defect == (action != "accept_transfer_without_change"),
            field="controlled_recoverable_defect",
            actual=has_defect,
            declared_action=action,
            **evidence,
        )
        checks.test(
            "L1-CTX",
            cell.regime_cell_id,
            condition is not None and observed_condition == condition and observed_action == action,
            field="data_integrity.condition_and_action",
            declared_condition=condition,
            actual_condition=observed_condition,
            declared_action=action,
            actual_action=observed_action,
            **evidence,
        )
    elif dataset.variant.context in {"C1", "C2"}:
        checks.not_bound["L1-CTX", cell.regime_cell_id].append(
            {**evidence, "field": "controlled_recoverable_defect", "reason": "No raw transfer in this view"}
        )
    if integrity:
        checks.test(
            "L1-CTX",
            cell.regime_cell_id,
            context.controlled_recoverable_defect,
            field="undeclared_integrity_metadata",
            **evidence,
        )
    if dataset.kind == "scored" and integrity and integrity.get("affected_domain"):
        domain = integrity["affected_domain"]
        checks.test(
            "L1-CTX",
            cell.regime_cell_id,
            repair.actions.get(domain) == integrity["resolution_action"],
            task=dataset.variant.task_id,
            field="data_integrity.resolution_action",
            actual=repair.actions.get(domain),
            declared=integrity["resolution_action"],
        )
        found = repair.affected_keys.get(domain, set())
        declared = {
            tuple(
                (
                    str(struct.unpack(">d", bytes.fromhex(value[1]))[0])
                    if value[0] in {"float64", "double"}
                    else str(value[1])
                )
                for value in json.loads(item)
            )
            for item in integrity["selected_observed_compound_keys"]
        }
        checks.test(
            "L1-CTX",
            cell.regime_cell_id,
            found == declared,
            task=dataset.variant.task_id,
            field="data_integrity.affected_keys",
            observed=sorted(found),
            declared=sorted(declared),
        )
        checks.test(
            "L1-CTX",
            cell.regime_cell_id,
            repair.affected_rows.get(domain, 0) == integrity["observed_affected_row_count"]
            and len(found) == integrity["observed_issue_group_count"],
            task=dataset.variant.task_id,
            field="data_integrity.counts",
            actual_rows=repair.affected_rows.get(domain, 0),
            actual_groups=len(found),
        )


def raw_transfer_checks(checks: ExactChecks, dataset: Dataset, cell: CellDeclaration) -> None:
    """Inspect C3/C4 originals before any repair can erase an undeclared defect.

    construct.py:754-843 injects transfer incidents only for C5.
    Ordinary raw transfers therefore contain neither copies nor version rows.
    """
    if dataset.variant.context not in {"C3", "C4"}:
        return
    raw = {name: table for name, table in dataset.tables.items() if name.startswith("data/raw/")}
    issues = [
        {
            "table": name,
            "duplicate_rows": int(table.duplicated(keep=False).sum()),
            "version_rows": len(table) if "record_version" in table else 0,
        }
        for name, table in raw.items()
        if table.duplicated(keep=False).any() or "record_version" in table
    ]
    checks.test(
        "L1-CTX",
        cell.regime_cell_id,
        bool(raw) and not issues,
        task=dataset.variant.task_id,
        kind=dataset.kind,
        index=dataset.index,
        field="controlled_recoverable_defect",
        issues=issues,
    )


def support_checks(
    checks: ExactChecks,
    dataset: Dataset,
    analysis: pd.DataFrame,
    cell: CellDeclaration,
    series: SeriesDeclaration,
    *,
    cluster_law: ClusterSizeLaw | None = None,
    facility_census: pd.Series | None = None,
) -> None:
    """Check declared baseline ranges, levels and recruitment support exactly."""
    for feature in series.clinical_profile.baseline_fields:
        if feature.field_id not in analysis:
            checks.test(
                "L1-SUPPORT",
                cell.regime_cell_id,
                False,
                task=dataset.variant.task_id,
                index=dataset.index,
                field=feature.field_id,
                reason="missing baseline",
            )
            continue
        values = analysis[feature.field_id]
        ok = values.notna()
        if feature.support:
            numeric = pd.to_numeric(values, errors="raise")
            ok &= numeric.ge(feature.support.lower)
            if feature.support.upper is not None:
                ok &= numeric.le(feature.support.upper)
        if feature.allowed_values:
            ok &= values.isin(feature.allowed_values)
        if feature.categories:
            ok &= values.isin(feature.categories)
        checks.test(
            "L1-SUPPORT",
            cell.regime_cell_id,
            bool(ok.all()),
            task=dataset.variant.task_id,
            kind=dataset.kind,
            index=dataset.index,
            field=feature.field_id,
            n_outside=int((~ok).sum()),
            examples=values.loc[~ok].astype(str).head(3).tolist(),
        )
    recruitment = series.clinical_profile.recruitment
    countries = set(analysis.country_code.dropna())
    checks.test(
        "L1-SUPPORT",
        cell.regime_cell_id,
        countries <= set(recruitment.country_codes),
        task=dataset.variant.task_id,
        index=dataset.index,
        field="countries",
        observed=sorted(countries),
    )
    evidence = dict(task=dataset.variant.task_id, kind=dataset.kind, index=dataset.index)
    # Release construction simulation.py:1704-1729 opens one site per
    # sampled cluster for facility cohorts, with capacity participant_count.
    # Integer-site cohorts alone use max(2, ceil(2 * participant_count / N)).
    facility = recruitment.site_count == "randomization_units"
    cohort_n = completed_cohort_size(dataset.documents.get("protocol.md", "")) if facility else None
    if facility:
        units = (
            cluster_law.count(series.evaluation_series_id, cohort_n)
            if cluster_law is not None and cohort_n is not None
            else None
        )
        clusters = int(analysis.cluster_id.nunique()) if "cluster_id" in analysis else 0
        sites = int(analysis.site_id.nunique())
        checks.test(
            "L1-SUPPORT",
            cell.regime_cell_id,
            clusters > 0 and sites == clusters,
            field="facility_sites_and_clusters",
            distinct_sites=sites,
            distinct_clusters=clusters,
            **evidence,
        )
        if units is None:
            checks.not_bound["L1-SUPPORT", cell.regime_cell_id].append(
                {**evidence, "field": "randomization_units", "reason": "Cluster-size law or protocol cohort n absent"}
            )
        else:
            checks.test(
                "L1-SUPPORT",
                cell.regime_cell_id,
                clusters == sites == units,
                field="randomization_units",
                declared=units,
                actual=clusters,
                distinct_sites=sites,
                cohort_n=cohort_n,
                **evidence,
            )
        expected_sites = units if units is not None else clusters
    else:
        expected_sites = int(recruitment.site_count)
    numbers = analysis.site_id.astype(str).str.extract(r"^site_(\d+)$", expand=False)
    numeric = pd.to_numeric(numbers, errors="coerce")
    within = numeric.between(1, expected_sites)
    if facility:
        within &= analysis.site_id.astype(str).eq(numeric.map(lambda n: f"site_{int(n):03d}" if pd.notna(n) else ""))
    checks.test(
        "L1-SUPPORT",
        cell.regime_cell_id,
        bool(within.all()) and analysis.site_id.nunique() <= expected_sites,
        field="site_ids",
        declared=expected_sites,
        invalid=analysis.loc[~within, "site_id"].astype(str).unique().tolist(),
        **evidence,
    )
    rotation = numeric.map(
        lambda k: recruitment.country_codes[(int(k) - 1) % len(recruitment.country_codes)] if pd.notna(k) else None
    )
    checks.test(
        "L1-SUPPORT",
        cell.regime_cell_id,
        bool(analysis.country_code.eq(rotation).all()),
        field="site_country_rotation",
        **evidence,
    )
    n = analysis.participant_id.nunique()
    if facility:
        if cohort_n is None or (dataset.variant.context in {"C1", "C2"} and facility_census is None):
            checks.not_bound["L1-SUPPORT", cell.regime_cell_id].append(
                {**evidence, "field": "site_capacity", "reason": "Protocol cohort n or matching C3 census absent"}
            )
            return
        capacity = cohort_n
    else:
        capacity = max(2, math.ceil(2 * n / expected_sites))
    sizes = analysis.groupby("site_id").participant_id.nunique()
    census_equal = (
        sizes.sort_index().equals(facility_census.sort_index())
        if facility and dataset.variant.context in {"C1", "C2"} and facility_census is not None
        else True
    )
    checks.test(
        "L1-SUPPORT",
        cell.regime_cell_id,
        bool(sizes.le(capacity).all()) and census_equal,
        field="site_capacity",
        declared=capacity,
        n_participants=int(n),
        over_capacity={str(k): int(v) for k, v in sizes[sizes.gt(capacity)].items()},
        c3_census_equal=census_equal,
        **evidence,
    )


def observation_checks(checks: ExactChecks, dataset: Dataset, analysis: pd.DataFrame, cell: CellDeclaration) -> None:
    """Check deterministic observation restrictions, leaving latent states untested."""
    evidence = dict(task=dataset.variant.task_id, kind=dataset.kind, index=dataset.index)
    if cell.observation_kind == "validation_subsample":
        selected = analysis.validation_selected.eq(True)
        present = analysis.adjudicated_endpoint.notna()
        checks.test(
            "L1-OBS", cell.regime_cell_id, bool(selected.eq(present).all()), field="validation_subsample", **evidence
        )
    if cell.observation is None:
        return
    observation = cell.observation
    if "observed" not in analysis:
        checks.unavailable["L1-OBS", cell.regime_cell_id].append(
            {**evidence, "reason": "Contact observation indicators are not in this analysis view"}
        )
        return
    ordered = analysis.sort_values(["participant_id", "occurrence_time"])
    if observation.monotone_dropout:
        returned = ordered.observed.astype(int).groupby(ordered.participant_id).diff().gt(0)
        checks.test("L1-OBS", cell.regime_cell_id, not bool(returned.any()), field="monotone_dropout", **evidence)
    if observation.observe_first_contact:
        checks.test(
            "L1-OBS",
            cell.regime_cell_id,
            bool(ordered.groupby("participant_id").head(1).observed.all()),
            field="observe_first_contact",
            **evidence,
        )
    if observation.terminal_missingness_only:
        checks.test(
            "L1-OBS",
            cell.regime_cell_id,
            bool(ordered.loc[ordered.occurrence_time.lt(ordered.occurrence_time.max())].observed.all()),
            field="terminal_missingness_only",
            **evidence,
        )
    if observation.terminal_recovery_probability == 0:
        checks.test(
            "L1-OBS",
            cell.regime_cell_id,
            not bool(ordered.endpoint_recovered.any()),
            field="terminal_recovery_probability",
            **evidence,
        )


def intercurrent_checks(checks: ExactChecks, dataset: Dataset, repair: Repair, cell: CellDeclaration) -> None:
    """Verify event recording and post-event exposure in raw participant evidence."""
    if not cell.intercurrent_events:
        return
    events = repair.tables.get("data/raw/event_history.parquet")
    exposure = repair.tables.get("data/raw/exposure_records.parquet")
    if events is None or exposure is None:
        checks.unavailable["L1-ICE", cell.regime_cell_id].append(
            {
                "task": dataset.variant.task_id,
                "index": dataset.index,
                "reason": "This context supplies no treatment-receipt contact records",
            }
        )
        return
    for event in cell.intercurrent_events:
        records = events.loc[events.event_id.eq(event.event_id)]
        checks.test(
            "L1-ICE",
            cell.regime_cell_id,
            not records.empty,
            task=dataset.variant.task_id,
            index=dataset.index,
            field="recorded_event",
            event_id=event.event_id,
        )
        times = records.groupby("participant_id").occurrence_time.min()
        when = exposure.participant_id.map(times)
        after = exposure.loc[when.notna() & exposure.occurrence_time.ge(when)]
        if event.exposure_action in {"stop_active", "control_arm"}:
            ok = not after.received_arm.eq("active").any()
        elif event.exposure_action == "none":
            ok = True
        else:
            checks.unavailable["L1-ICE", cell.regime_cell_id].append(
                {
                    "event_id": event.event_id,
                    "reason": f"Unsupported declared exposure action: {event.exposure_action}",
                }
            )
            continue
        checks.test(
            "L1-ICE",
            cell.regime_cell_id,
            bool(ok),
            task=dataset.variant.task_id,
            index=dataset.index,
            field="exposure_action",
            action=event.exposure_action,
        )
