"""Generate narrowly scoped public claims and a separate number-to-source table."""

from __future__ import annotations

import csv
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, TypeAdapter

from trialagentbench_verifier.artifacts.io import sha256_file
from trialagentbench_verifier.properties.contracts import (
    CheckResult,
    DeterministicResult,
    FileDigest,
    Gates,
    PublicSplitEvidence,
    Verdict,
    WithheldSplitEvidence,
)
from trialagentbench_verifier.verification.contracts import (
    ClaimNumber,
    ExternalRoutes,
    NB2CensusRow,
    NB2Receipt,
    ObservationReceipt,
    PositiveControl,
    PublicReport,
    RangeComparison,
    RealTrial,
    ReportInputs,
    ReportPolicy,
    ReproComparison,
    RescoreComparison,
    SequentialReceipt,
    SmokeReceipt,
)
from trialagentbench_verifier.verification.report_sources import (
    DeclarationCensus,
    DeterministicSummary,
    ExternalQuantity,
    ExternalSource,
    Numbers,
    PayloadIdentity,
    PipelineCell,
    Precision,
    RunMetadata,
    SequentialField,
    display_number,
    nb2_counts,
)

T = TypeVar("T", bound=BaseModel)
PARAMETERS = {"F-MECH", "F-OBS", "F-ICE", "F-CENS", "F-CAL"}
FAMILY_NAMES = {
    "F-MECH": "measurement mechanisms",
    "F-OBS": "observation processes",
    "F-ICE": "intercurrent events",
    "F-CENS": "censoring",
    "F-CAL": "calendar drift",
    "F-REC": "credited-route recovery",
    "F-CONSEQ": "A3 practical consequences",
    "F-REGRET": "A2 decision regret",
}
EXACT_CLAIMS = {
    "L1-ID": "Released public cell and series identities match their declarations.",
    "L1-SPLIT": "The released public split excludes the declared withheld cells.",
    "L1-COUNT": "Released public task and replay counts match their declarations.",
    "L1-CTX": "The C5 context consistency checks and C2–C5 reconstructions against C1 satisfy the checked declarations.",
    "L1-DESIGN": "Released public design evidence satisfies the checked obligations wherever the view states the procedure they depend on.",
    "L1-BIND": "Released public estimand and design bindings match the declaration fields.",
    "L1-SUPPORT": "Released public baseline values, countries, site identifiers, capacities and cluster counts lie in their declared support.",
    "L1-DESIGNTIME": "Released public contact and monitoring times follow their declared schedules.",
    "L1-DIAG": "In C1 and C3 views, default-route diagnostic instructions are present and every column those diagnostics use is released.",
    "L1-OBS": "Released public observation indicators follow the checked deterministic declarations",
    "L1-ICE": "Intercurrent events follow the checked deterministic declarations in C3–C5 views of the cells that declare them",
}


def bound_path(base: Path, binding: FileDigest) -> Path:
    """Check the exact file's digest before parsing its independently owned format."""
    path = (base / binding.path).resolve(strict=True)
    if sha256_file(path) != binding.sha256:
        raise ValueError("Verification input SHA256 mismatch")
    return path


def csv_records(path: Path, model: type[T]) -> list[T]:
    """Validate every consumed CSV row, including R's nullable diagnostics."""
    with path.open(newline="") as stream:
        return [
            model.model_validate({k: None if v in {"", "NA"} else v for k, v in row.items()})
            for row in csv.DictReader(stream)
        ]


def claim(
    identity: str, text: str, how: str, n: int, margin: str, *, withheld: tuple[int, int] | None = None
) -> Verdict:
    """Construct aggregate evidence without copying private values or digests."""
    return Verdict(
        criterion_id=identity,
        scope="public_aggregate",
        claim=text,
        how_verified=how,
        public_split=PublicSplitEvidence(n=n, n_verified=n, margin=margin, digests=()),
        withheld_split=(
            None if withheld is None else WithheldSplitEvidence(n=withheld[0], n_verified=withheld[1], digests=())
        ),
        statistic=None,
        interval=None,
        n=n,
    )


def rescore_claims(
    path: Path, pipeline: Path, relative_tolerance: float, numbers: Numbers | None = None
) -> list[Verdict]:
    """Derive the approved independent and conditional sentences from every re-scoring row."""
    rows = csv_records(path, RescoreComparison)
    counts = Counter(r.status for r in rows)
    if counts["mismatched"]:
        raise ValueError("Re-scoring contains disagreements; positive agreement claim cannot be emitted")
    numeric = [r for r in rows if r.status in {"matched_exact", "matched_1e-12"}]
    displays = [r for r in rows if r.status == "matched_display"]
    conditional = [r for r in rows if r.status == "conditional"]
    if (
        not numeric
        or not displays
        or not all(r.relative_1e12 for r in numeric)
        or not all(r.display_equal for r in displays)
    ):
        raise ValueError("Re-scoring agreement diagnostics do not support its classification")
    bit_exact = sum(r.binary64_equal is True for r in numeric)
    tolerance = f"{display_number(relative_tolerance)}"
    independent = (
        "Independently re-scored from the raw task submissions and trial records, every reproducible value agrees with the published one: "
        f"{len(numeric):,} numeric values within a relative {tolerance} ({bit_exact:,} bit-exact), "
        f"and all {len(displays):,} published display values at their printed precision. No value disagrees."
    )
    answer = [
        claim(
            "rescore-independent",
            independent,
            "Independent R re-scoring with the declared scoring definitions.",
            len(numeric) + len(displays),
            f"relative {tolerance}; printed precision for displays",
        )
    ]
    if numbers:
        for label, value, fields, op in (
            ("numeric", len(numeric), "status", "count matched_exact and matched_1e-12"),
            ("bit_exact", bit_exact, "binary64_equal", "count true among independent numeric rows"),
            ("displays", len(displays), "status,display_equal", "count matched_display"),
            ("comparisons", len(numeric) + len(displays), "status", "count independent numeric and display rows"),
        ):
            numbers.add("rescore-independent", label, value, "rescore", fields, op)
        numbers.add(
            "rescore-independent", "relative_tolerance", relative_tolerance, "policy", "rescore_relative_tolerance"
        )
    if conditional:
        if not all(r.relative_1e12 or (r.display_cell and r.display_equal) for r in conditional):
            raise ValueError("Conditional leaderboard comparison disagrees")
        cells = TypeAdapter(list[PipelineCell]).validate_json(pipeline.read_bytes())
        if not cells or len({r.task_id for r in cells}) != len(cells):
            raise ValueError("Pipeline task-score census is invalid")
        exact = sum(r.binary64_equal is True for r in conditional)
        printed = [r for r in conditional if not r.binary64_equal]
        if not all(r.display_cell and r.display_equal for r in printed):
            raise ValueError("Conditional remainder is not reproduced at printed precision")
        answer.append(
            claim(
                "rescore-conditional",
                f"Given the pipeline baseline's {len(cells)} stored task scores, all {len(conditional):,} leaderboard effects, intervals and ranks also reproduce: {exact} values bit-for-bit, and the other {len(printed)} published display values at their printed precision.",
                "Independent R aggregation given the declared stored pipeline task scores.",
                len(conditional),
                "binary64 equality; printed precision for displays",
            )
        )
        if numbers:
            numbers.add(
                "rescore-conditional", "tasks", len(cells), "pipeline_cells", "task_id", "count unique task rows"
            )
            for label, value, field, op in (
                ("comparisons", len(conditional), "status", "count conditional"),
                ("bit_exact", exact, "binary64_equal", "count true among conditional"),
                (
                    "displays",
                    len(printed),
                    "display_equal,display_cell,binary64_equal",
                    "count non-exact conditional displays",
                ),
            ):
                numbers.add("rescore-conditional", label, value, "rescore", field, op)
    return answer


def statistical_claims(
    statistics: list[CheckResult], gates: Gates, observation: list[ObservationReceipt], numbers: Numbers
) -> list[Verdict]:
    """Describe each family's actual tests, cells and construction margins."""
    rows = []
    alpha = gates.recovery_criteria.coverage_alpha
    recovery, consequence = gates.recovery_criteria, gates.consequence
    for family, name in FAMILY_NAMES.items():
        selected = [r for r in statistics if r.family == family]
        ps = [p.holm_p for r in selected for p in r.parts] if family in PARAMETERS else [r.holm_p for r in selected]
        tested = [p for p in ps if p is not None]
        if not tested or not all(p >= alpha for p in tested):
            continue
        cells = len(
            {r.scope_id for r in selected if any(p.holm_p is not None for p in r.parts) or r.holm_p is not None}
        )
        scope = f"{len(tested)} test{'s' if len(tested) != 1 else ''} in {cells} cell{'s' if cells != 1 else ''}"
        numbers.add(
            family,
            "tests",
            len(tested),
            "statistical",
            "family,parts.holm_p" if family in PARAMETERS else "family,holm_p",
            "count non-null adjusted tests in this family",
        )
        numbers.add(
            family,
            "cells",
            cells,
            "statistical",
            "family,scope_id,holm_p,parts.holm_p",
            "count distinct tested cells in this family",
        )
        numbers.add(family, "alpha", alpha, "gates", "recovery_criteria.coverage_alpha")
        margin = "the declared parameter values"
        if family == "F-REC":
            routes = sum(len(r.route_p) for r in selected)
            scope = f"{len(tested)} composites over {routes} credited routes"
            numbers.add(family, "routes", routes, "statistical", "family,route_p", "sum route counts for family")
            margin = (
                f"bias {display_number(recovery.bias_margin_sd)} SD pooled / {display_number(recovery.tertile_bias_margin_sd)} SD by effect; "
                f"coverage {display_number(recovery.coverage_lower_bound)}; fit support {display_number(recovery.support_lower_bound)}"
            )
            for key in ("bias_margin_sd", "tertile_bias_margin_sd", "coverage_lower_bound", "support_lower_bound"):
                numbers.add(family, key, getattr(recovery, key), "gates", "recovery_criteria." + key)
        elif family == "F-CONSEQ":
            margin = (
                f"shift {display_number(consequence.TRIALEVAL_CONSEQUENTIAL_SHIFT_SD)} SD, inner bound "
                f"{display_number(consequence.TRIALEVAL_CONSEQUENTIAL_SHIFT_BOUND_SD)} SD or width ratio "
                f"{display_number(consequence.TRIALEVAL_CONSEQUENTIAL_WIDTH_RATIO)}; no inversion; benign twin"
            )
            for key in (
                "TRIALEVAL_CONSEQUENTIAL_SHIFT_SD",
                "TRIALEVAL_CONSEQUENTIAL_SHIFT_BOUND_SD",
                "TRIALEVAL_CONSEQUENTIAL_WIDTH_RATIO",
            ):
                numbers.add(family, key, getattr(consequence, key), "gates", "consequence." + key)
        elif family == "F-REGRET":
            points = {
                (p.economics_point.value, p.economics_point.cost)
                for r in selected
                for p in r.parts
                if p.economics_point
            }
            scope = (
                f"{cells} cells × {len(points)} economics point{'s' if len(points) != 1 else ''} ({len(tested)} tests)"
            )
            margin = f"mean regret at least {display_number(gates.over_correction_margin)} × US${display_number(consequence.TRIALEVAL_CONSEQUENTIAL_REGRET_MUSD)}M × the declared value multiplier"
            numbers.add(
                family,
                "points",
                len(points),
                "statistical",
                "family,parts.economics_point",
                "count distinct value/cost pairs",
            )
            numbers.add(family, "multiplier", gates.over_correction_margin, "gates", "over_correction_margin")
            numbers.add(
                family,
                "regret_MUSD",
                consequence.TRIALEVAL_CONSEQUENTIAL_REGRET_MUSD,
                "gates",
                "consequence.TRIALEVAL_CONSEQUENTIAL_REGRET_MUSD",
            )
        how = "Independent R estimates and declared falsification rules. Holm adjusts across all declared tests within each family."
        if family == "F-OBS":
            if not observation or not all(c.availability == "available" for r in observation for c in r.components):
                raise ValueError("Observation-model receipts do not support the family claim")
            if sum(len(r.components) for r in observation) != len(tested):
                raise ValueError("Observation-model component census disagrees with the family tests")
            how = "Independent R fits of the declared longitudinal observation models from released contact histories. Holm adjusts across all declared tests within each family."
        rows.append(
            claim(
                family,
                f"For {name}, no declared property was rejected across {scope} on the public split.",
                how,
                len(tested),
                f"Holm α {display_number(alpha)}; {margin}",
            )
        )
    return rows


def reproduction_claim(repro: list[ReproComparison], numbers: Numbers) -> Verdict | None:
    """Count records, fields, routes and actual replay indices from comparisons."""
    if not repro or not all(r.passed and r.margin >= 0 for r in repro):
        return None
    records: dict[tuple[str, str | None, str, int | None, str], set[str]] = defaultdict(set)
    for r in repro:
        key = (r.cell, r.route_id, r.task_id, r.replication, r.source)
        if r.field in records[key]:
            raise ValueError("Duplicate stored-reference comparison")
        records[key].add(r.field)
    source_counts = Counter(key[-1] for key in records)
    replay = [r for r in repro if r.source == "replay"]
    indices = sorted({r.replication for r in replay if r.replication is not None})
    replay_text = ""
    if indices:
        span = (
            f"{indices[0]} to {indices[-1]}"
            if len(indices) > 1 and indices == list(range(indices[0], indices[-1] + 1))
            else ", ".join(map(str, indices))
        )
        replay_text = f", replays {span}"
        numbers.add(
            "reference-reproduction",
            "replay_indices",
            span,
            "repro",
            "source,replication",
            "sorted distinct indices where source=replay",
        )
    scope_parts = []
    for source, label in (
        ("replay", "replay reference"),
        ("key", "credited key value"),
        ("key_plan", "plan readout"),
    ):
        if source_counts[source]:
            n = source_counts[source]
            scope_parts.append(f"{n:,} {label}{'s' if n != 1 else ''}" + (replay_text if source == "replay" else ""))
            numbers.add(
                "reference-reproduction",
                source,
                n,
                "repro",
                "source,cell,route_id,task_id,replication",
                "count distinct record keys for source=" + source,
            )
    fields = {r.field for r in repro}
    rectangular = all(value == fields for value in records.values())
    scope = (
        f"{len(records):,} record{'s' if len(records) != 1 else ''} × {len(fields)} field{'s' if len(fields) != 1 else ''}"
        if rectangular
        else f"{len(records):,} record{'s' if len(records) != 1 else ''} across {len(fields)} field{'s' if len(fields) != 1 else ''}"
    )
    how = (
        f"{scope}: "
        + "; ".join(scope_parts)
        + ". Non-null references only; declared procedures and tolerances, with release rounding for sequential stored values."
    )
    if replay:
        tasks = {r.cell: {x.task_id for x in replay if x.cell == r.cell} for r in replay}
        routes = {r.cell: {x.route_id for x in replay if x.cell == r.cell} for r in replay}
        task_counts, route_counts = {len(v) for v in tasks.values()}, {len(v) for v in routes.values()}
        if len(task_counts) == len(route_counts) == 1:
            n_tasks, n_routes = next(iter(task_counts)), next(iter(route_counts))
            how += f" Replay scope: {n_tasks} fixed task and {n_routes} compared route per cell."
            numbers.add(
                "reference-reproduction",
                "tasks_per_cell",
                n_tasks,
                "repro",
                "source,cell,task_id",
                "distinct tasks per replay cell; uniform across cells",
            )
            numbers.add(
                "reference-reproduction",
                "routes_per_cell",
                n_routes,
                "repro",
                "source,cell,route_id",
                "distinct routes per replay cell; uniform across cells",
            )
    key_tasks = len({r.task_id for r in repro if r.source == "key"})
    if key_tasks:
        how += f" Credited keys cover {key_tasks} tasks."
        numbers.add(
            "reference-reproduction", "key_tasks", key_tasks, "repro", "source,task_id", "distinct key task IDs"
        )
    for label, value, field, op in (
        ("comparisons", len(repro), "field", "count all comparison rows"),
        ("records", len(records), "source,cell,route_id,task_id,replication", "count distinct record keys"),
        ("fields", len(fields), "field", "count distinct compared fields"),
    ):
        numbers.add("reference-reproduction", label, value, "repro", field, op)
    return claim(
        "reference-reproduction",
        "The checked stored public replay references, credited key values and plan readouts reproduce in independent R.",
        how,
        len(repro),
        "all comparison margins nonnegative under the declared route-specific tolerances",
    )


def build_report(
    inputs: ReportInputs, base: Path, sources: dict[str, dict[str, ClaimNumber]] | None = None
) -> PublicReport:
    """Aggregate typed records and bind every displayed number to its derivation."""
    if inputs.context is None or inputs.context.run_metadata is None or inputs.context.deterministic_summary is None:
        raise ValueError("Report requires bound run metadata, summary and scope declarations")
    bindings = {name: value for name, value in inputs if isinstance(value, FileDigest)}
    bindings.update({name: value for name, value in inputs.context if isinstance(value, FileDigest)})
    paths = {name: bound_path(base, value) for name, value in bindings.items()}
    numbers = Numbers(bindings)
    policy = ReportPolicy.model_validate_json(paths["policy"].read_bytes())
    summary = DeterministicSummary.model_validate_json(paths["deterministic_summary"].read_bytes())
    declarations = DeclarationCensus.model_validate_json(paths["cell_declarations"].read_bytes())
    precision = Precision.model_validate_json(paths["profiles"].read_bytes()).release_numeric_decimal_places
    metadata = RunMetadata.model_validate_json(paths["run_metadata"].read_bytes())
    payload = PayloadIdentity.model_validate_json(paths["payload"].read_bytes())
    gates = Gates.model_validate_json(paths["gates"].read_bytes())
    provenance = (
        f"Release `{summary.release.release_id}` (HF payload revision `{payload.revision}`); "
        f"verifier commit `{metadata.candidate_commit}`; configuration SHA256 `{metadata.input_manifest_sha256}`. "
        f"Independent R {metadata.R} ran without networking in image `{metadata.image_digest}`. "
        "Each row states the verified scope, method, counts and margin."
    )
    for label, provenance_value, source, field in (
        ("release", summary.release.release_id, "deterministic_summary", "release.release_id"),
        ("payload_revision", payload.revision, "payload", "revision"),
        ("verifier_commit", metadata.candidate_commit, "run_metadata", "candidate_commit"),
        ("configuration", metadata.input_manifest_sha256, "run_metadata", "input_manifest_sha256"),
        ("R", metadata.R, "run_metadata", "R"),
        ("image", metadata.image_digest, "run_metadata", "image_digest"),
    ):
        numbers.add("provenance", label, provenance_value, source, field)
    exact = [
        DeterministicResult.model_validate_json(line) for line in paths["deterministic"].read_bytes().splitlines()
    ]
    statistics = TypeAdapter(list[CheckResult]).validate_json(paths["statistical"].read_bytes())
    rows: list[Verdict] = []
    for check_id, text in EXACT_CLAIMS.items():
        selected = [r for r in exact if r.check_id == check_id]
        if not selected or any(r.n_failed for r in selected):
            continue
        n = sum(r.n_checked for r in selected)
        numbers.add(
            check_id,
            "checks",
            n,
            "deterministic",
            "check_id,n_checked,n_failed",
            "sum n_checked for passing check_id=" + check_id,
        )
        how = "Exact checks on released tables, parsed documents and declaration fields."
        margin = "exact equality"
        if check_id == "L1-ID":
            cell_ids = {r.regime_cell_id for r in declarations.cells}
            cell_n = sum(r.n_checked for r in selected if r.cell_ref in cell_ids)
            series_n = n - cell_n
            how = f"{cell_n} cell-reference checks plus {series_n} series-reference checks."
            for label, count, op in (
                ("cells", cell_n, "sum checks whose cell_ref is in declarations.cells"),
                ("series", series_n, "sum remaining identity checks"),
            ):
                numbers.add(
                    check_id,
                    label,
                    count,
                    ("deterministic", "cell_declarations"),
                    ("check_id,cell_ref,n_checked", "cells.regime_cell_id"),
                    op,
                )
        elif check_id == "L1-SPLIT":
            how = f"{summary.n_variants:,} views ({n:,} checks)."
            numbers.add(check_id, "views", summary.n_variants, "deterministic_summary", "n_variants")
        elif check_id == "L1-CTX":
            contexts = len(declarations.context_definitions)
            if not contexts or summary.n_datasets % contexts:
                raise ValueError("Context dataset census is inconsistent")
            reconstructed = summary.n_datasets // contexts * (contexts - 1)
            how = f"{n:,} checks in total, of which {reconstructed:,} are C2–C5 reconstructions against C1."
            margin = f"relative tolerance {display_number(policy.context_relative_tolerance)} × max({display_number(policy.context_scale_floor)}, |x|) for numerical reconstructions; other checks exact"
            numbers.add(
                check_id,
                "reconstructions",
                reconstructed,
                ("deterministic_summary", "cell_declarations"),
                ("n_datasets", "context_definitions"),
                "n_datasets / context count × noncanonical context count",
            )
            numbers.add(
                check_id,
                "relative_tolerance",
                policy.context_relative_tolerance,
                "policy",
                "context_relative_tolerance",
            )
            numbers.add(check_id, "relative_scale_floor", policy.context_scale_floor, "policy", "context_scale_floor")
        elif check_id == "L1-BIND":
            fields = policy.binding_fields
            if len(set(fields)) != len(fields) or n != summary.n_variants * len(fields):
                raise ValueError("Binding checks disagree with the view and field census")
            how = (
                f"{summary.n_variants:,} views × {len(fields)} fields: "
                + ", ".join(f"`{field}`" for field in fields)
                + "."
            )
            numbers.add(check_id, "views", summary.n_variants, "deterministic_summary", "n_variants")
            numbers.add(check_id, "fields", len(fields), "policy", "binding_fields", "count distinct field names")
        elif check_id == "L1-DESIGNTIME":
            margin = f"times exact; familywise α within {display_number(policy.schedule_alpha_tolerance)} of {display_number(policy.schedule_alpha)}; boundary constants within the {precision}-decimal release rounding"
            numbers.add(
                check_id, "alpha_tolerance", policy.schedule_alpha_tolerance, "policy", "schedule_alpha_tolerance"
            )
            numbers.add(check_id, "alpha", policy.schedule_alpha, "policy", "schedule_alpha")
            numbers.add(check_id, "decimal_places", precision, "profiles", "release_numeric_decimal_places")
        elif check_id in {"L1-OBS", "L1-ICE"}:
            cells = len({r.cell_ref for r in selected if r.n_checked})
            text += f" (in {cells} cells)."
            numbers.add(
                check_id,
                "cells",
                cells,
                "deterministic",
                "check_id,cell_ref,n_checked",
                "distinct cell_ref with positive n_checked",
            )
        rows.append(claim(check_id, text, how, n, margin))
    observation = [
        ObservationReceipt.model_validate_json(bound_path(base, b).read_bytes()) for b in inputs.observation
    ]
    rows.extend(statistical_claims(statistics, gates, observation, numbers))
    reproduced = reproduction_claim(csv_records(paths["repro"], ReproComparison), numbers)
    if reproduced:
        rows.append(reproduced)
    controls = csv_records(paths["positive_control"], PositiveControl)
    alpha = gates.recovery_criteria.coverage_alpha
    if controls and all(r.cell_state == "FALSIFIED" and r.cell_p < alpha for r in controls):
        multipliers = {r.and_multiplier for r in controls}
        if len(multipliers) != 1:
            raise ValueError("Positive controls have inconsistent multiplicity")
        multiplicity = next(iter(multipliers))
        rows.append(
            claim(
                "consequence-control",
                "The A3 consequence rule rejects when the property is absent in the positive control.",
                f"Consequence and non-inversion tests on the A1/A2 twins with ×{multiplicity} multiplicity; the benign conjunct is not applicable.",
                len(controls),
                f"α {display_number(alpha)} with ×{multiplicity} multiplicity",
            )
        )
        numbers.add(
            "consequence-control",
            "cells",
            len(controls),
            "positive_control",
            "cell_state,cell_p",
            "count falsified control rows",
        )
        numbers.add(
            "consequence-control",
            "multiplicity",
            multiplicity,
            "positive_control",
            "and_multiplier",
            "unique multiplier",
        )
        numbers.add("consequence-control", "alpha", alpha, "gates", "recovery_criteria.coverage_alpha")
    rows.extend(rescore_claims(paths["rescore"], paths["pipeline_cells"], policy.rescore_relative_tolerance, numbers))
    external = ExternalRoutes.model_validate_json(paths["external_routes"].read_bytes())
    quantities = csv_records(paths["external_quantities"], ExternalQuantity)
    external_source = ExternalSource.model_validate_json(paths["external_source"].read_bytes())
    if external.accepted and external.failed_count == 0 and external.passed_count == external.scheduled_count:
        if (
            not quantities
            or not all(r.quantity_pass for r in quantities)
            or len({r.route for r in quantities}) != external.scheduled_count
        ):
            raise ValueError("External numerical census disagrees with route summary")
        rows.append(
            claim(
                "external-reference",
                "The external-reference routes agree with their declared comparators.",
                f"Run at the {policy.external_release} release source (`{external_source.commit}`): {external.passed_count}/{external.scheduled_count} routes and {len(quantities)}/{len(quantities)} numerical comparisons.",
                external.passed_count,
                "each route's declared tolerance",
            )
        )
        for label, external_value, source, field, op in (
            ("passed_routes", external.passed_count, "external_routes", "passed_count", "identity"),
            ("scheduled_routes", external.scheduled_count, "external_routes", "scheduled_count", "identity"),
            (
                "comparisons",
                len(quantities),
                "external_quantities",
                "quantity,quantity_pass",
                "count passing numerical comparisons",
            ),
            ("release", policy.external_release, "policy", "external_release", "identity"),
            ("source", external_source.commit, "external_source", "commit", "identity"),
        ):
            numbers.add("external-reference", label, external_value, source, field, op)
    real = RealTrial.model_validate_json(paths["real_trial"].read_bytes())
    if real.status == "passed" and real.r_fit_failures == 0 and real.passed_comparisons == real.numeric_comparisons:
        rows.append(
            claim(
                "real-trial-refit",
                f"Independent R refits reproduce the reconstructed results for {real.trials} real trials and {real.complete_participants:,} complete-case participants.",
                "OLS, HC1 standard errors, t intervals and the arm-reversed control in the pinned R image.",
                real.passed_comparisons,
                f"atol {display_number(real.atol)} + rtol {display_number(real.rtol)}; maximum used tolerance fraction {display_number(real.maximum_tolerance_fraction)}",
            )
        )
        for key in (
            "trials",
            "complete_participants",
            "passed_comparisons",
            "atol",
            "rtol",
            "maximum_tolerance_fraction",
        ):
            numbers.add("real-trial-refit", key, getattr(real, key), "real_trial", key)
    ranges = TypeAdapter(list[RangeComparison]).validate_json(paths["realism_ranges"].read_bytes())
    compared = [
        r
        for r in ranges
        if r.inside_outside in {"inside", "outside"} and r.classification in {"declared", "implied_population"}
    ]
    if compared and all(r.inside_outside == "inside" for r in compared):
        public = [r for r in compared if not r.private]
        private = [r for r in compared if r.private]
        cells = len({r.cell for r in public})
        rows.append(
            claim(
                "realism-ranges",
                f"In the A3 cells, every declared measurement-mechanism parameter and identified implied population quantity that has a cited range falls inside it ({len(public)}/{len(public)} in {cells} cells).",
                "Compare each declared or identified implied quantity with its cited range.",
                len(public),
                "the supplied parameter-specific cited range",
                withheld=(len(private), len(private)),
            )
        )
        numbers.add(
            "realism-ranges",
            "comparisons",
            len(public),
            "realism_ranges",
            "private,classification,inside_outside",
            "count public compared quantities",
        )
        numbers.add(
            "realism-ranges",
            "cells",
            cells,
            "realism_ranges",
            "cell,private,classification,inside_outside",
            "count distinct public compared cells",
        )
        numbers.add(
            "realism-ranges",
            "withheld",
            len(private),
            "realism_ranges",
            "private,classification,inside_outside",
            "count private compared quantities",
        )
    sequential = SequentialReceipt.model_validate_json(paths["sequential"].read_bytes())
    public_seq = [r for r in sequential.results if r.split == "public"]
    private_seq = [r for r in sequential.results if r.split == "withheld"]
    seq_fields = [r for r in csv_records(paths["sequential_fields"], SequentialField) if r.split == "public"]
    if public_seq and all(r.passed for r in public_seq):
        if not seq_fields or not all(r.complete_check_passed for r in seq_fields):
            raise ValueError("Sequential field evidence does not support the summary")
        rows.append(
            claim(
                "sequential-panel",
                f"At the declared precision, R reproduces the stored scored reference for the {policy.sequential_dataset} and agrees with Python within the declared sequential tolerance.",
                "Independent R sequential fit against stored readouts and the declared per-field R–Python allowances.",
                len(public_seq),
                f"{precision}-decimal stored precision; declared per-field sequential tolerances",
                withheld=(len(private_seq), sum(r.passed for r in private_seq)),
            )
        )
        numbers.add(
            "sequential-panel",
            "public",
            len(public_seq),
            "sequential",
            "results.split,results.passed",
            "count passing public results",
        )
        numbers.add(
            "sequential-panel", "withheld", len(private_seq), "sequential", "results.split", "count withheld results"
        )
        numbers.add(
            "sequential-panel",
            "withheld_passed",
            sum(r.passed for r in private_seq),
            "sequential",
            "results.split,results.passed",
            "count passing withheld results",
        )
        numbers.add("sequential-panel", "decimal_places", precision, "profiles", "release_numeric_decimal_places")
        numbers.add("sequential-panel", "dataset", policy.sequential_dataset, "policy", "sequential_dataset")
    nb2 = NB2Receipt.model_validate_json(paths["nb2"].read_bytes())
    fit_n, readout_n = nb2_counts(csv_records(paths["nb2_census"], NB2CensusRow), nb2)
    if fit_n and nb2.failed == 0 and nb2.passed == fit_n:
        rows.append(
            claim(
                "nb2-panel",
                f"Independent R reproduces the estimate and SE on every stored TrialDev NB2 fit ({nb2.passed}/{fit_n}, in {readout_n} readouts).",
                "Reconcile the complete fit census with the comparison summary; compare only the estimate and SE.",
                fit_n,
                "the per-field NB2 comparison tolerances recorded in the census",
            )
        )
        numbers.add(
            "nb2-panel",
            "fits",
            fit_n,
            ("nb2", "nb2_census"),
            ("compared,passed,failed,counts", "readout_root,readout_id,arm_id,field,status"),
            "count distinct fits with estimate and SE comparisons; reconcile all summary counts",
        )
        numbers.add(
            "nb2-panel",
            "readouts",
            readout_n,
            "nb2_census",
            "readout_root,readout_id,field,status",
            "count distinct readouts with compared fits",
        )
    if inputs.smoke:
        smoke = SmokeReceipt.model_validate_json(paths["smoke"].read_bytes())
        if smoke.status == "passed" and smoke.executed_comparisons:
            rows.append(
                claim(
                    "generated-smoke",
                    "The generated-data R comparison smoke reproduces its declared numerical targets.",
                    "All declared smoke configurations and convention fixtures in the pinned image.",
                    smoke.executed_comparisons,
                    "the declared per-configuration tolerances",
                )
            )
            numbers.add("generated-smoke", "comparisons", smoke.executed_comparisons, "smoke", "executed_comparisons")
    if sources is not None:
        sources.update(numbers.rows)
    return PublicReport(claims=tuple(rows), provenance=provenance)


def render(report: PublicReport) -> str:
    """Render sectioned public tables and a withheld paragraph containing counts."""
    lines = ["# Verification", "", report.provenance, ""]
    sections = (
        ("Released data match their declarations", set(EXACT_CLAIMS)),
        ("Statistical properties", {*FAMILY_NAMES, "consequence-control"}),
        (
            "Independent reproduction in R",
            {
                "reference-reproduction",
                "external-reference",
                "real-trial-refit",
                "sequential-panel",
                "nb2-panel",
                "generated-smoke",
            },
        ),
        ("Independent re-scoring", {"rescore-independent", "rescore-conditional"}),
        ("Realism ranges", {"realism-ranges"}),
    )
    for title, identities in sections:
        lines.extend([f"## {title}", "", "| Claim | How verified | Result |", "| --- | --- | --- |"])
        for row in report.claims:
            if row.criterion_id not in identities:
                continue
            public = row.public_split
            result = "" if public is None else f"{public.n_verified:,}/{public.n:,}; {public.margin}"
            lines.append(
                "| "
                + " | ".join(v.replace("|", "\\|").replace("\n", " ") for v in (row.claim, row.how_verified, result))
                + " |"
            )
        lines.append("")
    withheld = []
    for row in report.claims:
        if row.withheld_split is not None:
            w = row.withheld_split
            label = {"realism-ranges": "realism-range comparisons", "sequential-panel": "sequential comparisons"}[
                row.criterion_id
            ]
            withheld.append(f"{w.n_verified:,}/{w.n:,} {label}")
    lines.extend(
        [
            "## Withheld split",
            "",
            "Withheld checks: " + "; ".join(withheld) + "." if withheld else "No withheld comparisons are reported.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(manifest: Path, output: Path, *, claim_table: Path | None = None) -> PublicReport:
    """Write the generated document, public panel and machine-readable derivations."""
    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    if claim_table is None:
        claim_table = output.with_name(output.name + "-claim-sources.json")
    if output.resolve().is_relative_to(repo) or claim_table.resolve().is_relative_to(repo):
        raise ValueError("Report output must be outside the checkout")
    if claim_table.resolve().is_relative_to(output.resolve()) or claim_table.exists():
        raise ValueError("Use a new claim table outside the public panel")
    inputs = ReportInputs.model_validate_json(manifest.read_bytes())
    sources: dict[str, dict[str, ClaimNumber]] = {}
    report = build_report(inputs, manifest.resolve().parent, sources)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (output / "report.json").write_text(report.model_dump_json(indent=2) + "\n")
    (output / "report.md").write_text(render(report))
    claim_table.write_text(
        json.dumps(
            {
                row: {label: item.model_dump(mode="json") for label, item in values.items()}
                for row, values in sources.items()
            },
            indent=2,
        )
        + "\n"
    )
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(output / name)}  {name}\n" for name in ("report.json", "report.md"))
    )
    return report
