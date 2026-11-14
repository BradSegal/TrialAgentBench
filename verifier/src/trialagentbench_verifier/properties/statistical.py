"""Run statistical property falsification on authenticated release observations."""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from trialagentbench_verifier.artifacts.io import sha256_file
from trialagentbench_verifier.development_rules.decision_consequence import (
    DecisionEconomicsV1,
    DecisionReportV1,
    decision_consequence_v1,
)
from trialagentbench_verifier.properties.consequence import consequence, regret_test
from trialagentbench_verifier.properties.contracts import (
    CellDeclaration,
    CellDeclarations,
    CheckResult,
    Gates,
    REstimate,
    REstimateRequest,
    StatisticalPart,
    TierOnlyVerdict,
)
from trialagentbench_verifier.properties.inputs import (
    BoundInputs,
    PublicInputs,
    load_declarations,
    load_run_inputs,
    load_task_assignments,
    parse_dataset,
    public_bundle,
    write_r_payloads,
)
from trialagentbench_verifier.properties.observables import reconstruct
from trialagentbench_verifier.properties.r_runner import RRunner
from trialagentbench_verifier.properties.statistics import FitRecord, TestEvidence, adjust_families, recovery
from trialagentbench_verifier.reporting.layer_priors import LAYER_PRIOR_SHA256, load_layer_prior

FAMILIES = {
    "L1-MECH": "F-MECH",
    "L1-OBS": "F-OBS",
    "L1-ICE": "F-ICE",
    "L1-CENS": "F-CENS",
    "L1-CAL": "F-CAL",
    "L1-REC": "F-REC",
    "L1-CONSEQ-A3": "F-CONSEQ",
    "L1-REGRET-A2": "F-REGRET",
}
ASSUMPTION_HOLDS_TIERS = ("A1", "A2")
# The declared economic value follows the benchmark scoring contract.
# No producer module is imported.


def write_json(path: Path, payload: Any) -> None:
    """Serialize finite JSON with stable field order."""
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def checksums(output: Path) -> None:
    """Bind every retained run artifact without a self-referential digest."""
    files = sorted(p for p in output.rglob("*") if p.is_file() and p != output / "SHA256SUMS")
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(p)}  {p.relative_to(output).as_posix()}\n" for p in files)
    )


def public_results(results: tuple[CheckResult, ...], tier_only: set[str], alpha: float) -> list[dict[str, Any]]:
    """Project every tier-only hypothesis to its four permitted fields."""
    rows: list[dict[str, Any]] = []
    for result in results:
        if result.scope_id not in tier_only:
            rows.append(result.model_dump(mode="json"))
            continue
        family = FAMILIES[result.check_id]
        values = (
            [part.holm_p for part in result.parts]
            if result.check_id in {"L1-MECH", "L1-OBS", "L1-ICE", "L1-CENS", "L1-CAL"} and result.parts
            else [result.holm_p]
        )
        rows.extend(
            TierOnlyVerdict(
                cell=result.scope_id,
                family=family,
                verdict="unavailable" if p is None else "fail" if p < alpha else "pass",
                holm_p=p,
            ).model_dump(mode="json")
            for p in values
        )
    return rows


def bound_gates(bound: BoundInputs, directory: Path) -> Gates:
    """Read bound gates and reject disagreement with the duplicate constants export."""
    gates = Gates.model_validate_json(bound.files[(directory / "gates.json").resolve()])
    constants = json.loads(bound.files[(directory / "qualification_constants.json").resolve()])
    for field, value in gates.consequence.model_dump().items():
        if constants[field] != value:
            raise ValueError(f"inconsistent bound consequence constant: {field}")
    return gates


def _extra_request(root: Path, donor: str, route: str, index: int) -> Path:
    fit = root / f"extra-{index:02d}"
    fit.mkdir()
    tables = {}
    for source in (root / "csv").glob("*.csv"):
        os.link(source, fit / source.name)
        tables[source.stem] = source.name
    request = REstimateRequest(
        type="estimate", tables={"data": tables["analysis"], **tables}, cell_id=donor, route_id=route
    )
    (fit / "request.json").write_text(request.model_dump_json(indent=2) + "\n")
    return fit


def prepare_cell(
    inputs: PublicInputs,
    release: Path,
    variants: tuple[Any, ...],
    cell: Any,
    declarations: Any,
    bound: BoundInputs,
    directory: Path,
    output: Path,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    """Prepare exactly four C1 tasks and their corresponding raw observations."""
    cell_dir = output / cell.regime_cell_id
    cell_dir.mkdir()
    assignments = {a.task_id: a for a in load_task_assignments(bound)}
    by_ordinal = {a.task_key: a.task_id for a in assignments.values() if a.context == "C3"}
    chosen = sorted(
        (v for v in variants if v.context == "C1" and assignments[v.task_id].scenario_label == cell.regime_cell_id),
        key=lambda v: int(assignments[v.task_id].task_key.rsplit("#", 1)[1]),
    )
    if len(chosen) != cell.replicates_per_regime_context:
        raise ValueError("statistical runner requires all four declared tasks")
    series = next(s for s in declarations.series if s.evaluation_series_id == cell.evaluation_series_id)
    profile = declarations.process_profiles[cell.evaluation_series_id]
    variants_by_id = {v.task_id: v for v in variants}
    extra: list[tuple[str, str]] = []
    if cell.negative_control:
        extra.append((cell.regime_cell_id, cell.negative_control.route_id))
    same_series = [c for c in declarations.cells if c.evaluation_series_id == cell.evaluation_series_id]
    # A2 temptation is always the A3 credited route, including private A3 cells.
    from trialagentbench_verifier.properties.inputs import load_private_cells

    full = load_private_cells(directory, bound)
    a3 = next(
        (c for c in full if c.evaluation_series_id == cell.evaluation_series_id and c.assumption_tier == "A3"), None
    )
    if cell.assumption_tier == "A2" and a3:
        extra.extend((a3.regime_cell_id, r.route_id) for r in a3.reference_analyses)
    for adverse in same_series:
        if adverse.assumption_tier == "A3" and adverse.negative_control:
            extra.append((adverse.regime_cell_id, adverse.negative_control.route_id))
    extra = list(dict.fromkeys(extra))
    rows, worlds, keys = [], [], {}
    for effect, variant in enumerate(chosen):
        keys[variant.task_key] = variant.key_record["target"]
        raw_variant = variants_by_id[by_ordinal[assignments[variant.task_id].task_key]]
        if variant.key_record["base_world_id"] != raw_variant.key_record["base_world_id"]:
            raise ValueError("C1/C3 raw evidence is not the same base world")
        with inputs.replays(variant) as bank, inputs.replays(raw_variant) as raw_bank:
            if sorted(int(p.name) for p in bank.iterdir() if p.is_dir()) != list(
                range(1, inputs.manifest.replays_per_law + 1)
            ):
                raise ValueError("incomplete replay bank")
            for index in range(1, inputs.manifest.replays_per_law + 1):
                dataset = parse_dataset(bank / f"{index:02d}", variant, "replay", index)
                analysis = reconstruct(dataset)
                path = cell_dir / variant.task_id / f"{index:02d}"
                write_r_payloads(dataset, analysis, declarations, path)
                requests = sorted(path.glob("route-*/request.json"))
                request_dirs = [p.parent for p in requests]
                selected_routes = [r.route_id for r in cell.reference_analyses]
                for donor, route in extra:
                    if route not in selected_routes:
                        request_dirs.append(_extra_request(path, donor, route, len(request_dirs)))
                        selected_routes.append(route)
                raw_paths = {}
                if cell.intercurrent_events or cell.evaluation_series_id in {"TE-S10", "TE-S11", "TE-S12", "TE-S13"}:
                    raw = parse_dataset(raw_bank / f"{index:02d}", raw_variant, "replay", index)
                    # Keep the independently authenticated raw evidence, never an inferred
                    # missingness history. The deterministic verifier owns table reconstruction.
                    for name, table in raw.tables.items():
                        if name.startswith("data/raw/"):
                            target = path / "csv" / f"raw-{Path(name).stem}.csv"
                            table.to_csv(target, index=False, na_rep="NA", float_format="%.17g")
                            raw_paths[Path(name).stem] = str(target)
                for fit_dir in request_dirs:
                    request = REstimateRequest.model_validate_json((fit_dir / "request.json").read_bytes())
                    rows.append(
                        {
                            "directory": str(fit_dir),
                            "cell": cell.regime_cell_id,
                            "task_key": variant.task_key,
                            "task_id": variant.task_id,
                            "effect_index": effect,
                            "replication": index,
                            "route_id": request.route_id,
                            "truth": float(inputs.replay_records[variant.task_id, index]["truth"]),
                            "grading_sd": variant.key_record["target"]["reference_sampling_standard_deviation"],
                            "point_scored": variant.key_record["target"]["point_scored"],
                            "ratio": variant.key_record["target"]["binding"]["effect_measure"]
                            in {"rate_ratio", "risk_ratio", "hazard_ratio"},
                        }
                    )
                worlds.append(
                    {
                        "task_id": variant.task_id,
                        "replication": index,
                        "analysis": str(path / "csv/analysis.csv"),
                        "requests": [str(p) for p in request_dirs],
                        "raw": raw_paths,
                    }
                )
    realism_path = directory / "realism_observation_layer.json"
    if public_bundle(bound) is not None:
        realism_path = next(
            path for path in bound.files if bound.digests[str(path)] == bound.manifest.declarations.sha256
        )
    job = {
        "cell": cell.model_dump(mode="json"),
        "series": series.model_dump(mode="json"),
        "profile": profile.model_dump(mode="json"),
        "cluster": series.design_profile_id in {"TE-DP05", "TE-DP06"},
        "fit_workers": 4,
        "worlds": worlds,
        "result": str(cell_dir / "parameters.json"),
        "realism_path": str(realism_path),
        "realism_sha256": bound.digests[str(realism_path.resolve())],
    }
    path = cell_dir / "job.json"
    write_json(path, job)
    write_json(cell_dir / "fit-inputs.json", rows)
    write_json(cell_dir / "targets.json", keys)
    return path, rows, keys


def collect_fits(rows: list[dict[str, Any]]) -> tuple[FitRecord, ...]:
    """Validate every R response and convert native ratios to comparison units."""
    fits = []
    for row in rows:
        data = dict(row)
        result = REstimate.model_validate_json((Path(data.pop("directory")) / "response.json").read_bytes())
        ratio = data.pop("ratio")
        if ratio:
            # Replay-bank truth already uses comparison units (log for ratios).
            if result.status == "ok":
                assert result.estimate is not None and result.lower is not None and result.upper is not None
                if min(result.estimate, result.lower, result.upper) <= 0:
                    result = REstimate(
                        type="estimate",
                        status="failed",
                        estimate=None,
                        lower=None,
                        upper=None,
                        se=None,
                        message="native ratio interval is not positive",
                        versions=result.versions,
                    )
                else:
                    # The R estimator returns the native ratio SE. Transform that SE by
                    # the delta method along with the interval endpoints.
                    assert result.estimate is not None and result.lower is not None and result.upper is not None
                    result = result.model_copy(
                        update={
                            "estimate": math.log(result.estimate),
                            "lower": math.log(result.lower),
                            "upper": math.log(result.upper),
                            "se": None if result.se is None else result.se / result.estimate,
                        }
                    )
        fits.append(FitRecord(**data, fit=result))
    return tuple(fits)


def parameter_checks(cell: str, path: Path) -> tuple[list[CheckResult], list[TestEvidence], list[dict[str, Any]]]:
    """Retain each declared parameter as its own family hypothesis and evidence row."""
    results, evidence, unavailable = [], [], []
    for record in json.loads(path.read_bytes()):
        check = record["check_id"]
        response = record["result"]
        parts = []
        for component in response["components"]:
            name = "selection_share" if component["parameter"] == "selection" else component["parameter"]
            estimate = component.get("estimate", component.get("realized"))
            n = component.get("n", record["n"])
            p = component["p"]
            parts.append(
                StatisticalPart(
                    name=name,
                    parameter=name,
                    statistic=estimate,
                    p=p,
                    n=n,
                    arm_id=component.get("arm"),
                    stratum=component.get("stratum"),
                )
            )
            se = component.get("se")
            if (
                se is None
                and p is not None
                and n
                and name in {"selection_share", "sensitivity", "specificity", "capture"}
            ):
                se = math.sqrt(component["declared"] * (1 - component["declared"]) / n)
            if se is None and name == "loss_rate":
                se = math.sqrt(component["declared"] / component["exposure_years"])
            evidence.append(
                TestEvidence(
                    cell=cell,
                    check_id=check,
                    name=name,
                    parameter=name,
                    arm_id=component.get("arm"),
                    stratum=component.get("stratum"),
                    declaration_field=component.get("declaration_field", record["declaration_field"] + "." + name),
                    declared=component["declared"],
                    estimate=estimate,
                    se=se,
                    p=p,
                    n=n,
                    numerator=component.get("numerator"),
                    message=component.get("message", ""),
                )
            )
        if not parts:
            unavailable.append({"cell": cell, "check_id": check, "reason": response["message"]})
            if response["message"].startswith("No "):
                results.append(
                    CheckResult(
                        check_id=check,
                        scope="cell",
                        scope_id=cell,
                        family=None,
                        parts=(),
                        route_p={},
                        cell_p=None,
                        holm_p=None,
                        status="unavailable",
                    )
                )
                continue
            # A named unavailable hypothesis is retained in the family census.
            parts.append(
                StatisticalPart(name="unavailable", parameter="unavailable", statistic=None, p=None, n=record["n"])
            )
        if check == "L1-MECH" and parts[0].name == "unavailable":
            # A mechanism with no components has no testable stratum dimension.
            unavailable.append({"cell": cell, "check_id": check, "reason": "no estimable mechanism parameters"})
            continue
        results.append(
            CheckResult(
                check_id=check,
                scope="cell",
                scope_id=cell,
                family=FAMILIES[check],
                parts=tuple(parts),
                route_p={},
                cell_p=None,
                holm_p=None,
                status="unavailable" if any(p.p is None for p in parts) else "pass",
            )
        )
    return results, evidence, unavailable


def price_regret(
    cell: Any, fits: tuple[FitRecord, ...], keys: dict[str, Any], tempting_route: str, gates: Gates
) -> tuple[list[CheckResult], list[TestEvidence]]:
    """Price all replications, including failed answers and absent credited fits."""
    results, evidence = [], []
    grouped: dict[tuple[str, int], list[FitRecord]] = defaultdict(list)
    for fit in fits:
        grouped[fit.task_key, fit.replication].append(fit)
    credited_ids = {r.route_id for r in cell.reference_analyses}

    def report(record: FitRecord) -> DecisionReportV1 | None:
        if record.fit.status != "ok":
            return None
        assert record.fit.estimate is not None and record.fit.lower is not None and record.fit.upper is not None
        return DecisionReportV1(estimate=record.fit.estimate, lower=record.fit.lower, upper=record.fit.upper)

    for point in gates.economics_band:
        regrets, n_ok = [], 0
        for (task, _), rows in sorted(grouped.items()):
            target = keys[task]
            declaration = target["consequence"]["economics"]
            economics = DecisionEconomicsV1(**{k: declaration[k] for k in DecisionEconomicsV1.model_fields}).scaled(
                value=point.value, cost=point.cost
            )
            tempting = next(r for r in rows if r.route_id == tempting_route)
            answer = report(tempting)
            n_ok += answer is not None
            credited = tuple(value for r in rows if r.route_id in credited_ids and (value := report(r)) is not None)
            regrets.append(
                0.0
                if not credited
                else decision_consequence_v1(
                    answer,
                    credited,
                    truth=tempting.truth,
                    scale=tempting.grading_sd,
                    participants=target["consequence"]["participants"],
                    favorable_direction=target["binding"]["favorable_direction"],
                    economics=economics,
                    prior=load_layer_prior(target["binding"]["estimand_id"]),
                    consequential_regret_musd=gates.consequence.TRIALEVAL_CONSEQUENTIAL_REGRET_MUSD,
                ).regret_musd
            )
        result, item = regret_test(cell.regime_cell_id, tempting_route, regrets, n_ok, point, gates)
        results.append(result)
        evidence.append(item)
    return results, evidence


def evaluate(
    declarations: CellDeclarations,
    full: tuple[CellDeclaration, ...],
    all_fits: dict[str, tuple[FitRecord, ...]],
    targets: dict[str, dict[str, Any]],
    gates: Gates,
    runtime: Path,
) -> tuple[tuple[CheckResult, ...], list[TestEvidence], list[dict[str, Any]]]:
    """Evaluate declared families from typed fits without rerunning an estimator."""
    results: list[CheckResult] = []
    evidence: list[TestEvidence] = []
    unavailable: list[dict[str, Any]] = []
    for cell in declarations.cells:
        fits = all_fits[cell.regime_cell_id]
        credited = tuple(f for f in fits if f.route_id in {r.route_id for r in cell.reference_analyses})
        result, recovery_evidence = recovery(cell.regime_cell_id, credited, gates.recovery_criteria)
        results.append(result)
        evidence.extend(recovery_evidence)
        pars, parameter_evidence, missing = parameter_checks(
            cell.regime_cell_id, runtime / cell.regime_cell_id / "parameters.json"
        )
        results.extend(pars)
        evidence.extend(parameter_evidence)
        unavailable.extend(missing)
    for cell in declarations.cells:
        fits = all_fits[cell.regime_cell_id]
        credited = tuple(f for f in fits if f.route_id in {r.route_id for r in cell.reference_analyses})
        if cell.assumption_tier == "A2":
            a3 = next(
                c for c in full if c.evaluation_series_id == cell.evaluation_series_id and c.assumption_tier == "A3"
            )
            if len(a3.reference_analyses) != 1:
                raise ValueError("A2 tempting route is ambiguous")
            scored, regret_evidence = price_regret(
                cell, fits, targets[cell.regime_cell_id], a3.reference_analyses[0].route_id, gates
            )
            results.extend(scored)
            evidence.extend(regret_evidence)
        elif cell.assumption_tier == "A3":
            assert cell.negative_control is not None
            twin = next(
                c
                for tier in ASSUMPTION_HOLDS_TIERS
                for c in declarations.cells
                if c.evaluation_series_id == cell.evaluation_series_id and c.assumption_tier == tier
            )
            benign = all_fits[twin.regime_cell_id]
            excluded = tuple(f for f in fits if f.route_id == cell.negative_control.route_id)
            benign_excluded = tuple(f for f in benign if f.route_id == cell.negative_control.route_id)
            benign_credited = tuple(
                f
                for f in benign
                if f.route_id in {r.route_id for r in twin.reference_analyses}
                and f.route_id != cell.negative_control.route_id
            )
            result, items = consequence(
                cell.regime_cell_id, excluded, credited, benign_excluded, benign_credited, gates
            )
            results.append(result)
            evidence.extend(items)
    return adjust_families(results, gates.recovery_criteria.coverage_alpha), evidence, unavailable


def run(
    release: Path,
    manifest: Path,
    split: str,
    output: Path,
    *,
    anchor_sha256: str,
    sums: Path,
    r_package: Path,
    r_dependencies: Path,
) -> dict[str, Any]:
    """Run the full statistical split, with unchanged declared margins and offline R."""
    raw_manifest = json.loads(manifest.read_bytes())
    directory = (manifest.parent / raw_manifest["declarations"]["path"]).resolve().parent
    bound = load_run_inputs(manifest, directory)
    candidate_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    source_dirty = bool(
        subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True).stdout.strip()
    )
    if bound.manifest.split != split:
        raise ValueError("requested split differs from bound run-input manifest")
    gates = bound_gates(bound, directory)
    declarations = load_declarations(directory, inputs=bound)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    if split == "withheld":
        from trialagentbench_verifier.properties.withheld import run_withheld

        return run_withheld(bound, directory, output, gates)
    inputs = PublicInputs(release, anchor_sha256=anchor_sha256, sums=sums)
    if inputs.identity.release_id != bound.manifest.release_id:
        raise ValueError("release and manifest identities differ")
    runtime = output.parent / f"private-runtime-{output.name}"
    runtime.mkdir(mode=0o700)
    from trialagentbench_verifier.properties.inputs import load_private_cells

    full = load_private_cells(directory, bound)
    registry = declarations.model_dump(mode="json")
    registry["cells"] = [c.model_dump(mode="json") for c in full]
    write_json(runtime / "route-declarations.json", registry)
    runner = RRunner(r_package, runtime, directory, r_dependencies, runtime / "route-declarations.json")
    all_fits: dict[str, tuple[FitRecord, ...]] = {}
    targets: dict[str, dict[str, Any]] = {}
    with inputs.opened() as (root, variants):
        for public_cell in sorted(declarations.cells, key=lambda c: c.regime_cell_id):
            cell = next(c for c in full if c.regime_cell_id == public_cell.regime_cell_id)
            print(f"Preparing {cell.regime_cell_id}", flush=True)
            job, rows, keys = prepare_cell(inputs, root, variants, cell, declarations, bound, directory, runtime)
            print(f"Fitting {cell.regime_cell_id}: {len(rows)} requests", flush=True)
            runner.run_cell(job)
            fits = collect_fits(rows)
            expected_versions = {"R": runner.pins.R, **runner.pins.packages}
            if any(f.fit.versions.get(name) != value for f in fits for name, value in expected_versions.items()):
                raise ValueError("R response versions differ from the declared package pins")
            all_fits[cell.regime_cell_id], targets[cell.regime_cell_id] = fits, keys
            write_json(job.parent / "fits.json", [f.model_dump(mode="json") for f in fits])
    adjusted, evidence, unavailable = evaluate(declarations, full, all_fits, targets, gates, runtime)
    write_json(runtime / "internal-checks.json", [r.model_dump(mode="json") for r in adjusted])
    write_json(runtime / "internal-evidence.json", [e.model_dump(mode="json") for e in evidence])
    write_json(runtime / "internal-unavailable.json", unavailable)
    tier_only = {c.regime_cell_id for c in declarations.cells if c.declaration_scope == "tier_only"}
    public_evidence = [e for e in evidence if e.cell not in tier_only]
    write_json(output / "checks.json", public_results(adjusted, tier_only, gates.recovery_criteria.coverage_alpha))
    write_json(output / "evidence.json", [e.model_dump(mode="json") for e in public_evidence])
    write_json(output / "unavailable.json", [row for row in unavailable if row["cell"] not in tier_only])
    run_record = {
        "candidate_commit": candidate_commit,
        "source_dirty": source_dirty,
        "input_hashes": {**bound.digests, **inputs.digests},
        "split": split,
        "python": platform.python_version(),
        "R": runner.pins.R,
        "pins": runner.pins.model_dump(),
        "r_source": runner.source_identity(),
        "layer_prior_sha256": LAYER_PRIOR_SHA256,
        "S4_allowlist": ["GitHub", "Hugging Face", "Docker Hub", "Posit CRAN snapshot 2025-10-30", "PyPI"],
        "assumption_holds_tiers": ASSUMPTION_HOLDS_TIERS,
        "stopped_cells": sorted({r.scope_id for r in adjusted if r.status == "fail"}),
    }
    write_json(output / "run.json", run_record)
    checksums(output)
    return {
        "cells": len(declarations.cells),
        "replications_per_cell": inputs.manifest.replays_per_law * 4,
        "checks": len(adjusted),
        "stopped_cells": run_record["stopped_cells"],
    }
