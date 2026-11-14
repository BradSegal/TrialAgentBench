"""Private checks on retained scored inputs, without inventing withheld replays."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any

from trialagentbench_verifier.artifacts.anchor import ReleaseInputResolver
from trialagentbench_verifier.artifacts.archive import opened_release
from trialagentbench_verifier.properties.contracts import Gates, TaskVariant
from trialagentbench_verifier.properties.design import design_checks, diagnostic_checks, timing_checks
from trialagentbench_verifier.properties.deterministic import (
    ExactChecks,
    binding_checks,
    context_checks,
    intercurrent_checks,
    observation_checks,
    raw_transfer_checks,
    support_checks,
)
from trialagentbench_verifier.properties.inputs import (
    BoundInputs,
    load_declarations,
    load_instantiations,
    load_private_cells,
    load_sequential_plan,
    parse_dataset,
)
from trialagentbench_verifier.properties.observables import reconstruct
from trialagentbench_verifier.properties.repair import resolve_transfer


def run_withheld(bound: BoundInputs, directory: Path, output: Path, gates: Gates) -> dict[str, Any]:
    """Check retained scored datasets and report every replay family unavailable."""
    from trialagentbench_verifier.properties.statistical import FAMILIES, checksums, write_json

    private = bound.manifest.withheld_release
    if private is None:
        raise ValueError("withheld run requires a bound private release")
    root = Path(private.root)
    resolver = ReleaseInputResolver(root, anchor_sha256=private.anchor_sha256 or private.sums_sha256)
    catalog = root / "trialeval-withheld-evaluator.assets.json"
    digest = resolver.require(catalog)
    declarations = load_declarations(directory, inputs=bound)
    full = load_private_cells(directory, bound)
    cells = {c.regime_cell_id: c for c in full}
    payload = declarations.model_dump(mode="json")
    payload.update(split="withheld", cells=[c.model_dump(mode="json") for c in full])
    declarations = declarations.model_validate(payload)
    series = {s.evaluation_series_id: s for s in declarations.series}
    law, estimands = load_instantiations(directory, bound)
    plan = load_sequential_plan(directory, bound)
    checks = ExactChecks()
    tasks = []
    with opened_release(catalog, expected_sha256=digest, verify_files=True) as opened:
        release_manifest = json.loads((opened / "release_manifest.json").read_bytes())
        if release_manifest["release_id"] != bound.manifest.release_id:
            raise ValueError("withheld release and run-input manifest identities differ")
        construct = json.loads((opened / "evaluation/construct.json").read_bytes())
        trials = {t["trial_id"]: t for t in construct["trials"]}
        keys = [json.loads(line) for line in (opened / "grading_key/records.jsonl").read_text().splitlines()]
        for key in keys:
            trial = trials[key["trial_id"]]
            cell = cells[trial["difficulty"]["regime_cell_id"]]
            declaration = series[cell.evaluation_series_id]
            task_dir = opened / "participant/tasks" / key["task_id"]
            task = json.loads((task_dir / "task.json").read_bytes())
            variant = TaskVariant(
                task_id=key["task_id"],
                cell_ref=key["cell_ref"],
                context=key["context"],
                task_key=f"{key['cell_ref']}#{key['target']['value']}",
                key_record=key,
                materials={m["artifact_id"]: m["relative_path"] for m in task["available_materials"]},
            )
            dataset = parse_dataset(task_dir, variant, "scored", 0)
            raw_transfer_checks(checks, dataset, cell)
            repair = resolve_transfer(dataset)
            context_checks(checks, dataset, cell, declarations, repair)
            try:
                analysis = reconstruct(dataset, repair=repair)
            except ValueError as error:
                checks.unavailable["L1-CTX", cell.regime_cell_id].append(
                    {"task": key["task_id"], "reason": str(error), "scope": "retained scored dataset"}
                )
                continue
            binding_checks(checks, dataset, cell, declaration, declarations)
            observation_checks(checks, dataset, analysis, cell)
            intercurrent_checks(checks, dataset, repair, cell)
            support_checks(checks, dataset, analysis, cell, declaration, cluster_law=law)
            design_checks(
                checks,
                dataset,
                analysis,
                cell,
                declaration,
                declarations,
                estimand_instantiation=estimands[declaration.evaluation_series_id],
            )
            diagnostic_checks(checks, dataset, analysis, cell, declarations)
            timing_checks(checks, dataset, analysis, repair, cell, declaration, declarations, sequential_plan=plan)
            tasks.append(
                {
                    "task_id": key["task_id"],
                    "cell": cell.regime_cell_id,
                    "rows": len(analysis),
                    "context": key["context"],
                    "scope": "retained scored dataset",
                }
            )
    deterministic, results = checks.results()
    write_json(output / "single-dataset-deterministic.json", [r.model_dump(mode="json") for r in deterministic])
    write_json(output / "single-dataset-checks.json", [r.model_dump(mode="json") for r in results])
    write_json(output / "single-dataset-tasks.json", tasks)
    families = [
        {"family": family, "status": "unavailable", "reason": "no retained withheld replay bank", "replications": 0}
        for family in FAMILIES.values()
    ]
    write_json(output / "replay-family-summary.json", families)
    summary = {
        "scope": "retained withheld scored datasets only",
        "scored_datasets": len(tasks),
        "cells": len({t["cell"] for t in tasks}),
        "deterministic_assertions": sum(r.n_checked for r in deterministic),
        "deterministic_findings": sum(r.n_failed for r in deterministic),
        "replay_families": families,
    }
    write_json(
        output / "run.json",
        {
            "release_id": bound.manifest.release_id,
            "split": "withheld",
            "candidate_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
            ).stdout.strip(),
            "python": platform.python_version(),
            "R_execution": "not run; retained scored deterministic checks only",
            "S4_allowlist": ["GitHub", "Hugging Face", "Docker Hub", "Posit CRAN snapshot 2025-10-30", "PyPI"],
            "input_hashes": {**bound.digests, str(catalog): digest},
            "summary": summary,
        },
    )
    checksums(output)
    return summary
