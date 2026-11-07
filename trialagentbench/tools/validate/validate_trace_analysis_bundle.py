"""Validate a publication-neutral observable trace bundle."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

from trialagentbench.contracts.trace.bundle import (
    OBSERVABLE_TRACE_TABLE_MODELS,
    ObservableTraceBundleManifestV2,
)
from trialagentbench.io import sha256_file
from trialagentbench.io.csv_contracts import iter_contract_csv

MANIFEST_NAME = "trace_bundle_manifest.json"


def validate_trace_analysis_bundle(root: Path) -> ObservableTraceBundleManifestV2:
    """Validate exact members, schemas, counts, and hashes of a trace bundle.

    Parameters
    ----------
    root
        Dedicated observable trace bundle directory.

    Returns
    -------
    ObservableTraceBundleManifestV2
        The validated bundle manifest.

    Raises
    ------
    FileNotFoundError
        If the bundle or a declared artifact is absent.
    ValueError
        If members, schemas, row counts, identities, or checksums disagree.
    """

    bundle_root = Path(root)
    if not bundle_root.is_dir():
        raise FileNotFoundError(f"observable trace bundle does not exist: {bundle_root}")
    manifest_path = bundle_root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"observable trace manifest does not exist: {manifest_path}")
    manifest = ObservableTraceBundleManifestV2.model_validate_json(manifest_path.read_text(encoding="utf-8"))

    expected_members = {MANIFEST_NAME, *OBSERVABLE_TRACE_TABLE_MODELS}
    actual_members = {path.name for path in bundle_root.iterdir() if path.is_file()}
    if actual_members != expected_members:
        missing = sorted(expected_members - actual_members)
        unknown = sorted(actual_members - expected_members)
        raise ValueError(f"observable trace bundle member mismatch: missing={missing!r}, unknown={unknown!r}")
    if any(path.is_dir() for path in bundle_root.iterdir()):
        raise ValueError("observable trace bundle must not contain nested directories")

    declared = {table.path: table for table in manifest.tables}
    if set(declared) != set(OBSERVABLE_TRACE_TABLE_MODELS):
        raise ValueError("observable trace manifest does not declare the exact canonical table set")

    for name, model in OBSERVABLE_TRACE_TABLE_MODELS.items():
        table = declared[name]
        path = bundle_root / name
        if sha256_file(path) != table.sha256:
            raise ValueError(f"observable trace table checksum mismatch: {name}")
        schema_id = model.model_fields["schema_id"].default
        if table.row_schema_id != schema_id:
            raise ValueError(f"observable trace row schema mismatch: {name}")

    feature_units: set[tuple[str, str, str, str | None, str | None, str | None, str | None]] = set()
    feature_projections: set[tuple[str, str, str, str | None, str | None, str | None, str | None]] = set()
    observed_models: set[str] = set()
    observed_runs: set[str] = set()
    observed_suites: set[str] = set()
    feature_count = 0
    feature_model = OBSERVABLE_TRACE_TABLE_MODELS["unit_features.csv"]
    for model_row in iter_contract_csv(bundle_root / "unit_features.csv", model=feature_model):
        row = model_row.model_dump(mode="json")
        unit = (
            str(row["benchmark"]),
            str(row["model_id"]),
            str(row["run_id"]),
            str(row["task_id"]) if row.get("task_id") is not None else None,
            str(row["assignment_id"]) if row.get("assignment_id") is not None else None,
            str(row["program_id"]) if row.get("program_id") is not None else None,
            str(row["phase_id"]) if row.get("phase_id") is not None else None,
        )
        if unit in feature_units:
            raise ValueError("observable trace feature rows must identify unique benchmark units")
        feature_units.add(unit)
        optional_dimensions = tuple((None, value) if value is not None else (None,) for value in unit[3:])
        feature_projections.update(
            (unit[0], unit[1], unit[2], task_id, assignment_id, program_id, phase_id)
            for task_id, assignment_id, program_id, phase_id in product(*optional_dimensions)
        )
        observed_suites.add(unit[0])
        observed_models.add(unit[1])
        observed_runs.add(unit[2])
        feature_count += 1
    if feature_count != declared["unit_features.csv"].row_count:
        raise ValueError("observable trace row count mismatch: unit_features.csv")
    if not feature_units:
        raise ValueError("observable trace bundle contains no observable benchmark units")
    if tuple(sorted(observed_models)) != manifest.model_ids or tuple(sorted(observed_runs)) != manifest.run_ids:
        raise ValueError("observable trace manifest model/run identities disagree with feature rows")
    if tuple(sorted(observed_suites)) != manifest.benchmark_suites:
        raise ValueError("observable trace manifest suites disagree with feature rows")

    for name, model in OBSERVABLE_TRACE_TABLE_MODELS.items():
        if name == "unit_features.csv":
            continue
        row_count = 0
        for model_row in iter_contract_csv(bundle_root / name, model=model):
            row_count += 1
            row = model_row.model_dump(mode="json")
            if name == "evidence_use.csv":
                evidence_unit = (
                    str(row["benchmark"]),
                    str(row["model_id"]),
                    str(row["run_id"]),
                    str(row["task_id"]) if row.get("task_id") is not None else None,
                    str(row["assignment_id"]) if row.get("assignment_id") is not None else None,
                    str(row["program_id"]) if row.get("program_id") is not None else None,
                    str(row["phase_id"]) if row.get("phase_id") is not None else None,
                )
                if evidence_unit not in feature_projections:
                    raise ValueError("evidence-use row references an unknown observable unit")
        if row_count != declared[name].row_count:
            raise ValueError(f"observable trace row count mismatch: {name}")

    return manifest


def main(argv: list[str] | None = None) -> int:
    """Validate one observable trace bundle from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = validate_trace_analysis_bundle(args.bundle)
    print(
        json.dumps(
            {
                "status": "pass",
                "models": len(manifest.model_ids),
                "runs": len(manifest.run_ids),
                "tables": len(manifest.tables),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["validate_trace_analysis_bundle", "main"]
