"""Authenticated, streaming access to released property-verification inputs."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from trialagentbench_verifier.artifacts.anchor import ReleaseInputResolver
from trialagentbench_verifier.artifacts.archive import opened_release
from trialagentbench_verifier.artifacts.io import sha256_file
from trialagentbench_verifier.properties.contracts import (
    CellDeclaration,
    CellDeclarations,
    PrivateTaskAssignment,
    ProcessProfiles,
    Replication,
    REstimateRequest,
    RunInputManifest,
    SequentialPlan,
    TaskVariant,
    VerifiedRelease,
)
from trialagentbench_verifier.properties.instantiation import ClusterSizeLaw, EstimandInstantiation


class ReplayManifest(BaseModel):
    """Published replay census, including its exact records binding."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_id: str
    release_id: str
    replays_per_law: int = Field(gt=0)
    seed_namespace: str
    records_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    variant: str


@dataclass
class Dataset:
    """Parsed tables and documents from one scored or replay context view."""

    variant: TaskVariant
    kind: str
    index: int
    tables: dict[str, pd.DataFrame]
    documents: dict[str, str]
    sources: dict[str, str]
    null_columns: set[tuple[str, str]] | None = None


@dataclass(frozen=True)
class BoundInputs:
    """Manifest and authenticated byte snapshots, checked before parsing."""

    manifest: RunInputManifest
    files: dict[Path, bytes]
    digests: dict[str, str]
    task_assignments_path: Path | None
    s0_task_assignment_path: Path | None = None
    s0_flag_reference_path: Path | None = None


def load_run_inputs(path: Path, directory: Path) -> BoundInputs:
    """Validate every binding and require coverage of all declaration files.

    Relative paths resolve from the manifest directory. Snapshots prevent a
    second, unvalidated read between digest verification and parsing.
    """
    body = path.read_bytes()
    manifest = RunInputManifest.model_validate_json(body)
    bindings = [
        manifest.declarations,
        manifest.crosswalk,
        manifest.tier_table,
        *manifest.declaration_files,
        *manifest.trialdev_records,
        *(manifest.gate_records or ()),
    ]
    if manifest.task_assignments:
        bindings.append(manifest.task_assignments)
    bindings.extend(binding for binding in (manifest.s0_task_assignment, manifest.s0_flag_reference) if binding)
    files: dict[Path, bytes] = {}
    digests = {str(path.resolve()): hashlib.sha256(body).hexdigest()}
    for binding in bindings:
        source = (path.parent / binding.path).resolve()
        snapshot = files.get(source)
        if snapshot is None:
            snapshot = source.read_bytes()
        digest = hashlib.sha256(snapshot).hexdigest()
        if digest != binding.sha256:
            raise ValueError(f"Run-input SHA256 mismatch: {source}")
        files[source] = snapshot
        digests[str(source)] = digest
    missing = {p.resolve() for p in directory.iterdir() if p.is_file()} - files.keys()
    if missing:
        raise ValueError(f"Unbound declaration files: {sorted(map(str, missing))}")
    assignment_path = (path.parent / manifest.task_assignments.path).resolve() if manifest.task_assignments else None
    return BoundInputs(
        manifest,
        files,
        digests,
        assignment_path,
        (path.parent / manifest.s0_task_assignment.path).resolve() if manifest.s0_task_assignment else None,
        (path.parent / manifest.s0_flag_reference.path).resolve() if manifest.s0_flag_reference else None,
    )


def load_task_assignments(inputs: BoundInputs) -> tuple[PrivateTaskAssignment, ...]:
    """Read the independent private catalog assignment from authenticated bytes."""
    if inputs.task_assignments_path is None:
        return ()
    import csv

    return tuple(
        PrivateTaskAssignment.model_validate(row)
        for row in csv.DictReader(io.StringIO(inputs.files[inputs.task_assignments_path].decode()))
    )


def load_private_cells(directory: Path, inputs: BoundInputs) -> tuple[CellDeclaration, ...]:
    """Read all 23 private inventory cells, retaining their split membership."""
    inventory = json.loads(inputs.files[(directory / "inventory_full.json").resolve()])
    cells = []
    for record in inventory["regime_cells"]:
        record = {k: v for k, v in record.items() if k not in {"realism_source", "negative_control_reason"}}
        cells.append(CellDeclaration.model_validate({**record, "declaration_scope": "full"}))
    return tuple(cells)


def public_bundle(inputs: BoundInputs) -> dict[str, Any] | None:
    """Read a bundled public declaration from its authenticated manifest binding."""
    snapshot = next(
        body for path, body in inputs.files.items() if inputs.digests[str(path)] == inputs.manifest.declarations.sha256
    )
    payload: dict[str, Any] = json.loads(snapshot)
    return payload if payload.get("schema_id") == "trialagentbench.public_verification_declarations/v1" else None


def load_declarations(directory: Path, *, inputs: BoundInputs, private: bool = False) -> CellDeclarations:
    """Parse authenticated declarations and attach the separately exported process profiles.

    Parameters
    ----------
    directory
        Directory containing the public declarations and process-profile export.
    private
        Restore tier-only mechanisms from the private data export for internal checks.
    inputs
        Authenticated snapshots from the supplied run-input manifest.

    Returns
    -------
    CellDeclarations
        Validated declarations with the declared release precision and profiles.
    """
    bundle = public_bundle(inputs)
    if bundle is not None:
        if private:
            raise ValueError("The public declaration bundle cannot supply private cell parameters")
        return CellDeclarations.model_validate(bundle["declarations"])
    declarations = CellDeclarations.model_validate_json(
        next(
            body
            for path, body in inputs.files.items()
            if inputs.digests[str(path)] == inputs.manifest.crosswalk.sha256
        )
    )
    profiles = ProcessProfiles.model_validate_json(inputs.files[(directory / "process_profiles.json").resolve()])
    payload = declarations.model_dump(mode="json", exclude_unset=True)
    profile_values = profiles.model_dump(mode="json", by_alias=True)
    payload["release_numeric_decimal_places"] = profile_values.pop("release_numeric_decimal_places")
    payload["process_profiles"] = profile_values
    if private:
        full = {cell.regime_cell_id: cell for cell in load_private_cells(directory, inputs)}
        payload["cells"] = [full[cell.regime_cell_id].model_dump(mode="json") for cell in declarations.cells]
    return CellDeclarations.model_validate(payload)


def load_sequential_plan(directory: Path, inputs: BoundInputs) -> SequentialPlan:
    """Read the authenticated completed-cohort maturity forecast."""
    bundle = public_bundle(inputs)
    payload = (
        bundle["sequential_plan"]
        if bundle is not None
        else json.loads(inputs.files[(directory / "s09_sequential_plan.json").resolve()])
    )
    return SequentialPlan.model_validate({name: payload[name] for name in SequentialPlan.model_fields})


def load_instantiations(
    directory: Path, inputs: BoundInputs
) -> tuple[ClusterSizeLaw, dict[str, EstimandInstantiation]]:
    """Read cluster and estimand exports solely from authenticated snapshots."""
    bundle = public_bundle(inputs)
    law = (
        bundle["cluster_size_law"]
        if bundle is not None
        else json.loads(inputs.files[(directory / "cluster_size_law.json").resolve()])
    )
    estimands = (
        bundle["estimand_instantiations"]
        if bundle is not None
        else json.loads(inputs.files[(directory / "estimand_instantiations.json").resolve()])
    )
    return (
        ClusterSizeLaw.model_validate({name: law[name] for name in ClusterSizeLaw.model_fields}),
        {name: EstimandInstantiation.model_validate(value) for name, value in estimands["series"].items()},
    )


def parse_dataset(root: Path, variant: TaskVariant, kind: str, index: int) -> Dataset:
    """Open the actual tables and documents of one variant.

    Parameters
    ----------
    root
        Logical task directory, including an archive-backed path.
    variant
        Authenticated scored identity shared by the replay law.
    kind, index
        Dataset kind and replay index (zero for the scored dataset).

    Returns
    -------
    Dataset
        Parsed bytes; no source filename is treated as proof of readability.
    """
    tables: dict[str, pd.DataFrame] = {}
    documents: dict[str, str] = {}
    sources: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix == ".parquet":
            tables[relative] = pd.read_parquet(io.BytesIO(path.read_bytes()))
            sources[relative] = str(path)
        elif path.name == "standards.zip":
            with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as archive:
                for member in ("AE.parquet", "SUPPAE.parquet"):
                    if member in archive.namelist():
                        tables[f"{relative}#{member}"] = pd.read_parquet(io.BytesIO(archive.read(member)))
        elif relative.startswith("documents/"):
            documents[path.name] = path.read_text(encoding="utf-8")
    return Dataset(variant, kind, index, tables, documents, sources)


class PublicInputs:
    """Bind catalog, keys, replay manifest and archives through the authenticated release resolver.

    Parameters
    ----------
    root
        Published HF directory (or a store directory containing that view).
    anchor_sha256, sums
        Trusted checksum census identity and optional external census location.
    """

    def __init__(self, root: Path, *, anchor_sha256: str, sums: Path | None = None) -> None:
        if not (root / "full/trialeval-evaluator.assets.json").exists():
            root = root / "releases/v1.1.0/hf"
        self.root = root
        self.resolver = ReleaseInputResolver(root, anchor_sha256=anchor_sha256, sums=sums)
        self.catalog = root / "full/trialeval-evaluator.assets.json"
        self.catalog_sha256 = self.resolver.require(self.catalog)
        manifest = root / "replay-bank/manifest.json"
        self.resolver.require(manifest)
        self.manifest = ReplayManifest.model_validate_json(manifest.read_bytes())
        self.resolver.require(root / "replay-bank/records.jsonl", expected_sha256=self.manifest.records_sha256)
        self.replay_indices: dict[str, list[int]] = {}
        self.replay_records: dict[tuple[str, int], dict[str, Any]] = {}
        for line in (root / "replay-bank/records.jsonl").read_text().splitlines():
            record = json.loads(line)
            self.replay_indices.setdefault(record["task_id"], []).append(record["replay_index"])
            self.replay_records[record["task_id"], record["replay_index"]] = record
        self.identity = VerifiedRelease(
            root=str(root),
            anchor_sha256=anchor_sha256,
            sums_sha256=anchor_sha256,
            release_id=self.manifest.release_id,
            split="public",
        )
        self.digests: dict[str, str] = {
            str(self.catalog): self.catalog_sha256,
            str(manifest): sha256_file(manifest),
            str(root / "replay-bank/records.jsonl"): self.manifest.records_sha256,
        }

    @contextmanager
    def opened(self) -> Iterator[tuple[Path, tuple[TaskVariant, ...]]]:
        """Authenticate catalog payloads and yield all released task identities."""
        with opened_release(self.catalog, expected_sha256=self.catalog_sha256, verify_files=True) as root:
            variants = []
            seen: set[str] = set()
            for line in (root / "grading_key/records.jsonl").read_text().splitlines():
                key: dict[str, Any] = json.loads(line)
                task_id = key["task_id"]
                if task_id in seen:
                    raise ValueError(f"Duplicate released task: {task_id}")
                if key["release_id"] != self.identity.release_id:
                    raise ValueError("Replay and scored release identities differ")
                seen.add(task_id)
                task = json.loads((root / "participant/tasks" / task_id / "task.json").read_bytes())
                variants.append(
                    TaskVariant(
                        task_id=task_id,
                        cell_ref=key["cell_ref"],
                        context=key["context"],
                        task_key=f"{key['cell_ref']}#{key['target']['value']}",
                        materials={m["artifact_id"]: m["relative_path"] for m in task["available_materials"]},
                        key_record=key,
                    )
                )
            yield root, tuple(variants)

    @contextmanager
    def replays(self, variant: TaskVariant) -> Iterator[Path]:
        """Authenticate one replay ZIP before exposing any of its members."""
        path = self.root / "replay-bank/tasks" / f"{variant.task_id}.zip"
        digest = self.resolver.require(path)
        self.digests[str(path)] = digest
        with opened_release(path, expected_sha256=digest) as root:
            yield root / variant.task_id


def write_r_payloads(dataset: Dataset, analysis: pd.DataFrame, declarations: CellDeclarations, output: Path) -> int:
    """Write lossless CSVs and R requests for all credited routes of a C1 dataset.

    Parameters
    ----------
    dataset, analysis
        Authenticated dataset and its analysis table, without rounding.
    declarations
        Normative route instructions.
    output
        New dataset directory. Existing payloads are never overwritten.

    Returns
    -------
    int
        Number of serialized estimation requests.
    """
    if dataset.variant.context != "C1":
        raise ValueError("R payloads use the canonical C1 observations")
    output.mkdir(parents=True, exist_ok=False)
    crosswalk = {x.cell_ref: x.regime_cell_id for x in declarations.crosswalk}
    cell = next(x for x in declarations.cells if x.regime_cell_id == crosswalk[dataset.variant.cell_ref])
    tables = {"analysis": analysis}
    for name, table in dataset.tables.items():
        if name != "data/analysis.parquet" and "#" not in name:
            tables[Path(name).stem] = table
    if "monitoring_plan.csv" in dataset.documents:
        tables["monitoring_plan"] = pd.read_csv(
            io.StringIO(dataset.documents["monitoring_plan.csv"]), float_precision="round_trip"
        )
    # Use one directory per fit. CSVs are hardlinked, so additional routes cost no data copies.
    csv_root = output / "csv"
    csv_root.mkdir()
    for name, table in tables.items():
        csv_table = table.copy()
        # R's CSV type conversion recognizes TRUE/FALSE. Python's title-case
        # spelling otherwise turns an observed-event indicator into text.
        for column in csv_table:
            values = csv_table[column]
            present = values.dropna()
            if pd.api.types.is_bool_dtype(values.dtype) or (
                pd.api.types.is_object_dtype(values.dtype)
                and len(present) > 0
                and all(isinstance(value, (bool, np.bool_)) for value in present)
            ):
                csv_table[column] = values.map({True: "TRUE", False: "FALSE"})
        csv_table.to_csv(csv_root / f"{name}.csv", index=False, na_rep="NA", float_format="%.17g")
    import os

    for index, route in enumerate(cell.reference_analyses):
        fit = output / f"route-{index:02d}"
        fit.mkdir()
        inputs = {name: f"{name}.csv" for name in tables}
        for name in tables:
            os.link(csv_root / inputs[name], fit / inputs[name])
        request = REstimateRequest(
            type="estimate",
            tables={
                "data": (
                    inputs["monitoring_results"]
                    if route.procedure.partition == "monitoring_look"
                    else inputs["analysis"]
                ),
                **inputs,
            },
            cell_id=cell.regime_cell_id,
            route_id=route.route_id,
        )
        (fit / "request.json").write_text(request.model_dump_json(indent=2) + "\n")
    replication = Replication.model_validate(
        {"variant": dataset.variant, "kind": dataset.kind, "index": dataset.index, "table_paths": dataset.sources}
    )
    (output / "replication.json").write_text(replication.model_dump_json(indent=2) + "\n")
    return len(cell.reference_analyses)
