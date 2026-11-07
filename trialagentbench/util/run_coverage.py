"""Prospective schedule and atomic completion tracking for benchmark runs."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from trialagentbench.contracts.core.runs import RunCoverageV1
from trialagentbench.io import canonical_payload_sha256, read_json_model


def run_coverage_v1(*, run_identity_sha256: str, unit_ids: tuple[str, ...]) -> RunCoverageV1:
    """Create the immutable denominator for one benchmark run."""

    return RunCoverageV1(
        run_identity_sha256=run_identity_sha256,
        schedule_sha256=canonical_payload_sha256(list(unit_ids)),
        unit_ids=unit_ids,
    )


def load_or_create_run_coverage_v1(
    path: Path,
    *,
    run_identity_sha256: str,
    unit_ids: tuple[str, ...],
    resume: bool,
) -> RunCoverageV1:
    """Load an identity-matched schedule or create it before execution."""

    expected = run_coverage_v1(run_identity_sha256=run_identity_sha256, unit_ids=unit_ids)
    if resume:
        recorded = read_json_model(RunCoverageV1, path)
        if (
            recorded.run_identity_sha256 != expected.run_identity_sha256
            or recorded.schedule_sha256 != expected.schedule_sha256
            or recorded.unit_ids != expected.unit_ids
        ):
            raise ValueError("Persisted run coverage conflicts with the immutable schedule.")
        return recorded
    write_run_coverage_v1(path, expected)
    return expected


def reconcile_run_coverage_v1(
    path: Path,
    coverage: RunCoverageV1,
    *,
    completed_unit_ids: tuple[str, ...],
) -> RunCoverageV1:
    """Persist the ordered completed subset of a prospective schedule."""

    reconciled = coverage.model_copy(update={"completed_unit_ids": completed_unit_ids})
    if reconciled != coverage:
        write_run_coverage_v1(path, reconciled)
    return reconciled


def write_run_coverage_v1(path: Path, coverage: RunCoverageV1) -> None:
    """Atomically replace a validated coverage view."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = json.dumps(coverage.model_dump(mode="json"), indent=2, ensure_ascii=True, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = [
    "load_or_create_run_coverage_v1",
    "reconcile_run_coverage_v1",
    "run_coverage_v1",
    "write_run_coverage_v1",
]
