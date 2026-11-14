"""Typed report projections and reconciliation of the independent fit census."""

from __future__ import annotations

from collections import defaultdict
from math import isclose

from pydantic import Field

from trialagentbench_verifier.properties.contracts import FileDigest, VerifiedRelease
from trialagentbench_verifier.verification.contracts import (
    ClaimNumber,
    NB2CensusRow,
    NB2Receipt,
    SourceProjection,
)


class RunMetadata(SourceProjection):
    """Execution identity, without importing any private configuration values."""

    candidate_commit: str
    input_manifest_sha256: str
    image_digest: str
    R: str


class PayloadIdentity(SourceProjection):
    """Pinned public payload revision, supplied by the release manifest."""

    revision: str = Field(pattern=r"^[0-9a-f]{40}$")


class DeterministicSummary(SourceProjection):
    """Dataset and view census produced by the deterministic runner."""

    release: VerifiedRelease
    n_datasets: int = Field(gt=0)
    n_variants: int = Field(gt=0)
    n_tasks: int = Field(gt=0)


class CellReference(SourceProjection):
    """Cell identifier only; declarations cannot enter the public panel."""

    regime_cell_id: str


class PipelineCell(SourceProjection):
    """Stored pipeline task-score census identifier."""

    task_id: str


class ContextReference(SourceProjection):
    """Declared context label."""

    tier: str


class DeclarationCensus(SourceProjection):
    """Identifier sets used to classify exact-check scopes."""

    cells: tuple[CellReference, ...]
    context_definitions: tuple[ContextReference, ...]


class Precision(SourceProjection):
    """Release precision supplied by the process declaration."""

    release_numeric_decimal_places: int = Field(ge=0)


class ExternalQuantity(SourceProjection):
    """Independent external route's comparison and its declared allowance."""

    route: str
    quantity: str
    quantity_pass: bool
    absolute_tolerance: float = Field(ge=0)


class ExternalSource(SourceProjection):
    """Exact release-source commit used by the external route panel."""

    commit: str


class SequentialField(SourceProjection):
    """Per-field sequential comparison allowance without fitted values."""

    split: str
    field: str
    tolerance: float = Field(ge=0)
    complete_check_passed: bool


def display_number(value: float) -> str:
    """Use Python's general format with six significant digits for report floats."""
    return format(value, ".6g")


class Numbers:
    """Collect public number derivations separately from the public report."""

    def __init__(self, bindings: dict[str, FileDigest]) -> None:
        self.bindings = bindings
        self.rows: dict[str, dict[str, ClaimNumber]] = {}

    def add(
        self,
        row: str,
        label: str,
        value: int | float | str,
        sources: str | tuple[str, ...],
        fields: str | tuple[str, ...],
        operation: str = "identity",
    ) -> None:
        """Bind a named displayed number to its source field and reduction."""
        names = (sources,) if isinstance(sources, str) else sources
        self.rows.setdefault(row, {})[label] = ClaimNumber(
            value=value,
            sources=tuple(self.bindings[name] for name in names),
            fields=(fields,) if isinstance(fields, str) else fields,
            operation=operation,
            display=display_number(value) if isinstance(value, float) else None,
            display_format=".6g" if isinstance(value, float) else None,
        )


def nb2_counts(rows: list[NB2CensusRow], summary: NB2Receipt) -> tuple[int, int]:
    """Reconcile every census fit and compared field before counting readouts."""
    if not rows:
        raise ValueError("NB2 census is empty")
    fits: dict[tuple[str, str, str], dict[str, NB2CensusRow]] = defaultdict(dict)
    unbound: set[tuple[str, str]] = set()
    for row in rows:
        if row.field is None:
            if row.status != "unbindable" or (row.readout_root, row.readout_id) in unbound:
                raise ValueError("NB2 census has an invalid unbound readout")
            unbound.add((row.readout_root, row.readout_id))
            continue
        if not row.arm_id or row.status == "unbindable":
            raise ValueError("NB2 census fit lacks its arm binding")
        key = (row.readout_root, row.readout_id, row.arm_id)
        if row.field in fits[key]:
            raise ValueError("NB2 census contains a duplicate fit field")
        fits[key][row.field] = row
    compared = passed = failed = partial = crashed = nonconverged = 0
    readouts: set[tuple[str, str]] = set()
    for key, fields in fits.items():
        if fields.keys() != {"estimate", "standard_error", "interval_lower", "interval_upper"}:
            raise ValueError("NB2 census fit has an incomplete field schema")
        checks = {name: row for name, row in fields.items() if row.status in {"passed", "failed"}}
        stored = {name for name, row in fields.items() if row.stored is not None}
        if stored != checks.keys():
            raise ValueError("NB2 stored fit fields lack comparisons")
        if checks:
            if checks.keys() != {"estimate", "standard_error"}:
                raise ValueError("NB2 comparison must cover exactly estimate and SE")
            for row in checks.values():
                if any(value is None for value in (row.r, row.stored, row.difference, row.tolerance)):
                    raise ValueError("NB2 compared field lacks numeric evidence")
                assert row.r is not None and row.stored is not None
                assert row.difference is not None and row.tolerance is not None
                # The census compares raw R fits by absolute difference; its
                # tolerance already contains the producer's atol + rtol * abs(r).
                difference = abs(row.r - row.stored)
                # Allow only binary64/serialization noise: max(1e-15, 1e-12
                # times the larger discrepancy), independently of fit tolerance.
                if not isclose(difference, row.difference, rel_tol=1e-12, abs_tol=1e-15):
                    raise ValueError("NB2 supplied discrepancy disagrees with its R and stored values")
                if (difference <= row.tolerance) != (row.status == "passed"):
                    raise ValueError("NB2 field status disagrees with its tolerance")
            compared += 1
            fit_passed = all(row.status == "passed" for row in checks.values())
            passed += fit_passed
            failed += not fit_passed
            partial += len(checks) < len(fields)
            readouts.add(key[:2])
        crashed += any(row.status == "crashed" for row in fields.values())
        nonconverged += any(row.status == "non-converged" for row in fields.values())
    bound = {key[:2] for key in fits}
    if bound & unbound:
        raise ValueError("NB2 readout is both bound and unbindable")
    counts = dict(
        bound=len(bound),
        unbindable=len(unbound),
        fits=len(fits),
        compared=compared,
        passed=passed,
        failed=failed,
        unavailable=len(fits) - compared,
        non_converged=nonconverged,
        crashed=crashed,
        partially_compared=partial,
    )
    if summary.counts != counts or (summary.compared, summary.passed, summary.failed) != (compared, passed, failed):
        raise ValueError("NB2 census disagrees with the comparison summary")
    return compared, len(readouts)
