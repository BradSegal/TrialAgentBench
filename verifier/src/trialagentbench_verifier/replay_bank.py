"""The TrialEval replay bank: fresh datasets of each public task, for scoring at the true effect.

A TrialEval answer is scored by replaying its saved script unchanged on fresh datasets drawn from its task's
data-generating process (same true effect, new data, the task's own evidence view) and valuing each replay's
decision at the true effect against the scripted reference route's decision on the same dataset. The bank holds those datasets
and, per dataset, every route's report. It is an evaluator-side artifact published beside the frozen task
release; its identity is the SHA-256 of ``manifest.json``, checked once when loaded.

A bank has one of two variants. The operator's ``full`` bank records every route, including the fixed policies whose
anchors the operator scores. The ``public`` bank (:func:`public_replay_bank_v1`) records per replay only the true
effect and the primary reference report, under the role name ``reference`` rather than its method. It has the
same schema for every task, names no method and carries no fixed policy. Where several reference analyses are valid,
public-bank regret is an upper bound on the operator's minimum regret over reference analyses.

Layout::

    manifest.json                                  ReplayBankManifestV1
    records.jsonl                                  one ReplayBankRecordV1 per (task, replay)
    tasks/<task_id>/<replay_index:02d>/            the participant task: task.json, documents/, data/
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

REPLAY_BANK_SCHEMA_ID: Literal["trialagentbench.trialeval.replay_bank/v1"] = "trialagentbench.trialeval.replay_bank/v1"
PUBLIC_REFERENCE_ROUTE_ID: Literal["reference"] = "reference"
"""The public bank's only route: the scripted reference, named by its role rather than its method."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ReplayBankReportV1(_Model):
    """One route's estimate and 95% interval on the comparison scale (log scale for ratios)."""

    estimate: float
    lower: float
    upper: float


class ReplayBankRecordV1(_Model):
    """One fresh dataset of a task and every route's report on it."""

    task_id: str
    replay_index: int = Field(ge=1)
    truth: float = Field(description="The law's true effect on the comparison scale.")
    reference_route_id: str = Field(description="The first credited route, retained for bank compatibility.")
    credited_route_ids: tuple[str, ...] = Field(
        default=(),
        exclude_if=lambda value: not value,
        description="All credited routes; empty only for single-reference records.",
    )
    routes: dict[str, ReplayBankReportV1 | None] = Field(description="Route reports; None when a route failed.")

    @model_validator(mode="after")
    def _reference_present(self) -> ReplayBankRecordV1:
        if self.reference_route_id not in self.routes:
            raise ValueError("a replay record must report its scripted reference route")
        if self.credited_route_ids and (
            len(set(self.credited_route_ids)) != len(self.credited_route_ids)
            or self.reference_route_id not in self.credited_route_ids
            or any(route not in self.routes for route in self.credited_route_ids)
        ):
            raise ValueError("credited replay routes must be unique, present, and include the reference")
        return self

    @property
    def reference_route_ids(self) -> tuple[str, ...]:
        """Return every reference route, including a legacy record's sole reference."""
        return self.credited_route_ids or (self.reference_route_id,)


class ReplayBankManifestV1(_Model):
    """The bank's identity: the release it extends, its replay count and the digest of its records."""

    schema_id: Literal["trialagentbench.trialeval.replay_bank/v1"] = REPLAY_BANK_SCHEMA_ID
    release_id: str
    replays_per_law: int = Field(ge=1)
    seed_namespace: str
    records_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    variant: Literal["full", "public"] = Field(
        default="full",
        description="full: every route, which fixed-policy anchors need; public: the truth and the reference only.",
    )


class ReplayBankV1(_Model):
    """A loaded bank: its root, manifest and records by task in replay order."""

    root: Path
    manifest: ReplayBankManifestV1
    records: dict[str, tuple[ReplayBankRecordV1, ...]]

    def task_root(self, task_id: str, replay_index: int) -> Path:
        """Return the participant task directory of one replay."""
        return self.root / "tasks" / task_id / f"{replay_index:02d}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_replay_bank(root: Path, *, expected_manifest_sha256: str) -> ReplayBankV1:
    """Load a replay bank after checking its identity once.

    Parameters
    ----------
    root
        The bank directory.
    expected_manifest_sha256
        The published SHA-256 of ``manifest.json``.

    Returns
    -------
    ReplayBankV1
        The manifest and every task's records, ordered by replay index.

    Raises
    ------
    ValueError
        If the manifest or records differ from their declared digests, or a task lacks a replay index.
    """
    manifest_path = root / "manifest.json"
    if _sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("replay bank manifest differs from the expected digest")
    manifest = ReplayBankManifestV1.model_validate_json(manifest_path.read_bytes())
    records_path = root / "records.jsonl"
    if _sha256(records_path) != manifest.records_sha256:
        raise ValueError("replay bank records differ from the manifest digest")
    by_task: dict[str, list[ReplayBankRecordV1]] = defaultdict(list)
    for line in records_path.read_text(encoding="utf-8").splitlines():
        record = ReplayBankRecordV1.model_validate_json(line)
        by_task[record.task_id].append(record)
    expected = list(range(1, manifest.replays_per_law + 1))
    records = {}
    for task_id, items in by_task.items():
        ordered = sorted(items, key=lambda item: item.replay_index)
        if [item.replay_index for item in ordered] != expected:
            raise ValueError(f"replay bank task {task_id} lacks replays 1..{manifest.replays_per_law}")
        records[task_id] = tuple(ordered)
    if manifest.variant == "public" and any(
        record.reference_route_id != PUBLIC_REFERENCE_ROUTE_ID
        or set(record.routes) != {PUBLIC_REFERENCE_ROUTE_ID}
        or record.credited_route_ids
        for items in records.values()
        for record in items
    ):
        raise ValueError("a public replay bank may record only the reference route")
    return ReplayBankV1(root=root, manifest=manifest, records=records)


def public_replay_bank_v1(bank: ReplayBankV1) -> tuple[bytes, bytes]:
    """Derive the public bank's ``manifest.json`` and ``records.jsonl`` from the operator's full bank.

    Each public record keeps the replay's task, index and true effect and the scripted reference's report, under
    :data:`PUBLIC_REFERENCE_ROUTE_ID`; every other route, including each fixed policy, and the reference's method name
    are dropped. Tasks with one reference analysis score identically on either variant; for tasks with several reference analyses the public
    score is an upper bound on the operator score. The bank's task datasets and public bytes are unchanged.

    Parameters
    ----------
    bank
        The loaded full bank.

    Returns
    -------
    tuple of bytes
        The public manifest and records, deterministic for a given full bank.

    Raises
    ------
    ValueError
        If ``bank`` is already the public variant.
    """
    if bank.manifest.variant != "full":
        raise ValueError("the public replay bank is derived from the full bank")
    records = "".join(
        ReplayBankRecordV1(
            task_id=record.task_id,
            replay_index=record.replay_index,
            truth=record.truth,
            reference_route_id=PUBLIC_REFERENCE_ROUTE_ID,
            routes={PUBLIC_REFERENCE_ROUTE_ID: record.routes[record.reference_route_id]},
        ).model_dump_json()
        + "\n"
        for task_id in sorted(bank.records)
        for record in bank.records[task_id]
    ).encode("utf-8")
    manifest = bank.manifest.model_copy(
        update={"records_sha256": hashlib.sha256(records).hexdigest(), "variant": "public"}
    )
    return (manifest.model_dump_json(indent=2) + "\n").encode("utf-8"), records


__all__ = [
    "PUBLIC_REFERENCE_ROUTE_ID",
    "REPLAY_BANK_SCHEMA_ID",
    "ReplayBankManifestV1",
    "ReplayBankRecordV1",
    "ReplayBankReportV1",
    "ReplayBankV1",
    "load_replay_bank",
    "public_replay_bank_v1",
]
