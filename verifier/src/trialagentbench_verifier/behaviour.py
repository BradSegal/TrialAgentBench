"""Reproduce aggregate behaviour tables and vector figures without the main application.

The input remains controlled raw custody. Outputs disclose only model-labelled
counts, never paths, messages, identifiers, oracle values or provider payloads.
Counts describe observed actions and completion, not comprehension or quality.
"""

from __future__ import annotations

import csv
from collections import Counter
from html import escape
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_verifier.io import sha256_file


class _Table(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: Literal["action_events.csv", "evidence_use.csv", "trialdev_decisions.csv", "unit_features.csv"]
    row_schema_id: str
    row_count: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_schema(self) -> _Table:
        expected = {
            "action_events.csv": "ariadne.action_trace_event/v1",
            "evidence_use.csv": "ariadne.evidence_use_row/v1",
            "trialdev_decisions.csv": "trialagentbench.trialdev_decision_row/v1",
            "unit_features.csv": "ariadne.trace_feature_row/v1",
        }
        if self.row_schema_id != expected[self.path]:
            raise ValueError("Unknown row schema in the v2 behaviour contract")
        return self


class _Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_id: Literal["trialagentbench.observable_trace_bundle/v2"]
    benchmark_suites: tuple[Literal["trialdev", "trialeval"], ...]
    model_ids: tuple[str, ...]
    run_ids: tuple[str, ...]
    tables: tuple[_Table, ...]


class _Row(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$")
    schema_id: str
    event_type: str = ""
    execution_status: str = ""
    trace_coverage_status: str = ""
    submission_status: str = ""
    action: str = ""


def _svg(title: str, rows: list[tuple[str, str, int]]) -> str:
    """Draw every aggregate count with labels and an accessible description."""
    height = 108 + max(1, len(rows)) * 30
    maximum = max((count for _, _, count in rows), default=1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" viewBox="0 0 1000 {height}" role="img" aria-labelledby="title description">',
        f'<title id="title">{escape(title)}</title>',
        '<desc id="description">Observed counts across the complete supplied task inventory. Actions do not establish reasoning quality or scientific success.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="sans-serif" fill="#172b3a">',
        f'<text x="24" y="32" font-size="21">{escape(title)}</text>',
        '<text x="24" y="57" font-size="13">Observed counts; no inference about comprehension or decision quality.</text>',
    ]
    for index, (model, category, count) in enumerate(rows):
        y = 82 + index * 30
        width = 400 * count / max(1, maximum)
        parts.extend(
            [
                f'<text x="24" y="{y+16}" font-size="13">{escape(model)} · {escape(category.replace("_", " "))}</text>',
                f'<rect x="520" y="{y}" width="{width:.2f}" height="20" fill="#237c86"/>',
                f'<text x="{530+width:.2f}" y="{y+15}" font-size="13">{count}</text>',
            ]
        )
    if not rows:
        parts.append('<text x="24" y="96" font-size="14">No observed decisions in this inventory.</text>')
    return "\n".join([*parts, "</g></svg>\n"])


def reproduce_behaviour_figures(source: Path, *, expected_manifest_sha256: str, output: Path) -> Path:
    """Authenticate a v2 trace bundle and disclose aggregate tables and two SVG figures.

    Parameters
    ----------
    source
        Controlled trace bundle produced from native run custody.
    expected_manifest_sha256
        Separately retained SHA-256 of its manifest.
    output
        New directory for the minimal aggregate disclosure.

    Returns
    -------
    Path
        The aggregate disclosure directory.

    Raises
    ------
    ValueError
        If byte identity, table census, row schema or allowed categories differ.
    FileExistsError
        If the output already exists; retained evidence is never overwritten.
    """
    source = source.resolve(strict=True)
    manifest_path = source / "trace_bundle_manifest.json"
    if manifest_path.is_symlink() or sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("Behaviour manifest differs from the retained identity")
    manifest = _Manifest.model_validate_json(manifest_path.read_bytes())
    names = {table.path for table in manifest.tables}
    required = {"action_events.csv", "evidence_use.csv", "trialdev_decisions.csv", "unit_features.csv"}
    if names != required or len(manifest.tables) != len(required):
        raise ValueError("Behaviour table census is incomplete or duplicated")
    if {path.name for path in source.iterdir()} != required | {manifest_path.name}:
        raise ValueError("Behaviour directory contains undeclared files")
    counts: Counter[tuple[str, str, str]] = Counter()
    for table in manifest.tables:
        path = source / table.path
        if path.is_symlink() or sha256_file(path) != table.sha256:
            raise ValueError("Behaviour table bytes differ from their manifest")
        observed = 0
        with path.open(newline="", encoding="utf-8") as handle:
            for values in csv.DictReader(handle):
                observed += 1
                row = _Row.model_validate(values)
                if row.schema_id != table.row_schema_id or row.model_id not in manifest.model_ids:
                    raise ValueError("Behaviour row identity differs from its manifest")
                if table.path == "unit_features.csv":
                    status = row.execution_status or "status_not_projected"
                    if status not in {
                        "completed",
                        "model_noncompletion",
                        "evaluator_error",
                        "not_started",
                        "incomplete_run",
                        "status_not_projected",
                        "provider_stop",
                        "provider_unavailable",
                        "malformed_output",
                        "replay_failure",
                        "budget_exhausted",
                        "pre_run_identity_exclusion",
                    }:
                        raise ValueError("Unknown task execution status in behaviour disclosure")
                    counts[row.model_id, "task_status", status] += 1
                    if row.trace_coverage_status not in {
                        "full_conversation_trace",
                        "submission_only_trace",
                        "scorer_only_trace",
                        "missing_trace",
                    }:
                        raise ValueError("Unknown trace coverage status")
                    counts[row.model_id, "trace_coverage", row.trace_coverage_status] += 1
                elif table.path == "trialdev_decisions.csv":
                    if row.submission_status not in {
                        "accepted",
                        "committed",
                        "graded",
                        "rejected",
                        "ingress_without_grade",
                    }:
                        raise ValueError("Unknown decision submission status")
                    counts[row.model_id, "submission_status", row.submission_status] += 1
                    if (row.submission_status in {"accepted", "committed", "graded"}) != bool(row.action):
                        raise ValueError("Accepted decisions must identify their observed action")
                    if row.action:
                        if row.action not in {
                            "nominate",
                            "defer",
                            "stop",
                            "switch",
                            "backtrack",
                            "run_trial",
                            "continue",
                            "complete",
                            "report",
                            "start_study",
                            "readout",
                            "continue_recruitment",
                            "close_study",
                            "drop_arm",
                            "reduce_control",
                            "wait",
                            "advance",
                        }:
                            raise ValueError("Unknown TrialDev action in behaviour disclosure")
                        observation = "graded_action" if row.submission_status == "graded" else "committed_action"
                        counts[row.model_id, observation, row.action] += 1
        if observed != table.row_count:
            raise ValueError("Behaviour row denominator differs from its manifest")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model_id", "observation", "category", "count"))
        writer.writerows((*key, count) for key, count in sorted(counts.items()))
    for name, title, groups in (
        ("completion.svg", "Task status and trace coverage", {"task_status", "trace_coverage"}),
        (
            "decisions.svg",
            "TrialDev submissions and observed actions",
            {"submission_status", "graded_action", "committed_action"},
        ),
    ):
        rows = [
            (model, f"{group}: {category}", count)
            for (model, group, category), count in sorted(counts.items())
            if group in groups
        ]
        (output / name).write_text(_svg(title, rows), encoding="utf-8")
    (output / "source.sha256").write_text(expected_manifest_sha256 + "\n", encoding="ascii")
    return output
