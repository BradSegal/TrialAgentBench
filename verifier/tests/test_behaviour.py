"""Standalone observable-count reproduction never exports raw custody fields."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

import pytest

from trialagentbench_verifier.behaviour import reproduce_behaviour_figures
from trialagentbench_verifier.io import sha256_file


def bundle(root: Path) -> Path:
    root.mkdir()
    tables = []
    content = {
        "unit_features.csv": [
            {
                "schema_id": "ariadne.trace_feature_row/v1",
                "model_id": "fixture/model",
                "execution_status": "completed",
                "trace_coverage_status": "full_conversation_trace",
                "endpoint_valid": "true",
                "development_value": "-3",
                "source_path": "/private/SECRET_HOST",
            }
        ],
        "trialdev_decisions.csv": [
            {
                "schema_id": "trialagentbench.trialdev_decision_row/v1",
                "model_id": "fixture/model",
                "submission_status": "graded",
                "action": "stop",
                "world_id": "SECRET_WORLD",
                "source_relative_path": "SECRET_REFERENCE",
            }
        ],
        "action_events.csv": [],
        "evidence_use.csv": [],
    }
    for name, rows in content.items():
        path = root / name
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["schema_id", "model_id"])
            writer.writeheader()
            writer.writerows(rows)
        tables.append(
            {
                "path": name,
                "row_schema_id": (
                    rows[0]["schema_id"]
                    if rows
                    else {
                        "action_events.csv": "ariadne.action_trace_event/v1",
                        "evidence_use.csv": "ariadne.evidence_use_row/v1",
                    }[name]
                ),
                "row_count": len(rows),
                "sha256": sha256_file(path),
            }
        )
    manifest = root / "trace_bundle_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_id": "trialagentbench.observable_trace_bundle/v2",
                "benchmark_suites": ["trialdev"],
                "model_ids": ["fixture/model"],
                "run_ids": ["run"],
                "tables": tables,
            }
        )
    )
    return manifest


def test_aggregate_disclosure_keeps_counts_and_excludes_raw_paths_and_values(tmp_path: Path) -> None:
    manifest = bundle(tmp_path / "controlled")
    output = reproduce_behaviour_figures(
        manifest.parent, expected_manifest_sha256=sha256_file(manifest), output=tmp_path / "public"
    )
    assert {path.name for path in output.iterdir()} == {
        "counts.csv",
        "completion.svg",
        "decisions.svg",
        "source.sha256",
    }
    for path in output.iterdir():
        assert b"SECRET" not in path.read_bytes()
        assert b"development_value" not in path.read_bytes()
    with (output / "counts.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert {row["category"] for row in rows} == {"completed", "full_conversation_trace", "graded", "stop"}
    assert all(row["count"] == "1" for row in rows)
    for path in output.glob("*.svg"):
        root = ElementTree.parse(path).getroot()
        assert root.find("{http://www.w3.org/2000/svg}title") is not None
        assert root.find("{http://www.w3.org/2000/svg}desc") is not None


def test_mutated_or_undeclared_custody_cannot_produce_a_disclosure(tmp_path: Path) -> None:
    manifest = bundle(tmp_path / "controlled")
    task = manifest.parent / "unit_features.csv"
    task.write_text(task.read_text().replace("completed", "model_noncompletion"))
    with pytest.raises(ValueError, match="bytes differ"):
        reproduce_behaviour_figures(
            manifest.parent, expected_manifest_sha256=sha256_file(manifest), output=tmp_path / "public"
        )
    assert not (tmp_path / "public").exists()


def test_foreign_directory_reproduction_has_no_harness_or_simulator_import(tmp_path: Path) -> None:
    manifest = bundle(tmp_path / "controlled")
    script = """
import importlib.abc, sys
from pathlib import Path
class PublicOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "trialagentbench" or fullname.startswith(("trialagentbench.", "ariadne", "trialagentbench_construction", "trialagentbench_qualification")):
            raise RuntimeError("Private application import: " + fullname)
sys.meta_path.insert(0, PublicOnly())
from trialagentbench_verifier.behaviour import reproduce_behaviour_figures
reproduce_behaviour_figures(Path(sys.argv[1]), expected_manifest_sha256=sys.argv[2], output=Path(sys.argv[3]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(manifest.parent), sha256_file(manifest), str(tmp_path / "public")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "public" / "decisions.svg").is_file()
