"""Tests for role-separated TrialEval release custody."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from trialagentbench.release import (
    ParticipantReleaseError,
    load_evaluation_release,
)


def _write_evaluation_release(root: Path) -> Path:
    evaluation = root / "evaluation"
    grader = evaluation / "grader"
    grader.mkdir(parents=True)
    artifact = grader / "scoring_keys.jsonl"
    artifact.write_text('{"item_id":"te_0123456789abcdef0123"}\n', encoding="utf-8")
    artifacts = []
    for path in sorted(grader.iterdir()):
        payload = path.read_bytes()
        artifacts.append(
            {
                "relative_path": f"evaluation/grader/{path.name}",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "byte_length": len(payload),
            }
        )
    manifest = {
        "schema_id": "ariadne.benchmark_role_manifest/v1",
        "release_id": f"release_{'0' * 64}",
        "role": "evaluation",
        "trial_count": 1,
        "task_count": 1,
        "artifacts": artifacts,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    manifest["checksum"] = hashlib.sha256(canonical).hexdigest()
    (evaluation / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return evaluation


def test_evaluation_release_rejects_changed_bytes(tmp_path: Path) -> None:
    evaluation = _write_evaluation_release(tmp_path)

    loaded = load_evaluation_release(evaluation)
    assert loaded.manifest.role == "evaluation"

    (evaluation / "grader" / "scoring_keys.jsonl").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ParticipantReleaseError, match="artifact changed"):
        load_evaluation_release(evaluation)


def test_evaluation_release_rejects_undeclared_bytes(tmp_path: Path) -> None:
    evaluation = _write_evaluation_release(tmp_path)
    (evaluation / "grader" / "extra.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ParticipantReleaseError, match="files differ"):
        load_evaluation_release(evaluation)


def test_evaluation_release_rejects_missing_bytes(tmp_path: Path) -> None:
    """Require every artifact named by the role manifest."""

    evaluation = _write_evaluation_release(tmp_path)
    (evaluation / "grader" / "scoring_keys.jsonl").unlink()

    with pytest.raises(ParticipantReleaseError, match="artifact is missing"):
        load_evaluation_release(evaluation)


def test_evaluation_release_rejects_symbolic_link_artifacts(tmp_path: Path) -> None:
    """Reject artifacts that escape byte custody through symbolic links."""

    evaluation = _write_evaluation_release(tmp_path)
    artifact = evaluation / "grader" / "scoring_keys.jsonl"
    external = tmp_path / "external.jsonl"
    external.write_bytes(artifact.read_bytes())
    artifact.unlink()
    artifact.symlink_to(external)

    with pytest.raises(ParticipantReleaseError, match="artifact is missing"):
        load_evaluation_release(evaluation)
