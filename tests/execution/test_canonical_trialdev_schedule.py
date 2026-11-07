"""Tests for canonical TrialDev selection and resumable terminal coverage."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from trialagentbench.contracts.trialdev.release import TrialDevStream
from trialagentbench.execution_policy import TRIALDEV_RELEASE_BUDGET_V1
from trialagentbench.tools.run import canonical_trialdev
from trialagentbench.tools.run.canonical_trialdev import _selected_tasks


def _release():
    tasks = (
        SimpleNamespace(
            task_id="td-000000000001",
            case_id="case-a",
            data_view="raw",
            stream=TrialDevStream.EVIDENCE_ANALYSIS,
        ),
        SimpleNamespace(
            task_id="td-000000000002",
            case_id="case-b",
            data_view="raw",
            stream=TrialDevStream.PORTFOLIO_MANAGEMENT,
        ),
    )
    return SimpleNamespace(participant=SimpleNamespace(tasks=tasks))


def test_trialdev_selection_rejects_any_unmatched_or_duplicate_selector() -> None:
    release = _release()

    with pytest.raises(ValueError, match="matched no released task"):
        _selected_tasks(release.participant.tasks, ("td-000000000001", "typo"))
    with pytest.raises(ValueError, match="must be unique"):
        _selected_tasks(release.participant.tasks, ("case-a", "case-a"))
    assert tuple(task.task_id for task in _selected_tasks(release.participant.tasks, ("evidence_analysis",))) == (
        "td-000000000001",
    )


def test_trialdev_preparation_retains_a_resume_bound_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Create prospective custody without constructing a provider or running a task."""
    release = _release()
    release.manifest = SimpleNamespace(release_id="release_" + "d" * 64)
    release.participant.release_manifest_sha256 = "a" * 64
    release.analysis_configuration = lambda task: None
    release.authored_configuration = lambda task: None
    monkeypatch.setattr(canonical_trialdev, "load_trialdev_evaluator_release", lambda _: release)
    monkeypatch.setattr(
        canonical_trialdev,
        "resolve_trusted_executor_environment",
        lambda **_: SimpleNamespace(image_id="sha256:" + "0" * 64, limits=TRIALDEV_RELEASE_BUDGET_V1.code_execution),
    )

    def forbidden_provider(*args: object, **kwargs: object) -> None:
        raise AssertionError("preparation contacted a provider")

    monkeypatch.setattr(canonical_trialdev, "get_provider", forbidden_provider)
    args = canonical_trialdev._parser().parse_args(
        [
            "--release",
            "unused.zip",
            "--expected-sha256",
            "b" * 64,
            "--model",
            "fixture/model",
            "--provider",
            "openai",
            "--expected-executor-image-id",
            "sha256:" + "0" * 64,
            "--expected-executor-environment-sha256",
            "c" * 64,
            "--output-root",
            str(tmp_path / "runs"),
            "--prepare-only",
        ]
    )
    assert canonical_trialdev._run_materialized(args, release_root=tmp_path, release_archive_sha256="b" * 64) == 0
    invocation = next((tmp_path / "runs").rglob("invocation.json"))
    before = invocation.read_bytes()
    assert len(json.loads(before)["scheduled_task_ids"]) == 2
    assert not (invocation.parent / "tasks").exists()
    args.resume_run_dir = invocation.parent
    assert canonical_trialdev._run_materialized(args, release_root=tmp_path, release_archive_sha256="b" * 64) == 0
    assert invocation.read_bytes() == before
    args.max_tokens += 1
    assert canonical_trialdev._run_materialized(args, release_root=tmp_path, release_archive_sha256="b" * 64) == 2
    assert invocation.read_bytes() == before


def test_default_trialdev_panel_is_raw_and_alternatives_are_explicit() -> None:
    release = _release()
    ready = SimpleNamespace(
        task_id="td-000000000003",
        case_id="case-a",
        stream=TrialDevStream.EVIDENCE_ANALYSIS,
        data_view="analysis_ready",
    )
    release.participant.tasks += (ready,)
    assert tuple(task.task_id for task in _selected_tasks(release.participant.tasks, ())) == (
        "td-000000000001",
        "td-000000000002",
    )
    assert _selected_tasks(release.participant.tasks, (), data_view="analysis_ready") == (ready,)
    assert len(_selected_tasks(release.participant.tasks, (), data_view="all")) == 3
    assert _selected_tasks(release.participant.tasks, (ready.task_id,)) == (ready,)
    with pytest.raises(ValueError, match="matched no released task"):
        _selected_tasks(release.participant.tasks, (ready.task_id,), data_view="raw")


@pytest.mark.parametrize("stream", ["analysis_configuration", "authored_configuration"])
def test_inconsistent_reporting_context_is_rejected_before_executor_or_provider(tmp_path, monkeypatch, stream):
    """Even preparation rejects internally inconsistent release custody."""
    release = _release()
    release.analysis_configuration = lambda task: None
    release.authored_configuration = lambda task: None

    def invalid(task):
        raise ValueError("replay evidence differs from its committed snapshot")

    setattr(release, stream, invalid)
    monkeypatch.setattr(canonical_trialdev, "load_trialdev_evaluator_release", lambda _: release)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid release reached an execution service")

    monkeypatch.setattr(canonical_trialdev, "resolve_trusted_executor_environment", forbidden)
    monkeypatch.setattr(canonical_trialdev, "get_provider", forbidden)
    args = SimpleNamespace(tasks=None, data_view=None, prepare_only=True)
    assert canonical_trialdev._run_materialized(args, release_root=tmp_path, release_archive_sha256="a" * 64) == 2
    assert not tuple(tmp_path.iterdir())
