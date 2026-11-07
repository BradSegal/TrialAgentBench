"""Custody and temporal-evidence boundaries for whole-run development grading."""

from __future__ import annotations

from pathlib import Path

import pytest

from trialagentbench_verifier.development import _Artifact, _checked_artifacts, _Run, _verify_numerical_capture
from trialagentbench_verifier.io import sha256_file


def _run(artifacts: tuple[_Artifact, ...], **changes: object) -> _Run:
    payload = {
        "schema_id": "ariadne.trialdev.task_run/v1",
        "release_id": "control",
        "release_manifest_sha256": "a" * 64,
        "task_id": "td-0123456789ab",
        "stream": "evidence_analysis",
        "execution_status": "model_noncompletion",
        "report_relative_paths": [],
        "ingress_relative_paths": [],
        "grade_relative_paths": [],
        "artifacts": [row.model_dump() for row in artifacts],
        "final_state": None,
        "error": "control interrupted",
        "checksum": "b" * 64,
    }
    return _Run.model_validate({**payload, **changes})


def test_task_artifacts_reject_replacement_duplicate_and_escape(tmp_path: Path) -> None:
    """Declared identities bind actual bytes and cannot alias outside task custody."""
    path = tmp_path / "report.md"
    path.write_text("original")
    item = _Artifact(relative_path=path.name, sha256=sha256_file(path), byte_length=path.stat().st_size)
    assert _checked_artifacts(tmp_path, _run((item,))) == {path.name: item}
    with pytest.raises(ValueError, match="repeats an artifact"):
        _checked_artifacts(tmp_path, _run((item, item)))
    escaping = item.model_copy(update={"relative_path": "../report.md"})
    with pytest.raises(ValueError):
        _checked_artifacts(tmp_path, _run((escaping,)))


@pytest.mark.parametrize("attack", ["absent", "future", "duplicate", "modified"])
def test_accepted_replay_requires_exact_cutoff_data(tmp_path: Path, attack: str) -> None:
    """Even internally rebound custody cannot admit future or substituted data."""
    from trialagentbench_verifier.execution_contracts import ReplayInputArtifactV1
    from trialagentbench_verifier.numerical import NumericalReplay, NumericalReportingContext

    entry = ReplayInputArtifactV1(relative_path="data/observed.csv", sha256="a" * 64, byte_length=4)
    context = NumericalReportingContext(checkpoint_id="initial", target_ids=("effect",), evidence_artifacts=(entry,))
    inputs = () if attack == "absent" else (entry,)
    if attack == "future":
        inputs += (entry.model_copy(update={"relative_path": "data/confirmatory.csv"}),)
    elif attack == "duplicate":
        inputs += (entry,)
    elif attack == "modified":
        inputs = (entry.model_copy(update={"sha256": "f" * 64}),)
    replay = NumericalReplay(
        status="invalid_program",
        entrypoint="",
        checkpoint_id="initial",
        target_ids=context.target_ids,
        evidence_artifacts=inputs,
    )
    with pytest.raises(ValueError, match="differs from its reporting context"):
        _verify_numerical_capture(
            task_root=tmp_path,
            project=None,
            context=context,
            replay=replay,
            artifacts={},
            executor_image="sha256:" + "c" * 64,
        )


def test_ranking_means_keep_streams_views_and_world_denominators_separate() -> None:
    from trialagentbench_verifier.development import ranking_means
    from trialagentbench_verifier.response import TrialDevRankingItemV1

    rows = [
        TrialDevRankingItemV1(
            task_id=f"task-{index}",
            world_id=world,
            stream=stream,
            data_view=view,
            clinical_profile_id="clinical",
            scenario_profile_id="scenario",
            asset_count=3,
            execution_status="completed",
            run_checksum="a" * 64,
            observational_loss=recovery,
            initial_target_statuses={"effect": "reported"},
        )
        for index, (world, stream, view, recovery) in enumerate(
            (
                ("w1", "evidence_analysis", "raw", 0.0),
                ("w1", "evidence_analysis", "raw", 1.0),
                ("w2", "evidence_analysis", "raw", 1.0),
                ("w1", "evidence_analysis", "analysis_ready", 0.2),
                ("w1", "portfolio_management", "raw", 0.4),
            )
        )
    ]
    means = {(row.stream, row.data_view): row for row in ranking_means(rows)}
    raw = means["evidence_analysis", "raw"]
    assert (raw.scheduled_tasks, raw.scheduled_worlds, raw.mean) == (3, 2, 0.75)
    assert (raw.scored_tasks, raw.scored_worlds, raw.observed_mean) == (3, 2, 0.75)
    assert raw.bounded_recovery == pytest.approx(1 / 1.75)
    assert means["evidence_analysis", "analysis_ready"].mean == 0.2
    assert means["portfolio_management", "raw"].mean == 0.4
    authored = rows[-1].model_copy(update={"authored_programme": True, "net_development_value_usd": 2_000_000})
    # The unfinished programme is scored at its incurred costs and still counted as noncompletion.
    interrupted = authored.model_copy(
        update={
            "task_id": "unfinished",
            "world_id": "w2",
            "execution_status": "model_noncompletion",
            "net_development_value_usd": -500_000,
            "observational_loss": 0.0,
        }
    )
    means_by_metric = {row.metric: row for row in ranking_means([authored, interrupted])}
    value = means_by_metric["net_development_value_usd"]
    assert (value.scheduled_tasks, value.scheduled_worlds, value.mean) == (2, 2, 750_000)
    assert (value.scored_tasks, value.scored_worlds, value.unscored_task_counts) == (2, 2, {})
    assert value.model_noncompletion == means_by_metric["observational_loss"].model_noncompletion == 1
    assert means_by_metric["observational_loss"].mean == 0.2
    (empty,) = (
        row
        for row in ranking_means([interrupted.model_copy(update={"observational_loss": None})])
        if row.metric == "observational_loss"
    )
    assert empty.mean is empty.observed_mean is empty.bounded_recovery is None
    assert empty.scored_tasks == empty.scored_worlds == 0


def test_task_discovery_supports_runner_layout_and_rejects_duplicate_copies(tmp_path: Path) -> None:
    from trialagentbench_verifier.io import trialdev_task_run_paths

    direct = tmp_path / "td-0123456789ab/run.json"
    nested = tmp_path / "tasks/td-123456789abc/run.json"
    for path in (direct, nested):
        path.parent.mkdir(parents=True)
        path.write_text("{}")
    assert trialdev_task_run_paths(tmp_path) == {
        direct.parent.name: direct,
        nested.parent.name: nested,
    }
    duplicate = tmp_path / "tasks" / direct.parent.name / "run.json"
    duplicate.parent.mkdir()
    duplicate.write_text("{}")
    with pytest.raises(ValueError, match="duplicate task"):
        trialdev_task_run_paths(tmp_path)


def test_task_discovery_rejects_redirected_directories(tmp_path: Path) -> None:
    from trialagentbench_verifier.io import trialdev_task_run_paths

    source = tmp_path / "source/td-0123456789ab"
    source.mkdir(parents=True)
    (source / "run.json").write_text("{}")
    root = tmp_path / "run"
    root.mkdir()
    (root / source.name).symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="regular confined"):
        trialdev_task_run_paths(root)
