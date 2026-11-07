"""Campaign replay scoring stores each answer's replays once and reuses them."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from trialagentbench_verifier.execution_contracts import CodeExecutionLimitsV1, ReplayInputArtifactV1
from trialagentbench_verifier.replay_bank import ReplayBankRecordV1, ReplayBankReportV1
from trialagentbench_verifier.replay_scoring import ReplayOutcomeV1, saved_source_sha256

from trialagentbench.tools import finish_campaign_analysis as finish


def _bank(root: Path, *, replays: int = 1) -> str:
    root.mkdir()
    records = [
        ReplayBankRecordV1(
            task_id="banked",
            replay_index=index,
            truth=0.2,
            reference_route_id="r",
            routes={"r": ReplayBankReportV1(estimate=0.2, lower=0.0, upper=0.4)},
        )
        for index in range(1, replays + 1)
    ]
    (root / "records.jsonl").write_text("".join(record.model_dump_json() + "\n" for record in records))
    manifest = {
        "release_id": "release_fixture",
        "replays_per_law": replays,
        "seed_namespace": "fixture",
        "records_sha256": hashlib.sha256((root / "records.jsonl").read_bytes()).hexdigest(),
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()


def _row(task_id: str, source: str) -> SimpleNamespace:
    return SimpleNamespace(
        task_id=task_id,
        replay_program_artifacts=(
            ReplayInputArtifactV1(relative_path="scratch/analysis.py", sha256=source * 64, byte_length=1),
        ),
        target=SimpleNamespace(binding=SimpleNamespace(estimand_id="primary")),
    )


@pytest.fixture(params=["v1", "clarified"])
def reporting_interface(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture(params=[1])
def campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest, reporting_interface: str
) -> dict[str, object]:
    rows: list[SimpleNamespace] = [_row("banked", "a"), _row("unbanked", "b")]
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        finish,
        "load_trial_eval_campaign_grade_source_v1",
        lambda *_, **__: SimpleNamespace(rows=tuple(rows)),
    )
    monkeypatch.setattr(finish, "reproduced_answer_v1", lambda row: (0.2, 0.0, 0.4))
    monkeypatch.setattr(
        finish,
        "read_json_model",
        lambda model, path: SimpleNamespace(
            image_id="sha256:" + "0" * 64,
            limits=CodeExecutionLimitsV1(),
            experiment_condition=SimpleNamespace(reporting_interface=reporting_interface),
        ),
    )

    def checked(workspace: Path, entrypoint: str, *, task_id: str, source_sha256: str) -> None:
        calls.append((task_id, source_sha256))
        assert workspace == tmp_path / "grade" / "logs" / f"{task_id}_workspace" / "item"

    def replay(*, timed_out: bool = True, **arguments: object) -> ReplayOutcomeV1:
        # The estimate identifies the saved source and replay, so stored outcomes show their grouping.
        source = Path(str(arguments["workspace"])).parent.name
        index = int(str(arguments["replay_index"]))
        marker = 0.1 * index + (0.01 if source.startswith("banked") else 0.0)
        return ReplayOutcomeV1(replay_index=index, status="answered", estimate=marker, lower=0.0, upper=1.0)

    def rerun(**arguments: object) -> ReplayOutcomeV1:
        reruns.append(int(str(arguments["replay_index"])))
        return replay(**arguments, timed_out=False)

    def first(**arguments: object) -> ReplayOutcomeV1:
        assert arguments["interface"] == reporting_interface
        if int(str(arguments["replay_index"])) in timeouts:
            return ReplayOutcomeV1(replay_index=int(str(arguments["replay_index"])), status="timeout", error="slow")
        return replay(**arguments)

    timeouts: set[int] = set()
    reruns: list[int] = []
    monkeypatch.setattr(finish, "require_saved_source_v1", checked)
    monkeypatch.setattr(finish, "replay_once_v1", first)
    monkeypatch.setattr(finish, "rerun_timeout_v1", rerun)
    sha = _bank(tmp_path / "bank", replays=request.param)
    plan = SimpleNamespace(
        configuration=SimpleNamespace(operations=SimpleNamespace(executor_capacity=1, grading_workers=1))
    )

    def run(workers: int | None = None) -> object:
        return finish.replay_trialeval_answers(
            tmp_path / "campaign",
            plan,  # type: ignore[arg-type]
            (tmp_path / "grade",),
            bank=tmp_path / "bank",
            bank_manifest_sha256=sha,
            release_id="release_fixture",
            release_manifest_sha256="1" * 64,
            workers=workers,
        )

    return {"rows": rows, "calls": calls, "run": run, "timeouts": timeouts, "reruns": reruns}


def test_finished_answers_are_reused_and_only_new_answers_are_replayed(campaign: dict[str, object]) -> None:
    run, calls, rows = campaign["run"], campaign["calls"], campaign["rows"]
    first = run()  # type: ignore[operator]
    banked = saved_source_sha256(_row("banked", "a").replay_program_artifacts)
    # Only the reproduced answer to a banked task is replayed; the unbanked task is scored on its own dataset.
    assert calls == [("banked", banked)]
    assert [(item.task_id, item.source_sha256) for item in first.answers] == [("banked", banked)]

    assert run().answers == first.answers  # type: ignore[operator]
    assert len(calls) == 1  # re-finishing replays nothing

    rows.append(_row("banked", "c"))  # type: ignore[attr-defined]
    grown = run()  # type: ignore[operator]
    assert len(calls) == 2 and calls[-1][1] == saved_source_sha256(_row("banked", "c").replay_program_artifacts)
    assert len(grown.answers) == 2


def test_a_replacement_script_in_another_retry_run_gets_its_own_missing_bank_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, campaign: dict[str, object]
) -> None:
    """An old primary cache must never stand in for a selected retry's different saved program."""
    previous = campaign["run"]()  # type: ignore[operator]
    (original,) = previous.answers
    replacement = _row("banked", "c")
    retry_grade = tmp_path / "retry-grade"

    def source(*_, grade_roots, **__):
        rows = (replacement,) if grade_roots == (retry_grade,) else tuple(campaign["rows"])
        return SimpleNamespace(rows=rows)

    calls = []

    def check(workspace, entrypoint, *, task_id, source_sha256):
        assert workspace == retry_grade / "logs" / "banked_workspace" / "item"
        assert source_sha256 != original.source_sha256
        calls.append((task_id, source_sha256))

    monkeypatch.setattr(finish, "load_trial_eval_campaign_grade_source_v1", source)
    monkeypatch.setattr(finish, "require_saved_source_v1", check)
    bank_sha = hashlib.sha256((tmp_path / "bank/manifest.json").read_bytes()).hexdigest()

    def run():
        return finish.replay_trialeval_answers(
            tmp_path / "campaign",
            SimpleNamespace(configuration=SimpleNamespace(operations=SimpleNamespace(executor_capacity=1))),
            (tmp_path / "grade", retry_grade),
            bank=tmp_path / "bank",
            bank_manifest_sha256=bank_sha,
            release_id="release_fixture",
            release_manifest_sha256="1" * 64,
        )

    evidence = run()
    assert original in evidence.answers
    assert calls == [("banked", saved_source_sha256(replacement.replay_program_artifacts))]
    assert len(evidence.answers) == 2
    assert run().answers == evidence.answers and len(calls) == 1


def test_a_bank_with_another_digest_stops_before_any_replay(tmp_path: Path, campaign: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="expected digest"):
        finish.replay_trialeval_answers(
            tmp_path / "campaign",
            SimpleNamespace(),  # type: ignore[arg-type]
            (tmp_path / "grade",),
            bank=tmp_path / "bank",
            bank_manifest_sha256="f" * 64,
            release_id="release_fixture",
            release_manifest_sha256="1" * 64,
        )
    assert campaign["calls"] == []


@pytest.mark.parametrize("campaign", [3], indirect=True)
def test_replays_run_in_parallel_and_are_stored_per_answer_in_replay_order(campaign: dict[str, object]) -> None:
    rows = campaign["rows"]
    rows.append(_row("banked", "c"))  # type: ignore[attr-defined]
    evidence = campaign["run"](workers=3)  # type: ignore[operator]
    assert len(evidence.answers) == 2
    for answer in evidence.answers:
        assert [item.replay_index for item in answer.replays] == [1, 2, 3]
        assert [item.estimate for item in answer.replays] == pytest.approx([0.11, 0.21, 0.31])


@pytest.mark.parametrize("campaign", [3], indirect=True)
def test_a_timed_out_replay_is_rerun_once_after_the_parallel_replays(campaign: dict[str, object]) -> None:
    campaign["timeouts"].add(2)  # type: ignore[attr-defined]
    evidence = campaign["run"](workers=1)  # type: ignore[operator]
    assert campaign["reruns"] == [2]
    (answer,) = evidence.answers
    assert [item.status for item in answer.replays] == ["answered"] * 3
    assert [item.estimate for item in answer.replays] == pytest.approx([0.11, 0.21, 0.31])


def test_a_bank_of_another_release_stops_before_any_replay(tmp_path: Path, campaign: dict[str, object]) -> None:
    sha = hashlib.sha256((tmp_path / "bank" / "manifest.json").read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="not the graded release"):
        finish.replay_trialeval_answers(
            tmp_path / "campaign",
            SimpleNamespace(),  # type: ignore[arg-type]
            (tmp_path / "grade",),
            bank=tmp_path / "bank",
            bank_manifest_sha256=sha,
            release_id="release_other",
            release_manifest_sha256="1" * 64,
        )
    assert campaign["calls"] == []
