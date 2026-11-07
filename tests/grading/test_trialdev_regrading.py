"""Recorded task concurrency preserves regrading custody and failure atomicity."""

from multiprocessing import Barrier, Event, Lock, Value
from pathlib import Path
from unittest.mock import Mock

import pytest
from trialagentbench_verifier.response import TrialDevRunRankingV1
from trialagentbench_verifier.scorer import current_scorer

from trialagentbench.contracts.core.config import ExperimentConditionV1
from trialagentbench.contracts.trialdev.run import (
    TrialDevRunIndexV1,
    TrialDevRunInvocationV1,
    TrialDevRunResultV1,
    TrialDevTaskRunV1,
)
from trialagentbench.io import read_json_model, write_json_model
from trialagentbench.ports import CodeExecutionLimitsV1
from trialagentbench.trialdev import regrade


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("fail", [False, True])
def test_recorded_regrade_concurrency_preserves_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workers: int, fail: bool
) -> None:
    tasks = ("td-000000000001", "td-000000000002")
    invocation = TrialDevRunInvocationV1(
        release_id="release_" + "a" * 64,
        release_archive_sha256="b" * 64,
        release_manifest_sha256="c" * 64,
        scheduled_task_ids=tasks[::-1],
        scheduled_task_streams={task: "evidence_analysis" for task in tasks},
        model="fixture",
        provider="openai",
        experiment_condition=ExperimentConditionV1(
            condition_id="primary", request_replicate_id="request-1", maximum_turns_per_step=3
        ),
        send_temperature=False,
        maximum_completion_tokens=1000,
        maximum_context_characters=50000,
        watchdog_seconds=120,
        request_timeout_seconds=60,
        workers=workers,
        executor_image="sha256:" + "d" * 64,
        executor_limits=CodeExecutionLimitsV1(),
    )
    runs = tuple(
        TrialDevTaskRunV1(
            release_id=invocation.release_id,
            release_manifest_sha256=invocation.release_manifest_sha256,
            task_id=task,
            stream="evidence_analysis",
            execution_status="model_noncompletion",
            error="No numerical response",
            report_relative_paths=(),
            ingress_relative_paths=(),
            grade_relative_paths=(),
        )
        for task in tasks
    )
    index = TrialDevRunIndexV1(
        release_id=invocation.release_id,
        scheduled_task_ids=invocation.scheduled_task_ids,
        model=invocation.model,
        invocation_checksum=invocation.checksum,
        results=tuple(
            TrialDevRunResultV1(
                task_id=run.task_id, stream=run.stream, execution_status=run.execution_status, checksum=run.checksum
            )
            for run in reversed(runs)
        ),
    )
    write_json_model(tmp_path / "invocation.json", invocation)
    write_json_model(tmp_path / "run_index.json", index)
    for run in runs:
        write_json_model(tmp_path / "tasks" / run.task_id / "run.json", run)
    before = (tmp_path / "run_index.json").read_bytes()
    barrier, second_finished, lock = Barrier(2, timeout=10), Event(), Lock()
    active, peak = Value("i", 0), Value("i", 0)

    def execute(*, task_root: Path, **kwargs: object) -> TrialDevTaskRunV1:
        assert kwargs["executor_limits"] == invocation.executor_limits
        assert kwargs["executor_image"] == invocation.executor_image
        with lock:
            active.value += 1
            peak.value = max(peak.value, active.value)
        try:
            barrier.wait()
            if task_root.name == tasks[0]:
                assert second_finished.wait(timeout=10)
            else:
                second_finished.set()
            if fail and task_root.name == tasks[0]:
                raise ValueError("Rejected changed task custody")
            return read_json_model(TrialDevTaskRunV1, task_root / "run.json")
        finally:
            with lock:
                active.value -= 1

    rank = Mock(
        return_value=TrialDevRunRankingV1(
            scorer=current_scorer(),
            release_id=invocation.release_id,
            release_manifest_sha256=invocation.release_manifest_sha256,
            invocation_checksum=invocation.checksum,
            model_id=invocation.model,
            scope="scheduled_panel",
            items=(),
            profile_means=(),
        )
    )
    monkeypatch.setattr(regrade, "regrade_trialdev_task_directory", execute)
    monkeypatch.setattr(regrade, "rank_trialdev_run_v1", rank)
    if fail:
        with pytest.raises(ValueError, match="changed task custody"):
            regrade.regrade_trialdev_run(release=Mock(), run_root=tmp_path)
        rank.assert_not_called()
        assert not (tmp_path / "ranking.json").exists()
    else:
        assert regrade.regrade_trialdev_run(release=Mock(), run_root=tmp_path) == runs
        rank.assert_called_once()
    assert peak.value == 2
    assert active.value == 0
    assert (tmp_path / "run_index.json").read_bytes() == before
