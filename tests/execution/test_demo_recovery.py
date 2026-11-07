"""Process-death recovery through the ordinary disclosed demonstration CLI."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from trialagentbench.io import read_json_model, sha256_file, write_json_model
from trialagentbench.trialdev.resources import TrialDevBudgetLedger
from trialagentbench.trialdev.run import TrialDevProgrammeRun
from trialagentbench.trialdev.task_agent import TrialDevAgentCheckpoint

pytestmark = [pytest.mark.executor, pytest.mark.slow]


def _files(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in root.rglob("*") if path.is_file()}


def _owned_containers(process_id: int) -> tuple[str, ...]:
    """Find only container clients descended from this test's interrupted process."""
    pending = [process_id]
    visited: set[int] = set()
    names = []
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        process = Path(f"/proc/{current}")
        try:
            arguments = (process / "cmdline").read_bytes().decode().split("\0")
            for children in (process / "task").glob("*/children"):
                pending.extend(int(value) for value in children.read_text().split())
        except FileNotFoundError:
            continue
        if arguments and Path(arguments[0]).name == "docker" and "--name" in arguments:
            names.append(arguments[arguments.index("--name") + 1])
    return tuple(sorted(names))


def test_demo_resumes_an_acquired_episode_after_process_death(tmp_path: Path) -> None:
    """SIGKILL at a clean checkpoint preserves evidence, actions, numerical results, time and value."""
    reference = tmp_path / "uninterrupted"
    recovered = tmp_path / "recovered"

    def command(output: Path, *, resume: bool = False) -> list[str]:
        return [
            sys.executable,
            "-m",
            "trialagentbench.cli",
            "demo",
            "run",
            "--output",
            str(output),
            *(["--resume"] if resume else []),
        ]

    def complete(output: Path, *, resume: bool = False) -> None:
        log_path = tmp_path / ("resume.log" if resume else "uninterrupted.log")
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command(output, resume=resume), cwd=tmp_path, stdout=log, stderr=subprocess.STDOUT, timeout=180
            )
        assert result.returncode == 0, log_path.read_text()

    complete(reference)
    log_path = tmp_path / "interrupted.log"
    with log_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen(command(recovered), cwd=tmp_path, stdout=log, stderr=subprocess.STDOUT) as child:
            try:
                deadline = time.monotonic() + 120
                acquired: Path | None = None
                while time.monotonic() < deadline and child.poll() is None:
                    for checkpoint in (recovered / "trialdev").glob("*/private/agent_state.json"):
                        # Freeze before inspecting: killing during charged work must
                        # refuse resume, so this recovery case needs a clean clock.
                        child.send_signal(signal.SIGSTOP)
                        state = read_json_model(TrialDevAgentCheckpoint, checkpoint)
                        context = state.reporting_context
                        if (
                            context is not None
                            and context.checkpoint_id.startswith("evidence/")
                            and state.agent_time is not None
                            and not state.agent_time.charged_operation_in_progress
                        ):
                            acquired = checkpoint
                            break
                        child.send_signal(signal.SIGCONT)
                    if acquired is not None:
                        break
                    time.sleep(0.02)
                assert acquired is not None, log_path.read_text()
                containers = _owned_containers(child.pid)
                assert containers, "The acquired episode must still own a live executor at interruption"
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=10)
            assert child.returncode == -signal.SIGKILL

    # The native interactive owner exits on loss of its host input. The test does
    # not remove containers to manufacture a passing cleanup result.
    deadline = time.monotonic() + 15
    remaining = set(containers)
    while remaining and time.monotonic() < deadline:
        remaining = {
            name
            for name in remaining
            if subprocess.run(["docker", "inspect", name], capture_output=True).returncode == 0
        }
        if remaining:
            time.sleep(0.05)
    assert not remaining, f"Killed episode left owned containers: {sorted(remaining)}"

    task_root = acquired.parent.parent
    checkpoint = read_json_model(TrialDevAgentCheckpoint, acquired)
    committed = read_json_model(TrialDevBudgetLedger, task_root / "private/budget.json")
    assert committed.terminal is None
    assert len(committed.recruitment_commitments) == 1
    assert sum(row.kind == "readout" for row in committed.receipts) == 1
    assert not (task_root / "run.json").exists()
    context = checkpoint.reporting_context
    assert context is not None and context.checkpoint_id.startswith("evidence/")
    assert checkpoint.agent_time is not None and not checkpoint.agent_time.charged_operation_in_progress
    saved_seconds = checkpoint.agent_time.remaining_seconds
    assert saved_seconds is not None and 0 < saved_seconds < 600
    frozen = {
        artifact.relative_path: sha256_file(task_root / "workspace" / artifact.relative_path)
        for artifact in context.evidence_artifacts
    }
    assert frozen == {artifact.relative_path: artifact.sha256 for artifact in context.evidence_artifacts}
    commitment_bytes = _files(task_root / "private/commitments")
    trialeval_bytes = _files(recovered / "trialeval")
    write_json_model(tmp_path / "interrupted-checkpoint.json", checkpoint)
    write_json_model(tmp_path / "committed-budget.json", committed)

    complete(recovered, resume=True)
    resumed_checkpoint = read_json_model(TrialDevAgentCheckpoint, acquired)
    assert resumed_checkpoint.agent_time is not None
    assert 0 < resumed_checkpoint.agent_time.remaining_seconds <= saved_seconds
    assert _files(recovered / "trialeval") == trialeval_bytes
    assert all(sha256_file(task_root / "workspace" / name) == digest for name, digest in frozen.items())
    assert all(_files(task_root / "private/commitments")[name] == digest for name, digest in commitment_bytes.items())
    original = reference / "trialdev" / task_root.name
    expected_ledger = read_json_model(TrialDevBudgetLedger, original / "private/budget.json")
    actual_ledger = read_json_model(TrialDevBudgetLedger, task_root / "private/budget.json")
    assert actual_ledger == expected_ledger
    assert actual_ledger.terminal is not None and actual_ledger.terminal.decision == "stop"
    assert len(actual_ledger.recruitment_commitments) == 1
    assert sum(row.kind == "readout" for row in actual_ledger.receipts) == 1
    for folder in ("private/commitments", "workspace/evidence", "workspace/analysis_ready"):
        assert _files(task_root / folder) == _files(original / folder)
    histories = []
    for root in (original, task_root):
        state = read_json_model(TrialDevAgentCheckpoint, root / "private/agent_state.json")
        identifiers = [call["id"] for message in state.messages for call in message.get("tool_calls", []) or []]
        assert len(identifiers) == len(set(identifiers))
        histories.append(identifiers)
        assert not state.pending_calls
        assert json.loads((root / "run.json").read_text())["execution_status"] == "completed"
    assert histories[0] == histories[1]
    expected = read_json_model(TrialDevProgrammeRun, original / "programme_run.json")
    actual = read_json_model(TrialDevProgrammeRun, task_root / "programme_run.json")
    assert actual.execution_status == expected.execution_status == "completed"
    assert actual.evaluation.value == expected.evaluation.value
    assert actual.evaluation.initial_scores == expected.evaluation.initial_scores
    assert len(actual.evaluation.studies) == len(expected.evaluation.studies) == 1
    for observed, reference_study in zip(actual.evaluation.studies, expected.evaluation.studies, strict=True):
        assert observed.report == reference_study.report
        assert observed.scores == reference_study.scores
        assert observed.replay.reproducible and reference_study.replay.reproducible
        assert observed.replay.estimates == reference_study.replay.estimates
        assert observed.replay.submitted_csv_sha256 == reference_study.replay.submitted_csv_sha256
    assert {
        path.relative_to(task_root).as_posix(): path.read_bytes()
        for path in (task_root / "analysis_sources/cutoffs").rglob("estimates.csv")
    } == {
        path.relative_to(original).as_posix(): path.read_bytes()
        for path in (original / "analysis_sources/cutoffs").rglob("estimates.csv")
    }
