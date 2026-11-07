"""Host-control timeouts preserve completed submissions across resume."""

from __future__ import annotations

import json
import subprocess

import pytest
from execution import test_trialeval_run_resume as R
from execution import test_trialeval_terminal_telemetry as T
from execution.test_trialeval_run_resume import _isolated_executor  # noqa: F401
from trialagentbench_verifier.executor import CodeExecutionResultV1

from trialagentbench.ports import LLMResponse, RuntimeDeadlineExceeded, ToolCall
from trialagentbench.tools.run import trialeval
from trialagentbench.trialeval import agent

HOST = ["docker", "exec", "--workdir", "/", "session", "python", "-I", "-c", "<scratch tar>"]


class _Base(T._Session):
    def execute_result(self, code, **kwargs):
        return CodeExecutionResultV1(status="success", output="", elapsed_seconds=0.0)


def _terminals(path):
    return [
        json.loads(line)["failure_type"]
        for line in path.read_text().splitlines()
        if json.loads(line)["event_type"] == "step_terminal"
    ]


class _CodeThenSubmit(T._NoSubmissionProvider):
    def generate_turn(self, **kwargs):
        super().generate_turn(**kwargs)
        if len(self.tools) == 1:
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="c1", name="execute_code", arguments='{"code":"x=1","language":"python","purpose":"p"}'
                    )
                ]
            )
        return LLMResponse(
            tool_calls=[ToolCall(id="s", name="submit_response", arguments='{"content":"# Report\\n\\nDone."}')]
        )


def _run(tmp_path, provider, max_turns=3):
    return agent.run_agent(
        T._item(tmp_path),
        provider,
        max_turns=max_turns,
        analysis_specification="protocol_only",
        verbose=False,
        provider_log_path=tmp_path / "TASK1_provider_responses.jsonl",
        event_log_path=tmp_path / "TASK1_events.jsonl",
    )


def _patch_check(monkeypatch):
    monkeypatch.setattr(
        agent,
        "check_analysis_tool_v1",
        lambda call, **kw: (CodeExecutionResultV1(status="success", output="ok", elapsed_seconds=0.0), kw["session"]),
    )


@pytest.mark.parametrize("cmd, expected", [(HOST, "evaluator_error"), (["python", "analysis.py"], None)])
def test_tool_call_timeout_after_paid_turn(tmp_path, monkeypatch, cmd, expected):
    """Mid-conversation: qualified host timeout -> retained evaluator_error; unqualified -> propagates."""
    _patch_check(monkeypatch)

    class Session(_Base):
        def execute_result(self, code, **kwargs):
            if "x=1" in code:
                raise subprocess.TimeoutExpired(cmd, timeout=1)
            return super().execute_result(code)

    monkeypatch.setattr(agent, "DockerCodeSession", lambda **kw: Session())
    provider = _CodeThenSubmit()
    if expected:
        out = _run(tmp_path, provider)
        assert (out["status"], out["failure_type"]) == ("failed", expected)
        assert _terminals(tmp_path / "TASK1_events.jsonl") == [expected]
    else:
        with pytest.raises(subprocess.TimeoutExpired):
            _run(tmp_path, provider)
        assert _terminals(tmp_path / "TASK1_events.jsonl") == ["runtime_error"]
    assert len(provider.tools) == 1  # model was paid once; no second turn


@pytest.mark.parametrize("cmd", [HOST, ["docker", "rm", "--force", "session"]])
def test_close_host_timeout_preserves_completed_submission(tmp_path, monkeypatch, cmd):
    """Qualified teardown faults preserve the already-published completed report."""
    _patch_check(monkeypatch)

    class Session(_Base):
        def close(self):
            raise subprocess.TimeoutExpired(cmd, timeout=1)

    monkeypatch.setattr(agent, "DockerCodeSession", lambda **kw: Session())
    result = _run(tmp_path, _CodeThenSubmit())
    assert result["status"] == "success"
    assert result["report"] == "# Report\n\nDone."
    assert [event["failure_type"] for event in result["events"] if event["event_type"] == "step_terminal"] == [None]
    assert _terminals(tmp_path / "TASK1_events.jsonl") == [None]  # 'completed' terminal already published


class _MainSubmitProvider:
    """Offline provider for trialeval.main: one paid turn that submits."""

    model = "fixture/model"
    telemetry_route = "openai"

    def __init__(self, *a, **k):
        pass

    def generate_turn(self, *args, **kwargs):
        return LLMResponse(
            tool_calls=[ToolCall(id="s", name="submit_response", arguments='{"content":"# Report\\n\\nDone."}')]
        )

    def close(self):
        pass


def test_close_host_timeout_retains_completed_item_across_resume(tmp_path, monkeypatch):
    """Native persistence and repeat resume retain the completed report without another model call."""
    _patch_check(monkeypatch)
    closes = []

    class Session(_Base):
        def close(self):
            closes.append(1)
            if len(closes) == 1:
                raise subprocess.TimeoutExpired(HOST, timeout=300)

    monkeypatch.setattr(agent, "DockerCodeSession", lambda **kw: Session())
    arguments = R._args(R._participant_release(tmp_path), tmp_path / "runs")
    assert trialeval.main([*arguments, "--prepare-only"]) == 0
    native = next((tmp_path / "runs" / "fixture_model").iterdir())
    monkeypatch.setattr(trialeval, "execution_slice", lambda **kwargs: ("TASK1001",))
    monkeypatch.setattr(trialeval, "get_provider", lambda *a, **k: _MainSubmitProvider())
    resumed = [*arguments, "--resume-run-dir", str(native)]
    assert trialeval.main(resumed) == 0
    item_path = native / "items/TASK1001.json"
    retained = item_path.read_bytes()
    result = json.loads(retained)["agent_output"]
    assert result["status"] == "success"
    assert result["report"] == "# Report\n\nDone."
    monkeypatch.setattr(trialeval, "get_provider", lambda *a, **k: (_ for _ in ()).throw(AssertionError("rerun")))
    assert trialeval.main(resumed) == 0
    assert item_path.read_bytes() == retained
    assert closes == [1]


@pytest.mark.parametrize("cmd", [HOST, ["docker", "rm", "--force", "session"], ["python", "analysis.py"]])
@pytest.mark.parametrize("completed", [False, True])
def test_close_timeout_requires_host_control_origin(tmp_path, monkeypatch, cmd, completed):
    """Completed and turn-limit outcomes survive qualified faults; unqualified timeouts propagate."""
    from trialagentbench.contracts.scoring.trialeval_scores import trialeval_infrastructure_outcome_v1

    _patch_check(monkeypatch)

    class Session(_Base):
        def close(self):
            raise subprocess.TimeoutExpired(cmd, timeout=1)

    monkeypatch.setattr(agent, "DockerCodeSession", lambda **kw: Session())
    provider = _CodeThenSubmit() if completed else T._NoSubmissionProvider()
    if cmd[0] == "docker":
        result = _run(tmp_path, provider)
        if completed:
            assert result["status"] == "success"
            expected = None
        else:
            assert (result["status"], result["failure_type"], result["report"]) == (
                "max_turns_reached",
                "turn_limit_no_submission",
                None,
            )
            assert trialeval_infrastructure_outcome_v1(result["failure_type"]) is None
            expected = "turn_limit_no_submission"
        assert [event["failure_type"] for event in result["events"] if event["event_type"] == "step_terminal"] == [
            expected
        ]
    else:
        with pytest.raises(subprocess.TimeoutExpired):
            _run(tmp_path, provider)
        expected = None if completed else "turn_limit_no_submission"
    assert _terminals(tmp_path / "TASK1_events.jsonl") == [expected]


@pytest.mark.parametrize(
    "error,expected",
    [
        (KeyboardInterrupt(), "cancelled"),
        (RuntimeDeadlineExceeded("task budget"), "timeout"),
        (TimeoutError("unqualified operation"), "runtime_error"),
        (subprocess.TimeoutExpired(["docker", "rm", "--force", "fixture"], 10), "evaluator_error"),
    ],
)
def test_real_agent_exception_terminal(tmp_path, monkeypatch, error, expected):
    """Qualified host-control timeouts retain an infrastructure primary."""

    class Provider(T._NoSubmissionProvider):
        def generate_turn(self, **kwargs):
            raise error

    monkeypatch.setattr(agent, "DockerCodeSession", lambda **kwargs: T._Session())
    if expected in {"timeout", "evaluator_error"}:
        result = _run(tmp_path, Provider())
        assert result["failure_type"] == expected
    else:
        with pytest.raises(type(error)):
            _run(tmp_path, Provider())
    assert _terminals(tmp_path / "TASK1_events.jsonl") == [expected]
