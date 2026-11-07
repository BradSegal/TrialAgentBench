"""Executor teardown faults preserve settled outcomes and interruption signals."""

from __future__ import annotations

import json
import subprocess

import pytest
from execution import test_trialeval_run_resume as R
from execution import test_trialeval_terminal_telemetry as T
from execution.test_trialeval_run_resume import _isolated_executor  # noqa: F401
from trialagentbench_verifier.executor import CodeExecutionResultV1, CodeExecutionSessionTerminated

from trialagentbench.ports import LLMResponse, RuntimeDeadlineExceeded, ToolCall
from trialagentbench.tools.run import trialeval
from trialagentbench.trialeval import agent
from trialagentbench.util.campaign_control import CampaignCapacityStop

TAR = ["docker", "exec", "--workdir", "/", "session", "python", "-I", "-c", "<scratch tar>"]
RM = ["docker", "rm", "--force", "session"]
QUALIFIED = {"tar": TAR, "rm": RM}
UNQUALIFIED = {
    "python": ["python", "analysis.py"],
    "docker_run": ["docker", "run", "--rm", "img"],
    "bare_docker": ["docker"],
    "string": "docker exec session tar",
}


def _terminals(path):
    return [
        json.loads(line)["failure_type"]
        for line in path.read_text().splitlines()
        if json.loads(line)["event_type"] == "step_terminal"
    ]


def _returned_terminals(result):
    return [e["failure_type"] for e in result["events"] if e["event_type"] == "step_terminal"]


def _session_cls(close_cmd, *, execute_error=None):
    class Session(T._Session):
        closes = 0

        def execute_result(self, code, **kwargs):
            if execute_error is not None and "x=1" in code:
                raise execute_error
            return CodeExecutionResultV1(status="success", output="", elapsed_seconds=0.0)

        def close(self):
            type(self).closes += 1
            if close_cmd is not None:
                raise subprocess.TimeoutExpired(close_cmd, timeout=300)

    return Session


class _Submit(T._NoSubmissionProvider):
    def generate_turn(self, **kwargs):
        super().generate_turn(**kwargs)
        return LLMResponse(
            tool_calls=[ToolCall(id="s", name="submit_response", arguments='{"content":"# Report\\n\\nDone."}')]
        )


class _Raise(T._NoSubmissionProvider):
    def __init__(self, error):
        super().__init__()
        self.error = error

    def generate_turn(self, **kwargs):
        super().generate_turn(**kwargs)
        raise self.error


class _CodeFirst(T._NoSubmissionProvider):
    def generate_turn(self, **kwargs):
        super().generate_turn(**kwargs)
        return LLMResponse(
            tool_calls=[
                ToolCall(id="c1", name="execute_code", arguments='{"code":"x=1","language":"python","purpose":"p"}')
            ]
        )


class _Refuse(T._NoSubmissionProvider):
    def generate_turn(self, **kwargs):
        super().generate_turn(**kwargs)
        return LLMResponse(model_outcome="malformed_response")


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


@pytest.fixture
def check(monkeypatch):
    monkeypatch.setattr(
        agent,
        "check_analysis_tool_v1",
        lambda call, **kw: (CodeExecutionResultV1(status="success", output="ok", elapsed_seconds=0.0), kw["session"]),
    )


# --- 1. qualified close timeouts, with and without a published completed report ---------------


@pytest.mark.parametrize("site", sorted(QUALIFIED))
@pytest.mark.parametrize("completed", [True, False])
def test_qualified_close(tmp_path, monkeypatch, check, site, completed):
    Session = _session_cls(QUALIFIED[site])
    monkeypatch.setattr(agent, "DockerCodeSession", lambda session_type=Session, **kw: session_type())
    result = _run(tmp_path, _Submit() if completed else T._NoSubmissionProvider())
    persisted = [json.loads(line) for line in (tmp_path / "TASK1_events.jsonl").read_text().splitlines()]
    assert result["events"] == persisted  # returned trace == append-only trace
    assert Session.closes == 1
    if completed:
        assert (result["status"], result["report"]) == ("success", "# Report\n\nDone.")
        assert _terminals(tmp_path / "TASK1_events.jsonl") == [None]
    else:
        assert (result["status"], result["failure_type"], result["report"]) == (
            "max_turns_reached",
            "turn_limit_no_submission",
            None,
        )
        assert _terminals(tmp_path / "TASK1_events.jsonl") == ["turn_limit_no_submission"]


# --- 2. unqualified close timeouts propagate ------------------------------------------------------


@pytest.mark.parametrize("site", sorted(UNQUALIFIED))
@pytest.mark.parametrize("completed", [True, False])
def test_unqualified_close_propagates(tmp_path, monkeypatch, check, site, completed):
    Session = _session_cls(UNQUALIFIED[site])
    monkeypatch.setattr(agent, "DockerCodeSession", lambda session_type=Session, **kw: session_type())
    with pytest.raises(subprocess.TimeoutExpired):
        _run(tmp_path, _Submit() if completed else T._NoSubmissionProvider())
    assert _terminals(tmp_path / "TASK1_events.jsonl") == ([None] if completed else ["turn_limit_no_submission"])


# --- 3. a model-decided outcome in flight when close() times out --------------------------------
# Teardown cannot convert an established model outcome into an infrastructure retry,
# and cannot swallow a budget stop / operator interrupt.


@pytest.mark.parametrize("site", sorted(QUALIFIED))
@pytest.mark.parametrize(
    "case, expected",
    [
        ("participant_deadline_timeout", "timeout"),
        ("model_noncompletion", "contract_error"),
        ("participant_session_terminated", "participant_session_terminated"),
        ("turn_limit_no_submission", "turn_limit_no_submission"),
    ],
)
def test_model_outcome_survives_close_timeout(tmp_path, monkeypatch, check, caplog, site, case, expected):
    execute_error = (
        CodeExecutionSessionTerminated("participant OOM") if case == "participant_session_terminated" else None
    )
    Session = _session_cls(QUALIFIED[site], execute_error=execute_error)
    monkeypatch.setattr(agent, "DockerCodeSession", lambda session_type=Session, **kw: session_type())
    provider = {
        "participant_deadline_timeout": lambda: _Raise(RuntimeDeadlineExceeded("item budget exhausted")),
        "model_noncompletion": _Refuse,
        "participant_session_terminated": _CodeFirst,
        "turn_limit_no_submission": T._NoSubmissionProvider,
    }[case]()
    # Preserve the original model outcome after cleanup.
    result = _run(tmp_path, provider)
    observed = (result["failure_type"], _terminals(tmp_path / "TASK1_events.jsonl"))
    print("RESULT", json.dumps({"case": case, "site": site, "observed": observed}))
    assert observed == (expected, [expected])
    assert "Qualified executor teardown timeout suppressed" in caplog.text


@pytest.mark.parametrize(
    "case",
    [
        "participant_deadline_timeout",
        "model_noncompletion",
        "participant_session_terminated",
        "turn_limit_no_submission",
    ],
)
def test_model_outcome_control_clean_close(tmp_path, monkeypatch, check, case):
    expected = {
        "participant_deadline_timeout": "timeout",
        "model_noncompletion": "contract_error",
        "participant_session_terminated": "participant_session_terminated",
        "turn_limit_no_submission": "turn_limit_no_submission",
    }[case]
    execute_error = (
        CodeExecutionSessionTerminated("participant OOM") if case == "participant_session_terminated" else None
    )
    Session = _session_cls(None, execute_error=execute_error)
    monkeypatch.setattr(agent, "DockerCodeSession", lambda session_type=Session, **kw: session_type())
    provider = {
        "participant_deadline_timeout": lambda: _Raise(RuntimeDeadlineExceeded("item budget exhausted")),
        "model_noncompletion": _Refuse,
        "participant_session_terminated": _CodeFirst,
        "turn_limit_no_submission": T._NoSubmissionProvider,
    }[case]()
    result = _run(tmp_path, provider)
    assert (result["failure_type"], _terminals(tmp_path / "TASK1_events.jsonl")) == (expected, [expected])


@pytest.mark.parametrize("site", sorted(QUALIFIED))
@pytest.mark.parametrize("stop", ["capacity_stop", "keyboard_interrupt"])
def test_stop_survives_close_timeout(tmp_path, monkeypatch, check, caplog, site, stop):
    """Budget stops and operator interrupts propagate despite qualified teardown faults."""
    error = (
        CampaignCapacityStop("model budget exhausted", campaign_wide=False)
        if stop == "capacity_stop"
        else KeyboardInterrupt()
    )
    Session = _session_cls(QUALIFIED[site])
    monkeypatch.setattr(agent, "DockerCodeSession", lambda session_type=Session, **kw: session_type())
    outcome = None
    try:
        result = _run(tmp_path, _Raise(error))
        outcome = ("returned", result["failure_type"])
    except BaseException as raised:  # noqa: BLE001
        assert raised is error
        outcome = ("raised", type(raised).__name__)
    observed = (outcome, _terminals(tmp_path / "TASK1_events.jsonl"))
    print("RESULT", json.dumps({"stop": stop, "site": site, "observed": observed}))
    assert observed == (("raised", type(error).__name__), ["cancelled"])
    assert "Qualified executor teardown timeout suppressed" in caplog.text


# --- 4. end to end through trialeval.main: retention, idempotent resume, swallowed stop ----------


class _MainProvider:
    model = "fixture/model"
    telemetry_route = "openai"
    calls = 0

    def __init__(self, behaviour):
        self.behaviour = behaviour

    def generate_turn(self, *args, **kwargs):
        type(self).calls += 1
        if self.behaviour == "stop":
            raise CampaignCapacityStop("model budget exhausted", campaign_wide=False)
        return LLMResponse(content="analysis without submission")

    def close(self):
        pass


def _main_setup(tmp_path, monkeypatch, close_cmds):
    closes = []

    class Session(T._Session):
        def close(self):
            closes.append(1)
            if len(closes) <= len(close_cmds) and close_cmds[len(closes) - 1] is not None:
                raise subprocess.TimeoutExpired(close_cmds[len(closes) - 1], timeout=300)

    monkeypatch.setattr(agent, "DockerCodeSession", lambda session_type=Session, **kw: session_type())
    arguments = R._args(R._participant_release(tmp_path), tmp_path / "runs")
    assert trialeval.main([*arguments, "--prepare-only"]) == 0
    native = next((tmp_path / "runs" / "fixture_model").iterdir())
    monkeypatch.setattr(trialeval, "execution_slice", lambda **kwargs: ("TASK1001",))
    return [*arguments, "--resume-run-dir", str(native)], native, closes


def _tree(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.mark.parametrize("site", sorted(QUALIFIED))
def test_main_turn_limit_close_timeout_retained_and_resume_idempotent(tmp_path, monkeypatch, check, site):
    resumed, native, closes = _main_setup(tmp_path, monkeypatch, [QUALIFIED[site]])
    monkeypatch.setattr(trialeval, "get_provider", lambda *a, **k: _MainProvider("none"))
    assert trialeval.main(resumed) == 0
    item = native / "items/TASK1001.json"
    out = json.loads(item.read_text())["agent_output"]
    assert (out["status"], out["failure_type"]) == ("max_turns_reached", "turn_limit_no_submission")
    monkeypatch.setattr(trialeval, "get_provider", lambda *a, **k: (_ for _ in ()).throw(AssertionError("rerun")))
    first = _tree(native)
    for _ in range(3):
        assert trialeval.main(resumed) == 0
        assert _tree(native) == first
    assert closes == [1]


@pytest.mark.parametrize("site", sorted(QUALIFIED))
def test_main_budget_stop_with_close_timeout(tmp_path, monkeypatch, check, site):
    """Budget stops propagate without retaining an infrastructure primary."""
    resumed, native, closes = _main_setup(tmp_path, monkeypatch, [QUALIFIED[site]])
    monkeypatch.setattr(trialeval, "get_provider", lambda *a, **k: _MainProvider("stop"))
    outcome = None
    try:
        outcome = ("returned", trialeval.main(resumed))
    except BaseException as raised:  # noqa: BLE001
        outcome = ("raised", type(raised).__name__)
    item = native / "items/TASK1001.json"
    failure = json.loads(item.read_text())["agent_output"]["failure_type"] if item.exists() else None
    print("RESULT", json.dumps({"site": site, "outcome": outcome, "item_failure": failure}))
    assert outcome == ("raised", "CampaignCapacityStop")
    assert failure != "evaluator_error"


def test_main_budget_stop_control_clean_close(tmp_path, monkeypatch, check):
    resumed, native, closes = _main_setup(tmp_path, monkeypatch, [None])
    monkeypatch.setattr(trialeval, "get_provider", lambda *a, **k: _MainProvider("stop"))
    outcome = None
    try:
        outcome = ("returned", trialeval.main(resumed))
    except BaseException as raised:  # noqa: BLE001
        outcome = ("raised", type(raised).__name__)
    item = native / "items/TASK1001.json"
    failure = json.loads(item.read_text())["agent_output"]["failure_type"] if item.exists() else None
    print("RESULT", json.dumps({"control": True, "outcome": outcome, "item_failure": failure}))
    assert outcome == ("raised", "CampaignCapacityStop")


class _MainDeadline(_MainProvider):
    def generate_turn(self, *args, **kwargs):
        type(self).calls += 1
        raise RuntimeDeadlineExceeded("item budget exhausted")


@pytest.mark.parametrize("site", sorted(QUALIFIED))
@pytest.mark.parametrize("case", ["deadline", "turn_limit"])
def test_main_model_outcome_after_close_timeout_and_resume(tmp_path, monkeypatch, check, site, case):
    """End to end: the retained item after the fault (and one forbidden-provider resume) must carry
    the model's own outcome, never an infrastructure primary."""
    resumed, native, closes = _main_setup(tmp_path, monkeypatch, [QUALIFIED[site]])
    provider = _MainDeadline("x") if case == "deadline" else _MainProvider("none")
    monkeypatch.setattr(trialeval, "get_provider", lambda *a, **k: provider)
    try:
        first = ("returned", trialeval.main(resumed))
    except BaseException as raised:  # noqa: BLE001
        first = ("raised", type(raised).__name__)
    monkeypatch.setattr(trialeval, "get_provider", lambda *a, **k: (_ for _ in ()).throw(AssertionError("rerun")))
    trialeval.main(resumed)
    failure = json.loads((native / "items/TASK1001.json").read_text())["agent_output"]["failure_type"]
    print("RESULT", json.dumps({"e2e": case, "site": site, "first": first, "retained": failure}))
    assert failure == ("timeout" if case == "deadline" else "turn_limit_no_submission")


@pytest.mark.parametrize("close_kind", ["qualified_tar", "unqualified_python"])
def test_provider_error_and_unqualified_close_labels(tmp_path, monkeypatch, check, close_kind):
    """A cleanup fault preserves the original provider/model classification while propagating."""
    from trialagentbench.ports import ProviderRequestError, RetryTelemetry

    cmd = TAR if close_kind == "qualified_tar" else ["python", "analysis.py"]
    rows = []
    for name, error in [
        ("provider_error", ProviderRequestError("transport", retry_telemetry=RetryTelemetry(1, 1, 0))),
        ("deadline", RuntimeDeadlineExceeded("budget")),
    ]:
        d = tmp_path / name
        d.mkdir()
        Session = _session_cls(cmd)
        monkeypatch.setattr(agent, "DockerCodeSession", lambda session_type=Session, **kw: session_type())
        try:
            out = ("returned", _run(d, _Raise(error))["failure_type"])
        except BaseException as raised:  # noqa: BLE001
            out = ("raised", type(raised).__name__)
        rows.append((name, out, _terminals(d / "TASK1_events.jsonl")))
    print("RESULT", json.dumps({"close": close_kind, "rows": rows}))
    assert rows == [
        (
            "provider_error",
            ("raised", "ProviderRequestError" if close_kind == "qualified_tar" else "TimeoutExpired"),
            ["provider_error"],
        ),
        (
            "deadline",
            ("returned", "timeout") if close_kind == "qualified_tar" else ("raised", "TimeoutExpired"),
            ["timeout"],
        ),
    ]


@pytest.mark.parametrize("surface", ["prompt", "schema"])
def test_authored_preflight_failure_is_retained_as_infrastructure(tmp_path, monkeypatch, surface):
    """Invalid authored content aborts before any paid model turn and remains retryable."""
    from trialagentbench.ports.tool_input import HarnessConfigurationError, parse_tool_arguments

    monkeypatch.setattr(agent, "DockerCodeSession", lambda **kw: T._Session())
    provider = T._NoSubmissionProvider()
    if surface == "prompt":
        monkeypatch.setattr(agent, "_build_system_prompt", lambda *args, **kwargs: parse_tool_arguments.__doc__)
    else:

        def invalid_schema(*args, **kwargs):
            raise HarnessConfigurationError("Invalid authored schema")

        monkeypatch.setattr(agent, "_get_tools", invalid_schema)
    result = _run(tmp_path, provider)
    assert result["failure_type"] == "evaluator_error"
    assert _returned_terminals(result) == ["evaluator_error"]
    assert not provider.tools
