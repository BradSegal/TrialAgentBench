"""Agent-time accounting must survive interruption without buying extra time."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError
from trialagentbench_verifier.numerical import NumericalReportingContext

from trialagentbench.io import read_json_model
from trialagentbench.ports import CodeExecutionResultV1, LLMResponse, ToolCall
from trialagentbench.trialdev import task_agent as owner
from trialagentbench.util import runtime_context
from trialagentbench.util.agent_runtime import assistant_message_v1
from trialagentbench.util.runtime_context import RuntimeDeadline, RuntimeDeadlineExceeded


class Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    value = Clock()
    monkeypatch.setattr(runtime_context.time, "monotonic", lambda: value.now)
    return value


def test_nested_harness_work_and_offline_downtime_do_not_refill_or_spend_budget(clock):
    deadline = RuntimeDeadline.after(100, label="test")
    clock.advance(7)  # Provider request.
    with deadline.harness_work():
        clock.advance(300)
        with deadline.harness_work():
            clock.advance(400)
            assert deadline.snapshot_remaining() == 93
    clock.advance(6)  # Interactive tool.
    saved = deadline.snapshot_remaining()
    assert saved == 87
    clock.advance(8_000)  # No running process.
    resumed = RuntimeDeadline.after(100, label="test")
    resumed.restore_remaining(saved)
    assert resumed.remaining() == 87
    clock.advance(87)
    assert resumed.snapshot_remaining() == 0
    with pytest.raises(RuntimeDeadlineExceeded):
        resumed.remaining()


@pytest.mark.parametrize("seconds", [-1, float("inf"), float("nan")])
def test_saved_time_rejects_nonfinite_or_negative_values(seconds):
    with pytest.raises(ValidationError):
        owner.TrialDevAgentTimeCheckpoint(remaining_seconds=seconds)
    with pytest.raises(ValueError):
        RuntimeDeadline.after(100, label="test").restore_remaining(seconds)


def _arguments(tmp_path, monkeypatch, clock):
    monkeypatch.setattr(owner, "DockerCodeSession", lambda **kwargs: Mock(image_id="image-sha"))
    return dict(
        provider=Mock(),
        workdir=tmp_path / "workspace",
        system_prompt="Analyse the evidence.",
        program_id="programme",
        scenario_id="case",
        objective_id="evidence_analysis",
        max_turns_per_step=5,
        deadline_monotonic=clock.now + 100,
        agent_time_budget_seconds=100,
        reporting_context=NumericalReportingContext(
            checkpoint_id="initial", target_ids=("effect",), evidence_artifacts=()
        ),
    )


def test_native_resume_retains_charged_time_turns_and_model_visible_allowance(tmp_path, monkeypatch, clock):
    arguments = _arguments(tmp_path, monkeypatch, clock)
    agent = owner.TrialDevTaskAgent(**arguments)
    agent.begin_step(phase_id="analysis", step_id="initial")
    agent.append_user_message("Estimate effect.")
    notices = []
    calls = [
        ToolCall(id="code", name="execute_code", arguments='{"language":"python","code":"pass"}'),
        ToolCall(id="check", name="check_analysis", arguments="{}"),
    ]

    def provider_turn(**kwargs):
        notices.append(agent.messages[-1]["content"])
        if kwargs["turn"] == 2:
            clock.advance(7)
            raise OSError("provider interrupted")
        clock.advance(5)
        response = LLMResponse(content=None, tool_calls=calls, usage={})
        return SimpleNamespace(response=response, message=assistant_message_v1(response))

    def execute(call):
        duration = 12 if call.name == "execute_code" else 500
        clock.advance(duration)
        # A checkpoint written inside excluded work must use the paused clock.
        agent._persist_checkpoint()
        return CodeExecutionResultV1(status="success", output="done", elapsed_seconds=duration)

    monkeypatch.setattr(agent, "_provider_turn", provider_turn)
    monkeypatch.setattr(agent, "_execute_checkpointed_tool", execute)
    with pytest.raises(OSError, match="provider interrupted"):
        agent.run_until_report()
    agent.close()
    assert "100 seconds total; 100 seconds remaining" in notices[0]
    assert notices[1] == "[Agent time: 83 of 100 seconds remaining.]"
    assert "Model requests and analysis/workspace tools consume this time" in notices[0]
    assert "previews, study starts, adaptations, waiting, readouts and terminal decisions" in notices[0]
    assert "do not" in notices[0] and "simulated programme days" in notices[0]
    checkpoint = read_json_model(owner.TrialDevAgentCheckpoint, agent._checkpoint_path())
    assert checkpoint.agent_time.remaining_seconds == 76
    assert checkpoint.agent_time.charged_operation_in_progress is False
    assert checkpoint.turns_used == 1 and checkpoint.pending_calls == ()

    clock.advance(8_000)
    resumed = owner.TrialDevTaskAgent(**(arguments | {"deadline_monotonic": clock.now + 100}), resume=True)
    assert resumed.deadline.remaining() == 76
    assert resumed.turns_used == 1

    def fail_again(**kwargs):
        assert kwargs["turn"] == 2
        assert kwargs["timeout_seconds"] == 76
        assert "76 seconds remaining" in resumed.messages[-1]["content"]
        clock.advance(3)
        raise OSError("second interruption")

    monkeypatch.setattr(resumed, "_provider_turn", fail_again)
    with pytest.raises(OSError, match="second interruption"):
        resumed.run_until_report()
    resumed.close()
    assert read_json_model(owner.TrialDevAgentCheckpoint, agent._checkpoint_path()).agent_time.remaining_seconds == 73


@pytest.mark.parametrize("defect", ["missing_agent_time", "unclean", "changed_limit"])
def test_resume_refuses_unknown_consumed_time_or_changed_limit(tmp_path, monkeypatch, clock, defect):
    arguments = _arguments(tmp_path, monkeypatch, clock)
    agent = owner.TrialDevTaskAgent(**arguments)
    agent.begin_step(phase_id="analysis", step_id="initial")
    agent.append_user_message("Estimate effect.")
    agent.close()
    path = agent._checkpoint_path()
    payload = json.loads(path.read_text())
    if defect == "missing_agent_time":
        del payload["agent_time"]
        error = "no saved agent-time budget"
    elif defect == "unclean":
        payload["agent_time"]["charged_operation_in_progress"] = True
        error = "unclean interruption"
    else:
        arguments["agent_time_budget_seconds"] = 200
        error = "execution limits"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=error):
        owner.TrialDevTaskAgent(**arguments, resume=True)


def test_exhausted_checkpoint_cannot_request_another_turn(tmp_path, monkeypatch, clock):
    arguments = _arguments(tmp_path, monkeypatch, clock)
    agent = owner.TrialDevTaskAgent(**arguments)
    agent.begin_step(phase_id="analysis", step_id="initial")
    agent.append_user_message("Estimate effect.")
    clock.advance(101)
    agent.close()
    clock.advance(1_000)
    resumed = owner.TrialDevTaskAgent(**(arguments | {"deadline_monotonic": clock.now + 100}), resume=True)
    provider = Mock(side_effect=AssertionError("expired task must not call provider"))
    monkeypatch.setattr(resumed, "_provider_turn", provider)
    with pytest.raises(RuntimeDeadlineExceeded):
        resumed.run_until_report()
    resumed.close()
    provider.assert_not_called()


def test_checkpoint_and_executor_lifecycle_costs_do_not_change_saved_allowance(tmp_path, monkeypatch, clock):
    arguments = _arguments(tmp_path, monkeypatch, clock)
    original_write = owner.write_json_model

    def slow_write(*args, **kwargs):
        clock.advance(500)
        original_write(*args, **kwargs)

    def slow_session(**kwargs):
        clock.advance(500)
        return Mock(image_id="image-sha", close=lambda: clock.advance(500))

    monkeypatch.setattr(owner, "write_json_model", slow_write)
    monkeypatch.setattr(owner, "DockerCodeSession", slow_session)
    agent = owner.TrialDevTaskAgent(**arguments)
    agent.begin_step(phase_id="analysis", step_id="initial")
    agent.append_user_message("Estimate effect.")
    assert agent.deadline.remaining() == 100
    clock.advance(7)
    agent.close()
    assert agent.deadline.remaining() == 93
    checkpoint = read_json_model(owner.TrialDevAgentCheckpoint, agent._checkpoint_path())
    assert checkpoint.agent_time.remaining_seconds == 93
    clock.advance(8000)
    resumed = owner.TrialDevTaskAgent(**(arguments | {"deadline_monotonic": clock.now + 100}), resume=True)
    assert resumed.deadline.remaining() == 93
    resumed.close()


def test_code_recovery_snapshots_are_outside_the_interactive_tool_watchdog(tmp_path, monkeypatch, clock):
    arguments = _arguments(tmp_path, monkeypatch, clock)
    agent = owner.TrialDevTaskAgent(**arguments)
    agent.begin_step(phase_id="analysis", step_id="initial")
    agent.append_user_message("Estimate effect.")
    (agent.workdir / "scratch").mkdir()
    monkeypatch.setattr(agent.session, "snapshot_scratch", lambda: clock.advance(500))
    observed = []

    def bounded(operation, **kwargs):
        observed.append(agent.deadline.remaining())
        return operation()

    def execute(call):
        clock.advance(7)
        return CodeExecutionResultV1(status="success", output="done", elapsed_seconds=7)

    monkeypatch.setattr(agent.deadline, "run_blocking", bounded)
    monkeypatch.setattr(agent, "_dispatch_local_tool", execute)
    call = ToolCall(id="code", name="execute_code", arguments='{"language":"python","code":"pass"}')
    first = agent._execute_checkpointed_tool(call)
    assert observed == [100]
    assert agent.deadline.remaining() == 93
    assert agent._execute_checkpointed_tool(call) == first
    assert observed == [100]  # Committed results do not run or charge the code again.
    assert agent.deadline.remaining() == 93
    agent.close()


def test_process_death_leaves_a_checkpoint_that_refuses_native_resume(tmp_path, monkeypatch, clock):
    import subprocess
    import sys

    arguments = _arguments(tmp_path, monkeypatch, clock)
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch
from trialagentbench.trialdev.task_agent import TrialDevTaskAgent
from trialagentbench_verifier.numerical import NumericalReportingContext
with patch("trialagentbench.trialdev.task_agent.DockerCodeSession", return_value=Mock(image_id="image-sha")):
    agent = TrialDevTaskAgent(
        provider=Mock(), workdir=Path(sys.argv[1]), system_prompt="Analyse the evidence.",
        program_id="programme", scenario_id="case", objective_id="evidence_analysis", max_turns_per_step=5,
        deadline_monotonic=time.monotonic() + 100, agent_time_budget_seconds=100,
        reporting_context=NumericalReportingContext(checkpoint_id="initial", target_ids=("effect",), evidence_artifacts=()),
    )
    agent.begin_step(phase_id="analysis", step_id="initial")
    agent.append_user_message("Estimate effect.")
    with agent._charged_work():
        os._exit(91)
""",
            str(arguments["workdir"]),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert child.returncode == 91, child.stderr
    with pytest.raises(ValueError, match="unclean interruption"):
        owner.TrialDevTaskAgent(**arguments, resume=True)


@pytest.mark.parametrize("outcome", ["refusal", "malformed_response", "empty_response"])
def test_resume_preserves_a_paid_model_noncompletion_without_another_request(tmp_path, monkeypatch, clock, outcome):
    from trialagentbench.ports import ModelResponseNoncompletion

    arguments = _arguments(tmp_path, monkeypatch, clock)
    provider = Mock(model="fixture", telemetry_route="offline")
    provider.generate_turn.return_value = LLMResponse(content="retained", model_outcome=outcome)
    arguments.update(provider=provider, provider_log_path=tmp_path / "provider_responses.jsonl")
    agent = owner.TrialDevTaskAgent(**arguments)
    agent.begin_step(phase_id="analysis", step_id="initial")
    agent.append_user_message("Estimate effect.")
    with pytest.raises(ModelResponseNoncompletion):
        agent.run_until_report()
    agent.close()
    resumed = owner.TrialDevTaskAgent(**arguments, resume=True)
    try:
        with pytest.raises(ModelResponseNoncompletion, match=outcome):
            resumed.run_until_report()
    finally:
        resumed.close()
    assert provider.generate_turn.call_count == 1


@pytest.mark.parametrize("failed,missing_response_received", [(False, False), (False, True), (True, False)])
def test_resume_refuses_a_received_response_before_its_conversation_checkpoint(
    tmp_path, monkeypatch, clock, failed, missing_response_received
):
    from contextlib import nullcontext

    from trialagentbench.ports import ProviderRequestError, RetryTelemetry

    arguments = _arguments(tmp_path, monkeypatch, clock)
    provider = Mock(model="fixture", telemetry_route="offline")
    provider.generate_turn.return_value = LLMResponse(content="paid response")
    error = ProviderRequestError(
        "paid unusable response",
        retry_telemetry=RetryTelemetry(request_attempts=1, transient_failure_count=0, backoff_seconds=0),
    )
    error.response_received = True
    error.raw_response = {"id": "paid"}
    if failed:
        provider.generate_turn.side_effect = error
    arguments.update(provider=provider, provider_log_path=tmp_path / "provider_responses.jsonl")
    agent = owner.TrialDevTaskAgent(**arguments)
    agent.begin_step(phase_id="analysis", step_id="initial")
    agent.append_user_message("Estimate effect.")
    # Exact disk state after charged time was saved, before the response was
    # committed to conversation/pending calls. Process death here is clean-time
    # recovery, but must still never buy the same primary response again.
    with pytest.raises(ProviderRequestError) if failed else nullcontext():
        with agent._charged_work():
            agent._provider_turn(tools=[], turn=1, timeout_seconds=100)
    agent.close()
    if missing_response_received:
        path = arguments["provider_log_path"]
        events = [json.loads(line) for line in path.read_text().splitlines()]
        for event in events:
            event.pop("response_received", None)
        path.write_text("".join(json.dumps(event) + "\n" for event in events))
    with pytest.raises(ValueError, match="repeating a paid primary"):
        owner.TrialDevTaskAgent(**arguments, resume=True)
    assert provider.generate_turn.call_count == 1


def test_resume_accepts_a_response_committed_to_the_conversation_checkpoint(tmp_path, monkeypatch, clock):
    arguments = _arguments(tmp_path, monkeypatch, clock)
    provider = Mock(model="fixture", telemetry_route="offline")
    provider.generate_turn.return_value = LLMResponse(content="paid response")
    arguments.update(provider=provider, provider_log_path=tmp_path / "provider_responses.jsonl")
    agent = owner.TrialDevTaskAgent(**arguments)
    agent.begin_step(phase_id="analysis", step_id="initial")
    agent.append_user_message("Estimate effect.")
    with agent._charged_work():
        result = agent._provider_turn(tools=[], turn=1, timeout_seconds=100)
    agent.turns_used = 1
    agent.messages.append(result.message)
    agent._persist_checkpoint()
    agent.close()
    resumed = owner.TrialDevTaskAgent(**arguments, resume=True)
    assert resumed.turns_used == 1
    resumed.close()
    assert provider.generate_turn.call_count == 1


def test_provider_outage_checkpoint_resumes_same_turn_without_refunding_prior_work(tmp_path, monkeypatch, clock):
    """A lost transport window is excluded; successful turns and earlier time survive recovery."""
    import httpx
    from openai import APITimeoutError

    from trialagentbench.adapters.llm_providers import _retry
    from trialagentbench.ports import ProviderUnavailableError
    from trialagentbench.util.provider_telemetry import read_provider_terminal_events_v1

    class Provider:
        model = "test/model"
        telemetry_route = "test"

        def __init__(self, succeed_first):
            self.calls = 0
            self.succeed_first = succeed_first
            self.allowances = []

        def generate_turn(self, **kwargs):
            self.calls += 1
            self.allowances.append(kwargs["timeout_seconds"])
            if self.succeed_first and self.calls == 1:
                clock.advance(5)
                return LLMResponse(
                    content="Continue analysing.",
                    usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                )

            def request():
                clock.advance(20)
                raise APITimeoutError(request=httpx.Request("POST", "https://example.test/completion"))

            return _retry(
                request,
                max_attempts=3,
                base_delay=0,
                timeout_seconds=kwargs["timeout_seconds"],
                unavailability_policy="defer_unanswered_v1",
                request_timeout_seconds=20,
            )[0]

    arguments = _arguments(tmp_path, monkeypatch, clock) | {
        "provider": Provider(succeed_first=True),
        "provider_log_path": tmp_path / "provider_responses.jsonl",
        "provider_unavailability_policy": "defer_unanswered_v1",
    }
    agent = owner.TrialDevTaskAgent(**arguments)
    agent.begin_step(phase_id="analysis", step_id="initial")
    agent.append_user_message("Estimate effect.")
    with pytest.raises(ProviderUnavailableError):
        agent.run_until_report()
    agent.close()
    checkpoint = read_json_model(owner.TrialDevAgentCheckpoint, agent._checkpoint_path())
    assert checkpoint.turns_used == 1
    assert checkpoint.agent_time.remaining_seconds == 95
    assert checkpoint.agent_time.charged_operation_in_progress is False
    assert any(message.get("content") == "Continue analysing." for message in checkpoint.messages)
    events = read_provider_terminal_events_v1(arguments["provider_log_path"])
    assert [(event.turn_index, event.status) for event in events] == [(1, "succeeded"), (2, "failed")]
    assert events[-1].scientific_clock_excluded_seconds == 20

    clock.advance(8000)
    resumed_provider = Provider(succeed_first=False)
    resumed = owner.TrialDevTaskAgent(
        **(
            arguments
            | {
                "deadline_monotonic": clock.now + 100,
                "provider": resumed_provider,
            }
        ),
        resume=True,
    )
    assert resumed.deadline.remaining() == 95
    assert resumed.turns_used == 1
    with pytest.raises(ProviderUnavailableError):
        resumed.run_until_report()
    resumed.close()
    assert resumed_provider.allowances == [95]
    assert resumed.turns_used == 1
    checkpoint = read_json_model(owner.TrialDevAgentCheckpoint, resumed._checkpoint_path())
    assert checkpoint.agent_time.remaining_seconds == 95
    events = read_provider_terminal_events_v1(arguments["provider_log_path"])
    assert [(event.turn_index, event.status) for event in events] == [(1, "succeeded"), (2, "failed"), (2, "failed")]
    assert len({event.request_id for event in events}) == 3


def test_resume_rejects_changed_provider_clock_policy(tmp_path, monkeypatch, clock):
    arguments = _arguments(tmp_path, monkeypatch, clock)
    agent = owner.TrialDevTaskAgent(**arguments)
    agent.begin_step(phase_id="analysis", step_id="initial")
    agent.append_user_message("Estimate effect.")
    agent.close()
    with pytest.raises(ValueError, match="execution limits"):
        owner.TrialDevTaskAgent(**(arguments | {"provider_unavailability_policy": "defer_unanswered_v1"}), resume=True)
