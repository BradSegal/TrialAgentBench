from __future__ import annotations

import json
from pathlib import Path

import pytest
from trialagentbench_test_helpers import (
    minimal_participant_output_contract,
    write_minimal_trialeval_release_dictionaries,
)
from trialagentbench_verifier.numerical import NumericalReportingContext

from trialagentbench.ports import (
    CodeExecutionResultV1,
    LLMResponse,
    ProviderRequestError,
    RetryTelemetry,
    ToolCall,
)
from trialagentbench.trialdev import task_agent
from trialagentbench.trialdev.task_agent import TrialDevTaskAgent
from trialagentbench.trialeval import agent
from trialagentbench.trialeval.conditions import INCOMPLETE_SUBMISSION_NOTE
from trialagentbench.trialeval.schema import BenchmarkItem
from trialagentbench.util.runtime_context import FINAL_TURN_SUBMISSION_REMINDER, RuntimeDeadlineExceeded


@pytest.fixture
def successful_capture(monkeypatch):
    """Isolate replay from message, schema and resource-accounting tests."""

    def check(call, **kwargs):
        return CodeExecutionResultV1(status="success", output="reproduced", elapsed_seconds=0.0), kwargs["session"]

    monkeypatch.setattr(agent, "check_analysis_tool_v1", check)
    monkeypatch.setattr(task_agent, "check_analysis_tool_v1", check)


class _Session:
    image_id = "sha256:" + "0" * 64

    def execute(self, code: str) -> str:
        return ""

    def execute_result(self, code: str) -> CodeExecutionResultV1:
        return CodeExecutionResultV1(status="success", output="", elapsed_seconds=0.0)

    def close(self) -> None:
        return None

    def snapshot_scratch(self) -> Path:
        return Path(".")


class _FailingProvider:
    model = "test"
    telemetry_route = "test"

    def generate_turn(self, **_: object) -> LLMResponse:
        raise RuntimeDeadlineExceeded("Provider request deadline was exhausted before transport.")


class _ProviderRequestFailingProvider:
    model = "test"
    telemetry_route = "test"

    def generate_turn(self, **_: object) -> LLMResponse:
        raise ProviderRequestError(
            "provider rejected request",
            retry_telemetry=RetryTelemetry(
                request_attempts=1,
                transient_failure_count=0,
                backoff_seconds=0.0,
            ),
        )


class _NoSubmissionProvider:
    model = "test"
    telemetry_route = "test"

    def __init__(self) -> None:
        self.timeout_seconds: float | None = None
        self.messages: list[dict[str, object]] = []
        self.tools: list[list[dict[str, object]]] = []
        self.tool_choices: list[object] = []

    def generate_turn(self, **kwargs: object) -> LLMResponse:
        timeout = kwargs.get("timeout_seconds")
        assert timeout is None or isinstance(timeout, float)
        self.timeout_seconds = timeout
        messages = kwargs.get("messages")
        assert isinstance(messages, list)
        self.messages = messages
        tools = kwargs.get("tools")
        assert isinstance(tools, list)
        self.tools.append(tools)
        self.tool_choices.append(kwargs.get("tool_choice"))
        return LLMResponse(content="analysis without submission")


class _ToolThenNoSubmissionProvider(_NoSubmissionProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def generate_turn(self, **kwargs: object) -> LLMResponse:
        self.calls += 1
        super().generate_turn(**kwargs)
        if self.calls == 1:
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="execute",
                        name="execute_code",
                        arguments='{"code":"print(42)"}',
                    )
                ]
            )
        return LLMResponse(content="analysis without submission")


class _InvalidSubmissionProvider(_NoSubmissionProvider):
    def generate_turn(self, **kwargs: object) -> LLMResponse:
        super().generate_turn(**kwargs)
        return LLMResponse(
            tool_calls=[
                ToolCall(
                    id="submit",
                    name="submit_response",
                    arguments='{"task_id":"TASK1"}',
                )
            ]
        )


class _TrialDevReportProvider(_NoSubmissionProvider):
    def generate_turn(self, **kwargs: object) -> LLMResponse:
        super().generate_turn(**kwargs)
        return LLMResponse(
            tool_calls=[
                ToolCall(
                    id="submit-report",
                    name="submit_analysis_report",
                    arguments='{"content":"# Analysis\\n\\nThe evidence remains uncertain."}',
                )
            ]
        )


def _prepare_trialdev_result(workspace: Path) -> None:
    (workspace / "scratch").mkdir(exist_ok=True)
    (workspace / "scratch" / "estimates.csv").write_text(
        "target_id,estimate,lower,upper\neffect,1000000000,999999999,1000000001\n"
    )


@pytest.mark.parametrize("repair", (False, True))
@pytest.mark.parametrize("invalid_kind", ("missing_fields", "unknown_target", "duplicate_targets"))
def test_trialdev_public_syntax_feedback_preserves_budget_and_rejected_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, successful_capture, repair: bool, invalid_kind: str
) -> None:
    """Correct syntax within two turns without exposing or correcting scientific validity."""
    monkeypatch.setattr(task_agent, "DockerCodeSession", lambda **_: _Session())
    workspace = tmp_path / "workspace"

    class Provider(_TrialDevReportProvider):
        calls = 0

        def generate_turn(self, **kwargs: object) -> LLMResponse:
            self.calls += 1
            if self.calls == 2 and repair:
                _prepare_trialdev_result(workspace)
            return super().generate_turn(**kwargs)

    provider = Provider()
    loop = TrialDevTaskAgent(
        reporting_context=NumericalReportingContext(
            checkpoint_id="initial", target_ids=("effect",), evidence_artifacts=()
        ),
        provider=provider,
        workdir=workspace,
        system_prompt="Report numerical estimates and intervals.",
        program_id="PROGRAM1",
        scenario_id="SCENARIO1",
        objective_id="OBJECTIVE1",
        max_turns_per_step=2,
    )
    _prepare_trialdev_result(workspace)
    valid = (workspace / "scratch" / "estimates.csv").read_text()
    invalid = "target_id,estimate\neffect,1\n"
    if invalid_kind == "unknown_target":
        invalid = valid.replace("effect,", "unknown,")
    elif invalid_kind == "duplicate_targets":
        invalid = valid + valid.splitlines(keepends=True)[1]
    (workspace / "scratch" / "estimates.csv").write_text(invalid)
    loop.begin_step(phase_id="phase-1", step_id="step-1")
    loop.append_user_message("Submit your analysis.")
    if repair:
        capture = loop.run_until_report()
        assert (capture.path.parent / "estimates.csv").read_text() == valid
    else:
        with pytest.raises(task_agent.AgentTurnLimitExceeded):
            loop.run_until_report()
    loop.close()
    assert provider.calls == loop.turns_used == 2
    snapshots = list((workspace.parent / "analysis_sources").glob("submission-*/estimates.csv"))
    assert len(snapshots) == 2
    assert invalid in [path.read_text() for path in snapshots]
    assert any("Invalid scratch/estimates.csv" in str(message.get("content")) for message in loop.messages)


class _ManagedTrialEvalSubmissionProvider(_NoSubmissionProvider):
    manages_context = True

    def generate_turn(self, **kwargs: object) -> LLMResponse:
        super().generate_turn(**kwargs)
        return LLMResponse(
            tool_calls=[
                ToolCall(
                    id="submit",
                    name="submit_response",
                    arguments='{"content":"# Analysis\\n\\nThe estimated effect is uncertain."}',
                )
            ]
        )


class _ManagedTrialDevReportProvider(_TrialDevReportProvider):
    manages_context = True


def _item(tmp_path: Path) -> BenchmarkItem:
    visible = tmp_path / "visible"
    data = visible / "data"
    hidden = tmp_path / "hidden"
    data.mkdir(parents=True)
    hidden.mkdir()
    write_minimal_trialeval_release_dictionaries(tmp_path)
    return BenchmarkItem(
        item_id="TASK1",
        task_id="TASK1",
        trial_name="fixture",
        design_tier="undisclosed",
        design_subtype="individual_randomized",
        assumption_tier="undisclosed",
        context_tier="C4",
        data_preparation="raw_domains",
        analysis_specification="protocol_only",
        visible_dir=visible,
        data_dir=data,
        task={},
        submission_contract=minimal_participant_output_contract(
            "TASK1",
            data_preparation="raw_domains",
        ),
        suite_dir=tmp_path,
    )


def test_trialeval_preflight_replaces_session_and_records_execution(tmp_path, monkeypatch):
    original, replacement = _Session(), _Session()
    monkeypatch.setattr(agent, "DockerCodeSession", lambda **_: original)
    checks = []

    def check(call, **kwargs):
        assert kwargs["session"] is (replacement if checks else original)
        assert kwargs["context"].target_ids == ("effect",)
        assert kwargs["workspace"] == tmp_path / "workspace" / "item"
        checks.append(call.id)
        return CodeExecutionResultV1(status="success", output="reproduced", elapsed_seconds=1.2), replacement

    monkeypatch.setattr(agent, "check_analysis_tool_v1", check)

    class Provider(_NoSubmissionProvider):
        def generate_turn(self, **kwargs):
            super().generate_turn(**kwargs)
            if len(self.tools) == 1:
                return LLMResponse(tool_calls=[ToolCall(id="check", name="check_analysis", arguments="{}")])
            return LLMResponse(
                tool_calls=[
                    ToolCall(id="submit", name="submit_response", arguments='{"content":"Analysis complete."}')
                ]
            )

    item = _item(tmp_path)
    item.task["target_ids"] = ["effect"]
    provider = Provider()
    output = agent.run_agent(
        item,
        provider,
        max_turns=3,
        analysis_specification="protocol_only",
        verbose=False,
        item_workspace=tmp_path / "workspace",
        event_log_path=tmp_path / "events.jsonl",
    )
    assert checks == ["check", "submit"]
    assert output["status"] == "success"
    messages = [m for m in provider.messages if m.get("role") == "tool"]
    assert "reproduced" in messages[0]["content"]
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert any(e.get("tool_name") == "check_analysis" and e["event_type"] == "code_execution" for e in events)


@pytest.mark.parametrize("tool_name", ["check_analysis", "submit_response"])
def test_trialeval_replay_watchdog_returns_terminal_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool_name: str
) -> None:
    """A task deadline during either replay path is recorded instead of aborting the scheduler."""
    closed = []

    class Session(_Session):
        def close(self):
            closed.append(True)

    monkeypatch.setattr(agent, "DockerCodeSession", lambda **_: Session())
    captures = []

    def timeout_check(call, **kwargs):
        captures.append(kwargs.get("capture_result", False))
        raise RuntimeDeadlineExceeded("TrialEval item exhausted while blocking on analysis replay check.")

    monkeypatch.setattr(agent, "check_analysis_tool_v1", timeout_check)

    class Provider(_NoSubmissionProvider):
        def generate_turn(self, **kwargs):
            super().generate_turn(**kwargs)
            arguments = "{}" if tool_name == "check_analysis" else '{"content":"Candidate report."}'
            return LLMResponse(tool_calls=[ToolCall(id="replay", name=tool_name, arguments=arguments)])

    event_path = tmp_path / "events.jsonl"
    conversation_path = tmp_path / "conversation.json"
    output = agent.run_agent(
        _item(tmp_path),
        Provider(),
        max_turns=3,
        max_elapsed_seconds=100,
        verbose=False,
        item_workspace=tmp_path / "workspace",
        event_log_path=event_path,
        conversation_log_path=conversation_path,
    )
    assert len(captures) == 1
    assert output["status"] == "failed"
    assert output["failure_type"] == "timeout"
    assert output["report"] is None
    assert output["turns_used"] == 1
    assert closed
    assert output["conversation"] == json.loads(conversation_path.read_text())
    events = [json.loads(line) for line in event_path.read_text().splitlines()]
    terminals = [event for event in events if event["event_type"] == "step_terminal"]
    assert len(terminals) == 1
    assert terminals[0]["terminal_status"] == "failed"
    assert terminals[0]["failure_type"] == "timeout"
    assert output["events"] == events


def test_trialeval_provider_failure_finalizes_request_and_assignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent, "DockerCodeSession", lambda **_: _Session())
    provider_path = tmp_path / "TASK1_provider_responses.jsonl"
    event_path = tmp_path / "TASK1_events.jsonl"

    output = agent.run_agent(
        _item(tmp_path),
        _FailingProvider(),
        max_turns=1,
        analysis_specification="protocol_only",
        verbose=False,
        provider_log_path=provider_path,
        event_log_path=event_path,
    )

    # A time limit ends only this item; the runner records it and continues with siblings.
    assert (output["status"], output["failure_type"], output["report"]) == ("failed", "timeout", None)

    provider_events = [json.loads(line) for line in provider_path.read_text(encoding="utf-8").splitlines()]
    assert [event["status"] for event in provider_events] == ["started", "failed"]
    assert provider_events[-1]["failure_type"] == "timeout"
    runtime_events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    assert [event["event_id"] for event in runtime_events] == [
        f"trialeval:TASK1:{index:06d}" for index in range(len(runtime_events))
    ]
    assert all(event["timestamp"].endswith("Z") for event in runtime_events)
    assert all(event["source_artifact_path"] for event in runtime_events)
    assert all(len(event["source_payload_sha256"]) == 64 for event in runtime_events)
    terminals = [event for event in runtime_events if event["event_type"] == "step_terminal"]
    assert len(terminals) == 1
    assert terminals[0]["terminal_status"] == "failed"
    assert terminals[0]["failure_type"] == "timeout"


def test_provider_request_failure_has_one_cross_stream_classification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent, "DockerCodeSession", lambda **_: _Session())
    monkeypatch.setattr(task_agent, "DockerCodeSession", lambda **_: _Session())
    provider = _ProviderRequestFailingProvider()
    trialeval_provider_path = tmp_path / "trialeval_provider.jsonl"
    trialeval_event_path = tmp_path / "trialeval_events.jsonl"

    with pytest.raises(ProviderRequestError, match="provider rejected request"):
        agent.run_agent(
            _item(tmp_path / "trialeval"),
            provider,
            max_turns=1,
            analysis_specification="protocol_only",
            verbose=False,
            provider_log_path=trialeval_provider_path,
            event_log_path=trialeval_event_path,
        )

    trialdev_provider_path = tmp_path / "trialdev_provider.jsonl"
    trialdev_event_path = tmp_path / "trialdev_events.jsonl"
    trialdev_agent = TrialDevTaskAgent(
        reporting_context=NumericalReportingContext(
            checkpoint_id="initial", target_ids=("effect",), evidence_artifacts=()
        ),
        provider=provider,
        workdir=tmp_path / "trialdev" / "workspace",
        system_prompt="Analyse the available trial evidence.",
        program_id="PROGRAM1",
        scenario_id="SCENARIO1",
        objective_id="OBJECTIVE1",
        max_turns_per_step=1,
        provider_log_path=trialdev_provider_path,
        event_log_path=trialdev_event_path,
    )
    trialdev_agent.begin_step(phase_id="phase-1", step_id="step-1")
    trialdev_agent.append_user_message("Estimate the treatment effect.")
    with pytest.raises(ProviderRequestError, match="provider rejected request"):
        trialdev_agent.run_until_report()
    trialdev_agent.close()

    for provider_path, event_path in (
        (trialeval_provider_path, trialeval_event_path),
        (trialdev_provider_path, trialdev_event_path),
    ):
        provider_events = [json.loads(line) for line in provider_path.read_text(encoding="utf-8").splitlines()]
        terminal_events = [
            json.loads(line)
            for line in event_path.read_text(encoding="utf-8").splitlines()
            if json.loads(line)["event_type"] == "step_terminal"
        ]
        assert provider_events[-1]["failure_type"] == "provider_error"
        assert terminal_events[-1]["failure_type"] == "provider_error"


def test_trialdev_resource_close_does_not_invent_a_completed_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    successful_capture,
) -> None:
    monkeypatch.setattr(task_agent, "DockerCodeSession", lambda **_: _Session())
    event_path = tmp_path / "trialdev_events.jsonl"
    trialdev_agent = TrialDevTaskAgent(
        reporting_context=NumericalReportingContext(
            checkpoint_id="initial", target_ids=("effect",), evidence_artifacts=()
        ),
        provider=_TrialDevReportProvider(),
        workdir=tmp_path / "workspace",
        system_prompt="Analyse the available trial evidence.",
        program_id="PROGRAM1",
        scenario_id="SCENARIO1",
        objective_id="OBJECTIVE1",
        max_turns_per_step=1,
        event_log_path=event_path,
    )
    _prepare_trialdev_result(trialdev_agent.workdir)
    trialdev_agent.begin_step(phase_id="phase-1", step_id="step-1")
    trialdev_agent.append_user_message("Estimate the treatment effect.")

    capture = trialdev_agent.run_until_report()
    trialdev_agent.close()

    assert capture.content.startswith("# Analysis")
    runtime_events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    assert not [event for event in runtime_events if event["event_type"] == "step_terminal"]
    with pytest.raises(RuntimeError, match="prior step has a recorded outcome"):
        trialdev_agent.begin_step(phase_id="phase-2", step_id="step-2")


def test_managed_session_context_reaches_both_benchmark_task_loops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    successful_capture,
) -> None:
    monkeypatch.setattr(agent, "DockerCodeSession", lambda **_: _Session())
    monkeypatch.setattr(task_agent, "DockerCodeSession", lambda **_: _Session())
    trialeval_provider = _ManagedTrialEvalSubmissionProvider()

    result = agent.run_agent(
        _item(tmp_path / "trialeval"),
        trialeval_provider,
        max_turns=1,
        max_context_chars=1,
        analysis_specification="protocol_only",
        verbose=False,
        event_log_path=tmp_path / "trialeval_events.jsonl",
    )

    assert result["status"] == "success"
    assert trialeval_provider.messages[0]["role"] == "system"
    assert any(
        message.get("content") == FINAL_TURN_SUBMISSION_REMINDER + INCOMPLETE_SUBMISSION_NOTE
        for message in trialeval_provider.messages
    )

    trialdev_provider = _ManagedTrialDevReportProvider()
    trialdev_agent = TrialDevTaskAgent(
        reporting_context=NumericalReportingContext(
            checkpoint_id="initial", target_ids=("effect",), evidence_artifacts=()
        ),
        provider=trialdev_provider,
        workdir=tmp_path / "trialdev" / "workspace",
        system_prompt="Analyse the available trial evidence.",
        program_id="PROGRAM1",
        scenario_id="SCENARIO1",
        objective_id="OBJECTIVE1",
        max_turns_per_step=1,
        max_context_chars=1,
    )
    _prepare_trialdev_result(trialdev_agent.workdir)
    trialdev_agent.begin_step(phase_id="phase-1", step_id="step-1")
    trialdev_agent.append_user_message("Estimate the treatment effect.")

    capture = trialdev_agent.run_until_report()
    trialdev_agent.close()

    assert capture.content.startswith("# Analysis")
    assert trialdev_provider.messages[0]["role"] == "system"
    assert any(
        message.get("content") == FINAL_TURN_SUBMISSION_REMINDER + task_agent._INCOMPLETE_SUBMISSION_NOTE
        for message in trialdev_provider.messages
    )


def test_trialeval_executor_initialization_failure_still_finalizes_assignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_session(**_: object) -> _Session:
        raise RuntimeError("executor unavailable")

    monkeypatch.setattr(agent, "DockerCodeSession", fail_session)
    event_path = tmp_path / "TASK1_events.jsonl"

    with pytest.raises(RuntimeError, match="executor unavailable"):
        agent.run_agent(
            _item(tmp_path),
            _FailingProvider(),
            max_turns=1,
            analysis_specification="protocol_only",
            verbose=False,
            event_log_path=event_path,
        )

    runtime_events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    terminals = [event for event in runtime_events if event["event_type"] == "step_terminal"]
    assert len(terminals) == 1
    assert terminals[0]["terminal_status"] == "failed"
    assert terminals[0]["failure_type"] == "runtime_error"


def test_trialeval_turn_limit_is_failed_terminal_and_workspace_is_reusable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent, "DockerCodeSession", lambda **_: _Session())
    workspace = tmp_path / "item-workspace"
    event_path = tmp_path / "events.jsonl"
    provider = _NoSubmissionProvider()
    item = _item(tmp_path)

    output = agent.run_agent(
        item,
        provider,
        max_turns=1,
        analysis_specification="protocol_only",
        verbose=False,
        event_log_path=event_path,
        item_workspace=workspace,
        max_elapsed_seconds=10.0,
    )

    assert output["status"] == "max_turns_reached"
    assert provider.messages[-1] == {
        "role": "user",
        "content": FINAL_TURN_SUBMISSION_REMINDER + INCOMPLETE_SUBMISSION_NOTE,
    }
    assert provider.timeout_seconds is not None
    assert 0.0 < provider.timeout_seconds <= 10.0
    terminals = [
        json.loads(line)
        for line in event_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["event_type"] == "step_terminal"
    ]
    assert terminals == [
        {
            **terminals[0],
            "terminal_status": "failed",
            "failure_type": "turn_limit_no_submission",
        }
    ]
    prompt_events = [
        json.loads(line)
        for line in event_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["event_type"] == "prompt"
    ]
    assert len(prompt_events) == 3  # Task, current agent-time allowance, final-turn reminder.
    assert (workspace / "item").is_dir()
    audit_marker = workspace / "item" / "scratch" / "audit-marker.txt"
    audit_marker.parent.mkdir(parents=True)
    audit_marker.write_text("retained", encoding="utf-8")

    second = agent.run_agent(
        item,
        _NoSubmissionProvider(),
        max_turns=1,
        analysis_specification="protocol_only",
        verbose=False,
        item_workspace=workspace,
    )
    assert second["status"] == "max_turns_reached"
    assert audit_marker.read_text(encoding="utf-8") == "retained"


def test_trialeval_sends_and_records_remaining_time_even_after_context_compaction(tmp_path, monkeypatch):
    from trialagentbench.util import runtime_context

    clock = [1_000.0]
    monkeypatch.setattr(runtime_context.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(agent, "DockerCodeSession", lambda **_: _Session())
    seen = []

    class Provider(_NoSubmissionProvider):
        def generate_turn(self, **kwargs):
            response = super().generate_turn(**kwargs)
            notices = [
                message["content"]
                for message in kwargs["messages"]
                if str(message.get("content", "")).startswith(("[Agent-time budget:", "[Agent time:"))
            ]
            seen.append((notices[-1], kwargs["timeout_seconds"]))
            clock[0] += 10
            return response

    log = tmp_path / "conversation.json"
    output = agent.run_agent(
        _item(tmp_path),
        Provider(),
        max_turns=3,
        max_elapsed_seconds=100,
        max_context_chars=500,
        analysis_specification="protocol_only",
        verbose=False,
        conversation_log_path=log,
        item_workspace=tmp_path / "workspace",
    )
    assert output["status"] == "max_turns_reached"
    assert [timeout for _, timeout in seen] == [100, 90, 80]
    # The full allowance is stated once; later turns carry a one-line remainder.
    assert seen[0][0].startswith("[Agent-time budget: 100 seconds total; 100 seconds remaining.")
    assert [notice for notice, _ in seen[1:]] == [
        "[Agent time: 90 of 100 seconds remaining.]",
        "[Agent time: 80 of 100 seconds remaining.]",
    ]
    logged = json.loads(log.read_text())
    assert [
        message["content"]
        for message in logged
        if str(message.get("content", "")).startswith(("[Agent-time", "[Agent time"))
    ] == [notice for notice, _ in seen]


def test_trialeval_tool_reply_exposes_remaining_turn_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent, "DockerCodeSession", lambda **_: _Session())
    provider = _ToolThenNoSubmissionProvider()

    output = agent.run_agent(
        _item(tmp_path),
        provider,
        max_turns=2,
        analysis_specification="protocol_only",
        verbose=False,
        item_workspace=tmp_path / "workspace",
    )

    assert output["status"] == "max_turns_reached"
    tool_messages = [message for message in provider.messages if message.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert "was not offered on this turn" in str(tool_messages[0]["content"])
    assert str(tool_messages[0]["content"]).endswith("[turn 1/2, 1 left; submission only]")


def test_trialeval_terminal_turn_exposes_only_submission_transports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent, "DockerCodeSession", lambda **_: _Session())
    provider = _NoSubmissionProvider()

    output = agent.run_agent(
        _item(tmp_path),
        provider,
        max_turns=3,
        analysis_specification="protocol_only",
        verbose=False,
        item_workspace=tmp_path / "workspace",
    )

    assert output["status"] == "max_turns_reached"
    assert len(provider.tools) == 3
    assert {tool["function"]["name"] for tool in provider.tools[0]} > {
        "submit_response",
        "submit_response_file",
    }
    assert all(
        [tool["function"]["name"] for tool in request_tools] == ["submit_response", "submit_response_file"]
        for request_tools in provider.tools[1:]
    )
    assert provider.tool_choices == ["auto", "required", "required"]


def test_trialeval_trace_retains_parseable_rejected_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent, "DockerCodeSession", lambda **_: _Session())
    conversation_path = tmp_path / "conversation.json"

    output = agent.run_agent(
        _item(tmp_path),
        _InvalidSubmissionProvider(),
        max_turns=1,
        analysis_specification="protocol_only",
        verbose=False,
        conversation_log_path=conversation_path,
        item_workspace=tmp_path / "workspace",
    )

    assert output["status"] == "max_turns_reached"
    conversation = json.loads(conversation_path.read_text(encoding="utf-8"))
    rejected = [message for message in conversation if message.get("role") == "tool"]
    assert len(rejected) == 1
    assert rejected[0]["args"] == {"task_id": "TASK1"}
    assert "accepts only one string content field" in rejected[0]["output"]
    (assistant,) = [message for message in conversation if message.get("role") == "assistant"]
    assert assistant["message"]["tool_calls"][0]["id"] == rejected[0]["tool_call_id"]


def test_trialeval_runner_uses_explicit_item_specification_when_not_overridden(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct callers must not inherit a locked-SAP default on protocol-only tasks."""

    monkeypatch.setattr(agent, "DockerCodeSession", lambda **_: _Session())
    output = agent.run_agent(
        _item(tmp_path),
        _NoSubmissionProvider(),
        max_turns=1,
        verbose=False,
        item_workspace=tmp_path / "workspace",
    )

    assert output["status"] == "max_turns_reached"
    assert output["condition_provenance"]["analysis_specification"] == "protocol_only"


def test_trialeval_runner_rejects_explicit_specification_mismatch(tmp_path: Path) -> None:
    """An explicit override cannot contradict the immutable participant surface."""

    with pytest.raises(ValueError, match="immutable participant evidence surface"):
        agent.run_agent(
            _item(tmp_path),
            _NoSubmissionProvider(),
            max_turns=1,
            analysis_specification="locked_sap",
            verbose=False,
        )


class _ScratchWriteProvider(_NoSubmissionProvider):
    def generate_turn(self, **kwargs: object) -> LLMResponse:
        super().generate_turn(**kwargs)
        return LLMResponse(
            tool_calls=[
                ToolCall(id="write", name="write_workspace_file", arguments='{"path":"scratch/a.py","content":"x=1"}'),
                ToolCall(id="write-short", name="write_workspace_file", arguments='{"path":"b.py","content":"x=2"}'),
            ]
        )


def test_trialeval_events_record_one_scratch_path_for_either_spelling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The instructed scratch/ path and the bare shorthand both record as scratch/<file>."""
    monkeypatch.setattr(agent, "DockerCodeSession", lambda **_: _Session())
    agent.run_agent(
        _item(tmp_path),
        _ScratchWriteProvider(),
        max_turns=3,  # the last two turns offer only submission tools
        analysis_specification="protocol_only",
        verbose=False,
        event_log_path=tmp_path / "events.jsonl",
        item_workspace=tmp_path / "workspace",
    )
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    accessed = [
        event["file_accessed"]
        for event in events
        if event.get("tool_name") == "write_workspace_file" and event.get("file_accessed")
    ]
    assert set(accessed) == {"scratch/a.py", "scratch/b.py"}


class _MissingReadProvider(_NoSubmissionProvider):
    def generate_turn(self, **kwargs: object) -> LLMResponse:
        super().generate_turn(**kwargs)
        return LLMResponse(
            tool_calls=[ToolCall(id="read", name="read_workspace_file", arguments='{"path":"scratch/none.py"}')]
        )


class _MissingFileSession(_Session):
    def execute_result(self, code: str) -> CodeExecutionResultV1:
        status = "execution_error" if "none.py" in code else "success"
        return CodeExecutionResultV1(status=status, output="No file at scratch/none.py", elapsed_seconds=0.0)


def test_trialeval_read_events_carry_the_read_outcome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent, "DockerCodeSession", lambda **_: _MissingFileSession())
    agent.run_agent(
        _item(tmp_path),
        _MissingReadProvider(),
        max_turns=3,
        analysis_specification="protocol_only",
        verbose=False,
        event_log_path=tmp_path / "events.jsonl",
        item_workspace=tmp_path / "workspace",
    )
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    reads = [event for event in events if event["event_type"] == "file_inspection" and event["status"] == "observed"]
    assert [event["execution_status"] for event in reads] == ["execution_error"]


def test_harness_replay_time_is_not_charged_to_the_trialdev_watchdog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Saved-script checks and the report capture outlast the watchdog without exhausting it."""
    import time

    def slow_check(call, **kwargs):
        time.sleep(0.4)
        return CodeExecutionResultV1(status="success", output="reproduced", elapsed_seconds=0.4), kwargs["session"]

    monkeypatch.setattr(task_agent, "DockerCodeSession", lambda **_: _Session())
    monkeypatch.setattr(task_agent, "check_analysis_tool_v1", slow_check)

    class Provider(_TrialDevReportProvider):
        calls = 0

        def generate_turn(self, **kwargs: object) -> LLMResponse:
            self.calls += 1
            if self.calls <= 2:
                return LLMResponse(
                    tool_calls=[ToolCall(id=f"check-{self.calls}", name="check_analysis", arguments="{}")]
                )
            return super().generate_turn(**kwargs)

    loop = TrialDevTaskAgent(
        reporting_context=NumericalReportingContext(
            checkpoint_id="initial", target_ids=("effect",), evidence_artifacts=()
        ),
        provider=Provider(),
        workdir=tmp_path / "workspace",
        system_prompt="Report numerical estimates and intervals.",
        program_id="PROGRAM1",
        scenario_id="SCENARIO1",
        objective_id="OBJECTIVE1",
        max_turns_per_step=3,
        deadline_monotonic=time.monotonic() + 1.0,
    )
    _prepare_trialdev_result(loop.workdir)
    loop.begin_step(phase_id="phase-1", step_id="step-1")
    loop.append_user_message("Submit your analysis.")
    try:
        capture = loop.run_until_report()
    finally:
        loop.close()
    assert capture.content.startswith("# Analysis")
    remaining = loop.deadline.remaining()
    assert remaining is not None and remaining > 0.5


def test_trialdev_read_events_carry_the_read_outcome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(task_agent, "DockerCodeSession", lambda **_: _MissingFileSession())
    event_path = tmp_path / "trialdev_events.jsonl"
    loop = TrialDevTaskAgent(
        reporting_context=NumericalReportingContext(
            checkpoint_id="initial", target_ids=("effect",), evidence_artifacts=()
        ),
        provider=_MissingReadProvider(),
        workdir=tmp_path / "trialdev" / "workspace",
        system_prompt="Analyse the available trial evidence.",
        program_id="PROGRAM1",
        scenario_id="SCENARIO1",
        objective_id="OBJECTIVE1",
        max_turns_per_step=2,  # the final turn offers only the report tools
        event_log_path=event_path,
    )
    loop.begin_step(phase_id="phase-1", step_id="step-1")
    loop.append_user_message("Estimate the treatment effect.")
    with pytest.raises(task_agent.AgentTurnLimitExceeded):
        loop.run_until_report()
    loop.close()
    events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    reads = [event for event in events if event["event_type"] == "file_inspection" and event["status"] == "observed"]
    assert [event["execution_status"] for event in reads] == ["execution_error"]


def test_trialeval_unqualified_operation_timeout_propagates_without_automatic_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operation's own TimeoutError propagates without proving infrastructure origin."""

    class _OperationTimeoutProvider(_FailingProvider):
        def generate_turn(self, **_: object) -> LLMResponse:
            raise TimeoutError("socket read timed out before the item deadline")

    monkeypatch.setattr(agent, "DockerCodeSession", lambda **_: _Session())
    event_path = tmp_path / "TASK1_events.jsonl"
    with pytest.raises(TimeoutError, match="socket read") as raised:
        agent.run_agent(
            _item(tmp_path),
            _OperationTimeoutProvider(),
            max_turns=1,
            analysis_specification="protocol_only",
            verbose=False,
            provider_log_path=tmp_path / "TASK1_provider_responses.jsonl",
            event_log_path=event_path,
        )
    assert not isinstance(raised.value, RuntimeDeadlineExceeded)
    terminals = [
        json.loads(line)
        for line in event_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["event_type"] == "step_terminal"
    ]
    assert [(event["terminal_status"], event["failure_type"]) for event in terminals] == [("failed", "runtime_error")]


def test_runtime_failure_taxonomy_separates_the_deadline_from_operation_timeouts() -> None:
    import subprocess

    from trialagentbench.util.runtime_failures import common_runtime_failure_type_v1

    assert common_runtime_failure_type_v1(RuntimeDeadlineExceeded("budget")) == "timeout"
    assert common_runtime_failure_type_v1(TimeoutError("operation")) == "runtime_error"
    assert (
        common_runtime_failure_type_v1(subprocess.TimeoutExpired(["docker", "rm", "container"], 10))
        == "evaluator_error"
    )
    assert common_runtime_failure_type_v1(subprocess.TimeoutExpired(["python", "program.py"], 10)) == "runtime_error"


@pytest.mark.parametrize("where", ["execute_code", "check_analysis"])
def test_trialeval_terminated_participant_session_ends_only_the_item(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    where: str,
    termination_code: str = "import os; os._exit(1)",
) -> None:
    """A participant session killed by its own code ends only the item as scored noncompletion.

    An earlier campaign stopped with 45 tasks pending when check_analysis tried to snapshot a terminated
    session; TrialDev already records the same event as noncompletion.
    """
    from trialagentbench_verifier.executor import CodeExecutionSessionTerminated

    class DeadSession(_Session):
        def execute_result(self, code: str, **_: object) -> CodeExecutionResultV1:
            return CodeExecutionResultV1(
                status="session_terminated", output="Code execution session has terminated.", elapsed_seconds=0.0
            )

        def snapshot_scratch(self) -> Path:
            raise CodeExecutionSessionTerminated("Cannot snapshot a terminated code-execution session.")

    partial = tmp_path / "workspace" / "item" / "scratch" / "partial-result.json"

    def dead_session(**_):
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text('{"estimate": 0.1, "lower": -0.1, "upper": 0.3}\n')
        return DeadSession()

    monkeypatch.setattr(agent, "DockerCodeSession", dead_session)

    class Provider(_NoSubmissionProvider):
        def generate_turn(self, **kwargs):
            super().generate_turn(**kwargs)
            arguments = (
                json.dumps({"code": termination_code, "language": "python"}) if where == "execute_code" else "{}"
            )
            return LLMResponse(tool_calls=[ToolCall(id="dead", name=where, arguments=arguments)])

    event_path = tmp_path / "events.jsonl"
    output = agent.run_agent(
        _item(tmp_path),
        Provider(),
        max_turns=3,
        max_elapsed_seconds=100,
        verbose=False,
        item_workspace=tmp_path / "workspace",
        event_log_path=event_path,
        conversation_log_path=tmp_path / "conversation.json",
    )
    assert (output["status"], output["failure_type"], output["report"]) == (
        "failed",
        "participant_session_terminated",
        None,
    )
    assert json.loads(partial.read_bytes())["estimate"] == 0.1
    terminals = [
        json.loads(line)
        for line in event_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["event_type"] == "step_terminal"
    ]
    assert [(event["terminal_status"], event["failure_type"]) for event in terminals] == [
        ("failed", "participant_session_terminated")
    ]


def test_trialdev_compaction_retains_both_final_turn_reminder_and_time_notice(tmp_path, monkeypatch):
    import time

    monkeypatch.setattr(task_agent, "DockerCodeSession", lambda **_: _Session())
    provider = _NoSubmissionProvider()
    loop = TrialDevTaskAgent(
        provider=provider,
        workdir=tmp_path / "workspace",
        system_prompt="Analyse the evidence.",
        program_id="programme",
        scenario_id="case",
        objective_id="analysis",
        max_turns_per_step=1,
        max_context_chars=100,
        deadline_monotonic=time.monotonic() + 100,
        agent_time_budget_seconds=100,
    )
    loop.begin_step(phase_id="analysis", step_id="initial")
    loop.append_user_message("Initial task")
    loop.messages.append({"role": "assistant", "content": "earlier analysis " * 1000})
    try:
        with pytest.raises(task_agent.AgentTurnLimitExceeded):
            loop.run_until_report()
        contents = [message.get("content") for message in provider.messages]
        assert FINAL_TURN_SUBMISSION_REMINDER + task_agent._INCOMPLETE_SUBMISSION_NOTE in contents
        assert str(contents[-1]).startswith("[Agent-time budget: 100 seconds total;")
        assert not any("earlier analysis" in str(content) for content in contents)
    finally:
        loop.close()


@pytest.mark.parametrize("outcome", ["refusal", "malformed_response", "empty_response"])
def test_retained_model_noncompletion_ends_only_its_task_and_preserves_usage(tmp_path, monkeypatch, outcome):
    monkeypatch.setattr(agent, "DockerCodeSession", lambda **_: _Session())

    class Provider(_NoSubmissionProvider):
        calls = 0

        def generate_turn(self, **kwargs):
            self.calls += 1
            return LLMResponse(
                content="retained refusal",
                model_outcome=outcome,
                usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                raw_response={"refusal": "retained refusal"},
            )

    provider = Provider()
    path = tmp_path / "TASK1_provider_responses.jsonl"
    result = agent.run_agent(
        _item(tmp_path),
        provider,
        max_turns=3,
        analysis_specification="protocol_only",
        verbose=False,
        provider_log_path=path,
    )
    assert provider.calls == 1
    assert (result["status"], result["failure_type"], result["report"]) == ("failed", "contract_error", None)
    assert json.loads(path.read_text().splitlines()[-1])["total_tokens"] == 12
    assert result["events"][-1]["terminal_status"] == "failed"


def test_verbose_provider_and_tool_text_cannot_control_terminal_but_logs_remain_raw(tmp_path, monkeypatch, capsys):
    from trialagentbench.util.terminal import terminal_text

    hostile = "text\x1b[31m\r\x00\x07\x7f\x9b\u202e\ud800\n\tend"
    clean = terminal_text(hostile)
    assert clean == "text[31m\n\tend"

    class Session(_Session):
        def execute_result(self, code, *, language="python"):
            return CodeExecutionResultV1(status="success", output=hostile, elapsed_seconds=0)

    monkeypatch.setattr(agent, "DockerCodeSession", lambda **_: Session())
    provider = _NoSubmissionProvider()
    monkeypatch.setattr(provider, "generate_turn", lambda **_: LLMResponse(content=hostile))
    path = tmp_path / "conversation.json"
    agent.run_agent(
        _item(tmp_path),
        provider,
        max_turns=1,
        analysis_specification="protocol_only",
        verbose=True,
        conversation_log_path=path,
    )
    call = ToolCall(id="tool", name="execute_code", arguments=json.dumps({"code": "pass", "purpose": hostile}))
    output, _, _ = agent._handle_tool_call(call, Session(), True)
    assert output == hostile
    monkeypatch.setattr(task_agent, "DockerCodeSession", lambda **_: Session())
    loop = TrialDevTaskAgent(
        provider=provider,
        workdir=tmp_path / "dev",
        system_prompt="analyse",
        program_id="programme",
        scenario_id="case",
        objective_id="analysis",
        verbose=True,
    )
    loop.session = Session()
    try:
        assert loop._dispatch_local_tool(call).output == hostile
    finally:
        loop.close()
    printed = capsys.readouterr().out
    assert clean in printed
    assert all(char not in printed for char in "\x1b\r\x00\x07\x7f\x9b\u202e\ud800")
    assert any(message.get("content") == hostile for message in json.loads(path.read_text()))


def test_trialeval_turn_counter_names_the_last_analysis_turn() -> None:
    assert agent._turn_tag(1, 50) == "\n[turn 1/50, 49 left; analysis tools close after turn 48]"
    assert agent._turn_tag(48, 50) == "\n[turn 48/50, 2 left; analysis tools close after turn 48]"
    assert agent._turn_tag(49, 50) == "\n[turn 49/50, 1 left; submission only]"


@pytest.mark.parametrize("programme", [False, True])
def test_trialdev_prompt_states_the_turn_budget_and_final_turn_tools(tmp_path, monkeypatch, programme) -> None:
    monkeypatch.setattr(task_agent, "DockerCodeSession", lambda **_: _Session())
    loop = TrialDevTaskAgent(
        provider=_NoSubmissionProvider(),
        workdir=tmp_path / "workspace",
        system_prompt="Analyse the evidence.",
        program_id="programme",
        scenario_id="case",
        objective_id="analysis",
        max_turns_per_step=7,
    )
    try:
        if programme:
            loop.authored_programme = object()  # type: ignore[assignment]
        rules = loop._run_rules()
    finally:
        loop.close()
    if programme:
        assert (
            "You have 7 turns for the whole programme; turn 7 offers only stop_development and advance_candidate"
            in rules
        )
        assert "valued as a stop on its current day" in rules
    else:
        assert "You have 7 turns for each report; turn 7 offers only the report submission tools" in rules
        assert "recorded as missing" in rules
    assert "Keep exactly one saved script, scratch/analysis.py or scratch/analysis.R." in rules
    assert rules in loop.system_prompt or programme


@pytest.mark.parametrize(
    "command, retained", [(["docker", "exec", "session", "python"], True), (["python", "analysis.py"], False)]
)
def test_trialeval_executor_timeout_requires_host_control_origin(tmp_path, monkeypatch, command, retained):
    """A known host-control timeout is retained, while an unqualified subprocess error propagates."""
    import subprocess

    closed = []

    class Session(_Session):
        def execute(self, *args, **kwargs):
            raise subprocess.TimeoutExpired(command, timeout=1)

        def close(self):
            closed.append(True)

    def forbidden(**kwargs):
        raise AssertionError("The failed executor setup contacted the model")

    monkeypatch.setattr(agent, "DockerCodeSession", lambda **kwargs: Session())
    provider = _NoSubmissionProvider()
    monkeypatch.setattr(provider, "generate_turn", forbidden)
    event_path = tmp_path / "TASK1_events.jsonl"

    def run():
        return agent.run_agent(
            _item(tmp_path),
            provider,
            max_turns=1,
            analysis_specification="protocol_only",
            verbose=False,
            provider_log_path=tmp_path / "TASK1_provider_responses.jsonl",
            event_log_path=event_path,
        )

    if retained:
        result = run()
        assert result["status"] == "failed" and result["failure_type"] == "evaluator_error"
    else:
        with pytest.raises(subprocess.TimeoutExpired):
            run()
    terminals = [
        json.loads(line)
        for line in event_path.read_text().splitlines()
        if json.loads(line)["event_type"] == "step_terminal"
    ]
    assert [row["failure_type"] for row in terminals] == ["evaluator_error" if retained else "runtime_error"]
    assert closed == [True]
