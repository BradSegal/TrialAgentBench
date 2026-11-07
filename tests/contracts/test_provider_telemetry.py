from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from trialagentbench.contracts.core.runs import (
    ProviderRequestEventV1,
    ProviderTelemetrySummaryV1,
    RunCoverageV1,
)
from trialagentbench.ports import LLMResponse, LLMResponseMetadata, RetryTelemetry
from trialagentbench.ports.llm_provider import JsonObject, ToolCall
from trialagentbench.util.campaign_control import CampaignCapacityStop
from trialagentbench.util.provider_telemetry import (
    fail_provider_request_v1,
    generate_provider_turn_v1,
    start_provider_request_v1,
    succeed_provider_request_v1,
    summarize_provider_telemetry_v1,
)
from trialagentbench.util.reported_cost import (
    ReportedCostThresholdReached,
    ReportedCostUnavailable,
    RunStopRequested,
)


@pytest.mark.parametrize(
    "error",
    [
        KeyboardInterrupt(),
        SystemExit(),
        RunStopRequested("synthetic operator stop"),
        CampaignCapacityStop("synthetic campaign stop"),
        ReportedCostThresholdReached("synthetic operator budget guard"),
        ReportedCostUnavailable("synthetic cost-custody guard"),
    ],
)
def test_every_local_request_stop_is_cancelled_telemetry(tmp_path, error):
    """All four harness-defined stop classes and native interrupts remain local failures."""
    path = tmp_path / "provider.jsonl"
    with pytest.raises(type(error)):
        generate_provider_turn_v1(
            provider=_Provider(error=error),
            messages=[{"role": "user", "content": "synthetic request"}],
            tools=None,
            temperature=0.0,
            max_tokens=10,
            timeout_seconds=10,
            tool_choice="auto",
            log_path=path,
            benchmark="trialeval",
            unit_id="synthetic-task",
            phase_id="task",
            step_id="analysis",
            turn_index=1,
        )
    terminal = json.loads(path.read_text().splitlines()[-1])
    assert terminal["failure_type"] == "cancelled"
    assert terminal["exception_type"] == type(error).__name__
    assert not terminal["response_received"]


@pytest.mark.parametrize(
    "error",
    [
        *(RunStopRequested(f"synthetic runner stop {index}") for index in range(5)),
        CampaignCapacityStop("synthetic operator pause"),
        CampaignCapacityStop("synthetic dispatch cancellation"),
        CampaignCapacityStop("synthetic backup health guard stop"),
        CampaignCapacityStop("synthetic custody health guard stop"),
    ],
)
def test_synthetic_collaborator_stop_inventory_is_infrastructure(error):
    """Represent five runner and four capacity stops without retaining private campaign content."""
    from types import SimpleNamespace

    from trialagentbench_verifier.campaign import _attempt_outcome

    from trialagentbench.contracts.scoring.trialeval_scores import trialeval_observed_attempt_outcome_v1
    from trialagentbench.util.provider_telemetry import provider_failure_type_v1

    assert provider_failure_type_v1(error) == "cancelled"
    missing = SimpleNamespace(status="missing")
    assert (
        trialeval_observed_attempt_outcome_v1(
            agent_status="failed",
            failure_type="cancelled",
            outcome=SimpleNamespace(completion=missing),
        )
        == "evaluator_error"
    )
    assert (
        _attempt_outcome(
            agent_status="failed",
            failure_type="cancelled",
            completion=missing,
            reproducibility=missing,
        )
        == "evaluator_error"
    )


class _Provider:
    model = "test/model"
    telemetry_route = "test"

    def __init__(self, error: Exception | None = None, *, response: LLMResponse | None = None) -> None:
        self.error = error
        self.response = response
        self.call_count = 0

    def generate_turn(
        self,
        messages: Sequence[JsonObject],
        tools: Sequence[JsonObject] | None = None,
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float | None = None,
        tool_choice: Literal["auto", "required"] = "auto",
    ) -> LLMResponse:
        self.call_count += 1
        if self.error is not None:
            raise self.error
        return self.response if self.response is not None else _response()


@pytest.mark.parametrize("benchmark", ["trialeval", "trialdev"])
@pytest.mark.parametrize("surface", ["prompt", "tool"])
def test_documentation_content_is_infrastructure_before_provider_dispatch(benchmark, surface):
    """A preflight error cannot be charged to a participant or dispatched to a provider."""
    from trialagentbench.ports.tool_input import HarnessConfigurationError, parse_tool_arguments
    from trialagentbench.util.agent_runtime import run_agent_turn_v1
    from trialagentbench.util.runtime_failures import common_runtime_failure_type_v1

    text = "Leading context: " + parse_tool_arguments.__doc__ + " More context."
    provider = _Provider()
    messages = [
        {"role": "system", "content": text if surface == "prompt" else "Explicit request."},
        {"role": "user", "content": "Explicit task."},
    ]
    tools = [{"function": {"name": "sample", "description": text}}] if surface == "tool" else []
    with pytest.raises(HarnessConfigurationError) as failure:
        run_agent_turn_v1(
            provider=provider,
            messages=messages,
            session=object(),
            active_prompt_index=1,
            max_context_chars=100000,
            tools=tools,
            temperature=0.0,
            max_tokens=10,
            timeout_seconds=10,
            tool_choice="auto",
            provider_log_path=None,
            benchmark=benchmark,
            unit_id="synthetic-task",
            phase_id="task",
            step_id="analysis",
            turn_index=1,
        )
    assert common_runtime_failure_type_v1(failure.value) == "evaluator_error"
    assert provider.call_count == 0


@pytest.mark.parametrize("benchmark", ["trialeval", "trialdev"])
@pytest.mark.parametrize("surface", ["assistant", "arguments", "tool", "reminder"])
def test_episode_content_never_changes_provider_outcome(benchmark, surface, caplog):
    """Agent text is ignored; matching episode replies only record a provenance warning."""
    from trialagentbench.ports.tool_input import parse_tool_arguments
    from trialagentbench.util.agent_runtime import run_agent_turn_v1

    text = parse_tool_arguments.__doc__
    messages = [
        {"role": "system", "content": "Explicit request."},
        {"role": "user", "content": "Explicit task."},
        {"role": "assistant", "content": "Working."},
    ]
    if surface == "assistant":
        messages[-1]["content"] = text
    elif surface == "arguments":
        messages[-1]["tool_calls"] = [{"id": "call-1", "function": {"name": "sample", "arguments": text}}]
        messages.append({"role": "tool", "tool_call_id": "call-1", "content": "Explicit reply."})
    elif surface == "tool":
        messages[-1]["tool_calls"] = [{"id": "call-1", "function": {"name": "sample", "arguments": "{}"}}]
        messages.append({"role": "tool", "tool_call_id": "call-1", "content": text})
    else:
        messages.append({"role": "user", "content": text})
    provider = _Provider()
    with caplog.at_level("WARNING"):
        result = run_agent_turn_v1(
            provider=provider,
            messages=messages,
            session=object(),
            active_prompt_index=1,
            max_context_chars=100000,
            tools=[],
            temperature=0.0,
            max_tokens=10,
            timeout_seconds=10,
            tool_choice="auto",
            provider_log_path=None,
            benchmark=benchmark,
            unit_id="synthetic-task",
            phase_id="task",
            step_id="analysis",
            turn_index=2,
        )
    assert provider.call_count == 1
    assert result.response == _response()
    assert ("content_sha256=" in caplog.text) == (surface in {"tool", "reminder"})


def _response() -> LLMResponse:
    return LLMResponse(
        content="ok",
        usage={"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        metadata=LLMResponseMetadata(
            response_id="response-1",
            returned_model="deepseek/deepseek-v4-flash",
            upstream_provider="DeepSeek",
            finish_reason="stop",
            created_unix=123,
            reported_cost_usd=0.001,
            request_attempts=3,
            transient_failure_count=2,
            backoff_seconds=6.0,
        ),
    )


def _start(tmp_path: Path):
    return start_provider_request_v1(
        path=tmp_path / "logs" / "TASK1_provider_responses.jsonl",
        benchmark="trialeval",
        unit_id="TASK1",
        phase_id="task",
        step_id="analysis",
        turn_index=1,
        requested_model="deepseek/deepseek-v4-flash",
        provider_route="openrouter:DeepSeek",
    )


def _coverage() -> RunCoverageV1:
    return RunCoverageV1(
        run_identity_sha256="r" * 64,
        schedule_sha256="s" * 64,
        unit_ids=("TASK1",),
        completed_unit_ids=("TASK1",),
    )


def test_resumed_turn_retains_cancelled_and_successful_requests(tmp_path: Path) -> None:
    cancelled = _start(tmp_path)
    fail_provider_request_v1(cancelled, elapsed_seconds=0.1, failure_type="cancelled")
    resumed = _start(tmp_path)
    succeed_provider_request_v1(resumed, elapsed_seconds=0.5, response=_response())

    assert cancelled.request_id != resumed.request_id
    summary = summarize_provider_telemetry_v1(run_root=tmp_path, coverage=_coverage())
    assert summary.failed_request_count == 1
    assert summary.response_count == 1


def test_shared_provider_turn_records_one_complete_lifecycle(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "TASK1_provider_responses.jsonl"

    result = generate_provider_turn_v1(
        provider=_Provider(),
        messages=({"role": "user", "content": "analyse"},),
        tools=None,
        temperature=0.0,
        max_tokens=100,
        timeout_seconds=30.0,
        tool_choice="auto",
        log_path=log_path,
        benchmark="trialeval",
        unit_id="TASK1",
        phase_id="task",
        step_id="analysis",
        turn_index=1,
    )

    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert result.response.content == "ok"
    assert result.elapsed_seconds >= 0.0
    assert [event["status"] for event in events] == ["started", "succeeded"]
    assert events[0]["context_management"] == "local_complete"
    assert events[0]["source_context_messages"] == 1
    assert events[0]["transmitted_context_messages"] == 1
    assert events[0]["requested_completion_tokens"] == 100
    assert events[1]["completion_limit_reached"] is False
    assert events[0]["provider_capability"] is None


def test_raw_provider_payload_is_retained_without_changing_scored_telemetry(tmp_path: Path) -> None:
    handle = _start(tmp_path)
    response = _response()
    response.raw_response = {
        "id": "response-1",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {"reasoning_details": [{"type": "reasoning.text", "text": "Returned text."}]},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3, "completion_tokens_details": {"reasoning_tokens": 2}},
    }
    succeed_provider_request_v1(handle, elapsed_seconds=1.5, response=response)
    payload_path = tmp_path / "logs" / "TASK1_provider_payloads.raw.jsonl"
    payload = json.loads(payload_path.read_text())
    assert payload["request_id"] == handle.request_id
    assert payload["elapsed_seconds"] == 1.5
    assert payload["response"] == response.raw_response
    summary = summarize_provider_telemetry_v1(run_root=tmp_path, coverage=_coverage())
    assert summary.response_count == 1
    assert summary.total_tokens == 13


def test_reasoning_only_failure_retains_returned_payload_for_diagnosis(tmp_path: Path) -> None:
    from trialagentbench.adapters.llm_providers import TransientProviderResponseError

    handle = _start(tmp_path)
    payload = {"choices": [{"finish_reason": "length", "message": {"content": None, "reasoning": "Returned text."}}]}
    error = TransientProviderResponseError("No public answer", raw_response=payload)
    fail_provider_request_v1(handle, elapsed_seconds=2, failure_type="provider_error", error=error)
    saved = json.loads((tmp_path / "logs" / "TASK1_provider_payloads.raw.jsonl").read_text())
    assert saved["response"] == payload
    events = [json.loads(line) for line in handle.path.read_text().splitlines()]
    assert events[-1]["status"] == "failed"


@pytest.mark.parametrize("requested", (100, 101))
def test_source_bound_provider_capacity_is_recorded_and_completion_enforced(tmp_path: Path, requested: int) -> None:
    """Character accounting cannot substitute for a declared token capacity."""
    from datetime import UTC, datetime

    from trialagentbench.contracts.core.config import ProviderModelCapabilityV1

    capability = ProviderModelCapabilityV1(
        provider_transport="openai",
        model_id="test/model",
        context_window_tokens=4096,
        maximum_completion_tokens=100,
        source_url="https://example.test/model-capacity",
        source_retrieved_utc=datetime(2026, 9, 7, tzinfo=UTC),
        source_payload_sha256="a" * 64,
    )
    log = tmp_path / "provider.jsonl"

    def call() -> None:
        generate_provider_turn_v1(
            provider=_Provider(),
            messages=({"role": "user", "content": "analyse"},),
            tools=None,
            temperature=0,
            max_tokens=requested,
            timeout_seconds=30,
            tool_choice="auto",
            log_path=log,
            benchmark="trialeval",
            unit_id="task",
            phase_id="analysis",
            step_id="1",
            turn_index=1,
            provider_capability=capability,
        )

    if requested == 100:
        call()
    else:
        with pytest.raises(ValueError, match="exceeds the source-bound"):
            call()
    events = [ProviderRequestEventV1.model_validate_json(line) for line in log.read_text().splitlines()]
    assert [e.status for e in events] == ["started", "succeeded" if requested == 100 else "failed"]
    assert all(e.provider_capability == capability for e in events)
    assert events[0].transmitted_context_characters != capability.context_window_tokens


def test_shared_provider_turn_records_timeout_and_propagates(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "TASK1_provider_responses.jsonl"

    with pytest.raises(TimeoutError, match="deadline"):
        generate_provider_turn_v1(
            provider=_Provider(TimeoutError("deadline")),
            messages=({"role": "user", "content": "analyse"},),
            tools=None,
            temperature=0.0,
            max_tokens=100,
            timeout_seconds=30.0,
            tool_choice="auto",
            log_path=log_path,
            benchmark="trialdev",
            unit_id="TASK1",
            phase_id="portfolio_management",
            step_id="decision-001",
            turn_index=1,
        )

    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [event["status"] for event in events] == ["started", "failed"]
    assert events[-1]["failure_type"] == "timeout"


def test_provider_request_is_persisted_before_terminal_success_and_aggregates(tmp_path: Path) -> None:
    handle = _start(tmp_path)
    path = handle.path
    started = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [event["status"] for event in started] == ["started"]

    succeed_provider_request_v1(handle, elapsed_seconds=0.5, response=_response())
    summary = summarize_provider_telemetry_v1(run_root=tmp_path, coverage=_coverage())

    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [event["status"] for event in events] == ["started", "succeeded"]
    assert events[0]["request_id"] == events[1]["request_id"]
    assert events[1]["usage_status"] == "reported"
    assert summary.response_count == 1
    assert summary.total_tokens == 13
    assert summary.reported_cost_usd == pytest.approx(0.001)
    assert summary.upstream_providers == ["DeepSeek"]
    assert summary.source_files == ["logs/TASK1_provider_responses.jsonl"]
    assert summary.request_attempt_count == 3
    assert summary.transient_failure_count == 2
    assert summary.backoff_seconds == 6.0


def test_provider_telemetry_separates_compaction_usage(tmp_path: Path) -> None:
    handle = _start(tmp_path)
    response = _response()
    response.usage = {
        "prompt_tokens": 20,
        "completion_tokens": 7,
        "total_tokens": 27,
        "compaction_requests": 1,
        "compaction_input_tokens": 10,
        "compaction_output_tokens": 4,
    }

    event = succeed_provider_request_v1(handle, elapsed_seconds=0.5, response=response)
    summary = summarize_provider_telemetry_v1(run_root=tmp_path, coverage=_coverage())

    assert event.compaction_requests == 1
    assert event.compaction_input_tokens == 10
    assert event.compaction_output_tokens == 4
    assert summary.compaction_requests == 1
    assert summary.compaction_input_tokens == 10
    assert summary.compaction_output_tokens == 4


def test_provider_telemetry_records_archived_context_and_completion_limit(tmp_path: Path) -> None:
    response = _response()
    response.metadata = LLMResponseMetadata(
        response_id="response-1",
        returned_model="deepseek/deepseek-v4-flash",
        upstream_provider="DeepSeek",
        finish_reason="incomplete:max_output_tokens",
    )
    handle = start_provider_request_v1(
        path=tmp_path / "logs" / "TASK1_provider_responses.jsonl",
        benchmark="trialeval",
        unit_id="TASK1",
        phase_id="task",
        step_id="analysis",
        turn_index=1,
        requested_model="model",
        provider_route="route",
        context_management="local_archived",
        source_context_messages=8,
        transmitted_context_messages=5,
        source_context_characters=10_000,
        transmitted_context_characters=4_000,
        requested_completion_tokens=512,
    )

    terminal = succeed_provider_request_v1(handle, elapsed_seconds=0.5, response=response)
    summary = summarize_provider_telemetry_v1(run_root=tmp_path, coverage=_coverage())

    assert terminal.completion_limit_reached is True
    assert terminal.context_management == "local_archived"
    assert summary.locally_archived_request_count == 1
    assert summary.completion_limit_reached_count == 1
    assert summary.source_context_characters == 10_000
    assert summary.transmitted_context_characters == 4_000
    assert summary.requested_completion_tokens == 512


def test_provider_summary_includes_decision_boundary_resume_logs(tmp_path: Path) -> None:
    handle = start_provider_request_v1(
        path=tmp_path / "tasks" / "TASK1" / "provider_responses.resume-002.jsonl",
        benchmark="trialdev",
        unit_id="TASK1",
        phase_id="portfolio_management",
        step_id="decision-002",
        turn_index=1,
        requested_model="analysis-model",
        provider_route="openai",
    )
    succeed_provider_request_v1(handle, elapsed_seconds=0.5, response=_response())

    summary = summarize_provider_telemetry_v1(run_root=tmp_path, coverage=_coverage())

    assert summary.response_count == 1
    assert summary.reported_cost_usd == pytest.approx(0.001)
    assert summary.source_files == ["tasks/TASK1/provider_responses.resume-002.jsonl"]


def test_provider_request_finalizes_typed_failure_without_usage(tmp_path: Path) -> None:
    handle = _start(tmp_path)
    error = TimeoutError("provider deadline")
    error.status_code = 504  # type: ignore[attr-defined]
    fail_provider_request_v1(handle, elapsed_seconds=0.25, failure_type="timeout", error=error)

    summary = summarize_provider_telemetry_v1(run_root=tmp_path, coverage=_coverage())
    terminal = ProviderRequestEventV1.model_validate_json(handle.path.read_text(encoding="utf-8").splitlines()[-1])

    assert terminal.status == "failed"
    assert terminal.failure_type == "timeout"
    assert terminal.exception_type == "TimeoutError"
    assert terminal.http_status_code == 504
    assert terminal.usage_status == "not_applicable"
    assert summary.response_count == 0
    assert summary.failed_request_count == 1
    assert summary.failure_type_counts == {"timeout": 1, "provider_error": 0, "cancelled": 0}
    assert summary.request_attempt_count == 1


def test_exhausted_retry_failure_retains_actual_retry_observations(tmp_path: Path) -> None:
    handle = _start(tmp_path)
    error = TimeoutError("retry deadline")
    error.retry_telemetry = RetryTelemetry(  # type: ignore[attr-defined]
        request_attempts=4,
        transient_failure_count=4,
        backoff_seconds=7.5,
    )
    fail_provider_request_v1(
        handle,
        elapsed_seconds=8.0,
        failure_type="timeout",
        error=error,
    )

    summary = summarize_provider_telemetry_v1(run_root=tmp_path, coverage=_coverage())

    assert summary.request_attempt_count == 4
    assert summary.transient_failure_count == 4
    assert summary.backoff_seconds == 7.5


def test_provider_summary_excludes_archived_failed_attempt_logs(tmp_path: Path) -> None:
    authoritative = _start(tmp_path)
    succeed_provider_request_v1(authoritative, elapsed_seconds=0.5, response=_response())
    archived = start_provider_request_v1(
        path=tmp_path / "archived_failed_attempts" / "TASK1_provider_responses.jsonl",
        benchmark="trialeval",
        unit_id="TASK1",
        phase_id="task",
        step_id="analysis",
        turn_index=1,
        requested_model="deepseek/deepseek-v4-flash",
        provider_route="openrouter:DeepSeek",
    )
    fail_provider_request_v1(archived, elapsed_seconds=99.0, failure_type="provider_error")

    summary = summarize_provider_telemetry_v1(run_root=tmp_path, coverage=_coverage())

    assert summary.response_count == 1
    assert summary.elapsed_seconds == 0.5
    assert summary.source_files == ["logs/TASK1_provider_responses.jsonl"]
    assert summary.archived_request_count == 1
    assert summary.archived_failed_request_count == 1
    assert summary.archived_failure_type_counts == {
        "timeout": 0,
        "provider_error": 1,
        "cancelled": 0,
    }
    assert summary.archived_elapsed_seconds == 99.0
    assert summary.archived_source_files == ["archived_failed_attempts/TASK1_provider_responses.jsonl"]


def test_provider_telemetry_rejects_duplicate_terminal_id_across_files(tmp_path: Path) -> None:
    handle = _start(tmp_path)
    terminal = fail_provider_request_v1(handle, elapsed_seconds=0.1, failure_type="provider_error")
    duplicate_path = tmp_path / "other" / "OTHER_provider_responses.jsonl"
    duplicate_path.parent.mkdir(parents=True)
    duplicate_path.write_text(terminal.model_dump_json() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate provider request failed event"):
        summarize_provider_telemetry_v1(run_root=tmp_path, coverage=_coverage())


def test_provider_telemetry_retains_interrupted_request_without_inventing_usage(tmp_path: Path) -> None:
    handle = _start(tmp_path)
    original = handle.path.read_bytes()
    summary = summarize_provider_telemetry_v1(run_root=tmp_path, coverage=_coverage())
    assert handle.path.read_bytes() == original
    assert summary.interrupted_request_count == 1
    assert summary.log_recoveries[0].interrupted_request_ids == (handle.request_id,)
    assert summary.request_attempt_count == summary.total_tokens == summary.response_count == 0
    assert summary.reported_cost_usd == summary.elapsed_seconds == 0.0


def test_provider_telemetry_preserves_cost_for_incomplete_run_coverage(tmp_path: Path) -> None:
    handle = _start(tmp_path)
    succeed_provider_request_v1(handle, elapsed_seconds=0.5, response=_response())
    incomplete = _coverage().model_copy(update={"completed_unit_ids": ()})

    summary = summarize_provider_telemetry_v1(run_root=tmp_path, coverage=incomplete)

    assert summary.unit_ids == ("TASK1",)
    assert summary.completed_unit_ids == ()
    assert summary.response_count == 1
    assert summary.reported_cost_usd == 0.001


def test_provider_telemetry_rejects_identity_or_route_join_mismatch(tmp_path: Path) -> None:
    handle = _start(tmp_path)
    terminal = fail_provider_request_v1(handle, elapsed_seconds=0.1, failure_type="provider_error")
    lines = handle.path.read_text(encoding="utf-8").splitlines()
    payload = terminal.model_dump(mode="json")
    payload["provider_route"] = "openai"
    handle.path.write_text(lines[0] + "\n" + json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="identity or route mismatch"):
        summarize_provider_telemetry_v1(run_root=tmp_path, coverage=_coverage())


@pytest.mark.parametrize("field", ["status", "provider_route", "usage_status", "total_tokens"])
def test_provider_event_rejects_missing_status_route_or_usage(field: str) -> None:
    payload = {
        "request_id": "request-1",
        "status": "succeeded",
        "benchmark": "trialeval",
        "unit_id": "TASK1",
        "phase_id": "task",
        "step_id": "analysis",
        "turn_index": 1,
        "elapsed_seconds": 0.1,
        "requested_model": "model",
        "provider_route": "openai",
        "usage_status": "reported",
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
        "request_attempts": 1,
        "transient_failure_count": 0,
        "backoff_seconds": 0.0,
    }
    del payload[field]

    with pytest.raises(ValidationError):
        ProviderRequestEventV1.model_validate(payload)


def test_provider_telemetry_rejects_malformed_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "provider_responses.jsonl"
    path.write_text('{"schema_id":"wrong"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid provider telemetry"):
        summarize_provider_telemetry_v1(run_root=tmp_path, coverage=_coverage())


def test_provider_telemetry_rejects_inconsistent_retry_counts() -> None:
    with pytest.raises(ValidationError, match="transient_failure_count"):
        ProviderRequestEventV1(
            request_id="request-1",
            status="succeeded",
            benchmark="trialeval",
            unit_id="TASK1",
            phase_id="task",
            step_id="analysis",
            turn_index=1,
            elapsed_seconds=0.1,
            requested_model="model",
            provider_route="openai",
            usage_status="not_reported",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            request_attempts=3,
            transient_failure_count=1,
            backoff_seconds=0.0,
        )


def test_provider_failure_records_the_chained_message_and_response_body(tmp_path: Path) -> None:
    from trialagentbench.ports import ProviderRequestError

    upstream = RuntimeError("Error code: 400")
    upstream.body = {"error": {"message": "Invalid schema for function 'start_study'"}}  # type: ignore[attr-defined]
    error = ProviderRequestError(
        "Provider request failed with HTTP status 400.",
        retry_telemetry=RetryTelemetry(request_attempts=1, transient_failure_count=0, backoff_seconds=0.0),
    )
    error.__cause__ = upstream
    handle = _start(tmp_path)
    fail_provider_request_v1(handle, elapsed_seconds=0.1, failure_type="provider_error", error=error)

    terminal = ProviderRequestEventV1.model_validate_json(handle.path.read_text(encoding="utf-8").splitlines()[-1])
    assert terminal.error_detail is not None
    assert "HTTP status 400" in terminal.error_detail and "Invalid schema for function" in terminal.error_detail
    long = RuntimeError("x" * 5000)
    fail_provider_request_v1(_start(tmp_path), elapsed_seconds=0.1, failure_type="provider_error", error=long)
    last = ProviderRequestEventV1.model_validate_json(handle.path.read_text(encoding="utf-8").splitlines()[-1])
    assert last.error_detail is not None and len(last.error_detail) == 2000


def test_provider_success_records_cached_and_reasoning_tokens_and_returned_reasoning(tmp_path: Path) -> None:
    response = _response()
    response.usage = {**response.usage, "cached_tokens": 8, "reasoning_tokens": 2}
    response.reasoning = "Compare the arms first."
    handle = _start(tmp_path)
    event = succeed_provider_request_v1(handle, elapsed_seconds=0.5, response=response)

    assert (event.cached_tokens, event.reasoning_tokens, event.reasoning) == (8, 2, "Compare the arms first.")
    with pytest.raises(ValidationError, match="cannot exceed"):
        ProviderRequestEventV1.model_validate(event.model_dump() | {"cached_tokens": 11})


def _generate_recorded_turn(tmp_path: Path, provider: _Provider):
    return generate_provider_turn_v1(
        provider=provider,
        messages=({"role": "user", "content": "analyse"},),
        tools=({"type": "function", "function": {"name": "read_file"}},),
        temperature=0,
        max_tokens=100,
        timeout_seconds=30,
        tool_choice="auto",
        log_path=tmp_path / "logs" / "TASK1_provider_responses.jsonl",
        benchmark="trialdev",
        unit_id="TASK1",
        phase_id="portfolio_management",
        step_id="decision-001",
        turn_index=1,
    )


_USAGE_COUNTERS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cached_tokens",
    "reasoning_tokens",
    "compaction_requests",
    "compaction_input_tokens",
    "compaction_output_tokens",
)


@pytest.mark.parametrize(
    "usage",
    [
        pytest.param(
            {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13, "cached_tokens": 11},
            id="cached-exceeds-prompt",
        ),
        pytest.param(
            {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13, "reasoning_tokens": 4},
            id="reasoning-exceeds-completion",
        ),
        pytest.param(
            {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 12},
            id="inconsistent-total",
        ),
        pytest.param(
            {"prompt_tokens": -1, "completion_tokens": 3, "total_tokens": 2},
            id="negative-count",
        ),
        pytest.param(
            {"prompt_tokens": "10", "completion_tokens": 3, "total_tokens": 13},
            id="string-count",
        ),
        pytest.param(
            {"prompt_tokens": 10, "completion_tokens": 3.5, "total_tokens": 13.5},
            id="fractional-count",
        ),
        pytest.param(
            {"prompt_tokens": True, "completion_tokens": 3, "total_tokens": 4},
            id="boolean-count",
        ),
        pytest.param(
            {"prompt_tokens": 10, "completion_tokens": None, "total_tokens": 13},
            id="null-count",
        ),
        pytest.param({"completion_tokens": 3, "total_tokens": 13}, id="missing-prompt"),
        pytest.param({"prompt_tokens": 10, "total_tokens": 13}, id="missing-completion"),
        pytest.param({"prompt_tokens": 10, "completion_tokens": 3}, id="missing-total"),
        pytest.param(
            {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13, "cached_tokens": "bad"},
            id="malformed-optional-count",
        ),
        pytest.param(
            {
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "total_tokens": 13,
                "compaction_requests": 1,
                "compaction_input_tokens": 11,
                "compaction_output_tokens": 2,
            },
            id="inconsistent-compaction",
        ),
    ],
)
@pytest.mark.parametrize("include_raw_response", [True, False], ids=["raw-response", "usage-only"])
def test_invalid_usage_keeps_answer_and_diagnostics_without_retry(
    tmp_path: Path, usage: JsonObject, include_raw_response: bool
) -> None:
    response = _response()
    response.content = "I will read the published study."
    response.tool_calls = [ToolCall(id="call-1", name="read_file", arguments='{"path":"study.json"}')]
    response.reasoning = "Use the available study before choosing a final action."
    response.usage = usage
    response.raw_response = {"id": "response-1", "usage": dict(usage)} if include_raw_response else None
    provider = _Provider(response=response)

    result = _generate_recorded_turn(tmp_path, provider)

    # The agent must receive the actual answer, not a reconstructed response or a paid retry.
    assert provider.call_count == 1
    assert result.response is response
    assert result.response.usage is usage
    assert result.response.tool_calls == [ToolCall(id="call-1", name="read_file", arguments='{"path":"study.json"}')]
    assert result.response.reasoning == "Use the available study before choosing a final action."
    log_path = tmp_path / "logs" / "TASK1_provider_responses.jsonl"
    events = [ProviderRequestEventV1.model_validate_json(line) for line in log_path.read_text().splitlines()]
    assert [event.status for event in events] == ["started", "succeeded"]
    terminal = events[-1]
    assert terminal.usage_status == "invalid"
    assert terminal.usage_error
    assert all(getattr(terminal, field) == 0 for field in _USAGE_COUNTERS)
    assert terminal.reported_cost_usd == pytest.approx(0.001)
    assert (terminal.request_attempts, terminal.transient_failure_count, terminal.backoff_seconds) == (3, 2, 6)
    assert terminal.reasoning == response.reasoning
    assert terminal.failure_type is None

    payload_path = tmp_path / "logs" / "TASK1_provider_payloads.raw.jsonl"
    payloads = [json.loads(line) for line in payload_path.read_text().splitlines()]
    assert len(payloads) == 1
    assert payloads[0]["request_id"] == terminal.request_id
    assert payloads[0]["response"] == response.raw_response
    assert payloads[0]["usage"] == usage

    summary = summarize_provider_telemetry_v1(run_root=tmp_path, coverage=_coverage())
    assert summary.response_count == 1
    assert summary.failed_request_count == 0
    assert summary.responses_with_invalid_usage == 1
    assert summary.responses_with_usage == 0
    assert (summary.prompt_tokens, summary.completion_tokens, summary.total_tokens) == (0, 0, 0)
    assert summary.responses_with_reported_cost == 1
    assert summary.reported_cost_usd == pytest.approx(0.001)
    assert (summary.request_attempt_count, summary.transient_failure_count, summary.backoff_seconds) == (3, 2, 6)


def test_mixed_usage_summary_keeps_valid_token_totals_and_all_reported_costs(tmp_path: Path) -> None:
    _generate_recorded_turn(tmp_path, _Provider())
    response = _response()
    response.usage = {**response.usage, "reasoning_tokens": 100}
    _generate_recorded_turn(tmp_path, _Provider(response=response))

    summary = summarize_provider_telemetry_v1(run_root=tmp_path, coverage=_coverage())

    assert summary.response_count == 2
    assert summary.responses_with_usage == 1
    assert summary.responses_with_invalid_usage == 1
    assert (summary.prompt_tokens, summary.completion_tokens, summary.total_tokens) == (10, 3, 13)
    assert summary.reported_cost_usd == pytest.approx(0.002)
    assert summary.responses_with_reported_cost == 2


@pytest.mark.parametrize("invalid_usage", [False, True])
def test_unrelated_invalid_metadata_still_fails_with_diagnostics_and_one_raw_payload(
    tmp_path: Path, invalid_usage: bool
) -> None:
    response = _response()
    response.metadata = replace(response.metadata, reported_cost_usd=-1)
    if invalid_usage:
        response.usage = {**response.usage, "cached_tokens": 20}
    response.raw_response = {"id": "response-1", "usage": dict(response.usage)}
    provider = _Provider(response=response)

    with pytest.raises(ValidationError, match="reported_cost_usd"):
        _generate_recorded_turn(tmp_path, provider)

    assert provider.call_count == 1
    log_path = tmp_path / "logs" / "TASK1_provider_responses.jsonl"
    events = [ProviderRequestEventV1.model_validate_json(line) for line in log_path.read_text().splitlines()]
    assert [event.status for event in events] == ["started", "failed"]
    assert events[-1].exception_type == "ValidationError"
    assert events[-1].error_detail and "reported_cost_usd" in events[-1].error_detail
    payloads = [
        json.loads(line) for line in (tmp_path / "logs" / "TASK1_provider_payloads.raw.jsonl").read_text().splitlines()
    ]
    assert len(payloads) == 1
    assert payloads[0]["response"] == response.raw_response
    assert payloads[0]["usage"] == response.usage


@pytest.mark.parametrize("field", [None, *_USAGE_COUNTERS], ids=["missing-error", *_USAGE_COUNTERS])
def test_invalid_usage_contract_requires_diagnostic_and_zero_placeholders(tmp_path: Path, field: str | None) -> None:
    event = succeed_provider_request_v1(_start(tmp_path), elapsed_seconds=0.5, response=_response())
    payload = event.model_dump() | {key: 0 for key in _USAGE_COUNTERS}
    payload.update(usage_status="invalid", usage_error="Provider counts are inconsistent.")
    ProviderRequestEventV1.model_validate(payload)
    if field is None:
        del payload["usage_error"]
    else:
        payload[field] = 1

    with pytest.raises(ValidationError):
        ProviderRequestEventV1.model_validate(payload)


@pytest.mark.parametrize("state", ["reported", "not_reported", "started", "failed"])
def test_usage_diagnostic_is_allowed_only_for_invalid_successful_usage(tmp_path: Path, state: str) -> None:
    handle = _start(tmp_path)
    if state == "started":
        event = ProviderRequestEventV1.model_validate_json(handle.path.read_text().splitlines()[0])
    elif state == "failed":
        event = fail_provider_request_v1(handle, elapsed_seconds=0.5, failure_type="provider_error")
    else:
        response = _response()
        if state == "not_reported":
            response.usage = None
        event = succeed_provider_request_v1(handle, elapsed_seconds=0.5, response=response)

    with pytest.raises(ValidationError):
        ProviderRequestEventV1.model_validate(event.model_dump() | {"usage_error": "Misplaced diagnostic."})


def test_events_without_usage_diagnostics_remain_readable(tmp_path: Path) -> None:
    _generate_recorded_turn(tmp_path, _Provider())
    path = tmp_path / "logs" / "TASK1_provider_responses.jsonl"
    events_without_diagnostics = [json.loads(line) for line in path.read_text().splitlines()]
    for event in events_without_diagnostics:
        event.pop("usage_error", None)
    path.write_text("".join(json.dumps(event) + "\n" for event in events_without_diagnostics), encoding="utf-8")

    summary = summarize_provider_telemetry_v1(run_root=tmp_path, coverage=_coverage())
    summary_without_diagnostics = summary.model_dump(mode="json")
    summary_without_diagnostics.pop("responses_with_invalid_usage", None)
    restored = ProviderTelemetrySummaryV1.model_validate_json(json.dumps(summary_without_diagnostics))

    assert restored.responses_with_invalid_usage == 0
    assert restored.responses_with_usage == 1
    assert restored.total_tokens == 13
    assert restored.reported_cost_usd == pytest.approx(0.001)


@pytest.mark.parametrize("archived", [False, True])
@pytest.mark.parametrize("tail", [b'{"request_id":', b'{"text":"\xe2\x82'])
def test_torn_provider_tail_is_preserved_and_future_requests_use_a_new_segment(tmp_path, archived, tail):
    root = tmp_path / "archives" / "interrupted" if archived else tmp_path
    interrupted = _start(root)
    with interrupted.path.open("ab") as stream:
        stream.write(tail)
    original = interrupted.path.read_bytes()
    resumed = _start(root)
    assert resumed.path != interrupted.path
    succeed_provider_request_v1(resumed, elapsed_seconds=0.5, response=_response())
    summary = summarize_provider_telemetry_v1(run_root=tmp_path, coverage=_coverage())
    assert interrupted.path.read_bytes() == original
    assert summary.interrupted_request_count == (0 if archived else 1)
    assert summary.archived_interrupted_request_count == (1 if archived else 0)
    assert summary.response_count == (0 if archived else 1)
    (recovery,) = summary.log_recoveries
    assert recovery.torn_tail_bytes == len(tail)
    import hashlib

    assert recovery.torn_tail_sha256 == hashlib.sha256(tail).hexdigest()
    assert recovery.interrupted_request_ids == (interrupted.request_id,)


def test_newline_terminated_corruption_is_not_a_recoverable_tail(tmp_path):
    handle = _start(tmp_path)
    with handle.path.open("ab") as stream:
        stream.write(b'{"broken":\n')
    with pytest.raises(ValueError, match="Invalid provider telemetry"):
        summarize_provider_telemetry_v1(run_root=tmp_path, coverage=_coverage())
