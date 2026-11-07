"""Prospective deferral distinguishes an outage from a spent scientific allowance."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError
from pydantic import ValidationError

from trialagentbench.adapters.llm_providers import TransientProviderResponseError, _retry
from trialagentbench.contracts.core.config import ProviderRetryPolicyV1
from trialagentbench.contracts.core.runs import ProviderRequestEventV1
from trialagentbench.ports import LLMResponse, ProviderRequestError, ProviderUnavailableError, RetryTelemetry
from trialagentbench.util import runtime_context
from trialagentbench.util.provider_telemetry import generate_provider_turn_v1, read_provider_terminal_events_v1
from trialagentbench.util.runtime_context import RuntimeDeadline, RuntimeDeadlineExceeded


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(runtime_context.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(runtime_context.time, "sleep", lambda seconds: setattr(clock, "now", clock.now + seconds))
    return clock


def _error(kind):
    request = httpx.Request("POST", "https://example.test/completion")
    if kind == "timeout":
        return APITimeoutError(request=request)
    if kind == "connection":
        return APIConnectionError(request=request)
    response = httpx.Response(kind, request=request, headers={"retry-after": "17"})
    return APIStatusError("Provider unavailable", response=response, body={"error": "unavailable"})


def _call(error, *, clock, elapsed=20.0, remaining=100.0, policy="defer_unanswered_v1"):
    attempts = []

    def request():
        attempts.append(1)
        clock.now += elapsed
        raise error

    try:
        return _retry(
            request,
            max_attempts=3,
            base_delay=0,
            timeout_seconds=remaining,
            unavailability_policy=policy,
            request_timeout_seconds=20,
        )
    finally:
        error.observed_attempts = len(attempts)


def test_default_policy_is_omitted_from_serialization_and_opt_in_is_explicit():
    assert "unavailability_policy" not in ProviderRetryPolicyV1().model_dump(mode="json")
    assert (
        ProviderRetryPolicyV1(unavailability_policy="defer_unanswered_v1").model_dump(mode="json")[
            "unavailability_policy"
        ]
        == "defer_unanswered_v1"
    )
    with pytest.raises(ValidationError):
        ProviderRetryPolicyV1(unavailability_policy="anything_else")


@pytest.mark.parametrize("kind", ["timeout", "connection", 429, 500, 502, 503, 504])
def test_unanswered_outage_defers_once_and_retains_attempt_evidence(clock, kind):
    original = _error(kind)
    with pytest.raises(ProviderUnavailableError) as caught:
        _call(original, clock=clock)
    error = caught.value
    assert original.observed_attempts == 1
    assert error.response_received is False
    assert error.retry_telemetry.request_attempts == 1
    assert error.retry_telemetry.transient_failure_count == 1
    assert len(error.retry_telemetry.attempt_failures) == 1
    assert error.retry_telemetry.backoff_seconds == 0
    assert error.unanswered_transport_seconds == 20
    assert error.__cause__ is original
    if isinstance(kind, int):
        assert error.retry_after_seconds == 17


def test_full_timeout_stays_task_noncompletion_with_default_policy(clock):
    with pytest.raises(RuntimeDeadlineExceeded):
        _call(_error("timeout"), clock=clock, elapsed=20, remaining=20, policy="charge_all_v1")


def test_timeout_shortened_by_remaining_allowance_is_not_an_outage(clock):
    with pytest.raises(RuntimeDeadlineExceeded):
        _call(_error("timeout"), clock=clock, elapsed=5, remaining=5)


@pytest.mark.parametrize("kind", [400, 401, 403, 404, 422])
def test_permanent_request_errors_do_not_gain_deferral(clock, kind):
    with pytest.raises(ProviderRequestError) as caught:
        _call(_error(kind), clock=clock, elapsed=1)
    assert not isinstance(caught.value, ProviderUnavailableError)


@pytest.mark.parametrize("evidence", ["response_flag", "payload"])
def test_ambiguous_or_received_response_never_gets_clock_refund(clock, evidence):
    error = _error("timeout")
    if evidence == "response_flag":
        error.response_received = True
    else:
        error.raw_response = {"id": "already-returned"}
    with pytest.raises(RuntimeDeadlineExceeded):
        _call(error, clock=clock, elapsed=20, remaining=20)


def test_prior_unusable_response_cannot_be_erased_by_later_outage(clock):
    failures = [TransientProviderResponseError("Unusable output", raw_response={"id": "received"}), _error("timeout")]

    def request():
        clock.now += 20
        raise failures.pop(0)

    with pytest.raises(RuntimeDeadlineExceeded) as caught:
        _retry(
            request,
            max_attempts=3,
            base_delay=0,
            timeout_seconds=40,
            unavailability_policy="defer_unanswered_v1",
            request_timeout_seconds=20,
        )
    assert caught.value.response_received is True
    assert caught.value.raw_response == {"id": "received"}


def _generate(tmp_path, provider, deadline):
    return generate_provider_turn_v1(
        provider=provider,
        messages=({"role": "user", "content": "Analyse."},),
        tools=None,
        temperature=0,
        max_tokens=100,
        timeout_seconds=deadline.remaining(),
        tool_choice="auto",
        log_path=tmp_path / "provider_responses.jsonl",
        benchmark="trialdev",
        unit_id="programme",
        phase_id="analysis",
        step_id="initial",
        turn_index=2,
        deadline=deadline,
    )


def _provider(clock, error=None):
    def request(**kwargs):
        clock.now += 20
        if error is not None:
            raise error
        return LLMResponse(content="answer", usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})

    return SimpleNamespace(model="test/model", telemetry_route="test", generate_turn=request)


def _deferred():
    return ProviderUnavailableError(
        "Unavailable", retry_telemetry=RetryTelemetry(1, 1, 0), failure_kind="timeout", unanswered_transport_seconds=20
    )


def test_only_unanswered_wait_is_excluded_and_recorded(tmp_path, clock):
    deadline = RuntimeDeadline.after(100, label="test", provider_unavailability_policy="defer_unanswered_v1")
    clock.now += 7  # Earlier successful work is not refunded.
    with pytest.raises(ProviderUnavailableError) as caught:
        _generate(tmp_path, _provider(clock, _deferred()), deadline)
    assert deadline.remaining() == 93
    assert caught.value.elapsed_seconds == 20
    assert caught.value.scientific_clock_excluded_seconds == 20
    (event,) = read_provider_terminal_events_v1(tmp_path / "provider_responses.jsonl")
    assert event.elapsed_seconds == event.scientific_clock_excluded_seconds == 20
    assert event.unavailability_policy == "defer_unanswered_v1"
    assert not event.response_received
    _generate(tmp_path, _provider(clock), deadline)
    assert deadline.remaining() == 73  # Successful provider latency still counts.
    events = read_provider_terminal_events_v1(tmp_path / "provider_responses.jsonl")
    assert len(events) == 2 and events[0].request_id != events[1].request_id
    assert events[1].scientific_clock_excluded_seconds == 0


@pytest.mark.parametrize("defect", ["received", "raw_payload", "clock_policy"])
def test_inconsistent_deferral_fails_closed_but_retains_a_complete_log(tmp_path, clock, defect):
    error = _deferred()
    if defect == "received":
        error.response_received = True
    if defect == "raw_payload":
        error.raw_response = {"id": "already-returned"}
    deadline = RuntimeDeadline.after(
        100,
        label="test",
        provider_unavailability_policy=("charge_all_v1" if defect == "clock_policy" else "defer_unanswered_v1"),
    )
    with pytest.raises(ProviderRequestError) as caught:
        _generate(tmp_path, _provider(clock, error), deadline)
    assert not isinstance(caught.value, ProviderUnavailableError)
    assert deadline.remaining() == 80
    (event,) = read_provider_terminal_events_v1(tmp_path / "provider_responses.jsonl")
    assert event.unavailability_policy is None
    assert event.scientific_clock_excluded_seconds == 0
    if defect == "raw_payload":
        payload = json.loads((tmp_path / "provider_payloads.raw.jsonl").read_text())
        assert payload["response"] == {"id": "already-returned"}


def test_telemetry_rejects_unbacked_refunds(tmp_path, clock):
    deadline = RuntimeDeadline.after(100, label="test", provider_unavailability_policy="defer_unanswered_v1")
    with pytest.raises(ProviderUnavailableError):
        _generate(tmp_path, _provider(clock, _deferred()), deadline)
    (event,) = read_provider_terminal_events_v1(tmp_path / "provider_responses.jsonl")
    saved = event.model_dump(mode="json")
    for update in (
        {"scientific_clock_excluded_seconds": 21},
        {"response_received": True},
        {"unavailability_policy": None},
        {"status": "succeeded"},
    ):
        with pytest.raises(ValidationError):
            ProviderRequestEventV1.model_validate(saved | update)


def test_successful_provider_preparation_is_not_refunded(tmp_path, clock):
    deadline = RuntimeDeadline.after(100, label="test", provider_unavailability_policy="defer_unanswered_v1")
    error = _deferred()
    error.unanswered_transport_seconds = 15  # Five seconds of successful preparation preceded transport.
    with pytest.raises(ProviderUnavailableError):
        _generate(tmp_path, _provider(clock, error), deadline)
    assert deadline.remaining() == 95
    (event,) = read_provider_terminal_events_v1(tmp_path / "provider_responses.jsonl")
    assert event.elapsed_seconds == 20 and event.scientific_clock_excluded_seconds == 15


def test_outage_does_not_release_ambiguous_spending_reservation(tmp_path, clock):
    import sqlite3

    from execution.test_campaign import configuration, request

    from trialagentbench.util.campaign_control import CampaignCapacityStop, CampaignRequestControl

    config = configuration(tmp_path)
    first = config.models[0].model_copy(update={"spend_limit_micro_usd": 1000})
    config = config.model_copy(update={"models": (first, config.models[1])})
    control = CampaignRequestControl(database=tmp_path / "requests.sqlite", configuration=config, model_name="first")

    def paid_request():
        control.reserve(request())
        clock.now += 20
        raise _error("timeout")

    with pytest.raises(ProviderUnavailableError):
        _retry(
            paid_request,
            max_attempts=3,
            base_delay=0,
            timeout_seconds=100,
            unavailability_policy="defer_unanswered_v1",
            request_timeout_seconds=20,
        )
    with sqlite3.connect(control.database) as connection:
        assert connection.execute("SELECT charge, settled FROM attempts").fetchall() == [(1000, 0)]
    with pytest.raises(CampaignCapacityStop, match="exhausted"):
        control.reserve(request())


@pytest.mark.parametrize(
    "body",
    [
        {"choices": [{"message": {"content": "paid answer"}}]},
        {"usage": {"cost": 0.02}},
        {"error": {"metadata": {"raw": '{"output":[{"text":"paid answer"}]}'}}},
    ],
)
def test_http_error_with_output_or_usage_never_retries_or_defers(clock, body):
    original = _error(503)
    original.body = body
    with pytest.raises(ProviderRequestError) as caught:
        _call(original, clock=clock, elapsed=1)
    error = caught.value
    assert not isinstance(error, ProviderUnavailableError)
    assert error.response_received is True
    assert error.raw_response == body
    assert original.observed_attempts == 1
    assert error.retry_telemetry.request_attempts == 1


def test_sdk_stripped_error_envelope_cannot_hide_sibling_usage(clock):
    body = {"error": {"message": "upstream failure"}, "usage": {"cost": 0.02}}
    response = httpx.Response(503, request=httpx.Request("POST", "https://example.test/completion"), json=body)
    original = APIStatusError("Provider failure", response=response, body=body["error"])
    with pytest.raises(ProviderRequestError) as caught:
        _call(original, clock=clock, elapsed=1)
    assert not isinstance(caught.value, ProviderUnavailableError)
    assert caught.value.raw_response == body
    assert caught.value.response_received is True
    assert original.observed_attempts == 1
