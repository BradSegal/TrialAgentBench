"""Offline HTTP probes for cancellation and diagnostic credential custody."""

from __future__ import annotations

import gzip
import json
import threading
import time
import traceback
import zipfile
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from trialagentbench.adapters.llm_providers import ChatCompletionsProvider, ProviderRouting
from trialagentbench.contracts.core.config import ProviderRetryPolicyV1
from trialagentbench.ports import ProviderRequestError, RuntimeDeadlineExceeded
from trialagentbench.util.provider_redaction import redact_provider_text
from trialagentbench.util.provider_telemetry import generate_provider_turn_v1


@contextmanager
def _server(mode, body=None):
    calls = []
    disconnected = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            calls.append(self.path)
            try:
                if mode == "headers":
                    self.wfile.write(b"HTTP/1.1 200 OK\r\nX-Drip: ")
                elif mode == "body" or (mode == "body_then_success" and len(calls) == 1):
                    self.send_response(200)
                    self.send_header("Content-Length", "10000")
                    self.end_headers()
                else:
                    content = json.dumps(body).encode()
                    if mode == "gzip":
                        content = gzip.compress(content)
                    self.send_response(401 if mode == "error" else 200)
                    self.send_header("Content-Type", "application/json")
                    if mode == "gzip":
                        self.send_header("Content-Encoding", "gzip")
                    self.send_header("Content-Length", str(len(content)))
                    self.end_headers()
                    self.wfile.write(content)
                    return
                for _ in range(100):
                    self.wfile.write(b"x")
                    self.wfile.flush()
                    time.sleep(0.02)
            except (BrokenPipeError, ConnectionResetError):
                disconnected.set()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", calls, disconnected
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def _provider(url, *, api_key="dummy-offline-key"):
    provider = ChatCompletionsProvider(
        model="fixture",
        base_url=url,
        api_key=api_key,
        routing=ProviderRouting("openai"),
        timeout_s=10,
        send_temperature=False,
        retry_policy=ProviderRetryPolicyV1(maximum_attempts=3, base_delay_seconds=0),
    )
    # Exclude one-time SDK lazy imports from the subsecond I/O experiment.
    _ = provider._client.chat.completions
    return provider


def _recorded_turn(provider, tmp_path):
    return generate_provider_turn_v1(
        provider=provider,
        messages=[{"role": "user", "content": "offline"}],
        tools=[],
        temperature=0,
        max_tokens=10,
        timeout_seconds=5,
        tool_choice="auto",
        log_path=tmp_path / "provider_responses.jsonl",
        benchmark="trialeval",
        unit_id="task",
        phase_id="task",
        step_id="analysis",
        turn_index=1,
    )


@pytest.mark.parametrize("mode", ["body", "headers"])
def test_absolute_deadline_cancels_slow_drip_without_background_requests(mode, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with _server(mode) as (url, calls, disconnected):
        provider = _provider(url)
        try:
            started = time.monotonic()
            with pytest.raises(RuntimeDeadlineExceeded):
                provider.generate_turn(
                    [{"role": "user", "content": "offline"}],
                    temperature=0,
                    max_tokens=10,
                    timeout_seconds=0.2,
                )
            assert time.monotonic() - started < 1
            assert len(calls) == 1
            assert disconnected.wait(1), "Cancellation must close the socket before the adapter returns."
            time.sleep(0.1)
            assert len(calls) == 1
        finally:
            provider.close()
            provider.close()


def test_mock_401_echoes_no_credentials_into_logs_exports_or_traceback(tmp_path, monkeypatch):
    # All credentials are deliberately fake; do not inspect operator credentials.
    key = "dummy-provider-credential-401"
    custom = "dummy-custom-provider-credential"
    monkeypatch.setenv("OPENAI_API_KEY", key)
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-openrouter-key")
    monkeypatch.setenv("CUSTOM_PROVIDER_API_KEY", custom)
    message = f"bad credentials {key}; {custom}; Bearer arbitrary-token; sk-masked***last"
    with _server("error", {"error": {"message": message, "code": "unauthorized"}}) as (url, calls, _):
        provider = _provider(url, api_key=key)
        try:
            with pytest.raises(ProviderRequestError) as caught:
                _recorded_turn(provider, tmp_path)
        finally:
            provider.close()
    assert len(calls) == 1
    rendered = "".join(traceback.format_exception(caught.value))
    rendered += "".join(path.read_text() for path in tmp_path.rglob("*") if path.is_file())
    for secret in (key, custom, "arbitrary-token", "sk-masked***last"):
        assert secret not in rendered
    assert "[REDACTED]" in rendered
    # Use the public exporter's full-custody member path: it copies diagnostic
    # sidecars into the portable result archive without a second redaction pass.
    from trialagentbench.tools.export_results import _write_member

    sources = tuple(path for path in tmp_path.rglob("*") if path.is_file())
    exported = tmp_path / "diagnostic-export.zip"
    with zipfile.ZipFile(exported, "w") as archive:
        for source in sources:
            _write_member(archive, member=source.name, source=source, profile="full_custody")
    with zipfile.ZipFile(exported) as archive:
        content = "".join(archive.read(name).decode() for name in archive.namelist())
    for secret in (key, custom, "arbitrary-token", "sk-masked***last"):
        assert secret not in content


def test_redaction_precedes_truncation_and_covers_masked_forms(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-prefix-secret-suffix")
    text = "x" * 1998 + "dummy-prefix-secret-suffix sk-or-v1-abcd...wxyz Bearer abc123"
    redacted = redact_provider_text(text)[:2000]
    assert redacted.endswith("[R")
    assert "dummy" not in redacted


@pytest.mark.parametrize(
    "message,outcome",
    [
        ({"role": "assistant", "content": None, "refusal": "I cannot answer this request."}, "refusal"),
        (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call", "type": "function", "function": None}],
            },
            "malformed_response",
        ),
        (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call",
                        "type": "function",
                        "function": {"name": "execute_code", "arguments": {"code": "pass"}},
                    }
                ],
            },
            "malformed_response",
        ),
        ({"role": "assistant", "content": " "}, "empty_response"),
    ],
)
def test_paid_refusal_or_malformed_tool_envelope_is_retained_without_retry(tmp_path, monkeypatch, message, outcome):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-offline-key")
    payload = {
        "id": "paid-response",
        "model": "fixture",
        "object": "chat.completion",
        "created": 1,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12, "cost": 0.01},
    }
    with _server("response", payload) as (url, calls, _):
        provider = _provider(url)
        try:
            turn = _recorded_turn(provider, tmp_path)
        finally:
            provider.close()
    assert len(calls) == 1
    assert turn.response.model_outcome == outcome
    terminal = json.loads((tmp_path / "provider_responses.jsonl").read_text().splitlines()[-1])
    assert (terminal["status"], terminal["model_outcome"], terminal["total_tokens"]) == ("succeeded", outcome, 12)
    assert terminal["reported_cost_usd"] == 0.01
    raw = json.loads((tmp_path / "provider_payloads.raw.jsonl").read_text())
    assert raw["response"]["id"] == "paid-response"
    for key, value in message.items():
        assert raw["response"]["choices"][0]["message"][key] == value


def test_raw_payload_and_usage_error_also_redact_credentials(tmp_path, monkeypatch):
    from trialagentbench.ports import LLMResponse
    from trialagentbench.util.provider_telemetry import start_provider_request_v1, succeed_provider_request_v1

    secret = "dummy-raw-payload-credential"
    monkeypatch.setenv("CUSTOM_PROVIDER_API_KEY", secret)
    handle = start_provider_request_v1(
        path=tmp_path / "provider_responses.jsonl",
        benchmark="trialeval",
        unit_id="task",
        phase_id="task",
        step_id="analysis",
        turn_index=1,
        requested_model="fixture",
        provider_route="offline",
    )
    succeed_provider_request_v1(
        handle,
        elapsed_seconds=0.1,
        response=LLMResponse(
            content="retained",
            raw_response={"echo": secret, "authorization": "Bearer arbitrary-token"},
            usage={"prompt_tokens": secret, "completion_tokens": 0, "total_tokens": 0},
        ),
    )
    for path in tmp_path.iterdir():
        text = path.read_text()
        assert secret not in text and "arbitrary-token" not in text


@pytest.mark.parametrize("mode", ["response", "gzip", "body_then_success"])
def test_cancellable_client_can_be_reused_with_a_fresh_deadline_after_success_or_timeout(mode):
    payload = {
        "id": "fixture",
        "model": "fixture",
        "object": "chat.completion",
        "created": 1,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "retained"}, "finish_reason": "stop"}],
    }
    with _server(mode, payload) as (url, calls, disconnected):
        provider = _provider(url)
        provider._retry_policy = ProviderRetryPolicyV1(maximum_attempts=1)
        try:
            if mode == "body_then_success":
                provider._request_timeout_s = 0.15
                with pytest.raises(ProviderRequestError):
                    provider.generate_turn([], temperature=0, max_tokens=10, timeout_seconds=5)
                assert disconnected.wait(1)
                provider._request_timeout_s = 2
            else:
                assert (
                    provider.generate_turn([], temperature=0, max_tokens=10, timeout_seconds=2).content == "retained"
                )
            assert provider.generate_turn([], temperature=0, max_tokens=10, timeout_seconds=2).content == "retained"
            assert len(calls) == 2
        finally:
            provider.close()


@pytest.mark.parametrize("masked", ["abcd...wxyz", "abcd****wxyz", "sk-…abcd", "sk-or-v1-abcd***wxyz"])
def test_masked_credentials_are_redacted_without_requiring_the_unmasked_key(masked):
    assert masked not in redact_provider_text(f"Incorrect API key provided: {masked}")


@pytest.mark.parametrize("kind", ["refusal", "malformed_tool"])
def test_responses_transport_retains_noncompletion_and_usage_from_real_http(tmp_path, kind):
    from trialagentbench.adapters.llm_providers import ResponsesProvider

    output = (
        {
            "id": "msg",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "refusal", "refusal": "I cannot answer."}],
        }
        if kind == "refusal"
        else {"id": "fc", "type": "function_call", "call_id": "call", "name": None, "arguments": "{}"}
    )
    payload = {
        "id": "resp",
        "object": "response",
        "created_at": 1,
        "model": "fixture",
        "status": "completed",
        "output": [output],
        "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
    }
    with _server("response", payload) as (url, calls, _):
        provider = ResponsesProvider(
            model="fixture",
            api_key="dummy-offline-key",
            routing=ProviderRouting("openai_responses"),
            timeout_s=5,
            send_temperature=False,
        )
        provider._client.base_url = url + "/"
        try:
            result = _recorded_turn(provider, tmp_path)
        finally:
            provider.close()
    assert len(calls) == 1
    assert result.response.model_outcome == ("refusal" if kind == "refusal" else "malformed_response")
    assert result.response.usage["total_tokens"] == 12
    raw = json.loads((tmp_path / "provider_payloads.raw.jsonl").read_bytes())
    assert raw["response"]["output"][0]["id"] == output["id"]


def test_paid_response_with_invalid_metadata_retains_payload_and_receipt_without_retry(tmp_path):
    payload = {
        "id": "paid",
        "created": "invalid-timestamp",
        "model": "fixture",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}],
    }
    with _server("response", payload) as (url, calls, _):
        provider = _provider(url)
        try:
            with pytest.raises(TypeError, match="timestamp"):
                _recorded_turn(provider, tmp_path)
        finally:
            provider.close()
    assert len(calls) == 1
    terminal = json.loads((tmp_path / "provider_responses.jsonl").read_text().splitlines()[-1])
    assert terminal["response_received"] is True
    raw = json.loads((tmp_path / "provider_payloads.raw.jsonl").read_bytes())
    assert raw["response"]["id"] == "paid" and raw["response"]["created"] == "invalid-timestamp"


@pytest.mark.parametrize("kind", ["chat", "responses"])
def test_paid_empty_envelope_exhaustion_retains_receipt_and_raw_usage(tmp_path, kind):
    from trialagentbench.adapters.llm_providers import ResponsesProvider

    payload = {
        "id": "paid-empty-envelope",
        "object": "chat.completion",
        "created": 1,
        "model": "fixture",
        "choices": [],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    if kind == "responses":
        payload = {
            "id": "paid-empty-envelope",
            "object": "response",
            "created_at": 1,
            "model": "fixture",
            "status": "completed",
            "output": [],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }
    with _server("response", payload) as (url, calls, _):
        if kind == "chat":
            provider = _provider(url)
        else:
            provider = ResponsesProvider(
                model="fixture",
                api_key="dummy-offline-key",
                routing=ProviderRouting("openai_responses"),
                timeout_s=5,
                send_temperature=False,
                retry_policy=ProviderRetryPolicyV1(maximum_attempts=3, base_delay_seconds=0),
            )
            provider._client.base_url = url + "/"
        try:
            with pytest.raises(ProviderRequestError):
                _recorded_turn(provider, tmp_path)
        finally:
            provider.close()
    assert len(calls) == 3  # Preserve the declared transient-response retry policy.
    terminal = json.loads((tmp_path / "provider_responses.jsonl").read_text().splitlines()[-1])
    assert terminal["response_received"] is True
    raw = json.loads((tmp_path / "provider_payloads.raw.jsonl").read_text())
    assert raw["response"]["id"] == "paid-empty-envelope"
    assert raw["response"]["usage"]["total_tokens"] == 2


@pytest.mark.parametrize("terminal", ["transport", "deadline", "rate_limit", "local"])
def test_later_failure_cannot_erase_an_earlier_paid_retry_receipt(terminal):
    import httpx2
    from openai import APIConnectionError, RateLimitError

    from trialagentbench.adapters.llm_providers import TransientProviderResponseError, _retry
    from trialagentbench.util.campaign_control import CampaignCapacityStop

    calls = 0
    payload = {"id": "earlier-paid", "usage": {"total_tokens": 2}}
    request = httpx2.Request("POST", "http://127.0.0.1/offline")

    def attempt():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TransientProviderResponseError("unusable paid response", raw_response=payload)
        if terminal == "rate_limit":
            raise RateLimitError("rate limit", response=httpx2.Response(429, request=request), body=None)
        if terminal == "local":
            raise APIConnectionError(request=request) from OSError("offline disk failure")
        raise APIConnectionError(request=request)

    expected = {
        "transport": ProviderRequestError,
        "deadline": RuntimeDeadlineExceeded,
        "rate_limit": CampaignCapacityStop,
        "local": OSError,
    }[terminal]
    with pytest.raises(expected) as caught:
        _retry(
            attempt,
            max_attempts=2,
            base_delay=1 if terminal == "deadline" else 0,
            timeout_seconds=0.01 if terminal == "deadline" else 2,
            pause_on_rate_limit=terminal == "rate_limit",
        )
    assert caught.value.response_received is True
    assert caught.value.raw_response == payload
    assert calls == (1 if terminal == "deadline" else 2)
