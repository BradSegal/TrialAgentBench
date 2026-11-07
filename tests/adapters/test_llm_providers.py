"""Tests for live-provider adapter contracts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
from openai import APIConnectionError, BadRequestError
from openai.types.chat import ChatCompletion

from trialagentbench.adapters.compacting_responses import OpenAIAgentsCompactingProvider
from trialagentbench.adapters.llm_providers import (
    ChatCompletionsProvider,
    ProviderRouting,
    ResponsesProvider,
    TransientProviderResponseError,
    _is_attributed_openrouter_upstream_error,
    _retry,
    get_provider,
)
from trialagentbench.contracts.core.config import ProviderRetryPolicyV1
from trialagentbench.ports import ProviderRequestError

# Assembled at run time so the published source carries no key-shaped literal (release credential scan).
_OPAQUE_CONTINUATION = "opaque-sk-" + "feature-continuation"


class _Completions:
    def __init__(self, *, upstream_provider: str, content: str | None = "Complete.") -> None:
        self.upstream_provider = upstream_provider
        self.content = content
        self.kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> Any:
        self.kwargs = kwargs
        return ChatCompletion.model_validate(
            {
                "id": "response-1",
                "object": "chat.completion",
                "model": "model-1",
                "created": 1,
                "provider": self.upstream_provider,
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 0.001},
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": self.content}, "finish_reason": "stop"}
                ],
            }
        )


def _provider(*, upstream_provider: str) -> tuple[ChatCompletionsProvider, _Completions]:
    provider = ChatCompletionsProvider.__new__(ChatCompletionsProvider)
    provider.model = "model-1"
    provider._routing = ProviderRouting(provider="openrouter", openrouter_provider="PinnedProvider")
    provider._send_temperature = False
    provider._decoding_seed = None
    provider._bad_request_error_type = BadRequestError
    provider._request_timeout_s = 300.0
    completions = _Completions(upstream_provider=upstream_provider)
    provider._client = cast(Any, SimpleNamespace(chat=SimpleNamespace(completions=completions)))
    return provider, completions


class _ResponseItem(SimpleNamespace):
    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return dict(self.raw)


class _Responses:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> Any:
        self.kwargs = kwargs
        reasoning = _ResponseItem(
            type="reasoning",
            raw={
                "id": "reasoning-1",
                "type": "reasoning",
                "summary": [],
                "encrypted_content": "opaque-reasoning-state",
            },
        )
        function_call = _ResponseItem(
            type="function_call",
            call_id="call-1",
            name="execute_code",
            arguments='{"code":"print(1)"}',
            raw={
                "id": "function-1",
                "type": "function_call",
                "call_id": "call-1",
                "name": "execute_code",
                "arguments": '{"code":"print(1)"}',
                "status": "completed",
            },
        )
        usage = SimpleNamespace(input_tokens=3, output_tokens=5, total_tokens=8)
        return SimpleNamespace(
            id="resp-1",
            model="model-1",
            created_at=123.0,
            status="completed",
            error=None,
            incomplete_details=None,
            output=[reasoning, function_call],
            usage=usage,
        )


def _responses_provider() -> tuple[ResponsesProvider, _Responses]:
    provider = ResponsesProvider.__new__(ResponsesProvider)
    provider.model = "model-1"
    provider._routing = ProviderRouting(provider="openai_responses")
    provider.telemetry_route = "openai_responses"
    provider._send_temperature = False
    provider._request_timeout_s = 300.0
    responses = _Responses()
    provider._client = cast(Any, SimpleNamespace(responses=responses))
    return provider, responses


def test_openrouter_pin_disables_provider_fallbacks() -> None:
    provider, completions = _provider(upstream_provider="PinnedProvider")

    response = provider.generate_turn([{"role": "user", "content": "test"}], temperature=0.0, max_tokens=8)

    assert response.metadata.upstream_provider == "PinnedProvider"
    assert completions.kwargs is not None
    assert completions.kwargs["extra_body"] == {"provider": {"only": ["PinnedProvider"], "allow_fallbacks": False}}


def _untyped_schema_nodes(node: object, path: str = "") -> list[str]:
    """Return schema nodes that a provider could default to a string: nesting without an explicit type."""
    found: list[str] = []
    if isinstance(node, dict):
        nested = any(key in node for key in ("$ref", "anyOf", "oneOf", "allOf", "properties", "items"))
        if nested and not any(key in node for key in ("type", "enum", "const")):
            found.append(path or "<root>")
        for key, value in node.items():
            found += _untyped_schema_nodes(value, f"{path}/{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found += _untyped_schema_nodes(value, f"{path}[{index}]")
    return found


def test_openrouter_transmits_programme_tools_with_explicit_types_at_every_depth() -> None:
    """Providers that default an untyped nested argument to a string stringified programme requests.

    Every programme and terminal tool reaches the provider unchanged, and no parameter at any depth relies on
    $ref, a union or nesting without an explicit type.
    """
    from trialagentbench.trialdev.actions import trialdev_programme_tools, trialdev_terminal_tools

    tools = [*trialdev_programme_tools(), *trialdev_terminal_tools()]
    provider, completions = _provider(upstream_provider="PinnedProvider")
    provider.generate_turn(
        [{"role": "user", "content": "Preview a study."}], tools=tools, temperature=0.0, max_tokens=8
    )
    assert completions.kwargs is not None
    assert completions.kwargs["tools"] == tools
    for tool in completions.kwargs["tools"]:
        parameters = tool["function"]["parameters"]
        assert parameters["type"] == "object"
        assert _untyped_schema_nodes(parameters["properties"]) == [], tool["function"]["name"]


def test_provider_uses_the_declared_retry_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run-level single-attempt policy must prevent implicit adapter retries."""

    provider, completions = _provider(upstream_provider="PinnedProvider")
    provider._retry_policy = ProviderRetryPolicyV1(maximum_attempts=1, base_delay_seconds=0.0)
    completions.create = lambda **_: SimpleNamespace(choices=[])  # type: ignore[method-assign]
    sleeps: list[float] = []
    monkeypatch.setattr("trialagentbench.adapters.llm_providers.time.sleep", sleeps.append)

    with pytest.raises(ProviderRequestError) as raised:
        provider.generate_turn([{"role": "user", "content": "test"}], temperature=0.0, max_tokens=8)

    assert raised.value.retry_telemetry.request_attempts == 1
    assert sleeps == []


def test_openrouter_reasoning_condition_preserves_the_exact_route_pin() -> None:
    provider, completions = _provider(upstream_provider="PinnedProvider")
    provider._reasoning_effort = "high"
    provider._exclude_reasoning = True

    provider.generate_turn([{"role": "user", "content": "test"}], temperature=0.0, max_tokens=8)

    assert completions.kwargs is not None
    assert completions.kwargs["extra_body"] == {
        "provider": {"only": ["PinnedProvider"], "allow_fallbacks": False},
        "reasoning": {"effort": "high", "exclude": True},
    }


def test_returned_reasoning_and_token_details_leave_the_replayed_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reasoning is requested and kept for the provider log; the replayed assistant message omits it."""

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    provider = get_provider(
        "model-1",
        routing=ProviderRouting(provider="openrouter", openrouter_provider="PinnedProvider"),
        send_temperature=False,
        timeout_s=30.0,
        reasoning_effort="medium",
    )
    completions = _Completions(upstream_provider="PinnedProvider")
    provider._client = cast(Any, SimpleNamespace(chat=SimpleNamespace(completions=completions)))  # type: ignore[attr-defined]
    original = completions.create

    def create(**kwargs: object) -> Any:
        response = original(**kwargs)
        message = response.choices[0].message
        message.model_dump = lambda mode, **_: {
            "role": "assistant",
            "content": "Complete.",
            "reasoning": "Compare the arms first.",
            "reasoning_details": [{"type": "reasoning.summary", "summary": "Compare the arms first."}],
        }
        response.usage.prompt_tokens_details = SimpleNamespace(cached_tokens=1)
        response.usage.completion_tokens_details = SimpleNamespace(reasoning_tokens=1)
        return response

    completions.create = create  # type: ignore[method-assign]
    response = provider.generate_turn([{"role": "user", "content": "test"}], temperature=0.0, max_tokens=8)

    assert completions.kwargs is not None
    assert cast(dict[str, Any], completions.kwargs["extra_body"])["reasoning"] == {
        "effort": "medium",
        "exclude": False,
    }
    assert response.reasoning == "Compare the arms first."
    assert response.raw == {"role": "assistant", "content": "Complete."}
    assert response.usage is not None
    assert (response.usage["cached_tokens"], response.usage["reasoning_tokens"]) == (1, 1)


@pytest.mark.parametrize("effort", [None, "high"])
def test_openrouter_returned_reasoning_survives_logging_and_tool_roundtrip(effort) -> None:
    import json

    from trialagentbench.util.agent_runtime import assistant_message_v1

    provider, completions = _provider(upstream_provider="PinnedProvider")
    provider._exclude_reasoning = False
    provider._replay_reasoning = True
    provider._reasoning_effort = effort
    ordinary_response = completions.create
    details = [{"type": "reasoning.text", "text": "Provider-returned diagnostic text.", "index": 0}]

    def create(**kwargs):
        payload = ordinary_response(**kwargs).model_dump(mode="json")
        payload["choices"][0].update(
            finish_reason="tool_calls",
            message={
                "role": "assistant",
                "content": None,
                "reasoning": "Provider-returned diagnostic text.",
                "reasoning_details": details,
                "tool_calls": [
                    {"id": "call-1", "type": "function", "function": {"name": "inspect", "arguments": "{}"}}
                ],
            },
        )
        payload["usage"]["completion_tokens_details"] = {"reasoning_tokens": 1}
        return ChatCompletion.model_validate(payload)

    completions.create = create
    first = provider.generate_turn([{"role": "user", "content": "Inspect."}], temperature=0, max_tokens=32)
    # OpenRouter returns reasoning by default; the block is sent only to set an effort.
    if effort is None:
        assert "reasoning" not in completions.kwargs["extra_body"]
    else:
        assert completions.kwargs["extra_body"]["reasoning"] == {"exclude": False, "effort": effort}
    assert first.raw_response["usage"]["completion_tokens_details"]["reasoning_tokens"] == 1
    assert first.metadata.finish_reason == "tool_calls"
    restored = json.loads(json.dumps(assistant_message_v1(first)))
    provider.generate_turn(
        [
            {"role": "user", "content": "Inspect."},
            restored,
            {"role": "tool", "tool_call_id": "call-1", "content": "Observed."},
        ],
        temperature=0,
        max_tokens=32,
    )
    replayed = completions.kwargs["messages"][1]
    assert replayed["reasoning_details"] == details
    assert "reasoning" not in replayed  # only the structured details carry reasoning state
    assert "usage" not in replayed


@pytest.mark.parametrize("content", [None, "", "   "])
def test_chat_provider_rejects_reasoning_only_turn_without_public_output(content: str | None) -> None:
    """Prevent an invalid null assistant message from entering replay state."""

    provider, completions = _provider(upstream_provider="PinnedProvider")
    completions.content = content

    response = provider.generate_turn(
        [{"role": "user", "content": "test"}],
        temperature=0.0,
        max_tokens=8,
    )
    assert response.model_outcome == "empty_response"
    assert response.content is None and response.tool_calls == []
    assert response.usage["total_tokens"] == 2
    assert response.raw_response is not None


def test_responses_provider_translates_tools_and_retains_replay_state() -> None:
    provider, responses = _responses_provider()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "execute_code",
                "description": "Run Python.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            },
        }
    ]

    response = provider.generate_turn(
        [{"role": "system", "content": "Analyse."}, {"role": "user", "content": "Begin."}],
        tools=tools,
        temperature=0.0,
        max_tokens=128,
        tool_choice="required",
    )

    assert responses.kwargs is not None
    assert responses.kwargs["store"] is False
    assert responses.kwargs["include"] == ["reasoning.encrypted_content"]
    assert responses.kwargs["max_output_tokens"] == 128
    assert responses.kwargs["tool_choice"] == "required"
    assert responses.kwargs["tools"] == [
        {
            "type": "function",
            "name": "execute_code",
            "description": "Run Python.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        }
    ]
    assert response.tool_calls[0].id == "call-1"
    assert response.usage == {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}
    assert response.metadata.response_id == "resp-1"
    assert response.metadata.created_unix == 123
    assert isinstance(response.raw, dict)
    assert response.raw["provider_state"][0]["encrypted_content"] == "opaque-reasoning-state"


def test_responses_provider_replays_output_items_before_tool_results() -> None:
    provider, responses = _responses_provider()
    first = provider.generate_turn(
        [{"role": "user", "content": "Begin."}],
        temperature=0.0,
        max_tokens=128,
    )
    assert isinstance(first.raw, dict)

    provider.generate_turn(
        [
            {"role": "user", "content": "Begin."},
            first.raw,
            {"role": "tool", "tool_call_id": "call-1", "content": "1"},
        ],
        temperature=0.0,
        max_tokens=128,
    )

    assert responses.kwargs is not None
    input_items = cast(list[dict[str, object]], responses.kwargs["input"])
    assert input_items[1]["type"] == "reasoning"
    assert input_items[2]["type"] == "function_call"
    assert input_items[3] == {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": "1",
    }


def test_responses_provider_sends_declared_reasoning_effort() -> None:
    provider, responses = _responses_provider()
    provider._reasoning_effort = "high"

    provider.generate_turn(
        [{"role": "user", "content": "Begin."}],
        temperature=0.0,
        max_tokens=128,
    )

    assert responses.kwargs is not None
    assert responses.kwargs["reasoning"] == {"effort": "high"}


def test_compaction_rejects_non_responses_transport() -> None:
    with pytest.raises(ValueError, match="requires --provider openai_responses"):
        get_provider(
            "gpt-5.6",
            routing=ProviderRouting(provider="openai"),
            send_temperature=False,
            timeout_s=30.0,
            memory_strategy="openai_responses_compaction",
        )


def test_provider_factory_selects_one_compacting_session_per_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    provider = get_provider(
        "gpt-5.6",
        routing=ProviderRouting(provider="openai_responses"),
        send_temperature=False,
        timeout_s=30.0,
        memory_strategy="openai_responses_compaction",
    )

    assert isinstance(provider, OpenAIAgentsCompactingProvider)
    assert provider.manages_context is True
    assert provider.telemetry_route == "openai_responses_compaction"


def test_provider_sends_declared_decoding_seed() -> None:
    provider, completions = _provider(upstream_provider="PinnedProvider")
    provider._decoding_seed = 17

    provider.generate_turn([{"role": "user", "content": "test"}], temperature=0.0, max_tokens=8)

    assert completions.kwargs is not None
    assert completions.kwargs["seed"] == 17


def test_provider_omits_undeclared_decoding_seed() -> None:
    provider, completions = _provider(upstream_provider="PinnedProvider")

    provider.generate_turn([{"role": "user", "content": "test"}], temperature=0.0, max_tokens=8)

    assert completions.kwargs is not None
    assert "seed" not in completions.kwargs


def test_provider_requires_one_of_the_offered_tools_when_requested() -> None:
    provider, completions = _provider(upstream_provider="PinnedProvider")
    tools = [{"type": "function", "function": {"name": "submit_response", "parameters": {"type": "object"}}}]

    provider.generate_turn(
        [{"role": "user", "content": "submit"}],
        tools=tools,
        temperature=0.0,
        max_tokens=8,
        tool_choice="required",
    )

    assert completions.kwargs is not None
    assert completions.kwargs["tools"] == tools
    assert completions.kwargs["tool_choice"] == "required"


def test_provider_rejects_required_tool_choice_without_tools() -> None:
    provider, _ = _provider(upstream_provider="PinnedProvider")

    with pytest.raises(ValueError, match="requires at least one provider tool"):
        provider.generate_turn(
            [{"role": "user", "content": "submit"}],
            temperature=0.0,
            max_tokens=8,
            tool_choice="required",
        )


def test_chat_completions_rejects_responses_provider_state() -> None:
    provider, _ = _provider(upstream_provider="PinnedProvider")

    with pytest.raises(ValueError, match="cannot be sent through Chat Completions"):
        provider.generate_turn(
            [
                {"role": "user", "content": "test"},
                {
                    "role": "assistant",
                    "content": None,
                    "provider_state": [{"type": "reasoning", "encrypted_content": "opaque"}],
                },
            ],
            temperature=0.0,
            max_tokens=8,
        )


def test_openrouter_pin_rejects_returned_provider_drift() -> None:
    provider, _ = _provider(upstream_provider="FallbackProvider")

    with pytest.raises(RuntimeError, match="violated the requested upstream-provider pin"):
        provider.generate_turn([{"role": "user", "content": "test"}], temperature=0.0, max_tokens=8)


def test_openrouter_pin_accepts_provider_display_casing() -> None:
    provider, _ = _provider(upstream_provider="pInNeDpRoViDeR")

    response = provider.generate_turn([{"role": "user", "content": "test"}], temperature=0.0, max_tokens=8)

    assert response.metadata.upstream_provider == "pInNeDpRoViDeR"


def test_retry_returns_exact_attempt_and_backoff_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = 0
    sleeps: list[float] = []

    def request() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise APIConnectionError(request=httpx.Request("POST", "https://provider.invalid"))
        return "response"

    monkeypatch.setattr("trialagentbench.adapters.llm_providers.time.sleep", sleeps.append)

    response, telemetry = _retry(request, max_attempts=5, base_delay=2.0)

    assert response == "response"
    assert telemetry.request_attempts == 3
    assert telemetry.transient_failure_count == 2
    assert telemetry.backoff_seconds == 6.0
    assert sleeps == [2.0, 4.0]


def test_provider_retries_empty_completion_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, completions = _provider(upstream_provider="PinnedProvider")
    valid_create = completions.create
    attempts = 0
    sleeps: list[float] = []

    def create(**kwargs: object) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return SimpleNamespace(choices=None)
        return valid_create(**kwargs)

    completions.create = create  # type: ignore[method-assign]
    monkeypatch.setattr("trialagentbench.adapters.llm_providers.time.sleep", sleeps.append)

    response = provider.generate_turn([{"role": "user", "content": "test"}], temperature=0.0, max_tokens=8)

    assert attempts == 2
    assert response.metadata.request_attempts == 2
    assert response.metadata.transient_failure_count == 1
    assert response.metadata.backoff_seconds == 2.0
    assert sleeps == [2.0]
    assert response.metadata.attempt_failures[0]["exception_type"] == "TransientProviderResponseError"
    assert response.metadata.attempt_failures[0]["summary"] == (
        "Provider returned no completion choices for a successful chat request."
    )


@pytest.mark.parametrize("recover", [True, False])
def test_http_retry_causes_are_saved_without_authentication_material(tmp_path, monkeypatch, recover) -> None:
    import json

    from openai import InternalServerError

    from trialagentbench.util.provider_telemetry import generate_provider_turn_v1

    provider, completions = _provider(upstream_provider="PinnedProvider")
    provider.telemetry_route = "openrouter:PinnedProvider"
    provider._retry_policy = ProviderRetryPolicyV1(maximum_attempts=2, base_delay_seconds=0)
    ordinary_response = completions.create
    attempts = 0
    # Assembled at run time so the published source carries no key-shaped literal (release credential scan).
    secret = "sk-" + "test-credential-must-never-be-logged"

    def create(**kwargs):
        nonlocal attempts
        attempts += 1
        if not recover or attempts == 1:
            request = httpx.Request("POST", "https://provider.invalid", headers={"Authorization": f"Bearer {secret}"})
            raise InternalServerError(
                f"Sensitive SDK exception text: {secret}",
                response=httpx.Response(503, request=request, headers={"set-cookie": secret}),
                body={
                    "error": {
                        "code": "upstream_overloaded",
                        "type": "server_error",
                        "message": secret,
                        "request": {"headers": {"Authorization": secret}},
                        "metadata": {"provider_name": "PinnedProvider", "raw": secret},
                    }
                },
            )
        return ordinary_response(**kwargs)

    completions.create = create
    monkeypatch.setattr("trialagentbench.adapters.llm_providers.time.sleep", lambda _: None)
    log = tmp_path / "provider_responses.jsonl"

    def run():
        return generate_provider_turn_v1(
            provider=provider,
            messages=[{"role": "user", "content": "test"}],
            tools=None,
            temperature=0,
            max_tokens=16,
            timeout_seconds=30,
            tool_choice="auto",
            log_path=log,
            benchmark="trialeval",
            unit_id="task",
            phase_id="task",
            step_id="analysis",
            turn_index=1,
        )

    if recover:
        result = run()
        assert result.response.metadata.request_attempts == 2
    else:
        with pytest.raises(ProviderRequestError):
            run()
    payload_text = (tmp_path / "provider_payloads.raw.jsonl").read_text()
    payload = json.loads(payload_text)
    failures = payload["attempt_failures"]
    assert [row["attempt"] for row in failures] == ([1] if recover else [1, 2])
    assert all(row["exception_type"] == "InternalServerError" for row in failures)
    assert all(row["http_status_code"] == 503 for row in failures)
    assert failures[0]["provider_error"] == {
        "code": "upstream_overloaded",
        "type": "server_error",
        "provider_name": "PinnedProvider",
    }
    assert secret not in payload_text
    assert "Authorization" not in payload_text and "set-cookie" not in payload_text
    event = json.loads(log.read_text().splitlines()[-1])
    assert event["status"] == ("succeeded" if recover else "failed")
    assert "attempt_failures" not in event  # The scored telemetry schema stays unchanged.


def test_provider_retries_completion_inference_error(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, completions = _provider(upstream_provider="PinnedProvider")
    valid_create = completions.create
    attempts = 0
    sleeps: list[float] = []

    def create(**kwargs: object) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            response = valid_create(**kwargs)
            response.choices[0].finish_reason = "error"
            return response
        return valid_create(**kwargs)

    completions.create = create  # type: ignore[method-assign]
    monkeypatch.setattr("trialagentbench.adapters.llm_providers.time.sleep", sleeps.append)

    response = provider.generate_turn([{"role": "user", "content": "test"}], temperature=0.0, max_tokens=8)

    assert attempts == 2
    assert response.metadata.request_attempts == 2
    assert response.metadata.transient_failure_count == 1
    assert response.metadata.backoff_seconds == 2.0
    assert sleeps == [2.0]


def test_provider_attempt_timeout_is_capped_to_remaining_deadline() -> None:
    provider, completions = _provider(upstream_provider="PinnedProvider")

    provider.generate_turn(
        [{"role": "user", "content": "test"}],
        temperature=0.0,
        max_tokens=8,
        timeout_seconds=12.0,
    )

    assert completions.kwargs is not None
    assert 0.0 < cast(float, completions.kwargs["timeout"]) <= 12.0


def test_retry_exhausts_on_empty_provider_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("trialagentbench.adapters.llm_providers.time.sleep", lambda _: None)

    with pytest.raises(ProviderRequestError, match="exhausting transient-error retries") as raised:
        _retry(
            lambda: (_ for _ in ()).throw(TransientProviderResponseError("Provider returned no completion choices.")),
            max_attempts=2,
            base_delay=0.0,
        )
    telemetry = raised.value.retry_telemetry
    assert telemetry.request_attempts == 2
    assert telemetry.transient_failure_count == 2
    assert telemetry.backoff_seconds == 0.0


def test_retry_caps_backoff_to_remaining_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = 10.0
    sleeps: list[float] = []

    def monotonic() -> float:
        return clock

    def sleep(seconds: float) -> None:
        nonlocal clock
        sleeps.append(seconds)
        clock += seconds

    monkeypatch.setattr("trialagentbench.adapters.llm_providers.time.monotonic", monotonic)
    monkeypatch.setattr("trialagentbench.adapters.llm_providers.time.sleep", sleep)

    with pytest.raises(TimeoutError, match="retry backoff") as raised:
        _retry(
            lambda: (_ for _ in ()).throw(
                APIConnectionError(request=httpx.Request("POST", "https://provider.invalid"))
            ),
            max_attempts=5,
            base_delay=2.0,
            timeout_seconds=1.25,
        )

    assert sleeps == [1.25]
    assert raised.value.retry_telemetry.request_attempts == 1
    assert raised.value.retry_telemetry.transient_failure_count == 1
    assert raised.value.retry_telemetry.backoff_seconds == 1.25


def _bad_request(*, body: object) -> BadRequestError:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    return BadRequestError(
        "bad request",
        response=httpx.Response(400, request=request),
        body=body,
    )


def test_openrouter_upstream_error_requires_exact_provider_attribution() -> None:
    attributed = _bad_request(
        body={
            "message": "Provider returned error",
            "metadata": {"provider_name": "PinnedProvider"},
        }
    )
    local_error = _bad_request(body={"message": "Invalid model identifier"})
    wrong_provider = _bad_request(
        body={
            "message": "Provider returned error",
            "metadata": {"provider_name": "OtherProvider"},
        }
    )

    assert _is_attributed_openrouter_upstream_error(attributed, expected_provider="PinnedProvider")
    assert not _is_attributed_openrouter_upstream_error(local_error, expected_provider="PinnedProvider")
    assert not _is_attributed_openrouter_upstream_error(wrong_provider, expected_provider="PinnedProvider")
    assert not _is_attributed_openrouter_upstream_error(attributed, expected_provider=None)


def test_provider_retries_error_attributed_to_pinned_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, completions = _provider(upstream_provider="PinnedProvider")
    valid_create = completions.create
    attempts = 0
    sleeps: list[float] = []

    def create(**kwargs: object) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _bad_request(
                body={
                    "message": "Provider returned error",
                    "metadata": {"provider_name": "PinnedProvider"},
                }
            )
        return valid_create(**kwargs)

    completions.create = create  # type: ignore[method-assign]
    monkeypatch.setattr("trialagentbench.adapters.llm_providers.time.sleep", sleeps.append)

    response = provider.generate_turn([{"role": "user", "content": "test"}], temperature=0.0, max_tokens=8)

    assert attempts == 2
    assert response.metadata.request_attempts == 2
    assert response.metadata.transient_failure_count == 1
    assert sleeps == [2.0]


def test_provider_does_not_retry_unattributed_bad_request(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, completions = _provider(upstream_provider="PinnedProvider")
    attempts = 0

    def create(**_: object) -> Any:
        nonlocal attempts
        attempts += 1
        raise _bad_request(body={"message": "Invalid model identifier"})

    completions.create = create  # type: ignore[method-assign]
    monkeypatch.setattr("trialagentbench.adapters.llm_providers.time.sleep", lambda _: None)

    with pytest.raises(ProviderRequestError, match="HTTP status 400") as raised:
        provider.generate_turn([{"role": "user", "content": "test"}], temperature=0.0, max_tokens=8)
    assert attempts == 1
    assert isinstance(raised.value.__cause__, BadRequestError)


def test_provider_disables_sdk_retries_and_applies_explicit_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("openai.OpenAI", _Client)

    ChatCompletionsProvider(
        model="model-1",
        routing=ProviderRouting(provider="openai"),
        send_temperature=False,
        timeout_s=123.0,
    )

    assert captured["timeout"] == 123.0
    assert captured["max_retries"] == 0


@pytest.mark.parametrize("decoding_seed", [True, -1])
def test_provider_rejects_invalid_decoding_seed(
    monkeypatch: pytest.MonkeyPatch,
    decoding_seed: object,
) -> None:
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: SimpleNamespace())

    with pytest.raises(ValueError, match="decoding seed"):
        ChatCompletionsProvider(
            model="model-1",
            routing=ProviderRouting(provider="openai"),
            send_temperature=False,
            timeout_s=123.0,
            decoding_seed=cast(Any, decoding_seed),
        )


def test_responses_provider_rejects_unsupported_decoding_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: SimpleNamespace())

    with pytest.raises(ValueError, match="does not support a decoding seed"):
        ResponsesProvider(
            model="model-1",
            routing=ProviderRouting(provider="openai_responses"),
            send_temperature=False,
            timeout_s=123.0,
            decoding_seed=17,
        )


def test_provider_factory_selects_direct_responses_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: SimpleNamespace())

    provider = get_provider(
        "model-1",
        routing=ProviderRouting(provider="openai_responses"),
        send_temperature=False,
        timeout_s=123.0,
    )

    assert isinstance(provider, ResponsesProvider)
    assert provider.telemetry_route == "openai_responses"


@pytest.mark.parametrize("transport", ["chat", "responses"])
@pytest.mark.parametrize("include_credential", [False, True])
def test_diagnostic_patterns_do_not_change_executed_code_or_replayed_messages(transport, include_credential):
    """Diagnostic token matching cannot rewrite scientific code or its continuation context."""
    import json

    key = "dummy-credential-for-input"
    text = "analysis...continues; sk-feature; Bearer documentation-example"
    if include_credential:
        text += "; " + key
    arguments = json.dumps({"code": "print(" + repr(text) + ")"})
    if transport == "chat":
        provider, endpoint = _provider(upstream_provider="PinnedProvider")
        original = endpoint.create
        provider._replay_reasoning = True

        def create(**kwargs):
            response = original(**kwargs)
            from openai.types.chat import ChatCompletionMessageFunctionToolCall

            response.choices[0].message.content = text
            response.choices[0].message.reasoning_details = [
                {"type": "reasoning.encrypted", "data": _OPAQUE_CONTINUATION}
            ]
            response.choices[0].message.tool_calls = [
                ChatCompletionMessageFunctionToolCall(
                    id="call", type="function", function={"name": "execute_code", "arguments": arguments}
                )
            ]
            return response

    else:
        provider, endpoint = _responses_provider()
        original = endpoint.create

        def create(**kwargs):
            response = original(**kwargs)
            response.output[0].raw["encrypted_content"] = _OPAQUE_CONTINUATION
            response.output[-1].arguments = arguments
            response.output[-1].raw["arguments"] = arguments
            return response

    provider._credentials = (key,)
    endpoint.create = create
    result = provider.generate_turn([], temperature=0, max_tokens=10)
    expected = arguments.replace(key, "[REDACTED]")
    assert result.tool_calls[0].arguments == expected
    assert result.raw["tool_calls"][0]["function"]["arguments"] == expected
    if transport == "chat":
        assert result.content == text.replace(key, "[REDACTED]")
        assert result.raw["reasoning_details"][0]["data"] == _OPAQUE_CONTINUATION
    else:
        assert result.raw["provider_state"][-1]["arguments"] == expected
        assert result.raw["provider_state"][0]["encrypted_content"] == _OPAQUE_CONTINUATION
