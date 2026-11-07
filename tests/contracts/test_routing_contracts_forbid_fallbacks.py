from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from trialagentbench.adapters import llm_providers
from trialagentbench.adapters.llm_providers import (
    ChatCompletionsProvider,
    ProviderRouting,
    validate_provider_configuration,
)
from trialagentbench.contracts.core.config import RoutingConfigV1


def test_routing_config_forbids_openrouter_fallbacks_field_negative() -> None:
    payload = {
        "schema_id": "trialagentbench_routing_config_v1",
        "schema_version": 1,
        "provider": "openrouter",
        "openrouter_provider": "SomeProvider",
        "request_timeout_seconds": 300.0,
        "openrouter_allow_fallbacks": True,
    }
    with pytest.raises(ValidationError):
        RoutingConfigV1.model_validate(payload)


def test_routing_config_requires_explicit_request_timeout() -> None:
    with pytest.raises(ValidationError):
        RoutingConfigV1.model_validate({"provider": "openai"})


def test_openrouter_config_requires_provider_pin() -> None:
    with pytest.raises(ValueError, match="explicit upstream provider pin"):
        validate_provider_configuration(
            model="example/model",
            routing=ProviderRouting(provider="openrouter", openrouter_provider=None),
        )

    with pytest.raises(ValidationError, match="explicit upstream provider pin"):
        RoutingConfigV1.model_validate(
            {
                "provider": "openrouter",
                "request_timeout_seconds": 300.0,
            }
        )


def test_openrouter_pin_is_invalid_for_direct_openai_model() -> None:
    with pytest.raises(ValueError, match="only be set when provider is 'openrouter'"):
        validate_provider_configuration(
            model="example/model",
            routing=ProviderRouting(provider="openai", openrouter_provider="SomeProvider"),
        )


def test_model_identifier_does_not_select_transport() -> None:
    validate_provider_configuration(
        model="lab/model-with-a-slash",
        routing=ProviderRouting(provider="openai"),
    )


def test_openai_responses_is_an_explicit_direct_transport() -> None:
    validate_provider_configuration(
        model="example/model",
        routing=ProviderRouting(provider="openai_responses"),
    )
    config = RoutingConfigV1.model_validate(
        {
            "provider": "openai_responses",
            "request_timeout_seconds": 300.0,
        }
    )
    assert config.provider == "openai_responses"


def test_openai_responses_rejects_openrouter_pin() -> None:
    with pytest.raises(ValueError, match="only be set when provider is 'openrouter'"):
        validate_provider_configuration(
            model="example/model",
            routing=ProviderRouting(provider="openai_responses", openrouter_provider="SomeProvider"),
        )
    with pytest.raises(ValidationError, match="valid only for the OpenRouter"):
        RoutingConfigV1.model_validate(
            {
                "provider": "openai_responses",
                "openrouter_provider": "SomeProvider",
                "request_timeout_seconds": 300.0,
            }
        )


@pytest.mark.parametrize("send_temperature", [True, False])
def test_decoding_parameters_are_explicit_not_model_name_inferred(
    send_temperature: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def retry_once(
        fn: Callable[[], object],
        *,
        max_attempts: int,
        base_delay: float,
        timeout_seconds: float | None = None,
        pause_on_rate_limit: bool = False,
        credentials: tuple[str, ...] = (),
        unavailability_policy: str = "charge_all_v1",
        request_timeout_seconds: float | None = None,
    ) -> tuple[object, llm_providers.RetryTelemetry]:
        assert pause_on_rate_limit is False
        assert unavailability_policy == "charge_all_v1"
        assert request_timeout_seconds == 300.0
        captured["retry_maximum_attempts"] = max_attempts
        captured["retry_base_delay_seconds"] = base_delay
        captured["retry_timeout_seconds"] = timeout_seconds
        return (
            fn(),
            llm_providers.RetryTelemetry(
                request_attempts=1,
                transient_failure_count=0,
                backoff_seconds=0.0,
            ),
        )

    monkeypatch.setattr(
        llm_providers,
        "_retry",
        retry_once,
    )

    class _Completions:
        def create(self, **kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)

            class _Message:
                content = "ok"
                tool_calls = None

                def model_dump(self, *, mode: str) -> dict[str, object]:
                    assert mode == "json"
                    return {"role": "assistant", "content": self.content}

            return SimpleNamespace(
                id="response-1",
                model="returned-model",
                created=123,
                choices=[SimpleNamespace(message=_Message(), finish_reason="stop")],
                usage=None,
                model_dump=lambda mode: {
                    "id": "response-1",
                    "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                },
            )

    provider = object.__new__(ChatCompletionsProvider)
    provider.model = "arbitrary/reasoning-looking-model"
    provider._routing = ProviderRouting(provider="openai")
    provider._send_temperature = send_temperature
    provider._decoding_seed = None
    provider._bad_request_error_type = RuntimeError
    provider._request_timeout_s = 300.0
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))

    response = provider.generate_turn([], temperature=0.0, max_tokens=321)

    assert captured["max_completion_tokens"] == 321
    assert ("temperature" in captured) is send_temperature
    assert captured["retry_maximum_attempts"] == 5
    assert captured["retry_base_delay_seconds"] == 2.0
    assert captured["retry_timeout_seconds"] is None
    assert response.raw == {"role": "assistant", "content": "ok"}
    assert response.metadata.response_id == "response-1"
    assert response.metadata.returned_model == "returned-model"
