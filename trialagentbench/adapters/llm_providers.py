"""LLM provider adapters for explicit OpenAI-compatible transports.

These are *adapters*, not ports. Import from CLI entrypoints only.
Offline grading and deterministic analysis workflows must not import this module.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from math import isfinite
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

from pydantic import BaseModel

if TYPE_CHECKING:
    from trialagentbench.util.campaign_control import CampaignRequestControl

from trialagentbench.contracts.core.config import MemoryStrategyV1, ProviderRetryPolicyV1, ReasoningEffortV1
from trialagentbench.ports import (
    ClosableLLMProvider,
    LLMProvider,
    LLMResponse,
    LLMResponseMetadata,
    ProviderRequestError,
    ProviderUnavailableError,
    RetryTelemetry,
    RuntimeDeadlineExceeded,
    ToolCall,
)
from trialagentbench.ports.llm_provider import JsonObject, JsonValue
from trialagentbench.util.provider_redaction import (
    redact_provider_exception,
    redact_provider_text,
    redact_provider_value,
)

R = TypeVar("R")


class TransientProviderResponseError(RuntimeError):
    """Provider response was successful at transport level but unusable."""

    def __init__(self, message: str, *, raw_response: JsonObject | None = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.response_received = True


def _same_provider_name(observed: object, expected: str | None) -> bool:
    """Compare provider labels without treating display casing as route drift."""
    return (
        isinstance(observed, str)
        and expected is not None
        and observed.strip().casefold() == expected.strip().casefold()
    )


def _terminal_provider_error(
    error: BaseException,
    *,
    detail: str,
    request_attempts: int,
    transient_failure_count: int,
    backoff_seconds: float,
    attempt_failures: tuple[JsonObject, ...] = (),
) -> ProviderRequestError:
    """Translate a third-party transport exception into the harness boundary."""
    translated = ProviderRequestError(
        detail,
        retry_telemetry=RetryTelemetry(
            request_attempts=request_attempts,
            transient_failure_count=transient_failure_count,
            backoff_seconds=backoff_seconds,
            attempt_failures=attempt_failures,
        ),
    )
    translated.__cause__ = error
    return translated


def _raw_payload(
    response: object, *, credentials: tuple[str, ...] = (), match_patterns: bool = True
) -> JsonObject | None:
    """Return a provider response's JSON for the diagnostic sidecar, or None for a non-model response."""
    dump = getattr(response, "model_dump", None)
    if not callable(dump):
        return None
    try:
        payload = dump(mode="json", warnings=False) if isinstance(response, BaseModel) else dump(mode="json")
    except (AttributeError, TypeError, ValueError):
        return None
    return (
        cast(JsonObject, redact_provider_value(payload, credentials=credentials, match_patterns=match_patterns))
        if isinstance(payload, dict)
        else None
    )


@contextmanager
def _retain_received_response(
    response: object, *, credentials: tuple[str, ...], retry: RetryTelemetry
) -> Iterator[None]:
    """Preserve paid-response evidence when ingress validation raises, without retrying it."""

    completed = False
    try:
        yield
        completed = True
    finally:
        error = sys.exception()
        if not completed and error is not None:
            # Third-party ingress exceptions are extensible, like retry telemetry.
            error.raw_response = _raw_payload(response, credentials=credentials)  # type: ignore[attr-defined]
            error.response_received = True  # type: ignore[attr-defined]
            error.retry_telemetry = retry  # type: ignore[attr-defined]
            redact_provider_exception(error, credentials=credentials)


def _retry_failure(error: BaseException, *, attempt: int) -> JsonObject:
    """Keep bounded error diagnostics without serializing SDK requests, headers or arbitrary bodies."""
    cause = error if getattr(error, "status_code", None) is not None else error.__cause__ or error
    status = getattr(cause, "status_code", None)
    if not isinstance(status, int) or isinstance(status, bool) or not 100 <= status <= 599:
        status = None
    safe_summaries = {
        "OpenRouter attributed a failed completion to the pinned upstream provider.",
        "Provider returned no completion choices for a successful chat request.",
        "Provider returned a completion choice without an assistant message.",
        "Provider returned a completion choice with an inference error.",
        "Provider returned no public assistant text or tool call for a successful chat request.",
        "OpenAI Responses returned an inference error.",
        "OpenAI Responses returned terminal status 'failed'.",
        "OpenAI Responses returned terminal status 'cancelled'.",
        "OpenAI Responses returned no output items.",
    }
    # Only fixed adapter descriptions are copied; SDK exception strings can
    # contain credentials, request headers or whole provider bodies.
    if isinstance(error, TransientProviderResponseError) and str(error) in safe_summaries:
        summary = str(error)
    elif status is not None:
        summary = f"Provider HTTP request failed with status {status}."
    elif isinstance(error, TransientProviderResponseError):
        summary = "Provider returned an unusable completion."
    else:
        summary = "Provider transport request failed."
    diagnostic: dict[str, JsonValue] = {
        "attempt": attempt,
        "exception_type": type(error).__name__,
        "cause_exception_type": type(cause).__name__,
        "http_status_code": status,
        "summary": summary,
    }
    body = getattr(cause, "body", None)
    if isinstance(body, dict):
        body = body.get("error", body)
    if isinstance(body, dict):
        metadata = body.get("metadata")
        fields = {"code": body.get("code"), "type": body.get("type")}
        if isinstance(metadata, dict):
            fields["provider_name"] = metadata.get("provider_name")
        # Retain machine-readable labels only. Free-text messages and nested/raw
        # provider payloads can echo authentication material and are not copied.
        safe_fields: dict[str, JsonValue] = {}
        for key, value in fields.items():
            if isinstance(value, int) and not isinstance(value, bool):
                safe_fields[key] = value
            elif (
                isinstance(value, str)
                and re.fullmatch(r"[A-Za-z0-9_. /:-]{1,128}", value)
                and not re.search(r"(?i)bearer|authorization|api.?key|secret|password|sk-", value)
            ):
                safe_fields[key] = value
        if safe_fields:
            diagnostic["provider_error"] = safe_fields
    return diagnostic


def _is_attributed_openrouter_upstream_error(
    error: object,
    *,
    expected_provider: str | None,
) -> bool:
    """Return whether OpenRouter attributes an error to the pinned upstream."""
    if expected_provider is None:
        return False
    body = getattr(error, "body", None)
    if not isinstance(body, dict) or body.get("message") != "Provider returned error":
        return False
    metadata = body.get("metadata")
    return isinstance(metadata, dict) and _same_provider_name(metadata.get("provider_name"), expected_provider)


def _has_model_response_evidence(value: object, *, depth: int = 0) -> bool:
    """Recognize ambiguous model output/usage hidden inside an SDK HTTP error body."""
    if depth > 12:
        return True  # An uninspectably nested payload cannot establish an unanswered request.
    if isinstance(value, dict):
        if {
            "choices",
            "output",
            "output_text",
            "usage",
            "reasoning",
            "reasoning_details",
            "prompt_tokens",
            "completion_tokens",
            "input_tokens",
            "output_tokens",
        }.intersection(value):
            return True
        return any(_has_model_response_evidence(child, depth=depth + 1) for child in value.values())
    if isinstance(value, list):
        return any(_has_model_response_evidence(child, depth=depth + 1) for child in value)
    if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            decoded = json.loads(value)
        except (ValueError, RecursionError):
            return True  # Malformed nested JSON is ambiguous, not proof of no returned output.
        return _has_model_response_evidence(decoded, depth=depth + 1)
    return False


def _retry_after(error: object) -> float:
    """Honor an HTTP Retry-After delay without extending the caller's deadline."""
    response = getattr(error, "response", None)
    header = None if response is None else response.headers.get("retry-after")
    if not isinstance(header, str):
        return 0.0
    try:
        seconds = float(header)
    except ValueError:
        try:
            seconds = parsedate_to_datetime(header).timestamp() - time.time()
        except (ValueError, TypeError, OverflowError):
            return 0.0
    return max(0.0, seconds) if isfinite(seconds) else 0.0


def _retry(
    fn: Callable[[], R],
    *,
    max_attempts: int = 5,
    base_delay: float = 2.0,
    timeout_seconds: float | None = None,
    pause_on_rate_limit: bool = False,
    credentials: tuple[str, ...] = (),
    unavailability_policy: Literal["charge_all_v1", "defer_unanswered_v1"] = "charge_all_v1",
    request_timeout_seconds: float | None = None,
) -> tuple[R, RetryTelemetry]:
    """Run fn() with exponential backoff on transient HTTP/provider errors."""
    # Keep exception surface explicit: do not swallow unexpected errors.
    #
    # This adapter is imported only for opt-in network-backed workflows, so it
    # is acceptable to import provider exceptions lazily at call time.
    from openai import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        RateLimitError,
    )

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")
    if base_delay < 0.0:
        raise ValueError("base_delay must be non-negative.")
    if unavailability_policy not in {"charge_all_v1", "defer_unanswered_v1"}:
        raise ValueError("Unknown provider-unavailability policy.")
    if unavailability_policy == "defer_unanswered_v1" and (
        request_timeout_seconds is None or not isfinite(request_timeout_seconds) or request_timeout_seconds <= 0.0
    ):
        raise ValueError("Deferring unanswered requests requires a declared request timeout.")
    if timeout_seconds is not None and timeout_seconds <= 0.0:
        raise RuntimeDeadlineExceeded("Provider request deadline was exhausted before the first attempt.")
    retry_started = time.monotonic()
    deadline = retry_started + timeout_seconds if timeout_seconds is not None else None
    last_exc: Exception | None = None
    backoff_seconds = 0.0
    attempt_failures: list[JsonObject] = []
    response_received = False
    received_payload: JsonObject | None = None
    completed = False

    def deadline_exceeded(message: str, *, request_attempts: int) -> RuntimeDeadlineExceeded:
        error = RuntimeDeadlineExceeded(message)
        _attach_retry_telemetry(
            error,
            request_attempts=request_attempts,
            transient_failure_count=request_attempts,
            backoff_seconds=backoff_seconds,
            attempt_failures=tuple(attempt_failures),
        )
        return error

    def back_off(error: Exception, *, attempt: int, retriable: bool, detail: str) -> None:
        """Record a failed attempt, then raise its terminal error or wait for the next attempt."""
        nonlocal backoff_seconds, last_exc, response_received, received_payload
        redact_provider_exception(error, credentials=credentials)
        body = getattr(error, "body", None)
        if unavailability_policy == "defer_unanswered_v1" and isinstance(error, APIStatusError):
            # The SDK strips a top-level {"error": ...} envelope from .body.
            # Inspect only its already-buffered JSON too, so sibling usage or
            # choices cannot disappear. This never reads a live response stream.
            http_response = getattr(error, "response", None)
            if http_response is not None and getattr(http_response, "is_stream_consumed", False):
                try:
                    full_body = http_response.json()
                except (ValueError, RuntimeError):
                    full_body = None
                if _has_model_response_evidence(full_body):
                    body = redact_provider_value(full_body, credentials=credentials)
        ambiguous_http_body = (
            unavailability_policy == "defer_unanswered_v1"
            and isinstance(error, APIStatusError)
            and _has_model_response_evidence(body)
        )
        if ambiguous_http_body:
            # An HTTP failure is not evidence of an unanswered request when the
            # provider also returned output or billing evidence. Preserve it and
            # require review instead of retrying, refunding time or deferring.
            error.response_received = True  # type: ignore[attr-defined]
            error.raw_response = body if isinstance(body, dict) else {"ambiguous_http_body": body}  # type: ignore[attr-defined]
        response_received = response_received or getattr(error, "response_received", False) is True
        raw_response = getattr(error, "raw_response", None)
        if isinstance(raw_response, dict):
            received_payload = cast(JsonObject, raw_response)
        attempt_failures.append(_retry_failure(error, attempt=attempt + 1))
        if ambiguous_http_body:
            raise _terminal_provider_error(
                error,
                detail="Provider HTTP failure included output or usage evidence; operator review required.",
                request_attempts=attempt + 1,
                transient_failure_count=attempt,
                backoff_seconds=backoff_seconds,
                attempt_failures=tuple(attempt_failures),
            ) from error
        delay = max(base_delay * (2**attempt), _retry_after(error))
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        status = getattr(error, "status_code", None)
        # The opt-in policy defers at the first observable no-output transport
        # failure. It never buys another response inside this logical request.
        # A timeout shortened by the task's remaining allowance is its ordinary
        # scientific time limit, not evidence that provider capacity disappeared.
        complete_timeout_window = timeout_seconds is None or (
            request_timeout_seconds is not None and timeout_seconds >= request_timeout_seconds
        )
        unavailable = (
            (isinstance(error, APITimeoutError) and complete_timeout_window)
            or (isinstance(error, APIConnectionError) and not isinstance(error, APITimeoutError))
            or (isinstance(error, APIStatusError) and status in {429, 500, 502, 503, 504})
        )
        if (
            unavailability_policy == "defer_unanswered_v1"
            and unavailable
            and not response_received
            and received_payload is None
        ):
            deferred = ProviderUnavailableError(
                "Provider unavailable without a returned model response; preserve and defer this work.",
                retry_telemetry=RetryTelemetry(
                    request_attempts=attempt + 1,
                    transient_failure_count=attempt + 1,
                    backoff_seconds=backoff_seconds,
                    attempt_failures=tuple(attempt_failures),
                ),
                failure_kind=(
                    "timeout" if isinstance(error, APITimeoutError) else f"http_{status}" if status else "connection"
                ),
                retry_after_seconds=delay,
                unanswered_transport_seconds=time.monotonic() - retry_started,
            )
            if isinstance(status, int):
                deferred.status_code = status
            raise deferred from error
        if (
            pause_on_rate_limit
            and getattr(error, "status_code", None) == 429
            and (attempt == max_attempts - 1 or (remaining is not None and delay >= remaining))
        ):
            from trialagentbench.util.campaign_control import CampaignCapacityStop

            stopped = CampaignCapacityStop(
                "Provider rate limit persists; retain pending work and resume after the provider window recovers",
                campaign_wide=False,
            )
            _attach_retry_telemetry(
                stopped,
                request_attempts=attempt + 1,
                transient_failure_count=attempt + 1,
                backoff_seconds=backoff_seconds,
                attempt_failures=tuple(attempt_failures),
            )
            raise stopped from error
        if remaining is not None and remaining <= 0:
            raise deadline_exceeded(
                "Provider request deadline was exhausted during transport.", request_attempts=attempt + 1
            ) from error
        if not retriable or attempt == max_attempts - 1:
            raise _terminal_provider_error(
                error,
                detail=detail,
                request_attempts=attempt + 1,
                transient_failure_count=attempt + 1 if retriable else attempt,
                backoff_seconds=backoff_seconds,
                attempt_failures=tuple(attempt_failures),
            ) from error
        last_exc = error
        # Deterministic backoff: no jitter. This keeps reruns predictable and
        # avoids introducing non-reproducible timing into provider-backed
        # workflows.
        if remaining is not None:
            if delay >= remaining:
                if remaining:
                    time.sleep(remaining)
                    backoff_seconds += remaining
                raise deadline_exceeded(
                    "Provider request deadline was exhausted during retry backoff.", request_attempts=attempt + 1
                ) from error
        backoff_seconds += delay
        time.sleep(delay)

    try:
        for attempt in range(max_attempts):
            if deadline is not None and time.monotonic() >= deadline:
                raise deadline_exceeded(
                    "Provider request deadline was exhausted before a retry attempt.", request_attempts=attempt
                ) from last_exc
            try:
                result = fn()
                completed = True
                return (
                    result,
                    RetryTelemetry(
                        request_attempts=attempt + 1,
                        transient_failure_count=attempt,
                        backoff_seconds=backoff_seconds,
                        attempt_failures=tuple(attempt_failures),
                    ),
                )
            except (APIConnectionError, APITimeoutError, RateLimitError, TransientProviderResponseError) as e:
                # The SDK also wraps errors raised by our spending boundary as
                # connection errors. Retrying those can buy the same completion
                # repeatedly after malformed usage or a local ledger failure.
                redact_provider_exception(e, credentials=credentials)
                if isinstance(e, APIConnectionError) and isinstance(e.__cause__, (ValueError, OSError, sqlite3.Error)):
                    cause = e.__cause__
                    safe_detail = redact_provider_text(str(cause), credentials=credentials)
                    if safe_detail != str(cause):
                        # ValidationError and some OS errors render internal fields,
                        # rather than args. Retain a safe boundary error, with no
                        # unsafe rich exception in its printed cause chain.
                        raise ValueError(safe_detail) from None
                    raise cause from None
                back_off(
                    e,
                    attempt=attempt,
                    retriable=True,
                    detail="Provider request failed after exhausting transient-error retries.",
                )
            except APIStatusError as e:
                # Status-code driven retry policy. Do not retry client errors:
                # those are almost always permanent (bad request, invalid model id,
                # auth failure, etc.). Retry rate limits and transient server
                # failures only.
                status = getattr(e, "status_code", None)
                back_off(
                    e,
                    attempt=attempt,
                    retriable=status in {429, 500, 502, 503, 504},
                    detail=f"Provider request failed with HTTP status {status or 'unknown'}.",
                )
        raise RuntimeError("retry loop exhausted without returning or raising")
    finally:
        error = sys.exception()
        if not completed and error is not None and response_received:
            # A later timeout or transport error cannot erase an earlier paid
            # response from this logical request's retry sequence.
            error.response_received = True  # type: ignore[attr-defined]
            if not isinstance(getattr(error, "raw_response", None), dict) and received_payload is not None:
                error.raw_response = received_payload  # type: ignore[attr-defined]


def _attach_retry_telemetry(
    error: BaseException,
    *,
    request_attempts: int,
    transient_failure_count: int,
    backoff_seconds: float,
    attempt_failures: tuple[JsonObject, ...] = (),
) -> None:
    """Attach typed observations to the exact exception propagated to the caller."""

    error.retry_telemetry = RetryTelemetry(  # type: ignore[attr-defined]
        request_attempts=request_attempts,
        transient_failure_count=transient_failure_count,
        backoff_seconds=backoff_seconds,
        attempt_failures=attempt_failures,
    )


@dataclass
class ProviderRouting:
    """Explicit API transport and OpenRouter upstream routing pin."""

    provider: Literal["openai", "openai_responses", "openrouter", "offline_control"]
    openrouter_provider: str | None = None


def validate_provider_configuration(
    *,
    model: str,
    routing: ProviderRouting,
) -> None:
    """Validate provider routing before a live request is sent.

    Parameters
    ----------
    model:
        Exact model identifier sent to the selected provider.
    routing:
        Exact provider transport and upstream route.

    Raises
    ------
    ValueError
        If routing is inconsistent or cannot identify the OpenRouter upstream.
    """
    if not model.strip():
        raise ValueError("model must be a non-empty exact provider model identifier")
    if routing.openrouter_provider and routing.provider != "openrouter":
        raise ValueError("openrouter_provider can only be set when provider is 'openrouter'")
    if routing.provider == "openrouter" and not routing.openrouter_provider:
        raise ValueError("OpenRouter runs require an explicit upstream provider pin")


def _validate_decoding_seed(decoding_seed: int | None) -> None:
    """Reject values that cannot be transported as provider decoding seeds."""

    if isinstance(decoding_seed, bool) or (
        decoding_seed is not None and (not isinstance(decoding_seed, int) or decoding_seed < 0)
    ):
        raise ValueError("Provider decoding seed must be a non-negative integer or None.")


def _provider_retry_policy(provider: object) -> ProviderRetryPolicyV1:
    """Return the configured retry policy, including for minimal adapter fixtures."""

    policy = getattr(provider, "_retry_policy", None)
    if policy is None:
        return ProviderRetryPolicyV1()
    if not isinstance(policy, ProviderRetryPolicyV1):
        raise TypeError("Provider retry policy must use ProviderRetryPolicyV1.")
    return policy


class ChatCompletionsProvider:
    """OpenAI Chat Completions-compatible provider (OpenAI or OpenRouter)."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        routing: ProviderRouting,
        send_temperature: bool,
        timeout_s: float,
        decoding_seed: int | None = None,
        reasoning_effort: ReasoningEffortV1 | None = None,
        exclude_reasoning: bool = False,
        replay_reasoning: bool = False,
        retry_policy: ProviderRetryPolicyV1 | None = None,
        request_control: CampaignRequestControl | None = None,
    ) -> None:
        _validate_decoding_seed(decoding_seed)
        if reasoning_effort is not None and routing.provider != "openrouter":
            raise ValueError("Explicit reasoning effort is supported only through the OpenRouter transport.")
        if replay_reasoning and (routing.provider != "openrouter" or exclude_reasoning):
            raise ValueError("Replaying returned reasoning requires OpenRouter with reasoning returned.")
        # Import lazily so offline paths can import the harness without the
        # `providers` extra installed.
        from openai import BadRequestError, OpenAI

        self.model = model
        self._routing = routing
        self.telemetry_route = (
            f"openrouter:{routing.openrouter_provider}"
            if routing.openrouter_provider is not None
            else routing.provider
        )
        self._send_temperature = send_temperature
        self._decoding_seed = decoding_seed
        self._reasoning_effort = reasoning_effort
        self._exclude_reasoning = exclude_reasoning
        self._replay_reasoning = replay_reasoning
        self._retry_policy = retry_policy or ProviderRetryPolicyV1()
        self._pause_on_rate_limit = request_control is not None
        self._bad_request_error_type = BadRequestError
        if timeout_s <= 0.0:
            raise ValueError("Provider request timeout must be positive.")
        self._request_timeout_s = float(timeout_s)
        client_kwargs: dict[str, Any] = {
            "timeout": self._request_timeout_s,
            # The harness owns and records retries. SDK retries would be
            # invisible in the per-response provenance.
            "max_retries": 0,
        }
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if request_control is not None:
            from trialagentbench.util.campaign_control import controlled_http_client

            if routing.provider != "openrouter":
                raise ValueError("Durable campaign reservations currently support pinned OpenRouter routes")
            client_kwargs["http_client"] = controlled_http_client(request_control)
        if "http_client" not in client_kwargs:
            import httpx2

            from trialagentbench.adapters.http_transport import DeadlineTransport

            client_kwargs["http_client"] = httpx2.Client(transport=DeadlineTransport())
        self._client = OpenAI(**client_kwargs)
        self._credentials = (api_key,) if api_key is not None else ()

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
        if tool_choice == "required" and not tools:
            raise ValueError("tool_choice='required' requires at least one provider tool.")
        if any(message.get("provider_state") is not None for message in messages):
            raise ValueError("Responses provider state cannot be sent through Chat Completions.")
        request_deadline = time.monotonic() + timeout_seconds if timeout_seconds is not None else None
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "max_completion_tokens": int(max_tokens),
        }
        if self._send_temperature:
            kwargs["temperature"] = float(temperature)
        if self._decoding_seed is not None:
            kwargs["seed"] = self._decoding_seed
        if tools:
            kwargs["tools"] = list(tools)
            if tool_choice == "required":
                kwargs["tool_choice"] = "required"

        extra_body: dict[str, Any] = {}
        if self._routing.openrouter_provider:
            extra_body["provider"] = {
                "only": [self._routing.openrouter_provider],
                "allow_fallbacks": False,
            }
        reasoning_effort = getattr(self, "_reasoning_effort", None)
        exclude_reasoning = getattr(self, "_exclude_reasoning", False)
        # OpenRouter returns reasoning by default; the block is sent only to set an effort or request exclusion,
        # so request bodies are unchanged for the default condition.
        if reasoning_effort is not None or (self._routing.provider == "openrouter" and exclude_reasoning):
            extra_body["reasoning"] = {"exclude": exclude_reasoning}
            if reasoning_effort is not None:
                extra_body["reasoning"]["effort"] = reasoning_effort
        if extra_body:
            kwargs["extra_body"] = extra_body

        def request() -> Any:
            try:
                request_kwargs = dict(kwargs)
                if request_deadline is not None:
                    remaining = request_deadline - time.monotonic()
                    if remaining <= 0.0:
                        raise RuntimeDeadlineExceeded("Provider request deadline was exhausted before transport.")
                    request_kwargs["timeout"] = min(self._request_timeout_s, remaining)
                from trialagentbench.adapters.http_transport import provider_request_deadline

                token = provider_request_deadline.set(request_deadline)
                try:
                    response = self._client.chat.completions.create(**request_kwargs)
                finally:
                    provider_request_deadline.reset(token)
            except self._bad_request_error_type as error:
                if not _is_attributed_openrouter_upstream_error(
                    error,
                    expected_provider=self._routing.openrouter_provider,
                ):
                    raise
                raise TransientProviderResponseError(
                    "OpenRouter attributed a failed completion to the pinned upstream provider."
                ) from error
            choices = getattr(response, "choices", None)
            if not isinstance(choices, list) or not choices:
                raise TransientProviderResponseError(
                    "Provider returned no completion choices for a successful chat request.",
                    raw_response=_raw_payload(response, credentials=getattr(self, "_credentials", ())),
                )
            if getattr(choices[0], "message", None) is None:
                raise TransientProviderResponseError(
                    "Provider returned a completion choice without an assistant message.",
                    raw_response=_raw_payload(response, credentials=getattr(self, "_credentials", ())),
                )
            if getattr(choices[0], "finish_reason", None) == "error":
                raise TransientProviderResponseError(
                    "Provider returned a completion choice with an inference error.",
                    raw_response=_raw_payload(response, credentials=getattr(self, "_credentials", ())),
                )
            return response

        retry_policy = _provider_retry_policy(self)
        response, retry = _retry(
            request,
            max_attempts=retry_policy.maximum_attempts,
            base_delay=retry_policy.base_delay_seconds,
            timeout_seconds=timeout_seconds,
            pause_on_rate_limit=getattr(self, "_pause_on_rate_limit", False),
            credentials=getattr(self, "_credentials", ()),
            unavailability_policy=retry_policy.unavailability_policy,
            request_timeout_seconds=self._request_timeout_s,
        )
        with _retain_received_response(response, credentials=getattr(self, "_credentials", ()), retry=retry):
            return self._parse_response(response, retry=retry)

    def _parse_response(self, response: Any, *, retry: RetryTelemetry) -> LLMResponse:
        """Validate SDK envelopes after the transport retry boundary."""

        choice = response.choices[0]
        msg = choice.message

        tool_calls: list[ToolCall] = []
        outcome: Literal["refusal", "malformed_response", "empty_response"] | None = None
        refusal = getattr(msg, "refusal", None)
        content = msg.content if isinstance(msg.content, str) and msg.content.strip() else None
        if getattr(choice, "finish_reason", None) == "content_filter":
            outcome = "refusal"
        if isinstance(refusal, str) and refusal.strip():
            content = refusal
            outcome = "refusal"
        envelopes = getattr(msg, "tool_calls", None)
        if envelopes is not None:
            if not isinstance(envelopes, list):
                outcome = "malformed_response"
            else:
                for tc in envelopes:
                    function = getattr(tc, "function", None)
                    call_id = getattr(tc, "id", None)
                    name = getattr(function, "name", None)
                    arguments = getattr(function, "arguments", None)
                    if (
                        getattr(tc, "type", None) != "function"
                        or not isinstance(call_id, str)
                        or not call_id.strip()
                        or not isinstance(name, str)
                        or not name.strip()
                        or not isinstance(arguments, str)
                        or call_id in {call.id for call in tool_calls}
                    ):
                        outcome = "malformed_response"
                        tool_calls.clear()
                        break
                    tool_calls.append(ToolCall(id=call_id, name=name, arguments=arguments))
        if content is None and not tool_calls and outcome is None:
            outcome = "empty_response"

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                **_detail_tokens(
                    cached=getattr(response.usage, "prompt_tokens_details", None),
                    reasoning=getattr(response.usage, "completion_tokens_details", None),
                ),
            }
        raw_response_extra = getattr(response, "model_extra", None)
        response_extra = raw_response_extra if isinstance(raw_response_extra, dict) else {}
        raw_usage_extra = getattr(response.usage, "model_extra", None) if response.usage is not None else None
        usage_extra = raw_usage_extra if isinstance(raw_usage_extra, dict) else {}
        provider_name = response_extra.get("provider")
        reported_cost = usage_extra.get("cost")
        metadata = LLMResponseMetadata(
            response_id=_optional_string(getattr(response, "id", None), credentials=getattr(self, "_credentials", ())),
            returned_model=_optional_string(
                getattr(response, "model", None), credentials=getattr(self, "_credentials", ())
            ),
            upstream_provider=_optional_string(provider_name, credentials=getattr(self, "_credentials", ())),
            finish_reason=_optional_string(
                getattr(choice, "finish_reason", None), credentials=getattr(self, "_credentials", ())
            ),
            created_unix=_optional_integer(getattr(response, "created", None)),
            reported_cost_usd=_optional_nonnegative_float(reported_cost),
            request_attempts=retry.request_attempts,
            transient_failure_count=retry.transient_failure_count,
            backoff_seconds=retry.backoff_seconds,
            attempt_failures=retry.attempt_failures,
        )
        if self._routing.openrouter_provider and not _same_provider_name(
            provider_name,
            self._routing.openrouter_provider,
        ):
            raise RuntimeError(
                "OpenRouter response violated the requested upstream-provider pin: "
                f"requested={self._routing.openrouter_provider!r}, returned={metadata.upstream_provider!r}."
            )
        raw_message = _raw_payload(msg, credentials=getattr(self, "_credentials", ()), match_patterns=False)
        if not isinstance(raw_message, dict):
            raise TypeError("Provider message serialization must produce a JSON object.")
        # Returned reasoning goes to the provider log; the agent's context omits it unless the condition replays
        # the provider's reasoning details (reasoning-state continuity across tool turns).
        replay = getattr(self, "_replay_reasoning", False)
        reasoning = raw_message.pop("reasoning", None)
        details = raw_message.get("reasoning_details") or ()
        if not (replay and details):
            raw_message.pop("reasoning_details", None)
        if not isinstance(reasoning, str) or not reasoning:
            texts = [part.get("text") or part.get("summary") for part in details if isinstance(part, dict)]
            reasoning = "\n".join(text for text in texts if isinstance(text, str) and text) or None
        if content is not None:
            content = redact_provider_text(
                content, credentials=getattr(self, "_credentials", ()), match_patterns=False
            )
        for call in tool_calls:
            call.id = redact_provider_text(
                call.id, credentials=getattr(self, "_credentials", ()), match_patterns=False
            )
            call.name = redact_provider_text(
                call.name, credentials=getattr(self, "_credentials", ()), match_patterns=False
            )
            call.arguments = redact_provider_text(
                call.arguments, credentials=getattr(self, "_credentials", ()), match_patterns=False
            )
        if usage is not None:
            usage = cast(
                dict[str, JsonValue],
                redact_provider_value(
                    dict(usage), credentials=getattr(self, "_credentials", ()), match_patterns=False
                ),
            )
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            metadata=metadata,
            raw=raw_message,
            reasoning=reasoning,
            raw_response=_raw_payload(response, credentials=getattr(self, "_credentials", ())),
            model_outcome=outcome,
        )

    def close(self) -> None:
        """Close the underlying HTTP transport."""

        self._client.close()


def _responses_tools(tools: Sequence[JsonObject] | None) -> list[dict[str, JsonValue]]:
    """Translate Chat Completions function declarations to Responses tools."""

    translated: list[dict[str, JsonValue]] = []
    for index, tool in enumerate(tools or ()):
        if tool.get("type") != "function":
            raise ValueError(f"Responses tool {index} must be a function declaration.")
        function = tool.get("function")
        if not isinstance(function, dict):
            raise ValueError(f"Responses tool {index} is missing its function object.")
        name = function.get("name")
        parameters = function.get("parameters")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Responses tool {index} requires a non-empty function name.")
        if not isinstance(parameters, dict):
            raise ValueError(f"Responses tool {name!r} requires an object-valued parameters schema.")
        declaration: dict[str, JsonValue] = {
            "type": "function",
            "name": name,
            "parameters": parameters,
        }
        description = function.get("description")
        if description is not None:
            if not isinstance(description, str):
                raise ValueError(f"Responses tool {name!r} description must be a string.")
            declaration["description"] = description
        strict = function.get("strict")
        if strict is not None:
            if not isinstance(strict, bool):
                raise ValueError(f"Responses tool {name!r} strict flag must be boolean.")
            declaration["strict"] = strict
        translated.append(declaration)
    return translated


def responses_input_items(messages: Sequence[JsonObject]) -> list[dict[str, JsonValue]]:
    """Translate the canonical harness conversation to Responses input items."""

    translated: list[dict[str, JsonValue]] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        if role in {"system", "user"}:
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError(f"Responses {role} message {index} requires string content.")
            translated.append({"role": role, "content": content})
            continue
        if role == "tool":
            call_id = message.get("tool_call_id")
            output = message.get("content")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError(f"Responses tool message {index} requires a tool_call_id.")
            if not isinstance(output, str):
                raise ValueError(f"Responses tool message {index} requires string content.")
            translated.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                }
            )
            continue
        if role != "assistant":
            raise ValueError(f"Unsupported harness message role for Responses input: {role!r}.")

        provider_state = message.get("provider_state")
        if provider_state is not None:
            if not isinstance(provider_state, list) or not provider_state:
                raise ValueError(f"Responses assistant message {index} has invalid provider_state.")
            for item in provider_state:
                if not isinstance(item, dict) or not isinstance(item.get("type"), str):
                    raise ValueError(f"Responses assistant message {index} contains invalid provider state.")
                translated.append(item)
            continue

        content = message.get("content")
        if content is not None:
            if not isinstance(content, str):
                raise ValueError(f"Responses assistant message {index} content must be a string or null.")
            translated.append({"role": "assistant", "content": content})
        raw_calls = message.get("tool_calls")
        if raw_calls is None:
            continue
        if not isinstance(raw_calls, list):
            raise ValueError(f"Responses assistant message {index} tool_calls must be a list.")
        for raw_call in raw_calls:
            if not isinstance(raw_call, dict):
                raise ValueError(f"Responses assistant message {index} contains an invalid tool call.")
            call_id = raw_call.get("id")
            function = raw_call.get("function")
            if not isinstance(call_id, str) or not call_id or not isinstance(function, dict):
                raise ValueError(f"Responses assistant message {index} contains an invalid function call.")
            name = function.get("name")
            arguments = function.get("arguments")
            if not isinstance(name, str) or not name or not isinstance(arguments, str):
                raise ValueError(f"Responses assistant message {index} contains an invalid function payload.")
            translated.append(
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                }
            )
    return translated


class ResponsesProvider:
    """Direct OpenAI Responses provider with manual, auditable state replay."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        routing: ProviderRouting,
        send_temperature: bool,
        timeout_s: float,
        decoding_seed: int | None = None,
        reasoning_effort: ReasoningEffortV1 | None = None,
        retry_policy: ProviderRetryPolicyV1 | None = None,
    ) -> None:
        _validate_decoding_seed(decoding_seed)
        if decoding_seed is not None:
            raise ValueError("The OpenAI Responses transport does not support a decoding seed.")
        if routing.provider != "openai_responses":
            raise ValueError("ResponsesProvider requires provider='openai_responses'.")
        from openai import OpenAI

        self.model = model
        self._routing = routing
        self.telemetry_route = "openai_responses"
        self._send_temperature = send_temperature
        self._retry_policy = retry_policy or ProviderRetryPolicyV1()
        self._reasoning_effort = reasoning_effort
        if timeout_s <= 0.0:
            raise ValueError("Provider request timeout must be positive.")
        self._request_timeout_s = float(timeout_s)
        client_kwargs: dict[str, Any] = {
            "timeout": self._request_timeout_s,
            "max_retries": 0,
        }
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if "http_client" not in client_kwargs:
            import httpx2

            from trialagentbench.adapters.http_transport import DeadlineTransport

            client_kwargs["http_client"] = httpx2.Client(transport=DeadlineTransport())
        self._client = OpenAI(**client_kwargs)
        self._credentials = (api_key,) if api_key is not None else ()

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
        """Generate one Responses turn and retain every output item for replay."""

        started = time.monotonic()
        input_items = responses_input_items(messages)
        remaining = None if timeout_seconds is None else timeout_seconds - (time.monotonic() - started)
        return self.generate_responses_turn(
            input_items,
            tools,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=remaining,
            tool_choice=tool_choice,
        )

    def close(self) -> None:
        """Close the underlying HTTP transport."""

        self._client.close()

    def generate_responses_turn(
        self,
        input_items: Sequence[JsonObject],
        tools: Sequence[JsonObject] | None = None,
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float | None = None,
        tool_choice: Literal["auto", "required"] = "auto",
    ) -> LLMResponse:
        """Generate from already prepared Responses input items."""

        if tool_choice == "required" and not tools:
            raise ValueError("tool_choice='required' requires at least one provider tool.")
        request_deadline = time.monotonic() + timeout_seconds if timeout_seconds is not None else None
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": list(input_items),
            "include": ["reasoning.encrypted_content"],
            "max_output_tokens": int(max_tokens),
            "store": False,
        }
        translated_tools = _responses_tools(tools)
        if translated_tools:
            kwargs["tools"] = translated_tools
            if tool_choice == "required":
                kwargs["tool_choice"] = "required"
        if self._send_temperature:
            kwargs["temperature"] = float(temperature)
        reasoning_effort = getattr(self, "_reasoning_effort", None)
        if reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": reasoning_effort}

        def request() -> Any:
            request_kwargs = dict(kwargs)
            if request_deadline is not None:
                remaining = request_deadline - time.monotonic()
                if remaining <= 0.0:
                    raise RuntimeDeadlineExceeded("Provider request deadline was exhausted before transport.")
                request_kwargs["timeout"] = min(self._request_timeout_s, remaining)
            from trialagentbench.adapters.http_transport import provider_request_deadline

            token = provider_request_deadline.set(request_deadline)
            try:
                response = self._client.responses.create(**request_kwargs)
            finally:
                provider_request_deadline.reset(token)
            if getattr(response, "error", None) is not None:
                raise TransientProviderResponseError(
                    "OpenAI Responses returned an inference error.",
                    raw_response=_raw_payload(response, credentials=getattr(self, "_credentials", ())),
                )
            status = getattr(response, "status", None)
            if status in {"failed", "cancelled"}:
                raise TransientProviderResponseError(
                    f"OpenAI Responses returned terminal status {status!r}.",
                    raw_response=_raw_payload(response, credentials=getattr(self, "_credentials", ())),
                )
            output = getattr(response, "output", None)
            if not isinstance(output, list) or not output:
                raise TransientProviderResponseError(
                    "OpenAI Responses returned no output items.",
                    raw_response=_raw_payload(response, credentials=getattr(self, "_credentials", ())),
                )
            return response

        retry_policy = _provider_retry_policy(self)
        response, retry = _retry(
            request,
            max_attempts=retry_policy.maximum_attempts,
            base_delay=retry_policy.base_delay_seconds,
            timeout_seconds=timeout_seconds,
            credentials=getattr(self, "_credentials", ()),
            unavailability_policy=retry_policy.unavailability_policy,
            request_timeout_seconds=self._request_timeout_s,
        )

        with _retain_received_response(response, credentials=getattr(self, "_credentials", ()), retry=retry):
            return self._parse_response(response, retry=retry)

    def _parse_response(self, response: Any, *, retry: RetryTelemetry) -> LLMResponse:
        """Validate SDK output envelopes while retaining the received response on failure."""

        provider_state: list[dict[str, JsonValue]] = []
        tool_calls: list[ToolCall] = []
        text_parts: list[str] = []
        outcome: Literal["refusal", "malformed_response", "empty_response"] | None = None
        for item in response.output:
            raw_item = _raw_payload(item, credentials=getattr(self, "_credentials", ()), match_patterns=False)
            if not isinstance(raw_item, dict):
                raise TypeError("Responses output-item serialization must produce a JSON object.")
            provider_state.append(raw_item)
            item_type = getattr(item, "type", None)
            if item_type == "function_call":
                call_id = getattr(item, "call_id", None)
                name = getattr(item, "name", None)
                arguments = getattr(item, "arguments", None)
                if not all(isinstance(value, str) and value.strip() for value in (call_id, name)) or call_id in {
                    call.id for call in tool_calls
                }:
                    outcome = "malformed_response"
                    break
                if not isinstance(arguments, str):
                    outcome = "malformed_response"
                    break
                tool_calls.append(
                    ToolCall(
                        id=cast(str, call_id),
                        name=cast(str, name),
                        arguments=arguments,
                    )
                )
            elif item_type == "message":
                for part in getattr(item, "content", ()):
                    part_type = getattr(part, "type", None)
                    if part_type == "output_text":
                        text = getattr(part, "text", None)
                        if not isinstance(text, str):
                            raise TypeError("Responses output text must be a string.")
                        text_parts.append(text)
                    elif part_type == "refusal":
                        refusal = getattr(part, "refusal", None)
                        if not isinstance(refusal, str):
                            raise TypeError("Responses refusal text must be a string.")
                        text_parts.append(refusal)
                        outcome = "refusal"
                    else:
                        raise TypeError(f"Unsupported Responses message content type: {part_type!r}.")
            elif item_type == "reasoning":
                if raw_item.get("encrypted_content") is None:
                    raise RuntimeError(
                        "Responses reasoning output omitted encrypted continuation state while store=false."
                    )
            else:
                raise TypeError(f"Unsupported Responses output item type: {item_type!r}.")

        content = "\n".join(part for part in text_parts if part) or None
        if content is None and not tool_calls and outcome is None:
            outcome = "empty_response"
        usage = None
        if response.usage is not None:
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
                **_detail_tokens(
                    cached=getattr(response.usage, "input_tokens_details", None),
                    reasoning=getattr(response.usage, "output_tokens_details", None),
                ),
            }
        created_at = getattr(response, "created_at", None)
        created_unix = None
        if created_at is not None:
            if not isinstance(created_at, (int, float)) or not isfinite(float(created_at)) or created_at < 0:
                raise TypeError("Responses created_at must be a non-negative numeric timestamp.")
            created_unix = int(created_at)
        status = _optional_string(getattr(response, "status", None), credentials=getattr(self, "_credentials", ()))
        incomplete = getattr(response, "incomplete_details", None)
        if incomplete is not None:
            reason = getattr(incomplete, "reason", None)
            if isinstance(reason, str) and reason:
                reason = redact_provider_text(reason, credentials=getattr(self, "_credentials", ()))
                status = f"{status}:{reason}" if status is not None else reason
        metadata = LLMResponseMetadata(
            response_id=_optional_string(getattr(response, "id", None), credentials=getattr(self, "_credentials", ())),
            returned_model=_optional_string(
                getattr(response, "model", None), credentials=getattr(self, "_credentials", ())
            ),
            finish_reason=status,
            created_unix=created_unix,
            request_attempts=retry.request_attempts,
            transient_failure_count=retry.transient_failure_count,
            backoff_seconds=retry.backoff_seconds,
            attempt_failures=retry.attempt_failures,
        )
        if content is not None:
            content = redact_provider_text(
                content, credentials=getattr(self, "_credentials", ()), match_patterns=False
            )
        for call in tool_calls:
            call.id = redact_provider_text(
                call.id, credentials=getattr(self, "_credentials", ()), match_patterns=False
            )
            call.name = redact_provider_text(
                call.name, credentials=getattr(self, "_credentials", ()), match_patterns=False
            )
            call.arguments = redact_provider_text(
                call.arguments, credentials=getattr(self, "_credentials", ()), match_patterns=False
            )
        if usage is not None:
            usage = cast(
                dict[str, JsonValue],
                redact_provider_value(
                    dict(usage), credentials=getattr(self, "_credentials", ()), match_patterns=False
                ),
            )

        raw_message: dict[str, JsonValue] = {
            "role": "assistant",
            "content": content,
            "provider_state": cast(JsonValue, provider_state),
        }
        if tool_calls:
            raw_message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments or "{}"},
                }
                for call in tool_calls
            ]
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            metadata=metadata,
            raw=raw_message,
            raw_response=_raw_payload(response, credentials=getattr(self, "_credentials", ())),
            model_outcome=outcome,
        )


def _detail_tokens(*, cached: object, reasoning: object) -> dict[str, JsonValue]:
    """Return the cached-prompt and reasoning token counts a provider reports."""
    counts = {
        "cached_tokens": getattr(cached, "cached_tokens", None),
        "reasoning_tokens": getattr(reasoning, "reasoning_tokens", None),
    }
    return {key: value for key, value in counts.items() if value is not None}


def _optional_string(value: object, *, credentials: tuple[str, ...] = ()) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TypeError(f"Provider response metadata must be a non-empty string, observed {value!r}.")
    return redact_provider_text(value, credentials=credentials)


def _optional_integer(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"Provider response timestamp must be a non-negative integer, observed {value!r}.")
    return value


def _optional_nonnegative_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Provider-reported cost must be numeric, observed {value!r}.")
    numeric = float(value)
    if not isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"Provider-reported cost must be non-negative, observed {numeric!r}.")
    return numeric


def get_provider(
    model: str,
    *,
    routing: ProviderRouting,
    send_temperature: bool,
    timeout_s: float,
    decoding_seed: int | None = None,
    reasoning_effort: ReasoningEffortV1 | None = None,
    exclude_reasoning: bool = False,
    replay_reasoning: bool = False,
    retry_policy: ProviderRetryPolicyV1 | None = None,
    memory_strategy: MemoryStrategyV1 = "bounded",
    request_control: CampaignRequestControl | None = None,
) -> LLMProvider:
    """Return the explicitly selected OpenAI-compatible provider."""
    _validate_decoding_seed(decoding_seed)
    validate_provider_configuration(model=model, routing=routing)
    if routing.provider == "offline_control":
        from trialagentbench.adapters.offline_control import OfflineControlProvider

        if reasoning_effort is not None or memory_strategy != "bounded" or replay_reasoning:
            raise ValueError("Offline controls have no reasoning or provider-managed memory")
        return OfflineControlProvider(model)
    if request_control is not None and routing.provider != "openrouter":
        raise ValueError("Campaign request reservations require the OpenRouter transport")
    if memory_strategy == "openai_responses_compaction" and routing.provider != "openai_responses":
        raise ValueError("OpenAI Responses compaction requires --provider openai_responses")
    if routing.provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set in environment")
        return ChatCompletionsProvider(
            model=model,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            routing=routing,
            send_temperature=send_temperature,
            decoding_seed=decoding_seed,
            reasoning_effort=reasoning_effort,
            exclude_reasoning=exclude_reasoning,
            replay_reasoning=replay_reasoning,
            retry_policy=retry_policy,
            timeout_s=timeout_s,
            request_control=request_control,
        )
    if routing.provider == "openai_responses":
        responses_provider = ResponsesProvider(
            model=model,
            routing=routing,
            send_temperature=send_temperature,
            decoding_seed=decoding_seed,
            reasoning_effort=reasoning_effort,
            retry_policy=retry_policy,
            timeout_s=timeout_s,
        )
        if memory_strategy == "openai_responses_compaction":
            from trialagentbench.adapters.compacting_responses import OpenAIAgentsCompactingProvider

            return OpenAIAgentsCompactingProvider(
                responses_provider,
                timeout_seconds=timeout_s,
            )
        return responses_provider
    # OpenAI direct
    return ChatCompletionsProvider(
        model=model,
        routing=routing,
        send_temperature=send_temperature,
        decoding_seed=decoding_seed,
        reasoning_effort=reasoning_effort,
        exclude_reasoning=exclude_reasoning,
        retry_policy=retry_policy,
        timeout_s=timeout_s,
    )


def close_provider(provider: LLMProvider) -> None:
    """Close a provider when its adapter owns explicit transport resources."""

    if isinstance(provider, ClosableLLMProvider):
        provider.close()
