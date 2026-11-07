"""LLM provider port (hexagonal boundary).

Offline workflows must not import provider implementations. This module defines
the minimal provider protocol used by agent loops and optional eval-time LLM
features.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = Mapping[str, JsonValue]
ModelResponseOutcome = Literal["refusal", "malformed_response", "empty_response"]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string


@dataclass(frozen=True)
class RetryTelemetry:
    """Observed transport attempts and cumulative retry delay."""

    request_attempts: int
    transient_failure_count: int
    backoff_seconds: float
    attempt_failures: tuple[JsonObject, ...] = ()


class RuntimeDeadlineExceeded(TimeoutError):
    """A task's declared wall-clock budget expired; the task ends as model noncompletion.

    Raised by the task deadline and by provider transport bounded by that deadline. An operation's own
    ``TimeoutError`` propagates; it requires origin evidence before infrastructure retry admission.
    """


class ProviderRequestError(RuntimeError):
    """A provider request failed at the external transport boundary."""

    response_received: bool = False
    raw_response: JsonObject | None = None
    status_code: int | None = None

    def __init__(self, message: str, *, retry_telemetry: RetryTelemetry) -> None:
        super().__init__(message)
        self.retry_telemetry = retry_telemetry


class ProviderUnavailableError(ProviderRequestError):
    """An opted-in, unanswered transport failure leaves scientific work pending.

    This is not a task deadline or a returned model outcome. Accounting reservations
    remain untouched; only the explicitly recorded failed-transport clock interval
    may be excluded by ``defer_unanswered_v1``.
    """

    policy = "defer_unanswered_v1"
    response_received = False

    def __init__(
        self,
        message: str,
        *,
        retry_telemetry: RetryTelemetry,
        failure_kind: str,
        retry_after_seconds: float = 0.0,
        unanswered_transport_seconds: float | None = None,
    ) -> None:
        super().__init__(message, retry_telemetry=retry_telemetry)
        self.failure_kind = failure_kind
        self.retry_after_seconds = retry_after_seconds
        self.unanswered_transport_seconds = unanswered_transport_seconds
        self.elapsed_seconds = 0.0
        self.scientific_clock_excluded_seconds = 0.0


class ProviderAccountingError(ValueError):
    """A received response failed spending validation; its payload remains in custody."""

    def __init__(self, message: str, *, raw_response: JsonObject) -> None:
        super().__init__(message)
        self.raw_response = raw_response


class ModelResponseNoncompletion(RuntimeError):
    """A retained, paid model response cannot continue the task."""

    def __init__(self, outcome: ModelResponseOutcome) -> None:
        super().__init__(f"Retained model outcome: {outcome}.")
        self.outcome = outcome


@dataclass(frozen=True)
class LLMResponseMetadata:
    """Provider response identity, routing metadata, and retry observations."""

    response_id: str | None = None
    returned_model: str | None = None
    upstream_provider: str | None = None
    finish_reason: str | None = None
    created_unix: int | None = None
    reported_cost_usd: float | None = None
    request_attempts: int = 1
    transient_failure_count: int = 0
    backoff_seconds: float = 0.0
    attempt_failures: tuple[JsonObject, ...] = ()


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: JsonObject | None = None
    metadata: LLMResponseMetadata = field(default_factory=LLMResponseMetadata)
    raw: object = None
    # Returned reasoning text: kept in the provider log, never in the conversation or agent context.
    reasoning: str | None = None
    # Complete provider JSON for diagnostics, separate from the replayable assistant message.
    raw_response: JsonObject | None = None
    model_outcome: ModelResponseOutcome | None = None


class LLMProvider(Protocol):
    @property
    def model(self) -> str: ...

    @property
    def telemetry_route(self) -> str: ...

    def generate_turn(
        self,
        messages: Sequence[JsonObject],
        tools: Sequence[JsonObject] | None = None,
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float | None = None,
        tool_choice: Literal["auto", "required"] = "auto",
    ) -> LLMResponse: ...


@runtime_checkable
class ManagedContextProvider(Protocol):
    """Provider that owns persistent context for exactly one conversation."""

    @property
    def manages_context(self) -> Literal[True]: ...


@runtime_checkable
class ClosableLLMProvider(Protocol):
    """Provider with an explicit transport lifecycle."""

    def close(self) -> None: ...
