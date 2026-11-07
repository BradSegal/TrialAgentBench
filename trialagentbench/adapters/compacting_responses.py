"""Responses transport with OpenAI Agents SDK session compaction."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast
from uuid import uuid4

from trialagentbench.ports.llm_provider import JsonObject, JsonValue, LLMResponse, RuntimeDeadlineExceeded

if TYPE_CHECKING:
    from openai import AsyncOpenAI

DEFAULT_COMPACTION_CANDIDATE_ITEMS = 10


class ResponsesTurnTransport(Protocol):
    """Transport required after a Responses input history is prepared."""

    @property
    def model(self) -> str: ...

    def generate_responses_turn(
        self,
        input_items: Sequence[JsonObject],
        tools: Sequence[JsonObject] | None,
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float | None,
        tool_choice: Literal["auto", "required"],
    ) -> LLMResponse: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class CompactionTelemetry:
    """Cumulative observations from SDK-managed compaction requests."""

    events: int = 0
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class OpenAIAgentsCompactingProvider:
    """Add SDK-managed Responses compaction to one participant conversation.

    The benchmark continues to own tools, turns, submissions, and grading. This
    adapter owns only the provider context carried between those turns.
    """

    telemetry_route = "openai_responses_compaction"
    manages_context = True

    def __init__(
        self,
        transport: ResponsesTurnTransport,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 300.0,
        minimum_candidate_items: int = DEFAULT_COMPACTION_CANDIDATE_ITEMS,
        client: AsyncOpenAI | None = None,
        session_id: str | None = None,
    ) -> None:
        if minimum_candidate_items < 1:
            raise ValueError("minimum_candidate_items must be positive")
        if timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        try:
            from agents.memory import OpenAIResponsesCompactionSession, SQLiteSession
            from openai import AsyncOpenAI
        except ImportError as error:
            raise ImportError(
                "OpenAI Responses compaction requires the 'providers' extra: "
                "from the checkout, pip install ./clinical-data ./verifier '.[providers]'."
            ) from error

        async_client = client
        if async_client is None:
            async_client = AsyncOpenAI(api_key=api_key, timeout=timeout_seconds, max_retries=0)

        self.model = transport.model
        self._transport = transport
        self._minimum_candidate_items = minimum_candidate_items
        self._triggered = False
        self._source_hashes: list[str] = []
        self._telemetry = CompactionTelemetry()
        self._runner = asyncio.Runner()
        self._owns_client = client is None
        self._closed = False
        self._async_client = async_client

        def should_compact(state: dict[str, Any]) -> bool:
            candidates = state.get("compaction_candidate_items")
            if not isinstance(candidates, list):
                raise TypeError("Agents SDK compaction candidates must be a list")
            self._triggered = len(candidates) >= self._minimum_candidate_items
            return self._triggered

        resolved_session_id = session_id or f"trialagentbench-{uuid4().hex}"
        underlying = SQLiteSession(resolved_session_id)
        self._session = OpenAIResponsesCompactionSession(
            resolved_session_id,
            underlying,
            client=async_client,
            model=self.model,
            compaction_mode="input",
            should_trigger_compaction=should_compact,
        )

    @property
    def telemetry(self) -> CompactionTelemetry:
        """Return cumulative compaction usage for this conversation."""

        return self._telemetry

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
        """Compact accumulated history when due, then generate one model turn."""

        if self._closed:
            raise RuntimeError("The compacting provider is closed")
        started = time.monotonic()
        try:
            input_items, delta = self._runner.run(asyncio.wait_for(self._prepare(messages), timeout=timeout_seconds))
        except TimeoutError as error:
            if timeout_seconds is None:
                raise
            raise RuntimeDeadlineExceeded("Agent time exhausted during provider context compaction.") from error
        remaining = None if timeout_seconds is None else timeout_seconds - (time.monotonic() - started)
        if remaining is not None and remaining <= 0:
            raise RuntimeDeadlineExceeded("Agent time exhausted during provider context compaction.")
        response = self._transport.generate_responses_turn(
            input_items,
            tools,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=remaining,
            tool_choice=tool_choice,
        )
        if delta.events:
            usage: dict[str, JsonValue] = dict(response.usage or {})
            model_input = _usage_integer(usage, "prompt_tokens")
            model_output = _usage_integer(usage, "completion_tokens")
            usage.update(
                {
                    "prompt_tokens": model_input + delta.input_tokens,
                    "completion_tokens": model_output + delta.output_tokens,
                    "total_tokens": model_input + model_output + delta.input_tokens + delta.output_tokens,
                    "model_prompt_tokens": model_input,
                    "model_completion_tokens": model_output,
                    "compaction_events": delta.events,
                    "compaction_requests": delta.requests,
                    "compaction_input_tokens": delta.input_tokens,
                    "compaction_output_tokens": delta.output_tokens,
                }
            )
            response.usage = usage
        return response

    def close(self) -> None:
        """Close synchronous and asynchronous provider resources."""

        if self._closed:
            return
        if self._owns_client:
            self._runner.run(self._async_client.close())
        self._runner.close()
        self._transport.close()
        self._closed = True

    async def _prepare(self, messages: Sequence[JsonObject]) -> tuple[list[JsonObject], CompactionTelemetry]:
        from agents.run_context import RunContextWrapper

        from trialagentbench.adapters.llm_providers import responses_input_items

        hashes = [_message_identity(message) for message in messages]
        if len(hashes) < len(self._source_hashes) or hashes[: len(self._source_hashes)] != self._source_hashes:
            raise RuntimeError("A compacting provider cannot be reused across divergent conversations")
        new_messages = messages[len(self._source_hashes) :]
        # New task instructions, terminal reminders and time notices must reach the
        # model verbatim. Compact history before appending the current user suffix.
        suffix_start = len(new_messages)
        while suffix_start and new_messages[suffix_start - 1].get("role") == "user":
            suffix_start -= 1
        if history := new_messages[:suffix_start]:
            await self._session.add_items(cast(Any, responses_input_items(history)))
            self._source_hashes = hashes[: len(hashes) - len(new_messages) + suffix_start]

        wrapper = RunContextWrapper(context=None)
        self._triggered = False
        await self._session.run_compaction({"store": False}, wrapper=wrapper)
        delta = CompactionTelemetry()
        if self._triggered:
            usage = wrapper.usage
            delta = CompactionTelemetry(
                events=1,
                requests=usage.requests,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            )
            self._telemetry = CompactionTelemetry(
                events=self._telemetry.events + delta.events,
                requests=self._telemetry.requests + delta.requests,
                input_tokens=self._telemetry.input_tokens + delta.input_tokens,
                output_tokens=self._telemetry.output_tokens + delta.output_tokens,
            )
        if current_prompts := new_messages[suffix_start:]:
            await self._session.add_items(cast(Any, responses_input_items(current_prompts)))
        self._source_hashes = hashes
        raw_items = await self._session.get_items()
        return [_json_object(item) for item in raw_items], delta


def _message_identity(message: Mapping[str, JsonValue]) -> str:
    serialized = json.dumps(message, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _json_object(value: object) -> JsonObject:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError("Agents SDK session items must serialize to JSON objects")
    return value


def _usage_integer(usage: Mapping[str, JsonValue], name: str) -> int:
    value = usage.get(name, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{name} must be a non-negative integer")
    return value


__all__ = [
    "CompactionTelemetry",
    "DEFAULT_COMPACTION_CANDIDATE_ITEMS",
    "OpenAIAgentsCompactingProvider",
]
