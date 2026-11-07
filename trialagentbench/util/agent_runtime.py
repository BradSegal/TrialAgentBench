"""Shared provider-turn mechanics for TrialAgentBench analysis agents."""

from __future__ import annotations

import json
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from pydantic.types import JsonValue

from trialagentbench.contracts.core.config import ProviderModelCapabilityV1
from trialagentbench.ports import (
    CodeExecutionSession,
    LLMProvider,
    LLMResponse,
    ManagedContextProvider,
)
from trialagentbench.ports.tool_input import validate_provider_text_v1
from trialagentbench.util.provider_telemetry import generate_provider_turn_v1
from trialagentbench.util.runtime_context import RuntimeDeadline, provider_context


@dataclass(frozen=True)
class AgentTurn:
    """One provider response rendered for the persistent conversation."""

    response: LLMResponse
    message: dict[str, JsonValue]
    elapsed_seconds: float


def assistant_message_v1(response: LLMResponse) -> dict[str, JsonValue]:
    """Render a provider response as one portable assistant message."""

    if isinstance(response.raw, dict):
        return cast(dict[str, JsonValue], response.raw)
    message: dict[str, JsonValue] = {"role": "assistant", "content": response.content or None}
    if response.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments or "{}"},
            }
            for call in response.tool_calls
        ]
    return message


def offered_tool_names_v1(tools: Sequence[dict[str, JsonValue]]) -> frozenset[str]:
    """Return validated function names from provider tool declarations."""

    names: set[str] = set()
    for tool in tools:
        function = tool.get("function")
        if not isinstance(function, dict):
            raise RuntimeError("Provider tool declaration is missing a function object.")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise RuntimeError("Provider tool declaration has an invalid function name.")
        if name in names:
            raise RuntimeError(f"Provider tool declaration repeats {name!r}.")
        names.add(name)
    return frozenset(names)


def run_agent_turn_v1(
    *,
    provider: LLMProvider,
    messages: Sequence[dict[str, JsonValue]],
    session: CodeExecutionSession,
    active_prompt_index: int,
    max_context_chars: int,
    tools: Sequence[dict[str, JsonValue]],
    temperature: float,
    max_tokens: int,
    timeout_seconds: float | None,
    tool_choice: Literal["auto", "required"],
    provider_log_path: Path | None,
    benchmark: Literal["trialeval", "trialdev"],
    unit_id: str,
    phase_id: str,
    step_id: str,
    turn_index: int,
    required_message_indices: Sequence[int] = (),
    provider_capability: ProviderModelCapabilityV1 | None = None,
    deadline: RuntimeDeadline | None = None,
) -> AgentTurn:
    """Execute one bounded, logged provider turn for either benchmark."""

    offered_tool_names_v1(tools)
    with deadline.harness_work() if deadline is not None else nullcontext():
        prompt_indices = {0, active_prompt_index}
        last_assistant = max(
            (index for index, message in enumerate(messages) if message.get("role") == "assistant"), default=-1
        )
        validate_provider_text_v1(
            messages=[message for index, message in enumerate(messages) if index in prompt_indices],
            tools=tools,
            episode_messages=[
                message
                for index, message in enumerate(messages)
                if index > last_assistant and index not in prompt_indices
            ],
            provenance=f"{benchmark}:{unit_id}:{phase_id}:{step_id}:{turn_index}",
        )
        transmitted_messages = provider_context(
            messages,
            provider=provider,
            session=session,
            active_prompt_index=active_prompt_index,
            max_chars=max_context_chars,
            required_message_indices=required_message_indices,
        )
        source_context_characters = sum(
            len(json.dumps(message, ensure_ascii=True, sort_keys=True, default=str)) for message in messages
        )
        context_management: Literal["provider_managed", "local_complete", "local_archived"]
        if isinstance(provider, ManagedContextProvider) and provider.manages_context is True:
            context_management = "provider_managed"
        elif transmitted_messages == list(messages):
            context_management = "local_complete"
        else:
            context_management = "local_archived"
    result = generate_provider_turn_v1(
        provider=provider,
        messages=transmitted_messages,
        tools=tools,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        tool_choice=tool_choice,
        log_path=provider_log_path,
        benchmark=benchmark,
        unit_id=unit_id,
        phase_id=phase_id,
        step_id=step_id,
        turn_index=turn_index,
        context_management=context_management,
        source_context_messages=len(messages),
        source_context_characters=source_context_characters,
        provider_capability=provider_capability,
        deadline=deadline,
    )
    return AgentTurn(
        response=result.response,
        message=assistant_message_v1(result.response),
        elapsed_seconds=result.elapsed_seconds,
    )


__all__ = ["AgentTurn", "assistant_message_v1", "offered_tool_names_v1", "run_agent_turn_v1"]
