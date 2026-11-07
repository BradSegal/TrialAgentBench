"""Tests for OpenAI Agents SDK Responses compaction."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from trialagentbench.adapters.compacting_responses import OpenAIAgentsCompactingProvider
from trialagentbench.ports import LLMResponse
from trialagentbench.util.runtime_context import provider_context


class _FakeCompactEndpoint:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def compact(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(kwargs)
        return SimpleNamespace(
            output=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Retained result: adjusted effect = 7."}],
                }
            ],
            usage=SimpleNamespace(
                input_tokens=12,
                output_tokens=3,
                total_tokens=15,
                input_tokens_details={"cached_tokens": 0, "cache_write_tokens": 0},
                output_tokens_details={"reasoning_tokens": 0},
            ),
        )


class _FakeClient:
    def __init__(self) -> None:
        self.responses = _FakeCompactEndpoint()


class _RecordingTransport:
    model = "gpt-5.6"

    def __init__(self) -> None:
        self.inputs: list[list[dict[str, object]]] = []
        self.timeouts: list[object] = []

    def generate_responses_turn(
        self,
        input_items: object,
        tools: object,
        **kwargs: object,
    ) -> LLMResponse:
        del tools
        self.timeouts.append(kwargs["timeout_seconds"])
        assert isinstance(input_items, list)
        self.inputs.append(input_items)
        return LLMResponse(
            content="Use the retained adjusted effect of 7.",
            usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        )

    def close(self) -> None:
        """Close the fake transport."""

        return None


def test_forced_compaction_preserves_prior_analysis_and_accounts_for_usage() -> None:
    transport = _RecordingTransport()
    client = _FakeClient()
    provider = OpenAIAgentsCompactingProvider(
        transport,
        client=cast(Any, client),
        minimum_candidate_items=1,
        session_id="continuity-test",
    )

    response = provider.generate_turn(
        [
            {"role": "system", "content": "Analyse the trial."},
            {"role": "user", "content": "The adjusted effect was 7. Retain it for the decision."},
        ],
        temperature=0.0,
        max_tokens=128,
    )

    assert len(client.responses.requests) == 1
    assert "Retained result: adjusted effect = 7." in str(transport.inputs[0])
    assert response.content == "Use the retained adjusted effect of 7."
    assert response.usage == {
        "prompt_tokens": 17,
        "completion_tokens": 5,
        "total_tokens": 22,
        "model_prompt_tokens": 5,
        "model_completion_tokens": 2,
        "compaction_events": 1,
        "compaction_requests": 1,
        "compaction_input_tokens": 12,
        "compaction_output_tokens": 3,
    }
    provider.close()
    provider.close()
    with pytest.raises(RuntimeError, match="closed"):
        provider.generate_turn(
            [{"role": "user", "content": "Continue."}],
            temperature=0.0,
            max_tokens=16,
        )


def test_compacting_provider_rejects_reuse_for_another_conversation() -> None:
    provider = OpenAIAgentsCompactingProvider(
        _RecordingTransport(),
        client=cast(Any, _FakeClient()),
        minimum_candidate_items=10,
        session_id="single-conversation-test",
    )
    provider.generate_turn(
        [{"role": "system", "content": "First."}, {"role": "user", "content": "Begin."}],
        temperature=0.0,
        max_tokens=16,
    )

    with pytest.raises(RuntimeError, match="divergent conversations"):
        provider.generate_turn(
            [{"role": "system", "content": "Second."}, {"role": "user", "content": "Begin."}],
            temperature=0.0,
            max_tokens=16,
        )
    provider.close()


def test_managed_context_bypasses_local_character_truncation() -> None:
    provider = OpenAIAgentsCompactingProvider(
        _RecordingTransport(),
        client=cast(Any, _FakeClient()),
        minimum_candidate_items=10,
        session_id="context-owner-test",
    )
    messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Prompt"},
        {"role": "assistant", "content": "Earlier reasoning"},
    ]

    assert (
        provider_context(
            messages,
            provider=provider,
            session=SimpleNamespace(),  # unused when the provider owns context
            active_prompt_index=1,
            max_chars=1,
        )
        == messages
    )
    provider.close()


def test_managed_compaction_keeps_current_reminder_and_notice_and_consumes_request_time(monkeypatch):
    from trialagentbench.adapters import compacting_responses as owner
    from trialagentbench.util.runtime_context import FINAL_TURN_SUBMISSION_REMINDER

    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(owner.time, "monotonic", lambda: clock.now)
    transport = _RecordingTransport()
    client = _FakeClient()
    compact = client.responses.compact

    async def slow_compact(**kwargs):
        clock.now += 20
        return await compact(**kwargs)

    client.responses.compact = slow_compact
    notice = "[Agent-time budget: 100 seconds total; 100 seconds remaining.]"
    provider = OpenAIAgentsCompactingProvider(
        transport, client=cast(Any, client), minimum_candidate_items=1, session_id="required-messages-test"
    )
    try:
        provider.generate_turn(
            [
                {"role": "system", "content": "Analyse the trial."},
                {"role": "user", "content": "Initial task"},
                {"role": "assistant", "content": "Earlier analysis"},
                {"role": "user", "content": FINAL_TURN_SUBMISSION_REMINDER},
                {"role": "user", "content": notice},
            ],
            temperature=0,
            max_tokens=128,
            timeout_seconds=100,
        )
        assert len(client.responses.requests) == 1
        assert FINAL_TURN_SUBMISSION_REMINDER in str(transport.inputs[0])
        assert notice in str(transport.inputs[0])
        assert notice not in str(client.responses.requests[0])
        assert transport.timeouts == [80]
    finally:
        provider.close()


def test_managed_compaction_cannot_overrun_the_agent_watchdog():
    import asyncio

    from trialagentbench.ports import RuntimeDeadlineExceeded

    transport = _RecordingTransport()
    client = _FakeClient()

    async def blocked_compact(**kwargs):
        await asyncio.sleep(10)
        raise AssertionError("Compaction was not cancelled")

    client.responses.compact = blocked_compact
    provider = OpenAIAgentsCompactingProvider(
        transport, client=cast(Any, client), minimum_candidate_items=1, session_id="compaction-timeout-test"
    )
    try:
        with pytest.raises(RuntimeDeadlineExceeded, match="compaction"):
            provider.generate_turn(
                [{"role": "system", "content": "Analyse."}, {"role": "user", "content": "Submit."}],
                temperature=0,
                max_tokens=128,
                timeout_seconds=0.01,
            )
        assert not transport.inputs
    finally:
        provider.close()


def test_failed_managed_compaction_does_not_duplicate_history_on_retry():
    transport = _RecordingTransport()
    client = _FakeClient()
    compact = client.responses.compact
    calls = []

    async def interrupted_compact(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise OSError("Compaction interrupted")
        return await compact(**kwargs)

    client.responses.compact = interrupted_compact
    provider = OpenAIAgentsCompactingProvider(
        transport, client=cast(Any, client), minimum_candidate_items=1, session_id="retry-compaction-test"
    )
    messages = [
        {"role": "system", "content": "Analyse."},
        {"role": "user", "content": "Initial task"},
        {"role": "assistant", "content": "Earlier analysis"},
        {"role": "user", "content": "[Agent-time budget: 80 seconds remaining.]"},
    ]
    try:
        with pytest.raises(OSError, match="Compaction interrupted"):
            provider.generate_turn(messages, temperature=0, max_tokens=128)
        provider.generate_turn(messages, temperature=0, max_tokens=128)
        assert len(calls) == 2
        assert str(calls[1]).count("Earlier analysis") == 1
        assert "80 seconds remaining" in str(transport.inputs[0])
    finally:
        provider.close()
