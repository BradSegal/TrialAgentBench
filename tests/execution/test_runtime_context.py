from __future__ import annotations

import ast
import base64
import json
import re
import time
from typing import Literal

import pytest

from trialagentbench.ports.code_execution import CodeExecutionResultV1
from trialagentbench.util.runtime_context import (
    RuntimeDeadline,
    RuntimeDeadlineExceeded,
    bounded_provider_context,
    persist_bulky_tool_output,
    turn_budget_tag,
)


class _Session:
    def __init__(
        self,
        *,
        status: Literal["success", "session_terminated"] = "success",
    ) -> None:
        self.files: dict[str, str] = {}
        self.status = status

    def execute_result(self, code: str) -> CodeExecutionResultV1:
        expression = ast.parse(code).body[0]
        assert isinstance(expression, ast.Expr)
        assert isinstance(expression.value, ast.Call)
        script = ast.literal_eval(expression.value.args[0])
        path_match = re.search(r"Path\('scratch'\) / ('.*?')", script)
        content_match = re.search(r"base64\.b64decode\(('.*?')\)", script)
        if self.status == "success":
            assert path_match is not None
            assert content_match is not None
            path = ast.literal_eval(path_match.group(1))
            encoded = ast.literal_eval(content_match.group(1))
            self.files[path] = base64.b64decode(encoded).decode("utf-8")
        return CodeExecutionResultV1(
            status=self.status,
            output="" if self.status == "success" else "session unavailable",
            elapsed_seconds=0.0,
        )

    def execute(self, code: str) -> str:
        return self.execute_result(code).output

    def close(self) -> None:
        return None


def _tool_unit(call_id: str, output: str) -> list[dict[str, object]]:
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": "execute_code", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": output},
    ]


def test_context_moves_complete_assistant_tool_pairs_to_earlier_turns_file() -> None:
    session = _Session()
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        *_tool_unit("call-1", "old evidence " * 10),
        *_tool_unit("call-2", "new evidence " * 10),
    ]

    context = bounded_provider_context(messages, session=session, active_prompt_index=1, max_chars=500)

    assert context[:2] == messages[:2]
    assert "scratch/earlier_turns.json" in str(context[2]["content"])
    assert context[3:] == messages[4:]
    assert json.loads(session.files["earlier_turns.json"]) == messages[2:4]
    assert not any(message.get("role") == "assistant" and "summary" in message for message in context)


def test_context_prefix_is_stable_between_compactions() -> None:
    """Each turn extends the previous transmitted context until one compaction moves the boundary."""
    session = _Session()
    messages: list[dict[str, object]] = [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}]
    contexts = []
    for turn in range(30):
        messages += _tool_unit(f"call-{turn}", f"evidence {turn} " * 20)
        contexts.append(bounded_provider_context(messages, session=session, active_prompt_index=1, max_chars=6_000))
    compactions = 0
    for previous, current in zip(contexts, contexts[1:], strict=False):
        if current[: len(previous)] != previous:
            compactions += 1
    assert 1 <= compactions <= 5  # a sliding window would rewrite the prefix on every turn after the first
    assert all(context[:2] == messages[:2] for context in contexts)


def test_context_rejects_unmatched_or_duplicate_tool_call_ids() -> None:
    session = _Session()
    unmatched = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "current"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "tool", "arguments": "{}"},
                }
            ],
        },
    ]
    with pytest.raises(ValueError, match="do not match"):
        bounded_provider_context(
            unmatched,
            session=session,
            active_prompt_index=1,
            max_chars=1000,
        )

    duplicate = [
        *unmatched[:2],
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "tool", "arguments": "{}"},
                },
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "tool", "arguments": "{}"},
                },
            ],
        },
    ]
    with pytest.raises(ValueError, match="Duplicate assistant tool_call_id"):
        bounded_provider_context(
            duplicate,
            session=session,
            active_prompt_index=1,
            max_chars=1000,
        )


def test_context_allows_provider_call_ids_to_repeat_on_later_turns() -> None:
    session = _Session()
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "current"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "provider-call",
                    "type": "function",
                    "function": {"name": "tool", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "provider-call", "content": "first"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "provider-call",
                    "type": "function",
                    "function": {"name": "tool", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "provider-call", "content": "second"},
    ]

    context = bounded_provider_context(
        messages,
        session=session,
        active_prompt_index=1,
        max_chars=10_000,
    )

    assert context == messages


def test_context_retains_required_terminal_message_beyond_soft_budget() -> None:
    session = _Session()
    final_reminder = {"role": "user", "content": "submit now"}
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "active prompt " * 100},
        {"role": "assistant", "content": "recent analysis " * 100},
        final_reminder,
    ]

    context = bounded_provider_context(
        messages,
        session=session,
        active_prompt_index=1,
        max_chars=20,
        required_message_indices=(3,),
    )

    assert final_reminder in context
    assert messages[2] not in context


def test_bulky_output_is_losslessly_persisted_with_head_tail_and_one_path() -> None:
    session = _Session()
    output = "head-" + "evidence-" * 100 + "-tail"

    rendered = persist_bulky_tool_output(output, session=session, artifact_id="turn-3-1", inline_chars=50)

    assert session.files == {"outputs/turn-3-1.txt": output}
    assert "scratch/outputs/turn-3-1.txt" in rendered
    assert rendered.startswith("head-") and rendered.endswith("-tail")


def test_runtime_workspace_persistence_fails_loudly() -> None:
    session = _Session(status="session_terminated")

    with pytest.raises(RuntimeError, match="Unable to persist runtime workspace artifact"):
        persist_bulky_tool_output(
            "evidence-" * 100,
            session=session,
            artifact_id="turn-1-1",
            inline_chars=20,
        )


def test_turn_budget_tag_is_bounded_and_suite_independent() -> None:
    assert turn_budget_tag(turn=1, maximum=3) == "\n[turn 1/3, 2 left]"
    assert turn_budget_tag(turn=3, maximum=3) == "\n[turn 3/3, 0 left]"

    with pytest.raises(ValueError, match="within"):
        turn_budget_tag(turn=4, maximum=3)


def test_blocking_operation_is_capped_to_remaining_deadline() -> None:
    deadline = RuntimeDeadline.after(0.02, label="test operation")
    timed_out: list[bool] = []

    with pytest.raises(RuntimeDeadlineExceeded, match="blocking"):
        deadline.run_blocking(
            lambda: time.sleep(0.2),
            operation_name="slow tool",
            on_timeout=lambda: timed_out.append(True),
        )

    assert timed_out == [True]


def test_expired_deadline_raises_watchdog_error_without_starting_work() -> None:
    deadline = RuntimeDeadline(monotonic=0.0, label="expired task")
    with pytest.raises(RuntimeDeadlineExceeded, match="wall-clock budget"):
        deadline.remaining()


def test_operation_timeout_is_not_a_task_watchdog_failure() -> None:
    """A callable's own timeout retains its cause and must not trigger watchdog cancellation."""
    deadline = RuntimeDeadline.after(1.0, label="test operation")
    original = TimeoutError("provider timed out before task deadline")
    cancelled = []

    def fail():
        raise original

    with pytest.raises(TimeoutError) as caught:
        deadline.run_blocking(fail, operation_name="provider operation", on_timeout=lambda: cancelled.append(True))
    assert caught.value is original
    assert not isinstance(caught.value, RuntimeDeadlineExceeded)
    assert cancelled == []


def test_blocking_operation_propagates_the_original_failure() -> None:
    """Unexpected operation failures retain their type and message."""

    deadline = RuntimeDeadline.after(1.0, label="test operation")

    def fail() -> None:
        raise LookupError("provider response is malformed")

    with pytest.raises(LookupError, match="provider response is malformed"):
        deadline.run_blocking(fail, operation_name="failing operation")


def test_harness_work_is_not_charged_to_the_agent_deadline() -> None:
    """Blocking harness work outlasts the agent's budget, which then resumes where it paused."""

    deadline = RuntimeDeadline.after(0.1, label="test operation")
    with deadline.harness_work():
        assert deadline.run_blocking(lambda: time.sleep(0.3) or "done", operation_name="replay") == "done"
    remaining = deadline.remaining()
    assert remaining is not None and 0.0 < remaining <= 0.1
    time.sleep(0.15)
    with pytest.raises(TimeoutError, match="exceeded"):
        deadline.remaining()
