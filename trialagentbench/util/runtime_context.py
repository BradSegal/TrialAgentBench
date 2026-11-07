"""Shared deadline and lossless provider-context utilities."""

from __future__ import annotations

import base64
import json
import math
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal, TypeVar

from trialagentbench.ports.code_execution import CodeExecutionSession
from trialagentbench.ports.llm_provider import (
    JsonObject,
    LLMProvider,
    ManagedContextProvider,
    RuntimeDeadlineExceeded,
)

R = TypeVar("R")

EARLIER_TURNS_FILE = "scratch/earlier_turns.json"

FINAL_TURN_SUBMISSION_REMINDER = (
    "This is the final available turn. Submit the best complete response supported by your executed work now. "
    "Do not start another analysis step; an unsubmitted response is recorded as missing."
)
SUBMISSION_WINDOW_REMINDER = (
    "The analysis-tool window is now closed. Use the remaining turns to submit the best complete response "
    "supported by your executed work; an unsubmitted response is recorded as missing."
)


def turn_budget_tag(*, turn: int, maximum: int) -> str:
    """Render suite-independent remaining-turn metadata."""

    if maximum < 1:
        raise ValueError("Maximum turns must be positive.")
    if not 1 <= turn <= maximum:
        raise ValueError("Current turn must be within the declared turn budget.")
    return f"\n[turn {turn}/{maximum}, {maximum - turn} left]"


@dataclass
class RuntimeDeadline:
    """One monotonic agent-time budget shared by transport and agent-run tools.

    Harness-owned work (bookkeeping, saved-script replays, readouts and decisions) runs inside
    ``harness_work``; its wall time is not charged to the agent's budget.
    """

    monotonic: float | None
    label: str
    budget_seconds: float | None = None
    provider_unavailability_policy: Literal["charge_all_v1", "defer_unanswered_v1"] = "charge_all_v1"
    _paused_at: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate finite deadline and allowance values."""

        if self.monotonic is not None and not math.isfinite(self.monotonic):
            raise ValueError("Deadline must be finite when provided.")
        if self.budget_seconds is not None and (not math.isfinite(self.budget_seconds) or self.budget_seconds <= 0.0):
            raise ValueError("Deadline duration must be finite and positive when provided.")
        if self.provider_unavailability_policy not in {"charge_all_v1", "defer_unanswered_v1"}:
            raise ValueError("Unknown provider-unavailability clock policy.")

    @classmethod
    def after(
        cls,
        seconds: float | None,
        *,
        label: str,
        provider_unavailability_policy: Literal["charge_all_v1", "defer_unanswered_v1"] = "charge_all_v1",
    ) -> RuntimeDeadline:
        """Construct a deadline from a positive duration."""

        if seconds is not None and seconds <= 0.0:
            raise ValueError("Deadline duration must be positive when provided.")
        return cls(
            monotonic=time.monotonic() + seconds if seconds is not None else None,
            label=label,
            budget_seconds=seconds,
            provider_unavailability_policy=provider_unavailability_policy,
        )

    def exclude_unanswered_transport(self, elapsed_seconds: float) -> None:
        """Exclude only a recorded opted-in failed request, never refill prior work.

        Called once at the provider boundary before persisting a deferred checkpoint.
        Successful response latency, interactive tools and task-deadline cutoffs retain
        their usual charge. The caller records the excluded interval in request telemetry.
        """
        if self.provider_unavailability_policy != "defer_unanswered_v1":
            raise ValueError("The frozen clock policy does not exclude unavailable-provider time.")
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0.0 or self._paused_at is not None:
            raise ValueError("Provider clock exclusion requires a finite active request interval.")
        if self.monotonic is not None:
            self.monotonic += elapsed_seconds

    def snapshot_remaining(self) -> float | None:
        """Record the remaining budget, including zero, without failing during cleanup."""

        if self.monotonic is None:
            return None
        now = time.monotonic() if self._paused_at is None else self._paused_at
        return max(0.0, self.monotonic - now)

    def restore_remaining(self, seconds: float | None) -> None:
        """Resume a saved allowance without charging the time the process was stopped."""

        if seconds is not None and (not math.isfinite(seconds) or seconds < 0.0):
            raise ValueError("Saved remaining agent time must be finite and nonnegative.")
        if (seconds is None) != (self.monotonic is None):
            raise ValueError("Saved and requested agent-time limits differ.")
        if self.budget_seconds is not None and seconds is not None and seconds > self.budget_seconds:
            raise ValueError("Saved remaining agent time exceeds the configured allowance.")
        self.monotonic = None if seconds is None else time.monotonic() + seconds
        self._paused_at = None

    def model_notice(self) -> str | None:
        """Disclose the current allowance immediately before a model request."""

        remaining = self.remaining()
        if remaining is None:
            return None
        total = "" if self.budget_seconds is None else f"{self.budget_seconds:g} seconds total; "
        availability = (
            "Under defer_unanswered_v1, a failed transport request with no returned model response is deferred; "
            "its recorded waiting time is excluded. Successful response waiting time still counts. "
            if self.provider_unavailability_policy == "defer_unanswered_v1"
            else ""
        )
        return (
            f"[Agent-time budget: {total}{math.floor(remaining)} seconds remaining. "
            "Model requests and analysis/workspace tools consume this time. Harness-owned analysis checks, "
            "saved-script replays, all programme-tool execution (previews, study starts, adaptations, waiting, "
            "readouts and terminal decisions) and local bookkeeping do not. This is separate from simulated programme days "
            "and the turn limit; time can expire before all turns are used. "
            f"{availability}Submit your response or terminal development decision before time runs out.]"
        )

    def remaining(self) -> float | None:
        """Return remaining agent seconds, failing once no agent work may start."""

        remaining = self.snapshot_remaining()
        if remaining is not None and remaining <= 0.0:
            raise RuntimeDeadlineExceeded(f"{self.label} exceeded its declared wall-clock budget.")
        return remaining

    @contextmanager
    def harness_work(self) -> Iterator[None]:
        """Run harness-owned work without charging its wall time to the agent.

        The deadline moves later by the work's duration. Blocking operations inside
        are not cut off at the agent's deadline; their own execution limits bound them.
        """

        deadline = self.monotonic
        if deadline is None or self._paused_at is not None:
            yield
            return
        started = self._paused_at = time.monotonic()
        try:
            yield
        finally:
            self.monotonic = deadline + time.monotonic() - started
            self._paused_at = None

    def run_blocking(
        self,
        operation: Callable[[], R],
        *,
        operation_name: str,
        on_timeout: Callable[[], None] | None = None,
    ) -> R:
        """Run a local blocking operation for no longer than the remaining agent budget."""

        remaining = self.remaining()
        if remaining is None or self._paused_at is not None:
            return operation()
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"trialagentbench-{operation_name}")
        future = executor.submit(operation)
        try:
            completed, _ = wait((future,), timeout=remaining)
            if not completed:
                if on_timeout is not None:
                    on_timeout()
                future.cancel()
                raise RuntimeDeadlineExceeded(f"{self.label} exhausted while blocking on {operation_name}.")
            # A completed operation may raise its own TimeoutError; that is not this task's deadline.
            return future.result()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)


def bounded_provider_context(
    messages: Sequence[JsonObject],
    *,
    session: CodeExecutionSession,
    active_prompt_index: int,
    max_chars: int,
    required_message_indices: Sequence[int] = (),
) -> list[JsonObject]:
    """Build a prefix-stable bounded context without splitting tool-call pairs.

    The transmitted context is append-only between compactions, so provider prompt
    caches stay valid. When the conversation outgrows ``max_chars``, the oldest
    unpinned interaction units are moved to ``scratch/earlier_turns.json`` until the
    context is at most half the budget; that boundary then stays fixed until the
    budget is exceeded again. The boundary is replayed from the messages alone, so
    the same conversation always yields the same context.
    """

    if max_chars < 1:
        raise ValueError("max_chars must be at least 1.")
    if not messages or messages[0].get("role") != "system":
        raise ValueError("Provider context must start with one system message.")
    if not 0 <= active_prompt_index < len(messages):
        raise ValueError("active_prompt_index is outside the conversation.")
    if messages[active_prompt_index].get("role") != "user":
        raise ValueError("The active-step prompt must be a user message.")
    if any(not 0 <= index < len(messages) for index in required_message_indices):
        raise ValueError("A required message index is outside the conversation.")

    units = _complete_message_units(messages)
    required_indices = {0, active_prompt_index, *required_message_indices}
    pinned = {index for index, unit in enumerate(units) if required_indices.intersection(unit)}
    sizes = [sum(_message_chars(messages[index]) for index in unit) for unit in units]
    dropped: set[int] = set()
    kept = 0
    for position, size in enumerate(sizes):
        kept += size
        if kept <= max_chars:
            continue
        # Compact once to half the budget, oldest unpinned units first; never drop the newest unit.
        for candidate in range(position):
            if kept <= max_chars // 2:
                break
            if candidate not in pinned and candidate not in dropped:
                dropped.add(candidate)
                kept -= sizes[candidate]

    if not dropped:
        return list(messages)
    omitted = [messages[index] for unit_index in sorted(dropped) for index in units[unit_index]]
    _write_session_text(
        session=session,
        relative_path=EARLIER_TURNS_FILE.removeprefix("scratch/"),
        content=json.dumps(omitted, indent=1, ensure_ascii=True, default=str) + "\n",
    )
    note: JsonObject = {
        "role": "user",
        "content": (
            f"[{len(omitted)} earlier messages were moved out of this conversation to {EARLIER_TURNS_FILE}; "
            "read that file if you need them.]"
        ),
    }
    context: list[JsonObject] = []
    for unit_index, unit in enumerate(units):
        if unit_index in dropped:
            continue
        for index in unit:
            context.append(messages[index])
            if index == active_prompt_index:
                context.append(note)
    return context


def provider_context(
    messages: Sequence[JsonObject],
    *,
    provider: LLMProvider,
    session: CodeExecutionSession,
    active_prompt_index: int,
    max_chars: int,
    required_message_indices: Sequence[int] = (),
) -> list[JsonObject]:
    """Select the one context owner configured for this conversation."""

    if isinstance(provider, ManagedContextProvider) and provider.manages_context is True:
        return list(messages)
    return bounded_provider_context(
        messages,
        session=session,
        active_prompt_index=active_prompt_index,
        max_chars=max_chars,
        required_message_indices=required_message_indices,
    )


def persist_bulky_tool_output(
    output: str,
    *,
    session: CodeExecutionSession,
    artifact_id: str,
    inline_chars: int,
) -> str:
    """Save an oversized tool output and return its head and tail with the saved path.

    ``artifact_id`` is a short readable name (for example ``turn-12-1``); the full
    output is saved at ``scratch/outputs/<artifact_id>.txt``, a path that works in
    analysis code and in the workspace file tools alike.
    """

    if inline_chars < 1:
        raise ValueError("inline_chars must be at least 1.")
    if len(output) <= inline_chars:
        return output
    if not artifact_id or any(not (character.isalnum() or character in "-_") for character in artifact_id):
        raise ValueError("artifact_id must be a short filename component.")
    relative = f"outputs/{artifact_id}.txt"
    _write_session_text(session=session, relative_path=relative, content=output)
    head = output[: inline_chars * 3 // 5]
    tail = output[len(output) - inline_chars * 2 // 5 :]
    return (
        f"{head}\n\n[... {len(output) - len(head) - len(tail):,} characters omitted; "
        f"full output saved to scratch/{relative} ...]\n\n{tail}"
    )


def _write_session_text(
    *,
    session: CodeExecutionSession,
    relative_path: str,
    content: str,
) -> None:
    """Write runtime-owned text through the authoritative isolated workspace."""

    encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
    script = (
        "import base64\n"
        "from pathlib import Path\n"
        f"_path = Path('scratch') / {relative_path!r}\n"
        "_path.parent.mkdir(parents=True, exist_ok=True)\n"
        f"_path.write_bytes(base64.b64decode({encoded!r}))"
    )
    result = session.execute_result(f"exec({script!r}, {{}})")
    if result.status != "success":
        raise RuntimeError(
            f"Unable to persist runtime workspace artifact {relative_path!r}: {result.output or result.status}"
        )


def _complete_message_units(messages: Sequence[JsonObject]) -> list[tuple[int, ...]]:
    units: list[tuple[int, ...]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        role = message.get("role")
        raw_calls = message.get("tool_calls")
        if role == "tool":
            raise ValueError(f"Unmatched tool response at conversation index {index}.")
        if role != "assistant" or not isinstance(raw_calls, list) or not raw_calls:
            units.append((index,))
            index += 1
            continue
        call_ids: list[str] = []
        for call in raw_calls:
            if not isinstance(call, dict) or not isinstance(raw_call_id := call.get("id"), str):
                raise ValueError(f"Invalid assistant tool call at conversation index {index}.")
            call_id = raw_call_id
            if call_id in call_ids:
                raise ValueError(f"Duplicate assistant tool_call_id in conversation: {call_id}.")
            call_ids.append(call_id)
        end = index + 1
        response_ids: list[str] = []
        while end < len(messages) and messages[end].get("role") == "tool":
            response_id = messages[end].get("tool_call_id")
            if not isinstance(response_id, str):
                raise ValueError(f"Tool response at conversation index {end} has no tool_call_id.")
            response_ids.append(response_id)
            end += 1
        if len(response_ids) != len(set(response_ids)):
            raise ValueError(f"Duplicate tool responses for assistant at conversation index {index}.")
        if set(response_ids) != set(call_ids):
            raise ValueError(
                f"Assistant/tool response IDs do not match at conversation index {index}: "
                f"calls={sorted(call_ids)}, responses={sorted(response_ids)}."
            )
        units.append(tuple(range(index, end)))
        index = end
    return units


def _message_chars(message: JsonObject) -> int:
    return len(json.dumps(message, ensure_ascii=True, sort_keys=True, default=str))


__all__ = [
    "RuntimeDeadlineExceeded",
    "EARLIER_TURNS_FILE",
    "FINAL_TURN_SUBMISSION_REMINDER",
    "SUBMISSION_WINDOW_REMINDER",
    "RuntimeDeadline",
    "bounded_provider_context",
    "persist_bulky_tool_output",
    "turn_budget_tag",
]
