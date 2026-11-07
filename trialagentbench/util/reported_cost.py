"""Thread-safe provider-reported cost control for live evaluation runs."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from trialagentbench.ports import ClosableLLMProvider, LLMProvider, LLMResponse
from trialagentbench.ports.llm_provider import JsonObject


class ReportedCostThresholdReached(RuntimeError):
    """A live request was blocked after the reported-cost threshold was reached."""


class ReportedCostUnavailable(RuntimeError):
    """A cost-controlled route returned a response without reported cost."""


class RunStopRequested(KeyboardInterrupt):
    """A provider request was blocked after an orderly run stop was requested."""


@dataclass
class RunStopSignal:
    """Thread-safe cooperative stop signal shared by one live run."""

    _event: threading.Event = field(default_factory=threading.Event, repr=False)

    def request(self) -> None:
        """Prevent subsequent provider requests from starting."""

        self._event.set()

    def before_request(self) -> None:
        """Raise when an orderly run stop has been requested."""

        if self._event.is_set():
            raise RunStopRequested("The run was interrupted before this provider request started.")


@dataclass(frozen=True)
class ReportedCostSnapshotV1:
    """Atomic observation of one run-level reported-cost budget."""

    threshold_usd: float
    observed_usd: float
    response_count: int
    threshold_reached: bool
    cost_complete: bool

    @property
    def overshoot_usd(self) -> float:
        """Return observed spend above the post-response stop threshold."""

        return max(0.0, self.observed_usd - self.threshold_usd)


@dataclass
class ReportedCostBudget:
    """Share one post-response cost stop across concurrent provider adapters."""

    threshold_usd: float
    _observed_usd: float = 0.0
    _response_count: int = 0
    _cost_complete: bool = True
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        """Validate the positive post-response stop threshold."""

        if self.threshold_usd <= 0.0:
            raise ValueError("Reported-cost stop threshold must be positive.")

    @classmethod
    def from_observation(
        cls,
        *,
        threshold_usd: float,
        observed_usd: float,
        response_count: int,
        cost_complete: bool,
    ) -> ReportedCostBudget:
        """Restore a shared budget from validated prior response custody."""

        if observed_usd < 0.0 or response_count < 0:
            raise ValueError("Restored reported-cost observations must be non-negative.")
        return cls(
            threshold_usd=threshold_usd,
            _observed_usd=observed_usd,
            _response_count=response_count,
            _cost_complete=cost_complete,
        )

    def before_request(self) -> None:
        """Reject a new request after cost loss or threshold crossing."""

        with self._lock:
            if not self._cost_complete:
                raise ReportedCostUnavailable(
                    "A cost-controlled provider response omitted reported cost; no further request is allowed."
                )
            if self._observed_usd >= self.threshold_usd:
                raise ReportedCostThresholdReached(
                    f"Reported-cost stop threshold reached: {self._observed_usd:.9f} >= "
                    f"{self.threshold_usd:.9f} USD."
                )

    def record_response(self, response: LLMResponse) -> None:
        """Add one successful response's provider-reported cost atomically."""

        cost = response.metadata.reported_cost_usd
        with self._lock:
            self._response_count += 1
            if cost is None:
                self._cost_complete = False
            else:
                self._observed_usd += cost

    def snapshot(self) -> ReportedCostSnapshotV1:
        """Return a consistent public projection of current cost observations."""

        with self._lock:
            return ReportedCostSnapshotV1(
                threshold_usd=self.threshold_usd,
                observed_usd=self._observed_usd,
                response_count=self._response_count,
                threshold_reached=self._observed_usd >= self.threshold_usd,
                cost_complete=self._cost_complete,
            )


@dataclass(frozen=True)
class ReportedCostBoundProvider:
    """Decorate one provider with a shared post-response reported-cost stop."""

    provider: LLMProvider
    budget: ReportedCostBudget

    @property
    def model(self) -> str:
        """Return the delegated provider model identifier."""

        return str(self.provider.model)

    @property
    def telemetry_route(self) -> str:
        """Return the delegated provider route identity."""

        return str(self.provider.telemetry_route)

    @property
    def manages_context(self) -> bool:
        """Preserve delegated context ownership through the cost boundary."""

        return bool(getattr(self.provider, "manages_context", False))

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
        """Generate one turn and charge its reported cost to the shared run budget."""

        self.budget.before_request()
        response = self.provider.generate_turn(
            messages,
            tools,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            tool_choice=tool_choice,
        )
        self.budget.record_response(response)
        return response

    def close(self) -> None:
        """Close the delegated provider when it owns resources."""

        if isinstance(self.provider, ClosableLLMProvider):
            self.provider.close()


@dataclass(frozen=True)
class StoppableProvider:
    """Decorate one provider with a cooperative run-level stop signal."""

    provider: LLMProvider
    signal: RunStopSignal

    @property
    def model(self) -> str:
        """Return the delegated provider model identifier."""

        return str(self.provider.model)

    @property
    def telemetry_route(self) -> str:
        """Return the delegated provider route identity."""

        return str(self.provider.telemetry_route)

    @property
    def manages_context(self) -> bool:
        """Preserve delegated context ownership through the stop boundary."""

        return bool(getattr(self.provider, "manages_context", False))

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
        """Generate one turn unless the parent run has requested an orderly stop."""

        self.signal.before_request()
        return self.provider.generate_turn(
            messages,
            tools,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            tool_choice=tool_choice,
        )

    def close(self) -> None:
        """Close the delegated provider when it owns resources."""

        if isinstance(self.provider, ClosableLLMProvider):
            self.provider.close()


__all__ = [
    "ReportedCostBoundProvider",
    "ReportedCostBudget",
    "ReportedCostSnapshotV1",
    "ReportedCostThresholdReached",
    "ReportedCostUnavailable",
    "RunStopRequested",
    "RunStopSignal",
    "StoppableProvider",
]
