"""Tests for evaluator cost and cooperative-stop boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from trialagentbench.ports import LLMResponse, LLMResponseMetadata
from trialagentbench.util.reported_cost import (
    ReportedCostBoundProvider,
    ReportedCostBudget,
    ReportedCostThresholdReached,
    ReportedCostUnavailable,
    RunStopRequested,
    RunStopSignal,
    StoppableProvider,
)


@dataclass
class _Provider:
    model: str = "test/model"
    telemetry_route: str = "test"
    reported_cost_usd: float | None = 0.4
    manages_context: bool = False
    calls: int = 0

    def generate_turn(self, *args: object, **kwargs: object) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            content="report",
            metadata=LLMResponseMetadata(reported_cost_usd=self.reported_cost_usd),
        )


def _request(provider: object) -> LLMResponse:
    return provider.generate_turn([], temperature=0.0, max_tokens=100)  # type: ignore[attr-defined,no-any-return]


def test_reported_cost_threshold_blocks_only_subsequent_requests() -> None:
    """Attribute bounded post-response overshoot without starting another call."""

    delegate = _Provider(reported_cost_usd=0.6)
    provider = ReportedCostBoundProvider(delegate, ReportedCostBudget(threshold_usd=1.0))

    _request(provider)
    _request(provider)
    snapshot = provider.budget.snapshot()
    assert snapshot.observed_usd == pytest.approx(1.2)
    assert snapshot.overshoot_usd == pytest.approx(0.2)
    with pytest.raises(ReportedCostThresholdReached):
        _request(provider)
    assert delegate.calls == 2


def test_missing_reported_cost_has_a_distinct_terminal_error() -> None:
    """Do not conflate incomplete evaluator accounting with participant analysis."""

    delegate = _Provider(reported_cost_usd=None)
    provider = ReportedCostBoundProvider(delegate, ReportedCostBudget(threshold_usd=1.0))

    _request(provider)
    assert not provider.budget.snapshot().cost_complete
    with pytest.raises(ReportedCostUnavailable):
        _request(provider)
    assert delegate.calls == 1


def test_cooperative_stop_does_not_call_the_provider() -> None:
    """Preserve an operator stop separately from provider or analytical failure."""

    delegate = _Provider()
    signal = RunStopSignal()
    signal.request()

    with pytest.raises(RunStopRequested):
        _request(StoppableProvider(delegate, signal))
    assert delegate.calls == 0


def test_provider_decorators_preserve_context_ownership() -> None:
    """Keep one context owner when runtime controls are composed."""

    delegate = _Provider(manages_context=True)
    budgeted = ReportedCostBoundProvider(delegate, ReportedCostBudget(threshold_usd=1.0))
    stoppable = StoppableProvider(budgeted, RunStopSignal())

    assert budgeted.manages_context is True
    assert stoppable.manages_context is True
