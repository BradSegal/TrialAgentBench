"""Tests for public benchmark capability declarations."""

from trialagentbench.contracts.capabilities import BENCHMARK_CAPABILITIES


def test_benchmark_capabilities_resolve_both_streams() -> None:
    """Keep the finite public stream inventory runnable and gradeable."""

    assert {item.stream for item in BENCHMARK_CAPABILITIES} == {"trialeval", "trialdev"}
    assert all(item.task and item.support_boundary for item in BENCHMARK_CAPABILITIES)
