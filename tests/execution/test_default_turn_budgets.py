"""Tests for suite-owned default execution budgets."""

from trialagentbench.execution_policy import TRIALDEV_RELEASE_BUDGET_V1, TRIALEVAL_RELEASE_BUDGET_V1
from trialagentbench.trialdev.task_agent import DEFAULT_MAX_TURNS_PER_STEP
from trialagentbench.trialeval.agent import DEFAULT_MAX_TURNS


def test_trialeval_default_supports_bounded_long_form_analysis() -> None:
    """The default must not truncate the qualified long-form analysis path."""

    assert DEFAULT_MAX_TURNS == 50
    assert TRIALEVAL_RELEASE_BUDGET_V1.code_execution.timeout_seconds == 300.0


def test_trialdev_default_supports_bounded_multistep_analysis() -> None:
    """Each semantic step receives the suite-owned bounded-run turn ceiling."""

    assert DEFAULT_MAX_TURNS_PER_STEP == 100
    assert TRIALDEV_RELEASE_BUDGET_V1.code_execution.timeout_seconds == 300.0
