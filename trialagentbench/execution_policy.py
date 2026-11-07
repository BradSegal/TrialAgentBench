"""Release execution budgets shared by the public harness and release metadata."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from trialagentbench_verifier.execution_contracts import (
    DEFAULT_COMPLETION_TOKENS,
    RELEASE_EXECUTION_LIMITS_V1,
    TRIALDEV_MAXIMUM_TURNS_PER_STEP,
    TRIALEVAL_MAXIMUM_TURNS,
    CodeExecutionLimitsV1,
)

TRIALEVAL_DEFAULT_WORKERS = 1
TRIALEVAL_DIAGNOSTIC_PROOF_DEFAULT_WORKERS = 8
TRIALEVAL_DIAGNOSTIC_WORKER_INVALID_INPUT_EXIT_CODE = 2


class ExecutionBudgetProfileV1(BaseModel):
    """One immutable suite-level execution budget."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["ariadne.execution_budget_profile/v1"] = "ariadne.execution_budget_profile/v1"
    profile_id: str = Field(min_length=1)
    suite_id: Literal["trialeval", "trialdev"]
    turn_scope: Literal["item", "submission_step"]
    maximum_turns: int = Field(ge=1)
    maximum_completion_tokens_per_turn: int = Field(ge=1)
    provider_request_timeout_seconds: float = Field(gt=0.0, le=900.0)
    wall_time_limit_seconds: int = Field(ge=1)
    maximum_context_characters: int = Field(ge=1)
    code_execution: CodeExecutionLimitsV1
    network_access: Literal[False] = False


TRIALEVAL_RELEASE_BUDGET_V1 = ExecutionBudgetProfileV1(
    profile_id="trialeval_release_default_v1",
    suite_id="trialeval",
    turn_scope="item",
    maximum_turns=TRIALEVAL_MAXIMUM_TURNS,
    maximum_completion_tokens_per_turn=DEFAULT_COMPLETION_TOKENS,
    provider_request_timeout_seconds=300.0,
    wall_time_limit_seconds=3600,
    maximum_context_characters=120_000,
    code_execution=RELEASE_EXECUTION_LIMITS_V1,
)

TRIALDEV_RELEASE_BUDGET_V1 = ExecutionBudgetProfileV1(
    profile_id="trialdev_release_default_v1",
    suite_id="trialdev",
    turn_scope="submission_step",
    maximum_turns=TRIALDEV_MAXIMUM_TURNS_PER_STEP,
    maximum_completion_tokens_per_turn=DEFAULT_COMPLETION_TOKENS,
    provider_request_timeout_seconds=300.0,
    wall_time_limit_seconds=1800,
    maximum_context_characters=120_000,
    code_execution=RELEASE_EXECUTION_LIMITS_V1,
)


__all__ = [
    "ExecutionBudgetProfileV1",
    "TRIALEVAL_DIAGNOSTIC_PROOF_DEFAULT_WORKERS",
    "TRIALEVAL_DIAGNOSTIC_WORKER_INVALID_INPUT_EXIT_CODE",
    "TRIALEVAL_RELEASE_BUDGET_V1",
    "TRIALEVAL_DEFAULT_WORKERS",
    "TRIALDEV_RELEASE_BUDGET_V1",
]
