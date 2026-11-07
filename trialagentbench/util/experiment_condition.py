"""Resolve one provider-bound experimental condition before execution."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from trialagentbench_verifier.execution_contracts import AnalysisMode, ReportingInterfaceV1

from trialagentbench.contracts.core.config import (
    ExperimentConditionV1,
    MemoryStrategyV1,
    ProcedureAssistanceV1,
    ProviderModelCapabilityV1,
    ReasoningConfigV1,
    ReasoningEffortV1,
    ToolChoiceV1,
)
from trialagentbench.io import read_json_model


def resolve_experiment_condition_v1(
    *,
    condition_id: str,
    request_replicate_id: str,
    reasoning_effort: ReasoningEffortV1 | None,
    provider_capability_snapshot: Path | None,
    provider: Literal["openai", "openai_responses", "openrouter", "offline_control"],
    model: str,
    openrouter_provider: str | None,
    procedure_assistance: ProcedureAssistanceV1,
    maximum_turns_per_step: int,
    tool_choice: ToolChoiceV1,
    memory_strategy: MemoryStrategyV1 = "bounded",
    analysis_language: AnalysisMode = "python_r",
    exclude_reasoning: bool = False,
    replay_reasoning: bool = False,
    reporting_interface: ReportingInterfaceV1 = "v1",
) -> ExperimentConditionV1:
    """Validate provider capability and bind the complete request condition."""

    if exclude_reasoning and replay_reasoning:
        raise ValueError("Excluded reasoning cannot be replayed.")
    if replay_reasoning and provider != "openrouter":
        raise ValueError("Replaying returned reasoning details requires --provider openrouter.")
    if memory_strategy == "openai_responses_compaction" and provider != "openai_responses":
        raise ValueError("OpenAI Responses compaction requires --provider openai_responses.")

    capability = None
    if provider_capability_snapshot is not None:
        capability = read_json_model(
            ProviderModelCapabilityV1,
            provider_capability_snapshot,
        )
        capability.validate_route(provider=provider, model=model, upstream_provider=openrouter_provider)
    return ExperimentConditionV1(
        condition_id=condition_id,
        request_replicate_id=request_replicate_id,
        reasoning=ReasoningConfigV1(
            effort=reasoning_effort,
            exclude_from_response=exclude_reasoning,
            replay_in_context=replay_reasoning,
        ),
        provider_capability=capability,
        procedure_assistance=procedure_assistance,
        maximum_turns_per_step=maximum_turns_per_step,
        tool_choice=tool_choice,
        memory_strategy=memory_strategy,
        analysis_language=analysis_language,
        reporting_interface=reporting_interface,
    )


__all__ = ["resolve_experiment_condition_v1"]
