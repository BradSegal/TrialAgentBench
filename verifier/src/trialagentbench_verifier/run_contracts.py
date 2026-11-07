"""Common configuration contracts used by harness artifacts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from trialagentbench_verifier.execution_contracts import AnalysisMode, ReportingInterfaceV1
from trialagentbench_verifier.io import canonical_sha256 as canonical_payload_sha256

ProcedureAssistanceV1: TypeAlias = Literal[
    "output_contract_only",
    "unordered_checklist",
]  # noqa: UP040
ToolChoiceV1: TypeAlias = Literal["auto", "required"]  # noqa: UP040
MemoryStrategyV1: TypeAlias = Literal["bounded", "openai_responses_compaction"]  # noqa: UP040

ReasoningEffortV1: TypeAlias = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]  # noqa: UP040
TrialEvalAgentStatusV1: TypeAlias = Literal["success", "max_turns_reached", "failed"]  # noqa: UP040


class ProviderUnavailabilityReceiptV1(BaseModel):
    """A retained unanswered request authorizing one prospective infrastructure recovery."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_id: Literal["ariadne.provider_unavailability_receipt/v1"] = "ariadne.provider_unavailability_receipt/v1"
    request_event: dict[str, JsonValue]
    request_event_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_unanswered_request(self) -> Self:
        event = self.request_event
        if canonical_payload_sha256(event) != self.request_event_sha256:
            raise ValueError("Provider unavailability receipt differs from its retained request event")
        required = {
            "schema_id": "ariadne.provider_request_event/v1",
            "benchmark": "trialeval",
            "status": "failed",
            "exception_type": "ProviderUnavailableError",
            "unavailability_policy": "defer_unanswered_v1",
            "response_received": False,
            "usage_status": "not_applicable",
            "request_attempts": 1,
        }
        if any(event.get(key) != value for key, value in required.items()):
            raise ValueError("Provider recovery requires a typed, unanswered request under defer_unanswered_v1")
        if event.get("failure_type") not in {"timeout", "provider_error"}:
            raise ValueError("Provider recovery requires transport unavailability")
        for field in ("request_id", "unit_id", "requested_model", "provider_route"):
            if not isinstance(event.get(field), str) or not event[field]:
                raise ValueError("Provider recovery is missing request identity")
        for field in (
            "response_id",
            "returned_model",
            "finish_reason",
            "model_outcome",
            "reasoning",
            "reported_cost_usd",
        ):
            if event.get(field) is not None:
                raise ValueError("A model response or ambiguous usage cannot authorize provider recovery")
        for field in ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens", "cached_tokens"):
            if event.get(field) != 0:
                raise ValueError("A usage-bearing request cannot authorize provider recovery")
        elapsed = event.get("elapsed_seconds")
        excluded = event.get("scientific_clock_excluded_seconds", 0.0)
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or isinstance(excluded, bool)
            or not isinstance(excluded, (int, float))
            or not 0 <= excluded <= elapsed < float("inf")
        ):
            raise ValueError("Provider recovery has invalid measured clock accounting")
        return self

    def require_run_binding(self, *, item_id: str, model: str, policy: str) -> None:
        """Bind this receipt to the task and prospectively frozen provider policy."""
        if (
            policy != "defer_unanswered_v1"
            or self.request_event["unit_id"] != item_id
            or self.request_event["requested_model"] != model
        ):
            raise ValueError("Provider recovery receipt differs from the frozen task, model or policy")


class DecodingConfigV1(BaseModel):
    """Explicit decoding configuration recorded for reproducibility."""

    model_config = ConfigDict(extra="forbid")
    schema_id: Literal["trialagentbench_decoding_config_v1"] = "trialagentbench_decoding_config_v1"
    schema_version: Literal[1] = 1
    temperature: float
    max_tokens: int
    send_temperature: bool
    decoding_seed: int | None = Field(default=None, strict=True, ge=0)
    # Keep the contract small and stable. If/when we add additional decoding
    # knobs (e.g., top_p), they should be added as explicit optional fields
    # rather than a free-form dict.


class RoutingConfigV1(BaseModel):
    """Routing pins and request boundary for the selected transport."""

    model_config = ConfigDict(extra="forbid")
    schema_id: Literal["trialagentbench_routing_config_v1"] = "trialagentbench_routing_config_v1"
    schema_version: Literal[1] = 1
    provider: Literal["openai", "openai_responses", "openrouter", "offline_reference", "offline_control"]
    openrouter_provider: str | None = None
    request_timeout_seconds: float = Field(gt=0.0, le=900.0)

    @model_validator(mode="after")
    def validate_exact_route(self) -> Self:
        """Require one reproducible route for the selected transport."""

        if self.provider == "openrouter" and not self.openrouter_provider:
            raise ValueError("OpenRouter runs require an explicit upstream provider pin.")
        if self.provider != "openrouter" and self.openrouter_provider is not None:
            raise ValueError("openrouter_provider is valid only for the OpenRouter transport.")
        return self


class ProviderRetryPolicyV1(BaseModel):
    """Explicit retry policy applied by an auditable provider adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["ariadne.provider_retry_policy/v1"] = "ariadne.provider_retry_policy/v1"
    maximum_attempts: int = Field(default=5, ge=1, le=20)
    base_delay_seconds: float = Field(default=2.0, ge=0.0, le=60.0)
    unavailability_policy: Literal["charge_all_v1", "defer_unanswered_v1"] = Field(
        default="charge_all_v1", exclude_if=lambda value: value == "charge_all_v1"
    )
    # Operational: how a received but rejected provider response is accounted (continuation cohort, 27 Oct).
    # It changes no task input, grading rule or score; recorded so those runs parse and regrade.
    rejection_policy: Literal["quarantine_rejected_v1"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class ProviderModelCapabilityV1(BaseModel):
    """Source-bound context, completion and reasoning limits for one model route."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["ariadne.provider_model_capability/v1"] = "ariadne.provider_model_capability/v1"
    provider_transport: Literal["openai", "openai_responses", "openrouter"]
    model_id: str = Field(..., min_length=1)
    upstream_provider: str | None = Field(default=None, min_length=1)
    supported_efforts: tuple[ReasoningEffortV1, ...] = ()
    context_window_tokens: int | None = Field(default=None, strict=True, gt=0)
    maximum_completion_tokens: int | None = Field(default=None, strict=True, gt=0)
    source_url: str = Field(..., min_length=1)
    source_retrieved_utc: datetime
    source_payload_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    def validate_route(self, *, provider: str, model: str, upstream_provider: str | None) -> None:
        """Reject applying a source record to a different analysis model route."""

        if (self.provider_transport, self.model_id, self.upstream_provider) != (provider, model, upstream_provider):
            raise ValueError("Provider capability model does not match the analysis model route.")

    @model_validator(mode="after")
    def validate_capability(self) -> Self:
        """Require coherent limits, a unique effort set, exact route and checksum."""

        if self.context_window_tokens is not None and self.maximum_completion_tokens is not None:
            if self.maximum_completion_tokens > self.context_window_tokens:
                raise ValueError("Provider completion capacity cannot exceed its context window.")

        if len(self.supported_efforts) != len(set(self.supported_efforts)):
            raise ValueError("Provider reasoning efforts must be unique.")
        if (self.provider_transport == "openrouter") != (self.upstream_provider is not None):
            raise ValueError("An upstream provider is required exactly for OpenRouter capability records.")
        digest = canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))
        if self.checksum is not None and self.checksum != digest:
            raise ValueError("Provider model capability checksum mismatch.")
        object.__setattr__(self, "checksum", digest)
        return self


class ReasoningConfigV1(BaseModel):
    """Requested reasoning configuration, validated by the model capability."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    effort: ReasoningEffortV1 | None = None
    # Returned reasoning text is kept only in the run's provider log, never in the agent's
    # context, conversation or grading; True (earlier runs) asked the route not to return it.
    exclude_from_response: bool = False
    # Returned reasoning details sent back with the assistant turn (provider reasoning-state continuity).
    replay_in_context: bool = Field(default=False, exclude_if=lambda value: value is False)


class ExperimentConditionV1(BaseModel):
    """Provider-neutral identity for one evaluated request condition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["ariadne.experiment_condition/v1"] = "ariadne.experiment_condition/v1"
    condition_id: str = Field(..., min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    request_replicate_id: str = Field(..., min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    reasoning: ReasoningConfigV1 = Field(default_factory=ReasoningConfigV1)
    provider_capability: ProviderModelCapabilityV1 | None = None
    procedure_assistance: ProcedureAssistanceV1 = "output_contract_only"
    maximum_turns_per_step: int = Field(..., ge=1)
    tool_choice: ToolChoiceV1 = "auto"
    memory_strategy: MemoryStrategyV1 = "bounded"
    analysis_language: AnalysisMode = Field(default="python_r", exclude_if=lambda value: value == "python_r")
    # Absent in 1.0 records, which used the v1 interface; omitted when v1 so their identities are unchanged.
    reporting_interface: ReportingInterfaceV1 = Field(default="v1", exclude_if=lambda value: value == "v1")

    @model_validator(mode="after")
    def validate_provider_reasoning(self) -> Self:
        """Bind requested reasoning to the same capability that records capacity."""

        if self.reasoning.effort is not None:
            if self.provider_capability is None:
                raise ValueError("A requested reasoning effort requires one exact capability record.")
            if self.reasoning.effort not in self.provider_capability.supported_efforts:
                raise ValueError(
                    f"Reasoning effort {self.reasoning.effort!r} is not supported by the exact model route."
                )
        return self


__all__ = [
    "DecodingConfigV1",
    "ExperimentConditionV1",
    "MemoryStrategyV1",
    "ProcedureAssistanceV1",
    "ProviderRetryPolicyV1",
    "ProviderModelCapabilityV1",
    "ReasoningConfigV1",
    "ReasoningEffortV1",
    "RoutingConfigV1",
    "ToolChoiceV1",
]
