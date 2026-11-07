"""Retrieve and retain the public limits for a pinned OpenRouter model route."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import get_args
from urllib.parse import quote
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field
from trialagentbench_verifier.campaign_contracts import CampaignRequestPricingV1
from trialagentbench_verifier.io import read_json_object

from trialagentbench.contracts.core.config import ProviderModelCapabilityV1, ReasoningEffortV1
from trialagentbench.io import read_json_model, write_json_model


class _Reasoning(BaseModel):
    model_config = ConfigDict(extra="ignore")
    supported_efforts: tuple[ReasoningEffortV1, ...] | None = ()
    mandatory: bool = False


class _ModelMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    reasoning: _Reasoning | None = None


class _Rates(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)
    prompt: Decimal = Field(default=Decimal(0), ge=0)
    completion: Decimal = Field(default=Decimal(0), ge=0)
    request: Decimal = Field(default=Decimal(0), ge=0)
    input_cache_read: Decimal = Field(default=Decimal(0), ge=0)
    input_cache_write: Decimal = Field(default=Decimal(0), ge=0)
    input_cache_write_1h: Decimal = Field(default=Decimal(0), ge=0)
    internal_reasoning: Decimal = Field(default=Decimal(0), ge=0)


class _Pricing(_Rates):
    prompt: Decimal = Field(ge=0)
    completion: Decimal = Field(ge=0)
    overrides: tuple[_Rates, ...] = ()


class _Endpoint(BaseModel):
    model_config = ConfigDict(extra="ignore")
    provider_name: str
    pricing: _Pricing | None = None
    context_length: int = Field(gt=0, strict=True)
    max_completion_tokens: int | None = Field(default=None, gt=0, strict=True)
    supported_parameters: tuple[str, ...]


class _Endpoints(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    endpoints: tuple[_Endpoint, ...]


class _ModelResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    data: _ModelMetadata


class _EndpointResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    data: _Endpoints


def _metadata(url: str) -> bytes:
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "TrialAgentBench/1.0"})
    if key := os.environ.get("OPENROUTER_API_KEY"):
        # urllib copies ordinary headers to any redirect target; unredirected ones stay with this URL.
        request.add_unredirected_header("Authorization", f"Bearer {key}")
    with urlopen(request, timeout=30) as response:
        payload = bytes(response.read(4 * 1024 * 1024 + 1))
    if len(payload) > 4 * 1024 * 1024:
        raise ValueError("Provider metadata exceeds the response size limit")
    return payload


def snapshot_openrouter_capability(*, model: str, upstream_provider: str, directory: Path) -> Path:
    """Fetch exact-route capacity and model effort metadata before model execution.

    Parameters
    ----------
    model, upstream_provider
        Exact model identifier and pinned upstream provider display name.
    directory
        Operator-owned location for immutable metadata and capability records.

    Returns
    -------
    pathlib.Path
        Content-addressed capability record, suitable for frozen native invocations.

    Raises
    ------
    ValueError
        If the model, route or required capacity is unavailable or ambiguous.
    OSError
        If metadata retrieval or persistence fails. No model request is made.
    """
    encoded = quote(model, safe="/")
    base = "https://openrouter.ai/api/v1"
    model_url, endpoint_url = f"{base}/model/{encoded}", f"{base}/models/{encoded}/endpoints"
    model_bytes, endpoint_bytes = _metadata(model_url), _metadata(endpoint_url)
    metadata = _ModelResponse.model_validate_json(model_bytes).data
    routes = _EndpointResponse.model_validate_json(endpoint_bytes).data
    if metadata.id != model or routes.id != model:
        raise ValueError("Provider metadata identifies a different model")
    endpoints = [row for row in routes.endpoints if row.provider_name.casefold() == upstream_provider.casefold()]
    if not endpoints or any(row.max_completion_tokens is None for row in endpoints):
        raise ValueError(
            "Pinned provider route has no complete capacity metadata; supply capability_snapshot explicitly"
        )
    if any(not {"tools", "tool_choice"}.issubset(row.supported_parameters) for row in endpoints):
        raise ValueError("Pinned provider route does not support the benchmark's tools")
    reasoning = metadata.reasoning
    efforts: tuple[ReasoningEffortV1, ...] = ()
    if reasoning is not None and all(
        {"reasoning", "reasoning_effort"}.intersection(row.supported_parameters) for row in endpoints
    ):
        efforts = (
            reasoning.supported_efforts if reasoning.supported_efforts is not None else get_args(ReasoningEffortV1)
        )
        if reasoning.mandatory:
            efforts = tuple(effort for effort in efforts if effort != "none")
    source = json.dumps(
        {model_url: json.loads(model_bytes), endpoint_url: json.loads(endpoint_bytes)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    source_digest = hashlib.sha256(source).hexdigest()
    capability = ProviderModelCapabilityV1(
        provider_transport="openrouter",
        model_id=model,
        upstream_provider=upstream_provider,
        supported_efforts=efforts,
        context_window_tokens=min(row.context_length for row in endpoints),
        maximum_completion_tokens=min(
            row.max_completion_tokens for row in endpoints if row.max_completion_tokens is not None
        ),
        source_url=base,
        source_retrieved_utc=datetime.now(UTC),
        source_payload_sha256=source_digest,
    )
    directory.mkdir(parents=True, exist_ok=True)
    source_path = directory / f"{source_digest}.source.json"
    if source_path.exists() and source_path.read_bytes() != source:
        raise ValueError("Retained provider metadata differs from its content identity")
    if not source_path.exists():
        source_path.write_bytes(source)
    path = directory / f"{capability.checksum}.json"
    write_json_model(path, capability)
    return path


def campaign_reservations(
    capability_path: Path, *, completion_tokens: int
) -> tuple[int, int, str, CampaignRequestPricingV1]:
    """Bound text-request cost using the retained route rates, including conditional rates."""
    capability = read_json_model(ProviderModelCapabilityV1, capability_path)
    source_path = capability_path.parent / f"{capability.source_payload_sha256}.source.json"
    if not source_path.is_file():
        raise ValueError(
            "Automatic reservations need retained route metadata; supply all reservation fields for a custom capability"
        )
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != capability.source_payload_sha256:
        raise ValueError("Provider pricing source differs from its recorded identity")
    source = read_json_object(source_path)
    url = f"{capability.source_url}/models/{quote(capability.model_id, safe='/')}/endpoints"
    endpoints = _EndpointResponse.model_validate(source.get(url)).data
    if endpoints.id != capability.model_id or capability.context_window_tokens is None:
        raise ValueError("Pricing requires the exact model and its context limit")
    routes = [
        row
        for row in endpoints.endpoints
        if row.provider_name.casefold() == str(capability.upstream_provider).casefold()
    ]
    if not routes or any(row.pricing is None for row in routes):
        raise ValueError(
            "Route pricing is unavailable; supply reservation_micro_usd, reservation_tokens and pricing_premise"
        )
    rates = [rate for row in routes if row.pricing is not None for rate in (row.pricing, *row.pricing.overrides)]
    # Include cache-write and reasoning charges without assuming discounts or cache hits.
    prompt_rate = max(rate.prompt for rate in rates) + max(
        max(rate.input_cache_read, rate.input_cache_write, rate.input_cache_write_1h) for rate in rates
    )
    completion_rate = max(rate.completion for rate in rates) + max(rate.internal_reasoning for rate in rates)
    request_rate = max(rate.request for rate in rates)
    prompt_tokens = capability.context_window_tokens
    pricing = CampaignRequestPricingV1(
        prompt_usd_per_token=prompt_rate, completion_usd_per_token=completion_rate, request_usd=request_rate
    )
    micro_usd = pricing.reserve_micro_usd(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    premise = (
        f"Retained route metadata {capability.source_payload_sha256}; "
        f"{prompt_tokens} input tokens at ${prompt_rate}/token + "
        f"{completion_tokens} output tokens at ${completion_rate}/token + ${request_rate}/request; "
        "maximum published rates including overrides and cache/reasoning charges, no discounts."
    )
    return micro_usd, prompt_tokens + completion_tokens, premise, pricing
