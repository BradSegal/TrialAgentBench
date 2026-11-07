"""Source-bound capability discovery without network-dependent tests."""

import hashlib
import io
import json
from email.message import Message
from urllib.request import HTTPRedirectHandler

import pytest

from trialagentbench.adapters import provider_capabilities as adapter
from trialagentbench.contracts.core.config import ProviderModelCapabilityV1


def metadata(monkeypatch, *, efforts=("high", "low"), providers=("OpenAI", "OpenAI")):
    responses = {
        "model": {"data": {"id": "fixture/model", "reasoning": {"supported_efforts": efforts}}},
        "endpoints": {
            "data": {
                "id": "fixture/model",
                "endpoints": [
                    {
                        "provider_name": provider,
                        "context_length": 20000 + i * 10000,
                        "max_completion_tokens": 2000 + i * 1000,
                        "supported_parameters": ["tools", "tool_choice", "reasoning"],
                    }
                    for i, provider in enumerate(providers)
                ],
            }
        },
    }
    calls = []

    def fetch(url):
        calls.append(url)
        return json.dumps(responses["endpoints" if url.endswith("/endpoints") else "model"]).encode()

    monkeypatch.setattr(adapter, "_metadata", fetch)
    return responses, calls


def test_snapshot_uses_pinned_route_limits_and_retains_exact_sources(tmp_path, monkeypatch):
    _, calls = metadata(monkeypatch)
    path = adapter.snapshot_openrouter_capability(
        model="fixture/model", upstream_provider="OpenAI", directory=tmp_path
    )
    record = ProviderModelCapabilityV1.model_validate_json(path.read_text())
    assert (record.context_window_tokens, record.maximum_completion_tokens) == (20000, 2000)
    assert record.supported_efforts == ("high", "low")
    source = tmp_path / f"{record.source_payload_sha256}.source.json"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == record.source_payload_sha256
    assert set(json.loads(source.read_bytes())) == set(calls)
    record.validate_route(provider="openrouter", model="fixture/model", upstream_provider="OpenAI")


@pytest.mark.parametrize("defect", ["wrong_model", "wrong_provider", "unknown_capacity", "missing_tools"])
def test_snapshot_fails_before_saving_unsupported_routes(tmp_path, monkeypatch, defect):
    responses, _ = metadata(monkeypatch)
    if defect == "wrong_model":
        responses["model"]["data"]["id"] = "fixture/other"
    else:
        endpoint = responses["endpoints"]["data"]["endpoints"][0]
        if defect == "wrong_provider":
            responses["endpoints"]["data"]["endpoints"] = []
        elif defect == "unknown_capacity":
            endpoint["max_completion_tokens"] = None
        else:
            endpoint["supported_parameters"] = []
    with pytest.raises(ValueError):
        adapter.snapshot_openrouter_capability(model="fixture/model", upstream_provider="OpenAI", directory=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_mandatory_reasoning_excludes_disable_even_when_gateway_allows_all(tmp_path, monkeypatch):
    responses, _ = metadata(monkeypatch, efforts=None)
    responses["model"]["data"]["reasoning"]["mandatory"] = True
    path = adapter.snapshot_openrouter_capability(
        model="fixture/model", upstream_provider="OpenAI", directory=tmp_path
    )
    record = ProviderModelCapabilityV1.model_validate_json(path.read_text())
    assert "high" in record.supported_efforts
    assert "none" not in record.supported_efforts


def test_reservations_use_route_upper_prices_and_reject_changed_source(tmp_path, monkeypatch):
    responses, _ = metadata(monkeypatch)
    for row in responses["endpoints"]["data"]["endpoints"]:
        row["pricing"] = {
            "prompt": "0.000001",
            "completion": "0.000004",
            "request": "0.01",
            "overrides": [{"min_prompt_tokens": 10000, "prompt": "0.000002"}],
        }
    path = adapter.snapshot_openrouter_capability(
        model="fixture/model", upstream_provider="OpenAI", directory=tmp_path
    )
    money, tokens, premise, pricing = adapter.campaign_reservations(path, completion_tokens=2000)
    assert (money, tokens) == (58000, 22000)
    assert "maximum published rates" in premise
    assert pricing.reserve_micro_usd(prompt_tokens=100, completion_tokens=10) == 10240
    record = ProviderModelCapabilityV1.model_validate_json(path.read_text())
    source = tmp_path / f"{record.source_payload_sha256}.source.json"
    source.write_text(source.read_text() + " ")
    with pytest.raises(ValueError, match="recorded identity"):
        adapter.campaign_reservations(path, completion_tokens=2000)


def test_metadata_key_is_not_forwarded_to_redirect_targets(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-test-key-123")
    sent = []

    def capture(request, timeout):
        sent.append(request)
        return io.BytesIO(b"{}")

    monkeypatch.setattr(adapter, "urlopen", capture)
    adapter._metadata("https://openrouter.ai/api/v1/models/fixture/model")
    (request,) = sent
    assert request.get_header("Authorization") == "Bearer dummy-test-key-123"
    redirected = HTTPRedirectHandler().redirect_request(
        request, io.BytesIO(), 302, "Found", Message(), "http://elsewhere.invalid/metadata"
    )
    assert redirected is not None
    assert redirected.get_header("Authorization") is None
    assert "dummy-test-key-123" not in repr(redirected.header_items())
