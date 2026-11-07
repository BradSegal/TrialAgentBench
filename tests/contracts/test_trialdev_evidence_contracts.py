"""Contract tests for variable-cardinality TrialDev evidence."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError
from trialagentbench_verifier.response import (
    TRIALDEV_SAMPLING_INTERVAL_DESCRIPTION_V1,
    TrialDevCandidateDiagnosticsV1,
    TrialDevEndpointEstimateV1,
)
from trialagentbench_verifier.units import compatible_units

from trialagentbench.contracts.core.config import (
    ExperimentConditionV1,
    ProviderModelCapabilityV1,
    ReasoningConfigV1,
)
from trialagentbench.contracts.trialdev.release import TrialDevMaterialV1, TrialDevTaskV1
from trialagentbench.contracts.trialdev.run import (
    TrialDevRunIndexV1,
    TrialDevRunInvocationV1,
    TrialDevRunResultV1,
)
from trialagentbench.io import read_json_model, write_json_model
from trialagentbench.ports import CodeExecutionLimitsV1


@pytest.mark.parametrize(
    ("reported", "reference"),
    (
        ("points", "SGRQ points"),
        ("score points", "SGRQ points"),
        ("SGRQ total score points", "SGRQ points"),
        ("DAS28 difference (points)", "DAS28 points"),
        ("risk-difference units", "risk difference"),
        ("difference in proportions", "risk difference"),
        ("proportion", "risk difference"),
        (
            "risk difference by Day 168 (proportion; candidate minus standard inhaled therapy)",
            "risk difference",
        ),
    ),
)
def test_trialdev_unit_labels_accept_same_scale_language(reported: str, reference: str) -> None:
    assert compatible_units(reported=reported, reference=reference)


@pytest.mark.parametrize(
    ("reported", "reference"),
    (
        ("percentage points", "risk difference"),
        ("risk difference reported in percentage points", "risk difference"),
        ("points", "mg/dL"),
        ("EQ-5D points", "SGRQ points"),
        ("hazard ratio", "risk difference"),
        ("log risk difference", "risk difference"),
        ("odds risk difference", "risk difference"),
        ("log DAS28 points", "DAS28 points"),
        ("ratio DAS28 points", "DAS28 points"),
        ("percent DAS28 points", "DAS28 points"),
    ),
)
def test_trialdev_unit_labels_reject_scale_changes(reported: str, reference: str) -> None:
    assert not compatible_units(reported=reported, reference=reference)


def test_point_estimates_do_not_require_invented_sampling_uncertainty() -> None:
    estimate = TrialDevEndpointEstimateV1(
        asset_id="asset_alpha",
        endpoint="clinical_outcome",
        unit="score points",
        estimate=-1.2,
    )

    assert estimate.interval_95 is None

    with pytest.raises(ValidationError, match="sensitivity intervals require a sampling confidence interval"):
        TrialDevEndpointEstimateV1(
            asset_id="asset_alpha",
            endpoint="clinical_outcome",
            unit="score points",
            estimate=-1.2,
            sensitivity_interval=(-2.0, -0.4),
        )


def test_sampling_interval_schema_states_its_inferential_meaning() -> None:
    schema = TrialDevEndpointEstimateV1.model_json_schema()

    assert schema["properties"]["interval_95"]["description"] == TRIALDEV_SAMPLING_INTERVAL_DESCRIPTION_V1


def _estimate(asset_id: str, endpoint: str) -> TrialDevEndpointEstimateV1:
    return TrialDevEndpointEstimateV1(
        asset_id=asset_id,
        endpoint=endpoint,
        unit="score points" if endpoint == "clinical_outcome" else "risk difference",
        estimate=0.0,
        standard_error=0.1,
        interval_95=(-0.2, 0.2),
    )


def _invocation(**updates: object) -> TrialDevRunInvocationV1:
    payload: dict[str, object] = {
        "release_id": f"release_{'a' * 64}",
        "release_archive_sha256": "c" * 64,
        "release_manifest_sha256": "b" * 64,
        "harness_source_sha256": "d" * 64,
        "scheduled_task_ids": ("td-123456789abc",),
        "scheduled_task_streams": {"td-123456789abc": "evidence_analysis"},
        "model": "analysis-model",
        "provider": "openai",
        "experiment_condition": ExperimentConditionV1(
            condition_id="primary",
            request_replicate_id="request-1",
            maximum_turns_per_step=12,
        ),
        "send_temperature": True,
        "maximum_completion_tokens": 4_096,
        "maximum_context_characters": 100_000,
        "watchdog_seconds": 900,
        "request_timeout_seconds": 120.0,
        "executor_image": "sha256:executor",
        "executor_limits": CodeExecutionLimitsV1(),
    }
    payload.update(updates)
    return TrialDevRunInvocationV1.model_validate(payload)


def test_run_invocation_binds_resume_identity_and_provider_route() -> None:
    """A continuation cannot silently change its task, model, or provider route."""

    invocation = _invocation()
    assert TrialDevRunInvocationV1.model_validate(invocation.model_dump()) == invocation
    with pytest.raises(ValidationError, match="checksum mismatch"):
        _invocation(checksum="0" * 64)
    with pytest.raises(ValidationError, match="provider routing"):
        _invocation(provider="openrouter")
    with pytest.raises(ValidationError, match="task denominator"):
        _invocation(scheduled_task_streams={"td-aaaaaaaaaaaa": "evidence_analysis"})


def test_run_invocation_stream_map_round_trips_independently_of_json_key_order(tmp_path: Path) -> None:
    """Canonical JSON key sorting must not invalidate a valid task schedule."""

    invocation = _invocation(
        scheduled_task_ids=("td-ffffffffffff", "td-000000000000"),
        scheduled_task_streams={
            "td-ffffffffffff": "portfolio_management",
            "td-000000000000": "evidence_analysis",
        },
    )
    path = tmp_path / "invocation.json"

    write_json_model(path, invocation)

    assert read_json_model(TrialDevRunInvocationV1, path) == invocation


def test_run_invocation_binds_reasoning_capability_to_the_exact_route() -> None:
    """Reasoning effort cannot be inferred from an unbound model label."""

    capability = ProviderModelCapabilityV1(
        provider_transport="openrouter",
        model_id="analysis-model",
        upstream_provider="PinnedProvider",
        supported_efforts=("low", "high"),
        source_url="https://example.test/capability",
        source_retrieved_utc=datetime(2026, 8, 24, tzinfo=UTC),
        source_payload_sha256="a" * 64,
    )
    invocation = _invocation(
        provider="openrouter",
        openrouter_provider="PinnedProvider",
        experiment_condition=ExperimentConditionV1(
            condition_id="high-reasoning",
            request_replicate_id="request-1",
            reasoning=ReasoningConfigV1(effort="high"),
            provider_capability=capability,
            maximum_turns_per_step=12,
        ),
    )

    assert invocation.experiment_condition.provider_capability == capability
    wrong_model = ProviderModelCapabilityV1(
        **{
            **capability.model_dump(mode="python", exclude={"checksum"}),
            "model_id": "other-model",
        }
    )
    with pytest.raises(ValidationError, match="analysis model route"):
        _invocation(
            provider="openrouter",
            openrouter_provider="PinnedProvider",
            experiment_condition=ExperimentConditionV1(
                condition_id="high-reasoning",
                request_replicate_id="request-1",
                reasoning=ReasoningConfigV1(effort="high"),
                provider_capability=wrong_model,
                maximum_turns_per_step=12,
            ),
        )


def test_run_index_requires_complete_ordered_task_coverage() -> None:
    """Keep analysis denominators inseparable from the scheduled invocation."""

    invocation = _invocation()
    result = TrialDevRunResultV1(
        task_id="td-123456789abc",
        stream="evidence_analysis",
        execution_status="completed",
        checksum="b" * 64,
    )
    index = TrialDevRunIndexV1(
        release_id=invocation.release_id,
        scheduled_task_ids=invocation.scheduled_task_ids,
        model=invocation.model,
        invocation_checksum=invocation.checksum,
        results=(result,),
    )
    assert index.checksum is not None
    with pytest.raises(ValidationError, match="cover the scheduled tasks"):
        TrialDevRunIndexV1(
            release_id=invocation.release_id,
            scheduled_task_ids=("td-aaaaaaaaaaaa",),
            model=invocation.model,
            invocation_checksum=invocation.checksum,
            results=(result,),
        )


def test_numerical_analysis_accepts_declared_candidate_cardinality() -> None:
    """The actual CSV contract supports the full assigned candidate census."""
    from trialagentbench_verifier.numerical import NumericalEstimate, NumericalReportingContext

    assets = ("asset_alpha", "asset_beta", "asset_gamma", "asset_delta")
    targets = tuple(f"{asset}.{endpoint}" for asset in assets for endpoint in ("clinical_outcome", "safety_risk"))
    context = NumericalReportingContext(checkpoint_id="initial", target_ids=targets, evidence_artifacts=())
    estimates = tuple(NumericalEstimate(target_id=target, estimate=0, lower=-1, upper=1) for target in targets)
    assert tuple(row.target_id for row in estimates) == context.target_ids
    assert len(estimates) == 2 * len(assets)


def test_asset_identity_remains_a_strict_public_contract() -> None:
    """General cardinality must not weaken portable candidate identities."""

    with pytest.raises(ValidationError, match="asset_id"):
        _estimate("candidate A", "clinical_outcome")


@pytest.mark.parametrize("relative_path", ("../world.json", "/tmp/world.json", "documents\\brief.md"))
def test_trialdev_material_rejects_nonportable_paths(relative_path: str) -> None:
    """Participant materials cannot escape or vary with the host filesystem."""

    with pytest.raises(ValidationError, match="not portable"):
        TrialDevMaterialV1(relative_path=relative_path, description="Trial brief")


def test_trialdev_task_rejects_duplicate_material_paths() -> None:
    """One participant path cannot have competing task descriptions."""

    material = TrialDevMaterialV1(relative_path="documents/brief.md", description="Trial brief")
    with pytest.raises(ValidationError, match="must be unique"):
        TrialDevTaskV1(
            task_id="td-123456789abc",
            case_id="tdw-0001",
            stream="evidence_analysis",
            title="Comparative evidence analysis",
            clinical_question="Which candidate is supported by the available evidence?",
            available_materials=(material, material),
            requested_outputs=("Report estimates and uncertainty.",),
        )


def test_candidate_diagnostics_reject_reversed_propensity_range() -> None:
    """Reject an impossible reported propensity range at the schema boundary."""

    with pytest.raises(ValidationError, match="minimum cannot exceed"):
        TrialDevCandidateDiagnosticsV1(
            asset_id="asset_alpha",
            propensity_minimum=0.8,
            propensity_maximum=0.2,
        )


def test_candidate_diagnostics_retain_complementary_effective_sample_sizes() -> None:
    """Retain every declared ESS value for subsequent artifact verification."""

    result = TrialDevCandidateDiagnosticsV1(
        asset_id="asset_alpha",
        effective_sample_size_ratio=0.8,
        effective_sample_size_candidate=40.0,
        effective_sample_size_control=40.0,
    )
    assert result.effective_sample_size_ratio == 0.8
    assert result.effective_sample_size_candidate == result.effective_sample_size_control == 40.0
