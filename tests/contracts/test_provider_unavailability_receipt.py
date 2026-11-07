"""Only retained prospective outage evidence authorizes an Eval replacement."""

from __future__ import annotations

import json

import pytest
from contracts.test_trialeval_score_source_contracts import _write_missing_outcome
from pydantic import ValidationError
from trialagentbench_verifier.campaign import (
    INFRASTRUCTURE_ATTEMPT_OUTCOMES_V1,
    _attempt_outcome,
    _ComponentV1,
)
from trialagentbench_verifier.run_contracts import ProviderUnavailabilityReceiptV1

from trialagentbench.analysis.trialeval_score_rows import iter_trialeval_score_rows
from trialagentbench.contracts.core.runs import ProviderRequestEventV1, TrialEvalItemResultV1, TrialEvalRunConfigV1
from trialagentbench.io.checksums import canonical_payload_sha256


def request_event():
    return ProviderRequestEventV1(
        request_id="request-1",
        status="failed",
        benchmark="trialeval",
        unit_id="TASK1",
        phase_id="analysis",
        step_id="analysis",
        turn_index=2,
        elapsed_seconds=20,
        requested_model="test-model",
        provider_route="openai",
        usage_status="not_applicable",
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        request_attempts=1,
        transient_failure_count=1,
        backoff_seconds=0,
        failure_type="provider_error",
        exception_type="ProviderUnavailableError",
        unavailability_policy="defer_unanswered_v1",
        scientific_clock_excluded_seconds=20,
        retry_after_seconds=0,
    ).model_dump(mode="json")


def receipt_payload(event=None):
    event = request_event() if event is None else event
    return {"request_event": event, "request_event_sha256": canonical_payload_sha256(event)}


def outage_item(tmp_path):
    path = tmp_path / "items/TASK1.json"
    _write_missing_outcome(path)
    item = json.loads(path.read_text())
    config = item["run_config"]
    config.pop("run_identity_sha256")
    config["retry"]["unavailability_policy"] = "defer_unanswered_v1"
    item["run_config"] = TrialEvalRunConfigV1.create(**config).model_dump(mode="json")
    item["agent_output"].update(
        status="failed",
        failure_type="provider_unavailable",
        report=None,
        provider_unavailability=receipt_payload(),
    )
    item["scores"]["agent_status"] = "failed"
    path.write_text(json.dumps(item))
    return path, item


def test_native_outage_row_preserves_receipt_and_is_recoverable(tmp_path):
    path, item = outage_item(tmp_path)
    record = TrialEvalItemResultV1.model_validate(item)
    row = next(iter_trialeval_score_rows(tmp_path))
    assert row.observed_attempt_outcome == "provider_unavailable"
    assert row.observed_attempt_outcome in INFRASTRUCTURE_ATTEMPT_OUTCOMES_V1
    assert record.agent_output.provider_unavailability.request_event == request_event()
    assert (
        _attempt_outcome(
            agent_status="failed",
            failure_type="provider_unavailable",
            completion=_ComponentV1.model_validate(item["scores"]["outcome"]["completion"]),
            reproducibility=_ComponentV1.model_validate(item["scores"]["outcome"]["reproducibility"]),
        )
        == row.observed_attempt_outcome
    )
    assert path.read_text()  # Source remains present; grading does not replace its receipt.


@pytest.mark.parametrize("defect", ["missing_receipt", "charge_all_policy", "wrong_task", "wrong_model", "wrong_hash"])
def test_native_ingestion_rejects_outage_without_bound_proof(tmp_path, defect):
    path, item = outage_item(tmp_path)
    if defect == "missing_receipt":
        item["agent_output"].pop("provider_unavailability")
    elif defect == "charge_all_policy":
        config = item["run_config"]
        config.pop("run_identity_sha256")
        config["retry"].pop("unavailability_policy")
        item["run_config"] = TrialEvalRunConfigV1.create(**config).model_dump(mode="json")
    elif defect == "wrong_hash":
        item["agent_output"]["provider_unavailability"]["request_event_sha256"] = "0" * 64
    else:
        event = request_event()
        event["unit_id" if defect == "wrong_task" else "requested_model"] = "another"
        item["agent_output"]["provider_unavailability"] = receipt_payload(event)
    path.write_text(json.dumps(item))
    with pytest.raises(ValidationError):
        next(iter_trialeval_score_rows(tmp_path))


@pytest.mark.parametrize(
    "field,value",
    [
        ("response_received", True),
        ("exception_type", "ProviderRequestError"),
        ("unavailability_policy", None),
        ("status", "succeeded"),
        ("reasoning", "partial response"),
        ("reported_cost_usd", 0.01),
        ("completion_tokens", 1),
        ("request_attempts", 2),
        ("scientific_clock_excluded_seconds", 21),
        ("elapsed_seconds", float("inf")),
    ],
)
def test_independent_receipt_rejects_ambiguous_or_unmeasured_outage(field, value):
    event = request_event()
    event[field] = value
    with pytest.raises((ValidationError, ValueError)):
        ProviderUnavailabilityReceiptV1.model_validate(receipt_payload(event))


@pytest.mark.parametrize(
    "failure,expected",
    [
        ("provider_error", "provider_stop"),
        ("timeout", "budget_exhausted"),
        ("contract_error", "malformed_output"),
        ("turn_limit_no_submission", "model_noncompletion"),
    ],
)
def test_provider_stop_and_model_failures_do_not_gain_retries(tmp_path, failure, expected):
    path, item = outage_item(tmp_path)
    item["agent_output"].pop("provider_unavailability")
    item["agent_output"]["failure_type"] = failure
    path.write_text(json.dumps(item))
    row = next(iter_trialeval_score_rows(tmp_path))
    assert row.observed_attempt_outcome == expected
    assert row.observed_attempt_outcome not in INFRASTRUCTURE_ATTEMPT_OUTCOMES_V1
    assert (
        _attempt_outcome(
            agent_status="failed",
            failure_type=failure,
            completion=_ComponentV1.model_validate(item["scores"]["outcome"]["completion"]),
            reproducibility=_ComponentV1.model_validate(item["scores"]["outcome"]["reproducibility"]),
        )
        == expected
    )


def test_receipt_is_forbidden_on_a_provider_stop(tmp_path):
    _, item = outage_item(tmp_path)
    item["agent_output"]["failure_type"] = "provider_error"
    with pytest.raises(ValidationError):
        TrialEvalItemResultV1.model_validate(item)


@pytest.mark.parametrize("failure", ["cancelled", "evaluator_error", "pre_run_identity_error"])
def test_operator_interruption_is_infrastructure_in_both_outcome_derivations(tmp_path, failure):
    path, item = outage_item(tmp_path)
    item["agent_output"].pop("provider_unavailability")
    item["agent_output"]["failure_type"] = failure
    path.write_text(json.dumps(item))
    row = next(iter_trialeval_score_rows(tmp_path))
    expected = "pre_run_identity_exclusion" if failure == "pre_run_identity_error" else "evaluator_error"
    assert row.observed_attempt_outcome == expected
    assert row.observed_attempt_outcome in INFRASTRUCTURE_ATTEMPT_OUTCOMES_V1
    assert (
        _attempt_outcome(
            agent_status="failed",
            failure_type=failure,
            completion=_ComponentV1.model_validate(item["scores"]["outcome"]["completion"]),
            reproducibility=_ComponentV1.model_validate(item["scores"]["outcome"]["reproducibility"]),
        )
        == expected
    )


def test_proven_late_outage_stays_unscored_even_if_partial_script_is_replayable():
    from types import SimpleNamespace

    from trialagentbench.contracts.scoring.trialeval_scores import trialeval_observed_attempt_outcome_v1

    completed = SimpleNamespace(status="satisfied")
    outcome = SimpleNamespace(completion=completed, reproducibility=completed)
    for failure, expected in (
        ("provider_unavailable", "provider_unavailable"),
        ("provider_error", "completed"),
        ("cancelled", "evaluator_error"),
        ("runtime_error", "completed"),
        ("participant_session_terminated", "model_noncompletion"),
        ("pre_run_identity_error", "pre_run_identity_exclusion"),
    ):
        assert (
            trialeval_observed_attempt_outcome_v1(agent_status="failed", failure_type=failure, outcome=outcome)
            == expected
        )
        assert (
            _attempt_outcome(
                agent_status="failed", failure_type=failure, completion=completed, reproducibility=completed
            )
            == expected
        )
