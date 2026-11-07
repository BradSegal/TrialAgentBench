"""TrialEval live-run artifact contract tests."""

import pytest
from pydantic import ValidationError

from trialagentbench.contracts.core.runs import TrialEvalConditionProvenanceV1


def _payload() -> dict[str, object]:
    return {
        "procedure_assistance": "output_contract_only",
        "analysis_specification": "protocol_only",
        "analysis_surface_sha256": "d" * 64,
        "max_turns": 10,
        "prompt_set_sha256": "a" * 64,
        "rendered_system_prompt_sha256": "b" * 64,
        "tool_schema_sha256": "c" * 64,
    }


def test_condition_provenance_records_prompt_turn_budget() -> None:
    provenance = TrialEvalConditionProvenanceV1.model_validate(_payload())

    assert provenance.max_turns == 10


def test_condition_provenance_rejects_missing_prompt_turn_budget() -> None:
    payload = _payload()
    del payload["max_turns"]

    with pytest.raises(ValidationError, match="max_turns"):
        TrialEvalConditionProvenanceV1.model_validate(payload)
