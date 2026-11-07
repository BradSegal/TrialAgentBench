"""Authored terminal-value wire records are independently self-consistent."""

import pytest
from pydantic import ValidationError

from trialagentbench_verifier.response import TrialDevFrozenProgrammeValue


def test_signed_safety_benefit_and_sunk_stop_costs() -> None:
    value = TrialDevFrozenProgrammeValue(
        decision="advance",
        approved=True,
        clinical_value_usd=100,
        safety_cost_usd=-20,
        study_cost_usd=30,
        time_cost_usd=10,
        net_value_usd=80,
    )
    assert TrialDevFrozenProgrammeValue.model_validate_json(value.model_dump_json()) == value
    stopped = TrialDevFrozenProgrammeValue(
        decision="stop",
        clinical_value_usd=0,
        safety_cost_usd=0,
        study_cost_usd=30,
        time_cost_usd=10,
        net_value_usd=-40,
    )
    assert stopped.net_value_usd == -40


@pytest.mark.parametrize(
    "change", [{"net_value_usd": 81}, {"decision": "stop"}, {"approved": False}, {"study_cost_usd": -30}]
)
def test_inconsistent_terminal_record_is_rejected(change: dict[str, str | int]) -> None:
    payload = dict(
        decision="advance",
        approved=True,
        clinical_value_usd=100,
        safety_cost_usd=-20,
        study_cost_usd=30,
        time_cost_usd=10,
        net_value_usd=80,
    )
    with pytest.raises(ValidationError):
        TrialDevFrozenProgrammeValue.model_validate(payload | change)
