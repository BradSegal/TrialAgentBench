"""Resource receipts are checkable without the simulation or harness packages."""

import pytest
from pydantic import ValidationError

from trialagentbench_verifier.resource_accounting import TrialDevAccountingCase


def _case():
    return dict(
        phase="early_safety",
        cost_multiplier=1.0,
        programme_day=28,
        enrollment_days=(0, 14, 28),
        sites=2,
        screened=5,
        follow_up_days=84,
        assessment_days=(28, 84),
        surveillance_days=(14, 28, 84),
        setup=150000,
        per_site=25000,
        screening=250,
        enrollment=1500,
        followup_day=15,
        assessment=200,
        accrued_usd=206980,
        remaining_liability_usd=4350,
    )


def test_partial_followup_preserves_unavoidable_bills_and_charges_shared_visits_once():
    case = TrialDevAccountingCase.model_validate(_case())
    # 205750 fixed/enrollment + 252 patient-days * 15 + 9 distinct visits * 200.
    assert case.accrued_usd + case.remaining_liability_usd == 211330
    final = TrialDevAccountingCase.model_validate(
        _case() | {"programme_day": 112, "accrued_usd": 211330, "remaining_liability_usd": 0}
    )
    assert final.accrued_usd == case.accrued_usd + case.remaining_liability_usd


@pytest.mark.parametrize("field", ["accrued_usd", "remaining_liability_usd", "screening", "followup_day"])
def test_changed_bill_or_unit_price_cannot_pass_independent_accounting(field):
    payload = _case()
    payload[field] += 1
    with pytest.raises(ValidationError, match="independent participant-level accounting"):
        TrialDevAccountingCase.model_validate(payload)
