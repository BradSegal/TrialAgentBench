"""Risk-adjusted terminal sponsor value from frozen contrasts, approval and programme resources."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from trialagentbench_verifier.response import TrialDevFrozenProgrammeValue as TrialDevFrozenProgrammeValue

from trialagentbench.contracts.trialdev.pools import TrialDevFrozenPool, TrialDevPoolContrast
from trialagentbench.io import read_json_model, sha256_file
from trialagentbench.trialdev.resources import TrialDevBudgetLedger


class TrialDevSponsorPreferences(BaseModel):
    """Public risk-adjusted sponsor economics in clinical units and USD.

    An approved treatment earns a constant annual operating profit from launch,
    one regulatory review after the terminal decision, until market exclusivity
    ends. Profit scales linearly with benefit in typical-pivotal-effect units and
    falls with excess safety risk. Cash flows are discounted to programme day zero,
    so delay costs a share of the asset's value through discounting and lost
    exclusivity. The defaults are directional estimates, not elicited utilities.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    outcome_unit: str = Field(min_length=1)
    benefit_unit: float = Field(gt=0, description="Typical pivotal-trial benefit on the outcome scale")
    annual_profit_per_benefit_unit_usd: float = Field(default=250_000_000, ge=0)
    annual_cost_per_safety_percentage_point_usd: float = Field(
        default=20_000_000, ge=0, description="Annual profit lost per percentage point of excess safety risk"
    )
    discount_rate_per_year: float = Field(default=0.10, ge=0, lt=1)
    exclusivity_end_day: int = Field(default=6575, gt=0, strict=True)
    regulatory_review_days: int = Field(default=365, ge=0, strict=True)
    programme_cost_per_day_usd: float = Field(default=15_000, ge=0)


def trialdev_market_years(preferences: TrialDevSponsorPreferences, decision_day: float) -> float:
    """Return discounted exclusive market years for a treatment advanced on a given day.

    Parameters
    ----------
    preferences
        Declared discounting, review and exclusivity terms.
    decision_day
        Programme day of the advancement decision.

    Returns
    -------
    float
        Present value at programme day zero of one dollar per year from launch to
        exclusivity end; zero if launch would follow exclusivity end.

    Raises
    ------
    ValueError
        If the decision day is not finite and nonnegative.

    Examples
    --------
    With 10% discounting, launch in year 4 and exclusivity to year 18 give about
    5.3 discounted years; each further year of delay loses roughly 12%.
    """
    if isinstance(decision_day, bool) or not math.isfinite(decision_day) or decision_day < 0:
        raise ValueError("market timing requires a finite nonnegative decision day")
    launch = (decision_day + preferences.regulatory_review_days) / 365.25
    end = preferences.exclusivity_end_day / 365.25
    if launch >= end:
        return 0.0
    rate = math.log1p(preferences.discount_rate_per_year)
    if rate == 0:
        return end - launch
    return (math.exp(-rate * launch) - math.exp(-rate * end)) / rate


def trialdev_annual_net_profit_usd(
    preferences: TrialDevSponsorPreferences, *, outcome_mean_difference: float, safety_risk_difference: float
) -> float:
    """Return annual operating profit for a launched treatment with given true effects.

    Parameters
    ----------
    preferences
        Declared benefit scale and prices.
    outcome_mean_difference
        Treatment-minus-control difference on the lower-is-better outcome scale.
    safety_risk_difference
        Treatment-minus-control safety-event risk difference.

    Returns
    -------
    float
        Profit from benefit less the cost of excess risk; negative for harm.

    Examples
    --------
    One typical benefit unit with no excess risk earns the full annual profit.
    """
    benefit = -outcome_mean_difference / preferences.benefit_unit
    return (
        benefit * preferences.annual_profit_per_benefit_unit_usd
        - safety_risk_difference * 100 * preferences.annual_cost_per_safety_percentage_point_usd
    )


def trialdev_frozen_programme_value(
    *,
    decision: Literal["stop", "advance"],
    target: TrialDevPoolContrast | None,
    preferences: TrialDevSponsorPreferences,
    ledger: TrialDevBudgetLedger,
    elapsed_days: float,
) -> TrialDevFrozenProgrammeValue:
    """Value an explicit terminal choice using fixed truth and actual resource obligations.

    Parameters
    ----------
    decision
        Explicit lawful terminal decision, never inferred from model noncompletion.
    target
        Selected asset/regimen's verified fixed-population contrast for advancement;
        None for stopping. Clinical outcomes use treatment-minus-control differences
        on the current four programmes' lower-is-better scales.
    preferences
        Declared benefit scale, safety price, discounting and programme overhead.
    ledger
        Authenticated terminal programme accounting, including the regulatory
        review recorded with an advancement.
    elapsed_days
        Actual terminal programme day, no earlier than recorded activity.

    Returns
    -------
    TrialDevFrozenProgrammeValue
        Discounted market value of an approved treatment minus its discounted safety
        cost, nominal study expenditure/liabilities and programme overhead. A
        rejected submission keeps every cost and earns no market value.

    Raises
    ------
    ValueError
        If decision, target units, review or terminal calendar are inconsistent.

    Examples
    --------
    Stopping has zero market value but retains all sunk and unavoidable costs.
    This function values a verified choice; it does not authenticate programme history
    or turn evaluator faults and unfinished runs into valid stopping decisions.
    """
    if decision not in ("stop", "advance") or (decision == "advance") != (target is not None):
        raise ValueError("terminal decision requires exactly its corresponding fixed target")
    if isinstance(elapsed_days, bool) or not math.isfinite(elapsed_days) or elapsed_days < 0:
        raise ValueError("terminal programme time must be finite and nonnegative")
    latest = ledger.current_day
    if elapsed_days < latest:
        raise ValueError("terminal value cannot precede recorded programme activity")
    if ledger.recruitment_commitments and ledger.resource_day != elapsed_days:
        raise ValueError("terminal valuation requires resource accounting at the terminal clock")
    terminal = ledger.terminal
    if ledger.recruitment_commitments and (terminal is None or terminal.decision != decision):
        raise ValueError("valuation requires the recorded terminal decision")
    approved = False
    clinical = safety = 0.0
    if target is not None:
        if terminal is None or terminal.regulatory_review is None:
            raise ValueError("advancement value requires the recorded regulatory review")
        if target.asset_id != terminal.asset_id or target.dose != terminal.dose:
            raise ValueError("fixed target does not match the terminal asset and regimen")
        if target.outcome_unit != preferences.outcome_unit:
            raise ValueError("sponsor benefit scale must match the fixed target's clinical units")
        approved = terminal.regulatory_review.approved
        if approved:
            years = trialdev_market_years(preferences, elapsed_days)
            clinical = (
                -target.outcome_mean_difference
                / preferences.benefit_unit
                * preferences.annual_profit_per_benefit_unit_usd
                * years
            )
            safety = (
                target.safety_risk_difference * 100 * preferences.annual_cost_per_safety_percentage_point_usd * years
            )
    study_cost = ledger.spent_usd + ledger.remaining_liability_usd
    time_cost = elapsed_days * preferences.programme_cost_per_day_usd
    return TrialDevFrozenProgrammeValue(
        decision=decision,
        approved=approved,
        clinical_value_usd=clinical,
        safety_cost_usd=safety,
        study_cost_usd=study_cost,
        time_cost_usd=time_cost,
        net_value_usd=clinical - safety - study_cost - time_cost,
    )


def grade_trialdev_finished_programme(
    *,
    pool_root: Path,
    ledger: TrialDevBudgetLedger,
    preferences: TrialDevSponsorPreferences,
) -> TrialDevFrozenProgrammeValue:
    """Grade a recorded terminal choice against its bound private finite population.

    Parameters
    ----------
    pool_root
        Evaluator-owned frozen pool directory, unavailable to the participant.
    ledger
        Evaluator-owned terminal programme history.
    preferences
        Sponsor preferences fixed by the task before the run.

    Returns
    -------
    TrialDevFrozenProgrammeValue
        Fixed-population benefit and safety value less actual resource costs.

    Raises
    ------
    ValueError
        If the programme is unfinished or its bound pool identity is inconsistent.

    Examples
    --------
    A stop selects no treatment target; an advancement selects exactly the recorded
    asset and regimen. Neither the participant nor the caller supplies a contrast.
    """
    terminal = ledger.terminal
    if terminal is None:
        raise ValueError("unfinished programmes require noncompletion handling, not terminal grading")
    manifest_path = pool_root / "manifest.json"
    if ledger.pool_manifest_sha256 != sha256_file(manifest_path):
        raise ValueError("terminal programme and frozen pool manifest identities disagree")
    manifest = read_json_model(TrialDevFrozenPool, manifest_path)
    target = None
    if terminal.decision == "advance":
        matches = tuple(
            row for row in manifest.targets if row.asset_id == terminal.asset_id and row.dose == terminal.dose
        )
        if len(matches) != 1:
            raise ValueError("terminal treatment has no unique fixed-population target")
        target = matches[0]
    return trialdev_frozen_programme_value(
        decision=terminal.decision,
        target=target,
        preferences=preferences,
        ledger=ledger,
        elapsed_days=terminal.programme_day,
    )
