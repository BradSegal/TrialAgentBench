"""Deterministic enrollment from finite observable patient frames."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Collection
from typing import Self

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from trialagentbench.contracts.trialdev.pools import TrialDevArmId, TrialDevRecruitmentFrame
from trialagentbench.trialdev.errors import TrialDevActionRejectedError


class TrialDevArmEnrollment(BaseModel):
    """Agent-selected enrollment cap for one concurrent randomized arm."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    arm_id: TrialDevArmId
    participant_count: int = Field(ge=1, le=3000, strict=True)


class TrialDevRecruitmentPreference(BaseModel):
    """Validated enrollment preference for one group."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        json_schema_extra={
            "description": "Share of screened candidates in one recruitment group (see the brief) who are enrolled."
        },
    )

    group_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    screening_fraction: float = Field(ge=0.1, le=1)


class TrialDevRecruitmentSelection(BaseModel):
    """Validated programme enrollment selection."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        json_schema_extra={
            "description": "Share of screened candidates who are enrolled, overall and by recruitment group."
        },
    )

    background_fraction: float = Field(
        default=1,
        ge=0.1,
        le=1,
        description="Fraction accepted for enrollment among screened candidates outside selected groups; defaults to every candidate.",
    )
    preferences: tuple[TrialDevRecruitmentPreference, ...] = Field(
        default=(),
        max_length=8,
        description="Group fractions override the background. On overlap, use the largest selected group fraction.",
    )

    @model_validator(mode="after")
    def validate_preferences(self) -> Self:
        """Reject duplicate group choices rather than applying an implicit override."""
        if len({item.group_id for item in self.preferences}) != len(self.preferences):
            raise ValueError("recruitment preferences must name each group at most once")
        return self


class TrialDevEnrollmentPlan(BaseModel):
    """Immutable exact arm caps and clinical screening choices for an authored study."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    study_id: str = Field(min_length=1)
    site_ids: tuple[str, ...] = Field(min_length=2)
    arms: tuple[TrialDevArmEnrollment, ...] = Field(min_length=2, max_length=7)
    recruitment: TrialDevRecruitmentSelection = Field(default_factory=TrialDevRecruitmentSelection)
    screen_failure_fraction: float = Field(ge=0, lt=1, description="Programme screen-failure share")

    @model_validator(mode="after")
    def validate_enrollment(self) -> Self:
        """Require unique sites and arms within the supported study size."""
        if len(set(self.site_ids)) != len(self.site_ids) or any(not site for site in self.site_ids):
            raise ValueError("enrollment requires unique nonempty activated site identities")
        arms = {arm.arm_id for arm in self.arms}
        if len(arms) != len(self.arms) or "control" not in arms:
            raise ValueError("enrollment requires unique arms including control")
        if sum(arm.participant_count for arm in self.arms) > 8000:
            raise ValueError("enrollment exceeds the 8000-person study limit")
        return self


def plan_trialdev_enrollment(
    frame: TrialDevRecruitmentFrame,
    *,
    study_id: str,
    arms: tuple[TrialDevArmEnrollment, ...],
    recruitment: TrialDevRecruitmentSelection,
) -> TrialDevEnrollmentPlan:
    """Bind exact arm sizes and clinical screening preferences to a frozen frame.

    Parameters
    ----------
    frame
        Public clinical group definitions and site capacities.
    study_id
        Immutable identity of the authored study.
    arms
        Concurrent arm caps in immutable treatment order, including control.
    recruitment
        Background and group screening fractions. On overlap the largest group
        fraction applies; outside selected groups the background fraction applies.

    Returns
    -------
    TrialDevEnrollmentPlan
        Exact arm caps without promised per-group counts or analysis support.

    Raises
    ------
    ValueError
        If a group is unknown, arms are duplicated or caps are invalid.

    Examples
    --------
    ``plan_trialdev_enrollment(frame, study_id="trial-1", arms=arms, recruitment=selection)``.
    """
    if {row.group_id for row in recruitment.preferences} - {row.group_id for row in frame.groups}:
        raise TrialDevActionRejectedError(
            f"unknown recruitment group; groups in this programme: {', '.join(row.group_id for row in frame.groups)}"
        )
    return TrialDevEnrollmentPlan(
        study_id=study_id,
        site_ids=tuple(site.site_id for site in frame.sites),
        arms=arms,
        recruitment=recruitment,
        screen_failure_fraction=frame.screen_failure_fraction,
    )


def _priority(seed: int, namespace: str, identity: str | int | list[str | int]) -> bytes:
    encoded = json.dumps([seed, namespace, identity], separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).digest()


def allocate_trialdev_patients(
    baseline: pd.DataFrame,
    plan: TrialDevEnrollmentPlan,
    *,
    seed: int,
    unavailable_participant_ids: Collection[str | int] = (),
    screening_round: int = 0,
) -> TrialDevStudyReservation:
    """Screen patients without replacement and randomize accepted patients to exact arm caps.

    Parameters
    ----------
    baseline
        Frozen participant identities, recorded clinical group memberships, sites and
        screening times. Other columns, including potential outcomes, are ignored.
    plan
        Exact arm caps, screening fractions and the programme screen-failure share from a
        validated authored study design.
    seed
        Private run seed. It decides which of a site's candidates presents at each
        of that site's frozen arrival times, screening acceptance and allocation to
        arms. Study names change none of these, and arrival clocks stay frozen.
    unavailable_participant_ids
        Patients already enrolled or reserved in this run, supplied by its ledger.
    screening_round
        Number of earlier studies in this run. Each study screens afresh, so a
        candidate who failed screening before passes again with the programme's
        passing probability rather than failing deterministically.

    Returns
    -------
    TrialDevStudyReservation
        Selected patients and randomized arms, with the complete screening clock
        through the final selected candidate. The caller enforces phase/programme
        limits and commits actual enrollment and charges atomically. Reserved
        future patients are not treated as already enrolled or screened.

    Raises
    ------
    ValueError
        If identifiers, support, prior-use state or available capacity are invalid.

    Examples
    --------
    ``allocation = allocate_trialdev_patients(baseline, plan, seed=42)`` is replayable.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("recruitment requires a nonnegative integer run seed")
    if isinstance(screening_round, bool) or not isinstance(screening_round, int) or screening_round < 0:
        raise ValueError("screening round must be a nonnegative integer")
    preferences = plan.recruitment.preferences
    membership_columns = [f"recruitment_{item.group_id}" for item in preferences]
    required = ["participant_id", "site_id", "screening_day", *membership_columns]
    if not set(required).issubset(baseline.columns):
        raise ValueError("recruitment frame is missing declared patient or group fields")
    frame = baseline.loc[:, required].copy()
    if frame.empty or frame.isna().any().any():
        raise ValueError("recruitment frame must contain complete patient identities and group memberships")
    if not frame.participant_id.map(
        lambda value: not isinstance(value, bool)
        and (isinstance(value, int) or (isinstance(value, str) and bool(value)))
    ).all():
        raise ValueError("recruitment participant identities must be integers or nonempty strings")
    if frame.participant_id.duplicated().any():
        raise ValueError("recruitment participant identities must be unique")
    if not frame.site_id.map(lambda value: isinstance(value, str) and bool(value)).all():
        raise ValueError("recruitment requires complete site identities")
    if not frame.screening_day.map(lambda value: math.isfinite(value) and value >= 0).all():
        raise ValueError("recruitment requires finite nonnegative frozen screening times")
    all_ids = set(frame.participant_id)
    unavailable = set(unavailable_participant_ids)
    if not unavailable.issubset(all_ids):
        raise ValueError("prior-use ledger contains patients outside this pool")
    if any(not pd.api.types.is_bool_dtype(frame[column]) for column in membership_columns):
        raise ValueError("frozen recruitment group memberships must be boolean")
    # The frozen clock fixes when each site's candidates present; the run seed fixes who
    # presents in each slot. Independent runs therefore recruit independent samples of
    # the fixed population, whose contrasts are the study targets.
    arrival = frame.assign(_order=[_priority(seed, "arrival", patient) for patient in frame.participant_id])
    slots = arrival.sort_values(["site_id", "screening_day", "participant_id"], kind="stable").screening_day
    frame.loc[arrival.sort_values(["site_id", "_order"], kind="stable").index, "screening_day"] = slots.to_numpy()
    activated_sites = set(plan.site_ids)
    available = frame.loc[~frame.participant_id.isin(unavailable) & frame.site_id.isin(activated_sites)]
    available = available.sort_values(["screening_day", "site_id", "participant_id"], kind="stable")
    screening_queue = available.participant_id.tolist()
    screening_ordinal = {patient: index + 1 for index, patient in enumerate(screening_queue)}
    fractions = pd.Series(plan.recruitment.background_fraction, index=available.index, dtype=float)
    matched = pd.Series(False, index=available.index)
    selected_fraction = pd.Series(0.0, index=available.index)
    for preference in preferences:
        members = available[f"recruitment_{preference.group_id}"]
        matched |= members
        selected_fraction.loc[members] = selected_fraction.loc[members].clip(lower=preference.screening_fraction)
    fractions.loc[matched] = selected_fraction.loc[matched]
    # A candidate is enrolled only after passing screening (an outcome-independent programme
    # screen-failure share) and the study's own screening fraction; failures still consume
    # screening effort and their arrival slot.
    passing = int((1 - plan.screen_failure_fraction) * 2**256)
    accepted = [
        patient
        for patient, fraction in zip(available.participant_id, fractions, strict=True)
        if int.from_bytes(_priority(seed, "screen-failure", [screening_round, patient]), "big") < passing
        and int.from_bytes(_priority(seed, "screening", patient), "big") < int(float(fraction) * 2**256)
    ]
    count = sum(arm.participant_count for arm in plan.arms)
    if len(accepted) < count:
        raise TrialDevActionRejectedError(
            f"only {len(accepted)} unenrolled candidates remain under these screening fractions; "
            f"the design needs {count}"
        )
    selected = accepted[:count]
    randomized = sorted(
        selected, key=lambda patient: (_priority(seed, "allocation", patient), str(type(patient)), str(patient))
    )
    arm_ids = [arm.arm_id for arm in plan.arms for _ in range(arm.participant_count)]
    assignments = dict(zip(randomized, arm_ids, strict=True))
    final_screen = screening_ordinal[selected[-1]]
    return TrialDevStudyReservation(
        plan=plan,
        patients=tuple(
            TrialDevReservedPatient(
                participant_id=patient, arm_id=assignments[patient], screening_ordinal=screening_ordinal[patient]
            )
            for patient in selected
        ),
        screening_days=tuple(float(day) for day in available.screening_day.iloc[:final_screen]),
    )


class TrialDevCommittedPatient(BaseModel):
    """Private enrollment identity bound to one public readout participant."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    participant_id: StrictInt | StrictStr
    participant_label: str
    arm_id: str
    enrollment_day: float = Field(ge=0)


class TrialDevStudyEnrollment(BaseModel):
    """Retained actual enrollment and fixed design identity for one study."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    study_id: str = Field(min_length=1)
    design_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    patients: tuple[TrialDevCommittedPatient, ...]
    screened: int = Field(default=0, ge=0, strict=True)


class TrialDevReservedPatient(BaseModel):
    """Selected future patient and randomized arm, before actual enrollment."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    participant_id: StrictInt | StrictStr
    arm_id: str
    screening_ordinal: int = Field(ge=1, strict=True)


class TrialDevStudyReservation(BaseModel):
    """Immutable patient allocation retained after a reservation is released."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    plan: TrialDevEnrollmentPlan
    patients: tuple[TrialDevReservedPatient, ...]
    screening_days: tuple[float, ...] = Field(min_length=1)
    closed: bool = False
    closed_arms: tuple[str, ...] = ()
    retained_control_ids: tuple[StrictInt | StrictStr, ...] | None = None

    @model_validator(mode="after")
    def validate_allocation(self) -> Self:
        """Require unique patients and exact agreement with the committed arm caps."""
        if len(set(self.closed_arms)) != len(self.closed_arms) or not set(self.closed_arms).issubset(
            {quota.arm_id for quota in self.plan.arms} - {"control"}
        ):
            raise ValueError("arm closure requires unique active arm identities")
        ids = [patient.participant_id for patient in self.patients]
        if len(ids) != len(set(ids)):
            raise ValueError("reservation patients must be unique")
        ordinals = [patient.screening_ordinal for patient in self.patients]
        if len(ordinals) != len(set(ordinals)):
            raise ValueError("reservation screening ordinals must be unique")
        if (
            not ordinals
            or any(not math.isfinite(day) or day < 0 for day in self.screening_days)
            or tuple(sorted(self.screening_days)) != self.screening_days
            or max(ordinals) != len(self.screening_days)
        ):
            raise ValueError("reservation must retain its complete ordered screening clock")
        counts: dict[str, int] = {}
        for patient in self.patients:
            counts[patient.arm_id] = counts.get(patient.arm_id, 0) + 1
        if counts != {arm.arm_id: arm.participant_count for arm in self.plan.arms}:
            raise ValueError("reservation allocation must match its exact arm caps")
        if self.retained_control_ids is not None:
            retained = set(self.retained_control_ids)
            controls = [patient for patient in self.patients if patient.arm_id == "control"]
            if (
                not retained
                or len(retained) != len(self.retained_control_ids)
                or not retained.issubset({p.participant_id for p in controls})
            ):
                raise ValueError("retained controls must be nonempty unique original control patients")
        return self

    @property
    def active_patients(self) -> tuple[TrialDevReservedPatient, ...]:
        """Return reservations still held for future enrollment, excluding released slots."""
        if self.closed:
            return ()
        controls = None if self.retained_control_ids is None else set(self.retained_control_ids)
        return tuple(
            patient
            for patient in self.patients
            if patient.arm_id not in self.closed_arms
            and (patient.arm_id != "control" or controls is None or patient.participant_id in controls)
        )
