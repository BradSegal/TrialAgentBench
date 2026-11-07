"""Standalone assembly of one assigned arm's accrued participant records."""

import os
import uuid
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import pandas as pd
from clinical_data_tables import available_clinical_records, validate_clinical_epochs
from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench.contracts.trialdev.pools import TrialDevArmId, TrialDevFrozenPool, TrialDevPoolPartition
from trialagentbench.contracts.trialdev.pools import TrialDevAssignedArm as TrialDevAssignedArm
from trialagentbench.contracts.trialdev.release import TrialDevDose
from trialagentbench.io import canonical_payload_sha256, read_json_model, sha256_file, write_json_model
from trialagentbench.io.atomic import _sync_directory
from trialagentbench.io.checksums import sha256_dir_files
from trialagentbench.io.directory import staged_directory
from trialagentbench.trialdev.design import TrialDevStudyRequest, bind_trialdev_study_request
from trialagentbench.trialdev.documents import write_readout_dictionary
from trialagentbench.trialdev.errors import TrialDevActionRejectedError
from trialagentbench.trialdev.observations import (
    trialdev_contact_records,
    trialdev_exposure_records,
    trialdev_intercurrent_records,
    trialdev_measurement_information_day,
    trialdev_measurement_records,
    trialdev_recurrent_information_day,
    trialdev_recurrent_records,
    trialdev_safety_records,
)
from trialagentbench.trialdev.recruitment import (
    TrialDevCommittedPatient,
    TrialDevEnrollmentPlan,
    TrialDevStudyEnrollment,
)
from trialagentbench.trialdev.resources import (
    MINIMUM_PHASE_PARTICIPANTS_PER_ARM,
    TrialDevBudgetLedger,
    TrialDevCostReceipt,
    TrialDevRecruitmentCommitment,
    TrialDevRecruitmentContinuation,
    TrialDevStudyCost,
    TrialDevStudyTariff,
    TrialDevTerminalDecision,
    advance_trialdev_clock,
    close_trialdev_arm,
    close_trialdev_reservation,
    commit_trialdev_recruitment,
    continue_trialdev_recruitment,
    finish_trialdev_programme,
    reconcile_trialdev_cost,
    reconcile_trialdev_enrollment,
    reduce_trialdev_control,
    reserve_trialdev_patients,
    trialdev_committed_accrual,
    trialdev_study_cost,
    validate_trialdev_phase_design,
)
from trialagentbench.util.run_ownership import exclusive_run_directory


def read_pool_table(
    root: Path,
    manifest: TrialDevFrozenPool,
    relative_path: str,
    *,
    participant_ids: list[int | str] | None = None,
) -> pd.DataFrame:
    """Read one integrity-verified frozen-pool table, optionally restricted to participants."""
    artifact = next((item for item in manifest.artifacts if item.relative_path == relative_path), None)
    path = root / relative_path
    if artifact is None or not path.resolve().is_relative_to(root):
        raise ValueError("readout artifact is outside the frozen pool inventory")
    if path.stat().st_size != artifact.byte_length:
        raise ValueError("frozen readout artifact failed integrity verification")
    filters = None if participant_ids is None else [("participant_id", "in", participant_ids)]
    return pd.read_parquet(path.open("rb"), filters=filters)


def reserve_trialdev_study(
    pool_root: Path, budget_path: Path, plan: TrialDevEnrollmentPlan, *, seed: int
) -> TrialDevBudgetLedger:
    """Persist a patient reservation under the programme's exclusive transaction lock.

    Parameters
    ----------
    pool_root
        Immutable evaluator-only patient pool.
    budget_path
        Existing private programme ledger, outside the pool.
    plan
        Exact study arm caps and clinical screening preferences.
    seed
        Private recruitment seed fixed for the programme.

    Returns
    -------
    TrialDevBudgetLedger
        Persisted reservation state, with spending and actual enrollment unchanged.

    Raises
    ------
    ValueError
        If integrity, capacity, study identity or exclusive ownership fails.

    Examples
    --------
    Retry with the same study design to recover its original allocation.
    """
    root = pool_root.resolve()
    if budget_path.resolve().is_relative_to(root):
        raise ValueError("programme ledger cannot modify immutable pool storage")
    with exclusive_run_directory(budget_path.parent):
        ledger = read_json_model(TrialDevBudgetLedger, budget_path)
        manifest_path = root / "manifest.json"
        manifest = read_json_model(TrialDevFrozenPool, manifest_path)
        baseline = read_pool_table(root, manifest, "baseline.parquet")
        updated = reserve_trialdev_patients(
            ledger, baseline, plan, pool_manifest_sha256=sha256_file(manifest_path), seed=seed
        )
        write_json_model(budget_path, updated)
        return updated


def commit_trialdev_study_recruitment(
    budget_path: Path, commitment: TrialDevRecruitmentCommitment, *, pool_root: Path
) -> TrialDevBudgetLedger:
    """Persist prospective recruitment parameters under exclusive run ownership.

    Parameters
    ----------
    budget_path
        Existing private ledger containing the study reservation.
    commitment
        Immutable treatment, observation and recruitment design.
    pool_root
        Frozen pool whose inventory must support the complete commitment.

    Returns
    -------
    TrialDevBudgetLedger
        Atomically persisted commitment and current resource accounting.

    Raises
    ------
    ValueError
        If the study is unknown, already active, changed, or concurrently owned.

    Examples
    --------
    Bind recruitment before calculating and publishing the first accrued readout.
    """
    with exclusive_run_directory(budget_path.parent):
        ledger = read_json_model(TrialDevBudgetLedger, budget_path)
        manifest_path = pool_root / "manifest.json"
        manifest = read_json_model(TrialDevFrozenPool, manifest_path)
        updated = _admit_trialdev_study(ledger, commitment, manifest, sha256_file(manifest_path))
        updated = advance_trialdev_clock(updated, updated.current_day)
        write_json_model(budget_path, updated)
        return updated


def _admit_trialdev_study(
    ledger: TrialDevBudgetLedger,
    commitment: TrialDevRecruitmentCommitment,
    manifest: TrialDevFrozenPool,
    pool_sha256: str,
) -> TrialDevBudgetLedger:
    if ledger.pool_manifest_sha256 != pool_sha256:
        raise ValueError("study commitment cannot change its reserved population")
    if any(arm.partition not in manifest.partitions for arm in commitment.arms):
        raise ValueError("committed treatments differ from the frozen pool")
    if not set((*commitment.assessment_days, *commitment.surveillance_days)).issubset({0, *manifest.collection_days}):
        raise TrialDevActionRejectedError(
            "assessment_days and surveillance_days must be visit-grid days no later than follow-up: "
            + ", ".join(str(day) for day in manifest.collection_days if day <= commitment.follow_up_days)
        )
    reservation = next((item for item in ledger.reservations if item.plan.study_id == commitment.study_id), None)
    if reservation is None:
        raise ValueError("study commitment requires an existing patient reservation")
    expected_sites = tuple(site.site_id for site in manifest.recruitment.sites[: commitment.sites])
    if len(expected_sites) != commitment.sites or reservation.plan.site_ids != expected_sites:
        raise ValueError("study commitment cannot change its activated programme sites")
    validate_trialdev_phase_design(commitment, reservation.plan, manifest.collection_days)
    return commit_trialdev_recruitment(ledger, commitment)


def start_trialdev_study(
    pool_root: Path,
    budget_path: Path,
    request: TrialDevStudyRequest,
    *,
    study_id: str,
    start_day: float,
    seed: int,
    validate_only: bool = False,
) -> TrialDevBudgetLedger:
    """Atomically admit an agent-authored study and reserve its frozen patients.

    Parameters
    ----------
    pool_root
        Private frozen population and treatment inventory.
    budget_path
        Existing private programme ledger.
    request
        Public numerical study choices without evaluator paths or patient identities.
    study_id
        Runner-owned study identity, retained across retries.
    start_day
        Runner-owned current programme calendar day.
    seed
        Private run recruitment seed, fixed across the programme.

    validate_only
        Apply the same admission rules in memory, without persisting a reservation
        or charging resources. The returned ledger is a hypothetical projection.

    Returns
    -------
    TrialDevBudgetLedger
        Persisted allocation and admitted study. Invalid designs leave no reservation,
        expenditure or participant output behind.

    Raises
    ------
    ValueError
        If design, phase, capacity, population or immutable run identity is invalid.

    Examples
    --------
    Follow successful admission with ``write_trialdev_committed_readout``.
    """
    root = pool_root.resolve()
    if budget_path.resolve().is_relative_to(root):
        raise ValueError("programme ledger cannot modify immutable pool storage")
    with exclusive_run_directory(budget_path.parent):
        ledger = read_json_model(TrialDevBudgetLedger, budget_path)
        manifest_path = root / "manifest.json"
        manifest = read_json_model(TrialDevFrozenPool, manifest_path)
        plan, commitment = bind_trialdev_study_request(request, manifest, study_id=study_id, start_day=start_day)
        baseline = read_pool_table(root, manifest, "baseline.parquet")
        reserved = reserve_trialdev_patients(
            ledger, baseline, plan, pool_manifest_sha256=sha256_file(manifest_path), seed=seed
        )
        admitted = _admit_trialdev_study(reserved, commitment, manifest, sha256_file(manifest_path))
        admitted = advance_trialdev_clock(admitted, admitted.current_day)
        if not validate_only:
            write_json_model(budget_path, admitted)
        return admitted


def continue_trialdev_study_recruitment(
    budget_path: Path, continuation: TrialDevRecruitmentContinuation
) -> TrialDevBudgetLedger:
    """Persist a paused study's continuation under exclusive programme ownership.

    Parameters
    ----------
    budget_path
        Private ledger containing the paused cohort and published readout receipt.
    continuation
        Study identity and immutable continuation date.

    Returns
    -------
    TrialDevBudgetLedger
        Atomically persisted continuation with enrollment and obligations reconciled.

    Raises
    ------
    ValueError
        If the decision changes, is backdated, lacks a paused cohort or conflicts with another process.

    Examples
    --------
    Recover the same continuation on retry before publishing later accrued records.
    """
    with exclusive_run_directory(budget_path.parent):
        updated = continue_trialdev_recruitment(read_json_model(TrialDevBudgetLedger, budget_path), continuation)
        updated = advance_trialdev_clock(updated, updated.current_day)
        write_json_model(budget_path, updated)
        return updated


def trialdev_next_readout_day(pool_root: Path, ledger: TrialDevBudgetLedger, study_id: str) -> float:
    """Resolve the next committed look from available primary records or its deadline.

    Parameters
    ----------
    pool_root
        Private frozen population; native generation is never called.
    ledger
        Current committed allocation, continuation and readout history.
    study_id
        Study whose next unconsumed look is requested.

    Returns
    -------
    float
        Threshold availability day, or the prospective deadline if the threshold
        remains unmet, never earlier than the current programme clock. A delayed
        request uses current accrued evidence and retains all incurred costs.
        A deadline readout may contain fewer complete observations than requested.

    Raises
    ------
    ValueError
        If no look remains, population identity changes or primary records are invalid.

    Examples
    --------
    Calendar looks resolve without reading outcomes; information looks count available records only.
    """
    design = next((item for item in ledger.recruitment_commitments if item.study_id == study_id), None)
    receipts = [item for item in ledger.receipts if item.study_id == study_id and item.kind == "readout"]
    if design is None or len(receipts) >= len(design.looks):
        raise TrialDevActionRejectedError(f"{study_id} has no remaining committed look")
    look = design.looks[len(receipts)]

    def current_cutoff(scheduled: float) -> float:
        actual = max(scheduled, ledger.current_day)
        if receipts and actual <= receipts[-1].programme_day:
            raise TrialDevActionRejectedError("next look requires a later observation cutoff")
        return actual

    if look.completed_participants is None or ledger.current_day >= look.deadline_day:
        return current_cutoff(look.deadline_day)
    root = pool_root.resolve()
    manifest = read_json_model(TrialDevFrozenPool, root / "manifest.json")
    if ledger.pool_manifest_sha256 != sha256_file(root / "manifest.json"):
        raise ValueError("information look cannot change its frozen population")
    accrued = trialdev_committed_accrual(ledger, study_id, programme_day=look.deadline_day)
    enrollment = pd.DataFrame(
        [patient.model_dump() for patient in accrued.patients],
        columns=["participant_id", "participant_label", "arm_id", "enrollment_day"],
    )
    if enrollment.empty:
        return current_cutoff(look.deadline_day)
    inventory = {item.relative_path for item in manifest.artifacts}
    tables = []
    recurrent_flags = []
    for arm in design.arms:
        patients = enrollment.loc[enrollment.arm_id.eq(arm.arm_id), "participant_id"].tolist()
        recurrent = f"{arm.partition.directory}/recurrent_observed.parquet" in inventory
        recurrent_flags.append(recurrent)
        filename = "recurrent_observed.parquet" if recurrent else "observations.parquet"
        tables.append(
            read_pool_table(root, manifest, f"{arm.partition.directory}/{filename}", participant_ids=patients)
        )
    if len(set(recurrent_flags)) != 1:
        raise ValueError("study arms must share the same primary outcome type")
    records = pd.concat(tables, ignore_index=True)
    outcomes = records.outcome_id.unique().tolist()
    if len(outcomes) != 1:
        raise ValueError("information look requires one protocol primary outcome")
    resolver = trialdev_recurrent_information_day if recurrent_flags[0] else trialdev_measurement_information_day
    day = resolver(
        records,
        enrollment,
        outcome_id=outcomes[0],
        participant_count=look.completed_participants,
        follow_up_days=design.follow_up_days,
        deadline_day=look.deadline_day,
    )
    return current_cutoff(look.deadline_day if day is None else day)


def close_trialdev_study_arm(budget_path: Path, study_id: str, arm_id: str) -> TrialDevBudgetLedger:
    """Persist an active-arm enrollment stop using the shared programme transaction.

    Parameters
    ----------
    budget_path
        Private ledger containing the study's accrued readout.
    study_id
        Study whose active arm stops recruiting.
    arm_id
        Active arm to drop while preserving enrolled participants.

    Returns
    -------
    TrialDevBudgetLedger
        Atomically persisted closure after reconciling actual enrollment and obligations.

    Raises
    ------
    ValueError
        If arm closure is invalid or another process owns the programme.

    Examples
    --------
    Identical retries preserve the original closure and cannot reopen enrollment.
    """
    with exclusive_run_directory(budget_path.parent):
        ledger = read_json_model(TrialDevBudgetLedger, budget_path)
        updated = close_trialdev_arm(advance_trialdev_clock(ledger, ledger.current_day), study_id, arm_id)
        write_json_model(budget_path, updated)
        return updated


def reduce_trialdev_study_control(budget_path: Path, study_id: str, participant_count: int) -> TrialDevBudgetLedger:
    """Persist a reduced future control cap after dropping an active arm.

    Parameters
    ----------
    budget_path
        Private programme ledger.
    study_id
        Active study with a dropped arm.
    participant_count
        New total control cap, including retained enrolled controls.

    Returns
    -------
    TrialDevBudgetLedger
        Atomically persisted reservation change after reconciling current resources.

    Raises
    ------
    ValueError
        If reduction is invalid or another process owns the programme.

    Examples
    --------
    Repeating the same cap does not select a fresh set of controls.
    """
    with exclusive_run_directory(budget_path.parent):
        ledger = read_json_model(TrialDevBudgetLedger, budget_path)
        updated = reduce_trialdev_control(
            advance_trialdev_clock(ledger, ledger.current_day), study_id, participant_count
        )
        write_json_model(budget_path, updated)
        return updated


def advance_trialdev_programme_clock(budget_path: Path, programme_day: float) -> TrialDevBudgetLedger:
    """Persist all study accrual and costs at a new clock without an information look.

    Parameters
    ----------
    budget_path
        Existing private programme ledger.
    programme_day
        New resource-accounting clock.

    Returns
    -------
    TrialDevBudgetLedger
        Atomically persisted state after all ongoing studies are reconciled.

    Raises
    ------
    ValueError
        If time regresses, obligations exceed budget or another process owns the run.

    Examples
    --------
    Refresh resources before valuing a terminal decision; this operation releases no data.
    """
    with exclusive_run_directory(budget_path.parent):
        updated = advance_trialdev_clock(read_json_model(TrialDevBudgetLedger, budget_path), programme_day)
        write_json_model(budget_path, updated)
        return updated


def close_trialdev_study_reservation(budget_path: Path, study_id: str) -> TrialDevBudgetLedger:
    """Persist recruitment closure without erasing accrued patients or liabilities.

    Parameters
    ----------
    budget_path
        Existing private programme ledger.
    study_id
        Previously reserved study whose future recruitment stops.

    Returns
    -------
    TrialDevBudgetLedger
        Persisted state; repeated closure preserves the same reservation history.

    Raises
    ------
    ValueError
        If the study is unknown or another process owns the programme transaction.

    Examples
    --------
    Close recruitment before allocating unused capacity to a later study.
    """
    with exclusive_run_directory(budget_path.parent):
        ledger = read_json_model(TrialDevBudgetLedger, budget_path)
        updated = close_trialdev_reservation(advance_trialdev_clock(ledger, ledger.current_day), study_id)
        write_json_model(budget_path, updated)
        return updated


def write_trialdev_arm_readout(
    pool_root: Path,
    partition: TrialDevPoolPartition,
    enrollment: pd.DataFrame,
    output_dir: Path,
    *,
    readout_day: float,
    follow_up_days: int,
    assessment_days: tuple[int, ...],
    surveillance_days: tuple[int, ...],
) -> None:
    """Publish an assigned arm's available records without native generation.

    Parameters
    ----------
    pool_root
        Evaluator-only frozen pool. This directory must remain outside agent access.
    partition
        Committed treatment partition from the pool inventory.
    enrollment
        Committed participant_id, enrollment_day and unique public participant_label.
    output_dir
        New participant output directory; existing output is never overwritten.
    readout_day
        Programme readout day.
    follow_up_days
        Committed maximum participant follow-up.
    assessment_days
        Selected primary measurement dates from the supported pool grid.
    surveillance_days
        Selected safety surveillance dates, ending at committed follow-up.

    Returns
    -------
    None
        Publishes whitelisted Parquet records atomically after successful assembly.

    Raises
    ------
    ValueError
        If treatment, identities, schedules or artifact integrity are invalid.
    FileExistsError
        If the output directory already exists.

    Examples
    --------
    Call from the evaluator's committed study transaction, never from an agent process.
    """
    root = pool_root.resolve()
    if output_dir.resolve().is_relative_to(root):
        raise ValueError("participant output cannot modify immutable pool storage")
    manifest = read_json_model(TrialDevFrozenPool, root / "manifest.json")
    if partition not in manifest.partitions:
        raise ValueError("assigned treatment is not in this frozen pool")
    if not set(assessment_days).issubset({0, *manifest.collection_days}):
        raise ValueError("assessment dates are outside the frozen collection grid")
    if not set(surveillance_days).issubset({0, *manifest.collection_days}):
        raise ValueError("surveillance dates are outside the frozen contact grid")
    labels = enrollment.loc[:, ["participant_id", "participant_label"]]
    if (
        labels.isna().any().any()
        or labels.participant_id.duplicated().any()
        or labels.participant_label.duplicated().any()
    ):
        raise ValueError("readout requires complete unique private and public participant identities")
    if not labels.participant_label.map(lambda value: isinstance(value, str) and bool(value)).all():
        raise ValueError("public participant labels must be nonempty strings")
    public_ids = labels.set_index("participant_id").participant_label
    inventory = {item.relative_path: item for item in manifest.artifacts}

    def read_table(relative_path: str) -> pd.DataFrame:
        return read_pool_table(root, manifest, relative_path, participant_ids=labels.participant_id.tolist())

    baseline = read_table("baseline.parquet")
    if set(baseline.participant_id) != set(labels.participant_id) or baseline.participant_id.duplicated().any():
        raise ValueError("committed enrollment does not identify unique frozen patients")
    administrative = ["consent_time"] if "consent_time" in baseline else []
    baseline = baseline.loc[
        :, ["participant_id", *manifest.baseline_fields, "site_id", "country_code", *administrative]
    ]
    baseline = baseline.merge(enrollment.loc[:, ["participant_id", "enrollment_day"]], validate="one_to_one")
    contacts = trialdev_contact_records(
        read_table("contacts.parquet"),
        enrollment,
        readout_day=readout_day,
        follow_up_days=follow_up_days,
        surveillance_days=surveillance_days,
    )
    safety = trialdev_safety_records(
        read_table(f"{partition.directory}/safety_events.parquet"),
        enrollment,
        readout_day=readout_day,
        follow_up_days=follow_up_days,
        surveillance_days=surveillance_days,
    )
    exposure = trialdev_exposure_records(
        read_table(f"{partition.directory}/exposure.parquet"),
        enrollment,
        readout_day=readout_day,
        follow_up_days=follow_up_days,
    )
    exposure["received_arm"] = exposure.received_arm.replace(
        {"active": partition.asset_id, "control": manifest.partitions[0].asset_id}
    )
    with staged_directory(output_dir) as staged:

        def write_table(name: str, data: pd.DataFrame) -> None:
            public = data.copy()
            public["participant_id"] = public.participant_id.map(public_ids)
            if public.participant_id.isna().any():
                raise ValueError("readout contains a patient outside committed enrollment")
            public.to_parquet(staged / name, index=False)

        write_table("baseline.parquet", baseline)
        write_table("contacts.parquet", contacts)
        write_table("exposure.parquet", exposure)
        write_table("safety_events.parquet", safety)
        completion_path = f"{partition.directory}/completion_records.parquet"
        if completion_path in inventory:
            candidates = read_table(completion_path)
            completion = available_clinical_records(
                candidates.loc[candidates.occurrence_time.eq(follow_up_days)],
                enrollment,
                readout_day=readout_day,
                follow_up_days=follow_up_days,
            )
            clocks = ("occurrence_time", "observation_time", "documentation_time", "availability_time")
            for clock in clocks:
                completion[clock] += completion.enrollment_day
            write_table("completion_records.parquet", completion[["participant_id", *clocks, "status"]])
        intercurrent_path = f"{partition.directory}/intercurrent_events.parquet"
        if intercurrent_path in inventory:
            intercurrent = trialdev_intercurrent_records(
                read_table(intercurrent_path),
                enrollment,
                readout_day=readout_day,
                follow_up_days=follow_up_days,
            )
            write_table("intercurrent_events.parquet", intercurrent)
        recurrent_path = f"{partition.directory}/recurrent_observed.parquet"
        if recurrent_path in inventory:
            recurrent = trialdev_recurrent_records(
                read_table(recurrent_path), enrollment, readout_day=readout_day, follow_up_days=follow_up_days
            )
            # One time origin per readout: every released time is a programme day.
            recurrent = recurrent.merge(
                enrollment.loc[:, ["participant_id", "enrollment_day"]], on="participant_id", validate="many_to_one"
            )
            recurrent["occurrence_time"] = recurrent.enrollment_day + recurrent.event_time
            write_table(
                "recurrent_events.parquet", recurrent[["participant_id", "outcome_id", "occurrence_time", "event"]]
            )
        else:
            measurements = trialdev_measurement_records(
                read_table(f"{partition.directory}/observations.parquet"),
                enrollment,
                readout_day=readout_day,
                follow_up_days=follow_up_days,
                assessment_days=assessment_days,
                monotone_dropout=manifest.monotone_dropout,
                complete_measurements=(
                    read_table(f"{partition.directory}/complete_measurements.parquet")
                    if manifest.monotone_dropout
                    else None
                ),
            )
            write_table("measurements.parquet", measurements)


class TrialDevReadoutArm(BaseModel):
    """Public treatment identity accompanying an arm's raw data files."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    arm_id: str
    asset_id: str
    dose: TrialDevDose | None


class TrialDevReadoutTarget(BaseModel):
    """One active-minus-control contrast for all programme-eligible recruitment strata."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    target_id: str
    arm_id: str
    endpoint: Literal["clinical_outcome", "safety_risk"]
    unit: str
    contrast: Literal["active_minus_control"] = "active_minus_control"
    population: Literal["programme_eligible_population"] = "programme_eligible_population"
    confidence_level: float = 0.95


class TrialDevReadoutManifest(BaseModel):
    """Public readout dates and treatment identities, without private pool paths."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    calendar_origin: date | None = Field(default=None, description="Authored date corresponding to programme day zero")
    epochs: tuple[tuple[str, float, float], ...] = ()
    study_id: str = Field(min_length=1)
    outcome_object: str = Field(min_length=1)
    readout_day: float = Field(ge=0)
    follow_up_days: int = Field(gt=0)
    assessment_days: tuple[int, ...]
    surveillance_days: tuple[int, ...]
    arms: tuple[TrialDevReadoutArm, ...]
    targets: tuple[TrialDevReadoutTarget, ...]
    tariff: TrialDevStudyTariff
    sites: int
    screened: int
    cost: TrialDevStudyCost

    @model_validator(mode="after")
    def validate_epochs(self) -> "TrialDevReadoutManifest":
        """Keep planned epochs on the complete authored participant follow-up clock."""
        validate_clinical_epochs(self.epochs, horizon=self.follow_up_days)
        return self


class TrialDevReadoutCommitment(BaseModel):
    """Immutable evaluator-only inputs for a replayable study readout."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    study_id: str = Field(min_length=1)
    pool_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    readout: TrialDevReadoutManifest
    assignments: tuple[TrialDevAssignedArm, ...]
    patients: tuple[TrialDevCommittedPatient, ...]


def _commit_readout(path: Path, commitment: TrialDevReadoutCommitment) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        write_json_model(temporary, commitment)
        try:
            # Publish complete bytes exclusively; concurrent retries cannot overwrite.
            os.link(temporary, path)
            _sync_directory(path.parent)
        except FileExistsError:
            existing = read_json_model(TrialDevReadoutCommitment, path)
            if existing != commitment:
                raise ValueError("readout identity is already committed to different inputs") from None
    finally:
        temporary.unlink(missing_ok=True)


def _supported_readout_arms(
    directory: Path, arms: tuple[TrialDevAssignedArm, ...], follow_up_days: int
) -> tuple[TrialDevArmId, ...]:
    """Identify endpoint support from released records, never private outcomes."""
    supported = []
    for arm in arms:
        arm_path = directory / "arms" / arm.arm_id
        baseline = pd.read_parquet(arm_path / "baseline.parquet").set_index("participant_id")
        recurrent_path = arm_path / "recurrent_events.parquet"
        if recurrent_path.is_file():
            rows = pd.read_parquet(recurrent_path)
            follow_up_end = rows.participant_id.map(baseline.enrollment_day) + follow_up_days
            complete = rows.event.eq(0) & rows.occurrence_time.ge(follow_up_end)
        else:
            rows = pd.read_parquet(arm_path / "measurements.parquet")
            endpoint_time = rows.participant_id.map(baseline.enrollment_day) + follow_up_days
            complete = rows.observed & rows.occurrence_time.eq(endpoint_time)
        # Two observations establish elementary within-arm variation support,
        # not statistical significance, acceptable precision or clinical success.
        if rows.loc[complete, "participant_id"].nunique() >= MINIMUM_PHASE_PARTICIPANTS_PER_ARM:
            supported.append(arm.arm_id)
    return tuple(supported)


def write_trialdev_study_readout(
    pool_root: Path,
    arms: tuple[TrialDevAssignedArm, ...],
    enrollment: pd.DataFrame,
    output_dir: Path,
    *,
    commitment_path: Path,
    study_id: str,
    budget_path: Path,
    tariff: TrialDevStudyTariff,
    sites: int,
    screened: int,
    readout_day: float,
    follow_up_days: int,
    assessment_days: tuple[int, ...],
    surveillance_days: tuple[int, ...],
) -> TrialDevBudgetLedger:
    """Atomically assemble concurrent arms with one shared control and unique patients.

    Parameters
    ----------
    pool_root
        Evaluator-only frozen pool.
    arms
        Committed public arm identities and private treatment partitions.
    enrollment
        Committed arm_id, participant_id, participant_label and enrollment_day rows.
    output_dir
        New participant study-readout directory.
    commitment_path
        Evaluator-only immutable commitment file, outside participant output.
    study_id
        Stable programme study identity for cumulative resource accounting.
    budget_path
        Existing evaluator-only JSON budget ledger; its run directory is exclusively locked.
    tariff
        Public unit prices committed before study execution.
    sites
        Activated sites for this study.
    screened
        Actual candidates screened by this readout.
    readout_day
        Programme readout day.
    follow_up_days
        Committed participant follow-up duration.
    assessment_days
        Committed primary collection schedule.
    surveillance_days
        Committed safety contact schedule.

    Returns
    -------
    TrialDevBudgetLedger
        Persisted private accounting state, committed before participant publication.

    Raises
    ------
    ValueError
        If arms, control identity or enrollment are inconsistent.
    FileExistsError
        If the destination exists.

    Examples
    --------
    Use from the committed study transaction after resource and phase validation.
    """
    if budget_path.resolve().is_relative_to(pool_root.resolve()) or budget_path.resolve().is_relative_to(
        output_dir.resolve()
    ):
        raise ValueError("budget ledger must remain outside pool and participant storage")
    with exclusive_run_directory(budget_path.parent):
        budget_ledger = read_json_model(TrialDevBudgetLedger, budget_path)
        recruitment = next((item for item in budget_ledger.recruitment_commitments if item.study_id == study_id), None)
        if recruitment is not None and (
            recruitment.tariff != tariff
            or recruitment.sites != sites
            or tuple(sorted(recruitment.arms, key=lambda item: item.arm_id))
            != tuple(sorted(arms, key=lambda item: item.arm_id))
            or recruitment.follow_up_days != follow_up_days
            or recruitment.assessment_days != assessment_days
            or recruitment.surveillance_days != surveillance_days
        ):
            raise ValueError("readout cannot change the committed treatment, schedule or resource design")
        identities = [arm.arm_id for arm in arms]
        controls = [arm for arm in arms if arm.partition.dose is None]
        active_treatments = [
            (arm.partition.asset_id, arm.partition.dose) for arm in arms if arm.partition.dose is not None
        ]
        if not 2 <= len(arms) <= 7 or len(set(identities)) != len(identities):
            raise ValueError("study requires two through seven uniquely identified arms")
        if len(controls) != 1 or controls[0].arm_id != "control":
            raise ValueError("study requires exactly one shared control arm")
        if len(set(active_treatments)) != len(active_treatments):
            raise ValueError("study cannot duplicate an asset/regimen pair")
        if not set(enrollment.arm_id).issubset(identities):
            raise ValueError("enrollment contains an arm outside the committed study")
        if enrollment.participant_id.duplicated().any() or enrollment.participant_label.duplicated().any():
            raise ValueError("patients and public labels must be unique across all study arms")
        if commitment_path.resolve().is_relative_to(pool_root.resolve()) or output_dir.resolve().is_relative_to(
            pool_root.resolve()
        ):
            raise ValueError("readout publication cannot modify immutable pool storage")
        if commitment_path.resolve().is_relative_to(output_dir.resolve()):
            raise ValueError("private readout commitment cannot be placed inside participant output")
        cost = trialdev_study_cost(
            tariff,
            sites=sites,
            screened=screened,
            enrollment_days=tuple(float(day) for day in enrollment.enrollment_day),
            programme_day=readout_day,
            follow_up_days=follow_up_days,
            assessment_days=assessment_days,
            surveillance_days=surveillance_days,
        )
        pool = read_json_model(TrialDevFrozenPool, pool_root / "manifest.json")
        targets = []
        for arm in sorted(arms, key=lambda item: item.arm_id):
            if arm.partition.dose is None:
                continue
            contrast = next(
                item
                for item in pool.targets
                if item.asset_id == arm.partition.asset_id and item.dose == arm.partition.dose
            )
            endpoints: tuple[tuple[Literal["clinical_outcome", "safety_risk"], str], ...] = (
                ("clinical_outcome", contrast.outcome_unit),
                ("safety_risk", "risk difference"),
            )
            for endpoint, unit in endpoints:
                targets.append(
                    TrialDevReadoutTarget(
                        target_id=f"{arm.partition.asset_id}:{arm.partition.dose.value}:vs_control:{endpoint}",
                        arm_id=arm.arm_id,
                        endpoint=endpoint,
                        unit=unit,
                    )
                )
        public_readout = TrialDevReadoutManifest(
            outcome_object=pool.outcome_object,
            calendar_origin=pool.calendar_origin,
            epochs=tuple(
                (name, start, min(end, float(follow_up_days)))
                for name, start, end in pool.epochs
                if start < follow_up_days
            ),
            targets=tuple(targets),
            study_id=study_id,
            tariff=tariff,
            sites=sites,
            screened=screened,
            cost=cost,
            readout_day=readout_day,
            follow_up_days=follow_up_days,
            assessment_days=assessment_days,
            surveillance_days=surveillance_days,
            # Control first in agent-facing listings (study.json and the readout letter).
            arms=tuple(
                TrialDevReadoutArm(arm_id=arm.arm_id, asset_id=arm.partition.asset_id, dose=arm.partition.dose)
                for arm in sorted(arms, key=lambda item: (item.arm_id != "control", item.arm_id))
            ),
        )
        patients = tuple(
            TrialDevCommittedPatient.model_validate(row)
            for row in enrollment.loc[:, ["participant_id", "participant_label", "arm_id", "enrollment_day"]]
            .sort_values("participant_label")
            .to_dict("records")
        )
        commitment = TrialDevReadoutCommitment(
            study_id=study_id,
            pool_manifest_sha256=sha256_file(pool_root / "manifest.json"),
            readout=public_readout,
            assignments=tuple(sorted(arms, key=lambda item: item.arm_id)),
            patients=patients,
        )
        readout_id = canonical_payload_sha256(commitment.model_dump(mode="json"))
        if not any(receipt.readout_id == readout_id for receipt in budget_ledger.receipts):
            if recruitment is not None:
                if readout_day != trialdev_next_readout_day(pool_root, budget_ledger, study_id):
                    raise ValueError("readout must occur at the next committed look")
                accrued = trialdev_committed_accrual(budget_ledger, study_id, programme_day=readout_day)
                if screened != accrued.screened or patients != tuple(
                    sorted(accrued.patients, key=lambda item: item.participant_label)
                ):
                    raise ValueError("readout enrollment and screening must match the committed recruitment clock")
            budget_ledger = reconcile_trialdev_enrollment(
                budget_ledger,
                pool_manifest_sha256=commitment.pool_manifest_sha256,
                enrollment=TrialDevStudyEnrollment(
                    study_id=study_id,
                    design_sha256=canonical_payload_sha256(
                        recruitment.model_dump(mode="json")
                        if recruitment is not None
                        else public_readout.model_dump(mode="json", exclude={"readout_day", "screened", "cost"})
                    ),
                    patients=patients,
                    screened=screened,
                ),
            )
            if recruitment is not None:
                budget_ledger = advance_trialdev_clock(budget_ledger, readout_day)
        with staged_directory(output_dir) as staged:
            for arm in sorted(arms, key=lambda item: item.arm_id):
                write_trialdev_arm_readout(
                    pool_root,
                    arm.partition,
                    enrollment.loc[enrollment.arm_id.eq(arm.arm_id)],
                    staged / "arms" / arm.arm_id,
                    readout_day=readout_day,
                    follow_up_days=follow_up_days,
                    assessment_days=assessment_days,
                    surveillance_days=surveillance_days,
                )
            write_readout_dictionary(
                staged / "data_dictionary.csv",
                intercurrent=any(staged.glob("arms/*/intercurrent_events.parquet")),
                completion=any(staged.glob("arms/*/completion_records.parquet")),
                consent="consent_time" in pd.read_parquet(staged / "arms" / arms[0].arm_id / "baseline.parquet"),
                baseline_fields=pool.baseline_fields,
                outcome_unit=targets[0].unit,
                recurrent=(staged / "arms" / arms[0].arm_id / "recurrent_events.parquet").is_file(),
            )
            updated_budget = reconcile_trialdev_cost(
                budget_ledger,
                TrialDevCostReceipt(
                    study_id=study_id,
                    readout_id=readout_id,
                    cost=cost,
                    programme_day=readout_day,
                    supported_arms=(
                        _supported_readout_arms(staged, arms, follow_up_days) if recruitment is not None else ()
                    ),
                ),
            )
            _commit_readout(commitment_path, commitment)
            write_json_model(staged / "study.json", public_readout)
            write_json_model(budget_path, updated_budget)
        return updated_budget


def write_trialdev_committed_readout(
    pool_root: Path,
    budget_path: Path,
    study_id: str,
    output_dir: Path,
    *,
    commitment_path: Path,
) -> TrialDevBudgetLedger:
    """Publish the next authored-study readout entirely from persisted programme state.

    Parameters
    ----------
    pool_root
        Private frozen patient and treatment records.
    budget_path
        Private programme ledger containing the admitted study and decisions.
    study_id
        Committed study to observe.
    output_dir
        Participant-facing raw-data directory. Existing committed output is verified on retry.
    commitment_path
        Private durable identity for this readout. Reusing it recovers that exact
        readout, including after later programme activity.

    Returns
    -------
    TrialDevBudgetLedger
        Persisted enrollment, expenditure and readout receipt after publication.

    Raises
    ------
    ValueError
        If study identity is absent, recovery inputs conflict, or concurrent state
        changes invalidate the proposed readout before publication.

    Examples
    --------
    The runner supplies identities and destinations, never patient assignments,
    screening counts, tariffs or observation cutoffs.
    """
    with exclusive_run_directory(budget_path.parent):
        ledger = read_json_model(TrialDevBudgetLedger, budget_path)
        design = next((item for item in ledger.recruitment_commitments if item.study_id == study_id), None)
        if design is None:
            raise TrialDevActionRejectedError(f"unknown study {study_id!r}")
        if commitment_path.exists():
            previous = read_json_model(TrialDevReadoutCommitment, commitment_path)
            if previous.study_id != study_id or previous.pool_manifest_sha256 != ledger.pool_manifest_sha256:
                raise ValueError("readout recovery identity does not match the committed study")
            day = previous.readout.readout_day
            patients = previous.patients
            screened = previous.readout.screened
        else:
            day = trialdev_next_readout_day(pool_root, ledger, study_id)
            accrued = trialdev_committed_accrual(ledger, study_id, programme_day=day)
            patients = accrued.patients
            screened = accrued.screened
        enrollment = pd.DataFrame(
            [patient.model_dump() for patient in patients],
            columns=["participant_id", "participant_label", "arm_id", "enrollment_day"],
        )
        recovering_output = output_dir.exists() or output_dir.is_symlink()
        if recovering_output:
            if (
                output_dir.is_symlink()
                or not output_dir.is_dir()
                or any(path.is_symlink() for path in output_dir.rglob("*"))
            ):
                raise ValueError("readout recovery requires a regular evidence directory without symbolic links")
            if not commitment_path.is_file():
                raise ValueError("existing readout has no committed recovery identity")
            previous = read_json_model(TrialDevReadoutCommitment, commitment_path)
            identity = canonical_payload_sha256(previous.model_dump(mode="json"))
            if not any(row.readout_id == identity and row.study_id == study_id for row in ledger.receipts):
                raise ValueError("existing readout has no committed resource receipt")
    if recovering_output:
        # Reproject only the recorded patients/cutoff, without native generation or
        # a new observation look. Existing participant files are never overwritten.
        with TemporaryDirectory(prefix="trialdev-readout-recovery-") as temporary:
            recovered = Path(temporary) / "evidence"
            current = write_trialdev_committed_readout(
                pool_root, budget_path, study_id, recovered, commitment_path=commitment_path
            )
            if sha256_dir_files(output_dir) != sha256_dir_files(recovered):
                raise ValueError("existing readout differs from its committed participant evidence")
            return current
    # The publishing transaction reacquires ownership and revalidates all derived
    # inputs against current state. A competing change cannot silently alter them.
    return write_trialdev_study_readout(
        pool_root,
        design.arms,
        enrollment,
        output_dir,
        commitment_path=commitment_path,
        study_id=study_id,
        budget_path=budget_path,
        tariff=design.tariff,
        sites=design.sites,
        screened=screened,
        readout_day=day,
        follow_up_days=design.follow_up_days,
        assessment_days=design.assessment_days,
        surveillance_days=design.surveillance_days,
    )


def finish_trialdev_run(budget_path: Path, decision: TrialDevTerminalDecision) -> TrialDevBudgetLedger:
    """Persist one explicit terminal decision and settle remaining study liabilities.

    Parameters
    ----------
    budget_path
        Private evaluator-owned programme ledger.
    decision
        Explicit terminal choice and current programme day.

    Returns
    -------
    TrialDevBudgetLedger
        Atomically persisted frozen programme state.

    Raises
    ------
    ValueError
        If choice, time, evidence or resource limits are invalid.

    Examples
    --------
    Repeating a terminal decision cannot charge settlement twice or change the chosen asset.
    """
    with exclusive_run_directory(budget_path.parent):
        updated = finish_trialdev_programme(read_json_model(TrialDevBudgetLedger, budget_path), decision)
        write_json_model(budget_path, updated)
        return updated
