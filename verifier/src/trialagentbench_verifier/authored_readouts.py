"""Independent reconstruction of frozen-pool participant readouts."""

import math
from bisect import bisect_left
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from trialagentbench_verifier.io import sha256_file


class _Projection(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, allow_inf_nan=False)


class _Artifact(_Projection):
    relative_path: str
    sha256: str
    byte_length: int = Field(ge=0)


class _Partition(_Projection):
    asset_id: str
    dose: str | None
    directory: str = Field(pattern=r"^partitions/p[0-9]{3}$")


class _Pool(_Projection):
    outcome_object: str = Field(min_length=1)
    calendar_origin: date | None = None
    schema_id: Literal["trialdev.frozen-pool/v7", "trialdev.frozen-pool/v8"]
    monotone_dropout: bool = False
    baseline_fields: tuple[str, ...]
    artifacts: tuple[_Artifact, ...]
    partitions: tuple[_Partition, ...]


class _Arm(_Projection):
    arm_id: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$")
    asset_id: str
    dose: str | None


class _Assignment(_Projection):
    arm_id: str
    partition: _Partition


class _Readout(_Projection):
    outcome_object: str = Field(min_length=1)
    calendar_origin: date | None = None
    study_id: str
    readout_day: float = Field(ge=0)
    follow_up_days: int = Field(gt=0)
    arms: tuple[_Arm, ...]
    assessment_days: tuple[int, ...]
    surveillance_days: tuple[int, ...]


class _Patient(_Projection):
    participant_id: int
    participant_label: str = Field(min_length=1)
    arm_id: str
    enrollment_day: float = Field(ge=0)


class _Commitment(_Projection):
    study_id: str
    pool_manifest_sha256: str
    readout: _Readout
    assignments: tuple[_Assignment, ...]
    patients: tuple[_Patient, ...]


def verify_trialdev_readout_patients(*, pool_root: Path, commitment_path: Path, readout_root: Path) -> None:
    """Check all baseline rows against a bound pool and committed patient mapping.

    Parameters
    ----------
    pool_root
        Authenticated evaluator pool, inaccessible to the evaluated model.
    commitment_path
        Retained evaluator-owned patient/arm/cutoff commitment.
    readout_root
        Published readout containing study.json and arms/*/baseline.parquet.

    Raises
    ------
    ValueError
        If patients, treatments, baseline values, columns or enrollment dates differ.

    Notes
    -----
    This checks the patient mapping and baseline projection. It does not certify
    how patients were selected, or the subsequent outcome and event tables.
    """
    commitment = _Commitment.model_validate_json(commitment_path.read_bytes())
    manifest_path = pool_root / "manifest.json"
    if sha256_file(manifest_path) != commitment.pool_manifest_sha256:
        raise ValueError("Readout commitment identifies a different frozen pool")
    pool = _Pool.model_validate_json(manifest_path.read_bytes())
    if commitment.readout.calendar_origin != pool.calendar_origin:
        raise ValueError("Readout calendar origin differs from the frozen programme")
    if commitment.readout.outcome_object != pool.outcome_object:
        raise ValueError("Readout clinical object differs from the frozen programme")
    public = _Readout.model_validate_json((readout_root / "study.json").read_bytes())
    if public != commitment.readout or commitment.study_id != public.study_id:
        raise ValueError("Readout identity or cutoff differs from its commitment")
    assignments = {row.arm_id: row.partition for row in commitment.assignments}
    arms = {row.arm_id: row for row in public.arms}
    if (
        len(assignments) != len(commitment.assignments)
        or len(arms) != len(public.arms)
        or assignments.keys() != arms.keys()
    ):
        raise ValueError("Readout treatment-arm census differs from its commitment")
    for arm_id, partition in assignments.items():
        if partition not in pool.partitions or (partition.asset_id, partition.dose) != (
            arms[arm_id].asset_id,
            arms[arm_id].dose,
        ):
            raise ValueError("Readout treatment differs from its released partition")
    patients = commitment.patients
    if len({row.participant_id for row in patients}) != len(patients) or len(
        {row.participant_label for row in patients}
    ) != len(patients):
        raise ValueError("Readout repeats private patients or public participant labels")
    if any(row.arm_id not in arms or row.enrollment_day > public.readout_day for row in patients):
        raise ValueError("Readout includes an unassigned or future participant")
    inventory = [row for row in pool.artifacts if row.relative_path == "baseline.parquet"]
    baseline_path = pool_root / "baseline.parquet"
    if (
        len(inventory) != 1
        or sha256_file(baseline_path) != inventory[0].sha256
        or baseline_path.stat().st_size != inventory[0].byte_length
    ):
        raise ValueError("Readout baseline differs from the frozen pool inventory")
    baseline = pd.read_parquet(baseline_path.open("rb")).set_index("participant_id")
    if baseline.index.has_duplicates:
        raise ValueError("Frozen baseline repeats participant identities")
    administrative = ["consent_time"] if "consent_time" in baseline else []
    columns = ["participant_id", "enrollment_day", "site_id", "country_code", *pool.baseline_fields, *administrative]
    if len(set(columns)) != len(columns):
        raise ValueError("Readout clinical fields overwrite participant identities")
    for arm_id in arms:
        selected = [row for row in patients if row.arm_id == arm_id]
        identities = [row.participant_id for row in selected]
        if not set(identities) <= set(baseline.index):
            raise ValueError("Readout contains a participant outside the frozen pool")
        expected = baseline.loc[
            identities, [*pool.baseline_fields, "site_id", "country_code", *administrative]
        ].reset_index(drop=True)
        expected["participant_id"] = [row.participant_label for row in selected]
        expected["enrollment_day"] = [row.enrollment_day for row in selected]
        path = readout_root / "arms" / arm_id / "baseline.parquet"
        if path.is_symlink() or not path.resolve().is_relative_to(readout_root.resolve()):
            raise ValueError("Readout baseline crosses its committed directory")
        actual = pd.read_parquet(path.open("rb"))
        if set(actual.columns) != set(columns) or len(actual.columns) != len(columns):
            raise ValueError("Readout baseline has missing or private columns")
        try:
            pd.testing.assert_frame_equal(
                actual[columns].sort_values("participant_id").reset_index(drop=True),
                expected[columns].sort_values("participant_id").reset_index(drop=True),
                check_dtype=False,
                check_exact=True,
            )
        except AssertionError as error:
            raise ValueError("Readout baseline differs from committed patients and clinical records") from error


def _record_projection(
    source: pd.DataFrame,
    patients: tuple[_Patient, ...],
    readout: _Readout,
    kind: str,
    *,
    complete_measurements: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Reconstruct the observable record set from individual calendar limits."""
    days = {row.participant_id: row.enrollment_day for row in patients}
    labels = {row.participant_id: row.participant_label for row in patients}
    data = source.loc[source.participant_id.isin(days)].copy()
    data["enrolled"] = data.participant_id.map(days)
    if kind == "recurrent_events":
        ends = data.loc[data.event.eq(0), ["participant_id", "event_time"]]
        if ends.participant_id.duplicated().any() or set(ends.participant_id) != set(days):
            raise ValueError("Frozen recurrent data lack a unique follow-up end")
        cutoffs = {
            identity: min(float(end), readout.follow_up_days, readout.readout_day - days[identity])
            for identity, end in ends.itertuples(index=False, name=None)
        }
        cutoff = data.participant_id.map(cutoffs)
        data = data.loc[data.event.eq(0) | data.event_time.le(cutoff)].copy()
        data.loc[data.event.eq(0), "event_time"] = data.loc[data.event.eq(0), "participant_id"].map(cutoffs)
        # Released recurrent times are programme days, like every other readout time.
        data["occurrence_time"] = data.enrolled + data.event_time
        columns = ["participant_id", "outcome_id", "occurrence_time", "event"]
    else:
        if kind == "exposure":
            data["availability_time"] = data.occurrence_time
        elif kind == "contacts":
            data = data.loc[data.occurrence_time.isin(readout.surveillance_days)].copy()
        elif kind == "measurements":
            data = data.loc[data.occurrence_time.isin(readout.assessment_days)].copy()
            if complete_measurements is not None:
                attempts = data["_measurement_attempt_succeeded"]
                if not pd.api.types.is_bool_dtype(attempts) or attempts.isna().any():
                    raise ValueError("Frozen measurement attempts must be nonmissing booleans")
                if (attempts & ~data.attended).any():
                    raise ValueError("A successful measurement requires attendance")
                keys = ["participant_id", "outcome_id", "occurrence_time"]
                data = data.merge(
                    complete_measurements[keys + ["complete_recorded_outcome"]],
                    on=keys,
                    how="left",
                    validate="one_to_one",
                    indicator=True,
                ).sort_values(keys)
                if not data["_merge"].eq("both").all():
                    raise ValueError("Frozen potentials do not cover committed measurement attempts")
                # Replay chronological attempts before filtering by report availability.
                # An omitted visit never enters this history; a delayed report does.
                first_failure = (
                    data.occurrence_time.where(~data._measurement_attempt_succeeded)
                    .groupby([data.participant_id, data.outcome_id])
                    .transform("min")
                )
                data["observed"] = first_failure.isna() | data.occurrence_time.lt(first_failure)
                data["observed_outcome"] = data.complete_recorded_outcome.where(data.observed)
                if not data.loc[data.observed, "observed_outcome"].map(math.isfinite).all():
                    raise ValueError("An observed measurement requires a finite recorded potential")
            data.loc[~data.observed, "observed_outcome"] = float("nan")
        elif kind == "completion_records":
            data = data.loc[data.occurrence_time.eq(readout.follow_up_days)].copy()
            if (
                data.participant_id.duplicated().any()
                or not data.status.isin(("completed", "not_completed", "status_unresolved")).all()
            ):
                raise ValueError("Frozen closeout requires unique participants and explicit supported status")
        elif kind == "intercurrent_events":
            if not pd.api.types.is_bool_dtype(data.event_at_contact) or data.event_at_contact.isna().any():
                raise ValueError("Intercurrent incident flags must be nonmissing booleans")
            data = data.loc[data.event_at_contact].copy()
        elif kind == "safety_events":
            grid = (*readout.surveillance_days, float("inf"))

            def contact(day: float) -> float:
                return float("inf") if math.isnan(day) else grid[bisect_left(grid, day)]

            data = data.loc[data.event_at_contact & data.reported.eq(True)].copy()
            scheduled = data.occurrence_time.map(contact)
            spontaneous = data["_spontaneous_observation_time"].fillna(float("inf"))
            if not data.ascertainment.isin(("scheduled", "spontaneous", "urgent")).all():
                raise ValueError("Frozen safety records have an unknown ascertainment process")
            urgent = data.ascertainment.eq("urgent")
            if not spontaneous.loc[urgent].map(math.isfinite).all():
                raise ValueError("Urgent safety records require a finite clinical report time")
            observed_at = pd.concat([scheduled, spontaneous], axis=1).min(axis=1)
            observed_at.loc[urgent] = spontaneous.loc[urgent]
            data["availability_time"] = observed_at + data["_recording_delay_days"]
            resolved_at = data.resolution_time.map(contact) + data["_recording_delay_days"] + data.enrolled
            data["resolution_time"] = data.resolution_time + data.enrolled
            data.loc[
                (resolved_at > readout.readout_day)
                | (data.resolution_time > readout.readout_day)
                | (data.resolution_time > data.enrolled + readout.follow_up_days),
                "resolution_time",
            ] = float("nan")
        else:
            raise ValueError("Unsupported readout record kind")
        data = data.loc[
            data.occurrence_time.le(readout.follow_up_days)
            & (data.availability_time + data.enrolled).le(readout.readout_day)
        ].copy()
        data["occurrence_time"] = data.occurrence_time + data.enrolled
        data["availability_time"] = data.availability_time + data.enrolled
        if kind == "contacts":
            columns = ["participant_id", "occurrence_time", "availability_time"]
        elif kind == "exposure":
            columns = ["participant_id", "occurrence_time", "received_arm"]
        elif kind == "completion_records":
            for clock in ("observation_time", "documentation_time"):
                data[clock] += data.enrolled
            columns = [
                "participant_id",
                "occurrence_time",
                "observation_time",
                "documentation_time",
                "availability_time",
                "status",
            ]
        elif kind == "intercurrent_events":
            columns = ["participant_id", "event_id", "event_kind", "occurrence_time", "availability_time"]
        elif kind == "measurements":
            columns = [
                "participant_id",
                "outcome_id",
                "occurrence_time",
                "availability_time",
                "attended",
                "observed",
                "observed_outcome",
            ]
        else:
            for field in ("seriousness_criteria", "dictionary_term"):
                if field not in data:
                    data[field] = pd.NA
            columns = [
                "participant_id",
                "event_id",
                "occurrence_time",
                "availability_time",
                "medical_code",
                "coding_system",
                "dictionary_term",
                "severity",
                "serious",
                "seriousness_criteria",
                "resolution_time",
            ]
    data["participant_id"] = data.participant_id.map(labels)
    return data[columns].reset_index(drop=True)


def verify_trialdev_readout_records(*, pool_root: Path, commitment_path: Path, readout_root: Path) -> None:
    """Reconstruct all public raw readout tables from authenticated frozen records.

    Parameters
    ----------
    pool_root, commitment_path, readout_root
        Evaluator pool, retained cutoff commitment and published raw readout.

    Raises
    ------
    ValueError
        If patient mapping, record availability, values or public file/column inventories differ.

    Notes
    -----
    This checks materialization at a committed cutoff. Recruitment selection and
    whether the action history lawfully reached that cutoff require separate checks.
    """
    verify_trialdev_readout_patients(pool_root=pool_root, commitment_path=commitment_path, readout_root=readout_root)
    pool = _Pool.model_validate_json((pool_root / "manifest.json").read_bytes())
    commitment = _Commitment.model_validate_json(commitment_path.read_bytes())
    readout = commitment.readout
    for grid in (readout.assessment_days, readout.surveillance_days):
        if not grid or tuple(sorted(set(grid))) != grid or grid[0] < 0 or grid[-1] != readout.follow_up_days:
            raise ValueError("Readout has an invalid observation grid")
    inventory = {row.relative_path: row for row in pool.artifacts}
    if len(inventory) != len(pool.artifacts):
        raise ValueError("Frozen pool repeats file identities")
    controls = [row.asset_id for row in pool.partitions if row.dose is None]
    if len(controls) != 1:
        raise ValueError("Frozen pool lacks a unique control")
    if {path.name for path in (readout_root / "arms").iterdir()} != {arm.arm_id for arm in readout.arms}:
        raise ValueError("Readout contains an undeclared or missing treatment arm")

    def read_source(relative: str, patients: tuple[_Patient, ...]) -> pd.DataFrame:
        entry = inventory.get(relative)
        source_path = pool_root / relative
        if (
            entry is None
            or source_path.is_symlink()
            or sha256_file(source_path) != entry.sha256
            or source_path.stat().st_size != entry.byte_length
        ):
            raise ValueError("Frozen readout source differs from its pool inventory")
        return pd.read_parquet(
            source_path.open("rb"), filters=[("participant_id", "in", [row.participant_id for row in patients])]
        )

    for assignment in commitment.assignments:
        partition = assignment.partition
        patients = tuple(row for row in commitment.patients if row.arm_id == assignment.arm_id)
        recurrent = f"{partition.directory}/recurrent_observed.parquet" in inventory
        files = {
            "contacts": "contacts.parquet",
            "exposure": f"{partition.directory}/exposure.parquet",
            "safety_events": f"{partition.directory}/safety_events.parquet",
            "recurrent_events" if recurrent else "measurements": (
                f"{partition.directory}/recurrent_observed.parquet"
                if recurrent
                else f"{partition.directory}/observations.parquet"
            ),
        }
        for domain in ("intercurrent_events", "completion_records"):
            relative = f"{partition.directory}/{domain}.parquet"
            if relative in inventory:
                files[domain] = relative
        arm_root = readout_root / "arms" / assignment.arm_id
        if {path.name for path in arm_root.iterdir()} != {"baseline.parquet", *(f"{name}.parquet" for name in files)}:
            raise ValueError("Readout contains missing or undeclared raw files")

        for kind, relative in files.items():
            source = read_source(relative, patients)
            complete = (
                read_source(f"{partition.directory}/complete_measurements.parquet", patients)
                if kind == "measurements" and pool.monotone_dropout
                else None
            )
            expected = _record_projection(source, patients, readout, kind, complete_measurements=complete)
            if kind == "exposure":
                expected["received_arm"] = expected.received_arm.replace(
                    {"active": partition.asset_id, "control": controls[0]}
                )
            path = arm_root / f"{kind}.parquet"
            if path.is_symlink() or not path.resolve().is_relative_to(readout_root.resolve()):
                raise ValueError("Readout records cross the committed directory")
            actual = pd.read_parquet(path.open("rb"))
            columns = list(expected.columns)
            if set(actual.columns) != set(columns) or len(actual.columns) != len(columns):
                raise ValueError("Readout records contain missing or private columns")
            try:
                pd.testing.assert_frame_equal(
                    actual[columns].sort_values(columns).reset_index(drop=True),
                    expected.sort_values(columns).reset_index(drop=True),
                    check_dtype=False,
                    check_exact=True,
                )
            except AssertionError as error:
                raise ValueError(f"Readout {kind} differs from the committed observable records") from error
