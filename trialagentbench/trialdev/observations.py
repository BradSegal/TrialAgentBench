"""Time-limited participant observations from private frozen study records."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from clinical_data_tables import available_clinical_records, validate_enrollment_clock


def trialdev_intercurrent_records(
    records: pd.DataFrame, enrollment: pd.DataFrame, *, readout_day: float, follow_up_days: int
) -> pd.DataFrame:
    """Release incident intercurrent events available at the committed readout.

    Parameters
    ----------
    records
        Frozen native event history; private predictors and future event flags
        are excluded from the returned records.
    enrollment
        Committed participant identities and enrollment days.
    readout_day
        Actual programme readout day.
    follow_up_days
        Committed participant follow-up duration.

    Returns
    -------
    pandas.DataFrame
        Participant, event identity/kind and occurrence/availability timestamps
        on the programme calendar, including a typed empty table when needed.

    Raises
    ------
    ValueError
        If incident flags or enrollment/timing are invalid.
    """
    if not pd.api.types.is_bool_dtype(records.event_at_contact) or records.event_at_contact.isna().any():
        raise ValueError("intercurrent incident flags must be nonmissing booleans")
    data = available_clinical_records(
        records.loc[records.event_at_contact],
        enrollment,
        readout_day=readout_day,
        follow_up_days=follow_up_days,
    )
    for column in ("occurrence_time", "availability_time"):
        data[column] = data[column] + data.enrollment_day
    return data.loc[
        :, ["participant_id", "event_id", "event_kind", "occurrence_time", "availability_time"]
    ].reset_index(drop=True)


def trialdev_measurement_records(
    records: pd.DataFrame,
    enrollment: pd.DataFrame,
    *,
    readout_day: float,
    follow_up_days: int,
    assessment_days: tuple[int, ...],
    monotone_dropout: bool = False,
    complete_measurements: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Release only committed, available measurements and observable missingness.

    Parameters
    ----------
    records
        Frozen native observation table for the assigned treatment.
    enrollment
        Committed participant identities and programme enrollment days.
    readout_day
        Actual programme readout day.
    follow_up_days
        Committed participant follow-up duration.
    assessment_days
        Prospectively selected collection dates, including zero if requested.
    monotone_dropout
        Replay frozen measurement attempts over the committed visits. An
        uncommitted visit cannot trigger follow-up loss in the actual study.
    complete_measurements
        Private recorded measurement potentials, required for dropout replay.
        Only successful attempts retained by the committed schedule are released.

    Returns
    -------
    pandas.DataFrame
        Whitelisted raw measurements, with timestamps on the programme calendar.
        Outcome units and identifiers are defined by the clinical protocol.

    Raises
    ------
    ValueError
        If assessment dates are invalid or an observed value violates missingness.

    Examples
    --------
    ``data = trialdev_measurement_records(records, enrollment, readout_day=84,
    follow_up_days=84, assessment_days=(0, 28, 84))``.
    """
    if not assessment_days or tuple(sorted(set(assessment_days))) != assessment_days:
        raise ValueError("assessment dates must be nonempty, unique and increasing")
    if assessment_days[0] < 0 or assessment_days[-1] > follow_up_days:
        raise ValueError("assessment dates must lie within committed follow-up")
    data = available_clinical_records(records, enrollment, readout_day=readout_day, follow_up_days=follow_up_days)
    data = data.loc[data.occurrence_time.isin(assessment_days)].copy()
    if monotone_dropout:
        selected = records.loc[records.occurrence_time.isin(assessment_days)].copy()
        if complete_measurements is None or "_measurement_attempt_succeeded" not in selected:
            raise ValueError("dropout replay requires frozen measurement attempts and recorded potentials")
        if not pd.api.types.is_bool_dtype(selected._measurement_attempt_succeeded):
            raise ValueError("frozen measurement attempts must be boolean")
        if (selected._measurement_attempt_succeeded & ~selected.attended).any():
            raise ValueError("a successful measurement attempt requires an attended contact")
        keys = ["participant_id", "outcome_id", "occurrence_time"]
        selected = selected.merge(
            complete_measurements[keys + ["complete_recorded_outcome"]],
            on=keys,
            how="left",
            validate="one_to_one",
            indicator="_potential_merge",
        ).sort_values(keys, kind="stable")
        if not selected._potential_merge.eq("both").all():
            raise ValueError("recorded potentials do not cover every committed measurement attempt")
        selected["observed"] = (
            selected.groupby(["participant_id", "outcome_id"])["_measurement_attempt_succeeded"].cumprod().astype(bool)
        )
        selected["observed_outcome"] = selected.complete_recorded_outcome.where(selected.observed)
        data = available_clinical_records(selected, enrollment, readout_day=readout_day, follow_up_days=follow_up_days)
    elif complete_measurements is not None:
        raise ValueError("recorded potentials are only accepted for declared dropout replay")
    if not pd.api.types.is_bool_dtype(data.observed) or not pd.api.types.is_bool_dtype(data.attended):
        raise ValueError("measurement observation and attendance indicators must be boolean")
    if (data.observed & ~data.attended).any():
        raise ValueError("an observed measurement requires an attended contact")
    if not np.isfinite(data.loc[data.observed, "observed_outcome"].to_numpy(dtype=float)).all():
        raise ValueError("an observed measurement must contain a finite value")
    # Missing rows cannot accidentally carry a private underlying measurement.
    data.loc[~data.observed, "observed_outcome"] = np.nan
    for column in ("occurrence_time", "availability_time"):
        data[column] = data[column] + data.enrollment_day
    return data.loc[
        :,
        [
            "participant_id",
            "outcome_id",
            "occurrence_time",
            "availability_time",
            "attended",
            "observed",
            "observed_outcome",
        ],
    ].reset_index(drop=True)


def _surveillance_grid(days: tuple[int, ...], follow_up_days: int) -> npt.NDArray[np.float64]:
    if (
        not days
        or any(isinstance(day, bool) or not isinstance(day, int) for day in days)
        or tuple(sorted(set(days))) != days
        or days[0] < 0
        or days[-1] != follow_up_days
    ):
        raise ValueError("surveillance dates must be unique increasing integers ending at committed follow-up")
    return np.asarray((*days, np.inf), dtype=float)


def _scheduled_safety_records(
    records: pd.DataFrame, *, surveillance_days: tuple[int, ...], follow_up_days: int
) -> pd.DataFrame:
    grid = _surveillance_grid(surveillance_days, follow_up_days)
    required = {"_spontaneous_observation_time", "_recording_delay_days", "ascertainment"}
    if not required.issubset(records.columns):
        raise ValueError("changed surveillance requires frozen native observation-delay potentials")
    data = records.loc[records.event_at_contact & records.reported.eq(True)].copy()
    occurrence = data.occurrence_time.to_numpy(dtype=float)
    spontaneous = data._spontaneous_observation_time.to_numpy(dtype=float)
    lag = data._recording_delay_days.to_numpy(dtype=float)
    if (
        not np.isfinite(occurrence).all()
        or np.any(occurrence < 0)
        or not np.isfinite(lag).all()
        or np.any(lag < 0)
        or np.isinf(spontaneous).any()
        or np.any(spontaneous < occurrence)
    ):
        raise ValueError("native safety observation potentials contain invalid timing")
    contact = grid[np.searchsorted(grid, occurrence)]
    if not data.ascertainment.isin(("scheduled", "spontaneous", "urgent")).all():
        raise ValueError("native safety ascertainment must name a supported clinical reporting process")
    urgent = data.ascertainment.eq("urgent").to_numpy(dtype=bool)
    if np.isnan(spontaneous[urgent]).any():
        raise ValueError("urgent safety records require a frozen clinical report time")
    observation = np.where(
        urgent, spontaneous, np.minimum(np.where(np.isnan(spontaneous), np.inf, spontaneous), contact)
    )
    data["availability_time"] = observation + lag
    resolution = data.resolution_time.to_numpy(dtype=float)
    if np.isinf(resolution).any() or np.any(resolution < occurrence):
        raise ValueError("native safety resolution cannot precede occurrence")
    # A resolution is documented at the next committed surveillance contact.
    resolution_contact = grid[np.searchsorted(grid, np.where(np.isnan(resolution), np.inf, resolution))]
    data["_resolution_availability_time"] = resolution_contact + lag
    return data.loc[data.availability_time.lt(np.inf)].copy()


def trialdev_safety_records(
    records: pd.DataFrame,
    enrollment: pd.DataFrame,
    *,
    readout_day: float,
    follow_up_days: int,
    surveillance_days: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Release reported safety events without future first-event or potential columns.

    Parameters
    ----------
    records
        Frozen native safety table for the assigned treatment.
    enrollment
        Committed participant identities and enrollment dates.
    readout_day
        Actual programme readout day.
    follow_up_days
        Committed participant follow-up duration.
    surveillance_days
        Committed safety contacts. When supplied, replay detection from retained
        native delay potentials. Urgent clinical reports retain their independent
        clock; scheduled surveillance affects other detection and resolution
        updates. None preserves the frozen native schedule.

    Returns
    -------
    pandas.DataFrame
        Available event records only. Event-free observation time must be supplied
        separately by the study's follow-up records, not inferred from this table.

    Raises
    ------
    ValueError
        If enrollment or record timing is invalid.

    Examples
    --------
    ``events = trialdev_safety_records(records, enrollment, readout_day=84, follow_up_days=84)``.
    """
    if surveillance_days is not None:
        records = _scheduled_safety_records(
            records, surveillance_days=surveillance_days, follow_up_days=follow_up_days
        )
    data = available_clinical_records(records, enrollment, readout_day=readout_day, follow_up_days=follow_up_days)
    data = data.loc[data.event_at_contact & data.reported.eq(True)].copy()
    for field in ("seriousness_criteria", "dictionary_term"):
        if field not in data:
            data[field] = pd.NA
    data["resolution_time"] = data.resolution_time + data.enrollment_day
    data.loc[
        data.resolution_time.gt(readout_day) | data.resolution_time.gt(data.enrollment_day + follow_up_days),
        "resolution_time",
    ] = np.nan
    if surveillance_days is not None:
        data.loc[(data._resolution_availability_time + data.enrollment_day).gt(readout_day), "resolution_time"] = (
            np.nan
        )
    for column in ("occurrence_time", "availability_time"):
        data[column] = data[column] + data.enrollment_day
    return data.loc[
        :,
        [
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
        ],
    ].reset_index(drop=True)


def trialdev_recurrent_records(
    records: pd.DataFrame,
    enrollment: pd.DataFrame,
    *,
    readout_day: float,
    follow_up_days: int,
) -> pd.DataFrame:
    """Release accrued recurrent events and an observation-end row per patient.

    Parameters
    ----------
    records
        Native observed recurrent events, including exactly one event=0 row per
        patient at the observed follow-up end. Events are ascertained immediately
        in this supported native process; this function invents no reporting lag.
    enrollment
        Committed participant identities and programme enrollment dates.
    readout_day
        Actual programme readout day.
    follow_up_days
        Committed maximum participant follow-up.

    Returns
    -------
    pandas.DataFrame
        Raw event rows and one event=0 observation-end row per enrolled patient.
        event_time is days since the patient's enrolment, so exposure is read
        directly; the calendar placement remains the baseline enrollment_day.
        Future loss times are replaced by the current readout or committed study
        end, whichever occurs first.

    Raises
    ------
    ValueError
        If enrollment, event indicators or observed follow-up records are invalid.

    Examples
    --------
    ``events = trialdev_recurrent_records(records, enrollment, readout_day=182, follow_up_days=364)``.
    """
    enrolled = validate_enrollment_clock(enrollment, readout_day=readout_day, follow_up_days=follow_up_days)
    data = records.loc[:, ["participant_id", "outcome_id", "event_time", "event"]].merge(
        enrolled, on="participant_id", how="inner", validate="many_to_one"
    )
    times = data.event_time.to_numpy(dtype=float)
    if not np.isfinite(times).all() or np.any(times < 0) or not data.event.isin([0, 1]).all():
        raise ValueError("recurrent records require finite nonnegative times and binary event indicators")
    ends = data.loc[data.event.eq(0)].copy()
    if ends.participant_id.duplicated().any() or set(ends.participant_id) != set(enrolled.participant_id):
        raise ValueError("recurrent records require one observed follow-up end per enrolled patient")
    ends["cutoff"] = np.minimum(
        ends.event_time.to_numpy(), np.minimum(follow_up_days, readout_day - ends.enrollment_day.to_numpy())
    )
    events = data.loc[data.event.eq(1)].merge(
        ends.loc[:, ["participant_id", "outcome_id", "event_time", "cutoff"]].rename(
            columns={"event_time": "observed_end"}
        ),
        on=["participant_id", "outcome_id"],
        how="left",
        validate="many_to_one",
    )
    if events.observed_end.isna().any() or events.event_time.gt(events.observed_end).any():
        raise ValueError("recurrent event lies outside its observed follow-up")
    events = events.loc[events.event_time.le(events.cutoff)].copy()
    ends["event_time"] = ends.cutoff
    columns = ["participant_id", "outcome_id", "event_time", "event"]
    return pd.concat([events.loc[:, columns], ends.loc[:, columns]], ignore_index=True).sort_values(
        ["participant_id", "event_time", "event"], ascending=[True, True, False], ignore_index=True
    )


def trialdev_contact_records(
    records: pd.DataFrame,
    enrollment: pd.DataFrame,
    *,
    readout_day: float,
    follow_up_days: int,
    surveillance_days: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Expose accrued native surveillance contacts without asserting absent events.

    Parameters
    ----------
    records
        Native contact records on the committed surveillance grid.
    enrollment
        Committed participant identities and programme enrollment dates.
    readout_day
        Actual programme readout day.
    follow_up_days
        Committed maximum participant follow-up.
    surveillance_days
        Selected native surveillance contacts, or None for the original grid.

    Returns
    -------
    pandas.DataFrame
        Available contact and record-availability times on the programme calendar.
        Contacts establish surveillance opportunities, not complete event reporting
        or successful acquisition of a primary measurement.

    Raises
    ------
    ValueError
        If enrollment, contact timing or contact identities are invalid.

    Examples
    --------
    ``contacts = trialdev_contact_records(records, enrollment, readout_day=84, follow_up_days=168)``.
    """
    if surveillance_days is not None:
        _surveillance_grid(surveillance_days, follow_up_days)
        # No participant rows exist before enrollment. The materializer checks
        # the frozen pool's schedule even when an arm has no records yet.
        if not enrollment.empty and not set(surveillance_days).issubset(set(records.occurrence_time)):
            raise ValueError("surveillance dates are outside the frozen contact grid")
        records = records.loc[records.occurrence_time.isin(surveillance_days)]
    data = available_clinical_records(records, enrollment, readout_day=readout_day, follow_up_days=follow_up_days)
    if data.duplicated(["participant_id", "occurrence_time"]).any():
        raise ValueError("surveillance contact identities must be unique")
    for column in ("occurrence_time", "availability_time"):
        data[column] = data[column] + data.enrollment_day
    return data.loc[:, ["participant_id", "occurrence_time", "availability_time"]].reset_index(drop=True)


def trialdev_exposure_records(
    records: pd.DataFrame,
    enrollment: pd.DataFrame,
    *,
    readout_day: float,
    follow_up_days: int,
) -> pd.DataFrame:
    """Release accrued treatment exposure on the native clinical clock.

    Parameters
    ----------
    records
        Native participant_id, occurrence_time and received_arm records. The
        supported exposure process records receipt at occurrence without a
        separate reporting-delay mechanism.
    enrollment
        Committed participant identities and programme enrollment dates.
    readout_day
        Programme readout day.
    follow_up_days
        Committed participant follow-up duration.

    Returns
    -------
    pandas.DataFrame
        Received treatment and programme occurrence time, without private initial
        adherence flags or future exposure states.

    Raises
    ------
    ValueError
        If timing, exposure identities or received treatment values are invalid.

    Examples
    --------
    ``exposure = trialdev_exposure_records(records, enrollment, readout_day=84, follow_up_days=168)``.
    """
    observed = records.loc[:, ["participant_id", "occurrence_time", "received_arm"]].assign(
        availability_time=records.occurrence_time
    )
    data = available_clinical_records(observed, enrollment, readout_day=readout_day, follow_up_days=follow_up_days)
    if (
        data.duplicated(["participant_id", "occurrence_time"]).any()
        or not data.received_arm.isin(["active", "control"]).all()
    ):
        raise ValueError("exposure requires unique patient times and supported received treatments")
    data["occurrence_time"] = data.occurrence_time + data.enrollment_day
    return data.loc[:, ["participant_id", "occurrence_time", "received_arm"]].reset_index(drop=True)


def trialdev_measurement_information_day(
    records: pd.DataFrame,
    enrollment: pd.DataFrame,
    *,
    outcome_id: str,
    participant_count: int,
    follow_up_days: int,
    deadline_day: float,
) -> float | None:
    """Find when the requested number of primary measurements becomes available.

    Parameters
    ----------
    records
        Frozen native measurement records for the assigned patients and treatments.
    enrollment
        Committed patients enrolled by the prospective deadline.
    outcome_id
        Protocol primary outcome identifier.
    participant_count
        Prespecified number of distinct observed primary measurements required.
    follow_up_days
        Primary assessment time relative to each participant's enrollment.
    deadline_day
        Prospective programme deadline for reaching the information threshold.

    Returns
    -------
    float or None
        Calendar availability of the threshold measurement, including recording
        delays. None means the information threshold is not reached by the deadline;
        it must not be presented as a completed information look.

    Raises
    ------
    ValueError
        If the threshold, records, identities or observation timing are invalid.

    Examples
    --------
    Missing visits and measurements not yet reported never count as available primary outcomes.
    """
    if isinstance(participant_count, bool) or not isinstance(participant_count, int) or participant_count < 1:
        raise ValueError("information look requires a positive integer participant count")
    if not isinstance(outcome_id, str) or not outcome_id:
        raise ValueError("information look requires a primary outcome identifier")
    primary = records.loc[records.outcome_id.eq(outcome_id) & records.occurrence_time.eq(follow_up_days)]
    data = trialdev_measurement_records(
        primary,
        enrollment,
        readout_day=deadline_day,
        follow_up_days=follow_up_days,
        assessment_days=(follow_up_days,),
    )
    if data.participant_id.duplicated().any():
        raise ValueError("information look requires one primary measurement per participant")
    times = data.loc[data.observed, "availability_time"].sort_values().tolist()
    return None if len(times) < participant_count else float(times[participant_count - 1])


def trialdev_recurrent_information_day(
    records: pd.DataFrame,
    enrollment: pd.DataFrame,
    *,
    outcome_id: str,
    participant_count: int,
    follow_up_days: int,
    deadline_day: float,
) -> float | None:
    """Find when enough complete primary recurrent-outcome observation windows exist.

    Parameters
    ----------
    records
        Assigned native observed recurrent records, including observation-end rows.
    enrollment
        Committed patients enrolled by the prospective deadline.
    outcome_id
        Protocol recurrent primary outcome identifier.
    participant_count
        Required number of participants with complete primary follow-up.
    follow_up_days
        Committed length of each participant's primary observation window.
    deadline_day
        Prospective calendar deadline for reaching the threshold.

    Returns
    -------
    float or None
        Calendar day of the threshold completed window, or None if unreached.
        Zero-event windows count; early loss and incomplete windows do not.

    Raises
    ------
    ValueError
        If the requested threshold or observed event/follow-up records are invalid.

    Examples
    --------
    Completion is distinct from an event-count trigger or a statistical information fraction.
    Native recurrent observations currently assume immediate event ascertainment.
    """
    if isinstance(participant_count, bool) or not isinstance(participant_count, int) or participant_count < 1:
        raise ValueError("information look requires a positive integer participant count")
    if not isinstance(outcome_id, str) or not outcome_id:
        raise ValueError("information look requires a primary outcome identifier")
    data = trialdev_recurrent_records(
        records.loc[records.outcome_id.eq(outcome_id)],
        enrollment,
        readout_day=deadline_day,
        follow_up_days=follow_up_days,
    )
    ends = data.loc[data.event.eq(0)].merge(
        enrollment.loc[:, ["participant_id", "enrollment_day"]], on="participant_id", validate="one_to_one"
    )
    complete = ends.event_time.ge(follow_up_days)
    completed = (ends.loc[complete, "event_time"] + ends.loc[complete, "enrollment_day"]).sort_values().tolist()
    return None if len(completed) < participant_count else float(completed[participant_count - 1])
