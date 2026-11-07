"""Deterministic clinical tabulation and analysis projections without simulation."""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Literal, get_args

import numpy as np
import pandas as pd

_DAYS_PER_UNIT = {"hour": 1.0 / 24.0, "day": 1.0, "week": 7.0}


class StandardsProjectionError(ValueError):
    """Raised when observed clinical records cannot support their projection."""


SeriousnessCriterion = Literal["AESCONG", "AESDISAB", "AESDTH", "AESHOSP", "AESLIFE", "AESMIE"]


# Nonterminal study discontinuation terms from CDASHIG v2.1 NCOMPLT.
# Death and completion are distinct recorded sources, never withdrawal reasons.
WithdrawalReason = Literal[
    "ADVERSE EVENT",
    "DISEASE RELAPSE",
    "LACK OF EFFICACY",
    "LOST TO FOLLOW-UP",
    "NON-COMPLIANCE WITH STUDY DRUG",
    "PHYSICIAN DECISION",
    "PREGNANCY",
    "PROGRESSIVE DISEASE",
    "PROTOCOL DEVIATION",
    "RECOVERY",
    "SITE TERMINATED BY SPONSOR",
    "STUDY TERMINATED BY SPONSOR",
    "TECHNICAL PROBLEMS",
    "WITHDRAWAL BY PARENT/GUARDIAN",
    "WITHDRAWAL BY SUBJECT",
    "OTHER",
]


def sdtm_sex(values: pd.Series) -> pd.Series:
    """Encode recorded sex using CDISC SDTM C66731 (2026-03-27).

    Parameters
    ----------
    values : pandas.Series
        Recorded female, male, intersex or unknown values, or their submission
        codes. Missing and explicitly unknown values become U. Numeric codes
        require an explicit source dictionary before projection.

    Returns
    -------
    pandas.Series
        F, M, INTERSEX or U, preserving the source index.

    Raises
    ------
    StandardsProjectionError
        If a present value has no supported, unambiguous interpretation.

    Examples
    --------
    >>> sdtm_sex(pd.Series(["female", "M", None])).tolist()
    ['F', 'M', 'U']

    Notes
    -----
    Terminology source: NCI EVS, CDISC SDTM Terminology 2026-03-27,
    https://evs.nci.nih.gov/ftp1/CDISC/SDTM/Archive/SDTM%20Terminology%202026-03-27.txt.
    Gender is a distinct source concept and is not inferred here.
    """
    aliases = {
        "F": "F",
        "FEMALE": "F",
        "M": "M",
        "MALE": "M",
        "INTERSEX": "INTERSEX",
        "U": "U",
        "UNK": "U",
        "UNKNOWN": "U",
        "": "U",
    }
    normalized = values.map(lambda value: value.strip().upper() if isinstance(value, str) else value)
    invalid = normalized.notna() & ~normalized.isin(aliases)
    if invalid.any():
        raise StandardsProjectionError("recorded sex contains an unsupported value; provide its source coding")
    return normalized.map(aliases).fillna("U")


def sdtm_ae_seriousness(serious: pd.Series, criteria: pd.Series | None) -> pd.DataFrame:
    """Project recorded seriousness criteria without inferring a reason.

    Parameters
    ----------
    serious : pandas.Series
        Recorded boolean seriousness for each incident event.
    criteria : pandas.Series or None
        Pipe-separated SDTM criterion names. Missing means unknown; an empty
        string means the source explicitly records no seriousness criterion.

    Returns
    -------
    pandas.DataFrame
        Six criterion columns containing Y, N or an empty unknown value.

    Raises
    ------
    StandardsProjectionError
        If identities, criterion names or seriousness assertions conflict.

    Examples
    --------
    >>> sdtm_ae_seriousness(pd.Series([True]), pd.Series(["AESHOSP"]))["AESHOSP"].iloc[0]
    'Y'
    """
    columns = get_args(SeriousnessCriterion)
    if criteria is not None and not serious.index.equals(criteria.index):
        raise StandardsProjectionError("safety criteria and seriousness indexes differ")
    rows: list[list[str]] = []
    for position, status in enumerate(serious):
        if not isinstance(status, (bool, np.bool_)):
            raise StandardsProjectionError("safety seriousness requires an observed boolean")
        value = None if criteria is None else criteria.iloc[position]
        if not isinstance(value, str):
            if pd.api.types.is_scalar(value) and pd.isna(value):
                rows.append([""] * len(columns))
                continue
            raise StandardsProjectionError("safety criteria require pipe-separated criterion names")
        names = value.split("|") if value else []
        if len(names) != len(set(names)) or not set(names) <= set(columns):
            raise StandardsProjectionError("safety criteria contain duplicate or unknown names")
        if bool(names) != bool(status):
            raise StandardsProjectionError("safety criteria contradict recorded seriousness")
        rows.append(["Y" if column in names else "N" for column in columns])
    return pd.DataFrame(rows, index=serious.index, columns=columns)


def sdtm_ae_severity(value: str) -> str:
    """Project an explicit mild, moderate or severe grade without inferring it.

    Parameters
    ----------
    value : str
        Recorded severity label, which may instead describe a combined stratum
        or seriousness. Those labels do not identify an SDTM severity grade.

    Returns
    -------
    str
        Uppercase severity grade, or an empty value when no grade is identified.

    Examples
    --------
    >>> sdtm_ae_severity("moderate")
    'MODERATE'
    >>> sdtm_ae_severity("mild_or_moderate")
    ''
    """
    normalized = value.upper()
    return normalized if normalized in {"MILD", "MODERATE", "SEVERE"} else ""


def clinical_study_days(values: pd.Series, unit: str) -> pd.Series:
    """Convert elapsed clinical time to calendar study days without day zero.

    The reference date starts at elapsed time zero. Calendar dates are obtained
    by flooring elapsed days, not by rounding to the nearest midnight.

    Parameters
    ----------
    values : pandas.Series
        Elapsed times relative to the reference date; missing values are retained.
    unit : str
        Clinical time unit: ``hour``, ``day`` or ``week``.

    Returns
    -------
    pandas.Series
        Integer-valued study days, with missing values retained as NaN.

    Raises
    ------
    StandardsProjectionError
        If the time unit is unsupported or a time is infinite.

    Examples
    --------
    >>> clinical_study_days(pd.Series([-0.1, 0.0, 0.9, 1.0]), "day").tolist()
    [-1.0, 1.0, 1.0, 2.0]
    """
    if unit not in _DAYS_PER_UNIT:
        raise StandardsProjectionError(f"unsupported clinical time unit: {unit}")
    elapsed = values.to_numpy(dtype=float) * _DAYS_PER_UNIT[unit]
    if np.isinf(elapsed).any():
        raise StandardsProjectionError("clinical time must be finite or missing")
    calendar_day = np.floor(elapsed)
    return pd.Series(np.where(calendar_day >= 0, calendar_day + 1, calendar_day), index=values.index)


def study_dates(study_start: date | None, study_days: pd.Series) -> pd.Series:
    """Map study days without day zero to ISO 8601 calendar dates.

    Parameters
    ----------
    study_start : datetime.date or None
        Declared reference date; None retains unknown dates.
    study_days : pandas.Series
        Calendar study days, where day 1 is the reference date.

    Returns
    -------
    pandas.Series
        ISO dates, with empty strings for unavailable dates.
    """
    if study_start is None:
        return pd.Series("", index=study_days.index)
    return study_days.map(
        lambda value: "" if pd.isna(value) else (study_start + timedelta(days=int(value) - int(value > 0))).isoformat()
    )


def clinical_timing(
    values: pd.Series,
    unit: str,
    study_start: date | None,
    *,
    reference_days: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Project elapsed times through each participant's reference calendar.

    Parameters
    ----------
    values : pandas.Series
        Elapsed time from the participant's reference instant.
    unit : str
        Clinical time unit: hour, day or week.
    study_start : datetime.date or None
        Calendar origin for the trial; None retains unknown dates and study days.
    reference_days : pandas.Series or None
        Participant reference instants in days from the calendar origin, aligned
        to values. None means every participant starts at the calendar origin.

    Returns
    -------
    tuple[pandas.Series, pandas.Series]
        ISO dates and integer study days without day zero. Study days compare
        calendar dates, so a short elapsed interval may cross midnight.

    Raises
    ------
    StandardsProjectionError
        If reference instants are missing, infinite or misaligned, or the elapsed
        time unit is unsupported.

    Notes
    -----
    CDISC SDTMIG 3.3 section 4.4.4 defines study day from the date portions of
    the observation and participant reference dates, without a day zero.

    Examples
    --------
    >>> dates, days = clinical_timing(
    ...     pd.Series([0.3]), "day", date(2025, 1, 1), reference_days=pd.Series([10.8])
    ... )
    >>> dates.tolist(), days.tolist()
    (['2025-01-12'], [2.0])
    """
    if reference_days is None:
        days = clinical_study_days(values, unit)
        return study_dates(study_start, days), days if study_start is not None else days * np.nan
    if not reference_days.index.equals(values.index) or not np.isfinite(reference_days.to_numpy(dtype=float)).all():
        raise StandardsProjectionError("reference days must be finite and aligned to observation rows")
    if unit not in _DAYS_PER_UNIT:
        raise StandardsProjectionError(f"unsupported clinical time unit: {unit}")
    elapsed = values.astype(float) * _DAYS_PER_UNIT[unit] + reference_days
    dates = study_dates(study_start, clinical_study_days(elapsed, "day"))
    days = clinical_study_days(elapsed - np.floor(reference_days), "day")
    return dates, days if study_start is not None else days * np.nan


def sdtm_death_status(
    participant_ids: pd.Series,
    recorded_times: pd.Series,
    *,
    time_unit: str,
    study_start: date | None,
    reference_days: pd.Series | None = None,
) -> pd.DataFrame:
    """Project reported deaths, preserving unknown dates and unreported status.

    Parameters
    ----------
    participant_ids : pandas.Series
        Unique source participant identities in output order.
    recorded_times : pandas.Series
        Observed death times indexed by participant; NaN means death is known
        but its exact time is not. Absence means no observed death assertion.
    time_unit : str
        Source elapsed-time unit: hour, day or week.
    study_start : datetime.date or None
        Declared calendar origin, or None for unknown dates.
    reference_days : pandas.Series or None
        Participant entry offsets in days, aligned to participant_ids.

    Returns
    -------
    pandas.DataFrame
        DTHFL and DTHDTC in source participant order.

    Raises
    ------
    StandardsProjectionError
        If identities or recorded times are ambiguous or invalid.

    Examples
    --------
    Unknown exact times retain a Y flag and an empty date.
    """
    if (
        participant_ids.isna().any()
        or participant_ids.duplicated().any()
        or not recorded_times.index.is_unique
        or recorded_times.index.hasnans
        or not recorded_times.index.isin(participant_ids).all()
    ):
        raise StandardsProjectionError("recorded death refers to an unknown or ambiguous participant")
    if not pd.api.types.is_numeric_dtype(recorded_times) or recorded_times.lt(0).any():
        raise StandardsProjectionError("recorded death times must be nonnegative numbers or unknown")
    dates, _ = clinical_timing(
        participant_ids.map(recorded_times), time_unit, study_start, reference_days=reference_days
    )
    return pd.DataFrame(
        {"DTHFL": np.where(participant_ids.isin(recorded_times.index), "Y", ""), "DTHDTC": dates},
        index=participant_ids.index,
    )


def participation_end_times(
    deaths: pd.Series, completions: pd.Series, *, withdrawals: pd.Series | None = None
) -> pd.Series:
    """Combine recorded participation endings without extending withdrawn follow-up.

    Parameters
    ----------
    deaths, completions : pandas.Series
        Available source times indexed by participant identity, on the same
        elapsed clock. Unknown dates remain unknown. Neither treatment cessation
        nor planned observation end belongs in these sources.
    withdrawals : pandas.Series or None
        Available withdrawal occurrence times on the same clock. Later vital
        status does not extend participation; withdrawal and completion cannot
        both end the same participation episode.

    Returns
    -------
    pandas.Series
        Recorded participation end indexed by participant.

    Raises
    ------
    StandardsProjectionError
        If a participant has duplicate or contradictory participation endings.

    Examples
    --------
    An unreported biological death contributes no participation-end assertion.
    """
    result: pd.Series = pd.concat((deaths, completions))
    if result.index.hasnans or not result.index.is_unique:
        raise StandardsProjectionError("recorded participation endings require unique nonmissing participants")
    if not pd.api.types.is_numeric_dtype(result) or result.lt(0).any() or np.isinf(result.to_numpy(dtype=float)).any():
        raise StandardsProjectionError("recorded participation end must be nonnegative or unknown")
    if withdrawals is None:
        return result
    if withdrawals.index.hasnans or not withdrawals.index.is_unique:
        raise StandardsProjectionError("recorded withdrawals require unique nonmissing participants")
    if (
        not pd.api.types.is_numeric_dtype(withdrawals)
        or withdrawals.lt(0).any()
        or np.isinf(withdrawals.to_numpy(dtype=float)).any()
    ):
        raise StandardsProjectionError("recorded withdrawal must be nonnegative or unknown")
    if withdrawals.index.isin(completions.index).any():
        raise StandardsProjectionError("withdrawal and completion cannot both end study participation")
    shared = deaths.index.intersection(withdrawals.index)
    if deaths.loc[shared].lt(withdrawals.loc[shared]).any():
        raise StandardsProjectionError("recorded withdrawal cannot follow an earlier death")
    result = pd.concat((result.loc[~result.index.isin(withdrawals.index)], withdrawals))
    return result


def _disposition_occurrences(events: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Select unique first incident records with valid participant clocks."""
    selected = events.event_kind.eq(kind)
    if "event_at_contact" in events:
        selected &= events.event_at_contact.eq(True)
    rows = events.loc[selected].copy()
    if rows.participant_id.isna().any() or (
        not rows.empty and not pd.api.types.is_numeric_dtype(rows.occurrence_time)
    ):
        raise StandardsProjectionError("disposition requires a participant and numeric occurrence time")
    if not np.isfinite(rows.occurrence_time.to_numpy(dtype=float)).all() or rows.occurrence_time.lt(0).any():
        raise StandardsProjectionError("disposition occurrence must be finite and nonnegative")
    return rows.sort_values(["participant_id", "occurrence_time"], kind="stable").drop_duplicates(
        "participant_id", keep="first"
    )


def withdrawal_participation_times(events: pd.DataFrame) -> pd.Series:
    """Read participation end from observed withdrawal incidents.

    Parameters
    ----------
    events : pandas.DataFrame
        Available incident events with participant_id, event_kind and
        occurrence_time on one elapsed clock. If event_at_contact is present,
        only incident rows are used. The caller owns report availability.

    Returns
    -------
    pandas.Series
        Earliest recorded study withdrawal indexed by participant.

    Raises
    ------
    StandardsProjectionError
        If withdrawal identities or occurrence times are invalid.

    Examples
    --------
    Treatment discontinuation alone never supplies a participation-end date.
    """
    return (
        _disposition_occurrences(events, "study_withdrawal").set_index("participant_id").occurrence_time.astype(float)
    )


def intercurrent_disposition_records(events: pd.DataFrame) -> pd.DataFrame:
    """Project recorded treatment discontinuation and authored withdrawal reasons.

    Parameters
    ----------
    events : pandas.DataFrame
        Observed incidents with participant_id, event_kind and occurrence_time.
        Withdrawal reasons require both withdrawal_reason (NCOMPLT) and
        withdrawal_reason_text (verbatim). Missing reasons stay unknown.
        The caller owns report availability and clock units.

    Returns
    -------
    pandas.DataFrame
        First incident per participant and disposition kind, with explicit
        DS labels. Ordinary treatment discontinuation never ends participation.

    Raises
    ------
    StandardsProjectionError
        If identities, clocks or recorded withdrawal reasons are invalid.

    Examples
    --------
    Generic withdrawal supplies no reason-specific DS assertion. Explicit
    physician decision is never relabelled as withdrawal by the participant.
    """
    treatment = _disposition_occurrences(events, "treatment_discontinuation")
    treatment = treatment[["participant_id", "occurrence_time"]].assign(
        DSDECOD="DISCONTINUED", DSTERM="DISCONTINUED", DSSCAT="STUDY TREATMENT", DSCAT="DISPOSITION EVENT"
    )
    withdrawal = _disposition_occurrences(events, "study_withdrawal")
    reason = withdrawal.get("withdrawal_reason", pd.Series(index=withdrawal.index, dtype=object))
    text = withdrawal.get("withdrawal_reason_text", pd.Series(index=withdrawal.index, dtype=object))
    known = reason.notna()
    if (known != text.notna()).any() or not reason.loc[known].isin(get_args(WithdrawalReason)).all():
        raise StandardsProjectionError("withdrawal requires a supported reason and its verbatim text")
    if not text.loc[known].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise StandardsProjectionError("withdrawal reason text must be a nonblank string")
    withdrawal = withdrawal.loc[known, ["participant_id", "occurrence_time"]].assign(
        DSDECOD=reason.loc[known], DSTERM=text.loc[known], DSSCAT="STUDY PARTICIPATION", DSCAT="DISPOSITION EVENT"
    )
    return pd.concat((treatment, withdrawal), ignore_index=True)


def completion_disposition_records(records: pd.DataFrame) -> pd.DataFrame:
    """Project recorded completion or administrative end status without inference.

    Parameters
    ----------
    records : pandas.DataFrame
        Available completion forms with participant_id, occurrence_time and
        status. The caller owns attendance, terminal limits, report availability
        and clock conversion; this function never infers completion.

    Returns
    -------
    pandas.DataFrame
        Explicit completed/non-completed disposition or unresolved administrative review.

    Raises
    ------
    StandardsProjectionError
        If a form has an invalid identity, time, status or repeated participant.

    Examples
    --------
    An empty source produces no disposition, regardless of planned follow-up.
    """
    required = {"participant_id", "occurrence_time", "status"}
    if not required.issubset(records.columns):
        raise StandardsProjectionError("completion forms require participant, occurrence and status")
    if records.participant_id.isna().any() or records.participant_id.duplicated().any():
        raise StandardsProjectionError("completion forms require unique nonmissing participants")
    if (
        records.status.isna().any()
        or not records.status.isin(("completed", "not_completed", "status_unresolved")).all()
    ):
        raise StandardsProjectionError("completion forms require an explicit supported end status")
    times = records.occurrence_time
    if not records.empty and (
        times.isna().any()
        or not pd.api.types.is_numeric_dtype(times)
        or not np.isfinite(times).all()
        or times.lt(0).any()
    ):
        raise StandardsProjectionError("completion occurrence must be finite and nonnegative")
    previous = times
    for clock in ("observation_time", "documentation_time", "availability_time"):
        if clock not in records:
            continue
        values = records[clock]
        if (
            values.isna().any()
            or not pd.api.types.is_numeric_dtype(values)
            or not np.isfinite(values).all()
            or values.lt(previous).any()
        ):
            raise StandardsProjectionError("completion report clocks must be finite and chronological")
        previous = values
    unresolved = records.status.eq("status_unresolved")
    return records[["participant_id", "occurrence_time"]].assign(
        DSDECOD=records.status.map(
            {"completed": "COMPLETED", "not_completed": "OTHER", "status_unresolved": "END STATUS UNRESOLVED"}
        ),
        DSTERM=records.status.map(
            {
                "completed": "COMPLETED",
                "not_completed": "REQUIRED FINAL CONTACT NOT ATTENDED",
                "status_unresolved": "END STATUS UNRESOLVED",
            }
        ),
        DSCAT=np.where(unresolved, "OTHER EVENT", "DISPOSITION EVENT"),
        DSSCAT=np.where(records.status.eq("completed"), "STUDY PARTICIPATION", "ADMINISTRATIVE CLOSEOUT"),
    )


def sdtm_disposition(
    records: pd.DataFrame,
    *,
    study_id: str,
    subject_by_participant: pd.Series,
    study_start: date | None,
    time_unit: str,
    reference_days: pd.Series,
    epochs: tuple[tuple[str, float, float], ...] = (),
) -> pd.DataFrame:
    """Project recorded disposition events without inferring study completion.

    Parameters
    ----------
    records : pandas.DataFrame
        participant_id, occurrence_time, DSDECOD, DSCAT and DSSCAT records.
        Optional DSTERM preserves a recorded verbatim reason.
        Unknown occurrence times remain unknown; consent can precede entry.
    study_id : str
        Declared study identity.
    subject_by_participant : pandas.Series
        Unique public identities indexed by source participant.
    study_start : datetime.date or None
        Declared calendar origin, or None for unknown dates.
    time_unit : str
        Source elapsed-time unit: hour, day or week.
    reference_days : pandas.Series
        Participant entry offsets in days, indexed by participant identity.
    epochs : tuple
        Declared participant-clock name, start and end intervals.

    Returns
    -------
    pandas.DataFrame
        DS records ordered by participant, time and disposition term.

    Raises
    ------
    StandardsProjectionError
        If an event has no unambiguous participant or disposition category.

    Examples
    --------
    This projection consumes observed events; a planned last visit is not one.
    """
    if (
        not subject_by_participant.index.is_unique
        or subject_by_participant.index.hasnans
        or subject_by_participant.isna().any()
        or not subject_by_participant.is_unique
        or not records.participant_id.isin(subject_by_participant.index).all()
    ):
        raise StandardsProjectionError("disposition refers to an unknown or ambiguous participant")
    if records[["DSDECOD", "DSCAT", "DSSCAT"]].isna().any().any():
        raise StandardsProjectionError(
            "disposition labels must be explicit, with empty strings for unknown subcategory"
        )
    rows = records.sort_values(["participant_id", "occurrence_time", "DSDECOD"], kind="stable").reset_index(drop=True)
    dates, days = clinical_timing(
        rows.occurrence_time, time_unit, study_start, reference_days=rows.participant_id.map(reference_days)
    )
    return pd.DataFrame(
        {
            "STUDYID": study_id,
            "DOMAIN": "DS",
            "USUBJID": rows.participant_id.map(subject_by_participant),
            "DSSEQ": rows.groupby("participant_id").cumcount() + 1,
            "DSTERM": rows.get("DSTERM", rows.DSDECOD).fillna(rows.DSDECOD),
            "DSDECOD": rows.DSDECOD,
            "DSCAT": rows.DSCAT,
            "DSSCAT": rows.DSSCAT,
            "EPOCH": clinical_epochs(rows.occurrence_time, epochs),
            "DSSTDTC": dates,
            "DSSTDY": days,
        }
    )


def sdtm_adverse_events(
    events: pd.DataFrame,
    *,
    study_id: str,
    subject_by_participant: pd.Series,
    study_start: date | None,
    time_unit: str,
    reference_days: pd.Series | None = None,
    epochs: tuple[tuple[str, float, float], ...] = (),
) -> pd.DataFrame:
    """Project observed safety episodes without inventing dictionary terms or seriousness.

    Parameters
    ----------
    events : pandas.DataFrame
        Observed incident episodes with participant_id, event_id, occurrence_time,
        resolution_time, severity and serious. Optional dictionary_term and
        seriousness_criteria retain explicitly recorded coding; absence is unknown.
    study_id : str
        Declared study identifier.
    subject_by_participant : pandas.Series
        Public subject identifiers indexed by source participant identity.
    study_start : datetime.date or None
        Calendar origin, or None when dates are unknown.
    time_unit : str
        Source elapsed-time unit: hour, day or week.
    reference_days : pandas.Series or None
        Participant reference dates in elapsed calendar days, aligned to events.
    epochs : tuple
        Declared participant-clock periods as name, start and end triples.

    Returns
    -------
    pandas.DataFrame
        AE records with one sequence per subject and linked clinical dates.

    Raises
    ------
    StandardsProjectionError
        If subjects, seriousness, decoded terms or event chronology are invalid.

    Examples
    --------
    Supply already observed episodes; this projection never selects private events.
    """
    if (
        not subject_by_participant.index.is_unique
        or subject_by_participant.index.hasnans
        or subject_by_participant.isna().any()
        or not subject_by_participant.is_unique
        or not events.participant_id.isin(subject_by_participant.index).all()
    ):
        raise StandardsProjectionError("safety event refers to an unknown or ambiguous subject")
    if not events.serious.isin([True, False]).all():
        raise StandardsProjectionError("safety seriousness requires an observed boolean")
    if events.resolution_time.lt(events.occurrence_time).any():
        raise StandardsProjectionError("safety resolution precedes occurrence")
    decoded = events.get("dictionary_term", pd.Series("", index=events.index, dtype=object)).fillna("")
    if not decoded.map(lambda value: isinstance(value, str)).all():
        raise StandardsProjectionError("safety dictionary term must be recorded text or missing")
    start_dates, start_days = clinical_timing(
        events.occurrence_time, time_unit, study_start, reference_days=reference_days
    )
    end_dates, end_days = clinical_timing(
        events.resolution_time, time_unit, study_start, reference_days=reference_days
    )
    return pd.DataFrame(
        {
            "STUDYID": study_id,
            "DOMAIN": "AE",
            "USUBJID": events.participant_id.map(subject_by_participant),
            "AESEQ": events.groupby("participant_id").cumcount() + 1,
            "AETERM": events.event_id,
            "AEDECOD": decoded,
            "AESEV": events.severity.map(sdtm_ae_severity),
            "AESER": events.serious.map({True: "Y", False: "N"}),
            **sdtm_ae_seriousness(events.serious, events.get("seriousness_criteria")),
            "EPOCH": clinical_epochs(events.occurrence_time, epochs),
            "AESTDTC": start_dates,
            "AEENDTC": end_dates,
            "AESTDY": start_days,
            "AEENDY": end_days,
        }
    )


def sdtm_exposure(
    history: pd.DataFrame,
    *,
    study_id: str,
    subject_by_participant: pd.Series,
    study_start: date | None,
    time_unit: str,
    reference_days: pd.Series | None = None,
) -> pd.DataFrame:
    """Project recorded treatment receipt at individual observation times.

    Parameters
    ----------
    history : pandas.DataFrame
        Participant identity, occurrence_time and received_treatment. Rows are
        receipt observations, not inferred continuous dosing intervals.
    study_id : str
        Declared study identifier.
    subject_by_participant : pandas.Series
        Public subject identifiers indexed by source participant identity.
    study_start : datetime.date or None
        Declared reference date; unknown calendar dates remain empty.
    time_unit : str
        Unit of the source elapsed clock: hour, day or week.

    reference_days : pandas.Series or None
        Reference instants in calendar days, aligned to the history rows.

    Returns
    -------
    pandas.DataFrame
        Linked exposure observations without inferred dose, route or frequency.

    Raises
    ------
    StandardsProjectionError
        If subject linkage, receipt labels or observation times are invalid.
    """
    required = {"participant_id", "occurrence_time", "received_treatment"}
    if not required.issubset(history):
        raise StandardsProjectionError("exposure requires participant, time and received treatment")
    if (
        not subject_by_participant.index.is_unique
        or subject_by_participant.index.hasnans
        or subject_by_participant.isna().any()
        or not subject_by_participant.is_unique
        or not history.participant_id.isin(subject_by_participant.index).all()
        or history.duplicated(["participant_id", "occurrence_time"]).any()
        or not pd.api.types.is_numeric_dtype(history.occurrence_time)
        or not np.isfinite(history.occurrence_time).all()
        or not history.received_treatment.map(lambda value: isinstance(value, str) and bool(value.strip())).all()
    ):
        raise StandardsProjectionError("exposure has invalid subject linkage, timing or received treatment")
    dates, days = clinical_timing(history.occurrence_time, time_unit, study_start, reference_days=reference_days)
    return pd.DataFrame(
        {
            "STUDYID": study_id,
            "DOMAIN": "EX",
            "USUBJID": history.participant_id.map(subject_by_participant),
            "EXSEQ": history.groupby("participant_id").cumcount() + 1,
            "EXTRT": history.received_treatment,
            "EPOCH": "",
            "EXSTDTC": dates,
            "EXENDTC": dates,
            "EXSTDY": days,
            "EXENDY": days,
        }
    )


def exposure_date_bounds(
    history: pd.DataFrame,
    *,
    time_unit: str,
    study_start: date | None,
    reference_days: pd.Series | None = None,
) -> pd.DataFrame:
    """Derive first and last recorded treatment dates from exposure observations.

    Parameters
    ----------
    history : pandas.DataFrame
        Recorded participant_id and finite occurrence_time for treatment receipt.
    time_unit : str
        Elapsed clinical time unit: hour, day or week.
    study_start : datetime.date or None
        Calendar origin; unknown origins retain empty dates.
    reference_days : pandas.Series or None
        Participant reference instants in calendar days, aligned to history rows.

    Returns
    -------
    pandas.DataFrame
        RFXSTDTC and RFXENDTC indexed by participant_id. Subjects without receipt
        records are absent. Bounds describe recorded receipt, not inferred dosing
        between observations or a planned treatment period.

    Raises
    ------
    StandardsProjectionError
        If receipt identities, times or calendar references are invalid.

    Examples
    --------
    >>> history = pd.DataFrame({"participant_id": [1], "occurrence_time": [0.0]})
    >>> exposure_date_bounds(history, time_unit="day", study_start=None).loc[1].tolist()
    ['', '']
    """
    if not {"participant_id", "occurrence_time"}.issubset(history):
        raise StandardsProjectionError("exposure bounds require participant and receipt time")
    if history.participant_id.isna().any() or not np.isfinite(history.occurrence_time).all():
        raise StandardsProjectionError("exposure bounds require known participants and finite receipt times")
    dates, _ = clinical_timing(history.occurrence_time, time_unit, study_start, reference_days=reference_days)
    return (
        history[["participant_id"]]
        .assign(date=dates)
        .groupby("participant_id")
        .agg(RFXSTDTC=("date", "min"), RFXENDTC=("date", "max"))
    )


def sdtm_findings(
    history: pd.DataFrame,
    *,
    study_id: str,
    domain: str,
    test_code: str,
    test_name: str,
    subject_by_participant: pd.Series,
    study_start: date | None,
    time_unit: str,
    reference_days: pd.Series | None = None,
    result_unit: str,
    value_kind: Literal["numeric", "character"] = "numeric",
    object_name: str | None = None,
) -> pd.DataFrame:
    """Project observed values to a declared SDTM findings domain.

    Parameters
    ----------
    history : pandas.DataFrame
        Participant, assigned_arm, occurrence_time, observed_outcome and observed
        records. Occurrence time is elapsed relative to the reference date.
    study_id, domain, test_code, test_name : str
        Study and finding identities; domain is QS, LB, VS or FA.
    subject_by_participant : pandas.Series
        Public subject identifiers indexed by source participant identity.
    study_start : datetime.date or None
        Declared reference date. Unknown dates remain empty.
    time_unit : str
        Elapsed time unit: hour, day or week.
    result_unit : str
        Declared measurement unit, empty when no unit is available.
    object_name : str or None
        Authored event, intervention or condition being assessed. Required for FA;
        distinct from the test performed and never inferred from its name.
    value_kind : {"numeric", "character"}
        Whether observations represent quantitative values or recorded categories.

    reference_days : pandas.Series or None
        Reference instants in calendar days, aligned to the history rows.

    Returns
    -------
    pandas.DataFrame
        Findings with observed values, subject linkage and calendar study days.

    Raises
    ------
    StandardsProjectionError
        If the domain, identities, time unit or observed-value contract is invalid.
    """
    prefix = domain.upper()
    if prefix not in {"QS", "LB", "VS", "FA"}:
        raise StandardsProjectionError(f"unsupported findings domain: {domain}")
    if prefix == "FA" and (object_name is None or not object_name.strip()):
        raise StandardsProjectionError("findings about require an explicitly authored clinical object")
    _require_history(history, subject_by_participant, numeric=value_kind == "numeric")
    dates, study_days = clinical_timing(
        history["occurrence_time"], time_unit, study_start, reference_days=reference_days
    )
    visit_numbers = history.groupby("participant_id").cumcount()
    standardized_result_text = history["observed_outcome"].map(
        lambda value: "" if pd.isna(value) else format(float(value), ".12g") if value_kind == "numeric" else str(value)
    )
    standardized_result = pd.to_numeric(standardized_result_text, errors="coerce")
    return pd.DataFrame(
        {
            "STUDYID": study_id,
            "DOMAIN": prefix,
            "USUBJID": history["participant_id"].map(subject_by_participant),
            f"{prefix}SEQ": history.groupby("participant_id").cumcount() + 1,
            f"{prefix}TESTCD": test_code,
            f"{prefix}TEST": test_name,
            **({"FAOBJ": object_name} if prefix == "FA" else {}),
            f"{prefix}CAT": "CLINICAL OUTCOME ASSESSMENT",
            **({"QSSCAT": ""} if prefix == "QS" else {}),
            f"{prefix}ORRES": history["observed_outcome"].map(
                lambda value: "" if pd.isna(value) else str(float(value)) if value_kind == "numeric" else str(value)
            ),
            f"{prefix}STRESC": standardized_result_text,
            f"{prefix}STRESN": standardized_result,
            f"{prefix}STRESU": np.where(history["observed"], result_unit, ""),
            f"{prefix}STAT": np.where(history["observed"], "", "NOT DONE"),
            f"{prefix}REASND": np.where(history["observed"], "", "NOT OBSERVED"),
            **(
                {
                    "QSLOBXFL": np.where(history["occurrence_time"].eq(0) & history["observed"], "Y", ""),
                }
                if prefix == "QS"
                else {}
            ),
            "VISITNUM": visit_numbers,
            "VISIT": pd.Series(
                np.where(history["occurrence_time"].eq(0), "BASELINE", "ASSESSMENT"), index=history.index
            ),
            "EPOCH": "",
            f"{prefix}DTC": dates,
            f"{prefix}DY": study_days,
        }
    )


def adam_findings(
    history: pd.DataFrame,
    *,
    study_id: str,
    test_code: str,
    test_name: str,
    subject_by_participant: pd.Series,
    study_start: date | None,
    time_unit: str,
    reference_days: pd.Series | None = None,
    value_kind: Literal["numeric", "character"] = "numeric",
) -> pd.DataFrame:
    """Project observed values to an ADaM basic data structure.

    Parameters
    ----------
    history : pandas.DataFrame
        Participant, assigned_arm, occurrence_time, observed_outcome and observed
        records, with elapsed times relative to the reference date.
    study_id, test_code, test_name : str
        Declared study and analysis parameter identities.
    subject_by_participant : pandas.Series
        Public subject identifiers indexed by source participant identity.
    study_start : datetime.date or None
        Declared reference date. Unknown dates remain empty.
    time_unit : str
        Elapsed time unit: hour, day or week.
    value_kind : {"numeric", "character"}
        Whether the analysis value is quantitative or a recorded category.

    reference_days : pandas.Series or None
        Reference instants in calendar days, aligned to the history rows.

    Returns
    -------
    pandas.DataFrame
        Values, observation flags and calendar study days.
        Baseline and change exist only when a time-zero measurement was observed.

    Raises
    ------
    StandardsProjectionError
        If identities, time unit or observed-value contract is invalid.
    """
    _require_history(history, subject_by_participant, numeric=value_kind == "numeric")
    dates, study_days = clinical_timing(
        history["occurrence_time"], time_unit, study_start, reference_days=reference_days
    )
    visit_numbers = history.groupby("participant_id").cumcount()
    analysis_value = history["observed_outcome"].map(
        lambda value: np.nan if pd.isna(value) or value_kind == "character" else float(format(float(value), ".12g"))
    )
    baseline_record = history["occurrence_time"].eq(0) & history["observed"]
    baseline = analysis_value.where(baseline_record).groupby(history["participant_id"]).transform("first")
    return pd.DataFrame(
        {
            "STUDYID": study_id,
            "USUBJID": history["participant_id"].map(subject_by_participant),
            "TRTP": history["assigned_arm"],
            "PARAMCD": test_code,
            "PARAM": test_name,
            "AVISIT": pd.Series(
                np.where(history["occurrence_time"].eq(0), "BASELINE", "ASSESSMENT"), index=history.index
            ),
            "AVISITN": visit_numbers,
            "ADT": dates,
            "ADY": study_days,
            "AVAL": analysis_value,
            **({"AVALC": history["observed_outcome"].fillna("").astype(str)} if value_kind == "character" else {}),
            "BASE": baseline,
            "CHG": analysis_value - baseline,
            "ABLFL": np.where(history["occurrence_time"].eq(0) & history["observed"], "Y", ""),
            "ANL01FL": np.where(history["observed"], "Y", ""),
        }
    )


def _require_history(history: pd.DataFrame, subjects: pd.Series, *, numeric: bool) -> None:
    """Refuse unresolved identity, timing or observed-value inconsistencies."""
    required = {"participant_id", "occurrence_time", "observed_outcome", "observed", "assigned_arm"}
    if not required <= set(history.columns) or not subjects.index.is_unique or subjects.isna().any():
        raise StandardsProjectionError("findings require complete source fields and unique subject mapping")
    if history.participant_id.isna().any() or not history.participant_id.isin(subjects.index).all():
        raise StandardsProjectionError("finding refers to an unknown subject")
    if not history.observed.isin([True, False]).all():
        raise StandardsProjectionError("finding has invalid observation status")
    if not history.observed.astype(bool).equals(history.observed_outcome.notna()):
        raise StandardsProjectionError("finding value and observation status disagree")
    if numeric:
        values = history.observed_outcome.dropna().to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise StandardsProjectionError("observed finding must be finite")


def adam_time_to_event(
    records: pd.DataFrame, *, study_id: str, test_code: str, test_name: str, subject_by_participant: pd.Series
) -> pd.DataFrame:
    """Preserve exact elapsed follow-up and censoring, without inventing event dates.

    Parameters
    ----------
    records : pandas.DataFrame
        Observed participant_id, assigned_arm, observed_time and event_observed.
    study_id, test_code, test_name : str
        Declared study and analysis parameter identity.
    subject_by_participant : pandas.Series
        Unique public subject identifiers indexed by the source participant key.

    Returns
    -------
    pandas.DataFrame
        ADTTE with exact elapsed-day AVAL and CNSR=0 for an observed event.

    Raises
    ------
    StandardsProjectionError
        If identity, follow-up or censoring is invalid.
    """
    required = {"participant_id", "assigned_arm", "observed_time", "event_observed"}
    if not required <= set(records) or records.participant_id.duplicated().any():
        raise StandardsProjectionError("time-to-event analysis requires one complete record per subject")
    if not records.participant_id.isin(subject_by_participant.index).all():
        raise StandardsProjectionError("time-to-event record refers to an unknown subject")
    times = records.observed_time.to_numpy(dtype=float)
    if not np.isfinite(times).all() or (times < 0).any() or not records.event_observed.isin([True, False]).all():
        raise StandardsProjectionError("invalid observed follow-up or event status")
    event = records.event_observed.astype(bool)
    return pd.DataFrame(
        {
            "STUDYID": study_id,
            "USUBJID": records.participant_id.map(subject_by_participant),
            "TRTP": records.assigned_arm,
            "PARAMCD": test_code,
            "PARAM": test_name,
            "AVAL": times,
            "AVALU": "DAYS",
            "CNSR": (~event).astype(int),
            "EVNTDESC": np.where(event, test_name, "CENSORED"),
            "ANL01FL": "Y",
        }
    )


def validate_clinical_epochs(intervals: tuple[tuple[str, float, float], ...], *, horizon: float | None = None) -> None:
    """Validate declared study periods on one participant clock.

    Parameters
    ----------
    intervals
        Ordered (name, start, end) intervals; empty means unspecified.
    horizon
        If supplied, require the declaration to cover zero through this time.

    Raises
    ------
    StandardsProjectionError
        If periods are unnamed, nonfinite, unordered, noncontiguous or incomplete.

    Examples
    --------
    >>> validate_clinical_epochs((("TREATMENT", 0, 28),), horizon=28)
    """
    previous = None
    for name, start, end in intervals:
        if (
            not name.strip()
            or not np.isfinite([start, end]).all()
            or end <= start
            or (previous is not None and start != previous)
        ):
            raise StandardsProjectionError("epoch intervals must be named, finite, ordered and contiguous")
        previous = end
    if intervals and horizon is not None and (intervals[0][1] != 0 or intervals[-1][2] != horizon):
        raise StandardsProjectionError("study epochs must cover the declared clock horizon")


def elapsed_clinical_times(
    values: pd.Series, reference_days: pd.Series, *, declared_times: tuple[float, ...]
) -> pd.Series:
    """Recover elapsed days from recorded calendar times and declared boundaries.

    Parameters
    ----------
    values
        Recorded calendar days; unknown times remain missing.
    reference_days
        Participant origins, aligned to the observation index.
    declared_times
        Authored elapsed assessment, surveillance or epoch boundaries.

    Returns
    -------
    pandas.Series
        Calendar days minus the participant origin. A declared time is restored
        only when adding it to that origin exactly reproduces the recorded
        calendar value. No tolerance admits a distinct neighbouring timestamp.

    Raises
    ------
    StandardsProjectionError
        If origins are missing, nonfinite or not aligned, or a boundary is invalid.

    Examples
    --------
    >>> elapsed_clinical_times(pd.Series([40.1]), pd.Series([12.1]), declared_times=(28,)).tolist()
    [28.0]
    """
    if (
        not values.index.equals(reference_days.index)
        or not np.isfinite(reference_days).all()
        or not np.isfinite(declared_times).all()
    ):
        raise StandardsProjectionError("elapsed clocks require aligned finite origins and declared boundaries")
    elapsed = values - reference_days
    for boundary in declared_times:
        elapsed.loc[values.eq(reference_days + boundary)] = boundary
    return elapsed


def clinical_epochs(values: pd.Series, intervals: tuple[tuple[str, float, float], ...]) -> pd.Series:
    """Assign declared epoch intervals without inferring treatment from exposure.

    Parameters
    ----------
    values : pandas.Series
        Observation times on the same elapsed clock as the intervals.
    intervals : tuple of tuple
        Ordered contiguous (name, start, end) intervals. Internal boundaries
        belong to the next epoch; the final endpoint belongs to the last epoch.
        An empty declaration leaves labels unknown.

    Returns
    -------
    pandas.Series
        Epoch labels, empty for missing times or observations outside the plan.

    Raises
    ------
    StandardsProjectionError
        If intervals are invalid, unordered or noncontiguous.

    Examples
    --------
    >>> clinical_epochs(pd.Series([0, 7, 14]), (("TREATMENT", 0, 7), ("FOLLOW-UP", 7, 14))).tolist()
    ['TREATMENT', 'FOLLOW-UP', 'FOLLOW-UP']
    """
    validate_clinical_epochs(intervals)
    result = pd.Series("", index=values.index, dtype=str)
    for index, (name, start, end) in enumerate(intervals):
        selected = values.ge(start) & (values.le(end) if index == len(intervals) - 1 else values.lt(end))
        result.loc[selected] = name
    return result


def validate_enrollment_clock(enrollment: pd.DataFrame, *, readout_day: float, follow_up_days: float) -> pd.DataFrame:
    """Validate the committed population for a clinical calendar readout.

    Parameters
    ----------
    enrollment
        Complete unique participant_id and nonnegative enrollment_day records.
    readout_day
        Finite programme day at which every supplied participant is enrolled.
    follow_up_days
        Positive finite participant follow-up duration in days.

    Returns
    -------
    pandas.DataFrame
        Validated participant identity and enrollment-day columns.

    Raises
    ------
    ValueError
        If identities, dates or the declared readout/follow-up are invalid.

    Examples
    --------
    ``validate_enrollment_clock(enrollment, readout_day=90, follow_up_days=28)``.
    """
    if (
        isinstance(readout_day, bool)
        or not math.isfinite(readout_day)
        or readout_day < 0
        or isinstance(follow_up_days, bool)
        or not math.isfinite(follow_up_days)
        or follow_up_days <= 0
    ):
        raise ValueError("readout and follow-up must be valid nonnegative calendar durations")
    enrolled = enrollment.loc[:, ["participant_id", "enrollment_day"]]
    if enrolled.isna().any().any() or enrolled.participant_id.duplicated().any():
        raise ValueError("enrollment requires complete unique patient identities and dates")
    days = enrolled.enrollment_day.to_numpy(dtype=float)
    if not np.isfinite(days).all() or np.any(days < 0) or np.any(days > readout_day):
        raise ValueError("enrollment dates must precede the readout")
    return enrolled


def available_clinical_records(
    records: pd.DataFrame,
    enrollment: pd.DataFrame,
    *,
    readout_day: float,
    follow_up_days: float,
    unit: Literal["hour", "day", "week"] = "day",
) -> pd.DataFrame:
    """Select enrolled participants' records available by a calendar readout.

    Parameters
    ----------
    records
        Records with participant_id, occurrence_time and availability_time.
    enrollment
        Committed participant_id and enrollment_day rows for this population.
    readout_day
        Programme day of the actual readout, including its boundary.
    follow_up_days
        Maximum participant follow-up, in days, including its boundary.
    unit
        Unit of the record's elapsed times; enrollment and readout remain days.

    Returns
    -------
    pandas.DataFrame
        Eligible records with enrollment_day attached. Original record times and
        columns are retained. This intermediate may contain private fields and
        is not a participant-safe projection by itself.

    Raises
    ------
    ValueError
        If enrollment, timing or units are invalid.

    Examples
    --------
    ``available_clinical_records(records, enrollment, readout_day=90, follow_up_days=28)``.
    """
    if unit not in _DAYS_PER_UNIT:
        raise ValueError("clinical record time unit must be hour, day or week")
    enrolled = validate_enrollment_clock(enrollment, readout_day=readout_day, follow_up_days=follow_up_days)
    selected = records.merge(enrolled, on="participant_id", how="inner", validate="many_to_one")
    times = selected.loc[:, ["occurrence_time", "availability_time"]].to_numpy(dtype=float)
    if not np.isfinite(times).all() or np.any(times < 0) or np.any(times[:, 1] < times[:, 0]):
        raise ValueError("record availability cannot precede its occurrence")
    factor = _DAYS_PER_UNIT[unit]
    return selected.loc[
        (selected.occurrence_time * factor).le(follow_up_days)
        & (selected.availability_time * factor + selected.enrollment_day).le(readout_day)
    ].copy()
