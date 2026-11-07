"""Independent source-to-standard reconciliation using only pandas and NumPy."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def _calendar_dates(reference: pd.Series, offsets: pd.Series | NDArray[np.float64]) -> pd.Series:
    """Reconstruct dates only where both source date and elapsed offset are known."""
    offsets = pd.Series(offsets, index=reference.index)
    dates = pd.to_datetime(reference.replace("", pd.NA))
    known = dates.notna() & offsets.notna()
    result = pd.Series("", index=reference.index, dtype=object)
    result.loc[known] = (dates.loc[known] + pd.to_timedelta(offsets.loc[known], unit="D")).dt.strftime("%Y-%m-%d")
    return result


def read_tables(path: Path) -> dict[str, pd.DataFrame]:
    """Read the disclosed flat standards archive without extracting files."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(Path(name).name != name for name in names):
            raise ValueError("Standards archive must contain unique flat filenames")
        return {
            name: (
                pd.read_parquet(io.BytesIO(archive.read(name)))
                if name.endswith(".parquet")
                else pd.read_csv(io.BytesIO(archive.read(name)), keep_default_na=False)
            )
            for name in names
        }


def check_supplemental_links(tables: dict[str, pd.DataFrame]) -> int:
    """Verify every supplemental qualifier against its declared parent records."""
    checked = 0
    for name, table in tables.items():
        if not name.startswith("SUPP") or not name.endswith(".parquet"):
            continue
        domain = name.removeprefix("SUPP").removesuffix(".parquet")
        parent = tables.get(f"{domain}.parquet")
        required = ["STUDYID", "RDOMAIN", "USUBJID", "IDVAR", "IDVARVAL", "QNAM"]
        if parent is None or not set(required).issubset(table):
            raise ValueError("Supplemental table lacks its parent or linkage fields")
        if not table.RDOMAIN.eq(domain).all() or table[["STUDYID", "USUBJID", "QNAM"]].isna().any().any():
            raise ValueError("Supplemental parent domain or identity is invalid")
        if table.duplicated(required).any():
            raise ValueError("Supplemental qualifier identity is duplicated")
        for identifier, records in table.groupby(table.IDVAR.fillna(""), sort=False):
            if not isinstance(identifier, str):
                raise ValueError("Supplemental IDVAR must name a parent variable")
            keys = ["STUDYID", "USUBJID"]
            left = records[keys].copy()
            right = parent[keys].copy()
            if identifier:
                if identifier not in parent:
                    raise ValueError("Supplemental IDVAR is absent from its parent")
                source = parent[identifier]
                values = records.IDVARVAL
                if pd.api.types.is_numeric_dtype(source):
                    values = pd.to_numeric(values, errors="raise")
                    if not np.isfinite(values).all():
                        raise ValueError("Supplemental numeric parent identity must be finite")
                left["parent_value"] = values.to_numpy()
                right["parent_value"] = source.to_numpy()
            elif not records.IDVARVAL.fillna("").eq("").all():
                raise ValueError("Supplemental IDVARVAL requires a declared IDVAR")
            if not pd.MultiIndex.from_frame(left).isin(pd.MultiIndex.from_frame(right)).all():
                raise ValueError("Supplemental qualifier refers to an absent parent record")
        checked += len(table)
    return checked


def reconcile_tables(
    participants: pd.DataFrame,
    analysis: pd.DataFrame,
    archive: Path,
    *,
    observational: bool = False,
    endpoint_event: str | None = None,
    intercurrent: pd.DataFrame | None = None,
    completion: pd.DataFrame | None = None,
    safety: pd.DataFrame | None = None,
    exposure: pd.DataFrame | None = None,
    epochs: tuple[tuple[str, float, float], ...] | None = None,
) -> pd.DataFrame:
    """Check exported identities, values, missingness, timing and source linkage.

    No simulation, harness, projector or production grader is imported. Failure
    raises rather than producing a favourable row with an unexecuted comparison.
    """

    epoch_records = 0

    def check_epochs(table: pd.DataFrame, times: pd.Series) -> None:
        nonlocal epoch_records
        if epochs is None:
            return
        labels = np.full(len(times), "", dtype=object)
        previous = 0.0
        for index, (name, start, end) in enumerate(epochs):
            if not name.strip() or not np.isfinite([start, end]).all() or start != previous or end <= start:
                raise ValueError("Public study epochs are invalid")
            mask = times.ge(start) & (times.le(end) if index == len(epochs) - 1 else times.lt(end))
            labels[mask.to_numpy()] = name
            previous = end
        np.testing.assert_array_equal(table.EPOCH, labels)
        epoch_records += len(table)

    tables = read_tables(archive)
    supplemental_rows = check_supplemental_links(tables)
    dm, adsl = tables["DM.parquet"], tables["ADSL.parquet"]
    if dm.SUBJID.duplicated().any() or dm.USUBJID.duplicated().any():
        raise ValueError("Repeated source/standard subject identity")
    source = participants.assign(SUBJID=participants.participant_id.astype(str))
    paired = source.merge(dm, on="SUBJID", how="outer", validate="one_to_one", indicator=True)
    if not paired._merge.eq("both").all() or not paired.assigned_arm.astype(str).equals(paired.ACTARM):
        raise ValueError("DM subject population or treatment differs from source")
    # Independent decoding of the published C66731 source labels. Do not import
    # the production projection: altered demographic records must be detectable.
    expected_sex = pd.Series("U", index=paired.index)
    if "sex" in source:
        labels = paired.sex.fillna("").astype(str).str.strip().str.upper()
        codes = {
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
        if not labels.isin(codes).all():
            raise ValueError("Source sex coding is not declared")
        expected_sex = labels.map(codes)
    np.testing.assert_array_equal(paired.SEX, expected_sex)
    aligned_adsl = adsl.set_index("USUBJID").reindex(paired.USUBJID)
    np.testing.assert_array_equal(aligned_adsl.SEX, expected_sex)
    if "age_years" in source:
        np.testing.assert_allclose(paired.AGE, paired.age_years, rtol=0, atol=0, equal_nan=True)
        np.testing.assert_array_equal(paired.AGEU, np.where(paired.age_years.notna(), "YEARS", ""))
    for field, variable in (("site_id", "SITEID"), ("country_code", "COUNTRY")):
        if field in source:
            np.testing.assert_array_equal(paired[variable], paired[field])
            np.testing.assert_array_equal(aligned_adsl[variable], paired[field])
    if observational and (dm.ARM.ne("").any() or adsl.ITTFL.ne("").any()):
        raise ValueError("Observational evidence is incorrectly labelled randomized")
    if not observational and (dm.ARM.ne(dm.ACTARM).any() or not adsl.ITTFL.eq("Y").all()):
        raise ValueError("Randomized treatment or population flag differs")
    joined = paired.drop(columns=["_merge", "AGE", "AGEU", "SITEID", "COUNTRY"], errors="ignore").merge(
        adsl, on=["STUDYID", "USUBJID"], validate="one_to_one"
    )
    if len(joined) != len(participants) or not joined.TRT01A.equals(joined.ACTARM):
        raise ValueError("ADSL population or treatment differs from DM")
    dictionary = tables["variables.csv"]
    column_metadata = "entry_kind" in dictionary
    if not column_metadata:
        dictionary = dictionary.assign(
            entry_kind=np.where(dictionary.dataset.eq("ADBDS"), "parameter_mapping", "source_mapping"),
            parameter=np.where(dictionary.dataset.eq("ADBDS"), dictionary.variable, ""),
            label=dictionary.source_field,
        )
    elif not dictionary.entry_kind.isin({"column", "source_mapping", "parameter_mapping"}).all():
        raise ValueError("Unknown standards dictionary entry kind")
    if column_metadata:
        columns = dictionary.loc[dictionary.entry_kind.eq("column")]
        for name, table in tables.items():
            if not name.endswith(".parquet"):
                continue
            declared = columns.loc[columns.dataset.eq(name.removesuffix(".parquet"))]
            if declared.variable.duplicated().any() or set(declared.variable) != set(table.columns):
                raise ValueError("Standards column dictionary differs from its physical schema")
    for row in dictionary.loc[dictionary.dataset.eq("ADSL") & dictionary.entry_kind.eq("source_mapping")].itertuples():
        if not row.source_field or row.source_field not in joined or row.variable not in joined:
            raise ValueError("ADSL source mapping names an absent field")
        pd.testing.assert_series_equal(joined[row.source_field], joined[row.variable], check_names=False)
    ids = dm.set_index("SUBJID").USUBJID
    reference_dates = dm.set_index("USUBJID").RFSTDTC
    for name, table in tables.items():
        if not name.endswith(".parquet") or "DOMAIN" not in table or "USUBJID" not in table:
            continue
        domain = name.removesuffix(".parquet")
        for suffix in ("", "ST", "EN"):
            day_column, date_column = f"{domain}{suffix}DY", f"{domain}{suffix}DTC"
            if day_column not in table:
                continue
            if date_column not in table:
                raise ValueError(f"{day_column} lacks its source date column")
            reference = pd.to_datetime(table.USUBJID.map(reference_dates).replace("", pd.NA))
            recorded = pd.to_datetime(table[date_column].replace("", pd.NA))
            elapsed = (recorded - reference).dt.days
            expected_day = elapsed.where(elapsed.lt(0), elapsed + 1)
            np.testing.assert_allclose(table[day_column], expected_day, rtol=0, atol=0, equal_nan=True)
    expected_ids = analysis.participant_id.astype(str).map(ids)
    comparisons = []
    if endpoint_event not in (None, "death"):
        raise ValueError("Public endpoint event designation is unsupported")
    reported_deaths = (
        analysis.loc[analysis.event_observed.eq(True) & analysis.event_type.eq("primary_event")]
        if endpoint_event == "death"
        else pd.DataFrame(columns=["participant_id", "observed_time"])
    )
    if reported_deaths.participant_id.duplicated().any():
        raise ValueError("Public death records repeat a participant")
    death_times = reported_deaths.set_index(reported_deaths.participant_id.astype(str)).observed_time.astype(float)
    source_ids = paired.participant_id.astype(str)
    expected_flags = np.where(source_ids.isin(death_times.index), "Y", "")
    expected_times = source_ids.map(death_times)
    death_enrollment = paired.get("enrollment_day", pd.Series(0.0, index=paired.index))
    offsets = np.floor(death_enrollment - np.floor(death_enrollment) + expected_times)
    expected_dates = _calendar_dates(paired.RFSTDTC, offsets)
    np.testing.assert_array_equal(paired.get("DTHFL", pd.Series("", index=paired.index)), expected_flags)
    np.testing.assert_array_equal(paired.get("DTHDTC", pd.Series("", index=paired.index)), expected_dates)
    dispositions = tables.get("DS.parquet", pd.DataFrame(columns=["DSDECOD", "USUBJID", "DSTERM", "DSSCAT"]))
    deaths = dispositions.loc[dispositions.DSDECOD.eq("DEATH")]
    expected_subjects = set(paired.loc[source_ids.isin(death_times.index), "USUBJID"])
    if set(deaths.USUBJID) != expected_subjects or deaths.USUBJID.duplicated().any():
        raise ValueError("DS deaths differ from public observed death records")
    if not deaths.empty:
        assert deaths.DSCAT.eq("DISPOSITION EVENT").all()
        assert deaths.DSSCAT.eq("STUDY PARTICIPATION").all()
        assert deaths.DSTERM.eq("DEATH").all()
        date_by_subject = pd.Series(expected_dates.to_numpy(), index=paired.USUBJID)
        np.testing.assert_array_equal(deaths.DSSTDTC, deaths.USUBJID.map(date_by_subject))
        event_by_subject = pd.Series(expected_times.to_numpy(), index=paired.USUBJID)
        check_epochs(deaths, deaths.USUBJID.map(event_by_subject))
    if endpoint_event == "death":
        comparisons.append({"quantity": "recorded death and disposition", "rows": len(deaths), "maximum_error": 0.0})
    consent_times = paired.get("consent_time", pd.Series(np.nan, index=paired.index))
    consent_offsets = np.floor(death_enrollment - np.floor(death_enrollment) + consent_times)
    consent_dates = _calendar_dates(paired.RFSTDTC, consent_offsets)
    np.testing.assert_array_equal(paired.get("RFICDTC", pd.Series("", index=paired.index)), consent_dates)
    consents = dispositions.loc[dispositions.DSDECOD.eq("INFORMED CONSENT OBTAINED")]
    if (
        set(consents.USUBJID) != set(paired.loc[consent_times.notna(), "USUBJID"])
        or consents.USUBJID.duplicated().any()
    ):
        raise ValueError("DS consent differs from recorded participant consent")
    if not consents.empty:
        assert consents.DSCAT.eq("PROTOCOL MILESTONE").all()
        assert consents.DSSCAT.eq("").all()
        dates_by_subject = pd.Series(consent_dates.to_numpy(), index=paired.USUBJID)
        np.testing.assert_array_equal(consents.DSSTDTC, consents.USUBJID.map(dates_by_subject))
        comparisons.append({"quantity": "recorded consent", "rows": len(consents), "maximum_error": 0.0})
    completion_fields = [
        "completion_time",
        "completion_observation_time",
        "completion_documentation_time",
        "completion_availability_time",
    ]
    if (set(completion_fields) | {"completion_status"}) & set(paired):
        if completion is not None or not (set(completion_fields) | {"completion_status"}).issubset(paired):
            raise ValueError("Public participant completion must have one source and all report clocks")
        clocks = paired[completion_fields]
        if (clocks.notna().any(axis=1) & clocks.isna().any(axis=1)).any():
            raise ValueError("Public completion clocks have inconsistent missingness")
        known = clocks.loc[clocks.notna().all(axis=1)]
        completion_clock_values = known.to_numpy(dtype=float)
        if not np.isfinite(completion_clock_values).all() or (np.diff(completion_clock_values, axis=1) < 0).any():
            raise ValueError("Public completion report clocks are invalid")
        if not paired.completion_status.notna().eq(clocks.notna().all(axis=1)).all():
            raise ValueError("Public closeout status is missing or has no available form clocks")
        completion = paired.loc[known.index, ["participant_id", "completion_time", "completion_status"]].rename(
            columns={"completion_time": "occurrence_time", "completion_status": "status"}
        )
    completion_times = pd.Series(dtype=float)
    closeout_times = pd.Series(dtype=float)
    if completion is not None:
        if completion.participant_id.duplicated().any() or completion.participant_id.isna().any():
            raise ValueError("Public completion records repeat or omit a participant")
        if (
            completion.status.isna().any()
            or not completion.status.isin(("completed", "not_completed", "status_unresolved")).all()
        ):
            raise ValueError("Public closeout records have an unsupported status")
        report_clocks = [
            name
            for name in ("occurrence_time", "observation_time", "documentation_time", "availability_time")
            if name in completion
        ]
        clock_values = completion[report_clocks].to_numpy(dtype=float)
        if not np.isfinite(clock_values).all() or (np.diff(clock_values, axis=1) < 0).any():
            raise ValueError("Public completion report clocks are invalid")
        completion_times = completion.set_index(completion.participant_id.astype(str)).occurrence_time.astype(float)
        if not completion_times.index.isin(source_ids).all():
            raise ValueError("Public completion refers to an unknown participant")
        if completion_times.isna().any() or not np.isfinite(completion_times).all() or completion_times.lt(0).any():
            raise ValueError("Public completion has an invalid occurrence")
        closeout_times = completion_times.copy()
        completion_times = closeout_times.loc[
            completion.set_index(completion.participant_id.astype(str)).status.eq("completed")
        ]
        if completion_times.index.isin(death_times.index).any():
            raise ValueError("Public completion conflicts with a terminal death")
    completion_offsets = np.floor(death_enrollment - np.floor(death_enrollment) + source_ids.map(completion_times))
    completion_dates = _calendar_dates(paired.RFSTDTC, completion_offsets)
    participation_dates = completion_dates.where(~source_ids.isin(death_times.index), expected_dates)
    withdrawal_dispositions = dispositions.loc[
        ~dispositions.DSDECOD.isin(
            ("DEATH", "COMPLETED", "INFORMED CONSENT OBTAINED", "DISCONTINUED", "END STATUS UNRESOLVED")
        )
        & ~dispositions.DSSCAT.eq("ADMINISTRATIVE CLOSEOUT")
    ]
    if intercurrent is None and not withdrawal_dispositions.empty:
        raise ValueError("DS withdrawal reason has no public incident source")
    if intercurrent is not None:
        withdrawn = intercurrent.loc[intercurrent.event_kind.eq("study_withdrawal")].copy()
        if "event_at_contact" in withdrawn:
            withdrawn = withdrawn.loc[withdrawn.event_at_contact.eq(True)]
        if withdrawn.participant_id.isna().any():
            raise ValueError("withdrawal participant identity is missing")
        if not withdrawn.empty and (
            withdrawn.occurrence_time.isna().any()
            or not pd.api.types.is_numeric_dtype(withdrawn.occurrence_time)
            or not np.isfinite(withdrawn.occurrence_time).all()
            or withdrawn.occurrence_time.lt(0).any()
        ):
            raise ValueError("withdrawal occurrence must be finite and nonnegative")
        withdrawal_times = withdrawn.groupby(withdrawn.participant_id.astype(str)).occurrence_time.min()
        if not withdrawal_times.index.isin(source_ids).all():
            raise ValueError("withdrawal refers to an unknown public participant")
        if withdrawal_times.index.isin(completion_times.index).any():
            raise ValueError("withdrawal contradicts recorded completion")
        overlap = withdrawal_times.index.intersection(death_times.index)
        if death_times.loc[overlap].lt(withdrawal_times.loc[overlap]).any():
            raise ValueError("withdrawal occurs after recorded death")
        withdrawal_offsets = np.floor(death_enrollment - np.floor(death_enrollment) + source_ids.map(withdrawal_times))
        withdrawal_dates = _calendar_dates(paired.RFSTDTC, withdrawal_offsets)
        participation_dates = participation_dates.where(~source_ids.isin(withdrawal_times.index), withdrawal_dates)
        first = withdrawn.sort_values("occurrence_time", kind="stable").drop_duplicates("participant_id")
        withdrawal_codes = first.get("withdrawal_reason", pd.Series(index=first.index, dtype=object))
        withdrawal_texts = first.get("withdrawal_reason_text", pd.Series(index=first.index, dtype=object))
        if (withdrawal_codes.notna() != withdrawal_texts.notna()).any():
            raise ValueError("Public withdrawal reason and verbatim text are incomplete")
        known_reasons = withdrawal_codes.notna()
        reason_subjects = first.loc[known_reasons, "participant_id"].astype(str).map(ids)
        if (
            set(withdrawal_dispositions.USUBJID) != set(reason_subjects)
            or withdrawal_dispositions.USUBJID.duplicated().any()
        ):
            raise ValueError("DS withdrawal reasons differ from public incident sources")
        if not withdrawal_dispositions.empty:
            assert withdrawal_dispositions.DSCAT.eq("DISPOSITION EVENT").all()
            assert withdrawal_dispositions.DSSCAT.eq("STUDY PARTICIPATION").all()
            for field, reason_values in (("DSDECOD", withdrawal_codes), ("DSTERM", withdrawal_texts)):
                by_subject = pd.Series(reason_values.loc[known_reasons].to_numpy(), index=reason_subjects)
                np.testing.assert_array_equal(
                    withdrawal_dispositions[field], withdrawal_dispositions.USUBJID.map(by_subject)
                )
            dates_by_subject = pd.Series(withdrawal_dates.to_numpy(), index=paired.USUBJID)
            np.testing.assert_array_equal(
                withdrawal_dispositions.DSSTDTC, withdrawal_dispositions.USUBJID.map(dates_by_subject)
            )
            times_by_subject = pd.Series(source_ids.map(withdrawal_times).to_numpy(), index=paired.USUBJID)
            check_epochs(withdrawal_dispositions, withdrawal_dispositions.USUBJID.map(times_by_subject))
        comparisons.append({"quantity": "recorded withdrawal", "rows": len(withdrawal_times), "maximum_error": 0.0})
    np.testing.assert_array_equal(paired.get("RFPENDTC", pd.Series("", index=paired.index)), participation_dates)
    closeout_offsets = np.floor(death_enrollment - np.floor(death_enrollment) + source_ids.map(closeout_times))
    reference_end_dates = _calendar_dates(paired.RFSTDTC, closeout_offsets)
    reference_end_dates = reference_end_dates.where(participation_dates.eq(""), participation_dates)
    np.testing.assert_array_equal(paired.get("RFENDTC", pd.Series("", index=paired.index)), reference_end_dates)
    administrative = dispositions.loc[dispositions.DSSCAT.eq("ADMINISTRATIVE CLOSEOUT")]
    source_administrative = (
        pd.DataFrame(columns=["participant_id", "status", "occurrence_time"])
        if completion is None
        else completion.loc[~completion.status.eq("completed")]
    )
    if (
        set(administrative.USUBJID) != set(source_administrative.participant_id.astype(str).map(ids))
        or administrative.USUBJID.duplicated().any()
    ):
        raise ValueError("Administrative end status differs from public closeout forms")
    if not administrative.empty:
        statuses = pd.Series(
            source_administrative.status.to_numpy(), index=source_administrative.participant_id.astype(str).map(ids)
        )
        unresolved = administrative.USUBJID.map(statuses).eq("status_unresolved")
        np.testing.assert_array_equal(administrative.DSDECOD, np.where(unresolved, "END STATUS UNRESOLVED", "OTHER"))
        np.testing.assert_array_equal(
            administrative.DSTERM, np.where(unresolved, "END STATUS UNRESOLVED", "REQUIRED FINAL CONTACT NOT ATTENDED")
        )
        np.testing.assert_array_equal(administrative.DSCAT, np.where(unresolved, "OTHER EVENT", "DISPOSITION EVENT"))
        np.testing.assert_array_equal(administrative.DSSCAT, np.full(len(administrative), "ADMINISTRATIVE CLOSEOUT"))
        dates = pd.Series(_calendar_dates(paired.RFSTDTC, closeout_offsets).to_numpy(), index=paired.USUBJID)
        np.testing.assert_array_equal(administrative.DSSTDTC, administrative.USUBJID.map(dates))
    completed = dispositions.loc[dispositions.DSDECOD.eq("COMPLETED")]
    if (
        set(completed.USUBJID) != set(paired.loc[source_ids.isin(completion_times.index), "USUBJID"])
        or completed.USUBJID.duplicated().any()
    ):
        raise ValueError("DS completion differs from public recorded completion")
    if not completed.empty:
        assert completed.DSCAT.eq("DISPOSITION EVENT").all()
        assert completed.DSSCAT.eq("STUDY PARTICIPATION").all()
        assert completed.DSTERM.eq("COMPLETED").all()
        dates_by_subject = pd.Series(completion_dates.to_numpy(), index=paired.USUBJID)
        np.testing.assert_array_equal(completed.DSSTDTC, completed.USUBJID.map(dates_by_subject))
        times_by_subject = pd.Series(source_ids.map(completion_times).to_numpy(), index=paired.USUBJID)
        check_epochs(completed, completed.USUBJID.map(times_by_subject))
        comparisons.append({"quantity": "recorded completion", "rows": len(completed), "maximum_error": 0.0})
    if intercurrent is not None:
        stops = intercurrent.loc[intercurrent.event_kind.eq("treatment_discontinuation")].copy()
        stops["subject"] = stops.participant_id.astype(str).map(ids)
        if stops.subject.isna().any():
            raise ValueError("Treatment discontinuation refers to an unknown participant")
        first_stops = stops.groupby("subject").occurrence_time.min()
        treatment_rows = dispositions.loc[dispositions.DSDECOD.eq("DISCONTINUED")]
        if set(treatment_rows.USUBJID) != set(first_stops.index) or treatment_rows.USUBJID.duplicated().any():
            raise ValueError("DS treatment disposition differs from observed discontinuations")
        if not treatment_rows.empty:
            assert treatment_rows.DSCAT.eq("DISPOSITION EVENT").all()
            assert treatment_rows.DSSCAT.eq("STUDY TREATMENT").all()
            assert treatment_rows.DSTERM.eq("DISCONTINUED").all()
            follow_up = paired.USUBJID.map(first_stops)
            offsets = np.floor(death_enrollment - np.floor(death_enrollment) + follow_up)
            stop_dates = _calendar_dates(paired.RFSTDTC, offsets)
            stop_by_subject = pd.Series(stop_dates.to_numpy(), index=paired.USUBJID)
            np.testing.assert_array_equal(treatment_rows.DSSTDTC, treatment_rows.USUBJID.map(stop_by_subject))
            check_epochs(treatment_rows, treatment_rows.USUBJID.map(first_stops))
        comparisons.append(
            {"quantity": "recorded treatment discontinuation", "rows": len(first_stops), "maximum_error": 0.0}
        )
    if "ADTTE.parquet" in tables:
        tte = tables["ADTTE.parquet"].sort_values("SRCROW")
        if tte.SRCROW.tolist() != list(range(len(analysis))) or tte.USUBJID.tolist() != expected_ids.tolist():
            raise ValueError("ADTTE lost or substituted a source record")
        np.testing.assert_array_equal(tte.AVAL, analysis.observed_time)
        np.testing.assert_array_equal(tte.CNSR, (~analysis.event_observed.astype(bool)).astype(int))
        comparisons.append({"quantity": "exact follow-up and censoring", "rows": len(tte), "maximum_error": 0.0})
    if "ADBDS.parquet" in tables:
        bds = tables["ADBDS.parquet"]
        declared = dictionary.loc[dictionary.dataset.eq("ADBDS") & dictionary.entry_kind.eq("parameter_mapping")]
        if declared.parameter.duplicated().any():
            raise ValueError("Repeated source mapping for an analysis parameter")
        if not analysis.empty and set(bds.PARAMCD) != set(declared.parameter):
            raise ValueError("Undeclared or absent analysis parameter")
        for row in declared.itertuples():
            if (
                not isinstance(row.source_field, str)
                or not isinstance(row.unit, str)
                or not isinstance(row.label, str)
                or not row.source_field
            ):
                raise ValueError("Findings source field and unit must be strings")
            values = bds.loc[bds.PARAMCD.eq(row.parameter)].sort_values("SRCROW")
            if (
                values.SRCROW.tolist() != list(range(len(analysis)))
                or values.USUBJID.tolist() != expected_ids.tolist()
            ):
                raise ValueError("ADBDS lost or substituted a source record")
            if "occurrence_time" in analysis:
                np.testing.assert_array_equal(values.ELAPDAY, analysis.occurrence_time)
            elif "assessment_time" in analysis:
                np.testing.assert_array_equal(values.ELAPDAY, analysis.assessment_time)
            source_values = analysis[row.source_field].reset_index(drop=True)
            if values.empty:
                comparisons.append({"quantity": row.source_field, "rows": 0, "maximum_error": 0.0})
                continue
            np.testing.assert_array_equal(values.PARAM, np.full(len(values), row.label[:40]))
            domain = values.SRCDOM.iloc[0]
            raw = tables[f"{domain}.parquet"]
            raw = raw.loc[raw[f"{domain}TESTCD"].eq(row.parameter)]
            if domain == "FA":
                if not isinstance(row.clinical_object, str) or not row.clinical_object.strip():
                    raise ValueError("Findings About lacks its authored clinical object")
                if not raw.FAOBJ.eq(row.clinical_object).all():
                    raise ValueError("Findings About object differs from its source definition")
            traced = values.merge(
                raw,
                left_on=["USUBJID", "PARAMCD", "AVISITN"],
                right_on=["USUBJID", f"{domain}TESTCD", "VISITNUM"],
                validate="one_to_one",
                how="outer",
                indicator=True,
            )
            if not traced._merge.eq("both").all():
                raise ValueError("An analysis record has no source tabulation")
            has_result = traced.AVALC.ne("") if row.source_field == "validation_stratum" else traced.AVAL.notna()
            np.testing.assert_array_equal(traced[f"{domain}STRESU"], np.where(has_result, row.unit, ""))
            if column_metadata or domain != "FA" or row.parameter != "PRIMARY":
                np.testing.assert_array_equal(traced[f"{domain}TEST"], np.full(len(traced), row.label[:40]))
            np.testing.assert_array_equal(traced[f"{domain}STAT"], np.where(has_result, "", "NOT DONE"))
            np.testing.assert_array_equal(traced[f"{domain}REASND"], np.where(has_result, "", "NOT OBSERVED"))
            if row.source_field == "validation_stratum":
                np.testing.assert_array_equal(values.AVALC, source_values.fillna("").astype(str))
                np.testing.assert_array_equal(traced.AVALC, traced[f"{domain}STRESC"])
                error = 0.0
            else:
                expected = source_values.to_numpy(dtype=float, na_value=np.nan)
                actual = values.AVAL.to_numpy(dtype=float)
                np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11, equal_nan=True)
                np.testing.assert_array_equal(values.ANL01FL.eq("Y"), source_values.notna())
                np.testing.assert_allclose(traced.AVAL, traced[f"{domain}STRESN"], rtol=0, atol=0, equal_nan=True)
                finite = np.isfinite(expected)
                error = float(np.max(np.abs(actual[finite] - expected[finite]))) if finite.any() else 0.0
            if "ELAPDAY" in traced:
                check_epochs(traced, traced.ELAPDAY)
            np.testing.assert_allclose(traced.ADY, traced[f"{domain}DY"], rtol=0, atol=0, equal_nan=True)
            comparisons.append({"quantity": row.source_field, "rows": len(values), "maximum_error": error})
    if safety is not None:
        ae = tables["AE.parquet"]
        check_epochs(ae, safety.occurrence_time)
        np.testing.assert_array_equal(ae.USUBJID, safety.participant_id.astype(str).map(ids))
        np.testing.assert_array_equal(ae.AESER, safety.serious.map({True: "Y", False: "N"}))
        expected_terms = safety.get("dictionary_term", pd.Series("", index=safety.index)).fillna("")
        np.testing.assert_array_equal(ae.get("AEDECOD", pd.Series("", index=ae.index)), expected_terms)
        for column in ("AESCONG", "AESDISAB", "AESDTH", "AESHOSP", "AESLIFE", "AESMIE"):
            expected_criteria = [
                "" if pd.isna(value) else "Y" if column in value.split("|") else "N"
                for value in safety.get("seriousness_criteria", pd.Series(None, index=safety.index))
            ]
            np.testing.assert_array_equal(ae.get(column, pd.Series("", index=ae.index)), expected_criteria)
        enrollment = (
            safety.participant_id.map(participants.set_index("participant_id").enrollment_day).to_numpy()
            if "enrollment_day" in participants
            else np.zeros(len(safety))
        )
        expected = np.floor(enrollment + safety.occurrence_time.to_numpy()) - np.floor(enrollment)
        reference_dates = ae.USUBJID.map(dm.set_index("USUBJID").RFSTDTC)
        known_dates = reference_dates.notna() & reference_dates.ne("")
        expected_days = np.where(expected >= 0, expected + 1, expected).astype(float)
        expected_days[~known_dates.to_numpy()] = np.nan
        np.testing.assert_allclose(ae.AESTDY, expected_days, rtol=0, atol=0, equal_nan=True)
        if known_dates.any():
            expected_dates = pd.to_datetime(reference_dates[known_dates]) + pd.to_timedelta(
                expected[known_dates], unit="D"
            )
            np.testing.assert_array_equal(ae.loc[known_dates, "AESTDTC"], expected_dates.dt.strftime("%Y-%m-%d"))
        supp = tables["SUPPAE.parquet"]
        maximum_error = 0.0
        for code, field in [
            ("MEDCODE", "medical_code"),
            ("CODESYS", "coding_system"),
            ("ELAPDAY", "occurrence_time"),
            ("AVAILDAY", "availability_time"),
            ("RESOLDAY", "resolution_time"),
        ]:
            if field in safety:
                actual_supplement = supp.loc[supp.QNAM.eq(code), "QVAL"]
                expected_values = safety[field].dropna()
                if field.endswith("_time"):
                    observed = actual_supplement.to_numpy(dtype=float)
                    expected_values_array = expected_values.to_numpy(dtype=float)
                    np.testing.assert_allclose(observed, expected_values_array, rtol=0, atol=1e-11)
                    if len(observed):
                        maximum_error = max(maximum_error, float(np.max(np.abs(observed - expected_values_array))))
                else:
                    np.testing.assert_array_equal(actual_supplement, expected_values.astype(str))
        comparisons.append(
            {"quantity": "recorded safety and availability", "rows": len(ae), "maximum_error": maximum_error}
        )
    if exposure is not None:
        ex = tables["EX.parquet"]
        check_epochs(ex, exposure.occurrence_time)
        np.testing.assert_array_equal(ex.USUBJID, exposure.participant_id.astype(str).map(ids))
        np.testing.assert_array_equal(ex.EXTRT, exposure.received_arm)
        np.testing.assert_array_equal(ex.EXSEQ, exposure.groupby("participant_id").cumcount() + 1)
        for column, source_column, ascending, analysis_column in (
            ("RFXSTDTC", "EXSTDTC", True, "TRTSDT"),
            ("RFXENDTC", "EXENDTC", False, "TRTEDT"),
        ):
            # ISO date strings sort chronologically. Select their extrema without
            # creating a Python aggregation object for every subject.
            expected_dates = (
                ex.dropna(subset=[source_column])
                .sort_values(source_column, ascending=ascending)
                .drop_duplicates("USUBJID")
                .set_index("USUBJID")[source_column]
            )
            np.testing.assert_array_equal(dm[column], dm.USUBJID.map(expected_dates).fillna(""))
            np.testing.assert_array_equal(adsl[analysis_column], adsl.USUBJID.map(expected_dates).fillna(""))
        times = tables["SUPPEX.parquet"]
        if len(times) != len(ex) or not times.QNAM.eq("ELAPDAY").all():
            raise ValueError("Exposure elapsed-time records are incomplete")
        np.testing.assert_array_equal(times.USUBJID, ex.USUBJID)
        np.testing.assert_array_equal(times.IDVARVAL, ex.EXSEQ.astype(str))
        np.testing.assert_allclose(times.QVAL.to_numpy(dtype=float), exposure.occurrence_time, rtol=0, atol=0)
        comparisons.append({"quantity": "recorded treatment receipt", "rows": len(ex), "maximum_error": 0.0})
    elif "EX.parquet" in tables:
        raise ValueError("Exposure projection requires its observed source records")
    if not comparisons:
        raise ValueError("No endpoint comparison was performed")
    if epoch_records:
        comparisons.append({"quantity": "planned study periods", "rows": epoch_records, "maximum_error": 0.0})
    if supplemental_rows:
        comparisons.append(
            {"quantity": "supplemental parent linkage", "rows": supplemental_rows, "maximum_error": 0.0}
        )
    return pd.DataFrame(comparisons).assign(status="passed")


def reconcile_readout(readout: Path, archive: Path, *, extension: str = "parquet") -> pd.DataFrame:
    """Check a phased export against public arm records and its temporal cutoff."""
    import json

    study = json.loads((readout / "study.json").read_text())
    declared_times = sorted(
        {
            0,
            study["follow_up_days"],
            *study["assessment_days"],
            *study["surveillance_days"],
            *(bound for _, start, end in study.get("epochs", []) for bound in (start, end)),
        }
    )
    participants: list[pd.DataFrame] = []
    observations: list[pd.DataFrame] = []
    safety: list[pd.DataFrame] = []
    exposure: list[pd.DataFrame] = []
    intercurrent: list[pd.DataFrame] = []
    completion: list[pd.DataFrame] = []
    for arm in study["arms"]:
        directory = readout / "arms" / arm["arm_id"]

        def load(name: str, directory: Path = directory) -> pd.DataFrame:
            path = directory / f"{name}.{extension}"
            if extension == "parquet":
                return pd.read_parquet(path)
            # Empty criteria mean known absence; NA denotes an unknown assessment.
            return pd.read_csv(
                path,
                float_precision="round_trip",
                converters={"seriousness_criteria": lambda value: None if value == "NA" else value},
            )

        baseline = load("baseline").assign(
            study_id=study["study_id"], assigned_arm=arm["arm_id"], asset_id=arm["asset_id"], dose=arm["dose"]
        )
        if baseline.enrollment_day.gt(study["readout_day"]).any():
            raise ValueError("Public readout contains a future enrollment")
        participants.append(baseline)
        recurrent = (directory / f"recurrent_events.{extension}").exists()
        extra = [
            (name, destination)
            for name, destination in (("intercurrent_events", intercurrent), ("completion_records", completion))
            if (directory / f"{name}.{extension}").exists()
        ]
        for filename, collection in [
            ("recurrent_events" if recurrent else "measurements", observations),
            ("safety_events", safety),
            ("exposure", exposure),
            *extra,
        ]:
            records = load(filename).merge(baseline, on="participant_id", validate="many_to_one")
            if filename == "recurrent_events":
                records = records.rename(columns={"event": "clinical_event"})
            for clock in (
                "occurrence_time",
                "observation_time",
                "documentation_time",
                "availability_time",
                "resolution_time",
            ):
                if clock in records:
                    if records[clock].gt(study["readout_day"]).any():
                        raise ValueError("Public readout contains a future record")
                    # Invert the disclosed calendar addition at authored times.
                    # Exact matching preserves distinct neighbouring timestamps;
                    # subtraction alone can move the final visit outside its epoch.
                    records[clock] = [
                        next((t for t in declared_times if origin + t == day), day - origin)
                        for day, origin in zip(records[clock], records.enrollment_day, strict=True)
                    ]
            collection.append(records)
    population = pd.concat(participants, ignore_index=True)
    dm = read_tables(archive)["DM.parquet"].set_index("SUBJID")
    reference = dm.reindex(population.participant_id.astype(str)).RFSTDTC
    if study.get("calendar_origin") is None:
        if not reference.eq("").all():
            raise ValueError("Readout without a calendar origin contains reference dates")
    else:
        expected = pd.Timestamp(study["calendar_origin"]) + pd.to_timedelta(
            np.floor(population.enrollment_day), unit="D"
        )
        np.testing.assert_array_equal(reference, expected.dt.strftime("%Y-%m-%d"))
    return reconcile_tables(
        population,
        pd.concat(observations, ignore_index=True),
        archive,
        safety=pd.concat([frame for frame in safety if not frame.empty] or safety[:1], ignore_index=True),
        exposure=pd.concat(exposure, ignore_index=True),
        intercurrent=pd.concat(intercurrent, ignore_index=True) if intercurrent else None,
        completion=pd.concat(completion, ignore_index=True) if completion else None,
        epochs=tuple(tuple(epoch) for epoch in study.get("epochs", [])),
    )


def review_development(root: Path, output: Path) -> None:
    """Reconstruct observational and phased exports from disclosed public records."""
    initial = pd.read_csv(root / "initial_ready.csv").rename(columns={"asset_id": "assigned_arm"})
    endpoints = ["das28_score_end_of_follow_up", "infection_aesi_end_of_follow_up"]
    results = [
        reconcile_tables(
            initial.drop(columns=endpoints), initial, root / "initial-standards.zip", observational=True
        ).assign(study="initial observational")
    ]
    for readout in sorted((root / "readouts").iterdir()):
        if readout.is_dir():
            results.append(
                reconcile_readout(readout, readout / "standards.zip", extension="csv").assign(study=readout.name)
            )
    pd.concat(results, ignore_index=True).to_csv(output / "trialdev_standards_reconciliation.csv", index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Disclosed TrialDev directory.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    review_development(args.source, args.output)
