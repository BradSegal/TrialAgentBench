"""Participant documentation for accrued TrialDev source records."""

import csv
from pathlib import Path


def write_readout_dictionary(
    path: Path,
    *,
    baseline_fields: tuple[str, ...],
    outcome_unit: str,
    recurrent: bool,
    intercurrent: bool = False,
    completion: bool = False,
    consent: bool = False,
) -> None:
    """Write field meanings for the source files shared by every study arm.

    Parameters
    ----------
    path
        New CSV path inside the staged public readout.
    baseline_fields
        Public clinical column names, whose labels and units are in the initial
        observational data dictionary. Never pass private source identifiers.
    outcome_unit
        Clinical unit already declared by the public reporting target.
    recurrent
        Whether this study releases recurrent events instead of measurements.

    intercurrent
        Whether the study releases additional-therapy records.
    completion
        Whether explicitly recorded administrative end-status forms are released.

    consent
        Whether native consent relative to enrollment is present in baseline records.

    Raises
    ------
    FileExistsError
        If the destination already exists.

    Notes
    -----
    File names are relative to each arms/<arm_id> directory. Calendar timestamps
    use the programme origin; assessment schedules use days after enrollment.
    This describes records, not an analysis procedure or a missingness model.
    """
    calendar = "days from programme start"
    identifier = ("participant_id", "identifier", "Public participant key joining this study's arm files.")
    occurrence = (
        "occurrence_time",
        calendar,
        "Recorded occurrence date; subtract enrollment_day for time since enrollment.",
    )
    available = (
        "availability_time",
        calendar,
        "Date the record became available; no record is released before this date.",
    )
    fields = {
        "baseline.parquet": (
            identifier,
            *(
                (
                    name,
                    "see initial dictionary",
                    "Baseline clinical field defined in documents/observational_data_dictionary.csv.",
                )
                for name in baseline_fields
            ),
            ("site_id", "identifier", "Recruiting site in the programme network."),
            ("country_code", "ISO 3166-1 alpha-3", "Country of the recruiting site."),
            ("enrollment_day", calendar, "Participant's enrollment date and origin for the study follow-up schedule."),
        ),
        "contacts.parquet": (
            identifier,
            occurrence,
            available,
        ),
        "exposure.parquet": (
            identifier,
            occurrence,
            (
                "received_arm",
                "treatment identifier",
                "Treatment received at this record; randomized assignment is the enclosing arm in study.json.",
            ),
        ),
        "safety_events.parquet": (
            identifier,
            (
                "event_id",
                "identifier",
                "Recorded safety event identifier; interpret with participant and occurrence time.",
            ),
            occurrence,
            available,
            ("medical_code", "code", "Recorded adverse-event code in coding_system."),
            ("coding_system", "code system", "Terminology used for medical_code."),
            (
                "dictionary_term",
                "clinical term",
                "Explicit decoded term in coding_system; missing means no recorded decoding.",
            ),
            ("severity", "recorded category", "Recorded severity category."),
            ("serious", "boolean", "Whether the recorded event is classified as serious."),
            (
                "seriousness_criteria",
                "recorded categories",
                "Reasons separated by |: AESHOSP=hospitalisation, AESLIFE=life threatening, AESDTH=death, "
                "AESDISAB=disability, AESCONG=congenital anomaly, AESMIE=other medical importance. "
                "Empty means none; missing means not recorded.",
            ),
            (
                "resolution_time",
                calendar,
                "Documented resolution date; blank means resolution is not available at this cutoff.",
            ),
        ),
    }
    if consent:
        fields["baseline.parquet"] += (
            (
                "consent_time",
                "days from enrollment",
                "Recorded native consent; add enrollment_day for programme time.",
            ),
        )
    if completion:
        fields["completion_records.parquet"] = (
            identifier,
            occurrence,
            ("observation_time", calendar, "Date the final administrative action was observed."),
            ("documentation_time", calendar, "Date the administrative end-status form was documented."),
            available,
            (
                "status",
                "completed; not_completed; status_unresolved",
                "Recorded end status; only completed confirms clinical completion. Administrative review is not a patient contact.",
            ),
        )
    if intercurrent:
        fields["intercurrent_events.parquet"] = (
            identifier,
            ("event_id", "identifier", "Recorded additional-therapy identifier."),
            ("event_kind", "category", "Type of intercurrent event."),
            occurrence,
            available,
        )
    if recurrent:
        fields["recurrent_events.parquet"] = (
            identifier,
            ("outcome_id", "identifier", "Clinical endpoint identifier."),
            (
                "occurrence_time",
                calendar,
                "Date of the event, or of the participant's current observation end, according to event; "
                "subtract enrollment_day for time since enrollment.",
            ),
            (
                "event",
                "0 or 1",
                "1 denotes a recorded event; 0 marks this participant's current observation end, not an event.",
            ),
        )
    else:
        fields["measurements.parquet"] = (
            identifier,
            ("outcome_id", "identifier", "Clinical endpoint identifier."),
            occurrence,
            available,
            ("attended", "boolean", "Whether the scheduled measurement contact was attended."),
            (
                "observed",
                "boolean",
                "Whether the measurement was obtained; attendance alone does not guarantee observation.",
            ),
            (
                "observed_outcome",
                outcome_unit,
                "Recorded measurement; blank when observed is false. Future assessments are absent.",
            ),
        )
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("file", "field", "unit", "description"))
        for filename, columns in fields.items():
            for column, unit, description in columns:
                if filename == "contacts.parquet":
                    description += " A contact is a surveillance opportunity, not proof of event-free follow-up."
                writer.writerow((f"arms/<arm_id>/{filename}", column, unit, description))
