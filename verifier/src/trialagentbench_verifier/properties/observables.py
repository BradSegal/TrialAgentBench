"""Independent participant and endpoint reconstruction from released documents."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from trialagentbench_verifier.properties.inputs import Dataset
from trialagentbench_verifier.properties.repair import Repair, resolve_transfer


@dataclass(frozen=True)
class AuditFlag:
    """Per-dataset audit discrepancy counts, including the marginal Fisher flag."""

    n_validation: int
    n_discordant: int
    active_validation: int
    active_discordant: int
    control_validation: int
    control_discordant: int
    f1: int
    f2_p: float | None


def audit_flag(analysis: pd.DataFrame, endpoint: str) -> AuditFlag:
    """Count routine/reference discordance on observed selected assessments.

    Parameters
    ----------
    analysis
        Reconstructed analysis observations.
    endpoint
        Released endpoint name for non-survival outcomes.

    Returns
    -------
    AuditFlag
        Pooled counts within this dataset, without treating unobserved values as zero.
    """
    routine = "routine_endpoint" if "routine_endpoint" in analysis else endpoint
    required = {routine, "validation_selected", "adjudicated_endpoint"}
    if not required <= set(analysis):
        return AuditFlag(0, 0, 0, 0, 0, 0, 0, None)
    selected = analysis.validation_selected.fillna(False).astype(bool)
    valid = selected & analysis[routine].notna() & analysis.adjudicated_endpoint.notna()
    sample = analysis.loc[valid]
    discordance = sample[routine].ne(sample.adjudicated_endpoint)
    active = sample.assigned_arm.eq("active")
    control = sample.assigned_arm.eq("control")
    na, nc = int(active.sum()), int(control.sum())
    da, dc = int((discordance & active).sum()), int((discordance & control).sum())
    p = float(fisher_exact(np.array([[da, na - da], [dc, nc - dc]], dtype=int)).pvalue) if na and nc else None
    return AuditFlag(len(sample), int(discordance.sum()), na, da, nc, dc, int(discordance.any()), p)


def _source_columns(dataset: Dataset, table: str) -> list[str]:
    rows = csv.DictReader(io.StringIO(dataset.documents["data_dictionary.csv"]))
    return [
        row["variable"]
        for row in rows
        if row["table"] == table
        and not row["variable"].startswith("completion_")
        and row["derivation"] != "source transfer metadata"
    ]


def reconstruct(dataset: Dataset, *, repair: Repair | None = None) -> pd.DataFrame:
    """Reconstruct observations using only the supplied variant's source definitions.

    Parameters
    ----------
    dataset
        Opened variant, never another context's table or dictionary.
    repair
        Optional independently resolved transfer; otherwise resolve all rules.

    Returns
    -------
    pandas.DataFrame
        Participant-level or participant/contact-level analysis observations.
    """
    if "data/analysis.parquet" in dataset.tables:
        return dataset.tables["data/analysis.parquet"].copy()
    tables = (repair or resolve_transfer(dataset)).tables
    roster = tables["data/raw/participant_records.parquet"]
    roster = roster[
        [column for column in _source_columns(dataset, "data/raw/participant_records.parquet") if column in roster]
    ]
    if roster.participant_id.duplicated().any():
        raise ValueError("Participant roster is not unique")
    endpoints = tables.get("data/raw/endpoint_observations.parquet")
    events = tables["data/raw/event_history.parquet"]
    if endpoints is not None:
        source_columns = _source_columns(dataset, "data/raw/endpoint_observations.parquet")
        endpoints = endpoints[[column for column in source_columns if column in endpoints]]
        overlap = set(roster) & set(endpoints) - {"participant_id"}
        if overlap:
            raise ValueError(f"Ambiguous roster/endpoint fields: {sorted(str(x) for x in overlap)}")
        result = roster.merge(endpoints, on="participant_id", how="left", validate="one_to_many")
        if "validation_selected" in endpoints:
            outcome_columns = [
                col
                for col in endpoints
                if col
                not in {
                    "participant_id",
                    "assessment_time",
                    "validation_stratum",
                    "validation_selected",
                    "adjudicated_endpoint",
                }
            ]
            if len(outcome_columns) != 1:
                raise ValueError("Endpoint dictionary does not identify one routine endpoint")
            result["endpoint_observed"] = result[outcome_columns[0]].notna()
        return result
    assessments = events.loc[events.event_kind.eq("endpoint_assessment")]
    if not assessments.empty:
        routine = assessments.loc[assessments.event_id.str.startswith("routine_")]
        reference = assessments.loc[assessments.event_id.str.startswith("adjudicated_")]
        if routine.participant_id.duplicated().any() or reference.participant_id.duplicated().any():
            raise ValueError("Ambiguous validation assessment")
        result = roster.merge(
            routine[["participant_id", "event_by_contact"]].rename(columns={"event_by_contact": "routine_endpoint"}),
            on="participant_id",
            how="left",
            validate="one_to_one",
        )
        result = result.merge(
            reference[["participant_id", "event_by_contact"]].rename(
                columns={"event_by_contact": "adjudicated_endpoint"}
            ),
            on="participant_id",
            how="left",
            validate="one_to_one",
        )
        result["validation_selected"] = result.participant_id.isin(reference.participant_id)
        result["treatment_discontinuation_time"] = _event_times(
            events, result.participant_id, "treatment_discontinuation"
        )
        return result
    terminal = events.loc[events.event_kind.isin(["endpoint", "follow_up"])]
    if terminal.participant_id.duplicated().any():
        raise ValueError("Ambiguous terminal record")
    result = roster.merge(
        terminal[["participant_id", "occurrence_time", "event_id", "event_kind"]],
        on="participant_id",
        how="left",
        validate="one_to_one",
    )
    protocol = dataset.documents["protocol.md"]
    horizon = float(re.search(r"Follow-up: ([\d.]+) days", protocol)[1])  # type: ignore[index]
    result["entry_time"] = result.enrollment_day
    result["administrative_follow_up"] = (
        np.minimum(horizon, tables["data/monitoring_results.parquet"].information_time.iloc[-1] - result.entry_time)
        if "data/monitoring_results.parquet" in tables
        else horizon
    )
    result["observed_time"] = pd.to_numeric(result.pop("occurrence_time"))
    result["event_observed"] = result.event_kind.eq("endpoint")
    result["primary_endpoint"] = result.event_observed
    result["event_type"] = np.where(
        result.event_kind.eq("endpoint"),
        "primary_event",
        np.where(result.event_id.eq("administrative_follow_up_end"), "administrative_censoring", "censoring"),
    )
    result = result.drop(columns="event_id")
    result = result.drop(columns="event_kind")
    result["treatment_discontinuation_time"] = _event_times(events, result.participant_id, "treatment_discontinuation")
    lower = re.search(r"entry time at least ([\d.]+) days", protocol)
    upper = re.search(r"entry time less than ([\d.]+) days", protocol)
    bmi = re.search(r"Baseline BMI threshold: .*? at least ([\d.]+) kg/m2", protocol)
    if lower and upper:
        result["reference_population_member"] = result.entry_time.ge(float(lower[1])) & result.entry_time.lt(
            float(upper[1])
        )
    elif bmi:
        rows = csv.DictReader(io.StringIO(dataset.documents["data_dictionary.csv"]))
        field = next(row["variable"] for row in rows if "Body mass index (BMI) at randomization" in row["label"])
        result["reference_population_member"] = result[field].ge(float(bmi[1]))
    return result


def _event_times(events: pd.DataFrame, participants: pd.Series[Any], event: str) -> np.ndarray[Any, Any]:
    subset = events.loc[events.event_kind.eq(event)].groupby("participant_id").occurrence_time.min()
    return participants.map(subset).to_numpy()


def differences(
    expected: pd.DataFrame, actual: pd.DataFrame, columns: list[str] | None = None
) -> list[dict[str, Any]]:
    """Compare participant/contact coordinates with the declared float tolerance.

    Parameters
    ----------
    expected, actual
        Independently obtained observation tables.
    columns
        Explicit fields; by default all expected fields are required.

    Returns
    -------
    list of dict
        Field-level differences with bounded example coordinates.
    """
    keys = ["participant_id"] + (["occurrence_time"] if "outcome_id" in expected else [])
    failures: list[dict[str, Any]] = []
    if not set(keys) <= set(actual):
        return [{"field": "coordinates", "reason": "missing participant/contact keys"}]
    if expected.duplicated(keys).any() or actual.duplicated(keys).any():
        return [{"field": "coordinates", "reason": "duplicate participant/contact keys"}]
    left = expected.set_index(keys).sort_index()
    right = actual.set_index(keys).sort_index()
    coordinate_equal = left.index.equals(right.index)
    if not coordinate_equal and len(keys) == 2 and len(left) == len(right):
        ids_equal = left.index.get_level_values(0).equals(right.index.get_level_values(0))
        a_times = left.index.get_level_values(1).to_numpy(dtype=float)
        b_times = right.index.get_level_values(1).to_numpy(dtype=float)
        coordinate_equal = ids_equal and bool(
            (np.abs(a_times - b_times) <= 1e-12 * np.maximum(1.0, np.abs(a_times))).all()
        )
        if coordinate_equal:
            right.index = left.index
    if not coordinate_equal:
        return [
            {
                "field": "coordinates",
                "reason": "participant/contact sets differ",
                "expected": len(left),
                "actual": len(right),
            }
        ]
    for column in columns or list(left.columns):
        if column in keys:
            continue
        if column not in right:
            failures.append({"field": column, "reason": "missing reconstructed field"})
            continue
        a, b = left[column], right[column]
        both_missing = a.isna() & b.isna()
        if pd.api.types.is_numeric_dtype(a) or pd.api.types.is_bool_dtype(a):
            av = pd.to_numeric(a, errors="raise").to_numpy(dtype=float, na_value=np.nan)
            bv = pd.to_numeric(b, errors="raise").to_numpy(dtype=float, na_value=np.nan)
            equal = (np.abs(av - bv) <= 1e-12 * np.maximum(1.0, np.abs(av))) | both_missing.to_numpy()
        else:
            equal = (a.eq(b).fillna(False) | both_missing).to_numpy()
        if not equal.all():
            bad = np.flatnonzero(~equal)
            failures.append(
                {
                    "field": column,
                    "n": len(bad),
                    "examples": [str(left.index[i]) for i in bad[:3]],
                    "expected": [str(a.iloc[i]) for i in bad[:3]],
                    "actual": [str(b.iloc[i]) for i in bad[:3]],
                }
            )
    return failures
