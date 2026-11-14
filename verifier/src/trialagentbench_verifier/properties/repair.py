"""Apply released transfer rules without consulting the grading answer."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass

import pandas as pd

from trialagentbench_verifier.properties.inputs import Dataset

# Decode the released row-conflict markers before comparing evidence views.
# These are the six assigned conditions and their declared resolutions.
C5_ACTIONS = {
    "clean_transfer_v1": "accept_transfer_without_change",
    "exact_transfer_duplication_v1": "remove_exact_duplicate_copy",
    "versioned_replacement_v1": "apply_declared_version_precedence",
    "versioned_encoding_drift_v1": "normalize_units_and_apply_versions",
    "declared_encoding_drift_v1": "normalize_declared_encoding",
    "public_crosswalk_referential_repair_v1": "apply_released_crosswalk",
}


@dataclass
class Repair:
    """Resolved tables and independently observed source-issue coordinates."""

    tables: dict[str, pd.DataFrame]
    affected_keys: dict[str, set[tuple[str, ...]]]
    affected_rows: dict[str, int]
    actions: dict[str, str]


def resolve_transfer(dataset: Dataset) -> Repair:
    """Apply every applicable documented normalization and version rule.

    Parameters
    ----------
    dataset
        Parsed source tables and that variant's own dictionary and transfer plan.

    Returns
    -------
    Repair
        Normalized copies, with original issue coordinates retained for checking.

    Raises
    ------
    ValueError
        If units, identifier mapping or logical record identity are ambiguous.
    """
    tables = {name: frame.copy() for name, frame in dataset.tables.items()}
    dictionary = list(csv.DictReader(io.StringIO(dataset.documents.get("data_dictionary.csv", ""))))
    entries = {(row["table"], row["variable"]): row for row in dictionary}
    policy = json.loads(dataset.documents.get("data_integrity_policy.json", '{"domains":[]}'))
    key_map = {row["relative_path"]: row["compound_key_fields"] for row in policy["domains"]}
    affected_keys: dict[str, set[tuple[str, ...]]] = {}
    affected_rows: dict[str, int] = {}
    actions: dict[str, str] = {}
    crosswalk = tables.get("data/raw/identifier_crosswalk.parquet")
    for name, frame in tables.items():
        if "identifier_crosswalk" in name:
            continue
        changed = pd.Series(False, index=frame.index)
        original = frame.copy()
        mapped_any = False
        hours_any = False
        if crosswalk is not None and "participant_id" in frame:
            if crosswalk.source_identifier.duplicated().any():
                raise ValueError("Released identifier crosswalk is not a function")
            mapping = crosswalk.set_index("source_identifier").current_identifier.to_dict()
            mapped = frame.participant_id.map(mapping)
            changed |= mapped.notna()
            mapped_any = bool(mapped.notna().any())
            frame["participant_id"] = pd.to_numeric(mapped.where(mapped.notna(), frame.participant_id), errors="raise")
        for column in tuple(frame.columns):
            entry = entries.get((name, column), {})
            if column.endswith("_unit"):
                value = column.removesuffix("_unit")
                if "24 hours per day" not in entry.get("controlled_terminology", ""):
                    raise ValueError(f"No declared time conversion for {name}:{column}")
                if not frame[column].isin(["day", "hour"]).all():
                    raise ValueError("Unrecognized time unit")
                hours = frame[column].eq("hour")
                hours_any |= bool(hours.any())
                frame[value] = pd.to_numeric(frame[value], errors="raise") / frame[column].map(
                    {"day": 1.0, "hour": 24.0}
                )
                changed |= hours
                frame[column] = "day"
        keys = key_map.get(name)
        if "record_version" in frame:
            rule = entries.get((name, "record_version"), {}).get("label", "")
            if keys is None or "highest supplied version is current" not in rule:
                raise ValueError("Version precedence is not declared")
            duplicated = frame.duplicated(keys, keep=False)
            changed |= duplicated
            maximum = frame.groupby(keys, dropna=False).record_version.transform("max")
            current = frame.loc[frame.record_version.eq(maximum)].drop_duplicates()
            if current.duplicated(keys).any():
                raise ValueError("Conflicting records share the maximum version")
            resolved = current
        else:
            duplicated = frame.duplicated(keep=False)
            changed |= duplicated
            resolved = frame.drop_duplicates()
        if keys is not None:
            affected_keys[name] = {
                tuple(str(value) for value in row)
                for row in original.loc[changed, keys].itertuples(index=False, name=None)
            }
            affected_rows[name] = int(changed.sum())
            if not changed.any():
                actions[name] = "accept_transfer_without_change"
            elif mapped_any:
                actions[name] = "apply_released_crosswalk"
            elif "record_version" in frame:
                actions[name] = (
                    "normalize_units_and_apply_versions" if hours_any else "apply_declared_version_precedence"
                )
            elif hours_any:
                actions[name] = "normalize_declared_encoding"
            else:
                actions[name] = "remove_exact_duplicate_copy"
        tables[name] = resolved.reset_index(drop=True)
    return Repair(tables, affected_keys, affected_rows, actions)
