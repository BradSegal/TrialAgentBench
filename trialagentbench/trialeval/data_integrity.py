"""Condition-neutral source-data quality utilities for TrialEval."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import struct
from dataclasses import asdict, dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any, TypeAlias, cast

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from .data_integrity_rules import validate_data_integrity_disposition_v1

CanonicalScalarV1: TypeAlias = list[str | bool | None]

DATA_INTEGRITY_POLICY_FILENAME_V1 = "data_integrity_policy.json"
EXACT_TRANSPORT_DUPLICATE_MULTIPLICITY_V1 = 2

_MARKER_ISSUES = {
    "superseded-record::": "superseded_records",
    "hexfloat::": "encoding_nonconformance",
    "prior-subject::": "identifier_reconciliation",
    "source-query-required": "unresolved_source_values",
}
_SOURCE_VALUE_STATUS_FIELD = "source_value_status"
_SOURCE_VALUE_QUERY_OPEN = "source_query_open"
_ENDPOINT_OBSERVATION_FIELDS = frozenset(
    {
        "participant_id",
        "occurrence_time",
        "assessment_time",
        "observation_time",
        "documentation_time",
        "availability_time",
        "attended",
        "measurement_obtained",
        "observed",
        "censoring_status",
        "outcome_id",
        _SOURCE_VALUE_STATUS_FIELD,
    }
)

__all__ = [
    "DATA_INTEGRITY_POLICY_FILENAME_V1",
    "EXACT_TRANSPORT_DUPLICATE_MULTIPLICITY_V1",
    "audit_source_data_v1",
    "canonical_domain_content_sha256_v1",
    "canonical_json_array_bytes_v1",
    "canonical_rows_v1",
    "record_data_disposition_v1",
    "repair_exact_transport_row_duplication_v1",
]


@dataclass(frozen=True)
class DataDispositionRecordV1:
    """Participant-reported source-data disposition."""

    issue_type: str
    affected_domain: str
    compound_key_fields: tuple[str, ...]
    observed_issue_group_count: int
    observed_affected_row_count: int
    disposition: str
    resolution_status: str
    resolved_domain_content_sha256: str | None
    source_domain_file_sha256: str | None
    analysis_input_data_checksum: str


def canonical_json_array_bytes_v1(value: object) -> bytes:
    """Encode one canonical JSON array value as UTF-8 bytes."""

    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _is_missing(value: object) -> bool:
    missing = pd.isna(cast(Any, value))
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    raise TypeError("canonical scalars cannot contain array-valued cells")


def _canonical_scalar(value: object, dtype: object) -> CanonicalScalarV1:
    typed_dtype = cast(Any, dtype)
    typed_value = cast(Any, value)
    if _is_missing(value):
        return ["null", None]
    if isinstance(dtype, pd.CategoricalDtype):
        return _canonical_scalar(value, pd.Series([value]).dtype)
    if ptypes.is_bool_dtype(typed_dtype) or isinstance(value, (bool, np.bool_)):
        return ["boolean", bool(value)]
    if ptypes.is_unsigned_integer_dtype(typed_dtype):
        return [str(dtype).lower(), str(int(typed_value))]
    if ptypes.is_integer_dtype(typed_dtype) or isinstance(value, (int, np.integer)):
        tag = str(dtype).lower() if ptypes.is_integer_dtype(typed_dtype) else "int64"
        return [tag, str(int(typed_value))]
    if ptypes.is_float_dtype(typed_dtype) or isinstance(value, (float, np.floating)):
        number = float(typed_value)
        if not math.isfinite(number):
            raise ValueError("canonical scalars require finite floating-point values")
        if str(dtype).lower() == "float32" or isinstance(value, np.float32):
            return ["float32", struct.pack(">f", number).hex()]
        return ["float64", struct.pack(">d", number).hex()]
    if ptypes.is_datetime64_any_dtype(typed_dtype) or isinstance(value, (pd.Timestamp, datetime)):
        timestamp = pd.Timestamp(typed_value)
        tag = "timestamp[ns]" if timestamp.tzinfo is None else f"timestamp[ns,{timestamp.tzinfo}]"
        return [tag, str(int(timestamp.value))]
    if isinstance(value, date):
        return ["date32", str((value - date(1970, 1, 1)).days)]
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("canonical scalars require finite decimal values")
        exponent = int(value.as_tuple().exponent)
        scale = max(0, -exponent)
        return [f"decimal:{len(value.as_tuple().digits)}:{scale}", str(int(value.scaleb(scale)))]
    if isinstance(value, (bytes, bytearray, memoryview)):
        encoded = base64.urlsafe_b64encode(bytes(value)).decode("ascii").rstrip("=")
        return ["binary", encoded]
    if isinstance(value, str):
        value.encode("utf-8", errors="strict")
        return ["string", value]
    raise TypeError(f"unsupported canonical scalar type: {type(value).__name__}")


def _validate_columns(frame: pd.DataFrame, key_fields: tuple[str, ...]) -> None:
    if frame.empty:
        raise ValueError("the declared source domain must contain at least one row")
    if len(set(str(column) for column in frame.columns)) != len(frame.columns):
        raise ValueError("the declared source domain requires unique column names")
    missing = tuple(field for field in key_fields if field not in frame.columns)
    if missing:
        raise ValueError(f"the source domain is missing compound-key fields: {missing!r}")
    for field in key_fields:
        if frame[field].map(_is_missing).any():
            raise ValueError(f"the compound-key field contains missing values: {field}")


def canonical_rows_v1(frame: pd.DataFrame, fields: tuple[str, ...]) -> tuple[bytes, ...]:
    """Encode selected typed columns in one tabular pass."""

    _validate_columns(frame, fields)
    dtypes = tuple(frame[field].dtype for field in fields)
    return tuple(
        canonical_json_array_bytes_v1(
            tuple(_canonical_scalar(value, dtype) for value, dtype in zip(values, dtypes, strict=True))
        )
        for values in frame.loc[:, list(fields)].itertuples(index=False, name=None)
    )


def _canonical_records(frame: pd.DataFrame, *, key_fields: tuple[str, ...]) -> tuple[tuple[bytes, bytes], ...]:
    columns = tuple(str(column) for column in frame.columns)
    positions = {column: index for index, column in enumerate(columns)}
    key_positions = tuple(positions[field] for field in key_fields)
    dtypes = tuple(frame[column].dtype for column in columns)
    records: list[tuple[bytes, bytes]] = []
    for values in frame.itertuples(index=False, name=None):
        encoded = tuple(_canonical_scalar(value, dtype) for value, dtype in zip(values, dtypes, strict=True))
        key = canonical_json_array_bytes_v1(tuple(encoded[position] for position in key_positions))
        records.append((key, canonical_json_array_bytes_v1(encoded)))
    return tuple(records)


def canonical_domain_content_sha256_v1(frame: pd.DataFrame, *, key_fields: tuple[str, ...]) -> str:
    """Return the canonical cross-format and cross-layout checksum of a typed domain."""

    _validate_columns(frame, key_fields)
    canonical_columns = tuple(sorted(str(column) for column in frame.columns))
    content = b"\n".join(
        payload
        for _key, payload in sorted(
            _canonical_records(
                frame.loc[:, list(canonical_columns)],
                key_fields=key_fields,
            )
        )
    )
    return hashlib.sha256(content).hexdigest()


def repair_exact_transport_row_duplication_v1(
    frame: pd.DataFrame,
    *,
    key_fields: tuple[str, ...],
) -> tuple[pd.DataFrame, int, int]:
    """Remove exact duplicate copies and fail on conflicting same-key records."""

    _validate_columns(frame, key_fields)
    groups: dict[bytes, list[int]] = {}
    records = _canonical_records(frame, key_fields=key_fields)
    for index, (key, _payload) in enumerate(records):
        groups.setdefault(key, []).append(index)
    drop_indexes: list[int] = []
    duplicate_groups = 0
    for indexes in groups.values():
        if len(indexes) == 1:
            continue
        if len(indexes) != EXACT_TRANSPORT_DUPLICATE_MULTIPLICITY_V1:
            raise ValueError("compound-key multiplicity must be one or two")
        if len({records[index][1] for index in indexes}) != 1:
            raise ValueError("same-key payloads are not identical")
        duplicate_groups += 1
        drop_indexes.append(indexes[1])
    if duplicate_groups == 0:
        raise ValueError("no exact transfer duplicate was detected")
    return frame.drop(index=drop_indexes).reset_index(drop=True), duplicate_groups, len(drop_indexes)


def _load_plan(root: Path) -> tuple[dict[str, object], ...]:
    plan_path = root / DATA_INTEGRITY_POLICY_FILENAME_V1
    if not plan_path.is_file():
        raise FileNotFoundError(f"source-data quality plan is missing: {plan_path}")
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_id") != "ariadne.trialeval.data_quality_plan/v1":
        raise ValueError("source-data quality plan has an unsupported schema")
    domains = payload.get("domains")
    if not isinstance(domains, list) or not domains or not all(isinstance(value, dict) for value in domains):
        raise ValueError("source-data quality plan requires domain definitions")
    return tuple(cast(dict[str, object], value) for value in domains)


def _irreconcilable_missing_cells(frame: pd.DataFrame, *, relative_path: str) -> tuple[str | None, tuple[int, ...]]:
    """Find source values that are missing despite an observed record."""

    if _SOURCE_VALUE_STATUS_FIELD in frame:
        statuses = frame[_SOURCE_VALUE_STATUS_FIELD].astype("string")
        unexpected = set(statuses.dropna().unique()) - {"verified", _SOURCE_VALUE_QUERY_OPEN}
        if unexpected:
            raise ValueError(f"source-data verification status contains unsupported values: {sorted(unexpected)!r}")
        indexes = tuple(int(index) for index in frame.index[statuses.eq(_SOURCE_VALUE_QUERY_OPEN)])
        if indexes:
            return _SOURCE_VALUE_STATUS_FIELD, indexes
        return None, ()
    if relative_path == "data/raw/event_history.parquet":
        field = "occurrence_time"
        if field not in frame:
            return None, ()
        return field, tuple(int(index) for index in frame.index[frame[field].isna()])
    if relative_path != "data/raw/endpoint_observations.parquet":
        return None, ()
    endpoint_field: str | None = next(
        (
            str(column)
            for column in frame.columns
            if str(column) not in _ENDPOINT_OBSERVATION_FIELDS and ptypes.is_numeric_dtype(frame[column].dtype)
        ),
        None,
    )
    if endpoint_field is None:
        return None, ()
    missing = frame[endpoint_field].isna()
    if "observed" in frame:
        missing &= frame["observed"].fillna(False).astype(bool)
    elif "measurement_obtained" in frame:
        missing &= frame["measurement_obtained"].fillna(False).astype(bool)
    return endpoint_field, tuple(int(index) for index in frame.index[missing])


def audit_source_data_v1(*, root: str | Path = ".") -> dict[str, object]:
    """Report observable QC findings without selecting a disposition."""

    item_root = Path(root)
    observations: list[dict[str, object]] = []
    domain_records: list[dict[str, object]] = []
    for domain in _load_plan(item_root):
        relative_path = domain.get("relative_path")
        key_fields = domain.get("compound_key_fields")
        if not isinstance(relative_path, str) or not isinstance(key_fields, list):
            raise ValueError("source-data quality domain fields are invalid")
        keys = tuple(str(value) for value in key_fields)
        path = item_root.joinpath(*PurePosixPath(relative_path).parts)
        frame = pd.read_parquet(path)
        records = _canonical_records(frame, key_fields=keys)
        grouped: dict[bytes, list[int]] = {}
        for index, (key, _payload) in enumerate(records):
            grouped.setdefault(key, []).append(index)
        for key, indexes in grouped.items():
            if len(indexes) < 2:
                continue
            payload_count = len({records[index][1] for index in indexes})
            observations.append(
                {
                    "source_domain": relative_path,
                    "compound_key": key.decode("utf-8"),
                    "field_name": None,
                    "issue_type": (
                        "exact_duplicate_records" if payload_count == 1 else "conflicting_same_key_records"
                    ),
                    "affected_row_count": len(indexes),
                }
            )
        for field in frame.columns:
            if not (ptypes.is_string_dtype(frame[field].dtype) or ptypes.is_object_dtype(frame[field].dtype)):
                continue
            values = frame[field].astype("string")
            for prefix, issue_type in _MARKER_ISSUES.items():
                matched = values.str.startswith(prefix, na=False)
                for index in frame.index[matched]:
                    observations.append(
                        {
                            "source_domain": relative_path,
                            "compound_key": records[int(index)][0].decode("utf-8"),
                            "field_name": str(field),
                            "issue_type": issue_type,
                            "affected_row_count": 1,
                        }
                    )
        missing_field, missing_indexes = _irreconcilable_missing_cells(
            frame,
            relative_path=relative_path,
        )
        for index in missing_indexes:
            observations.append(
                {
                    "source_domain": relative_path,
                    "compound_key": records[index][0].decode("utf-8"),
                    "field_name": missing_field,
                    "issue_type": "unresolved_source_values",
                    "affected_row_count": 1,
                }
            )
        domain_records.append(
            {
                "source_domain": relative_path,
                "compound_key_fields": keys,
                "row_count": len(frame),
                "content_sha256": canonical_domain_content_sha256_v1(frame, key_fields=keys),
            }
        )
    return {"domains": domain_records, "observation_count": len(observations), "observations": observations}


def record_data_disposition_v1(
    *,
    analysis_input_path: str,
    affected_domain: str,
    compound_key_fields: list[str],
    issue_type: str,
    observed_issue_group_count: int,
    observed_affected_row_count: int,
    disposition: str,
    resolution_status: str,
    root: str | Path = ".",
) -> dict[str, object]:
    """Record an analyst-selected disposition without adjudicating its correctness."""

    validate_data_integrity_disposition_v1(
        issue_type=issue_type,
        disposition=disposition,
        resolution_status=resolution_status,
    )
    if observed_issue_group_count < 0 or observed_affected_row_count < 0:
        raise ValueError("observed issue counts must be nonnegative")
    domains = _load_plan(Path(root))
    declared = next((domain for domain in domains if domain.get("relative_path") == affected_domain), None)
    if declared is None or declared.get("compound_key_fields") != compound_key_fields:
        raise ValueError("reported domain identity differs from the source-data quality plan")
    relative_input = PurePosixPath(analysis_input_path)
    if (
        relative_input.is_absolute()
        or ".." in relative_input.parts
        or not relative_input.parts
        or relative_input.parts[0] != "scratch"
        or relative_input.suffix != ".parquet"
    ):
        raise ValueError("analysis_input_path must identify one Parquet file under scratch/")
    candidate_path = Path(root).joinpath(*relative_input.parts)
    if not candidate_path.is_file():
        raise FileNotFoundError(f"analysis input is missing: {candidate_path}")
    retained_nonmaterial = disposition == "retain_nonmaterial_source_records"
    if retained_nonmaterial:
        source_path = Path(root).joinpath(*PurePosixPath(affected_domain).parts)
        if not source_path.is_file():
            raise FileNotFoundError(f"affected source domain is missing: {source_path}")
        resolved_checksum = None
        source_checksum = hashlib.sha256(source_path.read_bytes()).hexdigest()
        analysis_checksum = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    else:
        resolved_checksum = canonical_domain_content_sha256_v1(
            pd.read_parquet(candidate_path),
            key_fields=tuple(compound_key_fields),
        )
        source_checksum = None
        analysis_checksum = resolved_checksum
    record = DataDispositionRecordV1(
        issue_type=issue_type,
        affected_domain=affected_domain,
        compound_key_fields=tuple(compound_key_fields),
        observed_issue_group_count=observed_issue_group_count,
        observed_affected_row_count=observed_affected_row_count,
        disposition=disposition,
        resolution_status=resolution_status,
        resolved_domain_content_sha256=resolved_checksum,
        source_domain_file_sha256=source_checksum,
        analysis_input_data_checksum=analysis_checksum,
    )
    return {"analysis_input_path": relative_input.as_posix(), "data_integrity_record": asdict(record)}
