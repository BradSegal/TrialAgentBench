"""Source-data disposition vocabulary shared by TrialEval runtime and grading."""

from __future__ import annotations

from typing import Final

INTEGRITY_DISPOSITIONS_BY_ISSUE_V1: Final[dict[str, frozenset[str]]] = {
    "no_material_issue": frozenset({"accept_without_change"}),
    "exact_duplicate_records": frozenset({"deduplicate_exact_records"}),
    "conflicting_same_key_records": frozenset({"quarantine_conflicting_records", "retain_nonmaterial_source_records"}),
    "superseded_records": frozenset({"apply_documented_precedence"}),
    "superseded_records_and_time_units": frozenset({"normalize_time_units_and_select_current_records"}),
    "encoding_nonconformance": frozenset({"normalize_declared_encoding"}),
    "identifier_reconciliation": frozenset({"apply_documented_identifier_crosswalk"}),
    "unresolved_source_values": frozenset({"quarantine_unresolved_records", "retain_nonmaterial_source_records"}),
}
INTEGRITY_STATUS_BY_DISPOSITION_V1: Final[dict[str, str]] = {
    "accept_without_change": "clean",
    "deduplicate_exact_records": "repaired",
    "quarantine_conflicting_records": "qualified_exclusion",
    "retain_nonmaterial_source_records": "unresolved",
    "apply_documented_precedence": "repaired",
    "normalize_time_units_and_select_current_records": "repaired",
    "normalize_declared_encoding": "repaired",
    "apply_documented_identifier_crosswalk": "repaired",
    "quarantine_unresolved_records": "qualified_exclusion",
}


def validate_data_integrity_disposition_v1(
    *,
    issue_type: str,
    disposition: str,
    resolution_status: str,
) -> None:
    """Reject a source-data disposition that is incoherent with its finding."""

    permitted = INTEGRITY_DISPOSITIONS_BY_ISSUE_V1.get(issue_type)
    if permitted is None:
        raise ValueError(f"unsupported issue_type: {issue_type!r}")
    if disposition not in INTEGRITY_STATUS_BY_DISPOSITION_V1:
        raise ValueError(f"unsupported disposition: {disposition!r}")
    if disposition not in permitted:
        raise ValueError(f"data-integrity disposition {disposition!r} is incompatible with issue type {issue_type!r}")
    expected_status = INTEGRITY_STATUS_BY_DISPOSITION_V1[disposition]
    if resolution_status != expected_status:
        raise ValueError(f"data-integrity disposition {disposition!r} requires resolution status {expected_status!r}")


__all__ = [
    "INTEGRITY_DISPOSITIONS_BY_ISSUE_V1",
    "INTEGRITY_STATUS_BY_DISPOSITION_V1",
    "validate_data_integrity_disposition_v1",
]
