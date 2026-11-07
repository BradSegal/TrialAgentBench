"""Source-data quality utility and runtime tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from trialagentbench.trialeval.agent import _build_system_prompt, _get_tools
from trialagentbench.trialeval.data_integrity import (
    audit_source_data_v1,
    canonical_domain_content_sha256_v1,
    record_data_disposition_v1,
    repair_exact_transport_row_duplication_v1,
)
from trialagentbench.trialeval.schema import BenchmarkItem


def _plan(*, task_id: str = "TASK001") -> dict[str, object]:
    return {
        "schema_id": "ariadne.trialeval.data_quality_plan/v1",
        "task_id": task_id,
        "domains": [
            {
                "relative_path": "data/raw/events.parquet",
                "compound_key_fields": ["USUBJID", "EVENT_DY"],
            }
        ],
        "primary_analysis_domain": "data/raw/events.parquet",
    }


def _clean_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "USUBJID": pd.Series(["S001", "S002"], dtype="string"),
            "EVENT_DY": pd.Series([10, 20], dtype="int64"),
            "EVENT": pd.Series([True, False], dtype="bool"),
            "VALUE": pd.Series([1.25, 2.5], dtype="float64"),
        }
    )


def _write_item(root: Path, frame: pd.DataFrame, *, task_id: str = "TASK001") -> None:
    affected = root / "data" / "raw" / "events.parquet"
    affected.parent.mkdir(parents=True)
    frame.to_parquet(affected, index=False)
    (root / "data_integrity_policy.json").write_text(
        json.dumps(_plan(task_id=task_id), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _item(root: Path, *, task_id: str = "TASK001") -> BenchmarkItem:
    return BenchmarkItem(
        item_id=task_id,
        task_id=task_id,
        trial_name=task_id,
        design_tier="undisclosed",
        design_subtype="individual_randomized",
        assumption_tier="undisclosed",
        context_tier="C5",
        visible_dir=root,
        data_dir=root / "data" / "raw",
        raw_data_dir=root / "data" / "raw",
        data_preparation="raw_domains_declared_defect",
        task={},
    )


def test_audit_distinguishes_clean_exact_and_conflicting_records(tmp_path: Path) -> None:
    clean = _clean_frame()
    _write_item(tmp_path, clean)
    assert audit_source_data_v1(root=tmp_path)["observations"] == []

    exact = pd.concat([clean, clean.iloc[[0]]], ignore_index=True)
    exact.to_parquet(tmp_path / "data" / "raw" / "events.parquet", index=False)
    exact_audit = audit_source_data_v1(root=tmp_path)
    assert {row["issue_type"] for row in exact_audit["observations"]} == {"exact_duplicate_records"}
    assert exact_audit["observation_count"] == 1
    assert exact_audit["observations"][0]["affected_row_count"] == 2

    conflict = exact.copy()
    conflict.loc[2, "VALUE"] = 9.0
    conflict.to_parquet(tmp_path / "data" / "raw" / "events.parquet", index=False)
    conflict_audit = audit_source_data_v1(root=tmp_path)
    assert {row["issue_type"] for row in conflict_audit["observations"]} == {"conflicting_same_key_records"}


def test_audit_reports_transfer_markers_without_prescribing_disposition(tmp_path: Path) -> None:
    frame = _clean_frame().astype({"VALUE": "string"})
    frame.loc[0, "VALUE"] = "source-query-required::1.25"
    _write_item(tmp_path, frame)

    result = audit_source_data_v1(root=tmp_path)

    assert result["observation_count"] == 1
    assert result["observations"][0]["issue_type"] == "unresolved_source_values"
    assert "disposition" not in result["observations"][0]


def test_audit_reports_missing_value_on_observed_endpoint_record(tmp_path: Path) -> None:
    """Expose unresolved numeric values without changing a column's type."""

    data_root = tmp_path / "data" / "raw"
    data_root.mkdir(parents=True)
    pd.DataFrame(
        {
            "participant_id": pd.Series([1, 2, 3], dtype="int64"),
            "observed": pd.Series([True, True, False], dtype="bool"),
            "endpoint_score": pd.Series([4.0, None, None], dtype="float64"),
            "source_value_status": pd.Series(
                ["verified", "source_query_open", "verified"],
                dtype="object",
            ),
        }
    ).to_parquet(data_root / "endpoint_observations.parquet", index=False)
    (tmp_path / "data_integrity_policy.json").write_text(
        json.dumps(
            {
                "schema_id": "ariadne.trialeval.data_quality_plan/v1",
                "task_id": "TASK001",
                "domains": [
                    {
                        "relative_path": "data/raw/endpoint_observations.parquet",
                        "compound_key_fields": ["participant_id"],
                    }
                ],
                "primary_analysis_domain": "data/raw/endpoint_observations.parquet",
            }
        ),
        encoding="utf-8",
    )

    result = audit_source_data_v1(root=tmp_path)

    assert result["observation_count"] == 1
    assert result["observations"][0]["issue_type"] == "unresolved_source_values"
    assert result["observations"][0]["field_name"] == "source_value_status"


def test_disposition_record_checks_identity_and_disposition_coherence(tmp_path: Path) -> None:
    clean = _clean_frame()
    _write_item(tmp_path, clean)
    analysis_path = tmp_path / "scratch" / "events.parquet"
    analysis_path.parent.mkdir()
    clean.to_parquet(analysis_path, index=False)
    checksum = canonical_domain_content_sha256_v1(clean, key_fields=("USUBJID", "EVENT_DY"))
    assert (
        canonical_domain_content_sha256_v1(
            clean.loc[:, list(reversed(clean.columns))],
            key_fields=("USUBJID", "EVENT_DY"),
        )
        == checksum
    )

    result = record_data_disposition_v1(
        root=tmp_path,
        analysis_input_path="scratch/events.parquet",
        affected_domain="data/raw/events.parquet",
        compound_key_fields=["USUBJID", "EVENT_DY"],
        issue_type="no_material_issue",
        observed_issue_group_count=0,
        observed_affected_row_count=0,
        disposition="accept_without_change",
        resolution_status="clean",
    )

    assert result["data_integrity_record"]["analysis_input_data_checksum"] == checksum
    assert result["data_integrity_record"]["issue_type"] == "no_material_issue"


def test_disposition_record_rejects_incoherent_issue_action(tmp_path: Path) -> None:
    """The public helper and submission schema must reject the same invalid action."""

    clean = _clean_frame()
    _write_item(tmp_path, clean)
    analysis_path = tmp_path / "scratch" / "events.parquet"
    analysis_path.parent.mkdir()
    clean.to_parquet(analysis_path, index=False)

    with pytest.raises(ValueError, match="incompatible with issue type"):
        record_data_disposition_v1(
            root=tmp_path,
            analysis_input_path="scratch/events.parquet",
            affected_domain="data/raw/events.parquet",
            compound_key_fields=["USUBJID", "EVENT_DY"],
            issue_type="unresolved_source_values",
            observed_issue_group_count=1,
            observed_affected_row_count=1,
            disposition="accept_without_change",
            resolution_status="clean",
        )


def test_disposition_record_can_retain_demonstrably_nonmaterial_unresolved_values(tmp_path: Path) -> None:
    """Unresolved source values may remain only under the explicit nonmaterial action."""

    source = _clean_frame().astype({"VALUE": "string"})
    source.loc[0, "VALUE"] = "source-query-required"
    _write_item(tmp_path, source)
    analysis_path = tmp_path / "scratch" / "analysis.parquet"
    analysis_path.parent.mkdir()
    pd.DataFrame({"USUBJID": ["S001", "S002"], "result": [1.0, 2.0]}).to_parquet(
        analysis_path,
        index=False,
    )

    result = record_data_disposition_v1(
        root=tmp_path,
        analysis_input_path="scratch/analysis.parquet",
        affected_domain="data/raw/events.parquet",
        compound_key_fields=["USUBJID", "EVENT_DY"],
        issue_type="unresolved_source_values",
        observed_issue_group_count=1,
        observed_affected_row_count=1,
        disposition="retain_nonmaterial_source_records",
        resolution_status="unresolved",
    )

    assert result["data_integrity_record"]["resolved_domain_content_sha256"] is None


def test_disposition_record_rejects_an_undeclared_domain(tmp_path: Path) -> None:
    _write_item(tmp_path, _clean_frame())
    analysis_path = tmp_path / "scratch" / "events.parquet"
    analysis_path.parent.mkdir()
    _clean_frame().to_parquet(analysis_path, index=False)

    with pytest.raises(ValueError, match="differs from the source-data quality plan"):
        record_data_disposition_v1(
            root=tmp_path,
            analysis_input_path="scratch/events.parquet",
            affected_domain="data/raw/other.parquet",
            compound_key_fields=["USUBJID", "EVENT_DY"],
            issue_type="no_material_issue",
            observed_issue_group_count=0,
            observed_affected_row_count=0,
            disposition="accept_without_change",
            resolution_status="clean",
        )


def test_disposition_record_distinguishes_preserved_source_from_analysis_input(tmp_path: Path) -> None:
    """Preserve nonmaterial conflicts without claiming that the source was repaired."""

    clean = _clean_frame()
    conflict = pd.concat([clean, clean.iloc[[0]].assign(VALUE=9.0)], ignore_index=True)
    _write_item(tmp_path, conflict)
    analysis_path = tmp_path / "scratch" / "analysis.parquet"
    analysis_path.parent.mkdir()
    pd.DataFrame({"USUBJID": ["S001", "S002"], "result": [1.0, 2.0]}).to_parquet(
        analysis_path,
        index=False,
    )

    result = record_data_disposition_v1(
        root=tmp_path,
        analysis_input_path="scratch/analysis.parquet",
        affected_domain="data/raw/events.parquet",
        compound_key_fields=["USUBJID", "EVENT_DY"],
        issue_type="conflicting_same_key_records",
        observed_issue_group_count=1,
        observed_affected_row_count=2,
        disposition="retain_nonmaterial_source_records",
        resolution_status="unresolved",
    )
    record = result["data_integrity_record"]

    assert record["resolved_domain_content_sha256"] is None
    assert (
        record["source_domain_file_sha256"]
        == hashlib.sha256((tmp_path / "data" / "raw" / "events.parquet").read_bytes()).hexdigest()
    )
    assert record["analysis_input_data_checksum"] == hashlib.sha256(analysis_path.read_bytes()).hexdigest()


def test_disposition_record_rejects_false_resolution_of_preserved_conflicts(tmp_path: Path) -> None:
    """Prevent the retained-source disposition from claiming a repaired source."""

    _write_item(tmp_path, _clean_frame())
    analysis_path = tmp_path / "scratch" / "analysis.parquet"
    analysis_path.parent.mkdir()
    _clean_frame().to_parquet(analysis_path, index=False)

    with pytest.raises(ValueError, match="requires resolution status 'unresolved'"):
        record_data_disposition_v1(
            root=tmp_path,
            analysis_input_path="scratch/analysis.parquet",
            affected_domain="data/raw/events.parquet",
            compound_key_fields=["USUBJID", "EVENT_DY"],
            issue_type="conflicting_same_key_records",
            observed_issue_group_count=1,
            observed_affected_row_count=2,
            disposition="retain_nonmaterial_source_records",
            resolution_status="repaired",
        )


def test_exact_duplicate_helper_rejects_conflicting_same_key_rows() -> None:
    frame = pd.DataFrame({"USUBJID": ["S001", "S001"], "EVENT_DY": [10, 10], "VALUE": [1.0, 2.0]})
    with pytest.raises(ValueError, match="same-key payloads are not identical"):
        repair_exact_transport_row_duplication_v1(frame, key_fields=("USUBJID", "EVENT_DY"))


def test_prompt_does_not_disclose_the_data_integrity_workflow(tmp_path: Path) -> None:
    _write_item(tmp_path, _clean_frame())

    prompt = _build_system_prompt(_item(tmp_path))

    assert "audit_source_data" not in prompt
    assert "affected record count" not in prompt
    assert "declares a record-integrity incident" not in prompt
    assert "required action" not in prompt.lower()


def test_prompt_rejects_a_plan_for_another_task(tmp_path: Path) -> None:
    _write_item(tmp_path, _clean_frame(), task_id="TASK001")
    with pytest.raises(ValueError, match="does not match the selected task"):
        _build_system_prompt(_item(tmp_path, task_id="TASK002"))


def test_analysis_tools_do_not_offer_a_source_correction_shortcut() -> None:
    names = {tool["function"]["name"] for tool in _get_tools()}
    assert "audit_source_data" not in names
    assert "validate_data_integrity" not in names


def test_prompt_states_when_the_analysis_tools_close(tmp_path: Path) -> None:
    _write_item(tmp_path, _clean_frame())

    prompt = _build_system_prompt(_item(tmp_path), max_turns=50)

    assert "You have 50 turns total. Analysis tools are available only during turns 1-48." in prompt
    assert "Save your analysis script and numerical results before turn 48 ends." in prompt
    assert "Turns 49-50 are reserved for submitting your final report" in prompt


def test_prompt_states_the_run_rules_the_tools_do_not_announce(tmp_path: Path) -> None:
    _write_item(tmp_path, _clean_frame())

    prompt = _build_system_prompt(_item(tmp_path))

    assert "Keep exactly one saved script, scratch/analysis.py or scratch/analysis.R." in prompt
    assert "not tables you saved" in prompt
    assert "The task ends if an interactive execution exceeds the memory limit" in prompt
    assert "check_analysis and the checks at submission do not" in prompt
    assert "(up to 64 KiB per execution)" in prompt
    assert "Tool results longer than 4,000 characters show only their first and last parts" in prompt


def test_turn_budget_names_a_single_analysis_turn_without_a_range() -> None:
    from trialagentbench.trialeval.agent import _turn_budget_instruction

    assert "available only during turn 1. Save" in _turn_budget_instruction(3)
