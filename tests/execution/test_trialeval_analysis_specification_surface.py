"""Black-box qualification of matched TrialEval analysis specifications."""

from __future__ import annotations

import json
from pathlib import Path

from trialagentbench_test_helpers import (
    minimal_participant_output_contract,
    write_minimal_trialeval_release_dictionaries,
)

from trialagentbench.trialeval.data import (
    discover_participant_items,
    load_visible_context,
    participant_analysis_surface_sha256,
    stage_participant_evidence,
)
from trialagentbench.trialeval.schema import BenchmarkItem


def _participant_items(tmp_path: Path) -> tuple[BenchmarkItem, BenchmarkItem]:
    root = tmp_path / "public"
    root.mkdir(parents=True)
    write_minimal_trialeval_release_dictionaries(root)
    factors = {
        "TASK1001": {
            "context_configuration": "C1",
            "data_preparation": "analysis_ready",
            "analysis_specification": "locked_sap",
        },
        "TASK1002": {
            "context_configuration": "C2",
            "data_preparation": "analysis_ready",
            "analysis_specification": "protocol_only",
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_id": "trial_analysis_public_bundle_manifest/v1",
                "applied_baseline_profile_id": None,
                "applied_baseline_profile_sha256": None,
                "task_ids": ["TASK1001", "TASK1002"],
                "task_evidence_factors": factors,
            }
        ),
        encoding="utf-8",
    )
    for task_id in factors:
        item = root / "items" / task_id
        (item / "data").mkdir(parents=True)
        task = {
            "schema_id": "trial_analysis_task_v1",
            "task_id": task_id,
            "design_subtype": "individual_randomized",
            "primary_endpoint_id": "death",
            "primary_paramcd": "death",
            "primary_endpoint_term": "All-cause death",
            "primary_estimand_id": "primary_itt",
            "primary_effect_scale": "rmst_difference_tau",
            "estimand_mode": "fixed_declared_estimand",
            "primary_effect_scale_options": ["rmst_difference_tau"],
            "primary_result_unit": "days",
            "primary_population_id": "itt",
            "primary_intercurrent_event_strategy_ids": ["discontinuation:treatment_policy"],
            "primary_tau_dy": 365.0,
            "primary_control_arm_id": "control",
            "primary_treated_arm_id": "treated",
            "primary_question": "Estimate the RMST difference through day 365.",
            "deliverables": ["Report the RMST difference."],
            "analysis_tasks_file": "analysis_tasks.md",
        }
        (item / "task.json").write_text(json.dumps(task), encoding="utf-8")
        (item / "submission_contract.json").write_text(
            json.dumps(minimal_participant_output_contract(task_id)),
            encoding="utf-8",
        )
        (item / "protocol_summary.json").write_text(
            json.dumps(
                {
                    "primary_population": "ITT",
                    "followup_horizon_dy": 365.0,
                    "arms": ["control", "treated"],
                }
            ),
            encoding="utf-8",
        )
        (item / "data" / "ADSL.parquet").write_bytes(b"stable-analysis-data")
    sap_item = root / "items" / "TASK1001"
    (sap_item / "analysis_plan.json").write_text(
        json.dumps(
            {
                "primary_estimand_id": "primary_itt",
                "lane_rules": [{"role": "primary", "effect_scale": "rmst_difference_tau"}],
            }
        ),
        encoding="utf-8",
    )
    (sap_item / "analysis_tasks.md").write_text(
        "Estimate the RMST difference through day 365.\n",
        encoding="utf-8",
    )
    items = discover_participant_items(root)
    return items["TASK1002"], items["TASK1001"]


def test_matched_protocol_and_sap_items_preserve_data_but_not_prescription(
    tmp_path: Path,
) -> None:
    protocol_item, sap_item = _participant_items(tmp_path)
    protocol_root = stage_participant_evidence(protocol_item, tmp_path / "protocol")
    sap_root = stage_participant_evidence(sap_item, tmp_path / "sap")

    assert (protocol_root / "data" / "ADSL.parquet").read_bytes() == (sap_root / "data" / "ADSL.parquet").read_bytes()
    assert (protocol_root / "protocol_summary.json").read_bytes() == (sap_root / "protocol_summary.json").read_bytes()
    assert not (protocol_root / "analysis_plan.json").exists()
    assert not (protocol_root / "analysis_tasks.md").exists()
    assert (sap_root / "analysis_plan.json").is_file()
    assert (sap_root / "analysis_tasks.md").is_file()

    protocol_task = json.loads((protocol_root / "task.json").read_text(encoding="utf-8"))
    sap_task = json.loads((sap_root / "task.json").read_text(encoding="utf-8"))
    assert protocol_task["primary_effect_scale"] == "rmst_difference_tau"
    assert protocol_task["primary_effect_scale_options"] == ["rmst_difference_tau"]
    assert protocol_task["estimand_mode"] == "fixed_declared_estimand"
    assert "planning" not in protocol_task
    assert protocol_task["primary_question"] == "Estimate the RMST difference through day 365."
    assert sap_task["primary_effect_scale"] == "rmst_difference_tau"
    assert sap_task["primary_question"] == "Estimate the RMST difference through day 365."

    protocol_context = load_visible_context(protocol_item)
    assert "rmst_difference_tau" in protocol_context
    assert "analysis_plan.json" not in protocol_context
    assert participant_analysis_surface_sha256(protocol_item) != participant_analysis_surface_sha256(sap_item)


def test_surface_identity_is_deterministic(tmp_path: Path) -> None:
    protocol_item, _ = _participant_items(tmp_path)
    assert participant_analysis_surface_sha256(protocol_item) == participant_analysis_surface_sha256(protocol_item)
