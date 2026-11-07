from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
from trialagentbench_test_helpers import (
    minimal_participant_output_contract,
    write_minimal_trialeval_release_dictionaries,
)

from trialagentbench.trialeval.agent import _build_system_prompt
from trialagentbench.trialeval.data import (
    discover_participant_items,
    load_visible_context,
    participant_task_factors,
    stage_participant_evidence,
)


def _evidence_factors(context: str) -> dict[str, str]:
    mapping = {
        "C1": ("analysis_ready", "locked_sap"),
        "C2": ("analysis_ready", "protocol_only"),
        "C3": ("raw_domains", "locked_sap"),
        "C4": ("raw_domains", "protocol_only"),
        "C5": ("raw_domains_declared_defect", "protocol_only"),
    }
    data_preparation, analysis_specification = mapping[context]
    return {
        "context_configuration": context,
        "data_preparation": data_preparation,
        "analysis_specification": analysis_specification,
    }


def _participant_release(root: Path) -> Path:
    participant = root / "participant"
    participant.mkdir(parents=True)
    write_minimal_trialeval_release_dictionaries(participant)
    item = participant / "items" / "TASK1"
    (item / "data").mkdir(parents=True)
    (participant / "manifest.json").write_text(
        json.dumps(
            {
                "schema_id": "trial_analysis_public_bundle_manifest/v1",
                "applied_baseline_profile_id": None,
                "applied_baseline_profile_sha256": None,
                "task_ids": ["TASK1"],
                "task_evidence_factors": {"TASK1": _evidence_factors("C1")},
            }
        ),
        encoding="utf-8",
    )
    (item / "task.json").write_text(
        json.dumps(
            {
                "schema_id": "trial_analysis_task_v1",
                "task_id": "TASK1",
                "design_subtype": "individual_randomized",
                "primary_endpoint_id": "endpoint",
                "primary_paramcd": "endpoint",
                "primary_estimand_id": "estimand",
                "primary_effect_scale": "risk_difference_tau",
                "estimand_mode": "fixed_declared_estimand",
                "primary_effect_scale_options": ["risk_difference_tau"],
                "primary_result_unit": "probability_difference",
                "primary_tau_dy": 365.0,
                "primary_population_id": "itt",
                "primary_intercurrent_event_strategy_ids": ["treatment_policy"],
                "primary_control_arm_id": "control",
                "primary_treated_arm_id": "treated",
            }
        ),
        encoding="utf-8",
    )
    (item / "study_brief.md").write_text("Participant study brief.", encoding="utf-8")
    (item / "submission_contract.json").write_text(
        json.dumps(minimal_participant_output_contract("TASK1")),
        encoding="utf-8",
    )
    (item / "data" / "ADSL.parquet").write_bytes(b"participant-data")
    return participant


def _canonical_participant_release(
    root: Path, *, sequential_c5: bool = False, json_material: bool = False, archive_material: bool = False
) -> Path:
    participant = root / "participant"
    task_id = "te_0123456789abcdef0123"
    trial_id = "trial_0123456789abcdef0123"
    task_root = participant / "tasks" / task_id
    trial_root = participant / "trials" / trial_id
    task_root.mkdir(parents=True)
    (trial_root / "data").mkdir(parents=True)
    (trial_root / "documents").mkdir()
    common_materials = [
        {
            "artifact_id": "analysis_request",
            "relative_path": f"tasks/{task_id}/documents/analysis_request.md",
            "description": "Analysis request",
        },
        {
            "artifact_id": "report_instructions",
            "relative_path": f"tasks/{task_id}/documents/report_instructions.md",
            "description": "Report instructions",
        },
        {
            "artifact_id": "protocol",
            "relative_path": f"trials/{trial_id}/documents/protocol.md",
            "description": "Protocol definitions",
        },
    ]
    if sequential_c5:
        available_materials = [
            *common_materials,
            {
                "artifact_id": "participant_records",
                "relative_path": f"tasks/{task_id}/data/raw/participant_records.parquet",
                "description": "Participant source records",
            },
            {
                "artifact_id": "event_history",
                "relative_path": f"tasks/{task_id}/data/raw/event_history.parquet",
                "description": "Event source records with a declared integrity incident",
            },
            {
                "artifact_id": "data_integrity_policy",
                "relative_path": f"tasks/{task_id}/documents/data_integrity_policy.md",
                "description": "Declared integrity incident and resolution",
            },
            {
                "artifact_id": "monitoring_plan",
                "relative_path": f"trials/{trial_id}/documents/monitoring_plan.csv",
                "description": "Sequential monitoring plan",
            },
            {
                "artifact_id": "monitoring_results",
                "relative_path": f"trials/{trial_id}/data/monitoring_results.parquet",
                "description": "Realized monitoring results",
            },
        ]
    else:
        available_materials = [
            *common_materials,
            {
                "artifact_id": "analysis_plan",
                "relative_path": f"trials/{trial_id}/documents/statistical_analysis_plan.md",
                "description": "Prespecified statistical analysis",
            },
            {
                "artifact_id": "analysis_data",
                "relative_path": f"trials/{trial_id}/data/analysis.parquet",
                "description": "Analysis-ready participant records",
            },
        ]
    if json_material:
        available_materials.append(
            {
                "artifact_id": "record_layout",
                "relative_path": f"trials/{trial_id}/documents/analysis_plan.json",
                "description": "Public analysis definitions",
            }
        )
        (trial_root / "documents" / "analysis_plan.json").write_text(
            json.dumps(
                {
                    "target_method_id": "private-method",
                    "route_reference_id": "private-route",
                    "endpoint": "death",
                    "lane_rules": [{"route_reference_id": "private-route", "window_days": 365}],
                }
            )
        )
    if archive_material:
        available_materials.append(
            {
                "artifact_id": "standards",
                "relative_path": f"tasks/{task_id}/data/standards.zip",
                "description": "Participant standards packet",
            }
        )
        (task_root / "data").mkdir(parents=True, exist_ok=True)
        with ZipFile(task_root / "data" / "standards.zip", "w") as archive:
            archive.writestr("README.md", "Participant standards")
    task = {
        "schema_id": "ariadne.trialeval.task/v1",
        "task_id": task_id,
        "title": "Primary analysis of death",
        "study_design": "Individually randomized concurrent parallel-group survival study",
        "question": "What is the treated-minus-control marginal death-risk difference at tau?",
        "available_materials": available_materials,
        "requested_outputs": ["Report the primary estimate and uncertainty."],
        "target_ids": ["marginal_death_risk_difference"],
    }
    (task_root / "task.json").write_text(json.dumps(task), encoding="utf-8")
    (task_root / "documents").mkdir()
    (task_root / "documents" / "analysis_request.md").write_text(
        "# Primary analysis of death\n\nWhat is the treated-minus-control marginal death-risk difference at tau?",
        encoding="utf-8",
    )
    (task_root / "documents" / "report_instructions.md").write_text(
        "# Report instructions\n\nSubmit a Markdown statistical report.",
        encoding="utf-8",
    )
    (trial_root / "documents" / "protocol.md").write_text("Protocol", encoding="utf-8")
    (trial_root / "documents" / "statistical_analysis_plan.json").write_text(
        json.dumps(
            {
                "private_route_id": "unavailable_reference_route",
                "purpose": "Machine-readable release custody record",
            }
        ),
        encoding="utf-8",
    )
    (trial_root / "data" / "undeclared_sibling.parquet").write_bytes(b"undeclared-data")
    (participant / "analysis_methods.csv").write_text(
        "analysis_method_id,description\nkm_rmst_greenwood,Restricted mean survival time\n",
        encoding="utf-8",
    )
    (participant / "diagnostic_methods.csv").write_text(
        "diagnostic_id,operation\ncensoring_followup_public,Assess follow-up by treatment arm\n",
        encoding="utf-8",
    )
    write_minimal_trialeval_release_dictionaries(participant)
    if sequential_c5:
        raw = task_root / "data" / "raw"
        raw.mkdir(parents=True)
        (raw / "participant_records.parquet").write_bytes(b"participant-data")
        (raw / "event_history.parquet").write_bytes(b"event-data")
        (task_root / "documents" / "data_integrity_policy.md").write_text(
            "# Source-data quality plan\n\nReview the declared keys and document any supported disposition.\n",
            encoding="utf-8",
        )
        (task_root / "documents" / "data_integrity_policy.json").write_text(
            json.dumps(
                {
                    "schema_id": "ariadne.trialeval.data_quality_plan/v1",
                    "task_id": task_id,
                    "domains": [
                        {
                            "relative_path": "data/raw/event_history.parquet",
                            "compound_key_fields": ["participant_id", "event_id", "occurrence_time"],
                        }
                    ],
                    "primary_analysis_domain": "data/raw/event_history.parquet",
                }
            ),
            encoding="utf-8",
        )
        (trial_root / "documents" / "monitoring_plan.csv").write_text(
            "look_id,information_fraction\n1,1.0\n", encoding="utf-8"
        )
        (trial_root / "data" / "monitoring_results.parquet").write_bytes(b"monitoring-data")
    else:
        (trial_root / "documents" / "statistical_analysis_plan.md").write_text("Analysis plan", encoding="utf-8")
        (trial_root / "data" / "analysis.parquet").write_bytes(b"analysis-data")
    artifacts = []
    for path in sorted(participant.rglob("*")):
        if not path.is_file():
            continue
        payload = path.read_bytes()
        artifacts.append(
            {
                "relative_path": f"participant/{path.relative_to(participant).as_posix()}",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "byte_length": len(payload),
            }
        )
    manifest = {
        "schema_id": "ariadne.benchmark_role_manifest/v1",
        "release_id": f"release_{'0' * 64}",
        "role": "participant",
        "trial_count": 1,
        "task_count": 1,
        "artifacts": artifacts,
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    manifest["checksum"] = hashlib.sha256(canonical).hexdigest()
    (participant / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    release_manifest = {
        "schema_id": "ariadne.trialeval.release/v1",
        "data_provenance": "generated_under_declared_trial_mechanisms",
        "release_id": manifest["release_id"],
        "construct_sha256": "0" * 64,
        "qualification_sha256": "1" * 64,
        "trial_count": 1,
        "task_count": 1,
        "participant_manifest_sha256": manifest["checksum"],
        "evaluation_manifest_sha256": "2" * 64,
    }
    release_canonical = json.dumps(
        release_manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    release_manifest["checksum"] = hashlib.sha256(release_canonical).hexdigest()
    (participant.parent / "release_manifest.json").write_text(
        json.dumps(release_manifest),
        encoding="utf-8",
    )
    return participant


def test_canonical_sequential_c5_stages_one_coherent_runtime_surface(tmp_path: Path) -> None:
    """Combine task-local integrity domains and trial-level monitoring inputs."""

    participant = _canonical_participant_release(tmp_path, sequential_c5=True)
    item = next(iter(discover_participant_items(participant).values()))

    assert item.workspace_data_paths == (
        "data/monitoring_results.parquet",
        "data/raw/event_history.parquet",
        "data/raw/participant_records.parquet",
    )
    staged = stage_participant_evidence(item, tmp_path / "workspace")
    assert all((staged / path).is_file() for path in item.workspace_data_paths)
    assert (staged / "data_integrity_policy.md").is_file()
    assert (staged / "data_integrity_policy.json").is_file()
    prompt = _build_system_prompt(item, analysis_specification=item.analysis_specification)
    # The lean system prompt states the role, rules and data files; the documents carry the request.
    assert "interface/targets.csv" in prompt and "marginal_death_risk_difference" not in prompt
    assert all(path in prompt for path in item.workspace_data_paths)
    assert "data_integrity_policy.md" in load_visible_context(item)
    assert not prompt.startswith("\\")
    assert "\\n  -" not in prompt
    assert not any(line.startswith("        ") for line in prompt.splitlines())


def test_participant_discovery_uses_only_public_manifest(tmp_path: Path) -> None:
    manifest = {
        "schema_id": "trial_analysis_public_bundle_manifest/v1",
        "applied_baseline_profile_id": None,
        "applied_baseline_profile_sha256": None,
        "task_ids": ["TASK2", "TASK1"],
        "task_evidence_factors": {
            "TASK1": _evidence_factors("C3"),
            "TASK2": _evidence_factors("C1"),
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "grader").mkdir()
    (tmp_path / "grader" / "item_index.json").write_text("not-json", encoding="utf-8")

    task_ids, factors = participant_task_factors(tmp_path)

    assert task_ids == ["TASK2", "TASK1"]
    assert {task_id: row.context_configuration for task_id, row in factors.items()} == {
        "TASK1": "C3",
        "TASK2": "C1",
    }


@pytest.mark.parametrize(
    "task_ids,contexts",
    [
        (["TASK1", "TASK1"], {"TASK1": "C1"}),
        (["TASK1"], {}),
        ([], {}),
    ],
)
def test_participant_discovery_rejects_ambiguous_manifest(
    tmp_path: Path,
    task_ids: list[str],
    contexts: dict[str, str],
) -> None:
    manifest = {
        "schema_id": "trial_analysis_public_bundle_manifest/v1",
        "applied_baseline_profile_id": None,
        "applied_baseline_profile_sha256": None,
        "task_ids": task_ids,
        "task_evidence_factors": {task_id: _evidence_factors(context) for task_id, context in contexts.items()},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="task_ids|task_evidence_factors"):
        participant_task_factors(tmp_path)


def test_participant_manifest_rejects_symlink_to_evaluator(tmp_path: Path) -> None:
    participant = tmp_path / "participant"
    participant.mkdir()
    write_minimal_trialeval_release_dictionaries(participant)
    evaluator_manifest = tmp_path / "evaluator" / "manifest.json"
    evaluator_manifest.parent.mkdir()
    evaluator_manifest.write_text("{}", encoding="utf-8")
    (participant / "manifest.json").symlink_to(evaluator_manifest)

    with pytest.raises(ValueError, match="must not be a symlink"):
        participant_task_factors(participant)


def test_participant_discovery_rejects_symlinked_item_directory(tmp_path: Path) -> None:
    participant = tmp_path / "participant"
    evaluator_item = tmp_path / "evaluator" / "items" / "TASK1"
    evaluator_item.mkdir(parents=True)
    participant.mkdir()
    write_minimal_trialeval_release_dictionaries(participant)
    (participant / "manifest.json").write_text(
        json.dumps(
            {
                "schema_id": "trial_analysis_public_bundle_manifest/v1",
                "applied_baseline_profile_id": None,
                "applied_baseline_profile_sha256": None,
                "task_ids": ["TASK1"],
                "task_evidence_factors": {"TASK1": _evidence_factors("C1")},
            }
        ),
        encoding="utf-8",
    )
    (participant / "items").mkdir()
    (participant / "items" / "TASK1").symlink_to(evaluator_item, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symlink"):
        discover_participant_items(participant, task_ids=("TASK1",))


def test_visible_context_rejects_allowed_text_symlink_to_evaluator(tmp_path: Path) -> None:
    participant = _participant_release(tmp_path)
    item = discover_participant_items(participant, task_ids=("TASK1",))["TASK1"]
    brief = item.visible_dir / "study_brief.md"
    brief.unlink()
    evaluator_brief = tmp_path / "evaluator" / "grader_notes.md"
    evaluator_brief.parent.mkdir()
    evaluator_brief.write_text("hidden evaluator notes", encoding="utf-8")
    brief.symlink_to(evaluator_brief)

    with pytest.raises(ValueError, match="must not be a symlink"):
        load_visible_context(item)


def test_staging_rejects_data_symlink_outside_participant_root(tmp_path: Path) -> None:
    participant = _participant_release(tmp_path)
    item = discover_participant_items(participant, task_ids=("TASK1",))["TASK1"]
    data_file = item.data_dir / "ADSL.parquet"
    data_file.unlink()
    outside_data = tmp_path / "evaluator" / "truth.parquet"
    outside_data.parent.mkdir()
    outside_data.write_bytes(b"evaluator-data")
    data_file.symlink_to(outside_data)
    destination = tmp_path / "staged"

    with pytest.raises(ValueError, match="must not be a symlink"):
        stage_participant_evidence(item, destination)
    assert not destination.exists()


def test_normal_participant_discovery_context_and_staging(tmp_path: Path) -> None:
    participant = _participant_release(tmp_path)
    dictionary = participant / "items" / "TASK1" / "data_dictionary.json"
    dictionary.write_text('{"large_reference":"mounted_not_inlined"}', encoding="utf-8")

    item = discover_participant_items(participant, task_ids=("TASK1",))["TASK1"]
    context = load_visible_context(item)
    staged = stage_participant_evidence(item, tmp_path / "staged")

    assert item.visible_dir == participant / "items" / "TASK1"
    assert "Participant study brief." in context
    assert "=== submission_contract.json ===" in context
    assert "ariadne.trialeval_semantic_submission_contract/v1" in context
    assert "mounted_not_inlined" not in context
    assert (staged / "data" / "ADSL.parquet").read_bytes() == b"participant-data"
    assert json.loads((staged / "submission_contract.json").read_text(encoding="utf-8")) == {
        **minimal_participant_output_contract("TASK1"),
    }
    assert json.loads((staged / "data_dictionary.json").read_text(encoding="utf-8")) == {
        "large_reference": "mounted_not_inlined"
    }
    assert not (staged / "method_dictionary.json").exists()
    assert not (staged / "diagnostic_dictionary.json").exists()


def test_canonical_participant_release_is_verified_and_projected(tmp_path: Path) -> None:
    participant = _canonical_participant_release(tmp_path)
    task_ids, factors = participant_task_factors(participant)
    item = discover_participant_items(participant)[task_ids[0]]
    staged = stage_participant_evidence(item, tmp_path / "staged")

    assert factors[task_ids[0]].context_configuration == "C1"
    assert item.submission_contract is not None
    assert item.submission_contract["required_deliverables"] == ["limitations", "primary_analysis"]
    context = load_visible_context(item)
    assert context.startswith("=== analysis_request.md ===")
    assert "treated-minus-control marginal death-risk difference" in context
    assert "task.json" not in context
    assert (staged / "data" / "analysis.parquet").read_bytes() == b"analysis-data"
    assert (staged / "protocol.md").read_text(encoding="utf-8") == "Protocol"
    assert not (staged / "task.json").exists()
    assert not (staged / "statistical_analysis_plan.json").exists()
    assert not (staged / "data" / "undeclared_sibling.parquet").exists()
    assert "unavailable_reference_route" not in context
    assert (staged / "analysis_request.md").is_file()
    assert (staged / "report_instructions.md").is_file()
    assert not (staged / "analysis_methods.csv").exists()
    assert not (staged / "diagnostic_methods.csv").exists()
    assert not (staged / "method_dictionary.json").exists()
    assert not (staged / "diagnostic_dictionary.json").exists()
    prompt = _build_system_prompt(item, analysis_specification=item.analysis_specification)
    assert "interface/targets.csv" in prompt
    assert "analysis_methods.csv" not in prompt
    assert "diagnostic_methods.csv" not in prompt
    assert "method_dictionary.json" not in prompt
    assert "diagnostic_dictionary.json" not in prompt


def test_canonical_participant_release_rejects_mutated_artifact(tmp_path: Path) -> None:
    participant = _canonical_participant_release(tmp_path)
    (participant / "trials" / "trial_0123456789abcdef0123" / "documents" / "protocol.md").write_text(
        "Changed",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="participant artifact changed"):
        discover_participant_items(participant)


@pytest.mark.parametrize("sequential_c5", [False, True])
@pytest.mark.parametrize("archive_material", [False, True])
def test_independent_replay_stages_the_exact_harness_evidence(tmp_path, sequential_c5, archive_material):
    """Both consumers project the same authenticated canonical materials and bytes."""
    from trialagentbench_verifier.execution import _stage_task_inputs, _task

    from trialagentbench.trialeval.data import participant_evidence_artifacts

    participant = _canonical_participant_release(
        tmp_path, sequential_c5=sequential_c5, json_material=True, archive_material=archive_material
    )
    item = next(iter(discover_participant_items(participant).values()))
    harness = stage_participant_evidence(item, tmp_path / "harness")
    independent = tmp_path / "independent"
    independent.mkdir()
    _stage_task_inputs(participant, independent, stream="trialeval", task=_task(participant, item.task_id))
    actual = {
        path.relative_to(independent).as_posix(): path.read_bytes()
        for path in independent.rglob("*")
        if path.is_file()
    }
    expected = {
        path.relative_to(harness).as_posix(): path.read_bytes() for path in harness.rglob("*") if path.is_file()
    }
    assert actual == expected
    if archive_material:
        assert actual["data/standards.zip"] == actual["data/raw/standards.zip"]
    assert b"private-method" not in actual["analysis_plan.json"]
    assert b"private-route" not in actual["analysis_plan.json"]
    for artifact in participant_evidence_artifacts(item):
        assert hashlib.sha256(actual[artifact.relative_path]).hexdigest() == artifact.sha256


# SHA-256 of the 1.0 system prompt for these fixtures (sequential C5 or not, 50 or 3 turns); the v1
# reporting interface must render it unchanged.
_V1_PROMPT_SHA256 = {
    (False, 50): "85a936d6ec224e730d277e5525babe257a00d4ece6869e06ab04cb18befb47da",
    (False, 3): "47a6fedf74e8e3b50a77b6d86b4df9075e5fe6d8e846569c072fc9ae15cdb2b6",
    (True, 50): "bd57290df84dd92cfae3214260056a00b87cfe66fd10a35d88e2a28005d5bbfe",
    (True, 3): "c15ef56504c81af8210e4d76dac485bca6e1ac3376b5e5dee03b07668b4ad90d",
}


@pytest.mark.parametrize(("sequential_c5", "max_turns"), sorted(_V1_PROMPT_SHA256))
def test_v1_reporting_interface_renders_the_1_0_prompt(tmp_path: Path, sequential_c5: bool, max_turns: int) -> None:
    participant = _canonical_participant_release(tmp_path, sequential_c5=sequential_c5)
    item = next(iter(discover_participant_items(participant).values()))
    default = _build_system_prompt(item, max_turns=max_turns, analysis_specification=item.analysis_specification)
    v1, clarified = (
        _build_system_prompt(
            item,
            max_turns=max_turns,
            analysis_specification=item.analysis_specification,
            reporting_interface=interface,
        )
        for interface in ("v1", "clarified")
    )
    assert hashlib.sha256(default.encode("utf-8")).hexdigest() == _V1_PROMPT_SHA256[(sequential_c5, max_turns)]
    assert v1 == default
    assert "`targets()` returns the target IDs as a list of strings" in clarified
    assert "`report(rows)` takes a data frame, or a list of dicts" in clarified
    assert "the analysis tools reopen for 2 more turns, once" in clarified
