"""Shared TrialAgentBench release-test fixtures."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from pydantic.types import JsonValue

from trialagentbench.contracts.release.benchmark_charter import (
    TrialAgentBenchCharterV1,
    render_benchmark_map_markdown,
)
from trialagentbench.contracts.release.trialeval_manifest import (
    TrialEvalParticipantDiagnosticDictionaryV1,
    TrialEvalParticipantMethodDictionaryV1,
    TrialEvalParticipantMethodV1,
)
from trialagentbench.contracts.scoring.diagnostic_registry import diagnostic_key_by_assumption_id_v1
from trialagentbench.contracts.trialeval_diagnostics import participant_diagnostic_dictionary_v1
from trialagentbench.grading.models import EstimatorProcedureV1
from trialagentbench.io.checksums import canonical_payload_sha256, sha256_file

FIXTURE_CELL_KEY = f"cell_{'5' * 20}"
FIXTURE_SERIES_KEY = f"series_{'6' * 20}"
FIXTURE_C5_INTEGRITY: dict[str, JsonValue] = {
    "affected_domain": "data/raw/endpoint_observations.parquet",
    "compound_key_fields": ["participant_id"],
    "resolution_action": "remove_exact_duplicate_copy",
    "resolution_status": "repaired",
    "observed_issue_group_count": 1,
    "observed_affected_row_count": 2,
    "selected_affected_row_count": 1,
    "selected_observed_compound_keys": ['[["int64","1"]]'],
    "nonmaterial_to_primary_analysis": False,
    "resolved_domain_content_sha256": "7" * 64,
}


def fixture_trialeval_key_record(
    number: int,
    *,
    context: str = "C1",
    cell_ref: str = FIXTURE_CELL_KEY,
    series_ref: str = FIXTURE_SERIES_KEY,
    release_id: str = f"release_{'a' * 64}",
) -> object:
    """Return a minimal grading-key record for task ``number`` with a mean-difference target."""
    from trialagentbench_verifier.trialeval import TrialEvalGradingKeyRecordV1

    from trialagentbench.grading.outcome import TrialEvalPointOracleTargetV1, TrialEvalQuestionBindingV1

    binding = TrialEvalQuestionBindingV1(
        analysis_population_id="itt",
        estimand_id="primary",
        treatment_id="active",
        comparator_id="control",
        endpoint_id="endpoint",
        effect_measure="mean_difference",
        analysis_unit="participant",
        population_basis="superpopulation",
        identification_status="point_identified",
        result_unit="points",
        uncertainty_type="central_confidence_interval",
        confidence_level=0.95,
        sidedness="two_sided_central",
        null_value=0.0,
        favorable_direction="higher",
    )
    return TrialEvalGradingKeyRecordV1.model_validate(
        {
            "schema_id": "trialagentbench.trialeval_grading_key_record/v1",
            "release_id": release_id,
            "task_id": f"te_{number:020x}",
            "trial_id": f"trial_{number:020x}",
            "base_world_id": f"world_{number:020x}",
            "cell_ref": cell_ref,
            "series_ref": series_ref,
            "context": context,
            "effect_state": "near_null",
            "design_tier": "D1",
            "design_subtype": "Individually randomized parallel-group trial",
            "design_family": "individual_randomized",
            "participant_data_sha256": "3" * 64,
            "participant_artifact_paths": ("data/example.parquet",),
            "target": TrialEvalPointOracleTargetV1(
                binding=binding,
                value=1.0,
                confidence_level=0.95,
                reference_sampling_standard_deviation=0.2,
                reference_sampling_basis_id="finite_world_oracle_contrast",
            ).model_dump(),
            "oracle_worlds": 10_000,
            "oracle_monte_carlo_standard_error": 0.001,
            "maximum_oracle_monte_carlo_standard_error": 0.01,
            "data_integrity": FIXTURE_C5_INTEGRITY if context == "C5" else None,
        }
    )


def write_trialeval_grading_key(root: Path, records: Sequence[object]) -> Path:
    """Write a release manifest and a verified grading key for fixture key records.

    Parameters
    ----------
    root
        Evaluator release root receiving ``release_manifest.json`` and ``grading_key/``.
    records
        Grading-key records of one release.

    Returns
    -------
    Path
        The written key records file.
    """
    from pydantic import BaseModel
    from trialagentbench_verifier.grading_key import (
        GRADING_KEY_MANIFEST,
        TRIALEVAL_GRADING_KEY_RECORDS,
        GradingKeyArtifactV1,
        GradingKeyManifestV1,
        grading_key_lines,
        grading_key_manifest_bytes,
    )
    from trialagentbench_verifier.trialeval import TrialEvalGradingKeyRecordV1

    public = sorted(
        (
            TrialEvalGradingKeyRecordV1.model_validate(cast(BaseModel, record).model_dump(mode="json"))
            for record in records
        ),
        key=lambda record: record.task_id,
    )
    release_id = public[0].release_id
    release = {
        "schema_id": "ariadne.trialeval.release/v1",
        "release_id": release_id,
        "trial_count": len({record.trial_id for record in public}),
        "task_count": len(public),
    }
    Path(root).mkdir(parents=True, exist_ok=True)
    (root / "release_manifest.json").write_text(json.dumps(release) + "\n", encoding="utf-8")
    body = grading_key_lines(public)
    manifest = GradingKeyManifestV1.create(
        schema_id="trialagentbench.trialeval_grading_key/v1",
        release_id=release_id,
        release_manifest_sha256=sha256_file(root / "release_manifest.json"),
        derivation_id="fixture",
        task_count=len(public),
        world_count=len({record.base_world_id for record in public}),
        cell_count=len({record.cell_ref for record in public}),
        series_count=len({record.series_ref for record in public}),
        artifacts=(
            GradingKeyArtifactV1(
                relative_path=TRIALEVAL_GRADING_KEY_RECORDS,
                sha256=hashlib.sha256(body).hexdigest(),
                byte_length=len(body),
            ),
        ),
    )
    (root / "grading_key").mkdir(exist_ok=True)
    (root / GRADING_KEY_MANIFEST).write_bytes(grading_key_manifest_bytes(manifest))
    (root / TRIALEVAL_GRADING_KEY_RECORDS).write_bytes(body)
    return root / TRIALEVAL_GRADING_KEY_RECORDS


def minimal_participant_output_contract(
    task_id: str,
    *,
    data_preparation: str = "analysis_ready",
) -> dict[str, object]:
    """Return one valid representation-neutral participant output contract."""

    required = {"limitations", "primary_analysis"}
    if data_preparation != "analysis_ready":
        required.add("reconstruction")
    payload: dict[str, object] = {
        "schema_id": "ariadne.trialeval_semantic_submission_contract/v1",
        "task_id": task_id,
        "submission_semantics_id": "ariadne.trialeval_submission/v1",
        "required_deliverables": sorted(required),
    }
    payload["checksum"] = canonical_payload_sha256(cast(JsonValue, payload))
    return payload


def minimal_estimator_procedure(*, identified_set: bool = False) -> EstimatorProcedureV1:
    """Return one complete estimator procedure for contract-focused fixtures."""

    return EstimatorProcedureV1(
        analysis_unit="participant",
        grouping="randomized_arm",
        partition="none",
        refitting="none",
        variance="Model-based variance for the fitted treatment contrast.",
        degrees_of_freedom="normal_reference",
        transformation="Apply the declared estimator on its reported effect scale.",
        combination="identified_set" if identified_set else "treated_minus_control",
        confidence_level=None if identified_set else 0.95,
        arm_quantities=("treated_quantity", "control_quantity"),
        contrast_quantity=(
            "identified set for treated_quantity - control_quantity"
            if identified_set
            else "treated_quantity - control_quantity"
        ),
    )


def write_minimal_trialdev_pool(pool: Path) -> Path:
    """Write a two-site, one-candidate frozen-pool manifest fixture."""
    from trialagentbench.contracts.trialdev.pools import TrialDevFrozenPool
    from trialagentbench.io import write_json_model

    pool.mkdir()
    manifest = TrialDevFrozenPool.model_validate(
        {
            "world_id": "fixture",
            "outcome_object": "Rheumatoid arthritis disease activity",
            "participant_count": 40,
            "collection_days": [28, 56, 84, 168, 336],
            "recruitment": {
                "groups": [
                    {
                        "group_id": "recorded",
                        "label": "Recorded baseline",
                        "conditions": [{"field_id": "severity", "operator": "greater_than", "value": 0}],
                        "participant_count": 40,
                    }
                ],
                "sites": [
                    {
                        "site_id": f"site_{i:03d}",
                        "country_code": "USA",
                        "opening_day": 0,
                        "screening_candidates_per_month": 8,
                        "candidate_count": 20,
                    }
                    for i in (1, 2)
                ],
                "screen_failure_fraction": 0.0,
            },
            "baseline_fields": ["severity"],
            "source_provenance_sha256": "a" * 64,
            "partitions": [
                {"asset_id": "control", "dose": None, "directory": "partitions/p000"},
                {"asset_id": "candidate", "dose": "standard", "directory": "partitions/p001"},
            ],
            "targets": [
                {
                    "asset_id": "candidate",
                    "dose": "standard",
                    "participant_count": 40,
                    "reference_days": 336,
                    "outcome_unit": "DAS28",
                    "outcome_mean_difference": 0,
                    "safety_risk_difference": 0,
                }
            ],
            "artifacts": [{"relative_path": "baseline.parquet", "sha256": "d" * 64, "byte_length": 1}],
        }
    )
    write_json_model(pool / "manifest.json", manifest)
    return pool


def write_minimal_trialeval_method_dictionary(root: Path) -> Path:
    """Write one valid participant-safe method dictionary for release fixtures."""

    payload = minimal_trialeval_method_dictionary()
    path = Path(root) / "method_dictionary.json"
    path.write_text(json.dumps(payload.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_minimal_trialeval_diagnostic_dictionary(root: Path, *, operation: str = "Fixture operation") -> Path:
    """Write one valid task-general diagnostic dictionary for release fixtures."""

    payload = minimal_trialeval_diagnostic_dictionary(operation=operation)
    path = Path(root) / "diagnostic_dictionary.json"
    path.write_text(json.dumps(payload.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_minimal_trialeval_release_dictionaries(root: Path) -> tuple[Path, Path]:
    """Write both required participant analysis dictionaries."""

    return (
        write_minimal_trialeval_method_dictionary(root),
        write_minimal_trialeval_diagnostic_dictionary(root),
    )


def minimal_trialeval_method_dictionary() -> TrialEvalParticipantMethodDictionaryV1:
    """Return one participant-safe method dictionary for release fixtures."""

    return TrialEvalParticipantMethodDictionaryV1(
        methods=(
            TrialEvalParticipantMethodV1(
                method_id="km_rmst_greenwood",
                estimator_family="km",
                objective="estimation",
                result_kind="numeric_point",
                effect_scale="rmst_difference_tau",
                design_modifiers=("ph_robust_fixed_horizon",),
                uncertainty_method_id="greenwood",
                possible_diagnostic_ids=("censoring_followup_public",),
                description="Kaplan-Meier fixed-horizon risk contrast with Greenwood uncertainty.",
                procedure=EstimatorProcedureV1(
                    analysis_unit="participant",
                    grouping="randomized_arm",
                    partition="none",
                    refitting="none",
                    variance="Greenwood variance for each arm-specific Kaplan-Meier curve.",
                    degrees_of_freedom="normal_reference",
                    transformation="Integrate each arm-specific Kaplan-Meier survival curve to the declared horizon.",
                    combination="treated_minus_control",
                    confidence_level=0.95,
                    arm_quantities=("treated_rmst", "control_rmst"),
                    contrast_quantity="treated_rmst - control_rmst",
                    fitting_specification=(
                        "Fit separate product-limit survival curves by randomized arm, truncate both at the "
                        "declared horizon, and integrate the right-continuous survival steps."
                    ),
                ),
            ),
        )
    )


def minimal_trialeval_diagnostic_dictionary(
    *, operation: str = "Fixture operation"
) -> TrialEvalParticipantDiagnosticDictionaryV1:
    """Return installed diagnostic semantics with fixture participant wording (construction owns the real text)."""

    return participant_diagnostic_dictionary_v1(
        {public_key: f"{operation} for {public_key}." for public_key in diagnostic_key_by_assumption_id_v1().values()}
    )


def route_reference_task_ranges(path: Path) -> list[dict[str, object]]:
    """Return contiguous task byte ranges for a test truth JSONL file."""

    ranges: list[dict[str, object]] = []
    offset = 0
    active_task: str | None = None
    active_offset = 0
    active_length = 0
    active_rows = 0
    with path.open("rb") as handle:
        for line in handle:
            task_id = str(json.loads(line)["task_id"])
            if active_task is not None and task_id != active_task:
                ranges.append(
                    {
                        "task_id": active_task,
                        "byte_offset": active_offset,
                        "byte_length": active_length,
                        "row_count": active_rows,
                    }
                )
                active_offset = offset
                active_length = 0
                active_rows = 0
            active_task = task_id
            active_length += len(line)
            active_rows += 1
            offset += len(line)
    if active_task is not None:
        ranges.append(
            {
                "task_id": active_task,
                "byte_offset": active_offset,
                "byte_length": active_length,
                "row_count": active_rows,
            }
        )
    return ranges


def minimal_estimator_family_map_payload() -> dict[str, object]:
    """Return a complete estimator-family partition for release fixtures."""

    estimands = {"km": "risk_difference", "coxph_binary": "global_cox_ph"}
    modifiers = {
        "km_ipcw": "procedure_modifier",
        "coxph_ipcw": "procedure_modifier",
        "validated_endpoint": "procedure_modifier",
        "cluster_parallel_participant_weighted": "design_modifier",
        "stepped_wedge_period_adjusted": "design_modifier",
        "group_sequential_adjustment": "procedure_modifier",
        "bounds": "procedure_modifier",
        "standardized_cox_g_computation": "procedure_modifier",
    }
    exclusions = {
        "coxph_time_interaction": "different_estimand",
        "piecewise_cox": "different_estimand",
        "weighted_logrank": "regime_invalid_diagnostic_only",
        "aft_parametric": "different_estimand",
        "milestone_risk": "failed_or_missing_qualification",
        "competing_risks": "different_estimand",
        "other": "failed_or_missing_qualification",
    }
    entries: list[dict[str, object]] = []
    for family, route_family in estimands.items():
        entries.append(
            {
                "estimator_family": family,
                "route_family": route_family,
                "role": "estimand_family",
                "requires_proportional_hazards_diagnostic": family == "coxph_binary",
                "requires_limitation": False,
                "credit_eligible_when": "Passed when the task register includes this estimand.",
                "rejected_when": "Rejected when the task register excludes this estimand.",
            }
        )
    for family, role in modifiers.items():
        entries.append(
            {
                "estimator_family": family,
                "role": role,
                "requires_proportional_hazards_diagnostic": family == "coxph_ipcw",
                "requires_limitation": family == "bounds",
                "credit_eligible_when": "Passed only when required by the method route.",
                "rejected_when": "Rejected outside a method route requiring this modifier.",
            }
        )
    for family, reason in exclusions.items():
        entries.append(
            {
                "estimator_family": family,
                "role": "unsupported",
                "requires_proportional_hazards_diagnostic": False,
                "requires_limitation": False,
                "exclusion_reason": reason,
                "credit_eligible_when": "Not passed in the current release.",
                "rejected_when": "Excluded from official primary scoring.",
            }
        )
    return {
        "schema_id": "ariadne.trialeval.estimator_route_family_map/v1",
        "entries": sorted(entries, key=lambda entry: str(entry["estimator_family"])),
    }


def minimal_estimator_registry_payload() -> dict[str, object]:
    """Return a checksum-bound source-method registry for release fixtures."""

    payload: dict[str, object] = {
        "methods": [
            {
                "method_id": "observed:coxph_binary_breslow",
                "family": "coxph_binary",
                "supported_effect_scales": ["log_hr"],
            },
            {
                "method_id": "observed:cox_ph",
                "family": "coxph_binary",
                "supported_effect_scales": ["log_hr"],
            },
        ]
    }
    payload["checksum"] = canonical_payload_sha256(cast(JsonValue, payload))
    return payload


def minimal_benchmark_charter_payload() -> dict[str, object]:
    """Return a compact charter satisfying the standalone release boundary."""

    design_axes = (
        ("D1", ("individual_randomized",)),
        ("D2", ("pragmatic",)),
        ("D3", ("covariate_structure", "endpoint_ascertainment")),
        ("D4", ("cluster_parallel", "stepped_wedge", "group_sequential")),
    )
    contexts = (
        ("C1", "analysis_ready", "locked_sap"),
        ("C2", "analysis_ready", "protocol_only"),
        ("C3", "raw_domains", "locked_sap"),
        ("C4", "raw_domains", "protocol_only"),
        ("C5", "raw_domains_declared_defect", "protocol_only"),
    )
    contrast_pairs = (
        ("C1-C2", "C1", "C2"),
        ("C3-C4", "C3", "C4"),
        ("C3-C1", "C3", "C1"),
        ("C4-C2", "C4", "C2"),
        ("C5-C4", "C5", "C4"),
    )
    payload: dict[str, object] = {
        "schema_id": "ariadne.charter/v1",
        "version": "v1",
        "suites": [
            {
                "suite": suite,
                "purpose": f"{suite} purpose",
                "capability_cascade": ["completion", "validity"],
                "secondary_lanes": [],
            }
            for suite in ("trialeval", "trialdev")
        ],
        "axes": [
            {
                "axis": "design",
                "code": code,
                "label": code,
                "operative_definition": f"{code} definition",
                "ordinal_scope": "never",
                "allowed_design_subtypes": list(subtypes),
            }
            for code, subtypes in design_axes
        ]
        + [
            {
                "axis": "assumption",
                "code": code,
                "label": code,
                "operative_definition": f"{code} definition",
                "ordinal_scope": "matched_evaluation_series_only",
                "allowed_design_subtypes": [],
            }
            for code in ("A1", "A2", "A3")
        ],
        "context_configurations": [
            {
                "code": code,
                "data_preparation": preparation,
                "analysis_specification": specification,
                "canonical_procedure_assistance": "output_contract_only",
                "capability_isolated": f"{code} capability",
            }
            for code, preparation, specification in contexts
        ],
        "procedure_assistance_levels": ["output_contract_only", "unordered_checklist"],
        "response_interfaces": ["structured", "narrative"],
        "participant_artifacts": [
            {
                "artifact": artifact,
                "definition": f"{artifact} definition",
                "prohibited_content": [],
            }
            for artifact in (
                "protocol",
                "locked_sap",
                "data_specification",
                "output_contract",
                "unordered_checklist",
                "declared_defect",
            )
        ],
        "matched_context_contrasts": [
            {
                "contrast_id": contrast_id,
                "minuend": minuend,
                "subtrahend": subtrahend,
                "interpretation": f"{contrast_id} interpretation",
                "held_fixed": ["base_trial"],
            }
            for contrast_id, minuend, subtrahend in contrast_pairs
        ],
        "trialeval_grading_policy": {
            "credit_eligible_set_closure": "Only accepted compatible routes are passed.",
            "participant_compatibility_visibility": "Every task compatibility dimension is participant-visible.",
            "analysis_specification_policy": (
                "Locked SAP and protocol-only configurations test different capabilities."
            ),
            "vocabulary_notes": "Primary and sensitivity lanes are distinct.",
        },
    }
    payload["checksum"] = canonical_payload_sha256(cast(JsonValue, payload))
    return payload


def minimal_benchmark_map_markdown() -> str:
    """Render the compact test charter as its required assessment map."""

    return render_benchmark_map_markdown(TrialAgentBenchCharterV1.model_validate(minimal_benchmark_charter_payload()))
