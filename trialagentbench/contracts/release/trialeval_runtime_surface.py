"""TrialEval model-facing runtime-surface policy."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic import JsonValue as PydanticJsonValue
from trialagentbench_verifier.trialeval_inputs import (
    TRIALEVAL_AGENT_ANALYSIS_PLAN_REDACT_KEYS as TRIALEVAL_AGENT_ANALYSIS_PLAN_REDACT_KEYS,
)
from trialagentbench_verifier.trialeval_inputs import (
    TRIALEVAL_AGENT_ENDPOINT_DEFINITION_REDACT_KEYS as TRIALEVAL_AGENT_ENDPOINT_DEFINITION_REDACT_KEYS,
)
from trialagentbench_verifier.trialeval_inputs import (
    JsonArray as JsonArray,
)
from trialagentbench_verifier.trialeval_inputs import (
    JsonObject as JsonObject,
)
from trialagentbench_verifier.trialeval_inputs import (
    JsonScalar as JsonScalar,
)
from trialagentbench_verifier.trialeval_inputs import (
    JsonValue as JsonValue,
)
from trialagentbench_verifier.trialeval_inputs import (
    canonical_trialeval_workspace_path as canonical_trialeval_workspace_path,
)
from trialagentbench_verifier.trialeval_inputs import (
    sanitize_trialeval_agent_analysis_plan_payload as sanitize_trialeval_agent_analysis_plan_payload,
)
from trialagentbench_verifier.trialeval_inputs import (
    sanitize_trialeval_agent_endpoint_definition_payload as sanitize_trialeval_agent_endpoint_definition_payload,
)
from trialagentbench_verifier.trialeval_inputs import (
    sanitize_trialeval_agent_json_payload as sanitize_trialeval_agent_json_payload,
)
from trialagentbench_verifier.trialeval_inputs import (
    sanitize_trialeval_agent_json_text as sanitize_trialeval_agent_json_text,
)
from trialagentbench_verifier.trialeval_inputs import (
    sanitize_trialeval_agent_task_payload as sanitize_trialeval_agent_task_payload,
)

from trialagentbench.contracts.core.trialeval_factors import (
    TrialEvalAnalysisSpecificationV1,
    TrialEvalDataPreparationV1,
    TrialEvalEvidenceFactorsV1,
)
from trialagentbench.io import canonical_payload_sha256


class TrialAnalysisTaskV1(BaseModel):
    """Score-critical subset of an item-layout participant task (``items/<task_id>/task.json``)."""

    model_config = ConfigDict(extra="allow", frozen=True)

    schema_id: Literal["trial_analysis_task_v1"]
    task_id: str = Field(min_length=1)
    design_subtype: Literal[
        "individual_randomized",
        "pragmatic",
        "covariate_structure",
        "endpoint_ascertainment",
        "cluster_parallel",
        "stepped_wedge",
        "group_sequential",
    ]
    primary_endpoint_id: str = Field(min_length=1)
    primary_paramcd: str = Field(min_length=1)
    primary_estimand_id: str = Field(min_length=1)
    primary_effect_scale: (
        Literal[
            "log_hr",
            "risk_difference_tau",
            "rmst_difference_tau",
            "standardized_risk_difference_tau_reference",
        ]
        | None
    ) = None
    estimand_mode: Literal["fixed_declared_estimand"]
    primary_effect_scale_options: tuple[
        Literal[
            "log_hr",
            "risk_difference_tau",
            "rmst_difference_tau",
            "standardized_risk_difference_tau_reference",
        ],
        ...,
    ] = Field(min_length=1)
    bounded_deviation_delta_options: tuple[float, ...] = Field(default_factory=tuple)
    primary_result_unit: str | None = Field(default=None, min_length=1)
    primary_population_id: str = Field(min_length=1)
    primary_intercurrent_event_strategy_ids: tuple[str, ...] = ()
    primary_tau_dy: float | None = Field(default=None, gt=0)
    primary_control_arm_id: str = Field(min_length=1)
    primary_treated_arm_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _scientific_bindings_are_unambiguous(self) -> TrialAnalysisTaskV1:
        if "planning" in (self.__pydantic_extra__ or {}):
            raise ValueError("Participant tasks cannot include an unsupported planning analysis.")
        strategies = self.primary_intercurrent_event_strategy_ids
        if len(strategies) != len(set(strategies)):
            raise ValueError("primary intercurrent-event strategy identifiers must be unique")
        if self.primary_control_arm_id == self.primary_treated_arm_id:
            raise ValueError("primary treatment and control arm identifiers must differ")
        scale_options = self.primary_effect_scale_options
        if len(scale_options) != len(set(scale_options)):
            raise ValueError("primary_effect_scale_options must be unique")
        deltas = self.bounded_deviation_delta_options
        if any(not 0.0 < value < 1.0 for value in deltas) or tuple(sorted(set(deltas))) != deltas:
            raise ValueError("bounded_deviation_delta_options must be unique, increasing, and in (0, 1)")
        if self.primary_effect_scale is None or self.primary_result_unit is None:
            raise ValueError("fixed_declared_estimand requires primary_effect_scale and primary_result_unit")
        if scale_options != (self.primary_effect_scale,):
            raise ValueError("fixed_declared_estimand requires exactly the declared primary effect scale")
        expected_unit = {
            "log_hr": "log_hazard_ratio",
            "risk_difference_tau": "probability_difference",
            "rmst_difference_tau": "days",
            "standardized_risk_difference_tau_reference": "probability_difference",
        }[self.primary_effect_scale]
        if self.primary_result_unit != expected_unit:
            raise ValueError("primary_result_unit must match primary_effect_scale")
        tau_scales = {
            "risk_difference_tau",
            "rmst_difference_tau",
            "standardized_risk_difference_tau_reference",
        }
        requires_tau = bool(tau_scales.intersection(scale_options))
        if requires_tau and self.primary_tau_dy is None:
            raise ValueError("Tau-based credit-eligible primary scales require primary_tau_dy")
        if not requires_tau and self.primary_tau_dy is not None:
            raise ValueError("primary_tau_dy is valid only when a tau-based primary scale is accepted")
        return self


TrialEvalSemanticDeliverableV1 = Literal[
    "primary_analysis",
    "evidence",
    "limitations",
    "reconstruction",
    "analysis_data",
]


class TrialEvalSemanticSubmissionContractV1(BaseModel):
    """Semantic output requirements shared by all response interfaces."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["ariadne.trialeval_semantic_submission_contract/v1"]
    task_id: str = Field(min_length=1)
    submission_semantics_id: Literal["ariadne.trialeval_submission/v1"]
    required_deliverables: tuple[TrialEvalSemanticDeliverableV1, ...] = Field(min_length=1)
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _requirements_are_coherent(self) -> TrialEvalSemanticSubmissionContractV1:
        if tuple(sorted(set(self.required_deliverables))) != self.required_deliverables:
            raise ValueError("required semantic deliverables must be unique and sorted")
        core = {"primary_analysis", "limitations"}
        if not core.issubset(set(self.required_deliverables)):
            raise ValueError("semantic submission contract must require primary analysis and limitations")
        if "analysis_data" in self.required_deliverables and "reconstruction" not in self.required_deliverables:
            raise ValueError("data resolutions require reconstruction")
        payload = self.model_dump(mode="json", exclude_none=True)
        payload.pop("checksum")
        if canonical_payload_sha256(cast(PydanticJsonValue, payload)) != self.checksum:
            raise ValueError("participant output contract checksum does not match its semantic requirements")
        return self

    def validate_data_preparation(
        self,
        data_preparation: TrialEvalDataPreparationV1,
    ) -> TrialEvalSemanticSubmissionContractV1:
        """Require the exact obligations implied by an explicit data surface."""
        expected = {"primary_analysis", "limitations"}
        if data_preparation != "analysis_ready":
            expected.add("reconstruction")
        observed = set(self.required_deliverables)
        if observed != expected:
            raise ValueError(
                "semantic output obligations disagree with data preparation: "
                f"expected={sorted(expected)!r} observed={sorted(observed)!r}"
            )
        return self


class VisibleRuntimeFileV1(BaseModel):
    """Content identity of one participant-visible runtime file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["ariadne.visible_runtime_file/v1"] = "ariadne.visible_runtime_file/v1"
    relative_path: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mode: int = Field(ge=0, le=0o777)

    @field_validator("relative_path")
    @classmethod
    def _relative_path_is_safe(cls, value: str) -> str:
        normalized = PurePosixPath(value)
        if normalized.is_absolute() or ".." in normalized.parts or normalized.as_posix() != value:
            raise ValueError("visible runtime paths must be normalized relative POSIX paths")
        return value


class RuntimePolicyV1(BaseModel):
    """Execution policy that can alter the participant-visible task state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["ariadne.runtime_policy/v1"] = "ariadne.runtime_policy/v1"
    suite: Literal["trialeval"]
    evidence_factors: TrialEvalEvidenceFactorsV1
    runner_contract_version: str = Field(min_length=1)
    network_access: bool
    writable_workdir: bool
    timeout_seconds: int = Field(gt=0)


class RuntimeSurfaceIdentityV1(BaseModel):
    """Stable identity of the exact model-facing task state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["ariadne.runtime_surface_identity/v1"] = "ariadne.runtime_surface_identity/v1"
    task_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: tuple[VisibleRuntimeFileV1, ...]
    policy: RuntimePolicyV1
    fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _file_paths_are_unique(self) -> RuntimeSurfaceIdentityV1:
        paths = [record.relative_path for record in self.files]
        if paths != sorted(paths):
            raise ValueError("visible runtime files must be sorted by relative path")
        if len(paths) != len(set(paths)):
            raise ValueError("visible runtime file paths must be unique")
        return self


def visible_runtime_file(*, root: Path, relative_path: str) -> VisibleRuntimeFileV1:
    """Hash one regular file while rejecting traversal and symlink escape."""

    root_resolved = root.resolve(strict=True)
    normalized = PurePosixPath(relative_path)
    if normalized.is_absolute() or ".." in normalized.parts or normalized.as_posix() != relative_path:
        raise ValueError("visible runtime paths must be normalized relative POSIX paths")
    candidate = root / Path(*normalized.parts)
    if candidate.is_symlink():
        raise ValueError(f"visible runtime files must not be symlinks: {relative_path}")
    resolved = candidate.resolve(strict=True)
    if root_resolved not in resolved.parents:
        raise ValueError(f"visible runtime path escapes root: {relative_path}")
    if not resolved.is_file():
        raise ValueError(f"visible runtime path is not a regular file: {relative_path}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = resolved.stat()
    return VisibleRuntimeFileV1(
        relative_path=relative_path,
        size_bytes=stat.st_size,
        sha256=digest.hexdigest(),
        mode=stat.st_mode & 0o777,
    )


def fingerprint_runtime_surface(
    *, task_text: str, files: Sequence[VisibleRuntimeFileV1], policy: RuntimePolicyV1
) -> RuntimeSurfaceIdentityV1:
    """Return a deterministic identity for one model-facing runtime surface."""

    ordered = tuple(sorted(files, key=lambda record: record.relative_path))
    paths = [record.relative_path for record in ordered]
    if len(paths) != len(set(paths)):
        raise ValueError("visible runtime file paths must be unique")
    task_digest = hashlib.sha256(task_text.encode("utf-8")).hexdigest()
    payload = {
        "task_text_sha256": task_digest,
        "files": [record.model_dump(mode="json") for record in ordered],
        "policy": policy.model_dump(mode="json"),
    }
    return RuntimeSurfaceIdentityV1(
        task_text_sha256=task_digest,
        files=ordered,
        policy=policy,
        fingerprint_sha256=canonical_payload_sha256(cast(PydanticJsonValue, payload)),
    )


TRIALEVAL_AGENT_FORBIDDEN_ZIP_PARTS: Final[frozenset[str]] = frozenset(
    {
        "grader",
        "hidden",
        "public_estimand_contracts",
        "target_manifests",
    }
)

TRIALEVAL_AGENT_FORBIDDEN_ZIP_FILENAMES: Final[frozenset[str]] = frozenset(
    {
        "evaluation_target_register.jsonl",
        "method_route_register.jsonl",
        "method_composition.jsonl",
        "parser_ontology.json",
        "route_scoring_scope.jsonl",
        "route_reference_inputs.jsonl",
    }
)

TRIALEVAL_AGENT_FORBIDDEN_JSON_KEYS: Final[frozenset[str]] = frozenset(
    {
        *TRIALEVAL_AGENT_ANALYSIS_PLAN_REDACT_KEYS,
        "affected_records",
        "analysis_impact",
        "credit_eligible_route_families",
        "accepted_answer_family",
        "accepted_answer_families",
        "accepted_route_family",
        "accepted_route_families",
        "credit_eligible_method_families",
        "credit_eligible_route_reference_ids",
        "route_family",
        "evaluation_target_register",
        "intended_resolution",
        "method_route_id",
        "method_route_ids",
        "method_route_register",
        "point_analysis_status",
        "required_response",
        "disclosed_limitation",
        "supported_primary_result_kind",
        "unsupported_validation_strata",
        "unsupported_stratum_method_id",
        "point_estimation_method_id",
        "route_reference_input",
        "route_reference_inputs",
        "route_reference_id",
        "route_reference_ids",
        "defect_id",
        "evaluation_class",
    }
)

TRIALEVAL_AGENT_FORBIDDEN_SEMANTIC_FRAGMENTS: Final[tuple[str, ...]] = (
    "do not report a point estimate",
    "point analysis is not supported",
    "point estimate is not warranted",
    "report lower and upper identified-set bounds",
    "required response is",
)


class TrialEvalItemMemberRoleV1(str, Enum):
    """Custody role of one TrialEval item member."""

    TASK = "task"
    PROTOCOL = "protocol"
    ANALYSIS_INSTRUCTIONS = "analysis_instructions"
    ENDPOINT = "endpoint"
    INTERCURRENT_EVENT_STRATEGY = "intercurrent_event_strategy"
    RECONSTRUCTION_RECIPE = "reconstruction_recipe"
    DATA_DICTIONARY = "data_dictionary"
    PROVENANCE = "provenance"
    DOCUMENTATION = "documentation"
    OUTPUT_CONTRACT = "output_contract"
    INTERFACE_TOOL = "interface_tool"
    PREPARED_ANALYSIS_DATA = "prepared_analysis_data"
    SOURCE_DOMAIN_DATA = "source_domain_data"


TRIALEVAL_PARTICIPANT_TEXT_FILE_ROLES: Final[MappingProxyType[str, TrialEvalItemMemberRoleV1]] = MappingProxyType(
    {
        "README.md": TrialEvalItemMemberRoleV1.DOCUMENTATION,
        "analysis_request.md": TrialEvalItemMemberRoleV1.ANALYSIS_INSTRUCTIONS,
        "analysis_result_guide.md": TrialEvalItemMemberRoleV1.OUTPUT_CONTRACT,
        "analysis_plan.json": TrialEvalItemMemberRoleV1.ANALYSIS_INSTRUCTIONS,
        "analysis_population_guidance.json": TrialEvalItemMemberRoleV1.ANALYSIS_INSTRUCTIONS,
        "analysis_tasks.md": TrialEvalItemMemberRoleV1.ANALYSIS_INSTRUCTIONS,
        "ascertainment_model.json": TrialEvalItemMemberRoleV1.ENDPOINT,
        "data_dictionary.json": TrialEvalItemMemberRoleV1.DATA_DICTIONARY,
        "data_dictionary.csv": TrialEvalItemMemberRoleV1.DATA_DICTIONARY,
        "data_integrity_policy.md": TrialEvalItemMemberRoleV1.RECONSTRUCTION_RECIPE,
        "data_integrity_policy.json": TrialEvalItemMemberRoleV1.RECONSTRUCTION_RECIPE,
        "data_specification.md": TrialEvalItemMemberRoleV1.DATA_DICTIONARY,
        "endpoint_definition.json": TrialEvalItemMemberRoleV1.ENDPOINT,
        "intercurrent_event_strategy.json": TrialEvalItemMemberRoleV1.INTERCURRENT_EVENT_STRATEGY,
        "protocol_summary.json": TrialEvalItemMemberRoleV1.PROTOCOL,
        "protocol.md": TrialEvalItemMemberRoleV1.PROTOCOL,
        "reconstruction_task.json": TrialEvalItemMemberRoleV1.RECONSTRUCTION_RECIPE,
        "source_trace.json": TrialEvalItemMemberRoleV1.PROVENANCE,
        "study_brief.md": TrialEvalItemMemberRoleV1.PROTOCOL,
        "report_instructions.md": TrialEvalItemMemberRoleV1.OUTPUT_CONTRACT,
        "statistical_analysis_plan.md": TrialEvalItemMemberRoleV1.ANALYSIS_INSTRUCTIONS,
        "monitoring_plan.csv": TrialEvalItemMemberRoleV1.RECONSTRUCTION_RECIPE,
        "submission_contract.json": TrialEvalItemMemberRoleV1.OUTPUT_CONTRACT,
        "task.json": TrialEvalItemMemberRoleV1.TASK,
        "visible_manifest.json": TrialEvalItemMemberRoleV1.PROVENANCE,
    }
)

TRIALEVAL_PARTICIPANT_TEXT_FILENAMES: Final[tuple[str, ...]] = tuple(TRIALEVAL_PARTICIPANT_TEXT_FILE_ROLES)
TRIALEVAL_DATA_INTEGRITY_UTILITY_MEMBER_V1: Final = "interface/data_integrity.py"
TRIALEVAL_DATA_INTEGRITY_RULES_MEMBER_V1: Final = "interface/data_integrity_rules.py"
TRIALEVAL_PARTICIPANT_INTERFACE_FILE_ROLES: Final[MappingProxyType[str, TrialEvalItemMemberRoleV1]] = MappingProxyType(
    {
        "interface/analysis_result.schema.json": TrialEvalItemMemberRoleV1.OUTPUT_CONTRACT,
        "interface/estimates.csv": TrialEvalItemMemberRoleV1.OUTPUT_CONTRACT,
        "interface/targets.csv": TrialEvalItemMemberRoleV1.OUTPUT_CONTRACT,
        "interface/reporting.py": TrialEvalItemMemberRoleV1.INTERFACE_TOOL,
        "interface/reporting.R": TrialEvalItemMemberRoleV1.INTERFACE_TOOL,
        "interface/response_contract.json": TrialEvalItemMemberRoleV1.OUTPUT_CONTRACT,
        "interface/submission_shapes.json": TrialEvalItemMemberRoleV1.OUTPUT_CONTRACT,
        TRIALEVAL_DATA_INTEGRITY_UTILITY_MEMBER_V1: TrialEvalItemMemberRoleV1.INTERFACE_TOOL,
        TRIALEVAL_DATA_INTEGRITY_RULES_MEMBER_V1: TrialEvalItemMemberRoleV1.INTERFACE_TOOL,
    }
)
TRIALEVAL_AGENT_ALWAYS_EXCLUDED_FILES: Final[frozenset[str]] = frozenset(
    {"source_trace.json", "visible_manifest.json"}
)

_TRIALEVAL_COMMON_REQUIRED_MEMBERS: Final[frozenset[str]] = frozenset(
    {
        "README.md",
        "data_dictionary.json",
        "endpoint_definition.json",
        "intercurrent_event_strategy.json",
        "protocol_summary.json",
        "study_brief.md",
        "submission_contract.json",
        "task.json",
    }
)
_TRIALEVAL_ANALYSIS_READY_REQUIRED_MEMBERS: Final[frozenset[str]] = frozenset({"analysis_population_guidance.json"})
_TRIALEVAL_RECONSTRUCTION_REQUIRED_MEMBERS: Final[frozenset[str]] = frozenset({"reconstruction_task.json"})
_TRIALEVAL_LOCKED_SAP_REQUIRED_MEMBERS: Final[frozenset[str]] = frozenset({"analysis_plan.json", "analysis_tasks.md"})


def required_trialeval_participant_members(
    *,
    data_preparation: TrialEvalDataPreparationV1,
    analysis_specification: TrialEvalAnalysisSpecificationV1,
) -> frozenset[str]:
    """Return required non-data members for explicit evidence factors."""

    required = set(_TRIALEVAL_COMMON_REQUIRED_MEMBERS)
    if data_preparation == "analysis_ready":
        required.update(_TRIALEVAL_ANALYSIS_READY_REQUIRED_MEMBERS)
    else:
        required.update(_TRIALEVAL_RECONSTRUCTION_REQUIRED_MEMBERS)
    if data_preparation == "raw_domains_declared_defect":
        required.add("data_integrity_policy.json")
    if analysis_specification == "locked_sap":
        required.update(_TRIALEVAL_LOCKED_SAP_REQUIRED_MEMBERS)
    return frozenset(required)


def trialeval_agent_excluded_files() -> frozenset[str]:
    """Return all files excluded from the TrialEval model-visible filesystem."""

    return TRIALEVAL_AGENT_ALWAYS_EXCLUDED_FILES


def trialeval_agent_allows_item_member(
    *,
    item_relative_path: str,
    data_preparation: TrialEvalDataPreparationV1,
) -> bool:
    """Return whether a classified item member belongs on the participant surface."""

    return (
        PurePosixPath(item_relative_path).name not in TRIALEVAL_AGENT_ALWAYS_EXCLUDED_FILES
        and classify_trialeval_item_member(
            item_relative_path=item_relative_path,
            data_preparation=data_preparation,
        )
        != TrialEvalItemMemberRoleV1.PROVENANCE
    )


def classify_trialeval_item_member(
    *,
    item_relative_path: str,
    data_preparation: TrialEvalDataPreparationV1,
) -> TrialEvalItemMemberRoleV1:
    """Classify one item member, rejecting evidence-incompatible paths."""

    path = PurePosixPath(item_relative_path)
    if path.parts and path.parts[0] == "data" and path.suffix == ".parquet":
        if data_preparation == "analysis_ready" and len(path.parts) != 2:
            raise ValueError(f"Analysis-ready tasks require data directly under data/: {item_relative_path!r}")
        if data_preparation != "analysis_ready" and (len(path.parts) != 3 or path.parts[1] != "raw"):
            raise ValueError(f"Raw-domain tasks require data under data/raw/: {item_relative_path!r}")
    role = classify_trialeval_participant_member(item_relative_path)
    if data_preparation == "analysis_ready":
        if role == TrialEvalItemMemberRoleV1.SOURCE_DOMAIN_DATA:
            raise ValueError(f"Analysis-ready tasks cannot expose source-domain data: {item_relative_path!r}")
        return role
    if role == TrialEvalItemMemberRoleV1.PREPARED_ANALYSIS_DATA:
        raise ValueError(f"Raw-domain tasks cannot expose prepared analysis data: {item_relative_path!r}")
    return role


def classify_trialeval_participant_member(item_relative_path: str) -> TrialEvalItemMemberRoleV1:
    """Classify one TrialEval member by its participant-surface role."""

    path = PurePosixPath(item_relative_path)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != item_relative_path:
        raise ValueError("TrialEval item members must be normalized relative POSIX paths")
    if len(path.parts) == 1:
        try:
            return TRIALEVAL_PARTICIPANT_TEXT_FILE_ROLES[path.name]
        except KeyError as exc:
            raise ValueError(f"Unknown TrialEval item metadata member: {item_relative_path!r}") from exc
    if item_relative_path in TRIALEVAL_PARTICIPANT_INTERFACE_FILE_ROLES:
        return TRIALEVAL_PARTICIPANT_INTERFACE_FILE_ROLES[item_relative_path]
    if path.parts[0] != "data" or path.suffix != ".parquet":
        raise ValueError(f"Unknown TrialEval item member: {item_relative_path!r}")
    if len(path.parts) == 2:
        return TrialEvalItemMemberRoleV1.PREPARED_ANALYSIS_DATA
    if len(path.parts) == 3 and path.parts[1] == "raw":
        return TrialEvalItemMemberRoleV1.SOURCE_DOMAIN_DATA
    raise ValueError(f"Unknown TrialEval item data member: {item_relative_path!r}")


def trialeval_agent_forbidden_json_key_paths(
    value: object,
    *,
    prefix: str = "",
) -> list[str]:
    """Return paths to JSON keys forbidden on the model-visible TrialEval surface."""

    forbidden_keys = TRIALEVAL_AGENT_FORBIDDEN_JSON_KEYS
    findings: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in forbidden_keys:
                findings.append(path)
            findings.extend(trialeval_agent_forbidden_json_key_paths(item, prefix=path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            findings.extend(trialeval_agent_forbidden_json_key_paths(item, prefix=path))
    return findings


def trialeval_agent_semantic_leakage_paths(
    value: object,
    *,
    prefix: str = "",
) -> list[str]:
    """Return participant JSON paths containing answer-bearing instructions."""

    findings: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            findings.extend(trialeval_agent_semantic_leakage_paths(item, prefix=path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            findings.extend(trialeval_agent_semantic_leakage_paths(item, prefix=path))
    elif isinstance(value, str):
        normalized = " ".join(value.casefold().split())
        leaked_assumption_state = prefix.endswith(".assumption_class") and normalized == "stress"
        if leaked_assumption_state or any(
            fragment in normalized for fragment in TRIALEVAL_AGENT_FORBIDDEN_SEMANTIC_FRAGMENTS
        ):
            findings.append(prefix or "$")
    return findings
