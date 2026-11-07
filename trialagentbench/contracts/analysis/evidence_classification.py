"""Contracts for deterministic evidence-source classification."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from trialagentbench.contracts.trace.observable import EvidenceCategoryV1

EvidenceSourceRoleV1 = Literal[
    "release_public_file",
    "submitted_payload",
    "conversation_event",
    "agent_scratch_file",
    "run_internal_file",
    "hidden_or_grader_file",
    "shell_literal_or_pseudo_path",
    "not_available_by_design",
]
ScratchArtifactKindV1 = Literal[
    "schema_or_dictionary",
    "contract_or_request_copy",
    "summary_table",
    "model_result",
    "survival_result",
    "uncertainty_result",
    "diagnostic_listing",
    "code_fragment",
    "transient_unresolved_workfile",
    "not_scratch",
]
EvidenceClassificationBasisV1 = Literal[
    "release_manifest_resolution",
    "submission_path_resolution",
    "conversation_event_resolution",
    "hidden_path_boundary",
    "shell_literal_rule",
    "directory_role",
    "file_extension",
    "json_key_signature",
    "table_column_signature",
    "text_signature",
    "transient_scratch_reference",
]


class EvidenceClassificationResultV1(BaseModel):
    """Deterministic classification for one observed evidence source."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["ariadne.evidence_classification_result/v1"] = "ariadne.evidence_classification_result/v1"
    observed_path: str
    canonical_source_path: str | None = None
    source_role: EvidenceSourceRoleV1
    evidence_category: EvidenceCategoryV1
    scratch_artifact_kind: ScratchArtifactKindV1 = "not_scratch"
    basis: tuple[EvidenceClassificationBasisV1, ...]
    participant_facing: bool
    hidden_or_grader: bool
    supports_positive_method_claim: bool


__all__ = [
    "EvidenceClassificationBasisV1",
    "EvidenceClassificationResultV1",
    "EvidenceSourceRoleV1",
    "ScratchArtifactKindV1",
]
