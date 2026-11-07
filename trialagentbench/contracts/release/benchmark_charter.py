"""Canonical TrialAgentBench charter contract for standalone releases."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench.contracts.core.trialeval_conditions import ProcedureAssistanceV1
from trialagentbench.contracts.core.trialeval_factors import (
    TrialEvalAnalysisSpecificationV1,
    TrialEvalContextConfigurationV1,
    TrialEvalDataPreparationV1,
)
from trialagentbench.io.checksums import canonical_payload_sha256

DesignSubtypeV1 = Literal[
    "individual_randomized",
    "pragmatic",
    "covariate_structure",
    "endpoint_ascertainment",
    "cluster_parallel",
    "stepped_wedge",
    "group_sequential",
]


class _CharterModelV1(BaseModel):
    """Strict immutable base for standalone charter records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class BenchmarkSuiteDefinitionV1(_CharterModelV1):
    """Purpose and capability cascade for one benchmark suite."""

    suite: Literal["trialeval", "trialdev"]
    purpose: str = Field(min_length=1)
    capability_cascade: tuple[str, ...] = Field(min_length=2)
    secondary_lanes: tuple[str, ...] = ()


class BenchmarkAxisDefinitionV1(_CharterModelV1):
    """One design-archetype or construction-descriptor definition.

    Construction descriptors are not reported by or to the agent. The benchmark covers point-identified decisions.
    """

    axis: Literal["design", "assumption"]
    code: Literal["D1", "D2", "D3", "D4", "A1", "A2", "A3"]
    label: str = Field(min_length=1)
    operative_definition: str = Field(min_length=1)
    ordinal_scope: Literal["never", "matched_evaluation_series_only"]
    allowed_design_subtypes: tuple[DesignSubtypeV1, ...] = ()


class BenchmarkContextDefinitionV1(_CharterModelV1):
    """One non-ordinal evidence configuration."""

    code: TrialEvalContextConfigurationV1
    data_preparation: TrialEvalDataPreparationV1
    analysis_specification: TrialEvalAnalysisSpecificationV1
    canonical_procedure_assistance: Literal["output_contract_only"]
    capability_isolated: str = Field(min_length=1)


class BenchmarkParticipantArtifactDefinitionV1(_CharterModelV1):
    """Meaning and prohibited content for one participant artifact."""

    artifact: Literal[
        "protocol",
        "locked_sap",
        "data_specification",
        "output_contract",
        "unordered_checklist",
        "declared_defect",
    ]
    definition: str = Field(min_length=1)
    prohibited_content: tuple[str, ...] = ()


class BenchmarkMatchedContextContrastV1(_CharterModelV1):
    """One supported matched evidence-context contrast."""

    contrast_id: Literal["C1-C2", "C3-C4", "C3-C1", "C4-C2", "C5-C4"]
    minuend: TrialEvalContextConfigurationV1
    subtrahend: TrialEvalContextConfigurationV1
    interpretation: str = Field(min_length=1)
    held_fixed: tuple[str, ...] = Field(min_length=1)


class TrialEvalGradingPolicyV1(_CharterModelV1):
    """Public scientific policy bounding accepted TrialEval analyses."""

    credit_eligible_set_closure: str = Field(min_length=1)
    participant_compatibility_visibility: str = Field(min_length=1)
    analysis_specification_policy: str = Field(min_length=1)
    vocabulary_notes: str = Field(min_length=1)


class TrialAgentBenchCharterV1(_CharterModelV1):
    """Checksummed benchmark definition distributed with each release."""

    schema_id: Literal["ariadne.charter/v1"]
    version: Literal["v1"]
    suites: tuple[BenchmarkSuiteDefinitionV1, ...] = Field(min_length=2, max_length=2)
    axes: tuple[BenchmarkAxisDefinitionV1, ...] = Field(min_length=7, max_length=7)
    context_configurations: tuple[BenchmarkContextDefinitionV1, ...] = Field(min_length=5, max_length=5)
    procedure_assistance_levels: tuple[ProcedureAssistanceV1, ...] = Field(min_length=2, max_length=2)
    participant_artifacts: tuple[BenchmarkParticipantArtifactDefinitionV1, ...] = Field(min_length=6, max_length=6)
    matched_context_contrasts: tuple[BenchmarkMatchedContextContrastV1, ...] = Field(min_length=5, max_length=5)
    trialeval_grading_policy: TrialEvalGradingPolicyV1
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_charter(self) -> TrialAgentBenchCharterV1:
        if {row.suite for row in self.suites} != {"trialeval", "trialdev"}:
            raise ValueError("Benchmark charter must define TrialEval and TrialDev exactly once.")
        if {row.code for row in self.context_configurations} != {"C1", "C2", "C3", "C4", "C5"}:
            raise ValueError("Benchmark charter must define C1-C5 exactly once.")
        if {row.code for row in self.axes} != {"D1", "D2", "D3", "D4", "A1", "A2", "A3"}:
            raise ValueError("Benchmark charter must define D1-D4 and A1-A3 exactly once.")
        expected_factors = {
            "C1": ("analysis_ready", "locked_sap"),
            "C2": ("analysis_ready", "protocol_only"),
            "C3": ("raw_domains", "locked_sap"),
            "C4": ("raw_domains", "protocol_only"),
            "C5": ("raw_domains_declared_defect", "protocol_only"),
        }
        for row in self.context_configurations:
            if (row.data_preparation, row.analysis_specification) != expected_factors[row.code]:
                raise ValueError(f"Benchmark charter contains an invalid {row.code} factor mapping.")
        if {row.artifact for row in self.participant_artifacts} != {
            "protocol",
            "locked_sap",
            "data_specification",
            "output_contract",
            "unordered_checklist",
            "declared_defect",
        }:
            raise ValueError("Benchmark charter must define every participant artifact exactly once.")
        expected_contrasts = {"C1-C2", "C3-C4", "C3-C1", "C4-C2", "C5-C4"}
        if {row.contrast_id for row in self.matched_context_contrasts} != expected_contrasts:
            raise ValueError("Benchmark charter must define every supported context contrast exactly once.")
        payload = self.model_dump(mode="json", exclude={"checksum"})
        if canonical_payload_sha256(payload) != self.checksum:
            raise ValueError("Benchmark charter checksum does not match its payload.")
        return self


def render_benchmark_map_markdown(charter: TrialAgentBenchCharterV1) -> str:
    """Render a complete human-readable view of the released charter."""

    lines = [
        "# TrialAgentBench Benchmark Map",
        "",
        f"Charter schema: `{charter.schema_id}`  ",
        f"Charter checksum: `{charter.checksum}`",
        "",
        "This document is generated from `benchmark_charter.json`. The JSON record",
        "is the machine-readable benchmark definition.",
        "",
        "## Suite Purposes And Capability Cascades",
        "",
        "| Suite | Purpose | Capability cascade | Secondary lanes |",
        "|---|---|---|---|",
    ]
    for suite in charter.suites:
        lines.append(
            f"| `{suite.suite}` | {suite.purpose} | "
            f"{' -> '.join(f'`{value}`' for value in suite.capability_cascade)} | "
            f"{', '.join(f'`{value}`' for value in suite.secondary_lanes) or 'None'} |"
        )
    lines.extend(
        [
            "",
            "## TrialEval Design And Assumption Axes",
            "",
            "| Axis | Code | Label | Operative definition | Ordinal scope | Design subtypes |",
            "|---|---|---|---|---|---|",
        ]
    )
    for axis in charter.axes:
        subtypes = ", ".join(f"`{value}`" for value in axis.allowed_design_subtypes) or "Not applicable"
        lines.append(
            f"| {axis.axis} | `{axis.code}` | {axis.label} | {axis.operative_definition} | "
            f"`{axis.ordinal_scope}` | {subtypes} |"
        )
    lines.extend(
        [
            "",
            "## TrialEval Evidence Configurations",
            "",
            "C1-C5 are named, non-ordinal configurations. Procedure assistance and",
            "response interface are independent experimental factors.",
            "",
            "| Configuration | Data preparation | Analysis specification | "
            "Canonical assistance | Capability isolated |",
            "|---|---|---|---|---|",
        ]
    )
    for context in charter.context_configurations:
        lines.append(
            f"| `{context.code}` | `{context.data_preparation}` | "
            f"`{context.analysis_specification}` | `{context.canonical_procedure_assistance}` | "
            f"{context.capability_isolated} |"
        )
    lines.extend(
        [
            "",
            f"Procedure assistance levels: "
            f"{', '.join(f'`{value}`' for value in charter.procedure_assistance_levels)}.",
            "",
            "",
            "## Participant Artifact Glossary",
            "",
            "| Artifact | Definition | Prohibited content |",
            "|---|---|---|",
        ]
    )
    for artifact in charter.participant_artifacts:
        prohibited = "; ".join(artifact.prohibited_content) or "None"
        lines.append(f"| `{artifact.artifact}` | {artifact.definition} | {prohibited} |")
    lines.extend(
        [
            "",
            "## Supported Matched Context Contrasts",
            "",
            "| Contrast | Interpretation | Held fixed |",
            "|---|---|---|",
        ]
    )
    for contrast in charter.matched_context_contrasts:
        lines.append(
            f"| `{contrast.contrast_id}` | {contrast.interpretation} | "
            f"{', '.join(f'`{value}`' for value in contrast.held_fixed)} |"
        )
    policy = charter.trialeval_grading_policy
    lines.extend(
        [
            "",
            "## TrialEval Grading Policy",
            "",
            f"**Accepted-set closure.** {policy.credit_eligible_set_closure}",
            "",
            f"**Participant compatibility visibility.** {policy.participant_compatibility_visibility}",
            "",
            f"**Analysis-specification policy.** {policy.analysis_specification_policy}",
            "",
            f"**Vocabulary.** {policy.vocabulary_notes}",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "TrialAgentBenchCharterV1",
    "TrialEvalGradingPolicyV1",
    "render_benchmark_map_markdown",
]
