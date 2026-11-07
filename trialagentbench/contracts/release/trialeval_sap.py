"""Independent public locked-SAP contracts for TrialEval."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrialEvalPublicSAPPrimaryAnalysisV1(BaseModel):
    """Prespecified implementation of the locked primary analysis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    effect_scale: str = Field(min_length=1)
    method_id: str = Field(min_length=1)
    estimator_family: str = Field(min_length=1)
    implementation: str = Field(min_length=1)
    uncertainty_method: str = Field(min_length=1)
    required_method_modifiers: tuple[str, ...]
    baseline_covariate_strategy: Literal["unadjusted", "all_released"] = "unadjusted"
    nonlinear_prognostic_covariate: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _validate_model(self) -> TrialEvalPublicSAPPrimaryAnalysisV1:
        """Validate the declared baseline adjustment model."""

        if self.nonlinear_prognostic_covariate is not None and self.baseline_covariate_strategy != "all_released":
            raise ValueError("A nonlinear prognostic covariate requires adjustment for released baseline covariates.")
        return self


class TrialEvalPublicSAPDiagnosticRequirementV1(BaseModel):
    """Prespecified diagnostics for one analysis assumption."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assumption_id: str = Field(min_length=1)
    diagnostic_methods: tuple[str, ...] = Field(min_length=1)


class TrialEvalPublicSAPSensitivityAnalysisV1(BaseModel):
    """Prespecified method families for one scored sensitivity analysis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1)
    effect_scale: str = Field(min_length=1)
    method_families: tuple[str, ...] = Field(min_length=1)


class TrialEvalPublicSAPLaneRuleV1(BaseModel):
    """One participant-visible statistical-analysis-plan rule."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lane_id: str = Field(min_length=1)
    estimand_id: str | None = Field(default=None, min_length=1)
    endpoint_id: str | None = Field(default=None, min_length=1)
    effect_scale: str | None = Field(default=None, min_length=1)
    role: Literal["primary", "secondary", "sensitivity", "design", "reconstruction", "data_quality"]
    multiplicity_family_id: str = Field(min_length=1)
    alpha: float | None = Field(default=None, ge=0.0, le=1.0)
    mandatory: bool


class TrialEvalPublicSAPV1(BaseModel):
    """Checksummed participant-visible locked statistical analysis plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["trial_analysis_plan_contract_v1"]
    task_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    primary_estimand_id: str = Field(min_length=1)
    primary_endpoint_id: str = Field(min_length=1)
    primary_analysis: TrialEvalPublicSAPPrimaryAnalysisV1
    diagnostic_requirements: tuple[TrialEvalPublicSAPDiagnosticRequirementV1, ...] = Field(min_length=1)
    sensitivity_analyses: tuple[TrialEvalPublicSAPSensitivityAnalysisV1, ...]
    familywise_alpha: float = Field(gt=0.0, le=1.0)
    multiplicity_strategy: Literal["fixed_sequence", "holm", "none"]
    lane_rules: tuple[TrialEvalPublicSAPLaneRuleV1, ...] = Field(min_length=1)
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_plan(self) -> TrialEvalPublicSAPV1:
        """Validate stable ordering, primary-rule presence, and checksum."""

        if tuple(self.lane_rules) != tuple(sorted(self.lane_rules, key=lambda row: row.lane_id)):
            raise ValueError("Public SAP lane rules must be sorted by lane_id.")
        lane_ids = [rule.lane_id for rule in self.lane_rules]
        if len(lane_ids) != len(set(lane_ids)):
            raise ValueError("Public SAP lane identifiers must be unique.")
        primary = [rule for rule in self.lane_rules if rule.role == "primary" and rule.mandatory]
        if len(primary) != 1:
            raise ValueError("Public SAP requires exactly one mandatory primary rule.")
        primary_rule = primary[0]
        if primary_rule.estimand_id != self.primary_estimand_id:
            raise ValueError("Public SAP primary estimand is inconsistent.")
        if primary_rule.endpoint_id != self.primary_endpoint_id:
            raise ValueError("Public SAP primary endpoint is inconsistent.")
        if primary_rule.effect_scale != self.primary_analysis.effect_scale:
            raise ValueError("Public SAP primary effect scale is inconsistent.")
        if tuple(self.primary_analysis.required_method_modifiers) != tuple(
            sorted(set(self.primary_analysis.required_method_modifiers))
        ):
            raise ValueError("Public SAP primary-analysis modifiers must be unique and sorted.")
        if tuple(self.diagnostic_requirements) != tuple(
            sorted(self.diagnostic_requirements, key=lambda row: row.assumption_id)
        ):
            raise ValueError("Public SAP diagnostic requirements must be sorted by assumption_id.")
        if len({row.assumption_id for row in self.diagnostic_requirements}) != len(self.diagnostic_requirements):
            raise ValueError("Public SAP diagnostic assumptions must be unique.")
        for requirement in self.diagnostic_requirements:
            if tuple(requirement.diagnostic_methods) != tuple(sorted(set(requirement.diagnostic_methods))):
                raise ValueError("Public SAP diagnostic methods must be unique and sorted.")
        if tuple(self.sensitivity_analyses) != tuple(sorted(self.sensitivity_analyses, key=lambda row: row.task_id)):
            raise ValueError("Public SAP sensitivity analyses must be sorted by task_id.")
        if len({row.task_id for row in self.sensitivity_analyses}) != len(self.sensitivity_analyses):
            raise ValueError("Public SAP sensitivity task identifiers must be unique.")
        for analysis in self.sensitivity_analyses:
            if tuple(analysis.method_families) != tuple(sorted(set(analysis.method_families))):
                raise ValueError("Public SAP sensitivity method families must be unique and sorted.")
        payload = self.model_dump(mode="json", exclude={"checksum"}, exclude_none=True)
        expected = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if self.checksum != expected:
            raise ValueError("Public SAP checksum mismatch.")
        return self


__all__ = [
    "TrialEvalPublicSAPDiagnosticRequirementV1",
    "TrialEvalPublicSAPLaneRuleV1",
    "TrialEvalPublicSAPPrimaryAnalysisV1",
    "TrialEvalPublicSAPSensitivityAnalysisV1",
    "TrialEvalPublicSAPV1",
]
