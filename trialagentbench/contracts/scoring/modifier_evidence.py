"""Modifier evidence contracts for standalone TrialEval graders."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

METHOD_MODIFIERS_V1 = frozenset(
    {
        "ipcw_adjusted",
        "cluster_robust_inference",
        "participant_population_target",
        "stepped_wedge_period_adjusted",
        "group_sequential_adjustment",
        "misclassification_corrected",
        "reference_standardization",
        "flexible_model_form",
        "ph_robust_fixed_horizon",
    }
)

MethodModifierV1 = Literal[
    "ipcw_adjusted",
    "cluster_robust_inference",
    "participant_population_target",
    "stepped_wedge_period_adjusted",
    "group_sequential_adjustment",
    "misclassification_corrected",
    "reference_standardization",
    "flexible_model_form",
    "ph_robust_fixed_horizon",
]


class ModifierEvidenceBasisV1(BaseModel):
    """Public evidence that justifies one required method modifier."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["ariadne.trialeval.modifier_evidence_basis/v1"] = "ariadne.trialeval.modifier_evidence_basis/v1"
    modifier: MethodModifierV1
    public_rel_paths: tuple[str, ...] = Field(..., min_length=1)
    required_columns: tuple[str, ...] = Field(default_factory=tuple)
    rationale: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_evidence(self) -> ModifierEvidenceBasisV1:
        """Validate and canonicalize modifier evidence."""

        if self.modifier not in METHOD_MODIFIERS_V1:
            raise ValueError(f"Unknown TrialEval method modifier: {self.modifier!r}.")
        object.__setattr__(self, "public_rel_paths", tuple(sorted(set(self.public_rel_paths))))
        object.__setattr__(self, "required_columns", tuple(sorted(set(self.required_columns))))
        return self


__all__ = ["METHOD_MODIFIERS_V1", "MethodModifierV1", "ModifierEvidenceBasisV1"]
