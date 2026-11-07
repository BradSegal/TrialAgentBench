"""Scientific reference targets consumed by TrialDev evidence grading."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from trialagentbench.contracts.trialdev.release import TRIALDEV_ASSET_ID_PATTERN


class TrialDevEvidenceTargetV1(BaseModel):
    """Explicit scientific target and reference precision for one candidate endpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_id: str = Field(pattern=TRIALDEV_ASSET_ID_PATTERN)
    endpoint: Literal["clinical_outcome", "safety_risk"]
    unit: str = Field(min_length=1)
    decision_scale: float = Field(gt=0)
    comparison_sample_size: int = Field(ge=2)
    point_identified: Literal[True]
    estimand: Literal["standardized_observed_association"]
    oracle_monte_carlo_standard_error: float = Field(default=0.0, ge=0)
    oracle_value: float


__all__ = ["TrialDevEvidenceTargetV1"]
