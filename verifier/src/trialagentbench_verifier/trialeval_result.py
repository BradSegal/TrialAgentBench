"""Minimal numerical result emitted by a TrialEval analysis program."""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import Field, field_validator

from trialagentbench_verifier.response import StrictSubmissionModel

TrialEvalProgramEffectMeasureV2: TypeAlias = Literal[
    "mean_difference",
    "rate_ratio",
    "restricted_mean_survival_time_difference",
    "risk_difference",
]
TrialEvalProgramConclusionV2: TypeAlias = Literal[
    "favors_treatment",
    "favors_comparator",
    "no_clear_difference",
]


class TrialEvalProgramSamplingIntervalV2(StrictSubmissionModel):
    """Declared sampling interval for one point estimate.

    Scientific validity is checked when the runner binds this optional object to the
    strict grading contract. This lets an invalid interval be localized
    without discarding an otherwise valid point estimate.
    """

    lower: float | None = None
    upper: float | None = None
    confidence_level: float | None = None
    interval_type: str = Field(default="", description="central_confidence_interval or repeated_confidence_interval")
    sidedness: str = "two_sided_central"


class TrialEvalProgramPointEstimateV2(StrictSubmissionModel):
    """Point estimate with optional sampling uncertainty."""

    kind: Literal["point"] = "point"
    value: float
    effect_measure: TrialEvalProgramEffectMeasureV2
    unit: str = Field(min_length=1)
    sampling_interval: TrialEvalProgramSamplingIntervalV2 | None = None


TrialEvalProgramEstimateV2 = TrialEvalProgramPointEstimateV2


class TrialEvalProgramResultV2(StrictSubmissionModel):
    """Small task-agnostic quantitative result written by an analysis program."""

    schema_id: Literal["trialagentbench.trialeval_program_result/v2"] = "trialagentbench.trialeval_program_result/v2"
    result: TrialEvalProgramEstimateV2
    source_artifacts: tuple[str, ...] = Field(min_length=1)
    conclusion: TrialEvalProgramConclusionV2 = Field(
        description=(
            "Direction supported by the sampling interval. "
            "Use no_clear_difference when that uncertainty includes the null."
        )
    )

    @field_validator("source_artifacts")
    @classmethod
    def _canonical_sources(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Canonicalize an unordered evidence set and reject duplicates."""

        if len(set(values)) != len(values):
            raise ValueError("source artifact paths must be unique")
        return tuple(sorted(values))


__all__ = [
    "TrialEvalProgramConclusionV2",
    "TrialEvalProgramEffectMeasureV2",
    "TrialEvalProgramEstimateV2",
    "TrialEvalProgramPointEstimateV2",
    "TrialEvalProgramResultV2",
    "TrialEvalProgramSamplingIntervalV2",
]
