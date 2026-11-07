"""Participant-visible TrialEval data-integrity contract."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["TrialEvalDataQualityDomainV1", "TrialEvalPublicIntegrityPolicyV1"]


class TrialEvalDataQualityDomainV1(BaseModel):
    """One source domain and its ordinary record identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str = Field(pattern=r"^data/raw/[A-Za-z0-9_.-]+\.parquet$")
    compound_key_fields: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _canonical_key(self) -> Self:
        """Require one unambiguous row identity."""

        if len(set(self.compound_key_fields)) != len(self.compound_key_fields):
            raise ValueError("data-quality compound-key fields must be unique")
        return self


class TrialEvalPublicIntegrityPolicyV1(BaseModel):
    """Define condition-neutral source-data quality work."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["ariadne.trialeval.data_quality_plan/v1"] = "ariadne.trialeval.data_quality_plan/v1"
    task_id: str = Field(min_length=1)
    domains: tuple[TrialEvalDataQualityDomainV1, ...] = Field(min_length=1)
    primary_analysis_domain: str = Field(pattern=r"^data/raw/[A-Za-z0-9_.-]+\.parquet$")

    @model_validator(mode="after")
    def _canonical_fields(self) -> Self:
        """Require unique source domains."""

        paths = tuple(domain.relative_path for domain in self.domains)
        if len(set(paths)) != len(paths):
            raise ValueError("data-quality domains must be unique")
        if self.primary_analysis_domain not in paths:
            raise ValueError("primary analysis domain must have a declared key")
        return self
