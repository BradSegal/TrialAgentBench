"""Canonical TrialAgentBench diagnostic registry contracts."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

REGISTRY_RELATIVE_PATH = Path("diagnostic_registry_v1.json")
SeverityThresholdId = Literal["stressed", "fragile", "broken"]


class DiagnosticSeverityThresholdsV1(BaseModel):
    """Effect-scale boundaries for one empirically diagnosed mechanism."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric_name: str = Field(..., min_length=1)
    metric_unit: str = Field(..., min_length=1)
    decision_metric_names: dict[Literal["stressed", "fragile", "broken"], str] = Field(...)
    confidence_level: float = Field(..., gt=0.5, lt=1.0)
    multiplicity_adjustment: Literal["none", "bonferroni"]
    stressed: float = Field(..., ge=0.0)
    fragile: float = Field(..., ge=0.0)
    broken: float = Field(..., ge=0.0)

    @model_validator(mode="after")
    def _validate_order(self) -> DiagnosticSeverityThresholdsV1:
        if set(self.decision_metric_names) != {"stressed", "fragile", "broken"}:
            raise ValueError("Diagnostic decision metrics must cover stressed, fragile, and broken thresholds.")
        ordered: tuple[tuple[SeverityThresholdId, float], ...] = (
            ("stressed", self.stressed),
            ("fragile", self.fragile),
            ("broken", self.broken),
        )
        for (lower_id, lower), (upper_id, upper) in zip(ordered, ordered[1:], strict=False):
            if self.decision_metric_names[lower_id] == self.decision_metric_names[upper_id] and lower > upper:
                raise ValueError("Thresholds applied to the same diagnostic metric must be ordered.")
        return self


class DiagnosticKeySpecV1(BaseModel):
    """One diagnostic-key registry entry."""

    model_config = ConfigDict(extra="forbid")

    public_key: str
    diagnostic_scale: str
    threshold: str
    interpretation: str
    primary_credit_policy: Literal["always", "method_dependent", "design_modifier", "limitation"]
    metric_units: dict[str, str] = Field(default_factory=dict)
    nonidentified_evidence_class: Literal["empirical_diagnostic", "factual_premise"] | None = None
    severity_thresholds: DiagnosticSeverityThresholdsV1 | None = None
    primary_method_change_threshold: float | None = Field(default=None, ge=0.0)


class AssumptionTierOperatingDefinitionV1(BaseModel):
    """Scientific interpretation of one construction descriptor (decisions are point-identified)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = Field(..., min_length=1)
    operative_definition: str = Field(..., min_length=1)
    participant_evidence_requirement: str = Field(..., min_length=1)
    release_evidence_requirement: str = Field(..., min_length=1)
    scoring_constraint: str = Field(..., min_length=1)


class DiagnosticRegistryV1(BaseModel):
    """Canonical diagnostic registry used by release and scoring contracts."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["ariadne.diagnostic_registry/v1"]
    version: str
    assumption_tier_operating_definitions: dict[
        Literal["A1", "A2", "A3"],
        AssumptionTierOperatingDefinitionV1,
    ]
    diagnostic_keys: dict[str, DiagnosticKeySpecV1]

    @model_validator(mode="after")
    def _validate_registry(self) -> DiagnosticRegistryV1:
        if not self.diagnostic_keys:
            raise ValueError("diagnostic_keys must not be empty")
        if set(self.assumption_tier_operating_definitions) != {"A1", "A2", "A3"}:
            raise ValueError("Assumption-tier definitions must cover exactly A1 through A3.")
        public_keys = [entry.public_key for entry in self.diagnostic_keys.values()]
        if len(public_keys) != len(set(public_keys)):
            raise ValueError("diagnostic public keys must be unique")
        if any(
            not metric_id or not unit
            for entry in self.diagnostic_keys.values()
            for metric_id, unit in entry.metric_units.items()
        ):
            raise ValueError("diagnostic metric identities and units must be non-empty")
        return self


def diagnostic_registry_path_v1() -> Path:
    """Return the canonical diagnostic registry path."""

    path = Path(__file__).resolve().with_name(REGISTRY_RELATIVE_PATH.name)
    if not path.is_file():
        raise FileNotFoundError(f"Installed harness lacks diagnostic registry: {path}")
    return path


@lru_cache(maxsize=1)
def load_diagnostic_registry_v1() -> DiagnosticRegistryV1:
    """Load and validate the canonical diagnostic registry."""

    payload = json.loads(diagnostic_registry_path_v1().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Diagnostic registry must be a JSON object")
    return DiagnosticRegistryV1.model_validate(payload)


def diagnostic_key_by_assumption_id_v1() -> dict[str, str]:
    """Return assumption id to public diagnostic key mapping."""

    registry = load_diagnostic_registry_v1()
    return {assumption_id: spec.public_key for assumption_id, spec in registry.diagnostic_keys.items()}


def diagnostic_normal_critical_value_v1(*, assumption_id: str, comparisons: int) -> float:
    """Return the registry-declared simultaneous normal critical value."""
    if comparisons < 1:
        raise ValueError("Diagnostic comparisons must be at least one.")
    thresholds = load_diagnostic_registry_v1().diagnostic_keys[assumption_id].severity_thresholds
    if thresholds is None:
        raise ValueError(f"Assumption {assumption_id!r} has no empirical severity contract.")
    alpha = 1.0 - float(thresholds.confidence_level)
    if thresholds.multiplicity_adjustment == "bonferroni":
        alpha /= float(comparisons)
    return float(NormalDist().inv_cdf(1.0 - alpha / 2.0))


__all__ = [
    "AssumptionTierOperatingDefinitionV1",
    "DiagnosticKeySpecV1",
    "DiagnosticSeverityThresholdsV1",
    "DiagnosticRegistryV1",
    "REGISTRY_RELATIVE_PATH",
    "diagnostic_key_by_assumption_id_v1",
    "diagnostic_normal_critical_value_v1",
    "diagnostic_registry_path_v1",
    "load_diagnostic_registry_v1",
]
