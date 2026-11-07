"""Evaluator-side contracts for generated assumption evidence."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench.io import canonical_payload_sha256, read_jsonl_models

AssumptionStatusV1 = Literal["holds", "stressed", "broken", "not_classified"]
AssumptionSeverityBandV1 = Literal["holds", "mild", "fragile", "broken", "not_classified"]
SeverityThresholdId = Literal["stressed", "fragile", "broken"]
AssumptionDiagnosabilityV1 = Literal[
    "directly_diagnosable",
    "partially_diagnosable",
    "design_declared",
    "not_identifiable",
]


def _validate_public_evidence_path(path: str) -> str:
    value = str(path)
    pure = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or pure.is_absolute()
        or value != pure.as_posix()
        or any(part in {"", ".", ".."} or part.startswith(".") for part in pure.parts)
        or ":" in value
        or {"hidden", "grader"} & {part.lower() for part in pure.parts}
    ):
        raise ValueError(f"Invalid participant-relative public evidence path: {value!r}.")
    return value


class AssumptionEvidenceRecordV1(BaseModel):
    """One evaluator-side assumption-evidence record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assumption_id: str = Field(..., min_length=1)
    expected_status: AssumptionStatusV1
    computed_status: AssumptionStatusV1
    expected_band: AssumptionSeverityBandV1
    computed_band: AssumptionSeverityBandV1
    diagnosability: AssumptionDiagnosabilityV1
    severity_metric: float | None = None
    severity_metric_name: str | None = Field(default=None, min_length=1)
    threshold_stressed: float | None = Field(default=None, ge=0.0)
    threshold_fragile: float | None = Field(default=None, ge=0.0)
    threshold_broken: float | None = Field(default=None, ge=0.0)
    decision_metric_names: dict[Literal["stressed", "fragile", "broken"], str] = Field(default_factory=dict)
    supporting_metrics: dict[str, float] = Field(default_factory=dict)
    metric_units: dict[str, str] = Field(...)
    metric_public_evidence_basis: dict[str, tuple[str, ...]] = Field(...)
    factual_public_evidence_basis: tuple[str, ...] | None = None
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_bands(self) -> AssumptionEvidenceRecordV1:
        """Reject inconsistent severity thresholds and status bands."""

        metric_fields = (self.severity_metric, self.severity_metric_name)
        threshold_fields = (self.threshold_stressed, self.threshold_fragile, self.threshold_broken)
        has_metric = all(value is not None for value in metric_fields)
        has_thresholds = all(value is not None for value in threshold_fields)
        if any(value is not None for value in metric_fields) and not has_metric:
            raise ValueError("Assumption evidence must provide both a severity metric and its name.")
        if any(value is not None for value in threshold_fields) and not has_thresholds:
            raise ValueError("Assumption evidence must provide either all severity thresholds or none of them.")
        if not has_metric:
            if self.diagnosability not in {"design_declared", "not_identifiable"}:
                raise ValueError(
                    "Only design-declared or nonidentified obligations may omit empirical severity fields."
                )
            if has_thresholds:
                raise ValueError("Non-empirical obligations cannot contain severity thresholds.")
            if self.decision_metric_names:
                raise ValueError("Non-empirical obligations cannot contain empirical decision metrics.")
            if not self.metric_public_evidence_basis and not self.factual_public_evidence_basis:
                raise ValueError("A non-empirical obligation requires a public factual evidence basis.")
        elif has_thresholds:
            if self.threshold_stressed is None or self.threshold_fragile is None or self.threshold_broken is None:
                raise ValueError("Empirical assumption evidence requires complete severity thresholds.")
            if set(self.decision_metric_names) != {"stressed", "fragile", "broken"}:
                raise ValueError(
                    "Empirical assumption evidence requires decision metrics for "
                    "stressed, fragile, and broken thresholds."
                )
            ordered: tuple[tuple[SeverityThresholdId, float], ...] = (
                ("stressed", self.threshold_stressed),
                ("fragile", self.threshold_fragile),
                ("broken", self.threshold_broken),
            )
            for (lower_id, lower), (upper_id, upper) in zip(ordered, ordered[1:], strict=False):
                if self.decision_metric_names[lower_id] == self.decision_metric_names[upper_id] and lower > upper:
                    raise ValueError("Thresholds applied to the same assumption metric must be ordered.")
        else:
            if self.diagnosability not in {"directly_diagnosable", "partially_diagnosable"}:
                raise ValueError("Descriptive diagnostics must be empirically diagnosable.")
            if self.decision_metric_names:
                raise ValueError("Descriptive diagnostics cannot contain severity decision metrics.")
            if self.expected_status != "not_classified" or self.computed_status != "not_classified":
                raise ValueError("Descriptive diagnostics must not assert a categorical assumption status.")
            if self.expected_band != "not_classified" or self.computed_band != "not_classified":
                raise ValueError("Descriptive diagnostics must not assert a categorical severity band.")
        status_by_band = {
            "holds": "holds",
            "mild": "stressed",
            "fragile": "stressed",
            "broken": "broken",
            "not_classified": "not_classified",
        }
        if status_by_band[self.expected_band] != self.expected_status:
            raise ValueError("expected_band does not match expected_status.")
        if status_by_band[self.computed_band] != self.computed_status:
            raise ValueError("computed_band does not match computed_status.")
        metric_names = set(self.supporting_metrics)
        if self.severity_metric_name is not None:
            metric_names.add(self.severity_metric_name)
        unknown_decision_metrics = sorted(set(self.decision_metric_names.values()) - metric_names)
        if unknown_decision_metrics:
            raise ValueError(
                f"Assumption decision metrics must reference emitted public metrics: {unknown_decision_metrics!r}."
            )
        if set(self.metric_units) != metric_names or set(self.metric_public_evidence_basis) != metric_names:
            raise ValueError(
                "metric_units and metric_public_evidence_basis must exactly cover every generated metric."
            )
        if any(not str(unit) for unit in self.metric_units.values()):
            raise ValueError("Every assumption metric requires a non-empty unit.")
        for paths in self.metric_public_evidence_basis.values():
            if not paths:
                raise ValueError("Every assumption metric requires a public evidence basis.")
            for path in paths:
                _validate_public_evidence_path(path)
        factual_paths = tuple(sorted(set(self.factual_public_evidence_basis or ())))
        for path in factual_paths:
            _validate_public_evidence_path(path)
        if self.factual_public_evidence_basis is not None:
            object.__setattr__(self, "factual_public_evidence_basis", factual_paths)
        if self.severity_metric is not None and factual_paths:
            raise ValueError("Empirical assumption evidence must not declare a separate factual evidence basis.")
        return self

    def decision_metric(self, threshold_id: Literal["stressed", "fragile", "broken"]) -> tuple[str, float, float]:
        """Return the evaluator metric and boundary for one decision threshold."""

        if not self.decision_metric_names:
            raise ValueError("Design-declared assumption evidence has no empirical decision metric.")
        metric_name = self.decision_metric_names[threshold_id]
        metric_value = (
            self.severity_metric
            if metric_name == self.severity_metric_name
            else self.supporting_metrics.get(metric_name)
        )
        threshold_value = {
            "stressed": self.threshold_stressed,
            "fragile": self.threshold_fragile,
            "broken": self.threshold_broken,
        }[threshold_id]
        if metric_value is None or threshold_value is None:  # pragma: no cover - model validation is exhaustive
            raise ValueError(f"Incomplete empirical decision metric for threshold {threshold_id!r}.")
        return metric_name, float(metric_value), float(threshold_value)


class AssumptionEvidenceManifestV1(BaseModel):
    """One item-level generated assumption-evidence manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["v1"]
    schema_id: Literal["trial_benchmark_assumption_evidence_manifest_v1"]
    item_id: str = Field(..., min_length=1)
    base_case_id: str = Field(..., min_length=1)
    canonical_item_id: str = Field(..., min_length=1)
    variant_id: str = Field(..., min_length=1)
    context_tier: str = Field(..., min_length=1)
    replicate_index: int = Field(..., ge=0)
    records: tuple[AssumptionEvidenceRecordV1, ...] = Field(..., min_length=1)
    checksum: str = Field(..., min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_manifest(self) -> AssumptionEvidenceManifestV1:
        """Reject duplicate assumptions and a non-canonical embedded digest."""

        assumption_ids = tuple(record.assumption_id for record in self.records)
        if len(set(assumption_ids)) != len(assumption_ids):
            raise ValueError("Assumption-evidence records must be unique by assumption_id.")
        if assumption_ids != tuple(sorted(assumption_ids)):
            raise ValueError("Assumption-evidence records must use canonical assumption_id order.")
        payload = self.model_dump(mode="json", exclude={"checksum"}, exclude_none=True)
        if self.checksum != canonical_payload_sha256(payload):
            raise ValueError("Assumption-evidence manifest checksum mismatch.")
        return self


class _AssumptionEvidenceDomainPayloadV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["ariadne.trial_benchmark.assumption_evidence_manifest_row/v1"]
    manifest: AssumptionEvidenceManifestV1


class _AssumptionEvidenceDomainRowV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["ariadne.trial_benchmark.grader_domain_row/v1"]
    domain: Literal["assumption_evidence"]
    task_id: str = Field(min_length=1)
    payload: _AssumptionEvidenceDomainPayloadV1


def read_assumption_evidence_domains(*, release_root: Path) -> dict[str, AssumptionEvidenceManifestV1]:
    """Load item-level assumption evidence from an evaluator release."""

    path = Path(release_root) / "grader" / "domains" / "assumption_evidence.jsonl"
    manifests: dict[str, AssumptionEvidenceManifestV1] = {}
    for row in read_jsonl_models(_AssumptionEvidenceDomainRowV1, path):
        if row.task_id in manifests:
            raise ValueError(f"Duplicate assumption-evidence task_id: {row.task_id!r}.")
        manifests[row.task_id] = row.payload.manifest
    if not manifests:
        raise ValueError("Assumption-evidence domain cannot be empty.")
    return manifests


__all__ = [
    "AssumptionEvidenceManifestV1",
    "AssumptionEvidenceRecordV1",
    "read_assumption_evidence_domains",
]
