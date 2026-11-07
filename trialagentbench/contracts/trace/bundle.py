"""Contracts for a publication-neutral observable trace bundle."""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench.contracts.trace.observable import (
    EvidenceUseRowV1,
    ModelActionTraceEventV1,
    TraceFeatureRowV1,
    TrialDevDecisionRowV1,
)

OBSERVABLE_TRACE_TABLE_MODELS: Final[MappingProxyType[str, type[BaseModel]]] = MappingProxyType(
    {
        "action_events.csv": ModelActionTraceEventV1,
        "evidence_use.csv": EvidenceUseRowV1,
        "trialdev_decisions.csv": TrialDevDecisionRowV1,
        "unit_features.csv": TraceFeatureRowV1,
    }
)


class ObservableTraceTableV1(BaseModel):
    """One checksummed table in an observable trace bundle."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, pattern=r"^[a-z0-9_]+\.csv$")
    row_schema_id: str = Field(min_length=1)
    row_count: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ObservableTraceBundleManifestV2(BaseModel):
    """Deterministic manifest for generic trace-analysis output."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["trialagentbench.observable_trace_bundle/v2"] = "trialagentbench.observable_trace_bundle/v2"
    benchmark_suites: tuple[Literal["trialeval", "trialdev"], ...]
    model_ids: tuple[str, ...]
    run_ids: tuple[str, ...]
    tables: tuple[ObservableTraceTableV1, ...]

    @model_validator(mode="after")
    def _require_unique_sorted_inventory(self) -> ObservableTraceBundleManifestV2:
        if not self.benchmark_suites:
            raise ValueError("observable trace bundle requires at least one benchmark suite")
        if not self.model_ids or not self.run_ids:
            raise ValueError("observable trace bundle requires model and run identities")
        if tuple(sorted(set(self.benchmark_suites))) != self.benchmark_suites:
            raise ValueError("benchmark_suites must be unique and sorted")
        if tuple(sorted(set(self.model_ids))) != self.model_ids:
            raise ValueError("model_ids must be unique and sorted")
        if tuple(sorted(set(self.run_ids))) != self.run_ids:
            raise ValueError("run_ids must be unique and sorted")
        paths = tuple(table.path for table in self.tables)
        if tuple(sorted(set(paths))) != paths:
            raise ValueError("trace table paths must be unique and sorted")
        return self


__all__ = [
    "OBSERVABLE_TRACE_TABLE_MODELS",
    "ObservableTraceBundleManifestV2",
    "ObservableTraceTableV1",
]
