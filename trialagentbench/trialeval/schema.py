"""Shared data types for the TrialEval runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trialagentbench.contracts.core.trialeval_factors import TrialEvalAnalysisSpecificationV1


@dataclass
class BenchmarkItem:
    """One immutable benchmark task surface."""

    item_id: str  # Opaque identifier for the participant task
    trial_name: str
    design_tier: str  # Evaluator design label
    design_subtype: str
    assumption_tier: str  # Evaluator assumption label
    context_tier: str  # "C1" .. "C5"
    visible_dir: Path
    data_dir: Path  # visible_dir / "data" or visible_dir / "data" / "raw"
    task: dict[str, object]  # parsed task.json
    estimand_mode: str = ""
    data_preparation: str = ""
    analysis_specification: TrialEvalAnalysisSpecificationV1 = "locked_sap"

    data_version: str = "trialagentbench_v1"
    submission_contract: dict[str, object] | None = None  # parsed submission_contract.json
    reconstruction_task: dict[str, object] | None = None  # parsed reconstruction_task.json (C3/C4)
    raw_data_dir: Path | None = None  # visible_dir / "data" / "raw" when present
    workspace_data_paths: tuple[str, ...] = ()  # exact staged Parquet paths
    task_id: str = ""  # anonymised TASK id, e.g. "TASK137364"
    variant_id: str = ""  # Opaque variant identifier
    suite_dir: Path | None = None  # participant release root
