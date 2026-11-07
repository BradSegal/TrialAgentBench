"""Smoke tests shipped with the public TrialAgentBench harness."""

from __future__ import annotations

import importlib
import json
import tomllib
from datetime import date
from pathlib import Path


def test_public_runtime_modules_import() -> None:
    """Offline modules and live runner adapters import without credentials."""
    for module_name in (
        "trialagentbench.cli",
        "trialagentbench.contracts.trace.observable",
        "trialagentbench.tools.build.build_trace_analysis_bundle",
        "trialagentbench.tools.run.trialeval",
        "trialagentbench.tools.run.canonical_trialdev",
        "trialagentbench.tools.grade.grade_trialeval",
        "trialagentbench.tools.grade.canonical_trialdev",
        "trialagentbench.grading.outcome",
        "trialagentbench.grading.reporting",
        "trialagentbench.trialdev.action_trace",
        "trialagentbench.trialdev.targets",
        "trialagentbench.trialeval.action_trace",
        "trialagentbench.trialeval.outcome_submission",
    ):
        importlib.import_module(module_name)


def test_public_entrypoint_modules_exist() -> None:
    """Every advertised command resolves to source included in the package."""
    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    missing: list[str] = []
    for entrypoint in pyproject["project"]["scripts"].values():
        module_name = str(entrypoint).split(":", maxsplit=1)[0]
        module_path = root / Path(*module_name.split("."))
        if not module_path.with_suffix(".py").is_file() and not (module_path / "__init__.py").is_file():
            missing.append(module_name)
    assert missing == []


def test_public_clinical_projection_and_scoring(tmp_path: Path) -> None:
    """Project observed clinical outcomes and grade the recovered treatment contrast."""
    import pandas as pd
    from trialagentbench_verifier.numerical_scoring import numerical_loss

    from trialagentbench.clinical_tables import write_observed_study_tables

    participants = pd.DataFrame(
        {"participant_id": [1, 2, 3, 4], "assigned_arm": ["control", "control", "active", "active"]}
    )
    analysis = participants.assign(das28=[2.0, 3.0, 1.0, 1.5])
    output = tmp_path / "clinical-tables"
    write_observed_study_tables(
        output,
        study_id="observed-rheumatoid-arthritis",
        participants=participants,
        analysis=analysis,
        endpoint_id="das28",
        endpoint_object="Disease activity in 28 joints",
        endpoint_unit="DAS28 points",
        endpoint_kind="continuous",
        assessment_time=84,
        study_start=date(2024, 1, 1),
        epochs=(("TREATMENT", 0.0, 84.0),),
    )
    observed = pd.read_parquet(output / "ADBDS.parquet")
    assert len(observed) == len(participants)
    assert observed.ADY.tolist() == [85] * 4
    means = observed.groupby("TRTP").AVAL.mean()
    estimate = float(means["active"] - means["control"])
    loss = numerical_loss(estimate=estimate, lower=-2.0, upper=0.0, target=-1.0, scale=1.0, confidence_level=0.95)
    assert estimate == -1.25
    assert loss == 1.125
    (tmp_path / "clinical-smoke.json").write_text(
        json.dumps({"participants": len(observed), "estimate": estimate, "loss": loss}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
