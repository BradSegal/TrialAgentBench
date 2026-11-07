"""Executable capability boundary for the two public benchmark streams."""

from __future__ import annotations

import importlib
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator


class BenchmarkCapability(BaseModel):
    """One runnable and gradeable benchmark stream."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stream: Literal["trialeval", "trialdev"]
    task: str
    run_entry_point: str
    grade_entry_point: str
    participant_contract: str
    submission_contract: str
    score_contract: str
    support_boundary: str

    @model_validator(mode="after")
    def validate_entry_points(self) -> Self:
        """Require the declared run and grade functions to resolve."""

        for path in (self.run_entry_point, self.grade_entry_point):
            module_name, attribute = path.rsplit(".", 1)
            resolved = getattr(importlib.import_module(module_name), attribute)
            if not callable(resolved):
                raise TypeError(f"benchmark capability target is not callable: {path}")
        return self


BENCHMARK_CAPABILITIES: tuple[BenchmarkCapability, ...] = (
    BenchmarkCapability(
        stream="trialeval",
        task="Recover the question's numerical signal and uncertainty from participant-level trial evidence.",
        run_entry_point="trialagentbench.tools.run.trialeval.main",
        grade_entry_point="trialagentbench.tools.grade.grade_trialeval.grade_trialeval_run",
        participant_contract="trialagentbench.contracts.release.TrialEvalParticipantManifestV1",
        submission_contract="trialagentbench_verifier.numerical.NumericalEstimate",
        score_contract="trialagentbench.contracts.scoring.TrialEvalItemScoresV1",
        support_boundary=(
            "The released trial designs and evidence contexts, with task-owned targets and reproducible estimates.csv output."
        ),
    ),
    BenchmarkCapability(
        stream="trialdev",
        task="Analyse initial evidence, author phased trials and make development decisions under resource constraints.",
        run_entry_point="trialagentbench.tools.run.canonical_trialdev.main",
        grade_entry_point="trialagentbench.tools.grade.canonical_trialdev.main",
        participant_contract="trialagentbench.contracts.trialdev.release.TrialDevTaskV1",
        submission_contract="trialagentbench_verifier.numerical.NumericalEstimate",
        score_contract="trialagentbench.contracts.trialdev.run.TrialDevTaskRunV1",
        support_boundary=(
            "Released clinical worlds, authored study/recruitment actions, accrued evidence, numerical reports, "
            "stop/advance decisions and task-owned budgets and clocks."
        ),
    ),
)


__all__ = ["BENCHMARK_CAPABILITIES", "BenchmarkCapability"]
