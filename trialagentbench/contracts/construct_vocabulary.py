"""Reader-facing definitions for the TrialAgentBench measurement contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

BenchmarkDefinitionId = Literal[
    "trialeval",
    "trialdev",
    "design",
    "assumption_variant",
    "context",
    "estimand",
    "reference_analysis",
    "authored_study",
    "numerical_recovery",
    "development_value",
    "uncertainty",
]


class BenchmarkDefinitionV1(BaseModel):
    """One stable reader-facing benchmark definition.

    Attributes
    ----------
    definition_id:
        Stable machine identity used by documentation and presentation layers.
    term:
        Short reader-facing term.
    definition:
        Operational meaning used across benchmark surfaces.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    definition_id: BenchmarkDefinitionId
    term: str = Field(min_length=1)
    definition: str = Field(min_length=1)


def benchmark_definitions_v1() -> tuple[BenchmarkDefinitionV1, ...]:
    """Return the canonical reader-facing benchmark vocabulary.

    Returns
    -------
    tuple[BenchmarkDefinitionV1, ...]
        Ordered definitions shared by public evidence and presentation layers.
    """

    return (
        BenchmarkDefinitionV1(
            definition_id="trialeval",
            term="TrialEval",
            definition="Recover the declared clinical signal and its uncertainty from a completed trial's records using a reproducible numerical analysis.",
        ),
        BenchmarkDefinitionV1(
            definition_id="trialdev",
            term="TrialDev",
            definition="Analyse imperfect initial evidence, author phased studies, analyse acquired records and decide whether to stop or advance an asset and regimen under public resource constraints.",
        ),
        BenchmarkDefinitionV1(
            definition_id="design",
            term="Design (D)",
            definition="The study structure determining allocation, independent units, dependence, recruitment and observation times.",
        ),
        BenchmarkDefinitionV1(
            definition_id="assumption_variant",
            term="Scientific variation (A)",
            definition="A construction descriptor of the relation between the simulated law and a conventional analysis. It is not a hidden diagnosis the analyst must report or an assigned answer-method lane.",
        ),
        BenchmarkDefinitionV1(
            definition_id="context",
            term="Evidence presentation (C)",
            definition="The data preparation and analysis specification supplied for a fixed question. C5 adds consequential source discrepancies; reconstruction is graded through numerical consequences.",
        ),
        BenchmarkDefinitionV1(
            definition_id="estimand",
            term="Estimand",
            definition="The population contrast defined by treatments, endpoint, assessment window and effect scale. Initial TrialDev targets are standardized observed associations; adjustment does not assert causal identification.",
        ),
        BenchmarkDefinitionV1(
            definition_id="reference_analysis",
            term="Reference analysis",
            definition="A reproducibility control used to check construction and grading. Its estimate from observed data can differ from the frozen reference; it is not a method the analyst must select.",
        ),
        BenchmarkDefinitionV1(
            definition_id="authored_study",
            term="Authored study",
            definition="An analyst-specified trial with treatment arms, regimens, recruitment, allocation, sites, follow-up and readouts within public phase rules. The online runtime reveals accrued records from a patient pool generated offline.",
        ),
        BenchmarkDefinitionV1(
            definition_id="numerical_recovery",
            term="Numerical recovery",
            definition="Mean loss combines normalized estimation error and interval loss against a frozen reference; lower is better. A task receives numerical loss only when all assigned results are reproduced. Missing results remain visible in completion, and incomplete panels do not support complete-panel rankings. TrialEval leads with decision consequence instead: each answer's saved script is replayed unchanged on fresh datasets of its task's world law, each replayed report drives a confirmatory-trial decision under the disclosed prior and declared economics, and that decision is valued at the true effect against the scripted reference route's decision on the same dataset; a missing or unreproduced answer, or a failed replay, drives the prior-only decision. Bounded recovery, 1/(1 + aggregate mean loss), preserves the numerical loss ordering and completeness rule at each reporting level. Method names and correction narratives are not scoring inputs.",
        ),
        BenchmarkDefinitionV1(
            definition_id="development_value",
            term="Development value",
            definition="Terminal asset benefit minus signed safety burden, incurred and unavoidable study costs, and elapsed-time cost under public sponsor preferences. Stopping retains sunk costs; an unfinished programme has no terminal decision value.",
        ),
        BenchmarkDefinitionV1(
            definition_id="uncertainty",
            term="Uncertainty",
            definition="An interval at the task's declared confidence level, reported alongside an estimate. Realized recovery does not certify repeated-sampling coverage or eliminate uncertainty about development decisions.",
        ),
    )


__all__ = ["BenchmarkDefinitionId", "BenchmarkDefinitionV1", "benchmark_definitions_v1"]
