"""Score TrialEval answers by replay at the true effect.

An answer's saved script runs unchanged on each fresh dataset of its task's data-generating process in the
replay bank (``replay_bank``). Per replay, regret is reference-action value minus answer-action value at the true effect,
using the declared decision layer and economics. Operator scoring takes the minimum over every reference
analysis on that replay. The unchanged public bank uses its primary reference analysis, giving an upper bound.
Any answer at least as valuable as some reference analysis has nonpositive operator regret. A missing or
unreproduced answer is priced as the prior-only decision on the same replays; a replay that errors or
exports no valid primary result is priced as the prior-only decision for that replay. A task without bank records
is scored once, at the truth on its original dataset against the reference route there (``single_dataset_truth``).

Replay outcomes are stored as exported (native scale), so the score at any point of the economics band is a pure
function of the outcomes, the bank records and the grading target.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import math
import shutil
import statistics
import tempfile
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_verifier.decision_consequence import (
    CONFIRMATORY_SIZE_MULTIPLES,
    DecisionActionV1,
    DecisionEconomicsV1,
    DecisionReportV1,
    optimal_action_v1,
    realised_value_v1,
)
from trialagentbench_verifier.execution import _stage_task_inputs, _Task
from trialagentbench_verifier.execution_contracts import (
    CodeExecutionLimitsV1,
    ReplayInputArtifactV1,
    ReportingInterfaceV1,
)
from trialagentbench_verifier.executor import (
    DockerCodeSession,
    analysis_entrypoint_code,
    analysis_project_artifacts,
    offline_executor_capacity,
)
from trialagentbench_verifier.io import read_json_object
from trialagentbench_verifier.layer_priors import load_layer_prior
from trialagentbench_verifier.numerical import read_numerical_table, stage_reporting_interface
from trialagentbench_verifier.replay_bank import (
    PUBLIC_REFERENCE_ROUTE_ID,
    ReplayBankRecordV1,
    ReplayBankReportV1,
    ReplayBankV1,
    load_replay_bank,
)
from trialagentbench_verifier.trialeval import TrialEvalTargetV1, _Interval, _metrics, _PointResult

logger = logging.getLogger(__name__)

CONSEQUENTIAL_LOSS_MUSD = 5.0
"""Loss at the true effect (US$M, declared economics) at which a changed stop-or-confirm action is consequential."""
ANCHOR_POLICIES = ("scripted_reference", "follow_plan", "always_naive", "always_robust", "prior_only")
"""Fixed-policy anchors; every one but the reference and prior-only is a bank route alias ``policy:<name>``."""
ECONOMICS_BAND = {
    "declared": (1.0, 1.0),
    "value_x0.5": (0.5, 1.0),
    "value_x2": (2.0, 1.0),
    "cost_x0.5": (1.0, 0.5),
    "cost_x2": (1.0, 2.0),
}
"""Disclosed economics band points: (value multiplier, cost multiplier)."""
ANSWER_COLUMNS = (
    "model",
    "task_id",
    "cell",
    "context",
    "law_id",
    "scoring",
    "reproduced",
    "replay_regret_musd",
    "replay_regret_se",
    "consequential_share",
    "answered_share",
    "normalised_absolute_error",
    "interval_coverage",
    "interval_score",
    "value_answer_musd",
    "value_prior_only_musd",
    "value_truth_known_musd",
    "failure_cause",
    "value_multiplier",
    "cost_multiplier",
    "material_decision_error_rate",
    "material_decision_error_rate_se",
)
"""Columns of ``trialeval_answers.csv``, in order: one row per answerer, task and economics band point."""

Scoring = Literal["replay", "single_dataset_truth", "failure"]
FailureCause = Literal["no_answer", "unreproduced", "wrong_question"]
"""Why an answer is a failure: no primary estimate and interval, a saved script that did not reproduce it on the
original dataset (the execution proof), or a reproduced answer to a different question (the grader's
``wrong_question`` origin)."""
ReplayStatus = Literal["answered", "execution_error", "timeout", "missing_result", "invalid_result"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ReplayOutcomeV1(_Model):
    """One saved script's run on one bank dataset: its primary result as exported (native scale)."""

    replay_index: int = Field(ge=1)
    status: ReplayStatus
    estimate: float | None = None
    lower: float | None = None
    upper: float | None = None
    error: str | None = None

    @model_validator(mode="after")
    def _answered_iff_reported(self) -> Self:
        reported = None not in (self.estimate, self.lower, self.upper)
        if reported != (self.status == "answered"):
            raise ValueError("a replay reports a primary result exactly when it answered")
        return self


class AnswerReplaysV1(_Model):
    """The replay outcomes of one saved script on every bank dataset of its task (the compute cache record)."""

    schema_id: Literal["trialagentbench.trialeval.answer_replays/v1"] = "trialagentbench.trialeval.answer_replays/v1"
    task_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bank_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    replays: tuple[ReplayOutcomeV1, ...] = Field(min_length=1)


class ReplayDecisionValuesV1(_Model):
    """Values at the true effect of three decisions on one replay.

    Parameters
    ----------
    replay_index
        Bank replay index, starting at one.
    value_answer_musd, value_prior_only_musd, value_truth_known_musd
        Answer-action, prior-only-action and best true-effect action values in US$ millions,
        using the same declared economics and action set.

    Raises
    ------
    pydantic.ValidationError
        If the index is invalid, a value is nonfinite or unknown fields are supplied.

    Examples
    --------
    ``score_replays_v1`` retains one such value record in ``replay_values`` for every bank replay.
    """

    replay_index: int = Field(ge=1)
    value_answer_musd: float
    value_prior_only_musd: float
    value_truth_known_musd: float


class _ReplayPriceV1(ReplayDecisionValuesV1):
    regret_musd: float
    consequential: bool
    material_decision_error: bool


class ReplayScoreV1(_Model):
    """One answer's regret and action values at one economics point, averaged over its replays (US$M)."""

    scoring: Scoring
    reproduced: bool
    replay_regret_musd: float
    value_answer_musd: float
    value_prior_only_musd: float
    value_truth_known_musd: float
    replay_values: tuple[ReplayDecisionValuesV1, ...]
    replay_regret_se: float | None = Field(default=None, ge=0, description="Over replays; None for one dataset.")
    consequential_share: float = Field(ge=0, le=1)
    answered_share: float = Field(ge=0, le=1)
    normalised_absolute_error: float | None = Field(default=None, ge=0)
    interval_coverage: float | None = Field(default=None, ge=0, le=1)
    interval_score: float | None = Field(default=None, ge=0, description="Proper interval score in grading SDs.")
    material_decision_error_rate: float = Field(ge=0, le=1)
    """Share of all replays losing at least US$5M times the economics point's value multiplier against the reference, including sizing losses and prior-only decisions."""
    material_decision_error_rate_se: float | None = Field(default=None, ge=0)
    """Sample standard error of the replay indicator mean; None for one dataset."""


class TrialEvalAnswerRowV1(ReplayScoreV1):
    """One row of ``trialeval_answers.csv``: a model's answer, or ``policy:<name>`` for an anchor."""

    model: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    cell: str = Field(min_length=1)
    context: str = Field(min_length=1)
    law_id: str = Field(min_length=1)
    failure_cause: FailureCause | None = None
    value_multiplier: float = Field(gt=0)
    cost_multiplier: float = Field(gt=0)
    reference_posterior_regret_musd: float | None = Field(
        default=None,
        description="Reference-posterior regret retained in grade records and excluded from headline reporting.",
    )

    @model_validator(mode="after")
    def _cause_iff_failure(self) -> Self:
        if (self.failure_cause is not None) != (self.scoring == "failure"):
            raise ValueError("a failure cause is given exactly for a failed answer")
        return self


class TrialEvalReplayEvidenceV1(_Model):
    """A loaded bank and the replay outcomes of every answer scored against it."""

    bank: ReplayBankV1
    bank_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    answers: tuple[AnswerReplaysV1, ...] = ()

    @model_validator(mode="after")
    def _same_bank(self) -> Self:
        if any(item.bank_manifest_sha256 != self.bank_manifest_sha256 for item in self.answers):
            raise ValueError("answer replays were run against a different replay bank")
        keys = [(item.task_id, item.source_sha256) for item in self.answers]
        if len(set(keys)) != len(keys):
            raise ValueError("answer replays repeat one task and saved source")
        return self

    def replays_for(self, task_id: str, source_sha256: str) -> AnswerReplaysV1 | None:
        """Return the replay outcomes of one saved source on one task, if run."""
        return next(
            (item for item in self.answers if (item.task_id, item.source_sha256) == (task_id, source_sha256)),
            None,
        )


def saved_source_sha256(artifacts: Iterable[ReplayInputArtifactV1]) -> str:
    """Return the identity of a saved analysis project: the digest of its sorted (path, SHA-256) pairs.

    Parameters
    ----------
    artifacts
        The project's source files, as ``analysis_project_artifacts`` binds them.

    Returns
    -------
    str
        A SHA-256 hex digest, the same for byte-identical projects.

    Examples
    --------
    >>> row = ReplayInputArtifactV1(relative_path="scratch/analysis.py", sha256="0" * 64, byte_length=1)
    >>> len(saved_source_sha256((row,)))
    64
    """
    pairs = sorted((item.relative_path, item.sha256) for item in artifacts)
    return hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode()).hexdigest()


def _comparison(value: float, target: TrialEvalTargetV1) -> float:
    return math.log(value) if target.binding.effect_measure == "rate_ratio" else value


def _decision_report(estimate: float, lower: float, upper: float) -> DecisionReportV1:
    # The decision layer uses the estimate and the interval width separately (as campaign grading does).
    return DecisionReportV1(estimate=estimate, lower=-(upper - lower) / 2, upper=(upper - lower) / 2)


def _bank_report(report: ReplayBankReportV1 | None) -> DecisionReportV1 | None:
    return None if report is None else _decision_report(report.estimate, report.lower, report.upper)


def _outcome_report(outcome: ReplayOutcomeV1, target: TrialEvalTargetV1) -> DecisionReportV1 | None:
    """Return a replay's comparison-scale report, or None when it has no usable primary result."""
    if outcome.estimate is None or outcome.lower is None or outcome.upper is None:
        return None
    native = (outcome.estimate, outcome.lower, outcome.upper)
    if target.binding.effect_measure == "rate_ratio" and min(native) <= 0:
        return None  # a nonpositive ratio is an invalid export, priced as a missing answer
    return _decision_report(*(_comparison(value, target) for value in native))


def economics_v1(target: TrialEvalTargetV1, *, value: float = 1.0, cost: float = 1.0) -> DecisionEconomicsV1:
    """Return a target's declared decision economics at one band point.

    Parameters
    ----------
    target
        A grading target carrying its consequence reference.
    value, cost
        Multipliers of the value of success and of both confirmation costs.

    Returns
    -------
    DecisionEconomicsV1
        The scaled economics.

    Raises
    ------
    ValueError
        If the target declares no decision consequence.
    """
    if target.consequence is None:
        raise ValueError("a target without a decision consequence cannot be priced")
    declared = target.consequence.economics
    return DecisionEconomicsV1(
        value_of_success_musd=declared.value_of_success_musd,
        replicate_cost_musd=declared.replicate_cost_musd,
        fixed_trial_cost_musd=declared.fixed_trial_cost_musd,
    ).scaled(value=value, cost=cost)


@lru_cache(maxsize=8192)
def best_value_at_truth(benefit: float, *, scale: float, economics: DecisionEconomicsV1) -> float:
    """Return the largest expected value any declared action attains when the true benefit is known.

    Parameters
    ----------
    benefit
        The true benefit on the comparison scale (positive favours treatment).
    scale
        The target's grading normaliser.
    economics
        Decision economics.

    Returns
    -------
    float
        US$M; 0 (stop) when no confirmation pays.
    """
    values = [0.0]
    for multiple in CONFIRMATORY_SIZE_MULTIPLES:
        action = DecisionActionV1(confirm=True, size_multiple=multiple, expected_value_musd=0.0)
        values.append(realised_value_v1(action, true_benefit=benefit, scale=scale, economics=economics))
    return max(values)


class _Pricer:
    """Values actions at one truth under one target's decision layer and economics."""

    def __init__(self, target: TrialEvalTargetV1, *, value: float, cost: float) -> None:
        self.target = target
        self.economics = economics_v1(target, value=value, cost=cost)
        self.bar = CONSEQUENTIAL_LOSS_MUSD * value
        self.layer_prior = load_layer_prior(target.binding.estimand_id)
        self.prior = self.action(None)

    def action(self, report: DecisionReportV1 | None) -> DecisionActionV1:
        return optimal_action_v1(
            report,
            scale=self.target.reference_sampling_standard_deviation,
            favorable_direction=self.target.binding.favorable_direction,
            economics=self.economics,
            prior=self.layer_prior,
        )

    def replay(self, answer: DecisionReportV1 | None, record: ReplayBankRecordV1) -> _ReplayPriceV1:
        """Price one bank replay, refusing incomplete operator credit metadata."""
        if (
            self.target.consequence is not None
            and record.reference_route_id != PUBLIC_REFERENCE_ROUTE_ID
            and len(record.reference_route_ids) != len(self.target.consequence.credited)
        ):
            raise ValueError("replay bank must identify every credited route; refresh legacy multi-route records")
        return self.against_credited(
            answer,
            (_bank_report(record.routes[route]) for route in record.reference_route_ids),
            truth=record.truth,
            replay_index=record.replay_index,
        )

    def against_credited(
        self,
        answer: DecisionReportV1 | None,
        references: Iterable[DecisionReportV1 | None],
        *,
        truth: float,
        replay_index: int = 1,
    ) -> _ReplayPriceV1:
        """Price the answer against every reference analysis, retaining both truth-based value anchors."""
        benefit = truth if self.target.binding.favorable_direction == "higher" else -truth
        scale = self.target.reference_sampling_standard_deviation
        own = self.prior if answer is None else self.action(answer)
        own_value = realised_value_v1(own, true_benefit=benefit, scale=scale, economics=self.economics)
        prices = []
        for reference in references:
            other = self.prior if reference is None else self.action(reference)
            loss = realised_value_v1(other, true_benefit=benefit, scale=scale, economics=self.economics) - own_value
            prices.append((loss, own.confirm != other.confirm))
        loss = min(price for price, _ in prices)
        consequential = all(differs for _, differs in prices) and loss >= self.bar
        return _ReplayPriceV1(
            replay_index=replay_index,
            regret_musd=loss,
            consequential=consequential,
            material_decision_error=loss >= self.bar,
            value_answer_musd=own_value,
            value_prior_only_musd=realised_value_v1(
                self.prior, true_benefit=benefit, scale=scale, economics=self.economics
            ),
            value_truth_known_musd=best_value_at_truth(benefit, scale=scale, economics=self.economics),
        )


def _accuracy(
    target: TrialEvalTargetV1, reports: Iterable[tuple[float, tuple[float, float, float]]]
) -> tuple[float | None, float | None, float | None]:
    """Return mean normalised absolute error, coverage and interval score of native reports at their truths."""
    errors, covered, scores = [], [], []
    scale = target.reference_sampling_standard_deviation
    for truth, (estimate, lower, upper) in reports:
        native_truth = math.exp(truth) if target.binding.effect_measure == "rate_ratio" else truth
        metrics = _metrics(
            target.model_copy(update={"value": native_truth}),
            _PointResult(
                kind="point",
                value=estimate,
                effect_measure=target.binding.effect_measure,
                unit=target.binding.result_unit,
                interval=_Interval(lower=lower, upper=upper, confidence_level=target.confidence_level),
                conclusion="no_clear_difference",
            ),
        )
        error, score = metrics["normalized_absolute_error"], metrics["proper_interval_score"]
        if isinstance(error, float) and isinstance(score, float):
            errors.append(error)
            covered.append(float(bool(metrics["interval_covers_oracle"])))
            scores.append(score / scale)
    if not errors:
        return None, None, None
    return statistics.fmean(errors), statistics.fmean(covered), statistics.fmean(scores)


def _summary(
    scoring: Scoring,
    *,
    reproduced: bool,
    priced: list[_ReplayPriceV1],
    answered: int,
    accuracy: tuple[float | None, float | None, float | None],
) -> ReplayScoreV1:
    losses = [item.regret_musd for item in priced]
    material_errors = [float(item.material_decision_error) for item in priced]
    error, coverage, interval = accuracy
    return ReplayScoreV1(
        scoring=scoring,
        reproduced=reproduced,
        replay_regret_musd=statistics.fmean(losses),
        value_answer_musd=statistics.fmean(item.value_answer_musd for item in priced),
        value_prior_only_musd=statistics.fmean(item.value_prior_only_musd for item in priced),
        value_truth_known_musd=statistics.fmean(item.value_truth_known_musd for item in priced),
        replay_values=tuple(
            ReplayDecisionValuesV1.model_validate(item.model_dump(include=set(ReplayDecisionValuesV1.model_fields)))
            for item in priced
        ),
        replay_regret_se=statistics.stdev(losses) / math.sqrt(len(losses)) if len(losses) > 1 else None,
        consequential_share=statistics.fmean(float(item.consequential) for item in priced),
        answered_share=answered / len(priced),
        normalised_absolute_error=error,
        interval_coverage=coverage,
        interval_score=interval,
        material_decision_error_rate=statistics.fmean(material_errors),
        material_decision_error_rate_se=(
            statistics.stdev(material_errors) / math.sqrt(len(priced)) if len(priced) > 1 else None
        ),
    )


def score_replays_v1(
    target: TrialEvalTargetV1,
    records: tuple[ReplayBankRecordV1, ...],
    replays: AnswerReplaysV1 | None,
    *,
    value: float = 1.0,
    cost: float = 1.0,
) -> ReplayScoreV1:
    """Score one answer on its task's bank replays.

    Parameters
    ----------
    target
        The task's grading target: scale, favourable direction and declared economics.
    records
        The task's bank records, in replay order.
    replays
        The saved script's outcomes on those records; None for a failure (no answer, or not reproduced), which
        is priced as the prior-only decision on every replay.
    value, cost
        The economics band point; the consequential bar scales with ``value``.

    Returns
    -------
    ReplayScoreV1
        Mean minimum regret against the reference analyses and its standard error over replays, the
        consequential and answered shares, and accuracy over answered replays.

    Raises
    ------
    ValueError
        If the outcomes do not cover exactly the task's replay indices.
    """
    pricer = _Pricer(target, value=value, cost=cost)
    if replays is None:
        priced = [pricer.replay(None, record) for record in records]
        return _summary("failure", reproduced=False, priced=priced, answered=0, accuracy=(None, None, None))
    if replays.task_id != records[0].task_id or [item.replay_index for item in replays.replays] != [
        record.replay_index for record in records
    ]:
        raise ValueError(f"replay outcomes do not cover the bank replays of task {records[0].task_id}")
    priced, answered = [], []
    for record, outcome in zip(records, replays.replays, strict=True):
        report = _outcome_report(outcome, target)
        priced.append(pricer.replay(report, record))
        if report is not None:
            answered.append((record.truth, (outcome.estimate, outcome.lower, outcome.upper)))
    return _summary(
        "replay",
        reproduced=True,
        priced=priced,
        answered=len(answered),
        accuracy=_accuracy(target, answered),  # type: ignore[arg-type]
    )


def score_single_dataset_v1(
    target: TrialEvalTargetV1,
    answer: tuple[float, float, float] | None,
    *,
    value: float = 1.0,
    cost: float = 1.0,
) -> ReplayScoreV1:
    """Score an answer at the truth on its original dataset against the reference route there.

    Used only for a task without bank records. Regret is the minimum across all reference analyses of the
    grading target.

    Parameters
    ----------
    target
        The task's grading target, carrying the truth and the reference route's report.
    answer
        The reproduced answer (estimate, lower, upper) on the native scale, or None for a failure.
    value, cost
        The economics band point.

    Returns
    -------
    ReplayScoreV1
        One-dataset regret (no standard error), consequential flag as a share, and the answer's accuracy.

    Raises
    ------
    ValueError
        If the target declares no decision consequence.
    """
    if target.consequence is None:
        raise ValueError("a target without a decision consequence cannot be priced")
    pricer = _Pricer(target, value=value, cost=cost)
    references = tuple(
        _decision_report(*(_comparison(item, target) for item in credited)) for credited in target.consequence.credited
    )
    report = (
        None
        if answer is None
        else _outcome_report(
            ReplayOutcomeV1(replay_index=1, status="answered", estimate=answer[0], lower=answer[1], upper=answer[2]),
            target,
        )
    )
    truth = _comparison(target.value, target)
    priced = [pricer.against_credited(report, references, truth=truth)]
    accuracy = (None, None, None) if report is None or answer is None else _accuracy(target, [(truth, answer)])
    return _summary(
        "failure" if answer is None else "single_dataset_truth",
        reproduced=answer is not None,
        priced=priced,
        answered=int(report is not None),
        accuracy=accuracy,
    )


def anchor_scores_v1(
    target: TrialEvalTargetV1, records: tuple[ReplayBankRecordV1, ...], *, value: float = 1.0, cost: float = 1.0
) -> dict[str, ReplayScoreV1]:
    """Score every fixed-policy anchor on a task's bank replays from the bank's route reports.

    Parameters
    ----------
    target
        The task's grading target.
    records
        The task's bank records.
    value, cost
        The economics band point.

    Returns
    -------
    dict of str to ReplayScoreV1
        One score per anchor in ``ANCHOR_POLICIES``. The primary reference analysis has nonpositive operator
        regret; prior-only is scored as a failure (no answer), like a missing answer.

    Raises
    ------
    ValueError
        If the records come from the public bank, which carries no policy, or a record lacks a ``policy:<name>``
        route alias.
    """
    if any(set(record.routes) == {PUBLIC_REFERENCE_ROUTE_ID} for record in records):
        raise ValueError(
            "fixed-policy anchors need the operator's full replay bank; the public bank records only the reference"
        )
    pricer = _Pricer(target, value=value, cost=cost)
    scores = {}
    for policy in ANCHOR_POLICIES:
        priced, answered = [], []
        for record in records:
            if policy == "prior_only":
                report = None
            elif policy == "scripted_reference":
                report = record.routes[record.reference_route_id]
            elif f"policy:{policy}" in record.routes:
                report = record.routes[f"policy:{policy}"]
            else:
                raise ValueError(f"replay bank record {record.task_id}/{record.replay_index} lacks policy:{policy}")
            priced.append(pricer.replay(_bank_report(report), record))
            if report is not None:
                native = tuple(
                    math.exp(item) if target.binding.effect_measure == "rate_ratio" else item
                    for item in (report.estimate, report.lower, report.upper)
                )
                answered.append((record.truth, native))
        # Prior-only gives no answer: it is the failure price, scored and labelled as a failed answer is.
        scores[policy] = _summary(
            "failure" if policy == "prior_only" else "replay",
            reproduced=policy != "prior_only",
            priced=priced,
            answered=len(answered),
            accuracy=_accuracy(target, answered),  # type: ignore[arg-type]
        )
    return scores


def _stage_replay(
    task_dir: Path, workspace: Path, entrypoint: str, destination: Path, interface: ReportingInterfaceV1
) -> tuple[str, ...]:
    """Present one bank dataset as the harness presents its task, with the saved project in scratch/."""
    task = _Task.model_validate(read_json_object(task_dir / "task.json"))
    destination.mkdir(parents=True)
    _stage_task_inputs(task_dir, destination, stream="trialeval", task=task, task_root=True)
    (destination / "scratch").mkdir()
    for artifact in analysis_project_artifacts(workspace / "scratch" / entrypoint):
        target = destination / artifact.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(workspace / artifact.relative_path, target)
    stage_reporting_interface(destination, task.target_ids, interface=interface)
    return task.target_ids


def require_saved_source_v1(
    workspace: Path, entrypoint: Literal["analysis.py", "analysis.R"], *, task_id: str, source_sha256: str
) -> None:
    """Refuse a saved analysis project that differs from the graded source.

    Parameters
    ----------
    workspace
        The saved participant workspace holding ``scratch/<entrypoint>`` and its helpers.
    entrypoint
        The saved analysis entrypoint.
    task_id
        The answered task, named in the error.
    source_sha256
        ``saved_source_sha256`` of the graded project.

    Raises
    ------
    ValueError
        If the saved project's files differ from ``source_sha256``.
    """
    if saved_source_sha256(analysis_project_artifacts(workspace / "scratch" / entrypoint)) != source_sha256:
        raise ValueError(f"saved analysis project for {task_id} differs from its graded source")


def replay_once_v1(
    *,
    bank: ReplayBankV1,
    task_id: str,
    replay_index: int,
    primary_target_id: str,
    workspace: Path,
    entrypoint: Literal["analysis.py", "analysis.R"],
    image: str,
    limits: CodeExecutionLimitsV1,
    interface: ReportingInterfaceV1 = "v1",
    exclusive: bool = False,
) -> ReplayOutcomeV1:
    """Run one saved script unchanged, once, on one bank dataset of its task.

    A replay that exceeds the time limit is reported as ``timeout``; callers in an offline grading scope
    rerun it once with ``exclusive`` (``replay_answer_v1``, or after the parallel replays in a campaign
    finish), so host load is not scored as the script's failure.

    Parameters
    ----------
    bank
        The loaded replay bank.
    task_id, replay_index
        The bank dataset.
    primary_target_id
        The task's primary target, whose exported row is the replay's answer.
    workspace
        The saved participant workspace holding ``scratch/<entrypoint>`` and its helpers.
    entrypoint
        The saved analysis entrypoint.
    image, limits
        The release executor image and its limits.
    interface
        The reporting helpers of the run condition.
    exclusive
        Hold every executor slot of the offline grading scope while the script runs.

    Returns
    -------
    ReplayOutcomeV1
        The replay's primary result; errors, timeouts and missing or invalid exports are recorded, not raised.
    """
    language: Literal["python", "r"] = "python" if entrypoint == "analysis.py" else "r"
    with tempfile.TemporaryDirectory(prefix="trialeval-replay-") as temporary:
        root = Path(temporary) / "workspace"
        target_ids = _stage_replay(bank.task_root(task_id, replay_index), workspace, entrypoint, root, interface)
        session = DockerCodeSession(root, image=image, limits=limits, exclusive=exclusive)
        try:
            execution = session.execute_result(analysis_entrypoint_code(entrypoint, language), language=language)
            session.snapshot_scratch()
        finally:
            session.close()
        if execution.status == "timeout":
            return ReplayOutcomeV1(replay_index=replay_index, status="timeout", error=execution.output[-300:])
        if execution.status != "success":
            return ReplayOutcomeV1(replay_index=replay_index, status="execution_error", error=execution.output[-300:])
        result = root / "scratch" / "estimates.csv"
        if not result.exists():
            return ReplayOutcomeV1(replay_index=replay_index, status="missing_result")
        try:
            rows = read_numerical_table(result, target_ids=target_ids)
        except (ValueError, csv.Error, UnicodeDecodeError) as error:
            return ReplayOutcomeV1(replay_index=replay_index, status="invalid_result", error=str(error)[-300:])
    primary = next((row for row in rows if row.target_id == primary_target_id), None)
    if primary is None:
        return ReplayOutcomeV1(replay_index=replay_index, status="missing_result", error="primary target not reported")
    return ReplayOutcomeV1(
        replay_index=replay_index,
        status="answered",
        estimate=primary.estimate,
        lower=primary.lower,
        upper=primary.upper,
    )


def replay_answer_v1(
    *,
    bank: ReplayBankV1,
    bank_manifest_sha256: str,
    task_id: str,
    primary_target_id: str,
    workspace: Path,
    entrypoint: Literal["analysis.py", "analysis.R"],
    source_sha256: str,
    image: str,
    limits: CodeExecutionLimitsV1,
    interface: ReportingInterfaceV1 = "v1",
) -> AnswerReplaysV1:
    """Run one saved script unchanged on every bank dataset of its task, one replay after another.

    Parameters
    ----------
    bank
        The loaded replay bank.
    bank_manifest_sha256
        The bank's verified manifest digest, recorded with the outcomes.
    task_id
        The answered task.
    primary_target_id
        The task's primary target, whose exported row is the replay's answer.
    workspace
        The saved participant workspace holding ``scratch/<entrypoint>`` and its helpers.
    entrypoint
        The saved analysis entrypoint.
    source_sha256
        ``saved_source_sha256`` of the graded project; the saved files must still match it.
    image, limits
        The release executor image and its limits.
    interface
        The reporting helpers of the run condition.

    Returns
    -------
    AnswerReplaysV1
        One outcome per replay (``replay_once_v1``); in an offline grading scope a timeout is rerun once
        exclusively (``rerun_timeout_v1``).

    Raises
    ------
    ValueError
        If the saved project differs from the graded source or the bank has no records for the task.
    """
    require_saved_source_v1(workspace, entrypoint, task_id=task_id, source_sha256=source_sha256)
    records = bank.records.get(task_id)
    if not records:
        raise ValueError(f"replay bank has no records for task {task_id}")
    outcomes = []
    for record in records:
        arguments = {
            "bank": bank,
            "task_id": task_id,
            "replay_index": record.replay_index,
            "primary_target_id": primary_target_id,
            "workspace": workspace,
            "entrypoint": entrypoint,
            "image": image,
            "limits": limits,
            "interface": interface,
        }
        outcome = replay_once_v1(**arguments)  # type: ignore[arg-type]
        if outcome.status == "timeout" and offline_executor_capacity():
            outcome = rerun_timeout_v1(**arguments)
        outcomes.append(outcome)
    return AnswerReplaysV1(
        task_id=task_id,
        source_sha256=source_sha256,
        bank_manifest_sha256=bank_manifest_sha256,
        replays=tuple(outcomes),
    )


def rerun_timeout_v1(**arguments: object) -> ReplayOutcomeV1:
    """Rerun a timed-out replay once from a fresh workspace holding every executor slot.

    Offline grading shares the host across parallel replays, so a timeout there can be load rather than the
    script; only a script that cannot finish within the release limit alone still times out.

    Parameters
    ----------
    **arguments
        The ``replay_once_v1`` arguments of the timed-out replay.

    Returns
    -------
    ReplayOutcomeV1
        The exclusive rerun's outcome, which replaces the timed-out one.
    """
    logger.warning(
        "Rerunning replay %s/%02d exclusively after a timeout", arguments["task_id"], arguments["replay_index"]
    )
    return replay_once_v1(**arguments, exclusive=True)  # type: ignore[arg-type]


def answer_replays_path(directory: Path, *, bank_manifest_sha256: str, task_id: str, source_sha256: str) -> Path:
    """Return where one answer's replay outcomes are stored, keyed by bank, task and saved source.

    Parameters
    ----------
    directory
        The campaign's replay store.
    bank_manifest_sha256, task_id, source_sha256
        The outcomes' key.

    Returns
    -------
    Path
        ``directory/<bank manifest SHA-256>/<task_id>/<source SHA-256>.json``.
    """
    return directory / bank_manifest_sha256 / task_id / f"{source_sha256}.json"


def load_replay_evidence_v1(
    bank_root: Path, *, expected_manifest_sha256: str, expected_release_id: str, replays: Path | None
) -> TrialEvalReplayEvidenceV1:
    """Load a replay bank once and every stored answer replay against it.

    Parameters
    ----------
    bank_root
        The replay bank directory.
    expected_manifest_sha256
        The published SHA-256 of the bank's ``manifest.json``.
    expected_release_id
        The release of the graded answers; the bank must extend it.
    replays
        A replay store (``answer_replays_path`` layout), or None for no stored outcomes.

    Returns
    -------
    TrialEvalReplayEvidenceV1
        The bank and the outcomes stored for it.

    Raises
    ------
    ValueError
        If the bank differs from its digest or extends another release, or a stored record names another key
        than its path.
    """
    bank = load_replay_bank(bank_root, expected_manifest_sha256=expected_manifest_sha256)
    if bank.manifest.release_id != expected_release_id:
        raise ValueError(
            f"replay bank extends {bank.manifest.release_id}, not the graded release {expected_release_id}"
        )
    answers = () if replays is None else read_answer_replays_v1(replays, bank_manifest_sha256=expected_manifest_sha256)
    return TrialEvalReplayEvidenceV1(bank=bank, bank_manifest_sha256=expected_manifest_sha256, answers=answers)


def read_answer_replays_v1(directory: Path, *, bank_manifest_sha256: str) -> tuple[AnswerReplaysV1, ...]:
    """Read every answer's stored replay outcomes against one bank.

    Parameters
    ----------
    directory
        A replay store in the ``answer_replays_path`` layout.
    bank_manifest_sha256
        The bank whose outcomes are read.

    Returns
    -------
    tuple of AnswerReplaysV1
        The stored outcomes, in path order.

    Raises
    ------
    ValueError
        If a stored record names another key than its path.
    """
    answers = []
    for path in sorted((directory / bank_manifest_sha256).glob("*/*.json")):
        item = AnswerReplaysV1.model_validate_json(path.read_bytes())
        if (item.bank_manifest_sha256, item.task_id, item.source_sha256) != (
            bank_manifest_sha256,
            path.parent.name,
            path.stem,
        ):
            raise ValueError(f"stored answer replays differ from their key: {path}")
        answers.append(item)
    return tuple(answers)


def answers_csv(rows: Iterable[TrialEvalAnswerRowV1]) -> str:
    """Return ``trialeval_answers.csv``: one row per answer and anchor, in ``ANSWER_COLUMNS`` order.

    Parameters
    ----------
    rows
        Answer and anchor rows.

    Returns
    -------
    str
        CSV text with a header; missing values are empty.
    """
    table = io.StringIO(newline="")
    writer = csv.writer(table, lineterminator="\n")
    writer.writerow(ANSWER_COLUMNS)
    for row in rows:
        payload = row.model_dump()
        writer.writerow("" if payload[name] is None else payload[name] for name in ANSWER_COLUMNS)
    return table.getvalue()


__all__ = [
    "ANCHOR_POLICIES",
    "ANSWER_COLUMNS",
    "CONSEQUENTIAL_LOSS_MUSD",
    "ECONOMICS_BAND",
    "FailureCause",
    "AnswerReplaysV1",
    "ReplayOutcomeV1",
    "ReplayScoreV1",
    "TrialEvalAnswerRowV1",
    "TrialEvalReplayEvidenceV1",
    "anchor_scores_v1",
    "answer_replays_path",
    "answers_csv",
    "best_value_at_truth",
    "economics_v1",
    "load_replay_evidence_v1",
    "read_answer_replays_v1",
    "replay_answer_v1",
    "replay_once_v1",
    "rerun_timeout_v1",
    "require_saved_source_v1",
    "saved_source_sha256",
    "score_replays_v1",
    "score_single_dataset_v1",
]
