"""Canonical reader for freshly graded TrialEval item artifacts."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from trialagentbench_verifier.scorer import current_scorer

from trialagentbench.analysis.run_identity import trialeval_run_id
from trialagentbench.contracts.core.manifest import GradeManifestV1
from trialagentbench.contracts.core.runs import TrialEvalItemResultV1
from trialagentbench.contracts.scoring.trialeval_scores import (
    TrialEvalGradedItemRowV1,
    trialeval_observed_attempt_outcome_v1,
)
from trialagentbench.io import read_json_model, sha256_file


class TrialEvalScoreSourceError(RuntimeError):
    """Raised when a canonical graded TrialEval run is incomplete or malformed."""


def trialeval_graded_item_row_v1(
    *,
    source_json_path: str,
    source_json_sha256: str,
    run_id: str,
    record: TrialEvalItemResultV1,
) -> TrialEvalGradedItemRowV1:
    """Project one validated scored item into its canonical analysis row."""

    scores = record.scores
    if scores is None:
        raise TrialEvalScoreSourceError(f"TrialEval item has no canonical grade: {source_json_path}")
    if scores.outcome.estimand_performance.status == "observed" and not scores.replay.estimates:
        raise TrialEvalScoreSourceError(
            f"TrialEval numerical outcome lacks a replayed program result: {source_json_path}"
        )
    if scores.item_id != record.item_id or scores.task_id != record.item_id:
        raise TrialEvalScoreSourceError(f"TrialEval score wrapper identity mismatch: {source_json_path}")
    if scores.model != record.run_config.model:
        raise TrialEvalScoreSourceError(
            f"TrialEval score model disagrees with the immutable run config: {source_json_path}"
        )
    if scores.agent_status != record.agent_output.status:
        raise TrialEvalScoreSourceError(
            f"TrialEval score terminal status disagrees with the immutable item result: {source_json_path}"
        )
    factors = record.run_config.task_evidence_factors[record.item_id]
    condition = record.run_config.experiment_condition
    capability = condition.provider_capability
    return TrialEvalGradedItemRowV1(
        scorer=current_scorer(),
        release_id=scores.release_id,
        participant_manifest_checksum=record.run_config.participant_manifest_checksum,
        model_id=scores.model,
        run_id=run_id,
        run_identity_sha256=record.run_config.run_identity_sha256,
        task_id=record.item_id,
        trial_id=scores.trial_id,
        base_world_id=scores.base_world_id,
        condition_id=condition.condition_id,
        request_replicate_id=condition.request_replicate_id,
        reasoning_effort=condition.reasoning.effort,
        provider_capability_sha256=(None if capability is None else capability.checksum),
        procedure_assistance=condition.procedure_assistance,
        analysis_specification=factors.analysis_specification,
        trial_name=scores.trial_name or "",
        cell_ref=scores.cell_ref,
        effect_state=scores.effect_state,
        design_tier=scores.design_tier,
        design_subtype=scores.design_subtype,
        context_tier=scores.context_tier,
        effect_measure=scores.effect_measure,
        result_unit=scores.result_unit,
        estimand_mode=scores.estimand_mode,
        agent_status=scores.agent_status,
        terminal_failure_type=record.agent_output.failure_type,
        observed_attempt_outcome=trialeval_observed_attempt_outcome_v1(
            agent_status=scores.agent_status,
            failure_type=record.agent_output.failure_type,
            outcome=scores.outcome,
        ),
        turns_used=scores.turns_used,
        outcome=scores.outcome,
        additional_results=scores.additional_results,
        target=scores.target,
        oracle_worlds=scores.oracle_worlds,
        oracle_monte_carlo_standard_error=scores.oracle_monte_carlo_standard_error,
        maximum_oracle_monte_carlo_standard_error=scores.maximum_oracle_monte_carlo_standard_error,
        replay=scores.replay,
        plan_execution=scores.plan_execution,
        source_json_path=source_json_path,
        source_json_sha256=source_json_sha256,
    )


def iter_trialeval_score_rows(run_dir: Path) -> Iterator[TrialEvalGradedItemRowV1]:
    """Yield one canonical grade row for every graded item of a stored run.

    Every scheduled item must be present unless the run's grade manifest declares the missing ones
    unscored because a spending limit censored them before they ran. Items from a recovery run may
    record a subset of the campaign's scheduled surface; any other disagreement is an error.
    """

    item_dir = Path(run_dir) / "items"
    if not item_dir.is_dir():
        raise TrialEvalScoreSourceError(f"TrialEval JSON score item directory not found: {run_dir}")
    paths = sorted(item_dir.glob("*.json"))
    if not paths:
        raise TrialEvalScoreSourceError(f"TrialEval score item directory is empty: {item_dir}")
    run_id = trialeval_run_id(run_dir)
    records = tuple(read_json_model(TrialEvalItemResultV1, path) for path in paths)
    # A recovery run reschedules some of the campaign's tasks, so its items record that narrower
    # surface. The campaign surface is the largest one; every other must be a proper subset of it.
    task_surfaces = {tuple(record.run_config.task_ids) for record in records}
    scheduled = max(task_surfaces, key=len)
    if any(surface != scheduled and not set(surface) < set(scheduled) for surface in task_surfaces):
        raise TrialEvalScoreSourceError(f"TrialEval items disagree on the scheduled task surface: {run_dir}")
    observed = tuple(record.item_id for record in records)
    if len(set(observed)) != len(observed):
        raise TrialEvalScoreSourceError(f"TrialEval run contains duplicate item identities: {run_dir}")
    # The grader scores every scheduled item except budget-censored ones, which never ran, and its
    # manifest declares that denominator; only such a declared shortfall may be absent.
    expected = scheduled
    manifest_path = Path(run_dir) / "GRADE_MANIFEST.json"
    if set(observed) < set(scheduled) and manifest_path.is_file():
        manifest = read_json_model(GradeManifestV1, manifest_path)
        if manifest.suite_task_count == len(scheduled) and manifest.graded_task_count == len(observed):
            expected = tuple(task_id for task_id in scheduled if task_id in set(observed))
    if set(observed) != set(expected):
        raise TrialEvalScoreSourceError(
            "TrialEval item artifacts do not match the scheduled task surface: "
            f"missing={sorted(set(expected) - set(observed))!r}, "
            f"unexpected={sorted(set(observed) - set(expected))!r}."
        )
    by_id = {record.item_id: (path, record) for path, record in zip(paths, records, strict=True)}
    for task_id in expected:
        path, record = by_id[task_id]
        yield trialeval_graded_item_row_v1(
            source_json_path=path.as_posix(),
            source_json_sha256=sha256_file(path),
            run_id=run_id,
            record=record,
        )


__all__ = [
    "TrialEvalScoreSourceError",
    "iter_trialeval_score_rows",
    "trialeval_graded_item_row_v1",
]
