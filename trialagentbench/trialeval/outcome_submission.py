"""Bind replayed TrialEval numerical estimates to their grading-key records."""

from __future__ import annotations

from trialagentbench_verifier.execution_contracts import ReplayInputArtifactV1
from trialagentbench_verifier.numerical import NumericalReplay
from trialagentbench_verifier.trialeval import TrialEvalGradingKeyRecordV1

from trialagentbench.grading.outcome import (
    TrialEvalOutcomeGradingInputV1,
    TrialEvalPointEstimateV1,
    TrialEvalPointOracleTargetV1,
    _expected_conclusion,
)


def bind_trialeval_numerical_input(
    *,
    evaluator_record: TrialEvalGradingKeyRecordV1,
    replay: NumericalReplay,
    expected_evidence: tuple[ReplayInputArtifactV1, ...],
    target_id: str | None = None,
) -> TrialEvalOutcomeGradingInputV1:
    """Bind common CSV numbers to the assigned, repaired TrialEval point target.

    Parameters
    ----------
    evaluator_record
        Verified frozen target and declared participant artifact paths.
    replay
        Common numerical replay produced after submission commitment.
    expected_evidence
        Reporting-cutoff evidence inventory authenticated by the runner against
        this task's release, not supplied by the participant.
    target_id
        Required result to bind; omitted selects the retained primary result. The plan-as-executed result
        of a plan-given task binds with the primary question's scale.

    Returns
    -------
    TrialEvalOutcomeGradingInputV1
        Existing numerical grading input with task-owned scientific metadata.
        A missing or failed numerical result remains a failed required target.

    Raises
    ------
    ValueError
        If the target or evidence binding is inconsistent or the release still
        requires an unrepaired identified-set question.

    Examples
    --------
    A CSV row contains only target_id, estimate, lower and upper. Units, interval
    level and directional reporting are derived from the assigned question.
    """
    selected_id = target_id if target_id is not None else evaluator_record.target.binding.estimand_id
    plan = evaluator_record.plan_replay
    references = (
        (evaluator_record,)
        if plan is not None and selected_id == plan.target_id
        else tuple(
            reference
            for reference in evaluator_record.references
            if reference.target.binding.estimand_id == selected_id
        )
    )
    if len(references) != 1:
        raise ValueError("numerical result is not an assigned TrialEval target")
    # The key's wire record becomes the harness target that score records carry.
    target = TrialEvalPointOracleTargetV1.model_validate_json(references[0].target.model_dump_json())
    binding = target.binding
    if replay.target_ids != evaluator_record.reported_target_ids:
        raise ValueError("numerical replay does not bind the assigned TrialEval target")
    if sorted(replay.evidence_artifacts, key=lambda row: row.relative_path) != sorted(
        expected_evidence, key=lambda row: row.relative_path
    ):
        raise ValueError("numerical replay evidence differs from the assigned reporting cutoff")
    paths = tuple(row.relative_path for row in expected_evidence)
    if (
        len(paths) != len(set(paths))
        or not paths
        or not set(paths) <= set(evaluator_record.participant_artifact_paths)
    ):
        raise ValueError("numerical replay evidence is not declared by the assigned TrialEval task")
    estimates = replay.estimates or ()
    submitted_ids = tuple(row.target_id for row in estimates)
    if len(set(submitted_ids)) != len(submitted_ids) or not set(submitted_ids) <= set(
        evaluator_record.reported_target_ids
    ):
        raise ValueError("numerical estimates do not match the assigned TrialEval target")
    if not replay.reproducible or selected_id not in submitted_ids:
        # An unreported target of a partial answer fails alone; reported targets are still graded.
        return TrialEvalOutcomeGradingInputV1(
            context=evaluator_record.context,
            target=target,
            attempt_status="missing_required_output",
        )
    if binding.effect_measure not in (
        "mean_difference",
        "rate_ratio",
        "restricted_mean_survival_time_difference",
        "risk_difference",
    ) or binding.uncertainty_type not in ("central_confidence_interval", "repeated_confidence_interval"):
        raise ValueError("unsupported point-target metadata in the assigned TrialEval question")
    row = next(row for row in estimates if row.target_id == selected_id)
    if binding.effect_measure == "rate_ratio" and min(row.estimate, row.lower, row.upper) <= 0:
        # The CSV and execution remain retained, but logarithmic scoring cannot
        # interpret a nonpositive ratio as an eligible numerical answer.
        return TrialEvalOutcomeGradingInputV1(
            context=evaluator_record.context,
            target=target,
            attempt_status="missing_required_output",
        )
    result = TrialEvalPointEstimateV1.model_validate(
        {
            "value": row.estimate,
            "effect_measure": binding.effect_measure,
            "unit": binding.result_unit,
            "interval": {
                "lower": row.lower,
                "upper": row.upper,
                "confidence_level": binding.confidence_level,
                "interval_type": binding.uncertainty_type,
            },
            "conclusion": "no_clear_difference",
        }
    )
    result = result.model_copy(update={"conclusion": _expected_conclusion(result, binding)})
    return TrialEvalOutcomeGradingInputV1(
        context=evaluator_record.context,
        target=target,
        attempt_status="complete",
        submitted_result=result,
        evidence_eligible=True,
        reproducible=True,
    )


__all__ = [
    "bind_trialeval_numerical_input",
]
