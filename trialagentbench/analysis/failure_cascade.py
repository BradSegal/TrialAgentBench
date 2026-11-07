"""Ground-reference-linked failure-cascade diagnostics."""

from __future__ import annotations

from trialagentbench.contracts.trace.observable import (
    FailureCascadeRowV1,
    ObservableFailureTypeV1,
    TraceFeatureRowV1,
)


def phase_failure_cascade(feature: TraceFeatureRowV1, *, hidden_access: bool = False) -> FailureCascadeRowV1:
    """Classify the first observable failure for one benchmark phase or item."""
    if feature.score_link_id is None:
        raise ValueError("Trace feature score_link_id is required for phase cascades")
    endpoint_failed = feature.endpoint_state == "failed" or feature.endpoint_valid is False
    if hidden_access:
        failure: ObservableFailureTypeV1 = "hidden_or_grader_access"
    elif feature.endpoint_state == "evaluator_error":
        failure = "evaluator_failure"
    elif feature.endpoint_state == "not_attempted_noncompletion":
        failure = "model_noncompletion"
    elif feature.endpoint_state in {
        "score_export_absent",
        "submission_absent",
        "submission_present_score_absent",
        "score_present_submission_absent",
        "source_inconsistent_requires_adjudication",
    }:
        failure = "missing_output"
    elif not feature.submitted_answer and feature.endpoint_state not in {
        "not_reached_after_stop",
        "not_attempted_noncompletion",
        "not_scoreable_trace_only",
    }:
        failure = "missing_output"
    elif feature.endpoint_valid is False or feature.endpoint_state == "failed":
        failure = "endpoint_invalid_or_unusable"
    else:
        failure = "none_observed"
    return FailureCascadeRowV1(
        benchmark=feature.benchmark,
        model_id=feature.model_id,
        run_id=feature.run_id,
        task_id=feature.task_id,
        assignment_id=feature.assignment_id,
        program_id=feature.program_id,
        first_failure_phase=feature.phase_id,
        first_failure_type=failure,
        downstream_endpoint_failed=endpoint_failed,
        score_link_id=feature.score_link_id,
    )


__all__ = ["phase_failure_cascade"]
