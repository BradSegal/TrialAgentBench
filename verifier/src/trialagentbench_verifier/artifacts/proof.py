"""Authentication and narrow score binding for fresh execution proofs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

from trialagentbench_verifier.artifacts.io import (
    VerificationInputError,
    canonical_sha256,
    read_json_object,
    release_asset_scope,
    sha256_file,
)
from trialagentbench_verifier.contracts import ExecutionProofV1, NormalizedSubmissionV1
from trialagentbench_verifier.evaluation.numerical import NumericalEstimate, NumericalReplay, read_numerical_table
from trialagentbench_verifier.evaluation.trialeval import TrialEvalScoreInputV1, _PointResult
from trialagentbench_verifier.evaluation.trialeval_result import TrialEvalProgramResultV2
from trialagentbench_verifier.runtime.execution_contracts import ReplayInputArtifactV1
from trialagentbench_verifier.runtime.executor import analysis_project_artifacts


class AuthenticatedExecutionFactsV1(BaseModel):
    """Execution facts permitted to affect independent score verification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    program_artifacts: tuple[ReplayInputArtifactV1, ...] = ()
    program_sha256: str
    program_result_sha256: str
    normalized_program_result_sha256: str | None = None
    replayed_result_sha256s: tuple[str, str]
    replay_status: Literal["reproduced", "within_tolerance"]
    verified_diagnostic_asset_ids: frozenset[str]
    verified_analysis_data: Path | None = None


def numerical_score_projection_sha256(
    estimates: tuple[NumericalEstimate, ...], evidence: tuple[tuple[str, ...], ...]
) -> str:
    """Bind every scored CSV value and its source artifacts in declared target order."""
    if not estimates or len(estimates) != len(evidence):
        raise VerificationInputError("numerical proof requires estimates and matching evidence")
    targets: list[JsonValue] = [
        {
            **({"target_id": row.target_id} if len(estimates) > 1 else {}),
            "estimate": row.estimate,
            "lower": row.lower,
            "upper": row.upper,
            "evidence_source_artifacts": list(artifacts),
        }
        for row, artifacts in zip(estimates, evidence, strict=True)
    ]
    return canonical_sha256(targets[0] if len(targets) == 1 else {"targets": targets})


def submission_score_projection_sha256(submission: NormalizedSubmissionV1) -> str:
    """Hash only score-bearing fields that derive from the executed program result."""

    if submission.benchmark_stream == "trialeval":
        score_input = TrialEvalScoreInputV1.model_validate(submission.payload)
        committed = tuple(item for item in score_input.program_inputs if item.attempt_status == "complete")
        estimates: list[NumericalEstimate] = []
        for item in committed:
            value = item.submitted_result
            if not isinstance(value, _PointResult) or value.interval is None:
                raise VerificationInputError("execution proof requires every requested estimate and interval")
            estimates.append(
                NumericalEstimate(
                    target_id=item.target_id or "primary",
                    estimate=value.value,
                    lower=value.interval.lower,
                    upper=value.interval.upper,
                )
            )
        return numerical_score_projection_sha256(
            tuple(estimates), tuple(item.evidence_source_artifacts for item in committed)
        )
    if submission.submission_kind == "evidence":
        projection = cast(
            JsonValue,
            {
                name: value
                for name, value in submission.payload.items()
                if name not in {"schema_id", "task_id", "report_sha256"}
            },
        )
        return canonical_sha256(projection)
    decision = submission.payload.get("decision")
    if not isinstance(decision, dict):
        raise VerificationInputError("TrialDev portfolio score has no bound decision")
    projection = cast(
        JsonValue,
        {
            name: value
            for name, value in decision.items()
            if name not in {"schema_id", "task_id", "report_sha256", "state_sha256"}
        },
    )
    return canonical_sha256(projection)


def _require_trialeval_values(
    score_input: TrialEvalScoreInputV1, rows: tuple[NumericalEstimate, ...], target_ids: tuple[str, ...]
) -> None:
    """Compare committed answers exactly while preserving every missing-answer state."""
    if not target_ids:
        raise VerificationInputError("graded result has no assigned targets")
    inputs = score_input.program_inputs
    by_target = {row.target_id: row for row in rows}
    if len(by_target) != len(rows):
        raise VerificationInputError("graded numerical values repeat a target")
    input_targets = [item.target_id if item.target_id is not None else target_ids[0] for item in inputs]
    if len(set(input_targets)) != len(input_targets) or not set(by_target) <= set(input_targets):
        raise VerificationInputError("CSV numerical values differ from the supplied score input")
    for item, target_id in zip(inputs, input_targets, strict=True):
        if item.attempt_status != "complete":
            # Ineligible and nonfinite rows retain their missing state and denominator.
            continue
        row = by_target.get(target_id)
        result = item.submitted_result
        if (
            row is None
            or not isinstance(result, _PointResult)
            or result.interval is None
            or (row.estimate, row.lower, row.upper) != (result.value, result.interval.lower, result.interval.upper)
        ):
            raise VerificationInputError("CSV numerical values differ from the supplied score input")


def _require_reported_values(scores: dict[str, JsonValue], score_input: TrialEvalScoreInputV1) -> None:
    """Bind every reported point and interval, including complementary targets."""
    additional = scores.get("additional_results", [])
    if not isinstance(additional, list) or len(additional) != len(score_input.additional_inputs):
        raise VerificationInputError("graded target inventory differs from the supplied score input")
    for graded, item in zip((scores, *additional), score_input.numerical_inputs, strict=True):
        if not isinstance(graded, dict) or not isinstance(outcome := graded.get("outcome"), dict):
            raise VerificationInputError("graded target has no outcome")
        result = item.submitted_result
        point = outcome.get("point_performance")
        if point is not None and (
            not isinstance(point, dict)
            or not isinstance(result, _PointResult)
            or point.get("submitted_value") != result.value
        ):
            raise VerificationInputError("graded point values differ from the supplied score input")
        interval = outcome.get("interval_performance")
        if interval is not None and (
            not isinstance(interval, dict)
            or not isinstance(result, _PointResult)
            or result.interval is None
            or (interval.get("native_lower"), interval.get("native_upper"), interval.get("confidence_level"))
            != (result.interval.lower, result.interval.upper, result.interval.confidence_level)
        ):
            raise VerificationInputError("graded interval values differ from the supplied score input")


def bind_graded_submission(path: Path, submission: NormalizedSubmissionV1) -> None:
    """Bind a retained graded item's values to the authenticated normalized submission.

    Parameters
    ----------
    path
        Canonical graded-item JSON, portable TrialEval input, or normalized submission.
    submission
        Authenticated score input whose committed values must match the graded item.

    Raises
    ------
    VerificationInputError
        If the task or committed numerical values differ from the submission.
    """
    payload = read_json_object(path)
    if payload.get("schema_id") == "trialagentbench.trialeval_score_input/v1":
        graded = NormalizedSubmissionV1(
            benchmark_stream="trialeval",
            task_id=str(payload.get("task_id")),
            submission_kind="analysis",
            payload=payload,
        )
    elif "scores" not in payload:
        graded = NormalizedSubmissionV1.model_validate(payload)
    else:
        if payload.get("item_id") != submission.task_id:
            raise VerificationInputError("graded item identifies a different submission")
        scores = payload["scores"]
        if not isinstance(scores, dict):
            raise VerificationInputError("graded item has no comparable normalized submission")
        if submission.benchmark_stream == "trialeval":
            if submission.submission_failure_code is not None:
                # This input has no scored values; its retained failure code still owns its score and denominator.
                return
            score_input = TrialEvalScoreInputV1.model_validate(submission.payload)
            _require_reported_values(scores, score_input)
            if scores.get("replay") is None and all(
                item.attempt_status != "complete" for item in score_input.program_inputs
            ):
                return
        replay = NumericalReplay.model_validate(scores.get("replay"))
        if submission.benchmark_stream == "trialeval":
            _require_trialeval_values(score_input, replay.estimates or (), replay.target_ids)
        else:
            estimates = TypeAdapter(tuple[NumericalEstimate, ...]).validate_python(submission.payload.get("estimates"))
            if estimates != (replay.estimates or ()):
                raise VerificationInputError("graded item values differ from the normalized submission")
        return
    if graded != submission:
        raise VerificationInputError("graded item values differ from the normalized submission")


def authenticate_execution_proof(
    path: Path,
    *,
    expected_sha256: str,
    program_path: Path,
    program_result_path: Path,
    root: Path,
    stream: Literal["trialeval", "trialdev"],
    candidate_digest: str,
    release_digest: str,
    submission: NormalizedSubmissionV1,
) -> AuthenticatedExecutionFactsV1:
    """Authenticate one separately retained proof and bind its executed result."""

    if len(expected_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_sha256):
        raise VerificationInputError("expected execution-proof SHA-256 must be lowercase hexadecimal")
    actual = sha256_file(Path(path))
    if actual != expected_sha256:
        raise VerificationInputError("execution proof differs from its trusted expected SHA-256")
    proof = ExecutionProofV1.model_validate(read_json_object(Path(path)))
    if proof.status != "pass":
        raise VerificationInputError("execution proof does not establish successful fresh replay")
    if (
        proof.benchmark_stream != stream
        or proof.candidate_digest != candidate_digest
        or proof.release_digest != release_digest
        or proof.task_id != submission.task_id
    ):
        raise VerificationInputError("execution proof identity differs from the score input or release")
    if proof.result_format != "csv" and proof.score_projection_sha256 != submission_score_projection_sha256(
        submission
    ):
        raise VerificationInputError("execution proof result differs from the normalized score input")
    if not Path(program_path).is_file() or Path(program_path).is_symlink():
        raise VerificationInputError("executed program must be a regular file")
    if sha256_file(Path(program_path)) != proof.program_sha256:
        raise VerificationInputError("execution proof program differs from the supplied program")
    project = analysis_project_artifacts(Path(program_path))
    if proof.program_artifacts:
        if project != proof.program_artifacts:
            raise VerificationInputError("execution proof source project differs from the supplied modules")
    elif len(project) != 1:
        raise VerificationInputError("execution proof does not authenticate the supplied source modules")
    if not Path(program_result_path).is_file() or Path(program_result_path).is_symlink():
        raise VerificationInputError("executed program result must be a regular file")
    program_result_payload = None
    if proof.result_format == "csv":
        rows = read_numerical_table(Path(program_result_path), target_ids=proof.target_ids)
        if stream == "trialeval":
            score_input = TrialEvalScoreInputV1.model_validate(submission.payload)
            _require_trialeval_values(score_input, rows, proof.target_ids)
            inputs = score_input.program_inputs
            primary_evidence = tuple(inputs[0].evidence_source_artifacts)
            # The replay proof hashes every CSV row with the task's evidence set, as executed.
            if proof.score_projection_sha256 != numerical_score_projection_sha256(
                rows, tuple(primary_evidence for _ in rows)
            ):
                raise VerificationInputError("execution proof result differs from the normalized score input")
        else:
            if submission.submission_kind != "evidence":
                raise VerificationInputError("Initial CSV execution cannot authenticate a programme decision")
            estimates = TypeAdapter(tuple[NumericalEstimate, ...]).validate_python(submission.payload.get("estimates"))
            if estimates != rows:
                raise VerificationInputError("CSV numerical values differ from the supplied score input")
        program_result_sha256 = sha256_file(Path(program_result_path))
    else:
        if stream == "trialeval" and TrialEvalScoreInputV1.model_validate(submission.payload).additional_inputs:
            raise VerificationInputError("Paired TrialEval results require the complete CSV execution result")
        program_result_payload = read_json_object(Path(program_result_path))
        program_result_sha256 = canonical_sha256(program_result_payload)
    if program_result_sha256 != proof.program_result_sha256:
        raise VerificationInputError("execution proof result differs from the supplied program result")
    paired = Path(root) / "participant_archive_identity.json"
    if paired.is_file():
        identity = read_json_object(paired)
        scope = release_asset_scope(root)
        expected_participant = (
            identity.get("participant_archive_sha256")
            if scope is None
            else hashlib.sha256(scope.catalog.participant_catalog().to_bytes()).hexdigest()
        )
        if proof.input_kind != "archive" or expected_participant != proof.input_sha256:
            raise VerificationInputError("execution proof does not use the evaluator-bound participant archive")
    if (
        proof.program_sha256 is None
        or proof.program_result_sha256 is None
        or len(proof.replayed_result_sha256s) != 2
        or proof.replay_status not in {"reproduced", "within_tolerance"}
    ):
        raise VerificationInputError("execution proof omits required replay identities")
    verified_analysis_data = None
    if proof.analysis_data_reproduced is True:
        if proof.analysis_data_filename is None or proof.analysis_data_sha256 is None:
            raise VerificationInputError("analysis-data replay has no retained identity")
        data_path = Path(program_result_path).parent / proof.analysis_data_filename
        if not data_path.is_file() or data_path.is_symlink() or sha256_file(data_path) != proof.analysis_data_sha256:
            raise VerificationInputError("analysis data differs from the authenticated replay artifact")
        verified_analysis_data = data_path
    return AuthenticatedExecutionFactsV1(
        program_artifacts=proof.program_artifacts,
        program_sha256=proof.program_sha256,
        program_result_sha256=proof.program_result_sha256,
        normalized_program_result_sha256=(
            canonical_sha256(TrialEvalProgramResultV2.model_validate(program_result_payload).model_dump(mode="json"))
            if stream == "trialeval" and proof.result_format == "json"
            else None
        ),
        replayed_result_sha256s=(proof.replayed_result_sha256s[0], proof.replayed_result_sha256s[1]),
        replay_status=proof.replay_status,
        verified_diagnostic_asset_ids=frozenset(proof.verified_diagnostic_asset_ids),
        verified_analysis_data=verified_analysis_data,
    )


__all__ = ["AuthenticatedExecutionFactsV1", "authenticate_execution_proof", "submission_score_projection_sha256"]
