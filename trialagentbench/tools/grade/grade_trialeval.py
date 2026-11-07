"""Grade a TrialEval run against frozen estimand outcomes."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from trialagentbench_verifier.executor import resolve_trusted_executor_environment
from trialagentbench_verifier.io import release_asset_scope
from trialagentbench_verifier.numerical import NumericalReplay, replay_numerical_program
from trialagentbench_verifier.scorer import current_scorer

from trialagentbench.contracts.core.manifest import GradeManifestV1
from trialagentbench.contracts.core.runs import TrialEvalItemResultV1, TrialEvalRunConfigV1
from trialagentbench.contracts.release.archive_safety import materialized_release_archive
from trialagentbench.contracts.release.artifacts import TRIALEVAL_PARTICIPANT_ARCHIVE_IDENTITY_NAME
from trialagentbench.contracts.release.identity import PairedParticipantArchiveIdentityV1
from trialagentbench.contracts.scoring.trialeval_scores import TrialEvalItemScoresV1, TrialEvalTargetScoreV1
from trialagentbench.grading.outcome import (
    TrialEvalScoreInputV1,
    grade_trialeval_decision_v1,
    grade_trialeval_outcome_v1,
    grade_trialeval_plan_execution_v1,
)
from trialagentbench.grading.reporting import (
    TrialEvalExpectedItemV1,
    write_trialeval_grade_summary,
)
from trialagentbench.io import (
    read_json_model,
    sha256_path,
    staged_directory,
    write_json_model,
)
from trialagentbench.release import (
    LoadedTrialEvalGradingKey,
    load_participant_release,
    load_trialeval_grading_key,
)
from trialagentbench.trialeval.data import discover_participant_items, participant_evidence_artifacts
from trialagentbench.trialeval.outcome_submission import bind_trialeval_numerical_input
from trialagentbench.util import head_sha
from trialagentbench.util.source_identity import harness_source_sha256_v1, require_harness_scorer_v1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Canonical TrialEval run directory.")
    parser.add_argument("--evaluator-release", required=True, help="Complete TrialEval evaluator ZIP.")
    parser.add_argument("--expected-sha256", required=True, help="Trusted SHA-256 of the evaluator ZIP.")
    parser.add_argument("--out-dir", required=True, help="New directory for graded artifacts.")
    parser.add_argument(
        "--task-id",
        action="append",
        help="Grade only these retained tasks into a derived output, preserving the full original run identity.",
    )
    parser.add_argument(
        "--replay-executor-image-id",
        help="Locally attested immutable image ID with the same runtime semantics as the original invocation.",
    )
    parser.add_argument(
        "--budget-censored",
        action="append",
        default=[],
        metavar="TASK_ID",
        help="Scheduled task left unrun by an exhausted campaign spending limit; it is reported, not graded.",
    )
    return parser


def _flat_score_row(*, score: TrialEvalItemScoresV1) -> dict[str, object]:
    """Project one typed component grade into a stable rectangular row."""

    outcome = score.outcome
    point = outcome.point_performance
    interval = outcome.interval_performance
    # Each component reports the first unsatisfied or unevaluable status among the required targets, else the
    # primary's, so a missing complementary target is visible even when the primary target is satisfied.
    components = {
        name: next(
            (
                getattr(result.outcome, name).status
                for result in score.target_results
                if getattr(result.outcome, name).status in {"not_satisfied", "not_evaluable"}
            ),
            getattr(outcome, name).status,
        )
        for name in (
            "completion",
            "question_alignment",
            "evidence_eligibility",
            "reproducibility",
            "estimand_performance",
            "uncertainty",
        )
    }
    return {
        "item_id": score.item_id,
        "cell_ref": score.cell_ref,
        "effect_state": score.effect_state,
        "design_tier": score.design_tier,
        "design_subtype": score.design_subtype,
        "context_tier": score.context_tier,
        "effect_measure": score.effect_measure,
        "result_unit": score.result_unit,
        "model": score.model,
        "agent_status": score.agent_status,
        "turns_used": score.turns_used,
        "target_kind": outcome.target_kind,
        "signal_loss": score.signal_loss,
        "primary_signal_loss": outcome.signal_loss,
        "required_targets": len(score.target_results),
        "scored_targets": sum(result.outcome.signal_loss is not None for result in score.target_results),
        **components,
        "decision": "not_applicable" if score.decision is None else score.decision.status,
        "plan_execution_accurate": None if score.plan_execution is None else score.plan_execution.accurate,
        "failure_origins": sorted(
            {
                *outcome.failure_origins,
                *(origin for result in score.target_results for origin in result.outcome.failure_origins),
            }
        ),
        "native_signed_error": None if point is None else point.native_signed_error,
        "native_absolute_error": None if point is None else point.native_absolute_error,
        "native_squared_error": None if point is None else point.native_squared_error,
        "normalized_absolute_error": None if point is None else point.normalized_absolute_error,
        "normalized_squared_error": None if point is None else point.normalized_squared_error,
        "interval_covers_target": None if interval is None else interval.covers_target,
        "interval_width": None if interval is None else interval.native_width,
        "proper_interval_score": None if interval is None else interval.proper_interval_score,
        "program_replay_status": score.replay.status,
        "program_elapsed_seconds": sum(execution.elapsed_seconds for execution in score.replay.executions),
    }


def _validate_release_binding(
    *,
    run_config: TrialEvalRunConfigV1,
    participant_root: Path,
    grading_key: LoadedTrialEvalGradingKey,
    paired_identity_path: Path,
) -> None:
    """Require the run, participant role, and grading key to share one identity."""

    participant = load_participant_release(participant_root)
    paired = read_json_model(PairedParticipantArchiveIdentityV1, paired_identity_path)
    if participant.manifest.release_id != grading_key.manifest.release_id:
        raise ValueError("TrialEval participant role and grading key identify different releases.")
    scope = release_asset_scope(participant_root)
    expected_participant = (
        paired.participant_archive_sha256
        if scope is None
        else hashlib.sha256(scope.catalog.participant_catalog().to_bytes()).hexdigest()
    )
    if (
        paired.release_id != participant.manifest.release_id
        or expected_participant != run_config.participant_archive_sha256
        or paired.participant_manifest_checksum != participant.manifest.checksum
        or participant.manifest.checksum != run_config.participant_manifest_checksum
    ):
        raise ValueError("TrialEval run does not match the evaluator's paired participant archive.")


def _grade_materialized_run(
    *,
    source: Path,
    suite_dir: Path,
    suite_source: Path,
    output: Path,
    replay_executor_image_id: str | None = None,
    budget_censored: frozenset[str] = frozenset(),
    selected_task_ids: frozenset[str] | None = None,
) -> int:
    """Grade retained tasks without changing their original invocation or submission bytes."""

    run_config = read_json_model(TrialEvalRunConfigV1, source / "run_config.json")
    replay_executor = (
        run_config.executor
        if replay_executor_image_id is None
        else resolve_trusted_executor_environment(
            image=replay_executor_image_id,
            expected_image_id=replay_executor_image_id,
            expected_environment_sha256=run_config.executor.semantic_sha256(),
            limits=run_config.executor.limits,
        )
    )
    if not budget_censored <= set(run_config.task_ids):
        raise ValueError("Budget-censored TrialEval tasks must be scheduled")
    if selected_task_ids is not None and (
        not selected_task_ids
        or not selected_task_ids <= set(run_config.task_ids)
        or selected_task_ids & budget_censored
    ):
        raise ValueError("Selected TrialEval tasks must be nonempty, scheduled and not budget-censored")
    expected_ids = tuple(
        task_id
        for task_id in run_config.task_ids
        if task_id not in budget_censored and (selected_task_ids is None or task_id in selected_task_ids)
    )
    participant = load_participant_release(suite_dir / "participant")
    grading_key = load_trialeval_grading_key(suite_dir, expected_task_ids=expected_ids)
    _validate_release_binding(
        run_config=run_config,
        participant_root=participant.root,
        grading_key=grading_key,
        paired_identity_path=suite_dir / TRIALEVAL_PARTICIPANT_ARCHIVE_IDENTITY_NAME,
    )
    item_by_id = discover_participant_items(participant.root, task_ids=expected_ids)
    if set(expected_ids) - set(item_by_id):
        raise ValueError("TrialEval participant release is missing scheduled tasks.")
    item_files = {
        path.stem: path
        for path in sorted((source / "items").glob("*.json"))
        if selected_task_ids is None or path.stem in selected_task_ids
    }
    if set(item_files) != set(expected_ids):
        raise ValueError(
            "TrialEval run denominator mismatch: "
            f"missing={sorted(set(expected_ids) - set(item_files))}, "
            f"extra={sorted(set(item_files) - set(expected_ids))}"
        )
    expected_aggregation_items = tuple(
        TrialEvalExpectedItemV1(
            item_id=task_id,
            trial_id=(evaluator_record := grading_key.for_task(task_id)).trial_id,
            cell_ref=evaluator_record.cell_ref,
            effect_state=evaluator_record.effect_state,
            design_tier=evaluator_record.design_tier,
            context_tier=evaluator_record.context,
            effect_measure=evaluator_record.target.binding.effect_measure,
            result_unit=evaluator_record.target.binding.result_unit,
        )
        for task_id in expected_ids
    )

    with staged_directory(output) as staging:
        shutil.copytree(
            source,
            staging,
            dirs_exist_ok=True,
            ignore=lambda directory, names: (
                [name for name in names if Path(name).stem not in selected_task_ids]
                if selected_task_ids is not None and Path(directory) == source / "items"
                else []
            ),
        )
        write_json_model(staging / "replay_executor.json", replay_executor)

        def grade_item(task_id: str) -> dict[str, object]:
            record_path = staging / "items" / f"{task_id}.json"
            record = read_json_model(TrialEvalItemResultV1, item_files[task_id])
            if record.item_id != task_id:
                raise ValueError(f"TrialEval item identity mismatch: {task_id}")
            evaluator_record = grading_key.for_task(task_id)
            workspace = source / "logs" / f"{task_id}_workspace" / "item"
            expected_evidence = participant_evidence_artifacts(item_by_id[task_id])
            target_ids = evaluator_record.target_ids
            reported_ids = evaluator_record.reported_target_ids
            python_entrypoint = workspace / "scratch" / "analysis.py"
            r_entrypoint = workspace / "scratch" / "analysis.R"
            if python_entrypoint.exists() and r_entrypoint.exists():
                replay = NumericalReplay(
                    status="invalid_program",
                    entrypoint="",
                    target_ids=reported_ids,
                    evidence_artifacts=expected_evidence,
                    error="Submit exactly one analysis.py or analysis.R entrypoint.",
                )
            else:
                replay = replay_numerical_program(
                    workspace=workspace,
                    entrypoint="analysis.R" if r_entrypoint.exists() else "analysis.py",
                    target_ids=reported_ids,
                    evidence_artifacts=expected_evidence,
                    executor_limits=replay_executor.limits,
                    analysis_language=run_config.experiment_condition.analysis_language,
                    executor_image=replay_executor.image_id,
                    reporting_interface=run_config.experiment_condition.reporting_interface,
                )
            grading_inputs = tuple(
                bind_trialeval_numerical_input(
                    evaluator_record=evaluator_record,
                    replay=replay,
                    expected_evidence=expected_evidence,
                    target_id=target_id,
                )
                for target_id in reported_ids
            )
            if record.agent_output.status == "failed":
                grading_inputs = tuple(
                    value.model_copy(update={"attempt_status": "missing"}) if value.submitted_result is None else value
                    for value in grading_inputs
                )
            # The plan as executed (C1, C3) is reported last and graded apart from the loss and decision.
            plan_input = grading_inputs[len(target_ids) :]
            grading_inputs = grading_inputs[: len(target_ids)]
            outcomes = tuple(grade_trialeval_outcome_v1(value) for value in grading_inputs)
            decision = grade_trialeval_decision_v1(evaluator_record, grading_inputs, outcomes)
            plan_execution = grade_trialeval_plan_execution_v1(evaluator_record, plan_input[0]) if plan_input else None
            outcome = outcomes[0]
            score_payloads = tuple(
                {
                    **value.model_dump(mode="python", exclude={"schema_id", "context", "target", "evidence_eligible"}),
                    "evidence_source_artifacts": (
                        () if value.submitted_result is None else tuple(row.relative_path for row in expected_evidence)
                    ),
                    **({"target_id": target_id} if len(target_ids) > 1 or target_id not in target_ids else {}),
                }
                for target_id, value in zip(reported_ids, (*grading_inputs, *plan_input), strict=True)
            )
            score_input = TrialEvalScoreInputV1.model_validate(
                {
                    **score_payloads[0],
                    "schema_id": "trialagentbench.trialeval_score_input/v1",
                    "task_id": task_id,
                    "additional_inputs": score_payloads[1 : len(target_ids)],
                    **({"plan_input": score_payloads[-1]} if plan_input else {}),
                }
            )
            write_json_model(staging / "verification_inputs" / f"{task_id}.json", score_input)
            score = TrialEvalItemScoresV1(
                scorer=current_scorer(),
                item_id=task_id,
                task_id=task_id,
                release_id=evaluator_record.release_id,
                trial_id=evaluator_record.trial_id,
                base_world_id=evaluator_record.base_world_id,
                trial_name=item_by_id[task_id].trial_name,
                cell_ref=evaluator_record.cell_ref,
                effect_state=evaluator_record.effect_state,
                design_tier=evaluator_record.design_tier,
                design_subtype=evaluator_record.design_subtype,
                context_tier=evaluator_record.context,
                effect_measure=evaluator_record.target.binding.effect_measure,
                result_unit=evaluator_record.target.binding.result_unit,
                estimand_mode=item_by_id[task_id].estimand_mode,
                model=run_config.model,
                turns_used=record.agent_output.turns_used or 0,
                agent_status=record.agent_output.status,
                outcome=outcome,
                additional_results=tuple(
                    TrialEvalTargetScoreV1(
                        outcome=result,
                        target=bound.target,
                        oracle_worlds=reference.oracle_worlds,
                        oracle_monte_carlo_standard_error=reference.oracle_monte_carlo_standard_error,
                        maximum_oracle_monte_carlo_standard_error=reference.maximum_oracle_monte_carlo_standard_error,
                    )
                    for reference, bound, result in zip(
                        evaluator_record.additional_references, grading_inputs[1:], outcomes[1:], strict=True
                    )
                ),
                target=grading_inputs[0].target,
                oracle_worlds=evaluator_record.oracle_worlds,
                oracle_monte_carlo_standard_error=evaluator_record.oracle_monte_carlo_standard_error,
                maximum_oracle_monte_carlo_standard_error=(evaluator_record.maximum_oracle_monte_carlo_standard_error),
                replay=replay,
                decision=decision,
                plan_execution=plan_execution,
            )
            write_json_model(record_path, record.model_copy(update={"scores": score}))
            return _flat_score_row(score=score)

        from trialagentbench_verifier.executor import executor_session_capacity

        from trialagentbench.util.processes import process_map

        with executor_session_capacity(limits=replay_executor.limits, queue=True):
            rows = process_map(grade_item, expected_ids)

        write_trialeval_grade_summary(
            rows=rows,
            expected_items=expected_aggregation_items,
            scheduled_item_ids=expected_ids,
            evaluator_error_item_ids=(),
            output_dir=staging,
        )
        from trialagentbench import __version__ as harness_version

        manifest = GradeManifestV1(
            scorer=current_scorer(),
            suite="trialeval",
            harness_version=harness_version,
            harness_git_sha=head_sha(Path(__file__).resolve().parent) or None,
            harness_source_sha256=harness_source_sha256_v1(),
            timestamp_utc=run_config.timestamp_utc,
            input_run_dir=source.name,
            output_run_dir=output.name,
            release_manifest_sha256=sha256_path(suite_dir / "release_manifest.json"),
            score_profile_id="oracle_outcome_components_v1",
            suite_task_count=len(run_config.task_ids),
            graded_task_count=len(rows),
            notes=[
                "Original run identity is retained. Numerical components compare the fresh program replay "
                "with the frozen estimand outcome; estimator names do not gate scoring.",
                *(
                    [
                        f"Explicit derived grade selection (other tasks ungraded): {', '.join(sorted(selected_task_ids))}"
                    ]
                    if selected_task_ids is not None
                    else []
                ),
                *(
                    [f"Budget-censored and unscored (operator spending limit): {', '.join(sorted(budget_censored))}"]
                    if budget_censored
                    else []
                ),
            ],
        )
        write_json_model(staging / "GRADE_MANIFEST.json", manifest)
    print(f"graded {len(rows)} TrialEval items in {output}")
    return 0


def grade_trialeval_run(argv: list[str] | None = None) -> int:
    """Grade one canonical run against one explicit evaluator release."""

    args = _parser().parse_args(argv)
    require_harness_scorer_v1()
    source = Path(args.run_dir).resolve()
    suite_source = Path(args.evaluator_release).resolve()
    output = Path(args.out_dir).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"TrialEval run directory not found: {source}")
    if not suite_source.exists():
        raise FileNotFoundError(f"TrialEval evaluator release not found: {suite_source}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite grading output: {output}")
    run_config = read_json_model(TrialEvalRunConfigV1, source / "run_config.json")
    with materialized_release_archive(
        suite_source, expected_sha256=args.expected_sha256, in_place=True, task_ids=tuple(run_config.task_ids)
    ) as materialized:
        suite_dir = materialized.root
        if not (suite_dir / "participant").is_dir() or not (suite_dir / "grading_key").is_dir():
            raise FileNotFoundError("TrialEval evaluator release must contain the participant role and grading key.")
        return _grade_materialized_run(
            source=source,
            suite_dir=suite_dir,
            suite_source=suite_source,
            output=output,
            replay_executor_image_id=args.replay_executor_image_id,
            budget_censored=frozenset(args.budget_censored),
            selected_task_ids=None if args.task_id is None else frozenset(args.task_id),
        )


__all__ = ["grade_trialeval_run"]


if __name__ == "__main__":
    raise SystemExit(grade_trialeval_run())
