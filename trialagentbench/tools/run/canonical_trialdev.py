"""Run canonical TrialDev tasks from a role-separated evaluator release."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypeVar, cast

from pydantic import ValidationError
from trialagentbench_verifier.assets import AssetTask, read_asset_catalog
from trialagentbench_verifier.execution_contracts import AnalysisMode
from trialagentbench_verifier.executor import resolve_trusted_executor_environment

from trialagentbench.adapters import ProviderRouting, close_provider, get_provider
from trialagentbench.contracts.core.config import (
    ProviderRetryPolicyV1,
    ReasoningEffortV1,
)
from trialagentbench.contracts.release.archive_safety import materialized_release_archive
from trialagentbench.contracts.trialdev.release import TrialDevTaskV1
from trialagentbench.contracts.trialdev.run import (
    TrialDevReportedCostV1,
    TrialDevRunIndexV1,
    TrialDevRunInvocationV1,
    TrialDevRunResultV1,
)
from trialagentbench.execution_policy import TRIALDEV_RELEASE_BUDGET_V1
from trialagentbench.io import read_json_model, write_json_model
from trialagentbench.ports.llm_provider import ProviderUnavailableError
from trialagentbench.trialdev.release import load_trialdev_evaluator_release
from trialagentbench.trialdev.run import recover_terminal_trialdev_task, run_trialdev_task
from trialagentbench.trialdev.task_agent import DEFAULT_MAX_TURNS_PER_STEP
from trialagentbench.util.campaign_control import request_control_for_run
from trialagentbench.util.campaign_dispatch import execution_slice
from trialagentbench.util.experiment_condition import resolve_experiment_condition_v1
from trialagentbench.util.parallel import run_bounded_tasks
from trialagentbench.util.provider_environment import load_provider_dotenv
from trialagentbench.util.provider_telemetry import (
    read_provider_terminal_events_v1,
    summarize_provider_telemetry_v1,
)
from trialagentbench.util.reported_cost import (
    ReportedCostBoundProvider,
    ReportedCostBudget,
    RunStopSignal,
    StoppableProvider,
)
from trialagentbench.util.run_coverage import (
    load_or_create_run_coverage_v1,
    reconcile_run_coverage_v1,
)
from trialagentbench.util.run_ownership import exclusive_run_directory
from trialagentbench.util.source_identity import harness_source_sha256_v1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True, help="Complete TrialDev evaluator release ZIP.")
    parser.add_argument("--expected-sha256", required=True, help="Trusted SHA-256 of the evaluator ZIP.")
    parser.add_argument("--model", required=True, help="Exact analysis-model identifier.")
    parser.add_argument("--executor-image", help="Local reference for the qualified executor image.")
    parser.add_argument(
        "--expected-executor-image-id",
        required=True,
        help="Immutable sha256: Docker image ID from this host's authenticated executor attestation.",
    )
    parser.add_argument(
        "--expected-executor-environment-sha256",
        required=True,
        help="Externally trusted executor inventory SHA-256 from the release-set manifest.",
    )
    parser.add_argument(
        "--provider", choices=("openai", "openai_responses", "openrouter", "offline_control"), required=True
    )
    parser.add_argument(
        "--memory",
        choices=("bounded", "openai_responses_compaction"),
        default="bounded",
        help="Analysis conversation memory; Responses compaction requires --provider openai_responses.",
    )
    parser.add_argument("--openrouter-provider")
    parser.add_argument("--task", action="append", dest="tasks", help="Task or case ID; repeat to select more.")
    parser.add_argument(
        "--data-view",
        choices=("raw", "analysis_ready", "all"),
        help="Evidence presentation. Defaults to raw unless explicit task selectors are supplied.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("results/trialdev"))
    parser.add_argument("--max-turns-per-step", type=int, default=DEFAULT_MAX_TURNS_PER_STEP)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=TRIALDEV_RELEASE_BUDGET_V1.maximum_completion_tokens_per_turn,
    )
    parser.add_argument(
        "--max-context-characters",
        type=int,
        default=TRIALDEV_RELEASE_BUDGET_V1.maximum_context_characters,
    )
    parser.add_argument(
        "--watchdog-seconds",
        type=int,
        default=TRIALDEV_RELEASE_BUDGET_V1.wall_time_limit_seconds,
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=TRIALDEV_RELEASE_BUDGET_V1.provider_request_timeout_seconds,
    )
    parser.add_argument("--reported-cost-stop-usd", type=float)
    parser.add_argument("--reasoning-effort", choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"))
    parser.add_argument(
        "--exclude-returned-reasoning",
        action="store_true",
        help="Ask the route not to return reasoning; by default returned reasoning is kept in the provider log.",
    )
    parser.add_argument(
        "--replay-returned-reasoning",
        action="store_true",
        help="Send returned reasoning details back on later turns (OpenRouter reasoning-state continuity).",
    )
    parser.add_argument("--provider-capability-snapshot", type=Path)
    parser.add_argument("--analysis-language", choices=("python_r", "python", "r"), default="python_r")
    parser.add_argument("--condition-id", default="primary")
    parser.add_argument("--request-replicate-id", default="request-1")
    parser.add_argument("--provider-max-attempts", type=int, default=5)
    parser.add_argument("--provider-retry-base-delay-seconds", type=float, default=2.0)
    parser.add_argument(
        "--provider-unavailability-policy", choices=("charge_all_v1", "defer_unanswered_v1"), default="charge_all_v1"
    )
    parser.add_argument("--decoding-seed", type=int)
    parser.add_argument("--omit-temperature", action="store_true")
    parser.add_argument("--dotenv", action="store_true")
    parser.add_argument(
        "--resume-run-dir",
        type=Path,
        help="Existing run directory containing validated decision-boundary checkpoints.",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Concurrent independent tasks; each portfolio remains sequential."
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--campaign-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--campaign-job", help=argparse.SUPPRESS)
    parser.add_argument("--dispatch-task-id", action="append", help=argparse.SUPPRESS)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Freeze the validated invocation and task schedule without contacting a provider; use resume to execute.",
    )
    return parser


_SelectionTask = TypeVar("_SelectionTask", TrialDevTaskV1, AssetTask)


def _selected_tasks(
    available_tasks: Sequence[_SelectionTask],
    selectors: Sequence[str],
    *,
    data_view: Literal["raw", "analysis_ready", "all"] | None = None,
) -> tuple[_SelectionTask, ...]:
    """Resolve one task selection for metadata preflight and materialized execution."""
    view = data_view or ("all" if selectors else "raw")
    tasks = tuple(
        task
        for task in available_tasks
        if view == "all"
        or (("raw" if task.main else "analysis_ready") if isinstance(task, AssetTask) else task.data_view) == view
    )
    if not selectors:
        return tasks
    if len(selectors) != len(set(selectors)):
        raise ValueError("TrialDev task selectors must be unique")
    keys = {}
    for task in tasks:
        stream = task.stream if isinstance(task, AssetTask) else task.stream.value
        keys[task.task_id] = {task.task_id, task.case_id, stream, f"{task.case_id}:{stream}"}
    selected = set(selectors)
    matched = set().union(*(values & selected for values in keys.values()))
    unmatched = tuple(selector for selector in selectors if selector not in matched)
    if unmatched:
        raise ValueError(f"TrialDev task selectors matched no released task: {unmatched}")
    return tuple(task for task in tasks if keys[task.task_id] & selected)


def _reported_cost_budget(*, run_root: Path, threshold_usd: float, resume: bool) -> ReportedCostBudget:
    if not resume:
        return ReportedCostBudget(threshold_usd)
    events = tuple(
        event
        for path in sorted(run_root.rglob("provider_responses*.jsonl"))
        for event in read_provider_terminal_events_v1(path)
        if event.status == "succeeded"
    )
    costs = tuple(event.reported_cost_usd for event in events)
    return ReportedCostBudget.from_observation(
        threshold_usd=threshold_usd,
        observed_usd=sum(cost or 0.0 for cost in costs),
        response_count=len(costs),
        cost_complete=all(cost is not None for cost in costs),
    )


def _run_materialized(
    args: argparse.Namespace,
    *,
    release_root: Path,
    release_archive_sha256: str,
) -> int:
    """Run selected tasks from one verified materialized evaluator archive."""

    run_root: Path | None = None
    ownership = ExitStack()
    try:
        release = load_trialdev_evaluator_release(release_root)
        tasks = _selected_tasks(release.participant.tasks, args.tasks or (), data_view=args.data_view)
        if not tasks:
            raise ValueError("No TrialDev tasks matched the requested selectors")
        for task in tasks:
            release.analysis_configuration(task)
            release.authored_configuration(task)
        executor = resolve_trusted_executor_environment(
            image=args.executor_image,
            expected_image_id=args.expected_executor_image_id,
            expected_environment_sha256=args.expected_executor_environment_sha256,
            limits=TRIALDEV_RELEASE_BUDGET_V1.code_execution,
        )
        condition = resolve_experiment_condition_v1(
            condition_id=args.condition_id,
            analysis_language=cast(AnalysisMode, args.analysis_language),
            request_replicate_id=args.request_replicate_id,
            reasoning_effort=cast(ReasoningEffortV1 | None, args.reasoning_effort),
            provider_capability_snapshot=args.provider_capability_snapshot,
            provider=args.provider,
            model=args.model,
            openrouter_provider=args.openrouter_provider,
            procedure_assistance="output_contract_only",
            maximum_turns_per_step=args.max_turns_per_step,
            tool_choice="auto",
            memory_strategy=args.memory,
            exclude_reasoning=args.exclude_returned_reasoning,
            replay_reasoning=args.replay_returned_reasoning,
        )
        retry = ProviderRetryPolicyV1(
            maximum_attempts=args.provider_max_attempts,
            base_delay_seconds=args.provider_retry_base_delay_seconds,
            unavailability_policy=args.provider_unavailability_policy,
        )
        invocation = TrialDevRunInvocationV1(
            release_id=release.manifest.release_id,
            release_archive_sha256=release_archive_sha256,
            release_manifest_sha256=release.participant.release_manifest_sha256,
            harness_source_sha256=harness_source_sha256_v1(),
            scheduled_task_ids=tuple(task.task_id for task in tasks),
            scheduled_task_streams={task.task_id: task.stream for task in tasks},
            model=args.model,
            provider=args.provider,
            openrouter_provider=args.openrouter_provider,
            experiment_condition=condition,
            decoding_seed=args.decoding_seed,
            retry=retry,
            send_temperature=not args.omit_temperature,
            maximum_completion_tokens=args.max_tokens,
            maximum_context_characters=args.max_context_characters,
            watchdog_seconds=args.watchdog_seconds,
            request_timeout_seconds=args.request_timeout_seconds,
            reported_cost_stop_usd=args.reported_cost_stop_usd,
            workers=args.workers,
            executor_image=executor.image_id,
            executor_limits=executor.limits,
        )
        if args.resume_run_dir is None:
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            run_root = args.output_root.resolve() / f"run_{timestamp}"
            run_root.mkdir(parents=True, exist_ok=False)
            ownership.enter_context(exclusive_run_directory(run_root))
            write_json_model(run_root / "invocation.json", invocation)
        else:
            run_root = args.resume_run_dir.resolve(strict=True)
            if not run_root.is_dir() or run_root.is_symlink():
                raise ValueError("TrialDev resume run directory is invalid")
            ownership.enter_context(exclusive_run_directory(run_root))
            recorded = read_json_model(TrialDevRunInvocationV1, run_root / "invocation.json")
            if recorded != invocation:
                recorded_fields = recorded.model_dump(mode="json", exclude={"checksum"})
                requested_fields = invocation.model_dump(mode="json", exclude={"checksum"})
                changed_fields = sorted(
                    field for field, value in requested_fields.items() if recorded_fields.get(field) != value
                )
                raise ValueError(
                    "TrialDev resume arguments disagree with the recorded invocation: " + ", ".join(changed_fields)
                )
        if invocation.checksum is None:
            raise RuntimeError("TrialDev invocation has no custody checksum")
        coverage_path = run_root / "coverage.json"
        coverage = load_or_create_run_coverage_v1(
            coverage_path,
            run_identity_sha256=invocation.checksum,
            unit_ids=invocation.scheduled_task_ids,
            resume=args.resume_run_dir is not None,
        )
        if args.prepare_only:
            print(f"Prepared {len(tasks)} tasks without provider execution: {run_root}")
            return 0
        slice_ids = execution_slice(
            campaign_root=args.campaign_root,
            job_id=args.campaign_job,
            run_root=run_root,
            full_task_ids=invocation.scheduled_task_ids,
            requested_task_ids=args.dispatch_task_id,
        )
        by_id = {task.task_id: task for task in tasks}
        work_tasks = tuple(by_id[task_id] for task_id in slice_ids)
        completed_task_ids = set(coverage.completed_unit_ids)
        request_control = request_control_for_run(
            campaign_root=args.campaign_root, job_id=args.campaign_job, run_root=run_root
        )
        budget = (
            None
            if args.reported_cost_stop_usd is None
            else _reported_cost_budget(
                run_root=run_root,
                threshold_usd=args.reported_cost_stop_usd,
                resume=args.resume_run_dir is not None,
            )
        )
        stop = RunStopSignal()

        def execute(task: TrialDevTaskV1) -> TrialDevRunResultV1:
            completed = (
                None
                if args.resume_run_dir is None
                else recover_terminal_trialdev_task(
                    release=release,
                    task_id=task.task_id,
                    output_root=run_root / "tasks",
                )
            )
            if task.task_id in completed_task_ids and completed is None:
                raise ValueError(
                    f"TrialDev coverage claims completion without a valid task checkpoint: {task.task_id}"
                )
            if completed is not None:
                run = completed
            else:
                provider = get_provider(
                    args.model,
                    routing=ProviderRouting(args.provider, args.openrouter_provider),
                    send_temperature=not args.omit_temperature,
                    timeout_s=args.request_timeout_seconds,
                    decoding_seed=args.decoding_seed,
                    reasoning_effort=condition.reasoning.effort,
                    exclude_reasoning=condition.reasoning.exclude_from_response,
                    replay_reasoning=condition.reasoning.replay_in_context,
                    retry_policy=retry,
                    memory_strategy=args.memory,
                    request_control=request_control,
                )
                if budget is not None:
                    provider = ReportedCostBoundProvider(provider=provider, budget=budget)
                provider = StoppableProvider(provider=provider, signal=stop)
                try:
                    run = run_trialdev_task(
                        release=release,
                        task_id=task.task_id,
                        provider=provider,
                        output_root=run_root / "tasks",
                        max_turns_per_step=args.max_turns_per_step,
                        max_tokens=args.max_tokens,
                        max_context_characters=args.max_context_characters,
                        provider_capability=condition.provider_capability,
                        analysis_language=condition.analysis_language,
                        watchdog_seconds=args.watchdog_seconds,
                        tool_choice="auto",
                        executor_image=executor.image_id,
                        executor_limits=executor.limits,
                        resume=args.resume_run_dir is not None,
                        verbose=args.verbose,
                        provider_unavailability_policy=retry.unavailability_policy,
                    )
                finally:
                    close_provider(provider)
            if run.checksum is None:
                raise RuntimeError("TrialDev task run has no custody checksum")
            return TrialDevRunResultV1(
                task_id=run.task_id,
                stream=run.stream,
                execution_status=run.execution_status,
                checksum=run.checksum,
            )

        results: list[TrialDevRunResultV1] = []

        def consume(result: TrialDevRunResultV1) -> None:
            nonlocal coverage
            results.append(result)
            completed_task_ids.add(result.task_id)
            completed_in_order = tuple(
                task_id for task_id in invocation.scheduled_task_ids if task_id in completed_task_ids
            )
            coverage = reconcile_run_coverage_v1(coverage_path, coverage, completed_unit_ids=completed_in_order)
            print(f"[DONE {len(completed_task_ids)}/{len(tasks)}] {result.task_id}", flush=True)

        run_bounded_tasks(
            work_tasks, execute, workers=args.workers, consume=consume, stop=stop, request_control=request_control
        )
        if set(completed_task_ids) != set(invocation.scheduled_task_ids):
            print(f"Slice saved; {len(completed_task_ids)}/{len(tasks)} full-panel tasks complete.", flush=True)
            return 0
        # Earlier slices belong to the same native invocation. Reconstruct the
        # complete index from their immutable terminal checkpoints, never rerun.
        indexed = {result.task_id: result for result in results}
        for task in tasks:
            if task.task_id not in indexed:
                completed = recover_terminal_trialdev_task(
                    release=release, task_id=task.task_id, output_root=run_root / "tasks"
                )
                if completed is None or completed.checksum is None:
                    raise ValueError("Complete TrialDev coverage has no terminal task checkpoint")
                indexed[task.task_id] = TrialDevRunResultV1(
                    task_id=completed.task_id,
                    stream=completed.stream,
                    execution_status=completed.execution_status,
                    checksum=completed.checksum,
                )
        results = list(indexed.values())
        positions = {task.task_id: position for position, task in enumerate(tasks)}
        results.sort(key=lambda result: positions[result.task_id])
        cost = None if budget is None else budget.snapshot()
        summarize_provider_telemetry_v1(
            run_root=run_root,
            coverage=coverage,
        )
        write_json_model(
            run_root / "run_index.json",
            TrialDevRunIndexV1(
                release_id=release.manifest.release_id,
                scheduled_task_ids=tuple(task.task_id for task in tasks),
                model=args.model,
                invocation_checksum=invocation.checksum,
                results=tuple(results),
                reported_cost=(
                    None
                    if cost is None
                    else TrialDevReportedCostV1(
                        threshold_usd=cost.threshold_usd,
                        observed_usd=cost.observed_usd,
                        response_count=cost.response_count,
                        threshold_reached=cost.threshold_reached,
                        cost_complete=cost.cost_complete,
                        overshoot_usd=cost.overshoot_usd,
                    )
                ),
            ),
        )
    except ProviderUnavailableError:
        raise
    except (OSError, RuntimeError, ValueError, ValidationError) as error:
        sys.stderr.write(f"TrialDev run failed: {error}\n")
        return 2
    finally:
        ownership.close()
    if run_root is None:  # pragma: no cover - success-path invariant
        raise RuntimeError("TrialDev run completed without an output directory")
    sys.stdout.write(f"TrialDev results written to {run_root}.\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run TrialDev from an externally anchored evaluator ZIP."""

    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.dotenv:
        load_provider_dotenv()
    task_ids = None
    if args.release.name.endswith(".assets.json"):
        catalog = read_asset_catalog(args.release, expected_sha256=args.expected_sha256)
        if catalog.benchmark != "trialdev" or catalog.audience != "evaluator":
            raise ValueError("TrialDev execution requires evaluator assets")
        task_ids = tuple(
            task.task_id for task in _selected_tasks(catalog.tasks, args.tasks or (), data_view=args.data_view)
        )
    with materialized_release_archive(
        args.release, expected_sha256=args.expected_sha256, task_ids=task_ids
    ) as materialized:
        return _run_materialized(
            args,
            release_root=materialized.root,
            release_archive_sha256=materialized.archive_sha256,
        )


if __name__ == "__main__":
    raise SystemExit(main())
