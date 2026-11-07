"""Prospective task priorities without changing native invocation denominators."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path

from trialagentbench.contracts.campaign import CampaignPlanV1, CampaignRunV1
from trialagentbench.contracts.core.runs import RunCoverageV1
from trialagentbench.io import read_json_model
from trialagentbench.ports.llm_provider import ProviderUnavailableError
from trialagentbench.util.campaign_control import CampaignCapacityStop


def execution_slice(
    *,
    campaign_root: Path | None,
    job_id: str | None,
    run_root: Path,
    full_task_ids: Sequence[str],
    requested_task_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    """Authenticate an operational subset while retaining the full scientific panel.

    Spending control separately authenticates invocation hashes before execution.
    No slice is accepted for standalone commands or campaigns without a frozen order.
    """
    if requested_task_ids is None:
        return tuple(full_task_ids)
    if campaign_root is None or job_id is None or not requested_task_ids:
        raise ValueError("Execution slices require a prepared campaign and nonempty task IDs")
    plan = read_json_model(CampaignPlanV1, campaign_root / "campaign.json")
    run = next((row for row in plan.runs if row.job_id == job_id), None)
    dispatch = plan.configuration.dispatch
    if run is None or dispatch is None or (campaign_root / run.relative_run_root).resolve() != run_root.resolve():
        raise ValueError("Execution slice does not belong to a campaign with a frozen dispatch order")
    requested = tuple(requested_task_ids)
    if len(set(requested)) != len(requested) or not set(requested) <= set(full_task_ids):
        raise ValueError("Execution slice has duplicate or unscheduled task IDs")
    if len(requested) > dispatch.tasks_per_slice:
        raise ValueError("Execution slice exceeds its frozen size bound")
    ordered = tuple(row.task_id for row in dispatch.task_order if row.benchmark == run.benchmark)
    if not set(full_task_ids) <= set(ordered) or tuple(t for t in ordered if t in requested) != requested:
        raise ValueError("Execution slice conflicts with the frozen task order")
    return requested


def _event(root: Path, **fields: object) -> None:
    """Retain safe operational metadata; completion truth remains native coverage."""
    row = {"timestamp_utc": datetime.now(UTC).isoformat(), "time_unix": time.time(), **fields}
    with (root / "dispatch-events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _deferrals(root: Path) -> tuple[dict[tuple[str, str], int], dict[str, float], dict[str, int]]:
    counts: dict[tuple[str, str], int] = {}
    until: dict[str, float] = {}
    consecutive: dict[str, int] = {}
    path = root / "dispatch-events.jsonl"
    if path.exists():
        # A torn audit tail must be repaired explicitly; never silently erase retry history.
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["status"] == "provider_deferred":
                for task_id in row["pending_task_ids"]:
                    key = (row["job_id"], task_id)
                    counts[key] = counts.get(key, 0) + 1
                until[row["route"]] = max(until.get(row["route"], 0), row["resume_after_unix"])
                consecutive[row["route"]] = consecutive.get(row["route"], 0) + 1
            elif row["status"] == "completed":
                consecutive[row["route"]] = 0
    return counts, until, consecutive


def dispatch_campaign(
    *,
    root: Path,
    plan: CampaignPlanV1,
    execute: Callable[[CampaignRunV1, tuple[str, ...]], None],
    cancelled: Callable[[], None],
) -> None:
    """Run bounded slices fairly, retaining one native writer and ledger per job.

    A route is tried at most once after an outage in each invocation. Deferred
    work is eligible on a later resume after its recorded cooldown, up to the
    frozen per-task deferral bound. Other routes and later priorities proceed.
    """
    dispatch = plan.configuration.dispatch
    if dispatch is None:
        raise ValueError("A frozen dispatch order is required")
    models = {model.name: model for model in plan.configuration.models}
    model_names = tuple(models)
    ranks = {(row.benchmark, row.task_id): index for index, row in enumerate(dispatch.task_order)}
    ordered = {
        run.job_id: tuple(row.task_id for row in dispatch.task_order if row.benchmark == run.benchmark)
        for run in plan.runs
    }
    counts, cooldowns, consecutive = _deferrals(root)
    blocked_routes = {route for route, until in cooldowns.items() if until > time.time()}
    blocked_routes.update(
        route for route, count in consecutive.items() if count >= dispatch.maximum_consecutive_provider_deferrals
    )
    active_jobs: set[str] = set()
    pending: dict[str, set[str]] = {}

    def route(run: CampaignRunV1) -> str:
        model = models[run.model_name]
        return model.upstream_provider or model.name

    def refresh(run: CampaignRunV1) -> None:
        coverage = read_json_model(RunCoverageV1, root / run.relative_run_root / "coverage.json")
        expected = set(ordered[run.job_id])
        # Eligible infrastructure-retry jobs intentionally contain only a subset.
        if run.retry_of_job_id is None and set(coverage.unit_ids) != expected:
            raise ValueError("Native coverage differs from the frozen dispatch panel")
        if not set(coverage.unit_ids) <= expected:
            raise ValueError("Native coverage contains a task outside the frozen dispatch panel")
        pending[run.job_id] = set(coverage.unit_ids) - set(coverage.completed_unit_ids)

    for run in plan.runs:
        refresh(run)
        if any(
            counts.get((run.job_id, task_id), 0) >= dispatch.maximum_provider_deferrals_per_task
            for task_id in pending[run.job_id]
        ):
            # Do not turn a sustained outage into one new failed attempt on
            # every remaining task. A depleted recovery allowance needs review.
            blocked_routes.add(route(run))

    def candidate() -> tuple[CampaignRunV1, tuple[str, ...]] | None:
        choices = []
        for run in plan.runs:
            if run.job_id in active_jobs or route(run) in blocked_routes:
                continue
            ids = tuple(
                t
                for t in ordered[run.job_id]
                if t in pending[run.job_id]
                and counts.get((run.job_id, t), 0) < dispatch.maximum_provider_deferrals_per_task
            )
            if not ids:
                continue
            rank = ranks[run.benchmark, ids[0]]
            model_rank = (model_names.index(run.model_name) - rank) % len(model_names)
            # All primary jobs precede their predeclared retry jobs at the same priority.
            key = (run.repetition, rank, run.retry_of_job_id is not None, model_rank)
            choices.append((key, run, ids[: dispatch.tasks_per_slice]))
        if not choices:
            return None
        _, run, ids = min(choices, key=lambda row: row[0])
        return run, ids

    pool = ThreadPoolExecutor(max_workers=plan.configuration.operations.model_workers)
    active: dict[Future[None], tuple[CampaignRunV1, tuple[str, ...]]] = {}
    succeeded = False
    try:
        while True:
            while len(active) < plan.configuration.operations.model_workers:
                next_slice = candidate()
                if next_slice is None:
                    break
                run, ids = next_slice
                active_jobs.add(run.job_id)
                _event(root, status="started", job_id=run.job_id, task_ids=ids)
                active[pool.submit(execute, run, ids)] = run, ids
            if not active:
                break
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                run, ids = active.pop(future)
                active_jobs.remove(run.job_id)
                try:
                    future.result()
                except ProviderUnavailableError as error:
                    refresh(run)
                    blocked_routes.add(route(run))
                    remaining = tuple(t for t in ids if t in pending[run.job_id])
                    for t in remaining:
                        counts[run.job_id, t] = counts.get((run.job_id, t), 0) + 1
                    delay = max(dispatch.defer_seconds, error.retry_after_seconds or 0)
                    _event(
                        root,
                        status="provider_deferred",
                        job_id=run.job_id,
                        task_ids=ids,
                        pending_task_ids=remaining,
                        route=route(run),
                        resume_after_unix=time.time() + delay,
                        failure_kind=error.failure_kind,
                    )
                    print(f"Deferred provider for {run.job_id}; other routes continue.", flush=True)
                except CampaignCapacityStop as error:
                    if error.campaign_wide:
                        raise
                    refresh(run)
                    blocked_routes.add(route(run))
                    _event(root, status="capacity_deferred", job_id=run.job_id, task_ids=ids, route=route(run))
                else:
                    refresh(run)
                    if set(ids) & pending[run.job_id]:
                        raise RuntimeError("Native slice returned without terminal coverage or an explicit deferral")
                    _event(root, status="completed", job_id=run.job_id, task_ids=ids, route=route(run))
        remaining_count = sum(map(len, pending.values()))
        if remaining_count:
            print(
                f"Dispatch paused with {remaining_count} native tasks pending; retain checkpoints and resume later.",
                flush=True,
            )
        # A provider-deferred Eval slice may save the final failed primary before
        # raising. Publish native full-run summaries offline on the same custody.
        for run in plan.runs:
            if not pending[run.job_id]:
                execute(run, ())
        succeeded = True
    finally:
        if not succeeded:
            cancelled()
        pool.shutdown(wait=True, cancel_futures=True)
