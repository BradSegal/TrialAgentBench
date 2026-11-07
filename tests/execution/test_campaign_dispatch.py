"""Offline priority, restart, deferral and bounded dispatch invariants."""

from __future__ import annotations

import json
import threading

import pytest

from trialagentbench.contracts.campaign import CampaignConfigurationV1, CampaignPlanV1, CampaignRunV1
from trialagentbench.contracts.core.runs import RunCoverageV1
from trialagentbench.io import read_json_model, write_json_model
from trialagentbench.ports.llm_provider import ProviderUnavailableError, RetryTelemetry
from trialagentbench.util import campaign_dispatch as owner
from trialagentbench.util.parallel import run_bounded_tasks
from trialagentbench.util.reported_cost import RunStopSignal
from trialagentbench.util.run_coverage import reconcile_run_coverage_v1, run_coverage_v1, write_run_coverage_v1


def configuration(root, *, workers=1, task_workers=1, slices=1):
    return CampaignConfigurationV1.model_validate(
        {
            "release_directory": str(root),
            "expected_release_manifest_sha256": "a" * 64,
            "executor_image_id": "sha256:" + "b" * 64,
            "models": [
                {
                    "name": n,
                    "model": f"fixture/{n}",
                    "upstream_provider": n,
                    "maximum_completion_tokens": 100,
                    "spend_limit_micro_usd": 10000,
                    "reservation_micro_usd": 100,
                    "reservation_tokens": 1000,
                    "pricing_premise": "offline test",
                    "requests_per_minute": 100,
                    "tokens_per_minute": 10000,
                }
                for n in ["first", "second"]
            ],
            "panel": {"benchmarks": ["trialdev"], "trialdev_tasks": ["a", "b", "c"]},
            "operations": {"model_workers": workers, "task_workers": task_workers, "spend_limit_micro_usd": 20000},
            "dispatch": {
                "task_order": [{"benchmark": "trialdev", "task_id": t} for t in ["b", "a", "c"]],
                "tasks_per_slice": slices,
                "defer_seconds": 10,
                "maximum_provider_deferrals_per_task": 2,
            },
        }
    )


def plan(root, **kwargs):
    config = configuration(root, **kwargs)
    runs = tuple(
        CampaignRunV1(
            job_id=n,
            benchmark="trialdev",
            model_name=n,
            repetition=1,
            relative_run_root=f"runs/{n}",
            invocation_sha256="a" * 64,
        )
        for n in ["first", "second"]
    )
    result = CampaignPlanV1(configuration=config, runs=runs)
    for run in runs:
        p = root / run.relative_run_root / "coverage.json"
        write_run_coverage_v1(p, run_coverage_v1(run_identity_sha256="a" * 64, unit_ids=("a", "b", "c")))
    write_json_model(root / "campaign.json", result)
    return result


def complete(root, run, ids):
    p = root / run.relative_run_root / "coverage.json"
    c = read_json_model(RunCoverageV1, p)
    done = set(c.completed_unit_ids) | set(ids)
    reconcile_run_coverage_v1(p, c, completed_unit_ids=tuple(t for t in c.unit_ids if t in done))


def unavailable():
    return ProviderUnavailableError(
        "offline unavailable", failure_kind="connection_error", retry_telemetry=RetryTelemetry(1, 1, 0)
    )


def test_fair_order_resumes_native_coverage_without_duplicate_primaries(tmp_path):
    p = plan(tmp_path)
    calls = []
    cancelled = []

    def execute(run, ids):
        if ids:
            calls.append((run.model_name, ids))
            complete(tmp_path, run, ids)

    owner.dispatch_campaign(root=tmp_path, plan=p, execute=execute, cancelled=lambda: cancelled.append(1))
    assert calls == [
        ("first", ("b",)),
        ("second", ("b",)),
        ("second", ("a",)),
        ("first", ("a",)),
        ("first", ("c",)),
        ("second", ("c",)),
    ]
    before = list(calls)
    owner.dispatch_campaign(root=tmp_path, plan=p, execute=execute, cancelled=lambda: cancelled.append(1))
    assert calls == before and not cancelled


def test_outage_defers_one_route_allows_other_work_and_bounded_later_resume(tmp_path, monkeypatch):
    p = plan(tmp_path)
    calls = []
    failures = True
    now = [100.0]
    monkeypatch.setattr(owner.time, "time", lambda: now[0])

    def execute(run, ids):
        if not ids:
            return
        calls.append((run.model_name, ids))
        if run.model_name == "first" and failures:
            raise unavailable()
        complete(tmp_path, run, ids)

    owner.dispatch_campaign(
        root=tmp_path, plan=p, execute=execute, cancelled=lambda: pytest.fail("route cancelled campaign")
    )
    assert calls == [("first", ("b",)), ("second", ("b",)), ("second", ("a",)), ("second", ("c",))]
    owner.dispatch_campaign(root=tmp_path, plan=p, execute=execute, cancelled=lambda: None)
    assert len(calls) == 4  # cooldown persists across process-style invocations
    now[0] += 11
    owner.dispatch_campaign(root=tmp_path, plan=p, execute=execute, cancelled=lambda: None)
    assert len(calls) == 5  # second bounded deferral
    now[0] += 11
    failures = False
    owner.dispatch_campaign(root=tmp_path, plan=p, execute=execute, cancelled=lambda: None)
    assert ("first", ("b",)) not in calls[5:]  # exhausted task is still pending, no infinite retries
    c = read_json_model(RunCoverageV1, tmp_path / "runs/first/coverage.json")
    assert set(c.completed_unit_ids) == set()  # do not cascade a sustained outage across the panel
    events = [json.loads(x) for x in (tmp_path / "dispatch-events.jsonl").read_text().splitlines()]
    assert sum(x["status"] == "provider_deferred" for x in events) == 2


def test_provider_recovery_resumes_original_pending_identity(tmp_path, monkeypatch):
    p = plan(tmp_path)
    failed = [False]
    calls = []
    now = [100.0]
    monkeypatch.setattr(owner.time, "time", lambda: now[0])

    def execute(run, ids):
        if not ids:
            return
        calls.append((run.job_id, ids))
        if run.job_id == "first" and not failed[0]:
            failed[0] = True
            raise unavailable()
        complete(tmp_path, run, ids)

    owner.dispatch_campaign(root=tmp_path, plan=p, execute=execute, cancelled=lambda: None)
    now[0] += 11
    owner.dispatch_campaign(root=tmp_path, plan=p, execute=execute, cancelled=lambda: None)
    assert calls.count(("first", ("b",))) == 2
    assert all(calls.count(("second", (t,))) == 1 for t in ["a", "b", "c"])
    assert len({x[0] for x in calls}) == 2  # no replacement primary runs created


def test_dispatch_contract_and_slice_reject_panel_drift(tmp_path):
    p = plan(tmp_path)
    payload = p.configuration.model_dump()
    payload["dispatch"]["task_order"] = payload["dispatch"]["task_order"][:-1]
    with pytest.raises(ValueError, match="complete panel"):
        CampaignConfigurationV1.model_validate(payload)
    kwargs = dict(
        campaign_root=tmp_path, job_id="first", run_root=tmp_path / "runs/first", full_task_ids=("a", "b", "c")
    )
    assert owner.execution_slice(**kwargs, requested_task_ids=("b",)) == ("b",)
    with pytest.raises(ValueError, match="unscheduled"):
        owner.execution_slice(**kwargs, requested_task_ids=("outside",))
    with pytest.raises(ValueError, match="prepared campaign"):
        owner.execution_slice(**{**kwargs, "campaign_root": None}, requested_task_ids=("b",))
    with pytest.raises(ValueError, match="size bound"):
        owner.execution_slice(**kwargs, requested_task_ids=("b", "a"))


def test_fatal_interruption_keeps_completed_coverage_and_cancels_once(tmp_path):
    p = plan(tmp_path)
    calls = []
    cancelled = []

    def execute(run, ids):
        if not ids:
            return
        calls.append((run.job_id, ids))
        if run.job_id == "second":
            raise RuntimeError("offline fault")
        complete(tmp_path, run, ids)

    with pytest.raises(RuntimeError, match="offline fault"):
        owner.dispatch_campaign(root=tmp_path, plan=p, execute=execute, cancelled=lambda: cancelled.append(1))
    assert cancelled == [1]
    assert read_json_model(RunCoverageV1, tmp_path / "runs/first/coverage.json").completed_unit_ids == ("b",)


def test_parallel_worker_consumes_bounded_input_and_preserves_other_completion():
    reached = threading.Event()
    release = threading.Event()
    yielded = []
    consumed = []

    def tasks():
        for i in range(100):
            yielded.append(i)
            yield i

    def execute(i):
        if i == 0:
            reached.wait(2)
            assert yielded == [0, 1]
            release.set()
            raise unavailable()
        reached.set()
        release.wait(2)
        return i

    with pytest.raises(ProviderUnavailableError):
        run_bounded_tasks(tasks(), execute, workers=2, consume=consumed.append, stop=RunStopSignal())
    assert yielded == [0, 1] and consumed == [1]


def test_order_interleaves_benchmarks_and_rotates_models(tmp_path):
    config = configuration(tmp_path)
    payload = config.model_dump(mode="json")
    payload["panel"]["benchmarks"] = ["trialeval", "trialdev"]
    payload["panel"]["trialeval_tasks"] = ["e1", "e2"]
    priorities = [("trialeval", "e1"), ("trialdev", "b"), ("trialeval", "e2"), ("trialdev", "a"), ("trialdev", "c")]
    payload["dispatch"]["task_order"] = [dict(benchmark=b, task_id=t) for b, t in priorities]
    config = CampaignConfigurationV1.model_validate(payload)
    runs = []
    for model in config.models:
        for benchmark in config.panel.benchmarks:
            name = model.name + "-" + benchmark
            run = CampaignRunV1(
                job_id=name,
                benchmark=benchmark,
                model_name=model.name,
                repetition=1,
                relative_run_root="runs/" + name,
                invocation_sha256="a" * 64,
                analysis_plan_path="plans/" + name + ".json" if benchmark == "trialeval" else None,
                analysis_plan_sha256="b" * 64 if benchmark == "trialeval" else None,
            )
            runs.append(run)
            ids = config.panel.trialeval_tasks if benchmark == "trialeval" else config.panel.trialdev_tasks
            write_run_coverage_v1(
                tmp_path / run.relative_run_root / "coverage.json",
                run_coverage_v1(run_identity_sha256="a" * 64, unit_ids=ids),
            )
    p = CampaignPlanV1(configuration=config, runs=tuple(runs))
    calls = []

    def execute(run, ids):
        if ids:
            calls.append((run.model_name, run.benchmark, ids[0]))
            complete(tmp_path, run, ids)

    owner.dispatch_campaign(root=tmp_path, plan=p, execute=execute, cancelled=lambda: None)
    assert [(b, t) for _, b, t in calls] == [x for x in priorities for _ in range(2)]
    assert [m for m, _, _ in calls[:4]] == ["first", "second", "second", "first"]


def test_terminal_provider_failures_also_exhaust_route_recovery_bound(tmp_path, monkeypatch):
    p = plan(tmp_path)
    now = [100.0]
    calls = []
    cfg = p.configuration.model_copy(
        update={"dispatch": p.configuration.dispatch.model_copy(update={"maximum_consecutive_provider_deferrals": 2})}
    )
    p = CampaignPlanV1(configuration=cfg, runs=p.runs)
    monkeypatch.setattr(owner.time, "time", lambda: now[0])

    def execute(run, ids):
        if not ids:
            return
        calls.append((run.model_name, ids))
        complete(tmp_path, run, ids)
        if run.model_name == "first":
            raise unavailable()  # Eval-style retained failed primary

    for _ in range(3):
        owner.dispatch_campaign(root=tmp_path, plan=p, execute=execute, cancelled=lambda: None)
        now[0] += 11
    assert sum(m == "first" for m, _ in calls) == 2
    assert read_json_model(RunCoverageV1, tmp_path / "runs/first/coverage.json").completed_unit_ids == ("a", "b")


def test_slices_bound_concurrency_and_never_overlap_native_writer(tmp_path):
    p = plan(tmp_path, workers=2, task_workers=2, slices=2)
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active = set()
    maximum = [0]
    seen = []

    def execute(run, ids):
        if not ids:
            return
        with lock:
            assert run.job_id not in active
            active.add(run.job_id)
            maximum[0] = max(maximum[0], len(active))
            seen.extend((run.job_id, t) for t in ids)
        if len(ids) == 2:
            barrier.wait(timeout=3)
        complete(tmp_path, run, ids)
        with lock:
            active.remove(run.job_id)

    owner.dispatch_campaign(root=tmp_path, plan=p, execute=execute, cancelled=lambda: None)
    assert maximum == [2]
    assert len(seen) == len(set(seen)) == 6
