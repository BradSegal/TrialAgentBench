"""Campaign custody and durable commercial boundaries without provider network calls."""

from __future__ import annotations

import csv
import json
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from trialagentbench_verifier.scorer import current_scorer

from trialagentbench.contracts.campaign import CampaignConfigurationV1
from trialagentbench.contracts.trialdev.release import TrialDevStream
from trialagentbench.execution_policy import TRIALDEV_RELEASE_BUDGET_V1
from trialagentbench.tools import campaign
from trialagentbench.tools.run import canonical_trialdev
from trialagentbench.util.campaign_control import (
    CampaignCapacityStop,
    CampaignRequestControl,
    controlled_http_client,
    exhausted_budgets,
)


def configuration(tmp_path: Path) -> CampaignConfigurationV1:
    for name, route in (("first", "route-a"), ("second", "route-b")):
        (tmp_path / f"{name}-capability.json").write_text(
            json.dumps(
                {
                    "provider_transport": "openrouter",
                    "model_id": f"fixture/{name}",
                    "upstream_provider": route,
                    "context_window_tokens": 1000000,
                    "maximum_completion_tokens": 1000,
                    "source_url": "https://provider.example/deterministic-control",
                    "source_retrieved_utc": "2026-09-13T00:00:00Z",
                    "source_payload_sha256": "a" * 64,
                }
            )
        )
    return CampaignConfigurationV1.model_validate(
        {
            "release_directory": str(tmp_path),
            "expected_release_manifest_sha256": "a" * 64,
            "executor_image_id": "sha256:" + "b" * 64,
            "models": [
                {
                    "name": name,
                    "model": f"fixture/{name}",
                    "upstream_provider": route,
                    "capability_snapshot": str(tmp_path / f"{name}-capability.json"),
                    "maximum_completion_tokens": 100,
                    "spend_limit_micro_usd": 5000,
                    "reservation_micro_usd": 1000,
                    "reservation_tokens": 1000,
                    "pricing_premise": "Deterministic transport control; no real billing quote",
                    "requests_per_minute": 100,
                    "tokens_per_minute": 100000,
                }
                for name, route in (("first", "route-a"), ("second", "route-b"))
            ],
            "panel": {"benchmarks": ["trialdev"]},
            "operations": {"model_workers": 2, "executor_workers": 4, "spend_limit_micro_usd": 7000},
        }
    )


@pytest.mark.parametrize("where", ["execute_code", "check_analysis"])
@pytest.mark.parametrize(
    "termination_code",
    [
        "import os; os._exit(1)",
        "allocation = bytearray(8 * 1024**3)",
        "import os, signal; os.kill(os.getpid(), signal.SIGKILL)",
    ],
)
def test_participant_session_loss_with_partial_answer_cannot_schedule_a_retry(
    tmp_path, monkeypatch, where, termination_code
):
    """Exercise the real agent terminal, checkpoint writer, scheduler and independent classifier."""
    from contextlib import nullcontext

    from contracts.test_trialeval_score_source_contracts import _run_config
    from execution.test_trialeval_terminal_telemetry import (
        test_trialeval_terminated_participant_session_ends_only_the_item,
    )
    from trialagentbench_verifier.campaign import INFRASTRUCTURE_ATTEMPT_OUTCOMES_V1, _attempt_outcome

    from trialagentbench.contracts.campaign import CampaignPlanV1, CampaignRunV1
    from trialagentbench.contracts.core.runs import TrialEvalConditionProvenanceV1
    from trialagentbench.contracts.scoring.trialeval_scores import (
        trialeval_infrastructure_outcome_v1,
        trialeval_observed_attempt_outcome_v1,
    )
    from trialagentbench.io import write_json_model
    from trialagentbench.tools import finish_campaign_analysis
    from trialagentbench.tools.run.trialeval import _save_item_result
    from trialagentbench.trialeval import agent
    from trialagentbench.util.run_coverage import load_or_create_run_coverage_v1, reconcile_run_coverage_v1

    outputs = []
    run_agent = agent.run_agent

    def capture(*args, **kwargs):
        output = run_agent(*args, **kwargs)
        outputs.append(output)
        return output

    monkeypatch.setattr(agent, "run_agent", capture)
    test_trialeval_terminated_participant_session_ends_only_the_item(tmp_path, monkeypatch, where, termination_code)
    (output,) = outputs
    native_config = _run_config(task_ids=["TASK1"])
    native_fields = native_config.model_dump(exclude={"run_identity_sha256"})
    provenance = TrialEvalConditionProvenanceV1.model_validate(output["condition_provenance"])
    native_fields["prompt_set_sha256"] = provenance.prompt_set_sha256
    native_fields["experiment_condition"].update(
        maximum_turns_per_step=provenance.max_turns,
        procedure_assistance=provenance.procedure_assistance,
        reporting_interface=provenance.reporting_interface,
    )
    native_fields["task_evidence_factors"]["TASK1"] = {
        "context_configuration": "C4",
        "data_preparation": "raw_domains",
        "analysis_specification": provenance.analysis_specification,
    }
    native_config = type(native_config).create(**native_fields)
    native = tmp_path / "runs/primary"
    coverage = load_or_create_run_coverage_v1(
        native / "coverage.json",
        run_identity_sha256=native_config.run_identity_sha256,
        unit_ids=("TASK1",),
        resume=False,
    )
    write_json_model(native / "run_config.json", native_config)
    _save_item_result(output_dir=native, item_id="TASK1", run_config=native_config, agent_output=output)
    reconcile_run_coverage_v1(native / "coverage.json", coverage, completed_unit_ids=("TASK1",))
    config = configuration(tmp_path)
    config = config.model_copy(
        update={
            "models": config.models[:1],
            "panel": config.panel.model_copy(update={"benchmarks": ("trialeval",)}),
        }
    )
    primary = CampaignRunV1(
        job_id="primary",
        benchmark="trialeval",
        model_name=config.models[0].name,
        repetition=1,
        relative_run_root="runs/primary",
        invocation_sha256="a" * 64,
        analysis_plan_path="plans/primary.json",
        analysis_plan_sha256="b" * 64,
    )
    plan = CampaignPlanV1(configuration=config, runs=(primary,))
    monkeypatch.setattr(campaign, "load_campaign", lambda *_: plan)
    monkeypatch.setattr(campaign, "_require_host_capacity", lambda *_: None)
    monkeypatch.setattr(campaign, "_campaign_inputs", lambda *_: object())
    monkeypatch.setattr(campaign, "_campaign_asset_selections", lambda *_: ())
    monkeypatch.setattr(campaign, "release_assets", lambda *_, **__: nullcontext())
    monkeypatch.setattr(finish_campaign_analysis, "campaign_censoring", lambda *_: None)

    def forbidden(*_, **__):
        pytest.fail("A model-origin session loss must not prepare or execute any retry")

    monkeypatch.setattr(campaign, "_prepare_native_run", forbidden)
    real_run_campaign = campaign.run_campaign
    monkeypatch.setattr(campaign, "run_campaign", forbidden)
    campaign.retry_campaign(tmp_path)
    assert trialeval_infrastructure_outcome_v1(output["failure_type"]) is None
    assert trialeval_infrastructure_outcome_v1("runtime_error") is None
    # Even a reproducible partial answer cannot reclassify the participant's
    # terminal session loss as an evaluator outage.
    completed = SimpleNamespace(status="satisfied")
    outcome = SimpleNamespace(completion=completed, reproducibility=completed)
    assert (
        trialeval_observed_attempt_outcome_v1(
            agent_status="failed", failure_type=output["failure_type"], outcome=outcome
        )
        == "model_noncompletion"
    )
    observed = _attempt_outcome(
        agent_status="failed",
        failure_type=output["failure_type"],
        completion=completed,
        reproducibility=completed,
    )
    assert observed == "model_noncompletion" and observed not in INFRASTRUCTURE_ATTEMPT_OUTCOMES_V1
    assert (tmp_path / "workspace/item/scratch/partial-result.json").is_file()
    # A hand-prepared plan must not bypass the same check on resume.
    retry = primary.model_copy(
        update={
            "job_id": "retry",
            "relative_run_root": "runs/retry",
            "retry_of_job_id": "primary",
        }
    )
    retained = CampaignPlanV1(configuration=config, runs=(primary, retry))
    monkeypatch.setattr(campaign, "load_campaign", lambda *_: retained)
    retry_root = tmp_path / retry.relative_run_root
    retry_root.mkdir(parents=True)
    write_json_model(retry_root / "run_config.json", native_config)
    with pytest.raises(ValueError, match="only after an infrastructure failure"):
        real_run_campaign(tmp_path)


@pytest.mark.parametrize(
    "benchmark,default,flag",
    [("trialeval", 3600, "--item-watchdog-seconds"), ("trialdev", 1800, "--watchdog-seconds")],
)
@pytest.mark.parametrize("override", [None, 77])
def test_campaign_watchdog_uses_suite_default_or_explicit_override(tmp_path, benchmark, default, flag, override):
    config = configuration(tmp_path)
    config = config.model_copy(update={"panel": config.panel.model_copy(update={"watchdog_seconds": override})})
    manifest = SimpleNamespace(
        artifacts=tuple(
            SimpleNamespace(filename=name, sha256="d" * 64)
            for name in ("trialeval-participant.zip", "trialdev-evaluator.zip")
        ),
        executor=SimpleNamespace(environment_sha256="c" * 64),
    )
    args = campaign._native_arguments(
        config, manifest, model=config.models[0], benchmark=benchmark, repetition=1, output=tmp_path / "run"
    )
    assert args[args.index(flag) + 1] == str(default if override is None else override)


@pytest.mark.parametrize("benchmark", ["trialeval", "trialdev"])
def test_withheld_split_runs_against_the_withheld_role_archives(tmp_path, benchmark):
    config = configuration(tmp_path)
    config = config.model_copy(update={"panel": config.panel.model_copy(update={"split": "withheld"})})
    names = (
        "trialeval-participant.zip",
        "trialdev-evaluator.zip",
        "trialeval-withheld-participant.zip",
        "trialdev-withheld-evaluator.zip",
    )
    manifest = SimpleNamespace(
        artifacts=tuple(SimpleNamespace(filename=name, sha256="d" * 64) for name in names),
        executor=SimpleNamespace(environment_sha256="c" * 64),
    )
    args = campaign._native_arguments(
        config, manifest, model=config.models[0], benchmark=benchmark, repetition=1, output=tmp_path / "run"
    )
    flag = "--participant-release" if benchmark == "trialeval" else "--release"
    assert Path(args[args.index(flag) + 1]).name == (
        "trialeval-withheld-participant.zip" if benchmark == "trialeval" else "trialdev-withheld-evaluator.zip"
    )
    public = config.model_copy(update={"panel": config.panel.model_copy(update={"split": "public"})})
    assert "split" not in public.panel.model_dump(mode="json")


@pytest.mark.parametrize("split", ["public", "withheld"])
def test_campaign_finalizer_preserves_the_selected_split(tmp_path, monkeypatch, split):
    from trialagentbench.contracts.campaign import CampaignPlanV1, CampaignRunV1
    from trialagentbench.tools import finish_campaign_analysis, finish_run

    config = configuration(tmp_path)
    config = config.model_copy(
        update={"panel": config.panel.model_copy(update={"split": split}), "models": config.models[:1]}
    )
    run = CampaignRunV1(
        job_id="first-trialdev-1",
        benchmark="trialdev",
        model_name="first",
        repetition=1,
        relative_run_root="runs/first",
        invocation_sha256="a" * 64,
    )
    plan = CampaignPlanV1(configuration=config, runs=(run,))
    calls = []
    monkeypatch.setattr(finish_campaign_analysis, "campaign_censoring", lambda *_: None)
    monkeypatch.setattr(finish_run, "main", lambda arguments, **kwargs: calls.append(arguments))
    campaign._finish_native_runs(tmp_path, plan, SimpleNamespace(), plan.runs)
    assert len(calls) == 1
    assert (calls[0][calls[0].index("--split") + 1] if "--split" in calls[0] else "public") == split


@pytest.mark.parametrize("split", ["public", "withheld"])
def test_native_finalizer_grades_against_the_selected_split(tmp_path, monkeypatch, split):
    from trialagentbench.io import sha256_file
    from trialagentbench.tools import finish_run

    run = tmp_path / "run"
    run.mkdir()
    invocation = run / "invocation.json"
    invocation.write_text("{}")
    manifest = SimpleNamespace(
        artifacts=tuple(
            SimpleNamespace(filename=f"trialdev-{role}.zip", sha256=digest * 64)
            for role, digest in (("evaluator", "a"), ("withheld-evaluator", "b"))
        )
    )
    monkeypatch.setattr(finish_run, "authenticate_release_set_manifest", lambda *args, **kwargs: manifest)

    def grade(arguments):
        role, digest = ("withheld-evaluator", "b") if split == "withheld" else ("evaluator", "a")
        assert Path(arguments[arguments.index("--release") + 1]).name == f"trialdev-{role}.zip"
        assert arguments[arguments.index("--expected-sha256") + 1] == digest * 64
        raise RuntimeError("grading intercepted after release selection")

    monkeypatch.setattr(finish_run, "harness_main", grade)
    with pytest.raises(RuntimeError, match="grading intercepted"):
        finish_run.main(
            [
                "trialdev",
                "--release-dir",
                str(tmp_path),
                "--expected-manifest-sha256",
                "c" * 64,
                "--run-root",
                str(run),
                "--output",
                str(tmp_path / "finished"),
                "--expected-invocation-sha256",
                sha256_file(invocation),
                "--split",
                split,
            ],
            qualified_manifest=manifest,
        )


def request(model="first", route="route-a") -> bytes:
    return json.dumps(
        {"model": f"fixture/{model}", "provider": {"only": [route], "allow_fallbacks": False}, "max_tokens": 100}
    ).encode()


@pytest.mark.parametrize("benchmark", ["trialeval", "trialdev"])
@pytest.mark.parametrize("mode", ["default", "exclude", "replay"])
def test_campaign_records_and_forwards_returned_reasoning_condition(tmp_path, benchmark, mode):
    """Returned reasoning is retained by default; exclusion and replay are explicit, recorded conditions."""
    from trialagentbench.tools.run import trialeval
    from trialagentbench.util.experiment_condition import resolve_experiment_condition_v1

    config = configuration(tmp_path)
    model = config.models[0].model_copy(
        update={"exclude_reasoning": mode == "exclude", "replay_reasoning": mode == "replay"}
    )
    manifest = SimpleNamespace(
        artifacts=tuple(
            SimpleNamespace(filename=name, sha256="d" * 64)
            for name in ("trialeval-participant.zip", "trialdev-evaluator.zip")
        ),
        executor=SimpleNamespace(environment_sha256="c" * 64),
    )
    arguments = campaign._native_arguments(
        config, manifest, model=model, benchmark=benchmark, repetition=1, output=tmp_path / "run"
    )
    assert ("--exclude-returned-reasoning" in arguments) is (mode == "exclude")
    assert ("--replay-returned-reasoning" in arguments) is (mode == "replay")
    owner = trialeval if benchmark == "trialeval" else canonical_trialdev
    parsed = owner._parser().parse_args(arguments)
    condition = resolve_experiment_condition_v1(
        condition_id="diagnostic",
        request_replicate_id="1",
        reasoning_effort=None,
        provider_capability_snapshot=model.capability_snapshot,
        provider=model.provider,
        model=model.model,
        openrouter_provider=model.upstream_provider,
        procedure_assistance="output_contract_only",
        maximum_turns_per_step=12,
        tool_choice="auto",
        exclude_reasoning=parsed.exclude_returned_reasoning,
        replay_reasoning=parsed.replay_returned_reasoning,
    )
    reasoning = condition.model_dump()["reasoning"]
    assert reasoning["exclude_from_response"] is (mode == "exclude")
    # Defaults are not serialized, so existing campaign and condition identities are unchanged.
    assert ("replay_in_context" in reasoning) is (mode == "replay")
    assert ("exclude_reasoning" in model.model_dump()) is (mode == "exclude")
    assert ("replay_reasoning" in model.model_dump()) is (mode == "replay")


def test_campaign_rejects_replaying_excluded_reasoning(tmp_path):
    config = configuration(tmp_path)
    model = config.models[0]
    with pytest.raises(ValueError, match="cannot be replayed"):
        type(model).model_validate(model.model_dump() | {"exclude_reasoning": True, "replay_reasoning": True})


@pytest.mark.parametrize(
    "changes",
    [
        {"model": "other/model"},
        {"max_tokens": 101},
        {"stream": True},
        {"plugins": [{}]},
        {"plugins": [{"id": "web"}]},
        {"web_search_options": {"search_context_size": "low"}},
        {"tools": [{"type": "openrouter:web_search"}]},
        {"tools": [{"type": "web_search_preview"}]},
        {"model": "fixture/first:online"},
        {"service_tier": "priority"},
        {"provider": {"only": ["route-a"], "allow_fallbacks": True}},
    ],
)
def test_campaign_rejects_unplanned_billable_features(tmp_path: Path, changes: dict[str, object]) -> None:
    from pydantic import ValidationError

    config = configuration(tmp_path)
    control = CampaignRequestControl(database=tmp_path / "requests.sqlite", configuration=config, model_name="first")
    payload = json.loads(request()) | changes
    with pytest.raises((ValueError, ValidationError)):
        control.reserve(json.dumps(payload).encode())
    with sqlite3.connect(control.database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM attempts").fetchone() == (0,)


@pytest.mark.parametrize(
    "route", ["openai/gpt-5:online", "perplexity/sonar-pro", "openai/gpt-4o-search-preview", "openai/o3-deep-research"]
)
def test_campaign_configuration_rejects_web_enabled_routes(tmp_path: Path, route: str) -> None:
    from pydantic import ValidationError

    payload = configuration(tmp_path).model_dump(mode="json")
    payload["models"][0]["model"] = route
    with pytest.raises(ValidationError, match="built-in web retrieval"):
        type(configuration(tmp_path)).model_validate(payload)


def test_campaign_reservations_are_atomic_and_retain_ambiguous_attempts(tmp_path: Path) -> None:
    config = configuration(tmp_path)
    control = CampaignRequestControl(database=tmp_path / "requests.sqlite", configuration=config, model_name="first")
    with ThreadPoolExecutor(max_workers=5) as pool:
        attempts = list(pool.map(lambda _: control.reserve(request()), range(5)))
    assert len(set(attempts)) == 5
    with pytest.raises(CampaignCapacityStop, match="exhausted"):
        control.reserve(request())
    control.settle(attempts[0], b'{"choices":[{}],"usage":{}}')
    restarted = CampaignRequestControl(database=control.database, configuration=config, model_name="first")
    with pytest.raises(CampaignCapacityStop, match="exhausted"):
        restarted.reserve(request())
    restarted.settle(attempts[0], b'{"choices":[{}],"usage":{"cost":0.0000001}}')
    with sqlite3.connect(control.database) as connection:
        assert connection.execute("SELECT SUM(charge) FROM attempts").fetchone() == (4001,)
    second = CampaignRequestControl(database=control.database, configuration=config, model_name="second")
    second.reserve(request("second", "route-b"))
    second.reserve(request("second", "route-b"))
    with pytest.raises(CampaignCapacityStop, match="exhausted"):
        second.reserve(request("second", "route-b"))
    with pytest.raises(ValueError, match="already settled"):
        control.settle(attempts[0], b'{"choices":[{}],"usage":{"cost":0}}')


@pytest.mark.parametrize("campaign_wide", [False, True])
def test_active_transport_settlement_admits_a_waiting_worker(tmp_path, monkeypatch, campaign_wide):
    from threading import Event

    config = configuration(tmp_path)
    if campaign_wide:
        config = config.model_copy(
            update={"operations": config.operations.model_copy(update={"spend_limit_micro_usd": 1200})}
        )
    else:
        config = config.model_copy(
            update={"models": (config.models[0].model_copy(update={"spend_limit_micro_usd": 1200}), config.models[1])}
        )
    first = CampaignRequestControl(database=tmp_path / "requests.sqlite", configuration=config, model_name="first")
    waiting = CampaignRequestControl(database=first.database, configuration=config, model_name="first")
    active = first.reserve(request(), timeout_seconds=2, active_transport=True)
    blocked = Event()
    original = waiting.try_reserve

    def observe(**kwargs):
        result = original(**kwargs)
        if result is None:
            blocked.set()
        return result

    monkeypatch.setattr(waiting, "try_reserve", observe)
    with ThreadPoolExecutor(max_workers=1) as workers:
        future = workers.submit(waiting.reserve, request(), timeout_seconds=2)
        assert blocked.wait(1)
        assert not future.done()
        first.settle(active, b'{"choices":[{}],"usage":{"cost":0.0001}}')
        assert future.result(timeout=2) > active
    with sqlite3.connect(first.database) as connection:
        assert connection.execute("SELECT SUM(charge) FROM attempts").fetchone() == (1100,)


@pytest.mark.parametrize("campaign_wide", [False, True])
def test_permanent_budget_exhaustion_does_not_wait_for_active_transport(tmp_path, monkeypatch, campaign_wide):
    import time

    config = configuration(tmp_path)
    if campaign_wide:
        config = config.model_copy(
            update={"operations": config.operations.model_copy(update={"spend_limit_micro_usd": 1200})}
        )
    else:
        config = config.model_copy(
            update={"models": (config.models[0].model_copy(update={"spend_limit_micro_usd": 1200}), config.models[1])}
        )
    control = CampaignRequestControl(database=tmp_path / "requests.sqlite", configuration=config, model_name="first")
    model = config.models[0]
    control.try_reserve(model=model, now=time.monotonic(), micro_usd=300)
    control.try_reserve(model=model, now=time.monotonic(), micro_usd=900, active_until=time.monotonic() + 60)
    monkeypatch.setattr(time, "sleep", lambda _: pytest.fail("Permanent exhaustion must not wait"))
    with pytest.raises(CampaignCapacityStop, match="retained 300.*requested 1000") as failure:
        control.reserve(request())
    assert failure.value.campaign_wide is campaign_wide


def test_spending_limit_exhaustion_is_recorded_and_never_reopens(tmp_path: Path, capsys) -> None:
    """Budgets never refill: a recorded exhaustion refuses every later request of its scope."""
    import time

    from trialagentbench.util.campaign_control import CAMPAIGN_BUDGET_SCOPE, exhausted_budgets, print_spending_status

    config = configuration(tmp_path)
    first = CampaignRequestControl(database=tmp_path / "requests.sqlite", configuration=config, model_name="first")
    second = CampaignRequestControl(database=first.database, configuration=config, model_name="second")
    attempts = [first.try_reserve(model=config.models[0], now=time.monotonic(), micro_usd=1000) for _ in range(5)]
    assert exhausted_budgets(first.database, config) == {}
    with pytest.raises(CampaignCapacityStop, match="Model spending reservations exhausted") as stop:
        first.try_reserve(model=config.models[0], now=time.monotonic(), micro_usd=1000)
    assert stop.value.campaign_wide is False
    assert set(exhausted_budgets(first.database, config)) == {"first"}
    # A later settlement lowers retained spending, but cannot reopen the exhausted model budget.
    attempt = attempts[0]
    assert attempt is not None
    first.settle(attempt, b'{"choices":[{}],"usage":{"cost":0.000001}}')
    with pytest.raises(CampaignCapacityStop, match="Model spending reservations exhausted"):
        first.try_reserve(model=config.models[0], now=time.monotonic(), micro_usd=1)
    for _ in range(2):
        assert second.try_reserve(model=config.models[1], now=time.monotonic(), micro_usd=1000) is not None
    with pytest.raises(CampaignCapacityStop, match="Campaign spending reservations exhausted") as stop:
        second.try_reserve(model=config.models[1], now=time.monotonic(), micro_usd=1000)
    assert stop.value.campaign_wide is True
    with pytest.raises(CampaignCapacityStop, match="Campaign spending reservations exhausted"):
        second.try_reserve(model=config.models[1], now=time.monotonic(), micro_usd=1)
    assert set(exhausted_budgets(first.database, config)) == {"first", CAMPAIGN_BUDGET_SCOPE}
    print_spending_status(first.database, config)
    status = capsys.readouterr().out
    assert "Spending limit exhausted (model first)" in status and "Spending limit exhausted (campaign)" in status


def test_finish_censors_only_pending_work_behind_an_exhausted_budget(tmp_path: Path) -> None:
    """Pending tasks are budget-censored only when their spending limit is recorded as exhausted."""
    import time

    from trialagentbench.contracts.campaign import CampaignPlanV1, CampaignRunV1
    from trialagentbench.contracts.core.runs import RunCoverageV1
    from trialagentbench.io import write_json_model
    from trialagentbench.tools.finish_campaign_analysis import campaign_censoring

    config = configuration(tmp_path)
    plan = CampaignPlanV1(
        configuration=config,
        runs=tuple(
            CampaignRunV1(
                job_id=f"{model.name}-trialdev-1",
                benchmark="trialdev",
                model_name=model.name,
                repetition=1,
                relative_run_root=f"runs/{model.name}",
                invocation_sha256="e" * 64,
            )
            for model in config.models
        ),
    )
    root = tmp_path / "campaign"
    tasks = ("td-1", "td-2", "td-3", "td-4")

    def coverage(name: str, completed: tuple[str, ...]) -> None:
        (root / "runs" / name).mkdir(parents=True, exist_ok=True)
        write_json_model(
            root / "runs" / name / "coverage.json",
            RunCoverageV1(
                run_identity_sha256="a" * 64, schedule_sha256="b" * 64, unit_ids=tasks, completed_unit_ids=completed
            ),
        )

    coverage("first", tasks)
    coverage("second", tasks[:2])
    assert campaign_censoring(root, plan.model_copy(update={"runs": plan.runs[:1]})) is None
    # A crashed or interrupted run can resume, so finishing it is refused.
    with pytest.raises(ValueError, match="second-trialdev-1 has 2 pending tasks without an exhausted spending limit"):
        campaign_censoring(root, plan)
    control = CampaignRequestControl(database=root / "requests.sqlite", configuration=config, model_name="second")
    for _ in range(5):
        control.try_reserve(model=config.models[1], now=time.monotonic(), micro_usd=1000)
    with pytest.raises(CampaignCapacityStop):
        control.try_reserve(model=config.models[1], now=time.monotonic(), micro_usd=1)
    censoring = campaign_censoring(root, plan)
    assert censoring is not None
    assert censoring.exhausted_models == ("second",) and not censoring.campaign_budget_exhausted
    assert censoring.censored_by_job("trialdev") == {
        "first-trialdev-1": frozenset(),
        "second-trialdev-1": frozenset({"td-3", "td-4"}),
    }
    assert censoring.paired_task_ids("trialdev") == {"td-1", "td-2"}
    assert [(row.model_id, row.budget_censored_tasks, row.censoring_rate) for row in censoring.models] == [
        ("fixture/first", 0, 0.0),
        ("fixture/second", 2, 0.5),
    ]
    assert len(censoring.warnings) == 1 and "differs materially" in censoring.warnings[0]
    # Another model's exhaustion never explains this model's unfinished work.
    coverage("first", tasks[:3])
    with pytest.raises(ValueError, match="first-trialdev-1 has 1 pending tasks"):
        campaign_censoring(root, plan)


def test_active_reservation_timeout_and_expiry_never_release_liability(tmp_path, monkeypatch):
    import time

    config = configuration(tmp_path)
    config = config.model_copy(
        update={"models": (config.models[0].model_copy(update={"spend_limit_micro_usd": 1200}), config.models[1])}
    )
    control = CampaignRequestControl(database=tmp_path / "requests.sqlite", configuration=config, model_name="first")
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(time, "time", lambda: 1000 + clock.now)
    monkeypatch.setattr(time, "sleep", lambda seconds: setattr(clock, "now", clock.now + seconds))
    control.reserve(request(), timeout_seconds=1, active_transport=True)
    with pytest.raises(CampaignCapacityStop, match="capacity window"):
        control.reserve(request(), timeout_seconds=0.5)
    assert clock.now == 0.5
    # An abandoned/crashed transport lease becomes permanent, never free capacity.
    clock.now = 1.1
    with pytest.raises(CampaignCapacityStop, match="Model spending reservations exhausted"):
        control.reserve(request(), timeout_seconds=0.5)
    assert clock.now == 1.1
    with sqlite3.connect(control.database) as connection:
        assert connection.execute("SELECT charge,settled,active_until FROM attempts").fetchall() == [(1000, 0, 1)]


@pytest.mark.parametrize("signal", ["cancel", "overrun"])
def test_waiting_reservation_observes_stop_or_overrun(tmp_path, monkeypatch, signal):
    import time

    config = configuration(tmp_path)
    config = config.model_copy(
        update={"models": (config.models[0].model_copy(update={"spend_limit_micro_usd": 1200}), config.models[1])}
    )
    control = CampaignRequestControl(database=tmp_path / "requests.sqlite", configuration=config, model_name="first")
    active = control.reserve(request(), timeout_seconds=2, active_transport=True)

    def interrupt_wait(_):
        if signal == "cancel":
            control.set_cancelled(True)
        else:
            with pytest.raises(CampaignCapacityStop, match="Observed provider usage"):
                control.settle(active, b'{"choices":[{}],"usage":{"cost":0.002}}')

    monkeypatch.setattr(time, "sleep", interrupt_wait)
    with pytest.raises(CampaignCapacityStop, match="stop requested" if signal == "cancel" else "previous charge"):
        control.reserve(request(), timeout_seconds=1)
    with sqlite3.connect(control.database) as connection:
        assert connection.execute("SELECT COUNT(*),SUM(charge) FROM attempts").fetchone() == (
            1,
            1000 if signal == "cancel" else 2000,
        )


def test_explicit_zero_http_timeout_never_transports_or_reserves(tmp_path, monkeypatch):
    import httpx2

    control = CampaignRequestControl(
        database=tmp_path / "requests.sqlite", configuration=configuration(tmp_path), model_name="first"
    )
    monkeypatch.setattr(
        httpx2.Client, "send", lambda *args, **kwargs: pytest.fail("Expired request reached transport")
    )
    with controlled_http_client(control) as client:
        with pytest.raises(CampaignCapacityStop, match="capacity window"):
            client.post("https://openrouter.ai/api/v1/chat/completions", content=request(), timeout=0)
    with sqlite3.connect(control.database) as connection:
        assert connection.execute("SELECT COUNT(*),COALESCE(SUM(charge),0) FROM attempts").fetchone() == (0, 0)


def test_capacity_lock_wait_respects_http_deadline_without_reserving(tmp_path, monkeypatch):
    import time

    import httpx2

    control = CampaignRequestControl(
        database=tmp_path / "requests.sqlite", configuration=configuration(tmp_path), model_name="first"
    )
    monkeypatch.setattr(
        httpx2.Client, "send", lambda *args, **kwargs: pytest.fail("Expired request reached transport")
    )

    def send():
        client.post("https://openrouter.ai/api/v1/chat/completions", content=request(), timeout=0.05)

    with controlled_http_client(control) as client, ThreadPoolExecutor(max_workers=1) as workers:
        lock = sqlite3.connect(control.database)
        lock.execute("BEGIN IMMEDIATE")
        started = time.monotonic()
        try:
            future = workers.submit(send)
            with pytest.raises(CampaignCapacityStop, match="capacity window"):
                future.result(timeout=0.5)
            assert time.monotonic() - started < 0.25
        finally:
            lock.rollback()
            lock.close()
    with sqlite3.connect(control.database) as connection:
        assert connection.execute("SELECT COUNT(*),COALESCE(SUM(charge),0) FROM attempts").fetchone() == (0, 0)


def test_capacity_lock_released_before_deadline_admits_transport(tmp_path, monkeypatch):
    from threading import Event

    import httpx2

    control = CampaignRequestControl(
        database=tmp_path / "requests.sqlite", configuration=configuration(tmp_path), model_name="first"
    )
    waiting = Event()
    original = control.try_reserve

    def observe(**kwargs):
        result = original(**kwargs)
        if result is None:
            waiting.set()
        return result

    def transport(self, outgoing, **kwargs):
        assert 0 < outgoing.extensions["timeout"]["read"] < 1
        return httpx2.Response(200, request=outgoing, json={"choices": [{}], "usage": {"cost": 0.0001}})

    def send():
        return client.post("https://openrouter.ai/api/v1/chat/completions", content=request(), timeout=1)

    monkeypatch.setattr(control, "try_reserve", observe)
    monkeypatch.setattr(httpx2.Client, "send", transport)
    with controlled_http_client(control) as client, ThreadPoolExecutor(max_workers=1) as workers:
        lock = sqlite3.connect(control.database)
        lock.execute("BEGIN IMMEDIATE")
        try:
            future = workers.submit(send)
            assert waiting.wait(0.5)
        finally:
            lock.rollback()
            lock.close()
        assert future.result(timeout=2).is_success
    with sqlite3.connect(control.database) as connection:
        assert connection.execute("SELECT charge,settled,active_until FROM attempts").fetchall() == [(100, 1, 0)]


def test_request_expiring_inside_transaction_never_inserts_or_sleeps(tmp_path, monkeypatch):
    import time

    control = CampaignRequestControl(
        database=tmp_path / "requests.sqlite", configuration=configuration(tmp_path), model_name="first"
    )
    clock = iter((0.0, 0.0, 0.0))
    monkeypatch.setattr(time, "monotonic", lambda: next(clock, 2.0))
    monkeypatch.setattr(time, "sleep", lambda _: pytest.fail("Expired request must not sleep"))
    with pytest.raises(CampaignCapacityStop, match="capacity window"):
        control.reserve(request(), timeout_seconds=1, active_transport=True)
    with sqlite3.connect(control.database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM attempts").fetchone() == (0,)


def test_unexpected_spending_database_error_is_not_retried(tmp_path):
    control = CampaignRequestControl(
        database=tmp_path / "requests.sqlite", configuration=configuration(tmp_path), model_name="first"
    )
    with sqlite3.connect(control.database) as connection:
        connection.execute("DROP TABLE attempts")
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        control.reserve(request(), timeout_seconds=0.1)


def test_process_death_and_competing_workers_preserve_request_liabilities(tmp_path: Path) -> None:
    """A killed transport owner still consumes capacity after independent workers restart."""
    config = configuration(tmp_path)
    configuration_path = tmp_path / "configuration.json"
    configuration_path.write_text(config.model_dump_json(), encoding="utf-8")
    database = tmp_path / "requests.sqlite"
    # Initialize the same durable boundary a campaign creates before dispatching workers.
    CampaignRequestControl(database=database, configuration=config, model_name="first")
    script = """
import sys
from pathlib import Path
from trialagentbench.contracts.campaign import CampaignConfigurationV1
from trialagentbench.util.campaign_control import CampaignCapacityStop, CampaignRequestControl
configuration = CampaignConfigurationV1.model_validate_json(Path(sys.argv[1]).read_text())
control = CampaignRequestControl(database=Path(sys.argv[2]), configuration=configuration, model_name='first')
try:
    attempt = control.reserve(b'{"model":"fixture/first","provider":{"only":["route-a"],"allow_fallbacks":false},"max_tokens":100}')
except CampaignCapacityStop:
    print('capacity', flush=True)
else:
    if sys.argv[3] == 'hold':
        from trialagentbench.util.provider_telemetry import start_provider_request_v1
        log = Path(sys.argv[2]).parent / 'provider_responses.jsonl'
        start_provider_request_v1(path=log, benchmark='trialeval', unit_id='task', phase_id='task',
            step_id='analysis', turn_index=1, requested_model='fixture/first', provider_route='openrouter:route-a')
        with log.open('ab') as stream:
            stream.write(b'{"request_id":')
            stream.flush()
    print(attempt, flush=True)
    if sys.argv[3] == 'hold':
        sys.stdin.read()
"""
    arguments = [sys.executable, "-c", script, str(configuration_path), str(database)]
    with subprocess.Popen(
        [*arguments, "hold"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ) as interrupted:
        try:
            assert interrupted.stdout is not None
            committed = int(interrupted.stdout.readline())
        finally:
            interrupted.kill()
            interrupted.communicate(timeout=10)

    from trialagentbench.contracts.core.runs import RunCoverageV1
    from trialagentbench.util.provider_telemetry import summarize_provider_telemetry_v1

    log = tmp_path / "provider_responses.jsonl"
    preserved = log.read_bytes()
    summary = summarize_provider_telemetry_v1(
        run_root=tmp_path,
        coverage=RunCoverageV1(
            run_identity_sha256="a" * 64,
            schedule_sha256="b" * 64,
            unit_ids=("task",),
            completed_unit_ids=(),
        ),
    )
    assert summary.interrupted_request_count == 1
    assert summary.log_recoveries[0].torn_tail_bytes == len(b'{"request_id":')
    assert summary.reported_cost_usd == summary.total_tokens == 0
    assert log.read_bytes() == preserved

    def reserve(_: int) -> str:
        return subprocess.check_output([*arguments, "reserve"], text=True, timeout=30).strip()

    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(workers.map(reserve, range(8)))
    assert results.count("capacity") == 4
    accepted = [int(result) for result in results if result != "capacity"]
    assert len(set([committed, *accepted])) == 5
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*), SUM(charge) FROM attempts").fetchone() == (5, 5000)
    recovered = CampaignRequestControl(database=database, configuration=config, model_name="first")
    with pytest.raises(CampaignCapacityStop, match="exhausted"):
        recovered.reserve(request())


def test_interruption_drains_workers_and_durably_stops_new_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from threading import Event, Lock

    from trialagentbench.util.parallel import run_bounded_tasks
    from trialagentbench.util.reported_cost import RunStopSignal

    config = configuration(tmp_path)
    control = CampaignRequestControl(database=tmp_path / "requests.sqlite", configuration=config, model_name="first")
    cancelled, companion_started = Event(), Event()
    lock = Lock()
    active = 0
    original = control.set_cancelled

    def cancel(value: bool) -> None:
        original(value)
        if value:
            cancelled.set()

    monkeypatch.setattr(control, "set_cancelled", cancel)

    def execute(index: int) -> None:
        nonlocal active
        with lock:
            active += 1
        try:
            if index == 0:
                assert companion_started.wait(5)
                raise RuntimeError("injected interruption")
            companion_started.set()
            assert cancelled.wait(5)
            with pytest.raises(CampaignCapacityStop):
                control.reserve(request())
        finally:
            with lock:
                active -= 1

    with pytest.raises(RuntimeError, match="injected interruption"):
        run_bounded_tasks(
            range(20), execute, workers=2, consume=lambda _: None, stop=RunStopSignal(), request_control=control
        )
    assert active == 0 and cancelled.is_set()
    restarted = CampaignRequestControl(database=control.database, configuration=config, model_name="first")
    with pytest.raises(CampaignCapacityStop):
        restarted.reserve(request())
    restarted.set_cancelled(False)
    assert restarted.reserve(request()) > 0


def test_campaign_rate_windows_routes_and_excess_cost(tmp_path: Path) -> None:
    config = configuration(tmp_path)
    model = config.models[0].model_copy(update={"requests_per_minute": 1})
    config = config.model_copy(update={"models": (model, config.models[1])})
    first = CampaignRequestControl(database=tmp_path / "requests.sqlite", configuration=config, model_name="first")
    attempt = first.try_reserve(model=model, now=100)
    assert attempt is not None
    assert first.try_reserve(model=model, now=159) is None
    second = CampaignRequestControl(database=first.database, configuration=config, model_name="second")
    assert second.try_reserve(model=config.models[1], now=159) is not None
    assert first.try_reserve(model=model, now=161) is not None
    with pytest.raises(CampaignCapacityStop, match="exceeded"):
        first.settle(attempt, b'{"choices":[{}],"usage":{"cost":0.002}}')
    with pytest.raises(CampaignCapacityStop, match="previous charge"):
        second.reserve(request("second", "route-b"))
    changed = config.model_copy(update={"analysis": config.analysis.model_copy(update={"seed": 42})})
    with pytest.raises(ValueError, match="different frozen campaign"):
        CampaignRequestControl(database=first.database, configuration=changed, model_name="first")


def test_status_retains_unknown_charges_without_mutating_ledger(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from trialagentbench.util.campaign_control import print_spending_status

    config = configuration(tmp_path)
    database = tmp_path / "requests.sqlite"
    print_spending_status(database, config)
    assert not database.exists()
    control = CampaignRequestControl(database=database, configuration=config, model_name="first")
    control.reserve(request())
    before = database.read_bytes()
    print_spending_status(database, config)
    output = capsys.readouterr().out
    assert "unsettled attempts=1" in output and "charged-or-reserved microUSD=1000" in output
    assert database.read_bytes() == before


def test_preview_rejects_unknown_tasks_before_preparation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from zipfile import ZipFile

    config = configuration(tmp_path)
    task = {
        "task_id": "td-000000000001",
        "case_id": "tdw-0001",
        "stream": "evidence_analysis",
        "title": "Control",
        "clinical_question": "Compare the observed candidates.",
        "available_materials": [{"relative_path": "cohort.parquet", "description": "Observed cohort"}],
        "requested_outputs": ["comparative estimates"],
    }
    with ZipFile(tmp_path / "trialdev-evaluator.zip", "w") as archive:
        archive.writestr("participant/tasks/td-000000000001/task.json", json.dumps(task))
        ready = task | {"task_id": "td-000000000002", "data_view": "analysis_ready"}
        archive.writestr("participant/tasks/td-000000000002/task.json", json.dumps(ready))
    monkeypatch.setattr(
        campaign,
        "_campaign_inputs",
        lambda _: SimpleNamespace(artifacts=(SimpleNamespace(filename="trialdev-evaluator.zip", sha256="d" * 64),)),
    )
    campaign.preview_panel(config)
    assert "2 scheduled attempts" in capsys.readouterr().out
    explicit = config.model_copy(
        update={"panel": config.panel.model_copy(update={"trialdev_tasks": (task["task_id"], ready["task_id"])})}
    )
    inventory = tmp_path / "selection.csv"
    campaign.preview_panel(explicit, inventory=inventory)
    preview = capsys.readouterr().out
    assert "4 scheduled attempts" in preview
    assert "analysis_ready: 1 tasks" in preview
    with inventory.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["task_id"] for row in rows] == [task["task_id"], ready["task_id"]]
    assert [row["presentation"] for row in rows] == ["raw", "analysis_ready"]
    assert all(row["attempts"] == "2" for row in rows)
    assert set(rows[0]) == {
        "benchmark",
        "task_id",
        "world_or_case",
        "presentation",
        "stream",
        "models",
        "repetitions",
        "attempts",
    }
    campaign.preview_panel(explicit, inventory=inventory)
    changed = config.model_copy(update={"panel": config.panel.model_copy(update={"trialdev_tasks": ("unknown",)})})
    with pytest.raises(ValueError, match="matched no released task"):
        campaign.preview_panel(changed)


def test_offline_preview_authenticates_only_selected_participant_bytes(tmp_path, monkeypatch, capsys):
    """An unconfigured model can preview local tasks without provider or evaluator access."""
    from zipfile import ZipFile

    from trialagentbench_verifier.io import sha256_file

    task = {
        "task_id": "td-000000000001",
        "case_id": "tdw-0001",
        "stream": "evidence_analysis",
        "title": "Control",
        "clinical_question": "Compare the observed candidates.",
        "available_materials": [{"relative_path": "cohort.parquet", "description": "Observed cohort"}],
        "requested_outputs": ["comparative estimates"],
    }
    archive_path = tmp_path / "trialdev-participant.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("tasks/td-000000000001/task.json", json.dumps(task))
    manifest_path = tmp_path / "release-set-manifest.json"
    manifest_path.write_text("{}")
    path = tmp_path / "campaign.json"
    path.write_text(
        json.dumps(
            {
                "release_directory": ".",
                "expected_release_manifest_sha256": sha256_file(manifest_path),
                "panel": {"benchmarks": ["trialdev"]},
                "models": [{"name": "my-agent", "model": "NOT_CONFIGURED_YET"}],
            }
        )
    )
    metadata = SimpleNamespace(
        artifacts=(SimpleNamespace(filename=archive_path.name, sha256=sha256_file(archive_path)),)
    )
    monkeypatch.setattr(campaign, "authenticate_release_set_manifest", lambda *args, **kwargs: metadata)

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline preview entered execution preparation")

    monkeypatch.setattr(campaign, "load_configuration", forbidden)
    monkeypatch.setattr(campaign, "validate_model_capabilities", forbidden)
    inventory = tmp_path / "selection.csv"
    assert campaign.main(["preview", "--config", str(path), "--inventory", str(inventory)]) == 0
    assert "1 scheduled attempts" in capsys.readouterr().out
    assert task["task_id"] in inventory.read_text()
    assert not (tmp_path / "trialdev-evaluator.zip").exists()


def test_preview_uses_canonical_trialeval_context_without_assumption_filter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from zipfile import ZipFile

    from trialagentbench_test_helpers import fixture_trialeval_key_record, write_trialeval_grading_key

    config = configuration(tmp_path)
    config = config.model_copy(update={"panel": config.panel.model_copy(update={"benchmarks": ("trialeval",)})})
    records = tuple(
        fixture_trialeval_key_record(number, context=context, cell_ref=f"cell_{cell:020x}")
        for number, context, cell in ((1, "C1", 1), (2, "C5", 1), (3, "C5", 2))
    )
    entries = [{"task_id": record.task_id, "context_tier": record.context} for record in records]
    key_root = tmp_path / "key"
    write_trialeval_grading_key(key_root, records)
    with ZipFile(tmp_path / "trialeval-evaluator.zip", "w") as archive:
        for path in sorted(key_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(key_root).as_posix())
    monkeypatch.setattr(
        campaign,
        "_campaign_inputs",
        lambda _: SimpleNamespace(artifacts=(SimpleNamespace(filename="trialeval-evaluator.zip", sha256="d" * 64),)),
    )
    inventory = tmp_path / "trialeval-selection.csv"
    campaign.preview_panel(config, inventory=inventory)
    assert "2 tasks/model/repetition; 4 scheduled attempts" in capsys.readouterr().out
    with inventory.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["task_id"] for row in rows] == [entries[1]["task_id"], entries[2]["task_id"]]
    assert "effect_state" not in inventory.read_text()
    explicit = config.model_copy(
        update={"panel": config.panel.model_copy(update={"trialeval_tasks": (entries[0]["task_id"],)})}
    )
    campaign.preview_panel(explicit)
    assert "1 tasks/model/repetition; 2 scheduled attempts" in capsys.readouterr().out

    participant = tmp_path / "trialeval-participant.zip"
    with ZipFile(participant, "w") as archive:
        for row in entries:
            task_id = row["task_id"]
            artifacts = (("protocol", "documents/protocol.md"),) + (
                (
                    ("analysis_data", "data/analysis.parquet"),
                    ("analysis_plan", "documents/statistical_analysis_plan.md"),
                )
                if row["context_tier"] == "C1"
                else (
                    ("participant_records", "data/raw/participant_records.parquet"),
                    ("event_history", "data/raw/event_history.parquet"),
                    ("data_integrity_policy", "documents/data_integrity_policy.md"),
                )
            )
            task = {
                "schema_id": "ariadne.trialeval.task/v1",
                "task_id": task_id,
                "title": "Clinical source-record analysis",
                "study_design": "Randomized trial",
                "question": "Estimate the treatment contrast.",
                "available_materials": [
                    {"artifact_id": name, "relative_path": f"tasks/{task_id}/{path}", "description": name}
                    for name, path in artifacts
                ],
                "requested_outputs": ["Estimate and interval"],
                "target_ids": ["primary"],
            }
            archive.writestr(f"tasks/{task_id}/task.json", json.dumps(task))
    participant_inventory = tmp_path / "participant-selection.csv"
    campaign.preview_panel(config, inventory=participant_inventory, sources={"trialeval": participant})
    with participant_inventory.open(newline="") as handle:
        participant_rows = list(csv.DictReader(handle))
    assert [row["task_id"] for row in participant_rows] == [row["task_id"] for row in rows]
    assert all(row["world_or_case"] == "" for row in participant_rows)


def test_preflight_rejects_route_and_capacity_drift_before_creating_runs(tmp_path: Path) -> None:
    config = configuration(tmp_path)
    path = config.models[1].capability_snapshot
    original = json.loads(path.read_text())
    for changes, message in (
        ({"upstream_provider": "other"}, "does not match"),
        ({"maximum_completion_tokens": 50}, "exceed"),
    ):
        path.write_text(json.dumps(original | changes))
        with pytest.raises(ValueError, match=message):
            campaign.prepare_campaign(config, tmp_path / "not-created")
        assert not (tmp_path / "not-created").exists()


def _counted_sessions(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Replace container launch with a counter of simultaneously open sessions."""
    import threading

    from trialagentbench_verifier import executor

    counts = {"active": 0, "peak": 0}
    mutex = threading.Lock()

    def launch(self):
        with mutex:
            counts["active"] += 1
            counts["peak"] = max(counts["active"], counts["peak"])

    def terminate(self, **kwargs):
        with mutex:
            counts["active"] -= 1

    monkeypatch.setattr(executor.shutil, "which", lambda _: "docker")
    monkeypatch.setattr(executor.DockerCodeSession, "_resolve_image_id", lambda _: "sha256:" + "0" * 64)
    monkeypatch.setattr(executor.DockerCodeSession, "_launch_runtime", launch)
    monkeypatch.setattr(executor.DockerCodeSession, "_prime_runtime", lambda _: None)
    monkeypatch.setattr(executor.DockerCodeSession, "_terminate_runtime", terminate)
    return counts


def _open_session(path: Path) -> None:
    import time

    from trialagentbench_verifier import executor

    path.mkdir()
    with executor.DockerCodeSession(path):
        time.sleep(0.02)


def test_executor_capacity_bounds_actual_session_launches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from trialagentbench_verifier import executor

    counts = _counted_sessions(monkeypatch)
    with executor.executor_session_capacity(2), ThreadPoolExecutor(max_workers=4) as pool:
        with executor.executor_session_capacity(8):  # a native run inside a campaign keeps the campaign bound
            list(pool.map(_open_session, (tmp_path / str(index) for index in range(8))))
    assert counts == {"active": 0, "peak": 2}


def test_offline_grading_queues_for_capacity_beyond_the_execution_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import time

    from trialagentbench_verifier import executor
    from trialagentbench_verifier.execution_contracts import CodeExecutionLimitsV1

    counts = _counted_sessions(monkeypatch)

    def hold(path: Path) -> None:
        path.mkdir()
        # A one-second timeout would fail the queued sessions; offline grading waits instead.
        with executor.DockerCodeSession(path, limits=CodeExecutionLimitsV1(timeout_seconds=1.0)):
            time.sleep(1.5)

    with executor.executor_session_capacity(1, queue=True), ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(hold, (tmp_path / str(index) for index in range(3))))
    assert counts == {"active": 0, "peak": 1}


def test_offline_exclusive_session_holds_every_executor_slot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    from trialagentbench_verifier import executor

    counts = _counted_sessions(monkeypatch)
    alone: list[int] = []

    def hold(index: int) -> None:
        path = tmp_path / str(index)
        path.mkdir()
        with executor.DockerCodeSession(path, exclusive=index == 2):
            if index == 2:
                alone.append(counts["active"])
            time.sleep(0.2)

    with executor.executor_session_capacity(3, queue=True), ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(hold, range(5)))
    assert alone == [1] and counts["active"] == 0
    # Live runs have no exclusive mode: their capacity gate keeps its deadline.
    with pytest.raises(ValueError, match="offline grading"), executor.executor_session_capacity(3):
        executor.DockerCodeSession(tmp_path, exclusive=True)


def test_standalone_runs_cap_executor_sessions_per_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from trialagentbench.util.parallel import run_bounded_tasks
    from trialagentbench.util.reported_cost import RunStopSignal

    counts = _counted_sessions(monkeypatch)

    def task(index: int) -> None:
        # One task oversubscribes the executor: four simultaneous sessions against a two-slot share.
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(_open_session, (tmp_path / f"{index}-{part}" for part in range(4))))

    run_bounded_tasks(range(3), task, workers=1, consume=lambda _: None, stop=RunStopSignal())
    assert counts == {"active": 0, "peak": 2}


def test_executor_unavailability_leaves_only_that_task_pending() -> None:
    from trialagentbench_verifier.executor import ExecutorUnavailableError

    from trialagentbench.util.parallel import run_bounded_tasks
    from trialagentbench.util.reported_cost import RunStopSignal

    def task(index: int) -> int:
        if index == 2:
            raise ExecutorUnavailableError("Executor capacity unavailable within the declared execution timeout")
        return index

    consumed: list[int] = []
    stop = RunStopSignal()
    with pytest.raises(CampaignCapacityStop, match="1 task") as stopped:
        run_bounded_tasks(range(6), task, workers=2, consume=consumed.append, stop=stop)
    assert sorted(consumed) == [0, 1, 3, 4, 5]
    assert stopped.value.campaign_wide is False
    stop.before_request()  # the other routes and later requests are not stopped


@pytest.mark.parametrize("failure", ["timeout", "http_error", "malformed", "missing_cost"])
def test_actual_transport_reserves_before_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure
) -> None:
    import httpx2

    from trialagentbench.ports.llm_provider import ProviderAccountingError

    control = CampaignRequestControl(
        database=tmp_path / "requests.sqlite", configuration=configuration(tmp_path), model_name="first"
    )

    def interrupted(self, request, **kwargs):
        with sqlite3.connect(control.database) as connection:
            assert connection.execute("SELECT active_until>started FROM attempts").fetchone() == (1,)
        if failure == "timeout":
            raise httpx2.ReadTimeout("Interrupted after possible provider receipt", request=request)
        if failure == "malformed":
            return httpx2.Response(200, request=request, content=b"not-json")
        if failure == "http_error":
            return httpx2.Response(503, request=request, json={"error": "unavailable"})
        return httpx2.Response(200, request=request, json={"choices": [{}], "usage": {}})

    monkeypatch.setattr(httpx2.Client, "send", interrupted)
    with controlled_http_client(control) as client:
        if failure in {"timeout", "malformed"}:
            with pytest.raises(httpx2.ReadTimeout if failure == "timeout" else ProviderAccountingError):
                client.post("https://openrouter.ai/api/v1/chat/completions", content=request())
        else:
            client.post("https://openrouter.ai/api/v1/chat/completions", content=request())
    with sqlite3.connect(control.database) as connection:
        assert connection.execute("SELECT charge,settled,active_until FROM attempts").fetchall() == [(1000, 0, 0)]


def test_transport_capacity_wait_uses_the_original_request_deadline(tmp_path, monkeypatch):
    import time

    import httpx2

    config = configuration(tmp_path)
    config = config.model_copy(
        update={"models": (config.models[0].model_copy(update={"spend_limit_micro_usd": 1200}), config.models[1])}
    )
    control = CampaignRequestControl(database=tmp_path / "requests.sqlite", configuration=config, model_name="first")
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(time, "time", lambda: 1000 + clock.now)
    active = control.reserve(request(), timeout_seconds=2, active_transport=True)

    def settle_while_waiting(seconds):
        clock.now += seconds
        control.settle(active, b'{"choices":[{}],"usage":{"cost":0.0001}}')

    def transport(self, outgoing, **kwargs):
        assert outgoing.extensions["timeout"] == dict(connect=0.75, read=0.75, write=0.75, pool=0.75)
        with sqlite3.connect(control.database) as connection:
            assert connection.execute("SELECT active_until FROM attempts WHERE id!=?", (active,)).fetchone() == (1,)
        return httpx2.Response(200, request=outgoing, json={"choices": [{}], "usage": {"cost": 0.00005}})

    monkeypatch.setattr(time, "sleep", settle_while_waiting)
    monkeypatch.setattr(httpx2.Client, "send", transport)
    with controlled_http_client(control) as client:
        assert client.post("https://openrouter.ai/api/v1/chat/completions", content=request(), timeout=1).is_success
    with sqlite3.connect(control.database) as connection:
        assert connection.execute("SELECT SUM(charge),SUM(active_until) FROM attempts").fetchone() == (150, 0)


def test_real_sdk_adapter_pins_route_and_reserves_each_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx2

    from trialagentbench.adapters import ProviderRouting, close_provider, get_provider

    control = CampaignRequestControl(
        database=tmp_path / "requests.sqlite", configuration=configuration(tmp_path), model_name="first"
    )
    requests = []
    sleeps = []

    def transport(self, outgoing, **kwargs):
        payload = json.loads(outgoing.content)
        requests.append(payload)
        assert payload["provider"] == {"only": ["route-a"], "allow_fallbacks": False}
        if len(requests) == 1:
            return httpx2.Response(
                429, request=outgoing, headers={"retry-after": "7"}, json={"error": {"message": "controlled throttle"}}
            )
        return httpx2.Response(
            200,
            request=outgoing,
            json={
                "id": "control",
                "model": "fixture/first",
                "provider": "route-a",
                "object": "chat.completion",
                "created": 1,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11, "cost": 0.0001},
            },
        )

    monkeypatch.setenv("OPENROUTER_API_KEY", "nonsecret-test-fixture")
    monkeypatch.setattr(httpx2.Client, "send", transport)
    monkeypatch.setattr("trialagentbench.adapters.llm_providers.time.sleep", sleeps.append)
    provider = get_provider(
        "fixture/first",
        routing=ProviderRouting("openrouter", "route-a"),
        send_temperature=False,
        timeout_s=30,
        request_control=control,
    )
    try:
        response = provider.generate_turn([{"role": "user", "content": "control"}], temperature=0, max_tokens=100)
    finally:
        close_provider(provider)
    assert response.content == "done"
    assert len(requests) == 2 and sleeps == [7.0]
    with sqlite3.connect(control.database) as connection:
        assert connection.execute("SELECT charge,settled FROM attempts ORDER BY id").fetchall() == [
            (1000, 0),
            (100, 1),
        ]


@pytest.mark.parametrize("failure", ["usage", "json", "disk", "ledger"])
def test_sdk_does_not_retry_local_accounting_failures(tmp_path, monkeypatch, failure):
    """A rejected primary usage or local IO failure must never buy another completion."""
    import errno

    import httpx2

    from trialagentbench.adapters.llm_providers import ChatCompletionsProvider, ProviderRouting
    from trialagentbench.contracts.core.config import ProviderRetryPolicyV1

    control = CampaignRequestControl(
        database=tmp_path / "requests.sqlite", configuration=configuration(tmp_path), model_name="first"
    )
    transported = []

    def transport(self, outgoing, **kwargs):
        transported.append(outgoing)
        if failure == "json":
            return httpx2.Response(200, request=outgoing, content=b"partial-json{")
        return httpx2.Response(
            200,
            request=outgoing,
            json={
                "choices": [{}],
                "usage": {"cost": 0.0001, "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 3},
            },
        )

    def broken_settle(*args):
        if failure == "disk":
            raise OSError(errno.ENOSPC, "No space left on device")
        raise sqlite3.OperationalError("database is locked")

    if failure in {"disk", "ledger"}:
        monkeypatch.setattr(control, "settle", broken_settle)
    monkeypatch.setattr(httpx2.Client, "send", transport)
    provider = ChatCompletionsProvider(
        model="fixture/first",
        base_url="https://openrouter.ai/api/v1",
        api_key="offline-fixture",
        routing=ProviderRouting("openrouter", "route-a"),
        send_temperature=False,
        timeout_s=1,
        retry_policy=ProviderRetryPolicyV1(maximum_attempts=3, base_delay_seconds=0),
        request_control=control,
    )
    try:
        with pytest.raises((ValueError, OSError, sqlite3.OperationalError)):
            provider.generate_turn([{"role": "user", "content": "offline"}], temperature=0, max_tokens=100)
    finally:
        provider.close()
    assert len(transported) == 1
    with sqlite3.connect(control.database) as connection:
        assert connection.execute("SELECT charge,settled,active_until FROM attempts").fetchall() == [(1000, 0, 0)]


@pytest.mark.parametrize("bounded_dispatch", [False, True])
@pytest.mark.parametrize("long_retry_after", [False, True])
def test_persistent_provider_rate_limit_pauses_only_its_route(
    tmp_path, monkeypatch, bounded_dispatch, long_retry_after
):
    """Exhausted HTTP 429 retries leave work pending without draining another model's route."""
    import httpx2

    from trialagentbench.adapters.llm_providers import ChatCompletionsProvider, ProviderRouting
    from trialagentbench.contracts.core.config import ProviderRetryPolicyV1
    from trialagentbench.util.campaign_control import exhausted_budgets
    from trialagentbench.util.parallel import run_bounded_tasks
    from trialagentbench.util.reported_cost import RunStopSignal

    config = configuration(tmp_path)
    control = CampaignRequestControl(database=tmp_path / "requests.sqlite", configuration=config, model_name="first")
    transported = []

    def transport(self, outgoing, **kwargs):
        transported.append(outgoing)
        return httpx2.Response(
            429,
            request=outgoing,
            headers={"retry-after": "3600" if long_retry_after else "0"},
            json={"error": {"message": "offline rate limit"}},
        )

    monkeypatch.setattr(httpx2.Client, "send", transport)
    provider = ChatCompletionsProvider(
        model="fixture/first",
        base_url="https://openrouter.ai/api/v1",
        api_key="offline-fixture",
        routing=ProviderRouting("openrouter", "route-a"),
        send_temperature=False,
        timeout_s=1,
        retry_policy=ProviderRetryPolicyV1(maximum_attempts=3, base_delay_seconds=0),
        request_control=control,
    )
    try:

        def execute(_):
            return provider.generate_turn(
                [{"role": "user", "content": "offline"}], temperature=0, max_tokens=100, timeout_seconds=5
            )

        with pytest.raises(CampaignCapacityStop) as raised:
            if bounded_dispatch:
                run_bounded_tasks(
                    [None], execute, workers=1, consume=lambda _: None, stop=RunStopSignal(), request_control=control
                )
            else:
                execute(None)
        assert not raised.value.campaign_wide
        assert raised.value.retry_telemetry.request_attempts == (1 if long_retry_after else 3)
    finally:
        provider.close()
    assert len(transported) == (1 if long_retry_after else 3)
    other = CampaignRequestControl(database=control.database, configuration=config, model_name="second")
    assert other.reserve(request("second", "route-b"), timeout_seconds=1) is not None
    assert not exhausted_budgets(control.database, config)


@pytest.mark.parametrize("stratified_dispatch", [False, True])
@pytest.mark.parametrize("pause_first_model", [False, True])
def test_native_campaign_preparation_and_recovery_preserve_primary_custody(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pause_first_model: bool,
    capsys: pytest.CaptureFixture[str],
    stratified_dispatch: bool,
) -> None:
    config = configuration(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "nonsecret-test-fixture")
    monkeypatch.setattr(campaign, "_host_resources", lambda: (65536, 16))
    manifest = SimpleNamespace(
        artifacts=(SimpleNamespace(filename="trialdev-evaluator.zip", sha256="d" * 64),),
        executor=SimpleNamespace(environment_sha256="c" * 64),
    )
    monkeypatch.setattr(campaign, "verify_release_set", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(campaign, "authenticate_release_set_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(campaign, "preview_panel", lambda _: None)
    tasks = tuple(
        SimpleNamespace(
            task_id=f"td-{index:012d}",
            case_id=f"case-{index}",
            stream=TrialDevStream.EVIDENCE_ANALYSIS,
            data_view="raw",
        )
        for index in (1, 2)
    )
    release = SimpleNamespace(
        manifest=SimpleNamespace(release_id="release_" + "d" * 64),
        participant=SimpleNamespace(tasks=tasks, release_manifest_sha256="a" * 64),
    )
    release.analysis_configuration = lambda task: None
    release.authored_configuration = lambda task: None
    monkeypatch.setattr(canonical_trialdev, "load_trialdev_evaluator_release", lambda _: release)
    monkeypatch.setattr(
        canonical_trialdev,
        "resolve_trusted_executor_environment",
        lambda **_: SimpleNamespace(
            image_id=config.executor_image_id, limits=TRIALDEV_RELEASE_BUDGET_V1.code_execution
        ),
    )

    @contextmanager
    def materialized(*args, **kwargs):
        yield SimpleNamespace(root=tmp_path, archive_sha256="d" * 64)

    monkeypatch.setattr(canonical_trialdev, "materialized_release_archive", materialized)
    provider_calls = []

    def provider(*args, **kwargs):
        provider_calls.append(args)
        assert kwargs["request_control"] is not None
        return SimpleNamespace()

    monkeypatch.setattr(canonical_trialdev, "get_provider", provider)
    retained = []
    monkeypatch.setattr(campaign, "_retain_campaign_executor", lambda *args: retained.append(args))
    root = tmp_path / "campaign"
    if stratified_dispatch:
        payload = config.model_dump(mode="json")
        payload["panel"]["trialdev_tasks"] = [task.task_id for task in tasks]
        payload["dispatch"] = {
            "task_order": [{"benchmark": "trialdev", "task_id": task.task_id} for task in reversed(tasks)]
        }
        config = CampaignConfigurationV1.model_validate(payload)
    plan = campaign.prepare_campaign(config, root)
    assert retained == [(config, root, plan)]
    assert len(plan.runs) == 2 and not provider_calls
    # Status must remain readable while a transport event is being written.
    # Completed-run telemetry validation belongs to campaign finish.
    (root / plan.runs[0].relative_run_root / "provider_responses.jsonl").write_text('{"request_id":')
    assert campaign.main(["status", "--campaign", str(root)]) == 0
    assert "terminal checkpoints=0/2; pending checkpoints=2" in capsys.readouterr().out
    before = (root / "campaign.json").read_bytes()
    completed = {}
    pause_pending = pause_first_model

    def run_task(**kwargs):
        nonlocal pause_pending
        if pause_pending and "first-trialdev-1" in str(kwargs["output_root"]):
            pause_pending = False
            raise CampaignCapacityStop("controlled route pause", campaign_wide=False)
        task_id = kwargs["task_id"]
        result = SimpleNamespace(
            task_id=task_id, stream=TrialDevStream.EVIDENCE_ANALYSIS, execution_status="completed", checksum="e" * 64
        )
        completed[kwargs["output_root"], task_id] = result
        return result

    monkeypatch.setattr(canonical_trialdev, "run_trialdev_task", run_task)
    monkeypatch.setattr(
        canonical_trialdev,
        "recover_terminal_trialdev_task",
        lambda **kwargs: completed.get((kwargs["output_root"], kwargs["task_id"])),
    )
    monkeypatch.setattr(canonical_trialdev, "summarize_provider_telemetry_v1", lambda **kwargs: None)
    with monkeypatch.context() as small_host:
        small_host.setattr(campaign, "_host_resources", lambda: (8192, 64))
        with pytest.raises(ValueError, match="exceeds this host's 8192 MB"):
            campaign.run_campaign(root)
    assert not provider_calls
    campaign.run_campaign(root)
    assert len(provider_calls) == (3 if pause_first_model else 4)
    retained_results = dict(completed)
    with sqlite3.connect(root / "requests.sqlite") as connection:
        assert connection.execute("SELECT cancelled FROM control").fetchone() == (0,)
    campaign.run_campaign(root)
    assert len(provider_calls) == (5 if pause_first_model else 4)
    assert all(completed[key] is value for key, value in retained_results.items())
    assert campaign.main(["status", "--campaign", str(root)]) == 0
    assert "terminal checkpoints=2/2; pending checkpoints=0" in capsys.readouterr().out
    assert (root / "campaign.json").read_bytes() == before
    with campaign.campaign_lock(root), pytest.raises(ValueError, match="Another process"):
        campaign.run_campaign(root)
    invocation = root / plan.runs[0].relative_run_root / "invocation.json"
    invocation.write_text(invocation.read_text() + " ")
    with pytest.raises(ValueError, match="invocation differs"):
        campaign.load_campaign(root)


def test_combined_trialeval_plan_fixes_repetition_one_before_outcomes(tmp_path: Path) -> None:
    """Link repetitions and the infrastructure retry without outcome selection or duplicate unit labels."""
    from trialagentbench_verifier.campaign import TrialEvalCampaignPlanV1
    from trialagentbench_verifier.io import sha256_file

    from trialagentbench.contracts.campaign import CampaignPlanV1, CampaignRunV1
    from trialagentbench.tools.finish_campaign_analysis import combined_trialeval_plan

    config = configuration(tmp_path)
    config = config.model_copy(
        update={"panel": config.panel.model_copy(update={"benchmarks": ("trialeval",), "repetitions": 2})}
    )
    runs = []
    for model, repetition, retry_of in (
        *((model, repetition, None) for model in config.models for repetition in (1, 2)),
        (config.models[0], 1, f"{config.models[0].name}-1"),
    ):
        job_id = f"{model.name}-{repetition}" if retry_of is None else f"{retry_of}-retry"
        path = tmp_path / "plans" / f"{job_id}.json"
        path.parent.mkdir(exist_ok=True)
        source = TrialEvalCampaignPlanV1.model_validate(
            {
                "release_id": "release",
                "release_manifest_sha256": "a" * 64,
                "participant_manifest_checksum": "b" * 64,
                "prospective_analysis_plan_sha256": "c" * 64,
                "cells": [{"series_ref": "series_" + "c" * 20, "cell_ref": "cell_" + "c" * 20}],
                "attempts": [
                    {
                        "attempt_id": job_id,
                        "scheduled_unit_id": "shared-old-task-label",
                        "attempt_role": "primary",
                        "model_id": model.model,
                        "run_identity_sha256": "d" * 64,
                        "task_id": "task",
                        "trial_id": "trial_" + "a" * 20,
                        "base_world_id": "world_" + "b" * 20,
                        "series_ref": "series_" + "c" * 20,
                        "cell_ref": "cell_" + "c" * 20,
                        "effect_state": "near_null",
                        "context": "C1",
                        "condition_id": "primary",
                        "request_replicate_id": str(repetition) if retry_of is None else "retry-1",
                        "procedure_assistance": "output_contract_only",
                        "analysis_specification": "locked_sap",
                        "design_family": "individual_randomized",
                        "reconstruction_family": None,
                        "target_ids": ["effect"],
                        "target_kind": "point",
                        "effect_measure": "risk_difference",
                        "result_unit": "probability",
                    }
                ],
            }
        )
        path.write_text(source.model_dump_json(indent=2))
        runs.append(
            CampaignRunV1(
                job_id=job_id,
                benchmark="trialeval",
                model_name=model.name,
                repetition=repetition,
                relative_run_root=f"runs/{job_id}",
                invocation_sha256="e" * 64,
                analysis_plan_path=path.relative_to(tmp_path).as_posix(),
                analysis_plan_sha256=sha256_file(path),
                retry_of_job_id=retry_of,
            )
        )
    plan = CampaignPlanV1(configuration=config, runs=tuple(runs))
    combined = combined_trialeval_plan(tmp_path, plan)
    assert len({row.scheduled_unit_id for row in combined.attempts}) == 4
    assert [row.attempt_role for row in combined.attempts] == ["primary", "stochastic_duplicate"] * 2 + ["retry"]
    retry = combined.attempts[-1]
    assert retry.retry_of_attempt_id == retry.scheduled_unit_id == "first-1"
    assert retry.request_replicate_id == "retry-1"
    assert combined.model_comparisons == (("fixture/first", "fixture/second"),)
    with pytest.raises(ValueError, match="require a reclassification audit"):
        CampaignPlanV1(configuration=config, runs=(*runs, runs[-1].model_copy(update={"job_id": "second-retry"})))
    with pytest.raises(ValueError, match="must repeat one TrialEval primary"):
        CampaignPlanV1(
            configuration=config, runs=(*runs[:-1], runs[-1].model_copy(update={"retry_of_job_id": "first-2"}))
        )

    # A later correction may discover another interrupted task. Its original
    # primary and the already executed retry remain frozen in the combined plan.
    from contracts.test_trialeval_score_source_contracts import _run_config, _write_missing_outcome
    from trialagentbench_verifier import campaign as verifier

    primary_path = tmp_path / str(runs[0].analysis_plan_path)
    primary_source = TrialEvalCampaignPlanV1.model_validate_json(primary_path.read_bytes())
    extra = primary_source.attempts[0].model_copy(update={"attempt_id": "late-primary", "task_id": "late-task"})
    primary_source = TrialEvalCampaignPlanV1.model_validate(
        {**primary_source.model_dump(exclude={"checksum"}), "attempts": (*primary_source.attempts, extra)}
    )
    primary_path.write_text(primary_source.model_dump_json())
    runs[0] = runs[0].model_copy(update={"analysis_plan_sha256": sha256_file(primary_path)})
    audit = tmp_path / "interruptions.json"
    audit.write_text('{"task": "late-task", "reason": "reclassified after retry-1"}')
    supplementary_path = tmp_path / "plans/supplementary.json"
    supplementary_source = TrialEvalCampaignPlanV1.model_validate(
        {
            **primary_source.model_dump(exclude={"checksum", "attempts"}),
            "attempts": [extra.model_copy(update={"attempt_id": "late-retry", "request_replicate_id": "retry-2"})],
        }
    )
    supplementary_path.write_text(supplementary_source.model_dump_json())
    supplementary = runs[-1].model_copy(
        update={
            "job_id": "supplementary",
            "relative_run_root": "runs/supplementary",
            "analysis_plan_path": "plans/supplementary.json",
            "analysis_plan_sha256": sha256_file(supplementary_path),
            "reclassification_audit_path": "interruptions.json",
            "reclassification_audit_sha256": sha256_file(audit),
        }
    )
    for index, run in enumerate([*runs, supplementary]):
        source = TrialEvalCampaignPlanV1.model_validate_json((tmp_path / str(run.analysis_plan_path)).read_bytes())
        native = tmp_path / run.relative_run_root / "run_config.json"
        native.parent.mkdir(parents=True, exist_ok=True)
        native.write_text(_run_config(task_ids=[row.task_id for row in source.attempts]).model_dump_json())
        updated = run.model_copy(update={"invocation_sha256": sha256_file(native)})
        if index < len(runs):
            runs[index] = updated
        else:
            supplementary = updated
    before_plan = combined_trialeval_plan(tmp_path, CampaignPlanV1(configuration=config, runs=tuple(runs)))
    expanded = CampaignPlanV1(configuration=config, runs=(*runs, supplementary))
    (tmp_path / "campaign.json").write_text(expanded.model_dump_json())
    (tmp_path / "configuration.json").write_text(config.model_dump_json())
    assert campaign.load_campaign(tmp_path) == expanded
    after_plan = combined_trialeval_plan(tmp_path, expanded)

    # Native retry scheduling excludes every earlier retry's tasks, labels the
    # supplementary run, and binds its audit before contacting a provider.
    from contextlib import nullcontext
    from unittest.mock import patch

    initial = CampaignPlanV1(configuration=config, runs=tuple(runs))
    (tmp_path / "campaign.json").write_text(initial.model_dump_json())
    prepared = []

    def prepare(*args, **kwargs):
        prepared.append(kwargs)
        assert kwargs["retry_task_ids"] == ("late-task",)
        assert kwargs["retry_replicate_id"] == "retry-2"
        assert kwargs["reclassification_audit_path"] == audit.name
        return supplementary

    from trialagentbench.contracts.core.runs import TrialEvalItemResultV1, TrialEvalRunConfigV1
    from trialagentbench.io import read_json_model, write_json_model
    from trialagentbench.tools import finish_campaign_analysis
    from trialagentbench.util.run_coverage import load_or_create_run_coverage_v1, reconcile_run_coverage_v1

    template_path = tmp_path / "checkpoint-template.json"
    _write_missing_outcome(template_path)
    template = read_json_model(TrialEvalItemResultV1, template_path)
    for run in runs:
        native = tmp_path / run.relative_run_root
        config_record = read_json_model(TrialEvalRunConfigV1, native / "run_config.json")
        coverage = load_or_create_run_coverage_v1(
            native / "coverage.json",
            run_identity_sha256=config_record.run_identity_sha256,
            unit_ids=tuple(config_record.task_ids),
            resume=False,
        )
        reconcile_run_coverage_v1(native / "coverage.json", coverage, completed_unit_ids=tuple(config_record.task_ids))
        for task_id in config_record.task_ids:
            checkpoint_path = native / "items" / f"{task_id}.json"
            write_json_model(
                checkpoint_path,
                template.model_copy(
                    update={
                        "item_id": task_id,
                        "run_config": config_record,
                        "scores": None,
                        "agent_output": template.agent_output.model_copy(
                            update={
                                "status": "failed",
                                "report": None,
                                "failure_type": "evaluator_error" if run.job_id == "first-1" else "provider_error",
                            }
                        ),
                    }
                ),
            )

    with (
        patch.object(campaign, "_require_host_capacity"),
        patch.object(campaign, "_campaign_inputs", return_value=object()),
        patch.object(campaign, "_campaign_asset_selections", return_value=()),
        patch.object(campaign, "release_assets", return_value=nullcontext()),
        patch.object(
            campaign, "_finish_native_runs", side_effect=AssertionError("Retry selection regraded the panel")
        ),
        patch.object(finish_campaign_analysis, "campaign_censoring", return_value=None),
        patch.object(campaign, "_prepare_native_run", side_effect=prepare),
        patch.object(campaign, "run_campaign"),
    ):
        with pytest.raises(ValueError, match="require a reclassification audit"):
            campaign.retry_campaign(tmp_path)
        campaign.retry_campaign(tmp_path, reclassification_audit=audit)
    assert len(prepared) == 1
    assert campaign.load_campaign(tmp_path) == expanded

    def selected(plan):
        # Only selection is under test; numerical grading has its own tests.
        bound = []
        for attempt in plan.attempts:
            item = verifier._BoundAttempt(attempt, None)
            item.row = SimpleNamespace(
                observed_attempt_outcome=(
                    "evaluator_error" if attempt.attempt_id in {"first-1", "late-primary"} else "completed"
                ),
                targets=(),
            )
            bound.append(item)
        return {
            row.plan.scheduled_unit_id: row.plan.attempt_id for row in verifier._scientific_primaries(tuple(bound))
        }

    before, after = selected(before_plan), selected(after_plan)
    assert before["late-primary"] == "late-primary" and after["late-primary"] == "late-retry"
    assert {key: value for key, value in before.items() if key != "late-primary"} == {
        key: value for key, value in after.items() if key != "late-primary"
    }
    # Authentication rejects overlapping native schedules before any execution.
    native = tmp_path / supplementary.relative_run_root / "run_config.json"
    native.write_text(_run_config(task_ids=["task"]).model_dump_json())
    overlapping = supplementary.model_copy(update={"invocation_sha256": sha256_file(native)})
    (tmp_path / "campaign.json").write_text(
        CampaignPlanV1(configuration=config, runs=(*runs, overlapping)).model_dump_json()
    )
    with pytest.raises(ValueError, match="already been retried"):
        campaign.load_campaign(tmp_path)
    overlapping_source = TrialEvalCampaignPlanV1.model_validate(
        {
            **supplementary_source.model_dump(exclude={"checksum", "attempts"}),
            "attempts": [supplementary_source.attempts[0].model_copy(update={"task_id": "task"})],
        }
    )
    supplementary_path.write_text(overlapping_source.model_dump_json())
    overlapping = supplementary.model_copy(update={"analysis_plan_sha256": sha256_file(supplementary_path)})
    with pytest.raises(ValueError, match="one conditional retry per scheduled unit"):
        combined_trialeval_plan(tmp_path, CampaignPlanV1(configuration=config, runs=(*runs, overlapping)))
    path.write_text("{}")
    with pytest.raises(ValueError):
        combined_trialeval_plan(tmp_path, plan)


def test_finished_diagnostic_campaign_retains_public_unavailable_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Withheld panel inference must still produce a portable completion record."""
    from trialagentbench_verifier.campaign import TrialEvalCampaignUnavailableV1, UnsupportedTrialEvalPanelError
    from trialagentbench_verifier.replay_scoring import TrialEvalAnswerRowV1

    from trialagentbench.tools import finish_campaign_analysis as finish

    config = configuration(tmp_path)
    config = config.model_copy(update={"panel": config.panel.model_copy(update={"benchmarks": ("trialeval",)})})
    run = SimpleNamespace(
        benchmark="trialeval", job_id="primary", relative_run_root="runs/primary", analysis_plan_path="plan.json"
    )
    plan = SimpleNamespace(configuration=config, runs=(run,))
    monkeypatch.setattr(finish, "campaign_censoring", lambda *_: None)
    monkeypatch.setattr(finish, "read_json_model", lambda *_: SimpleNamespace(attempts=()))
    combined = SimpleNamespace(
        release_id="release",
        release_manifest_sha256="a" * 64,
        checksum="b" * 64,
        model_dump_json=lambda **_: "{}",
    )
    monkeypatch.setattr(finish, "combined_trialeval_plan", lambda *_: combined)

    @contextmanager
    def staged(*args, **kwargs):
        yield SimpleNamespace(root=tmp_path)

    export_calls = []

    def exported(*, output_dir, **kwargs):
        export_calls.append(output_dir)
        output_dir.mkdir()
        (output_dir / "fixture.zip").write_bytes(b"isolated export-owner fixture")

    def unsupported(*args, **kwargs):
        assert kwargs["grade_roots"] == (tmp_path / "results" / plan.runs[0].job_id / "grade",)
        raise UnsupportedTrialEvalPanelError("diagnostic panel lacks crossed inference support")

    monkeypatch.setattr(finish, "materialized_release_archive", staged)
    monkeypatch.setattr(finish, "export_results_v1", exported)

    def answers(**kwargs):
        assert kwargs["grade_roots"] == (tmp_path / "results" / plan.runs[0].job_id / "grade",)
        return (
            TrialEvalAnswerRowV1(
                model="fixture/first",
                task_id="te_fixture",
                cell="cell",
                context="C5",
                law_id="law",
                scoring="replay",
                reproduced=True,
                replay_regret_musd=-1.5,
                value_answer_musd=1.5,
                value_prior_only_musd=0.0,
                value_truth_known_musd=2.0,
                replay_values=(),
                consequential_share=0.0,
                answered_share=1.0,
                material_decision_error_rate=0.0,
                value_multiplier=1.0,
                cost_multiplier=1.0,
            ),
        )

    monkeypatch.setattr(finish, "analyse_trial_eval_campaign_v1", unsupported)
    monkeypatch.setattr(finish, "trial_eval_answer_rows_v1", answers)
    manifest = SimpleNamespace(artifacts=(SimpleNamespace(filename="trialeval-evaluator.zip", sha256="a" * 64),))
    finish.finish_comparisons(tmp_path, plan, manifest)
    destination = tmp_path / "comparisons" / "trialeval"
    # Every observed answer is scored even without paired inference (a C5-only campaign).
    (row,) = csv.DictReader((destination / "trialeval_answers.csv").open())
    assert (row["model"], row["task_id"], row["replay_regret_musd"]) == ("fixture/first", "te_fixture", "-1.5")
    public_path = destination / "public-results.json"
    result = TrialEvalCampaignUnavailableV1.model_validate_json(public_path.read_text())
    assert result.result_export_sha256 is None
    assert result.campaign_plan_sha256 == combined.checksum
    assert public_path.read_bytes() == (destination / "analysis.json").read_bytes()
    # Offline finalisation is repeatable; an unavailable comparison is terminal.
    finish.finish_comparisons(tmp_path, plan, manifest)
    assert TrialEvalCampaignUnavailableV1.model_validate_json(public_path.read_text()) == result
    assert len(export_calls) == 1


def test_cli_capacity_pause_is_explicit_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A planned commercial pause is recoverable but cannot look like completion."""

    def paused(root: Path) -> None:
        raise CampaignCapacityStop("campaign reservation limit reached")

    monkeypatch.setattr(campaign, "run_campaign", paused)
    assert campaign.main(["run", "--campaign", str(tmp_path)]) == 3
    assert "Campaign paused" in capsys.readouterr().out


def test_development_campaign_authenticates_inputs_without_qualification(tmp_path: Path) -> None:
    """Development selection binds archive bytes without accepting a release set."""
    from trialagentbench_verifier.campaign_inputs import (
        DevelopmentArchiveV1,
        DevelopmentCampaignInputsV1,
        DevelopmentExecutorV1,
    )
    from trialagentbench_verifier.io import sha256_file

    config = configuration(tmp_path)
    root = tmp_path / "development"
    root.mkdir()
    archive = root / "trialdev-evaluator.zip"
    archive.write_bytes(b"development archive authentication fixture")
    manifest = DevelopmentCampaignInputsV1(
        artifacts=(
            DevelopmentArchiveV1(
                filename=archive.name, sha256=sha256_file(archive), byte_length=archive.stat().st_size
            ),
        ),
        executor=DevelopmentExecutorV1(image_id=config.executor_image_id, environment_sha256="c" * 64),
    )
    path = root / "development-inputs.json"
    path.write_text(manifest.model_dump_json())
    assert "input_mode" not in config.model_dump(mode="json")
    config = config.model_copy(
        update={
            "input_mode": "development",
            "release_directory": root,
            "expected_release_manifest_sha256": sha256_file(path),
        }
    )
    assert campaign._campaign_inputs(config) == manifest
    assert config.model_dump(mode="json")["input_mode"] == "development"
    # A local rebuild keeps the semantic identity; every native run verifies the environment hash.
    assert campaign._campaign_inputs(config.model_copy(update={"executor_image_id": "sha256:" + "f" * 64})) == manifest
    with pytest.raises(ValueError):
        campaign._campaign_inputs(config.model_copy(update={"input_mode": "qualified"}))
    archive.write_bytes(b"changed archive")
    with pytest.raises(ValueError, match="archive changed"):
        campaign._campaign_inputs(config)


def test_development_aggregate_cannot_parse_as_qualified_public_result(tmp_path: Path) -> None:
    from pydantic import ValidationError
    from trialagentbench_verifier.campaign import TrialEvalCampaignUnavailableV1

    from trialagentbench.tools.finish_campaign_analysis import _DevelopmentResults, _write_aggregate

    config = configuration(tmp_path).model_copy(update={"input_mode": "development"})
    # The aggregate writer only consumes the already-frozen configuration.
    plan = SimpleNamespace(configuration=config)
    result = TrialEvalCampaignUnavailableV1(
        scorer=current_scorer(),
        reason="diagnostic subset",
        result_export_sha256="b" * 64,
        campaign_plan_sha256="c" * 64,
    )
    _write_aggregate(tmp_path, result, plan)
    assert not (tmp_path / "public-results.json").exists()
    payload = json.loads((tmp_path / "development-results.json").read_text())
    assert _DevelopmentResults.model_validate(payload).status == "unqualified"
    assert payload["input_manifest_sha256"] == config.expected_release_manifest_sha256
    with pytest.raises(ValidationError):
        TrialEvalCampaignUnavailableV1.model_validate(payload)


def test_token_capacity_uses_request_bound_then_complete_observed_usage(tmp_path: Path) -> None:
    config = configuration(tmp_path)
    config = config.model_copy(
        update={
            "models": (
                config.models[0].model_copy(update={"tokens_per_minute": 1000}),
                config.models[1],
            )
        }
    )
    control = CampaignRequestControl(database=tmp_path / "requests.sqlite", configuration=config, model_name="first")
    content = request()
    attempt = control.reserve(content)
    with sqlite3.connect(control.database) as connection:
        assert connection.execute("SELECT tokens FROM attempts").fetchone() == (len(content) + 100,)
    assert control.reserve(content, timeout_seconds=0.01) > attempt
    control.settle(
        attempt, b'{"choices":[{}],"usage":{"cost":0.0001,"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}'
    )
    with sqlite3.connect(control.database) as connection:
        assert connection.execute("SELECT tokens,settled FROM attempts").fetchone() == (12, 1)
    incomplete = control.reserve(content)
    control.settle(incomplete, b'{"choices":[{}],"usage":{"cost":0.0001,"total_tokens":12}}')
    with sqlite3.connect(control.database) as connection:
        assert connection.execute("SELECT tokens FROM attempts WHERE id=?", (incomplete,)).fetchone() == (
            len(content) + 100,
        )
    invalid = control.reserve(content)
    with pytest.raises(ValueError, match="total tokens differ"):
        control.settle(
            invalid,
            b'{"choices":[{}],"usage":{"cost":0.0001,"prompt_tokens":10,"completion_tokens":2,"total_tokens":5}}',
        )
    with sqlite3.connect(control.database) as connection:
        assert connection.execute("SELECT tokens,settled FROM attempts WHERE id=?", (invalid,)).fetchone() == (
            len(content) + 100,
            0,
        )


@pytest.mark.parametrize("automatic_pricing", [False, True])
def test_configuration_resolves_download_sidecar_and_provider_records(tmp_path, monkeypatch, automatic_pricing):
    """Normal editable configuration need not copy digests, image IDs or API metadata."""
    from trialagentbench_verifier.campaign_inputs import (
        DevelopmentArchiveV1,
        DevelopmentCampaignInputsV1,
        DevelopmentExecutorV1,
    )
    from trialagentbench_verifier.io import sha256_file

    from trialagentbench.adapters import provider_capabilities

    config = configuration(tmp_path)
    release = tmp_path / "downloads"
    release.mkdir()
    archive = release / "trialdev-evaluator.zip"
    archive.write_bytes(b"fixture")
    manifest = DevelopmentCampaignInputsV1(
        artifacts=(DevelopmentArchiveV1(filename=archive.name, sha256=sha256_file(archive), byte_length=7),),
        executor=DevelopmentExecutorV1(image_id=config.executor_image_id, environment_sha256="c" * 64),
    )
    manifest_path = release / "development-inputs.json"
    manifest_path.write_text(manifest.model_dump_json())
    digest = sha256_file(manifest_path)
    (release / "development-inputs.json.sha256").write_text(f"{digest}  development-inputs.json\n")
    payload = config.model_dump(mode="json", exclude={"expected_release_manifest_sha256", "executor_image_id"})
    payload.update(input_mode="development", release_directory="downloads")
    for row in payload["models"]:
        row.pop("capability_snapshot")
        if automatic_pricing:
            for key in ("reservation_micro_usd", "reservation_tokens", "pricing_premise"):
                row.pop(key)
    if automatic_pricing:
        from trialagentbench_verifier.campaign_contracts import CampaignRequestPricingV1

        pricing = CampaignRequestPricingV1(
            prompt_usd_per_token="0.000001", completion_usd_per_token="0.000002", request_usd="0"
        )
        monkeypatch.setattr(
            provider_capabilities,
            "campaign_reservations",
            lambda *args, **kwargs: (100000, 1000000, "retained route prices", pricing),
        )
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(payload))
    calls = []

    def discover(*, model, upstream_provider, directory):
        calls.append((model, upstream_provider, directory))
        return tmp_path / f"{model.split('/')[-1]}-capability.json"

    monkeypatch.setattr(provider_capabilities, "snapshot_openrouter_capability", discover)
    resolved = campaign.load_configuration(path)
    campaign.validate_model_capabilities(resolved)
    assert resolved.expected_release_manifest_sha256 == digest
    assert resolved.executor_image_id == config.executor_image_id
    assert resolved.release_directory == release
    assert len(calls) == 2
    if automatic_pricing:
        assert resolved.models[0].reservation_micro_usd == 5000
        assert resolved.models[0].reservation_tokens == 100000
        assert resolved.models[0].request_pricing is not None
    # Changed manifests are rejected before metadata discovery or provider execution.
    calls.clear()
    manifest_path.write_text(manifest_path.read_text() + " ")
    with pytest.raises(ValueError, match="pinned digest"):
        campaign.load_configuration(path)
    assert calls == []


def test_default_capacity_follows_task_parallelism_and_is_reused_for_grading(tmp_path: Path) -> None:
    """Normal concurrency needs no manual executor-slot arithmetic."""
    from trialagentbench_verifier.campaign_contracts import CampaignOperationsV1

    operations = CampaignOperationsV1(model_workers=2, task_workers=3, spend_limit_micro_usd=1000000)
    assert operations.executor_capacity == 12
    assert (operations.grading_workers or operations.executor_capacity) == 12
    with pytest.raises(ValueError, match="Two executor slots"):
        CampaignOperationsV1(model_workers=2, task_workers=3, executor_workers=11, spend_limit_micro_usd=1000000)


@pytest.mark.parametrize(
    ("memory_mb", "cpus", "message"),
    [(4 * 4096, 8, None), (4 * 4096 - 1, 64, "4096 MB per container"), (1 << 20, 7, "2 CPUs per container")],
)
def test_host_capacity_refuses_containers_the_host_cannot_hold(memory_mb: int, cpus: int, message: str | None) -> None:
    """Capacity x the fixed per-container limits must fit the host's memory and CPUs."""
    if message is None:
        campaign.check_host_capacity(4, host_memory_mb=memory_mb, host_cpus=cpus)
        return
    with pytest.raises(ValueError, match=message) as refused:
        campaign.check_host_capacity(4, host_memory_mb=memory_mb, host_cpus=cpus)
    assert "operations.executor_workers" in str(refused.value)


def test_one_task_at_a_time_runs_on_any_host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The default single-task capacity (two executor slots) is never refused, even on a small host."""
    campaign.check_host_capacity(2, host_memory_mb=2048, host_cpus=1)

    def unreadable() -> tuple[int, int]:
        raise AssertionError("one task at a time must not read the host")

    monkeypatch.setattr(campaign, "_host_resources", unreadable)
    from trialagentbench_verifier.campaign_contracts import CampaignOperationsV1

    base = configuration(tmp_path)
    operations = CampaignOperationsV1(
        model_workers=1, task_workers=1, spend_limit_micro_usd=base.operations.spend_limit_micro_usd
    )
    single = base.model_copy(update={"operations": operations})
    assert single.operations.executor_capacity == 2
    campaign._require_host_capacity(single)
    with pytest.raises(ValueError, match="4096 MB per container"):
        campaign.check_host_capacity(4, host_memory_mb=2048, host_cpus=64)


def test_actual_request_pricing_reserves_small_request_and_detects_excess(tmp_path):
    from trialagentbench_verifier.campaign_contracts import CampaignRequestPricingV1

    config = configuration(tmp_path)
    pricing = CampaignRequestPricingV1(
        prompt_usd_per_token="0.000001", completion_usd_per_token="0.000002", request_usd="0"
    )
    model = config.models[0].model_copy(update={"request_pricing": pricing})
    config = config.model_copy(update={"models": (model, config.models[1])})
    control = CampaignRequestControl(database=tmp_path / "costs.sqlite", configuration=config, model_name=model.name)
    content = request()
    attempt = control.reserve(content)
    with sqlite3.connect(control.database) as connection:
        reserved = connection.execute("SELECT charge FROM attempts WHERE id=?", (attempt,)).fetchone()[0]
    assert reserved == len(content) + 200 < model.reservation_micro_usd
    with pytest.raises(CampaignCapacityStop, match="exceeded"):
        control.settle(
            attempt,
            json.dumps(
                {"choices": [{"message": {"content": "ok"}}], "usage": {"cost": (reserved + 1) / 1_000_000}}
            ).encode(),
        )


def test_finished_campaign_retains_csv_bytes_and_rejects_normalized_replay(tmp_path: Path) -> None:
    """The exported table must retain the exact bytes bound by its manifest."""
    import hashlib

    from trialagentbench.tools.finish_run import _write_once

    original = b"task_id,result\r\ntrial,complete\r\n"
    expected = hashlib.sha256(original).hexdigest()
    output = tmp_path / "actions.csv"
    _write_once(output, original)
    assert hashlib.sha256(output.read_bytes()).hexdigest() == expected
    _write_once(output, original)
    with pytest.raises(ValueError, match="differs from verified inputs"):
        _write_once(output, original.decode().replace("\r\n", "\n"))


def test_finalization_write_failure_is_recoverable_without_partial_result(tmp_path: Path, monkeypatch) -> None:
    """Failed durable storage cannot publish a partial canonical report."""
    from trialagentbench.tools import finish_run

    output = tmp_path / "score.json"

    def fail_sync(descriptor: int) -> None:
        raise OSError("injected storage failure")

    with monkeypatch.context() as failure:
        failure.setattr(finish_run.os, "fsync", fail_sync)
        with pytest.raises(OSError, match="injected storage failure"):
            finish_run._write_once(output, b'{"status":"pass"}\n')
    assert list(tmp_path.iterdir()) == []
    finish_run._write_once(output, b'{"status":"pass"}\n')
    assert output.read_bytes() == b'{"status":"pass"}\n'


def test_finalization_concurrent_first_result_cannot_be_overwritten(tmp_path: Path, monkeypatch) -> None:
    """A writer arriving between inspection and publication keeps its exact bytes."""
    from trialagentbench.tools import finish_run

    output = tmp_path / "score.json"

    def publish_first(source: Path, target: Path) -> None:
        target.write_bytes(b"first result\n")
        raise FileExistsError(target)

    monkeypatch.setattr(finish_run.os, "link", publish_first)
    with pytest.raises(ValueError, match="differs from verified inputs"):
        finish_run._write_once(output, b"second result\n")
    assert output.read_bytes() == b"first result\n"
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("published", (False, True))
def test_finalization_recovers_only_owned_staging_after_process_death(tmp_path: Path, published: bool) -> None:
    """A killed publisher leaves no committed report and cannot conceal unrelated files."""
    from trialagentbench.tools.finish_run import _discard_incomplete_writes, _write_once

    output = tmp_path / "run-custody.sha256"
    script = r"""
import os, sys
from pathlib import Path
from trialagentbench.tools import finish_run
publish = finish_run.os.link
def die(source, target):
    if sys.argv[2] == 'True':
        publish(source, target)
    os._exit(73)
finish_run.os.link = die
finish_run._write_once(Path(sys.argv[1]), b'complete custody\n')
"""
    killed = subprocess.run([sys.executable, "-c", script, str(output), str(published)], check=False, timeout=30)
    assert killed.returncode == 73
    assert output.exists() == published
    assert len(list(tmp_path.iterdir())) == 1 + int(published)
    _discard_incomplete_writes(tmp_path)
    assert list(tmp_path.iterdir()) == ([output] if published else [])
    _write_once(output, b"complete custody\n")
    unrelated = tmp_path / ".notes.tmp"
    unrelated.write_text("must remain", encoding="utf-8")
    _discard_incomplete_writes(tmp_path)
    assert unrelated.read_text(encoding="utf-8") == "must remain"
    assert output.read_bytes() == b"complete custody\n"
    impostor = tmp_path / f".score.json.{'a' * 32}.tmp"
    impostor.symlink_to(output)
    with pytest.raises(ValueError, match="not a regular file"):
        _discard_incomplete_writes(tmp_path)
    assert output.read_bytes() == b"complete custody\n"


def test_campaign_index_preserves_diagnostic_status_and_portable_links(tmp_path: Path) -> None:
    """A diagnostic campaign has useful navigation without fabricated inference."""
    import re
    from urllib.parse import unquote

    from trialagentbench.contracts.campaign import CampaignPlanV1, CampaignRunV1
    from trialagentbench.contracts.core.runs import RunCoverageV1
    from trialagentbench.io import write_json_model
    from trialagentbench.tools.finish_campaign_analysis import write_campaign_index

    config = configuration(tmp_path)
    runs = tuple(
        CampaignRunV1(
            job_id=model.name,
            benchmark="trialdev",
            model_name=model.name,
            repetition=1,
            relative_run_root=f"runs/{model.name}",
            invocation_sha256="b" * 64,
        )
        for model in config.models
    )
    plan = CampaignPlanV1(configuration=config, runs=runs)
    for run in runs:
        native = tmp_path / run.relative_run_root
        native.mkdir(parents=True)
        write_json_model(
            native / "coverage.json",
            RunCoverageV1(
                run_identity_sha256="a" * 64,
                schedule_sha256="b" * 64,
                unit_ids=("a", "b"),
                completed_unit_ids=("a", "b"),
            ),
        )
        grade = tmp_path / "results" / run.job_id / "grade"
        grade.mkdir(parents=True)
        (grade / "ranking.json").write_text('{"status":"unavailable"}')
    comparison = tmp_path / "comparisons" / "trialdev"
    comparison.mkdir(parents=True)
    public = comparison / "public-results.json"
    public.write_text('{"status":"unavailable","reason":"diagnostic panel"}')
    review = comparison / "review"
    review.mkdir()
    (review / "comparisons.csv").write_text("standard_error,minimal_detectable_difference\n1,2.8016\n")
    original = public.read_bytes()
    report = write_campaign_index(tmp_path, plan)
    html = report.read_text()
    assert "terminal checkpoints 2/2" in html
    assert "do not establish reproduced analyses or scientific success" in html
    assert "Public summary" in html and "reference answers" in html
    assert "Paired comparisons — standard errors and detectable differences" in html
    assert "Detectable differences belong to the paired comparisons" in html
    assert 'href="comparisons/trialdev/review/comparisons.csv"' in html
    for href in re.findall(r'href="([^"]+)"', html):
        assert not Path(href).is_absolute()
        assert (tmp_path / unquote(href)).is_file()
    assert public.read_bytes() == original
    assert write_campaign_index(tmp_path, plan).read_text() == html


def test_trialdev_leaderboard_outputs_carry_standard_errors(tmp_path: Path) -> None:
    """The shareable leaderboard JSON and CSV carry each TrialDev headline's bootstrap standard error."""
    from trialagentbench_verifier.campaign_contracts import LeaderboardRowV1
    from trialagentbench_verifier.public_results import CampaignLeaderboardV1

    from trialagentbench.contracts.campaign import CampaignPlanV1, CampaignRunV1
    from trialagentbench.tools.finish_campaign_analysis import _write_leaderboard

    config = configuration(tmp_path)
    runs = tuple(
        CampaignRunV1(
            job_id=f"{model.name}-trialdev-1",
            benchmark="trialdev",
            model_name=model.name,
            repetition=1,
            relative_run_root=f"runs/{model.name}",
            invocation_sha256="b" * 64,
        )
        for model in config.models
    )
    row = LeaderboardRowV1(
        scorer=current_scorer(),
        benchmark="trialdev",
        track="portfolio_management/raw",
        model_id="fixture/first",
        metric="net_development_value_usd",
        direction="higher_is_better",
        estimate=2_000_000.0,
        interval_low=1_000_000.0,
        interval_high=3_000_000.0,
        standard_error=500_000.0,
        scheduled_tasks=4,
        scored_tasks=4,
        independent_worlds=4,
    )
    _write_leaderboard(tmp_path, CampaignPlanV1(configuration=config, runs=runs), (row,), None)
    leaderboard = CampaignLeaderboardV1.model_validate_json((tmp_path / "leaderboard.json").read_bytes())
    assert leaderboard.rows == (row,)
    with (tmp_path / "leaderboard.csv").open(encoding="utf-8") as handle:
        (written,) = csv.DictReader(handle)
    assert float(written["standard_error"]) == 500_000.0


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("development", [False, True])
def test_prepare_replaces_stale_attestation_and_accepts_an_equivalent_local_image(
    tmp_path, monkeypatch, local, development
):
    """Missing Docker IDs are re-attested; environment-equivalent local IDs need no rebuild."""
    from unittest.mock import Mock

    from trialagentbench_verifier.campaign_inputs import DevelopmentCampaignInputsV1
    from trialagentbench_verifier.execution_contracts import (
        RELEASE_EXECUTION_LIMITS_V1,
        ExecutorEnvironmentV1,
        ExecutorPackageV1,
    )
    from trialagentbench_verifier.release_set import ExecutorAttestationV1, ReleaseSetManifestV1

    from trialagentbench.io import read_json_model, write_json_model

    config = configuration(tmp_path)
    payload = config.model_dump(mode="json", exclude={"executor_image_id"})
    payload["input_mode"] = "development" if development else "qualified"
    source = tmp_path / "editable.json"
    source.write_text(json.dumps(payload))
    old_env = ExecutorEnvironmentV1(
        image_reference="old:tag",
        image_id="sha256:" + "1" * 64,
        python_version="3.12",
        packages=(ExecutorPackageV1(name="numpy", version="2.0"),),
        limits=RELEASE_EXECUTION_LIMITS_V1,
    )
    new_env = old_env.model_copy(update={"image_reference": "custom:local", "image_id": "sha256:" + "2" * 64})

    def attestation(environment):
        return ExecutorAttestationV1(
            toolkit_sha256="d" * 64,
            context_sha256="e" * 64,
            image_reference=environment.image_reference,
            image_id=environment.image_id,
            environment_sha256=environment.semantic_sha256(),
            environment=environment,
        )

    stale = attestation(old_env)
    stale_path = tmp_path / "executor-attestation.json"
    write_json_model(stale_path, stale)
    original = stale_path.read_bytes()
    manifest_class = DevelopmentCampaignInputsV1 if development else ReleaseSetManifestV1
    manifest = manifest_class.model_construct(
        executor=SimpleNamespace(
            image_reference="release:tag", environment_sha256=new_env.semantic_sha256(), context_sha256="e" * 64
        ),
        artifacts=(SimpleNamespace(kind="public_toolkit", filename="toolkit.zip", sha256="d" * 64),),
    )
    monkeypatch.setattr(campaign, "_configuration_payload", lambda _: (tmp_path, dict(payload)))
    monkeypatch.setattr(
        campaign,
        "read_json_model",
        lambda kind, path: manifest if kind is manifest_class else read_json_model(kind, path),
    )
    monkeypatch.setattr(campaign, "authenticate_release_set_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(campaign, "_local_executor_image_exists", lambda image: local and image == "custom:local")
    monkeypatch.setenv("TRIALAGENTBENCH_EXECUTOR_IMAGE", "custom:local")
    probe = Mock(return_value=new_env)
    build = Mock(return_value=attestation(new_env))
    monkeypatch.setattr(campaign, "resolve_executor_environment", probe)
    monkeypatch.setattr(campaign, "attest_executor_from_toolkit", build)
    with pytest.raises(ValueError, match="Stale executor attestation.*executor-attestation.json"):
        campaign.load_configuration(source)
    assert stale_path.read_bytes() == original
    if development:
        with pytest.raises(ValueError, match="Stale executor attestation.*executor-attestation.json"):
            campaign.load_configuration(source, prepare_executor=True)
        assert stale_path.read_bytes() == original
        return
    resolved = campaign.load_configuration(source, prepare_executor=True)
    assert resolved.executor_image_id == new_env.image_id
    assert read_json_model(ExecutorAttestationV1, stale_path) == attestation(new_env)
    backups = tuple(tmp_path.glob("executor-attestation.*.json"))
    assert len(backups) == 1 and backups[0].read_bytes() == original
    assert build.call_count == int(not local) and probe.call_count == int(local)


def test_campaign_retention_tags_the_pinned_image_and_prints_a_save_hint(tmp_path, monkeypatch, capsys):
    """Distinct campaign directories retain their images independently; failed tags fail preparation."""
    from unittest.mock import Mock

    config = configuration(tmp_path)
    run = Mock()
    monkeypatch.setattr(campaign.subprocess, "run", run)
    plan = SimpleNamespace(checksum="a" * 64)
    first = campaign._retain_campaign_executor(config, tmp_path / "one", plan)
    second = campaign._retain_campaign_executor(config, tmp_path / "two", plan)
    assert first != second
    assert run.call_args_list[0].args[0] == ["docker", "image", "tag", config.executor_image_id, first]
    assert run.call_args_list[0].kwargs == {"check": True, "timeout": campaign.DOCKER_CONTROL_TIMEOUT_SECONDS}
    message = capsys.readouterr().out
    assert "docker image save --output" in message and "executor-image.tar" in message and first in message
    run.side_effect = subprocess.CalledProcessError(1, "docker image tag")
    with pytest.raises(subprocess.CalledProcessError):
        campaign._retain_campaign_executor(config, tmp_path / "three", plan)


def test_docker_daemon_failure_is_not_treated_as_a_missing_image(monkeypatch):
    monkeypatch.setattr(
        campaign.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 1, "", "Cannot connect to Docker daemon"),
    )
    with pytest.raises(RuntimeError, match="Docker image inspection failed"):
        campaign._local_executor_image_exists("executor:tag")


@pytest.mark.parametrize("defect", ["cached", "reasoning", "primary"])
def test_invalid_telemetry_usage_cannot_bypass_campaign_spending(tmp_path, monkeypatch, defect):
    import httpx2

    from trialagentbench.adapters import ProviderRouting, close_provider, get_provider
    from trialagentbench.contracts.core.config import ProviderRetryPolicyV1
    from trialagentbench.util.provider_telemetry import generate_provider_turn_v1, read_provider_terminal_events_v1

    control = CampaignRequestControl(
        database=tmp_path / "requests.sqlite", configuration=configuration(tmp_path), model_name="first"
    )
    usage = {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11, "cost": 0.0001}
    if defect == "cached":
        usage["prompt_tokens_details"] = {"cached_tokens": 11}
    elif defect == "reasoning":
        usage["completion_tokens_details"] = {"reasoning_tokens": 2}
    else:
        usage["total_tokens"] = 12
    calls = []

    def transport(self, outgoing, **kwargs):
        calls.append(outgoing)
        return httpx2.Response(
            200,
            request=outgoing,
            json={
                "id": "invalid-usage",
                "model": "fixture/first",
                "provider": "route-a",
                "object": "chat.completion",
                "created": 1,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "kept"}, "finish_reason": "stop"}
                ],
                "usage": usage,
            },
        )

    monkeypatch.setenv("OPENROUTER_API_KEY", "nonsecret-test-fixture")
    monkeypatch.setattr(httpx2.Client, "send", transport)
    provider = get_provider(
        "fixture/first",
        routing=ProviderRouting("openrouter", "route-a"),
        request_control=control,
        timeout_s=30,
        send_temperature=False,
        retry_policy=ProviderRetryPolicyV1(maximum_attempts=1),
    )
    log_path = tmp_path / "provider_responses.jsonl"

    def generate():
        return generate_provider_turn_v1(
            provider=provider,
            messages=[{"role": "user", "content": "analyse"}],
            tools=[],
            temperature=0,
            max_tokens=100,
            timeout_seconds=30,
            tool_choice="auto",
            log_path=log_path,
            benchmark="trialeval",
            unit_id="task",
            phase_id="task",
            step_id="analysis",
            turn_index=1,
        )

    try:
        if defect == "primary":
            with pytest.raises(ValueError, match="total tokens"):
                generate()
        else:
            assert generate().response.content == "kept"
    finally:
        close_provider(provider)
    assert len(calls) == 1
    (event,) = read_provider_terminal_events_v1(log_path)
    with sqlite3.connect(control.database) as connection:
        charge, settled = connection.execute("SELECT charge,settled FROM attempts").fetchone()
    if defect == "primary":
        assert (charge, settled) == (1000, 0)  # Full reservation survives the rejected response.
        assert event.status == "failed"
        assert event.response_received is True
        assert "total tokens" in event.error_detail
        retained = json.loads((tmp_path / "provider_payloads.raw.jsonl").read_text())
        assert retained["response"]["choices"][0]["message"]["content"] == "kept"
        assert retained["response"]["usage"]["total_tokens"] == 12
    else:
        assert (charge, settled) == (100, 1)
        assert event.usage_status == "invalid" and event.reported_cost_usd == 0.0001


@pytest.mark.parametrize("wall_jump", [1000000, -1000000])
def test_campaign_clock_jumps_do_not_expire_live_leases_or_reset_rate_windows(tmp_path, monkeypatch, wall_jump):
    import time

    config = configuration(tmp_path)
    model = config.models[0].model_copy(update={"spend_limit_micro_usd": 1200, "requests_per_minute": 1})
    config = config.model_copy(update={"models": (model, config.models[1])})
    clock = SimpleNamespace(now=1000.0, wall=1000.0)
    monkeypatch.setattr(time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(time, "time", lambda: clock.wall)
    first = CampaignRequestControl(database=tmp_path / "requests.sqlite", configuration=config, model_name="first")
    second = CampaignRequestControl(database=first.database, configuration=config, model_name="first")
    active = first.reserve(request(), timeout_seconds=100, active_transport=True)
    clock.wall += wall_jump
    clock.now += 0.01
    assert second.try_reserve(model=model, now=clock.now) is None
    assert exhausted_budgets(first.database, config) == {}
    first.settle(active, b'{"choices":[{}],"usage":{"cost":0.0001}}')
    assert second.try_reserve(model=model, now=clock.now) is None  # Rate window remains full.
    clock.now += 60
    assert second.try_reserve(model=model, now=clock.now) is not None
    assert exhausted_budgets(first.database, config) == {}


def test_unknown_clock_domain_retains_funds_and_rate_window(tmp_path, monkeypatch):
    import time

    config = configuration(tmp_path)
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(time, "monotonic", lambda: clock.now)
    first = CampaignRequestControl(database=tmp_path / "requests.sqlite", configuration=config, model_name="first")
    attempt = first.reserve(request(), active_transport=True)
    with sqlite3.connect(first.database) as connection:
        # An empty clock domain has unknown monotonic timing.
        connection.execute("UPDATE attempts SET clock_domain='', started=99999999, active_until=99999999")
    model = config.models[0].model_copy(update={"requests_per_minute": 1})
    assert first.try_reserve(model=model, now=100) is None
    clock.now = 161
    assert first.try_reserve(model=model, now=clock.now) is not None
    with sqlite3.connect(first.database) as connection:
        assert connection.execute("SELECT charge,settled FROM attempts WHERE id=?", (attempt,)).fetchone() == (1000, 0)


def test_crash_telemetry_recovery_never_settles_an_interrupted_reservation(tmp_path):
    from trialagentbench.contracts.core.runs import RunCoverageV1
    from trialagentbench.util.provider_telemetry import start_provider_request_v1, summarize_provider_telemetry_v1

    config = configuration(tmp_path)
    control = CampaignRequestControl(database=tmp_path / "requests.sqlite", configuration=config, model_name="first")
    handle = start_provider_request_v1(
        path=tmp_path / "logs" / "provider_responses.jsonl",
        benchmark="trialeval",
        unit_id="task",
        phase_id="task",
        step_id="analysis",
        turn_index=1,
        requested_model="fixture/first",
        provider_route="openrouter:route-a",
    )
    attempt = control.reserve(request(), active_transport=True)
    with handle.path.open("ab") as stream:
        stream.write(b'{"request_id":')
    raw = handle.path.read_bytes()
    summary = summarize_provider_telemetry_v1(
        run_root=tmp_path,
        coverage=RunCoverageV1(
            run_identity_sha256="a" * 64,
            schedule_sha256="b" * 64,
            unit_ids=("task",),
            completed_unit_ids=(),
        ),
    )
    assert summary.interrupted_request_count == 1
    assert summary.reported_cost_usd == 0 and summary.total_tokens == 0
    assert handle.path.read_bytes() == raw
    with sqlite3.connect(control.database) as connection:
        assert connection.execute("SELECT charge,settled FROM attempts WHERE id=?", (attempt,)).fetchone() == (1000, 0)


def test_monotonic_clock_epoch_is_shared_across_process_controllers_without_a_kernel_boot_id(tmp_path, monkeypatch):
    import time

    from trialagentbench.util import campaign_control as owner

    monkeypatch.setattr(owner, "_CLOCK_DOMAIN", None)
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(time, "monotonic", lambda: clock.now)
    config = configuration(tmp_path)
    model = config.models[0].model_copy(update={"spend_limit_micro_usd": 1200})
    config = config.model_copy(update={"models": (model, config.models[1])})
    first = CampaignRequestControl(database=tmp_path / "requests.sqlite", configuration=config, model_name="first")
    second = CampaignRequestControl(database=first.database, configuration=config, model_name="first")
    active = first.reserve(request(), timeout_seconds=100, active_transport=True)
    clock.now += 0.01
    assert second.try_reserve(model=model, now=clock.now) is None
    assert first._clock_domain == second._clock_domain
    assert exhausted_budgets(first.database, config) == {}
    first.settle(active, b'{"choices":[{}],"usage":{"cost":0.0001}}')
    assert second.try_reserve(model=model, now=clock.now) is not None
    assert exhausted_budgets(first.database, config) == {}


def test_shared_clock_epoch_rotates_after_reboot_without_releasing_unknown_charges(tmp_path, monkeypatch):
    import time

    from trialagentbench.util import campaign_control as owner

    monkeypatch.setattr(owner, "_CLOCK_DOMAIN", None)
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(time, "monotonic", lambda: clock.now)
    config = configuration(tmp_path)
    model = config.models[0].model_copy(update={"requests_per_minute": 1})
    config = config.model_copy(update={"models": (model, config.models[1])})
    first = CampaignRequestControl(database=tmp_path / "requests.sqlite", configuration=config, model_name="first")
    paid = first.reserve(request(), timeout_seconds=10, active_transport=True)
    first.settle(paid, b'{"choices":[{}],"usage":{"cost":0}}')
    epoch = first._clock_domain
    clock.now = 1001
    assert first.try_reserve(model=model, now=clock.now) is None
    clock.now = 1061
    unresolved = first.reserve(request(), timeout_seconds=10, active_transport=True)
    clock.now = 5  # Kernel restart; monotonic values belong to a new epoch.
    recovered = CampaignRequestControl(database=first.database, configuration=config, model_name="first")
    assert recovered._clock_domain != epoch
    assert recovered.try_reserve(model=model, now=clock.now) is None
    clock.now = 66
    assert recovered.try_reserve(model=model, now=clock.now) is not None
    with sqlite3.connect(first.database) as connection:
        assert connection.execute(
            "SELECT charge,settled,clock_domain FROM attempts WHERE id=?", (unresolved,)
        ).fetchone() == (1000, 0, epoch)
