"""Current authored runs retain observable actions and negative value in exports."""

import csv
import json
import shutil
from pathlib import Path
from unittest.mock import Mock

import pytest
from trialagentbench_verifier.behaviour import reproduce_behaviour_figures
from trialagentbench_verifier.response import TrialDevRankingItemV1, TrialDevRunRankingV1
from trialagentbench_verifier.scorer import current_scorer

from trialagentbench.contracts.core.config import ExperimentConditionV1
from trialagentbench.contracts.trialdev.run import TrialDevRunInvocationV1, TrialDevTaskRunV1
from trialagentbench.io import read_json_model, sha256_file, write_json_model
from trialagentbench.ports import CodeExecutionLimitsV1, LLMResponse, ToolCall
from trialagentbench.tools.build.build_trace_analysis_bundle import build_trace_analysis_bundle
from trialagentbench.trialdev.actions import TrialDevActionSession
from trialagentbench.trialdev.resources import TrialDevBudgetLedger
from trialagentbench.trialdev.run import (
    TrialDevProgrammeEvaluation,
    TrialDevProgrammeRun,
    _numerical_artifact_paths,
    trialdev_task_artifact_v1,
)
from trialagentbench.trialdev.task_agent import TrialDevTaskAgent
from trialagentbench.trialdev.value import TrialDevFrozenProgrammeValue


def _authored_run(root: Path, monkeypatch: pytest.MonkeyPatch, *, nested: bool, replicate: str = "request-1") -> Path:
    task_id = "td-0123456789ab"
    task = root / "tasks" / task_id if nested else root / task_id
    budget = task / "private/budget.json"
    write_json_model(budget, TrialDevBudgetLedger(budget_usd=1000))
    monkeypatch.setattr("trialagentbench.trialdev.task_agent.DockerCodeSession", lambda **kwargs: Mock())

    class Provider:
        model = "control"
        telemetry_route = "test"
        index = 0

        def generate_turn(self, **kwargs):
            actions = (
                ("wait_until", {"programme_day": 90.5}),
                ("wait_until", {"programme_day": 50}),
                ("stop_development", {}),
            )
            name, arguments = actions[self.index]
            self.index += 1
            return LLMResponse(
                tool_calls=[ToolCall(id=f"call-{self.index}", name=name, arguments=json.dumps(arguments))]
            )

    agent = TrialDevTaskAgent(
        provider=Provider(),
        workdir=task / "workspace",
        system_prompt="Development control.",
        program_id=task_id,
        scenario_id="tdw-aabb",
        objective_id="portfolio_management",
        max_turns_per_step=3,
        conversation_log_path=task / "conversation.json",
        event_log_path=task / "events.jsonl",
        authored_programme=TrialDevActionSession(
            task / "pool",
            budget,
            task / "workspace/evidence",
            task / "private/commitments",
            7,
        ),
    )
    agent.begin_step(phase_id="development", step_id="programme")
    agent.append_user_message("Complete the programme.")
    agent.run_programme()
    agent.close()
    # A retained numerical result for testing projection, not a scientific score experiment.
    result = TrialDevProgrammeRun(
        execution_status="completed",
        error=None,
        evaluation=TrialDevProgrammeEvaluation(
            programme_complete=True,
            studies=(),
            value=TrialDevFrozenProgrammeValue(
                decision="stop",
                clinical_value_usd=0,
                safety_cost_usd=0,
                study_cost_usd=0,
                time_cost_usd=905,
                net_value_usd=-905,
            ),
        ),
    )
    write_json_model(task / "programme_run.json", result)
    run = TrialDevTaskRunV1(
        release_id="release_" + "a" * 64,
        release_manifest_sha256="b" * 64,
        task_id=task_id,
        stream="portfolio_management",
        execution_status="completed",
        report_relative_paths=(),
        ingress_relative_paths=(),
        grade_relative_paths=(),
        programme_relative_path="programme_run.json",
        artifacts=tuple(
            trialdev_task_artifact_v1(task, path) for path in _numerical_artifact_paths(task, programme=True)
        ),
    )
    write_json_model(task / "run.json", run)
    missing = "td-123456789abc"
    invocation = TrialDevRunInvocationV1(
        release_id=run.release_id,
        release_archive_sha256="c" * 64,
        release_manifest_sha256="b" * 64,
        scheduled_task_ids=(task_id, missing),
        scheduled_task_streams={name: "portfolio_management" for name in (task_id, missing)},
        model="control",
        provider="openai",
        send_temperature=True,
        experiment_condition=ExperimentConditionV1(
            condition_id="primary", request_replicate_id=replicate, maximum_turns_per_step=3
        ),
        maximum_completion_tokens=1000,
        maximum_context_characters=50000,
        watchdog_seconds=120,
        request_timeout_seconds=60,
        executor_image="sha256:" + "e" * 64,
        executor_limits=CodeExecutionLimitsV1(),
    )
    write_json_model(root / "invocation.json", invocation)
    ranking = TrialDevRunRankingV1(
        scorer=current_scorer(),
        release_id=run.release_id,
        release_manifest_sha256="b" * 64,
        invocation_checksum=invocation.checksum,
        model_id="control",
        scope="complete_release_task_inventory",
        profile_means=(),
        items=(
            TrialDevRankingItemV1(
                task_id=task_id,
                world_id="tdw-aabb",
                stream="portfolio_management",
                clinical_profile_id="control",
                scenario_profile_id="control",
                asset_count=3,
                execution_status="completed",
                run_checksum=run.checksum,
                observational_loss=2.4641292395641474,
                initial_target_statuses={"effect": "reported"},
                authored_programme=True,
                net_development_value_usd=-905,
            ),
            TrialDevRankingItemV1(
                task_id=missing,
                world_id="tdw-aacc",
                stream="portfolio_management",
                clinical_profile_id="control",
                scenario_profile_id="control",
                asset_count=3,
                execution_status="not_started",
                run_checksum=None,
                observational_loss=0,
                initial_target_statuses={"effect": "reported"},
                authored_programme=True,
                net_development_value_usd=None,
            ),
        ),
    )
    write_json_model(root / "ranking.json", ranking)
    return task


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("unfinished", [False, True])
def test_authored_trace_exports_actions_missing_tasks_and_negative_value(tmp_path, monkeypatch, nested, unfinished):
    root = tmp_path / "run"
    _authored_run(root, monkeypatch, nested=nested)
    if unfinished:
        pending = (root / "tasks" if nested else root) / "td-123456789abc"
        pending.mkdir()
        # Unsealed records must not be parsed into authenticated action claims.
        (pending / "events.jsonl").write_text("unsealed partial event")
    output = tmp_path / "trace"
    build_trace_analysis_bundle(out_dir=output, trialdev_root=root)
    with (output / "unit_features.csv").open() as handle:
        features = list(csv.DictReader(handle))
    assert len(features) == 2
    assert features[0]["development_value"] == "-905.0"
    assert float(features[0]["observational_loss"]) == 2.4641292395641474
    assert features[0]["endpoint_valid"] == "True"
    assert features[1]["execution_status"] == ("incomplete_run" if unfinished else "not_started")
    if unfinished:
        assert features[1]["observational_loss"] == ""
        assert features[1]["endpoint_valid"] == ""
        assert features[1]["endpoint_state"] == "not_scoreable_trace_only"
    if nested:
        from trialagentbench.analysis.calibration import _trialdev_units

        units = _trialdev_units(root)
        assert units[1].execution_status == features[1]["execution_status"]
        assert "scientific_status" not in units[1].model_dump()
    with (output / "trialdev_decisions.csv").open() as handle:
        decisions = list(csv.DictReader(handle))
    assert [row["submission_status"] for row in decisions] == ["committed", "rejected", "committed"]
    assert [row["action"] for row in decisions] == ["wait", "", "stop"]
    assert float(decisions[0]["day_after"]) == 90.5
    assert float(decisions[2]["day_before"]) == 90.5
    assert decisions[2]["terminal_reason"] == "stop"
    reproduce_behaviour_figures(
        output,
        expected_manifest_sha256=sha256_file(output / "trace_bundle_manifest.json"),
        output=tmp_path / "figures",
    )


def test_authored_trace_rejects_changed_conversation(tmp_path, monkeypatch):
    root = tmp_path / "run"
    task = _authored_run(root, monkeypatch, nested=False)
    (task / "conversation.json").write_text("[]")
    with pytest.raises(ValueError, match="artifact changed"):
        build_trace_analysis_bundle(out_dir=tmp_path / "trace", trialdev_root=root)


def test_authored_repetitions_keep_distinct_portable_trace_identities(tmp_path, monkeypatch):
    root = tmp_path / "results"
    for replicate in ("request-1", "request-2"):
        _authored_run(root / replicate / "grade", monkeypatch, nested=True, replicate=replicate)
    output = build_trace_analysis_bundle(out_dir=tmp_path / "trace", trialdev_root=root)
    with (output / "unit_features.csv").open() as handle:
        features = list(csv.DictReader(handle))
    identities = {
        f"trialdev:{read_json_model(TrialDevRunInvocationV1, path).checksum}" for path in root.rglob("invocation.json")
    }
    assert len(identities) == 2
    assert len(features) == 4
    assert {row["run_id"] for row in features} == identities
    with (output / "trialdev_decisions.csv").open() as handle:
        decisions = list(csv.DictReader(handle))
    assert len(decisions) == 6
    assert {row["run_id"] for row in decisions} == identities
    moved = tmp_path / "relocated"
    shutil.copytree(root, moved)
    relocated = build_trace_analysis_bundle(out_dir=tmp_path / "relocated-trace", trialdev_root=moved)
    for name in ("unit_features.csv", "trialdev_decisions.csv"):
        assert (output / name).read_bytes() == (relocated / name).read_bytes()
    shutil.copytree(root / "request-1", root / "duplicate")
    with pytest.raises(ValueError, match="unique benchmark units"):
        build_trace_analysis_bundle(out_dir=tmp_path / "duplicate-trace", trialdev_root=root)
