"""Custody tests for TrialEval interruption and exact append/resume."""

from __future__ import annotations

import errno
import hashlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic import JsonValue
from trialagentbench_test_helpers import (
    minimal_participant_output_contract,
    write_minimal_trialeval_release_dictionaries,
)

from trialagentbench.contracts.core.runs import (
    ExecutorEnvironmentV1,
    ExecutorPackageV1,
)
from trialagentbench.execution_policy import TRIALEVAL_RELEASE_BUDGET_V1
from trialagentbench.io.checksums import canonical_payload_sha256
from trialagentbench.tools.run import trialeval
from trialagentbench.trialeval.conditions import prompt_set_sha256_v1

EXECUTOR_ENVIRONMENT = ExecutorEnvironmentV1(
    image_reference="trialagentbench/executor:1.0.1",
    image_id=f"sha256:{'0' * 64}",
    python_version="3.12.0",
    packages=(ExecutorPackageV1(name="pandas", version="2.3.0"),),
    limits=TRIALEVAL_RELEASE_BUDGET_V1.code_execution,
)
EXECUTOR_ENVIRONMENT_SHA256 = canonical_payload_sha256(cast(JsonValue, EXECUTOR_ENVIRONMENT.model_dump(mode="json")))


@pytest.fixture(autouse=True)
def _isolated_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep run-custody tests independent of the host Docker inventory."""

    def resolve_trusted_executor_environment(
        *,
        image: str | None,
        expected_image_id: str,
        expected_environment_sha256: str,
        limits: object,
    ) -> ExecutorEnvironmentV1:
        assert image is None
        assert expected_image_id == EXECUTOR_ENVIRONMENT.image_id
        assert expected_environment_sha256 == EXECUTOR_ENVIRONMENT_SHA256
        assert limits == TRIALEVAL_RELEASE_BUDGET_V1.code_execution
        return EXECUTOR_ENVIRONMENT

    monkeypatch.setattr(trialeval, "resolve_trusted_executor_environment", resolve_trusted_executor_environment)
    monkeypatch.setattr(
        trialeval,
        "load_participant_release",
        lambda root: SimpleNamespace(root=root, manifest=SimpleNamespace(checksum="f" * 64)),
    )


def _participant_release(root: Path) -> Path:
    participant = root / "participant"
    task_ids = ("TASK1001", "TASK1002")
    (participant / "manifest.json").parent.mkdir(parents=True)
    (participant / "manifest.json").write_text(
        json.dumps(
            {
                "schema_id": "trial_analysis_public_bundle_manifest/v1",
                "applied_baseline_profile_id": None,
                "applied_baseline_profile_sha256": None,
                "task_ids": list(task_ids),
                "task_evidence_factors": {
                    task_id: {
                        "context_configuration": "C1",
                        "data_preparation": "analysis_ready",
                        "analysis_specification": "locked_sap",
                    }
                    for task_id in task_ids
                },
            }
        ),
        encoding="utf-8",
    )
    for task_id in task_ids:
        item = participant / "items" / task_id
        (item / "data").mkdir(parents=True)
        (item / "data" / "analysis.parquet").write_bytes(b"fixture-analysis-data")
        (item / "task.json").write_text(
            json.dumps(
                {
                    "schema_id": "trial_analysis_task_v1",
                    "task_id": task_id,
                    "design_subtype": "individual_randomized",
                    "primary_endpoint_id": "endpoint",
                    "primary_paramcd": "endpoint",
                    "primary_estimand_id": "estimand",
                    "primary_effect_scale": "risk_difference_tau",
                    "estimand_mode": "fixed_declared_estimand",
                    "primary_effect_scale_options": ["risk_difference_tau"],
                    "primary_result_unit": "probability_difference",
                    "primary_tau_dy": 365.0,
                    "primary_population_id": "itt",
                    "primary_intercurrent_event_strategy_ids": ["treatment_policy"],
                    "primary_control_arm_id": "control",
                    "primary_treated_arm_id": "treated",
                }
            ),
            encoding="utf-8",
        )
        (item / "submission_contract.json").write_text(
            json.dumps(minimal_participant_output_contract(task_id)),
            encoding="utf-8",
        )
    write_minimal_trialeval_release_dictionaries(participant)
    return participant


def _participant_archive(participant: Path) -> tuple[Path, str]:
    archive_path = participant.parent / "TrialEvalBench_participant.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(path for path in participant.rglob("*") if path.is_file()):
            member = zipfile.ZipInfo(source.relative_to(participant).as_posix(), (2020, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o100644 << 16
            archive.writestr(member, source.read_bytes())
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    return archive_path, digest


def _args(participant: Path, output: Path) -> list[str]:
    archive, digest = _participant_archive(participant)
    return [
        "--participant-release",
        str(archive),
        "--expected-sha256",
        digest,
        "--model",
        "fixture/model",
        "--expected-executor-image-id",
        EXECUTOR_ENVIRONMENT.image_id,
        "--expected-executor-environment-sha256",
        EXECUTOR_ENVIRONMENT_SHA256,
        "--provider",
        "openrouter",
        "--openrouter-provider",
        "GMICloud",
        "--output-dir",
        str(output),
        "--turns",
        "1",
        "--quiet",
        "--context",
        "all",
    ]


def test_preparation_freezes_identity_without_provider_and_allows_exact_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prespecify a real native schedule before credentials or model outcomes."""
    participant = _participant_release(tmp_path)

    def forbidden_provider(*args: object, **kwargs: object) -> None:
        raise AssertionError("preparation contacted a provider")

    monkeypatch.setattr(trialeval, "get_provider", forbidden_provider)
    arguments = _args(participant, tmp_path / "runs")
    assert trialeval.main([*arguments, "--prepare-only"]) == 0
    config_path = next((tmp_path / "runs").rglob("run_config.json"))
    before = config_path.read_bytes()
    coverage = json.loads((config_path.parent / "coverage.json").read_bytes())
    assert len(coverage["unit_ids"]) == 2
    assert coverage["completed_unit_ids"] == []
    assert trialeval.main([*arguments, "--prepare-only", "--resume-run-dir", str(config_path.parent)]) == 0
    assert config_path.read_bytes() == before
    with pytest.raises(AssertionError, match="contacted a provider"):
        trialeval.main([*arguments, "--resume-run-dir", str(config_path.parent)])


def test_trialeval_has_one_report_plus_program_interface() -> None:
    """The public runner has no interface-selection factor."""

    args = trialeval._parser().parse_args(
        [
            "--participant-release",
            "participant.zip",
            "--expected-sha256",
            "0" * 64,
            "--model",
            "fixture/model",
            "--expected-executor-image-id",
            EXECUTOR_ENVIRONMENT.image_id,
            "--expected-executor-environment-sha256",
            EXECUTOR_ENVIRONMENT_SHA256,
            "--provider",
            "openai",
        ]
    )

    assert not hasattr(args, "submission_interface")
    assert args.max_tokens == TRIALEVAL_RELEASE_BUDGET_V1.maximum_completion_tokens_per_turn


def test_trialeval_exposes_the_completion_budget_as_a_run_condition() -> None:
    """Reasoning-budget experiments can vary tokens without changing source."""

    args = trialeval._parser().parse_args(
        [
            "--participant-release",
            "participant.zip",
            "--expected-sha256",
            "0" * 64,
            "--model",
            "fixture/model",
            "--expected-executor-image-id",
            EXECUTOR_ENVIRONMENT.image_id,
            "--expected-executor-environment-sha256",
            EXECUTOR_ENVIRONMENT_SHA256,
            "--provider",
            "openai",
            "--max-tokens",
            "8192",
        ]
    )

    assert args.max_tokens == 8192


def test_trialeval_rejects_a_nonpositive_completion_budget(tmp_path: Path) -> None:
    """Invalid provider budgets fail before release or credential access."""

    participant = _participant_release(tmp_path)

    with pytest.raises(ValueError, match="--max-tokens must be at least 1"):
        trialeval.main(
            [
                *_args(participant, tmp_path / "runs"),
                "--max-tokens",
                "0",
            ]
        )


def _agent_payload(item, **kwargs):
    return {
        "status": "max_turns_reached",
        "failure_type": "turn_limit_no_submission",
        "report": None,
        "turns_used": 1,
        "conversation": [],
        "events": [],
        "condition_provenance": {
            "procedure_assistance": kwargs["procedure_assistance"],
            "analysis_specification": kwargs["analysis_specification"],
            "analysis_surface_sha256": "d" * 64,
            "max_turns": kwargs["max_turns"],
            "prompt_set_sha256": prompt_set_sha256_v1(),
            "rendered_system_prompt_sha256": "b" * 64,
            "tool_schema_sha256": "c" * 64,
        },
    }


def _capability(path: Path, *, model: str = "fixture/model") -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_id": "ariadne.provider_model_capability/v1",
                "provider_transport": "openrouter",
                "model_id": model,
                "upstream_provider": "GMICloud",
                "supported_efforts": ["low", "medium", "high"],
                "source_url": "https://openrouter.ai/models/fixture/model",
                "source_retrieved_utc": "2026-08-04T00:00:00Z",
                "source_payload_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_trialeval_rejects_reasoning_route_mismatch_before_output(
    tmp_path: Path,
) -> None:
    participant = _participant_release(tmp_path)
    capability = _capability(tmp_path / "capability.json", model="fixture/other")
    output = tmp_path / "runs"

    with pytest.raises(ValueError, match="model does not match"):
        trialeval.main(
            [
                *_args(participant, output),
                "--reasoning-effort",
                "high",
                "--provider-capability-snapshot",
                str(capability),
            ]
        )

    assert not output.exists()


@pytest.mark.parametrize("mode", ["default", "exclude", "replay"])
def test_trialeval_persists_and_forwards_reasoning_condition(
    tmp_path: Path,
    monkeypatch,
    mode,
) -> None:
    participant = _participant_release(tmp_path)
    capability = _capability(tmp_path / "capability.json")
    provider_arguments: list[dict[str, object]] = []

    def provider(*args, **kwargs):
        del args
        provider_arguments.append(kwargs)
        return object()

    monkeypatch.setattr(trialeval, "get_provider", provider)
    monkeypatch.setattr(
        trialeval,
        "run_agent",
        lambda item, provider, **kwargs: (_ for _ in ()).throw(RuntimeError("interrupted")),
    )
    args = [
        *_args(participant, tmp_path / "runs"),
        "--condition-id",
        "luna-high",
        "--request-replicate-id",
        "request-2",
        "--reasoning-effort",
        "high",
        "--provider-capability-snapshot",
        str(capability),
    ]
    if mode != "default":
        args.append(f"--{mode}-returned-reasoning")

    with pytest.raises(RuntimeError, match="interrupted"):
        trialeval.main(args)

    run_root = next((tmp_path / "runs" / "fixture_model").iterdir())
    run_config = json.loads((run_root / "run_config.json").read_text(encoding="utf-8"))
    condition = run_config["experiment_condition"]
    assert condition["condition_id"] == "luna-high"
    assert condition["request_replicate_id"] == "request-2"
    assert condition["reasoning"]["effort"] == "high"
    assert len(condition["provider_capability"]["checksum"]) == 64
    assert provider_arguments[0]["reasoning_effort"] == "high"
    assert provider_arguments[0]["exclude_reasoning"] is (mode == "exclude")
    assert provider_arguments[0]["replay_reasoning"] is (mode == "replay")
    assert condition["reasoning"]["exclude_from_response"] is (mode == "exclude")
    assert condition["reasoning"].get("replay_in_context", False) is (mode == "replay")


def test_trialeval_resumes_from_immutable_item_checkpoint(tmp_path: Path, monkeypatch) -> None:
    participant = _participant_release(tmp_path)
    monkeypatch.setattr(trialeval, "get_provider", lambda *args, **kwargs: object())
    calls = 0

    def interrupted(item, provider, **kwargs):
        nonlocal calls
        del provider
        calls += 1
        if calls == 2:
            conversation_path = kwargs["conversation_log_path"]
            conversation_path.parent.mkdir(parents=True, exist_ok=True)
            (conversation_path.parent / "TASK1002_provider_payloads.raw.jsonl").write_text('{"raw":true}\n')
            workspace = conversation_path.parent / "TASK1002_workspace"
            workspace.mkdir(parents=True)
            (workspace / "stale.txt").write_text("must not be reused\n", encoding="utf-8")
            raise RuntimeError("interrupted")
        return _agent_payload(item, **kwargs)

    monkeypatch.setattr(trialeval, "run_agent", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        trialeval.main(_args(participant, tmp_path / "runs"))
    run_root = next((tmp_path / "runs" / "fixture_model").iterdir())
    checkpoint = run_root / "items" / "TASK1001.json"
    checkpoint_bytes = checkpoint.read_bytes()

    resumed_calls = 0

    def resumed(item, provider, **kwargs):
        nonlocal resumed_calls
        del provider
        resumed_calls += 1
        workspace = kwargs["conversation_log_path"].parent / f"{item.item_id}_workspace"
        assert not workspace.exists()
        return _agent_payload(item, **kwargs)

    monkeypatch.setattr(trialeval, "run_agent", resumed)
    assert trialeval.main([*_args(participant, tmp_path / "unused"), "--resume-run-dir", str(run_root)]) == 0
    assert resumed_calls == 1
    assert checkpoint.read_bytes() == checkpoint_bytes
    coverage = json.loads((run_root / "coverage.json").read_text(encoding="utf-8"))
    assert coverage["unit_ids"] == ["TASK1001", "TASK1002"]
    assert coverage["completed_unit_ids"] == ["TASK1001", "TASK1002"]
    assert (
        run_root / "failed_attempts" / "TASK1002" / "attempt-1" / "TASK1002_workspace.archived" / "stale.txt"
    ).is_file()
    assert not (run_root / "logs" / "TASK1002_provider_payloads.raw.jsonl").exists()
    assert (
        run_root / "failed_attempts" / "TASK1002" / "attempt-1" / "TASK1002_provider_payloads.raw.archived.jsonl"
    ).read_text() == '{"raw":true}\n'


@pytest.mark.parametrize("failed_file", ["TASK1001.json", "TASK1001_conversation.json", "TASK1001_events.jsonl"])
def test_interrupted_result_publication_can_resume_without_a_partial_checkpoint(tmp_path, monkeypatch, failed_file):
    """Disk exhaustion in a final log or checkpoint cannot publish task completion."""
    from datetime import UTC, datetime

    from trialagentbench.contracts.trace.observable import BenchmarkRuntimeTraceEventV1

    participant = _participant_release(tmp_path)
    monkeypatch.setattr(trialeval, "get_provider", lambda *args, **kwargs: object())

    def completed(item, provider, **kwargs):
        terminal = BenchmarkRuntimeTraceEventV1(
            event_id=f"trialeval:{item.item_id}:000000",
            timestamp=datetime.now(UTC),
            source_artifact_path=str(kwargs["conversation_log_path"]),
            source_payload_sha256="a" * 64,
            benchmark="trialeval",
            task_id=item.item_id,
            event_index=0,
            phase_id="task",
            step_id="analysis",
            event_type="step_terminal",
            terminal_status="completed",
        )
        return _agent_payload(item, **kwargs) | {
            "conversation": [{"role": "assistant", "content": "finished"}],
            "events": [terminal.model_dump(mode="json")],
        }

    monkeypatch.setattr(trialeval, "run_agent", completed)
    arguments = _args(participant, tmp_path / "runs")
    assert trialeval.main([*arguments, "--prepare-only"]) == 0
    run_root = next((tmp_path / "runs" / "fixture_model").iterdir())
    original_open = Path.open

    class InterruptedWriter:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def write(self, text):
            self.handle.write(text[:16])
            self.handle.flush()
            raise OSError(errno.ENOSPC, "injected full disk")

    def partial_write(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        if args and args[0] in {"x", "w"} and (path.name == failed_file or path.name.startswith(f".{failed_file}.")):
            return InterruptedWriter(handle)
        return handle

    with monkeypatch.context() as interrupted:
        interrupted.setattr(Path, "open", partial_write)
        with pytest.raises(OSError, match="injected full disk"):
            trialeval.main([*arguments, "--resume-run-dir", str(run_root)])
    assert not (run_root / "items" / "TASK1001.json").exists()
    assert trialeval.main([*arguments, "--resume-run-dir", str(run_root)]) == 0
    checkpoint = run_root / "items" / "TASK1001.json"
    retained = checkpoint.read_bytes()
    assert json.loads(retained)["item_id"] == "TASK1001"
    assert trialeval.main([*arguments, "--resume-run-dir", str(run_root)]) == 0
    assert checkpoint.read_bytes() == retained


def test_trialeval_watchdog_checkpoints_failure_and_continues_the_schedule(tmp_path: Path, monkeypatch) -> None:
    """A replay watchdog is a terminal scientific attempt, not a campaign interruption."""
    from trialagentbench.ports import CodeExecutionResultV1, LLMResponse, ToolCall
    from trialagentbench.trialeval import agent
    from trialagentbench.util.runtime_context import RuntimeDeadlineExceeded

    class Session:
        image_id = "sha256:" + "e" * 64

        def execute(self, code):
            return ""

        def execute_result(self, code):
            return CodeExecutionResultV1(status="success", output="", elapsed_seconds=0.0)

        def close(self):
            pass

    class Provider:
        model = "fixture/model"
        telemetry_route = "test"

        def __init__(self):
            self.calls = 0

        def generate_turn(self, *args, **kwargs):
            self.calls += 1
            call = (
                ToolCall(id="check", name="check_analysis", arguments="{}")
                if self.calls == 1
                else ToolCall(id="submit", name="submit_response", arguments='{"content":"Completed report."}')
            )
            return LLMResponse(tool_calls=[call])

    checked = []

    def check(call, **kwargs):
        task_id = kwargs["context"].checkpoint_id
        checked.append(task_id)
        if task_id == "TASK1001":
            raise RuntimeDeadlineExceeded("TrialEval item exhausted while blocking on analysis replay check.")
        return CodeExecutionResultV1(status="success", output="reproduced", elapsed_seconds=0.0), kwargs["session"]

    monkeypatch.setattr(agent, "DockerCodeSession", lambda **_: Session())
    monkeypatch.setattr(agent, "check_analysis_tool_v1", check)
    monkeypatch.setattr(trialeval, "get_provider", lambda *args, **kwargs: Provider())
    participant = _participant_release(tmp_path)
    assert trialeval.main([*_args(participant, tmp_path / "runs"), "--turns", "3"]) == 0
    run_root = next((tmp_path / "runs" / "fixture_model").iterdir())
    first = json.loads((run_root / "items" / "TASK1001.json").read_text())["agent_output"]
    second = json.loads((run_root / "items" / "TASK1002.json").read_text())["agent_output"]
    assert first["status"] == "failed"
    assert first["failure_type"] == "timeout"
    assert first["turns_used"] == 1
    assert second["status"] == "success"
    assert second["failure_type"] is None
    assert checked == ["TASK1001", "TASK1002", "TASK1002"]
    coverage = json.loads((run_root / "coverage.json").read_text())
    assert coverage["completed_unit_ids"] == coverage["unit_ids"] == ["TASK1001", "TASK1002"]
    events = [json.loads(line) for line in (run_root / "logs" / "TASK1001_events.jsonl").read_text().splitlines()]
    terminals = [event for event in events if event["event_type"] == "step_terminal"]
    assert len(terminals) == 1
    assert terminals[0]["failure_type"] == "timeout"
    assert (run_root / "provider_telemetry_summary.json").is_file()


def test_trialeval_records_seed_and_rejects_resume_seed_drift(tmp_path: Path, monkeypatch) -> None:
    participant = _participant_release(tmp_path)
    provider_arguments: list[dict[str, object]] = []

    def provider(*args, **kwargs):
        del args
        provider_arguments.append(kwargs)
        return object()

    monkeypatch.setattr(trialeval, "get_provider", provider)
    monkeypatch.setattr(
        trialeval,
        "run_agent",
        lambda item, provider, **kwargs: (_ for _ in ()).throw(RuntimeError("interrupted")),
    )
    seeded_args = [*_args(participant, tmp_path / "runs"), "--decoding-seed", "29"]
    with pytest.raises(RuntimeError, match="interrupted"):
        trialeval.main(seeded_args)
    run_root = next((tmp_path / "runs" / "fixture_model").iterdir())
    run_config = json.loads((run_root / "run_config.json").read_text(encoding="utf-8"))

    assert run_config["decoding"]["decoding_seed"] == 29
    assert provider_arguments == [
        {
            "routing": trialeval.ProviderRouting(provider="openrouter", openrouter_provider="GMICloud"),
            "send_temperature": True,
            "decoding_seed": 29,
            "reasoning_effort": None,
            "exclude_reasoning": False,
            "replay_reasoning": False,
            "timeout_s": 300.0,
            "retry_policy": trialeval.ProviderRetryPolicyV1(),
            "memory_strategy": "bounded",
            "request_control": None,
        }
    ]
    with pytest.raises(ValueError, match="identity"):
        trialeval.main(
            [
                *_args(participant, tmp_path / "unused"),
                "--decoding-seed",
                "30",
                "--resume-run-dir",
                str(run_root),
            ]
        )


def test_trialeval_resume_rejects_release_mutation_and_denominator_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    participant = _participant_release(tmp_path)
    monkeypatch.setattr(trialeval, "get_provider", lambda *args, **kwargs: object())

    def interrupted(item, provider, **kwargs):
        del item, provider, kwargs
        raise RuntimeError("interrupted")

    monkeypatch.setattr(trialeval, "run_agent", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        trialeval.main(_args(participant, tmp_path / "runs"))
    run_root = next((tmp_path / "runs" / "fixture_model").iterdir())

    changed_model_args = _args(participant, tmp_path / "unused")
    changed_model_args[changed_model_args.index("fixture/model")] = "fixture/other-model"
    with pytest.raises(ValueError, match="identity"):
        trialeval.main([*changed_model_args, "--resume-run-dir", str(run_root)])

    with pytest.raises(ValueError, match="denominator|identity"):
        trialeval.main(
            [
                *_args(participant, tmp_path / "unused"),
                "--task-id",
                "TASK1001",
                "--resume-run-dir",
                str(run_root),
            ]
        )

    task = participant / "items" / "TASK1001" / "task.json"
    payload = json.loads(task.read_text(encoding="utf-8"))
    payload["primary_endpoint_id"] = "mutated"
    task.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        trialeval.main([*_args(participant, tmp_path / "unused"), "--resume-run-dir", str(run_root)])


def test_trialeval_persists_and_forwards_analysis_contract(tmp_path: Path, monkeypatch) -> None:
    participant = _participant_release(tmp_path)
    observed: list[tuple[str, str]] = []
    monkeypatch.setattr(trialeval, "get_provider", lambda *args, **kwargs: object())

    def run(item, provider, **kwargs):
        del provider
        observed.append((kwargs["procedure_assistance"], kwargs["analysis_specification"]))
        return _agent_payload(item, **kwargs)

    monkeypatch.setattr(trialeval, "run_agent", run)
    assert trialeval.main(_args(participant, tmp_path / "runs")) == 0
    run_root = next((tmp_path / "runs" / "fixture_model").iterdir())
    run_config = json.loads((run_root / "run_config.json").read_text(encoding="utf-8"))

    assert run_config["experiment_condition"] == {
        "schema_id": "ariadne.experiment_condition/v1",
        "condition_id": "primary",
        "request_replicate_id": "request-1",
        "reasoning": {
            "effort": None,
            "exclude_from_response": False,
        },
        "provider_capability": None,
        "procedure_assistance": "output_contract_only",
        "maximum_turns_per_step": 1,
        "tool_choice": "auto",
        "memory_strategy": "bounded",
    }
    assert run_config["task_evidence_factors"] == {
        task_id: {
            "context_configuration": "C1",
            "data_preparation": "analysis_ready",
            "analysis_specification": "locked_sap",
        }
        for task_id in ("TASK1001", "TASK1002")
    }
    assert run_config["budget_profile_id"] == TRIALEVAL_RELEASE_BUDGET_V1.profile_id
    assert run_config["maximum_context_characters"] == TRIALEVAL_RELEASE_BUDGET_V1.maximum_context_characters
    assert observed == [("output_contract_only", "locked_sap")] * 2

    changed = _args(participant, tmp_path / "unused")
    turns_index = changed.index("--turns") + 1
    changed[turns_index] = "2"
    with pytest.raises(ValueError, match="identity"):
        trialeval.main([*changed, "--resume-run-dir", str(run_root)])


def test_trialeval_runs_the_unordered_checklist_ablation(tmp_path: Path, monkeypatch) -> None:
    """The documented assistance comparison uses the ordinary public runner."""

    participant = _participant_release(tmp_path)
    observed: list[str] = []
    monkeypatch.setattr(trialeval, "get_provider", lambda *args, **kwargs: object())

    def run(item, provider, **kwargs):
        del provider
        observed.append(kwargs["procedure_assistance"])
        return _agent_payload(item, **kwargs)

    monkeypatch.setattr(trialeval, "run_agent", run)
    arguments = [
        *_args(participant, tmp_path / "runs"),
        "--procedure-assistance",
        "unordered_checklist",
    ]

    assert trialeval.main(arguments) == 0

    run_root = next((tmp_path / "runs" / "fixture_model").iterdir())
    run_config = json.loads((run_root / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["experiment_condition"]["procedure_assistance"] == "unordered_checklist"
    assert observed == ["unordered_checklist", "unordered_checklist"]


def test_canonical_default_rejects_a_release_without_c5_but_explicit_ids_remain_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    participant = _participant_release(tmp_path)
    arguments = _args(participant, tmp_path / "runs")[:-2]

    def forbidden_provider(*args: object, **kwargs: object) -> None:
        raise AssertionError("selection and preparation must not contact a provider")

    monkeypatch.setattr(trialeval, "get_provider", forbidden_provider)
    with pytest.raises(ValueError, match="No TrialEval tasks match"):
        trialeval.main([*arguments, "--prepare-only"])
    assert trialeval.main([*arguments, "--task-id", "TASK1001", "--prepare-only"]) == 0
    config = json.loads(next((tmp_path / "runs").rglob("run_config.json")).read_text())
    assert config["task_ids"] == ["TASK1001"]
    with pytest.raises(ValueError, match="conflict"):
        trialeval.main([*arguments, "--task-id", "TASK1001", "--context", "C5", "--prepare-only"])


@pytest.mark.parametrize("outcome", ["refusal", "malformed_response", "empty_response"])
def test_resume_recovers_paid_noncompletion_before_item_publication_without_repeating_primary(
    tmp_path, monkeypatch, outcome
):
    from trialagentbench.ports import LLMResponse, LLMResponseMetadata
    from trialagentbench.util.provider_telemetry import start_provider_request_v1, succeed_provider_request_v1

    participant = _participant_release(tmp_path)
    monkeypatch.setattr(trialeval, "get_provider", lambda *args, **kwargs: object())
    arguments = [*_args(participant, tmp_path / "runs"), "--task-id", "TASK1001"]

    def death_after_response(item, provider, **kwargs):
        handle = start_provider_request_v1(
            path=kwargs["provider_log_path"],
            benchmark="trialeval",
            unit_id=item.item_id,
            phase_id="task",
            step_id="analysis",
            turn_index=1,
            requested_model="fixture/model",
            provider_route="openai",
        )
        succeed_provider_request_v1(
            handle,
            elapsed_seconds=0.1,
            response=LLMResponse(
                model_outcome=outcome,
                content="retained model outcome",
                usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                metadata=LLMResponseMetadata(reported_cost_usd=0.01),
            ),
        )
        raise RuntimeError("simulated process death before item publication")

    monkeypatch.setattr(trialeval, "run_agent", death_after_response)
    with pytest.raises(RuntimeError, match="simulated process death"):
        trialeval.main(arguments)
    run_root = next((tmp_path / "runs" / "fixture_model").iterdir())
    raw = (run_root / "logs" / "TASK1001_provider_responses.jsonl").read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("Resume repeated a paid primary")

    monkeypatch.setattr(trialeval, "get_provider", forbidden)
    monkeypatch.setattr(trialeval, "run_agent", forbidden)
    assert trialeval.main([*arguments, "--resume-run-dir", str(run_root)]) == 0
    result = json.loads((run_root / "items" / "TASK1001.json").read_text())
    assert (result["agent_output"]["status"], result["agent_output"]["failure_type"]) == ("failed", "contract_error")
    assert (run_root / "logs" / "TASK1001_provider_responses.jsonl").read_bytes() == raw
    summary = json.loads((run_root / "provider_telemetry_summary.json").read_text())
    assert summary["reported_cost_usd"] == 0.01 and summary["total_tokens"] == 12
    checkpoint = (run_root / "items" / "TASK1001.json").read_bytes()
    assert trialeval.main([*arguments, "--resume-run-dir", str(run_root)]) == 0
    assert (run_root / "items" / "TASK1001.json").read_bytes() == checkpoint


@pytest.mark.parametrize(
    "observed",
    [
        "start_only",
        "torn_tail",
        "paid_partial",
        "paid_failure",
        "paid_timeout",
        "cancelled_unanswered",
        "cancelled_after_response",
        "system_exit",
        "run_stop",
        "capacity_stop",
        "cancelled_after_model_outcome",
        "reported_cost_threshold",
        "reported_cost_unavailable",
        "backup_guard",
        "custody_guard",
        "participant_terminal",
        "evaluator_terminal",
        "operator_terminal",
        "timeout_terminal",
    ],
)
def test_resume_retains_an_interrupted_primary_for_separately_labelled_infrastructure_retry(
    tmp_path, monkeypatch, observed
):
    from datetime import UTC, datetime

    from trialagentbench.adapters.llm_providers import TransientProviderResponseError
    from trialagentbench.contracts.trace.observable import (
        BenchmarkRuntimeTraceEventV1,
        runtime_event_source_payload_v1,
    )
    from trialagentbench.ports import LLMResponse, RuntimeDeadlineExceeded
    from trialagentbench.trialeval.action_trace import _runtime_events, _validate_runtime_events
    from trialagentbench.util.campaign_control import CampaignCapacityStop
    from trialagentbench.util.provider_telemetry import (
        fail_provider_request_v1,
        provider_failure_type_v1,
        start_provider_request_v1,
        succeed_provider_request_v1,
    )
    from trialagentbench.util.reported_cost import (
        ReportedCostThresholdReached,
        ReportedCostUnavailable,
        RunStopRequested,
    )

    participant = _participant_release(tmp_path)
    arguments = [*_args(participant, tmp_path / "runs"), "--task-id", "TASK1001"]
    assert trialeval.main([*arguments, "--prepare-only"]) == 0
    run_root = next((tmp_path / "runs" / "fixture_model").iterdir())
    handle = start_provider_request_v1(
        path=run_root / "logs" / "TASK1001_provider_responses.jsonl",
        benchmark="trialeval",
        unit_id="TASK1001",
        phase_id="task",
        step_id="analysis",
        turn_index=1,
        requested_model="fixture/model",
        provider_route="openai",
    )
    if observed == "torn_tail":
        with handle.path.open("ab") as stream:
            stream.write(b'{"request_id":')
    elif observed in {
        "paid_partial",
        "participant_terminal",
        "evaluator_terminal",
        "operator_terminal",
        "timeout_terminal",
    }:
        succeed_provider_request_v1(handle, elapsed_seconds=0.1, response=LLMResponse(content="incomplete analysis"))
    elif observed in {"paid_failure", "paid_timeout"}:
        error = (
            RuntimeDeadlineExceeded("paid retry sequence reached its deadline")
            if observed == "paid_timeout"
            else TransientProviderResponseError("empty paid envelope", raw_response={"id": "retained"})
        )
        error.response_received = True
        fail_provider_request_v1(
            handle,
            elapsed_seconds=0.1,
            failure_type="timeout" if observed == "paid_timeout" else "provider_error",
            error=error,
        )
    else:
        cancellations = {
            "cancelled_unanswered": KeyboardInterrupt(),
            "cancelled_after_response": KeyboardInterrupt(),
            "system_exit": SystemExit(),
            "run_stop": RunStopRequested("operator stop"),
            "capacity_stop": CampaignCapacityStop("campaign stopped"),
            "cancelled_after_model_outcome": KeyboardInterrupt(),
            "reported_cost_threshold": ReportedCostThresholdReached("operator cost threshold"),
            "reported_cost_unavailable": ReportedCostUnavailable("operator cost evidence unavailable"),
            "backup_guard": CampaignCapacityStop("synthetic backup health guard stop"),
            "custody_guard": CampaignCapacityStop("synthetic custody health guard stop"),
        }
        if observed in cancellations:
            if observed != "cancelled_unanswered":
                succeed_provider_request_v1(
                    handle,
                    elapsed_seconds=0.1,
                    response=LLMResponse(
                        content="retained response",
                        model_outcome="refusal" if observed == "cancelled_after_model_outcome" else None,
                    ),
                )
                handle = start_provider_request_v1(
                    path=handle.path,
                    benchmark="trialeval",
                    unit_id="TASK1001",
                    phase_id="task",
                    step_id="analysis",
                    turn_index=2,
                    requested_model="fixture/model",
                    provider_route="openai",
                )
            error = cancellations[observed]
            fail_provider_request_v1(
                handle,
                elapsed_seconds=0.0001,
                failure_type=provider_failure_type_v1(error),
                error=error,
            )
    raw = handle.path.read_bytes()
    conversation_path = run_root / "logs" / "TASK1001_conversation.json"
    conversation_path.write_text("[]\n")
    payload = runtime_event_source_payload_v1(
        benchmark="trialeval",
        task_id="TASK1001",
        program_id=None,
        scenario_id=None,
        objective_id=None,
        phase_id="task",
        step_id="analysis",
        event_type="step_started",
        terminal_status=None,
        failure_type=None,
        conversation_message=None,
    )
    started = BenchmarkRuntimeTraceEventV1(
        event_id="trialeval:TASK1001:000000",
        timestamp=datetime.now(UTC),
        source_artifact_path=str(conversation_path),
        source_payload_sha256=canonical_payload_sha256(payload),
        benchmark="trialeval",
        task_id="TASK1001",
        phase_id="task",
        step_id="analysis",
        event_type="step_started",
        event_index=0,
    )
    runtime_path = run_root / "logs" / "TASK1001_events.jsonl"
    runtime_path.write_text(started.model_dump_json() + "\n")
    if observed in {"participant_terminal", "evaluator_terminal", "operator_terminal", "timeout_terminal"}:
        failure = {
            "participant_terminal": "participant_session_terminated",
            "evaluator_terminal": "evaluator_error",
            "operator_terminal": "cancelled",
            "timeout_terminal": "timeout",
        }[observed]
        terminal_payload = runtime_event_source_payload_v1(
            benchmark="trialeval",
            task_id="TASK1001",
            program_id=None,
            scenario_id=None,
            objective_id=None,
            phase_id="task",
            step_id="analysis",
            event_type="step_terminal",
            terminal_status="failed",
            failure_type=failure,
            conversation_message=None,
        )
        terminal = started.model_copy(
            update={
                "event_id": "trialeval:TASK1001:000001",
                "event_index": 1,
                "event_type": "step_terminal",
                "terminal_status": "failed",
                "failure_type": failure,
                "source_payload_sha256": canonical_payload_sha256(terminal_payload),
            }
        )
        with runtime_path.open("a") as stream:
            stream.write(terminal.model_dump_json() + "\n")
    incomplete_trace = runtime_path.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("Resume repeated a possibly paid primary")

    monkeypatch.setattr(trialeval, "get_provider", forbidden)
    assert trialeval.main([*arguments, "--resume-run-dir", str(run_root)]) == 0
    assert handle.path.read_bytes() == raw
    checkpoint = json.loads((run_root / "items" / "TASK1001.json").read_bytes())
    assert checkpoint["agent_output"]["failure_type"] == {
        "paid_failure": "provider_error",
        "paid_timeout": "timeout",
        "paid_partial": "runtime_error",
        "cancelled_after_model_outcome": "contract_error",
        "participant_terminal": "participant_session_terminated",
        "operator_terminal": "cancelled",
        "timeout_terminal": "timeout",
    }.get(observed, "evaluator_error")
    summary = json.loads((run_root / "provider_telemetry_summary.json").read_bytes())
    assert summary["interrupted_request_count"] == (1 if observed in {"start_only", "torn_tail"} else 0)
    assert runtime_path.read_bytes().startswith(incomplete_trace)
    events = _runtime_events(runtime_path)
    _validate_runtime_events(events, task_id="TASK1001", conversation_path=conversation_path)
    assert events[-1].failure_type == checkpoint["agent_output"]["failure_type"]
    terminal_trace = runtime_path.read_bytes()
    assert trialeval.main([*arguments, "--resume-run-dir", str(run_root)]) == 0
    assert runtime_path.read_bytes() == terminal_trace


@pytest.mark.parametrize(
    "failure,declared_budget,expected",
    [("provider_error", False, "provider_error"), ("timeout", True, "timeout"), ("timeout", False, "provider_error")],
)
@pytest.mark.parametrize("torn", [False, True])
def test_resume_retains_unanswered_failed_request_before_runtime_terminal(
    tmp_path, monkeypatch, failure, declared_budget, expected, torn
):
    from trialagentbench_verifier.campaign import _attempt_outcome

    from trialagentbench.contracts.core.runs import TrialEvalRunConfigV1
    from trialagentbench.io import read_json_model
    from trialagentbench.ports import RuntimeDeadlineExceeded
    from trialagentbench.tools import campaign
    from trialagentbench.util.provider_telemetry import fail_provider_request_v1, start_provider_request_v1

    arguments = [*_args(_participant_release(tmp_path), tmp_path / "runs"), "--task-id", "TASK1001"]
    assert trialeval.main([*arguments, "--prepare-only"]) == 0
    native = next((tmp_path / "runs" / "fixture_model").iterdir())
    request = start_provider_request_v1(
        path=native / "logs/TASK1001_provider_responses.jsonl",
        benchmark="trialeval",
        unit_id="TASK1001",
        phase_id="task",
        step_id="analysis",
        turn_index=1,
        requested_model="fixture/model",
        provider_route="openai",
    )
    fail_provider_request_v1(
        request,
        elapsed_seconds=0.1,
        failure_type=failure,
        error=(
            RuntimeDeadlineExceeded("declared task budget exhausted")
            if declared_budget
            else TimeoutError("unqualified operation failure")
        ),
    )
    if torn:
        with request.path.open("ab") as stream:
            stream.write(b'{"request_id":')
    raw = request.path.read_bytes()
    assert not (native / "logs/TASK1001_events.jsonl").exists()

    def forbidden(*args, **kwargs):
        raise AssertionError("Resume re-queued a failed unanswered primary")

    monkeypatch.setattr(trialeval, "get_provider", forbidden)
    assert trialeval.main([*arguments, "--resume-run-dir", str(native)]) == 0
    checkpoint = native / "items/TASK1001.json"
    saved = checkpoint.read_bytes()
    assert json.loads(saved)["agent_output"]["failure_type"] == expected
    assert request.path.read_bytes() == raw
    assert not (native / "failed_attempts").exists()
    config = read_json_model(TrialEvalRunConfigV1, native / "run_config.json")
    assert not campaign._retryable_trialeval_primary(native, config, "TASK1001")
    missing = SimpleNamespace(status="not_satisfied")
    assert _attempt_outcome(
        agent_status="failed", failure_type=expected, completion=missing, reproducibility=missing
    ) == ("budget_exhausted" if expected == "timeout" else "provider_stop")
    assert trialeval.main([*arguments, "--resume-run-dir", str(native)]) == 0
    assert checkpoint.read_bytes() == saved


@pytest.mark.parametrize(
    "failure,request_state,expected",
    [
        ("runtime_error", "none", "runtime_error"),
        ("runtime_error", "started", "runtime_error"),
        ("runtime_error", "failed", "runtime_error"),
        ("runtime_error", "paid", "runtime_error"),
        ("provider_error", "outage", "provider_unavailable"),
        ("provider_error", "outage_torn", "provider_unavailable"),
        ("participant_session_terminated", "outage", "participant_session_terminated"),
        ("timeout", "outage", "timeout"),
        ("runtime_error", "outage", "runtime_error"),
        ("runtime_error", "outage_torn", "runtime_error"),
        ("provider_error", "model_outcome", "contract_error"),
    ],
)
def test_resume_reconciles_failure_terminal_before_queuing_primary(
    tmp_path, monkeypatch, failure, request_state, expected
):
    from datetime import UTC, datetime

    from trialagentbench.contracts.trace.observable import (
        BenchmarkRuntimeTraceEventV1,
        runtime_event_source_payload_v1,
    )
    from trialagentbench.ports import LLMResponse
    from trialagentbench.ports.llm_provider import ProviderUnavailableError, RetryTelemetry
    from trialagentbench.util.provider_telemetry import (
        fail_provider_request_v1,
        start_provider_request_v1,
        succeed_provider_request_v1,
    )

    arguments = [
        *_args(_participant_release(tmp_path), tmp_path / "runs"),
        "--task-id",
        "TASK1001",
        "--provider-unavailability-policy",
        "defer_unanswered_v1",
    ]
    assert trialeval.main([*arguments, "--prepare-only"]) == 0
    native = next((tmp_path / "runs" / "fixture_model").iterdir())
    if request_state != "none":
        handle = start_provider_request_v1(
            path=native / "logs/TASK1001_provider_responses.jsonl",
            benchmark="trialeval",
            unit_id="TASK1001",
            phase_id="task",
            step_id="analysis",
            turn_index=1,
            requested_model="fixture/model",
            provider_route="openai",
        )
        if request_state in {"paid", "model_outcome"}:
            succeed_provider_request_v1(
                handle,
                elapsed_seconds=0.1,
                response=LLMResponse(
                    content="retained partial response",
                    model_outcome="refusal" if request_state == "model_outcome" else None,
                ),
            )
        elif request_state in {"failed", "outage", "outage_torn"}:
            error = (
                ProviderUnavailableError(
                    "unanswered outage",
                    retry_telemetry=RetryTelemetry(1, 1, 0),
                    failure_kind="connection_error",
                    unanswered_transport_seconds=0,
                )
                if request_state != "failed"
                else TimeoutError("unqualified operation timeout")
            )
            fail_provider_request_v1(handle, elapsed_seconds=0.1, failure_type="provider_error", error=error)
            if request_state == "outage_torn":
                with handle.path.open("ab") as stream:
                    stream.write(b'{"request_id":')
    conversation = native / "logs/TASK1001_conversation.json"
    conversation.parent.mkdir(exist_ok=True)
    conversation.write_text("[]\n")
    payload = runtime_event_source_payload_v1(
        benchmark="trialeval",
        task_id="TASK1001",
        program_id=None,
        scenario_id=None,
        objective_id=None,
        phase_id="task",
        step_id="analysis",
        event_type="step_terminal",
        terminal_status="failed",
        failure_type=failure,
        conversation_message=None,
    )
    terminal = BenchmarkRuntimeTraceEventV1(
        event_id="trialeval:TASK1001:000000",
        timestamp=datetime.now(UTC),
        source_artifact_path=str(conversation),
        source_payload_sha256=canonical_payload_sha256(payload),
        benchmark="trialeval",
        task_id="TASK1001",
        event_index=0,
        phase_id="task",
        step_id="analysis",
        event_type="step_terminal",
        terminal_status="failed",
        failure_type=failure,
    )
    (native / "logs/TASK1001_events.jsonl").write_text(terminal.model_dump_json() + "\n")
    retained_logs = {path: path.read_bytes() for path in (native / "logs").glob("*") if path.is_file()}

    def forbidden(*args, **kwargs):
        raise AssertionError("Resume re-queued a terminal primary")

    monkeypatch.setattr(trialeval, "get_provider", forbidden)
    assert trialeval.main([*arguments, "--resume-run-dir", str(native)]) == 0
    result = json.loads((native / "items/TASK1001.json").read_bytes())
    assert result["agent_output"]["failure_type"] == expected
    if expected == "provider_unavailable":
        assert result["agent_output"]["provider_unavailability"]["request_event"]["response_received"] is False
    assert not (native / "failed_attempts").exists()
    assert all(path.read_bytes().startswith(raw) for path, raw in retained_logs.items())


def _slice_campaign(tmp_path, run_root):
    from trialagentbench.contracts.campaign import CampaignConfigurationV1, CampaignPlanV1, CampaignRunV1
    from trialagentbench.io import write_json_model

    config = CampaignConfigurationV1.model_validate(
        {
            "release_directory": str(tmp_path),
            "expected_release_manifest_sha256": "a" * 64,
            "executor_image_id": EXECUTOR_ENVIRONMENT.image_id,
            "models": [
                {
                    "name": "fixture",
                    "model": "fixture/model",
                    "upstream_provider": "GMICloud",
                    "spend_limit_micro_usd": 10000,
                    "reservation_micro_usd": 100,
                    "reservation_tokens": 40000,
                    "pricing_premise": "offline fixture",
                    "requests_per_minute": 100,
                    "tokens_per_minute": 100000,
                }
            ],
            "panel": {"benchmarks": ["trialeval"], "trialeval_tasks": ["TASK1001", "TASK1002"]},
            "operations": {"spend_limit_micro_usd": 10000},
            "dispatch": {"task_order": [{"benchmark": "trialeval", "task_id": t} for t in ["TASK1002", "TASK1001"]]},
        }
    )
    run = CampaignRunV1(
        job_id="fixture-trialeval-1",
        benchmark="trialeval",
        model_name="fixture",
        repetition=1,
        relative_run_root=run_root.relative_to(tmp_path).as_posix(),
        invocation_sha256=hashlib.sha256((run_root / "run_config.json").read_bytes()).hexdigest(),
        analysis_plan_path="plans/fixture.json",
        analysis_plan_sha256="a" * 64,
    )
    write_json_model(tmp_path / "campaign.json", CampaignPlanV1(configuration=config, runs=(run,)))
    return ["--campaign-root", str(tmp_path), "--campaign-job", run.job_id]


def test_native_eval_slices_keep_full_identity_and_only_finalize_full_coverage(tmp_path, monkeypatch):
    participant = _participant_release(tmp_path)
    arguments = _args(participant, tmp_path / "runs")
    assert trialeval.main([*arguments, "--prepare-only"]) == 0
    config_path = next((tmp_path / "runs").rglob("run_config.json"))
    before = config_path.read_bytes()
    root = config_path.parent
    flags = _slice_campaign(tmp_path, root)
    called = []
    monkeypatch.setattr(trialeval, "get_provider", lambda *a, **k: object())

    def run(item, provider, **kwargs):
        called.append(item.item_id)
        return _agent_payload(item, **kwargs)

    monkeypatch.setattr(trialeval, "run_agent", run)
    resume = [*arguments, "--resume-run-dir", str(root), *flags]
    assert trialeval.main([*resume, "--dispatch-task-id", "TASK1002"]) == 0
    assert called == ["TASK1002"]
    coverage = json.loads((root / "coverage.json").read_text())
    assert coverage["unit_ids"] == ["TASK1001", "TASK1002"]
    assert coverage["completed_unit_ids"] == ["TASK1002"]
    assert not (root / "provider_telemetry_summary.json").exists()
    assert trialeval.main([*resume, "--dispatch-task-id", "TASK1001"]) == 0
    assert called == ["TASK1002", "TASK1001"]
    assert (root / "provider_telemetry_summary.json").exists()
    assert trialeval.main(resume) == 0
    assert called == ["TASK1002", "TASK1001"] and config_path.read_bytes() == before


def test_eval_provider_deferral_retains_failed_primary_and_never_restarts_it(tmp_path, monkeypatch):
    from trialagentbench.ports.llm_provider import ProviderUnavailableError, RetryTelemetry
    from trialagentbench.util.provider_telemetry import fail_provider_request_v1, start_provider_request_v1

    participant = _participant_release(tmp_path)
    arguments = [*_args(participant, tmp_path / "runs"), "--provider-unavailability-policy", "defer_unanswered_v1"]
    assert trialeval.main([*arguments, "--prepare-only"]) == 0
    root = next((tmp_path / "runs").rglob("run_config.json")).parent
    flags = _slice_campaign(tmp_path, root)
    called = []
    monkeypatch.setattr(trialeval, "get_provider", lambda *a, **k: object())

    def run(item, provider, **kwargs):
        called.append(item.item_id)
        if item.item_id == "TASK1002":
            error = ProviderUnavailableError(
                "offline failure",
                retry_telemetry=RetryTelemetry(1, 1, 0),
                failure_kind="connection_error",
                unanswered_transport_seconds=0,
            )
            handle = start_provider_request_v1(
                path=kwargs["provider_log_path"],
                benchmark="trialeval",
                unit_id=item.item_id,
                phase_id="task",
                step_id="analysis",
                turn_index=1,
                requested_model="fixture/model",
                provider_route="openrouter:GMICloud",
            )
            fail_provider_request_v1(handle, elapsed_seconds=0.1, failure_type="provider_error", error=error)
            raise error
        return _agent_payload(item, **kwargs)

    monkeypatch.setattr(trialeval, "run_agent", run)
    resume = [*arguments, "--resume-run-dir", str(root), *flags]
    with pytest.raises(ProviderUnavailableError):
        trialeval.main([*resume, "--dispatch-task-id", "TASK1002"])
    result = json.loads((root / "items/TASK1002.json").read_text())
    assert result["agent_output"]["status"] == "failed"
    assert result["agent_output"]["failure_type"] == "provider_unavailable"
    assert result["agent_output"]["provider_unavailability"]["request_event"]["unit_id"] == "TASK1002"
    assert trialeval.main([*resume, "--dispatch-task-id", "TASK1001"]) == 0
    assert trialeval.main(resume) == 0
    assert called == ["TASK1002", "TASK1001"]


@pytest.mark.parametrize(
    "scope, capacity_stop, received, censored",
    [
        ("fixture", True, False, True),
        ("campaign", True, False, True),
        (None, True, False, False),
        ("fixture", False, False, False),
        ("fixture", True, True, False),
    ],
)
def test_resume_preserves_only_proven_pretransport_budget_stops(
    tmp_path, monkeypatch, scope, capacity_stop, received, censored
):
    """Only an unanswered capacity stop backed by the campaign ledger remains pending."""
    import sqlite3

    from trialagentbench.contracts.campaign import CampaignPlanV1
    from trialagentbench.io import read_json_model
    from trialagentbench.util.campaign_control import (
        CAMPAIGN_BUDGET_SCOPE,
        CampaignCapacityStop,
        request_control_for_run,
    )
    from trialagentbench.util.provider_telemetry import fail_provider_request_v1, start_provider_request_v1

    arguments = _args(_participant_release(tmp_path), tmp_path / "runs")
    assert trialeval.main([*arguments, "--prepare-only"]) == 0
    native = next((tmp_path / "runs" / "fixture_model").iterdir())
    campaign_args = _slice_campaign(tmp_path, native)
    control = request_control_for_run(campaign_root=tmp_path, job_id="fixture-trialeval-1", run_root=native)
    assert control is not None
    if scope is not None:
        ledger_scope = CAMPAIGN_BUDGET_SCOPE if scope == "campaign" else scope
        with sqlite3.connect(control.database) as connection:
            connection.execute("INSERT INTO exhausted VALUES (?,?)", (ledger_scope, "controlled exhaustion"))
    # The real campaign identity and ledger stay bound; select one of its two tasks.
    monkeypatch.setattr(trialeval, "execution_slice", lambda **kwargs: ("TASK1001",))
    handle = start_provider_request_v1(
        path=native / "logs/TASK1001_provider_responses.jsonl",
        benchmark="trialeval",
        unit_id="TASK1001",
        phase_id="task",
        step_id="analysis",
        turn_index=1,
        requested_model="fixture/model",
        provider_route="openai",
    )
    error = CampaignCapacityStop("controlled exhaustion") if capacity_stop else KeyboardInterrupt()
    if received:
        error.raw_response = {"id": "received"}
    fail_provider_request_v1(handle, elapsed_seconds=0.1, failure_type="cancelled", error=error)
    before = {p: p.read_bytes() for p in (native / "logs").glob("*")}

    def forbidden(*args, **kwargs):
        raise AssertionError("Resume started a new model conversation")

    monkeypatch.setattr(trialeval, "get_provider", forbidden)
    resumed = [*arguments, *campaign_args, "--resume-run-dir", str(native)]
    if censored:
        with pytest.raises(CampaignCapacityStop, match="controlled exhaustion"):
            trialeval.main(resumed)
        assert not (native / "items/TASK1001.json").exists()
        assert json.loads((native / "coverage.json").read_text())["completed_unit_ids"] == []
        assert read_json_model(CampaignPlanV1, tmp_path / "campaign.json").runs[0].retry_of_job_id is None
    else:
        assert trialeval.main(resumed) == 0
        assert (
            json.loads((native / "items/TASK1001.json").read_text())["agent_output"]["failure_type"]
            == "evaluator_error"
        )
    assert all(p.read_bytes() == raw for p, raw in before.items())
