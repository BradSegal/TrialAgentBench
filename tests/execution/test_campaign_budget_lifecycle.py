"""Budget-capped and infrastructure-retried campaigns finish through the real native lifecycle.

Both offline-control routes run the scripted teaching analysis, whose study design fits the
small teaching world; every turn reserves one microdollar through the campaign's real spending
ledger, so a model spending limit censors part of the campaign exactly as a provider route would. The declared input set is the shipped
teaching corpus, pinned as development inputs in ``tmp_path``; the test requires Docker and the
local executor image.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import time
import zipfile
from pathlib import Path

import pytest
from trialagentbench_verifier.campaign import TrialEvalCampaignUnavailableV1
from trialagentbench_verifier.campaign_contracts import BudgetCensoringV1
from trialagentbench_verifier.campaign_inputs import (
    DevelopmentArchiveV1,
    DevelopmentCampaignInputsV1,
    DevelopmentExecutorV1,
)
from trialagentbench_verifier.development_campaign import TrialDevCampaignAnalysisV1
from trialagentbench_verifier.execution_contracts import RELEASE_EXECUTION_LIMITS_V1
from trialagentbench_verifier.executor import resolve_executor_environment
from trialagentbench_verifier.io import sha256_file
from trialagentbench_verifier.public_results import PublicTrialDevCampaignV1

from trialagentbench.tools import campaign
from trialagentbench.tools.demo import ScriptedDemoProvider, demo_assets
from trialagentbench.tools.run import canonical_trialdev, trialeval
from trialagentbench.util.campaign_control import CampaignRequestControl

pytestmark = pytest.mark.executor
# Scripted turns: the first programme takes 6; the r control's limit stops it inside the second.
R_TURN_LIMIT = 8
TEACHING_TRIALEVAL = demo_assets() / "demo-trialeval-evaluator.zip"
TEACHING_TRIALDEV = demo_assets() / "demo-trialdev-evaluator.zip"


def _teaching_task_ids(archive_path: Path, **factors: str) -> tuple[str, ...]:
    """Select current task IDs from a bundled archive's participant metadata."""
    with zipfile.ZipFile(archive_path) as archive:
        tasks = [
            json.loads(archive.read(name))
            for name in archive.namelist()
            if name.startswith("participant/tasks/") and name.endswith("/task.json")
        ]
    return tuple(
        sorted(task["task_id"] for task in tasks if all(task[key] == value for key, value in factors.items()))
    )


(TRIALEVAL_TASK,) = _teaching_task_ids(TEACHING_TRIALEVAL)
FIRST, SECOND = _teaching_task_ids(TEACHING_TRIALDEV, stream="portfolio_management", data_view="analysis_ready")[:2]


class _MeteredControl:
    """Reserve one microdollar per scripted turn in the campaign ledger before delegating."""

    def __init__(self, provider: object, control: CampaignRequestControl) -> None:
        self._provider = provider
        self._control = control

    def __getattr__(self, name: str) -> object:
        return getattr(self._provider, name)

    def generate_turn(self, *args: object, **kwargs: object) -> object:
        model = self._control.model
        assert model is not None
        if self._control.try_reserve(model=model, now=time.time(), tokens=1, micro_usd=1) is None:
            raise AssertionError("Offline control capacity windows are unbounded")
        return self._provider.generate_turn(*args, **kwargs)  # type: ignore[attr-defined]


def _metered(model: str, *, request_control: CampaignRequestControl | None = None, **kwargs: object) -> object:
    assert request_control is not None, "campaign execution binds the spending ledger"
    return _MeteredControl(ScriptedDemoProvider(), request_control)


def _model(name: str, limit: int) -> dict[str, object]:
    return {
        "name": name,
        "model": f"smoke-zero-{name}",
        "provider": "offline_control",
        "maximum_completion_tokens": 1000,
        "spend_limit_micro_usd": limit,
        "reservation_micro_usd": 1,
        "reservation_tokens": 200000,
        "pricing_premise": "Offline control metered at one microdollar per turn to exercise spending limits.",
        "requests_per_minute": 100000,
        "tokens_per_minute": 10000000,
    }


def _development_inputs(
    root: Path,
    *,
    trialeval_evaluator: Path = TEACHING_TRIALEVAL,
    trialdev_evaluator: Path | None = TEACHING_TRIALDEV,
) -> tuple[str, str]:
    """Pin evaluator archives (the teaching ones by default) and the local executor as development inputs."""
    root.mkdir()
    if trialdev_evaluator is not None:
        shutil.copyfile(trialdev_evaluator, root / "trialdev-evaluator.zip")
    evaluator = root / "trialeval-evaluator.zip"
    shutil.copyfile(trialeval_evaluator, evaluator)
    # Project the paired participant archive with the release writer's canonical member encoding.
    with zipfile.ZipFile(evaluator) as source, zipfile.ZipFile(root / "trialeval-participant.zip", "x") as target:
        identity = json.loads(source.read("participant_archive_identity.json"))
        names = [name for name in source.namelist() if name.startswith("participant/")] + ["release_manifest.json"]
        for name in names:
            info = zipfile.ZipInfo(name.removeprefix("participant/"), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type, info.external_attr = zipfile.ZIP_DEFLATED, 0o100644 << 16
            target.writestr(info, source.read(name), compresslevel=6)
    assert sha256_file(root / "trialeval-participant.zip") == identity["participant_archive_sha256"]
    executor = resolve_executor_environment(image=None, limits=RELEASE_EXECUTION_LIMITS_V1)
    manifest = DevelopmentCampaignInputsV1(
        artifacts=tuple(
            DevelopmentArchiveV1(filename=path.name, sha256=sha256_file(path), byte_length=path.stat().st_size)
            for path in sorted(root.glob("*.zip"))
        ),
        executor=DevelopmentExecutorV1(image_id=executor.image_id, environment_sha256=executor.semantic_sha256()),
    )
    (root / "development-inputs.json").write_text(manifest.model_dump_json())
    return sha256_file(root / "development-inputs.json"), executor.image_id


def test_budget_capped_campaign_finishes_with_paired_comparisons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    release = tmp_path / "development"
    manifest_sha256, image_id = _development_inputs(release)
    config = tmp_path / "campaign-config.json"
    config.write_text(
        json.dumps(
            {
                "release_directory": str(release),
                "input_mode": "development",
                "expected_release_manifest_sha256": manifest_sha256,
                "executor_image_id": image_id,
                "models": [_model("python", 1000), _model("r", R_TURN_LIMIT)],
                "panel": {
                    "benchmarks": ["trialdev", "trialeval"],
                    "trialdev_tasks": [FIRST, SECOND],
                    "trialeval_tasks": [TRIALEVAL_TASK],
                    "trialdev_turns_per_step": 12,
                    "trialeval_turns": 12,
                    "watchdog_seconds": 900,
                    "condition_id": "budget-censoring-lifecycle",
                },
                "operations": {
                    "model_workers": 1,
                    "task_workers": 1,
                    "executor_workers": 2,
                    "grading_workers": 1,
                    "spend_limit_micro_usd": 2000,
                },
            }
        )
    )
    monkeypatch.setattr(canonical_trialdev, "get_provider", _metered)
    monkeypatch.setattr(trialeval, "get_provider", _metered)
    root = tmp_path / "campaign"
    assert campaign.main(["prepare", "--config", str(config), "--campaign", str(root)]) == 0
    assert campaign.main(["run", "--campaign", str(root)]) == 3

    def ledger() -> list[tuple[str, int]]:
        with sqlite3.connect(root / "requests.sqlite") as connection:
            return connection.execute("SELECT model,COUNT(*) FROM attempts GROUP BY model ORDER BY model").fetchall()

    spent = ledger()
    assert spent == [("python", spent[0][1]), ("r", R_TURN_LIMIT)]
    # Budgets never refill: resuming re-censors without a single new request.
    assert campaign.main(["resume", "--campaign", str(root)]) == 3
    assert ledger() == spent
    capsys.readouterr()

    assert campaign.main(["finish", "--campaign", str(root)]) == 0
    assert "WARNING: trialdev: budget censoring differs materially" in capsys.readouterr().out
    comparisons = root / "comparisons"
    censoring = BudgetCensoringV1.model_validate_json((comparisons / "censoring.json").read_bytes())
    assert censoring.exhausted_models == ("r",)
    assert censoring.censored_by_job("trialdev") == {"python-trialdev-1": frozenset(), "r-trialdev-1": {SECOND}}
    assert censoring.censored_by_job("trialeval") == {
        "python-trialeval-1": frozenset(),
        "r-trialeval-1": {TRIALEVAL_TASK},
    }
    assert censoring.paired_task_ids("trialdev") == {FIRST}

    development = TrialDevCampaignAnalysisV1.model_validate_json((comparisons / "trialdev/analysis.json").read_bytes())
    assert development.budget_censoring == censoring
    assert development.comparisons and {row.worlds for row in development.comparisons} == {1}
    assert (comparisons / "trialdev/comparisons.csv").is_file()
    unavailable = TrialEvalCampaignUnavailableV1.model_validate_json(
        (comparisons / "trialeval/analysis.json").read_bytes()
    )
    assert "completed by every model" in unavailable.reason

    # Development inputs publish an unqualified aggregate instead of a leaderboard; it carries the record.
    results = json.loads((comparisons / "trialdev/development-results.json").read_bytes())
    assert results["status"] == "unqualified" and results["input_manifest_sha256"] == manifest_sha256
    assert PublicTrialDevCampaignV1.model_validate(results["result"]).budget_censoring == censoring
    index = (root / "index.html").read_text()
    assert "Budget censoring" in index and "budget-censored 1" in index
    # Finishing is repeatable: retained stages of censored runs are verified and reused byte for byte.
    retained = (comparisons / "trialdev/development-results.json").read_bytes()
    assert campaign.main(["finish", "--campaign", str(root)]) == 0
    assert (comparisons / "trialdev/development-results.json").read_bytes() == retained


def _run_faulted_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_kind: str
) -> tuple[Path, Path, bytes]:
    """Prepare and execute one primary with a controlled fault."""
    from trialagentbench.ports.llm_provider import (
        LLMResponse,
        ProviderRequestError,
        ProviderUnavailableError,
        RetryTelemetry,
    )

    release = tmp_path / "development"
    manifest_sha256, image_id = _development_inputs(release, trialdev_evaluator=None)
    config = tmp_path / "campaign-config.json"
    config.write_text(
        json.dumps(
            {
                "release_directory": str(release),
                "input_mode": "development",
                "expected_release_manifest_sha256": manifest_sha256,
                "executor_image_id": image_id,
                "models": [
                    {
                        **_model("python", 1000),
                        "retry": {
                            "unavailability_policy": (
                                "charge_all_v1" if failure_kind.startswith("executor_") else "defer_unanswered_v1"
                            )
                        },
                    }
                ],
                "panel": {
                    "benchmarks": ["trialeval"],
                    "trialeval_tasks": [TRIALEVAL_TASK],
                    "trialeval_turns": 12,
                    "condition_id": "infrastructure-retry-lifecycle",
                },
                "dispatch": {
                    "task_order": [{"benchmark": "trialeval", "task_id": TRIALEVAL_TASK}],
                    "defer_seconds": 1,
                },
                "operations": {
                    "model_workers": 1,
                    "task_workers": 1,
                    "executor_workers": 2,
                    "grading_workers": 1,
                    "spend_limit_micro_usd": 1000,
                },
            }
        )
    )
    lost_sessions: list[str] = []

    class _LostSession:
        """End the first primary's execution session, as a host fault would."""

        def __init__(self, provider: object) -> None:
            self._provider = provider
            self._calls = 0

        def __getattr__(self, name: str) -> object:
            return getattr(self._provider, name)

        def generate_turn(self, *args: object, **kwargs: object) -> object:
            self._calls += 1
            if failure_kind == "executor_close_turn_limit":
                return LLMResponse(content="Analysis incomplete.")
            if failure_kind == "model_contract_error":
                return LLMResponse(model_outcome="malformed_response")
            if failure_kind == "generic_provider_error":
                if self._calls == 1:
                    return self._provider.generate_turn(*args, **kwargs)
                raise ProviderRequestError(
                    "generic failure after a received response", retry_telemetry=RetryTelemetry(1, 1, 0)
                )
            if failure_kind.startswith("provider_unavailable"):
                if failure_kind == "provider_unavailable_after_analysis" and self._calls == 1:
                    return self._provider.generate_turn(*args, **kwargs)
                control = self._provider._control
                assert control.try_reserve(model=control.model, now=time.time(), tokens=1, micro_usd=1) is not None
                raise ProviderUnavailableError(
                    "controlled unanswered provider outage",
                    retry_telemetry=RetryTelemetry(1, 1, 0),
                    failure_kind="connection",
                    unanswered_transport_seconds=0,
                )
            return self._provider.generate_turn(*args, **kwargs)

    host_faults = []
    if failure_kind == "executor_loss":
        from trialagentbench.trialeval import agent as trialeval_agent

        execute = trialeval_agent.DockerCodeSession.execute

        def host_timeout(session, *args, **kwargs):
            if not host_faults:
                host_faults.append("docker exec")
                raise subprocess.TimeoutExpired(cmd=["docker", "exec", "controlled-session", "python"], timeout=1)
            return execute(session, *args, **kwargs)

        monkeypatch.setattr(trialeval_agent.DockerCodeSession, "execute", host_timeout)
    elif failure_kind.startswith("executor_close"):
        from trialagentbench.trialeval import agent as trialeval_agent

        close = trialeval_agent.DockerCodeSession.close

        def close_timeout(session):
            # Release the real test container, then inject the host-control failure.
            close(session)
            if not host_faults:
                host_faults.append("docker rm")
                raise subprocess.TimeoutExpired(cmd=["docker", "rm", "--force", "controlled-session"], timeout=1)

        monkeypatch.setattr(trialeval_agent.DockerCodeSession, "close", close_timeout)

    def provider(model: str, **kwargs: object) -> object:
        metered = _metered(model, **kwargs)  # type: ignore[arg-type]
        # The scripted implementation is unchanged; telemetry must identify the declared test route.
        metered.model = model
        if lost_sessions:
            return metered
        lost_sessions.append(model)
        return _LostSession(metered)

    monkeypatch.setattr(trialeval, "get_provider", provider)
    root = tmp_path / "campaign"
    assert campaign.main(["prepare", "--config", str(config), "--campaign", str(root)]) == 0
    if failure_kind == "generic_provider_error":
        with pytest.raises(ProviderRequestError):
            campaign.main(["run", "--campaign", str(root)])
        assert campaign.main(["resume", "--campaign", str(root)]) == 0
    else:
        assert campaign.main(["run", "--campaign", str(root)]) == 0
    assert lost_sessions == ["smoke-zero-python"]
    if failure_kind.startswith("executor_"):
        assert host_faults == ["docker exec" if failure_kind == "executor_loss" else "docker rm"]
    primary_run = campaign.load_campaign(root).runs[0]
    primary_path = root / primary_run.relative_run_root / "items" / f"{TRIALEVAL_TASK}.json"
    primary_bytes = primary_path.read_bytes()
    return root, primary_path, primary_bytes


@pytest.mark.parametrize(
    "failure_kind",
    [
        "executor_loss",
        "executor_close",
        "provider_unavailable",
        "provider_unavailable_after_analysis",
        "generic_provider_error",
        "model_contract_error",
    ],
)
def test_infrastructure_failed_primary_is_retried_once_and_recovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], failure_kind: str
) -> None:
    """Typed infrastructure faults retain their primary and recover through native grading/retry once."""
    from trialagentbench_verifier.campaign import TrialEvalCampaignAnalysisV1, TrialEvalCampaignPlanV1

    root, primary_path, primary_bytes = _run_faulted_primary(tmp_path, monkeypatch, failure_kind)
    if failure_kind.startswith("provider_unavailable"):
        saved = json.loads(primary_bytes)["agent_output"]
        assert saved["failure_type"] == "provider_unavailable"
        assert saved["provider_unavailability"]["request_event"]["exception_type"] == "ProviderUnavailableError"
        assert saved["provider_unavailability"]["request_event"]["response_received"] is False
        time.sleep(1.05)  # New command is eligible after the configured route cooldown.
    if failure_kind in {"generic_provider_error", "model_contract_error"}:
        assert json.loads(primary_bytes)["agent_output"]["failure_type"] == (
            "provider_error" if failure_kind == "generic_provider_error" else "contract_error"
        )
        with sqlite3.connect(root / "requests.sqlite") as connection:
            original_ledger = connection.execute("SELECT COUNT(*), SUM(charge) FROM attempts").fetchone()
        for _ in range(2):
            assert campaign.main(["retry", "--campaign", str(root)]) == 0
            assert len(campaign.load_campaign(root).runs) == 1
            assert primary_path.read_bytes() == primary_bytes
            with sqlite3.connect(root / "requests.sqlite") as connection:
                assert connection.execute("SELECT COUNT(*), SUM(charge) FROM attempts").fetchone() == original_ledger
        assert campaign.main(["finish", "--campaign", str(root)]) == 0
        analysis = TrialEvalCampaignAnalysisV1.model_validate_json(
            (root / "comparisons/trialeval/analysis.json").read_bytes()
        )
        assert analysis.ledger.recovered_primary_attempt_ids == ()
        assert analysis.ledger.primary_scientifically_unscored_attempt_ids == ()
        # The teaching provider already wrote valid estimates before its generic failure.
        # Native grading retains reproduced estimates and the failed primary's bytes;
        # it starts no second conversation.
        assert (
            getattr(
                analysis.ledger.outcomes,
                "completed" if failure_kind == "generic_provider_error" else "malformed_output",
            )
            == 1
        )
        return
    capsys.readouterr()
    assert campaign.main(["retry", "--campaign", str(root)]) == 0
    assert "python-trialeval-1: retrying 1 infrastructure-failed primaries" in capsys.readouterr().out
    plan = campaign.load_campaign(root)
    assert [run.retry_of_job_id for run in plan.runs] == [None, "python-trialeval-1"]
    with sqlite3.connect(root / "requests.sqlite") as connection:
        before_repeat = connection.execute("SELECT COUNT(*), SUM(charge) FROM attempts").fetchone()
    assert primary_path.read_bytes() == primary_bytes
    # A repeated retry resumes the existing retry run and never schedules a second one.
    assert campaign.main(["retry", "--campaign", str(root)]) == 0
    assert campaign.load_campaign(root) == plan
    with sqlite3.connect(root / "requests.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*), SUM(charge) FROM attempts").fetchone() == before_repeat
    assert primary_path.read_bytes() == primary_bytes

    assert campaign.main(["finish", "--campaign", str(root)]) == 0
    combined = TrialEvalCampaignPlanV1.model_validate_json((root / "comparisons/trialeval/plan.json").read_bytes())
    primary, retry = combined.attempts
    assert (primary.attempt_role, retry.attempt_role) == ("primary", "retry")
    assert retry.retry_of_attempt_id == primary.attempt_id and retry.request_replicate_id == "retry-1"
    analysis = TrialEvalCampaignAnalysisV1.model_validate_json(
        (root / "comparisons/trialeval/analysis.json").read_bytes()
    )
    assert analysis.scientific_analysis_status == "unavailable_due_insufficient_independent_worlds"
    assert analysis.unavailable_reason == "insufficient_independent_worlds"
    assert analysis.ledger.recovered_primary_attempt_ids == (primary.attempt_id,)
    assert analysis.ledger.primary_scientifically_unscored_attempt_ids == ()
    expected_fault = "provider_unavailable" if failure_kind.startswith("provider_unavailable") else "evaluator_error"
    assert (getattr(analysis.ledger.outcomes, expected_fault), analysis.ledger.outcomes.completed) == (1, 1)
    with pytest.raises(ValueError, match="must precede campaign finish"):
        campaign.retry_campaign(root)


def test_settled_turn_limit_with_teardown_close_fault_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A settled turn-limit primary survives cleanup, cannot retry and permits campaign finish."""
    root, primary_path, primary_bytes = _run_faulted_primary(tmp_path, monkeypatch, "executor_close_turn_limit")
    output = json.loads(primary_bytes)["agent_output"]
    assert output["failure_type"] == "turn_limit_no_submission"
    capsys.readouterr()
    for _ in range(2):
        assert campaign.main(["retry", "--campaign", str(root)]) == 0
        assert [run.retry_of_job_id for run in campaign.load_campaign(root).runs] == [None]
        assert primary_path.read_bytes() == primary_bytes
    text = capsys.readouterr().out
    assert "No TrialEval primary ended in an infrastructure failure; nothing to retry." in text
    assert "retrying 1 infrastructure-failed" not in text
    assert campaign.main(["finish", "--campaign", str(root)]) == 0
