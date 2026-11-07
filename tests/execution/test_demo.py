"""First-run teaching assets execute and replay through the public native owners."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

from trialagentbench.contracts.release.archive_safety import materialized_release_archive
from trialagentbench.contracts.trialdev.pools import TrialDevFrozenPool
from trialagentbench.io import sha256_file
from trialagentbench.tools.demo import demo_assets
from trialagentbench.trialdev.release import load_trialdev_evaluator_release


def test_demo_assets_are_bounded_and_external_adapter_is_available() -> None:
    """The offline installed surface contains a bounded corpus its runtime can load."""
    root = demo_assets()
    archives = tuple(root.glob("demo-*-evaluator.zip"))
    assert len(archives) == 2
    assert sum(path.stat().st_size for path in archives) < 4_000_000
    assert (root / "run_external_provider.py").is_file()
    assert (root / "observed_analysis.py").is_file()
    with ZipFile(root / "demo-trialdev-evaluator.zip") as archive:
        pools = [name for name in archive.namelist() if name.endswith("/pool/manifest.json")]
        assert pools
        for name in pools:
            TrialDevFrozenPool.model_validate_json(archive.read(name))
        tasks = [
            json.loads(archive.read(name))
            for name in archive.namelist()
            if name.startswith("participant/tasks/") and name.endswith("/task.json")
        ]
        assert {(task["stream"], task["data_view"]) for task in tasks} == {
            (stream, view)
            for stream in ("evidence_analysis", "portfolio_management")
            for view in ("raw", "analysis_ready")
        }
        for view in ("raw", "analysis_ready"):
            budgets = {
                task["resources"]["budget_usd"]
                for task in tasks
                if task["stream"] == "portfolio_management" and task["data_view"] == view
            }
            assert len(budgets) == 3


def test_trialdev_demo_uses_current_frozen_pool_read_contract() -> None:
    """The shipped teaching pool must prepare under the current consumer contract."""
    archive = demo_assets() / "demo-trialdev-evaluator.zip"
    with materialized_release_archive(archive, expected_sha256=sha256_file(archive)) as materialized:
        release = load_trialdev_evaluator_release(materialized.root)
        task = next(task for task in release.participant.tasks if task.stream.value == "portfolio_management")
        assert release.authored_configuration(task) is not None


@pytest.mark.executor
def test_demo_native_reporting_acquisition_and_completed_resume(tmp_path: Path) -> None:
    """Run without provider SDK imports, then resume without another provider turn."""
    output = tmp_path / "teaching"
    driver = """
import importlib.abc
import sys
from pathlib import Path
class NoProviderSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'openai', 'httpx2', 'agents'}:
            raise AssertionError('Demo imported a provider SDK: ' + fullname)
sys.meta_path.insert(0, NoProviderSDK())
from trialagentbench.tools.demo import ScriptedDemoProvider, run_demo
class CompletedProvider(ScriptedDemoProvider):
    def generate_turn(self, *args, **kwargs):
        raise AssertionError('Completed resume requested another provider turn')
resume = len(sys.argv) > 2
run_demo(output=Path(sys.argv[1]), provider=CompletedProvider() if resume else ScriptedDemoProvider(),
         executor_image=None, resume=resume)
"""
    completed = subprocess.run(
        [sys.executable, "-c", driver, str(output)], capture_output=True, text=True, timeout=180, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    te = json.loads((output / "trialeval/replay.json").read_text())
    assert te["status"] == "reproduced"
    task = next((output / "trialdev").iterdir())
    programme = json.loads((task / "programme_run.json").read_text())
    assert programme["execution_status"] == "completed"
    assert len(programme["evaluation"]["studies"]) == 1
    assert programme["evaluation"]["initial_replay"]["status"] == "reproduced"
    assert programme["evaluation"]["studies"][0]["replay"]["status"] == "reproduced"
    estimates = list(csv.DictReader((task / "workspace/scratch/estimates.csv").read_text().splitlines()))
    assert len(estimates) == 2
    assert any(float(row["estimate"]) != 0 for row in estimates)
    assert all(float(row["lower"]) < float(row["upper"]) for row in estimates)
    events = [json.loads(line) for line in (task / "events.jsonl").read_text().splitlines()]
    assert any((event.get("cpu_seconds") or 0) > 0 for event in events)
    transcript = (task / "conversation.json").read_bytes()
    ledger = (task / "private/budget.json").read_bytes()
    resumed = subprocess.run(
        [sys.executable, "-c", driver, str(output), "resume"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert transcript == (task / "conversation.json").read_bytes()
    assert ledger == (task / "private/budget.json").read_bytes()
    report = (output / "index.html").read_text()
    grade = json.loads((output / "trialeval/grade-0.json").read_text())
    assert f'{grade["signal_loss"]:.4g}' in report
    assert f'{grade["point_performance"]["oracle_value"]:.4g}'.replace("-", "−") in report
    value = programme["evaluation"]["value"]["net_value_usd"]
    assert f"{'−' if value < 0 else ''}${abs(value):,.0f}" in report
    assert "365 days" in report and "$-" not in report
    assert "Development decision driven by the answer" in report
    assert "Regret at the true effect" in report
    from trialagentbench_verifier.decision_consequence import (
        DecisionEconomicsV1,
        DecisionReportV1,
        decision_consequence_v1,
    )
    from trialagentbench_verifier.layer_priors import load_layer_prior

    from trialagentbench.contracts.release.archive_safety import materialized_release_archive
    from trialagentbench.io import sha256_file
    from trialagentbench.release import load_trialeval_grading_key
    from trialagentbench.trialeval.data import discover_participant_items

    archive = demo_assets() / "demo-trialeval-evaluator.zip"
    with materialized_release_archive(archive, expected_sha256=sha256_file(archive)) as materialized:
        item = next(
            row
            for row in discover_participant_items(materialized.root / "participant").values()
            if row.context_tier == "C1"
        )
        target = (
            load_trialeval_grading_key(materialized.root, expected_task_ids=(item.item_id,))
            .for_task(item.item_id)
            .target
        )
    reference = target.consequence
    assert reference is not None
    expected = decision_consequence_v1(
        DecisionReportV1(
            estimate=grade["point_performance"]["submitted_value"],
            lower=grade["interval_performance"]["native_lower"],
            upper=grade["interval_performance"]["native_upper"],
        ),
        tuple(
            DecisionReportV1(estimate=estimate, lower=lower, upper=upper)
            for estimate, lower, upper in reference.credited
        ),
        truth=target.value,
        scale=target.reference_sampling_standard_deviation,
        participants=reference.participants,
        favorable_direction=target.binding.favorable_direction,
        prior=load_layer_prior(target.binding.estimand_id),
        economics=DecisionEconomicsV1(
            value_of_success_musd=reference.economics.value_of_success_musd,
            replicate_cost_musd=reference.economics.replicate_cost_musd,
            fixed_trial_cost_musd=reference.economics.fixed_trial_cost_musd,
        ),
    )
    assert f"${expected.regret_musd * 1_000_000:,.0f}" in report
    assert ("Run a confirmatory trial" if expected.agent.confirm else "Stop development") in report
    assert "Agresti–Caffo" in report
    assert 'tabindex="0" role="region" aria-label=' in report
    assert "max-width:75ch" in report
    assert "Compare its clinical estimates with the world-derived targets" in report
    assert "<summary>Execution details</summary>" in report
    assert "DAS28 points" in report and "risk proportions" in report and "mg/g" in report
    assert "Download every observed row as CSV" in report
    for relative in re.findall(r'href="([^"]+)"', report):
        assert (output / relative).is_file()


@pytest.mark.executor
def test_demo_preserves_interrupted_trialeval_work_before_restart(tmp_path: Path, capsys) -> None:
    """A restarted TrialEval attempt retains its original transcript, usage and workspace."""
    from trialagentbench.tools.demo import ScriptedDemoProvider, run_demo

    class InterruptedProvider(ScriptedDemoProvider):
        def generate_turn(self, messages, tools=None, **kwargs):
            if any(message.get("tool_calls") for message in messages):
                raise RuntimeError("interrupted after the first saved analysis")
            return super().generate_turn(messages, tools, **kwargs)

    output = tmp_path / "demo"
    with pytest.raises(RuntimeError, match="interrupted after"):
        run_demo(output=output, provider=InterruptedProvider(), executor_image=None)
    partial = output / "trialeval"
    before = {path.relative_to(partial): path.read_bytes() for path in partial.rglob("*") if path.is_file()}
    assert json.loads(before[Path("conversation.json")])[-1]["role"] == "tool"
    assert b'"status": "succeeded"' in before[Path("provider_responses.jsonl")]
    run_demo(output=output, provider=ScriptedDemoProvider(), executor_image=None, resume=True)
    archived = output / "failed_attempts/trialeval/attempt-1/trialeval.archived"
    assert {path.relative_to(archived): path.read_bytes() for path in archived.rglob("*") if path.is_file()} == before
    assert "Restarting TrialEval from its initial evidence" in capsys.readouterr().out
    report = (output / "index.html").read_text()
    assert "Interrupted TrialEval attempts: 1" in report
    assert "Include their provider charges" in report
    for relative in re.findall(r'href="([^"]+)"', report):
        assert (output / relative).is_file()


def test_executor_r_helpers_ship_the_repository_license():
    """The installed R helper and public source use the same Apache license."""
    root = Path(__file__).resolve().parents[2]
    package = root / "executor/r-package"
    assert "License: Apache License (== 2.0) | file LICENSE" in (package / "DESCRIPTION").read_text()
    assert (package / "LICENSE").read_bytes() == (root / "LICENSE").read_bytes()


def test_bundled_trialdev_pools_use_current_strict_acquisition_provenance():
    """Every teaching pool uses the unchanged v8 reader and native inventory validation."""
    from zipfile import ZipFile

    from trialagentbench.contracts.trialdev.pools import TrialDevFrozenPool

    pools = 0
    for path in demo_assets().glob("*.zip"):
        with ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.endswith("/pool/manifest.json"):
                    continue
                payload = json.loads(archive.read(name))
                pool = TrialDevFrozenPool.model_validate(payload)
                pools += 1
                assert pool.schema_id == "trialdev.frozen-pool/v8"
                assert re.fullmatch(r"[0-9a-f]{64}", pool.source_provenance_sha256)
                assert pool.participant_count == 2048
                assert pool.targets
                assert {row.participant_count for row in pool.targets} == {2048}
                assert pool.artifacts
                for key in ("clinical_source_sha256", "construction_source_sha256", "foundation_source_sha256"):
                    assert key not in payload
                    with pytest.raises(ValueError):
                        TrialDevFrozenPool.model_validate(payload | {key: "a" * 64})
                with pytest.raises(ValueError):
                    TrialDevFrozenPool.model_validate(payload | {"schema_id": "trialdev.frozen-pool/v7"})
                with pytest.raises(ValueError):
                    TrialDevFrozenPool.model_validate(
                        {key: value for key, value in payload.items() if key != "source_provenance_sha256"}
                    )
    assert pools > 0
