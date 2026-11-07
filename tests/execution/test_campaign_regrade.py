"""Retained-output regrading preserves original custody and uses the native finalizer."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from trialagentbench_verifier.scorer import ScorerIdentityV1, current_scorer

from trialagentbench.contracts.campaign import CampaignConfigurationV1, CampaignPlanV1, CampaignRunV1
from trialagentbench.io import canonical_payload_sha256, sha256_file, write_json_model
from trialagentbench.tools import campaign, finish_campaign_analysis
from trialagentbench.tools import regrade_campaign as regrade_module
from trialagentbench.tools.regrade_campaign import CampaignRegradeV1, regrade_campaign


def _retained_campaign(root: Path, *, missing_scorer_identity: bool) -> CampaignPlanV1:
    root.mkdir()
    example = Path(__file__).resolve().parents[2] / "examples" / "smoke.json"
    payload = json.loads(example.read_bytes())
    payload.update(
        release_directory=str(root), expected_release_manifest_sha256="a" * 64, executor_image_id="sha256:" + "b" * 64
    )
    payload["panel"]["benchmarks"] = ["trialdev"]
    for name, model in zip(("first", "second"), payload["models"], strict=True):
        model["name"] = name
    config = CampaignConfigurationV1.model_validate(payload)
    write_json_model(root / "configuration.json", config)
    runs = []
    for model in config.models:
        native = root / "runs" / model.name
        native.mkdir(parents=True)
        (native / "invocation.json").write_text('{"frozen": "original invocation"}\n')
        (native / "submitted.py").write_text('print("retained agent output")\n')
        grade = root / "results" / model.name / "grade"
        grade.mkdir(parents=True)
        payload = (
            {} if missing_scorer_identity else {"scorer": {"version": "earlier-scorer", "sources_sha256": "a" * 64}}
        )
        (grade / "ranking.json").write_text(json.dumps(payload))
        runs.append(
            CampaignRunV1(
                job_id=model.name,
                benchmark="trialdev",
                model_name=model.name,
                repetition=1,
                relative_run_root=f"runs/{model.name}",
                invocation_sha256=sha256_file(native / "invocation.json"),
            )
        )
    plan = CampaignPlanV1(configuration=config, runs=tuple(runs))
    write_json_model(root / "campaign.json", plan)
    return plan


def _bytes(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("missing_scorer_identity", [False, True])
def test_regrade_snapshots_retained_outputs_and_records_previous_scorer(
    tmp_path, monkeypatch, missing_scorer_identity
):
    source, output = tmp_path / "original", tmp_path / "regraded"
    plan = _retained_campaign(source, missing_scorer_identity=missing_scorer_identity)
    original = _bytes(source)
    calls = []
    monkeypatch.setattr(finish_campaign_analysis, "campaign_censoring", lambda *_: None)
    monkeypatch.setattr(campaign, "run_campaign", lambda *_: pytest.fail("Regrade must never run agents"))

    def finish(root, **_):
        assert campaign.load_campaign(root) == plan
        assert not (root / "results").exists()
        for run in plan.runs:
            assert (root / run.relative_run_root / "submitted.py").read_bytes() == original[
                f"{run.relative_run_root}/submitted.py"
            ]
        calls.append(root)

    monkeypatch.setattr(campaign, "finish_campaign", finish)
    assert campaign.main(["regrade", "--campaign", str(source), "--output", str(output)]) == 0
    assert calls == [output]
    assert _bytes(source) == original
    for run in plan.runs:  # retained runs are shared, not copied
        retained = f"{run.relative_run_root}/submitted.py"
        assert (output / retained).stat().st_ino == (source / retained).stat().st_ino
    assert (output / "campaign.json").stat().st_ino != (source / "campaign.json").stat().st_ino
    receipt = CampaignRegradeV1.model_validate_json((output / "regrade.json").read_bytes())
    assert receipt.scorer == current_scorer()
    assert receipt.source_campaign_sha256 == sha256_file(source / "campaign.json")
    for run in receipt.runs:
        expected = (
            None if missing_scorer_identity else ScorerIdentityV1(version="earlier-scorer", sources_sha256="a" * 64)
        )
        assert run.previous_scorer == expected
        assert run.previous_grade_sha256 == sha256_file(source / str(run.previous_grade_path))
    with pytest.raises(FileExistsError):
        regrade_campaign(source, output)
    assert _bytes(source) == original


def test_regrade_carries_operator_exclusions_to_the_new_campaign(tmp_path, monkeypatch):
    source, output = tmp_path / "original", tmp_path / "regraded"
    _retained_campaign(source, missing_scorer_identity=False)
    (source / "operator-exclusions.json").write_text('{"tasks": {"first": ["task-a"]}}\n')
    monkeypatch.setattr(finish_campaign_analysis, "campaign_censoring", lambda *_: None)
    monkeypatch.setattr(campaign, "finish_campaign", lambda root, **_: None)
    regrade_campaign(source, output)
    assert (output / "operator-exclusions.json").read_bytes() == (source / "operator-exclusions.json").read_bytes()


@pytest.mark.parametrize("explicit", [False, True])
def test_regrade_preserves_derived_primary_selection_for_future_table_regrades(tmp_path, monkeypatch, explicit):
    from trialagentbench.tools.primary_selection import (
        PRIMARY_SELECTION,
        TrialEvalEarlierCompletionV1,
        TrialEvalPrimarySelectionV1,
    )

    source, output = tmp_path / "original", tmp_path / "regraded"
    _retained_campaign(source, missing_scorer_identity=False)
    selection = TrialEvalPrimarySelectionV1(
        audit_path=tmp_path / "origin-audit.json",
        audit_sha256="a" * 64,
        selections=(
            TrialEvalEarlierCompletionV1(
                target_job_id="first",
                task_id="TASK1",
                target_run_config_sha256="b" * 64,
                source_run_root=tmp_path / "earlier-native",
                source_run_config_sha256="c" * 64,
                source_item_sha256="d" * 64,
            ),
        ),
    )
    path = tmp_path / "selection.json" if explicit else source / PRIMARY_SELECTION
    write_json_model(path, selection)
    original = _bytes(source)
    monkeypatch.setattr(finish_campaign_analysis, "campaign_censoring", lambda *_: None)
    monkeypatch.setattr(campaign, "finish_campaign", lambda *_, **__: None)
    regrade_campaign(source, output, primary_selection=path if explicit else None)
    assert (output / PRIMARY_SELECTION).read_bytes() == path.read_bytes()
    receipt = CampaignRegradeV1.model_validate_json((output / "regrade.json").read_bytes())
    assert receipt.primary_selection_sha256 == sha256_file(path)
    assert _bytes(source) == original


@pytest.mark.parametrize("destination", ["original", "original/child", "."])
def test_regrade_refuses_overlapping_roots_before_writing(tmp_path, destination):
    source = tmp_path / "original"
    _retained_campaign(source, missing_scorer_identity=False)
    original = _bytes(source)
    with pytest.raises(ValueError, match="disjoint"):
        regrade_campaign(source, tmp_path / destination)
    assert _bytes(source) == original


def test_regrade_refuses_linked_custody_before_writing(tmp_path, monkeypatch):
    source, output = tmp_path / "original", tmp_path / "regraded"
    _retained_campaign(source, missing_scorer_identity=False)
    monkeypatch.setattr(finish_campaign_analysis, "campaign_censoring", lambda *_: None)
    (source / "runs" / "first" / "linked").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="symbolic link"):
        regrade_campaign(source, output)
    assert not output.exists()


def test_finishing_refuses_harness_source_drift_before_reading_scores(tmp_path, monkeypatch):
    from trialagentbench.util import source_identity

    monkeypatch.setattr(source_identity, "harness_scoring_source_sha256_v1", lambda: "0" * 64)
    with pytest.raises(RuntimeError, match="differ from the declared scorer"):
        campaign.finish_campaign(tmp_path)


def _release_sets(tmp_path, monkeypatch, *, participant="c", environment="e"):
    """Serve the original release ``a…`` and a rebuilt one whose toolkit always differs."""
    original, rebuilt = tmp_path / "original", tmp_path / "rebuilt"
    rebuilt.mkdir()
    (rebuilt / "release-set-manifest.json.sha256").write_text("d" * 64 + "  release-set-manifest.json\n")

    def manifest(toolkit, catalog, executor):
        artifacts = (
            ("trialagentbench-toolkit-v1.zip", toolkit),
            ("trialdev-participant.assets.json", catalog),
            ("trialdev-evaluator.assets.json", "c" * 64),
        )
        return SimpleNamespace(
            artifacts=tuple(SimpleNamespace(filename=name, sha256=digest) for name, digest in artifacts),
            executor=SimpleNamespace(environment_sha256=executor * 64),
        )

    manifests = {
        (original, "a" * 64): manifest("1" * 64, "c" * 64, "e"),
        (rebuilt, "d" * 64): manifest("2" * 64, participant * 64, environment),
    }
    monkeypatch.setattr(
        regrade_module,
        "authenticate_release_set",
        lambda root, *, expected_manifest_sha256: manifests[(Path(root), expected_manifest_sha256)],
    )
    monkeypatch.setattr(finish_campaign_analysis, "campaign_censoring", lambda *_: None)
    return original, rebuilt


def test_regrade_rebinds_runs_to_a_release_serving_identical_inputs(tmp_path, monkeypatch):
    source, rebuilt = _release_sets(tmp_path, monkeypatch)
    output = tmp_path / "regraded"
    plan = _retained_campaign(source, missing_scorer_identity=False)
    with sqlite3.connect(source / "requests.sqlite") as ledger:
        ledger.execute("CREATE TABLE identity (digest TEXT PRIMARY KEY)")
        ledger.execute(
            "INSERT INTO identity VALUES (?)", (canonical_payload_sha256(plan.configuration.model_dump(mode="json")),)
        )
    original = _bytes(source)
    finished = []
    monkeypatch.setattr(campaign, "finish_campaign", lambda root, **_: finished.append(campaign.load_campaign(root)))
    assert (
        campaign.main(["regrade", "--campaign", str(source), "--output", str(output), "--release", str(rebuilt)]) == 0
    )
    (rebound,) = finished
    assert rebound.configuration.release_directory == rebuilt
    assert rebound.configuration.expected_release_manifest_sha256 == "d" * 64
    assert rebound.configuration.executor_image_id == plan.configuration.executor_image_id
    assert rebound.runs == plan.runs and rebound.checksum != plan.checksum
    with sqlite3.connect(output / "requests.sqlite") as ledger:
        assert ledger.execute("SELECT digest FROM identity").fetchall() == [
            (canonical_payload_sha256(rebound.configuration.model_dump(mode="json")),)
        ]
    receipt = CampaignRegradeV1.model_validate_json((output / "regrade.json").read_bytes())
    assert receipt.source_campaign_sha256 == sha256_file(source / "campaign.json")
    assert _bytes(source) == original


@pytest.mark.parametrize(
    ("difference", "message"),
    [({"participant": "f"}, "identical campaign inputs"), ({"environment": "f"}, "executor environment")],
)
def test_regrade_refuses_rebinding_to_other_inputs_before_writing(tmp_path, monkeypatch, difference, message):
    source, rebuilt = _release_sets(tmp_path, monkeypatch, **difference)
    _retained_campaign(source, missing_scorer_identity=False)
    original = _bytes(source)
    with pytest.raises(ValueError, match=message):
        regrade_campaign(source, tmp_path / "regraded", release=rebuilt)
    assert not (tmp_path / "regraded").exists()
    assert _bytes(source) == original
