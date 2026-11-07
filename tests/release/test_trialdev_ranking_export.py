"""Portable full-custody TrialDev ranking preserves a prospectively empty run."""

import hashlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from trialagentbench_verifier.development import ranking_means, verify_trialdev_ranking
from trialagentbench_verifier.grading_key import (
    GRADING_KEY_MANIFEST,
    TRIALDEV_WORLD_STRATA,
    GradingKeyArtifactV1,
    GradingKeyManifestV1,
    TrialDevWorldStrataV1,
    TrialDevWorldStratumV1,
    grading_key_manifest_bytes,
    trialdev_world_strata_bytes,
)
from trialagentbench_verifier.response import TrialDevRankingItemV1, TrialDevRunRankingV1
from trialagentbench_verifier.scorer import current_scorer

from trialagentbench.contracts.core.config import ExperimentConditionV1
from trialagentbench.contracts.trialdev.release import TrialDevStream
from trialagentbench.contracts.trialdev.run import TrialDevRunInvocationV1
from trialagentbench.io import sha256_file, write_json_model
from trialagentbench.ports import CodeExecutionLimitsV1
from trialagentbench.tools.export_results import export_results_v1
from trialagentbench.trialdev.ranking import rank_trialdev_run_v1


@pytest.mark.parametrize("unfinished", [False, True])
def test_full_archive_preserves_unstarted_authored_task_denominators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unfinished: bool
) -> None:
    """Exercise real ZIP/export/custody/ranking; release qualification alone is stubbed."""
    release = tmp_path / "release"
    release.mkdir()
    release_id = "release_" + "a" * 64
    (release / "release_manifest.json").write_text(
        json.dumps(
            {
                "schema_id": "ariadne.trialdev.release/v1",
                "release_id": release_id,
            }
        )
    )
    digest = sha256_file(release / "release_manifest.json")
    monkeypatch.setattr(
        "trialagentbench.tools.export_results.load_trialdev_evaluator_release",
        lambda _: SimpleNamespace(manifest=SimpleNamespace(release_id=release_id, checksum="b" * 64)),
    )
    monkeypatch.setattr(
        "trialagentbench_verifier.development._release_identity",
        lambda *args, **kwargs: ("trialdev", "c" * 64, digest),
    )
    task_ids = ("td-0123456789ab", "td-123456789abc")
    world_id = "tdw-0123456789ab"
    scenario_ref = "scenario_" + "1" * 20
    strata = trialdev_world_strata_bytes(
        TrialDevWorldStrataV1(
            schema_id="trialagentbench.trialdev_world_strata/v1",
            release_id=release_id,
            worlds=(
                TrialDevWorldStratumV1(
                    world_id=world_id,
                    clinical_profile_id="rheumatoid_arthritis",
                    candidate_count=3,
                    scenario_ref=scenario_ref,
                ),
            ),
        )
    )
    (release / "grading_key").mkdir()
    (release / TRIALDEV_WORLD_STRATA).write_bytes(strata)
    (release / GRADING_KEY_MANIFEST).write_bytes(
        grading_key_manifest_bytes(
            GradingKeyManifestV1.create(
                schema_id="trialagentbench.trialdev_grading_key/v1",
                release_id=release_id,
                release_manifest_sha256=digest,
                derivation_id="test",
                task_count=len(task_ids),
                world_count=1,
                artifacts=(
                    GradingKeyArtifactV1(
                        relative_path=TRIALDEV_WORLD_STRATA,
                        sha256=hashlib.sha256(strata).hexdigest(),
                        byte_length=len(strata),
                    ),
                ),
            )
        )
    )
    initial = {
        "context": {
            "checkpoint_id": "initial",
            "target_ids": ["candidate:clinical_outcome", "candidate:safety_risk"],
            "evidence_artifacts": [],
        },
        "targets": [
            {
                "asset_id": "candidate",
                "endpoint": endpoint,
                "unit": unit,
                "decision_scale": 1,
                "comparison_sample_size": 100,
                "point_identified": True,
                "oracle_value": 0,
                "estimand": "standardized_observed_association",
            }
            for endpoint, unit in (("clinical_outcome", "DAS28"), ("safety_risk", "risk difference"))
        ],
    }
    for task_id in task_ids:
        root = release / "participant/tasks" / task_id
        root.mkdir(parents=True)
        (root / "task.json").write_text(
            json.dumps(
                {
                    "task_id": task_id,
                    "case_id": world_id,
                    "stream": "portfolio_management",
                    "available_materials": [],
                    "data_view": "raw",
                    "resources": {"budget_usd": 15_000_000},
                }
            )
        )
        configuration = release / "evaluation/tasks" / task_id / "authored_programme.json"
        configuration.parent.mkdir(parents=True)
        configuration.write_text(
            json.dumps(
                {
                    "schema_id": "trialdev.authored-programme/v1",
                    "pool_manifest_sha256": "a" * 64,
                    "budget_usd": 15_000_000,
                    "preferences": {"outcome_unit": "DAS28", "benefit_unit": 1},
                    "initial_evidence": initial,
                }
            )
        )
    invocation = TrialDevRunInvocationV1(
        release_id=release_id,
        release_archive_sha256="d" * 64,
        release_manifest_sha256=digest,
        scheduled_task_ids=task_ids,
        scheduled_task_streams={task_id: "portfolio_management" for task_id in task_ids},
        model="test",
        provider="openai",
        experiment_condition=ExperimentConditionV1(
            condition_id="primary", request_replicate_id="request-1", maximum_turns_per_step=7
        ),
        send_temperature=True,
        maximum_completion_tokens=1000,
        maximum_context_characters=50000,
        watchdog_seconds=120,
        request_timeout_seconds=60,
        executor_image="sha256:" + "e" * 64,
        executor_limits=CodeExecutionLimitsV1(),
    )
    run, grade, verification = (tmp_path / name for name in ("run", "grade", "verification"))
    for root in (run, grade):
        write_json_model(root / "invocation.json", invocation)
        if unfinished:
            pending = root / "tasks" / task_ids[0]
            pending.mkdir(parents=True)
            (pending / "events.jsonl").write_text("unsealed interrupted output\n")
    items = tuple(
        TrialDevRankingItemV1(
            task_id=task_id,
            world_id=world_id,
            stream="portfolio_management",
            data_view="raw",
            clinical_profile_id="rheumatoid_arthritis",
            scenario_profile_id=scenario_ref,
            asset_count=3,
            execution_status="incomplete_run" if unfinished and task_id == task_ids[0] else "not_started",
            run_checksum=None,
            observational_loss=None,
            initial_target_statuses={
                identity: "incomplete_run" if unfinished and task_id == task_ids[0] else "not_started"
                for identity in initial["context"]["target_ids"]
            },
            authored_programme=True,
            net_development_value_usd=None,
            budget_usd=15_000_000,
        )
        for task_id in task_ids
    )
    ranking = TrialDevRunRankingV1(
        scorer=current_scorer(),
        release_id=release_id,
        release_manifest_sha256=digest,
        invocation_checksum=invocation.checksum,
        model_id="test",
        scope="complete_release_task_inventory",
        items=items,
        profile_means=ranking_means(list(items)),
    )
    projected_release = SimpleNamespace(
        world_stratum=TrialDevWorldStrataV1.model_validate_json(strata).stratum,
        root=release,
        manifest=SimpleNamespace(release_id=release_id, task_count=len(task_ids)),
        participant=SimpleNamespace(
            release_manifest_sha256=digest,
            tasks=tuple(
                SimpleNamespace(
                    task_id=task_id,
                    case_id=world_id,
                    stream=TrialDevStream.PORTFOLIO_MANAGEMENT,
                    data_view="raw",
                    resources=SimpleNamespace(budget_usd=15_000_000),
                )
                for task_id in task_ids
            ),
        ),
        analysis_configuration=lambda task: None,
        authored_configuration=lambda task: SimpleNamespace(
            initial_evidence=SimpleNamespace(
                context=SimpleNamespace(target_ids=tuple(initial["context"]["target_ids"]))
            )
        ),
    )
    assert rank_trialdev_run_v1(release=projected_release, run_root=run, task_roots=()) == ranking
    projected_release.manifest.task_count = len(task_ids) + 1
    assert rank_trialdev_run_v1(release=projected_release, run_root=run, task_roots=()).scope == "scheduled_panel"
    projected_release.manifest.task_count = len(task_ids)
    write_json_model(grade / "ranking.json", ranking)
    verification.mkdir()
    (verification / "report.json").write_text(
        json.dumps(
            {
                "schema_id": "trialagentbench.verification_report/v1",
                "release_digest": digest,
            }
        )
    )
    archive = tmp_path / "release.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in release.rglob("*"):
            if path.is_file():
                handle.write(path, path.relative_to(release).as_posix())
    exported = export_results_v1(
        release_root=release,
        output_dir=tmp_path / "export",
        run_roots=(run,),
        grade_roots=(grade,),
        verification_roots=(verification,),
    )
    result_zip = next(exported.glob("*.zip"))
    arguments = dict(
        expected_release_sha256=sha256_file(archive),
        expected_result_export_sha256=sha256_file(result_zip),
        expected_invocation_sha256=sha256_file(run / "invocation.json"),
    )
    report = verify_trialdev_ranking(archive, result_zip, **arguments)
    assert report.status == "pass"
    assert report.input_sha256 == sha256_file(result_zip)
    unanchored = verify_trialdev_ranking(archive, result_zip, **{**arguments, "expected_result_export_sha256": None})
    assert unanchored.status == "pass" and unanchored.input_sha256 == sha256_file(result_zip)
    assert unanchored.expected_input_sha256 is None
    for field in ("expected_release_sha256", "expected_result_export_sha256"):
        with pytest.raises(ValueError, match="externally trusted SHA-256"):
            verify_trialdev_ranking(archive, result_zip, **{**arguments, field: "f" * 64})
    computed = TrialDevRunRankingV1.model_validate(report.checks[0].observed)
    value = next(row for row in computed.profile_means if row.metric == "net_development_value_usd")
    assert (value.mean, value.scheduled_tasks, value.scheduled_worlds) == (None, 2, 1)
    assert len(computed.items) == 2
    recovery = next(row for row in computed.profile_means if row.metric == "observational_loss")
    assert recovery.target_status_counts == (
        {"incomplete_run": 2, "not_started": 2} if unfinished else {"not_started": 4}
    )
    assert all(row.acquired_target_statuses is None for row in computed.items)
    assert computed.items[0].execution_status == ("incomplete_run" if unfinished else "not_started")
    if unfinished:
        assert computed.items[0].observational_loss is None
        assert all(row.mean is None for row in computed.profile_means)
    with pytest.raises(ValueError, match="prospectively trusted invocation"):
        verify_trialdev_ranking(archive, result_zip, **{**arguments, "expected_invocation_sha256": "f" * 64})
