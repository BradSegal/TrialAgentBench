"""Campaign preparation and finishing carry publisher trust to independent analysis."""

from __future__ import annotations

import json
import shutil
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from grading.test_canonical_grade_contract import (
    _evaluator_record,
    _numerical_replay,
    _run_config_payload,
)
from trialagentbench_test_helpers import write_trialeval_grading_key
from trialagentbench_verifier.campaign import TrialEvalCampaignAnalysisV1, load_trial_eval_campaign_grade_source_v1
from trialagentbench_verifier.io import sha256_file
from trialagentbench_verifier.release_set import (
    ExecutorTrustAnchorV1,
    ReleaseSetArtifactV1,
    ReleaseSetManifestV1,
    SoftwareCandidateReference,
    _base_release_sha256,
    _canonical_json,
    authenticate_release_set_manifest,
)
from trialagentbench_verifier.trialeval import TrialEvalDecisionRuleV1

from trialagentbench.contracts.campaign import CampaignConfigurationV1, CampaignRunV1
from trialagentbench.contracts.core.runs import TrialEvalItemResultV1, TrialEvalRunConfigV1
from trialagentbench.contracts.release.identity import PairedParticipantArchiveIdentityV1
from trialagentbench.io import write_json_model
from trialagentbench.tools import campaign, finish_campaign_analysis, finish_run
from trialagentbench.tools.grade import grade_trialeval


def _configuration(root: Path) -> CampaignConfigurationV1:
    payload = json.loads((Path(__file__).resolve().parents[2] / "examples/smoke.json").read_bytes())
    payload.update(
        release_directory=str(root),
        expected_release_manifest_sha256="a" * 64,
        executor_image_id="sha256:" + "e" * 64,
        grading_key=str(root / "records.jsonl"),
        grading_key_sha256="d" * 64,
    )
    payload["models"] = payload["models"][:1]
    payload["panel"] = {"benchmarks": ["trialeval"]}
    return CampaignConfigurationV1.model_validate(payload)


def _anchor(root: Path, files: tuple[Path, ...]) -> dict[str, object]:
    sums = root / "publisher-SHA256SUMS"
    sums.write_text("".join(f"{sha256_file(path)}  {path.relative_to(root)}\n" for path in files))
    return dict(sums=sums, anchor_sha256=sha256_file(sums))


@pytest.mark.parametrize("anchor_kind", ("sha256", "url"))
def test_prepare_development_campaign_verifies_anchored_evaluator(
    tmp_path: Path, monkeypatch, anchor_kind: str
) -> None:
    """Run real development authentication and archive verification before freezing a plan."""
    from trialagentbench_verifier.artifacts import anchor
    from trialagentbench_verifier.campaign_inputs import (
        DevelopmentArchiveV1,
        DevelopmentCampaignInputsV1,
        DevelopmentExecutorV1,
    )

    from trialagentbench.tools.demo import demo_assets

    archive = tmp_path / "trialdev-evaluator.zip"
    shutil.copyfile(demo_assets() / "demo-trialdev-evaluator.zip", archive)
    config = _configuration(tmp_path)
    manifest = DevelopmentCampaignInputsV1(
        artifacts=(
            DevelopmentArchiveV1(
                filename=archive.name, sha256=sha256_file(archive), byte_length=archive.stat().st_size
            ),
        ),
        executor=DevelopmentExecutorV1(image_id=config.executor_image_id, environment_sha256="f" * 64),
    )
    manifest_path = tmp_path / "development-inputs.json"
    manifest_path.write_text(manifest.model_dump_json())
    trust = _anchor(tmp_path, (archive,))
    if anchor_kind == "url":
        declaration = json.dumps(
            dict(
                schema_version="trialagentbench.release_dataset/v1",
                repository="BradSegal/TrialAgentBench",
                revision="v1.1.0",
                payload_sha256="a" * 64,
                sha256sums_sha256=trust["anchor_sha256"],
            )
        ).encode()
        monkeypatch.setattr(anchor, "urlopen", lambda *_args, **_kwargs: BytesIO(declaration))
        trust.update(
            anchor_sha256=None,
            anchor_url="https://github.com/BradSegal/TrialAgentBench/blob/main/.github/release-dataset.json@"
            + "b" * 40,
        )
    config = config.model_copy(
        update=dict(
            input_mode="development",
            grading_key=None,
            grading_key_sha256=None,
            expected_release_manifest_sha256=sha256_file(manifest_path),
            panel=config.panel.model_copy(update={"benchmarks": ("trialdev",)}),
            **trust,
        )
    )
    monkeypatch.setattr(campaign, "_retain_campaign_executor", lambda *_: None)
    monkeypatch.setattr(
        campaign,
        "_prepare_native_run",
        lambda *_args, **kwargs: CampaignRunV1(
            job_id=kwargs["job_id"],
            benchmark="trialdev",
            model_name=config.models[0].name,
            repetition=1,
            relative_run_root="runs/fixture",
            invocation_sha256="a" * 64,
        ),
    )
    plan = campaign.prepare_campaign(config, tmp_path / "campaign")
    assert plan.configuration == config and len(plan.runs) == 1
    assert (tmp_path / "campaign/campaign.json").is_file()


def _qualified_configuration(
    tmp_path: Path, *, key: Path | None = None, evaluator_sha256: str = "b" * 64
) -> tuple[CampaignConfigurationV1, ReleaseSetManifestV1]:
    """Write a pinned manifest and key while leaving scientific checks to their owned suites."""
    config = _configuration(tmp_path)
    artifacts = tuple(
        sorted(
            (
                ReleaseSetArtifactV1(
                    filename="trialagentbench-toolkit-v1.zip",
                    sha256="a" * 64,
                    byte_length=1,
                    kind="public_toolkit",
                    audience="public",
                ),
                *(
                    ReleaseSetArtifactV1(
                        filename=f"{benchmark}-{role}.assets.json",
                        sha256=(evaluator_sha256 if (benchmark, role) == ("trialeval", "evaluator") else "b" * 64),
                        byte_length=1,
                        kind="controlled_role",
                        audience=f"{benchmark}_{role}",
                        benchmark_stream=benchmark,
                    )
                    for benchmark in ("trialeval", "trialdev")
                    for role in ("participant", "evaluator")
                ),
            ),
            key=lambda row: row.filename,
        )
    )
    software = SoftwareCandidateReference(
        candidate_sha256="a" * 64,
        source_sha256="b" * 64,
        wheelhouse_sha256="c" * 64,
        source_distributions_sha256="d" * 64,
    )
    executor = ExecutorTrustAnchorV1(
        image_reference="fixture",
        image_id=config.executor_image_id,
        environment_sha256="e" * 64,
        context_sha256="f" * 64,
    )
    content = dict(
        schema_version="trialagentbench.release_set/v2",
        software_candidate=software.model_dump(mode="json"),
        executor=executor.model_dump(mode="json"),
        base_release_sha256=_base_release_sha256(software, executor, artifacts),
        reproducibility_status="not_yet_available",
        artifacts=[row.model_dump(mode="json") for row in artifacts],
    )
    import hashlib

    manifest = ReleaseSetManifestV1.model_validate(
        dict(content, release_set_id="release_set_" + hashlib.sha256(_canonical_json(content)).hexdigest())
    )
    manifest_path = tmp_path / "release-set-manifest.json"
    manifest_path.write_text(manifest.model_dump_json())
    if key is None:
        key = tmp_path / "records.jsonl"
        key.write_text(_evaluator_record().model_dump_json() + "\n")
    config = config.model_copy(
        update=dict(
            expected_release_manifest_sha256=sha256_file(manifest_path),
            grading_key=key,
            grading_key_sha256=sha256_file(key),
            **_anchor(tmp_path, (manifest_path, key)),
        )
    )
    return config, manifest


def test_prepare_campaign_authenticates_configured_anchor_and_key(tmp_path: Path, monkeypatch) -> None:
    """Reject an unfinishable TrialEval schedule before freezing any native invocation."""
    config, _ = _qualified_configuration(tmp_path)
    key = config.grading_key
    assert key is not None
    editable = tmp_path / "editable.json"
    payload = config.model_dump(mode="json")
    payload.update(sums=config.sums.name, grading_key=key.name, release_directory=".")
    editable.write_text(json.dumps(payload))
    assert campaign.load_configuration(editable) == config
    calls = []

    def verify(root, **kwargs):
        calls.append(kwargs)
        # The expensive scientific/toolkit checks have their own suites; authentication stays real.
        return authenticate_release_set_manifest(
            root,
            **{name: kwargs[name] for name in ("expected_manifest_sha256", "anchor_sha256", "anchor_url", "sums")},
        )

    monkeypatch.setattr(campaign, "verify_release_set", verify)
    monkeypatch.setattr(campaign, "_campaign_asset_selections", lambda *_: {})
    monkeypatch.setattr(campaign, "preview_panel", lambda *_: None)
    monkeypatch.setattr(campaign, "release_assets", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(campaign, "_retain_campaign_executor", lambda *_: None)
    monkeypatch.setattr(
        campaign,
        "_prepare_native_run",
        lambda *_args, **kwargs: CampaignRunV1(
            job_id=kwargs["job_id"],
            benchmark="trialeval",
            model_name=config.models[0].name,
            repetition=1,
            relative_run_root="runs/fixture",
            invocation_sha256="a" * 64,
            analysis_plan_path="plans/fixture.json",
            analysis_plan_sha256="b" * 64,
        ),
    )
    missing_key = dict(payload)
    missing_key.pop("grading_key")
    missing_key.pop("grading_key_sha256")
    editable.write_text(json.dumps(missing_key))
    with pytest.raises(ValueError, match="TrialEval.*authenticated grading key"):
        campaign.main(["prepare", "--config", str(editable), "--campaign", str(tmp_path / "missing-key")])
    assert not (tmp_path / "missing-key").exists()
    assert calls == []
    editable.write_text(json.dumps(payload))
    plan = campaign.prepare_campaign(campaign.load_configuration(editable), tmp_path / "campaign")
    assert plan.configuration == config and len(calls) == 1
    assert calls[0]["grading_key"] == key
    assert calls[0]["grading_key_sha256"] == sha256_file(key)
    assert json.loads((tmp_path / "campaign/configuration.json").read_text())["grading_key"] == str(key)
    key.write_text(key.read_text() + " ")
    with pytest.raises(ValueError, match="trusted expected SHA-256"):
        campaign.prepare_campaign(config, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()


def test_native_producer_and_both_finalizers_use_authenticated_key(tmp_path: Path, monkeypatch) -> None:
    """A prepared keyed schedule reaches real native grading and both independent analysis callers."""
    suite = tmp_path / "suite"
    participant = suite / "participant"
    participant.mkdir(parents=True)
    record = _evaluator_record().model_copy(
        update={"decision": TrialEvalDecisionRuleV1(kind="superiority", margin=0.0, reference_decisions=((True,),))}
    )
    (suite / "release_manifest.json").write_text(json.dumps({"release_id": record.release_id}))
    write_json_model(
        suite / "participant_archive_identity.json",
        PairedParticipantArchiveIdentityV1.create(
            release_id=record.release_id, participant_archive_sha256="a" * 64, participant_manifest_checksum="f" * 64
        ),
    )
    write_trialeval_grading_key(suite, (record,))
    key = suite / "grading_key/records.jsonl"
    native = tmp_path / "native"
    native_config = TrialEvalRunConfigV1.model_validate(_run_config_payload(task_id=record.task_id))
    write_json_model(native / "run_config.json", native_config)
    write_json_model(
        native / "items" / f"{record.task_id}.json",
        TrialEvalItemResultV1(
            item_id=record.task_id,
            timestamp_utc="2026-11-11T00:00:00Z",
            run_config=native_config,
            agent_output=dict(
                status="success",
                report="Estimated effect with uncertainty.",
                turns_used=1,
                condition_provenance=dict(
                    procedure_assistance="output_contract_only",
                    analysis_specification="locked_sap",
                    analysis_surface_sha256="a" * 64,
                    max_turns=25,
                    prompt_set_sha256="b" * 64,
                    rendered_system_prompt_sha256="c" * 64,
                    tool_schema_sha256="d" * 64,
                ),
            ),
        ),
    )
    from trialagentbench_verifier.execution_contracts import ReplayInputArtifactV1

    replay = _numerical_replay().model_copy(
        update={
            "program_artifacts": (
                ReplayInputArtifactV1(relative_path="scratch/analysis.py", sha256="e" * 64, byte_length=1),
            )
        }
    )
    monkeypatch.setattr(
        grade_trialeval,
        "load_participant_release",
        lambda *_: SimpleNamespace(
            root=participant, manifest=SimpleNamespace(release_id=record.release_id, checksum="f" * 64)
        ),
    )
    monkeypatch.setattr(
        grade_trialeval,
        "discover_participant_items",
        lambda *_args, **_kwargs: {record.task_id: SimpleNamespace(trial_name="fixture", estimand_mode="")},
    )
    monkeypatch.setattr(grade_trialeval, "participant_evidence_artifacts", lambda *_: replay.evidence_artifacts)
    monkeypatch.setattr(grade_trialeval, "replay_numerical_program", lambda **_: replay)
    monkeypatch.setattr(grade_trialeval, "head_sha", lambda *_: "")
    evaluator = tmp_path / "trialeval-evaluator.zip"
    evaluator.write_bytes(b"authenticated evaluator transport fixture")
    config, manifest = _qualified_configuration(tmp_path, key=key, evaluator_sha256=sha256_file(evaluator))

    def materialized(*_args, **_kwargs):
        return nullcontext(SimpleNamespace(root=suite))

    monkeypatch.setattr(campaign, "materialized_release_archive", materialized)

    def verify(root, **kwargs):
        return authenticate_release_set_manifest(
            root,
            **{name: kwargs[name] for name in ("expected_manifest_sha256", "anchor_sha256", "anchor_url", "sums")},
        )

    def prepare_native(_benchmark, arguments):
        assert "--prepare-only" in arguments
        shutil.copytree(native, Path(arguments[arguments.index("--output-dir") + 1]))

    monkeypatch.setattr(campaign, "verify_release_set", verify)
    monkeypatch.setattr(campaign, "_campaign_asset_selections", lambda *_: {})
    monkeypatch.setattr(campaign, "preview_panel", lambda *_: None)
    monkeypatch.setattr(campaign, "release_assets", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(campaign, "_retain_campaign_executor", lambda *_: None)
    monkeypatch.setattr(campaign, "_invoke", prepare_native)
    root = tmp_path / "campaign"
    prepared = campaign.prepare_campaign(config, root)
    native = root / prepared.runs[0].relative_run_root
    path = root / prepared.runs[0].analysis_plan_path
    from trialagentbench_verifier.campaign import TrialEvalCampaignPlanV1

    scientific_plan = TrialEvalCampaignPlanV1.model_validate_json(path.read_bytes())

    def grade(arguments):
        return grade_trialeval._grade_materialized_run(
            source=native,
            suite_dir=suite,
            suite_source=evaluator,
            output=Path(arguments[arguments.index("--out-dir") + 1]),
        )

    monkeypatch.setattr(finish_run, "harness_main", grade)
    monkeypatch.setattr(finish_run, "materialized_release_archive", materialized)
    monkeypatch.setattr(finish_run, "export_results_v1", lambda **kwargs: kwargs["output_dir"].mkdir())
    output = tmp_path / "results/primary"
    arguments = [
        "trialeval",
        "--release-dir",
        str(tmp_path),
        "--expected-manifest-sha256",
        config.expected_release_manifest_sha256,
        "--run-root",
        str(native),
        "--output",
        str(output),
        "--executor-image-id",
        config.executor_image_id,
        "--campaign-plan",
        str(path),
        "--expected-plan-sha256",
        sha256_file(path),
        "--grading-key",
        str(key),
        "--grading-key-sha256",
        sha256_file(key),
        "--anchor-sha256",
        config.anchor_sha256,
        "--sums",
        str(config.sums),
    ]
    assert finish_run.main(arguments, qualified_manifest=manifest) == 0
    assert TrialEvalCampaignAnalysisV1.model_validate_json((output / "analysis.json").read_bytes())
    grade_root = output / "grade"
    source = load_trial_eval_campaign_grade_source_v1(
        "",
        grading_key=key,
        grading_key_sha256=sha256_file(key),
        expected_release_id=record.release_id,
        expected_release_manifest_sha256=scientific_plan.release_manifest_sha256,
        expected_archive_sha256=None,
        grade_roots=(grade_root,),
    )
    assert source.rows[0].outcome.decision.status == "satisfied"
    assert "decision_rule" not in json.loads((grade_root / "items" / f"{record.task_id}.json").read_text())["scores"]
    before = {file: file.read_bytes() for file in output.rglob("*") if file.is_file()}
    assert finish_run.main([*arguments, "--resume"], qualified_manifest=manifest) == 0
    assert all(file.read_bytes() == value for file, value in before.items())

    run = SimpleNamespace(
        benchmark="trialeval",
        job_id="primary",
        relative_run_root=native.relative_to(tmp_path).as_posix(),
        analysis_plan_path=path.relative_to(tmp_path).as_posix(),
    )
    plan = SimpleNamespace(configuration=config, runs=(run,), checksum="a" * 64)
    monkeypatch.setattr(finish_campaign_analysis, "campaign_censoring", lambda *_: None)
    monkeypatch.setattr(finish_campaign_analysis, "combined_trialeval_plan", lambda *_: scientific_plan)
    monkeypatch.setattr(finish_campaign_analysis, "materialized_release_archive", materialized)
    monkeypatch.setattr(finish_campaign_analysis, "export_results_v1", lambda **kwargs: kwargs["output_dir"].mkdir())
    # Economics are tested separately; here the independent key-bound reader must finish the same native grade.
    monkeypatch.setattr(finish_campaign_analysis, "trial_eval_answer_rows_v1", lambda **kwargs: ())
    finish_campaign_analysis.finish_comparisons(tmp_path, plan, manifest)
    assert TrialEvalCampaignAnalysisV1.model_validate_json(
        (tmp_path / "comparisons/trialeval/analysis.json").read_bytes()
    )


@pytest.mark.parametrize("defect", ("changed", "missing"))
def test_development_preparation_authenticates_key_before_native_work(tmp_path, monkeypatch, defect):
    """Both editable and direct Python preparation reject an unauthenticated development key."""
    from trialagentbench_verifier.campaign_inputs import (
        DevelopmentArchiveV1,
        DevelopmentCampaignInputsV1,
        DevelopmentExecutorV1,
    )

    evaluator = tmp_path / "trialeval-evaluator.zip"
    evaluator.write_bytes(b"not materialized before key authentication")
    participant = tmp_path / "trialeval-participant.zip"
    participant.write_bytes(b"participant transport")
    key = tmp_path / "records.jsonl"
    key.write_text(_evaluator_record().model_dump_json() + "\n")
    config = _configuration(tmp_path)
    manifest = DevelopmentCampaignInputsV1(
        artifacts=(
            DevelopmentArchiveV1(
                filename=evaluator.name, sha256=sha256_file(evaluator), byte_length=evaluator.stat().st_size
            ),
            DevelopmentArchiveV1(
                filename=participant.name, sha256=sha256_file(participant), byte_length=participant.stat().st_size
            ),
        ),
        executor=DevelopmentExecutorV1(image_id=config.executor_image_id, environment_sha256="f" * 64),
    )
    manifest_path = tmp_path / "development-inputs.json"
    manifest_path.write_text(manifest.model_dump_json())
    config = config.model_copy(
        update=dict(
            input_mode="development",
            expected_release_manifest_sha256=sha256_file(manifest_path),
            grading_key_sha256=sha256_file(key),
            **_anchor(tmp_path, (evaluator, key)),
        )
    )
    assert campaign._campaign_inputs(config) == manifest
    editable = tmp_path / "editable.json"
    editable.write_text(config.model_dump_json())
    if defect == "changed":
        key.write_text(key.read_text() + " ")
    else:
        key.unlink()
    native_calls = []
    monkeypatch.setattr(campaign, "_prepare_native_run", lambda *a, **k: native_calls.append((a, k)))
    for action in (
        lambda: campaign.prepare_campaign(config, tmp_path / "python-preparation"),
        lambda: campaign.main(["prepare", "--config", str(editable), "--campaign", str(tmp_path / "cli-preparation")]),
    ):
        with pytest.raises((ValueError, FileNotFoundError), match="trusted expected SHA-256|missing|records.jsonl"):
            action()
    assert native_calls == []
    assert not (tmp_path / "python-preparation").exists()
    assert not (tmp_path / "cli-preparation").exists()


def test_executor_campaign_fixture_authenticates_its_key_and_archives_without_docker(tmp_path, monkeypatch):
    """The executor-only lifecycle and reference-agent callers use real authenticated preparation inputs."""
    import test_campaign_budget_lifecycle as lifecycle

    monkeypatch.setattr(
        lifecycle,
        "resolve_executor_environment",
        lambda **_: SimpleNamespace(image_id="sha256:" + "e" * 64, semantic_sha256=lambda: "f" * 64),
    )
    root = tmp_path / "development"
    manifest_sha256, image_id = lifecycle._development_inputs(root)
    payload = _configuration(root).model_dump(mode="json") | dict(
        input_mode="development",
        expected_release_manifest_sha256=manifest_sha256,
        executor_image_id=image_id,
        **lifecycle._development_verifier_inputs(root),
    )
    configuration = CampaignConfigurationV1.model_validate(payload)
    manifest = campaign._campaign_inputs(configuration, full_verification=True)
    assert manifest.executor.image_id == image_id
    assert configuration.grading_key.read_bytes() == (root / "records.jsonl").read_bytes()
