"""Standalone comparison CLIs retain campaign finish's budget censoring."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock

import pytest

from trialagentbench_verifier import campaign, cli, development_campaign
from trialagentbench_verifier.campaign import TrialEvalCampaignPlanV1
from trialagentbench_verifier.campaign_contracts import BudgetCensoringV1, CampaignPlanV1, CensoredRunV1
from trialagentbench_verifier.io import sha256_file
from trialagentbench_verifier.scorer import current_scorer


@pytest.fixture
def campaign_files(tmp_path: Path):
    """Freeze one native plan per benchmark and the same censoring record finish writes."""
    native = TrialEvalCampaignPlanV1.model_validate(
        dict(
            release_id="release",
            release_manifest_sha256="a" * 64,
            participant_manifest_checksum="b" * 64,
            prospective_analysis_plan_sha256="c" * 64,
            cells=[dict(cell_ref="cell_" + "1" * 20, series_ref="series_" + "1" * 20)],
            attempts=[
                dict(
                    attempt_id=f"attempt-{i}",
                    scheduled_unit_id=f"attempt-{i}",
                    attempt_role="primary",
                    model_id="smoke-zero-python",
                    run_identity_sha256="d" * 64,
                    task_id=f"task-{i}",
                    trial_id="trial_" + str(i) * 20,
                    base_world_id="world_" + str(i) * 20,
                    series_ref="series_" + "1" * 20,
                    cell_ref="cell_" + "1" * 20,
                    effect_state="favourable",
                    context="C1",
                    condition_id="primary",
                    request_replicate_id="request-1",
                    procedure_assistance="output_contract_only",
                    analysis_specification="locked_sap",
                    design_family="parallel",
                    reconstruction_family=None,
                    target_ids=["effect"],
                    target_kind="point",
                    effect_measure="mean_difference",
                    result_unit="points",
                )
                for i in (1, 2)
            ],
        )
    )
    (tmp_path / "plans").mkdir()
    native_path = tmp_path / "plans/control.json"
    native_path.write_text(native.model_dump_json())
    index = CampaignPlanV1.model_validate(
        dict(
            configuration=dict(
                release_directory=str(tmp_path),
                expected_release_manifest_sha256="a" * 64,
                executor_image_id="sha256:" + "b" * 64,
                models=[
                    dict(
                        name="control",
                        model="smoke-zero-python",
                        provider="offline_control",
                        spend_limit_micro_usd=10,
                        reservation_micro_usd=1,
                        reservation_tokens=1000,
                        maximum_completion_tokens=100,
                        pricing_premise="Offline fixture",
                        requests_per_minute=10,
                        tokens_per_minute=1000,
                    )
                ],
                operations=dict(spend_limit_micro_usd=10),
            ),
            runs=[
                dict(
                    job_id=f"control-{benchmark}-1",
                    benchmark=benchmark,
                    model_name="control",
                    repetition=1,
                    relative_run_root=f"runs/{benchmark}",
                    invocation_sha256="c" * 64,
                    **(
                        dict(analysis_plan_path="plans/control.json", analysis_plan_sha256=sha256_file(native_path))
                        if benchmark == "trialeval"
                        else {}
                    ),
                )
                for benchmark in ("trialeval", "trialdev")
            ],
        )
    )
    index_path = tmp_path / "campaign.json"
    index_path.write_text(index.model_dump_json())
    censoring = BudgetCensoringV1(
        campaign_plan_sha256=index.checksum,
        exhausted_models=("control",),
        runs=tuple(
            CensoredRunV1(
                job_id=f"control-{benchmark}-1",
                model_name="control",
                model_id="smoke-zero-python",
                benchmark=benchmark,
                repetition=1,
                scheduled_task_ids=("task-1", "task-2"),
                censored_task_ids=("task-2",),
            )
            for benchmark in ("trialeval", "trialdev")
        ),
    )
    censor_path = tmp_path / "censoring.json"
    censor_path.write_text(censoring.model_dump_json())
    return index_path, native_path, censor_path, native, censoring


def test_trialeval_cli_binds_and_forwards_budget_censoring(campaign_files, tmp_path, monkeypatch):
    index, native, censor, plan, _ = campaign_files
    export = tmp_path / "results.zip"
    export.write_bytes(b"fixture")
    analyse = Mock(return_value=cli.TrialEvalCampaignAnalysisV1.model_construct())
    monkeypatch.setattr(cli, "analyse_trial_eval_campaign_v1", analyse)
    assert (
        cli.main(
            [
                "analyse-trialeval",
                "--campaign-plan",
                str(native),
                "--expected-campaign-plan-sha256",
                sha256_file(native),
                "--result-export",
                str(export),
                "--expected-result-export-sha256",
                sha256_file(export),
                "--expected-release-id",
                "release",
                "--expected-release-manifest-sha256",
                "a" * 64,
                "--budget-censoring",
                str(censor),
                "--campaign-index",
                str(index),
                "--expected-campaign-index-sha256",
                sha256_file(index),
                "--output",
                str(tmp_path / "analysis.json"),
            ]
        )
        == 0
    )
    assert analyse.call_args.kwargs["budget_censored_attempt_ids"] == frozenset({"attempt-2"})
    assert analyse.call_args.kwargs["plan"] == plan


def test_trialeval_cli_scores_a_c5_only_campaign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A C5-only plan has no crossed panel: the CLI records why and still writes every answer's score."""
    series = "series_" + "1" * 20
    cells = ["cell_" + "1" * 20, "cell_" + "2" * 20]
    plan = TrialEvalCampaignPlanV1.model_validate(
        dict(
            release_id="release",
            release_manifest_sha256="a" * 64,
            participant_manifest_checksum="b" * 64,
            prospective_analysis_plan_sha256="c" * 64,
            cells=[dict(cell_ref=cell, series_ref=series) for cell in cells],
            attempts=[
                dict(
                    attempt_id=f"attempt-{i}",
                    scheduled_unit_id=f"attempt-{i}",
                    attempt_role="primary",
                    model_id="smoke-zero-python",
                    run_identity_sha256="d" * 64,
                    task_id=f"task-{i}",
                    trial_id="trial_" + str(i) * 20,
                    base_world_id="world_" + str(i) * 20,
                    series_ref=series,
                    cell_ref=cell,
                    effect_state="favourable",
                    context="C5",
                    condition_id="primary",
                    request_replicate_id="request-1",
                    procedure_assistance="output_contract_only",
                    analysis_specification="protocol_only",
                    design_family="parallel",
                    reconstruction_family="table_reconstruction",
                    target_ids=["effect"],
                    target_kind="point",
                    effect_measure="mean_difference",
                    result_unit="points",
                )
                for i, cell in zip((1, 2), cells, strict=True)
            ],
        )
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.model_dump_json())
    export = tmp_path / "results.zip"
    export.write_bytes(b"fixture")
    # The grades are stubbed; the panel check that rejects a C5-only selection is the real one.
    source = campaign.TrialEvalAuthenticatedGradeSourceV1.model_construct(
        scorer=current_scorer(), release_id="release", release_manifest_sha256="a" * 64, member_sha256s=(), rows=()
    )
    monkeypatch.setattr(campaign, "load_trial_eval_campaign_grade_source_v1", Mock(return_value=source))
    output, answers = tmp_path / "analysis.json", tmp_path / "trialeval_answers.csv"
    arguments = [
        "analyse-trialeval",
        "--campaign-plan",
        str(plan_path),
        "--expected-campaign-plan-sha256",
        sha256_file(plan_path),
        "--result-export",
        str(export),
        "--expected-result-export-sha256",
        sha256_file(export),
        "--expected-release-id",
        "release",
        "--expected-release-manifest-sha256",
        "a" * 64,
        "--output",
        str(output),
        "--answers",
        str(answers),
    ]

    assert cli.main(arguments) == 0
    report = campaign.TrialEvalCampaignUnavailableV1.model_validate_json(output.read_text())
    assert report.reason == "Selected TrialEval tasks omit linked cell/world/context views"
    assert (report.campaign_plan_sha256, report.result_export_sha256) == (plan.checksum, sha256_file(export))
    assert answers.read_text().startswith("model,task_id,")


@pytest.mark.parametrize("changed", ["index", "censoring", "native", "combined", "missing_index"])
def test_trialeval_rejects_unbound_censoring(campaign_files, changed):
    index, native, censor, plan, record = campaign_files
    args = Namespace(budget_censoring=censor, campaign_index=index, expected_campaign_index_sha256=sha256_file(index))
    if changed == "index":
        index.write_text(index.read_text() + " ")
    elif changed == "censoring":
        censor.write_text(record.model_copy(update={"campaign_plan_sha256": "f" * 64}).model_dump_json())
    elif changed == "native":
        native.write_text(native.read_text() + " ")
    elif changed == "combined":
        plan = plan.model_copy(update={"attempts": plan.attempts[:1]})
    else:
        args.campaign_index = None
    with pytest.raises(ValueError):
        cli._trialeval_censored_attempts(args, plan)


def test_trialdev_cli_forwards_the_finish_record(campaign_files, tmp_path, monkeypatch):
    index, _, censor, _, record = campaign_files
    export = tmp_path / "results.zip"
    export.write_bytes(b"fixture")
    sources = tmp_path / "exports.json"
    sources.write_text('[{"job_id":"control-trialdev-1","path":"results.zip","sha256":"' + "a" * 64 + '"}]')
    analyse = Mock(return_value=object())
    render = Mock()
    monkeypatch.setattr(development_campaign, "analyse_trialdev_campaign", analyse)
    monkeypatch.setattr(development_campaign, "render_trialdev_comparisons", render)
    assert (
        cli.main(
            [
                "compare-trialdev",
                "release.zip",
                "--expected-release-sha256",
                "b" * 64,
                "--campaign-plan",
                str(index),
                "--expected-plan-sha256",
                sha256_file(index),
                "--exports",
                str(sources),
                "--budget-censoring",
                str(censor),
                "--output",
                str(tmp_path / "analysis"),
            ]
        )
        == 0
    )
    assert analyse.call_args.kwargs["censoring"] == record
    assert analyse.call_args.kwargs["exports"][0].path == export
    render.assert_called_once()
