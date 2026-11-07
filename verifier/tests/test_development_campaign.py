"""Independent finite-world counterexamples for paired TrialDev comparisons."""

from __future__ import annotations

import hashlib

import pytest

from trialagentbench_verifier.campaign_contracts import CampaignAnalysisV1
from trialagentbench_verifier.development_campaign import compare_trialdev_rankings
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

STRATA = (("clinical", "scenario", 3), ("clinical", "scenario", 5))
SCENARIO_KEY = "scenario_" + "0" * 20


def test_bounded_headline_withholds_incomplete_primary_loss():
    from trialagentbench_verifier.development_campaign import trialdev_leaderboard_rows

    run = single_stratum("a", (0.0, 1.0))
    run = run.model_copy(
        update={
            "items": (
                run.items[0],
                run.items[1].model_copy(
                    update={"observational_loss": None, "execution_status": "model_noncompletion"}
                ),
            )
        }
    )
    (row,) = trialdev_leaderboard_rows((run,), required_strata=STRATA[:1], policy=CampaignAnalysisV1())
    assert row.estimate is row.interval_low is row.interval_high is row.standard_error is None
    assert row.unavailable_reason == "missing_metric"
    assert (row.scheduled_tasks, row.scored_tasks) == (2, 1)


@pytest.mark.parametrize("estimate", [None, 0.0])
@pytest.mark.parametrize("standard_error,mdd", [(None, None), (0.0, 0.0), (0.2, 0.56), (0.2, None), (None, 0.56)])
def test_paired_uncertainty_requires_an_estimate_and_both_fields(estimate, standard_error, mdd):
    from pydantic import ValidationError

    from trialagentbench_verifier.development_campaign import TrialDevComparisonV1

    payload = dict(
        metric="observational_loss",
        stream="evidence_analysis",
        data_view="raw",
        model_left="a",
        model_right="b",
        worlds=4,
        strata=1,
        estimate=estimate,
        confidence_low=None,
        confidence_high=None,
        p_value=None,
        unavailable_reason="fixture",
        standard_error=standard_error,
        minimal_detectable_difference=mdd,
    )
    valid = (standard_error is None and mdd is None) or (
        estimate is not None and standard_error is not None and mdd is not None
    )
    if valid:
        row = TrialDevComparisonV1(scorer=current_scorer(), **payload)
        assert TrialDevComparisonV1.model_validate_json(row.model_dump_json()) == row
    else:
        with pytest.raises(ValidationError, match="Paired uncertainty|withheld estimate"):
            TrialDevComparisonV1(scorer=current_scorer(), **payload)


def _key_files(worlds: tuple[str, ...]) -> dict[str, bytes]:
    """Return a grading key holding the ranking stratum of each named world."""
    release_id = "release_" + "a" * 64
    strata = trialdev_world_strata_bytes(
        TrialDevWorldStrataV1(
            schema_id="trialagentbench.trialdev_world_strata/v1",
            release_id=release_id,
            worlds=tuple(
                TrialDevWorldStratumV1(
                    world_id=world, clinical_profile_id="clinical", candidate_count=3, scenario_ref=SCENARIO_KEY
                )
                for world in worlds
            ),
        )
    )
    manifest = GradingKeyManifestV1.create(
        schema_id="trialagentbench.trialdev_grading_key/v1",
        release_id=release_id,
        release_manifest_sha256="0" * 64,
        derivation_id="fixture",
        task_count=4,
        world_count=len(worlds),
        artifacts=(
            GradingKeyArtifactV1(
                relative_path=TRIALDEV_WORLD_STRATA, sha256=hashlib.sha256(strata).hexdigest(), byte_length=len(strata)
            ),
        ),
    )
    return {TRIALDEV_WORLD_STRATA: strata, GRADING_KEY_MANIFEST: grading_key_manifest_bytes(manifest)}


def ranking(model: str, *, delta: float = 0, fault: bool = False) -> TrialDevRunRankingV1:
    rows = []
    for size, count in ((3, 2), (5, 4)):
        for world in range(count):
            for stream in ("evidence_analysis", "portfolio_management"):
                value = (0.1 * world if size == 3 else 0.5) + delta
                rows.append(
                    TrialDevRankingItemV1(
                        task_id=f"{size}-{world}-{stream}",
                        world_id=f"world-{size}-{world}",
                        stream=stream,
                        clinical_profile_id="clinical",
                        scenario_profile_id="scenario",
                        asset_count=size,
                        execution_status="evaluator_error" if fault and world == 0 else "completed",
                        run_checksum="a" * 64,
                        observational_loss=value,
                        initial_target_statuses={"effect": "reported"},
                    )
                )
    return TrialDevRunRankingV1(
        scorer=current_scorer(),
        release_id="release",
        release_manifest_sha256="a" * 64,
        invocation_checksum="b" * 64,
        model_id=model,
        scope="scheduled_panel",
        items=tuple(rows),
        profile_means=(),
    )


def evidence_only(row: TrialDevRunRankingV1) -> TrialDevRunRankingV1:
    return row.model_copy(update={"items": tuple(item for item in row.items if item.stream == "evidence_analysis")})


def test_campaign_profiles_preserve_completion_and_equal_world_weight(tmp_path):
    import csv
    import json

    from trialagentbench_verifier.development_campaign import (
        TrialDevCampaignAnalysisV1,
        render_trialdev_comparisons,
        summarize_trialdev_campaign_profiles,
    )
    from trialagentbench_verifier.public_results import public_trialdev_campaign

    a = evidence_only(ranking("a"))
    incomplete = a.model_copy(
        update={
            "items": (
                a.items[0].model_copy(
                    update={
                        "execution_status": "model_noncompletion",
                        "observational_loss": None,
                        "initial_target_statuses": {"effect": "missing"},
                    }
                ),
                *a.items[1:],
            )
        }
    )
    summaries = summarize_trialdev_campaign_profiles((a, incomplete))
    three = next(row for row in summaries if row.asset_count == 3)
    assert three.scheduled_tasks == 4 and three.scored_tasks == 3
    assert three.scheduled_worlds == three.scored_worlds == 2
    assert three.unscored_task_counts == {"model_noncompletion": 1}
    assert three.model_noncompletion == 1
    assert three.target_status_counts == {"reported": 3, "missing": 1}
    assert three.mean is None
    assert three.observed_mean == pytest.approx(0.05)
    complete = next(row for row in summaries if row.asset_count == 5)
    assert complete.mean == complete.observed_mean == pytest.approx(0.5)
    assert complete.scheduled_tasks == 8 and complete.scheduled_worlds == 4
    empty = a.model_copy(
        update={
            "items": tuple(
                row.model_copy(
                    update={
                        "execution_status": "model_noncompletion",
                        "observational_loss": None,
                        "initial_target_statuses": {"effect": "missing"},
                    }
                )
                for row in a.items
            )
        }
    )
    missing = summarize_trialdev_campaign_profiles((empty,))[0].model_dump()
    assert missing["scored_worlds"] == missing["scored_tasks"] == 0
    assert missing["unscored_task_counts"] == {"model_noncompletion": 2}
    assert missing["mean"] is None and missing["observed_mean"] is None
    report = TrialDevCampaignAnalysisV1(
        scorer=current_scorer(),
        campaign_plan_sha256="a" * 64,
        release_sha256="b" * 64,
        exports=(),
        comparisons=(),
        profile_summaries=summaries,
    )
    assert public_trialdev_campaign(report, policy=CampaignAnalysisV1(), split="public").profile_summaries == summaries
    render_trialdev_comparisons(report, tmp_path / "review")
    with (tmp_path / "review/profiles.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["model_id"] == "a"
    assert json.loads(rows[0]["unscored_task_counts"]) == {"model_noncompletion": 1}
    assert rows[0]["model_noncompletion"] == "1"
    assert json.loads(rows[0]["target_status_counts"]) == {"reported": 3, "missing": 1}
    assert rows[0]["mean"] == "" and float(rows[0]["observed_mean"]) == pytest.approx(0.05)


def compare(*rows: TrialDevRunRankingV1):
    return compare_trialdev_rankings(tuple(rows), required_strata=STRATA, policy=CampaignAnalysisV1())


def test_paired_covariance_and_unequal_world_counts():
    from pathlib import Path

    from trialagentbench_verifier.development_campaign import TrialDevCampaignAnalysisV1, TrialDevCampaignExportV1
    from trialagentbench_verifier.public_results import public_trialdev_campaign

    a, b = evidence_only(ranking("a", delta=0.1)), evidence_only(ranking("b"))
    row = compare(a, b)[0]
    assert row.estimate == pytest.approx(0.1)
    assert row.confidence_low == pytest.approx(0.1)
    assert row.confidence_high == pytest.approx(0.1)
    assert row.worlds == 6
    assert row.strata == 2
    assert row.adjusted_p_value == pytest.approx(1 / 2001)
    # Duplicating the same within-world observations changes neither support nor uncertainty.
    repeated = compare(a, a, b, b)[0]
    assert repeated == row
    report = TrialDevCampaignAnalysisV1(
        scorer=current_scorer(),
        campaign_plan_sha256="a" * 64,
        release_sha256="b" * 64,
        exports=(TrialDevCampaignExportV1(job_id="a", path=Path("/private/custody.zip"), sha256="c" * 64),),
        comparisons=(row,),
    )
    public = public_trialdev_campaign(report, policy=CampaignAnalysisV1(), split="public")
    assert public.comparisons == (row,)
    assert public.export_sha256 == ("c" * 64,)
    native = report.model_copy(
        update={"exports": tuple(row.model_copy(update={"sha256": None}) for row in report.exports)}
    )
    native_public = public_trialdev_campaign(native, policy=CampaignAnalysisV1(), split="public")
    assert native_public.comparisons == public.comparisons
    assert "export_sha256" not in native_public.model_dump()
    assert "/private" not in public.model_dump_json()


def test_withheld_public_trialdev_results_and_leaderboard_name_no_tasks():
    from pathlib import Path

    from pydantic import ValidationError

    from trialagentbench_verifier.development_campaign import TrialDevCampaignAnalysisV1, TrialDevCampaignExportV1
    from trialagentbench_verifier.public_results import (
        CampaignLeaderboardV1,
        PublicTrialDevCampaignV1,
        public_trialdev_campaign,
    )

    censoring = _censoring(a=(), b=("3-1-evidence_analysis",))
    report = TrialDevCampaignAnalysisV1(
        scorer=current_scorer(),
        campaign_plan_sha256="a" * 64,
        release_sha256="b" * 64,
        exports=(TrialDevCampaignExportV1(job_id="a", path=Path("/private/custody.zip"), sha256="c" * 64),),
        comparisons=(compare(evidence_only(ranking("a")), evidence_only(ranking("b")))[0],),
        budget_censoring=censoring,
    )
    task_ids = {task for run in censoring.runs for task in run.scheduled_task_ids}
    withheld = public_trialdev_campaign(report, policy=CampaignAnalysisV1(), split="withheld")
    payload = withheld.model_dump_json()
    assert withheld.budget_censoring is None and '"split":"withheld"' in payload
    assert task_ids and not any(task in payload for task in task_ids)
    with pytest.raises(ValidationError, match="cannot name budget-censored tasks"):
        PublicTrialDevCampaignV1.model_validate({**withheld.model_dump(), "budget_censoring": censoring})
    public = public_trialdev_campaign(report, policy=CampaignAnalysisV1(), split="public")
    assert public.budget_censoring == censoring and '"split"' not in public.model_dump_json()

    leaderboard = {
        "campaign_plan_sha256": "a" * 64,
        "release_set_manifest_sha256": "b" * 64,
        "confidence_level": 0.95,
        "bootstrap_resamples": 100,
        "bootstrap_seed": 1,
        "rows": (),
    }
    assert CampaignLeaderboardV1(scorer=current_scorer(), **leaderboard, split="public", budget_censoring=censoring)
    assert CampaignLeaderboardV1(scorer=current_scorer(), **leaderboard, split="withheld").budget_censoring is None
    with pytest.raises(ValidationError, match="cannot name budget-censored tasks"):
        CampaignLeaderboardV1(scorer=current_scorer(), **leaderboard, split="withheld", budget_censoring=censoring)


def test_strata_are_equally_weighted_and_ties_do_not_force_ordering():
    a = evidence_only(ranking("a"))
    b = evidence_only(ranking("b"))
    b = b.model_copy(update={"items": tuple(row.model_copy(update={"observational_loss": 0.0}) for row in b.items)})
    assert compare(a, b)[0].estimate == pytest.approx((0.05 + 0.5) / 2)
    tied = compare(a, a.model_copy(update={"model_id": "b"}))[0]
    assert tied.estimate == 0
    assert tied.confidence_low == tied.confidence_high == 0
    assert tied.p_value == tied.adjusted_p_value == 1


def test_faults_missing_strata_and_pending_primaries_are_explicit():
    a, b = evidence_only(ranking("a")), evidence_only(ranking("b", fault=True))
    assert compare(a, b)[0].unavailable_reason == "evaluator_fault"
    assert compare(a, b)[0].estimate is None
    b = evidence_only(ranking("b"))
    shortened = tuple(row.model_copy(update={"items": row.items[:1]}) for row in (a, b))
    assert compare(*shortened)[0].unavailable_reason == "missing_required_stratum"
    with pytest.raises(ValueError, match="same scheduled task panel"):
        compare(a, shortened[1])
    for status in ("not_started", "incomplete_run"):
        pending = b.model_copy(
            update={"items": (b.items[0].model_copy(update={"execution_status": status}), *b.items[1:])}
        )
        with pytest.raises(ValueError, match="terminal primary"):
            compare(a, pending)


def test_model_noncompletion_withholds_loss_and_retains_holm_family():
    rows = []
    for model in ("a", "b", "c"):
        row = evidence_only(ranking(model))
        rows.append(
            row.model_copy(
                update={
                    "items": tuple(
                        item.model_copy(
                            update={
                                "execution_status": "model_noncompletion",
                                "observational_loss": None,
                                "initial_target_statuses": {"effect": "missing"},
                            }
                        )
                        for item in row.items
                    )
                }
            )
        )
    result = compare(*rows)
    recovery = [row for row in result if row.metric == "observational_loss"]
    assert len(recovery) == 3
    assert all(row.estimate is None and row.adjusted_p_value is None for row in recovery)
    assert all(row.unavailable_reason == "model_noncompletion" for row in recovery)
    fabricated = rows[0].model_copy(
        update={"items": tuple(row.model_copy(update={"observational_loss": 0.0}) for row in rows[0].items)}
    )
    with pytest.raises(ValueError, match="uncompleted evidence analysis"):
        compare(fabricated, rows[1])


def test_common_world_sensitivity_preserves_missing_panel_and_equal_strata():
    a, b = evidence_only(ranking("a")), evidence_only(ranking("b"))
    b = b.model_copy(update={"items": tuple(row.model_copy(update={"observational_loss": 0.0}) for row in b.items)})
    a = a.model_copy(
        update={
            "items": (
                a.items[0].model_copy(
                    update={
                        "execution_status": "model_noncompletion",
                        "observational_loss": None,
                        "initial_target_statuses": {"effect": "missing"},
                    }
                ),
                *a.items[1:],
            )
        }
    )
    row = compare(a, b)[0]
    assert row.estimate is None and row.unavailable_reason == "model_noncompletion"
    assert row.worlds == 6 and row.strata == 2
    assert row.common_worlds == 5 and row.common_strata == 2
    assert row.common_world_estimate == pytest.approx((0.1 + 0.5) / 2)
    repeated = compare(a, a, b, b)[0]
    assert repeated == row
    fault = evidence_only(ranking("fault", fault=True))
    blocked = compare(a, fault)[0]
    assert blocked.common_world_estimate is None
    assert blocked.common_worlds == blocked.common_strata == 0


def test_programme_metric_uses_portfolio_only_and_evidence_fault_is_local():
    runs = []
    for model, advantage in (("a", 4.0), ("b", 0.0)):
        run = ranking(model)
        items = []
        for row in run.items:
            if row.stream == "portfolio_management":
                items.append(
                    row.model_copy(
                        update={
                            "authored_programme": True,
                            "net_development_value_usd": advantage + row.asset_count,
                        }
                    )
                )
            else:
                items.append(row.model_copy(update={"execution_status": "evaluator_error"}))
        runs.append(run.model_copy(update={"items": tuple(items)}))
    recovery, programme_recovery, value = compare(*runs)
    assert recovery.unavailable_reason == "evaluator_fault"
    assert programme_recovery.unavailable_reason is None
    assert programme_recovery.estimate == 0
    assert value.estimate == value.confidence_low == value.confidence_high == 4
    assert value.unavailable_reason is None


def test_single_model_and_small_complete_strata():
    assert compare(evidence_only(ranking("a"))) == ()
    rows = []
    for model in ("a", "b"):
        run = evidence_only(ranking(model))
        rows.append(run.model_copy(update={"items": (run.items[0], run.items[2])}))
    result = compare(*rows)[0]
    assert result.estimate is None
    assert result.common_world_estimate == 0
    assert result.unavailable_reason == "insufficient_independent_worlds"
    assert result.confidence_low is result.confidence_high is result.p_value is None


def single_stratum(model: str, losses: tuple[float, ...]) -> TrialDevRunRankingV1:
    """Place one evidence analysis per world in the three-asset stratum."""
    run = evidence_only(ranking(model))
    template = run.items[0]
    return run.model_copy(
        update={
            "items": tuple(
                template.model_copy(
                    update={"task_id": f"task-{world}", "world_id": f"world-{world}", "observational_loss": loss}
                )
                for world, loss in enumerate(losses)
            )
        }
    )


@pytest.mark.parametrize("worlds", [2, 4])
def test_small_stratum_bootstrap_standard_error_is_unbiased(worlds: int):
    """Plain world resampling gives a stratum mean s * sqrt((n - 1) / n) / sqrt(n); corrected, s / sqrt(n)."""
    from math import sqrt
    from statistics import mean, stdev

    from trialagentbench_verifier.development_campaign import trialdev_leaderboard_rows
    from trialagentbench_verifier.numerical_scoring import bounded_recovery

    differences = (0.0, 0.3, 0.1, 0.7)[:worlds]
    policy = CampaignAnalysisV1(bootstrap_repetitions=20_000)
    left = single_stratum("a", tuple(0.2 + value for value in differences))
    right = single_stratum("b", (0.2,) * worlds)
    (row,) = compare_trialdev_rankings((left, right), required_strata=STRATA[:1], policy=policy)
    assert row.unavailable_reason is None and row.estimate == pytest.approx(mean(differences))
    assert row.standard_error == pytest.approx(stdev(differences) / sqrt(worlds), rel=0.03)
    if worlds == 2:
        # Two worlds give three atoms; the 2.5% limit is the lower one, sqrt(2) times as far from the mean.
        assert row.confidence_low == pytest.approx(0.15 - sqrt(2) * 0.15)
        assert row.confidence_high == pytest.approx(0.15 + sqrt(2) * 0.15)
    (headline,) = trialdev_leaderboard_rows((left,), required_strata=STRATA[:1], policy=policy)
    from trialagentbench_verifier.development_campaign import _world_bootstrap

    losses = [0.2 + value for value in differences]
    _, low, high, error = _world_bootstrap({STRATA[0]: losses}, policy)
    recovery = bounded_recovery(mean(losses))
    assert headline.estimate == pytest.approx(recovery)
    assert headline.interval_low == pytest.approx(bounded_recovery(max(0.0, high)))
    assert headline.interval_high == pytest.approx(bounded_recovery(max(0.0, low)))
    assert headline.standard_error == pytest.approx(error * recovery**2)
    assert headline.standard_error == pytest.approx(stdev(losses) / sqrt(worlds) * recovery**2, rel=0.03)


def test_minimal_detectable_difference_and_standard_errors_reach_every_output(tmp_path):
    import csv
    import json
    from pathlib import Path

    from trialagentbench_verifier.campaign_contracts import LeaderboardRowV1
    from trialagentbench_verifier.development_campaign import (
        MDD_MULTIPLIER,
        TrialDevCampaignAnalysisV1,
        TrialDevCampaignExportV1,
        render_trialdev_comparisons,
    )
    from trialagentbench_verifier.public_results import public_trialdev_campaign

    assert MDD_MULTIPLIER == pytest.approx(2.8016, abs=1e-3)
    a = evidence_only(ranking("a"))
    b = a.model_copy(
        update={
            "model_id": "b",
            "items": tuple(
                row.model_copy(update={"observational_loss": 0.05 * index}) for index, row in enumerate(a.items)
            ),
        }
    )
    row = compare(a, b)[0]
    assert row.standard_error is not None and row.standard_error > 0
    assert row.minimal_detectable_difference == pytest.approx(2.8016 * row.standard_error, abs=1e-3)
    # Withheld inference withholds its standard error and detectable difference too.
    small = compare(*(run.model_copy(update={"items": (run.items[0], run.items[2])}) for run in (a, b)))[0]
    assert small.unavailable_reason == "insufficient_independent_worlds"
    assert small.standard_error is small.minimal_detectable_difference is None
    report = TrialDevCampaignAnalysisV1(
        scorer=current_scorer(),
        campaign_plan_sha256="a" * 64,
        release_sha256="b" * 64,
        exports=(TrialDevCampaignExportV1(job_id="a", path=Path("custody.zip"), sha256="c" * 64),),
        comparisons=(row, small),
    )
    render_trialdev_comparisons(report, tmp_path / "review")
    with (tmp_path / "review/comparisons.csv").open() as handle:
        table = list(csv.DictReader(handle))
    assert float(table[0]["standard_error"]) == pytest.approx(row.standard_error)
    assert float(table[0]["minimal_detectable_difference"]) == pytest.approx(row.minimal_detectable_difference)
    assert table[1]["standard_error"] == table[1]["minimal_detectable_difference"] == ""
    analysis = json.loads((tmp_path / "review/analysis.json").read_text())
    assert analysis["comparisons"][0]["minimal_detectable_difference"] == row.minimal_detectable_difference
    assert public_trialdev_campaign(report, policy=CampaignAnalysisV1(), split="public").comparisons == (row, small)
    with pytest.raises(ValueError, match="cannot carry an interval or standard error"):
        LeaderboardRowV1(
            scorer=current_scorer(),
            benchmark="trialdev",
            track="portfolio_management/raw",
            model_id="a",
            metric="net_development_value_usd",
            direction="higher_is_better",
            estimate=None,
            standard_error=1.0,
            scheduled_tasks=1,
            scored_tasks=0,
            independent_worlds=1,
        )


def authored_ranking(model: str, delta: float = 0) -> TrialDevRunRankingV1:
    """Phased results carry no decision object."""
    run = ranking(model)
    return run.model_copy(
        update={
            "items": tuple(
                row.model_copy(
                    update={
                        "authored_programme": True,
                        "net_development_value_usd": row.asset_count * 1000 + delta,
                    }
                )
                for row in run.items
                if row.stream == "portfolio_management"
            )
        }
    )


def test_authored_campaign_preserves_dollars_and_charges_unfinished_programmes():
    """Abandoning a failing programme keeps its world, scored exactly as a stop at that point."""
    a, b = authored_ranking("a", 250_000), authored_ranking("b")
    result = compare(a, b)
    value = next(row for row in result if row.metric == "net_development_value_usd")
    assert value.estimate == pytest.approx(250_000)
    assert value.confidence_low == value.confidence_high == pytest.approx(250_000)
    assert value.worlds == 6
    assert not any(row.metric == "development_value" for row in result)
    # b abandons world 3-0 after incurring $2M of study costs and overhead.
    abandoned = b.model_copy(
        update={
            "items": (
                b.items[0].model_copy(
                    update={"execution_status": "model_noncompletion", "net_development_value_usd": -2_000_000}
                ),
                *b.items[1:],
            )
        }
    )
    result = compare(a, abandoned)
    value = next(row for row in result if row.metric == "net_development_value_usd")
    assert value.unavailable_reason is None
    assert value.worlds == value.common_worlds == 6 and value.common_strata == 2
    assert value.estimate == pytest.approx(((253_000 + 2_000_000 + 250_000) / 2 + 250_000) / 2)
    assert value.estimate > 250_000  # Dropping the abandoned world would have flattered b.
    # Abandoning cannot score higher than an explicit stop at the same point.
    stopped = abandoned.model_copy(
        update={"items": (abandoned.items[0].model_copy(update={"execution_status": "completed"}), *b.items[1:])}
    )
    assert compare(a, abandoned) == compare(a, stopped)
    unvalued = abandoned.model_copy(
        update={"items": (abandoned.items[0].model_copy(update={"net_development_value_usd": None}), *b.items[1:])}
    )
    with pytest.raises(ValueError, match="lacks its development value"):
        compare(a, unvalued)


@pytest.mark.parametrize("field,value", [("data_view", "analysis_ready"), ("budget_usd", 15_000_000)])
def test_campaign_pairing_binds_presentation_and_resources(field: str, value: str | int):
    a, b = authored_ranking("a"), authored_ranking("b")
    changed = b.model_copy(update={"items": (b.items[0].model_copy(update={field: value}), *b.items[1:])})
    with pytest.raises(ValueError, match="same scheduled task panel"):
        compare(a, changed)


def test_authored_value_fault_is_not_noncompletion_or_normalized_value():
    a, b = authored_ranking("a"), authored_ranking("b")
    faulty = b.model_copy(
        update={"items": (b.items[0].model_copy(update={"execution_status": "evaluator_error"}), *b.items[1:])}
    )
    assert all(row.estimate is None and row.unavailable_reason == "evaluator_fault" for row in compare(a, faulty))
    mixed = a.model_copy(update={"items": (a.items[0].model_copy(update={"authored_programme": False}), *a.items[1:])})
    with pytest.raises(ValueError, match="requires authored development values"):
        compare(mixed, mixed.model_copy(update={"model_id": "b"}))


def test_rankings_reject_unsupported_programme_schema():
    """An unsupported ranking schema cannot enter dollar-valued comparisons."""
    from pydantic import ValidationError

    current = authored_ranking("a").model_dump(mode="json")
    assert current["schema_id"] == "trialagentbench.trialdev_run_ranking/v6"
    for version in (1, 5):
        unsupported = {**current, "schema_id": f"trialagentbench.trialdev_run_ranking/v{version}"}
        with pytest.raises(ValidationError, match="schema_id"):
            TrialDevRunRankingV1.model_validate(unsupported)
    current["items"][0]["programme"] = {"ranking_value": 0.8}
    with pytest.raises(ValidationError, match="programme"):
        TrialDevRunRankingV1.model_validate(current)


def test_opposite_presentation_and_stream_effects_do_not_cancel():
    """Separate easy-view gains and standalone recovery from raw development."""
    models = []
    for model in ("a", "b"):
        source = authored_ranking(model)
        items = []
        for row in source.items:
            for stream in ("evidence_analysis", "portfolio_management"):
                for view in ("raw", "analysis_ready"):
                    difference = (0.2 if view == "raw" else -0.2) * (1 if stream == "evidence_analysis" else -1)
                    items.append(
                        row.model_copy(
                            update={
                                "task_id": f"{row.task_id}-{stream}-{view}",
                                "stream": stream,
                                "data_view": view,
                                "observational_loss": 0.5 + (difference if model == "a" else 0),
                            }
                        )
                    )
        models.append(source.model_copy(update={"items": tuple(items)}))
    results = compare(*models)
    recovery = {(row.stream, row.data_view): row for row in results if row.metric == "observational_loss"}
    assert len(results) == 6
    assert len(recovery) == 4
    for (stream, view), result in recovery.items():
        expected = (0.2 if view == "raw" else -0.2) * (1 if stream == "evidence_analysis" else -1)
        assert result.estimate == pytest.approx(expected)
        assert result.worlds == 6


def test_completed_initial_analysis_survives_later_programme_noncompletion():
    runs = []
    for model, cost in (("a", 1_000_000), ("b", 3_000_000)):
        row = ranking(model)
        runs.append(
            row.model_copy(
                update={
                    "items": tuple(
                        item.model_copy(
                            update={
                                "execution_status": "model_noncompletion",
                                "authored_programme": True,
                                "net_development_value_usd": -cost,
                            }
                        )
                        for item in row.items
                        if item.stream == "portfolio_management"
                    )
                }
            )
        )
    numerical, value = compare(*runs)
    assert numerical.estimate == 0
    assert numerical.unavailable_reason is None
    # Unfinished everywhere, each model is still compared on the costs it incurred.
    assert value.estimate == pytest.approx(2_000_000)
    assert value.unavailable_reason is None and value.common_worlds == 6


def test_development_missing_metrics_retain_distinct_execution_states():
    """An omitted analysis does not erase a programme's value; noncompletion stays counted."""
    from trialagentbench_verifier.development import ranking_means

    source = ranking("a").items[1]
    values = {"completed": 100.0, "model_noncompletion": -40.0}
    rows = [
        source.model_copy(
            update={
                "task_id": f"task-{index}",
                "execution_status": status,
                "authored_programme": True,
                "observational_loss": None,
                "net_development_value_usd": values.get(status),
            }
        )
        for index, status in enumerate(
            ("completed", "model_noncompletion", "evaluator_error", "incomplete_run", "not_started")
        )
    ]
    summaries = {row.metric: row for row in ranking_means(rows)}
    recovery = summaries["observational_loss"]
    value = summaries["net_development_value_usd"]
    assert recovery.scored_tasks == 0 and recovery.observed_mean is None
    assert recovery.unscored_task_counts == {
        "unscored": 1,
        "model_noncompletion": 1,
        "evaluator_error": 1,
        "incomplete_run": 1,
        "not_started": 1,
    }
    # The unfinished programme is scored at its incurred costs and still reported as noncompletion.
    assert value.scored_tasks == 2 and value.observed_mean == 30.0
    assert value.unscored_task_counts == {"evaluator_error": 1, "incomplete_run": 1, "not_started": 1}
    assert value.model_noncompletion == recovery.model_noncompletion == 1
    assert recovery.mean is value.mean is None
    with pytest.raises(ValueError, match="every missing metric"):
        type(value).model_validate({**value.model_dump(), "unscored_task_counts": {}})


def test_partial_targets_are_counted_without_a_partial_task_score():
    from trialagentbench_verifier.development import ranking_means

    row = (
        ranking("a")
        .items[0]
        .model_copy(
            update={
                "observational_loss": None,
                "initial_target_statuses": {"benefit": "reported", "safety": "missing"},
            }
        )
    )
    (summary,) = ranking_means([row])
    assert summary.target_status_counts == {"reported": 1, "missing": 1}
    assert summary.scored_tasks == 0
    assert summary.mean is summary.observed_mean is None
    with pytest.raises(ValueError, match="every initial target"):
        type(row).model_validate({**row.model_dump(), "observational_loss": 0.1})


def test_campaign_rejects_different_initial_target_inventory():
    a, b = evidence_only(ranking("a")), evidence_only(ranking("b"))
    changed = b.model_copy(
        update={
            "items": (
                b.items[0].model_copy(update={"initial_target_statuses": {"different": "reported"}}),
                *b.items[1:],
            )
        }
    )
    with pytest.raises(ValueError, match="same scheduled task panel"):
        compare(a, changed)


@pytest.mark.parametrize(
    ("selectors", "expected", "expected_error"),
    [
        ((), (0, 2), None),
        (("td-000000000001",), (1,), None),
        (("tdw-0000",), (0, 1), None),
        (("evidence_analysis",), (0, 1), None),
        (("tdw-0001:portfolio_management",), (2, 3), None),
        (("missing",), (0,), "selectors match no task"),
        ((), (0,), "frozen model or task panel"),
        ((), (0, 1, 2, 3), "frozen model or task panel"),
        (("tdw-0000",), (0,), "frozen model or task panel"),
    ],
)
@pytest.mark.parametrize("transport", ["zip", "assets", "incomplete_assets"])
def test_campaign_archive_resolves_native_selectors(
    tmp_path, monkeypatch, selectors, expected, expected_error, transport
):
    """Default and shorthand panels must reconcile with authenticated native ranks."""
    import json
    from types import SimpleNamespace
    from unittest.mock import Mock
    from zipfile import ZipFile

    from trialagentbench_verifier import development_campaign as owner
    from trialagentbench_verifier.campaign_contracts import CampaignPlanV1
    from trialagentbench_verifier.io import sha256_file

    tasks = [
        dict(
            task_id=f"td-{index:012x}",
            case_id=f"tdw-{index // 2:04x}",
            stream="evidence_analysis" if index < 2 else "portfolio_management",
            data_view="raw" if index % 2 == 0 else "analysis_ready",
            available_materials=[],
        )
        for index in range(4)
    ]
    release = tmp_path / "release.zip"
    # A selective view must carry every world's stratum; the incomplete key omits the second world.
    keyed_worlds = ("tdw-0000", "tdw-0001") if transport == "assets" else ("tdw-0000",)
    with ZipFile(release, "w") as archive:
        for task in tasks:
            archive.writestr(f"participant/tasks/{task['task_id']}/task.json", json.dumps(task))
        for name, body in _key_files(keyed_worlds).items():
            archive.writestr(name, body)
    if transport != "zip":
        from trialagentbench_verifier.assets import pack_release_assets

        with ZipFile(release, "a") as archive:
            archive.writestr("release_manifest.json", '{"schema_id":"ariadne.trialdev.release/v1"}')
            archive.writestr("participant/manifest.json", "{}")
            archive.writestr("evaluation/manifest.json", "{}")
            for index in range(2):
                archive.writestr(f"participant/worlds/tdw-{index:04x}/observed.csv", "patient,value\n1,0\n")
        _, release = pack_release_assets(release, expected_sha256=sha256_file(release), output=tmp_path / "assets")
    plan = CampaignPlanV1.model_validate(
        dict(
            configuration=dict(
                release_directory=tmp_path,
                expected_release_manifest_sha256="a" * 64,
                executor_image_id="sha256:" + "b" * 64,
                models=[
                    dict(
                        name="control",
                        model="fixture/control",
                        upstream_provider="fixture",
                        spend_limit_micro_usd=10,
                        reservation_micro_usd=1,
                        reservation_tokens=1000,
                        maximum_completion_tokens=100,
                        pricing_premise="Offline test",
                        requests_per_minute=10,
                        tokens_per_minute=1000,
                    )
                ],
                panel=dict(benchmarks=["trialdev"], trialdev_tasks=selectors),
                operations=dict(spend_limit_micro_usd=10),
            ),
            runs=[
                dict(
                    job_id="control-trialdev-1",
                    benchmark="trialdev",
                    model_name="control",
                    repetition=1,
                    relative_run_root="runs/control/run",
                    invocation_sha256="c" * 64,
                )
            ],
        )
    )
    items = tuple(
        ranking("fixture/control")
        .items[0]
        .model_copy(
            update=dict(
                task_id=tasks[index]["task_id"],
                world_id=tasks[index]["case_id"],
                stream=tasks[index]["stream"],
                data_view=tasks[index]["data_view"],
                authored_programme=index >= 2,
                net_development_value_usd=0.0 if index >= 2 else None,
                scenario_profile_id=SCENARIO_KEY,
            )
        )
        for index in expected
    )
    observed = ranking("fixture/control").model_copy(update=dict(items=items))
    verify = Mock(
        return_value=SimpleNamespace(status="pass", checks=[SimpleNamespace(observed=observed.model_dump())])
    )
    monkeypatch.setattr(owner, "verify_trialdev_ranking", verify)
    export = owner.TrialDevCampaignExportV1(
        job_id="control-trialdev-1", path=tmp_path / "results.zip", sha256="d" * 64
    )

    def analyse():
        return owner.analyse_trialdev_campaign(
            release,
            expected_release_sha256=sha256_file(release),
            plan=plan,
            expected_plan_checksum=plan.checksum,
            exports=(export,),
        )

    if transport == "incomplete_assets" and selectors != ("missing",):
        expected_error = "complete shared world metadata"
    if expected_error:
        with pytest.raises(ValueError, match=expected_error):
            analyse()
        return
    result = analyse()
    assert sum(row.scheduled_tasks for row in result.profile_summaries if row.metric == "observational_loss") == len(
        expected
    )
    assert verify.call_args.kwargs["expected_invocation_sha256"] == "c" * 64


def test_leaderboard_transforms_complete_primary_loss_and_world_clustered_intervals():
    from statistics import mean

    from trialagentbench_verifier.development_campaign import trialdev_leaderboard_rows

    policy = CampaignAnalysisV1()
    complete = evidence_only(ranking("model-a"))
    (row,) = trialdev_leaderboard_rows((complete,), required_strata=STRATA, policy=policy)
    expected = 1 / (1 + mean((mean((0.0, 0.1)), 0.5)))
    assert (row.metric, row.track, row.scheduled_tasks, row.independent_worlds) == (
        "bounded_recovery",
        "evidence_analysis/raw",
        6,
        6,
    )
    assert row.estimate == pytest.approx(expected)
    assert row.interval_low is not None and row.interval_high is not None
    assert row.interval_low <= row.estimate <= row.interval_high

    missing_item = complete.items[0].model_copy(
        update={
            "execution_status": "model_noncompletion",
            "observational_loss": None,
            "initial_target_statuses": {"effect": "missing"},
        }
    )
    incomplete = complete.model_copy(update={"items": (missing_item, *complete.items[1:])})
    (row,) = trialdev_leaderboard_rows((incomplete,), required_strata=STRATA, policy=policy)
    assert row.scored_tasks == 5
    assert row.estimate is row.interval_low is row.interval_high is row.standard_error is None
    assert row.unavailable_reason == "missing_metric"

    authored = authored_ranking("model-a")
    value = next(
        row
        for row in trialdev_leaderboard_rows((authored,), required_strata=STRATA, policy=policy)
        if row.metric == "net_development_value_usd"
    )
    assert (value.estimate, value.interval_low, value.interval_high) == (4000.0, 4000.0, 4000.0)

    def value_row(first: dict):
        changed = authored.model_copy(
            update={"items": (authored.items[0].model_copy(update=first), *authored.items[1:])}
        )
        return next(
            row
            for row in trialdev_leaderboard_rows((changed,), required_strata=STRATA, policy=policy)
            if row.metric == "net_development_value_usd"
        )

    # An unfinished programme counts at its incurred costs; only an evaluator fault withholds the headline.
    unfinished = value_row({"execution_status": "model_noncompletion", "net_development_value_usd": -1000.0})
    assert unfinished.scored_tasks == unfinished.scheduled_tasks
    assert unfinished.estimate == pytest.approx(mean((mean((-1000.0, 3000.0)), 5000.0)))
    fault = value_row({"execution_status": "evaluator_error", "net_development_value_usd": None})
    assert (fault.estimate, fault.interval_low, fault.scored_tasks) == (None, None, fault.scheduled_tasks - 1)


def _censoring(**censored: tuple[str, ...]):
    from trialagentbench_verifier.campaign_contracts import BudgetCensoringV1, CensoredRunV1

    tasks = tuple(row.task_id for row in evidence_only(ranking("a")).items)
    return BudgetCensoringV1(
        campaign_plan_sha256="c" * 64,
        exhausted_models=tuple(name for name, rows in censored.items() if rows),
        runs=tuple(
            CensoredRunV1(
                job_id=f"{name}-trialdev-1",
                model_name=name,
                model_id=name,
                benchmark="trialdev",
                repetition=1,
                scheduled_task_ids=tasks,
                censored_task_ids=rows,
            )
            for name, rows in censored.items()
        ),
    )


def _unrun(row: TrialDevRunRankingV1, task_ids: set[str]) -> TrialDevRunRankingV1:
    return row.model_copy(
        update={
            "items": tuple(
                (
                    item.model_copy(
                        update={
                            "execution_status": "incomplete_run",
                            "observational_loss": None,
                            "initial_target_statuses": {"effect": "incomplete_run"},
                        }
                    )
                    if item.task_id in task_ids
                    else item
                )
                for item in row.items
            )
        }
    )


def test_budget_censoring_compares_models_on_the_paired_panel_only():
    """Censored tasks leave the paired panel; participant failures withhold numerical rankings."""
    from trialagentbench_verifier.development_campaign import paired_trialdev_results, trialdev_leaderboard_rows

    policy = CampaignAnalysisV1(bootstrap_repetitions=200)
    complete_a, complete_b = evidence_only(ranking("a")), evidence_only(ranking("b", delta=0.2))
    censored = {"5-2-evidence_analysis", "5-3-evidence_analysis"}
    failed = complete_b.items[0].model_copy(
        update={
            "execution_status": "model_noncompletion",
            "observational_loss": None,
            "initial_target_statuses": {"effect": "missing"},
        }
    )
    complete_b = complete_b.model_copy(update={"items": (failed, *complete_b.items[1:])})
    rankings = {"a-trialdev-1": complete_a, "b-trialdev-1": _unrun(complete_b, censored)}
    censoring = _censoring(a=(), b=tuple(sorted(censored)))
    comparisons, profiles, leaderboard = paired_trialdev_results(
        rankings, censoring=censoring, required_strata=STRATA, policy=policy
    )
    paired = frozenset(row.task_id for row in complete_a.items) - censored
    assert censoring.paired_task_ids("trialdev") == paired
    retained = tuple(
        row.model_copy(update={"items": tuple(i for i in row.items if i.task_id in paired)})
        for row in (complete_a, complete_b)
    )
    assert comparisons == compare_trialdev_rankings(retained, required_strata=STRATA, policy=policy)
    expected = trialdev_leaderboard_rows(retained, required_strata=STRATA, policy=policy)
    assert [
        row.model_copy(update={"budget_censored_tasks": 0, "population": base.population, "warnings": base.warnings})
        for row, base in zip(leaderboard, expected, strict=True)
    ] == list(expected)
    assert all(row.population == "paired_completed_panel" and row.warnings for row in leaderboard)
    by_model = {row.model_id: row for row in leaderboard}
    assert (by_model["a"].scheduled_tasks, by_model["a"].budget_censored_tasks) == (4, 0)
    # The participant failure withholds recovery; censored tasks are excluded.
    assert by_model["b"].estimate is None
    assert (by_model["b"].scheduled_tasks, by_model["b"].scored_tasks, by_model["b"].budget_censored_tasks) == (
        4,
        3,
        2,
    )
    # Single-model summaries cover each model's own completed tasks.
    completed = {
        model: sum(
            row.scheduled_tasks for row in profiles if row.model_id == model and row.metric == "observational_loss"
        )
        for model in ("a", "b")
    }
    assert completed == {"a": 6, "b": 4}
    assert not any(row.unscored_task_counts.get("incomplete_run") for row in profiles)
    assert [(row.model_id, row.censoring_rate) for row in censoring.models] == [("a", 0.0), ("b", 1 / 3)]
    assert censoring.warnings and "differs materially" in censoring.warnings[0]

    with pytest.raises(ValueError, match="only recorded budget censoring"):
        paired_trialdev_results(rankings, censoring=None, required_strata=STRATA, policy=policy)
    with pytest.raises(ValueError, match="only recorded budget censoring"):
        paired_trialdev_results(
            {"a-trialdev-1": complete_a, "b-trialdev-1": complete_b},
            censoring=censoring,
            required_strata=STRATA,
            policy=policy,
        )
    everything = _censoring(a=(), b=tuple(sorted(row.task_id for row in complete_a.items)))
    unrun = {"a-trialdev-1": complete_a, "b-trialdev-1": _unrun(complete_b, set(everything.runs[1].censored_task_ids))}
    comparisons, profiles, leaderboard = paired_trialdev_results(
        unrun, censoring=everything, required_strata=STRATA, policy=policy
    )
    assert comparisons == () and leaderboard == () and {row.model_id for row in profiles} == {"a"}
    assert any("no scheduled task was completed by every model" in text for text in everything.warnings)


def test_budget_censoring_record_requires_recorded_exhaustion_and_derives_its_summaries():
    from trialagentbench_verifier.campaign_contracts import BudgetCensoringV1

    # Equal censoring rates need no informative-censoring warning; the paired panel drops both tasks.
    equal = _censoring(a=("3-0-evidence_analysis",), b=("3-1-evidence_analysis",))
    assert equal.warnings == () and len(equal.paired_task_ids("trialdev")) == 4
    record = _censoring(a=("3-0-evidence_analysis",), b=())
    assert "a 17%, b 0%" in record.warnings[0]
    payload = record.model_dump(mode="json")
    with pytest.raises(ValueError, match="recorded spending-limit exhaustion"):
        BudgetCensoringV1.model_validate({**payload, "exhausted_models": []})
    assert BudgetCensoringV1.model_validate(
        {**payload, "exhausted_models": [], "campaign_budget_exhausted": True}
    ).paired_task_ids("trialdev") == frozenset(record.runs[0].scheduled_task_ids) - {"3-0-evidence_analysis"}
    with pytest.raises(ValueError, match="differ from their runs"):
        BudgetCensoringV1.model_validate({**payload, "warnings": ["edited"]})
    with pytest.raises(ValueError, match="unique scheduled tasks"):
        _censoring(a=("not-scheduled",), b=())
    assert BudgetCensoringV1.model_validate_json(record.model_dump_json()) == record


@pytest.mark.parametrize(
    "condition,reason",
    [
        ("stratum", "missing_required_stratum"),
        ("worlds", "insufficient_independent_worlds"),
        ("pending", "missing_scheduled_outcome"),
        ("zero", None),
    ],
)
def test_trialdev_headlines_disclose_missing_support(condition, reason):
    """A supported zero is a score; incomplete release support is unavailable with explicit denominators."""
    from trialagentbench_verifier.development_campaign import trialdev_leaderboard_rows

    source = authored_ranking("a")
    if condition == "stratum":
        source = source.model_copy(update={"items": tuple(row for row in source.items if row.asset_count == 3)})
    elif condition == "worlds":
        world = source.items[0].world_id
        source = source.model_copy(update={"items": tuple(row for row in source.items if row.world_id != world)})
    elif condition == "pending":
        source = _unrun(source, {source.items[0].task_id})
    else:
        source = source.model_copy(
            update={"items": tuple(row.model_copy(update={"net_development_value_usd": 0.0}) for row in source.items)}
        )
    rows = trialdev_leaderboard_rows((source,), required_strata=STRATA, policy=CampaignAnalysisV1())
    row = next(row for row in rows if row.metric == "net_development_value_usd")
    assert row.unavailable_reason == reason
    assert row.uncertainty_unavailable_reason == reason
    assert row.scheduled_tasks == len([item for item in source.items if item.stream == "portfolio_management"])
    if reason:
        assert row.estimate is row.interval_low is row.interval_high is row.standard_error is None
        assert row.warnings and "required strata" in row.warnings[0]
        assert row.minimum_worlds_per_stratum == 2
    else:
        assert row.estimate == row.interval_low == row.interval_high == 0.0
        assert not row.warnings
    if condition == "stratum":
        assert row.population == "partial_release_strata" and row.missing_strata == STRATA[1:]


@pytest.mark.parametrize("seed", range(20))
def test_random_report_distortions_preserve_primary_order_in_all_trialdev_companions(seed):
    """Random honest/distorted report panels retain their primary ordering at every level."""
    from statistics import mean

    import numpy as np

    from trialagentbench_verifier.development import ranking_means
    from trialagentbench_verifier.development_campaign import trialdev_leaderboard_rows
    from trialagentbench_verifier.numerical_scoring import numerical_loss

    rng = np.random.RandomState(seed)
    truths = rng.uniform(-1, 1, size=12)
    shift, width = rng.uniform(-0.5, 0.5), rng.uniform(0.3, 1.4)
    panels = []
    for model, estimate, lower, upper in (
        ("honest", 0, -0.95, 0.95),
        ("distorted", shift, shift - width, shift + width),
    ):
        losses = tuple(
            numerical_loss(estimate=estimate, lower=lower, upper=upper, target=truth, scale=1, confidence_level=0.95)
            for truth in truths
        )
        run = single_stratum(model, losses)
        (summary,) = ranking_means(list(run.items))
        (headline,) = trialdev_leaderboard_rows(
            (run,), required_strata=STRATA[:1], policy=CampaignAnalysisV1(bootstrap_repetitions=100)
        )
        primary = mean(losses)
        assert summary.mean == pytest.approx(primary)
        assert summary.bounded_recovery == pytest.approx(1 / (1 + primary))
        assert headline.estimate == pytest.approx(summary.bounded_recovery)
        panels.append((primary, headline.estimate))
    honest, distorted = panels
    assert (honest[0] < distorted[0]) == (honest[1] > distorted[1])


def test_oracle_capture_uses_ratio_of_means_and_preserves_failure_costs(tmp_path):
    from trialagentbench_verifier.development_campaign import (
        TrialDevOracleValuesV1,
        TrialDevOracleValueV1,
        bind_trialdev_oracle_values,
        load_trialdev_oracle_values,
        trialdev_leaderboard_rows,
    )

    run = single_stratum("a", (0.0, 0.0, 0.0, 0.0))
    run = run.model_copy(
        update={
            "items": tuple(
                row.model_copy(
                    update={
                        "stream": "portfolio_management",
                        "authored_programme": True,
                        "net_development_value_usd": value,
                        "execution_status": "model_noncompletion" if value < 0 else "completed",
                    }
                )
                for row, value in zip(run.items, (10.0, 30.0, -4.0, 0.0), strict=True)
            )
        }
    )
    oracle = TrialDevOracleValuesV1(
        release_id=run.release_id,
        release_manifest_sha256=run.release_manifest_sha256,
        policy_source_sha256="b" * 64,
        source_artifacts={"construction.json": "c" * 64},
        values=tuple(
            TrialDevOracleValueV1(
                task_id=row.task_id, world_id=row.world_id, net_value_usd=value, reference_value_usd=value / 2
            )
            for row, value in zip(run.items, (20.0, 100.0, 0.0, -1.0), strict=True)
        ),
    )
    path = tmp_path / "oracle.json"
    path.write_text(oracle.model_dump_json())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_trialdev_oracle_values(path, expected_sha256=digest) == oracle
    with pytest.raises(ValueError, match="pinned digest"):
        load_trialdev_oracle_values(path, expected_sha256="0" * 64)
    bound = bind_trialdev_oracle_values(run, oracle)
    rows = trialdev_leaderboard_rows((bound,), required_strata=STRATA[:1], policy=CampaignAnalysisV1())
    capture = next(row for row in rows if row.metric == "oracle_value_capture_pct")
    assert capture.estimate == pytest.approx(100 * 36 / 120)
    assert capture.scheduled_tasks == capture.scored_tasks == 4
    assert capture.independent_worlds == 4 and capture.nonpositive_oracle_worlds == 2
    assert capture.uncertainty_unavailable_reason == "zero_oracle_bootstrap_denominator"
    assert capture.standard_error is capture.interval_low is capture.interval_high is None
    # Duplicated linked conditions/repetitions do not change either world mean.
    assert (
        next(
            row
            for row in trialdev_leaderboard_rows(
                (bound, bound), required_strata=STRATA[:1], policy=CampaignAnalysisV1()
            )
            if row.metric == "oracle_value_capture_pct"
        ).estimate
        == capture.estimate
    )
    for invalid in (
        oracle.model_copy(update={"release_id": "other"}),
        oracle.model_copy(update={"values": oracle.values[:1]}),
        oracle.model_copy(
            update={"values": (oracle.values[0].model_copy(update={"world_id": "other"}), *oracle.values[1:])}
        ),
    ):
        with pytest.raises(ValueError, match="release|scheduled programme|different world"):
            bind_trialdev_oracle_values(run, invalid)


@pytest.mark.parametrize("values", [(0.0, 0.0), (-1.0, 0.0)])
def test_zero_oracle_denominator_is_explicit(values):
    from trialagentbench_verifier.development_campaign import trialdev_leaderboard_rows

    run = single_stratum("a", (0.0, 0.0))
    run = run.model_copy(
        update={
            "items": tuple(
                row.model_copy(
                    update={
                        "stream": "portfolio_management",
                        "authored_programme": True,
                        "net_development_value_usd": -1.0,
                        "oracle_development_value_usd": value,
                    }
                )
                for row, value in zip(run.items, values, strict=True)
            )
        }
    )
    capture = next(
        row
        for row in trialdev_leaderboard_rows((run,), required_strata=STRATA[:1], policy=CampaignAnalysisV1())
        if row.metric == "oracle_value_capture_pct"
    )
    assert capture.estimate is capture.standard_error is None
    assert capture.unavailable_reason == "zero_oracle_value" and capture.nonpositive_oracle_worlds == 2


def test_oracle_ratio_bootstrap_resamples_numerator_and_denominator_together():
    from trialagentbench_verifier.development_campaign import _capture_bootstrap

    groups = {STRATA[0]: [10.0, 20.0], STRATA[1]: [100.0, 200.0, 300.0, 400.0]}
    denominators = {key: [2 * x for x in values] for key, values in groups.items()}
    estimate, low, high, error, reason = _capture_bootstrap(groups, denominators, CampaignAnalysisV1())
    assert estimate == low == high == 50.0 and error == 0.0 and reason is None


def test_reference_developer_uses_the_pinned_oracle_panel():
    from trialagentbench_verifier.development_campaign import (
        TrialDevOracleValuesV1,
        TrialDevOracleValueV1,
        bind_trialdev_oracle_values,
        paired_trialdev_results,
    )

    run = single_stratum("a", (0.0, 0.0))
    run = run.model_copy(
        update={
            "items": tuple(
                row.model_copy(
                    update={
                        "stream": "portfolio_management",
                        "authored_programme": True,
                        "net_development_value_usd": 0.0,
                    }
                )
                for row in run.items
            )
        }
    )
    oracle = TrialDevOracleValuesV1(
        release_id=run.release_id,
        release_manifest_sha256=run.release_manifest_sha256,
        policy_source_sha256="b" * 64,
        source_artifacts={"construction.json": "c" * 64},
        values=tuple(
            TrialDevOracleValueV1(
                task_id=row.task_id, world_id=row.world_id, net_value_usd=100.0, reference_value_usd=50.0
            )
            for row in run.items
        ),
    )
    bound = bind_trialdev_oracle_values(run, oracle)
    _, _, rows = paired_trialdev_results(
        {"job": bound}, censoring=None, required_strata=STRATA[:1], policy=CampaignAnalysisV1(), oracle=oracle
    )
    assert next(r for r in rows if r.metric == "oracle_value_capture_pct" and r.model_id == "a").estimate == 0.0
    reference = next(r for r in rows if r.metric == "oracle_value_capture_pct" and r.model_id == "policy:reference")
    assert reference.estimate == reference.interval_low == reference.interval_high == 50.0
    assert reference.standard_error == 0.0


def test_oracle_zero_rule_is_applied_after_linked_conditions_are_averaged_within_world():
    from trialagentbench_verifier.development_campaign import trialdev_leaderboard_rows

    run = single_stratum("a", (0.0, 0.0, 0.0, 0.0))
    run = run.model_copy(
        update={
            "items": tuple(
                row.model_copy(
                    update={
                        "world_id": f"world-{i//2}",
                        "stream": "portfolio_management",
                        "authored_programme": True,
                        "net_development_value_usd": value,
                        "oracle_development_value_usd": oracle,
                    }
                )
                for i, (row, value, oracle) in enumerate(
                    zip(run.items, (20.0, 30.0, 20.0, 30.0), (100.0, -50.0, 100.0, -50.0), strict=True)
                )
            )
        }
    )
    capture = next(
        row
        for row in trialdev_leaderboard_rows((run,), required_strata=STRATA[:1], policy=CampaignAnalysisV1())
        if row.metric == "oracle_value_capture_pct"
    )
    assert capture.estimate == capture.interval_low == capture.interval_high == 100.0
    assert capture.independent_worlds == 2 and capture.nonpositive_oracle_worlds == 0
