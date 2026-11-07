"""Campaign point estimates and intervals share canonical finite-cell weights."""

import math
import statistics

import pytest

from trialagentbench_verifier.campaign import (
    EffectState,
    TrialEvalCampaignCellV1,
    _bootstrap,
    _cell_mean,
    minimal_detectable_difference_v1,
)
from trialagentbench_verifier.scorer import current_scorer

CELL = "cell_" + "1" * 20
SERIES = "series_" + "1" * 20


def test_unbalanced_effect_labels_resample_worlds_within_the_series() -> None:
    """Effect labels are descriptive; worlds of a series are exchangeable under the continuous prior."""
    values: dict[tuple[str, EffectState, str], float] = {
        (CELL, "favourable", "a"): 0,
        (CELL, "favourable", "b"): 0,
        (CELL, "near_null", "c"): 1,
        (CELL, "unfavourable", "d"): 1,
    }
    assert _cell_mean(values) == 0.5
    cell = TrialEvalCampaignCellV1(cell_ref=CELL, series_ref=SERIES)
    estimate, low, high, p_value, _ = _bootstrap(values, cells=(cell,), resamples=200, confidence_level=0.95, seed=7)
    assert estimate == 0.5 and p_value is not None
    # The small-sample rescaling may carry an endpoint slightly past the observed range.
    assert low is not None and high is not None and low < 0.5 < high


def test_a_singleton_series_withholds_its_interval() -> None:
    values: dict[tuple[str, EffectState, str], float] = {(CELL, "favourable", "a"): 0.3}
    cell = TrialEvalCampaignCellV1(cell_ref=CELL, series_ref=SERIES)
    assert _bootstrap(values, cells=(cell,), resamples=100, confidence_level=0.95, seed=7) == (
        pytest.approx(0.3),
        None,
        None,
        None,
        None,
    )


@pytest.mark.parametrize("worlds", [2, 4])
def test_small_series_bootstrap_has_the_unbiased_standard_error(worlds: int) -> None:
    """Rescaled draws give the stratum mean the usual s / sqrt(n), not the plug-in (n - 1) / n of its variance."""
    observed = [1.0, 4.0, 2.0, 7.0][:worlds]
    values: dict[tuple[str, EffectState, str], float] = {
        (CELL, "favourable", f"w{index}"): value for index, value in enumerate(observed)
    }
    cell = TrialEvalCampaignCellV1(cell_ref=CELL, series_ref=SERIES)
    result = _bootstrap(values, cells=(cell,), resamples=40_000, confidence_level=0.95, seed=11)
    expected = statistics.stdev(observed) / math.sqrt(worlds)
    assert result.standard_error == pytest.approx(expected, rel=0.03)


def test_nested_cells_of_one_series_resample_their_own_worlds() -> None:
    """A series whose cells present disjoint worlds (the withheld design) is resampled within each cell."""
    other = "cell_" + "2" * 20
    values: dict[tuple[str, EffectState, str], float] = {
        (CELL, "favourable", "a"): 0.0,
        (CELL, "near_null", "b"): 2.0,
        (other, "favourable", "c"): 10.0,
        (other, "unfavourable", "d"): 14.0,
    }
    cells = (
        TrialEvalCampaignCellV1(cell_ref=CELL, series_ref=SERIES),
        TrialEvalCampaignCellV1(cell_ref=other, series_ref=SERIES),
    )
    result = _bootstrap(values, cells=cells, resamples=40_000, confidence_level=0.95, seed=3)
    # Equal-cell mean of independent cell means: var = (s1^2 / 2 + s2^2 / 2) / 4.
    expected = math.sqrt((statistics.variance([0.0, 2.0]) / 2 + statistics.variance([10.0, 14.0]) / 2) / 4)
    assert result.estimate == pytest.approx(6.5)
    assert result.standard_error == pytest.approx(expected, rel=0.03)


def test_cells_sharing_some_but_not_all_worlds_are_refused() -> None:
    other = "cell_" + "2" * 20
    values: dict[tuple[str, EffectState, str], float] = {
        (CELL, "favourable", "a"): 0.0,
        (CELL, "favourable", "b"): 1.0,
        (other, "favourable", "b"): 2.0,
        (other, "favourable", "c"): 3.0,
    }
    cells = (
        TrialEvalCampaignCellV1(cell_ref=CELL, series_ref=SERIES),
        TrialEvalCampaignCellV1(cell_ref=other, series_ref=SERIES),
    )
    with pytest.raises(ValueError, match="base worlds differ across cells"):
        _bootstrap(values, cells=cells, resamples=100, confidence_level=0.95, seed=3)


def test_minimal_detectable_difference_is_about_2_8_standard_errors() -> None:
    assert minimal_detectable_difference_v1(1.0, alpha=0.05) == pytest.approx(2.8016, abs=1e-3)
    with pytest.raises(ValueError):
        minimal_detectable_difference_v1(-1.0, alpha=0.05)


def test_trialeval_leaderboard_leads_with_decision_consequence_and_keeps_loss_secondary():
    from types import SimpleNamespace

    from trialagentbench_verifier.campaign import TrialEvalLossRankingV1
    from trialagentbench_verifier.public_results import trialeval_leaderboard_rows

    ranking = TrialEvalLossRankingV1(
        model_id="model-a",
        target_kind="point",
        scheduled_attempts=4,
        scored_attempts=3,
        scheduled_targets=4,
        scored_targets=3,
        unscored_targets=1,
        target_failure_counts={"not_started": 1},
        observed_mean_loss=0.5,
        scored_worlds=2,
        scored_cell_refs=(CELL,),
        independent_worlds=2,
        cell_refs=(CELL,),
        mean_loss=None,
        display_recovery=None,
        interval_low=None,
        interval_high=None,
        bounded_recovery=None,
        bounded_recovery_low=None,
        bounded_recovery_high=None,
    )

    def summary(metric_id: str, estimate: float, low: float, high: float, context: str = "C5"):  # type: ignore[no-untyped-def]
        return SimpleNamespace(
            model_id="model-a",
            context=context,
            metric=SimpleNamespace(metric_id=metric_id),
            scheduled_attempts=4,
            observed_attempts=4,
            equal_cell_effect_mean=estimate,
            interval_low=low,
            interval_high=high,
            standard_error=(high - low) / 3.92,
        )

    summaries = (
        summary("decision_regret_musd", 12.5, 3.0, 22.0),
        summary("consequential_error", 0.25, 0.0, 0.5),
        summary("decision_regret_musd", 99.0, 90.0, 110.0, context="C2"),
    )
    regret, errors, loss = trialeval_leaderboard_rows(
        SimpleNamespace(
            scorer=current_scorer(),
            between_model_contrasts=(),
            loss_rankings=(ranking,),
            metric_summaries=summaries,
            unavailable_reason=None,
        )
    )
    # The headline prices answers by their decisions on the main C5 view; missing answers drive the
    # prior-only decision, so the decision rows count every scheduled answer.
    assert (regret.metric, regret.estimate, regret.interval_low, regret.interval_high, regret.scored_tasks) == (
        "decision_regret_musd",
        12.5,
        3.0,
        22.0,
        4,
    )
    assert regret.standard_error == pytest.approx(19.0 / 3.92)
    assert (errors.metric, errors.estimate, errors.direction) == ("consequential_error_rate", 0.25, "lower_is_better")
    # One missing attempt withholds the complete-panel loss, the secondary row.
    assert (loss.metric, loss.estimate, loss.scored_tasks, loss.scheduled_tasks) == ("mean_loss", None, 3, 4)
    assert "minimal_detectable_difference" not in regret.model_dump()


def test_reported_endpoints_stay_inside_the_metric_support() -> None:
    """Rescaled draws may pass a rate's bounds; the reported interval is clipped, its standard error is not."""
    values: dict[tuple[str, EffectState, str], float] = {
        (CELL, "favourable", "a"): 0.0,
        (CELL, "favourable", "b"): 1.0,
    }
    cell = TrialEvalCampaignCellV1(cell_ref=CELL, series_ref=SERIES)
    free = _bootstrap(values, cells=(cell,), resamples=2000, confidence_level=0.95, seed=5)
    clipped = _bootstrap(values, cells=(cell,), resamples=2000, confidence_level=0.95, seed=5, support=(0.0, 1.0))
    assert free.low is not None and free.low < 0.0 and free.high is not None and free.high > 1.0
    assert (clipped.low, clipped.high) == (0.0, 1.0)
    assert clipped.standard_error == free.standard_error


def test_public_campaign_retains_c5_decision_uncertainty_without_diagnostics_or_private_labels():
    from types import SimpleNamespace

    from trialagentbench_verifier.campaign import TrialEvalBetweenModelContrastV1, TrialEvalMetricDefinitionV1
    from trialagentbench_verifier.public_results import public_trialeval_campaign

    def contrast(metric: str, context: str):
        return TrialEvalBetweenModelContrastV1(
            scorer=current_scorer(),
            minuend_model_id="model-a",
            subtrahend_model_id="model-b",
            context=context,
            metric=TrialEvalMetricDefinitionV1(
                metric_id=metric,
                domain="numerical_performance",
                direction="lower_is_better",
                target_kind="point",
            ),
            status="estimated",
            scheduled_pairs=4,
            observed_pairs=4,
            common_cell_refs=(CELL,),
            estimate=0.5,
            interval_low=0.1,
            interval_high=0.9,
            standard_error=0.2,
            minimal_detectable_difference=0.56,
        )

    retained = (
        contrast("point_signal_loss", "all"),
        contrast("point_signal_loss", "C1"),
        contrast("decision_regret_musd", "C5"),
        contrast("consequential_error", "C5"),
    )
    excluded = (
        contrast("decision_regret_musd", "C1"),
        contrast("consequential_error", "all"),
        contrast("primary_point_normalized_absolute_error", "C5"),
        contrast("plan_execution_accuracy", "C1"),
    )
    report = SimpleNamespace(
        scorer=current_scorer(),
        checksum="a" * 64,
        release_id="release",
        campaign_plan_checksum="b" * 64,
        source_archive_sha256="c" * 64,
        scientific_analysis_status="complete",
        unavailable_reason=None,
        confidence_level=0.95,
        bootstrap_resamples=100,
        bootstrap_seed=7,
        loss_rankings=(),
        cell_loss_summaries=(),
        difficulty_summaries=(),
        target_summaries=(),
        between_model_contrasts=(*retained, *excluded),
        metric_summaries=(),
        oracle_provenance=("private oracle",),
        cell_labels=("fixture-cell-label",),
        tiers=("A1",),
    )
    public = public_trialeval_campaign(report, split="public")
    assert public.between_model_contrasts == retained
    payload = public.model_dump_json()
    assert CELL in payload
    assert not any(value in payload for value in ("fixture-cell-label", '"tiers"', "oracle_provenance", "cell_labels"))
    assert all(
        row.standard_error == 0.2 and row.minimal_detectable_difference == 0.56
        for row in public.between_model_contrasts
    )


def test_metric_support_follows_the_definition_not_the_sample() -> None:
    """A continuous loss whose values happen to be 0 and 1 keeps an open upper bound; a rate is bounded."""
    from trialagentbench_verifier.campaign import TrialEvalMetricDefinitionV1, _metric_support

    def metric(metric_id: str) -> TrialEvalMetricDefinitionV1:
        return TrialEvalMetricDefinitionV1(
            metric_id=metric_id, domain="numerical_performance", direction="lower_is_better", target_kind="point"
        )

    assert _metric_support(metric("point_signal_loss")) == (0.0, None)
    assert _metric_support(metric("consequential_error")) == (0.0, 1.0)
    # Replay regret is signed: an answer can beat the scripted reference at the truth.
    assert _metric_support(metric("decision_regret_musd_value_x2")) == (None, None)
    assert _metric_support(metric("some_signed_diagnostic")) == (None, None)


@pytest.mark.parametrize("operation", ["_summaries", "_between_models", "_c_families"])
def test_additive_material_metric_preserves_existing_aggregate_bytes_and_bootstrap_seeds(monkeypatch, operation):
    """Adding the rate cannot change old estimates, intervals, SEs, order or seed declarations."""
    from types import SimpleNamespace

    from trialagentbench_verifier import campaign

    metrics = tuple(
        campaign.TrialEvalMetricDefinitionV1(
            metric_id=name, domain="decision_consequence", direction="lower_is_better", target_kind="point"
        )
        for name in ("consequential_error", "replay_answered_share", "material_decision_error_rate")
    )
    observations = []
    primaries = []
    for model in ("a", "b"):
        for world in range(3):
            for context in campaign.CONTEXTS_V1:
                identity = f"{model}-{world}-{context}"
                plan = SimpleNamespace(
                    model_id=model,
                    context=context,
                    target_kind="point",
                    effect_measure="mean_difference",
                    result_unit="score",
                    cell_ref=CELL,
                    effect_state="favourable",
                    base_world_id=f"w{world}",
                    attempt_id=identity,
                )
                primaries.append(SimpleNamespace(plan=plan))
                for metric in metrics:
                    value = float((world + (model == "b")) % 2)
                    observations.append(
                        campaign.TrialEvalMetricObservationV1(
                            attempt_id=identity,
                            model_id=model,
                            task_id=identity,
                            trial_id=identity,
                            base_world_id=f"w{world}",
                            cell_ref=CELL,
                            effect_state="favourable",
                            context=context,
                            metric=metric,
                            value=value,
                        )
                    )
    plan = SimpleNamespace(
        contexts=campaign.CONTEXTS_V1,
        cells=(TrialEvalCampaignCellV1(cell_ref=CELL, series_ref=SERIES),),
        model_comparisons=(("a", "b"),),
    )
    monkeypatch.setattr(campaign, "_scientific_primaries", lambda bound: tuple(primaries))
    original = tuple(row for row in observations if row.metric.metric_id != "material_decision_error_rate")
    monkeypatch.setattr(campaign, "_observations", lambda bound: original)
    function = getattr(campaign, operation)
    baseline = function(tuple(primaries), plan, resamples=100, confidence=0.95, seed=73)
    monkeypatch.setattr(campaign, "_observations", lambda bound: tuple(observations))
    extended = function(tuple(primaries), plan, resamples=100, confidence=0.95, seed=73)
    others = tuple(row for row in extended if row.metric.metric_id != "material_decision_error_rate")
    assert [row.model_dump_json() for row in others] == [row.model_dump_json() for row in baseline]
    added = [row for row in extended if row.metric.metric_id == "material_decision_error_rate"]
    assert added
    if operation == "_summaries":
        assert all(
            row.standard_error is not None and row.scheduled_attempts == row.observed_attempts == 3 for row in added
        )


def test_material_rate_is_an_additive_public_headline_with_its_own_uncertainty():
    """The new public rate is adjacent to the unchanged flip-only rate and carries its SE."""
    from types import SimpleNamespace

    from trialagentbench_verifier.public_results import trialeval_leaderboard_rows

    ranking = SimpleNamespace(
        target_kind="point",
        model_id="m",
        scheduled_attempts=3,
        scored_attempts=3,
        independent_worlds=3,
        mean_loss=0.1,
        interval_low=0.0,
        interval_high=0.2,
        standard_error=0.03,
    )
    summaries = tuple(
        SimpleNamespace(
            model_id="m",
            context="C5",
            metric=SimpleNamespace(metric_id=name),
            scheduled_attempts=3,
            observed_attempts=3,
            equal_cell_effect_mean=value,
            interval_low=0.0,
            interval_high=1.0,
            standard_error=0.1,
        )
        for name, value in (
            ("decision_regret_musd", 2.0),
            ("consequential_error", 0.25),
            ("material_decision_error_rate", 0.5),
        )
    )
    report = SimpleNamespace(
        scorer=current_scorer(),
        between_model_contrasts=(),
        loss_rankings=(ranking,),
        metric_summaries=summaries,
        unavailable_reason=None,
    )
    rows = trialeval_leaderboard_rows(report)
    assert [row.metric for row in rows] == [
        "decision_regret_musd",
        "consequential_error_rate",
        "material_decision_error_rate",
        "mean_loss",
    ]
    assert rows[2].estimate == 0.5 and rows[2].standard_error == 0.1
