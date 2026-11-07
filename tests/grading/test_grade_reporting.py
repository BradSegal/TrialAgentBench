"""Tests for denominator-complete TrialEval grade summaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from trialagentbench_verifier.scorer import current_scorer

from trialagentbench.contracts.release_scope import (
    TRIALEVAL_WITHHELD_CONTEXT_V1,
    TRIALEVAL_WITHHELD_TRIAL_COUNT_V1,
)
from trialagentbench.grading.reporting import (
    TrialEvalExpectedItemV1,
    build_trialeval_balanced_summary_v1,
    write_trialeval_grade_summary,
)

_CELL_KEYS = tuple(f"cell_{index:020x}" for index in range(21))
"""The 21 public cells, by opaque reference."""
_WITHHELD_ONLY_CELL_KEYS = (f"cell_{21:020x}",)
"""The withheld split's held-out cell."""
_CONTEXTS = ("C1", "C2", "C3", "C4", "C5")


def _graded_row(
    *,
    item_id: str = "task-1",
    cell_ref: str = _CELL_KEYS[0],
    context_tier: str = "C1",
    effect_measure: str = "mean_difference",
    result_unit: str = "points",
) -> dict[str, object]:
    return {
        "item_id": item_id,
        "signal_loss": 0.5,
        "cell_ref": cell_ref,
        "effect_state": "favourable",
        "design_tier": "D1",
        "design_subtype": "parallel",
        "context_tier": context_tier,
        "effect_measure": effect_measure,
        "result_unit": result_unit,
        "model": "test-model",
        "agent_status": "success",
        "turns_used": 2,
        "target_kind": "point",
        "conclusion_class": "favors_treatment",
        "completion": "satisfied",
        "question_alignment": "observed",
        "evidence_eligibility": "satisfied",
        "reproducibility": "satisfied",
        "estimand_performance": "observed",
        "uncertainty": "observed",
        "decision": "not_applicable",
        "failure_origins": [],
        "native_signed_error": 0.0,
        "native_absolute_error": 0.0,
        "native_squared_error": 0.0,
        "normalized_absolute_error": 0.0,
        "normalized_squared_error": 0.0,
        "interval_covers_target": True,
        "interval_width": 1.0,
        "proper_interval_score": 1.0,
        "program_replay_status": "reproduced",
        "program_elapsed_seconds": 0.1,
    }


def _expected(row: dict[str, object]) -> TrialEvalExpectedItemV1:
    return TrialEvalExpectedItemV1(
        item_id=str(row["item_id"]),
        trial_id=f"trial-{row['item_id']}",
        cell_ref=str(row["cell_ref"]),
        effect_state=str(row["effect_state"]),
        design_tier=str(row["design_tier"]),
        context_tier=str(row["context_tier"]),
        effect_measure=str(row["effect_measure"]),
        result_unit=str(row["result_unit"]),
    )


def _canonical_rows() -> tuple[list[dict[str, object]], tuple[TrialEvalExpectedItemV1, ...]]:
    rows: list[dict[str, object]] = []
    expected: list[TrialEvalExpectedItemV1] = []
    for cell_index, cell_ref in enumerate(_CELL_KEYS):
        for context in _CONTEXTS:
            for replicate in range(1, 5):
                item_id = f"{cell_ref}-{context}-{replicate}"
                row = _graded_row(
                    item_id=item_id,
                    cell_ref=cell_ref,
                    context_tier=context,
                )
                row["normalized_absolute_error"] = float(cell_index)
                rows.append(row)
                expected.append(_expected(row))
    return rows, tuple(expected)


def test_grade_summary_retains_evaluator_errors_in_declared_denominator(tmp_path: Path) -> None:
    first = _graded_row()
    second = _graded_row(item_id="task-2")
    write_trialeval_grade_summary(
        rows=[first],
        expected_items=(_expected(first), _expected(second)),
        scheduled_item_ids=("task-1", "task-2"),
        evaluator_error_item_ids=("task-2",),
        output_dir=tmp_path,
    )

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["scorer"] == current_scorer().model_dump(mode="json")
    assert summary["n_items"] == 2
    assert summary["n_graded_items"] == 1
    assert summary["n_scored_items"] == 1
    assert summary["n_evaluator_errors"] == 1
    assert summary["evaluator_error_item_ids"] == ["task-2"]
    assert summary["component_counts"]["estimand_performance"] == {"observed": 1}
    with (tmp_path / "results_full.csv").open(newline="") as handle:
        exported = list(csv.DictReader(handle))
    assert json.loads(exported[0]["scorer"]) == current_scorer().model_dump(mode="json")
    assert len(exported) == 1 and float(exported[0]["signal_loss"]) == 0.5
    rendered = (tmp_path / "RESULTS_SUMMARY.md").read_text(encoding="utf-8")
    assert "Declared denominator: **2**" in rendered
    assert "Evaluator errors: **1/2**" in rendered
    assert "Numerical losses available: **1/2**" in rendered
    assert f"| {_CELL_KEYS[0]} / C1 | unavailable | 0.5 | unavailable | 1 / 2 |" in rendered
    balanced = json.loads((tmp_path / "balanced_summary.json").read_text(encoding="utf-8"))
    (cell,) = balanced["context_cells"]
    recovery = next(row for row in cell["comparable_metrics"] if row["quantity_id"] == "bounded_recovery")
    # The companion has exactly the primary loss's availability.
    assert recovery == {
        "quantity_id": "bounded_recovery",
        "scheduled_count": 2,
        "observed_count": 1,
        "mean": None,
    }


def test_grade_summary_allows_every_item_to_have_an_evaluator_error(tmp_path: Path) -> None:
    first = _graded_row()
    second = _graded_row(item_id="task-2")
    write_trialeval_grade_summary(
        rows=[],
        expected_items=(_expected(first), _expected(second)),
        scheduled_item_ids=("task-1", "task-2"),
        evaluator_error_item_ids=("task-1", "task-2"),
        output_dir=tmp_path,
    )

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_items"] == 2
    assert summary["n_graded_items"] == 0
    assert summary["n_scored_items"] == 0
    assert summary["n_evaluator_errors"] == 2
    assert (tmp_path / "results_full.csv").read_text(encoding="utf-8").count("\n") == 1


def test_balanced_summary_averages_trials_then_contexts() -> None:
    first = _graded_row()
    second = {**_graded_row(item_id="task-2"), "completion": "not_satisfied"}
    third = _graded_row(item_id="task-3", context_tier="C2")

    summary = build_trialeval_balanced_summary_v1(
        [first, second, third],
        expected_items=tuple(_expected(row) for row in (first, second, third)),
    )

    cell = summary.regime_cells[0]
    completion = next(value for value in cell.components if value.quantity_id == "completion")
    assert cell.context_count == 2
    assert completion.mean == pytest.approx(0.75)
    assert completion.mean != pytest.approx(2 / 3)
    assert summary.effect_state_strata[0].effect_state == "favourable"
    assert summary.effect_state_strata[0].scheduled_trial_count == 3
    assert summary.effect_state_strata[0].conclusion_class_counts == {"favors_treatment": 3}
    assert not summary.complete_regime_grid
    assert not summary.overall_components


def test_balanced_summary_weights_every_cell_equally() -> None:
    rows, expected = _canonical_rows()
    summary = build_trialeval_balanced_summary_v1(rows, expected_items=expected)

    assert summary.complete_regime_grid
    assert summary.observed_regime_cell_count == len(_CELL_KEYS)
    normalized = next(value for value in summary.overall_metrics if value.quantity_id == "normalized_absolute_error")
    assert normalized.scheduled_count == len(_CELL_KEYS)
    assert normalized.observed_count == len(_CELL_KEYS)
    assert normalized.mean == pytest.approx(sum(range(len(_CELL_KEYS))) / len(_CELL_KEYS))


def test_withheld_split_c5_view_with_held_out_cell_receives_overall_scores() -> None:
    """The withheld split releases only C5 and one cell per law: a series' cells partition its four laws."""
    rows, expected = _canonical_rows()
    # Seventeen series of four laws: most hold one public cell, five hold two, and the held-out cell joins the first.
    series = [
        index for index, count in enumerate((1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1)) for _ in range(count)
    ]
    cells_by_series: dict[int, list[str]] = {}
    for cell, owner in zip((*_CELL_KEYS, *_WITHHELD_ONLY_CELL_KEYS), (*series, 0), strict=True):
        cells_by_series.setdefault(owner, []).append(cell)
    laws = {
        cell: range(1 + position * 2, 3 + position * 2) if len(cells) == 2 else range(1, 5)
        for cells in cells_by_series.values()
        for position, cell in enumerate(sorted(cells))
    }
    selected = [
        row
        for row in rows
        if row["context_tier"] == TRIALEVAL_WITHHELD_CONTEXT_V1
        and int(str(row["item_id"]).rsplit("-", 1)[1]) in laws[str(row["cell_ref"])]
    ]
    for cell_ref in _WITHHELD_ONLY_CELL_KEYS:
        for replicate in laws[cell_ref]:
            row = _graded_row(item_id=f"{cell_ref}-C5-{replicate}", cell_ref=cell_ref, context_tier="C5")
            row["normalized_absolute_error"] = float(len(_CELL_KEYS))
            selected.append(row)
    identities = {str(row["item_id"]) for row in selected}
    inventory = (
        *(item for item in expected if item.item_id in identities),
        *(_expected(row) for row in selected if str(row["cell_ref"]) in _WITHHELD_ONLY_CELL_KEYS),
    )
    summary = build_trialeval_balanced_summary_v1(selected, expected_items=inventory)

    assert summary.declared_trial_count == TRIALEVAL_WITHHELD_TRIAL_COUNT_V1 == 68
    assert summary.observed_regime_cell_count == len(_CELL_KEYS) + len(_WITHHELD_ONLY_CELL_KEYS)
    assert summary.complete_regime_grid
    assert {row.context_tier for row in summary.context_cells} == {"C5"}
    assert {row.scheduled_trial_count for row in summary.context_cells} == {2, 4}
    normalized = next(value for value in summary.overall_metrics if value.quantity_id == "normalized_absolute_error")
    cell_count = len(_CELL_KEYS) + len(_WITHHELD_ONLY_CELL_KEYS)
    assert normalized.mean == pytest.approx(sum(range(cell_count)) / cell_count)


def test_balanced_summary_rejects_unbalanced_replicates_as_incomplete() -> None:
    rows, expected = _canonical_rows()
    dropped = str(rows[0]["item_id"])
    summary = build_trialeval_balanced_summary_v1(
        [row for row in rows if row["item_id"] != dropped],
        expected_items=tuple(item for item in expected if item.item_id != dropped),
    )

    assert not summary.complete_regime_grid
    assert not summary.overall_components


def test_balanced_summary_rejects_sparse_context_grid_as_incomplete() -> None:
    rows, expected = _canonical_rows()
    excluded = rows[0]["cell_ref"]
    kept_rows = [row for row in rows if not (row["context_tier"] == "C5" and row["cell_ref"] == excluded)]
    kept_ids = {str(row["item_id"]) for row in kept_rows}
    kept_expected = tuple(item for item in expected if item.item_id in kept_ids)

    summary = build_trialeval_balanced_summary_v1(kept_rows, expected_items=kept_expected)

    assert summary.observed_regime_cell_count == len(_CELL_KEYS)
    assert not summary.complete_regime_grid
    assert not summary.overall_components
    assert not summary.overall_metrics


def test_evaluator_error_reduces_overall_completion() -> None:
    rows, expected = _canonical_rows()
    failed_id = expected[0].item_id
    rows = [row for row in rows if row["item_id"] != failed_id]

    summary = build_trialeval_balanced_summary_v1(
        rows,
        expected_items=expected,
        evaluator_error_item_ids=(failed_id,),
    )

    completion = next(value for value in summary.overall_components if value.quantity_id == "completion")
    assert completion.mean == pytest.approx((20 * len(_CELL_KEYS) - 1) / (20 * len(_CELL_KEYS)))


def test_cross_family_overall_metrics_exclude_native_units() -> None:
    rows, expected = _canonical_rows()
    for index, row in enumerate(rows):
        if index % 2:
            row.update(effect_measure="restricted_mean_survival_time_difference", result_unit="days")
        else:
            row.update(effect_measure="risk_difference", result_unit="proportion")
        row["native_absolute_error"] = 100.0 if row["result_unit"] == "days" else 0.1
    expected = tuple(_expected(row) for row in rows)

    summary = build_trialeval_balanced_summary_v1(rows, expected_items=expected)

    assert "native_absolute_error" not in {metric.quantity_id for metric in summary.overall_metrics}
    strata = {(row.effect_measure, row.result_unit): row for row in summary.native_metric_strata}
    assert set(strata) == {
        ("restricted_mean_survival_time_difference", "days"),
        ("risk_difference", "proportion"),
    }
    days = next(
        metric
        for metric in strata[("restricted_mean_survival_time_difference", "days")].metrics
        if metric.quantity_id == "native_absolute_error"
    )
    proportion = next(
        metric
        for metric in strata[("risk_difference", "proportion")].metrics
        if metric.quantity_id == "native_absolute_error"
    )
    assert days.mean == 100.0
    assert proportion.mean == 0.1


@pytest.mark.parametrize("contexts", [("C5",), ("C2",), ("C2", "C5")])
def test_complete_selected_presentations_receive_balanced_overall_scores(contexts: tuple[str, ...]) -> None:
    rows, expected = _canonical_rows()
    selected = [row for row in rows if row["context_tier"] in contexts]
    identities = {str(row["item_id"]) for row in selected}
    summary = build_trialeval_balanced_summary_v1(
        selected, expected_items=tuple(item for item in expected if item.item_id in identities)
    )
    assert summary.complete_regime_grid
    assert summary.declared_trial_count == 4 * len(_CELL_KEYS) * len(contexts)
    normalized = next(value for value in summary.overall_metrics if value.quantity_id == "normalized_absolute_error")
    assert normalized.mean == pytest.approx(sum(range(len(_CELL_KEYS))) / len(_CELL_KEYS))


def test_primary_recovery_withholds_evaluator_faults_but_counts_model_failures() -> None:
    rows, expected = _canonical_rows()
    selected = [row for row in rows if row["context_tier"] == "C5"]
    identities = {str(row["item_id"]) for row in selected}
    inventory = tuple(item for item in expected if item.item_id in identities)
    complete = build_trialeval_balanced_summary_v1(selected, expected_items=inventory)
    assert next(m.mean for m in complete.overall_metrics if m.quantity_id == "signal_loss") == 0.5
    selected[0] = {**selected[0], "signal_loss": None}
    failed_model = build_trialeval_balanced_summary_v1(selected, expected_items=inventory)
    assert next(m.mean for m in failed_model.overall_metrics if m.quantity_id == "signal_loss") is None
    assert next(m.mean for m in failed_model.overall_metrics if m.quantity_id == "observed_signal_loss") == 0.5
    fault = build_trialeval_balanced_summary_v1(
        selected[1:], expected_items=inventory, evaluator_error_item_ids=(str(selected[0]["item_id"]),)
    )
    assert fault.evaluator_error_count == 1
    assert next(m.mean for m in fault.overall_metrics if m.quantity_id == "signal_loss") is None
    assert next(m.mean for m in fault.overall_metrics if m.quantity_id == "observed_signal_loss") == 0.5
    empty = build_trialeval_balanced_summary_v1(
        [{**row, "signal_loss": None} for row in selected], expected_items=inventory
    )
    assert next(m.mean for m in empty.overall_metrics if m.quantity_id == "observed_signal_loss") is None


@pytest.mark.parametrize("balanced", [False, True])
@pytest.mark.parametrize("seed", range(10))
def test_random_report_panels_transform_loss_after_each_balanced_aggregation(seed, balanced):
    """Companions preserve ordering through unequal trial, context and cell weights."""
    import numpy as np
    from trialagentbench_verifier.numerical_scoring import numerical_loss

    rng = np.random.RandomState(seed)
    rows, expected = _canonical_rows()
    if not balanced:
        rows = [row for index, row in enumerate(rows) if row["context_tier"] != "C1" or index % 2 == 0]
        expected = tuple(item for item in expected if item.item_id in {row["item_id"] for row in rows})
    truths = rng.uniform(-1, 1, size=len(rows))
    shift, width = rng.uniform(-0.5, 0.5), rng.uniform(0.3, 1.4)
    panels = []
    for estimate, lower, upper in ((0, -0.95, 0.95), (shift, shift - width, shift + width)):
        for row, truth in zip(rows, truths, strict=True):
            row["signal_loss"] = numerical_loss(
                estimate=estimate, lower=lower, upper=upper, target=truth, scale=1, confidence_level=0.95
            )
        summary = build_trialeval_balanced_summary_v1(rows, expected_items=expected)
        assert summary.complete_regime_grid == balanced
        levels = [
            *(cell.comparable_metrics for cell in summary.context_cells),
            *(cell.comparable_metrics for cell in summary.regime_cells),
        ]
        if balanced:
            levels.append(summary.overall_metrics)
        else:
            assert not summary.overall_metrics
        for level in levels:
            metrics = {value.quantity_id: value for value in level}
            primary, companion = metrics["signal_loss"], metrics["bounded_recovery"]
            assert companion.mean == pytest.approx(1 / (1 + primary.mean))
            assert (companion.observed_count, companion.scheduled_count) == (
                primary.observed_count,
                primary.scheduled_count,
            )
        overall = {value.quantity_id: value.mean for value in levels[-1]}
        panels.append((overall["signal_loss"], overall["bounded_recovery"]))
    honest, distorted = panels
    assert (honest[0] < distorted[0]) == (honest[1] > distorted[1])
