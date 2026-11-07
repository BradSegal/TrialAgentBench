"""Compact reporting for estimator-neutral TrialEval outcome grades."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from trialagentbench_verifier.numerical_scoring import bounded_recovery
from trialagentbench_verifier.scorer import CurrentScorer, current_scorer

from trialagentbench.contracts.release_scope import (
    TRIALEVAL_CONTEXT_COUNT_V1,
    TRIALEVAL_REGIME_CELL_COUNT_V1,
    TRIALEVAL_WITHHELD_ONLY_CELL_COUNT_V1,
)
from trialagentbench.io import canonical_payload_sha256, write_json


def _failure_origins(row: dict[str, object]) -> tuple[str, ...]:
    value = row["failure_origins"]
    if not isinstance(value, list):
        raise TypeError("failure_origins must be a list")
    if not all(isinstance(code, str) for code in value):
        raise TypeError("failure_origins entries must be strings")
    return tuple(value)


_COMPONENTS = (
    "completion",
    "question_alignment",
    "evidence_eligibility",
    "reproducibility",
    "estimand_performance",
    "uncertainty",
    "decision",
)

_COMPARABLE_METRICS = (
    "signal_loss",
    "observed_signal_loss",
    "bounded_recovery",
    "normalized_absolute_error",
    "normalized_squared_error",
    "interval_covers_target",
)
_NATIVE_METRICS = (
    "native_absolute_error",
    "interval_width",
    "proper_interval_score",
)
_CONCLUSION_CLASSES = (
    "favors_treatment",
    "favors_comparator",
    "no_clear_difference",
    "not_identified",
)


class _SummaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class TrialEvalAggregateValueV1(_SummaryModel):
    """One denominator-explicit component or metric aggregate."""

    quantity_id: str = Field(min_length=1)
    scheduled_count: int = Field(ge=1)
    observed_count: int = Field(ge=0)
    mean: float | None = None

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if self.observed_count > self.scheduled_count:
            raise ValueError("aggregate observations exceed their scheduled denominator")
        if self.observed_count == 0 and self.mean is not None:
            raise ValueError("aggregate mean requires observations")
        if self.observed_count == self.scheduled_count and self.mean is None:
            raise ValueError("complete aggregate observations require a mean")
        if self.quantity_id == "signal_loss" and self.observed_count < self.scheduled_count and self.mean is not None:
            raise ValueError("primary loss requires every scheduled observation")
        return self


class TrialEvalExpectedItemV1(_SummaryModel):
    """Released identity and score strata for one scheduled TrialEval item."""

    item_id: str = Field(min_length=1)
    trial_id: str = Field(min_length=1)
    cell_ref: str = Field(pattern=r"^cell_[0-9a-f]{20}$")
    effect_state: Literal["favourable", "near_null", "unfavourable"]
    design_tier: Literal["D1", "D2", "D3", "D4"]
    context_tier: Literal["C1", "C2", "C3", "C4", "C5"]
    effect_measure: str = Field(min_length=1)
    result_unit: str = Field(min_length=1)


class TrialEvalNativeMetricStratumV1(_SummaryModel):
    """Native-unit metrics retained within one effect-measure and unit stratum."""

    effect_measure: str = Field(min_length=1)
    result_unit: str = Field(min_length=1)
    scheduled_trial_count: int = Field(ge=1)
    graded_trial_count: int = Field(ge=0)
    evaluator_error_count: int = Field(ge=0)
    metrics: tuple[TrialEvalAggregateValueV1, ...]

    @model_validator(mode="after")
    def _counts_reconcile(self) -> TrialEvalNativeMetricStratumV1:
        if self.graded_trial_count + self.evaluator_error_count != self.scheduled_trial_count:
            raise ValueError("TrialEval native metric stratum counts do not reconcile")
        return self


class TrialEvalEffectStateStratumV1(_SummaryModel):
    """Outcome components and comparable metrics within one generating effect state."""

    effect_state: Literal["favourable", "near_null", "unfavourable"]
    scheduled_trial_count: int = Field(ge=1)
    graded_trial_count: int = Field(ge=0)
    evaluator_error_count: int = Field(ge=0)
    conclusion_class_counts: dict[str, int]
    components: tuple[TrialEvalAggregateValueV1, ...]
    comparable_metrics: tuple[TrialEvalAggregateValueV1, ...]

    @model_validator(mode="after")
    def _counts_reconcile(self) -> TrialEvalEffectStateStratumV1:
        if self.graded_trial_count + self.evaluator_error_count != self.scheduled_trial_count:
            raise ValueError("TrialEval effect-state counts do not reconcile")
        if not set(self.conclusion_class_counts).issubset(_CONCLUSION_CLASSES):
            raise ValueError("TrialEval effect-state summary contains an unknown conclusion class")
        if any(count < 0 for count in self.conclusion_class_counts.values()):
            raise ValueError("TrialEval conclusion counts cannot be negative")
        if sum(self.conclusion_class_counts.values()) > self.graded_trial_count:
            raise ValueError("TrialEval conclusion counts exceed graded results")
        return self


class TrialEvalContextRegimeCellSummaryV1(_SummaryModel):
    """Trial-average results within one released context and scenario cell."""

    cell_ref: str = Field(pattern=r"^cell_[0-9a-f]{20}$")
    design_tier: Literal["D1", "D2", "D3", "D4"]
    context_tier: Literal["C1", "C2", "C3", "C4", "C5"]
    scheduled_trial_count: int = Field(ge=1)
    graded_trial_count: int = Field(ge=0)
    evaluator_error_count: int = Field(ge=0)
    components: tuple[TrialEvalAggregateValueV1, ...]
    comparable_metrics: tuple[TrialEvalAggregateValueV1, ...]

    @model_validator(mode="after")
    def _counts_reconcile(self) -> TrialEvalContextRegimeCellSummaryV1:
        if self.graded_trial_count + self.evaluator_error_count != self.scheduled_trial_count:
            raise ValueError("TrialEval context-regime counts do not reconcile")
        return self


class TrialEvalRegimeCellSummaryV1(_SummaryModel):
    """Equal-context result for one canonical released scenario cell."""

    cell_ref: str = Field(pattern=r"^cell_[0-9a-f]{20}$")
    design_tier: Literal["D1", "D2", "D3", "D4"]
    context_count: int = Field(ge=1, le=TRIALEVAL_CONTEXT_COUNT_V1)
    scheduled_trial_count: int = Field(ge=1)
    graded_trial_count: int = Field(ge=0)
    evaluator_error_count: int = Field(ge=0)
    components: tuple[TrialEvalAggregateValueV1, ...]
    comparable_metrics: tuple[TrialEvalAggregateValueV1, ...]

    @model_validator(mode="after")
    def _counts_reconcile(self) -> TrialEvalRegimeCellSummaryV1:
        if self.graded_trial_count + self.evaluator_error_count != self.scheduled_trial_count:
            raise ValueError("TrialEval regime-cell counts do not reconcile")
        return self


_RELEASED_REGIME_CELL_COUNTS = (
    TRIALEVAL_REGIME_CELL_COUNT_V1,
    TRIALEVAL_REGIME_CELL_COUNT_V1 + TRIALEVAL_WITHHELD_ONLY_CELL_COUNT_V1,
)
"""The public split releases the public cells; the withheld split adds its held-out cells."""


def _complete_context_grid(rows: tuple[TrialEvalContextRegimeCellSummaryV1, ...], expected_cells: int) -> bool:
    """Require every released scientific cell in the same selected presentation set, balanced in replicates.

    The public split releases every context, with every world in every scenario. The withheld split releases only
    the C5 view and presents each world in one scenario of its clinical question.
    """
    contexts = {row.context_tier for row in rows}
    cells = {row.cell_ref for row in rows}
    identities = {(row.cell_ref, row.context_tier) for row in rows}
    counts: dict[str, set[int]] = {}
    for row in rows:
        counts.setdefault(row.cell_ref, set()).add(row.scheduled_trial_count)
    return (
        bool(contexts)
        and expected_cells in _RELEASED_REGIME_CELL_COUNTS
        and len(cells) == expected_cells
        and len(identities) == len(rows) == len(cells) * len(contexts)
        and all(len(values) == 1 for values in counts.values())
        and (len(contexts) == 1 or len(set().union(*counts.values())) == 1)
    )


class TrialEvalBalancedSummaryV1(_SummaryModel):
    """Released-scenario TrialEval summary with explicit partial-run behavior."""

    scorer: CurrentScorer

    schema_id: Literal["trialagentbench.trialeval_balanced_summary/v3"] = (
        "trialagentbench.trialeval_balanced_summary/v3"
    )
    declared_trial_count: int = Field(ge=1)
    graded_trial_count: int = Field(ge=0)
    evaluator_error_count: int = Field(ge=0)
    expected_regime_cell_count: int = TRIALEVAL_REGIME_CELL_COUNT_V1
    observed_regime_cell_count: int = Field(ge=1)
    complete_regime_grid: bool
    context_cells: tuple[TrialEvalContextRegimeCellSummaryV1, ...]
    regime_cells: tuple[TrialEvalRegimeCellSummaryV1, ...]
    effect_state_strata: tuple[TrialEvalEffectStateStratumV1, ...]
    native_metric_strata: tuple[TrialEvalNativeMetricStratumV1, ...]
    overall_components: tuple[TrialEvalAggregateValueV1, ...]
    overall_metrics: tuple[TrialEvalAggregateValueV1, ...]
    checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_summary(self) -> Self:
        if self.graded_trial_count + self.evaluator_error_count != self.declared_trial_count:
            raise ValueError("balanced summary counts do not reconcile to the declared denominator")
        if self.observed_regime_cell_count > self.expected_regime_cell_count:
            raise ValueError("balanced summary observes more cells than the release declares")
        expected_complete = (
            _complete_context_grid(self.context_cells, self.expected_regime_cell_count)
            and self.declared_trial_count == sum(row.scheduled_trial_count for row in self.context_cells)
            and self.observed_regime_cell_count == self.expected_regime_cell_count
        )
        if self.complete_regime_grid != expected_complete:
            raise ValueError("balanced summary regime-grid disposition differs from released cells")
        if expected_complete != bool(self.overall_components):
            raise ValueError("overall component estimates require the complete released regime grid")
        expected = canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))
        if self.checksum is not None and self.checksum != expected:
            raise ValueError("balanced summary checksum mismatch")
        return self

    def with_checksum(self) -> TrialEvalBalancedSummaryV1:
        """Return the summary with its canonical content identity."""

        return self.model_copy(
            update={"checksum": canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))}
        )


def _numeric(value: object) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _component_value(status: object) -> float | None:
    if status in {"satisfied", "observed"}:
        return 1.0
    if status in {"not_satisfied", "not_evaluable"}:
        return 0.0
    if status == "not_applicable":
        return None
    raise ValueError(f"unknown TrialEval component status: {status!r}")


def _trial_aggregates(
    rows: tuple[dict[str, object], ...],
    quantities: tuple[str, ...],
    *,
    component: bool,
    scheduled_count: int,
    evaluator_error_count: int = 0,
) -> tuple[TrialEvalAggregateValueV1, ...]:
    aggregates: list[TrialEvalAggregateValueV1] = []
    for quantity in quantities:
        source = "signal_loss" if quantity in {"observed_signal_loss", "bounded_recovery"} else quantity
        values = list(
            value
            for row in rows
            if (value := _component_value(row[source]) if component else _numeric(row.get(source))) is not None
        )
        if component and quantity == "completion":
            values.extend(0.0 for _ in range(evaluator_error_count))
        estimate = (
            None
            if not values or (quantity in {"signal_loss", "bounded_recovery"} and len(values) < scheduled_count)
            else mean(values)
        )
        aggregates.append(
            TrialEvalAggregateValueV1(
                quantity_id=quantity,
                scheduled_count=scheduled_count,
                observed_count=len(values),
                mean=bounded_recovery(estimate) if quantity == "bounded_recovery" else estimate,
            )
        )
    return tuple(aggregates)


def _mean_context_aggregates(
    rows: tuple[TrialEvalContextRegimeCellSummaryV1 | TrialEvalRegimeCellSummaryV1, ...],
    *,
    attribute: Literal["components", "comparable_metrics"],
) -> tuple[TrialEvalAggregateValueV1, ...]:
    by_quantity = {value.quantity_id for row in rows for value in getattr(row, attribute)}
    results: list[TrialEvalAggregateValueV1] = []
    for quantity in sorted(by_quantity):
        source = "signal_loss" if quantity == "bounded_recovery" else quantity
        values = tuple(
            aggregate.mean
            for row in rows
            for aggregate in getattr(row, attribute)
            if aggregate.quantity_id == source and aggregate.mean is not None
        )
        estimate = None if not values or (source == "signal_loss" and len(values) < len(rows)) else mean(values)
        results.append(
            TrialEvalAggregateValueV1(
                quantity_id=quantity,
                scheduled_count=len(rows),
                observed_count=len(values),
                mean=bounded_recovery(estimate) if quantity == "bounded_recovery" else estimate,
            )
        )
    return tuple(results)


def _expected_inventory(
    expected_items: tuple[TrialEvalExpectedItemV1, ...],
) -> dict[str, TrialEvalExpectedItemV1]:
    if not expected_items:
        raise ValueError("TrialEval aggregation requires a non-empty released item inventory")
    by_item = {item.item_id: item for item in expected_items}
    if len(by_item) != len(expected_items):
        raise ValueError("TrialEval aggregation inventory contains duplicate item identities")
    trial_ids = tuple(item.trial_id for item in expected_items)
    if len(trial_ids) != len(set(trial_ids)):
        raise ValueError("TrialEval aggregation inventory contains duplicate trial identities")
    for cell_ref in {item.cell_ref for item in expected_items}:
        classifications = {item.design_tier for item in expected_items if item.cell_ref == cell_ref}
        if len(classifications) != 1:
            raise ValueError(f"TrialEval regime cell has inconsistent classification: {cell_ref}")
    return by_item


def _validate_observed_rows(
    rows: list[dict[str, object]],
    *,
    expected_by_item: dict[str, TrialEvalExpectedItemV1],
    evaluator_error_item_ids: tuple[str, ...],
) -> dict[str, dict[str, object]]:
    by_item = {str(row["item_id"]): row for row in rows}
    if len(by_item) != len(rows):
        raise ValueError("TrialEval balanced aggregation requires unique item identities")
    evaluator_errors = set(evaluator_error_item_ids)
    if len(evaluator_errors) != len(evaluator_error_item_ids):
        raise ValueError("TrialEval evaluator-error item IDs must be unique")
    if set(by_item) & evaluator_errors or set(by_item) | evaluator_errors != set(expected_by_item):
        raise ValueError("Every released TrialEval item must be graded or have an evaluator error")
    for item_id, row in by_item.items():
        expected = expected_by_item[item_id]
        observed = (
            str(row["cell_ref"]),
            str(row["effect_state"]),
            str(row["design_tier"]),
            str(row["context_tier"]),
            str(row["effect_measure"]),
            str(row["result_unit"]),
        )
        declared = (
            expected.cell_ref,
            expected.effect_state,
            expected.design_tier,
            expected.context_tier,
            expected.effect_measure,
            expected.result_unit,
        )
        if observed != declared:
            raise ValueError(f"TrialEval graded row classification differs from its release inventory: {item_id}")
    return by_item


def _native_metric_strata(
    *,
    expected_items: tuple[TrialEvalExpectedItemV1, ...],
    rows_by_item: dict[str, dict[str, object]],
) -> tuple[TrialEvalNativeMetricStratumV1, ...]:
    keys = sorted({(item.effect_measure, item.result_unit) for item in expected_items})
    strata: list[TrialEvalNativeMetricStratumV1] = []
    for effect_measure, result_unit in keys:
        expected = tuple(
            item
            for item in expected_items
            if item.effect_measure == effect_measure and item.result_unit == result_unit
        )
        rows = tuple(rows_by_item[item.item_id] for item in expected if item.item_id in rows_by_item)
        strata.append(
            TrialEvalNativeMetricStratumV1(
                effect_measure=effect_measure,
                result_unit=result_unit,
                scheduled_trial_count=len(expected),
                graded_trial_count=len(rows),
                evaluator_error_count=len(expected) - len(rows),
                metrics=_trial_aggregates(
                    rows,
                    _NATIVE_METRICS,
                    component=False,
                    scheduled_count=len(expected),
                ),
            )
        )
    return tuple(strata)


def _effect_state_strata(
    *,
    expected_items: tuple[TrialEvalExpectedItemV1, ...],
    rows_by_item: dict[str, dict[str, object]],
) -> tuple[TrialEvalEffectStateStratumV1, ...]:
    strata: list[TrialEvalEffectStateStratumV1] = []
    for effect_state in ("favourable", "near_null", "unfavourable"):
        expected = tuple(item for item in expected_items if item.effect_state == effect_state)
        if not expected:
            continue
        rows = tuple(rows_by_item[item.item_id] for item in expected if item.item_id in rows_by_item)
        strata.append(
            TrialEvalEffectStateStratumV1(
                effect_state=effect_state,
                scheduled_trial_count=len(expected),
                graded_trial_count=len(rows),
                evaluator_error_count=len(expected) - len(rows),
                conclusion_class_counts=dict(
                    sorted(
                        Counter(
                            conclusion for row in rows if isinstance((conclusion := row.get("conclusion_class")), str)
                        ).items()
                    )
                ),
                components=_trial_aggregates(
                    rows,
                    _COMPONENTS,
                    component=True,
                    scheduled_count=len(expected),
                    evaluator_error_count=len(expected) - len(rows),
                ),
                comparable_metrics=_trial_aggregates(
                    rows,
                    _COMPARABLE_METRICS,
                    component=False,
                    scheduled_count=len(expected),
                    evaluator_error_count=len(expected) - len(rows),
                ),
            )
        )
    return tuple(strata)


def build_trialeval_balanced_summary_v1(
    rows: list[dict[str, object]],
    *,
    expected_items: tuple[TrialEvalExpectedItemV1, ...],
    evaluator_error_item_ids: tuple[str, ...] = (),
) -> TrialEvalBalancedSummaryV1:
    """Aggregate trials, then contexts, then released scenario cells."""

    expected_by_item = _expected_inventory(expected_items)
    rows_by_item = _validate_observed_rows(
        rows,
        expected_by_item=expected_by_item,
        evaluator_error_item_ids=evaluator_error_item_ids,
    )
    expected_groups: dict[tuple[str, str], list[TrialEvalExpectedItemV1]] = {}
    for item in expected_items:
        expected_groups.setdefault((item.cell_ref, item.context_tier), []).append(item)
    context_cell_rows: list[TrialEvalContextRegimeCellSummaryV1] = []
    for (cell_ref, context), group in sorted(expected_groups.items()):
        graded = tuple(rows_by_item[item.item_id] for item in group if item.item_id in rows_by_item)
        context_cell_rows.append(
            TrialEvalContextRegimeCellSummaryV1(
                cell_ref=cell_ref,
                design_tier=group[0].design_tier,
                context_tier=context,
                scheduled_trial_count=len(group),
                graded_trial_count=len(graded),
                evaluator_error_count=len(group) - len(graded),
                components=_trial_aggregates(
                    graded,
                    _COMPONENTS,
                    component=True,
                    scheduled_count=len(group),
                    evaluator_error_count=len(group) - len(graded),
                ),
                comparable_metrics=_trial_aggregates(
                    graded,
                    _COMPARABLE_METRICS,
                    component=False,
                    scheduled_count=len(group),
                    evaluator_error_count=len(group) - len(graded),
                ),
            )
        )
    context_cells = tuple(context_cell_rows)
    cell_ids = tuple(sorted({row.cell_ref for row in context_cells}))
    cells = tuple(
        TrialEvalRegimeCellSummaryV1(
            cell_ref=cell_ref,
            design_tier=selected[0].design_tier,
            context_count=len(selected),
            scheduled_trial_count=sum(row.scheduled_trial_count for row in selected),
            graded_trial_count=sum(row.graded_trial_count for row in selected),
            evaluator_error_count=sum(row.evaluator_error_count for row in selected),
            components=_mean_context_aggregates(selected, attribute="components"),
            comparable_metrics=_mean_context_aggregates(selected, attribute="comparable_metrics"),
        )
        for cell_ref in cell_ids
        if (selected := tuple(row for row in context_cells if row.cell_ref == cell_ref))
    )
    expected_cells = len({item.cell_ref for item in expected_items})
    complete = _complete_context_grid(context_cells, expected_cells)
    overall_components = _mean_context_aggregates(cells, attribute="components") if complete else ()
    overall_metrics = _mean_context_aggregates(cells, attribute="comparable_metrics") if complete else ()
    return TrialEvalBalancedSummaryV1(
        scorer=current_scorer(),
        declared_trial_count=len(expected_items),
        graded_trial_count=len(rows),
        evaluator_error_count=len(evaluator_error_item_ids),
        expected_regime_cell_count=expected_cells,
        observed_regime_cell_count=len(cells),
        complete_regime_grid=complete,
        context_cells=context_cells,
        regime_cells=cells,
        effect_state_strata=_effect_state_strata(
            expected_items=expected_items,
            rows_by_item=rows_by_item,
        ),
        native_metric_strata=_native_metric_strata(
            expected_items=expected_items,
            rows_by_item=rows_by_item,
        ),
        overall_components=overall_components,
        overall_metrics=overall_metrics,
    ).with_checksum()


def write_trialeval_grade_summary(
    *,
    rows: list[dict[str, object]],
    expected_items: tuple[TrialEvalExpectedItemV1, ...],
    scheduled_item_ids: tuple[str, ...],
    evaluator_error_item_ids: tuple[str, ...],
    output_dir: Path,
) -> None:
    """Write stable machine-readable and human-readable run summaries."""

    if not scheduled_item_ids or len(scheduled_item_ids) != len(set(scheduled_item_ids)):
        raise ValueError("TrialEval scheduled item IDs must be non-empty and unique")
    if len(evaluator_error_item_ids) != len(set(evaluator_error_item_ids)):
        raise ValueError("TrialEval evaluator-error item IDs must be unique")
    graded_item_ids = {str(row["item_id"]) for row in rows}
    scheduled = set(scheduled_item_ids)
    evaluator_errors = set(evaluator_error_item_ids)
    if graded_item_ids & evaluator_errors or graded_item_ids | evaluator_errors != scheduled:
        raise ValueError("Every scheduled TrialEval item must be graded or have an evaluator error")
    if scheduled != {item.item_id for item in expected_items}:
        raise ValueError("TrialEval scheduled items differ from the released aggregation inventory")
    output = Path(output_dir)
    scheduled_count = len(scheduled_item_ids)
    fieldnames = (
        "scorer",
        "item_id",
        "cell_ref",
        "effect_state",
        "design_tier",
        "design_subtype",
        "context_tier",
        "effect_measure",
        "result_unit",
        "model",
        "agent_status",
        "turns_used",
        "target_kind",
        "signal_loss",
        "primary_signal_loss",
        "required_targets",
        "scored_targets",
        "conclusion_class",
        *_COMPONENTS,
        "plan_execution_accurate",
        "failure_origins",
        "native_signed_error",
        "native_absolute_error",
        "native_squared_error",
        "normalized_absolute_error",
        "normalized_squared_error",
        "interval_covers_target",
        "interval_width",
        "proper_interval_score",
        "program_replay_status",
        "program_elapsed_seconds",
    )
    component_counts = {component: Counter(str(row[component]) for row in rows) for component in _COMPONENTS}
    scored_count = sum(_numeric(row.get("signal_loss")) is not None for row in rows)
    balanced = build_trialeval_balanced_summary_v1(
        rows,
        expected_items=expected_items,
        evaluator_error_item_ids=evaluator_error_item_ids,
    )
    write_json(output / "balanced_summary.json", balanced.model_dump(mode="json"))
    with (output / "aggregation_weights.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "scorer",
                "cell_ref",
                "design_tier",
                "context_tier",
                "trial_count",
                "context_weight_within_cell",
                "cell_weight",
            ),
        )
        writer.writeheader()
        context_counts = {cell.cell_ref: cell.context_count for cell in balanced.regime_cells}
        for cell in balanced.context_cells:
            writer.writerow(
                {
                    "scorer": balanced.scorer.model_dump_json(),
                    "cell_ref": cell.cell_ref,
                    "design_tier": cell.design_tier,
                    "context_tier": cell.context_tier,
                    "trial_count": cell.scheduled_trial_count,
                    "context_weight_within_cell": 1.0 / context_counts[cell.cell_ref],
                    "cell_weight": (
                        1.0 / balanced.expected_regime_cell_count if balanced.complete_regime_grid else ""
                    ),
                }
            )
    write_json(
        output / "summary.json",
        {
            "scorer": balanced.scorer.model_dump(mode="json"),
            "schema_id": "ariadne.trialeval_grade_summary/v1",
            "n_items": scheduled_count,
            "n_graded_items": len(rows),
            "n_scored_items": scored_count,
            "n_evaluator_errors": len(evaluator_error_item_ids),
            "evaluator_error_item_ids": list(evaluator_error_item_ids),
            "component_counts": {
                component: dict(sorted(counts.items())) for component, counts in component_counts.items()
            },
            "items": rows,
        },
    )
    with (output / "results_full.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["scorer"] = balanced.scorer.model_dump_json()
            csv_row["failure_origins"] = "|".join(_failure_origins(row))
            writer.writerow(csv_row)

    failures = Counter(code for row in rows for code in _failure_origins(row))
    plan_rows = tuple(row for row in rows if row.get("plan_execution_accurate") is not None)
    plan_accurate = sum(row["plan_execution_accurate"] is True for row in plan_rows)
    lines = [
        "# TrialEvalBench grade summary",
        "",
        f"Scorer: `{balanced.scorer.version}`; sources SHA-256: `{balanced.scorer.sources_sha256}`.",
        "",
        f"- Declared denominator: **{scheduled_count}**",
        f"- Deterministically graded: **{len(rows)}/{scheduled_count}**",
        f"- Numerical losses available: **{scored_count}/{scheduled_count}**",
        f"- Evaluator errors: **{len(evaluator_error_item_ids)}/{scheduled_count}**",
        f"- Complete submissions: **{component_counts['completion']['satisfied']}/{scheduled_count}**",
        f"- Reproduced analyses: **{component_counts['reproducibility']['satisfied']}/{scheduled_count}**",
        "- Observed scale alignment: " f"**{component_counts['question_alignment']['observed']}/{scheduled_count}**",
        *(
            (
                f"- Plan executed within tolerance (plan-given items, scored apart): **{plan_accurate}/{len(plan_rows)}**",
            )
            if plan_rows
            else ()
        ),
        "",
        "## Component outcomes",
        "",
        "| component | satisfied | observed | not satisfied | not evaluable | not applicable |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        "| "
        + component.replace("_", " ")
        + " | "
        + " | ".join(
            str(component_counts[component][status])
            for status in ("satisfied", "observed", "not_satisfied", "not_evaluable", "not_applicable")
        )
        + " |"
        for component in _COMPONENTS
    )
    lines.extend(
        [
            "",
            "## Numerical recovery",
            "",
            "Complete-panel loss requires every scheduled result. Observed-case loss describes only "
            "the available results, retaining equal weighting within the available contexts and cells. "
            "Models completing different subsets cannot be ranked as complete-panel evaluations. "
            "Bounded recovery is 1/(1 + complete-panel mean loss), computed after aggregation "
            "with the same weights and availability as complete-panel loss; higher is better.",
            "",
            "| scope | complete-panel loss | observed-case loss | bounded recovery | scored / scheduled |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for cell in balanced.context_cells:
        metrics = {value.quantity_id: value for value in cell.comparable_metrics}
        primary = metrics["signal_loss"]
        observed = metrics["observed_signal_loss"]
        recovery = metrics["bounded_recovery"]
        primary_text = "unavailable" if primary.mean is None else f"{primary.mean:.6g}"
        observed_text = "unavailable" if observed.mean is None else f"{observed.mean:.6g}"
        recovery_text = "unavailable" if recovery.mean is None else f"{recovery.mean:.6g}"
        lines.append(
            f"| {cell.cell_ref} / {cell.context_tier} | {primary_text} | {observed_text} | "
            f"{recovery_text} | {observed.observed_count} / {observed.scheduled_count} |"
        )
    lines.extend(
        [
            "",
            "## Conclusions by generating effect state",
            "",
            "| effect state | conclusion | count | rate among graded items |",
            "|---|---|---:|---:|",
        ]
    )
    for stratum in balanced.effect_state_strata:
        for conclusion in _CONCLUSION_CLASSES:
            count = stratum.conclusion_class_counts.get(conclusion, 0)
            rate = count / stratum.graded_trial_count if stratum.graded_trial_count else 0.0
            lines.append(f"| {stratum.effect_state} | {conclusion.replace('_', ' ')} | {count} | {rate:.3f} |")
    lines.extend(
        [
            "",
            "## Failure origins",
            "",
            "| origin | count |",
            "|---|---:|",
        ]
    )
    lines.extend(f"| `{code}` | {count} |" for code, count in sorted(failures.items()))
    (output / "RESULTS_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "TrialEvalAggregateValueV1",
    "TrialEvalBalancedSummaryV1",
    "TrialEvalContextRegimeCellSummaryV1",
    "TrialEvalEffectStateStratumV1",
    "TrialEvalExpectedItemV1",
    "TrialEvalNativeMetricStratumV1",
    "TrialEvalRegimeCellSummaryV1",
    "build_trialeval_balanced_summary_v1",
    "write_trialeval_grade_summary",
]
