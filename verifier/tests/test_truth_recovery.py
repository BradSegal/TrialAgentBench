"""Truth recovery and share of oracle value aggregate the recorded scores by their published rules."""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trialagentbench_verifier.development_campaign import TrialDevOracleValuesV1, TrialDevOracleValueV1
from trialagentbench_verifier.replay_bank import ReplayBankRecordV1, ReplayBankReportV1
from trialagentbench_verifier.replay_scoring import AnswerReplaysV1, ReplayOutcomeV1, answer_replays_path
from trialagentbench_verifier.resources import truth_recovery as tr
from trialagentbench_verifier.response import TrialDevRankingItemV1, TrialDevRunRankingV1
from trialagentbench_verifier.scorer import current_scorer
from trialagentbench_verifier.trialeval import _ConsequenceEconomics, _ConsequenceReference, _PointTarget, _Question

COLUMNS = (
    "model",
    "task_id",
    "context",
    "scoring",
    "answered_share",
    "normalised_absolute_error",
    "interval_coverage",
    "value_multiplier",
    "cost_multiplier",
)
TASKS = tuple(f"t{world}" for world in range(4))
DESIGN = {task: ("cell", "question", f"world{index}") for index, task in enumerate(TASKS)}


def _row(model: str, task: str, error: float | None = 1.0, *, share: float = 1.0, **update: str) -> dict[str, str]:
    answered = error is not None
    row = {
        "model": model,
        "task_id": task,
        "context": "C5",
        "scoring": "replay" if answered else "failure",
        "answered_share": str(share if answered else 0.0),
        "normalised_absolute_error": "" if error is None else str(error),
        "interval_coverage": "0.9" if answered else "",
        "value_multiplier": "1.0",
        "cost_multiplier": "1.0",
    }
    return row | update


def _write(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _cell(answerer: str, task: str, error: float | None, *, share: float = 1.0) -> tr.TrialEvalCell:
    return tr.TrialEvalCell(
        answerer=answerer,
        task_id=task,
        status="answered" if error is not None else "missing",
        answered_share=share if error is not None else 0.0,
        interval_coverage=0.9 if error is not None else None,
        own_error=error,
    )


def _cells(**answers: tuple[float | str | None, ...]) -> dict[str, dict[str, tr.TrialEvalCell]]:
    """Return cells by answerer, ``__`` standing for ``:``; ``None`` is a missing answer, ``"out"`` outside the panel."""
    named = {answerer.replace("__", ":"): errors for answerer, errors in answers.items()}
    return {
        answerer: {
            task: _cell(answerer, task, error) for task, error in zip(TASKS, errors, strict=True) if error != "out"
        }
        for answerer, errors in named.items()
    }


def _target(measure: str = "mean_difference", value: float = 0.25) -> _PointTarget:
    return _PointTarget(
        kind="point",
        binding=_Question(
            schema_id="trialagentbench.trialeval_question_binding/v1",
            analysis_population_id="itt",
            estimand_id="primary",
            treatment_id="treatment",
            comparator_id="control",
            endpoint_id="y",
            effect_measure=measure,
            analysis_unit="participant",
            population_basis="superpopulation",
            identification_status="point_identified",
            result_unit="ratio" if measure == "rate_ratio" else "score",
            uncertainty_type="central_confidence_interval",
            confidence_level=0.95,
            sidedness="two_sided_central",
            null_value=1.0 if measure == "rate_ratio" else 0.0,
            favorable_direction="higher",
        ),
        value=value,
        confidence_level=0.95,
        reference_sampling_standard_deviation=0.2,
        reference_sampling_basis_id="finite_world_oracle_contrast",
        consequence=_ConsequenceReference(
            participants=100,
            credited=((0.2, 0.0, 0.4),),
            economics=_ConsequenceEconomics(
                cost_type="fixture", value_of_success_musd=1000.0, replicate_cost_musd=20.0, fixed_trial_cost_musd=10.0
            ),
        ),
    )


def test_a_failed_answer_is_missing_and_an_answer_without_fresh_trials_is_refused() -> None:
    assert tr.cell_from_row("m", _row("m", "t", None)).status == "missing"
    partial = tr.cell_from_row("m", _row("m", "t", 0.4, share=0.5))
    assert (partial.status, partial.answered_share, partial.own_error) == ("answered", 0.5, 0.4)
    with pytest.raises(ValueError, match="no fresh trials"):
        tr.cell_from_row("m", _row("m", "t", 0.4, scoring="single_dataset_truth"))
    with pytest.raises(ValueError, match="without an error"):
        tr.cell_from_row("m", _row("m", "t", 0.4, normalised_absolute_error=""))


def test_the_schedule_decides_the_panel_and_only_declared_main_rows_count(tmp_path: Path) -> None:
    answers = _write(
        tmp_path / "answers.csv",
        [
            _row("m", "t0", 1.0),
            _row("m", "t0", 9.0, context="C1"),
            _row("m", "t0", 9.0, value_multiplier="2.0"),
            _row("policy:scripted_reference", "t0", 0.5),
            _row("policy:prior_only", "t0", None),
        ],
    )
    cells = tr.read_cells([answers], {"m": frozenset({"t0", "t1"})})
    assert cells["m"]["t0"].own_error == 1.0
    assert cells["m"]["t1"].status == "missing"
    assert "t2" not in cells["m"]
    assert "policy:prior_only" not in cells


def test_an_unscheduled_answer_a_repeated_answer_and_a_disagreeing_fixed_analysis_are_refused(tmp_path: Path) -> None:
    schedule = {"m": frozenset({"t0"})}
    outside = _write(tmp_path / "outside.csv", [_row("other", "t0")])
    with pytest.raises(ValueError, match="outside the schedule"):
        tr.read_cells([outside], schedule)
    once = _write(tmp_path / "once.csv", [_row("m", "t0")])
    with pytest.raises(ValueError, match="more than one answer"):
        tr.read_cells([once, once], schedule)
    same = _write(tmp_path / "same.csv", [_row(tr.ROBUST, "t0", 0.5)])
    assert tr.read_cells([same, same], {})[tr.ROBUST]["t0"].own_error == 0.5
    other = _write(tmp_path / "other.csv", [_row(tr.ROBUST, "t0", 0.6)])
    with pytest.raises(ValueError, match="differs between answer files"):
        tr.read_cells([same, other], {})


def test_the_pipeline_is_read_as_its_own_fixed_analysis(tmp_path: Path) -> None:
    pipeline = _write(tmp_path / "pipeline.csv", [_row("detect", "t0", 0.7)])
    assert tr.read_cells([], {}, pipeline)[tr.PIPELINE]["t0"].own_error == 0.7


def test_the_missing_answer_rule_is_never_better_than_a_real_answer_or_guessing_no_effect() -> None:
    cells = _cells(a=(1.0, 3.0, 0.2, 0.2), b=(2.0, 1.0, 0.1, None), policy__always_naive=(99.0, 99.0, 99.0, 99.0))
    no_effect = {"t0": 5.0, "t1": 0.5, "t2": 0.1, "t3": 0.4}
    imputed = tr.imputed_errors(cells, no_effect)
    assert imputed == {"t0": 5.0, "t1": 3.0, "t2": 0.2, "t3": 0.4}
    assert tr.imputed_errors(cells, no_effect, "no_effect") == no_effect
    assert tr.imputed_errors(cells, no_effect, "worst_answer") == {"t0": 2.0, "t1": 3.0, "t2": 0.2, "t3": 0.2}
    assert tr.cell_error(cells["b"]["t3"], imputed["t3"]) == 0.4
    assert tr.cell_error(_cell("b", "t", 0.4, share=0.5), 3.0) == pytest.approx(1.7)
    with pytest.raises(ValueError, match="no model answer"):
        tr.worst_answer_errors({"a": {"t0": _cell("a", "t0", None)}}, ["t0"])


def test_no_effect_error_uses_the_comparison_scale() -> None:
    assert tr.no_effect_error(SimpleNamespace(task_id="t", target=_target())) == pytest.approx(1.25)
    ratio = SimpleNamespace(task_id="t", target=_target("rate_ratio", 2.0))
    assert tr.no_effect_error(ratio) == pytest.approx(math.log(2) / 0.2)


def test_a_partial_panel_is_adjusted_for_how_hard_its_tasks_are() -> None:
    values = np.array([[1.0, 3.0], [2.0, np.nan]])
    assert tr.two_way_fit(values).tolist() == pytest.approx([2.0, 3.0])
    with pytest.raises(ValueError, match="observed cell"):
        tr.two_way_fit(np.array([[1.0, np.nan], [2.0, np.nan]]))


def test_world_resamples_keep_a_shared_world_together_and_refuse_unequal_strata() -> None:
    tasks = ["a0", "a1", "b0", "b1"]
    design = {task: (f"cell_{task[0]}", "question", f"world{task[1]}") for task in tasks}
    columns, scale = tr.world_resamples(tasks, design, np.zeros((1, 4)), bootstrap=20, seed=1)
    assert scale == pytest.approx(math.sqrt(2))
    assert all(int(draw[k]) % 2 == int(draw[k + 1]) % 2 for draw in columns for k in (0, 2))
    uneven = {**design, "c0": ("cell_c", "other", "world9")}
    with pytest.raises(ValueError, match="world strata"):
        tr.world_resamples([*tasks, "c0"], uneven, np.zeros((1, 5)), bootstrap=1, seed=1)


def test_the_table_reports_error_serious_misses_and_paired_counts() -> None:
    cells = _cells(
        a=(1.0, 2.0, 1.0, 1.0),
        b=(1.0, 1.0, 1.0, 1.0),
        policy__scripted_reference=(1.0, 1.0, 1.0, 1.0),
        policy__always_robust=(1.01, 1.5, 0.5, 1.1),
    )
    imputed = dict.fromkeys(TASKS, 3.0)
    rows = {row.answerer: row for row in tr.truth_recovery_table(cells, DESIGN, imputed, bootstrap=50, seed=3)}
    assert rows["a"].error == pytest.approx(1.25)
    assert (rows["b"].rank, rows["a"].rank) == (1, 2)
    assert (rows["a"].serious_misses, rows["a"].serious_miss_share) == (1, 0.25)
    assert (rows["a"].wins, rows["a"].ties, rows["a"].losses) == (1, 1, 2)
    assert rows["a"].error_low <= rows["a"].error <= rows["a"].error_high
    assert rows[tr.ROBUST].kind == "fixed_analysis" and rows[tr.ROBUST].rank is None and rows[tr.ROBUST].wins is None
    assert rows["a"].interval_coverage == pytest.approx(0.9)


def test_a_serious_miss_starts_at_exactly_half_a_standard_error() -> None:
    cells = _cells(
        at=(1.5, 1.0, 1.0, 1.0),
        below=(1.25, 1.0, 1.0, 1.0),
        policy__scripted_reference=(1.0, 1.0, 1.0, 1.0),
    )
    rows = {
        row.answerer: row for row in tr.truth_recovery_table(cells, DESIGN, dict.fromkeys(TASKS, 3.0), bootstrap=9)
    }
    assert (rows["at"].serious_misses, rows["below"].serious_misses) == (1, 0)


def test_a_missing_answer_and_a_partial_panel_enter_the_table_by_their_rules() -> None:
    cells = _cells(a=(1.0, None, 1.0, 1.0), b=(1.0, 1.0, "out", "out"), policy__scripted_reference=(1.0,) * 4)
    rows = {
        row.answerer: row for row in tr.truth_recovery_table(cells, DESIGN, dict.fromkeys(TASKS, 3.0), bootstrap=9)
    }
    assert rows["a"].panel_tasks == 4 and rows["a"].missing_answer_tasks == 1
    assert rows["a"].error == pytest.approx(1.5)
    assert rows["a"].rerun_share == 0.75
    assert rows["b"].panel_tasks == 2
    assert rows["a"].wins is None, "without always corrected the counts stay empty"


def test_the_table_requires_the_reference_analysis_and_its_tasks() -> None:
    with pytest.raises(ValueError, match="process-aware analysis is absent"):
        tr.truth_recovery_table(_cells(a=(1.0,) * 4), DESIGN, dict.fromkeys(TASKS, 3.0), bootstrap=1)
    cells = _cells(a=(1.0,) * 4, policy__scripted_reference=(1.0, 1.0, "out", "out"))
    with pytest.raises(ValueError, match="outside the process-aware"):
        tr.truth_recovery_table(cells, DESIGN, dict.fromkeys(TASKS, 3.0), bootstrap=1)


def test_the_reference_analysis_comes_from_the_public_bank() -> None:
    def replay(index: int, report: ReplayBankReportV1 | None) -> ReplayBankRecordV1:
        return ReplayBankRecordV1(
            task_id="t", replay_index=index, truth=0.25, reference_route_id="ref", routes={"ref": report}
        )

    record = SimpleNamespace(task_id="t", target=_target())
    cell = tr.reference_cell(
        record,
        [
            replay(1, ReplayBankReportV1(estimate=0.35, lower=0.15, upper=0.55)),
            replay(2, ReplayBankReportV1(estimate=0.45, lower=0.40, upper=0.50)),
            replay(3, None),
            replay(4, ReplayBankReportV1(estimate=0.25, lower=0.05, upper=0.45)),
        ],
    )
    assert cell.answerer == tr.REFERENCE
    assert cell.answered_share == 0.75
    assert cell.own_error == pytest.approx((0.5 + 1.0 + 0.0) / 3)
    assert cell.interval_coverage == pytest.approx(2 / 3)


def test_counts_beyond_twice_rerun_noise_need_the_trials_and_call_a_noisy_gap_a_tie() -> None:
    cells = _cells(a=(1.0, 2.0, 0.5, None), policy__scripted_reference=(1.0,) * 4, policy__always_robust=(1.0,) * 4)
    robust = [1.0, 1.0, 1.0, 1.0]
    vectors = {
        "a": {
            "t0": [1.0, 1.0, 1.0, 1.0],  # the same answer: no noise, no gap
            "t1": [2.0, 2.0, 2.0, 2.0],  # a resolved loss
            "t2": [0.0, 1.0, 0.0, 1.0],  # a mean gap of 0.5, inside twice the noise of 0.5: a tie, a win on the band
            "t3": [None, None, None, None],  # missing: the imputed error on every trial, a resolved loss
        },
        tr.ROBUST: dict.fromkeys(TASKS, robust),
    }
    rows = tr.truth_recovery_table(cells, DESIGN, dict.fromkeys(TASKS, 3.0), vectors=vectors, bootstrap=9)
    row = next(r for r in rows if r.answerer == "a")
    assert (row.wins, row.ties, row.losses) == (1, 1, 2)
    assert (row.mc_wins, row.mc_ties, row.mc_losses) == (0, 2, 2)
    assert next(r for r in rows if r.answerer == tr.ROBUST).mc_wins is None
    bare = tr.truth_recovery_table(cells, DESIGN, dict.fromkeys(TASKS, 3.0), bootstrap=9)
    assert next(r for r in bare if r.answerer == "a").mc_wins is None, "without trials the counts stay empty"


def test_fresh_trial_errors_must_average_to_the_cell() -> None:
    cells = _cells(a=(1.0,) * 4, policy__scripted_reference=(1.0,) * 4, policy__always_robust=(1.0,) * 4)
    vectors = {"a": dict.fromkeys(TASKS, [2.0, 2.0]), tr.ROBUST: dict.fromkeys(TASKS, [1.0, 1.0])}
    with pytest.raises(ValueError, match="do not average"):
        tr.truth_recovery_table(cells, DESIGN, dict.fromkeys(TASKS, 3.0), vectors=vectors, bootstrap=1)


def _bank(variant: str = "full") -> SimpleNamespace:
    routes = {
        "ref": ReplayBankReportV1(estimate=0.45, lower=0.1, upper=0.6),
        tr.ROBUST: ReplayBankReportV1(estimate=0.25, lower=0.1, upper=0.6),
    }
    replay = ReplayBankRecordV1(task_id="t", replay_index=1, truth=0.25, reference_route_id="ref", routes=routes)
    other = replay.model_copy(update={"replay_index": 2})
    return SimpleNamespace(manifest=SimpleNamespace(variant=variant), records={"t": (replay, other)})


def _store(root: Path, name: str, estimate: float) -> Path:
    outcomes = [
        ReplayOutcomeV1(replay_index=index, status="answered", estimate=estimate, lower=0.0, upper=1.0)
        for index in (1, 2)
    ]
    path = answer_replays_path(root, bank_manifest_sha256="b" * 64, task_id="t", source_sha256=name * 64)
    path.parent.mkdir(parents=True, exist_ok=True)
    answer = AnswerReplaysV1(task_id="t", source_sha256=name * 64, bank_manifest_sha256="b" * 64, replays=outcomes)
    path.write_text(answer.model_dump_json(), encoding="utf-8")
    return root


def test_a_model_trials_come_from_the_replay_file_that_reproduces_its_score(tmp_path: Path) -> None:
    key = {"t": SimpleNamespace(task_id="t", target=_target())}
    store = _store(_store(tmp_path, "a", 0.45), "b", 0.65)
    cell = _cell("m", "t", 1.0)
    cells = {
        "m": {"t": cell},
        tr.REFERENCE: {"t": _cell(tr.REFERENCE, "t", 1.0)},
        tr.ROBUST: {"t": _cell(tr.ROBUST, "t", 0.0)},
    }
    vectors = tr.trial_vectors(cells, key, _bank(), "b" * 64, [store])
    assert vectors["m"]["t"] == pytest.approx([1.0, 1.0])
    assert vectors[tr.REFERENCE]["t"] == pytest.approx([1.0, 1.0])
    assert vectors[tr.ROBUST]["t"] == pytest.approx([0.0, 0.0])
    with pytest.raises(ValueError, match="0 replay files"):
        tr.trial_vectors({"m": {"t": _cell("m", "t", 9.0)}}, key, _bank(), "b" * 64, [store])
    with pytest.raises(ValueError, match="fixed-analysis table or the full replay bank"):
        tr.trial_vectors(cells, key, _bank("public"), "b" * 64, [store])


def _programme(task: str, stratum: str, net: float, oracle: float, *, status: str = "answered") -> tr.Programme:
    return tr.Programme(
        task_id=task,
        stratum=(stratum, "s", 3),
        status=status,
        net_value_usd=net,
        oracle_value_usd=oracle,
        rule_based_value_usd=oracle / 2,
    )


def test_the_share_of_oracle_value_is_a_ratio_of_equal_stratum_means_with_a_corrected_interval() -> None:
    panel = [
        _programme("a", "x", 10, 100),
        _programme("b", "x", 30, 100),
        _programme("c", "y", 90, 200),
        _programme("d", "y", 0, 200),
    ]
    row = tr.oracle_share_table({"m": panel})[0]
    assert tr.stratum_weights(panel) == [0.25] * 4
    assert row.oracle_share == pytest.approx(((10 + 30) / 2 + 45) / (100 + 200))
    assert row.rule_based_share == pytest.approx(0.5)

    # Two worlds per stratum: the correction widens the plain bootstrap's spread by sqrt(2).
    rng = np.random.default_rng(tr.SEED)
    plain = np.zeros(tr.BOOTSTRAP)
    for values in (np.array([10.0, 30.0]), np.array([90.0, 0.0])):
        plain += (values[rng.integers(0, 2, size=(tr.BOOTSTRAP, 2))].mean(axis=1) - values.mean()) * math.sqrt(2)
        plain += values.mean()
    expected = np.percentile(plain / 300, [2.5, 97.5])
    assert [row.oracle_share_low, row.oracle_share_high] == pytest.approx(expected.tolist())


def test_a_partial_panel_keeps_its_share_and_a_panel_without_oracle_value_has_no_row() -> None:
    panel = [_programme("a", "x", 10, 100), _programme("b", "x", 0, 100, status="missing")]
    assert tr.oracle_share_table({"m": panel})[0].oracle_share == pytest.approx(0.05)
    assert tr.oracle_share_table({"m": panel[:1]})[0].oracle_share_low is None
    assert tr.oracle_share_table({"m": [_programme("a", "x", 10, 0)]}) == ()


def _ranking(path: Path, statuses: dict[str, tuple[str, float | None]]) -> Path:
    items = [
        TrialDevRankingItemV1(
            task_id=task,
            world_id=f"world-{task}",
            stream="portfolio_management",
            clinical_profile_id="clinical",
            scenario_profile_id="scenario",
            execution_status=status,
            run_checksum="a" * 64 if status == "completed" else None,
            observational_loss=None,
            initial_target_statuses={"effect": "unscored"},
            authored_programme=True,
            net_development_value_usd=net,
        )
        for task, (status, net) in statuses.items()
    ]
    ranking = TrialDevRunRankingV1(
        scorer=current_scorer(),
        release_id="release",
        release_manifest_sha256="a" * 64,
        invocation_checksum="b" * 64,
        model_id="m",
        scope="scheduled_panel",
        items=tuple(items),
        profile_means=(),
    )
    path.write_text(ranking.model_dump_json(), encoding="utf-8")
    return path


def _oracle(*tasks: str) -> TrialDevOracleValuesV1:
    return TrialDevOracleValuesV1(
        release_id="release",
        release_manifest_sha256="a" * 64,
        policy_source_sha256="b" * 64,
        source_artifacts={"runs": "c" * 64},
        values=tuple(
            TrialDevOracleValueV1(task_id=t, world_id=f"world-{t}", net_value_usd=100.0, reference_value_usd=40.0)
            for t in tasks
        ),
    )


def test_programmes_come_from_verified_rankings_and_an_unfinished_attempt_stops_at_once(tmp_path: Path) -> None:
    path = _ranking(tmp_path / "r.json", {"a": ("completed", 25.0), "b": ("incomplete_run", None)})
    panel = {p.task_id: p for p in tr.read_programmes([path], _oracle("a", "b"))["m"]}
    assert (panel["a"].status, panel["a"].net_value_usd, panel["a"].rule_based_value_usd) == ("answered", 25.0, 40.0)
    assert (panel["b"].status, panel["b"].net_value_usd) == ("missing", 0.0)
    with pytest.raises(ValueError, match="more than one outcome"):
        tr.read_programmes([path, path], _oracle("a", "b"))
    with pytest.raises(ValueError, match="lack a scheduled programme"):
        tr.read_programmes([path], _oracle("a"))
    unvalued = _ranking(tmp_path / "u.json", {"a": ("completed", None)})
    with pytest.raises(ValueError, match="has no value"):
        tr.read_programmes([unvalued], _oracle("a"))


def test_the_command_writes_both_tables_from_campaign_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    records = tuple(
        SimpleNamespace(
            task_id=task, cell_ref="cell", series_ref="question", base_world_id=DESIGN[task][2], target=_target()
        )
        for task in TASKS
    )
    monkeypatch.setattr(tr, "read_grading_records", lambda root: {record.task_id: record for record in records})
    answers = [
        *(_row("m", task, 1.0) for task in TASKS[:3]),
        *(_row("n", task, 2.0) for task in TASKS),
        *(_row(tr.REFERENCE, task, 0.5) for task in TASKS),
    ]
    schedule = tmp_path / "schedule.csv"
    pairs = [(model, task) for model in "mn" for task in TASKS] + [("m", "a"), ("m", "b")]
    schedule.write_text("model,task_id\n" + "".join(f"{m},{t}\n" for m, t in pairs), encoding="utf-8")
    oracle = tmp_path / "oracle.json"
    oracle.write_text(_oracle("a", "b").model_dump_json(), encoding="utf-8")
    output = tmp_path / "out"
    tr.main(
        [
            "--answers",
            str(_write(tmp_path / "answers.csv", answers)),
            "--schedule",
            str(schedule),
            "--evaluator-release",
            str(tmp_path),
            "--output",
            str(output),
            "--trialdev-rankings",
            str(_ranking(tmp_path / "r.json", {"a": ("completed", 25.0), "b": ("completed", 25.0)})),
            "--oracle-values",
            str(oracle),
            "--expected-oracle-values-sha256",
            hashlib.sha256(oracle.read_bytes()).hexdigest(),
        ]
    )
    table = {row["answerer"]: row for row in csv.DictReader((output / "truth_recovery.csv").open())}
    assert table["m"]["missing_answer_tasks"] == "1" and table["m"]["panel_tasks"] == "4"
    assert float(table["m"]["error"]) == pytest.approx((3 * 1.0 + 2.0) / 4), "the missing task takes the worst answer"
    share = next(csv.DictReader((output / "oracle_share.csv").open()))
    assert float(share["oracle_share"]) == 0.25


def test_the_command_requires_a_source_for_the_reference_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tr, "read_grading_records", lambda root: {})
    empty = tmp_path / "schedule.csv"
    empty.write_text("model,task_id\n", encoding="utf-8")
    with pytest.raises(ValueError, match="process-aware analysis is absent"):
        tr.main(
            [
                "--answers",
                str(_write(tmp_path / "answers.csv", [])),
                "--schedule",
                str(empty),
                "--evaluator-release",
                str(tmp_path),
                "--output",
                str(tmp_path / "out"),
            ]
        )


def test_the_command_takes_its_answers_and_schedule_from_a_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = {
        task: SimpleNamespace(
            task_id=task, cell_ref="cell", series_ref="question", base_world_id=DESIGN[task][2], target=_target()
        )
        for task in TASKS
    }
    monkeypatch.setattr(tr, "read_grading_records", lambda root: records)
    plan = SimpleNamespace(
        configuration=SimpleNamespace(
            panel=SimpleNamespace(trialeval_tasks=TASKS),
            models=(SimpleNamespace(model="m"), SimpleNamespace(model="n")),
        )
    )
    monkeypatch.setattr(tr.CampaignPlanV1, "model_validate_json", lambda data: plan)
    campaign = tmp_path / "campaign"
    (campaign / "comparisons" / "trialeval").mkdir(parents=True)
    (campaign / "campaign.json").write_text("{}", encoding="utf-8")
    rows = [
        *(_row("m", task, 1.0) for task in TASKS[:3]),
        *(_row("n", task, 2.0) for task in TASKS),
        *(_row(tr.REFERENCE, task, 0.5) for task in TASKS),
    ]
    _write(campaign / "comparisons" / "trialeval" / "trialeval_answers.csv", rows)
    arguments = ["--campaign", str(campaign), "--evaluator-release", str(tmp_path), "--output", str(tmp_path / "out")]
    tr.main(arguments)
    table = {row["answerer"]: row for row in csv.DictReader((tmp_path / "out" / "truth_recovery.csv").open())}
    assert table["m"]["missing_answer_tasks"] == "1", "a scheduled task without a row is a missing answer"
    with pytest.raises(ValueError, match="pass --campaign"):
        tr.main(["--evaluator-release", str(tmp_path), "--output", str(tmp_path / "out")])


def test_the_release_tables_give_the_fixed_analyses_and_rerun_noise_with_the_public_bank(tmp_path: Path) -> None:
    fixed = tmp_path / "fixed"
    fixed.mkdir()
    rows = [_row(tr.REFERENCE, "t", 1.0), _row(tr.ROBUST, "t", 0.0)]
    _write(fixed / tr.FIXED_ANALYSIS_ROWS, rows)
    (fixed / tr.FIXED_ANALYSIS_TRIALS).write_text(
        "model,task_id,replay_index,error\n"
        f"{tr.REFERENCE},t,1,1.0\n{tr.REFERENCE},t,2,1.0\n{tr.ROBUST},t,1,0.0\n{tr.ROBUST},t,2,0.0\n",
        encoding="utf-8",
    )
    trials = tr.read_fixed_trials(fixed / tr.FIXED_ANALYSIS_TRIALS)
    key = {"t": SimpleNamespace(task_id="t", target=_target())}
    store = _store(tmp_path / "store", "a", 0.45)
    cells = tr.read_cells(
        [_write(tmp_path / "m.csv", [_row("m", "t", 1.0)]), fixed / tr.FIXED_ANALYSIS_ROWS], {"m": frozenset("t")}
    )
    vectors = tr.trial_vectors(cells, key, _bank("public"), "b" * 64, [store], trials)
    assert vectors["m"]["t"] == pytest.approx([1.0, 1.0])
    assert vectors[tr.ROBUST]["t"] == [0.0, 0.0]
    with pytest.raises(ValueError, match="fixed-analysis table or the full replay bank"):
        tr.trial_vectors(cells, key, _bank("public"), "b" * 64, [store])


def test_a_row_names_its_own_replay_store_when_campaigns_are_pooled(tmp_path: Path) -> None:
    own = _store(tmp_path / "own", "a", 0.45)
    _store(tmp_path / "other", "b", 0.45)
    answers = _write(tmp_path / "answers.csv", [_row("m", "t", 1.0)])
    text = answers.read_text(encoding="utf-8").splitlines()
    answers.write_text(f"{text[0]},replay_store\n{text[1]},{own}\n", encoding="utf-8")
    cells = tr.read_cells([answers], {"m": frozenset("t")})
    assert cells["m"]["t"].replay_store == str(own)
    key = {"t": SimpleNamespace(task_id="t", target=_target())}
    vectors = tr.trial_vectors(cells, key, _bank(), "b" * 64, [tmp_path / "other"])
    assert vectors["m"]["t"] == pytest.approx([1.0, 1.0])
