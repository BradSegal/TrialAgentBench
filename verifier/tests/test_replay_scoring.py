"""Replay scoring at the true effect on a tiny fixture bank."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

from trialagentbench_verifier import campaign, replay_scoring
from trialagentbench_verifier.decision_consequence import (
    CONFIRMATORY_SIZE_MULTIPLES,
    DecisionActionV1,
    DecisionEconomicsV1,
    UntruncatedSNRPriorV1,
    realised_value_v1,
)
from trialagentbench_verifier.execution_contracts import CodeExecutionLimitsV1, CodeExecutionResultV1
from trialagentbench_verifier.executor import analysis_project_artifacts
from trialagentbench_verifier.replay_bank import (
    PUBLIC_REFERENCE_ROUTE_ID,
    ReplayBankRecordV1,
    ReplayBankReportV1,
    load_replay_bank,
    public_replay_bank_v1,
)
from trialagentbench_verifier.replay_scoring import (
    AnswerReplaysV1,
    ReplayOutcomeV1,
    anchor_scores_v1,
    answers_csv,
    best_value_at_truth,
    load_replay_evidence_v1,
    replay_answer_v1,
    saved_source_sha256,
    score_replays_v1,
    score_single_dataset_v1,
)
from trialagentbench_verifier.trialeval import _ConsequenceEconomics, _ConsequenceReference, _PointTarget, _Question

TASK = "te_fixture"
REPLAYS = 6
TRUTH = 0.25
RELEASE = "release_fixture"


@pytest.fixture(autouse=True)
def synthetic_continuous_prior(monkeypatch):
    """Declare the unbounded prior of this tiny, synthetic ``primary`` question."""
    monkeypatch.setattr(replay_scoring, "load_layer_prior", lambda _: UntruncatedSNRPriorV1())
    monkeypatch.setattr(campaign, "load_layer_prior", lambda _: UntruncatedSNRPriorV1())

    summary = replay_scoring._summary

    def checked_summary(*args, **kwargs):
        score = summary(*args, **kwargs)
        assert score.material_decision_error_rate >= score.consequential_share
        return score

    monkeypatch.setattr(replay_scoring, "_summary", checked_summary)


# The scripted reference: difference in means with a normal 95% interval. The fixture bank stores its report, and
# the "reference" saved script below runs the same code, so its replays must reproduce the bank's reports exactly.
REFERENCE_SOURCE = """
import csv, math, pathlib, statistics
rows = list(csv.DictReader(open("data/raw/outcomes.csv")))
if pathlib.Path("protocol.md").read_text().startswith("CRASH") and CRASH_ON_MARKER:
    raise RuntimeError("cannot analyse this dataset")
arms = {arm: [float(row["y"]) for row in rows if row["arm"] == arm] for arm in ("treatment", "control")}
estimate = statistics.fmean(arms["treatment"]) - statistics.fmean(arms["control"])
se = math.sqrt(sum(statistics.variance(values) / len(values) for values in arms.values()))
with open("scratch/estimates.csv", "w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(("target_id", "estimate", "lower", "upper"))
    writer.writerow(("primary", repr(estimate), repr(estimate - 1.959963984540054 * se), repr(estimate + 1.959963984540054 * se)))
"""
CONSTANT_SOURCE = """
with open("scratch/estimates.csv", "w") as handle:
    handle.write("target_id,estimate,lower,upper\\nprimary,-0.4,-0.45,-0.35\\n")
"""


def _reference_report(rows: list[tuple[str, float]]) -> tuple[float, float, float]:
    """The same arithmetic as REFERENCE_SOURCE, so the bank stores exactly what the saved script reports."""
    arms = {arm: [y for name, y in rows if name == arm] for arm in ("treatment", "control")}
    estimate = statistics.fmean(arms["treatment"]) - statistics.fmean(arms["control"])
    se = math.sqrt(sum(statistics.variance(values) / len(values) for values in arms.values()))
    return estimate, estimate - 1.959963984540054 * se, estimate + 1.959963984540054 * se


def _target() -> _PointTarget:
    return _PointTarget(
        kind="point",
        binding=_Question(
            schema_id="trialagentbench.trialeval_question_binding/v1",
            analysis_population_id="itt",
            estimand_id="primary",
            treatment_id="treatment",
            comparator_id="control",
            endpoint_id="y",
            effect_measure="mean_difference",
            analysis_unit="participant",
            population_basis="superpopulation",
            identification_status="point_identified",
            result_unit="score",
            uncertainty_type="central_confidence_interval",
            confidence_level=0.95,
            sidedness="two_sided_central",
            null_value=0.0,
            favorable_direction="higher",
        ),
        value=TRUTH,
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


def _write_bank(root: Path, *, crash_replay: int | None = None) -> str:
    """Write a bank of one task with REPLAYS fresh datasets; return its manifest SHA-256."""
    generator = random.Random(7)
    records = []
    for index in range(1, REPLAYS + 1):
        task_dir = root / "tasks" / TASK / f"{index:02d}"
        (task_dir / "documents").mkdir(parents=True)
        (task_dir / "data" / "raw").mkdir(parents=True)
        (task_dir / "documents" / "protocol.md").write_text("CRASH\n" if index == crash_replay else "Protocol\n")
        rows = [
            (arm, generator.gauss(TRUTH if arm == "treatment" else 0.0, 1.0))
            for _ in range(50)
            for arm in ("treatment", "control")
        ]
        with (task_dir / "data" / "raw" / "outcomes.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("arm", "y"))
            writer.writerows((arm, repr(y)) for arm, y in rows)
        (task_dir / "task.json").write_text(
            json.dumps(
                {
                    "task_id": TASK,
                    "target_ids": ["primary"],
                    "available_materials": [
                        {"relative_path": f"tasks/{TASK}/documents/protocol.md"},
                        {"relative_path": f"tasks/{TASK}/data/raw/outcomes.csv"},
                    ],
                }
            )
        )
        estimate, lower, upper = _reference_report(rows)
        reference = ReplayBankReportV1(estimate=estimate, lower=lower, upper=upper)
        naive = ReplayBankReportV1(estimate=estimate - 0.3, lower=lower - 0.3, upper=upper - 0.3)
        records.append(
            ReplayBankRecordV1(
                task_id=TASK,
                replay_index=index,
                truth=TRUTH,
                reference_route_id="difference_in_means",
                routes={
                    "difference_in_means": reference,
                    "policy:follow_plan": reference,
                    "policy:always_naive": naive,
                    "policy:always_robust": None,
                },
            )
        )
    (root / "records.jsonl").write_text("".join(record.model_dump_json() + "\n" for record in records))
    records_sha = hashlib.sha256((root / "records.jsonl").read_bytes()).hexdigest()
    manifest = {
        "schema_id": "trialagentbench.trialeval.replay_bank/v1",
        "release_id": RELEASE,
        "replays_per_law": REPLAYS,
        "seed_namespace": "fixture",
        "records_sha256": records_sha,
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()


def _saved(tmp_path: Path, name: str, source: str) -> Path:
    workspace = tmp_path / name
    (workspace / "scratch").mkdir(parents=True)
    (workspace / "scratch" / "analysis.py").write_text(source)
    return workspace


class _LocalSession:
    """Run the staged script with the host interpreter in place of the container."""

    runs = 0
    timeouts = 0  # the first ``timeouts`` non-exclusive sessions time out, as under host load
    exclusive: list[bool] = []

    def __init__(self, cwd: Path, *, exclusive: bool = False, **_: object) -> None:
        self.root = Path(cwd)
        type(self).exclusive.append(exclusive)
        self.loaded = not exclusive and len([item for item in type(self).exclusive if not item]) <= type(self).timeouts

    def execute_result(self, code: str, *, language: str) -> CodeExecutionResultV1:
        type(self).runs += 1
        if self.loaded:
            return CodeExecutionResultV1(status="timeout", output="", elapsed_seconds=300.0)
        completed = subprocess.run(  # noqa: S603 - the fixture's own script
            [sys.executable, "scratch/analysis.py"], cwd=self.root, capture_output=True, text=True, check=False
        )
        return CodeExecutionResultV1(
            status="success" if completed.returncode == 0 else "execution_error",
            output=completed.stderr,
            elapsed_seconds=0.0,
        )

    def snapshot_scratch(self) -> Path:
        return self.root / "scratch"

    def close(self) -> None:
        pass


@pytest.fixture
def local_executor(monkeypatch: pytest.MonkeyPatch) -> type[_LocalSession]:
    _LocalSession.runs, _LocalSession.timeouts, _LocalSession.exclusive = 0, 0, []
    monkeypatch.setattr(replay_scoring, "DockerCodeSession", _LocalSession)
    return _LocalSession


def _replay(bank_root: Path, sha: str, workspace: Path) -> AnswerReplaysV1:
    evidence = load_replay_evidence_v1(
        bank_root, expected_manifest_sha256=sha, expected_release_id=RELEASE, replays=None
    )
    return replay_answer_v1(
        bank=evidence.bank,
        bank_manifest_sha256=sha,
        task_id=TASK,
        primary_target_id="primary",
        workspace=workspace,
        entrypoint="analysis.py",
        source_sha256=saved_source_sha256(analysis_project_artifacts(workspace / "scratch" / "analysis.py")),
        image="fixture",
        limits=CodeExecutionLimitsV1(),
    )


def test_the_scripted_reference_scores_zero_on_every_replay_and_a_data_blind_answer_loses(
    tmp_path: Path, local_executor: type[_LocalSession]
) -> None:
    sha = _write_bank(tmp_path / "bank")
    evidence = load_replay_evidence_v1(
        tmp_path / "bank", expected_manifest_sha256=sha, expected_release_id=RELEASE, replays=None
    )
    records = evidence.bank.records[TASK]
    target = _target()
    reference = _replay(
        tmp_path / "bank", sha, _saved(tmp_path, "reference", "CRASH_ON_MARKER = False\n" + REFERENCE_SOURCE)
    )
    assert [item.status for item in reference.replays] == ["answered"] * REPLAYS
    for record, outcome in zip(records, reference.replays, strict=True):
        single = AnswerReplaysV1(
            task_id=TASK, source_sha256=reference.source_sha256, bank_manifest_sha256=sha, replays=(outcome,)
        )
        assert score_replays_v1(target, (record,), single).replay_regret_musd == 0.0
    score = score_replays_v1(target, records, reference)
    assert (score.scoring, score.reproduced, score.replay_regret_musd) == ("replay", True, 0.0)
    assert score.consequential_share == 0.0 and score.answered_share == 1.0
    assert score.interval_coverage is not None and score.normalised_absolute_error is not None
    assert anchor_scores_v1(target, records)["scripted_reference"] == score

    constant = _replay(tmp_path / "bank", sha, _saved(tmp_path, "constant", CONSTANT_SOURCE))
    blind = score_replays_v1(target, records, constant)
    assert blind.replay_regret_musd > 5.0
    assert blind.consequential_share == 1.0
    assert blind.replay_regret_se is not None


def test_replays_are_deterministic_and_unreproduced_answers_are_priced_as_the_prior(
    tmp_path: Path, local_executor: type[_LocalSession]
) -> None:
    sha = _write_bank(tmp_path / "bank")
    workspace = _saved(tmp_path, "reference", "CRASH_ON_MARKER = False\n" + REFERENCE_SOURCE)
    first = _replay(tmp_path / "bank", sha, workspace)
    second = _replay(tmp_path / "bank", sha, workspace)
    assert first == second
    records = load_replay_evidence_v1(
        tmp_path / "bank", expected_manifest_sha256=sha, expected_release_id=RELEASE, replays=None
    ).bank.records[TASK]
    target = _target()
    assert score_replays_v1(target, records, first) == score_replays_v1(target, records, second)

    failure = score_replays_v1(target, records, None)
    prior = anchor_scores_v1(target, records)["prior_only"]
    assert (failure.scoring, failure.reproduced, failure.answered_share) == ("failure", False, 0.0)
    assert prior == failure  # the prior-only anchor is the failure price, labelled as a failed answer


def test_a_crashing_replay_is_priced_as_the_prior_for_that_replay(
    tmp_path: Path, local_executor: type[_LocalSession]
) -> None:
    sha = _write_bank(tmp_path / "bank", crash_replay=3)
    replays = _replay(
        tmp_path / "bank", sha, _saved(tmp_path, "fragile", "CRASH_ON_MARKER = True\n" + REFERENCE_SOURCE)
    )
    assert [item.status for item in replays.replays] == ["answered", "answered", "execution_error", *["answered"] * 3]
    records = load_replay_evidence_v1(
        tmp_path / "bank", expected_manifest_sha256=sha, expected_release_id=RELEASE, replays=None
    ).bank.records[TASK]
    target = _target()
    score = score_replays_v1(target, records, replays)
    assert score.answered_share == pytest.approx(5 / 6)
    crashed = AnswerReplaysV1(
        task_id=TASK, source_sha256=replays.source_sha256, bank_manifest_sha256=sha, replays=(replays.replays[2],)
    )
    assert score_replays_v1(target, (records[2],), crashed).replay_regret_musd == pytest.approx(
        score_replays_v1(target, (records[2],), None).replay_regret_musd
    )
    assert score.replay_regret_musd == pytest.approx(
        score_replays_v1(target, (records[2],), None).replay_regret_musd / 6
    )


def test_a_bank_with_another_digest_fails_fast(tmp_path: Path) -> None:
    _write_bank(tmp_path / "bank")
    with pytest.raises(ValueError, match="expected digest"):
        load_replay_evidence_v1(
            tmp_path / "bank", expected_manifest_sha256="0" * 64, expected_release_id=RELEASE, replays=None
        )


def test_a_bank_of_another_release_fails_fast(tmp_path: Path) -> None:
    sha = _write_bank(tmp_path / "bank")
    with pytest.raises(ValueError, match="not the graded release"):
        load_replay_evidence_v1(
            tmp_path / "bank", expected_manifest_sha256=sha, expected_release_id="release_other", replays=None
        )


@pytest.mark.parametrize("offline", [True, False])
def test_a_replay_timing_out_offline_is_rerun_once_exclusively(
    tmp_path: Path, local_executor: type[_LocalSession], monkeypatch: pytest.MonkeyPatch, offline: bool
) -> None:
    monkeypatch.setattr(replay_scoring, "offline_executor_capacity", lambda: offline)
    local_executor.timeouts = 1
    sha = _write_bank(tmp_path / "bank")
    replays = _replay(
        tmp_path / "bank", sha, _saved(tmp_path, "reference", "CRASH_ON_MARKER = False\n" + REFERENCE_SOURCE)
    )
    if offline:  # load is not scored: the exclusive rerun answers
        assert local_executor.exclusive[:2] == [False, True]
        assert [item.status for item in replays.replays] == ["answered"] * REPLAYS
    else:  # outside an offline grading scope a timeout is the script's
        assert True not in local_executor.exclusive
        assert [item.status for item in replays.replays] == ["timeout", *["answered"] * (REPLAYS - 1)]


def test_a_changed_saved_script_is_refused(tmp_path: Path, local_executor: type[_LocalSession]) -> None:
    sha = _write_bank(tmp_path / "bank")
    evidence = load_replay_evidence_v1(
        tmp_path / "bank", expected_manifest_sha256=sha, expected_release_id=RELEASE, replays=None
    )
    with pytest.raises(ValueError, match="differs from its graded source"):
        replay_answer_v1(
            bank=evidence.bank,
            bank_manifest_sha256=sha,
            task_id=TASK,
            primary_target_id="primary",
            workspace=_saved(tmp_path, "constant", CONSTANT_SOURCE),
            entrypoint="analysis.py",
            source_sha256="0" * 64,
            image="fixture",
            limits=CodeExecutionLimitsV1(),
        )
    assert local_executor.runs == 0


def test_a_task_without_bank_records_is_scored_once_at_the_truth() -> None:
    target = _target()
    own = score_single_dataset_v1(target, (0.2, 0.0, 0.4))
    assert (own.scoring, own.reproduced, own.replay_regret_musd, own.replay_regret_se) == (
        "single_dataset_truth",
        True,
        0.0,
        None,
    )
    failure = score_single_dataset_v1(target, None)
    assert (failure.scoring, failure.answered_share) == ("failure", 0.0)
    assert failure.replay_regret_musd > 0.0


def test_scores_scale_with_the_economics_band_and_the_answer_table_has_the_contract_columns() -> None:
    target = _target()
    record = ReplayBankRecordV1(
        task_id=TASK,
        replay_index=1,
        truth=TRUTH,
        reference_route_id="r",
        routes={"r": ReplayBankReportV1(estimate=0.3, lower=0.1, upper=0.5)},
    )
    outcome = ReplayOutcomeV1(replay_index=1, status="answered", estimate=-0.4, lower=-0.45, upper=-0.35)
    replays = AnswerReplaysV1(task_id=TASK, source_sha256="1" * 64, bank_manifest_sha256="2" * 64, replays=(outcome,))
    declared = score_replays_v1(target, (record,), replays)
    doubled = score_replays_v1(target, (record,), replays, value=2.0)
    assert doubled.replay_regret_musd > declared.replay_regret_musd > 0
    row = replay_scoring.TrialEvalAnswerRowV1(
        model="m",
        task_id=TASK,
        cell="cell",
        context="C5",
        law_id="trial",
        value_multiplier=1.0,
        cost_multiplier=1.0,
        **declared.model_dump(),
    )
    failed = replay_scoring.TrialEvalAnswerRowV1(
        model="m",
        task_id=TASK,
        cell="cell",
        context="C5",
        law_id="trial",
        failure_cause="unreproduced",
        value_multiplier=2.0,
        cost_multiplier=1.0,
        **score_replays_v1(target, (record,), None, value=2.0).model_dump(),
    )
    header, values, failure = list(csv.reader(answers_csv((row, failed)).splitlines()))
    assert tuple(header) == replay_scoring.ANSWER_COLUMNS
    assert values[:6] == ["m", TASK, "cell", "C5", "trial", "replay"]
    assert math.isclose(float(values[7]), declared.replay_regret_musd)
    assert values[-5:-2] == ["", "1.0", "1.0"] and failure[-5:-2] == ["unreproduced", "2.0", "1.0"]
    with pytest.raises(ValueError, match="failure cause"):
        replay_scoring.TrialEvalAnswerRowV1.model_validate(row.model_dump() | {"failure_cause": "no_answer"})


def test_best_value_at_truth_stops_without_benefit_and_confirms_with_a_large_one() -> None:
    economics = DecisionEconomicsV1(value_of_success_musd=1000.0, replicate_cost_musd=20.0, fixed_trial_cost_musd=10.0)
    assert best_value_at_truth(-1.0, scale=1.0, economics=economics) == 0.0
    best = best_value_at_truth(3.0, scale=1.0, economics=economics)
    actions = [
        DecisionActionV1(confirm=True, size_multiple=m, expected_value_musd=0.0) for m in CONFIRMATORY_SIZE_MULTIPLES
    ]
    assert best == pytest.approx(
        max(realised_value_v1(action, true_benefit=3.0, scale=1.0, economics=economics) for action in actions)
    )
    assert best > 0.0


def _write_public_bank(full: Path, sha: str, root: Path) -> str:
    """Derive the public bank of ``full`` beside it, with the same task datasets; return its manifest SHA-256."""
    manifest, records = public_replay_bank_v1(load_replay_bank(full, expected_manifest_sha256=sha))
    shutil.copytree(full / "tasks", root / "tasks")
    (root / "manifest.json").write_bytes(manifest)
    (root / "records.jsonl").write_bytes(records)
    return hashlib.sha256(manifest).hexdigest()


def test_the_public_bank_scores_answers_exactly_as_the_full_bank(
    tmp_path: Path, local_executor: type[_LocalSession]
) -> None:
    full_sha = _write_bank(tmp_path / "full", crash_replay=3)
    public_sha = _write_public_bank(tmp_path / "full", full_sha, tmp_path / "public")
    assert public_sha != full_sha
    target = _target()
    banks = {
        sha: load_replay_bank(root, expected_manifest_sha256=sha)
        for root, sha in ((tmp_path / "full", full_sha), (tmp_path / "public", public_sha))
    }
    for name, source in (
        ("reference", "CRASH_ON_MARKER = False\n" + REFERENCE_SOURCE),
        ("fragile", "CRASH_ON_MARKER = True\n" + REFERENCE_SOURCE),
        ("constant", CONSTANT_SOURCE),
    ):
        workspace = _saved(tmp_path, name, source)
        on_full = _replay(tmp_path / "full", full_sha, workspace)
        on_public = _replay(tmp_path / "public", public_sha, workspace)
        # The same datasets give the same outcomes (an error message names its temporary directory).
        assert [item.model_dump(exclude={"error"}) for item in on_public.replays] == [
            item.model_dump(exclude={"error"}) for item in on_full.replays
        ]
        for replays in (on_full, None):
            for value, cost in replay_scoring.ECONOMICS_BAND.values():
                full_score = score_replays_v1(target, banks[full_sha].records[TASK], replays, value=value, cost=cost)
                public_score = score_replays_v1(
                    target, banks[public_sha].records[TASK], replays, value=value, cost=cost
                )
                assert public_score.model_dump_json() == full_score.model_dump_json()


def test_the_public_bank_records_only_the_truth_and_the_reference(tmp_path: Path) -> None:
    full_sha = _write_bank(tmp_path / "full")
    public_sha = _write_public_bank(tmp_path / "full", full_sha, tmp_path / "public")
    text = (tmp_path / "public" / "records.jsonl").read_text()
    assert "policy:" not in text and "difference_in_means" not in text
    bank = load_replay_bank(tmp_path / "public", expected_manifest_sha256=public_sha)
    assert bank.manifest.variant == "public"
    for record in bank.records[TASK]:
        assert record.reference_route_id == PUBLIC_REFERENCE_ROUTE_ID and set(record.routes) == {"reference"}
    assert load_replay_bank(tmp_path / "full", expected_manifest_sha256=full_sha).manifest.variant == "full"
    with pytest.raises(ValueError, match="derived from the full bank"):
        public_replay_bank_v1(bank)


def test_fixed_policy_anchors_refuse_the_public_bank(tmp_path: Path) -> None:
    full_sha = _write_bank(tmp_path / "full")
    public_sha = _write_public_bank(tmp_path / "full", full_sha, tmp_path / "public")
    records = load_replay_bank(tmp_path / "public", expected_manifest_sha256=public_sha).records[TASK]
    with pytest.raises(ValueError, match="fixed-policy anchors need the operator's full replay bank"):
        anchor_scores_v1(_target(), records)


def test_a_public_bank_carrying_another_route_is_refused(tmp_path: Path) -> None:
    full_sha = _write_bank(tmp_path / "full")
    root = tmp_path / "public"
    _write_public_bank(tmp_path / "full", full_sha, root)
    (root / "records.jsonl").write_text(
        (root / "records.jsonl").read_text().replace('"routes":{', '"routes":{"policy:always_naive":null,')
    )
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["records_sha256"] = hashlib.sha256((root / "records.jsonl").read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="only the reference route"):
        load_replay_bank(
            root, expected_manifest_sha256=hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
        )


def test_the_answer_table_leaves_fixed_policy_anchors_to_the_operator_with_the_public_bank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from trialagentbench_verifier import campaign

    full_sha = _write_bank(tmp_path / "full")
    public_sha = _write_public_bank(tmp_path / "full", full_sha, tmp_path / "public")
    monkeypatch.setattr(campaign, "_decision_consequence", lambda row: None)
    monkeypatch.setattr(campaign, "failure_cause_v1", lambda row: "no_answer")
    target = _target()
    rows = {}
    for sha, root in ((full_sha, tmp_path / "full"), (public_sha, tmp_path / "public")):
        evidence = load_replay_evidence_v1(
            root, expected_manifest_sha256=sha, expected_release_id=RELEASE, replays=None
        )
        records = evidence.bank.records[TASK]
        scores = {
            name: score_replays_v1(target, records, None, value=value, cost=cost)
            for name, (value, cost) in replay_scoring.ECONOMICS_BAND.items()
        }
        row = SimpleNamespace(
            task_id=TASK, cell_ref="cell", context_tier="C5", trial_id="law", model_id="model", target=target
        )
        rows[sha] = campaign._answer_rows((SimpleNamespace(row=row, scores=scores),), evidence)
    models = {sha: {row.model for row in items} for sha, items in rows.items()}
    assert models[public_sha] == {"model"}
    assert models[full_sha] == {"model", *(f"policy:{name}" for name in replay_scoring.ANCHOR_POLICIES)}
    answers = [row for row in rows[full_sha] if row.model == "model"]
    assert [row.model_dump_json() for row in answers] == [row.model_dump_json() for row in rows[public_sha]]


@pytest.mark.parametrize("value,cost", replay_scoring.ECONOMICS_BAND.values())
def test_second_credited_analysis_and_public_upper_bound(value: float, cost: float) -> None:
    """Either credited route avoids positive operator regret, including the second route."""
    from trialagentbench_verifier.replay_bank import ReplayBankManifestV1, ReplayBankV1

    target = _target()
    assert target.consequence is not None
    first = ReplayBankReportV1(estimate=0.4, lower=0.2, upper=0.6)
    second = ReplayBankReportV1(estimate=-0.4, lower=-0.6, upper=-0.2)
    target = target.model_copy(
        update={
            "consequence": target.consequence.model_copy(
                update={
                    "credited": (
                        (first.estimate, first.lower, first.upper),
                        (second.estimate, second.lower, second.upper),
                    )
                }
            )
        }
    )
    record = ReplayBankRecordV1(
        task_id=TASK,
        replay_index=1,
        truth=TRUTH,
        reference_route_id="a",
        credited_route_ids=("a", "z"),
        routes={"a": first, "z": second},
    )
    outcomes = AnswerReplaysV1(
        task_id=TASK,
        source_sha256="0" * 64,
        bank_manifest_sha256="1" * 64,
        replays=(
            ReplayOutcomeV1(
                replay_index=1, status="answered", estimate=second.estimate, lower=second.lower, upper=second.upper
            ),
        ),
    )
    operator = score_replays_v1(target, (record,), outcomes, value=value, cost=cost)
    assert operator.replay_regret_musd == 0.0
    assert operator.consequential_share == 0.0
    assert (
        score_single_dataset_v1(
            target, (second.estimate, second.lower, second.upper), value=value, cost=cost
        ).replay_regret_musd
        == 0.0
    )
    bank = ReplayBankV1(
        root=Path("unused"),
        manifest=ReplayBankManifestV1(
            release_id=RELEASE, replays_per_law=1, seed_namespace="fixture", records_sha256="0" * 64
        ),
        records={TASK: (record,)},
    )
    _, encoded = public_replay_bank_v1(bank)
    public = ReplayBankRecordV1.model_validate_json(encoded)
    # Identical public schema for single- and multi-credited tasks; no route names or credit counts.
    assert set(public.model_dump(exclude_defaults=True)) == {
        "task_id",
        "replay_index",
        "truth",
        "reference_route_id",
        "routes",
    }
    assert set(public.routes) == {"reference"}
    assert "credited_route_ids" not in encoded.decode()
    for answer in (outcomes, None):
        full_score = score_replays_v1(target, (record,), answer, value=value, cost=cost)
        public_score = score_replays_v1(target, (public,), answer, value=value, cost=cost)
        assert public_score.replay_regret_musd >= full_score.replay_regret_musd
    with pytest.raises(ValueError, match="every credited route"):
        score_replays_v1(target, (record.model_copy(update={"credited_route_ids": ()}),), outcomes)


@pytest.mark.parametrize("credited", [("a", "missing"), ("a", "a"), ("b",)])
def test_credited_bank_routes_must_be_present_unique_and_include_primary(credited: tuple[str, ...]) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="credited replay routes"):
        ReplayBankRecordV1(
            task_id=TASK,
            replay_index=1,
            truth=TRUTH,
            reference_route_id="a",
            credited_route_ids=credited,
            routes={"a": None, "b": None},
        )


def test_truth_value_anchors_are_shared_by_failures_answers_and_fixed_policies(tmp_path: Path) -> None:
    bank_sha = _write_bank(tmp_path / "bank")
    records = load_replay_bank(tmp_path / "bank", expected_manifest_sha256=bank_sha).records[TASK]
    target = _target()
    anchors = anchor_scores_v1(target, records)
    failure = score_replays_v1(target, records, None)
    assert failure == anchors["prior_only"]
    assert failure.value_answer_musd == failure.value_prior_only_musd
    for score in anchors.values():
        assert score.value_prior_only_musd == failure.value_prior_only_musd
        assert score.value_truth_known_musd == failure.value_truth_known_musd
        assert len(score.replay_values) == len(records)
        assert [row.replay_index for row in score.replay_values] == [r.replay_index for r in records]
        for values in score.replay_values:
            assert values.value_truth_known_musd >= values.value_answer_musd
            assert values.value_truth_known_musd >= values.value_prior_only_musd
    economics = replay_scoring.economics_v1(target)
    for size in CONFIRMATORY_SIZE_MULTIPLES:
        action = DecisionActionV1(confirm=True, size_multiple=size, expected_value_musd=0)
        value = realised_value_v1(
            action, true_benefit=TRUTH, scale=target.reference_sampling_standard_deviation, economics=economics
        )
        assert failure.replay_values[0].value_truth_known_musd >= value


def test_operator_takes_the_credited_minimum_on_each_replay() -> None:
    target = _target()
    assert target.consequence is not None
    a = ReplayBankReportV1(estimate=0.4, lower=0.2, upper=0.6)
    b = ReplayBankReportV1(estimate=-0.4, lower=-0.6, upper=-0.2)
    target = target.model_copy(
        update={
            "consequence": target.consequence.model_copy(
                update={"credited": ((a.estimate, a.lower, a.upper), (b.estimate, b.lower, b.upper))}
            )
        }
    )
    records = tuple(
        ReplayBankRecordV1(
            task_id=TASK,
            replay_index=i,
            truth=TRUTH,
            reference_route_id="a",
            credited_route_ids=("a", "z"),
            routes={"a": x, "z": y},
        )
        for i, (x, y) in enumerate(((a, b), (b, a)), start=1)
    )
    result = score_replays_v1(target, records, None)
    pricer = replay_scoring._Pricer(target, value=1.0, cost=1.0)
    choices = [
        pricer.against_credited(None, (replay_scoring._bank_report(r.routes[route]),), truth=r.truth).regret_musd
        for r in records
        for route in r.reference_route_ids
    ]
    assert result.replay_regret_musd == statistics.fmean((min(choices[:2]), min(choices[2:])))
    assert result.replay_regret_musd < min(statistics.fmean(choices[::2]), statistics.fmean(choices[1::2]))


def test_consequential_flip_requires_disagreement_with_every_credited_analysis(monkeypatch):
    target = _target()
    assert target.consequence is not None
    economics = target.consequence.economics.model_copy(
        update={"replicate_cost_musd": 50.0, "fixed_trial_cost_musd": 100.0}
    )
    target = target.model_copy(update={"consequence": target.consequence.model_copy(update={"economics": economics})})
    pricer = replay_scoring._Pricer(target, value=1.0, cost=1.0)
    actions = [
        DecisionActionV1(confirm=True, size_multiple=size, expected_value_musd=0)
        for size in CONFIRMATORY_SIZE_MULTIPLES
    ]

    def value(action):
        return realised_value_v1(
            action, true_benefit=TRUTH, scale=target.reference_sampling_standard_deviation, economics=pricer.economics
        )

    worst, best = min(actions, key=value), max(actions, key=value)
    assert value(worst) < -5 < 0 < value(best)
    reports = [replay_scoring._decision_report(float(i), float(i) - 0.1, float(i) + 0.1) for i in range(3)]
    choices = {0.0: worst, 1.0: DecisionActionV1(confirm=False, size_multiple=0, expected_value_musd=0), 2.0: best}
    monkeypatch.setattr(pricer, "action", lambda report: choices[report.estimate])
    priced = pricer.against_credited(reports[0], reports[1:], truth=TRUTH)
    assert priced.regret_musd > 5
    assert not priced.consequential


_MATERIAL_FIELDS = {"material_decision_error_rate", "material_decision_error_rate_se"}


def _material_scores(tmp_path):
    """Price the fixed bank's reports, failures and policies at every economics point."""
    root = tmp_path / "material-bank"
    digest = _write_bank(root)
    records = load_replay_evidence_v1(
        root, expected_manifest_sha256=digest, expected_release_id=RELEASE, replays=None
    ).bank.records[TASK]
    target = _target()
    reports = [None, (-0.4, -0.45, -0.35), (0.15, 0.1, 0.2), (0.35, -0.05, 0.75), (1.0, 0.95, 1.05)]
    output = []
    for value, cost in ((1.0, 1.0), (0.5, 1.0), (2.0, 1.0), (1.0, 0.5), (1.0, 2.0)):
        for report in reports:
            outcomes = (
                None
                if report is None
                else AnswerReplaysV1(
                    task_id=TASK,
                    source_sha256="1" * 64,
                    bank_manifest_sha256=digest,
                    replays=tuple(
                        ReplayOutcomeV1(
                            replay_index=r.replay_index,
                            status="answered",
                            estimate=report[0],
                            lower=report[1],
                            upper=report[2],
                        )
                        for r in records
                    ),
                )
            )
            output.append(score_replays_v1(target, records, outcomes, value=value, cost=cost))
        partial = AnswerReplaysV1(
            task_id=TASK,
            source_sha256="1" * 64,
            bank_manifest_sha256=digest,
            replays=tuple(
                (
                    ReplayOutcomeV1(
                        replay_index=r.replay_index, status="answered", **r.routes[r.reference_route_id].model_dump()
                    )
                    if r.replay_index % 2
                    else ReplayOutcomeV1(replay_index=r.replay_index, status="timeout")
                )
                for r in records
            ),
        )
        output.append(score_replays_v1(target, records, partial, value=value, cost=cost))
        output.extend(anchor_scores_v1(target, records, value=value, cost=cost).values())
    return output


def test_the_material_rate_leaves_every_other_replay_field_byte_for_byte(tmp_path):
    """The complete serialized score without the material rate has a fixed digest."""
    scores = _material_scores(tmp_path)
    others = json.dumps(
        [score.model_dump(mode="json", exclude=_MATERIAL_FIELDS) for score in scores], separators=(",", ":")
    ).encode()
    assert hashlib.sha256(others).hexdigest() == "12f99782ded2bda5c039a6a847622d4e107efa67cdf44606f3aa9b9cdb9c507b"
    assert all(score.material_decision_error_rate >= score.consequential_share for score in scores)
    assert any(0 < score.material_decision_error_rate < 1 for score in scores)
    for score in scores:
        rate = score.material_decision_error_rate
        assert score.material_decision_error_rate_se == pytest.approx(math.sqrt(rate * (1 - rate) / (REPLAYS - 1)))
    import io

    rows = [
        replay_scoring.TrialEvalAnswerRowV1(
            model="m",
            task_id=TASK,
            cell="cell",
            context="C5",
            law_id="law",
            failure_cause="no_answer" if score.scoring == "failure" else None,
            value_multiplier=1,
            cost_multiplier=1,
            **score.model_dump(),
        )
        for score in scores[:6]
    ]
    legacy_csv = io.StringIO()
    csv.writer(legacy_csv, lineterminator="\n").writerows(
        row[:-2] for row in csv.reader(answers_csv(rows).splitlines())
    )
    assert (
        hashlib.sha256(legacy_csv.getvalue().encode()).hexdigest()
        == "4c985f31250b1cd67ed0bf69ce5449186b807d8f5472779cf110fc230fd17c24"
    )


@pytest.mark.parametrize("value,cost", [(1.0, 1.0), (0.5, 1.0), (2.0, 1.0), (1.0, 0.5), (1.0, 2.0)])
def test_material_rate_counts_sizing_losses_and_prior_only_at_every_economics_point(monkeypatch, value, cost):
    """Same-action sizing losses enter the new rate without changing the flip-only rate."""
    pricer = replay_scoring._Pricer(_target(), value=value, cost=cost)
    answer = replay_scoring.DecisionReportV1(estimate=0.1, lower=0.0, upper=0.2)
    reference = replay_scoring.DecisionReportV1(estimate=0.2, lower=0.1, upper=0.3)
    small = DecisionActionV1(confirm=True, size_multiple=0.5, expected_value_musd=0.0)
    large = DecisionActionV1(confirm=True, size_multiple=2.0, expected_value_musd=1.0)
    monkeypatch.setattr(pricer, "action", lambda report: small if report == answer else large)
    monkeypatch.setattr(pricer, "prior", small)
    monkeypatch.setattr(
        replay_scoring, "realised_value_v1", lambda action, **kwargs: (5.0 * value if action == large else 0.0)
    )
    monkeypatch.setattr(replay_scoring, "best_value_at_truth", lambda *args, **kwargs: 5.0 * value)
    priced = [
        pricer.against_credited(report, (reference,), truth=TRUTH, replay_index=i)
        for i, report in enumerate((answer, None), 1)
    ]
    score = replay_scoring._summary("replay", reproduced=True, priced=priced, answered=1, accuracy=(None, None, None))
    assert score.consequential_share == 0.0
    assert score.material_decision_error_rate == 1.0
    assert score.material_decision_error_rate_se == 0.0
    assert all(item.material_decision_error for item in priced)
    monkeypatch.setattr(
        replay_scoring, "realised_value_v1", lambda action, **kwargs: (4.999 * value if action == large else 0.0)
    )
    below = pricer.against_credited(answer, (reference,), truth=TRUTH)
    assert not below.material_decision_error and not below.consequential


def test_material_rate_equals_flip_rate_without_material_sizing_loss_at_declared_economics(tmp_path):
    """The scripted reference and prior anchor coincide when only decision flips exceed the bar."""
    root = tmp_path / "bank"
    sha = _write_bank(root)
    records = load_replay_evidence_v1(
        root, expected_manifest_sha256=sha, expected_release_id=RELEASE, replays=None
    ).bank.records[TASK]
    target = _target()
    reference = anchor_scores_v1(target, records)["scripted_reference"]
    stopped = AnswerReplaysV1(
        task_id=TASK,
        source_sha256="1" * 64,
        bank_manifest_sha256=sha,
        replays=tuple(
            ReplayOutcomeV1(
                replay_index=record.replay_index, status="answered", estimate=-0.4, lower=-0.45, upper=-0.35
            )
            for record in records
        ),
    )
    flipped = score_replays_v1(target, records, stopped)
    assert reference.material_decision_error_rate == reference.consequential_share == 0.0
    assert flipped.material_decision_error_rate == flipped.consequential_share == 1.0
