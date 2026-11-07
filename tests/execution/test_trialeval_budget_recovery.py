"""Budget-censored resumes preserve recorded outcomes, telemetry and task coverage."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from execution import test_trialeval_run_resume as R
from execution.test_trialeval_run_resume import _isolated_executor  # noqa: F401  (autouse fixture)

from trialagentbench.contracts.campaign import CampaignPlanV1
from trialagentbench.contracts.scoring.trialeval_scores import trialeval_infrastructure_outcome_v1
from trialagentbench.contracts.trace.observable import BenchmarkRuntimeTraceEventV1, runtime_event_source_payload_v1
from trialagentbench.io import read_json_model
from trialagentbench.io.checksums import canonical_payload_sha256
from trialagentbench.ports import LLMResponse
from trialagentbench.tools.run import trialeval
from trialagentbench.util.campaign_control import (
    CAMPAIGN_BUDGET_SCOPE,
    CampaignCapacityStop,
    request_control_for_run,
)
from trialagentbench.util.provider_telemetry import (
    fail_provider_request_v1,
    start_provider_request_v1,
    succeed_provider_request_v1,
)
from trialagentbench.util.reported_cost import RunStopRequested

CENSORED = "censored"
RETRY = "interrupted-retry"


@pytest.mark.skipif(os.getpriority(os.PRIO_PROCESS, 0) != 10, reason="Requires nice 10 scheduling")
def test_niceness():
    assert os.getpriority(os.PRIO_PROCESS, 0) == 10


def _setup(tmp_path: Path, monkeypatch, *, exhausted: tuple[str, ...], slice_ids=("TASK1001",)):
    arguments = R._args(R._participant_release(tmp_path), tmp_path / "runs")
    assert trialeval.main([*arguments, "--prepare-only"]) == 0
    native = next((tmp_path / "runs" / "fixture_model").iterdir())
    campaign_args = R._slice_campaign(tmp_path, native)
    control = request_control_for_run(campaign_root=tmp_path, job_id="fixture-trialeval-1", run_root=native)
    assert control is not None
    with sqlite3.connect(control.database) as connection:
        for scope in exhausted:
            connection.execute("INSERT INTO exhausted VALUES (?,?)", (scope, f"controlled exhaustion {scope}"))
    if slice_ids is not None:
        monkeypatch.setattr(trialeval, "execution_slice", lambda **kwargs: tuple(slice_ids))

    def forbidden(*args, **kwargs):
        raise AssertionError("Resume started a new model conversation")

    monkeypatch.setattr(trialeval, "get_provider", forbidden)
    return native, [*arguments, *campaign_args, "--resume-run-dir", str(native)], control


def _handle(native: Path, task: str, turn: int):
    return start_provider_request_v1(
        path=native / f"logs/{task}_provider_responses.jsonl",
        benchmark="trialeval",
        unit_id=task,
        phase_id="task",
        step_id="analysis",
        turn_index=turn,
        requested_model="fixture/model",
        provider_route="openai",
    )


def _stop(native: Path, task: str, turn: int, error: BaseException | None = None, received: bool = False):
    error = CampaignCapacityStop("controlled exhaustion") if error is None else error
    if received:
        error.raw_response = {"id": "received"}
    fail_provider_request_v1(_handle(native, task, turn), elapsed_seconds=0.1, failure_type="cancelled", error=error)


def _paid(native: Path, task: str, turn: int, outcome=None):
    succeed_provider_request_v1(
        _handle(native, task, turn),
        elapsed_seconds=0.1,
        response=LLMResponse(content="paid partial", model_outcome=outcome),
    )


def _runtime_terminal(native: Path, task: str, failure: str = "cancelled"):
    """Write a failed terminal trace for an interrupted item."""
    conversation = native / f"logs/{task}_conversation.json"
    conversation.parent.mkdir(exist_ok=True)
    conversation.write_text("[]\n")
    payload = runtime_event_source_payload_v1(
        benchmark="trialeval",
        task_id=task,
        program_id=None,
        scenario_id=None,
        objective_id=None,
        phase_id="task",
        step_id="analysis",
        event_type="step_terminal",
        terminal_status="failed",
        failure_type=failure,
        conversation_message=None,
    )
    terminal = BenchmarkRuntimeTraceEventV1(
        event_id=f"trialeval:{task}:000000",
        timestamp=datetime.now(UTC),
        source_artifact_path=str(conversation),
        source_payload_sha256=canonical_payload_sha256(payload),
        benchmark="trialeval",
        task_id=task,
        event_index=0,
        phase_id="task",
        step_id="analysis",
        event_type="step_terminal",
        terminal_status="failed",
        failure_type=failure,
    )
    (native / f"logs/{task}_events.jsonl").write_text(terminal.model_dump_json() + "\n")


def _snapshot(native: Path) -> dict[Path, bytes]:
    return {p: p.read_bytes() for p in (native / "logs").rglob("*") if p.is_file()}


def _terminal_of(native: Path, resumed: list[str], task: str = "TASK1001"):
    """Return (kind, failure_type, raised) after one resume."""
    raised = None
    try:
        code = trialeval.main(resumed)
        assert code == 0
    except CampaignCapacityStop as error:
        raised = error
    path = native / f"items/{task}.json"
    if not path.exists():
        return CENSORED if raised is not None else "pending", None, raised
    failure = json.loads(path.read_text())["agent_output"]["failure_type"]
    kind = RETRY if trialeval_infrastructure_outcome_v1(failure) is not None else "model-outcome"
    return kind, failure, raised


# --- 1. Matrix over event shape x ledger state -------------------------------------------------

STATES = {
    # name: (writer, runtime-terminal?)
    "stop_unanswered": (lambda n: _stop(n, "TASK1001", 1), True),
    "stop_unanswered_no_terminal": (lambda n: _stop(n, "TASK1001", 1), False),
    "stop_received": (lambda n: _stop(n, "TASK1001", 1, received=True), True),
    "keyboard_interrupt": (lambda n: _stop(n, "TASK1001", 1, error=KeyboardInterrupt()), True),
    "run_stop_requested": (lambda n: _stop(n, "TASK1001", 1, error=RunStopRequested("operator stop")), True),
    "paid_then_stop": (lambda n: (_paid(n, "TASK1001", 1), _stop(n, "TASK1001", 2)), True),
    "stop_then_started": (lambda n: (_stop(n, "TASK1001", 1), _handle(n, "TASK1001", 2)), True),
    "stop_then_paid": (lambda n: (_stop(n, "TASK1001", 1), _paid(n, "TASK1001", 2)), False),
    "stop_then_outcome": (lambda n: (_stop(n, "TASK1001", 1), _paid(n, "TASK1001", 2, "refusal")), False),
    "outcome_then_stop": (lambda n: (_paid(n, "TASK1001", 1, "refusal"), _stop(n, "TASK1001", 2)), True),
    "stop_torn_tail": (
        lambda n: (
            _stop(n, "TASK1001", 1),
            (n / "logs/TASK1001_provider_responses.jsonl").open("ab").write(b'{"request_id":'),
        ),
        True,
    ),
    "stop_start_only": (lambda n: _handle(n, "TASK1001", 1), True),  # crash before stop record
}

LEDGERS = {
    "none": (),
    "model": ("fixture",),
    "campaign": (CAMPAIGN_BUDGET_SCOPE,),
    "other_model": ("other",),
    "both": ("fixture", CAMPAIGN_BUDGET_SCOPE),
}


def _expected(state: str, ledger: str) -> str:
    exhausted = ledger in {"model", "campaign", "both"}
    if state == "stop_then_outcome":
        return "model-outcome"
    if state == "outcome_then_stop":
        # Recorded model outcomes take precedence over later requests.
        return "model-outcome"
    if state in {"stop_unanswered", "stop_unanswered_no_terminal", "paid_then_stop", "stop_torn_tail"} and exhausted:
        return CENSORED
    if state == "stop_then_paid":
        return "model-outcome-or-retry"
    return RETRY


@pytest.mark.parametrize("ledger", list(LEDGERS))
@pytest.mark.parametrize("state", list(STATES))
def test_guard_matrix(tmp_path, monkeypatch, state, ledger):
    native, resumed, _ = _setup(tmp_path, monkeypatch, exhausted=LEDGERS[ledger])
    writer, terminal = STATES[state]
    writer(native)
    if terminal:
        _runtime_terminal(native, "TASK1001")
    before = _snapshot(native)
    kind, failure, raised = _terminal_of(native, resumed)
    expected = _expected(state, ledger)
    record = {"state": state, "ledger": ledger, "kind": kind, "failure": failure, "expected": expected}
    print("RESULT", json.dumps(record))
    # Telemetry bytes are never rewritten.
    assert all(p.read_bytes().startswith(raw) for p, raw in before.items())
    if expected == "model-outcome-or-retry":
        assert kind in {"model-outcome", RETRY}
    else:
        assert kind == expected, record
    if kind == CENSORED:
        assert raised.campaign_wide is (ledger in {"campaign", "both"})
        assert json.loads((native / "coverage.json").read_text())["completed_unit_ids"] == []
        assert not (native / "failed_attempts").exists()
        # Idempotent second resume: same outcome, no bytes changed.
        snap = _snapshot(native)
        again = _terminal_of(native, resumed)
        assert again[0] == CENSORED and _snapshot(native) == snap
        assert read_json_model(CampaignPlanV1, native.parents[2] / "campaign.json").runs[0].retry_of_job_id is None
    else:
        first = (native / "items/TASK1001.json").read_bytes()
        assert _terminal_of(native, resumed)[1] == failure
        assert (native / "items/TASK1001.json").read_bytes() == first


# --- 2. Ledger authentication -------------------------------------------------------------------


def test_foreign_ledger_identity_fails_closed(tmp_path, monkeypatch):
    native, resumed, control = _setup(tmp_path, monkeypatch, exhausted=("fixture",))
    _stop(native, "TASK1001", 1)
    _runtime_terminal(native, "TASK1001")
    with sqlite3.connect(control.database) as connection:
        connection.execute("UPDATE identity SET digest='0'")
    with pytest.raises(ValueError, match="different frozen campaign"):
        trialeval.main(resumed)
    assert not (native / "items/TASK1001.json").exists()


def test_symlinked_ledger_fails_closed(tmp_path, monkeypatch):
    native, resumed, control = _setup(tmp_path, monkeypatch, exhausted=("fixture",))
    _stop(native, "TASK1001", 1)
    _runtime_terminal(native, "TASK1001")
    moved = control.database.with_name("elsewhere.sqlite")
    control.database.rename(moved)
    control.database.symlink_to(moved)
    with pytest.raises(ValueError):
        trialeval.main(resumed)
    assert not (native / "items/TASK1001.json").exists()


def test_missing_ledger_is_not_censored(tmp_path, monkeypatch):
    """A deleted ledger is recreated empty by the control constructor: no exhaustion, no censoring."""
    native, resumed, control = _setup(tmp_path, monkeypatch, exhausted=("fixture",))
    _stop(native, "TASK1001", 1)
    _runtime_terminal(native, "TASK1001")
    control.database.unlink()
    kind, failure, _ = _terminal_of(native, resumed)
    print("RESULT", json.dumps({"case": "missing_ledger", "kind": kind, "failure": failure}))
    assert kind == RETRY


def test_standalone_run_without_campaign_never_censors(tmp_path, monkeypatch):
    arguments = R._args(R._participant_release(tmp_path), tmp_path / "runs")
    assert trialeval.main([*arguments, "--prepare-only"]) == 0
    native = next((tmp_path / "runs" / "fixture_model").iterdir())
    monkeypatch.setattr(trialeval, "execution_slice", lambda **kwargs: ("TASK1001",))
    monkeypatch.setattr(trialeval, "get_provider", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    _stop(native, "TASK1001", 1)
    _runtime_terminal(native, "TASK1001")
    kind, failure, _ = _terminal_of(native, [*arguments, "--resume-run-dir", str(native)])
    assert kind == RETRY


# Recovery across a two-task slice.

PRE_TERMINAL = {
    # Interrupted sibling states and their expected terminal classifications.
    "refusal_no_terminal": (lambda n, t: _paid(n, t, 1, "refusal"), None, "contract_error"),
    "paid_crash_no_terminal": (lambda n, t: _paid(n, t, 1), None, "runtime_error"),
    "runtime_terminal_timeout": (lambda n, t: _paid(n, t, 1), "timeout", "timeout"),
}


@pytest.mark.parametrize("order", ["stop_first", "stop_last"])
@pytest.mark.parametrize("sibling", list(PRE_TERMINAL))
def test_budget_stop_does_not_hide_sibling_recovery(tmp_path, monkeypatch, order, sibling):
    """Recovery of one task must not depend on where a budget-stopped sibling sits in the slice."""
    stopped, other = ("TASK1001", "TASK1002") if order == "stop_first" else ("TASK1002", "TASK1001")
    native, resumed, _ = _setup(tmp_path, monkeypatch, exhausted=("fixture",), slice_ids=None)
    _stop(native, stopped, 1)
    _runtime_terminal(native, stopped)
    writer, terminal, expected = PRE_TERMINAL[sibling]
    writer(native, other)
    if terminal is not None:
        _runtime_terminal(native, other, terminal)
    try:
        trialeval.main(resumed)
    except CampaignCapacityStop:
        pass
    other_path = native / f"items/{other}.json"
    recovered = json.loads(other_path.read_text())["agent_output"]["failure_type"] if other_path.exists() else None
    # Campaign finish would classify every pending task of an exhausted model as budget-censored.
    from trialagentbench.tools.finish_campaign_analysis import campaign_censoring

    censoring = campaign_censoring(
        native.parents[2], read_json_model(CampaignPlanV1, native.parents[2] / "campaign.json")
    )
    censored = set() if censoring is None else set(censoring.runs[0].censored_task_ids)
    print(
        "RESULT",
        json.dumps({"sibling": sibling, "order": order, "recovered": recovered, "censored": sorted(censored)}),
    )
    assert not (native / f"items/{stopped}.json").exists()
    assert stopped in censored
    assert recovered == expected, f"{other} lost: finish censors {sorted(censored)} instead of scoring {expected}"
    assert other not in censored


@pytest.mark.parametrize("order", ["stop_first", "stop_last"])
def test_repeated_resume_preserves_recovered_coverage_in_both_orders(tmp_path, monkeypatch, order):
    """The first resume recovers every sibling and publishes coverage, independently of slice order."""
    from trialagentbench.tools.finish_campaign_analysis import campaign_censoring

    stopped, other = ("TASK1001", "TASK1002") if order == "stop_first" else ("TASK1002", "TASK1001")
    native, resumed, _ = _setup(tmp_path, monkeypatch, exhausted=("fixture",), slice_ids=None)
    _stop(native, stopped, 1)
    _runtime_terminal(native, stopped)
    _paid(native, other, 1, "refusal")
    seen = []
    for _ in range(3):
        with pytest.raises(CampaignCapacityStop):
            trialeval.main(resumed)
        plan = read_json_model(CampaignPlanV1, native.parents[2] / "campaign.json")
        seen.append(
            (
                (native / f"items/{other}.json").exists(),
                sorted(campaign_censoring(native.parents[2], plan).runs[0].censored_task_ids),
            )
        )
    print("RESULT", json.dumps({"case": "repeat", "order": order, "seen": seen}))
    assert seen[0] == seen[1] == seen[2] == (True, [stopped])
