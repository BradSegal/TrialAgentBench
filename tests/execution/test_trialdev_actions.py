"""Participant action boundaries exclude evaluator-owned allocation and accounting."""

import pytest
from trialagentbench_test_helpers import write_minimal_trialdev_pool as _write_pool

from trialagentbench.io import read_json_model, write_json_model
from trialagentbench.ports import LLMResponse
from trialagentbench.trialdev.actions import TRIALDEV_ACTION, TrialDevActionSession
from trialagentbench.trialdev.resources import TrialDevBudgetLedger
from trialagentbench.util.runtime_context import RuntimeDeadline


@pytest.fixture(autouse=True)
def action_capture_boundary(monkeypatch):
    """Isolate action transactions from native replay, which has executor tests."""
    import json

    from trialagentbench_verifier.numerical import read_numerical_table

    from trialagentbench.ports import CodeExecutionResultV1

    def capture(call, *, session, workspace, context, **kwargs):
        try:
            rows = read_numerical_table(workspace / "scratch" / "estimates.csv", target_ids=context.target_ids)
            missing = sorted(set(context.target_ids) - {row.target_id for row in rows})
            error = "Missing numerical targets: " + ", ".join(missing) if missing else None
        except (ValueError, FileNotFoundError) as failure:
            error = str(failure)
        return (
            CodeExecutionResultV1(
                status="success" if error is None else "execution_error",
                output=json.dumps({"error": error}),
                elapsed_seconds=0.01,
            ),
            session,
        )

    monkeypatch.setattr("trialagentbench.trialdev.task_agent.check_analysis_tool_v1", capture)


def _start(design: dict, *, study_id: str = "study-101"):
    """Resolve flat start_study arguments the way the session does for a one-compound pool."""
    from trialagentbench.trialdev.actions import TrialDevProgrammeCall, TrialDevStartStudy, TrialDevStudyProposal

    proposal = TrialDevStudyProposal.model_validate(design)
    arms = [
        {"arm_id": "control", "asset_id": "control", "dose": None, "participant_count": proposal.control_participants}
    ]
    arms += [
        {
            "arm_id": arm.name or f"{arm.compound}-{arm.dose.value}",
            "asset_id": arm.compound,
            "dose": arm.dose,
            "participant_count": arm.participants,
        }
        for arm in proposal.arms
    ]
    action = TrialDevStartStudy(
        action="start_study",
        study_id=study_id,
        phase=proposal.phase,
        arms=arms,
        sites=proposal.sites,
        follow_up_days=proposal.follow_up_days,
        assessment_days=proposal.assessment_days or (28, 56),
        surveillance_days=proposal.surveillance_days or (28, 56),
        looks=proposal.looks,
    )
    return TrialDevProgrammeCall(action=action, preview=proposal.preview)


def _mock_session(tmp_path, ledger=None):
    """A mocked programme session that persists requests like the real one."""
    from unittest.mock import Mock

    from trialagentbench.trialdev.actions import TrialDevActionRequest

    programme = Mock(spec=TrialDevActionSession)
    programme.commitment_root = tmp_path / "private/commitments"
    programme.evidence_root = tmp_path / "workspace" / "evidence"
    ledger = ledger or TrialDevBudgetLedger(budget_usd=1_000_000)

    def execute(action, *, action_id):
        write_json_model(
            programme.commitment_root / f"{action_id}.request.json", TrialDevActionRequest(request=action)
        )
        return ledger

    programme.execute.side_effect = execute
    return programme


def _decoded(action, **kwargs):
    from trialagentbench.trialdev.actions import TRIALDEV_ACTION, TrialDevProgrammeCall

    return TrialDevProgrammeCall(action=TRIALDEV_ACTION.validate_python(action), **kwargs)


def _bare_agent(tmp_path, programme, *, step_id="programme"):
    from unittest.mock import Mock

    from trialagentbench.trialdev.task_agent import TrialDevTaskAgent

    agent = object.__new__(TrialDevTaskAgent)
    agent.deadline = RuntimeDeadline.after(30, label="action unit test")
    agent.reporting_context = None
    agent.executor_limits = None
    agent.executor_image = None
    agent.analysis_language = "python_r"
    agent.session = Mock()
    agent._executor_image_id = "sha256:" + "0" * 64
    agent.workdir = tmp_path / "workspace"
    agent.step_id = step_id
    agent.executed_code = []
    agent.execution_outputs = []
    agent.authored_programme = programme
    return agent


def test_inline_study_preview_submission_and_retry_preserve_design(tmp_path):
    """A preview commits nothing; a retried call keeps the design it committed first."""
    import json

    from trialagentbench.ports import ToolCall

    design = {
        "phase": "early_safety",
        "sites": 2,
        "arms": [{"compound": "candidate", "dose": "standard", "participants": 30, "name": "active"}],
        "control_participants": 30,
        "follow_up_days": 56,
        "assessment_days": [28, 56],
        "surveillance_days": [56],
        "looks": [{"deadline_day": 180, "completed_participants": 60}],
    }
    programme = _mock_session(tmp_path)
    programme.decode.side_effect = lambda name, arguments: _start(arguments)
    programme.preview_study.return_value = TrialDevBudgetLedger(budget_usd=1_000_000)
    agent = _bare_agent(tmp_path, programme)

    def call(identity, **changes):
        return agent._dispatch_local_tool(
            ToolCall(id=identity, name="start_study", arguments=json.dumps(design | changes))
        )

    preview = call("preview", preview=True)
    assert "study-101 preview (not started)" in preview.output and "active (candidate standard) 30" in preview.output
    programme.execute.assert_not_called()
    assert not programme.commitment_root.exists()
    design["sites"] = 4
    started = call("submit")
    assert "study-101 started" in started.output
    assert programme.execute.call_args.args[0].sites == 4
    design["sites"] = 8
    call("submit")
    assert programme.execute.call_args.args[0].sites == 4
    call("new-submit")
    assert programme.execute.call_args.args[0].sites == 8


def test_actions_cannot_supply_hidden_state_or_backdate_terminal_choice(tmp_path):
    budget = tmp_path / "private" / "budget.json"
    write_json_model(budget, TrialDevBudgetLedger(budget_usd=1000))
    session = TrialDevActionSession(tmp_path / "pool", budget, tmp_path / "evidence", tmp_path / "commits", 7)
    for payload in (
        {"action": "stop", "programme_day": 0},
        {"action": "stop", "cost": 0},
        {"action": "readout", "study_id": "study", "participant_ids": [1, 2]},
        {"action": "readout", "study_id": "study", "output_dir": "../private"},
        {"action": "advance", "asset_id": "asset", "dose": "standard", "effect": -10},
        {"action": "unknown"},
    ):
        with pytest.raises(ValueError):
            TRIALDEV_ACTION.validate_python(payload)
    wait = TRIALDEV_ACTION.validate_python({"action": "wait", "programme_day": 100})
    with pytest.raises(ValueError, match="filename component"):
        session.execute(wait, action_id="../escape")
    assert read_json_model(TrialDevBudgetLedger, budget).current_day == 0
    session.execute(wait, action_id="wait-1")
    with pytest.raises(ValueError, match="different inputs"):
        session.execute(TRIALDEV_ACTION.validate_python({"action": "wait", "programme_day": 200}), action_id="wait-1")
    assert read_json_model(TrialDevBudgetLedger, budget).current_day == 100

    stop = TRIALDEV_ACTION.validate_python({"action": "stop"})
    final = session.execute(stop, action_id="final")
    assert final.terminal is not None
    assert final.terminal.programme_day == 100
    assert session.execute(stop, action_id="final") == final
    with pytest.raises(ValueError, match="immutable"):
        session.execute(TRIALDEV_ACTION.validate_python({"action": "wait", "programme_day": 101}), action_id="late")
    assert read_json_model(TrialDevBudgetLedger, budget) == final


def test_agent_programme_tool_returns_only_public_state(tmp_path):
    import json

    from trialagentbench.ports import ToolCall
    from trialagentbench.ports.tool_input import ToolInputError

    budget = tmp_path / "private" / "budget.json"
    write_json_model(budget, TrialDevBudgetLedger(budget_usd=1000))
    agent = _bare_agent(
        tmp_path,
        TrialDevActionSession(
            tmp_path / "pool", budget, tmp_path / "workspace" / "evidence", tmp_path / "commits", 12345
        ),
        step_id="decision-1",
    )

    def call(name, payload):
        return agent._dispatch_local_tool(ToolCall(id="provider-call", name=name, arguments=json.dumps(payload)))

    with pytest.raises(ToolInputError, match="Extra inputs"):
        call("stop_development", {"cost": 0})
    design = {
        "phase": "early_safety",
        "sites": 4,
        "arms": [{"compound": "candidate", "dose": "standard", "participants": 30, "name": "16973std"}],
        "control_participants": 30,
        "follow_up_days": 56,
        "looks": [{"deadline_day": 180, "completed_participants": 60}],
    }
    before = budget.read_bytes()
    with pytest.raises(ToolInputError, match="pattern"):
        call("start_study", design)
    with pytest.raises(ToolInputError, match="Unknown local tool"):
        call("programme_action", {"request": {"action": "stop"}})
    assert budget.read_bytes() == before
    assert not (tmp_path / "commits").exists()
    internal = {
        "action": "start_study",
        "study_id": "study-101",
        "phase": "early_safety",
        "sites": 4,
        "arms": [
            {"arm_id": "control", "asset_id": "control", "dose": None, "participant_count": 30},
            {"arm_id": "active", "asset_id": "candidate", "dose": "standard", "participant_count": 30},
        ],
        "follow_up_days": 56,
        "assessment_days": [28, 56],
        "surveillance_days": [56],
        "looks": [{"deadline_day": 180, "completed_participants": 60}],
    }
    with pytest.raises(ValueError, match="below total enrollment"):
        TRIALDEV_ACTION.validate_python(internal | {"pause_after_enrolled": 60})
    assert TRIALDEV_ACTION.validate_python(internal | {"pause_after_enrolled": 30}).pause_after_enrolled == 30
    assert read_json_model(TrialDevBudgetLedger, budget).terminal is None
    result = call("stop_development", {})
    assert result.status == "success"
    assert result.output.startswith("## Development stopped on programme day 0")
    assert "available $0.0M" in result.output
    assert agent._programme_state.terminal.decision == "stop"
    assert "12345" not in result.output
    assert "pool_manifest_sha256" not in result.output
    assert str(tmp_path) not in result.output
    agent.authored_programme = None
    with pytest.raises(ToolInputError, match="No authored programme"):
        call("stop_development", {})


def test_agent_readout_identifies_workspace_evidence(tmp_path):
    from pathlib import Path
    from unittest.mock import Mock

    from trialagentbench.ports import ToolCall
    from trialagentbench.trialdev.materialization import (
        TrialDevReadoutArm,
        TrialDevReadoutManifest,
        TrialDevReadoutTarget,
    )
    from trialagentbench.trialdev.resources import TrialDevStudyCost, trialdev_phase_tariff

    programme = _mock_session(tmp_path, TrialDevBudgetLedger(budget_usd=1000))
    programme.decode.return_value = _decoded(
        {"action": "readout", "study_id": "study-101", "evidence_folder": "study-101-look-1"}
    )
    agent = _bare_agent(tmp_path, programme, step_id="decision-1")
    target = TrialDevReadoutTarget(
        target_id="active:clinical_outcome", arm_id="active", endpoint="clinical_outcome", unit="DAS28"
    )
    manifest = TrialDevReadoutManifest(
        study_id="study-101",
        outcome_object="Disease activity",
        readout_day=400,
        follow_up_days=168,
        assessment_days=(28, 168),
        surveillance_days=(28, 168),
        arms=(
            TrialDevReadoutArm(arm_id="active", asset_id="candidate", dose="standard"),
            TrialDevReadoutArm(arm_id="control", asset_id="control", dose=None),
        ),
        targets=(target,),
        tariff=trialdev_phase_tariff("early_safety"),
        sites=2,
        screened=60,
        cost=TrialDevStudyCost(accrued_usd=1.0, remaining_liability_usd=0.0),
    )
    agent._assign_readout_context = Mock(return_value=manifest)
    result = agent._dispatch_local_tool(
        ToolCall(id="readout-call", name="request_readout", arguments='{"study_id": "study-101"}')
    )
    agent._assign_readout_context.assert_called_once_with(Path("evidence/study-101-look-1"))
    assert result.output.startswith("## study-101 readout 1 (programme day 400)")
    assert "`evidence/study-101-look-1/`" in result.output and "enrollment_day" in result.output
    assert "| `active:clinical_outcome` | active minus control, clinical endpoint | DAS28 |" in result.output
    assert agent._programme_state.readout_path == "evidence/study-101-look-1"
    assert str(tmp_path) not in result.output


def test_cutoff_is_accepted_only_after_action_success_and_retry_preserves_capture(tmp_path):
    import json
    from unittest.mock import Mock

    from trialagentbench_verifier.numerical import NumericalReportingContext

    from trialagentbench.ports import ToolCall
    from trialagentbench.trialdev.task_agent import TrialDevTaskAgent

    agent = object.__new__(TrialDevTaskAgent)
    agent.deadline = RuntimeDeadline.after(30, label="action unit test")
    agent.reporting_context = None
    agent.executor_limits = None
    agent.executor_image = None
    agent.analysis_language = "python_r"
    agent.workdir = tmp_path / "workspace"
    scratch = agent.workdir / "scratch"
    scratch.mkdir(parents=True)
    (scratch / "analysis.py").write_text("# original analysis\n")
    (scratch / "estimates.csv").write_text("target_id,estimate,lower,upper\neffect,1,0,2\n")
    agent.session = Mock()
    agent._executor_image_id = "sha256:" + "0" * 64
    agent.step_id = "final"
    agent.executed_code = []
    agent.execution_outputs = []
    agent.reporting_context = NumericalReportingContext(
        checkpoint_id="study-one", target_ids=("effect",), evidence_artifacts=()
    )
    programme = Mock(spec=TrialDevActionSession)
    programme.evidence_root = agent.workdir / "evidence"
    programme.commitment_root = tmp_path / "commits"
    programme.decode.return_value = _decoded({"action": "stop"})
    from trialagentbench.trialdev.resources import TrialDevTerminalDecision

    ledger = TrialDevBudgetLedger(budget_usd=1000, terminal=TrialDevTerminalDecision(decision="stop", programme_day=0))
    programme.execute.side_effect = [ValueError("rejected action"), ledger, ledger]
    agent.authored_programme = programme
    call = ToolCall(id="stop-call", name="stop_development", arguments=json.dumps({}))
    with pytest.raises(ValueError, match="rejected action"):
        agent._dispatch_local_tool(call)
    captures = tmp_path / "analysis_sources"
    assert not (captures / "cutoffs").exists()
    pending = tuple((captures / "pending_cutoffs").iterdir())
    assert len(pending) == 1
    (scratch / "analysis.py").write_text("# later analysis\n")
    agent.reporting_context = None  # Recover the staged capture even before in-memory context restoration.
    agent._dispatch_local_tool(call)
    accepted = captures / "cutoffs" / pending[0].name
    assert (accepted / "analysis.py").read_text() == "# original analysis\n"
    assert not pending[0].exists()
    agent._dispatch_local_tool(call)
    assert (accepted / "analysis.py").read_text() == "# original analysis\n"
    agent.session.snapshot_scratch.assert_called_once()


@pytest.mark.parametrize("initial_action", ["none", "report", "invalid_wait", "report_then_action"])
@pytest.mark.parametrize("terminal_action", [False, True])
def test_authored_agent_requires_explicit_terminal_action(tmp_path, monkeypatch, initial_action, terminal_action):
    import json
    from types import SimpleNamespace
    from unittest.mock import Mock

    from trialagentbench.ports import LLMResponse, ToolCall
    from trialagentbench.trialdev import task_agent as owner

    submit_first = initial_action in {"report", "report_then_action"}
    same_turn = initial_action == "report_then_action"
    turn_limit = 1 if same_turn or initial_action == "none" else 2
    budget = tmp_path / "private" / "budget.json"
    write_json_model(budget, TrialDevBudgetLedger(budget_usd=1000))
    monkeypatch.setattr(owner, "DockerCodeSession", lambda **kwargs: Mock())
    agent = owner.TrialDevTaskAgent(
        event_log_path=tmp_path / "events.jsonl",
        provider=Mock(),
        workdir=tmp_path / "workspace",
        system_prompt="Develop the assets.",
        program_id="programme",
        scenario_id="case",
        objective_id="development",
        max_turns_per_step=turn_limit,
        authored_programme=TrialDevActionSession(
            tmp_path / "pool", budget, tmp_path / "workspace" / "evidence", tmp_path / "commits", 7
        ),
    )
    call = ToolCall(id="stop", name="stop_development", arguments="{}")
    provider_turn = Mock(
        return_value=SimpleNamespace(
            response=LLMResponse(tool_calls=[call] if terminal_action else []),
            message={"role": "assistant", "content": "Stop development."},
        )
    )
    if submit_first:
        from trialagentbench_verifier.numerical import NumericalReportingContext

        agent.reporting_context = NumericalReportingContext(
            checkpoint_id="observed", target_ids=("effect",), evidence_artifacts=()
        )
        scratch = agent.workdir / "scratch"
        scratch.mkdir()
        (scratch / "estimates.csv").write_text("target_id,estimate,lower,upper\neffect,1,0,2\n")
        report_call = ToolCall(
            id="report", name="submit_analysis_report", arguments=json.dumps({"content": "Observed summary."})
        )
        provider_turn.side_effect = [
            SimpleNamespace(
                response=LLMResponse(tool_calls=[report_call]), message={"role": "assistant", "content": "Report."}
            ),
            provider_turn.return_value,
        ]
    if same_turn:
        provider_turn.side_effect = [
            SimpleNamespace(
                response=LLMResponse(tool_calls=[report_call, call] if terminal_action else [report_call]),
                message={"role": "assistant", "content": "Report and decide."},
            )
        ]
    if initial_action == "invalid_wait":
        rejected_call = ToolCall(id="invalid-wait", name="wait_until", arguments=json.dumps({"programme_day": 2191}))
        provider_turn.side_effect = [
            SimpleNamespace(
                response=LLMResponse(tool_calls=[rejected_call]), message={"role": "assistant", "content": "Wait."}
            ),
            provider_turn.return_value,
        ]
    monkeypatch.setattr(agent, "_provider_turn", provider_turn)
    agent.begin_step(phase_id="development", step_id="programme")
    agent.append_user_message("Make the development decision.")
    if not terminal_action:
        with pytest.raises(owner.AgentTurnLimitExceeded, match="terminal development decision"):
            agent.run_programme()
        assert agent.step_terminal
        final_event = json.loads((tmp_path / "events.jsonl").read_text().splitlines()[-1])
        assert final_event["failure_type"] == "turn_limit_no_terminal_decision"
        assert read_json_model(TrialDevBudgetLedger, budget).terminal is None
        assert agent.turns_used == turn_limit
        return
    assert agent.run_programme() is None
    assert provider_turn.call_count == turn_limit
    assert agent.turns_used == turn_limit
    assert agent.step_terminal
    rejected = [message.get("content", "") for message in agent.messages if message.get("role") == "tool"]
    if initial_action == "invalid_wait":
        assert any("beyond the programme horizon, day 2190. Nothing was committed." in text for text in rejected)
        assert read_json_model(TrialDevBudgetLedger, budget).current_day == 0
    if submit_first:
        # Programmes keep reports at readouts and the decision; a separate report tool is not offered.
        assert any("'submit_analysis_report' was not offered" in text for text in rejected)
    assert read_json_model(TrialDevBudgetLedger, budget).terminal.decision == "stop"
    offered = {item["function"]["name"] for item in provider_turn.call_args.kwargs["tools"]}
    assert {"stop_development", "advance_candidate"} <= offered and "submit_analysis_report" not in offered

    calls_before_recovery = provider_turn.call_count
    ledger_before_recovery = budget.read_bytes()
    agent.begin_step(phase_id="development", step_id="recovered")
    agent.append_user_message("Recover the interrupted programme.")
    assert agent.run_programme() is None
    assert agent.step_terminal and agent.turns_used == 0
    assert provider_turn.call_count == calls_before_recovery
    assert budget.read_bytes() == ledger_before_recovery


def test_unavailable_treatment_is_recoverable_but_integrity_errors_are_not(tmp_path):
    import json
    from unittest.mock import Mock

    from trialagentbench.ports import ToolCall
    from trialagentbench.ports.tool_input import ToolInputError
    from trialagentbench.trialdev.errors import TrialDevActionRejectedError
    from trialagentbench.trialdev.task_agent import TrialDevTaskAgent

    agent = object.__new__(TrialDevTaskAgent)
    agent.deadline = RuntimeDeadline.after(30, label="action unit test")
    agent.reporting_context = None
    agent.executor_limits = None
    agent.executor_image = None
    agent.analysis_language = "python_r"
    agent.session = Mock()
    agent._executor_image_id = "sha256:" + "0" * 64
    agent.workdir = tmp_path / "workspace"
    agent.step_id = "design"
    agent.authored_programme = Mock(spec=TrialDevActionSession)
    agent.authored_programme.commitment_root = tmp_path / "commits"
    agent.authored_programme.decode.return_value = _decoded({"action": "wait", "programme_day": 0})
    call = ToolCall(id="design-call", name="wait_until", arguments=json.dumps({"programme_day": 0}))
    agent.authored_programme.execute.side_effect = TrialDevActionRejectedError("unavailable treatment")
    with pytest.raises(ToolInputError, match="unavailable treatment"):
        agent._dispatch_local_tool(call)
    agent.authored_programme.execute.side_effect = ValueError("private manifest hash changed")
    with pytest.raises(ValueError, match="private manifest hash changed") as failure:
        agent._dispatch_local_tool(call)
    assert not isinstance(failure.value, ToolInputError)


def test_unavailable_look_and_excess_horizon_leave_programme_unchanged(tmp_path):
    from trialagentbench.trialdev.errors import TrialDevActionRejectedError
    from trialagentbench.trialdev.materialization import trialdev_next_readout_day

    ledger = TrialDevBudgetLedger(budget_usd=1000)
    budget = tmp_path / "private" / "budget.json"
    write_json_model(budget, ledger)
    session = TrialDevActionSession(tmp_path / "pool", budget, tmp_path / "evidence", tmp_path / "commits", 7)
    with pytest.raises(TrialDevActionRejectedError, match="remaining committed look"):
        trialdev_next_readout_day(tmp_path / "pool", ledger, "unknown")
    before = budget.read_bytes()
    with pytest.raises(TrialDevActionRejectedError, match="beyond the programme horizon"):
        session.execute(TRIALDEV_ACTION.validate_python({"action": "wait", "programme_day": 2191}), action_id="late")
    assert budget.read_bytes() == before
    session.execute(TRIALDEV_ACTION.validate_python({"action": "wait", "programme_day": 100}), action_id="valid")
    advanced = budget.read_bytes()
    with pytest.raises(TrialDevActionRejectedError, match="already at day 100"):
        session.execute(
            TRIALDEV_ACTION.validate_python({"action": "wait", "programme_day": 99}), action_id="backdated"
        )
    assert budget.read_bytes() == advanced


@pytest.mark.parametrize(
    "action_payload",
    [
        {"action": "continue_recruitment", "study_id": "unknown"},
        {"action": "drop_arm", "study_id": "unknown", "arm_id": "active"},
        {"action": "reduce_control", "study_id": "unknown", "participant_count": 10},
        {"action": "advance", "asset_id": "asset", "dose": "standard"},
    ],
)
def test_unavailable_operational_actions_do_not_change_accounting(tmp_path, action_payload):
    from trialagentbench.trialdev.errors import TrialDevActionRejectedError

    budget = tmp_path / "private" / "budget.json"
    write_json_model(budget, TrialDevBudgetLedger(budget_usd=1000))
    session = TrialDevActionSession(tmp_path / "pool", budget, tmp_path / "evidence", tmp_path / "commits", 7)
    before = budget.read_bytes()
    with pytest.raises(TrialDevActionRejectedError):
        session.execute(TRIALDEV_ACTION.validate_python(action_payload), action_id="rejected")
    assert budget.read_bytes() == before
    assert not session.evidence_root.exists()


def test_invalid_source_project_does_not_block_terminal_decision(tmp_path):
    import json
    from unittest.mock import Mock

    from trialagentbench_verifier.numerical import NumericalReportingContext

    from trialagentbench.ports import ToolCall
    from trialagentbench.trialdev.task_agent import TrialDevTaskAgent

    workspace = tmp_path / "workspace"
    scratch = workspace / "scratch"
    scratch.mkdir(parents=True)
    outside = tmp_path / "private_source.py"
    outside.write_text("# must never enter a replay project\n")
    (scratch / "analysis.py").symlink_to(outside)
    budget = tmp_path / "private" / "budget.json"
    write_json_model(budget, TrialDevBudgetLedger(budget_usd=1000))
    agent = object.__new__(TrialDevTaskAgent)
    agent.deadline = RuntimeDeadline.after(30, label="action unit test")
    agent.reporting_context = None
    agent.executor_limits = None
    agent.executor_image = None
    agent.analysis_language = "python_r"
    agent.workdir = workspace
    agent.session = Mock()
    agent._executor_image_id = "sha256:" + "0" * 64
    agent.step_id = "terminal"
    agent.executed_code = []
    agent.execution_outputs = []
    agent.reporting_context = NumericalReportingContext(
        checkpoint_id="study", target_ids=("effect",), evidence_artifacts=()
    )
    agent.authored_programme = TrialDevActionSession(
        tmp_path / "pool", budget, workspace / "evidence", tmp_path / "commits", 7
    )
    result = agent._dispatch_local_tool(
        ToolCall(id="stop", name="stop_development", arguments=json.dumps({"accept_incomplete": True}))
    )
    assert result.status == "success"
    assert read_json_model(TrialDevBudgetLedger, budget).terminal.decision == "stop"
    (captured,) = (tmp_path / "analysis_sources" / "cutoffs").iterdir()
    assert (captured / "source_error.txt").is_file()
    assert not (captured / "analysis.py").exists()
    assert outside.read_text() == "# must never enter a replay project\n"


@pytest.mark.parametrize("defect", ["cross_run", "symlink_root", "traversal", "unknown_readout"])
def test_cutoff_custody_is_checked_before_starting_executor(tmp_path, monkeypatch, defect):
    from unittest.mock import Mock

    from trialagentbench_verifier.numerical import NumericalReportingContext

    from trialagentbench.ports import CodeExecutionLimitsV1
    from trialagentbench.trialdev.run import evaluate_trialdev_cutoff
    from trialagentbench.trialdev.value import TrialDevSponsorPreferences

    workspace = tmp_path / "run" / "workspace"
    workspace.mkdir(parents=True)
    accepted_root = workspace.parent / "analysis_sources" / "cutoffs"
    accepted_root.parent.mkdir()
    if defect == "symlink_root":
        external = tmp_path / "external"
        external.mkdir()
        accepted_root.symlink_to(external, target_is_directory=True)
    captured = (tmp_path / "other_run" if defect == "cross_run" else accepted_root) / "report"
    captured.mkdir(parents=True)
    write_json_model(
        captured / "reporting_context.json",
        NumericalReportingContext(
            checkpoint_id="evidence/../evidence/look" if defect == "traversal" else "evidence/look",
            target_ids=("active:clinical_outcome",),
            evidence_artifacts=(),
        ),
    )
    programme = TrialDevActionSession(
        tmp_path / "pool", tmp_path / "budget.json", workspace / "evidence", tmp_path / "commits", 7
    )
    replay = Mock(side_effect=AssertionError("unbound captures must never launch a container"))
    monkeypatch.setattr("trialagentbench.trialdev.run.replay_numerical_submission", replay)
    expected = FileNotFoundError if defect == "unknown_readout" else ValueError
    with pytest.raises(expected):
        evaluate_trialdev_cutoff(
            workspace=workspace,
            captured=captured,
            programme=programme,
            preferences=TrialDevSponsorPreferences(outcome_unit="DAS28", benefit_unit=1),
            executor_limits=CodeExecutionLimitsV1(),
        )
    replay.assert_not_called()


@pytest.mark.parametrize("termination", ["deadline", "turn_limit", "executor_exit", "session_snapshot"])
def test_programme_runner_keeps_a_decision_committed_before_the_limit_was_observed(tmp_path, termination):
    """A stop committed to the ledger stands when the deadline or turn limit is observed afterwards."""
    from unittest.mock import Mock

    from trialagentbench.io import sha256_file
    from trialagentbench.ports import CodeExecutionLimitsV1, RuntimeDeadlineExceeded
    from trialagentbench.trialdev.actions import TrialDevStop
    from trialagentbench.trialdev.run import run_trialdev_programme
    from trialagentbench.trialdev.task_agent import AgentExecutionTerminated, AgentTurnLimitExceeded, TrialDevTaskAgent
    from trialagentbench.trialdev.value import TrialDevSponsorPreferences

    workspace = tmp_path / "run" / "workspace"
    workspace.mkdir(parents=True)
    budget = tmp_path / "budget.json"
    pool = _write_pool(tmp_path / "pool")
    write_json_model(
        budget, TrialDevBudgetLedger(budget_usd=1000, pool_manifest_sha256=sha256_file(pool / "manifest.json"))
    )
    loop = Mock(spec=TrialDevTaskAgent)
    loop.reporting_context = None
    loop.workdir = workspace
    loop.active_prompt_index = None
    session = TrialDevActionSession(tmp_path / "pool", budget, workspace / "evidence", tmp_path / "commits", 7)
    loop.authored_programme = session
    from trialagentbench_verifier.executor import CodeExecutionSessionTerminated

    limit = {
        "deadline": RuntimeDeadlineExceeded("TrialDev task exceeded its declared wall-clock budget."),
        "turn_limit": AgentTurnLimitExceeded("Turn budget exhausted"),
        "executor_exit": AgentExecutionTerminated("Participant executor exited"),
        "session_snapshot": CodeExecutionSessionTerminated("Cannot snapshot a terminated code-execution session."),
    }[termination]

    def commit_then_observe_limit():
        session.execute(TrialDevStop(action="stop"), action_id="stop-1")
        raise limit

    loop.run_programme.side_effect = commit_then_observe_limit
    result = run_trialdev_programme(
        loop=loop,
        prompt="Decide whether to develop the programme.",
        preferences=TrialDevSponsorPreferences(outcome_unit="DAS28", benefit_unit=1),
        executor_limits=CodeExecutionLimitsV1(),
    )
    assert result.execution_status == "completed"
    assert result.error is None
    assert result.evaluation.programme_complete
    loop.close.assert_called_once()


def test_unfinished_programme_is_valued_exactly_as_a_stop_at_its_last_day(tmp_path):
    """Noncompletion is charged the overhead and costs it incurred, never more than stopping there."""
    from trialagentbench.io import sha256_file
    from trialagentbench.ports import CodeExecutionLimitsV1
    from trialagentbench.trialdev.run import evaluate_trialdev_programme
    from trialagentbench.trialdev.value import TrialDevSponsorPreferences

    pool = _write_pool(tmp_path / "pool")
    budget = tmp_path / "private" / "budget.json"
    write_json_model(
        budget, TrialDevBudgetLedger(budget_usd=1000, pool_manifest_sha256=sha256_file(pool / "manifest.json"))
    )
    session = TrialDevActionSession(pool, budget, tmp_path / "workspace" / "evidence", tmp_path / "commits", 7)
    session.execute(TRIALDEV_ACTION.validate_python({"action": "wait", "programme_day": 100}), action_id="wait")
    preferences = TrialDevSponsorPreferences(outcome_unit="DAS28", benefit_unit=1)

    def value():
        return evaluate_trialdev_programme(
            workspace=tmp_path / "workspace",
            programme=session,
            preferences=preferences,
            executor_limits=CodeExecutionLimitsV1(),
        ).value

    unfinished = read_json_model(TrialDevBudgetLedger, budget)
    abandoned = value()
    assert read_json_model(TrialDevBudgetLedger, budget) == unfinished  # Valuation commits nothing.
    assert (abandoned.decision, abandoned.time_cost_usd, abandoned.net_value_usd) == ("stop", 1_500_000, -1_500_000)
    session.execute(TRIALDEV_ACTION.validate_python({"action": "stop"}), action_id="stop")
    assert value() == abandoned


@pytest.mark.parametrize("termination", ["deadline", "turn_limit", "executor_exit", "session_snapshot"])
def test_programme_runner_preserves_noncompletion_and_closes_session(tmp_path, termination):
    from unittest.mock import Mock

    from trialagentbench.ports import CodeExecutionLimitsV1, ProviderRequestError, RuntimeDeadlineExceeded
    from trialagentbench.ports.llm_provider import RetryTelemetry
    from trialagentbench.trialdev.run import TrialDevProgrammeRun, run_trialdev_programme
    from trialagentbench.trialdev.task_agent import AgentExecutionTerminated, AgentTurnLimitExceeded, TrialDevTaskAgent
    from trialagentbench.trialdev.value import TrialDevSponsorPreferences

    workspace = tmp_path / "run" / "workspace"
    workspace.mkdir(parents=True)
    budget = tmp_path / "budget.json"
    from trialagentbench.io import sha256_file

    pool = _write_pool(tmp_path / "pool")
    ledger = TrialDevBudgetLedger(budget_usd=1000, pool_manifest_sha256=sha256_file(pool / "manifest.json"))
    write_json_model(budget, ledger)
    loop = Mock(spec=TrialDevTaskAgent)
    loop.reporting_context = None
    loop.workdir = workspace
    loop.active_prompt_index = None
    loop.authored_programme = TrialDevActionSession(
        tmp_path / "pool", budget, workspace / "evidence", tmp_path / "commits", 7
    )
    from trialagentbench_verifier.executor import CodeExecutionSessionTerminated

    loop.run_programme.side_effect = {
        "deadline": RuntimeDeadlineExceeded("TrialDev task exceeded its declared wall-clock budget."),
        "turn_limit": AgentTurnLimitExceeded("No terminal decision"),
        "executor_exit": AgentExecutionTerminated("Participant executor exited"),
        "session_snapshot": CodeExecutionSessionTerminated("Cannot snapshot a terminated code-execution session."),
    }[termination]
    result = run_trialdev_programme(
        loop=loop,
        prompt="Decide whether to develop the programme.",
        preferences=TrialDevSponsorPreferences(outcome_unit="DAS28", benefit_unit=1),
        executor_limits=CodeExecutionLimitsV1(),
    )
    assert result.execution_status == "model_noncompletion"
    # Nothing was incurred before day 0, so the unfinished programme is valued as a day-0 stop.
    assert (result.evaluation.value.decision, result.evaluation.value.net_value_usd) == ("stop", 0)
    assert result.evaluation.studies == ()
    assert read_json_model(TrialDevBudgetLedger, budget) == ledger
    assert read_json_model(TrialDevProgrammeRun, workspace.parent / "programme_run.json") == result
    loop.close.assert_called_once()
    loop.begin_step.assert_called_once_with(phase_id="development", step_id="programme")
    assert not (workspace / "documents").exists()  # The runner writes no participant documents.
    context_file = workspace / "documents" / "committee_brief.md"
    context_file.parent.mkdir()
    context_file.write_text("# Committee brief\n")
    from trialagentbench_verifier.execution_contracts import ReplayInputArtifactV1
    from trialagentbench_verifier.numerical import NumericalReportingContext

    from trialagentbench.contracts.trialdev.grading_key import TrialDevEvidenceTargetV1
    from trialagentbench.trialdev.release import TrialDevInitialEvidence

    initial = TrialDevInitialEvidence(
        context=NumericalReportingContext(
            checkpoint_id="initial",
            target_ids=("candidate:clinical_outcome",),
            evidence_artifacts=(
                ReplayInputArtifactV1(
                    relative_path="documents/committee_brief.md",
                    sha256="0" * 64,
                    byte_length=context_file.stat().st_size,
                ),
            ),
        ),
        targets=(
            TrialDevEvidenceTargetV1(
                asset_id="candidate",
                endpoint="clinical_outcome",
                unit="DAS28",
                decision_scale=1,
                comparison_sample_size=40,
                point_identified=True,
                estimand="standardized_observed_association",
                oracle_value=0,
            ),
        ),
    )
    loop.run_programme.reset_mock()
    with pytest.raises(ValueError, match="committed snapshot"):
        run_trialdev_programme(
            loop=loop,
            prompt="Develop",
            preferences=TrialDevSponsorPreferences(outcome_unit="DAS28", benefit_unit=1),
            executor_limits=CodeExecutionLimitsV1(),
            initial_evidence=initial,
        )
    loop.run_programme.assert_not_called()
    context_before = context_file.read_bytes() + b"\n"
    context_file.write_bytes(context_before)
    # Provider faults remain retryable exceptions, not scientific outcomes.
    before = (workspace.parent / "programme_run.json").read_bytes()
    loop.run_programme.side_effect = ProviderRequestError(
        "provider unavailable", retry_telemetry=RetryTelemetry(1, 1, 0.0)
    )
    initial = initial.model_copy(
        update={
            "context": initial.context.model_copy(
                update={
                    "evidence_artifacts": (
                        ReplayInputArtifactV1(
                            relative_path="documents/committee_brief.md",
                            sha256=sha256_file(context_file),
                            byte_length=context_file.stat().st_size,
                        ),
                    ),
                }
            )
        }
    )
    loop.reporting_context = None
    with pytest.raises(ProviderRequestError):
        run_trialdev_programme(
            loop=loop,
            prompt="Continue.",
            preferences=TrialDevSponsorPreferences(outcome_unit="DAS28", benefit_unit=1),
            executor_limits=CodeExecutionLimitsV1(),
            initial_evidence=initial,
        )
    prompt = loop.append_user_message.call_args.args[0]
    assert (
        "| `candidate:clinical_outcome` | candidate minus control, case-mix-standardized mean difference (natural units) | DAS28 |"
        in prompt
    )
    assert "in this order (interface/targets.csv), each with a 95% confidence interval" in prompt
    assert "oracle_value" not in prompt and "comparison_sample_size" not in prompt
    assert loop.reporting_context == initial.context
    assert (workspace.parent / "programme_run.json").read_bytes() == before
    assert loop.close.call_count == 3
    assert context_file.read_bytes() == context_before

    for changed in (TrialDevSponsorPreferences(outcome_unit="different units", benefit_unit=1),):
        loop.run_programme.reset_mock()
        with pytest.raises(ValueError, match="units differ"):
            run_trialdev_programme(
                loop=loop,
                prompt="Continue.",
                preferences=changed,
                executor_limits=CodeExecutionLimitsV1(),
            )
        loop.run_programme.assert_not_called()
        assert context_file.read_bytes() == context_before
        assert (workspace.parent / "programme_run.json").read_bytes() == before

    # A pool replacement is an evaluator fault and cannot consume a provider turn.
    loop.run_programme.reset_mock()
    (pool / "manifest.json").write_text('{"replacement": true}')
    with pytest.raises(ValueError, match="bound to its released frozen pool"):
        run_trialdev_programme(
            loop=loop,
            prompt="Continue.",
            preferences=TrialDevSponsorPreferences(outcome_unit="DAS28", benefit_unit=1),
            executor_limits=CodeExecutionLimitsV1(),
        )
    loop.run_programme.assert_not_called()
    assert loop.close.call_count == 5
    assert (workspace.parent / "programme_run.json").read_bytes() == before


def test_authored_resume_restores_pending_call_and_original_turn(tmp_path, monkeypatch):
    """A restart continues the received tool batch without querying the provider again."""
    import json
    from types import SimpleNamespace
    from unittest.mock import Mock

    from trialagentbench.ports import LLMResponse, ToolCall
    from trialagentbench.trialdev import task_agent as owner

    budget = tmp_path / "private" / "budget.json"
    write_json_model(budget, TrialDevBudgetLedger(budget_usd=1000))
    monkeypatch.setattr(owner, "DockerCodeSession", lambda **kwargs: Mock())
    arguments = dict(
        provider=Mock(),
        workdir=tmp_path / "workspace",
        system_prompt="Develop assets.",
        program_id="programme",
        scenario_id="case",
        objective_id="development",
        max_turns_per_step=2,  # the final turn offers only the terminal decision tools
        authored_programme=TrialDevActionSession(
            tmp_path / "pool", budget, tmp_path / "workspace" / "evidence", tmp_path / "commits", 7
        ),
    )
    agent = owner.TrialDevTaskAgent(**arguments)
    calls = [
        ToolCall(id="wait", name="wait_until", arguments=json.dumps({"programme_day": 100})),
        ToolCall(id="stop", name="stop_development", arguments="{}"),
    ]
    provider = Mock(
        return_value=SimpleNamespace(
            response=LLMResponse(tool_calls=calls), message={"role": "assistant", "content": "Wait then stop."}
        )
    )
    monkeypatch.setattr(agent, "_provider_turn", provider)
    original = agent._dispatch_local_tool

    def interrupted(call):
        if call.id == "stop":
            raise OSError("injected pending-action interruption")
        return original(call)

    monkeypatch.setattr(agent, "_dispatch_local_tool", interrupted)
    agent.begin_step(phase_id="development", step_id="programme")
    agent.append_user_message("Decide.")
    with pytest.raises(OSError, match="pending-action"):
        agent.run_programme()
    agent.close()
    assert read_json_model(TrialDevBudgetLedger, budget).current_day == 100
    restored = owner.TrialDevTaskAgent(**arguments, resume=True)
    assert restored.messages == agent.messages
    assert restored.turns_used == 1 and restored.pending_tool_index == 1
    assert [call.id for call in restored.pending_calls] == ["stop"]
    monkeypatch.setattr(restored, "_provider_turn", Mock(side_effect=AssertionError("must not request another turn")))
    restored.run_programme()
    restored.close()
    ledger = read_json_model(TrialDevBudgetLedger, budget)
    assert ledger.terminal.decision == "stop" and ledger.current_day == 100
    assert restored.turns_used == 1 and not restored.pending_calls
    assert provider.call_count == 1
    with pytest.raises(ValueError, match="execution limits"):
        owner.TrialDevTaskAgent(**(arguments | {"max_turns_per_step": 3}), resume=True)
    with pytest.raises(ValueError, match="requested programme"):
        owner.TrialDevTaskAgent(**(arguments | {"program_id": "different-programme"}), resume=True)
    other_seed = TrialDevActionSession(
        tmp_path / "pool", budget, tmp_path / "workspace" / "evidence", tmp_path / "commits", 8
    )
    with pytest.raises(ValueError, match="requested programme"):
        owner.TrialDevTaskAgent(**(arguments | {"authored_programme": other_seed}), resume=True)


def test_authored_process_exit_preserves_pending_action_and_event_sequence(tmp_path):
    """Exit without finally/close and recover from durable state in a fresh process."""
    import json
    import subprocess
    import sys

    script = tmp_path / "process_resume.py"
    script.write_text("""
import json, os, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from trialagentbench.io import read_json_model, write_json_model
from trialagentbench.ports import ToolCall, LLMResponse
from trialagentbench.trialdev import task_agent as owner
from trialagentbench.trialdev.actions import TrialDevActionSession
from trialagentbench.trialdev.resources import TrialDevBudgetLedger
root=Path(sys.argv[1]); resume=sys.argv[2]=='resume'
budget=root/'private'/'budget.json'
if not resume: write_json_model(budget,TrialDevBudgetLedger(budget_usd=1000))
owner.DockerCodeSession=lambda **kwargs: Mock()
agent=owner.TrialDevTaskAgent(provider=Mock(),workdir=root/'workspace',system_prompt='Develop.',
    program_id='programme',scenario_id='case',objective_id='development',max_turns_per_step=2,
    event_log_path=root/'events.jsonl',conversation_log_path=root/'conversation.json',resume=resume,
    authored_programme=TrialDevActionSession(root/'pool',budget,root/'workspace'/'evidence',root/'commits',7))
if not resume:
    calls=[ToolCall(id='wait',name='wait_until',arguments=json.dumps({'programme_day':100})),
        ToolCall(id='stop',name='stop_development',arguments='{}')]
    agent._provider_turn=lambda **kwargs: SimpleNamespace(response=LLMResponse(tool_calls=calls),
        message={'role':'assistant','content':'Wait and stop.'})
    original=agent._dispatch_local_tool
    def dispatch(call):
        result=original(call)
        if call.id=='wait': os._exit(73)
        return result
    agent._dispatch_local_tool=dispatch
    agent.begin_step(phase_id='development',step_id='programme')
    agent.append_user_message('Decide.')
else:
    def forbidden(**kwargs): raise AssertionError('Unexpected provider request')
    agent._provider_turn=forbidden
agent.run_programme()
agent.close()
assert agent.turns_used==1 and not agent.pending_calls
ledger=read_json_model(TrialDevBudgetLedger,budget)
assert ledger.current_day==100 and ledger.terminal.decision=='stop'
""")
    first = subprocess.run(
        [sys.executable, str(script), str(tmp_path), "start"], capture_output=True, text=True, timeout=30
    )
    assert first.returncode == 73, first.stderr
    second = subprocess.run(
        [sys.executable, str(script), str(tmp_path), "resume"], capture_output=True, text=True, timeout=30
    )
    assert second.returncode == 0, second.stderr
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["event_index"] for event in events] == list(range(len(events)))
    assert len({event["event_id"] for event in events}) == len(events)


@pytest.mark.executor
def test_authored_code_files_are_durable_before_next_provider_turn(tmp_path, monkeypatch):
    """A real container's acknowledged code output is available to host recovery."""
    import json
    from types import SimpleNamespace
    from unittest.mock import Mock

    from trialagentbench.ports import LLMResponse, ToolCall
    from trialagentbench.trialdev.task_agent import TrialDevAgentCheckpoint, TrialDevTaskAgent

    budget = tmp_path / "private" / "budget.json"
    write_json_model(budget, TrialDevBudgetLedger(budget_usd=1000))
    agent = TrialDevTaskAgent(
        provider=Mock(),
        workdir=tmp_path / "workspace",
        system_prompt="Develop assets.",
        program_id="programme",
        scenario_id="case",
        objective_id="development",
        max_turns_per_step=2,
        authored_programme=TrialDevActionSession(
            tmp_path / "pool", budget, tmp_path / "workspace" / "evidence", tmp_path / "commits", 7
        ),
    )

    def provider_turn(**kwargs):
        if kwargs["turn"] == 1:
            call = ToolCall(
                id="code",
                name="execute_code",
                arguments=json.dumps(
                    {
                        "language": "python",
                        "code": "from pathlib import Path; Path('scratch/saved.txt').write_text('analysis state')",
                    }
                ),
            )
        else:
            assert (agent.workdir / "scratch" / "saved.txt").read_text() == "analysis state"
            checkpoint = read_json_model(TrialDevAgentCheckpoint, tmp_path / "private" / "agent_state.json")
            assert not checkpoint.pending_calls and checkpoint.turns_used == 1
            assert any(
                message.get("role") == "tool" and message.get("tool_call_id") == "code"
                for message in checkpoint.messages
            )
            call = ToolCall(id="stop", name="stop_development", arguments="{}")
        return SimpleNamespace(
            response=LLMResponse(tool_calls=[call]), message={"role": "assistant", "content": "Proceed."}
        )

    monkeypatch.setattr(agent, "_provider_turn", provider_turn)
    try:
        agent.begin_step(phase_id="development", step_id="programme")
        agent.append_user_message("Analyse then decide.")
        agent.run_programme()
    finally:
        agent.close()
    assert read_json_model(TrialDevBudgetLedger, budget).terminal.decision == "stop"


@pytest.mark.executor
@pytest.mark.parametrize("interruption", ["before_result_commit", "after_result_commit"])
def test_authored_code_recovery_does_not_apply_scratch_increment_twice(tmp_path, monkeypatch, interruption):
    """Recover unacknowledged code on either side of its durable result boundary."""
    import json
    from types import SimpleNamespace
    from unittest.mock import Mock

    from trialagentbench.ports import CodeExecutionResultV1, ToolCall
    from trialagentbench.trialdev import task_agent as owner

    budget = tmp_path / "private" / "budget.json"
    write_json_model(budget, TrialDevBudgetLedger(budget_usd=1000))
    arguments = dict(
        provider=Mock(),
        workdir=tmp_path / "workspace",
        system_prompt="Develop.",
        program_id="programme",
        scenario_id="case",
        objective_id="development",
        max_turns_per_step=2,
        authored_programme=TrialDevActionSession(
            tmp_path / "pool", budget, tmp_path / "workspace" / "evidence", tmp_path / "commits", 7
        ),
    )
    calls = []

    def provider_turn(**kwargs):
        calls.append(kwargs["turn"])
        call = (
            ToolCall(
                id="code",
                name="execute_code",
                arguments=json.dumps(
                    {
                        "language": "python",
                        "code": "from pathlib import Path; p=Path('scratch/count.txt'); p.write_text(str((int(p.read_text()) if p.exists() else 0)+1))",
                    }
                ),
            )
            if kwargs["turn"] == 1
            else ToolCall(id="stop", name="stop_development", arguments="{}")
        )
        return SimpleNamespace(
            response=LLMResponse(tool_calls=[call]), message={"role": "assistant", "content": "Proceed."}
        )

    agent = owner.TrialDevTaskAgent(**arguments)
    monkeypatch.setattr(agent, "_provider_turn", provider_turn)
    agent.begin_step(phase_id="development", step_id="programme")
    agent.append_user_message("Analyse then decide.")
    original_write = owner.write_json_model
    original_execute = agent._execute_checkpointed_tool

    def interrupted_write(path, model, **kwargs):
        if isinstance(model, CodeExecutionResultV1):
            raise OSError("injected result commit interruption")
        return original_write(path, model, **kwargs)

    def interrupted_execute(call):
        result = original_execute(call)
        if call.name == "execute_code":
            raise OSError("injected result commit interruption")
        return result

    try:
        with monkeypatch.context() as patch:
            if interruption == "before_result_commit":
                patch.setattr(owner, "write_json_model", interrupted_write)
            else:
                patch.setattr(agent, "_execute_checkpointed_tool", interrupted_execute)
            with pytest.raises(OSError, match="result commit interruption"):
                agent.run_programme()
    finally:
        agent.close()
    assert (agent.workdir / "scratch" / "count.txt").read_text() == "1"
    restored = owner.TrialDevTaskAgent(**arguments, resume=True)
    monkeypatch.setattr(restored, "_provider_turn", provider_turn)
    try:
        restored.run_programme()
    finally:
        restored.close()
    assert (restored.workdir / "scratch" / "count.txt").read_text() == "1"
    assert calls == [1, 2]
    assert read_json_model(TrialDevBudgetLedger, budget).terminal.decision == "stop"


@pytest.mark.parametrize("development", [False])  # programmes keep reports at cutoffs, not via a report tool
def test_numerical_resume_after_report_commit_preserves_first_numbers(tmp_path, monkeypatch, development):
    """An interrupted report acknowledgement neither replaces numbers nor spends a turn."""
    import json
    from types import SimpleNamespace
    from unittest.mock import Mock

    from trialagentbench_verifier.numerical import NumericalReportingContext

    from trialagentbench.ports import LLMResponse, ToolCall
    from trialagentbench.trialdev import task_agent as owner

    budget = tmp_path / "private" / "budget.json"
    write_json_model(budget, TrialDevBudgetLedger(budget_usd=1000))
    monkeypatch.setattr(owner, "DockerCodeSession", lambda **kwargs: Mock())
    arguments = dict(
        provider=Mock(),
        workdir=tmp_path / "workspace",
        system_prompt="Develop.",
        program_id="programme",
        scenario_id="case",
        objective_id="development",
        max_turns_per_step=1,
        reporting_context=NumericalReportingContext(
            checkpoint_id="initial", target_ids=("effect",), evidence_artifacts=()
        ),
        authored_programme=(
            TrialDevActionSession(
                tmp_path / "pool", budget, tmp_path / "workspace" / "evidence", tmp_path / "commits", 7
            )
            if development
            else None
        ),
    )
    agent = owner.TrialDevTaskAgent(**arguments)
    scratch = agent.workdir / "scratch"
    scratch.mkdir()
    original_numbers = "target_id,estimate,lower,upper\neffect,1,0,2\n"
    (scratch / "estimates.csv").write_text(original_numbers)
    calls = [
        ToolCall(id="report", name="submit_analysis_report", arguments=json.dumps({"content": "Estimate."})),
        ToolCall(id="stop", name="stop_development", arguments="{}"),
    ]
    if not development:
        calls = calls[:1]
    provider = Mock(
        return_value=SimpleNamespace(
            response=LLMResponse(tool_calls=calls), message={"role": "assistant", "content": "Report and stop."}
        )
    )
    monkeypatch.setattr(agent, "_provider_turn", provider)
    commit = agent._commit_numerical_report

    def interrupted(captured):
        commit(captured)
        raise OSError("injected report acknowledgement interruption")

    monkeypatch.setattr(agent, "_commit_numerical_report", interrupted)
    agent.begin_step(phase_id="development", step_id="programme")
    agent.append_user_message("Decide.")
    with pytest.raises(OSError, match="report acknowledgement"):
        agent.run_programme() if development else agent.run_until_report()
    agent.close()
    (scratch / "estimates.csv").write_text("target_id,estimate,lower,upper\neffect,9,8,10\n")
    restored = owner.TrialDevTaskAgent(**arguments, resume=True)
    monkeypatch.setattr(restored, "_provider_turn", Mock(side_effect=AssertionError("extra provider turn")))
    restored.run_programme() if development else restored.run_until_report()
    restored.close()
    captures = tuple((tmp_path / "analysis_sources").glob("submission-*"))
    assert len(captures) == 1
    assert (captures[0] / "estimates.csv").read_text() == original_numbers
    committed_kind = "cutoffs" if development else "submitted"
    cutoff = next((tmp_path / "analysis_sources" / committed_kind).glob("*/estimates.csv"))
    assert cutoff.read_text() == original_numbers
    assert provider.call_count == 1 and restored.turns_used == 1


@pytest.mark.executor
def test_terminated_executor_records_result_without_snapshot_or_retry(tmp_path, monkeypatch):
    """Participant process exit becomes durable noncompletion, not a snapshot crash."""
    import json
    from types import SimpleNamespace
    from unittest.mock import Mock

    from trialagentbench_verifier.numerical import NumericalReportingContext

    from trialagentbench.ports import LLMResponse, ToolCall
    from trialagentbench.trialdev import task_agent as owner

    agent = owner.TrialDevTaskAgent(
        provider=Mock(),
        workdir=tmp_path / "workspace",
        system_prompt="Analyse.",
        program_id="analysis",
        scenario_id="case",
        objective_id="effect",
        max_turns_per_step=2,
        reporting_context=NumericalReportingContext(
            checkpoint_id="initial", target_ids=("effect",), evidence_artifacts=()
        ),
    )
    call = ToolCall(id="exit", name="execute_code", arguments='{"language":"python","code":"import os; os._exit(17)"}')
    provider = Mock(
        return_value=SimpleNamespace(
            response=LLMResponse(tool_calls=[call]), message={"role": "assistant", "content": "Run code."}
        )
    )
    monkeypatch.setattr(agent, "_provider_turn", provider)
    agent.begin_step(phase_id="evidence_analysis", step_id="initial")
    agent.append_user_message("Analyse the supplied records.")
    try:
        with pytest.raises(owner.AgentExecutionTerminated):
            agent.run_until_report()
        assert provider.call_count == 1
        assert agent.step_terminal and not agent.pending_calls
        assert agent.execution_outputs[-1]["result"]["status"] == "session_terminated"
        transaction = agent._code_transaction(call)
        committed = json.loads((transaction / "after/result.json").read_text())
        assert committed["status"] == "session_terminated"
        # Replaying the retained tool result does not execute participant code again.
        assert agent._execute_checkpointed_tool(call).status == "session_terminated"
    finally:
        agent.close()


@pytest.mark.parametrize("action", ["readout", "advance", "stop"])
def test_malformed_automatic_report_is_visible_and_retryable_before_new_evidence(tmp_path, action):
    """An unknown result target cannot silently cross the next evidence cutoff."""
    import json
    from unittest.mock import Mock

    from trialagentbench_verifier.numerical import NumericalReportingContext

    from trialagentbench.ports import ToolCall
    from trialagentbench.ports.tool_input import ToolInputError
    from trialagentbench.trialdev.task_agent import TrialDevTaskAgent

    agent = object.__new__(TrialDevTaskAgent)
    agent.deadline = RuntimeDeadline.after(30, label="action unit test")
    agent.reporting_context = None
    agent.executor_limits = None
    agent.executor_image = None
    agent.analysis_language = "python_r"
    agent.workdir = tmp_path / "workspace"
    scratch = agent.workdir / "scratch"
    scratch.mkdir(parents=True)
    (scratch / "analysis.py").write_text("# source\n")
    csv = scratch / "estimates.csv"
    csv.write_text("target_id,estimate,lower,upper\nwrong,1,0,2\n")
    agent.session = Mock()
    agent._executor_image_id = "sha256:" + "0" * 64
    agent.step_id = "programme"
    agent.executed_code = []
    agent.execution_outputs = []
    agent.reporting_context = NumericalReportingContext(
        checkpoint_id="initial", target_ids=("effect",), evidence_artifacts=()
    )
    from trialagentbench.trialdev.resources import TrialDevTerminalDecision

    terminal = TrialDevTerminalDecision(decision="stop", programme_day=0)
    programme = _mock_session(tmp_path, TrialDevBudgetLedger(budget_usd=1000, terminal=terminal))
    agent.authored_programme = programme
    request = {"action": action}
    if action == "readout":
        request.update(study_id="study-101", evidence_folder="study-101-look-1")
    elif action == "advance":
        request.update(asset_id="candidate", dose="standard")
    name = {"readout": "request_readout", "advance": "advance_candidate", "stop": "stop_development"}[action]
    programme.decode.return_value = _decoded(request)
    call = ToolCall(id="cutoff", name=name, arguments="{}")
    with pytest.raises(ToolInputError, match="unknown numerical target"):
        agent._dispatch_local_tool(call)
    programme.execute.assert_not_called()
    assert not tuple((tmp_path / "analysis_sources/pending_cutoffs").glob("*/estimates.csv"))
    # An explicit incomplete stop remains a valid decision.
    if action == "stop":
        programme.decode.return_value = _decoded(request, accept_incomplete=True)
        result = agent._dispatch_local_tool(
            ToolCall(id="stop-incomplete", name=name, arguments=json.dumps({"accept_incomplete": True}))
        )
        assert "Note: the analysis kept at this cutoff was incomplete:" in result.output
        assert "unknown numerical target" in result.output
        programme.execute.assert_called_once()
    else:
        csv.write_text("target_id,estimate,lower,upper\neffect,0.1,-1.2,1.4\n")
        from trialagentbench.io import canonical_payload_sha256

        identity = "action-" + canonical_payload_sha256({"step": agent.step_id, "call": call.id})[:24]
        pending = agent._prepare_cutoff(identity, published=None, reject_invalid_result=True)
        assert pending is not None
        assert (pending / "estimates.csv").read_bytes() == csv.read_bytes()


def test_inline_defaults_preserve_scientific_choices_and_explicit_schedules(tmp_path):
    """Deterministic code names control, numbers studies and fills grid defaults; explicit choices stay."""
    from trialagentbench.trialdev.errors import TrialDevActionRejectedError

    budget = tmp_path / "private" / "budget.json"
    write_json_model(budget, TrialDevBudgetLedger(budget_usd=1000))
    session = TrialDevActionSession(
        _write_pool(tmp_path / "pool"), budget, tmp_path / "evidence", tmp_path / "commits", 7
    )
    choices = {
        "phase": "early_safety",
        "sites": 2,
        "arms": [{"compound": "CANDIDATE", "dose": "standard", "participants": 60}],
        "control_participants": 30,
        "follow_up_days": 56,
        "looks": [{"deadline_day": 180, "completed_participants": 90}],
    }
    resolved = session.decode("start_study", choices).action
    assert resolved.study_id == "study-101"
    assert [(arm.arm_id, arm.asset_id) for arm in resolved.arms] == [
        ("control", "control"),
        ("candidate-standard", "candidate"),
    ]
    assert [arm.participant_count for arm in resolved.arms] == [30, 60]
    assert resolved.assessment_days == resolved.surveillance_days == (28, 56)
    assert resolved.follow_up_days == 56 and resolved.looks[0].completed_participants == 90
    assert session.decode("start_study", choices | {"phase": "confirmation"}).action.study_id == "study-301"
    explicit = session.decode(
        "start_study",
        choices
        | {"assessment_days": [56], "surveillance_days": [28], "arms": [choices["arms"][0] | {"name": "chosen"}]},
    ).action
    assert explicit.assessment_days == (56,) and explicit.surveillance_days == (28,)
    assert explicit.arms[1].arm_id == "chosen"
    for change, message in (
        ({"arms": [choices["arms"][0] | {"compound": "other"}]}, "compounds are candidate"),
        ({"arms": [choices["arms"][0] | {"name": "control"}]}, "reserved for the control arm"),
    ):
        with pytest.raises(TrialDevActionRejectedError, match=message):
            session.decode("start_study", choices | change)
    with pytest.raises(TrialDevActionRejectedError, match="no study has been started"):
        session.decode("request_readout", {"study_id": "study-101"})
    advance = session.decode("advance_candidate", {"compound": "Candidate", "dose": "standard"}).action
    assert (advance.asset_id, advance.dose.value) == ("candidate", "standard")
