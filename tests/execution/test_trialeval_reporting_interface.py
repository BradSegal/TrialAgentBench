"""The TrialEval reporting interface is a declared condition: v1 keeps 1.0 behaviour, clarified adds repair turns."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from trialagentbench_test_helpers import (
    minimal_participant_output_contract,
    write_minimal_trialeval_release_dictionaries,
)
from trialagentbench_verifier.campaign_contracts import CampaignPanelV1
from trialagentbench_verifier.run_contracts import ExperimentConditionV1

from trialagentbench.ports import CodeExecutionResultV1, LLMResponse, ToolCall
from trialagentbench.trialeval import agent
from trialagentbench.trialeval.schema import BenchmarkItem


def test_absent_interface_means_v1_and_v1_is_not_serialised() -> None:
    """Campaigns and conditions recorded before v1.1 keep their meaning and their serialised bytes."""

    assert CampaignPanelV1().reporting_interface == "v1"
    assert "reporting_interface" not in CampaignPanelV1().model_dump(mode="json")
    clarified = CampaignPanelV1(reporting_interface="clarified")
    assert clarified.model_dump(mode="json")["reporting_interface"] == "clarified"
    condition = ExperimentConditionV1(condition_id="c", request_replicate_id="r", maximum_turns_per_step=50)
    assert condition.reporting_interface == "v1"
    assert "reporting_interface" not in condition.model_dump(mode="json")
    assert condition.model_copy(update={"reporting_interface": "clarified"}).model_dump(mode="json")[
        "reporting_interface"
    ] == ("clarified")


class _Session:
    image_id = "sha256:" + "0" * 64

    def execute(self, code: str) -> str:
        return ""

    def close(self) -> None:
        return None

    def snapshot_scratch(self) -> Path:
        return Path(".")


def _item(tmp_path: Path) -> BenchmarkItem:
    visible = tmp_path / "visible"
    data = visible / "data"
    data.mkdir(parents=True)
    write_minimal_trialeval_release_dictionaries(tmp_path)
    return BenchmarkItem(
        item_id="TASK1",
        task_id="TASK1",
        trial_name="fixture",
        design_tier="undisclosed",
        design_subtype="individual_randomized",
        assumption_tier="undisclosed",
        context_tier="C4",
        data_preparation="raw_domains",
        analysis_specification="protocol_only",
        visible_dir=visible,
        data_dir=data,
        task={"target_ids": ["effect"]},
        submission_contract=minimal_participant_output_contract("TASK1", data_preparation="raw_domains"),
        suite_dir=tmp_path,
    )


class _SubmitCheckSubmit:
    """Submit on the last analysis turn, then check and submit again, recording what each turn offered."""

    model = "test"
    telemetry_route = "test"

    def __init__(self) -> None:
        self.tools: list[list[str]] = []
        self.tool_outputs: list[str] = []

    def generate_turn(self, **kwargs: object) -> LLMResponse:
        tools = kwargs["tools"]
        assert isinstance(tools, list)
        self.tools.append([tool["function"]["name"] for tool in tools])
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        self.tool_outputs = [str(message["content"]) for message in messages if message.get("role") == "tool"]
        submit = ToolCall(id=f"submit-{len(self.tools)}", name="submit_response", arguments='{"content":"Report."}')
        if len(self.tools) == 2 and "execute_code" in self.tools[-1]:
            return LLMResponse(tool_calls=[ToolCall(id="check", name="check_analysis", arguments="{}")])
        return LLMResponse(tool_calls=[submit])


@pytest.mark.parametrize("interface", ["v1", "clarified"])
def test_failed_submission_replay_reopens_analysis_only_when_clarified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interface: str
) -> None:
    monkeypatch.setattr(agent, "DockerCodeSession", lambda **_: _Session())
    checks: list[str] = []

    def check(call: ToolCall, **kwargs: object) -> tuple[CodeExecutionResultV1, object]:
        assert kwargs["reporting_interface"] == interface
        checks.append(call.id)
        failed = len(checks) == 1
        result = CodeExecutionResultV1(
            status="execution_error" if failed else "success",
            output="scratch/analysis.py failed in a fresh session" if failed else "reproduced",
            elapsed_seconds=0.0,
        )
        return result, kwargs["session"]

    monkeypatch.setattr(agent, "check_analysis_tool_v1", check)
    provider = _SubmitCheckSubmit()
    output = agent.run_agent(
        _item(tmp_path),
        cast(Any, provider),
        max_turns=3,
        analysis_specification="protocol_only",
        verbose=False,
        item_workspace=tmp_path / "workspace",
        reporting_interface=interface,  # type: ignore[arg-type]
    )

    assert output["status"] == "success"
    first_reply = provider.tool_outputs[0]
    if interface == "v1":
        # 1.0 behaviour: no analysis turn remains, so the next turn offers only the submission tools.
        assert "No analysis turn remains" in first_reply
        assert provider.tools[1] == ["submit_response", "submit_response_file"]
        assert output["turns_used"] == 2
    else:
        assert "The analysis tools reopen for 2 repair turns" in first_reply
        assert "execute_code" in provider.tools[1] and "check_analysis" in provider.tools[1]
        assert "repair: analysis tools close after turn 3" in first_reply
        assert output["turns_used"] == 3
        assert checks == ["submit-1", "check", "submit-3"]
        assert cast(dict[str, object], output["condition_provenance"])["reporting_interface"] == "clarified"
