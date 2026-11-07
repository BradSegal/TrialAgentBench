"""Model-facing TrialDev timing rules state the programme clock explicitly."""

from __future__ import annotations

from trialagentbench.trialdev.actions import trialdev_programme_tools
from trialagentbench.trialdev.run import trialdev_programme_system_prompt


def test_system_prompt_states_deadline_origin_and_readout_semantics() -> None:
    prompt = trialdev_programme_system_prompt("raw")
    assert "deadline_day is a programme day counted from programme start, not a duration" in prompt
    assert "follow-up runs from each participant's enrolment" in prompt
    assert "uses its planned look even when few or no participants" in prompt
    assert "its files never change" in prompt


def test_look_schema_describes_deadline_origin_to_the_model() -> None:
    start = next(tool for tool in trialdev_programme_tools() if tool["function"]["name"] == "start_study")
    text = repr(start["function"]["parameters"])
    assert "counted from programme start (day 0), not a duration after the study starts" in text
    assert "Optional earlier trigger" in text
