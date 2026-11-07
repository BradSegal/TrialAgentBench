"""Participant-only deterministic installation control, with no model or network.

Zero estimates deliberately have no scientific interpretation. This adapter
exercises the same tools, persistent executors, result capture and study service
as a model while making no claim about model performance.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue

from trialagentbench.ports.llm_provider import JsonObject, LLMResponse, LLMResponseMetadata, ToolCall

# The control's early-safety study (30 per arm, 56-day follow-up, read out by day 365 or at 60
# completed) activates sites in the public activation order until they list enough candidates
# and forecast enough screening to fill it before the look (at least two sites). Under
# registry-rate accrual two sites can open late and screen a few candidates a year, leaving no
# outcome record by day 365, and hold fewer eligible candidates than the doubled adverse census
# study (120 participants) needs.
STUDY_SITE_CANDIDATES = 480
STUDY_FORECAST_SCREENED = 60
STUDY_LAST_ENROLMENT_DAY = 365 - 56
_CAPACITY_REJECTION = re.compile(r"Tool input rejected: only (\d+) unenrolled candidates remain")


class _Function(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str


class _Tool(BaseModel):
    model_config = ConfigDict(extra="ignore")
    function: _Function


class _Message(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str
    content: str | None = None
    tool_calls: tuple[_Tool, ...] = ()


class OfflineControlProvider:
    """Save a zero-result script and, for programmes, acquire one public study."""

    def __init__(self, model: str) -> None:
        if model not in {"smoke-zero-python", "smoke-zero-r"}:
            raise ValueError("Unknown offline installation control")
        self._model = model

    @property
    def model(self) -> str:
        """Identify the control explicitly in every retained run."""
        return self._model

    @property
    def telemetry_route(self) -> str:
        """Distinguish local control steps from provider requests."""
        return "offline_control"

    def generate_turn(
        self,
        messages: Sequence[JsonObject],
        tools: Sequence[JsonObject] | None = None,
        *,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float | None = None,
        tool_choice: Literal["auto", "required"] = "auto",
    ) -> LLMResponse:
        """Choose solely from participant-visible history and offered tools."""
        offered = {_Tool.model_validate(tool).function.name for tool in tools or ()}
        history = [_Message.model_validate(message) for message in messages]
        calls = [call for message in history for call in message.tool_calls]
        outputs = [message.content for message in history if message.role == "tool" and message.content]
        programme = "start_study" in offered
        arguments: dict[str, JsonValue]
        if not calls:
            language = "r" if self.model.endswith("-r") else "python"
            script = (
                "source('interface/reporting.R')\n"
                "report(data.frame(target_id=targets(),estimate=0,lower=-1,upper=1))\n"
                if language == "r"
                else "from interface.reporting import targets, report\n"
                "report([dict(target_id=t,estimate=0,lower=-1,upper=1) for t in targets()])\n"
            )
            filename = "analysis.R" if language == "r" else "analysis.py"
            code = (
                "from pathlib import Path\nimport csv, json\n"
                "assert not Path('evaluation').exists()\n"
                f"Path('scratch/{filename}').write_text({script!r})\n"
            )
            if programme:
                code += (
                    "c=next(csv.DictReader(open('interface/targets.csv')))['target_id'].split(':')[0]\n"
                    "s=sorted(csv.DictReader(open('documents/site_network.csv')),key=lambda r:int(r['activation_order']))\n"
                    "f=lambda r:float(r['screening_candidates_per_month'])"
                    f"*max(0.0,{STUDY_LAST_ENROLMENT_DAY}-float(r['opening_day']))/30.4375\n"
                    "n=next((k for k in range(2,len(s)+1) "
                    f"if sum(int(r['candidates']) for r in s[:k])>={STUDY_SITE_CANDIDATES} "
                    f"and sum(f(r) for r in s[:k])>={STUDY_FORECAST_SCREENED}),len(s))\n"
                    "print(json.dumps(dict(phase='early_safety',sites=n,control_participants=30,"
                    "arms=[dict(compound=c,dose='standard',participants=30)],"
                    "follow_up_days=56,looks=[dict(deadline_day=365,completed_participants=60)])))\n"
                )
            else:
                code += "print('Saved installation-control analysis for native result capture.')\n"
            if language == "r":
                code = (
                    "stopifnot(!dir.exists('evaluation'))\n"
                    f"writeLines({json.dumps(script)}, 'scratch/{filename}', useBytes=TRUE)\n"
                )
                if programme:
                    code += (
                        "c <- strsplit(read.csv('interface/targets.csv')$target_id[1], ':', fixed=TRUE)[[1]][1]\n"
                        "s <- read.csv('documents/site_network.csv')\n"
                        "s <- s[order(s$activation_order), ]\n"
                        "f <- s$screening_candidates_per_month * "
                        f"pmax(0, {STUDY_LAST_ENROLMENT_DAY} - s$opening_day) / 30.4375\n"
                        "eligible <- which(seq_len(nrow(s)) >= 2 & "
                        f"cumsum(s$candidates) >= {STUDY_SITE_CANDIDATES} & "
                        f"cumsum(f) >= {STUDY_FORECAST_SCREENED})\n"
                        "n <- if(length(eligible)) eligible[1] else nrow(s)\n"
                        "cat(jsonlite::toJSON(list(phase='early_safety', sites=n, control_participants=30, "
                        "arms=list(list(compound=c, dose='standard', participants=30)), follow_up_days=56, "
                        "looks=list(list(deadline_day=365, completed_participants=60))), "
                        "auto_unbox=TRUE, digits=17), '\\n')\n"
                    )
                else:
                    code += "cat('Saved installation-control analysis for native result capture.\\n')\n"
            name = "execute_code"
            arguments = {"code": code, "language": language, "purpose": "Save an offline zero-result control."}
        elif not programme:
            name = "submit_response" if "submit_response" in offered else "submit_analysis_report"
            arguments = {
                "content": "Offline zero-result installation control; these estimates are not clinical or model-performance evidence."
            }
        else:
            from trialagentbench.trialdev.actions import TrialDevStudyProposal

            prior = [call.function.name for call in calls[1:]]
            if not outputs:
                raise ValueError("Offline study control received no tool output")
            capacity_rejections = [match for output in outputs if (match := _CAPACITY_REJECTION.match(output))]
            if prior.count("start_study") < 2 + len(capacity_rejections):
                if outputs[-1].startswith("Tool input rejected:") and not _CAPACITY_REJECTION.match(outputs[-1]):
                    raise ValueError(f"Offline study control preview failed: {outputs[-1]}")
                design = TrialDevStudyProposal.model_validate_json(outputs[0].splitlines()[0])
                if capacity_rejections:
                    per_arm = min(30, *(int(match[1]) // 2 for match in capacity_rejections))
                    if per_arm < 10:
                        raise ValueError("Offline study control requires at least 20 eligible candidates")
                    design = design.model_copy(
                        update={
                            "control_participants": per_arm,
                            "arms": tuple(arm.model_copy(update={"participants": per_arm}) for arm in design.arms),
                            "looks": tuple(
                                look.model_copy(update={"completed_participants": 2 * per_arm})
                                for look in design.looks
                            ),
                        }
                    )
                name = "start_study"
                arguments = design.model_dump(mode="json", exclude_none=True) | {
                    "preview": prior.count("start_study") == len(capacity_rejections)
                }
            elif "request_readout" not in prior:
                name, arguments = "request_readout", {"study_id": "study-101"}
            else:
                if "evidence/study-101-look-1/" not in outputs[-1] or "| `" not in outputs[-1]:
                    raise ValueError("Offline study control did not acquire reportable records")
                name, arguments = "stop_development", {}
        if name not in offered:
            raise ValueError(f"Offline control tool is unavailable: {name}")
        return LLMResponse(
            tool_calls=[ToolCall(id=f"control-{len(calls)}", name=name, arguments=json.dumps(arguments))],
            metadata=LLMResponseMetadata(
                response_id=f"offline-{len(calls)}",
                returned_model=self.model,
                finish_reason="tool_calls",
                reported_cost_usd=0.0,
                request_attempts=1,
            ),
        )
