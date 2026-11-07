"""Scripted reference agent that uses only the participant-facing tools and documents.

The agent implements the ordinary ``LLMProvider`` protocol. It reads participant messages
and tool results, writes its analysis into scratch/ with the workspace tool, runs it, checks
replay, and submits (TrialEval) or designs, acquires and decides (TrialDev). It never opens
evaluator material, release internals or backend APIs. Every tool result is classified and
retained so that each friction point (rejected inputs, errors, retries, identifiers it had
to carry across tools, environment knowledge its analysis had to hard-code) is recorded.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from trialagentbench.ports.llm_provider import JsonObject, LLMResponse, LLMResponseMetadata, ToolCall

HERE = Path(__file__).resolve().parent
_ERROR = re.compile(r"Tool input rejected|Traceback \(most recent call last\)|^\w*Error\b", re.MULTILINE)


@dataclass
class FrictionLog:
    """Ordered tool interactions and friction points observed by one episode."""

    calls: list[dict[str, object]] = field(default_factory=list)
    friction: list[str] = field(default_factory=list)

    def note(self, text: str) -> None:
        """Record one friction point once (identifiers normalised), in order of first occurrence."""
        text = _IDENTIFIER.sub("<id>", text)
        if text not in self.friction:
            self.friction.append(text)


_IDENTIFIER = re.compile(r"action-[0-9a-f]{24}")
_PROGRAMME_TOOLS = {
    "start_study",
    "request_readout",
    "adapt_study",
    "wait_until",
    "stop_development",
    "advance_candidate",
}


def _is_error(tool: str, text: str) -> bool:
    """Classify a tool result as an error without matching field names such as *_error."""
    head = text.split("\n[turn")[0]
    if tool == "check_analysis" and head.startswith("{"):
        return json.loads(head.splitlines()[0]).get("status") != "reproduced"
    if tool in _PROGRAMME_TOOLS:
        return head.startswith("Tool input rejected") or "Note: the analysis kept" in head
    return bool(_ERROR.search(head[:2000]))


def _messages(messages: Sequence[JsonObject]) -> list[dict[str, object]]:
    return [dict(message) for message in messages]


def _tool_results(history: list[dict[str, object]]) -> dict[str, str]:
    results: dict[str, str] = {}
    for message in history:
        if message.get("role") == "tool":
            content = message.get("content")
            if isinstance(content, str):
                results[str(message.get("tool_call_id"))] = content
    return results


def _calls(history: list[dict[str, object]]) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    for message in history:
        for call in cast(list[dict[str, object]], message.get("tool_calls") or []):
            calls.append(dict(call))
    return calls


class ReferenceAgent:
    """Deterministic analyst and development policy driving the native tool loop."""

    model = "trialagentbench-reference-agent"
    telemetry_route = "local-scripted"

    def __init__(self) -> None:
        self.log = FrictionLog()
        self._step = 0
        self._programme: dict[str, Any] | None = None
        self._estimates: dict[str, dict[str, Any]] = {}
        self._stage = "observational"
        self._pending: dict[str, Any] | None = None
        self._candidates: list[tuple[str, str]] = []
        self._study_id: str | None = None
        self._rejections = 0
        self._last_action = ""

    # ------------------------------------------------------------------ protocol
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
        """Return the next ordinary tool call from public conversation state."""
        history = _messages(messages)
        offered = {str(cast(dict[str, Any], tool)["function"]["name"]) for tool in tools or ()}
        self._observe(history)
        if "start_study" in offered:
            name, arguments = self._trialdev(history, offered)
        elif "submit_response" in offered:
            name, arguments = self._trialeval(history, offered)
        else:
            name, arguments = self._trialdev_report(offered)
        if name not in offered:
            raise ValueError(f"Reference agent requested an unavailable tool: {name}")
        if name in _PROGRAMME_TOOLS:
            self._last_action = name + (":preview" if arguments.get("preview") else "")
        self._step += 1
        self.log.calls.append({"tool": name, "arguments": _summary(arguments)})
        return LLMResponse(
            tool_calls=[ToolCall(id=f"ref-{self._step}", name=name, arguments=json.dumps(arguments))],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            metadata=LLMResponseMetadata(
                response_id=f"ref-{self._step}", returned_model=self.model, finish_reason="tool_calls"
            ),
        )

    def _observe(self, history: list[dict[str, object]]) -> None:
        """Attach the latest tool result to its call and harvest friction lines."""
        results = _tool_results(history)
        for call in _calls(history):
            record = next((c for c in self.log.calls if c.get("id") == call["id"]), None)
            if record is None:
                index = int(str(call["id"]).removeprefix("ref-")) - 1
                if 0 <= index < len(self.log.calls) and "id" not in self.log.calls[index]:
                    record = self.log.calls[index]
                    record["id"] = call["id"]
            if record is not None and "result" not in record and call["id"] in results:
                text = results[str(call["id"])]
                record["result"] = text[:400]
                record["result_chars"] = len(text)
                record["error"] = _is_error(str(record["tool"]), text)
                for line in text.splitlines():
                    if line.startswith("FRICTION: "):
                        self.log.note("analysis: " + line.removeprefix("FRICTION: "))
                if record["error"]:
                    self.log.note(f"{record['tool']} error: {_IDENTIFIER.sub('<id>', text[:240])}")

    def _last_result(self, history: list[dict[str, object]]) -> str:
        for message in reversed(history):
            if message.get("role") == "tool":
                return str(message.get("content") or "")
        return ""

    # ------------------------------------------------------------------ TrialEval
    def _trialeval(self, history: list[dict[str, object]], offered: set[str]) -> tuple[str, dict[str, object]]:
        system = str(history[0].get("content", ""))
        data = re.findall(r"(data/[\w/.-]+\.parquet)", system.split("Data files:", 1)[1])
        plan: list[tuple[str, dict[str, object]]] = [
            ("read_workspace_file", {"path": "interface/targets.csv"}),
            *[("inspect_parquet", {"path": path, "language": "python"}) for path in data[:1]],
            ("write_workspace_file", {"path": "scratch/analysis.py", "content": _script("trialeval_analysis.py")}),
            ("execute_code", _run_script()),
            ("check_analysis", {"language": "python"}),
        ]
        done = len(_calls(history))
        if done < len(plan):
            return plan[done]
        output = next((c.get("result", "") for c in reversed(self.log.calls) if c["tool"] == "execute_code"), "")
        check = self._last_result(history)
        accepted = not _is_error("check_analysis", check)
        if not accepted:
            self.log.note(f"check_analysis did not confirm replay: {check[:240]}")
        estimates = "\n".join(line for line in str(output).splitlines() if ": " in line and "[" in line)
        report = (
            "# Statistical report\n\nReference analysis (scripted): conventional estimators applied to the "
            "documented analysis population after the source-record rules in the data-integrity policy.\n\n"
            f"```\n{estimates}\n```\n\nKaplan-Meier risks and restricted means, Wald proportions, Welch means, "
            "Poisson rate ratios with robust errors and cluster-robust linear models; 95% intervals.\n"
        )
        return "submit_response", {"content": report, "accept_incomplete": not accepted}

    # ------------------------------------------------------------------ TrialDev
    def _trialdev_report(self, offered: set[str]) -> tuple[str, dict[str, object]]:
        """Evidence-analysis stream: analyse, check and submit the report."""
        plan: list[tuple[str, dict[str, object]]] = [
            ("write_workspace_file", {"path": "scratch/analysis.py", "content": _script("trialdev_analysis.py")}),
            ("execute_code", _run_script()),
            ("check_analysis", {"language": "python"}),
        ]
        if self._step < len(plan):
            return plan[self._step]
        return "submit_analysis_report", {"content": "Reference observational analysis (regression standardisation)."}

    def _trialdev(self, history: list[dict[str, object]], offered: set[str]) -> tuple[str, dict[str, object]]:
        last = self._last_result(history)
        if self._step == 0:
            brief = re.search(r"`(documents/[^`]+/committee_brief\.md)`", str(history[1].get("content", "")))
            if brief is None:
                raise ValueError("The task message names no committee brief")
            return "execute_code", {
                "code": _PROGRAMME_SUMMARY.replace("BRIEF_PATH", brief.group(1)),
                "language": "python",
                "purpose": "Read the committee brief's design rules.",
            }
        if self._programme is None:
            line = next((x for x in last.splitlines() if x.startswith("PROGRAMME=")), None)
            if line is None:
                raise ValueError(f"Programme summary unavailable: {last[:300]}")
            self._programme = json.loads(line.removeprefix("PROGRAMME="))
            return "write_workspace_file", {"path": "scratch/analysis.py", "content": _script("trialdev_analysis.py")}
        previous = str(self.log.calls[-1]["tool"]) if self.log.calls else ""
        if self._stage in {"observational", "readout_analysis"} and previous == "write_workspace_file":
            return "execute_code", _run_script()
        if previous == "execute_code" and "ESTIMATES=" in last:
            line = next(x for x in last.splitlines() if x.startswith("ESTIMATES="))
            self._estimates = {row["target_id"]: row for row in json.loads(line.removeprefix("ESTIMATES="))}
            return self._decide()
        if previous == "execute_code":
            self.log.note(f"analysis failed before a decision: {last[:240]}")
            return "stop_development", {"accept_incomplete": True}
        if previous in _PROGRAMME_TOOLS:
            return self._after_action(last)
        raise AssertionError(f"Unhandled reference state after {previous}")

    def _decide(self) -> tuple[str, dict[str, object]]:
        assert self._programme is not None
        phases = {p["phase"]: p for p in self._programme["phases"]}
        benefit = {
            key.rsplit(":", 1)[0]: row for key, row in self._estimates.items() if key.endswith(":clinical_outcome")
        }
        safety = {key.rsplit(":", 1)[0]: row for key, row in self._estimates.items() if key.endswith(":safety_risk")}
        ranked = sorted(benefit, key=lambda k: benefit[k]["estimate"])  # lower is favourable in every programme
        if self._stage == "observational":
            self._stage = "early_safety"
            assets = [k.split(":")[0] for k in ranked[:2]]
            return self._propose("early_safety", [(a, "standard") for a in assets], phases)
        best = ranked[0]
        row, harm = benefit[best], safety.get(best)
        favourable = row["upper"] < 0
        safe = harm is None or harm["lower"] < 0.05
        asset, dose = best.split(":")[0], best.split(":")[1]
        if self._stage == "early_safety_done":
            if row["estimate"] >= 0 or not safe:
                return "stop_development", {}
            self._stage = "exploration"
            return self._propose("exploration", [(asset, dose)], phases)
        if self._stage == "exploration_done":
            if not favourable or not safe:
                return "stop_development", {}
            self._stage = "confirmation"
            sd = max((row["upper"] - row["lower"]) / (2 * 1.96) * math.sqrt(self._last_n / 2), 1e-6)
            n = int(math.ceil(2 * ((1.96 + 1.2816) * sd / abs(row["estimate"])) ** 2))
            n = max(n, int(self._programme["minimum_completed_treated"]) + 30)
            return self._propose("confirmation", [(asset, dose)], phases, per_arm=n)
        if self._stage == "confirmation_done":
            if favourable:
                return "advance_candidate", {"compound": asset, "dose": dose}
            return "stop_development", {}
        raise AssertionError(f"No decision rule for stage {self._stage}")

    _last_n = 100

    def _propose(
        self,
        phase: str,
        arms: list[tuple[str, str]],
        phases: dict[str, dict[str, Any]],
        per_arm: int | None = None,
    ) -> tuple[str, dict[str, object]]:
        assert self._programme is not None
        rule = phases[phase]
        low, high = int(rule["minimum"]), int(rule["maximum"])
        size = min(max(per_arm or {"early_safety": 30, "exploration": 120}.get(phase, low), low), high)
        self._last_n = size
        options = sorted(int(day) for day in rule["follow_up_days"])
        endpoint = int(self._programme["endpoint_day"])
        follow = next((day for day in options if day >= endpoint), options[-1])
        day = float(self._programme["programme_day"])
        startup = float(rule["startup_days"])
        horizon = float(self._programme["horizon_days"])
        if day + startup + follow + 180 > horizon:
            self.log.note(f"{phase} study cannot complete follow-up before the horizon; development stopped")
            return "stop_development", {}
        deadline = min(day + startup + follow + 540, horizon)
        self._pending = {
            "phase": phase,
            "arms": [{"compound": a, "dose": d, "participants": size} for a, d in arms],
            "control_participants": size,
            "sites": 40 if phase != "early_safety" else 10,
            "follow_up_days": follow,
            "looks": [{"deadline_day": deadline, "completed_participants": size * (len(arms) + 1)}],
        }
        return "start_study", self._pending | {"preview": True}

    def _after_action(self, text: str) -> tuple[str, dict[str, object]]:
        action = self._last_action
        if text.startswith("Tool input rejected"):
            self._rejections += 1
            if self._rejections > 2 or self._pending is None:
                return "stop_development", {}
            self._pending["sites"] = max(2, int(self._pending["sites"]) // 2)
            return "start_study", self._pending | {"preview": True}
        assert self._programme is not None
        day = re.search(r"Programme day ([\d,]+) of", text)
        if day is not None:
            self._programme["programme_day"] = float(day.group(1).replace(",", ""))
        if action == "start_study:preview":
            assert self._pending is not None
            return "start_study", self._pending
        if action == "start_study":
            started = re.search(r"^## (study-\d{3}) started", text, flags=re.MULTILINE)
            if started is None:
                raise ValueError(f"Start letter names no study: {text[:200]}")
            self._study_id = started.group(1)
            return "request_readout", {"study_id": self._study_id}
        if action == "request_readout":
            self._stage = self._stage + "_done"
            return "execute_code", _run_script()
        raise AssertionError(f"Unexpected programme letter after {action}")


def _summary(arguments: dict[str, object]) -> str:
    text = {k: (v if k != "content" and k != "code" else f"<{len(str(v))} chars>") for k, v in arguments.items()}
    return str(text)[:600]


def _script(name: str) -> str:
    return (HERE / name).read_text(encoding="utf-8")


def _run_script() -> dict[str, object]:
    return {
        "code": "import runpy, sys\nsys.path.insert(0, '.')\nrunpy.run_path('scratch/analysis.py', run_name='__main__')",
        "language": "python",
        "purpose": "Run the saved analysis.",
    }


_PROGRAMME_SUMMARY = r"""import json, re
from pathlib import Path
brief = Path('BRIEF_PATH').read_text()
phases = [
    dict(phase=m[0], minimum=int(m[1].replace(',', '')), maximum=int(m[2].replace(',', '')),
         follow_up_days=[int(x) for x in m[3].split(',')], startup_days=int(m[4]))
    for m in re.findall(r"^\| [^|]*\(`(\w+)`\) \| [^|]* \| ([\d,]+)–([\d,]+) \| ([\d, ]+) \| (\d+) \| \d+ \|$",
                        brief, flags=re.MULTILINE)
]
protocol = Path('documents/evidence_protocol.md').read_text()
summary = dict(
    phases=phases,
    horizon_days=int(re.search(r"Time horizon \| programme day ([\d,]+)", brief).group(1).replace(',', '')),
    minimum_completed_treated=int(re.search(r"at least ([\d,]+) completed participants", brief).group(1).replace(',', '')),
    endpoint_day=int(re.search(r'Clinical endpoint:[^\n]*?Day (\d+)', protocol).group(1)),
    programme_day=0.0,
)
assert len(phases) == 3, phases
print('PROGRAMME=' + json.dumps(summary))
"""
