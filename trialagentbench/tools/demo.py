"""Run a clinical research demonstration through the native agent/provider boundary."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from html import escape
from importlib.resources import files
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict
from trialagentbench_verifier.authored_results import (
    TrialDevNumericalRecovery,
    TrialDevProgrammeRun,
    mean_numerical_loss,
)
from trialagentbench_verifier.campaign import trialeval_target_consequence_v1
from trialagentbench_verifier.decision_consequence import DecisionReportV1
from trialagentbench_verifier.executor import resolve_executor_environment
from trialagentbench_verifier.numerical import NumericalReplay, replay_numerical_program

from trialagentbench.contracts.release.archive_safety import materialized_release_archive
from trialagentbench.execution_policy import TRIALDEV_RELEASE_BUDGET_V1, TRIALEVAL_RELEASE_BUDGET_V1
from trialagentbench.grading.outcome import TrialEvalOutcomeGradeV1, grade_trialeval_outcome_v1
from trialagentbench.io import read_json_model, sha256_file, write_json_model
from trialagentbench.ports.llm_provider import JsonObject, LLMProvider, LLMResponse, LLMResponseMetadata, ToolCall
from trialagentbench.release import load_trialeval_grading_key
from trialagentbench.tools.run.trialeval import _archive_partial_item
from trialagentbench.trialdev.release import load_trialdev_evaluator_release
from trialagentbench.trialdev.resources import TrialDevBudgetLedger
from trialagentbench.trialdev.run import run_trialdev_task
from trialagentbench.trialeval.agent import run_agent
from trialagentbench.trialeval.data import discover_participant_items, participant_evidence_artifacts
from trialagentbench.trialeval.outcome_submission import bind_trialeval_numerical_input
from trialagentbench.util.run_ownership import exclusive_run_directory


class _Function(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    name: str
    arguments: str


class _Call(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    id: str
    function: _Function


class _Message(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    role: str
    tool_calls: tuple[_Call, ...] | None = None
    tool_call_id: str | None = None
    content: str | None = None


def demo_assets() -> Path:
    """Locate the bundled demonstration archives in an installed wheel or the source checkout."""
    installed = Path(str(files("trialagentbench").joinpath("resources/examples/clinical")))
    return installed if installed.is_dir() else Path(__file__).resolve().parents[2] / "examples" / "clinical"


class ScriptedDemoProvider:
    """Example external provider using only participant messages and tool outputs.

    The supplied analysis calculates naive observed contrasts with uncertainty.
    It demonstrates the integration protocol, not statistical competence.
    It has no evaluator paths, frozen answers, hidden pool access, or API client.
    Its next action is derived from conversation history, including on resume.
    """

    model = "disclosed-scripted-teaching-analysis"
    telemetry_route = "local-scripted"

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
        """Return one ordinary tool call determined by public conversation state."""
        history = tuple(_Message.model_validate(message) for message in messages)
        calls = tuple(call for message in history for call in message.tool_calls or ())
        offered = set()
        for tool in tools or ():
            function = tool.get("function")
            if isinstance(function, dict):
                offered.add(str(function["name"]))
        programme = "start_study" in offered
        arguments: dict[str, object]
        if not calls:
            script = (demo_assets() / "observed_analysis.py").read_text(encoding="utf-8")
            code = f"from pathlib import Path\nPath('scratch/analysis.py').write_text({script!r})\n"
            code += "exec(compile(Path('scratch/analysis.py').read_text(), 'analysis.py', 'exec'))\n"
            if programme:
                code += (demo_assets() / "choose_study.py").read_text(encoding="utf-8")
            name, arguments = "execute_code", {"code": code, "language": "python"}
        elif not programme:
            name, arguments = "submit_response", {
                "content": "Disclosed teaching analysis: linear treatment-by-time contrast at Day365 with participant-clustered uncertainty. "
                "No adjustment for missingness or nonlinear time effects. "
                "This exercises native reporting and replay; it is not a qualified analysis or model performance result."
            }
        else:
            replies = {message.tool_call_id: message.content for message in history if message.role == "tool"}
            studies = [call for call in calls if call.function.name == "start_study"]
            if len(studies) < 2:
                text = replies.get(calls[0].id)
                if text is None:
                    raise ValueError("Demo is missing its public study-design reply")
                design_line = next(line for line in text.splitlines() if line.startswith("DEMO_STUDY="))
                request = json.loads(design_line.removeprefix("DEMO_STUDY="))
                name, arguments = "start_study", request | {"preview": not studies}
            elif not any(call.function.name == "request_readout" for call in calls):
                name, arguments = "request_readout", {"study_id": "study-101"}
            elif sum(call.function.name == "execute_code" for call in calls) == 1:
                name, arguments = "execute_code", {
                    "code": "from pathlib import Path\nexec(compile(Path('scratch/analysis.py').read_text(), 'analysis.py', 'exec'))",
                    "language": "python",
                }
            else:
                name, arguments = "stop_development", {}
        if name not in offered:
            raise ValueError(f"Demo requested unavailable tool: {name}")
        return LLMResponse(
            tool_calls=[ToolCall(id=f"demo-{len(calls) + 1}", name=name, arguments=json.dumps(arguments))],
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            metadata=LLMResponseMetadata(returned_model=self.model, reported_cost_usd=0.0, finish_reason="tool_calls"),
        )


def run_demo(*, output: Path, provider: LLMProvider, executor_image: str | None, resume: bool = False) -> None:
    """Execute and grade teaching tasks with native evidence, replay and checkpoints."""
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=resume)
    assets = demo_assets()
    with exclusive_run_directory(output):
        executor = resolve_executor_environment(
            image=executor_image, limits=TRIALEVAL_RELEASE_BUDGET_V1.code_execution
        )
        if resume and (output / "executor.json").is_file():
            if read_json_model(type(executor), output / "executor.json") != executor:
                raise ValueError("Demo resume executor differs from the recorded environment")
        write_json_model(output / "executor.json", executor)
        te_output = output / "trialeval"
        if not (resume and (te_output / "grade-0.json").is_file()):
            if resume:
                archived = _archive_partial_item(output, "trialeval", candidate_paths=(te_output,))
                if archived is not None:
                    print(f"Retained interrupted TrialEval attempt: {archived}")
                    print("Restarting TrialEval from its initial evidence; prior work and usage remain archived.")
            te_output.mkdir(exist_ok=resume)
            archive = assets / "demo-trialeval-evaluator.zip"
            with materialized_release_archive(archive, expected_sha256=sha256_file(archive)) as materialized:
                items = discover_participant_items(materialized.root / "participant")
                item = next(item for item in items.values() if item.context_tier == "C1")
                record = load_trialeval_grading_key(materialized.root, expected_task_ids=(item.item_id,)).for_task(
                    item.item_id
                )
                result = run_agent(
                    item,
                    provider,
                    max_turns=8,
                    verbose=False,
                    analysis_language="python",
                    item_workspace=te_output / "workspace",
                    conversation_log_path=te_output / "conversation.json",
                    event_log_path=te_output / "events.jsonl",
                    provider_log_path=te_output / "provider_responses.jsonl",
                    executor_image=executor.image_id,
                    executor_limits=executor.limits,
                )
                if result["status"] != "success":
                    raise RuntimeError(f"TrialEval demonstration failed: {result['status']}")
                evidence = participant_evidence_artifacts(item)
                replay = replay_numerical_program(
                    workspace=te_output / "workspace/item",
                    entrypoint="analysis.py",
                    target_ids=record.reported_target_ids,
                    evidence_artifacts=evidence,
                    executor_image=executor.image_id,
                    executor_limits=executor.limits,
                    analysis_language="python",
                )
                write_json_model(te_output / "replay.json", replay)
                for index, target in enumerate(record.target_ids):
                    grade = grade_trialeval_outcome_v1(
                        bind_trialeval_numerical_input(
                            evaluator_record=record, replay=replay, expected_evidence=evidence, target_id=target
                        )
                    )
                    write_json_model(te_output / f"grade-{index}.json", grade)
        archive = assets / "demo-trialdev-evaluator.zip"
        with materialized_release_archive(archive, expected_sha256=sha256_file(archive)) as materialized:
            release = load_trialdev_evaluator_release(materialized.root)
            task = next(
                t
                for t in release.participant.tasks
                if t.stream.value == "portfolio_management" and t.data_view == "analysis_ready"
            )
            run = run_trialdev_task(
                release=release,
                task_id=task.task_id,
                provider=provider,
                output_root=output / "trialdev",
                max_turns_per_step=12,
                max_tokens=2048,
                max_context_characters=500_000,
                watchdog_seconds=600,
                tool_choice="auto",
                executor_image=executor.image_id,
                executor_limits=TRIALDEV_RELEASE_BUDGET_V1.code_execution,
                analysis_language="python",
                resume=resume,
            )
            if run.execution_status != "completed":
                raise RuntimeError(f"TrialDev demonstration failed: {run.execution_status}: {run.error}")
        _write_demo_report(output, run.task_id)
        print(f"Clinical research demonstration completed: {output}")
        print(f"Open {output / 'index.html'}")
        print("Explore the clinical estimates, uncertainty and consequences of the research decisions.")


def _write_demo_report(output: Path, task_id: str) -> None:
    """Explain the native grades and expose the observed evidence without rescoring."""
    te_output = output / "trialeval"
    replay = read_json_model(NumericalReplay, te_output / "replay.json")
    programme_path = output / "trialdev" / task_id / "programme_run.json"
    programme = read_json_model(TrialDevProgrammeRun, programme_path)
    ledger = read_json_model(TrialDevBudgetLedger, programme_path.parent / "private/budget.json")

    def number(value: float | None) -> str:
        return "Unavailable" if value is None else f"{value:.4g}".replace("-", "−")

    def money(value: float) -> str:
        return f"{'−' if value < 0 else ''}${abs(value):,.0f}"

    def table(headings: tuple[str, ...], rows: Sequence[tuple[str, ...]]) -> str:
        head = "".join(f'<th scope="col">{escape(value)}</th>' for value in headings)
        body = "".join("<tr>" + "".join(f"<td>{escape(value)}</td>" for value in row) + "</tr>" for row in rows)
        return (
            f'<div class="table-scroll" tabindex="0" role="region" aria-label="{escape(headings[0])} results">'
            f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
        )

    te_rows: list[tuple[str, str]] = []
    archive = demo_assets() / "demo-trialeval-evaluator.zip"
    with materialized_release_archive(archive, expected_sha256=sha256_file(archive)) as materialized:
        item = next(
            item
            for item in discover_participant_items(materialized.root / "participant").values()
            if item.context_tier == "C1"
        )
        key = load_trialeval_grading_key(materialized.root, expected_task_ids=(item.item_id,)).for_task(item.item_id)
    for index in range(len(tuple(te_output.glob("grade-*.json")))):
        grade = read_json_model(TrialEvalOutcomeGradeV1, te_output / f"grade-{index}.json")
        point, interval = grade.point_performance, grade.interval_performance
        answer = (
            None
            if point is None or interval is None
            else DecisionReportV1(
                estimate=point.submitted_value, lower=interval.native_lower, upper=interval.native_upper
            )
        )
        consequence = trialeval_target_consequence_v1(key.references[index].target, answer)
        if consequence is None:
            raise ValueError("Demo grading key lacks the TrialEval decision-consequence reference")
        te_rows.extend(
            (
                (
                    "Development decision driven by the answer",
                    "Run a confirmatory trial" if consequence.agent.confirm else "Stop development",
                ),
                (
                    "Regret at the true effect — lower is better",
                    money(consequence.realised_regret_musd * 1_000_000),
                ),
            )
        )
        te_rows.extend(
            (
                ("Estimate (mg/g)", number(None if point is None else point.submitted_value)),
                (
                    "95% interval (mg/g)",
                    (
                        "Unavailable"
                        if interval is None
                        else f"{number(interval.native_lower)} to {number(interval.native_upper)}"
                    ),
                ),
                ("Population reference (mg/g)", number(None if point is None else point.oracle_value)),
                ("Absolute error (mg/g)", number(None if point is None else point.native_absolute_error)),
                ("Numerical loss — lower is better", number(grade.signal_loss)),
            )
        )

    def development_rows(
        observed: NumericalReplay | None, scores: tuple[TrialDevNumericalRecovery, ...]
    ) -> list[tuple[str, ...]]:
        estimates = {} if observed is None else {row.target_id: row for row in observed.estimates or ()}
        rows: list[tuple[str, ...]] = []
        for score in scores:
            estimate = estimates.get(score.target_id)
            rows.append(
                (
                    score.target_id.replace(":", " / ").replace("_", " "),
                    number(None if estimate is None else estimate.estimate),
                    "Unavailable" if estimate is None else f"{number(estimate.lower)} to {number(estimate.upper)}",
                    number(score.reference_value),
                    number(score.loss),
                    score.status,
                )
            )
        return rows

    evaluation = programme.evaluation
    initial_table = table(
        ("Starting-cohort target", "Estimate", "95% interval", "Reference", "Numerical loss", "Result"),
        development_rows(evaluation.initial_replay, evaluation.initial_scores),
    )
    study_tables = "".join(
        f"<h3>Acquired study {index + 1}</h3><p>Replay: <strong>{escape(study.replay.status)}</strong>. "
        f"Mean target loss: {number(mean_numerical_loss(study.scores))}.</p>"
        + table(
            ("Study target", "Estimate", "95% interval", "Reference", "Numerical loss", "Result"),
            development_rows(study.replay, study.scores),
        )
        for index, study in enumerate(evaluation.studies)
    )
    value = evaluation.value
    decision = value.decision if evaluation.programme_complete else "unfinished, valued as a stop"
    value_html = (
        f"<p>Decision: <strong>{escape(decision)}</strong>. "
        f"Programme time: <strong>{ledger.resource_day:,.2f}".rstrip("0").rstrip(".")
        + " days</strong>.</p>"
        + table(
            ("Native value component", "USD"),
            tuple(
                (label, money(amount))
                for label, amount in (
                    ("Treatment benefit", value.clinical_value_usd),
                    ("Safety cost", value.safety_cost_usd),
                    ("Study expenditure", value.study_cost_usd),
                    ("Delay cost", value.time_cost_usd),
                    ("Net development value — higher is better", value.net_value_usd),
                )
            ),
        )
        + "<p>A stop decision, or an unfinished programme, receives no advancement benefit and retains expenditure "
        "and delay already incurred. "
        "This one realized programme does not estimate the average value of a development policy.</p>"
    )
    tables_directory = output / "tables"
    tables_directory.mkdir(exist_ok=True)
    data_sources = (
        ("TrialEval observed follow-up", te_output / "workspace/item/data/analysis.parquet"),
        ("TrialDev starting cohort", programme_path.parent / "workspace/data/observational_cohort.parquet"),
    ) + tuple(
        (f"TrialDev acquired study {index + 1} measurements", path)
        for index, path in enumerate(
            sorted((programme_path.parent / "workspace/analysis_ready").glob("*/measurements.parquet"))
        )
    )
    data_html = []
    for index, (label, path) in enumerate(data_sources):
        frame = pd.read_parquet(path)
        csv_path = tables_directory / f"observed-{index + 1}.csv"
        frame.to_csv(csv_path, index=False)
        data_html.append(
            f"<h3>{escape(label)}</h3><p>{len(frame):,} observed rows, {len(frame.columns)} columns. "
            f'<a href="{csv_path.relative_to(output).as_posix()}">Download every observed row as CSV</a>.</p>'
            "<details><summary>Inspect the first eight rows (first eight columns)</summary>"
            f'<div class="table-scroll" tabindex="0" role="region" aria-label="{escape(label)} preview">'
            + frame.iloc[:8, :8].to_html(index=False, border=0, escape=True)
            + "</div></details>"
        )
    interrupted = tuple(sorted((output / "failed_attempts/trialeval").glob("attempt-*")))
    attempts_html = ""
    if interrupted:
        attempt_links = "".join(
            f'<li><a href="{path.relative_to(output).as_posix()}">{escape(path.parent.parent.name)}: {escape(path.name)}</a></li>'
            for attempt in interrupted
            for path in sorted(attempt.rglob("*"))
            if path.is_file() and path.name in {"conversation.json", "provider_responses.jsonl", "events.jsonl"}
        )
        attempts_html = (
            f"<section><h2>Interrupted TrialEval attempts: {len(interrupted)}</h2>"
            "<p>TrialEval restarted from its initial evidence after preserving these attempts. Their work, "
            "provider usage and errors are retained separately; they are not additional scientific successes. "
            "Include their provider charges when assessing the total cost of this run.</p>"
            f"<ul>{attempt_links}</ul></section>"
        )
    links = (
        ("TrialEval clinical question and supplied records", "trialeval/workspace/item/analysis_request.md"),
        ("TrialEval observed estimates", "trialeval/workspace/item/scratch/estimates.csv"),
        ("TrialEval submitted analysis", "trialeval/workspace/item/scratch/analysis.py"),
        ("TrialEval native numerical grade", "trialeval/grade-0.json"),
        ("TrialEval replay and execution proof", "trialeval/replay.json"),
        ("TrialDev acquired estimates", f"trialdev/{task_id}/workspace/scratch/estimates.csv"),
        ("TrialDev submitted analysis", f"trialdev/{task_id}/workspace/scratch/analysis.py"),
        ("TrialDev native programme grade", programme_path.relative_to(output).as_posix()),
        ("TrialDev participant conversation", f"trialdev/{task_id}/conversation.json"),
        ("TrialDev resource ledger", f"trialdev/{task_id}/private/budget.json"),
        ("Local executor identity", "executor.json"),
    )
    link_items = "".join(f'<li><a href="{escape(path)}">{escape(label)}</a></li>' for label, path in links)
    analysis = escape((te_output / "workspace/item/scratch/analysis.py").read_text(encoding="utf-8"))
    (output / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Clinical research demonstration · TrialAgentBench</title><style>"
        "body{font:17px/1.6 system-ui;color:#193b37;background:#faf9f5;max-width:75ch;margin:3rem auto;padding:0 1rem}"
        "h1,h2{line-height:1.2}section{border-top:1px solid #a9bbb6;margin-top:2rem;padding-top:1rem}"
        "table{border-collapse:collapse;width:100%;font-size:.9rem}td,th{text-align:left;padding:.6rem;border-bottom:1px solid #c7d4ce}"
        "th{vertical-align:bottom}td:first-child{max-width:18rem;overflow-wrap:anywhere}"
        ".table-scroll{overflow:auto}:focus-visible{outline:2px solid #075e73;outline-offset:3px}"
        "pre{font-size:.85rem;white-space:pre-wrap;overflow-wrap:anywhere}"
        "li{margin:.5rem 0}a{color:#075e73;overflow-wrap:anywhere}</style><main>"
        "<h1>Clinical answers, checked against known outcomes</h1>"
        "<p>A fixed script analyses a kidney-disease trial, then commissions a rheumatoid-arthritis study "
        "and decides whether to continue. Compare its clinical estimates with the world-derived targets, "
        "then inspect what the research cost and what it revealed.</p>"
        f"<details><summary>Execution details</summary><p>TrialEval replay: {escape(replay.status)}. "
        f"TrialDev execution: {escape(programme.execution_status)}.</p></details>"
        "<section><h2>1. Recover the kidney-treatment effect</h2>"
        "<p>The question asks for the treatment-minus-control difference in albuminuria at Day365, in mg/g. "
        "Lower albuminuria favours treatment. The script fits a linear treatment-by-time contrast with "
        "participant-clustered uncertainty. It does not correct missingness or nonlinear trajectories.</p>"
        + table(("TrialEval quantity", "Native result"), te_rows)
        + "<p>Regret is the value, at the world's true effect, that the answer's stop-or-confirm decision loses against "
        "the decision of the reference analysis of the same data, under the demo grading key's declared "
        "confirmation economics, in US dollars. The benchmark headline replays the saved script on 30 fresh datasets "
        "and averages this regret over them; the demo values its one dataset. This is one teaching task, not a "
        "benchmark ranking.</p>"
        "<p>Numerical loss is dimensionless; lower is better. Interval width contributes to loss even when the "
        "interval covers the population reference. A broad interval is not strong evidence of accurate estimation.</p></section>"
        "<section><h2>2. Analyse the rheumatoid-arthritis evidence</h2>"
        "<p>Clinical outcome differences use DAS28 points; negative values favour treatment. Safety differences use "
        "infection-risk proportions (0.01 means one percentage point); positive values indicate more infections. "
        "Continuous contrasts use Welch intervals; binary risk differences use Agresti–Caffo intervals. "
        "Neither uses confounding adjustment, missingness correction or "
        "population standardization. Starting-cohort references describe standardized observational associations.</p>"
        + "<p>On a narrow screen, scroll the target tables horizontally to inspect references and loss.</p>"
        + initial_table
        + study_tables
        + "<p>All target losses use the native normalization and grading rules; lower is better. "
        "Initial and acquired analyses answer different questions and are reported separately.</p></section>"
        "<section><h2>3. Account for the development decision</h2>" + value_html + "</section>"
        "<section><h2>Inspect the observed evidence and analysis</h2>"
        + "".join(data_html)
        + "<details><summary>Read the actual submitted teaching analysis</summary><pre>"
        + analysis
        + "</pre></details>"
        "<p>To improve this analysis, begin with the supplied clinical question and protocol. Check the outcome clock, "
        "missingness and target population before changing the statistical model. To evaluate a real agent, replace "
        "the external-provider example's teaching policy, then use explicit scored task IDs in a separate campaign.</p>"
        "</section>"
        + attempts_html
        + "<section><h2>Native records</h2><ul>"
        + link_items
        + "</ul></section></main></html>",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the installed, provider-free first-run walkthrough."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Integration example: examples/clinical/run_external_provider.py in the public source, "
        "or trialagentbench/resources/examples/clinical/ in the installed package. "
        "Use trialagentbench.tools.demo.run_demo(provider=your_provider, ...) with the public LLMProvider protocol.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executor-image")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    run_demo(
        output=args.output, provider=ScriptedDemoProvider(), executor_image=args.executor_image, resume=args.resume
    )
    return 0
