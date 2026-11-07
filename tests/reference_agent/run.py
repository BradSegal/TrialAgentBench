"""Run the scripted reference agent through the native runners on development archives.

Usage::

    python -m reference_agent.run --inputs DIR --output DIR [--trialeval-task ID ...] [--trialdev-task ID ...]

``DIR`` holds ``trialeval-evaluator.zip`` and/or ``trialdev-evaluator.zip`` (for example a
development campaign input directory). TrialEval tasks are replayed and graded exactly as
the demonstration does; TrialDev programmes run through ``run_trialdev_task``. The summary
(``reference_agent.json``) records, per task, completion, the tool sequence, every tool
error and every friction point the agent observed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from trialagentbench_verifier.executor import resolve_executor_environment
from trialagentbench_verifier.numerical import replay_numerical_program

from reference_agent.agent import ReferenceAgent
from trialagentbench.contracts.release.archive_safety import materialized_release_archive
from trialagentbench.execution_policy import TRIALDEV_RELEASE_BUDGET_V1, TRIALEVAL_RELEASE_BUDGET_V1
from trialagentbench.grading.outcome import grade_trialeval_outcome_v1
from trialagentbench.io import sha256_file
from trialagentbench.release import load_trialeval_grading_key
from trialagentbench.trialdev.release import load_trialdev_evaluator_release
from trialagentbench.trialdev.run import run_trialdev_task
from trialagentbench.trialeval.agent import run_agent
from trialagentbench.trialeval.data import discover_participant_items, participant_evidence_artifacts
from trialagentbench.trialeval.outcome_submission import bind_trialeval_numerical_input


def run_trialeval(archive: Path, task_ids: list[str], output: Path, image: str | None) -> list[dict[str, object]]:
    """Run, replay and grade each selected TrialEval task with a fresh reference agent."""
    executor = resolve_executor_environment(image=image, limits=TRIALEVAL_RELEASE_BUDGET_V1.code_execution)
    rows: list[dict[str, object]] = []
    with materialized_release_archive(archive, expected_sha256=sha256_file(archive)) as materialized:
        items = discover_participant_items(materialized.root / "participant")
        records = load_trialeval_grading_key(materialized.root, expected_task_ids=tuple(task_ids or items))
        for task_id in task_ids or sorted(items):
            agent, item, root = ReferenceAgent(), items[task_id], output / "trialeval" / task_id
            started = time.monotonic()
            result = run_agent(
                item,
                agent,
                max_turns=12,
                verbose=False,
                analysis_language="python",
                item_workspace=root / "workspace",
                conversation_log_path=root / "conversation.json",
                event_log_path=root / "events.jsonl",
                provider_log_path=root / "provider_responses.jsonl",
                executor_image=executor.image_id,
                executor_limits=executor.limits,
            )
            row: dict[str, object] = {"benchmark": "trialeval", "task_id": task_id, "status": result["status"]}
            record = records.for_task(task_id)
            replay = replay_numerical_program(
                workspace=root / "workspace/item",
                entrypoint="analysis.py",
                target_ids=record.reported_target_ids,
                evidence_artifacts=participant_evidence_artifacts(item),
                executor_image=executor.image_id,
                executor_limits=executor.limits,
                analysis_language="python",
            )
            row["replay"] = replay.status
            if replay.status == "reproduced":
                losses = [
                    grade_trialeval_outcome_v1(
                        bind_trialeval_numerical_input(
                            evaluator_record=record,
                            replay=replay,
                            expected_evidence=participant_evidence_artifacts(item),
                            target_id=target,
                        )
                    ).model_dump(mode="json")
                    for target in record.target_ids
                ]
                row["grades"] = [{k: g.get(k) for k in ("target_id", "signal_loss") if k in g} for g in losses]
            row |= {"seconds": round(time.monotonic() - started, 1), "turns": len(agent.log.calls)}
            row |= {"calls": agent.log.calls, "friction": agent.log.friction}
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("task_id", "status", "replay", "turns")}))
    return rows


def run_trialdev(archive: Path, task_ids: list[str], output: Path, image: str | None) -> list[dict[str, object]]:
    """Run each selected TrialDev task with a fresh reference agent through the native runner."""
    rows: list[dict[str, object]] = []
    with materialized_release_archive(archive, expected_sha256=sha256_file(archive)) as materialized:
        release = load_trialdev_evaluator_release(materialized.root)
        for task_id in task_ids:
            agent = ReferenceAgent()
            started = time.monotonic()
            run = run_trialdev_task(
                release=release,
                task_id=task_id,
                provider=agent,
                output_root=output / "trialdev",
                max_turns_per_step=TRIALDEV_RELEASE_BUDGET_V1.maximum_turns,
                max_tokens=2048,
                max_context_characters=500_000,
                watchdog_seconds=1800,
                tool_choice="auto",
                executor_image=image,
                executor_limits=TRIALDEV_RELEASE_BUDGET_V1.code_execution,
                analysis_language="python",
            )
            payload = run.model_dump(mode="json")
            row: dict[str, object] = {
                "benchmark": "trialdev",
                "task_id": task_id,
                "status": run.execution_status,
                "error": payload.get("error"),
                "seconds": round(time.monotonic() - started, 1),
                "turns": len(agent.log.calls),
                "calls": agent.log.calls,
                "friction": agent.log.friction,
            }
            programme = output / "trialdev" / task_id / "programme_run.json"
            if programme.is_file():
                evaluation = json.loads(programme.read_text())["evaluation"] or {}
                row["value"] = evaluation.get("value")
                row["initial_scores"] = [
                    {k: score.get(k) for k in ("target_id", "normalized_interval_loss")}
                    for score in evaluation.get("initial_scores") or ()
                ]
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("task_id", "status", "turns", "error")}))
    return rows


def main() -> None:
    """Run the selected tasks and write the reference-agent record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trialeval-task", action="append", default=[])
    parser.add_argument("--trialdev-task", action="append", default=[])
    parser.add_argument("--executor-image")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    if args.trialeval_task:
        rows += run_trialeval(
            args.inputs / "trialeval-evaluator.zip", args.trialeval_task, args.output, args.executor_image
        )
    if args.trialdev_task:
        rows += run_trialdev(
            args.inputs / "trialdev-evaluator.zip", args.trialdev_task, args.output, args.executor_image
        )
    (args.output / "reference_agent.json").write_text(json.dumps(rows, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
