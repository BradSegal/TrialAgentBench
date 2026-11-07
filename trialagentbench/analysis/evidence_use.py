"""Evidence-use rows derived from origin-first source classification."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

from trialagentbench.analysis.evidence_classifier import (
    EvidenceClassificationError,
    classify_evidence_source,
    is_hidden_or_grader_path,
)
from trialagentbench.contracts.trace.observable import (
    EvidenceUseRowV1,
    ModelActionTraceEventV1,
    is_successful_file_read,
)


def _program_dir_for_event(event: ModelActionTraceEventV1) -> Path | None:
    source_path = Path(event.source_path)
    if source_path.name in {"conversation.json", "events.jsonl"}:
        return source_path.parent
    return None


def _task_workspace(event: ModelActionTraceEventV1) -> Path | None:
    """Return the workspace beside a task's event log: TrialDev <task>/workspace, TrialEval <task>_workspace/item."""
    source = Path(event.source_path)
    if event.benchmark == "trialdev":
        return source.parent / "workspace"
    task = source.name.removesuffix("_events.jsonl")
    return None if task == source.name else source.parent / f"{task}_workspace" / "item"


def _scenario_id_for_event(event: ModelActionTraceEventV1) -> str | None:
    if event.benchmark == "trialdev" and event.scenario_id is None:
        raise ValueError("TrialDev evidence events require explicit scenario_id provenance.")
    return event.scenario_id


def evidence_rows_from_events(
    events: list[ModelActionTraceEventV1],
    *,
    trialdev_release_root: Path | None = None,
) -> list[EvidenceUseRowV1]:
    """Convert file-inspection and code-path trace events into evidence-use rows."""
    rows: list[EvidenceUseRowV1] = []
    for event in events:
        if not is_successful_file_read(event):
            continue
        path = event.file_accessed
        if not path:
            continue
        if event.tool_call_id is None:
            raise ValueError(f"File-inspection event lacks tool_call_id: {event.event_id}")
        classification = classify_evidence_source(
            path,
            event_source_path=event.source_path,
            program_dir=_program_dir_for_event(event),
            trialdev_release_root=trialdev_release_root,
            scenario_id=_scenario_id_for_event(event),
            participant_release_relative=event.benchmark == "trialeval",
            workspace=_task_workspace(event),
        )
        rows.append(
            EvidenceUseRowV1(
                benchmark=event.benchmark,
                model_id=event.model_id,
                run_id=event.run_id,
                task_id=event.task_id,
                assignment_id=event.assignment_id,
                program_id=event.program_id,
                phase_id=event.phase_id,
                evidence_category=classification.evidence_category,
                source=cast(
                    Literal["tool_call", "code_path", "structured_field", "text_citation", "validator"],
                    "tool_call",
                ),
                artifact_path=path,
                participant_facing=classification.participant_facing,
                leakage_violation=classification.hidden_or_grader,
            )
        )
    return rows


__all__ = [
    "EvidenceClassificationError",
    "evidence_rows_from_events",
    "is_hidden_or_grader_path",
]
