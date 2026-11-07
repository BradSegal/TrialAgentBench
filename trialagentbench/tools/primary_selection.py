"""Reproduce source-bound earliest completions across same-condition campaign restarts."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from trialagentbench_verifier.campaign import TrialEvalCampaignPlanV1
from trialagentbench_verifier.scorer import current_scorer

from trialagentbench.contracts.campaign import CampaignRunV1
from trialagentbench.contracts.core.manifest import GradeManifestV1
from trialagentbench.contracts.core.runs import TrialEvalItemResultV1, TrialEvalRunConfigV1
from trialagentbench.io import read_json_model, sha256_file, write_json_model

PRIMARY_SELECTION = "trialeval-primary-selection.json"


class TrialEvalEarlierCompletionV1(BaseModel):
    """One audited earlier native completion, never a caller-authored score."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    target_job_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    target_run_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_run_root: Path
    source_run_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_item_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class TrialEvalPrimarySelectionV1(BaseModel):
    """Derived selection custody; original campaign plans and native records remain intact."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_id: Literal["trialagentbench.trialeval_primary_selection/v1"] = (
        "trialagentbench.trialeval_primary_selection/v1"
    )
    audit_path: Path
    audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selections: tuple[TrialEvalEarlierCompletionV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_sources(self) -> Self:
        keys = [(row.target_job_id, row.task_id) for row in self.selections]
        if len(keys) != len(set(keys)):
            raise ValueError("Earlier-completion selection repeats a primary task")
        if not self.audit_path.is_absolute() or any(not row.source_run_root.is_absolute() for row in self.selections):
            raise ValueError("Earlier-completion custody paths must be absolute")
        return self


def validate_earlier_completion(
    row: TrialEvalEarlierCompletionV1, target: Path
) -> tuple[TrialEvalRunConfigV1, TrialEvalItemResultV1]:
    """Verify chronology, immutable source bytes and the task's scientific condition."""
    source = row.source_run_root
    for path, expected in (
        (target / "run_config.json", row.target_run_config_sha256),
        (source / "run_config.json", row.source_run_config_sha256),
        (source / "items" / f"{row.task_id}.json", row.source_item_sha256),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"Earlier-completion custody changed: {path}")
    earlier = read_json_model(TrialEvalRunConfigV1, source / "run_config.json")
    later = read_json_model(TrialEvalRunConfigV1, target / "run_config.json")
    item = read_json_model(TrialEvalItemResultV1, source / "items" / f"{row.task_id}.json")
    current = read_json_model(TrialEvalItemResultV1, target / "items" / f"{row.task_id}.json")
    if (
        item.item_id != row.task_id
        or item.run_config != earlier
        or current.run_config != later
        or item.agent_output.status != "success"
        or item.timestamp_utc >= current.timestamp_utc
    ):
        raise ValueError("Selection requires a retained earlier completed primary")

    def scientific(config: TrialEvalRunConfigV1) -> dict[str, object]:
        payload = config.model_dump(mode="json")
        for key in (
            "timestamp_utc",
            "run_identity_sha256",
            "workers",
            "participant_archive",
            "task_ids",
            "n_items",
            "scorer_source_sha256",
        ):
            payload.pop(key, None)
        # Routing/transport retry policy does not create a new model condition.
        payload.pop("retry", None)
        factors = config.task_evidence_factors.get(row.task_id)
        if factors is None:
            raise ValueError("Earlier-completion task is absent from its invocation")
        payload["task_evidence_factors"] = {row.task_id: factors.model_dump(mode="json")}
        condition = config.experiment_condition.model_dump(mode="json")
        condition.pop("condition_id", None)
        if config.experiment_condition.provider_capability is not None:
            capability = config.experiment_condition.provider_capability.model_dump(mode="json")
            for key in ("checksum", "source_payload_sha256", "source_retrieved_utc"):
                capability.pop(key, None)
            condition["provider_capability"] = capability
        payload["experiment_condition"] = condition
        return payload

    if scientific(earlier) != scientific(later):
        raise ValueError("Earlier completion differs in scientific condition or harness behavior")
    return earlier, item


def _grade_view(source: Path, destination: Path, tasks: frozenset[str]) -> Path:
    """Publish a read-only derived grade source containing exactly the selected rows."""
    destination.mkdir(parents=True)
    manifest = read_json_model(GradeManifestV1, source / "GRADE_MANIFEST.json")
    if manifest.scorer != current_scorer():
        raise ValueError("Earlier-completion grades require the current matched scorer")
    for name in ("run_config.json", "replay_executor.json"):
        os.link(source / name, destination / name)
    for task in sorted(tasks):
        for name in ("items", "verification_inputs"):
            (destination / name).mkdir(exist_ok=True)
            os.link(source / name / f"{task}.json", destination / name / f"{task}.json")
        for path in (source / "logs").glob(f"{task}_*"):
            target = destination / "logs" / path.name
            target.parent.mkdir(exist_ok=True)
            if path.is_dir():
                shutil.copytree(path, target, copy_function=os.link)
            else:
                os.link(path, target)
    write_json_model(
        destination / "GRADE_MANIFEST.json",
        manifest.model_copy(
            update={
                "graded_task_count": len(tasks),
                "notes": (*manifest.notes, "Derived earliest-completion selection; original grades retained."),
            }
        ),
    )
    return destination


def select_trialeval_primary_sources(
    root: Path,
    combined: TrialEvalCampaignPlanV1,
    runs: tuple[CampaignRunV1, ...],
    *,
    views: Path,
    evaluator: Path,
    evaluator_sha256: str,
) -> tuple[TrialEvalCampaignPlanV1, tuple[Path, ...], tuple[Path, ...]]:
    """Grade audited earlier completions and bind one primary-or-retry per scheduled task."""
    grades = tuple(root / "results" / run.job_id / "grade" for run in runs)
    natives = tuple(root / run.relative_run_root for run in runs)
    path = root / PRIMARY_SELECTION
    if not path.exists():
        return combined, grades, natives
    selection = read_json_model(TrialEvalPrimarySelectionV1, path)
    if sha256_file(selection.audit_path) != selection.audit_sha256:
        raise ValueError("Earlier-completion origin audit changed")
    by_job = {run.job_id: run for run in runs}
    selected: dict[str, dict[str, TrialEvalRunConfigV1]] = {}
    origins: dict[tuple[str, Path], list[str]] = {}
    for row in selection.selections:
        run = by_job.get(row.target_job_id)
        if run is None or run.retry_of_job_id is not None or run.repetition != 1:
            raise ValueError("Earlier-completion selection must name a scheduled primary job")
        config, _ = validate_earlier_completion(row, root / run.relative_run_root)
        selected.setdefault(run.job_id, {})[row.task_id] = config
        origins.setdefault((run.job_id, row.source_run_root), []).append(row.task_id)
    from trialagentbench.tools.grade.grade_trialeval import grade_trialeval_run

    first_grades = []
    for index, ((job, source), tasks) in enumerate(sorted(origins.items())):
        output = root / "results" / f"first-completion-{sha256_file(path)[:16]}-{index}" / "grade"
        if not output.exists():
            target = root / by_job[job].relative_run_root
            image = read_json_model(TrialEvalRunConfigV1, target / "run_config.json").executor.image_id
            grade_trialeval_run(
                [
                    str(source),
                    "--evaluator-release",
                    str(evaluator),
                    "--expected-sha256",
                    evaluator_sha256,
                    "--out-dir",
                    str(output),
                    "--replay-executor-image-id",
                    image,
                    *(value for task in sorted(tasks) for value in ("--task-id", task)),
                ]
            )
        if read_json_model(GradeManifestV1, output / "GRADE_MANIFEST.json").scorer != current_scorer():
            raise ValueError("Earlier-completion grades have a different scorer")
        first_grades.append(output)
    filtered = []
    for run, grade in zip(runs, grades, strict=True):
        excluded = selected.get(run.retry_of_job_id or run.job_id, {})
        retained_tasks = frozenset(file.stem for file in (grade / "items").glob("*.json")) - excluded.keys()
        if retained_tasks:
            filtered.append(_grade_view(grade, views / run.job_id, frozenset(retained_tasks)) if excluded else grade)
    replacements = {(config.model, task): config for tasks in selected.values() for task, config in tasks.items()}
    attempts = []
    for attempt in combined.attempts:
        replacement = replacements.get((attempt.model_id, attempt.task_id))
        if replacement is not None and attempt.attempt_role == "retry":
            continue
        attempts.append(
            attempt.model_copy(update={"run_identity_sha256": replacement.run_identity_sha256})
            if replacement is not None and attempt.attempt_role == "primary"
            else attempt
        )
    combined = TrialEvalCampaignPlanV1.model_validate(
        {**combined.model_dump(exclude={"checksum", "attempts"}), "attempts": attempts}
    )
    for row in selection.selections:
        validate_earlier_completion(row, root / by_job[row.target_job_id].relative_run_root)
    return combined, (*filtered, *first_grades), (*natives, *(source for _, source in origins))
