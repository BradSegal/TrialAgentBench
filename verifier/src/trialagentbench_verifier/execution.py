"""Fresh public replay through the standalone untrusted-code boundary."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Generator, Iterable
from contextlib import nullcontext
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from trialagentbench_verifier.contracts import ExecutionProofV1
from trialagentbench_verifier.execution_contracts import ReportingInterfaceV1
from trialagentbench_verifier.io import (
    VerificationInputError,
    canonical_sha256,
    materialized_release,
    read_json_object,
    release_archive,
)
from trialagentbench_verifier.numerical import NumericalReplay
from trialagentbench_verifier.proof import numerical_score_projection_sha256
from trialagentbench_verifier.release import (
    EvaluationReleaseError,
    _participant_identity,
    _verify_expected_input,
    _verify_participant_release,
)
from trialagentbench_verifier.trialeval_inputs import (
    canonical_trialeval_workspace_path,
    canonical_trialeval_workspace_paths,
    sanitize_trialeval_agent_json_text,
)

_PROGRAM_NAMES = ("analysis.py", "analysis.R")

if TYPE_CHECKING:
    from trialagentbench_verifier.execution_contracts import ExecutorEnvironmentV1


class _Model(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)


class _Material(_Model):
    relative_path: str = Field(min_length=1)


class _Task(_Model):
    task_id: str = Field(min_length=1)
    # TrialDev materials are relative to their world, worlds/<case_id>/.
    case_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]+$")
    available_materials: tuple[_Material, ...] = Field(min_length=1)
    stream: Literal["evidence_analysis", "portfolio_management"] | None = None
    target_ids: tuple[str, ...] = ()
    initial_target_ids: tuple[str, ...] = ()


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts or "\\" in value or path.as_posix() != value:
        raise EvaluationReleaseError(f"participant material path is not portable: {value}")
    return path


def _copy_regular(source: Path, destination: Path, *, source_root: Path) -> None:
    resolved_root = source_root.resolve(strict=True)
    try:
        relative = source.relative_to(source_root)
    except ValueError as error:
        raise EvaluationReleaseError(f"replay input escapes its authenticated root: {source}") from error
    current = source_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise EvaluationReleaseError(f"replay input path contains a symlink: {source}")
    if not source.is_file() or source.is_symlink() or not source.resolve(strict=True).is_relative_to(resolved_root):
        raise EvaluationReleaseError(f"required replay input is missing or unsafe: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Cache permissions belong to the source; replay owns a writable private copy.
    shutil.copyfile(source, destination)


def _task(root: Path, task_id: str) -> _Task:
    path = root / "tasks" / task_id / "task.json"
    if not path.is_file() or path.is_symlink():
        raise EvaluationReleaseError(f"participant release has no task {task_id!r}")
    task = _Task.model_validate(read_json_object(path))
    if task.task_id != task_id:
        raise EvaluationReleaseError("selected participant task has a different task identity")
    return task


def _stage_task_inputs(
    root: Path,
    destination: Path,
    *,
    stream: Literal["trialeval", "trialdev"],
    task: _Task,
    task_root: bool = False,
) -> None:
    """Stage a task's materials as the harness presents them.

    With ``task_root`` (a replay-bank dataset), ``root`` is the task's own directory, so release paths
    ``tasks/<task_id>/...`` are read below it.
    """
    destinations: set[PurePosixPath] = set()
    if stream == "trialdev" and task.case_id is None:
        raise EvaluationReleaseError(f"TrialDev task {task.task_id!r} does not name its world")
    material_root = root if stream == "trialeval" else root / "worlds" / str(task.case_id)
    for material in task.available_materials:
        relative = _safe_relative_path(material.relative_path)
        source = (
            root / relative.relative_to(PurePosixPath("tasks", task.task_id))
            if task_root
            else material_root / relative
        )
        projected_paths = (
            canonical_trialeval_workspace_paths(relative.as_posix(), task_id=task.task_id)
            if stream == "trialeval"
            else (relative.as_posix(),)
        )
        for projected_path in projected_paths:
            projected = PurePosixPath(projected_path)
            if projected in destinations:
                raise EvaluationReleaseError(f"participant materials collide in the replay workspace: {projected}")
            destinations.add(projected)
            _copy_regular(source, destination.joinpath(*projected.parts), source_root=root)
        if stream == "trialeval" and source.suffix.lower() == ".json":
            destination.joinpath(*projected.parts).write_text(
                sanitize_trialeval_agent_json_text(
                    archive_name=projected.as_posix(),
                    text=source.read_text(encoding="utf-8"),
                    available_item_members=frozenset(
                        canonical_trialeval_workspace_path(row.relative_path) for row in task.available_materials
                    ),
                ),
                encoding="utf-8",
            )
        if stream == "trialeval" and relative.name == "data_integrity_policy.md":
            policy_source = source.with_suffix(".json")
            policy_target = destination / "data_integrity_policy.json"
            _copy_regular(policy_source, policy_target, source_root=root)
            policy_target.write_text(
                sanitize_trialeval_agent_json_text(
                    archive_name=policy_target.name, text=policy_source.read_text(encoding="utf-8")
                ),
                encoding="utf-8",
            )


def _submission_root(submission: Path) -> Path:
    source = Path(submission)
    if not source.is_dir() or source.is_symlink():
        raise EvaluationReleaseError("submission must be a regular directory")
    nested = source / "scratch"
    if nested.is_dir() and not any((source / name).is_file() for name in _PROGRAM_NAMES):
        if nested.is_symlink():
            raise EvaluationReleaseError("submission scratch directory must not be a symlink")
        return nested
    return source


def _stage_submission(source: Path, destination: Path) -> Literal["python", "r"] | None:
    """Stage the same minimal numerical project for either benchmark."""
    from trialagentbench_verifier.executor import analysis_project_files

    scratch = destination / "scratch"
    scratch.mkdir(mode=0o700)
    programs = tuple(name for name in _PROGRAM_NAMES if (source / name).is_file() or (source / name).is_symlink())
    if programs:
        for module in analysis_project_files(source / programs[0]):
            _copy_regular(module, scratch / module.relative_to(source), source_root=source)
    result = source / "estimates.csv"
    if result.exists() or result.is_symlink():
        _copy_regular(result, scratch / result.name, source_root=source)
    programs = tuple(name for name in _PROGRAM_NAMES if (scratch / name).is_file())
    return "python" if programs == ("analysis.py",) else "r" if programs == ("analysis.R",) else None


def _stage_result_schema(destination: Path) -> None:
    """Expose only the common numerical table header."""
    path = destination / "interface" / "estimates.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("target_id,estimate,lower,upper\n", encoding="utf-8")


def _proof(
    *,
    replay: NumericalReplay,
    stream: Literal["trialeval", "trialdev"],
    task_id: str,
    input_kind: Literal["archive", "directory"],
    input_sha256: str,
    expected_sha256: str,
    release_digest: str,
    candidate_digest: str,
    environment: ExecutorEnvironmentV1,
    expected_executor_image_id: str,
    expected_executor_environment_sha256: str,
    language: Literal["python", "r"] | None,
    score_projection_sha256: str | None,
    program_result_sha256: str | None,
) -> ExecutionProofV1:
    replay_status = replay.status
    replayed_hashes = replay.replayed_csv_sha256s
    execution_status = None if not replay.executions else replay.executions[-1].status
    source = next(
        (row for row in replay.program_artifacts if row.relative_path == f"scratch/{replay.entrypoint}"), None
    )
    program_sha256 = None if source is None else source.sha256
    elapsed_seconds = sum(row.elapsed_seconds for row in replay.executions)
    cpu_values = tuple(row.cpu_seconds for row in replay.executions)
    cpu_seconds = (
        sum(value for value in cpu_values if value is not None)
        if cpu_values and all(value is not None for value in cpu_values)
        else None
    )
    memory = max(
        (row.session_peak_memory_bytes for row in replay.executions if row.session_peak_memory_bytes is not None),
        default=None,
    )
    processes = max(
        (row.session_peak_processes for row in replay.executions if row.session_peak_processes is not None),
        default=None,
    )
    mismatch_paths: tuple[str, ...] = ("estimates.csv",) if replay.status == "result_mismatch" else ()
    attempt_count = len(replay.executions)
    classification: Literal["exact", "within_tolerance", "irreproducible"] | None = None
    if replay_status in {"reproduced", "within_tolerance", "result_mismatch"}:
        classification = cast(
            Literal["exact", "within_tolerance", "irreproducible"],
            {
                "reproduced": "exact",
                "within_tolerance": "within_tolerance",
                "result_mismatch": "irreproducible",
            }[replay_status],
        )
    succeeded = replay_status in {"reproduced", "within_tolerance"}
    return ExecutionProofV1(
        status="pass" if succeeded else "fail",
        input_kind=input_kind,
        input_sha256=input_sha256,
        expected_input_sha256=expected_sha256,
        release_digest=release_digest,
        candidate_digest=candidate_digest,
        benchmark_stream=stream,
        task_id=task_id,
        language=language,
        program_sha256=program_sha256,
        program_artifacts=replay.program_artifacts,
        result_format="csv",
        target_ids=replay.target_ids,
        evidence_artifacts=replay.evidence_artifacts,
        program_result_sha256=program_result_sha256,
        score_projection_sha256=score_projection_sha256,
        replayed_result_sha256s=replayed_hashes,
        replay_status=replay_status,
        execution_status=execution_status,
        comparison_classification=classification,
        attempt_count=attempt_count,
        elapsed_seconds=elapsed_seconds,
        cpu_seconds=cpu_seconds,
        session_peak_memory_bytes=memory,
        session_peak_processes=processes,
        executor_image_reference=environment.image_reference,
        executor_image_id=environment.image_id,
        expected_executor_image_id=expected_executor_image_id,
        executor_environment_sha256=environment.semantic_sha256(),
        expected_executor_environment_sha256=expected_executor_environment_sha256,
        mismatch_paths=mismatch_paths,
        failure_code=None if succeeded else replay_status,
    )


class ExecutionSubmissionV1(BaseModel):
    """One task and submitted workspace for an independently isolated replay."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str = Field(min_length=1)
    submission: Path


def execute_submissions(
    release: Path,
    submissions: Iterable[ExecutionSubmissionV1],
    *,
    expected_sha256: str,
    expected_executor_image_id: str,
    expected_executor_environment_sha256: str,
    executor_image: str | None = None,
    reporting_interface: ReportingInterfaceV1 = "v1",
) -> Generator[ExecutionProofV1, None, None]:
    """Replay each submission twice using one authenticated archive and executor.

    Consume or close the iterator to release the private archive extraction.
    Every item still receives fresh isolated workspaces and its own proof.
    ``reporting_interface`` selects the run condition's reporting helpers.
    """

    input_kind, input_sha256 = _verify_expected_input(release, expected_sha256)
    if input_sha256 != expected_sha256:
        raise EvaluationReleaseError("participant input differs from its external trust anchor")
    selected_tasks = None
    if release.name.endswith(".assets.json"):
        submissions = tuple(submissions)
        selected_tasks = tuple(dict.fromkeys(request.task_id for request in submissions))
    with (
        (
            release_archive(release, expected_sha256=expected_sha256)
            if input_kind == "archive"
            else nullcontext(release)
        ) as authenticated,
        materialized_release(authenticated, task_ids=selected_tasks) as root,
    ):
        if (root / "participant").is_dir() or not (root / "manifest.json").is_file():
            raise EvaluationReleaseError("execution accepts only a participant release, never evaluator files")
        stream, candidate, release_digest = _participant_identity(root)
        _verify_participant_release(root, stream=stream, candidate=candidate)
        from trialagentbench_verifier.execution_contracts import RELEASE_EXECUTION_LIMITS_V1
        from trialagentbench_verifier.executor import resolve_executor_environment

        limits = RELEASE_EXECUTION_LIMITS_V1
        environment = resolve_executor_environment(image=executor_image, limits=limits)
        environment_sha256 = environment.semantic_sha256()
        if environment.image_id != expected_executor_image_id:
            raise VerificationInputError("executor image differs from its trusted expected immutable ID")
        if environment_sha256 != expected_executor_environment_sha256:
            raise VerificationInputError("executor environment differs from its trusted expected SHA-256")
        for request in submissions:
            task_id = request.task_id
            submission = request.submission
            task = _task(root, task_id)
            if stream == "trialdev" and (task.stream is None or not task.initial_target_ids):
                raise EvaluationReleaseError("TrialDev execution requires the current numerical task contract")
            if any(not identifier for identifier in task.initial_target_ids) or len(
                set(task.initial_target_ids)
            ) != len(task.initial_target_ids):
                raise EvaluationReleaseError("Numerical target identifiers must be nonempty and unique")
            with tempfile.TemporaryDirectory(prefix="trialagentbench-public-replay-") as temporary:
                workspace = Path(temporary) / "workspace"
                workspace.mkdir()
                _stage_task_inputs(root, workspace, stream=stream, task=task)
                _stage_result_schema(workspace)
                language = _stage_submission(_submission_root(submission), workspace)
                replay: NumericalReplay
                score_projection_sha256: str | None
                from trialagentbench_verifier.execution_contracts import ReplayInputArtifactV1
                from trialagentbench_verifier.io import sha256_file
                from trialagentbench_verifier.numerical import replay_numerical_program

                target_ids = task.target_ids if stream == "trialeval" else task.initial_target_ids
                if (
                    not target_ids
                    or any(not target for target in target_ids)
                    or len(set(target_ids)) != len(target_ids)
                ):
                    raise EvaluationReleaseError("Numerical target identifiers must be nonempty and unique")
                evidence = tuple(
                    ReplayInputArtifactV1(
                        relative_path=path.relative_to(workspace).as_posix(),
                        sha256=sha256_file(path),
                        byte_length=path.stat().st_size,
                    )
                    for path in sorted(workspace.rglob("*"))
                    if path.is_file() and path.relative_to(workspace).parts[0] not in {"scratch", "interface"}
                )
                if language is None:
                    numerical_replay = NumericalReplay(
                        status="invalid_program",
                        entrypoint="",
                        target_ids=target_ids,
                        evidence_artifacts=evidence,
                        error="Submit exactly one analysis.py or analysis.R entrypoint.",
                    )
                else:
                    numerical_replay = replay_numerical_program(
                        workspace=workspace,
                        entrypoint="analysis.py" if language == "python" else "analysis.R",
                        target_ids=target_ids,
                        evidence_artifacts=evidence,
                        executor_image=environment.image_id,
                        executor_limits=environment.limits,
                        reporting_interface=reporting_interface,
                    )
                replay = numerical_replay
                rows = numerical_replay.estimates or ()
                score_projection_sha256 = (
                    numerical_score_projection_sha256(
                        rows,
                        tuple(tuple(sorted(item.relative_path for item in evidence)) for _ in rows),
                    )
                    if stream == "trialeval" and rows
                    else canonical_sha256({"estimates": [row.model_dump(mode="json") for row in rows]})
                )
                program_result_sha256 = numerical_replay.submitted_csv_sha256
            yield _proof(
                replay=replay,
                stream=stream,
                task_id=task_id,
                input_kind=input_kind,
                input_sha256=input_sha256,
                expected_sha256=expected_sha256,
                release_digest=release_digest,
                candidate_digest=candidate,
                environment=environment,
                expected_executor_image_id=expected_executor_image_id,
                expected_executor_environment_sha256=expected_executor_environment_sha256,
                language=language,
                score_projection_sha256=score_projection_sha256,
                program_result_sha256=program_result_sha256,
            )


def execute_submission(
    release: Path,
    submission: Path,
    *,
    task_id: str,
    expected_sha256: str,
    expected_executor_image_id: str,
    expected_executor_environment_sha256: str,
    executor_image: str | None = None,
    reporting_interface: ReportingInterfaceV1 = "v1",
) -> ExecutionProofV1:
    """Replay one public submission twice from authenticated participant inputs."""

    proofs = execute_submissions(
        release,
        (ExecutionSubmissionV1(task_id=task_id, submission=submission),),
        expected_sha256=expected_sha256,
        expected_executor_image_id=expected_executor_image_id,
        expected_executor_environment_sha256=expected_executor_environment_sha256,
        executor_image=executor_image,
        reporting_interface=reporting_interface,
    )
    try:
        return next(proofs)
    finally:
        proofs.close()


__all__ = ["ExecutionSubmissionV1", "execute_submissions", "execute_submission"]
