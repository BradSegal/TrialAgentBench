"""Docker-backed isolated Python and R execution for model-generated code."""

from __future__ import annotations

import base64
import binascii
import functools
import json
import multiprocessing
import multiprocessing.synchronize
import os
import queue
import re
import shutil
import signal
import stat
import subprocess
import tarfile
import tempfile
import threading
import time
import unicodedata
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Final, Literal, TextIO, cast

from pydantic import BaseModel, ConfigDict, ValidationError

from trialagentbench_verifier.execution_contracts import (
    AnalysisLanguage,
    CodeExecutionLimitsV1,
    CodeExecutionResultV1,
    ExecutorEnvironmentV1,
    ReplayInputArtifactV1,
)
from trialagentbench_verifier.io import sha256_file

DEFAULT_EXECUTOR_IMAGE: Final = "trialagentbench/executor:1.0.1"
_IMAGE_ID: Final = re.compile(r"sha256:[0-9a-f]{64}")
# Variables that locate and authenticate the Docker daemon. Every Docker CLI call gets
# exactly these, so run, exec, rm and inspect reach the same daemon and none sees
# provider credentials; containers receive only explicit --env values regardless.
_DOCKER_CLIENT_VARIABLES: Final = (
    "PATH",
    "HOME",
    "XDG_RUNTIME_DIR",
    "DOCKER_HOST",
    "DOCKER_CONTEXT",
    "DOCKER_CONFIG",
    "DOCKER_CERT_PATH",
    "DOCKER_TLS_VERIFY",
)


_VERIFIED_SUBSTITUTES: set[str] = set()


def _executor_substitute(image: str) -> str | None:
    """Return a local image standing in for an unavailable recorded image ID, if one is declared.

    ``TRIALAGENTBENCH_EXECUTOR_SUBSTITUTE=<recorded sha256 ID>=<local sha256 ID>`` lets an offline
    regrade replay retained analyses when the recorded image is not present locally. The substitute must
    reproduce the recorded semantic environment (``TRIALAGENTBENCH_REPLAY_EXECUTOR_ENVIRONMENT_SHA256``);
    ``resolve_executor_environment`` enforces that. Records keep the recorded ID; only Docker runs the
    substitute.
    """
    declared = os.environ.get("TRIALAGENTBENCH_EXECUTOR_SUBSTITUTE")
    if not declared:
        return None
    recorded, _, substitute = declared.partition("=")
    return substitute if image == recorded and substitute else None


def _docker_environment() -> dict[str, str]:
    """Return the one minimal environment used for every Docker CLI call."""
    return {name: os.environ[name] for name in _DOCKER_CLIENT_VARIABLES if name in os.environ}


class CodeExecutionSessionTerminated(RuntimeError):
    """The participant's execution session ended (for example its own code killed it); scratch is unavailable."""


class _WorkerResultV1(BaseModel):
    """Authenticated result record emitted once per request by the supervisor."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    id: str
    status: Literal["success", "execution_error", "timeout"]
    output: str
    truncated: bool
    reset_required: bool = False


# Docker control commands (inspect, exec, rm) run against the daemon, which can stall for tens of
# seconds on a loaded host. A wedged daemon still fails loudly when this expires.
DOCKER_CONTROL_TIMEOUT_SECONDS = 120.0

_SESSION_CAPACITY: multiprocessing.synchronize.BoundedSemaphore | None = None
_CAPACITY_SLOTS = 0
# Serializes exclusive sessions so two of them never each hold part of the slots.
_EXCLUSIVE_CAPACITY: multiprocessing.synchronize.Lock | None = None
# Offline grading queues for a slot without a deadline: waiting is not execution time.
_CAPACITY_QUEUES = False
_CAPACITY_OWNER = threading.Lock()


class ExecutorUnavailableError(RuntimeError):
    """No executor slot became free in time: an infrastructure limit, never a model outcome."""


@contextmanager
def executor_session_capacity(
    maximum: int | None = None, *, limits: CodeExecutionLimitsV1 | None = None, queue: bool = False
) -> Iterator[None]:
    """Bound simultaneous containers across all threads of a run or campaign.

    A nested scope, such as a native run inside a campaign, keeps the outer bound. With ``queue``
    (offline grading), sessions wait for a slot without a deadline and may run exclusively; live runs
    keep the execution timeout as their wait limit.
    """
    global _SESSION_CAPACITY, _CAPACITY_SLOTS, _EXCLUSIVE_CAPACITY, _CAPACITY_QUEUES
    if maximum is None:
        if limits is None:
            raise ValueError("Offline executor capacity requires declared memory limits")
        available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        maximum = min(len(os.sched_getaffinity(0)), max(1, available // (limits.memory_mb * 1024**2)))
    if maximum < 1:
        raise ValueError("Executor capacity must be positive")
    if not _CAPACITY_OWNER.acquire(blocking=False):
        yield
        return
    context = multiprocessing.get_context("fork")
    _SESSION_CAPACITY = context.BoundedSemaphore(maximum)
    _CAPACITY_SLOTS = maximum
    _EXCLUSIVE_CAPACITY = context.Lock()
    _CAPACITY_QUEUES = queue
    try:
        yield
    finally:
        _SESSION_CAPACITY = None
        _CAPACITY_SLOTS = 0
        _EXCLUSIVE_CAPACITY = None
        _CAPACITY_QUEUES = False
        _CAPACITY_OWNER.release()


def offline_executor_capacity() -> bool:
    """Return whether sessions run inside an offline grading scope (``queue=True``)."""
    return _CAPACITY_QUEUES


def analysis_entrypoint_code(name: str, language: AnalysisLanguage) -> str:
    """Execute a scratch entrypoint with normal local-module lookup.

    Parameters
    ----------
    name
        Basename of the submitted entrypoint.
    language
        Supported Python or R interpreter.

    Returns
    -------
    str
        Code executed only inside the isolated analysis session.

    Raises
    ------
    ValueError
        If the entrypoint is not a supported source basename.
    """
    if name != ("analysis.py" if language == "python" else "analysis.R"):
        raise ValueError("unsupported analysis entrypoint")
    if language == "r":
        return f"sys.source('scratch/{name}', envir=.GlobalEnv)"
    return (
        "from pathlib import Path\nimport sys\n"
        f"_program = Path('scratch/{name}')\n"
        "sys.path.insert(0, str(_program.parent.resolve()))\n"
        "exec(compile(_program.read_text(encoding='utf-8'), str(_program), 'exec'), "
        "{'__name__': '__main__', '__file__': str(_program.resolve())})"
    )


def analysis_project_files(program: Path) -> tuple[Path, ...]:
    """Select regular Python/R sources beside an analysis entrypoint.

    Parameters
    ----------
    program
        Explicit Python or R entrypoint in the submitted scratch project.

    Returns
    -------
    tuple of Path
        Ordered source files, including nested local modules. Generated tables,
        submitted result files and interpreter caches are not replay inputs.

    Raises
    ------
    ValueError
        If the entrypoint or a project source uses an unsafe filesystem boundary.
    """
    if (
        program.parent.is_symlink()
        or program.is_symlink()
        or not program.is_file()
        or program.suffix.lower() not in {".py", ".r"}
    ):
        raise ValueError("analysis entrypoint must be a regular Python or R source file")
    paths = tuple(sorted(program.parent.rglob("*")))
    if any(path.is_symlink() for path in paths):
        raise ValueError("analysis projects cannot contain symbolic links")
    return tuple(path for path in paths if path.is_file() and path.suffix.lower() in {".py", ".r"})


def analysis_project_artifacts(program: Path) -> tuple[ReplayInputArtifactV1, ...]:
    """Bind every local Python/R source file used by an analysis entrypoint.

    Parameters
    ----------
    program
        Selected entrypoint in the scratch project.

    Returns
    -------
    tuple of ReplayInputArtifactV1
        Stable relative source paths, byte lengths and hashes.

    Raises
    ------
    ValueError
        If a source crosses an unsafe filesystem boundary.
    """
    return tuple(
        ReplayInputArtifactV1(
            relative_path=(Path("scratch") / path.relative_to(program.parent)).as_posix(),
            sha256=sha256_file(path),
            byte_length=path.stat().st_size,
        )
        for path in analysis_project_files(program)
    )


def verified_replay_inputs(source: Path, evidence_artifacts: tuple[ReplayInputArtifactV1, ...]) -> tuple[Path, ...]:
    """Resolve bound evidence, rejecting ambiguous paths and changed bytes.

    Parameters
    ----------
    source
        Assigned evidence workspace.
    evidence_artifacts
        Ordered paths, lengths and hashes recorded for the reporting cutoff.

    Returns
    -------
    tuple of Path
        Verified input paths in inventory order.

    Raises
    ------
    ValueError
        If identities repeat, paths escape the workspace or bytes have changed.

    Examples
    --------
    ``verified_replay_inputs(workspace, context.evidence_artifacts)`` checks the
    same inventory before execution and replay. Copied bytes are checked again
    because a source can change between validation and copying.
    """
    names = [artifact.relative_path for artifact in evidence_artifacts]
    if len(names) != len(set(names)):
        raise ValueError("replay evidence paths must be unique")
    paths = []
    for artifact in evidence_artifacts:
        relative = PurePosixPath(artifact.relative_path)
        if (
            relative.is_absolute()
            or relative.as_posix() != artifact.relative_path
            or not relative.parts
            or ".." in relative.parts
            or "\\" in artifact.relative_path
            or relative.parts[0] in {"scratch", "analysis_sources"}
        ):
            raise ValueError("replay evidence must be a canonical path outside generated scratch records")
        original = source.joinpath(*relative.parts)
        if any(source.joinpath(*relative.parts[:index]).is_symlink() for index in range(1, len(relative.parts) + 1)):
            raise ValueError("replay evidence cannot use symbolic links")
        if not original.is_file() or not original.resolve().is_relative_to(source.resolve()):
            raise ValueError("replay evidence must remain inside its assigned workspace")
        if original.stat().st_size != artifact.byte_length or sha256_file(original) != artifact.sha256:
            raise ValueError("replay evidence differs from its committed snapshot")
        paths.append(original)
    return tuple(paths)


def fresh_replay_workspace(
    *,
    source: Path,
    program: Path,
    destination: Path,
    evidence_artifacts: tuple[ReplayInputArtifactV1, ...] | None = None,
) -> Path:
    """Stage evidence and Python/R source modules without submitted answers or generated scratch data.

    Parameters
    ----------
    source, program, destination
        Original workspace, declared entrypoint and new replay directory.
    evidence_artifacts
        When provided, stage only these runner-bound evidence files, verifying their
        paths and bytes. Later or undeclared workspace files are not replay inputs.
        The project may then come from a separately captured submission directory.

    Returns
    -------
    Path
        Fresh workspace containing only source files in its scratch project.

    Raises
    ------
    OSError
        If input copying fails or the destination already exists.
    """
    project_files = analysis_project_files(program)
    if evidence_artifacts is None and program.parent.resolve() != (source / "scratch").resolve():
        raise ValueError("analysis entrypoint must belong to the selected workspace scratch project")
    if evidence_artifacts is None:
        shutil.copytree(
            source,
            destination,
            symlinks=True,
            ignore=lambda directory, names: (
                [name for name in names if name in {"scratch", "analysis_sources"}]
                if Path(directory) == source
                else []
            ),
        )
    else:
        originals = verified_replay_inputs(source, evidence_artifacts)
        destination.mkdir(parents=True, exist_ok=False)
        for artifact, original in zip(evidence_artifacts, originals, strict=True):
            relative = PurePosixPath(artifact.relative_path)
            copied = destination.joinpath(*relative.parts)
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, copied)
            if copied.stat().st_size != artifact.byte_length or sha256_file(copied) != artifact.sha256:
                raise ValueError("replay evidence differs from its committed snapshot")
    scratch = destination / "scratch"
    scratch.mkdir(mode=0o700)
    for path in project_files:
        target = scratch / path.relative_to(program.parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return destination


REQUIRED_ANALYSIS_PACKAGES: Final = (
    "jsonschema",
    "lifelines",
    "matplotlib",
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "scipy",
    "statsmodels",
)
REQUIRED_R_ANALYSIS_PACKAGES: Final = (
    "base",
    "jsonlite",
    "mass",
    "matrix",
    "sandwich",
    "survival",
    "survey",
    "trialagentbenchr",
)
_RESULT_PREFIX: Final = "__TRIALAGENTBENCH_RESULT__"
# Participant Python runs in this persistent child interpreter. It holds only
# its capture pipe (fds 1 and 2) and a request/completion pipe pair; the
# protocol token and the supervisor's result channel never enter it. Its
# completion message is advisory: the supervisor owns the clock, the bounded
# output and the authenticated result line.
_PYTHON_RUNNER = r"""
import base64
import json
import os
import signal
import sys
import traceback

requests = os.fdopen(int(sys.argv.pop()), "r", encoding="utf-8")
completions = os.fdopen(int(sys.argv.pop()), "w", encoding="utf-8")

class ExecutionTimedOut(Exception):
    pass

def timeout_handler(_signum, _frame):
    raise ExecutionTimedOut()

def reap_descendants():
    pending = [os.getpid()]
    descendants = []
    while pending:
        parent = pending.pop()
        try:
            children = [int(value) for value in open(f"/proc/{parent}/task/{parent}/children").read().split()]
        except (FileNotFoundError, ProcessLookupError):
            children = []
        descendants.extend(children)
        pending.extend(children)
    for process_id in reversed(descendants):
        try:
            os.kill(process_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for process_id in descendants:
        try:
            os.waitpid(process_id, 0)
        except (ChildProcessError, ProcessLookupError):
            pass

namespace = {"__name__": "__main__"}
original_stdout = sys.stdout
original_stderr = sys.stderr
for raw in requests:
    try:
        request = json.loads(raw)
        request_id = request["id"]
        timeout_seconds = float(request["timeout_seconds"])
        code = base64.b64decode(request["code_b64"]).decode("utf-8")
    except (KeyError, TypeError, ValueError):
        continue
    status = "success"
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
        try:
            exec(compile(code, "<agent-code>", "exec"), namespace, namespace)
        except ExecutionTimedOut:
            status = "timeout"
        except (Exception, SystemExit, KeyboardInterrupt) as error:
            status = "execution_error"
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            try:
                traceback.print_exc()
            except Exception:
                # Rendering consults participant objects: Python 3.12 NameError suggestions
                # iterate frame locals, which patsy formula namespaces cannot. The analyst
                # error must remain an execution error, not end the interpreter.
                sys.stderr.write(
                    "Traceback (most recent call last):\n"
                    + "".join(traceback.format_tb(error.__traceback__))
                    + f"{type(error).__name__}: {error}\n"
                )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (AttributeError, OSError, ValueError):
            pass
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    reap_descendants()
    completions.write(json.dumps({"id": request_id, "status": status}) + "\n")
    completions.flush()
"""
_WORKER = r"""
import base64
import ctypes
import json
import os
import pathlib
import select
import shutil
import signal
import stat
import subprocess
import sys
import time
import unicodedata

# The supervisor is PID 1 of the container, so participant processes cannot
# signal it. Without dumpability, same-UID processes also cannot read its
# memory, environment or file descriptors through /proc or ptrace.
if ctypes.CDLL(None, use_errno=True).prctl(4, 0, 0, 0, 0) != 0:
    raise RuntimeError("Executor supervisor could not disable process inspection")
result_prefix = "__TRIALAGENTBENCH_RESULT__" + sys.stdin.readline().strip() + ":"
PYTHON_RUNNER = __PYTHON_RUNNER__

def restore_scratch():
    source_root = pathlib.Path("/scratch-seed")
    destination_root = pathlib.Path("/workspace/scratch")
    destination_root.mkdir(parents=True, exist_ok=True)
    for source in source_root.rglob("*"):
        if source.is_symlink():
            raise RuntimeError(f"Scratch checkpoint contains a forbidden symlink: {source}")
        relative = source.relative_to(source_root)
        destination = destination_root / relative
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        source_info = source.stat(follow_symlinks=False)
        if not stat.S_ISREG(source_info.st_mode) or source_info.st_nlink != 1:
            raise RuntimeError(f"Scratch checkpoint contains a forbidden file: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

projected_evidence_roots = set()

def remove_projected_evidence(path):
    if not path.exists() and not path.is_symlink():
        return
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
        raise RuntimeError(f"Projected evidence contains a forbidden linked or special entry: {path}")
    if stat.S_ISREG(info.st_mode):
        if info.st_nlink != 1:
            raise RuntimeError(f"Projected evidence contains a forbidden hard link: {path}")
        path.unlink()
        return
    for entry in path.rglob("*"):
        entry_info = entry.lstat()
        if stat.S_ISLNK(entry_info.st_mode):
            raise RuntimeError(f"Projected evidence contains a forbidden symlink: {entry}")
        if stat.S_ISDIR(entry_info.st_mode):
            continue
        if not stat.S_ISREG(entry_info.st_mode) or entry_info.st_nlink != 1:
            raise RuntimeError(f"Projected evidence contains a forbidden linked or special entry: {entry}")
    shutil.rmtree(path)

def sync_evidence():
    global projected_evidence_roots
    source_root = pathlib.Path("/evidence")
    destination_root = pathlib.Path("/workspace")
    maximum_bytes = int(os.environ["TRIALAGENTBENCH_EVIDENCE_BYTES"])
    maximum_files = int(os.environ["TRIALAGENTBENCH_EVIDENCE_FILE_COUNT"])
    total_bytes = 0
    file_count = 0
    sources = []
    source_roots = set()
    reserved_roots = {"scratch", ".cache", ".config", ".trialagentbench-r-session.RData"}
    for source in sorted(source_root.rglob("*"), key=lambda item: item.as_posix()):
        relative = source.relative_to(source_root)
        if relative.parts and relative.parts[0] == "scratch":
            continue
        if relative.parts and relative.parts[0] in reserved_roots:
            raise RuntimeError(f"Participant evidence uses reserved runtime path: {relative.parts[0]}")
        if source.is_symlink():
            raise RuntimeError(f"Participant evidence contains a forbidden symlink: {source}")
        if relative.parts:
            source_roots.add(relative.parts[0])
        if source.is_dir():
            sources.append((source, relative, True))
            continue
        source_info = source.stat(follow_symlinks=False)
        if not stat.S_ISREG(source_info.st_mode):
            raise RuntimeError(f"Participant evidence contains a forbidden special file: {source}")
        if source_info.st_nlink != 1:
            raise RuntimeError(f"Participant evidence contains a forbidden hard link: {source}")
        file_count += 1
        total_bytes += source_info.st_size
        if file_count > maximum_files:
            raise RuntimeError("Participant evidence exceeds the configured file-count limit")
        if total_bytes > maximum_bytes:
            raise RuntimeError("Participant evidence exceeds the configured byte limit")
        sources.append((source, relative, False))

    for root_name in sorted(projected_evidence_roots | source_roots):
        remove_projected_evidence(destination_root / root_name)
    for source, relative, is_directory in sources:
        destination = destination_root / relative
        if is_directory:
            destination.mkdir(parents=True, exist_ok=False)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    projected_evidence_roots = source_roots

def validate_live_scratch():
    scratch_root = pathlib.Path("/workspace/scratch")
    maximum_bytes = int(os.environ["TRIALAGENTBENCH_WORKSPACE_BYTES"])
    maximum_entries = int(os.environ["TRIALAGENTBENCH_WORKSPACE_FILE_COUNT"])
    total_bytes = 0
    entry_count = 0
    # The host checkpoint rejects normalization collisions; end the session here instead.
    normalized_paths = set()
    for path in scratch_root.rglob("*"):
        entry_count += 1
        if entry_count > maximum_entries:
            raise RuntimeError("Agent scratch workspace exceeds the configured file-count limit")
        normalized = unicodedata.normalize("NFC", path.relative_to(scratch_root).as_posix())
        if normalized in normalized_paths:
            raise RuntimeError(f"Agent scratch workspace repeats a Unicode-normalized path: {path}")
        normalized_paths.add(normalized)
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeError(f"Agent scratch workspace contains a forbidden symlink: {path}")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError(f"Agent scratch workspace contains a forbidden linked or special file: {path}")
        total_bytes += info.st_size
        if total_bytes > maximum_bytes:
            raise RuntimeError("Agent scratch workspace exceeds the configured byte limit")

def unreaped_by(entry, parents):
    # A zombie whose parent is a retained process cannot be reaped here.
    try:
        fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return fields[0] == "Z" and int(fields[1]) in parents

def terminate_other_processes(retained):
    # As PID 1 the supervisor inherits every orphan; kill and reap all of them.
    # Returns whether killed processes remain as zombies of a retained process.
    victims = []
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in {os.getpid(), *retained}:
            continue
        try:
            os.kill(int(entry.name), signal.SIGKILL)
        except (PermissionError, ProcessLookupError):
            continue
        victims.append(entry)
    deadline = time.monotonic() + 5.0
    while victims and time.monotonic() < deadline:
        try:
            while os.waitpid(-1, os.WNOHANG)[0] > 0:
                pass
        except ChildProcessError:
            pass
        victims = [entry for entry in victims if entry.exists() and not unreaped_by(entry, retained)]
        if victims:
            time.sleep(0.01)
    return any(unreaped_by(entry, retained) for entry in pathlib.Path("/proc").iterdir() if entry.name.isdigit())

class BoundedCapture:
    def __init__(self, limit):
        self.limit = limit
        self.captured = bytearray()
        self.truncated = False

    def read(self, descriptor):
        try:
            chunk = os.read(descriptor, 65536)
        except BlockingIOError:
            return None
        remaining = self.limit - len(self.captured)
        if remaining > 0:
            self.captured.extend(chunk[:remaining])
        if len(chunk) > max(remaining, 0):
            self.truncated = True
        return chunk

def discard_pending(descriptor):
    try:
        while os.read(descriptor, 65536):
            pass
    except BlockingIOError:
        pass

python_runtime = None

def start_python_runtime():
    request_read, request_write = os.pipe()
    completion_read, completion_write = os.pipe()
    output_read, output_write = os.pipe()
    child = subprocess.Popen(
        [sys.executable, "-u", "-c", PYTHON_RUNNER, str(completion_write), str(request_read)],
        cwd="/workspace",
        stdin=subprocess.DEVNULL,
        stdout=output_write,
        stderr=output_write,
        pass_fds=(request_read, completion_write),
    )
    for descriptor in (request_read, completion_write, output_write):
        os.close(descriptor)
    for descriptor in (completion_read, output_read):
        os.set_blocking(descriptor, False)
    return child, os.fdopen(request_write, "w", encoding="utf-8"), completion_read, output_read

def execute_python_captured(request_id, code, limit, timeout_seconds):
    global python_runtime
    if python_runtime is None:
        python_runtime = start_python_runtime()
    child, requests, completion_read, output_read = python_runtime
    discard_pending(completion_read)
    discard_pending(output_read)
    os.kill(child.pid, signal.SIGCONT)
    capture = BoundedCapture(limit)
    started = time.monotonic()
    requests.write(
        json.dumps(
            {
                "id": request_id,
                "code_b64": base64.b64encode(code.encode("utf-8")).decode("ascii"),
                "timeout_seconds": timeout_seconds,
            }
        )
        + "\n"
    )
    requests.flush()
    pending = b""
    status = None
    hard_timeout = False
    watched = [output_read, completion_read]
    while status is None and child.poll() is None:
        # The analyst can ignore SIGALRM or block in a native extension. Enforce
        # its deadline here too, even when the host harness has died. Give the
        # interpreter's soft timeout a short grace period to retain its state.
        if time.monotonic() - started >= timeout_seconds + 1.0:
            hard_timeout = True
            break
        readable = select.select(watched, [], [], 0.25)[0]
        if output_read in readable and capture.read(output_read) == b"":
            watched.remove(output_read)
        if completion_read not in readable:
            continue
        try:
            chunk = os.read(completion_read, 4096)
        except BlockingIOError:
            continue
        if not chunk:
            # A closed completion pipe normally means the interpreter is exiting;
            # wait briefly so its exit ends the session instead of a later write.
            try:
                child.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                status = "execution_error"
            break
        # Claims naming another request are ignored; a claim for this request
        # is the participant's own exception outcome, never its time budget.
        *lines, pending = (pending + chunk).split(b"\n")
        pending = pending[-4096:]
        for line in lines:
            try:
                claimed = json.loads(line)
            except ValueError:
                continue
            if isinstance(claimed, dict) and claimed.get("id") == request_id:
                status = claimed.get("status")
                if status not in {"success", "execution_error", "timeout"}:
                    status = "execution_error"
                break
    elapsed = time.monotonic() - started
    if child.poll() is not None:
        raise RuntimeError("The Python analysis interpreter exited; in-memory variables were lost.")
    os.kill(child.pid, signal.SIGSTOP)
    lingering = terminate_other_processes({child.pid})
    while capture.read(output_read):
        pass
    if lingering or hard_timeout:
        # Unreaped children of a stopped interpreter would hold the process
        # limit; replace the interpreter so PID 1 inherits and reaps them.
        child.kill()
        child.wait()
        requests.close()
        for descriptor in (completion_read, output_read):
            os.close(descriptor)
        python_runtime = None
        terminate_other_processes(set())
        status = "timeout" if hard_timeout else "execution_error"
        reason = "exceeded its hard timeout" if hard_timeout else "left unreaped processes"
        capture.captured.extend(f"\nThe Python session {reason} and was reset; scratch files were retained but in-memory variables were lost.".encode())
    if elapsed >= timeout_seconds:
        status = "timeout"
    output = capture.captured.decode("utf-8", errors="replace")
    if status == "timeout":
        output += f"\nExecution timed out after {timeout_seconds:g}s."
    return status, output, capture.truncated, hard_timeout

def execute_r_captured(code, limit, timeout_seconds):
    code_path = pathlib.Path("/tmp/trialagentbench-agent-code.R")
    runner_path = pathlib.Path("/tmp/trialagentbench-r-runner.R")
    output_path = pathlib.Path("/tmp/trialagentbench-r-output.txt")
    state_path = pathlib.Path("/workspace/.trialagentbench-r-session.RData")
    code_path.write_text(code, encoding="utf-8")
    runner_path.write_text(
        "local({\n"
        "state_path <- Sys.getenv('TRIALAGENTBENCH_R_STATE')\n"
        "code_path <- Sys.getenv('TRIALAGENTBENCH_R_CODE')\n"
        "initial_environment <- Sys.getenv()\n"
        "if (file.exists(state_path)) load(state_path, envir = .GlobalEnv)\n"
        "if (exists('.trialagentbench_session', envir = .GlobalEnv, inherits = FALSE)) {\n"
        "  saved <- get('.trialagentbench_session', envir = .GlobalEnv)\n"
        "  for (package_name in rev(saved$packages)) {\n"
        "    if (package_name == 'base') next\n"
        "    search_name <- paste0('package:', package_name)\n"
        "    if (search_name %in% search()) detach(search_name, character.only = TRUE)\n"
        "    suppressPackageStartupMessages(library(package_name, character.only = TRUE))\n"
        "  }\n"
        "  for (search_name in setdiff(grep('^package:', search(), value = TRUE),\n"
        "                              paste0('package:', saved$packages))) {\n"
        "    detach(search_name, character.only = TRUE)\n"
        "  }\n"
        "  removed_options <- setdiff(names(options()), names(saved$options))\n"
        "  cleared_options <- rep(list(NULL), length(removed_options))\n"
        "  names(cleared_options) <- removed_options\n"
        "  options(cleared_options)\n"
        "  options(saved$options)\n"
        "  Sys.unsetenv(saved$unset_environment)\n"
        "  if (length(saved$environment)) do.call(Sys.setenv, as.list(saved$environment))\n"
        "  setwd(saved$directory)\n"
        "}\n"
        "execution_status <- 0L\n"
        "tryCatch(\n"
        "  sys.source(code_path, envir = .GlobalEnv),\n"
        "  error = function(condition) {\n"
        "    execution_status <<- 70L\n"
        "    message('Error: ', conditionMessage(condition))\n"
        "  }\n"
        ")\n"
        "environment <- Sys.getenv()\n"
        "changed <- !names(environment) %in% names(initial_environment) | "
        "environment != initial_environment[names(environment)]\n"
        "assign('.trialagentbench_session', list(\n"
        "  packages = sub('^package:', '', grep('^package:', search(), value = TRUE)),\n"
        "  options = options(), directory = getwd(), environment = environment[changed],\n"
        "  unset_environment = setdiff(names(initial_environment), names(environment))\n"
        "), envir = .GlobalEnv)\n"
        "state_names <- ls(envir = .GlobalEnv, all.names = TRUE)\n"
        "next_state_path <- paste0(state_path, '.next')\n"
        "save(list = state_names, file = next_state_path, envir = .GlobalEnv, version = 3)\n"
        "if (!file.rename(next_state_path, state_path)) stop('Unable to save the R session state')\n"
        "quit(save = 'no', status = execution_status)\n"
        "})\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["TRIALAGENTBENCH_R_STATE"] = str(state_path)
    environment["TRIALAGENTBENCH_R_CODE"] = str(code_path)
    status = "success"
    with output_path.open("wb") as output_stream:
        process = subprocess.Popen(
            ["Rscript", "--vanilla", str(runner_path)],
            cwd="/workspace",
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=output_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            status = "timeout"
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        else:
            if process.returncode < 0:
                raise RuntimeError("The R analysis interpreter exited; in-memory variables were lost.")
            if process.returncode != 0:
                status = "execution_error"
    terminate_other_processes(set() if python_runtime is None else {python_runtime[0].pid})
    output_size = output_path.stat().st_size
    with output_path.open("rb") as output_stream:
        captured = output_stream.read(limit)
    truncated = output_size > limit
    if status == "timeout":
        captured += f"Execution timed out after {timeout_seconds:g}s.\n".encode()
    for path in (code_path, runner_path, output_path):
        path.unlink(missing_ok=True)
    return status, captured.decode("utf-8", errors="replace"), truncated

restore_scratch()
for raw in sys.stdin:
    request = json.loads(raw)
    request_id = request["id"]
    code = base64.b64decode(request["code_b64"]).decode("utf-8")
    sync_evidence()
    language = request["language"]
    if language == "python":
        status, output, truncated, reset_required = execute_python_captured(
            request_id,
            code,
            int(os.environ["TRIALAGENTBENCH_OUTPUT_BYTES"]),
            float(request["timeout_seconds"]),
        )
    elif language == "r":
        reset_required = False
        status, output, truncated = execute_r_captured(
            code,
            int(os.environ["TRIALAGENTBENCH_OUTPUT_BYTES"]),
            float(request["timeout_seconds"]),
        )
    else:
        raise RuntimeError(f"Unsupported analysis language: {language!r}")
    validate_live_scratch()
    payload = json.dumps(
        {
            "id": request_id,
            "status": status,
            "output": output,
            "truncated": truncated,
            "reset_required": reset_required,
        }
    )
    print(result_prefix + base64.b64encode(payload.encode()).decode(), flush=True)
""".replace("__PYTHON_RUNNER__", repr(_PYTHON_RUNNER))

_PARQUET_BRIDGE = "/opt/trialagentbench/parquet_to_csv.py"

_ENVIRONMENT_PROBE = r"""
import hashlib
import importlib.metadata
import json
import platform
import stat
import subprocess
from pathlib import Path

packages = sorted(
    (
        {"name": distribution.metadata["Name"], "version": distribution.version}
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    ),
    key=lambda package: package["name"].casefold(),
)
r_probe = subprocess.run(
    [
        "Rscript",
        "--vanilla",
        "-e",
        "cat(paste(R.version$major, R.version$minor, sep='.')); cat('\\n'); "
        "write.table(installed.packages()[,c('Package','Version')], row.names=FALSE, col.names=FALSE, sep='\\t', quote=FALSE)",
    ],
    check=True,
    capture_output=True,
    text=True,
)
r_lines = r_probe.stdout.splitlines()
r_packages = [
    {"name": line.split("\t", 1)[0], "version": line.split("\t", 1)[1]}
    for line in r_lines[1:]
    if "\t" in line
]
r_packages.sort(key=lambda package: package["name"].casefold())
system_probe = subprocess.run(
    ["dpkg-query", "-W", "-f=${binary:Package}\t${Version}\n"],
    check=True,
    capture_output=True,
    text=True,
)
system_packages = [
    {"name": line.split("\t", 1)[0], "version": line.split("\t", 1)[1]}
    for line in system_probe.stdout.splitlines()
    if "\t" in line
]
system_packages.sort(key=lambda package: package["name"].casefold())
support_files = [
    {
        "path": str(path),
        "mode": format(stat.S_IMODE(path.stat().st_mode), "04o"),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    for path in sorted(Path("/opt/trialagentbench").rglob("*"))
    if path.is_file()
]
cgroup_metrics = {
    "cpu": Path("/sys/fs/cgroup/cpu.stat").is_file(),
    "memory_peak": Path("/sys/fs/cgroup/memory.peak").is_file(),
    "pids_peak": Path("/sys/fs/cgroup/pids.peak").is_file(),
}
print(
    json.dumps(
        {
            "python_version": platform.python_version(),
            "operating_system": platform.system(),
            "architecture": platform.machine(),
            "packages": packages,
            "r_version": r_lines[0],
            "r_packages": r_packages,
            "system_packages": system_packages,
            "support_files": support_files,
            "cgroup_metrics": cgroup_metrics,
        },
        separators=(",", ":"),
    )
)
"""


def resolve_executor_environment(
    *,
    image: str | None = None,
    limits: CodeExecutionLimitsV1 | None = None,
) -> ExecutorEnvironmentV1:
    """Resolve and validate the immutable isolated analysis environment."""

    if shutil.which("docker") is None:
        raise RuntimeError("Docker is required for isolated code execution; no host-Python fallback is permitted.")
    image_reference = image or os.environ.get("TRIALAGENTBENCH_EXECUTOR_IMAGE", DEFAULT_EXECUTOR_IMAGE)
    substitute = _executor_substitute(image_reference)
    inspection = subprocess.run(
        ["docker", "image", "inspect", substitute or image_reference],
        stdin=subprocess.DEVNULL,
        env=_docker_environment(),
        capture_output=True,
        text=True,
        check=False,
        timeout=DOCKER_CONTROL_TIMEOUT_SECONDS,
    )
    if inspection.returncode != 0:
        detail = inspection.stderr.strip() or "image not found"
        raise RuntimeError(
            f"Isolated executor image {image_reference!r} is unavailable ({detail}). "
            "Build executor/Dockerfile or set TRIALAGENTBENCH_EXECUTOR_IMAGE."
        )
    inspection_payload = json.loads(inspection.stdout)
    if not isinstance(inspection_payload, list) or len(inspection_payload) != 1:
        raise RuntimeError("Docker returned an invalid executor image inspection payload.")
    image_document = inspection_payload[0]
    if not isinstance(image_document, dict):
        raise RuntimeError("Docker returned a non-object executor image inspection record.")
    image_id = image_document.get("Id")
    if not isinstance(image_id, str):
        raise RuntimeError("Docker executor image inspection omitted its immutable image ID.")
    docker_image_id = image_id
    if substitute is not None:
        image_id = image_reference

    probe_payload = _probe_executor_image(docker_image_id)
    if not isinstance(probe_payload, dict):
        raise RuntimeError("Executor environment probe returned a non-object payload.")
    if probe_payload.get("cgroup_metrics") != {"cpu": True, "memory_peak": True, "pids_peak": True}:
        raise RuntimeError("Executor runtime does not expose required CPU, memory, and process accounting.")
    environment = ExecutorEnvironmentV1.model_validate(
        {
            "image_reference": image_reference,
            "image_id": image_id,
            "python_version": probe_payload.get("python_version"),
            "operating_system": probe_payload.get("operating_system"),
            "architecture": probe_payload.get("architecture"),
            "packages": probe_payload.get("packages"),
            "r_version": probe_payload.get("r_version"),
            "r_packages": probe_payload.get("r_packages"),
            "system_packages": probe_payload.get("system_packages"),
            "support_files": probe_payload.get("support_files"),
            "limits": (limits or CodeExecutionLimitsV1()).model_dump(mode="json"),
        }
    )
    installed = {package.name.casefold() for package in environment.packages}
    missing = sorted(set(REQUIRED_ANALYSIS_PACKAGES) - installed)
    if missing:
        raise RuntimeError(f"Executor image is missing required analysis packages: {', '.join(missing)}")
    installed_r = {package.name.casefold() for package in environment.r_packages}
    missing_r = sorted(set(REQUIRED_R_ANALYSIS_PACKAGES) - installed_r)
    if missing_r:
        raise RuntimeError(f"Executor image is missing required R analysis packages: {', '.join(missing_r)}")
    # Participant code runs as a non-root user, so the R parquet bridge must be
    # world-readable; its mode is part of the attested semantic environment.
    bridge = next((item for item in environment.support_files if item.path == _PARQUET_BRIDGE), None)
    if bridge is None or not int(bridge.mode, 8) & 0o004:
        raise RuntimeError(f"Executor image must provide a world-readable {_PARQUET_BRIDGE}.")
    if substitute is not None and environment.semantic_sha256() != os.environ.get(
        "TRIALAGENTBENCH_REPLAY_EXECUTOR_ENVIRONMENT_SHA256"
    ):
        raise RuntimeError("Substitute executor image does not reproduce the recorded semantic environment.")
    return environment


@functools.cache
def _probe_executor_image(docker_image_id: str) -> object:
    """Read the immutable environment of one executor image, once per process."""
    probe_name = f"trialagentbench-probe-{uuid.uuid4().hex}"
    try:
        probe = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--name",
                probe_name,
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--pids-limit",
                "16",
                "--memory",
                "256m",
                "--cpus",
                "1",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,nodev,size=64m",
                docker_image_id,
                "python",
                "-c",
                _ENVIRONMENT_PROBE,
            ],
            stdin=subprocess.DEVNULL,
            env=_docker_environment(),
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    finally:
        subprocess.run(
            ["docker", "rm", "--force", probe_name],
            stdin=subprocess.DEVNULL,
            env=_docker_environment(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=DOCKER_CONTROL_TIMEOUT_SECONDS,
        )

    if probe.returncode != 0:
        detail = probe.stderr.strip() or "environment probe failed"
        raise RuntimeError(f"Executor environment probe failed: {detail}")
    payload: object = json.loads(probe.stdout)
    return payload


def resolve_trusted_executor_environment(
    *,
    image: str | None,
    expected_image_id: str,
    expected_environment_sha256: str,
    limits: CodeExecutionLimitsV1,
) -> ExecutorEnvironmentV1:
    """Resolve the executor and compare it with external release trust anchors."""

    if not expected_image_id.startswith("sha256:") or len(expected_image_id) != 71:
        raise ValueError("expected executor image ID must be sha256 plus 64 lowercase hexadecimal characters")
    if any(character not in "0123456789abcdef" for character in expected_image_id.removeprefix("sha256:")):
        raise ValueError("expected executor image ID must be sha256 plus 64 lowercase hexadecimal characters")
    if len(expected_environment_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_environment_sha256
    ):
        raise ValueError("expected executor environment SHA-256 must be 64 lowercase hexadecimal characters")
    environment = resolve_executor_environment(image=image, limits=limits)
    observed_environment_sha256 = environment.semantic_sha256()
    if environment.image_id != expected_image_id:
        raise ValueError("executor image differs from its externally trusted immutable ID")
    if observed_environment_sha256 != expected_environment_sha256:
        raise ValueError("executor environment differs from its externally trusted SHA-256")
    return environment


def _validate_evidence_tree(root: Path, limits: CodeExecutionLimitsV1) -> None:
    """Reject evidence trees that cannot be mounted and copied safely."""
    normalized_paths: set[str] = set()
    file_count = 0
    total_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda candidate: candidate.as_posix()):
        relative = path.relative_to(root).as_posix()
        if Path(relative).parts[0] == "scratch":
            continue
        normalized = unicodedata.normalize("NFC", relative)
        if normalized in normalized_paths:
            raise ValueError(f"Participant evidence has a Unicode-normalized path collision: {relative}")
        normalized_paths.add(normalized)
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"Participant evidence contains a forbidden symlink: {relative}")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"Participant evidence contains a forbidden special file: {relative}")
        if info.st_nlink != 1:
            raise ValueError(f"Participant evidence contains a forbidden hard link: {relative}")
        file_count += 1
        total_bytes += info.st_size
        if file_count > limits.evidence_file_count:
            raise ValueError(
                f"Participant evidence exceeds the configured file-count limit ({limits.evidence_file_count})"
            )
        maximum_bytes = limits.evidence_mb * 1024 * 1024
        if total_bytes > maximum_bytes:
            raise ValueError(f"Participant evidence exceeds the configured byte limit ({limits.evidence_mb} MiB)")


def _validate_scratch_tree(root: Path, limits: CodeExecutionLimitsV1) -> None:
    """Reject scratch checkpoints that cannot be restored safely."""

    total_bytes = 0
    entry_count = 0
    for path in root.rglob("*"):
        entry_count += 1
        if entry_count > limits.workspace_file_count:
            raise ValueError(
                "Agent scratch workspace exceeds the configured file-count limit " f"({limits.workspace_file_count})"
            )
        info = path.lstat()
        relative = path.relative_to(root).as_posix()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"Agent scratch workspace contains a forbidden symlink: {relative}")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f"Agent scratch workspace contains a forbidden linked or special file: {relative}")
        total_bytes += info.st_size
        if total_bytes > limits.workspace_mb * 1024 * 1024:
            raise ValueError(f"Agent scratch workspace exceeds the configured limit ({limits.workspace_mb} MiB)")


def _docker_command(
    *,
    workdir: Path,
    scratch_root: Path,
    image: str,
    limits: CodeExecutionLimitsV1,
    container_name: str,
) -> list[str]:
    uid = os.getuid() if hasattr(os, "getuid") else 65534
    gid = os.getgid() if hasattr(os, "getgid") else 65534
    return [
        "docker",
        "run",
        "--rm",
        "--interactive",
        "--name",
        container_name,
        "--network",
        "none",
        # Docker otherwise copies the host's nameservers and search domains into
        # /etc/resolv.conf, which participant code can read despite having no network.
        "--dns",
        "127.0.0.1",
        "--dns-search",
        ".",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        str(limits.process_limit),
        "--memory",
        f"{limits.memory_mb}m",
        # Without an equal swap limit Docker allows the same amount again as swap,
        # so the declared memory limit would not bound the container on hosts with swap.
        "--memory-swap",
        f"{limits.memory_mb}m",
        "--cpus",
        str(limits.cpu_count),
        "--user",
        f"{uid}:{gid}",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=256m",
        "--mount",
        f"type=bind,src={workdir},dst=/evidence,readonly",
        "--tmpfs",
        f"/workspace:rw,noexec,nosuid,nodev,size={limits.workspace_mb}m,mode=0770,uid={uid},gid={gid}",
        "--mount",
        f"type=bind,src={scratch_root},dst=/scratch-seed,readonly",
        "--workdir",
        "/workspace",
        "--env",
        "PYTHONHASHSEED=0",
        # `python scratch/analysis.py` resolves `interface` as the in-session entrypoint does.
        "--env",
        "PYTHONPATH=/workspace",
        "--env",
        "HOME=/workspace",
        "--env",
        "XDG_CACHE_HOME=/workspace/.cache",
        "--env",
        "MPLCONFIGDIR=/workspace/.config/matplotlib",
        "--env",
        "OPENBLAS_NUM_THREADS=1",
        "--env",
        "OMP_NUM_THREADS=1",
        "--env",
        "MKL_NUM_THREADS=1",
        "--env",
        "NUMEXPR_NUM_THREADS=1",
        "--env",
        "VECLIB_MAXIMUM_THREADS=1",
        "--env",
        f"TRIALAGENTBENCH_OUTPUT_BYTES={limits.output_bytes}",
        "--env",
        f"TRIALAGENTBENCH_WORKSPACE_BYTES={limits.workspace_mb * 1024 * 1024}",
        "--env",
        f"TRIALAGENTBENCH_WORKSPACE_FILE_COUNT={limits.workspace_file_count}",
        "--env",
        f"TRIALAGENTBENCH_EVIDENCE_BYTES={limits.evidence_mb * 1024 * 1024}",
        "--env",
        f"TRIALAGENTBENCH_EVIDENCE_FILE_COUNT={limits.evidence_file_count}",
        image,
        "python",
        "-u",
        "-c",
        _WORKER,
    ]


class DockerCodeSession:
    """Persistent Python/R session contained by a locked-down Docker runtime."""

    _capacity_gate: multiprocessing.synchronize.BoundedSemaphore | None = None
    # A held gate is one slot; an exclusive session holds them all.
    _capacity_slots = 1

    def __init__(
        self,
        cwd: str | Path | None = None,
        timeout: int = 120,
        *,
        image: str | None = None,
        limits: CodeExecutionLimitsV1 | None = None,
        exclusive: bool = False,
    ) -> None:
        """Start a contained session.

        ``exclusive`` holds every executor slot of an offline grading scope, so the session runs
        without concurrent replays; it is refused outside such a scope.
        """
        if exclusive and not _CAPACITY_QUEUES:
            raise ValueError("Exclusive executor sessions require an offline grading capacity scope")
        workdir = Path(cwd or Path.cwd()).resolve(strict=True)
        if not workdir.is_dir():
            raise NotADirectoryError(f"Code-execution workdir is not a directory: {workdir}")
        if any(character in str(workdir) for character in (",", "\n", "\r")):
            raise ValueError("Code-execution workdir contains characters unsupported by Docker mount syntax.")
        self._limits = limits or CodeExecutionLimitsV1(timeout_seconds=float(timeout))
        _validate_evidence_tree(workdir, self._limits)
        self._workdir = workdir
        self._scratch_root = workdir / "scratch"
        self._scratch_root.mkdir(mode=0o700, exist_ok=True)
        if self._scratch_root.is_symlink() or not self._scratch_root.is_dir():
            raise ValueError("Agent scratch workspace must be a regular directory.")
        _validate_scratch_tree(self._scratch_root, self._limits)
        if shutil.which("docker") is None:
            raise RuntimeError("Docker is required for isolated code execution; no host-Python fallback is permitted.")
        self._image = image or os.environ.get("TRIALAGENTBENCH_EXECUTOR_IMAGE", DEFAULT_EXECUTOR_IMAGE)
        self._closed = False
        self.image_id = self._resolve_image_id()
        # Docker runs a declared substitute; the session still reports the recorded image ID.
        self._docker_image_id = self.image_id
        if (substitute := _executor_substitute(self.image_id)) is not None:
            if self.image_id not in _VERIFIED_SUBSTITUTES:
                resolve_executor_environment(image=self.image_id, limits=self._limits)
                _VERIFIED_SUBSTITUTES.add(self.image_id)
            self._docker_image_id = substitute
        capacity = _SESSION_CAPACITY
        if capacity is not None and exclusive:
            # No other holder waits while holding a slot, so draining every slot cannot deadlock.
            with cast(multiprocessing.synchronize.Lock, _EXCLUSIVE_CAPACITY):
                for _ in range(_CAPACITY_SLOTS):
                    capacity.acquire()
            self._capacity_slots = _CAPACITY_SLOTS
        elif capacity is not None:
            if not capacity.acquire(timeout=None if _CAPACITY_QUEUES else self._limits.timeout_seconds):
                raise ExecutorUnavailableError("Executor capacity unavailable within the declared execution timeout")
        self._capacity_gate = capacity
        ready = False
        try:
            self._launch_runtime()
            self._prime_runtime()
            ready = True
        finally:
            if not ready:
                if hasattr(self, "_proc"):
                    self.close()
                else:
                    self._release_capacity()

    def _launch_runtime(self) -> None:
        """Start one isolated interpreter against the retained scratch mount."""

        self._container_name = f"trialagentbench-{uuid.uuid4().hex}"
        self._protocol_token = uuid.uuid4().hex
        self._result_prefix = f"{_RESULT_PREFIX}{self._protocol_token}:"
        self._request_index = 0
        self._cpu_usage_usec = 0
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._proc = self._start(self._workdir)
        if self._proc.stdout is None or self._proc.stdin is None:
            self.close()
            raise RuntimeError("Docker executor did not expose its protocol pipes.")
        # The token travels only on the supervisor's private stdin, never in
        # the container environment that participant processes can read.
        self._proc.stdin.write(self._protocol_token + "\n")
        self._proc.stdin.flush()
        self._reader = threading.Thread(
            target=self._read_stdout,
            args=(self._proc.stdout, self._lines),
            daemon=True,
        )
        self._reader.start()

    def _restart_after_hard_timeout(self) -> None:
        """Replace an unresponsive interpreter while retaining scratch files."""

        self._terminate_runtime(checkpoint_scratch=True)
        self._launch_runtime()
        self._prime_runtime()

    def _prime_runtime(self) -> None:
        """Load trusted libraries under a separate, bounded startup allowance."""

        for language, code in (
            ("python", "import numpy as np\nimport pandas as pd"),
            ("r", "library(survival); library(survey); library(trialagentbenchR)"),
        ):
            result = self._execute_result(
                code, language=cast(AnalysisLanguage, language), timeout_seconds=30.0, restart_on_timeout=False
            )
            if result.status != "success":
                self.close()
                raise RuntimeError(f"Executor image cannot import required analysis libraries: {result.output}")

    def _resolve_image_id(self) -> str:
        """Bind the session to one immutable image, even if its tag later moves."""
        if _executor_substitute(self._image) is not None:
            return self._image
        result = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", self._image],
            stdin=subprocess.DEVNULL,
            env=_docker_environment(),
            capture_output=True,
            text=True,
            check=False,
            timeout=DOCKER_CONTROL_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or "image not found"
            raise RuntimeError(
                f"Isolated executor image {self._image!r} is unavailable ({detail}). "
                "Build executor/Dockerfile or set TRIALAGENTBENCH_EXECUTOR_IMAGE."
            )
        image_id = result.stdout.strip()
        if not _IMAGE_ID.fullmatch(image_id):
            raise RuntimeError(f"Docker returned an invalid immutable image ID for {self._image!r}.")
        return image_id

    def _start(self, workdir: Path) -> subprocess.Popen[str]:
        command = _docker_command(
            workdir=workdir,
            scratch_root=self._scratch_root,
            image=self._docker_image_id,
            limits=self._limits,
            container_name=self._container_name,
        )
        return subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
            env=_docker_environment(),
        )

    @staticmethod
    def _read_stdout(stream: TextIO, lines: queue.Queue[str | None]) -> None:
        try:
            for line in stream:
                lines.put(line.rstrip("\n"))
        finally:
            lines.put(None)

    def execute_result(self, code: str, *, language: AnalysisLanguage = "python") -> CodeExecutionResultV1:
        """Execute one block and return its typed bounded result."""
        try:
            return self._execute_result(
                code, language=language, timeout_seconds=self._limits.timeout_seconds, restart_on_timeout=True
            )
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            self.close()
            raise

    def _execution_usage(self) -> tuple[float, int, int]:
        """Read kernel counters without trusting the participant interpreter.

        Docker supplies a separate pipe and a different UID for the immutable
        reader. No participant Python modules, environment changes, result
        messages or file descriptors enter this accounting boundary. CPU covers
        the interval since the previous snapshot, including accounting overhead
        and any background session work between calls.
        """
        participant_uid = os.getuid() if hasattr(os, "getuid") else 65534
        probe = subprocess.run(
            [
                "docker",
                "exec",
                "--user",
                "0:0" if participant_uid != 0 else "65534:65534",
                "--workdir",
                "/",
                self._container_name,
                "/bin/cat",
                "/sys/fs/cgroup/cpu.stat",
                "/sys/fs/cgroup/memory.peak",
                "/sys/fs/cgroup/pids.peak",
            ],
            stdin=subprocess.DEVNULL,
            env=_docker_environment(),
            capture_output=True,
            text=True,
            check=False,
            timeout=DOCKER_CONTROL_TIMEOUT_SECONDS,
        )
        if probe.returncode != 0:
            raise RuntimeError(f"Executor kernel accounting failed: {probe.stderr.strip()}")
        lines = probe.stdout.splitlines()
        cpu_values = [line.removeprefix("usage_usec ") for line in lines[:-2] if line.startswith("usage_usec ")]
        values = (*cpu_values, *lines[-2:])
        if (
            len(cpu_values) != 1
            or len(values) != 3
            or any(not value.isascii() or not value.isdecimal() for value in values)
        ):
            raise RuntimeError("Executor kernel accounting returned invalid counters")
        cpu_usage_usec, memory_peak, process_peak = (int(value) for value in values)
        if cpu_usage_usec < self._cpu_usage_usec or memory_peak == 0 or process_peak == 0:
            raise RuntimeError("Executor kernel accounting counters decreased or omitted session peaks")
        cpu_seconds = (cpu_usage_usec - self._cpu_usage_usec) / 1_000_000.0
        self._cpu_usage_usec = cpu_usage_usec
        return cpu_seconds, memory_peak, process_peak

    def _execution_usage_after_timeout(self) -> tuple[float | None, int | None, int | None]:
        """Retain a hard timeout when a wedged executor also refuses accounting."""
        try:
            return self._execution_usage()
        except (RuntimeError, subprocess.TimeoutExpired):
            return None, None, None

    def _execute_result(
        self, code: str, *, language: AnalysisLanguage, timeout_seconds: float, restart_on_timeout: bool
    ) -> CodeExecutionResultV1:
        """Apply the selected trusted-startup or analyst-code execution budget."""

        started = time.monotonic()
        if self._closed or self._proc.poll() is not None or self._proc.stdin is None:
            return CodeExecutionResultV1(
                status="session_terminated",
                language=language,
                output="Code execution session has terminated.",
                elapsed_seconds=time.monotonic() - started,
            )
        self._request_index += 1
        request_id = str(self._request_index)
        request = {
            "id": request_id,
            "code_b64": base64.b64encode(code.encode("utf-8")).decode("ascii"),
            "language": language,
            "timeout_seconds": timeout_seconds,
        }
        try:
            self._proc.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            self.close()
            return CodeExecutionResultV1(
                status="session_terminated",
                language=language,
                output="Code execution session pipe closed.",
                elapsed_seconds=time.monotonic() - started,
            )

        incidental: list[str] = []
        # The worker owns the analysis timeout and returns a typed
        # result. This outer margin is only a hard kill for an unresponsive
        # interpreter or native extension.
        deadline = started + timeout_seconds + 5.0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                # A wedged or process-saturated container may refuse the
                # accounting probe; the participant timeout stays a timeout.
                usage = self._execution_usage_after_timeout()
                cpu_seconds, memory_peak, process_peak = usage
                if restart_on_timeout:
                    self._restart_after_hard_timeout()
                else:
                    self.close()
                return CodeExecutionResultV1(
                    status="timeout",
                    language=language,
                    output=(
                        f"Execution timed out after {timeout_seconds:g}s. "
                        + (
                            "The code session was reset; scratch files were retained but in-memory variables were lost."
                            if restart_on_timeout
                            else "Trusted startup failed; the session was closed."
                        )
                    ),
                    elapsed_seconds=time.monotonic() - started,
                    cpu_seconds=cpu_seconds,
                    session_peak_memory_bytes=memory_peak,
                    session_peak_processes=process_peak,
                )
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                self.close()
                detail = "\n".join(incidental).strip()
                return CodeExecutionResultV1(
                    status="session_terminated",
                    language=language,
                    output=detail or "Code execution session terminated before returning a result.",
                    elapsed_seconds=time.monotonic() - started,
                )
            if not line.startswith(self._result_prefix):
                incidental.append(line)
                continue
            try:
                payload = _WorkerResultV1.model_validate_json(
                    base64.b64decode(line.removeprefix(self._result_prefix), validate=True)
                )
            except (binascii.Error, ValidationError):
                payload = _WorkerResultV1(
                    id=request_id,
                    status="execution_error",
                    output="The executor returned a malformed result record.",
                    truncated=False,
                )
            if payload.id != request_id:
                continue
            output = "\n".join((*incidental, payload.output)).strip()
            encoded = output.encode("utf-8")
            truncated = payload.truncated or len(encoded) > self._limits.output_bytes
            if truncated:
                output = encoded[: self._limits.output_bytes].decode("utf-8", errors="ignore")
                output += "\n[output truncated]"
            cpu_seconds, memory_peak, process_peak = (
                self._execution_usage_after_timeout() if payload.reset_required else self._execution_usage()
            )
            if payload.reset_required:
                if restart_on_timeout:
                    self._restart_after_hard_timeout()
                else:
                    self.close()
            return CodeExecutionResultV1(
                status=payload.status,
                language=language,
                output=output,
                output_truncated=truncated,
                elapsed_seconds=time.monotonic() - started,
                cpu_seconds=cpu_seconds,
                session_peak_memory_bytes=memory_peak,
                session_peak_processes=process_peak,
            )

    def execute(self, code: str, *, language: AnalysisLanguage = "python") -> str:
        """Execute code and return the agent-facing text response."""
        result = self.execute_result(code, language=language)
        if result.output:
            return result.output
        if result.status == "success":
            return "(code executed successfully; no stdout produced)"
        return f"[{result.status}]"

    def close(self) -> None:
        """Kill the container and its client process group."""
        if self._closed:
            return
        self._closed = True
        try:
            self._terminate_runtime(checkpoint_scratch=True)
        finally:
            self._release_capacity()

    def _release_capacity(self) -> None:
        if self._capacity_gate is not None:
            for _ in range(self._capacity_slots):
                self._capacity_gate.release()
            self._capacity_gate = None

    def snapshot_scratch(self) -> Path:
        """Persist the live tmpfs scratch tree without ending the session."""

        if self._closed or self._proc.poll() is not None:
            raise CodeExecutionSessionTerminated("Cannot snapshot a terminated code-execution session.")
        checkpoint = self._copy_scratch_checkpoint()
        if checkpoint is None:
            raise CodeExecutionSessionTerminated("Code-execution session ended before scratch could be snapshotted.")
        self._install_scratch_checkpoint(checkpoint)
        return self._scratch_root

    def _terminate_runtime(self, *, checkpoint_scratch: bool) -> None:
        """Terminate the current container and local client process."""

        try:
            scratch_checkpoint = self._copy_scratch_checkpoint() if checkpoint_scratch else None
        finally:
            try:
                subprocess.run(
                    ["docker", "rm", "--force", self._container_name],
                    stdin=subprocess.DEVNULL,
                    env=_docker_environment(),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=DOCKER_CONTROL_TIMEOUT_SECONDS,
                )
            finally:
                # Docker can be unreachable; its owned local client must still
                # die, be reaped and release every protocol pipe and capacity slot.
                try:
                    if self._proc.poll() is None:
                        self._kill_client_process()
                    try:
                        self._proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self._kill_client_process()
                        self._proc.wait(timeout=5)
                finally:
                    for stream in (self._proc.stdin, self._proc.stdout):
                        if stream is not None:
                            try:
                                stream.close()
                            except BrokenPipeError:
                                pass
        if scratch_checkpoint is not None:
            self._install_scratch_checkpoint(scratch_checkpoint)

    def _kill_client_process(self) -> None:
        """Terminate the executor process group, falling back to its owned child."""

        try:
            os.killpg(self._proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except PermissionError:
            if self._proc.poll() is None:
                self._proc.kill()

    def _install_scratch_checkpoint(self, checkpoint: Path) -> None:
        """Atomically replace the host scratch seed with a validated snapshot.

        Staging directories sit beside the mounted workdir, on the same filesystem but
        outside the evidence mount, so an interrupted swap never projects a stale copy.
        """

        staging_root = self._workdir.parent
        prefix = f".{self._workdir.name}-{self._scratch_root.name}.{uuid.uuid4().hex}"
        staged = staging_root / f"{prefix}.tmp"
        previous = staging_root / f"{prefix}.previous"
        try:
            shutil.copytree(checkpoint, staged)
            _validate_scratch_tree(staged, self._limits)
            self._scratch_root.rename(previous)
            try:
                staged.rename(self._scratch_root)
            except OSError:
                previous.rename(self._scratch_root)
                raise
            shutil.rmtree(previous)
        finally:
            shutil.rmtree(checkpoint, ignore_errors=True)
            shutil.rmtree(staged, ignore_errors=True)

    def _copy_scratch_checkpoint(self) -> Path | None:
        """Copy the bounded scratch tmpfs before terminating a live container."""

        if self._proc.poll() is not None:
            return None
        checkpoint = Path(tempfile.mkdtemp(prefix="trialagentbench-scratch-"))
        archive_path = checkpoint.with_suffix(".tar")
        archive_script = (
            "import sys, tarfile\n"
            "with tarfile.open(fileobj=sys.stdout.buffer, mode='w|') as archive:\n"
            "    archive.add('/workspace/scratch', arcname='.')\n"
        )
        try:
            with archive_path.open("wb") as archive_stream:
                # Isolated mode: the participant can write /workspace (the container's
                # working directory and PYTHONPATH), so modules there must not be imported.
                result = subprocess.run(
                    [
                        "docker",
                        "exec",
                        "--workdir",
                        "/",
                        self._container_name,
                        "python",
                        "-I",
                        "-c",
                        archive_script,
                    ],
                    stdin=subprocess.DEVNULL,
                    env=_docker_environment(),
                    stdout=archive_stream,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=max(30.0, min(300.0, float(self._limits.workspace_mb))),
                )
            if result.returncode != 0:
                detail = result.stderr.decode("utf-8", errors="replace").strip() or "scratch export failed"
                # A container that ended itself (for example after an OOM kill)
                # has no live scratch; the host seed is the last checkpoint.
                if "No such container" in detail or not self._container_running():
                    shutil.rmtree(checkpoint)
                    archive_path.unlink(missing_ok=True)
                    return None
                raise RuntimeError(f"Unable to checkpoint the agent scratch workspace: {detail}")
            self._extract_scratch_archive(archive_path=archive_path, checkpoint=checkpoint)
            _validate_scratch_tree(checkpoint, self._limits)
        except (OSError, RuntimeError, subprocess.TimeoutExpired, tarfile.TarError, ValueError):
            shutil.rmtree(checkpoint)
            archive_path.unlink(missing_ok=True)
            raise
        archive_path.unlink()
        return checkpoint

    def _container_running(self) -> bool:
        """Report whether Docker still runs this session's container."""
        inspection = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", self._container_name],
            stdin=subprocess.DEVNULL,
            env=_docker_environment(),
            capture_output=True,
            text=True,
            check=False,
            timeout=DOCKER_CONTROL_TIMEOUT_SECONDS,
        )
        return inspection.returncode == 0 and inspection.stdout.strip() == "true"

    def _extract_scratch_archive(self, *, archive_path: Path, checkpoint: Path) -> None:
        """Extract regular scratch files without trusting archive paths or links."""

        total_bytes = 0
        entry_count = 0
        member_names: set[str] = set()
        with tarfile.open(archive_path, mode="r:") as archive:
            for member in archive:
                relative = Path(member.name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Scratch archive contains an unsafe path: {member.name}")
                if relative.parts:
                    normalized = unicodedata.normalize("NFC", relative.as_posix())
                    if normalized in member_names:
                        raise ValueError(f"Scratch archive repeats a normalized path: {member.name}")
                    member_names.add(normalized)
                    entry_count += 1
                    if entry_count > self._limits.workspace_file_count:
                        raise ValueError(
                            "Agent scratch workspace exceeds the configured file-count limit "
                            f"({self._limits.workspace_file_count})"
                        )
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError(f"Scratch archive contains a linked or special file: {member.name}")
                total_bytes += member.size
                if total_bytes > self._limits.workspace_mb * 1024 * 1024:
                    raise ValueError(
                        f"Agent scratch workspace exceeds the configured limit ({self._limits.workspace_mb} MiB)"
                    )

        with tarfile.open(archive_path, mode="r:") as archive:
            for member in archive:
                relative = Path(member.name)
                if member.isdir():
                    (checkpoint / relative).mkdir(parents=True, exist_ok=True)
                    continue
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Scratch archive omitted file content: {member.name}")
                destination = checkpoint / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)

    def __enter__(self) -> DockerCodeSession:
        """Return the active Docker-backed code session."""

        return self

    def __exit__(self, *_: object) -> None:
        """Close the Docker-backed Python session on context exit."""

        self.close()


__all__ = [
    "DEFAULT_EXECUTOR_IMAGE",
    "DOCKER_CONTROL_TIMEOUT_SECONDS",
    "REQUIRED_ANALYSIS_PACKAGES",
    "REQUIRED_R_ANALYSIS_PACKAGES",
    "DockerCodeSession",
    "ExecutorUnavailableError",
    "resolve_executor_environment",
    "resolve_trusted_executor_environment",
]
