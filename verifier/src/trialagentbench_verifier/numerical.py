"""Common target-bound estimate and uncertainty table for both benchmarks."""

import csv
import io
import logging
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_verifier.execution_contracts import (
    TRIALEVAL_REPLAY_ABS_TOLERANCE_V1,
    TRIALEVAL_REPLAY_REL_TOLERANCE_V1,
    AnalysisLanguage,
    AnalysisMode,
    CodeExecutionLimitsV1,
    CodeExecutionResultV1,
    ReplayInputArtifactV1,
    ReportingInterfaceV1,
    analysis_languages,
)
from trialagentbench_verifier.executor import (
    DockerCodeSession,
    analysis_entrypoint_code,
    analysis_project_artifacts,
    fresh_replay_workspace,
    offline_executor_capacity,
)
from trialagentbench_verifier.io import read_json_object, sha256_file

logger = logging.getLogger(__name__)


def stage_reporting_interface(
    workspace: Path, target_ids: tuple[str, ...], *, interface: ReportingInterfaceV1 = "v1"
) -> None:
    """Expose runner-owned target IDs and Python/R result writers without reference values.

    ``interface`` selects the run condition's helpers: ``"v1"`` writes the 1.0 helpers unchanged, and
    ``"clarified"`` also accepts a data frame or row mappings and names the expected columns on error.
    """
    _validate_target_ids(target_ids)
    directory = workspace / "interface"
    if directory.is_symlink():
        raise ValueError("Reporting interface cannot be a symbolic link")
    directory.mkdir(exist_ok=True)
    table = io.StringIO(newline="")
    writer = csv.writer(table, lineterminator="\n")
    writer.writerow(("target_id",))
    writer.writerows((target,) for target in target_ids)
    files = {
        "targets.csv": table.getvalue(),
        "reporting.py": (
            '"""Read public target IDs and write numerical results for this checkpoint."""\n'
            "import csv\n"
            "from pathlib import Path\n\n"
            "def targets():\n"
            "    with open('interface/targets.csv', newline='') as source:\n"
            "        return [row['target_id'] for row in csv.DictReader(source)]\n\n"
            "def report(rows):\n"
            "    # Rows for targets of an earlier checkpoint are left out; only current targets are written.\n"
            "    current = set(targets())\n"
            "    rows = [row for row in rows if row['target_id'] in current]\n"
            "    Path('scratch').mkdir(exist_ok=True)\n"
            "    with open('scratch/estimates.csv', 'w', newline='') as destination:\n"
            "        writer = csv.DictWriter(destination, fieldnames=['target_id','estimate','lower','upper'])\n"
            "        writer.writeheader()\n"
            "        writer.writerows(rows)\n"
        ),
        "reporting.R": (
            "targets <- function() read.csv('interface/targets.csv', stringsAsFactors=FALSE)$target_id\n"
            "report <- function(rows) {\n"
            "  dir.create('scratch', showWarnings=FALSE)\n"
            "  rows <- rows[rows$target_id %in% targets(), c('target_id','estimate','lower','upper')]\n"
            "  write.csv(rows, 'scratch/estimates.csv', row.names=FALSE)\n"
            "}\n"
        ),
    }
    if interface == "clarified":
        files.update(_CLARIFIED_HELPERS)
    for name, text in files.items():
        path = directory / name
        if path.is_symlink():
            raise ValueError("Reporting interface file cannot be a symbolic link")
        path.write_text(text, encoding="utf-8")


# Reporting helpers of the "clarified" interface. targets() still returns a list of ID strings; report() also
# takes a data frame, keeps only the four columns, and names them (and the targets) when a row lacks one.
_CLARIFIED_HELPERS = {
    "reporting.py": (
        '"""Read public target IDs and write numerical results for this checkpoint."""\n'
        "import csv\n"
        "from pathlib import Path\n\n"
        "COLUMNS = ['target_id', 'estimate', 'lower', 'upper']\n\n"
        "def targets():\n"
        '    """Return the target IDs, a list of strings, in reporting order."""\n'
        "    with open('interface/targets.csv', newline='') as source:\n"
        "        return [row['target_id'] for row in csv.DictReader(source)]\n\n"
        "def report(rows):\n"
        '    """Write one row per target from a data frame or a list of dicts with COLUMNS."""\n'
        "    if hasattr(rows, 'to_dict'):\n"
        "        rows = rows.to_dict('records')\n"
        "    rows = list(rows)\n"
        "    expected = targets()\n"
        "    for row in rows:\n"
        "        if not isinstance(row, dict) or any(key not in row for key in COLUMNS):\n"
        "            raise ValueError(\n"
        "                f'report() takes a data frame or a list of dicts, one row per target, with keys {COLUMNS}; '\n"
        "                f'the targets are {expected}; got {row!r}'\n"
        "            )\n"
        "    # Rows for targets of an earlier checkpoint are left out; only current targets are written.\n"
        "    current = set(expected)\n"
        "    rows = [{key: row[key] for key in COLUMNS} for row in rows if row['target_id'] in current]\n"
        "    Path('scratch').mkdir(exist_ok=True)\n"
        "    with open('scratch/estimates.csv', 'w', newline='') as destination:\n"
        "        writer = csv.DictWriter(destination, fieldnames=COLUMNS)\n"
        "        writer.writeheader()\n"
        "        writer.writerows(rows)\n"
    ),
    "reporting.R": (
        "targets <- function() read.csv('interface/targets.csv', stringsAsFactors=FALSE)$target_id\n"
        "report <- function(rows) {\n"
        "  columns <- c('target_id','estimate','lower','upper')\n"
        "  problem <- function(detail) stop(sprintf(\n"
        "    'report() takes a data frame or a list of lists, one row per target, with columns %s; %s; the targets are %s',\n"
        "    paste(columns, collapse=', '), detail, paste(targets(), collapse=', ')), call.=FALSE)\n"
        "  if (!is.data.frame(rows) && is.list(rows) && length(rows) > 0 && all(vapply(rows, is.list, logical(1)))) {\n"
        "    for (row in rows) if (!all(columns %in% names(row))) problem(sprintf('a row has only %s',\n"
        "      paste(names(row), collapse=', ')))\n"
        "    rows <- do.call(rbind, lapply(rows, function(row) data.frame(row[columns], stringsAsFactors=FALSE)))\n"
        "  }\n"
        "  if (!is.data.frame(rows)) problem(sprintf('got an object of class %s', class(rows)[1]))\n"
        "  missing <- setdiff(columns, names(rows))\n"
        "  if (length(missing) > 0) problem(sprintf('missing %s', paste(missing, collapse=', ')))\n"
        "  dir.create('scratch', showWarnings=FALSE)\n"
        "  rows <- rows[rows$target_id %in% targets(), columns]\n"
        "  write.csv(rows, 'scratch/estimates.csv', row.names=FALSE)\n"
        "}\n"
    ),
}


class NumericalEstimate(BaseModel):
    """One estimate and uncertainty interval in the task's declared units."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False, strict=True)
    target_id: str = Field(min_length=1)
    estimate: float
    lower: float
    upper: float

    @model_validator(mode="after")
    def ordered_interval(self) -> Self:
        """Reject reversed limits without forcing the estimate inside its interval."""
        if self.lower > self.upper:
            raise ValueError("uncertainty lower limit exceeds upper limit")
        return self


def _validate_target_ids(target_ids: tuple[str, ...]) -> None:
    """Reject inconsistent runner-owned target inventories before participant parsing."""
    if len(set(target_ids)) != len(target_ids) or any(not value for value in target_ids):
        raise ValueError("assigned numerical targets must be unique and nonempty")


class NumericalReportingContext(BaseModel):
    """Runner-owned scientific targets and evidence for one frozen report."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    checkpoint_id: str = Field(min_length=1)
    target_ids: tuple[str, ...]
    evidence_artifacts: tuple[ReplayInputArtifactV1, ...]

    @model_validator(mode="after")
    def unique_inventory(self) -> Self:
        """Reject ambiguous target or evidence identities before report capture."""
        _validate_target_ids(self.target_ids)
        paths = tuple(item.relative_path for item in self.evidence_artifacts)
        if len(set(paths)) != len(paths):
            raise ValueError("reporting evidence paths must be unique")
        return self


# Spellings of an unreported value written by pandas (blank) and R (``NA``).
_UNREPORTED_VALUES = frozenset({"", "NA"})


def read_numerical_table(path: Path, *, target_ids: tuple[str, ...]) -> tuple[NumericalEstimate, ...]:
    """Read target_id, estimate, lower and upper from a participant CSV export.

    Parameters
    ----------
    path
        Regular participant result file. Column order may vary.
    target_ids
        Unique target inventory assigned by the runner, in reporting order.

    Returns
    -------
    tuple of NumericalEstimate
        Present estimates in task order. Missing targets, and targets whose row has
        a blank, ``NA`` or non-finite number, remain absent so grading retains them
        as failures in the full task denominator; other rows are unaffected.

    Raises
    ------
    ValueError
        If the inventory, file, columns, numbers or target identities are invalid.
    csv.Error
        If CSV syntax is malformed.

    Examples
    --------
    A row may be ``primary_effect,-0.42,-0.71,-0.13``. Scientific definitions,
    units, confidence level and checkpoint identity belong to the task, not this file.
    """
    _validate_target_ids(target_ids)
    if path.is_symlink() or not path.is_file():
        raise ValueError("numerical export must be a regular non-symlinked CSV file")
    fields = {"target_id", "estimate", "lower", "upper"}
    rows: dict[str, NumericalEstimate] = {}
    unreported: set[str] = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        header = next(reader, [])
        if len(header) != len(fields) or set(header) != fields:
            raise ValueError("numerical export requires exactly target_id,estimate,lower,upper columns")
        positions = {name: header.index(name) for name in fields}
        for values in reader:
            if not values:
                continue
            if len(values) != len(header):
                raise ValueError("numerical export row has the wrong number of fields")
            target = values[positions["target_id"]]
            if target not in target_ids:
                raise ValueError(f"unknown numerical target: {target!r}")
            if target in rows or target in unreported:
                raise ValueError(f"duplicate numerical target: {target!r}")
            numbers = tuple(values[positions[name]].strip() for name in ("estimate", "lower", "upper"))
            if any(text in _UNREPORTED_VALUES for text in numbers) or not all(
                math.isfinite(float(text)) for text in numbers
            ):
                # A blank, NA or non-finite row leaves only its own target unreported.
                unreported.add(target)
                continue
            estimate, lower, upper = (float(text) for text in numbers)
            rows[target] = NumericalEstimate(target_id=target, estimate=estimate, lower=lower, upper=upper)
    return tuple(rows[target] for target in target_ids if target in rows)


class NumericalReplay(BaseModel):
    """Reproduction outcome for a common numerical export and its exact inputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    status: Literal[
        "reproduced",
        "within_tolerance",
        "invalid_program",
        "missing_result",
        "invalid_result",
        "execution_error",
        "result_mismatch",
    ]
    entrypoint: str
    checkpoint_id: str | None = Field(default=None, min_length=1)
    target_ids: tuple[str, ...]
    program_artifacts: tuple[ReplayInputArtifactV1, ...] = ()
    evidence_artifacts: tuple[ReplayInputArtifactV1, ...]
    submitted_csv_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    replayed_csv_sha256s: tuple[str, ...] = ()
    executions: tuple[CodeExecutionResultV1, ...] = ()
    estimates: tuple[NumericalEstimate, ...] | None = None
    error: str | None = None

    @model_validator(mode="after")
    def validate_reproduction(self) -> Self:
        """Require two successful executions before certifying numerical estimates."""
        _validate_target_ids(self.target_ids)
        if self.reproducible:
            if (
                self.estimates is None
                or self.submitted_csv_sha256 is None
                or len(self.replayed_csv_sha256s) != 2
                or len(self.executions) != 2
                or any(row.status != "success" for row in self.executions)
                or self.error is not None
            ):
                raise ValueError("reproduced estimates require two successful bound executions")
        elif self.estimates is not None:
            raise ValueError("failed reproduction cannot certify numerical estimates")
        return self

    @property
    def reproducible(self) -> bool:
        """Return whether both fresh executions recover the submitted numerical rows."""
        return self.status in {"reproduced", "within_tolerance"}


def replay_numerical_program(
    *,
    workspace: Path,
    entrypoint: Literal["analysis.py", "analysis.R"],
    target_ids: tuple[str, ...],
    evidence_artifacts: tuple[ReplayInputArtifactV1, ...],
    executor_limits: CodeExecutionLimitsV1,
    analysis_language: AnalysisMode = "python_r",
    executor_image: str | None = None,
    submitted_project: Path | None = None,
    capture_result: bool = False,
    reporting_interface: ReportingInterfaceV1 = "v1",
) -> NumericalReplay:
    """Reproduce a four-column CSV from an explicit project and frozen evidence.

    Parameters
    ----------
    workspace
        Participant workspace containing the submitted scratch project.
    entrypoint
        Explicit Python or R entrypoint; other local source files are helpers.
    target_ids
        Runner-assigned scientific target inventory, including unreported targets.
    evidence_artifacts
        Exact participant evidence available at the committed reporting cutoff.
        Files added later are never staged merely because they exist in workspace.
    executor_limits
        Per-execution resource limits used by the shared isolated executor. In an
        offline grading scope a run that times out is rerun once, alone on the
        executor, before the timeout counts against the submission.
    executor_image
        Executor image selected and bound by the run invocation.
    submitted_project
        Runner-owned captured source and CSV directory. When supplied, subsequent
        edits to workspace scratch are excluded from both submission and replay.
    capture_result
        For a live check or reporting cutoff, derive the result from two fresh
        runs of the saved script and never read the workspace's earlier
        scratch/estimates.csv; the first output is written there only after the
        second agrees. Historical grading leaves this false and verifies the
        original captured CSV without rewriting it.
    reporting_interface
        Reporting helpers the run condition staged; replays stage the same ones.

    Returns
    -------
    NumericalReplay
        Source/evidence identities, execution observations and reproduced estimates.
        Malformed or irreproducible submissions retain explicit failure outcomes.

    Raises
    ------
    ValueError
        If committed evidence or source identities change during replay.
    RuntimeError
        If the execution infrastructure cannot run the assigned analysis.

    Examples
    --------
    A project writes scratch/estimates.csv from its assigned raw records. Its output
    is excluded from both replay workspaces and must be recomputed each time.
    """
    _validate_target_ids(target_ids)
    project = workspace / "scratch" if submitted_project is None else submitted_project
    program = project / entrypoint
    result_path = project / "estimates.csv"
    sources: tuple[ReplayInputArtifactV1, ...] = ()
    submitted_hash = None
    hashes: list[str] = []
    executions: list[CodeExecutionResultV1] = []

    def outcome(
        status: Literal[
            "reproduced",
            "within_tolerance",
            "invalid_program",
            "missing_result",
            "invalid_result",
            "execution_error",
            "result_mismatch",
        ],
        *,
        error: str | None = None,
        estimates: tuple[NumericalEstimate, ...] | None = None,
    ) -> NumericalReplay:
        return NumericalReplay(
            status=status,
            entrypoint=entrypoint,
            target_ids=target_ids,
            program_artifacts=sources,
            evidence_artifacts=evidence_artifacts,
            submitted_csv_sha256=submitted_hash,
            replayed_csv_sha256s=tuple(hashes),
            executions=tuple(executions),
            estimates=estimates,
            error=error,
        )

    try:
        sources = analysis_project_artifacts(program)
    except ValueError as error:
        return outcome("invalid_program", error=str(error))
    submitted: tuple[NumericalEstimate, ...] | None = None
    captured_bytes: bytes | None = None
    if not capture_result:
        if not result_path.exists():
            return outcome("missing_result", error="scratch/estimates.csv is missing")
        try:
            submitted = read_numerical_table(result_path, target_ids=target_ids)
        except (ValueError, csv.Error, UnicodeDecodeError) as error:
            return outcome("invalid_result", error=str(error))
        submitted_hash = sha256_file(result_path)
    language: AnalysisLanguage = "python" if entrypoint == "analysis.py" else "r"
    if language not in analysis_languages(analysis_language):
        return NumericalReplay(
            status="invalid_program",
            entrypoint=entrypoint,
            target_ids=target_ids,
            evidence_artifacts=evidence_artifacts,
            error=f"Entrypoint is outside the {analysis_language} analysis condition",
        )

    def execute(destination: Path, *, exclusive: bool) -> tuple[Path, CodeExecutionResultV1]:
        root = fresh_replay_workspace(
            source=workspace,
            program=program,
            destination=destination,
            evidence_artifacts=evidence_artifacts,
        )
        if analysis_project_artifacts(root / "scratch" / entrypoint) != sources:
            raise ValueError("analysis sources differ from the committed project")
        stage_reporting_interface(root, target_ids, interface=reporting_interface)
        session = DockerCodeSession(root, image=executor_image, limits=executor_limits, exclusive=exclusive)
        try:
            execution = session.execute_result(analysis_entrypoint_code(entrypoint, language), language=language)
            session.snapshot_scratch()
        finally:
            session.close()
        return root, execution

    exact = True
    for _ in range(2):
        with TemporaryDirectory(prefix="numerical-replay-") as temporary:
            root, execution = execute(Path(temporary) / "workspace", exclusive=False)
            if execution.status == "timeout" and offline_executor_capacity():
                # Offline grading shares the host across parallel replays, so a timeout there can be
                # load rather than the script. Rerun once from a fresh workspace holding every slot:
                # only a script that cannot finish within the release limit alone still times out.
                logger.warning("Rerunning offline replay of scratch/%s exclusively after a timeout", entrypoint)
                root, execution = execute(Path(temporary) / "exclusive", exclusive=True)
            executions.append(execution)
            if execution.status != "success":
                return outcome(
                    "execution_error",
                    error=(
                        f"scratch/{entrypoint} {'timed out' if execution.status == 'timeout' else 'failed'} in a "
                        "fresh session, which has no interactive variables (see the execution's error output); "
                        "the script must load its own data and helpers"
                    ),
                )
            replayed_path = root / "scratch" / "estimates.csv"
            if not replayed_path.exists():
                return outcome(
                    "missing_result",
                    error=(
                        f"scratch/{entrypoint} ran but did not write scratch/estimates.csv; end it with "
                        "report(rows) from the interface reporting helper"
                    ),
                )
            try:
                observed = read_numerical_table(replayed_path, target_ids=target_ids)
            except (ValueError, csv.Error, UnicodeDecodeError) as error:
                return outcome(
                    "invalid_result",
                    error=(
                        f"scratch/estimates.csv written by scratch/{entrypoint} is invalid: {error}; report one "
                        "row per target in targets() (interface/targets.csv) with target_id, estimate, lower and upper"
                    ),
                )
            hashes.append(sha256_file(replayed_path))
            if submitted is None:
                submitted = observed
                captured_bytes = replayed_path.read_bytes()
                submitted_hash = hashes[-1]
            if tuple(row.target_id for row in observed) != tuple(row.target_id for row in submitted):
                return outcome(
                    "result_mismatch",
                    error=(
                        "the two fresh runs reported different targets"
                        if capture_result
                        else "replayed target inventory differs from the submitted result"
                    ),
                )
            for expected, actual in zip(submitted, observed, strict=True):
                for name in ("estimate", "lower", "upper"):
                    left, right = getattr(expected, name), getattr(actual, name)
                    exact = exact and left == right
                    if not math.isclose(
                        left,
                        right,
                        rel_tol=TRIALEVAL_REPLAY_REL_TOLERANCE_V1,
                        abs_tol=TRIALEVAL_REPLAY_ABS_TOLERANCE_V1,
                    ):
                        return outcome(
                            "result_mismatch",
                            error=(
                                f"the two fresh runs gave different {name} for {expected.target_id}; seed any "
                                "randomness and do not depend on file order or the clock"
                                if capture_result
                                else f"{expected.target_id}.{name} differs from the submitted result"
                            ),
                        )
    if capture_result:
        if captured_bytes is None:
            raise RuntimeError("Successful capture requires a replayed result")
        if result_path.is_symlink():
            raise ValueError("Captured result cannot overwrite a symbolic link")
        result_path.write_bytes(captured_bytes)
    return outcome("reproduced" if exact else "within_tolerance", estimates=submitted)


def replay_numerical_submission(
    *,
    workspace: Path,
    submitted_project: Path,
    executor_limits: CodeExecutionLimitsV1,
    analysis_language: AnalysisMode = "python_r",
    executor_image: str | None = None,
) -> NumericalReplay:
    """Replay a runner-captured report using its own frozen reporting context.

    Parameters
    ----------
    workspace
        Evidence workspace; only the captured inventory is eligible for replay.
    submitted_project
        Runner-owned source, CSV and reporting_context.json capture directory.
    executor_limits, executor_image
        Assigned execution resources and isolated runtime image.

    Returns
    -------
    NumericalReplay
        Checkpoint-bound result, including explicit missing or ambiguous program
        outcomes without executing an arbitrary choice of entrypoint.

    Raises
    ------
    ValueError
        If runner-owned context is invalid or committed evidence has changed.
    OSError
        If the captured reporting context cannot be read.

    Examples
    --------
    A later study may change workspace scratch without replacing the report's
    source, target inventory or evidence cutoff.
    """
    context = NumericalReportingContext.model_validate(read_json_object(submitted_project / "reporting_context.json"))
    names: tuple[Literal["analysis.py", "analysis.R"], ...] = ("analysis.py", "analysis.R")
    entrypoints = tuple(name for name in names if (submitted_project / name).is_file())
    if len(entrypoints) != 1:
        return NumericalReplay(
            status="invalid_program",
            entrypoint="",
            checkpoint_id=context.checkpoint_id,
            target_ids=context.target_ids,
            evidence_artifacts=context.evidence_artifacts,
            error="captured submission must contain exactly one analysis.py or analysis.R entrypoint",
        )
    result = replay_numerical_program(
        workspace=workspace,
        submitted_project=submitted_project,
        entrypoint=entrypoints[0],
        target_ids=context.target_ids,
        evidence_artifacts=context.evidence_artifacts,
        executor_limits=executor_limits,
        analysis_language=analysis_language,
        executor_image=executor_image,
    )
    return result.model_copy(update={"checkpoint_id": context.checkpoint_id})
