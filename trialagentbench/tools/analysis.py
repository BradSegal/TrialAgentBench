"""Common analysis tools exposed by TrialAgentBench tasks."""

from __future__ import annotations

import json
import textwrap
import time
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Literal, cast

from pydantic.types import JsonValue
from trialagentbench_verifier.execution_contracts import (
    RELEASE_EXECUTION_LIMITS_V1,
    AnalysisMode,
    CodeExecutionLimitsV1,
    ReportingInterfaceV1,
    analysis_languages,
)
from trialagentbench_verifier.executor import DockerCodeSession
from trialagentbench_verifier.numerical import NumericalReplay, NumericalReportingContext, replay_numerical_program

from trialagentbench.ports import AnalysisLanguage, CodeExecutionResultV1, CodeExecutionSession, ToolCall
from trialagentbench.ports.tool_input import ToolInputError, parse_tool_arguments

EXECUTE_CODE_TOOL: dict[str, JsonValue] = {
    "type": "function",
    "function": {
        "name": "execute_code",
        "description": (
            "Run Python or R in the persistent analysis session (separate state per language). Paths are relative "
            "to the task workspace; long output is shortened and saved under scratch/outputs/."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "code": {"type": "string", "description": "Python or R code to execute."},
                "language": {
                    "type": "string",
                    "enum": ["python", "r"],
                    "default": "python",
                    "description": "Analysis language; omitted calls use Python.",
                },
                "purpose": {"type": "string", "description": "Brief purpose for the analysis log."},
            },
            "required": ["code"],
        },
    },
}

INSPECT_PARQUET_TOOL: dict[str, JsonValue] = {
    "type": "function",
    "function": {
        "name": "inspect_parquet",
        "description": "Inspect the shape, columns, types, missingness, and summaries of a Parquet file.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to a Parquet file in the task workspace.",
                },
                "language": {
                    "type": "string",
                    "enum": ["python", "r"],
                    "default": "python",
                    "description": "Language used to inspect the file; omitted calls use Python.",
                },
            },
            "required": ["path"],
        },
    },
}

CHECK_ANALYSIS_TOOL: dict[str, JsonValue] = {
    "type": "function",
    "function": {
        "name": "check_analysis",
        "description": (
            "Rerun the saved analysis script twice in a fresh session, as happens at submission, a readout request "
            "or a final decision, and report whether it reproduces and covers every current target; a reproduced "
            "result is written to scratch/estimates.csv. It does not judge accuracy or submit anything. Interactive "
            "variables are reset."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"language": {"type": "string", "enum": ["python", "r"], "default": "python"}},
            "required": [],
        },
    },
}

ANALYSIS_TOOLS: tuple[dict[str, JsonValue], ...] = (EXECUTE_CODE_TOOL, INSPECT_PARQUET_TOOL, CHECK_ANALYSIS_TOOL)


def analysis_tools(mode: AnalysisMode = "python_r") -> tuple[dict[str, JsonValue], ...]:
    """Offer only the interpreters selected for the run."""
    if mode == "python_r":
        return ANALYSIS_TOOLS
    tools = deepcopy(ANALYSIS_TOOLS)
    for tool in tools:
        function = cast(dict[str, JsonValue], tool["function"])
        function["description"] = str(function["description"]).replace(
            "Python or R", "Python" if mode == "python" else "R"
        )
        parameters = cast(dict[str, JsonValue], function["parameters"])
        properties = cast(dict[str, JsonValue], parameters["properties"])
        properties["language"] = {"type": "string", "enum": [mode], "default": mode}
        if "code" in properties:
            properties["code"] = {"type": "string", "description": f"{mode.upper()} code to execute."}
    return tools


def analysis_instructions(
    mode: AnalysisMode = "python_r",
    *,
    limits: CodeExecutionLimitsV1 = RELEASE_EXECUTION_LIMITS_V1,
    reporting_interface: ReportingInterfaceV1 = "v1",
) -> str:
    """State the analysis environment and the reporting rule in a few lines.

    The "clarified" reporting interface also states the helpers' types; "v1" keeps the 1.0 wording.
    """
    languages = analysis_languages(mode)
    stacks = []
    if "python" in languages:
        stacks.append("Python (pandas, NumPy, SciPy, statsmodels, lifelines, scikit-learn)")
    if "r" in languages:
        stacks.append(
            "R (survival, survey, sandwich, nlme, mgcv, MASS; read Parquet with read_trialagentbench_parquet())"
        )
    scripts = " or ".join("scratch/analysis.py" if name == "python" else "scratch/analysis.R" for name in languages)
    helpers = []
    if "python" in languages:
        helpers.append("`from interface.reporting import targets, report`")
    if "r" in languages:
        helpers.append("`source('interface/reporting.R')`")
    recovery = (
        "R file connections do not persist across executions; reopen files within each call. "
        "After an R timeout, objects and settings revert to the last completed execution; rerun setup from "
        "the timed-out call.\n"
        if "r" in languages
        else ""
    )
    return (
        f"Analysis runs in {' and '.join(stacks)}, in a persistent session started in the task workspace, with no "
        f"network; each execution may use {limits.timeout_seconds:g} seconds, {limits.cpu_count:g} CPUs and "
        f"{limits.memory_mb:,} MB. Supplied files are read-only; keep your work in scratch/.\n"
        f"Report results from a saved script, {scripts}: {' or '.join(helpers)}, then `report(rows)` with one row "
        + (
            "per target in `targets()` (interface/targets.csv) giving estimate, lower and upper (95% interval). The "
            if reporting_interface == "v1"
            else "per target. `targets()` returns the target IDs as a list of strings (from interface/targets.csv); "
            "`report(rows)` takes a data frame, or a list of dicts (R: lists), with columns target_id, estimate, "
            "lower and upper (95% interval). The "
        )
        + "script is rerun twice in a "
        f"fresh session and must give the same numbers, so seed any randomness.\n{recovery}"
    )


def _require_argument_keys(
    arguments: dict[str, object],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> None:
    missing = required - arguments.keys()
    unexpected = arguments.keys() - required - optional
    if missing:
        raise ToolInputError(f"Missing tool arguments: {', '.join(sorted(missing))}")
    if unexpected:
        raise ToolInputError(f"Unexpected tool arguments: {', '.join(sorted(unexpected))}")


def check_analysis_tool_v1(
    call: ToolCall,
    *,
    session: CodeExecutionSession,
    workspace: Path,
    context: NumericalReportingContext,
    executor_limits: CodeExecutionLimitsV1,
    executor_image: str | None,
    analysis_language: AnalysisMode,
    reporting_cutoff: bool = False,
    reporting_interface: ReportingInterfaceV1 = "v1",
) -> tuple[CodeExecutionResultV1, CodeExecutionSession]:
    """Run the saved script twice in fresh sessions and keep its output as scratch/estimates.csv.

    The result always comes from the saved script, never from an earlier interactive
    scratch/estimates.csv. The interactive session is closed for the replays and a fresh
    one is returned. At a reporting cutoff the saved entrypoint selects the language and a
    missing target fails the capture. Replays are harness work: callers keep them outside
    the agent's watchdog with ``RuntimeDeadline.harness_work``.
    """
    arguments = parse_tool_arguments(call.arguments, tool_name=call.name)
    _require_argument_keys(arguments, required=frozenset(), optional=frozenset({"language"}))
    languages = analysis_languages(analysis_language)
    language = arguments.get("language", languages[0])
    if language not in languages:
        raise ToolInputError(f"This run accepts analysis code in {', '.join(languages)}")
    language = cast(AnalysisLanguage, language)
    entrypoint: Literal["analysis.py", "analysis.R"] = "analysis.py" if language == "python" else "analysis.R"
    started = time.monotonic()
    session.snapshot_scratch()
    session.close()
    programs = tuple(name for name in ("analysis.py", "analysis.R") if (workspace / "scratch" / name).is_file())
    if reporting_cutoff and len(programs) == 1:
        entrypoint = cast(Literal["analysis.py", "analysis.R"], programs[0])
        language = "python" if entrypoint == "analysis.py" else "r"
    if programs != (entrypoint,):
        replay = NumericalReplay(
            status="invalid_program",
            entrypoint=entrypoint,
            target_ids=context.target_ids,
            evidence_artifacts=context.evidence_artifacts,
            error=(
                f"Save exactly one entrypoint, scratch/{entrypoint}, for this language"
                if programs
                else f"No saved script: write scratch/{entrypoint} that ends with report(rows)"
            ),
        )
    else:
        replay = replay_numerical_program(
            workspace=workspace,
            entrypoint=entrypoint,
            target_ids=context.target_ids,
            evidence_artifacts=context.evidence_artifacts,
            executor_limits=executor_limits,
            executor_image=executor_image,
            analysis_language=analysis_language,
            capture_result=True,
            reporting_interface=reporting_interface,
        )
    reported = {row.target_id for row in replay.estimates or ()}
    missing = [target for target in context.target_ids if target not in reported]
    output = json.dumps(
        {
            "scope": (
                "saved-script result capture; public evidence only, no accuracy feedback"
                if reporting_cutoff
                else "public-evidence reproducibility only; not scientific accuracy or submission"
            ),
            "checkpoint": context.checkpoint_id,
            "status": replay.status,
            "reproducible": replay.reproducible,
            "error": replay.error,
            "missing_targets": missing,
            "executions": [
                {
                    "status": row.status,
                    "seconds": row.elapsed_seconds,
                    "error_output": row.output[:1200] if row.status != "success" else None,
                }
                for row in replay.executions
            ],
            "interactive_session": (
                "reset; scratch/estimates.csv now holds the saved script's output"
                if replay.reproducible
                else "reset; saved scratch/ files retained"
            ),
        }
    )
    replacement = DockerCodeSession(workspace, image=executor_image, limits=executor_limits)
    cpu = [row.cpu_seconds for row in replay.executions]
    memory = [row.session_peak_memory_bytes for row in replay.executions if row.session_peak_memory_bytes is not None]
    return (
        CodeExecutionResultV1(
            status="success" if replay.reproducible and not (reporting_cutoff and missing) else "execution_error",
            language=language,
            output=output,
            elapsed_seconds=time.monotonic() - started,
            cpu_seconds=sum(cast(list[float], cpu)) if cpu and all(value is not None for value in cpu) else None,
            session_peak_memory_bytes=max(memory) if memory else None,
        ),
        replacement,
    )


def execute_analysis_tool_v1(
    call: ToolCall,
    *,
    session: CodeExecutionSession,
    analysis_language: AnalysisMode = "python_r",
) -> CodeExecutionResultV1:
    """Execute one common analysis tool or reject an unknown tool name."""

    arguments = parse_tool_arguments(call.arguments, tool_name=call.name)
    allowed = analysis_languages(analysis_language)
    requested_language = arguments.get("language", allowed[0])
    if requested_language not in allowed:
        raise ToolInputError(f"This run accepts analysis code in {', '.join(allowed)}")
    arguments["language"] = requested_language
    if call.name == "execute_code":
        _require_argument_keys(
            arguments,
            required=frozenset({"code"}),
            optional=frozenset({"language", "purpose"}),
        )
        code = arguments.get("code")
        if not isinstance(code, str) or not code.strip():
            raise ToolInputError("execute_code requires non-empty string code")
        purpose = arguments.get("purpose")
        if purpose is not None and not isinstance(purpose, str):
            raise ToolInputError("execute_code purpose must be a string")
        language = arguments.get("language", "python")
        if language not in {"python", "r"}:
            raise ToolInputError("execute_code language must be 'python' or 'r'")
        return session.execute_result(code, language=cast(AnalysisLanguage, language))
    if call.name == "inspect_parquet":
        _require_argument_keys(arguments, required=frozenset({"path"}), optional=frozenset({"language"}))
        raw_path = arguments.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ToolInputError("inspect_parquet requires a relative Parquet path")
        path = PurePosixPath(raw_path)
        if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".parquet":
            raise ToolInputError("inspect_parquet requires a relative .parquet path inside the task workspace")
        language = arguments.get("language", "python")
        if language not in {"python", "r"}:
            raise ToolInputError("inspect_parquet language must be 'python' or 'r'")
        if language == "python":
            code = textwrap.dedent(f"""\
            import pandas as _pd
            _df = _pd.read_parquet({path.as_posix()!r})
            print(f"Shape: {{_df.shape}}")
            print(f"Columns: {{list(_df.columns)}}")
            print("Dtypes:")
            print(_df.dtypes)
            print("Head:")
            print(_df.head())
            print("Describe:")
            print(_df.describe(include="all"))
            print("Null counts:")
            print(_df.isnull().sum())
            del _df
            """)
        else:
            code = textwrap.dedent(f"""\
                library(trialagentbenchR)
                .tab_df <- read_trialagentbench_parquet({path.as_posix()!r})
                cat("Shape:", nrow(.tab_df), ncol(.tab_df), "\n")
                cat("Columns:", paste(names(.tab_df), collapse=", "), "\n")
                cat("Structure:\n")
                str(.tab_df)
                cat("Head:\n")
                print(utils::head(.tab_df))
                cat("Summary:\n")
                print(summary(.tab_df))
                cat("Null counts:\n")
                print(colSums(is.na(.tab_df)))
                rm(.tab_df)
                """)
        return session.execute_result(code, language=cast(AnalysisLanguage, language))
    raise ToolInputError(f"Unknown analysis tool: {call.name!r}")


__all__ = [
    "ANALYSIS_TOOLS",
    "EXECUTE_CODE_TOOL",
    "INSPECT_PARQUET_TOOL",
    "analysis_languages",
    "execute_analysis_tool_v1",
]
