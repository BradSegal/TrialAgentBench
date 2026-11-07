"""Tests for common analysis-tool schema and execution agreement."""

from __future__ import annotations

from pathlib import Path

import pytest

from trialagentbench.ports import AnalysisLanguage, CodeExecutionResultV1, ToolCall
from trialagentbench.ports.tool_input import ToolInputError
from trialagentbench.tools.analysis import execute_analysis_tool_v1


class _Session:
    def __init__(self) -> None:
        self.code: list[tuple[AnalysisLanguage, str]] = []

    def execute_result(self, code: str, *, language: AnalysisLanguage = "python") -> CodeExecutionResultV1:
        self.code.append((language, code))
        return CodeExecutionResultV1(status="success", language=language, output="ok", elapsed_seconds=0.0)

    def execute(self, code: str, *, language: AnalysisLanguage = "python") -> str:
        return self.execute_result(code, language=language).output

    def snapshot_scratch(self) -> Path:
        return Path("scratch")

    def close(self) -> None:
        return None


def test_execute_code_accepts_only_declared_arguments() -> None:
    session = _Session()

    result = execute_analysis_tool_v1(
        ToolCall(id="call-1", name="execute_code", arguments='{"code":"print(1)","purpose":"check"}'),
        session=session,
    )

    assert result.status == "success"
    assert session.code == [("python", "print(1)")]
    with pytest.raises(ToolInputError, match="Unexpected tool arguments: extra"):
        execute_analysis_tool_v1(
            ToolCall(id="call-2", name="execute_code", arguments='{"code":"print(1)","extra":42}'),
            session=session,
        )


def test_inspect_parquet_accepts_only_declared_relative_path() -> None:
    session = _Session()

    result = execute_analysis_tool_v1(
        ToolCall(id="call-1", name="inspect_parquet", arguments='{"path":"data/analysis.parquet"}'),
        session=session,
    )

    assert result.status == "success"
    assert "data/analysis.parquet" in session.code[0][1]
    with pytest.raises(ToolInputError, match="Unexpected tool arguments: extra"):
        execute_analysis_tool_v1(
            ToolCall(
                id="call-2",
                name="inspect_parquet",
                arguments='{"path":"data/analysis.parquet","extra":42}',
            ),
            session=session,
        )


def test_execute_code_and_parquet_inspection_support_r_explicitly() -> None:
    session = _Session()

    executed = execute_analysis_tool_v1(
        ToolCall(id="call-1", name="execute_code", arguments='{"language":"r","code":"print(1)"}'),
        session=session,
    )
    inspected = execute_analysis_tool_v1(
        ToolCall(
            id="call-2",
            name="inspect_parquet",
            arguments='{"language":"r","path":"data/analysis.parquet"}',
        ),
        session=session,
    )

    assert executed.language == "r"
    assert inspected.language == "r"
    assert session.code[0] == ("r", "print(1)")
    assert "read_trialagentbench_parquet" in session.code[1][1]


@pytest.mark.parametrize("language,other", [("python", "r"), ("r", "python")])
def test_single_language_condition_controls_tools_defaults_and_execution(language, other):
    import json

    from trialagentbench.tools.analysis import analysis_instructions, analysis_tools

    session = _Session()
    for tool in analysis_tools(language):
        assert tool["function"]["parameters"]["properties"]["language"]["enum"] == [language]
    result = execute_analysis_tool_v1(
        ToolCall(id="ok", name="execute_code", arguments='{"code":"print(1)"}'),
        session=session,
        analysis_language=language,
    )
    assert result.language == language
    with pytest.raises(ToolInputError, match="This run accepts"):
        execute_analysis_tool_v1(
            ToolCall(id="bad", name="execute_code", arguments=json.dumps({"code": "print(1)", "language": other})),
            session=session,
            analysis_language=language,
        )
    assert len(session.code) == 1
    assert ("scratch/analysis.py" if other == "python" else "scratch/analysis.R") not in analysis_instructions(
        language
    )


@pytest.mark.parametrize("mode", ["python", "r", "python_r"])
def test_reporting_helper_instructions_match_offered_languages(mode):
    from trialagentbench.tools.analysis import analysis_instructions

    instructions = analysis_instructions(mode)
    assert ("from interface.reporting import" in instructions) == (mode != "r")
    assert ("source('interface/reporting.R')" in instructions) == (mode != "python")
    assert "interface/targets.csv" in instructions


def test_analysis_instructions_use_the_actual_session_limits():
    from trialagentbench.ports import CodeExecutionLimitsV1
    from trialagentbench.tools.analysis import analysis_instructions

    default = analysis_instructions()
    assert "300 seconds" in default and "4,096 MB" in default and "2 CPUs" in default
    actual = analysis_instructions("r", limits=CodeExecutionLimitsV1(timeout_seconds=37, memory_mb=512, cpu_count=1))
    assert "37 seconds" in actual and "512 MB" in actual and "1 CPUs" in actual
    assert "300 seconds" not in actual and "language condition" not in actual
