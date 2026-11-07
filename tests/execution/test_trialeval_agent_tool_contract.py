from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from trialagentbench.ports import CodeExecutionResultV1, ToolCall
from trialagentbench.trialeval.agent import _handle_tool_call


class _Session:
    def execute(self, code: str) -> str:
        return code

    def execute_result(self, code: str, *, language: str = "python") -> CodeExecutionResultV1:
        del language
        return CodeExecutionResultV1(status="success", output=code, elapsed_seconds=0.1)

    def close(self) -> None:
        return None


@pytest.mark.parametrize(
    "tool_call",
    [
        ToolCall(id="1", name="unknown", arguments="{}"),
        ToolCall(id="2", name="execute_code", arguments='{"code": 3}'),
        ToolCall(id="3", name="inspect_parquet", arguments='{"path": "../grader/truth.parquet"}'),
        ToolCall(id="4", name="inspect_parquet", arguments='{"path": "table.csv"}'),
        ToolCall(id="5", name="write_workspace_file", arguments='{"path": "../truth.json", "content": "x"}'),
        ToolCall(id="6", name="read_workspace_file", arguments='{"path": "x", "start_line": 2, "end_line": 1}'),
    ],
)
def test_trial_eval_tools_reject_invalid_calls(tool_call: ToolCall, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _handle_tool_call(
            tool_call,
            _Session(),
            False,
        )


def test_inspect_parquet_accepts_only_relative_parquet_paths(tmp_path: Path) -> None:
    output, submission, execution = _handle_tool_call(
        ToolCall(id="1", name="inspect_parquet", arguments='{"path": "data/raw/adtte.parquet"}'),
        _Session(),
        False,
    )

    assert "_pd.read_parquet('data/raw/adtte.parquet')" in output
    assert submission is None
    assert execution is not None
    assert execution.status == "success"


def test_execute_code_preserves_failure_status_for_runtime_trace(tmp_path: Path) -> None:
    class FailureSession(_Session):
        def execute_result(self, code: str, *, language: str = "python") -> CodeExecutionResultV1:
            del code, language
            return CodeExecutionResultV1(
                status="execution_error",
                output="Traceback: failed",
                output_truncated=True,
                elapsed_seconds=0.25,
            )

    output, submission, execution = _handle_tool_call(
        ToolCall(id="1", name="execute_code", arguments='{"code": "raise RuntimeError()"}'),
        FailureSession(),
        False,
    )

    assert output == "Traceback: failed"
    assert submission is None
    assert execution is not None
    assert execution.status == "execution_error"
    assert execution.output_truncated is True


class _SubmissionSession(_Session):
    def __init__(self, text: str) -> None:
        self.text = text

    def execute_result(self, code: str, *, language: str = "python") -> CodeExecutionResultV1:
        del language
        assert "scratch/" in code
        encoded = base64.b64encode(self.text.encode("utf-8")).decode("ascii")
        return CodeExecutionResultV1(status="success", output=encoded, elapsed_seconds=0.1)


def test_response_file_rejects_workspace_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="remain under scratch"):
        _handle_tool_call(
            ToolCall(id="1", name="submit_response_file", arguments='{"path":"../submission.json"}'),
            _SubmissionSession("report"),
            False,
        )


def test_narrative_response_file_preserves_text(tmp_path: Path) -> None:
    report = "# Analysis\n\nPrimary estimate: -0.20 (95% CI -0.46 to 0.06).\n"
    output, submission, execution = _handle_tool_call(
        ToolCall(id="1", name="submit_response_file", arguments='{"path":"report.md"}'),
        _SubmissionSession(report),
        False,
    )

    assert output == "Response submitted."
    assert submission == {"report": report}
    assert execution is None


def test_narrative_response_file_rejects_blank_report(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty Markdown"):
        _handle_tool_call(
            ToolCall(id="1", name="submit_response_file", arguments='{"path":"report.md"}'),
            _SubmissionSession(" \n"),
            False,
        )


def test_direct_response_preserves_the_report_text(tmp_path: Path) -> None:
    del tmp_path
    report = "# Analysis\n\nEstimate 0.1."
    _, submission, _ = _handle_tool_call(
        ToolCall(id="1", name="submit_response", arguments=json.dumps({"content": report})),
        _Session(),
        False,
    )

    assert submission == {"report": report}
