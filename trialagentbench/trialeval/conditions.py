"""Immutable participant-facing conditions for TrialEval experiments."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from pydantic.types import JsonValue
from trialagentbench_verifier.execution_contracts import AnalysisMode

from trialagentbench.contracts.core.trialeval_conditions import (
    ProcedureAssistanceV1,
    procedure_assistance_exposure_v1,
)
from trialagentbench.io.checksums import canonical_payload_sha256
from trialagentbench.util.runtime_context import FINAL_TURN_SUBMISSION_REMINDER, SUBMISSION_WINDOW_REMINDER

# Appended to both end-of-run reminders: the analysis tools are closed, so a failing script cannot be repaired.
INCOMPLETE_SUBMISSION_NOTE = " If your saved script fails or omits a result, submit with accept_incomplete=true."

_PROCEDURE_COMPONENTS = (
    "define the scientific question, estimand, population, contrast, endpoint, horizon, and intercurrent-event handling",
    "inspect the study data, assignment mechanism, dependence structure, timing, monitoring, and missingness",
    "assess identification and the assumptions needed by candidate analyses",
    "execute a defensible primary analysis with uncertainty and relevant diagnostics or sensitivity analyses",
    "when the public task declares planning eligible, translate the result into the requested planning implication",
    "verify every submitted claim against executed code and cited participant evidence",
)
_UNORDERED_PROCEDURE_COMPONENTS = "; ".join(_PROCEDURE_COMPONENTS)
_PROCEDURE_ASSISTANCE_BOUNDARY = (
    "These requirements do not prescribe an estimator; method selection must follow the protocol and supplied data."
)
_PROCEDURE_ASSISTANCE: dict[ProcedureAssistanceV1, str] = {
    "output_contract_only": "",
    "unordered_checklist": (
        f"A complete analysis must perform these operations in any order: {_UNORDERED_PROCEDURE_COMPONENTS}. "
        f"{_PROCEDURE_ASSISTANCE_BOUNDARY}"
    ),
}
_REPORT_SUBMISSION = (
    "Submit the Markdown report with submit_response, or save it in scratch/ and call submit_response_file. "
    "Your saved script is rerun at submission; if it fails or omits a result, fix it, or resubmit with "
    "accept_incomplete=true."
)

_DIRECT_SUBMISSION_TOOL: dict[str, JsonValue] = {
    "type": "function",
    "function": {
        "name": "submit_response",
        "description": "Submit the completed Markdown analysis report.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "content": {"type": "string", "minLength": 1},
                "accept_incomplete": {"type": "boolean", "default": False},
            },
            "required": ["content"],
        },
    },
}
_FILE_SUBMISSION_TOOL: dict[str, JsonValue] = {
    "type": "function",
    "function": {
        "name": "submit_response_file",
        "description": "Submit the completed UTF-8 Markdown report from a file under scratch/.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Report path, for example scratch/report.md",
                },
                "accept_incomplete": {"type": "boolean", "default": False},
            },
            "required": ["path"],
        },
    },
}
_NUMERICAL_HEADER = "target_id,estimate,lower,upper\n"
_PROGRAM_RESULT_RELATIVE_PATH = Path("interface") / "estimates.csv"


def procedure_assistance_v1(assistance: ProcedureAssistanceV1) -> str:
    """Return the exact method-neutral procedure-assistance text."""

    procedure_assistance_exposure_v1(suite="trialeval", procedure_assistance=assistance)
    return _PROCEDURE_ASSISTANCE[assistance]


def submission_instruction_v1(analysis_language: AnalysisMode = "python_r") -> str:
    """Return the single report-plus-program submission instruction."""

    return _REPORT_SUBMISSION


def stage_numerical_result_template(root: Path) -> Path:
    """Stage the common numerical CSV header in one workspace."""

    path = Path(root) / _PROGRAM_RESULT_RELATIVE_PATH
    expected = _NUMERICAL_HEADER
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("TrialEval analysis-result contract path must not be a symlink.")
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != expected:
            raise ValueError("Existing TrialEval analysis-result contract differs from the canonical CSV header.")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(expected, encoding="utf-8")
    return path


def submission_tools_v1() -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    """Return direct and file transports for the Markdown report."""

    return _DIRECT_SUBMISSION_TOOL, _FILE_SUBMISSION_TOOL


def prompt_set_sha256_v1() -> str:
    """Hash the complete participant-facing prompt and result-contract surface."""

    return canonical_payload_sha256(
        cast(
            JsonValue,
            {
                "procedure_assistance": _PROCEDURE_ASSISTANCE,
                "procedure_assistance_exposure": {
                    assistance: procedure_assistance_exposure_v1(
                        suite="trialeval",
                        procedure_assistance=assistance,
                    ).model_dump(mode="json")
                    for assistance in _PROCEDURE_ASSISTANCE
                },
                "submission_instruction": _REPORT_SUBMISSION,
                "numerical_csv_header": _NUMERICAL_HEADER,
                "submission_tools": submission_tools_v1(),
                "final_turn_submission_reminder": FINAL_TURN_SUBMISSION_REMINDER,
                "submission_window_reminder": SUBMISSION_WINDOW_REMINDER,
                "incomplete_submission_note": INCOMPLETE_SUBMISSION_NOTE,
            },
        )
    )


__all__ = [
    "INCOMPLETE_SUBMISSION_NOTE",
    "procedure_assistance_v1",
    "prompt_set_sha256_v1",
    "stage_numerical_result_template",
    "submission_instruction_v1",
    "submission_tools_v1",
]
