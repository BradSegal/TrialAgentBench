"""Persistent text-file workspace tools shared by both benchmark agents."""

from __future__ import annotations

import base64
import binascii
from pathlib import PurePosixPath
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from trialagentbench.ports import CodeExecutionResultV1, CodeExecutionSession
from trialagentbench.ports.llm_provider import JsonValue
from trialagentbench.ports.tool_input import ToolInputError, provider_tool

_MAX_FILE_CHARS: Final = 262_144
_MAX_READ_LINES: Final = 400
_MAX_PATH_CHARS: Final = 240
_MAX_SUBMISSION_FILE_BYTES: Final = 65_536


class _WorkspaceArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _WriteWorkspaceArguments(_WorkspaceArguments):
    path: str = Field(description="File path under scratch/, e.g. scratch/analysis.py")
    content: str = Field(description="Complete UTF-8 file content")


class _ReadWorkspaceArguments(_WorkspaceArguments):
    path: str = Field(description="Path relative to the task workspace, e.g. scratch/analysis.py")
    start_line: int = Field(default=1, ge=1, strict=True, description="First line, inclusive")
    end_line: int | None = Field(default=None, ge=1, strict=True, description="Last line, inclusive")


class _ListWorkspaceArguments(_WorkspaceArguments):
    pass


_WORKSPACE_ARGUMENT_MODELS: Final[dict[str, type[_WorkspaceArguments]]] = {
    "write_workspace_file": _WriteWorkspaceArguments,
    "read_workspace_file": _ReadWorkspaceArguments,
    "list_workspace_files": _ListWorkspaceArguments,
}
_WORKSPACE_TOOL_DESCRIPTIONS: Final[dict[str, str]] = {
    "write_workspace_file": "Write a complete UTF-8 text file (script, notes, table) under scratch/.",
    "read_workspace_file": "Read numbered lines of a text file in the task workspace.",
    "list_workspace_files": "List the files under scratch/ with their sizes.",
}
WORKSPACE_TOOLS: Final[list[dict[str, JsonValue]]] = [
    provider_tool(name, _WORKSPACE_TOOL_DESCRIPTIONS[name], model)
    for name, model in _WORKSPACE_ARGUMENT_MODELS.items()
]


def handle_workspace_tool(
    *,
    name: str,
    arguments: dict[str, object],
    session: CodeExecutionSession,
) -> CodeExecutionResultV1:
    """Execute one validated workspace operation in the persistent container."""

    argument_model = _WORKSPACE_ARGUMENT_MODELS.get(name)
    if argument_model is None:
        raise ToolInputError(f"Unknown workspace tool: {name}")
    try:
        parsed = argument_model.model_validate(arguments)
    except ValidationError as error:
        raise ToolInputError(f"Invalid {name} arguments: {error}") from error

    if isinstance(parsed, _WriteWorkspaceArguments):
        path = _workspace_path(parsed.path)
        content = parsed.content
        if len(content) > _MAX_FILE_CHARS:
            raise ToolInputError(f"write_workspace_file content exceeds {_MAX_FILE_CHARS} characters.")
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        code = (
            "import base64\n"
            "from pathlib import Path\n"
            f"_path = Path({str(path)!r})\n"
            "_path.parent.mkdir(parents=True, exist_ok=True)\n"
            f"_content = base64.b64decode({encoded!r}).decode('utf-8')\n"
            "_path.write_text(_content, encoding='utf-8')\n"
            "print(f'Wrote {_path.as_posix()} ({len(_content)} characters).')"
        )
        return session.execute_result(code)

    if isinstance(parsed, _ReadWorkspaceArguments):
        path = _readable_path(parsed.path)
        start = parsed.start_line
        end = parsed.end_line if parsed.end_line is not None else start + _MAX_READ_LINES - 1
        if end < start:
            raise ToolInputError("read_workspace_file end_line must be greater than or equal to start_line.")
        if end - start + 1 > _MAX_READ_LINES:
            raise ToolInputError(f"read_workspace_file may return at most {_MAX_READ_LINES} lines.")
        code = (
            "from pathlib import Path\n"
            f"_path = Path({str(path)!r})\n"
            "_requested = _path\n"
            "if not _path.exists() and _path.parts[0] != 'scratch' and (Path('scratch') / _path).exists():\n"
            "    _path = Path('scratch') / _path\n"
            "if _path.is_symlink() or not _path.is_file():\n"
            "    _similar = sorted(p.as_posix() for p in Path('.').rglob(_requested.name) if p.is_file())[:5]\n"
            "    raise FileNotFoundError(f'No file at {_requested.as_posix()}'\n"
            "        + (f'; files with that name: {\", \".join(_similar)}' if _similar else ''))\n"
            "_lines = _path.read_text(encoding='utf-8').splitlines()\n"
            f"_start, _end = {start}, {end}\n"
            "for _number, _line in enumerate(_lines[_start - 1:_end], start=_start):\n"
            "    print(f'{_number:04d}: {_line}')"
        )
        return session.execute_result(code)

    if isinstance(parsed, _ListWorkspaceArguments):
        code = (
            "from pathlib import Path\n"
            "_root = Path('scratch')\n"
            "_root.mkdir(parents=True, exist_ok=True)\n"
            "_files = sorted(path for path in _root.rglob('*') if path.is_file())\n"
            "for _path in _files[:200]:\n"
            "    print(f'{_path.as_posix()}\\t{_path.stat().st_size} bytes')\n"
            "if len(_files) > 200:\n"
            "    print(f'... {len(_files) - 200} additional files')\n"
            "if not _files:\n"
            "    print('(scratch workspace is empty)')"
        )
        return session.execute_result(code)

    raise AssertionError(f"Unhandled workspace argument model: {type(parsed).__name__}")


def read_workspace_submission_text(
    *,
    path: object,
    session: CodeExecutionSession,
) -> str:
    """Read one bounded regular UTF-8 submission file from scratch/."""
    workspace_path = _workspace_path(path)
    code = (
        "import base64\n"
        "from pathlib import Path\n"
        f"_path = Path({str(workspace_path)!r})\n"
        "if _path.is_symlink() or not _path.is_file():\n"
        "    raise ValueError('Submission path must identify a regular non-symlink file.')\n"
        f"if _path.stat().st_size > {_MAX_SUBMISSION_FILE_BYTES}:\n"
        f"    raise ValueError('Submission file exceeds {_MAX_SUBMISSION_FILE_BYTES} bytes.')\n"
        "print(base64.b64encode(_path.read_bytes()).decode('ascii'))"
    )
    result = session.execute_result(code)
    if result.status != "success":
        raise ToolInputError(f"Unable to read workspace submission file: {result.output or result.status}")
    try:
        raw = base64.b64decode(result.output.strip(), validate=True)
        text = raw.decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise ToolInputError("Workspace submission file must contain valid UTF-8 text.") from exc
    if len(raw) > _MAX_SUBMISSION_FILE_BYTES:
        raise ToolInputError(f"Workspace submission file exceeds {_MAX_SUBMISSION_FILE_BYTES} bytes.")
    return text


def _workspace_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value.strip():
        raise ToolInputError("Workspace path must be a non-empty string.")
    if len(value) > _MAX_PATH_CHARS:
        raise ToolInputError(f"Workspace path exceeds {_MAX_PATH_CHARS} characters.")
    if "\\" in value or any(ord(character) < 32 for character in value):
        raise ToolInputError("Workspace path must use printable POSIX path characters.")
    relative = PurePosixPath(value)
    if relative.parts and relative.parts[0] == "scratch":
        relative = PurePosixPath(*relative.parts[1:])
    if (
        relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative == PurePosixPath(".")
    ):
        raise ToolInputError("Workspace path must be relative and remain under scratch/.")
    return PurePosixPath("scratch") / relative


def _readable_path(value: object) -> PurePosixPath:
    """Return one portable path within the participant task workspace."""
    if not isinstance(value, str) or not value.strip():
        raise ToolInputError("Workspace path must be a non-empty string.")
    if len(value) > _MAX_PATH_CHARS:
        raise ToolInputError(f"Workspace path exceeds {_MAX_PATH_CHARS} characters.")
    if "\\" in value or any(ord(character) < 32 for character in value):
        raise ToolInputError("Workspace path must use printable POSIX path characters.")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative == PurePosixPath(".")
    ):
        raise ToolInputError("Workspace path must be relative and remain inside the task workspace.")
    return relative


__all__ = ["WORKSPACE_TOOLS", "handle_workspace_tool", "read_workspace_submission_text"]
