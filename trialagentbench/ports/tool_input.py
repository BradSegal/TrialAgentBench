"""Validation boundary for untrusted model tool calls."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import logging
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import cast

from pydantic import BaseModel

from trialagentbench.ports.llm_provider import JsonObject, JsonValue

logger = logging.getLogger(__name__)


class HarnessConfigurationError(ValueError):
    """Invalid harness-authored provider content, never a participant failure."""


class ToolInputError(ValueError):
    """A model-supplied tool call that can be corrected within the run."""


class JsonObjectDecodeError(ValueError):
    """A stable failure while decoding an untrusted JSON object."""

    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def parse_tool_arguments(arguments: str, *, tool_name: str) -> dict[str, object]:
    """Parse one model tool payload as a JSON object."""

    return parse_json_object_text(arguments, label=f"Tool {tool_name!r} arguments") if arguments else {}


def parse_json_object_text(text: str, *, label: str) -> dict[str, object]:
    """Parse JSON object text while rejecting duplicate field names."""

    try:
        return decode_json_object_text(text)
    except JsonObjectDecodeError as exc:
        raise ToolInputError(f"{label} {exc}") from exc


def decode_json_object_text(text: str) -> dict[str, object]:
    """Decode one JSON object with stable fail-fast error categories."""

    try:
        payload = json.loads(text, object_pairs_hook=_reject_duplicate_json_keys)
    except json.JSONDecodeError as exc:
        raise JsonObjectDecodeError(code="json_syntax", message="must be valid JSON.") from exc
    if not isinstance(payload, dict):
        raise JsonObjectDecodeError(code="json_object_required", message="must be a JSON object.")
    return payload


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise JsonObjectDecodeError(
                code="duplicate_json_field",
                message=f"Duplicate JSON field: {key!r}.",
            )
        payload[key] = value
    return payload


# Deliberate normalized documentation matches require an explanation per entry.
_DOCSTRING_TEXT_ALLOWLIST: frozenset[str] = frozenset()
_PACKAGE_PREFIXES = ("trialagentbench", "trialagentbench_verifier")


@lru_cache(maxsize=512)
def _module_docstrings(path: Path, mtime_ns: int, byte_length: int) -> frozenset[str]:
    """Read definitions once per source revision while allowing ordinary documentation edits."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return frozenset(
        " ".join(doc.split())
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and (doc := ast.get_docstring(node, clean=False))
        and doc.strip()
    )


def _package_docstrings() -> frozenset[str]:
    """Walk every source module without importing optional transports or running entry points."""
    docs: set[str] = set()
    for package in _PACKAGE_PREFIXES:
        spec = importlib.util.find_spec(package)
        if spec is None or spec.submodule_search_locations is None:
            raise HarnessConfigurationError(f"Cannot locate documentation source for {package}")
        for location in spec.submodule_search_locations:
            for path in sorted(Path(location).rglob("*.py")):
                state = path.stat()
                docs.update(_module_docstrings(path, state.st_mtime_ns, state.st_size))
    docs.difference_update(_DOCSTRING_TEXT_ALLOWLIST)
    return frozenset(docs)


def _class_docstrings(value: object, seen: set[int]) -> set[str]:
    """Retain automatic schema documentation refusals for local and dependency classes."""
    if id(value) in seen:
        return set()
    seen.add(id(value))
    if isinstance(value, type):
        return {" ".join(base.__doc__.split()) for base in value.__mro__ if base.__doc__}
    children = value.values() if isinstance(value, dict) else value if isinstance(value, (list, tuple)) else ()
    return {doc for child in children for doc in _class_docstrings(child, seen)}


def _reject_docstring_text(value: object, docstrings: frozenset[str], *, descriptions_only: bool) -> None:
    """Refuse internal documentation contained in rendered descriptions or request text."""
    if isinstance(value, str) and not descriptions_only:
        normalized = " ".join(value.split())
        if any(doc in normalized for doc in docstrings):
            raise HarnessConfigurationError(
                "Provider tool descriptions must be explicit schema code; "
                "prompt text must also be distinct from internal docstrings"
            )
    elif isinstance(value, Mapping):
        for key, child in value.items():
            if not descriptions_only or key == "description" or isinstance(child, (Mapping, list, tuple)):
                _reject_docstring_text(child, docstrings, descriptions_only=descriptions_only and key != "description")
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_docstring_text(child, docstrings, descriptions_only=descriptions_only)


def validate_provider_text_v1(
    *,
    messages: Sequence[JsonObject],
    tools: Sequence[JsonObject] | None,
    episode_messages: Sequence[JsonObject] = (),
    provenance: str = "",
) -> None:
    """Preflight authored prompts/tools; warn on episode replies without changing outcomes.

    Only supplied system/task content is blocking. Assistant messages and tool-call
    arguments are never examined. Episode replies can contain participant material;
    any documentation match is a provenance warning and cannot fail the model.
    """
    docstrings = _package_docstrings()
    for message in messages:
        if message.get("role") in ("system", "user"):
            _reject_docstring_text(message.get("content"), docstrings, descriptions_only=False)
    _reject_docstring_text(tools, docstrings, descriptions_only=True)
    for message in episode_messages:
        if message.get("role") not in ("user", "tool"):
            continue
        content = message.get("content")
        try:
            _reject_docstring_text(content, docstrings, descriptions_only=False)
        except HarnessConfigurationError:
            digest = hashlib.sha256(json.dumps(content, ensure_ascii=True, sort_keys=True).encode()).hexdigest()
            logger.warning(
                "Provider content provenance warning; continuing unchanged: %s content_sha256=%s", provenance, digest
            )


def provider_tool(name: str, description: str, arguments: type[BaseModel]) -> dict[str, JsonValue]:
    """Declare a function tool from its argument model in a form every provider accepts.

    References are inlined, every schema node states its ``type``, a nullable field is a type list
    such as ``["integer", "null"]`` and generated titles are dropped.

    Raises
    ------
    ValueError
        If a rendered description contains internal package documentation, or needs a
        construct outside that form, such as a union of two non-null types.
    """
    schema = arguments.model_json_schema()
    definitions = cast(dict[str, dict[str, JsonValue]], schema.pop("$defs", {}))
    tool: dict[str, JsonValue] = {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": _typed(schema, definitions)},
    }
    _reject_docstring_text(
        tool,
        _package_docstrings() | _class_docstrings(arguments.__pydantic_core_schema__, set()),
        descriptions_only=True,
    )
    return tool


def _typed(schema: dict[str, JsonValue], definitions: dict[str, dict[str, JsonValue]]) -> dict[str, JsonValue]:
    """Return one schema node with references inlined and an explicit type."""
    if "$ref" in schema:
        reference = str(schema["$ref"]).removeprefix("#/$defs/")
        schema = {**definitions[reference], **{key: value for key, value in schema.items() if key != "$ref"}}
    if "anyOf" in schema:
        options = cast(list[dict[str, JsonValue]], schema["anyOf"])
        present = [option for option in options if option != {"type": "null"}]
        if len(options) != 2 or len(present) != 1:
            raise ValueError("provider tool schemas allow one type, optionally nullable")
        option = _typed(present[0], definitions)
        if not isinstance(option["type"], str):
            raise ValueError("provider tool schemas allow one type, optionally nullable")
        outer = {key: value for key, value in schema.items() if key != "anyOf"}
        schema = {**option, **outer, "type": [option["type"], "null"]}
    typed: dict[str, JsonValue] = {}
    for key, value in schema.items():
        if key == "title" and isinstance(value, str):
            continue
        if key == "properties":
            properties = cast(dict[str, dict[str, JsonValue]], value)
            typed[key] = {name: _typed(child, definitions) for name, child in properties.items()}
        elif key == "items":
            typed[key] = _typed(cast(dict[str, JsonValue], value), definitions)
        else:
            typed[key] = value
    if "type" not in typed or (typed["type"] == "array" and "items" not in typed):
        raise ValueError(f"provider tool schema node lacks an explicit type: {sorted(typed)}")
    return typed


__all__ = [
    "HarnessConfigurationError",
    "JsonObjectDecodeError",
    "ToolInputError",
    "decode_json_object_text",
    "parse_json_object_text",
    "parse_tool_arguments",
    "provider_tool",
    "validate_provider_text_v1",
]
