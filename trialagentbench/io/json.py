"""Strict JSON IO helpers (UTF-8, fail-fast, schema-validated)."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from trialagentbench.io.atomic import atomic_text_writer

TModel = TypeVar("TModel", bound=BaseModel)


def loads_json(text: str) -> Any:
    """Decode one JSON value while rejecting duplicates and non-finite values."""

    def reject_constant(value: str) -> float:
        raise ValueError(f"non-finite JSON constant: {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        payload: dict[str, object] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError(f"duplicate JSON object key: {key}")
            payload[key] = value
        return payload

    return json.loads(
        text,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate_keys,
    )


def read_json(path: Path) -> Any:
    """Read strict JSON from `path` with UTF-8 encoding.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    ValueError
        If the payload contains duplicate keys or non-finite numbers.
    """

    return loads_json(path.read_text(encoding="utf-8"))


def read_jsonl_models(model: type[TModel], path: Path) -> Iterator[TModel]:  # noqa: UP047
    """Stream non-empty JSONL records through one strict model contract."""

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield model.model_validate(loads_json(line))
            except (ValidationError, ValueError) as error:
                raise ValueError(f"invalid JSONL record at {path}:{line_number}") from error


def read_json_model(model: type[TModel], path: Path) -> TModel:  # noqa: UP047
    """Read JSON from `path` and validate against `model`.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    pydantic.ValidationError
        If the JSON payload does not validate against `model`.
    """
    payload = read_json(path)
    try:
        return model.model_validate(payload)
    except ValidationError:
        # Preserve pydantic error message; do not wrap into a generic exception.
        raise


def write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    """Atomically write stable UTF-8 JSON to `path`."""

    text = json.dumps(payload, indent=indent, ensure_ascii=True, sort_keys=True, allow_nan=False) + "\n"
    with atomic_text_writer(path) as handle:
        handle.write(text)


def write_json_model(path: Path, model: BaseModel, *, indent: int = 2) -> None:
    """Write a Pydantic model to JSON with stable formatting."""
    write_json(path, model.model_dump(mode="json"), indent=indent)


def append_jsonl_model(path: Path, model: BaseModel) -> None:
    """Append one validated model as a durable JSONL record."""

    append_jsonl(path, model.model_dump(mode="json"))


def append_jsonl(path: Path, payload: Any) -> None:
    """Append one JSON value as a durable JSONL record."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=True, sort_keys=True, allow_nan=False) + "\n"
    with target.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
