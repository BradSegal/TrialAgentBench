"""Durable atomic file replacement for harness artifacts."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

from pydantic import BaseModel, JsonValue

from trialagentbench.io.atomic import _sync_directory


def _make_directories(directory: Path) -> None:
    """Persist newly created ancestor links before publishing files beneath them."""

    missing: list[Path] = []
    parent = directory
    while not parent.exists():
        missing.append(parent)
        parent = parent.parent
    directory.mkdir(parents=True, exist_ok=True)
    for created in reversed(missing):
        _sync_directory(created.parent)


@contextmanager
def atomic_text_writer(path: Path, *, newline: str | None = None, exclusive: bool = False) -> Iterator[TextIO]:
    """Publish complete UTF-8 text durably, optionally refusing an existing target."""

    target = Path(path)
    _make_directories(target.parent)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline=newline) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            os.link(temporary, target)
        else:
            os.replace(temporary, target)
        _sync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path: Path, payload: JsonValue) -> None:
    """Durably append provider evidence, including newly created directory links."""

    target = Path(path)
    _make_directories(target.parent)
    line = json.dumps(payload, ensure_ascii=True, sort_keys=True, allow_nan=False) + "\n"
    with target.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
    _sync_directory(target.parent)


def append_jsonl_model(path: Path, model: BaseModel) -> None:
    """Append one validated provider evidence record durably."""

    append_jsonl(path, model.model_dump(mode="json"))


__all__ = ["atomic_text_writer", "append_jsonl", "append_jsonl_model"]
