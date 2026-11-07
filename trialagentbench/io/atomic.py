"""Durable atomic file replacement for harness artifacts."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO


def _sync_directory(directory: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def atomic_text_writer(path: Path, *, newline: str | None = None) -> Iterator[TextIO]:
    """Yield a UTF-8 writer and durably replace ``path`` after success."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline=newline) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _sync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = ["atomic_text_writer"]
