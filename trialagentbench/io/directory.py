"""Transactional directory publication."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def staged_directory(destination: Path) -> Iterator[Path]:
    """Yield a sibling staging directory and publish it only on success."""

    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        yield staging
        if destination.exists():
            raise FileExistsError(f"Output directory appeared during publication: {destination}")
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


__all__ = ["staged_directory"]
