"""Exclusive ownership of native run mutation, released automatically after process death."""

from __future__ import annotations

import fcntl
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_INHERITED_LOCK_WAIT_SECONDS = 5.0


@contextmanager
def exclusive_run_directory(root: Path, *, wait_seconds: float = _INHERITED_LOCK_WAIT_SECONDS) -> Iterator[None]:
    """Lock a run directory without adding operational files to scientific custody.

    A ``flock`` belongs to the open file description, which a child forked by any thread inherits until it
    executes. A subprocess spawned elsewhere in this process can therefore hold a just-released lock for a few
    milliseconds (12% of lock attempts under concurrent spawning). The lock is retried for ``wait_seconds``
    before another owner is declared; a genuine owner holds it far longer.
    """
    import os

    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        deadline, pause = time.monotonic() + wait_seconds, 0.001
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as error:
                if time.monotonic() >= deadline:
                    raise ValueError("Another process owns this native run") from error
                time.sleep(pause)
                pause = min(2 * pause, 0.05)
        yield
    finally:
        os.close(descriptor)
