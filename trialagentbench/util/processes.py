"""Ordered offline grading with forked workers and inherited read-only releases."""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from typing import TypeVar, cast

_T = TypeVar("_T")
_R = TypeVar("_R")
_ACTION: Callable[[object], object] | None = None


def _execute(item: object) -> object:
    if _ACTION is None:
        raise RuntimeError("Offline grading worker has no action")
    return _ACTION(item)


def process_map(action: Callable[[_T], _R], items: Iterable[_T], *, workers: int | None = None) -> list[_R]:
    """Grade in input order, propagating worker failures and draining before return.

    Fork preserves read-only loaded release data and the campaign's shared
    executor semaphore. Offline grading calls this from the owning main thread.
    """
    return list(process_imap(action, items, workers=workers))


def process_imap(action: Callable[[_T], _R], items: Iterable[_T], *, workers: int | None = None) -> Iterator[_R]:
    """Yield ``process_map`` results in input order as soon as each prefix is complete."""
    global _ACTION
    sequence = tuple(items)
    if not sequence:
        return
    maximum = min(len(sequence), workers or len(os.sched_getaffinity(0)))
    if maximum == 1:
        yield from (action(item) for item in sequence)
        return
    previous = _ACTION
    _ACTION = cast(Callable[[object], object], action)
    try:
        with ProcessPoolExecutor(max_workers=maximum, mp_context=multiprocessing.get_context("fork")) as pool:
            yield from cast(Iterator[_R], pool.map(_execute, sequence))
    finally:
        _ACTION = previous
