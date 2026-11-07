"""Run ownership survives descriptors inherited by concurrently spawned children."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from trialagentbench.util.run_ownership import exclusive_run_directory


def test_a_spawned_child_does_not_masquerade_as_another_owner(tmp_path: Path) -> None:
    stop = threading.Event()
    failures: list[Exception] = []

    def spawn() -> None:
        while not stop.is_set():
            subprocess.run(["/bin/true"], check=True)

    spawners = [threading.Thread(target=spawn) for _ in range(4)]
    for thread in spawners:
        thread.start()
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                with exclusive_run_directory(tmp_path):
                    pass
            except ValueError as error:
                failures.append(error)
    finally:
        stop.set()
        for thread in spawners:
            thread.join()
    assert not failures


def test_a_live_owner_in_another_process_is_still_refused(tmp_path: Path) -> None:
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import fcntl, os, sys, time; d = os.open(sys.argv[1], os.O_RDONLY | os.O_DIRECTORY); "
            "fcntl.flock(d, fcntl.LOCK_EX); print('held', flush=True); time.sleep(30)",
            str(tmp_path),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "held"
        with pytest.raises(ValueError, match="Another process owns this native run"):
            with exclusive_run_directory(tmp_path, wait_seconds=0.2):
                pass
    finally:
        holder.kill()
        holder.wait()
