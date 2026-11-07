"""Git helpers for provenance recording."""

from __future__ import annotations

import subprocess
from pathlib import Path


def head_sha(repo_root: Path) -> str:
    """Return current HEAD SHA for `repo_root`, or empty string if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return ""


__all__ = ["head_sha"]
