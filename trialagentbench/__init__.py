"""TrialAgentBench harness.

This repository contains the evaluation harness for:

- TrialEval: participant-level clinical-trial analysis tasks.
- TrialDev: sequential drug-development program tasks.

The public entrypoints are exposed via `[project.scripts]` in `pyproject.toml`.
"""

from __future__ import annotations

from importlib.metadata import version

__all__ = ["__version__"]

# pyproject.toml is the single source; an uninstalled checkout raises PackageNotFoundError.
__version__ = version("trial-agent-bench")
