"""Stable classification of failures shared by benchmark runtimes."""

from __future__ import annotations

import subprocess
from pathlib import Path

from trialagentbench_verifier.executor import CodeExecutionSessionTerminated

from trialagentbench.contracts.trace.observable import BenchmarkRuntimeFailureTypeV1
from trialagentbench.ports import ModelResponseNoncompletion, ProviderRequestError, RuntimeDeadlineExceeded
from trialagentbench.ports.tool_input import HarnessConfigurationError, ToolInputError


def common_runtime_failure_type_v1(
    error: BaseException | None,
) -> BenchmarkRuntimeFailureTypeV1:
    """Classify failures that have the same meaning in every benchmark runtime."""

    if isinstance(error, HarnessConfigurationError):
        return "evaluator_error"
    if isinstance(error, RuntimeDeadlineExceeded):
        return "timeout"
    if isinstance(error, ProviderRequestError):
        return "provider_error"
    if isinstance(error, (ToolInputError, ModelResponseNoncompletion)):
        return "contract_error"
    if isinstance(error, CodeExecutionSessionTerminated):
        return "participant_session_terminated"
    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        return "cancelled"
    if (
        isinstance(error, subprocess.TimeoutExpired)
        and isinstance(error.cmd, (list, tuple))
        and len(error.cmd) > 1
        and Path(error.cmd[0]).name == "docker"
        and error.cmd[1] in {"exec", "inspect", "image", "rm", "cp"}
    ):
        # These are host control/custody operations, not the participant's
        # bounded interpreter. An unqualified runtime error proves no origin.
        return "evaluator_error"
    return "runtime_error"


__all__ = ["common_runtime_failure_type_v1"]
