"""Compatibility import for ``trialagentbench_verifier.runtime.executor``."""

import sys as _compat_sys
from typing import TYPE_CHECKING

from trialagentbench_verifier.runtime import executor as _compat_module

if TYPE_CHECKING:
    # isort: off
    from trialagentbench_verifier.runtime.executor import (
        CodeExecutionSessionTerminated as CodeExecutionSessionTerminated,
        DEFAULT_EXECUTOR_IMAGE as DEFAULT_EXECUTOR_IMAGE,
        DOCKER_CONTROL_TIMEOUT_SECONDS as DOCKER_CONTROL_TIMEOUT_SECONDS,
        DockerCodeSession as DockerCodeSession,
        ExecutorUnavailableError as ExecutorUnavailableError,
        REQUIRED_ANALYSIS_PACKAGES as REQUIRED_ANALYSIS_PACKAGES,
        REQUIRED_R_ANALYSIS_PACKAGES as REQUIRED_R_ANALYSIS_PACKAGES,
        _CAPACITY_OWNER as _CAPACITY_OWNER,
        _CAPACITY_QUEUES as _CAPACITY_QUEUES,
        _CAPACITY_SLOTS as _CAPACITY_SLOTS,
        _DOCKER_CLIENT_VARIABLES as _DOCKER_CLIENT_VARIABLES,
        _ENVIRONMENT_PROBE as _ENVIRONMENT_PROBE,
        _EXCLUSIVE_CAPACITY as _EXCLUSIVE_CAPACITY,
        _IMAGE_ID as _IMAGE_ID,
        _PARQUET_BRIDGE as _PARQUET_BRIDGE,
        _PYTHON_RUNNER as _PYTHON_RUNNER,
        _RESULT_PREFIX as _RESULT_PREFIX,
        _SESSION_CAPACITY as _SESSION_CAPACITY,
        _VERIFIED_SUBSTITUTES as _VERIFIED_SUBSTITUTES,
        _WORKER as _WORKER,
        _WorkerResultV1 as _WorkerResultV1,
        _docker_command as _docker_command,
        _docker_environment as _docker_environment,
        _executor_substitute as _executor_substitute,
        _probe_executor_image as _probe_executor_image,
        _validate_evidence_tree as _validate_evidence_tree,
        _validate_scratch_tree as _validate_scratch_tree,
        analysis_entrypoint_code as analysis_entrypoint_code,
        analysis_project_artifacts as analysis_project_artifacts,
        analysis_project_files as analysis_project_files,
        executor_session_capacity as executor_session_capacity,
        fresh_replay_workspace as fresh_replay_workspace,
        offline_executor_capacity as offline_executor_capacity,
        resolve_executor_environment as resolve_executor_environment,
        resolve_trusted_executor_environment as resolve_trusted_executor_environment,
        verified_replay_inputs as verified_replay_inputs,
    )

    # isort: on

_compat_sys.modules[__name__] = _compat_module
