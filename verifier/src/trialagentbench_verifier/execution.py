"""Compatibility import for ``trialagentbench_verifier.runtime.execution``."""

import sys as _compat_sys
from typing import TYPE_CHECKING

from trialagentbench_verifier.runtime import execution as _compat_module

if TYPE_CHECKING:
    # isort: off
    from trialagentbench_verifier.runtime.execution import (
        ExecutionSubmissionV1 as ExecutionSubmissionV1,
        _Material as _Material,
        _Model as _Model,
        _PROGRAM_NAMES as _PROGRAM_NAMES,
        _Task as _Task,
        _copy_regular as _copy_regular,
        _proof as _proof,
        _safe_relative_path as _safe_relative_path,
        _stage_result_schema as _stage_result_schema,
        _stage_submission as _stage_submission,
        _stage_task_inputs as _stage_task_inputs,
        _submission_root as _submission_root,
        _task as _task,
        execute_submission as execute_submission,
        execute_submissions as execute_submissions,
    )

    # isort: on

_compat_sys.modules[__name__] = _compat_module
