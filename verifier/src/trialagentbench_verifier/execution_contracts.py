"""Compatibility import for ``trialagentbench_verifier.runtime.execution_contracts``."""

import sys as _compat_sys
from typing import TYPE_CHECKING

from trialagentbench_verifier.runtime import execution_contracts as _compat_module

if TYPE_CHECKING:
    # isort: off
    from trialagentbench_verifier.runtime.execution_contracts import (
        AnalysisLanguage as AnalysisLanguage,
        AnalysisMode as AnalysisMode,
        CodeExecutionLimitsV1 as CodeExecutionLimitsV1,
        CodeExecutionResultV1 as CodeExecutionResultV1,
        CodeExecutionSession as CodeExecutionSession,
        DEFAULT_COMPLETION_TOKENS as DEFAULT_COMPLETION_TOKENS,
        EXECUTOR_SLOTS_PER_TASK as EXECUTOR_SLOTS_PER_TASK,
        ExecutorEnvironmentV1 as ExecutorEnvironmentV1,
        ExecutorPackageV1 as ExecutorPackageV1,
        ExecutorSupportFileV1 as ExecutorSupportFileV1,
        RELEASE_EXECUTION_LIMITS_V1 as RELEASE_EXECUTION_LIMITS_V1,
        ReplayInputArtifactV1 as ReplayInputArtifactV1,
        ReportingInterfaceV1 as ReportingInterfaceV1,
        TRIALDEV_MAXIMUM_TURNS_PER_STEP as TRIALDEV_MAXIMUM_TURNS_PER_STEP,
        TRIALEVAL_MAXIMUM_TURNS as TRIALEVAL_MAXIMUM_TURNS,
        TRIALEVAL_REPLAY_ABS_TOLERANCE_V1 as TRIALEVAL_REPLAY_ABS_TOLERANCE_V1,
        TRIALEVAL_REPLAY_REL_TOLERANCE_V1 as TRIALEVAL_REPLAY_REL_TOLERANCE_V1,
        analysis_languages as analysis_languages,
    )

    # isort: on

_compat_sys.modules[__name__] = _compat_module
