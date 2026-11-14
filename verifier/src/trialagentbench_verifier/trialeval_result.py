"""Compatibility import for ``trialagentbench_verifier.evaluation.trialeval_result``."""

import sys as _compat_sys
from typing import TYPE_CHECKING

from trialagentbench_verifier.evaluation import trialeval_result as _compat_module

if TYPE_CHECKING:
    # isort: off
    from trialagentbench_verifier.evaluation.trialeval_result import (
        TrialEvalProgramConclusionV2 as TrialEvalProgramConclusionV2,
        TrialEvalProgramEffectMeasureV2 as TrialEvalProgramEffectMeasureV2,
        TrialEvalProgramEstimateV2 as TrialEvalProgramEstimateV2,
        TrialEvalProgramPointEstimateV2 as TrialEvalProgramPointEstimateV2,
        TrialEvalProgramResultV2 as TrialEvalProgramResultV2,
        TrialEvalProgramSamplingIntervalV2 as TrialEvalProgramSamplingIntervalV2,
    )

    # isort: on

_compat_sys.modules[__name__] = _compat_module
