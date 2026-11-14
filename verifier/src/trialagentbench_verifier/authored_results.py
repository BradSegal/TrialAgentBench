"""Compatibility import for ``trialagentbench_verifier.authored.authored_results``."""

import sys as _compat_sys
from typing import TYPE_CHECKING

from trialagentbench_verifier.authored import authored_results as _compat_module

if TYPE_CHECKING:
    # isort: off
    from trialagentbench_verifier.authored.authored_results import (
        TrialDevAnalysisRun as TrialDevAnalysisRun,
        TrialDevNumericalRecovery as TrialDevNumericalRecovery,
        TrialDevProgrammeEvaluation as TrialDevProgrammeEvaluation,
        TrialDevProgrammeRun as TrialDevProgrammeRun,
        TrialDevStudyEvaluation as TrialDevStudyEvaluation,
        TrialDevStudyReport as TrialDevStudyReport,
        mean_numerical_loss as mean_numerical_loss,
        numerical_target_statuses as numerical_target_statuses,
    )

    # isort: on

_compat_sys.modules[__name__] = _compat_module
