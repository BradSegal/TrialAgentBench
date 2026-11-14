"""Compatibility import for ``trialagentbench_verifier.authored.authored_scoring``."""

import sys as _compat_sys
from typing import TYPE_CHECKING

from trialagentbench_verifier.authored import authored_scoring as _compat_module

if TYPE_CHECKING:
    # isort: off
    from trialagentbench_verifier.authored.authored_scoring import (
        TrialEvalTargetV1 as TrialEvalTargetV1,
        _ReferencePool as _ReferencePool,
        _Study as _Study,
        _contrast as _contrast,
        compute_numerical_recovery as compute_numerical_recovery,
        verify_numerical_recovery as verify_numerical_recovery,
        verify_trialdev_prospective_scores as verify_trialdev_prospective_scores,
    )

    # isort: on

_compat_sys.modules[__name__] = _compat_module
