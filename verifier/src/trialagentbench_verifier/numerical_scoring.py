"""Compatibility import for ``trialagentbench_verifier.evaluation.numerical_scoring``."""

import sys as _compat_sys
from typing import TYPE_CHECKING

from trialagentbench_verifier.evaluation import numerical_scoring as _compat_module

if TYPE_CHECKING:
    # isort: off
    from trialagentbench_verifier.evaluation.numerical_scoring import (
        NUMERICAL_LOSS_CONTEXT as NUMERICAL_LOSS_CONTEXT,
        NUMERICAL_LOSS_PRECISION as NUMERICAL_LOSS_PRECISION,
        NUMERICAL_LOSS_ROUNDING as NUMERICAL_LOSS_ROUNDING,
        bounded_recovery as bounded_recovery,
        bounded_recovery_summary as bounded_recovery_summary,
        finite_diagnostic as finite_diagnostic,
        interval_score as interval_score,
        numerical_loss as numerical_loss,
        round_numerical_loss as round_numerical_loss,
    )

    # isort: on

_compat_sys.modules[__name__] = _compat_module
