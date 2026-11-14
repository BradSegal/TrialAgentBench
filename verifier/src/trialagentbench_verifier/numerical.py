"""Compatibility import for ``trialagentbench_verifier.evaluation.numerical``."""

import sys as _compat_sys
from typing import TYPE_CHECKING

from trialagentbench_verifier.evaluation import numerical as _compat_module

if TYPE_CHECKING:
    # isort: off
    from trialagentbench_verifier.evaluation.numerical import (
        NumericalEstimate as NumericalEstimate,
        NumericalReplay as NumericalReplay,
        NumericalReportingContext as NumericalReportingContext,
        _CLARIFIED_HELPERS as _CLARIFIED_HELPERS,
        _UNREPORTED_VALUES as _UNREPORTED_VALUES,
        _validate_target_ids as _validate_target_ids,
        logger as logger,
        read_numerical_table as read_numerical_table,
        replay_numerical_program as replay_numerical_program,
        replay_numerical_submission as replay_numerical_submission,
        stage_reporting_interface as stage_reporting_interface,
    )

    # isort: on

_compat_sys.modules[__name__] = _compat_module
