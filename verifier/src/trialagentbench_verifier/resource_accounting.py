"""Compatibility import for ``trialagentbench_verifier.development_rules.resource_accounting``."""

import sys as _compat_sys
from typing import TYPE_CHECKING

from trialagentbench_verifier.development_rules import resource_accounting as _compat_module

if TYPE_CHECKING:
    # isort: off
    from trialagentbench_verifier.development_rules.resource_accounting import (
        TrialDevAccountingCase as TrialDevAccountingCase,
        TrialDevResourceAccountingV1 as TrialDevResourceAccountingV1,
        _Commitment as _Commitment,
        _Configuration as _Configuration,
        _Contrast as _Contrast,
        _Enrollment as _Enrollment,
        _Ledger as _Ledger,
        _Patient as _Patient,
        _Pool as _Pool,
        _Preferences as _Preferences,
        _Projection as _Projection,
        _Review as _Review,
        _TRIALDEV_HORIZON_DAYS as _TRIALDEV_HORIZON_DAYS,
        _Terminal as _Terminal,
        reconstruct_trialdev_terminal_value as reconstruct_trialdev_terminal_value,
        verify_trialdev_terminal_value as verify_trialdev_terminal_value,
    )

    # isort: on

_compat_sys.modules[__name__] = _compat_module
