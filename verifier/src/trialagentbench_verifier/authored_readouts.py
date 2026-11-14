"""Compatibility import for ``trialagentbench_verifier.authored.authored_readouts``."""

import sys as _compat_sys
from typing import TYPE_CHECKING

from trialagentbench_verifier.authored import authored_readouts as _compat_module

if TYPE_CHECKING:
    # isort: off
    from trialagentbench_verifier.authored.authored_readouts import (
        _Arm as _Arm,
        _Artifact as _Artifact,
        _Assignment as _Assignment,
        _Commitment as _Commitment,
        _Partition as _Partition,
        _Patient as _Patient,
        _Pool as _Pool,
        _Projection as _Projection,
        _Readout as _Readout,
        _record_projection as _record_projection,
        verify_trialdev_readout_patients as verify_trialdev_readout_patients,
        verify_trialdev_readout_records as verify_trialdev_readout_records,
    )

    # isort: on

_compat_sys.modules[__name__] = _compat_module
