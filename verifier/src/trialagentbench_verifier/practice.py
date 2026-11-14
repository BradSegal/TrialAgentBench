"""Compatibility import for ``trialagentbench_verifier.releases.practice``."""

import sys as _compat_sys
from typing import TYPE_CHECKING

from trialagentbench_verifier.releases import practice as _compat_module

if TYPE_CHECKING:
    # isort: off
    from trialagentbench_verifier.releases.practice import (
        _Case as _Case,
        _Fixture as _Fixture,
        _IntervalCase as _IntervalCase,
        _LossCase as _LossCase,
        _Model as _Model,
        _UnitCase as _UnitCase,
        _observed as _observed,
        verify_practice_suite as verify_practice_suite,
    )

    # isort: on

_compat_sys.modules[__name__] = _compat_module
