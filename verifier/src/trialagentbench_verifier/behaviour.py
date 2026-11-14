"""Compatibility import for ``trialagentbench_verifier.reporting.behaviour``."""

import sys as _compat_sys
from typing import TYPE_CHECKING

from trialagentbench_verifier.reporting import behaviour as _compat_module

if TYPE_CHECKING:
    # isort: off
    from trialagentbench_verifier.reporting.behaviour import (
        _Manifest as _Manifest,
        _Row as _Row,
        _Table as _Table,
        _svg as _svg,
        reproduce_behaviour_figures as reproduce_behaviour_figures,
    )

    # isort: on

_compat_sys.modules[__name__] = _compat_module
