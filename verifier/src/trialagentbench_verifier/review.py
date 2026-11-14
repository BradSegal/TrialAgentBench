"""Compatibility import for ``trialagentbench_verifier.releases.review``."""

import sys as _compat_sys
from typing import TYPE_CHECKING

from trialagentbench_verifier.releases import review as _compat_module

if TYPE_CHECKING:
    # isort: off
    from trialagentbench_verifier.releases.review import (
        write_review as write_review,
    )

    # isort: on

_compat_sys.modules[__name__] = _compat_module
