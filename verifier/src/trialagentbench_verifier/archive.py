"""Compatibility import for ``trialagentbench_verifier.artifacts.archive``."""

import sys as _compat_sys
from typing import TYPE_CHECKING

from trialagentbench_verifier.artifacts import archive as _compat_module

if TYPE_CHECKING:
    # isort: off
    from trialagentbench_verifier.artifacts.archive import (
        ArchivePath as ArchivePath,
        _OPEN as _OPEN,
        _archive as _archive,
        _asset_scope as _asset_scope,
        _directories as _directories,
        _members as _members,
        opened_release as opened_release,
    )

    # isort: on

_compat_sys.modules[__name__] = _compat_module
