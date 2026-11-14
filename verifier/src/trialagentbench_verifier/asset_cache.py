"""Compatibility import for ``trialagentbench_verifier.artifacts.asset_cache``."""

import sys as _compat_sys
from typing import TYPE_CHECKING

from trialagentbench_verifier.artifacts import asset_cache as _compat_module

if TYPE_CHECKING:
    # isort: off
    from trialagentbench_verifier.artifacts.asset_cache import (
        AssetCacheFootprint as AssetCacheFootprint,
        _directory as _directory,
        _download as _download,
        _lock as _lock,
        _selection_key as _selection_key,
        _verify_view as _verify_view,
        acquire_asset_selection as acquire_asset_selection,
        cache_footprint as cache_footprint,
        cached_asset_selection as cached_asset_selection,
    )

    # isort: on

_compat_sys.modules[__name__] = _compat_module
