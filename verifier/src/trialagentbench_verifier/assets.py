"""Compatibility import for ``trialagentbench_verifier.artifacts.assets``."""

import sys as _compat_sys
from typing import TYPE_CHECKING

from trialagentbench_verifier.artifacts import assets as _compat_module

if TYPE_CHECKING:
    # isort: off
    from trialagentbench_verifier.artifacts.assets import (
        AssetFootprint as AssetFootprint,
        AssetMember as AssetMember,
        AssetShard as AssetShard,
        AssetTask as AssetTask,
        ReleaseAssetScope as ReleaseAssetScope,
        ReleaseAssets as ReleaseAssets,
        TRAINING_CANARY as TRAINING_CANARY,
        _KeyRecord as _KeyRecord,
        _Model as _Model,
        _ReleaseIdentity as _ReleaseIdentity,
        _ReleasedArtifact as _ReleasedArtifact,
        _RoleInventory as _RoleInventory,
        _TaskProjection as _TaskProjection,
        _group as _group,
        _link_payload as _link_payload,
        _verify_shard as _verify_shard,
        _write_payload as _write_payload,
        asset_footprint as asset_footprint,
        materialize_asset_selection as materialize_asset_selection,
        pack_release_assets as pack_release_assets,
        read_asset_catalog as read_asset_catalog,
    )

    # isort: on

_compat_sys.modules[__name__] = _compat_module
