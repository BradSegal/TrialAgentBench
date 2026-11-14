"""Compatibility import for ``trialagentbench_verifier.artifacts.io``."""

import sys as _compat_sys
from typing import TYPE_CHECKING

from trialagentbench_verifier.artifacts import io as _compat_module

if TYPE_CHECKING:
    # isort: off
    from trialagentbench_verifier.artifacts.io import (
        MAX_JSON_BYTES as MAX_JSON_BYTES,
        MAX_RELEASE_ARCHIVE_BYTES as MAX_RELEASE_ARCHIVE_BYTES,
        MAX_RELEASE_ARCHIVE_MEMBERS as MAX_RELEASE_ARCHIVE_MEMBERS,
        MAX_RELEASE_MEMBER_BYTES as MAX_RELEASE_MEMBER_BYTES,
        MAX_RELEASE_UNCOMPRESSED_BYTES as MAX_RELEASE_UNCOMPRESSED_BYTES,
        VerificationInputError as VerificationInputError,
        _ACTIVE_ASSETS as _ACTIVE_ASSETS,
        _ASSET_SCOPE_OWNER as _ASSET_SCOPE_OWNER,
        _AssetViews as _AssetViews,
        _extract_release as _extract_release,
        _safe_member as _safe_member,
        bounded_archive_file as bounded_archive_file,
        canonical_json_bytes as canonical_json_bytes,
        canonical_sha256 as canonical_sha256,
        materialized_release as materialized_release,
        read_json_lines as read_json_lines,
        read_json_object as read_json_object,
        registered_asset_scope as registered_asset_scope,
        release_archive as release_archive,
        release_asset_scope as release_asset_scope,
        release_assets as release_assets,
        sha256_file as sha256_file,
        trialdev_task_run_paths as trialdev_task_run_paths,
    )

    # isort: on

_compat_sys.modules[__name__] = _compat_module
