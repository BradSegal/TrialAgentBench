"""Compatibility import for ``trialagentbench_verifier.artifacts.grading_key``."""

import sys as _compat_sys
from typing import TYPE_CHECKING

from trialagentbench_verifier.artifacts import grading_key as _compat_module

if TYPE_CHECKING:
    # isort: off
    from trialagentbench_verifier.artifacts.grading_key import (
        GRADING_KEY_MANIFEST as GRADING_KEY_MANIFEST,
        GradingKeyArtifactV1 as GradingKeyArtifactV1,
        GradingKeyManifestV1 as GradingKeyManifestV1,
        TRIALDEV_WORLD_STRATA as TRIALDEV_WORLD_STRATA,
        TRIALEVAL_GRADING_KEY_RECORDS as TRIALEVAL_GRADING_KEY_RECORDS,
        TrialDevWorldStrataV1 as TrialDevWorldStrataV1,
        TrialDevWorldStratumV1 as TrialDevWorldStratumV1,
        _Model as _Model,
        _RELEASE_SCHEMAS as _RELEASE_SCHEMAS,
        _SCHEMAS as _SCHEMAS,
        _read_manifest as _read_manifest,
        _verify_file as _verify_file,
        grading_key_lines as grading_key_lines,
        grading_key_manifest_bytes as grading_key_manifest_bytes,
        read_trialdev_world_strata as read_trialdev_world_strata,
        read_trialeval_grading_key as read_trialeval_grading_key,
        trialdev_world_strata_bytes as trialdev_world_strata_bytes,
        verify_grading_key as verify_grading_key,
    )

    # isort: on

_compat_sys.modules[__name__] = _compat_module
