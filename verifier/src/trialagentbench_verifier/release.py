"""Compatibility import for ``trialagentbench_verifier.releases.release``."""

import sys as _compat_sys
from typing import TYPE_CHECKING

from trialagentbench_verifier.releases import release as _compat_module

if TYPE_CHECKING:
    # isort: off
    from trialagentbench_verifier.releases.release import (
        EvaluationReleaseError as EvaluationReleaseError,
        ScoreSubmissionV1 as ScoreSubmissionV1,
        _PORTABLE_GRADED_ITEM_PATH_FIELDS as _PORTABLE_GRADED_ITEM_PATH_FIELDS,
        _QUALIFICATION_NAMES as _QUALIFICATION_NAMES,
        _object_checksum as _object_checksum,
        _participant_identity as _participant_identity,
        _portable_graded_item as _portable_graded_item,
        _portable_graded_item_sha256 as _portable_graded_item_sha256,
        _qualification_bundle as _qualification_bundle,
        _qualification_checks as _qualification_checks,
        _release_identity as _release_identity,
        _report as _report,
        _scoped_artifacts as _scoped_artifacts,
        _score_materialized_submission as _score_materialized_submission,
        _score_trialdev as _score_trialdev,
        _score_trialeval as _score_trialeval,
        _standard_component_digest as _standard_component_digest,
        _verify_artifact_identity as _verify_artifact_identity,
        _verify_expected_input as _verify_expected_input,
        _verify_materialized_integrity as _verify_materialized_integrity,
        _verify_materialized_release as _verify_materialized_release,
        _verify_paired_participant_archive_identity as _verify_paired_participant_archive_identity,
        _verify_participant_release as _verify_participant_release,
        _verify_role_manifest as _verify_role_manifest,
        score_submission as score_submission,
        score_submissions as score_submissions,
        verify_release as verify_release,
    )

    # isort: on

_compat_sys.modules[__name__] = _compat_module
