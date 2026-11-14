"""Compatibility import for ``trialagentbench_verifier.identity.scorer``."""

import sys as _compat_sys
from typing import TYPE_CHECKING

from trialagentbench_verifier.identity import scorer as _compat_module

if TYPE_CHECKING:
    # isort: off
    from trialagentbench_verifier.identity.scorer import (
        CurrentScorer as CurrentScorer,
        SCORER_HARNESS_SOURCE_SHA256 as SCORER_HARNESS_SOURCE_SHA256,
        SCORER_VERSION as SCORER_VERSION,
        SOURCE_CODE_FORMAT as SOURCE_CODE_FORMAT,
        ScorerIdentityV1 as ScorerIdentityV1,
        current_scorer as current_scorer,
        python_code_bytes_v1 as python_code_bytes_v1,
        require_current_scorer as require_current_scorer,
        require_same_scorer as require_same_scorer,
    )

    # isort: on

_compat_sys.modules[__name__] = _compat_module
