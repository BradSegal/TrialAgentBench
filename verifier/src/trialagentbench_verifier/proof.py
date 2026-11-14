"""Compatibility import for ``trialagentbench_verifier.artifacts.proof``."""

import sys as _compat_sys
from typing import TYPE_CHECKING

from trialagentbench_verifier.artifacts import proof as _compat_module

if TYPE_CHECKING:
    # isort: off
    from trialagentbench_verifier.artifacts.proof import (
        AuthenticatedExecutionFactsV1 as AuthenticatedExecutionFactsV1,
        authenticate_execution_proof as authenticate_execution_proof,
        numerical_score_projection_sha256 as numerical_score_projection_sha256,
        submission_score_projection_sha256 as submission_score_projection_sha256,
    )

    # isort: on

_compat_sys.modules[__name__] = _compat_module
