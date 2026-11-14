"""Compatibility import for ``trialagentbench_verifier.releases.external_qualification``."""

import sys as _compat_sys
from typing import TYPE_CHECKING

from trialagentbench_verifier.releases import external_qualification as _compat_module

if TYPE_CHECKING:
    # isort: off
    from trialagentbench_verifier.releases.external_qualification import (
        EXTERNAL_QUALIFICATION_NAME as EXTERNAL_QUALIFICATION_NAME,
        ExternalTrialDevQualificationV1 as ExternalTrialDevQualificationV1,
        QUALIFICATION_PENDING as QUALIFICATION_PENDING,
        read_external_qualification as read_external_qualification,
    )

    # isort: on

_compat_sys.modules[__name__] = _compat_module
