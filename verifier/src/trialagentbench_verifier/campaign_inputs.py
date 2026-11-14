"""Compatibility import for ``trialagentbench_verifier.evaluation.campaign_inputs``."""

import sys as _compat_sys
from typing import TYPE_CHECKING

from trialagentbench_verifier.evaluation import campaign_inputs as _compat_module

if TYPE_CHECKING:
    # isort: off
    from trialagentbench_verifier.evaluation.campaign_inputs import (
        DevelopmentArchiveV1 as DevelopmentArchiveV1,
        DevelopmentCampaignInputsV1 as DevelopmentCampaignInputsV1,
        DevelopmentExecutorV1 as DevelopmentExecutorV1,
        _Model as _Model,
        authenticate_development_inputs as authenticate_development_inputs,
        verify_development_evaluator as verify_development_evaluator,
    )

    # isort: on

_compat_sys.modules[__name__] = _compat_module
