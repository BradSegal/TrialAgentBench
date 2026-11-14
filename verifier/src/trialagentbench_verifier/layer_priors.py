"""Compatibility import for ``trialagentbench_verifier.reporting.layer_priors``."""

import sys as _compat_sys
from typing import TYPE_CHECKING

from trialagentbench_verifier.reporting import layer_priors as _compat_module

if TYPE_CHECKING:
    # isort: off
    from trialagentbench_verifier.reporting.layer_priors import (
        LAYER_PRIOR_RESOURCE as LAYER_PRIOR_RESOURCE,
        LAYER_PRIOR_SHA256 as LAYER_PRIOR_SHA256,
        LayerPriorFileV1 as LayerPriorFileV1,
        MeanZMultiplierMapV1 as MeanZMultiplierMapV1,
        MeanZPriorV1 as MeanZPriorV1,
        PopulationEffectCurveV1 as PopulationEffectCurveV1,
        PriorSourceRecordV1 as PriorSourceRecordV1,
        QuestionLayerPriorV1 as QuestionLayerPriorV1,
        _Model as _Model,
        _checked as _checked,
        _public_file as _public_file,
        _scoring_priors as _scoring_priors,
        load_layer_prior as load_layer_prior,
        load_layer_prior_file as load_layer_prior_file,
    )

    # isort: on

_compat_sys.modules[__name__] = _compat_module
