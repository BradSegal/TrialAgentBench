"""Compatibility import for ``trialagentbench_verifier.reporting.layer_calibration``."""

import sys as _compat_sys
from typing import TYPE_CHECKING

from trialagentbench_verifier.reporting import layer_calibration as _compat_module

if TYPE_CHECKING:
    # isort: off
    from trialagentbench_verifier.reporting.layer_calibration import (
        LAYER_CALIBRATION_RESOURCE as LAYER_CALIBRATION_RESOURCE,
        LAYER_CALIBRATION_SHA256 as LAYER_CALIBRATION_SHA256,
        LayerCalibrationFileV1 as LayerCalibrationFileV1,
        QuestionReportCalibrationV1 as QuestionReportCalibrationV1,
        _checked as _checked,
        load_layer_calibration_file as load_layer_calibration_file,
        load_public_layer_calibration as load_public_layer_calibration,
    )

    # isort: on

_compat_sys.modules[__name__] = _compat_module
