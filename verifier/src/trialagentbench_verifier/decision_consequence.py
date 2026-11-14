"""Compatibility import for ``trialagentbench_verifier.development_rules.decision_consequence``."""

import sys as _compat_sys
from typing import TYPE_CHECKING

from trialagentbench_verifier.development_rules import decision_consequence as _compat_module

if TYPE_CHECKING:
    # isort: off
    from trialagentbench_verifier.development_rules.decision_consequence import (
        COCHRANE_SNR_MIXTURE_STANDARD_DEVIATIONS as COCHRANE_SNR_MIXTURE_STANDARD_DEVIATIONS,
        COCHRANE_SNR_MIXTURE_WEIGHTS as COCHRANE_SNR_MIXTURE_WEIGHTS,
        CONFIRMATORY_SIZE_MULTIPLES as CONFIRMATORY_SIZE_MULTIPLES,
        DecisionActionV1 as DecisionActionV1,
        DecisionConsequenceV1 as DecisionConsequenceV1,
        DecisionEconomicsV1 as DecisionEconomicsV1,
        DecisionPriorV1 as DecisionPriorV1,
        DecisionReportV1 as DecisionReportV1,
        DiscreteEffectPriorV1 as DiscreteEffectPriorV1,
        ReportCalibrationV1 as ReportCalibrationV1,
        ReportDensityComponentV1 as ReportDensityComponentV1,
        TruthCentredConsequenceV1 as TruthCentredConsequenceV1,
        UntruncatedSNRPriorV1 as UntruncatedSNRPriorV1,
        _Array as _Array,
        _Model as _Model,
        _SUCCESS_CRITICAL_Z as _SUCCESS_CRITICAL_Z,
        _Z95 as _Z95,
        _atom_log_posterior as _atom_log_posterior,
        _benefit as _benefit,
        _calibrated_log_weights as _calibrated_log_weights,
        _calibrated_tail_log_weights as _calibrated_tail_log_weights,
        _calibrated_weights as _calibrated_weights,
        _density_arrays as _density_arrays,
        _fixed_atom_powers as _fixed_atom_powers,
        _normal_atom_log_likelihood as _normal_atom_log_likelihood,
        _posterior as _posterior,
        _posterior_value as _posterior_value,
        _power as _power,
        _prior_arrays as _prior_arrays,
        _report_posterior as _report_posterior,
        _value_curve as _value_curve,
        decision_consequence_v1 as decision_consequence_v1,
        optimal_action_v1 as optimal_action_v1,
        realised_value_v1 as realised_value_v1,
        truth_centred_consequence_v1 as truth_centred_consequence_v1,
    )

    # isort: on

_compat_sys.modules[__name__] = _compat_module
