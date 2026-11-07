"""Analysis route families for TrialEval effect scales."""

from __future__ import annotations

_ROUTE_FAMILY_BY_EFFECT_SCALE = {
    "risk_difference_tau": "risk_difference",
    "standardized_risk_difference_tau_reference": "standardized_risk",
    "rmst_difference_tau": "rmst_contrast",
    "log_hr": "global_cox_ph",
    "hr": "global_cox_ph",
    "hazard_ratio": "global_cox_ph",
    "time_varying_log_hr": "time_varying_cox",
    "piecewise_log_hr_vector": "piecewise_cox",
    "weighted_logrank_test": "weighted_logrank",
    "log_time_ratio": "aft_parametric",
    "cif_difference_tau": "competing_risk",
    "milestone_risk_difference_tau": "milestone_risk",
    "bounds_interval": "partial_identification",
    "non_identification": "qualified_limitation",
}


def route_family_for_effect_scale_v1(effect_scale: str) -> str:
    """Return the analysis family represented by a supported effect scale."""

    try:
        return _ROUTE_FAMILY_BY_EFFECT_SCALE[effect_scale]
    except KeyError as error:
        raise ValueError(f"Unsupported TrialEval effect scale: {effect_scale!r}") from error


__all__ = [
    "route_family_for_effect_scale_v1",
]
