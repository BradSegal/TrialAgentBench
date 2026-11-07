"""Thin lifelines adapters for exact portable survival comparisons."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time


def _kaplan_meier_summary(
    durations: npt.NDArray[np.float64],
    events: npt.NDArray[np.int_],
    *,
    horizon: float,
) -> tuple[float, float, float]:
    """Return survival, risk, and restricted mean survival at one horizon."""

    if durations.ndim != 1 or events.ndim != 1 or durations.shape != events.shape:
        raise ValueError("durations and events must be aligned one-dimensional arrays")
    if durations.size == 0 or not np.isfinite(durations).all() or (durations < 0).any():
        raise ValueError("durations must be non-empty, finite, and non-negative")
    if not np.isin(events, (0, 1)).all() or not np.isfinite(horizon) or horizon <= 0:
        raise ValueError("events must be binary and horizon must be positive")
    fit = KaplanMeierFitter().fit(durations, event_observed=events)
    survival = float(fit.predict(horizon))
    rmst = float(restricted_mean_survival_time(fit, t=horizon))
    return survival, 1.0 - survival, rmst


def _cox_log_hazard_ratio(
    durations: npt.NDArray[np.float64],
    events: npt.NDArray[np.int_],
    treatment: npt.NDArray[np.int_],
) -> tuple[float, float]:
    """Return the Cox treatment log-hazard ratio and model standard error."""

    if durations.ndim != 1 or events.ndim != 1 or treatment.ndim != 1:
        raise ValueError("Cox inputs must be one-dimensional")
    if not durations.shape == events.shape == treatment.shape or durations.size == 0:
        raise ValueError("Cox inputs must be non-empty and aligned")
    if not np.isfinite(durations).all() or (durations < 0).any():
        raise ValueError("Cox durations must be finite and non-negative")
    if not np.isin(events, (0, 1)).all() or not np.isin(treatment, (0, 1)).all():
        raise ValueError("Cox event and treatment indicators must be binary")
    frame = pd.DataFrame({"duration": durations, "event": events, "treatment": treatment})
    fit = CoxPHFitter().fit(frame, duration_col="duration", event_col="event")
    return float(fit.params_["treatment"]), float(fit.standard_errors_["treatment"])


__all__ = ["_cox_log_hazard_ratio", "_kaplan_meier_summary"]
