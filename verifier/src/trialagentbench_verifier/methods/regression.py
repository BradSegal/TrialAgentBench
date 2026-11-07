"""Thin statsmodels adapters and explicit score-dependent weighting transforms."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import statsmodels.api as sm


def _linear_contrast(
    outcome: npt.NDArray[np.float64],
    treatment: npt.NDArray[np.int_],
    *,
    covariates: npt.NDArray[np.float64] | None = None,
    clusters: npt.NDArray[np.generic] | None = None,
) -> tuple[float, float]:
    """Return an adjusted treatment contrast with HC1 or cluster covariance."""

    if outcome.ndim != 1 or treatment.ndim != 1 or outcome.shape != treatment.shape:
        raise ValueError("outcome and treatment must be aligned one-dimensional arrays")
    if outcome.size == 0 or not np.isfinite(outcome).all() or not np.isin(treatment, (0, 1)).all():
        raise ValueError("linear-contrast inputs must be finite with binary treatment")
    columns = [np.ones(outcome.size), treatment.astype(float)]
    if covariates is not None:
        matrix = np.asarray(covariates, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        if matrix.shape[0] != outcome.size or not np.isfinite(matrix).all():
            raise ValueError("covariates must be finite and row-aligned")
        columns.extend(matrix.T)
    design = np.column_stack(columns)
    model = sm.OLS(outcome, design)
    if clusters is None:
        fit = model.fit(cov_type="HC1")
    else:
        group = np.asarray(clusters)
        if group.ndim != 1 or group.shape != outcome.shape or np.unique(group).size < 2:
            raise ValueError("cluster covariance requires at least two aligned clusters")
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": group})
    return float(fit.params[1]), float(fit.bse[1])


def _weighted_mean_difference(
    outcome: npt.NDArray[np.float64],
    treatment: npt.NDArray[np.int_],
    weights: npt.NDArray[np.float64],
) -> float:
    """Return the declared weighted arm-mean difference."""

    if outcome.ndim != 1 or treatment.ndim != 1 or weights.ndim != 1:
        raise ValueError("weighted-contrast inputs must be one-dimensional")
    if not outcome.shape == treatment.shape == weights.shape or outcome.size == 0:
        raise ValueError("weighted-contrast inputs must be non-empty and aligned")
    if not np.isfinite(outcome).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("outcomes and strictly positive weights must be finite")
    if not np.isin(treatment, (0, 1)).all() or np.unique(treatment).size != 2:
        raise ValueError("weighted contrast requires both binary treatment arms")
    treated = treatment == 1
    return float(np.average(outcome[treated], weights=weights[treated])) - float(
        np.average(outcome[~treated], weights=weights[~treated])
    )


__all__ = ["_linear_contrast", "_weighted_mean_difference"]
