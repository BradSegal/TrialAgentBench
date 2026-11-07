"""Small analytic transforms for released trial-design checks."""

from __future__ import annotations

import math

from scipy.stats import norm


def _cluster_design_effect(mean_cluster_size: float, intracluster_correlation: float) -> float:
    """Return the equal-size cluster design effect."""

    if not math.isfinite(mean_cluster_size) or mean_cluster_size < 1:
        raise ValueError("mean cluster size must be finite and at least one")
    if not math.isfinite(intracluster_correlation) or not 0 <= intracluster_correlation <= 1:
        raise ValueError("intracluster correlation must lie in [0, 1]")
    return 1.0 + (mean_cluster_size - 1.0) * intracluster_correlation


def _obrien_fleming_boundary(information_fraction: float, *, two_sided_alpha: float) -> float:
    """Return the canonical two-sided O'Brien-Fleming Z boundary."""

    if not math.isfinite(information_fraction) or not 0 < information_fraction <= 1:
        raise ValueError("information fraction must lie in (0, 1]")
    if not math.isfinite(two_sided_alpha) or not 0 < two_sided_alpha < 1:
        raise ValueError("two-sided alpha must lie in (0, 1)")
    return float(norm.ppf(1.0 - two_sided_alpha / 2.0) / math.sqrt(information_fraction))


__all__ = ["_cluster_design_effect", "_obrien_fleming_boundary"]
