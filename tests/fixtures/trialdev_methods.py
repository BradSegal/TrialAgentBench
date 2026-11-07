"""Executable method controls for a declared two-stratum observational population."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

method = sys.argv[1]
source = pd.read_parquet("data/observational_cohort.parquet")
q = source.groupby("risk").size() / len(source)
rows = []
for asset in ("asset_alpha", "asset_beta"):
    pair = source.loc[source.asset_id.isin((asset, "control"))].copy()
    treated = pair.asset_id.eq(asset).to_numpy(dtype=float)
    x = pair.risk.to_numpy(dtype=float)
    for endpoint, column, unit in (("clinical_outcome", "outcome", "points"), ("safety_risk", "safety", "proportion")):
        y = pair[column].to_numpy(dtype=float)
        if method == "regression_standardisation":
            design = np.column_stack([np.ones(len(pair)), x, treated, treated * x])
            fit = sm.OLS(y, design).fit(cov_type="HC1")
            gradient = np.array([0.0, 0.0, 1.0, float(source.risk.mean())])
            estimate = float(gradient @ fit.params)
            se = float(np.sqrt(gradient @ fit.cov_params() @ gradient))
        else:
            variance = 0.0
            estimate = 0.0
            for stratum, population_share in q.items():
                mask = x == stratum
                a, c = y[mask & (treated == 1)], y[mask & (treated == 0)]
                if method == "matching":
                    # Full exact matching: all pairs within a measured stratum.
                    # Reused participants remain the independent units for SE.
                    estimate += population_share * float((a[:, None] - c[None, :]).mean())
                elif method == "inverse_probability_weighting":
                    propensity = len(a) / (len(a) + len(c))
                    target_weight = population_share / mask.mean()
                    contribution = target_weight * (
                        treated[mask] * y[mask] / propensity - (1 - treated[mask]) * y[mask] / (1 - propensity)
                    )
                    estimate += float(contribution.sum() / len(pair))
                else:
                    raise ValueError("unknown prespecified method")
                variance += population_share**2 * (a.var(ddof=1) / len(a) + c.var(ddof=1) / len(c))
            se = float(np.sqrt(variance))
        if not np.isfinite([estimate, se]).all() or se <= 0:
            raise ValueError("the control lacks finite sampling uncertainty")
        margin = norm.ppf(0.975) * se
        rows.append(
            {
                "asset_id": asset,
                "endpoint": endpoint,
                "unit": unit,
                "estimate": float(estimate),
                "standard_error": se,
                "sampling_interval": {
                    "lower": float(estimate - margin),
                    "upper": float(estimate + margin),
                    "confidence_level": 0.95,
                    "interval_type": "central_frequentist_confidence_interval",
                },
            }
        )
result = {
    "stream": "portfolio_management",
    "action": "nominate",
    "asset_id": "asset_alpha",
    "comparative_analysis": {"analysis_method": method, "estimates": rows},
    "identification_limit_acknowledged": True,
    "preference_dependence_acknowledged": True,
    "uncertainty_statement": "Conditional on declared baseline strata; marginal 95% endpoint intervals only.",
    "rationale": "Evaluate the favorable candidate under the public costs and safety ceiling.",
}
Path("scratch/analysis_result.json").write_text(json.dumps(result, sort_keys=True))
