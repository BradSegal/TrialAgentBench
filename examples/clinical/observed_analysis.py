"""The demonstration's scripted analyst.

The TrialEval task runs the statistical analysis plan's prespecified primary model. The TrialDev readouts and the
observational cohort use naive observed contrasts for teaching: no confounding or missingness adjustment.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from interface.reporting import report, targets  # type: ignore[import-not-found]
from scipy.stats import t


def contrast(active: pd.Series, control: pd.Series) -> dict[str, float]:
    """Calculate an observed mean difference and a two-sided Welch interval.

    Binary outcomes use the Agresti-Caffo interval (one event and one non-event added to each arm),
    which stays defined when an arm has no events; its estimate is the adjusted difference.
    """
    active, control = active.dropna(), control.dropna()
    if min(len(active), len(control)) < 2:
        raise ValueError("The teaching contrast requires two observed participants per arm")
    if set(active) | set(control) <= {0.0, 1.0}:
        p_a, p_c = (active.sum() + 1) / (len(active) + 2), (control.sum() + 1) / (len(control) + 2)
        estimate = float(p_a - p_c)
        half = 1.959964 * math.sqrt(p_a * (1 - p_a) / (len(active) + 2) + p_c * (1 - p_c) / (len(control) + 2))
        return dict(estimate=estimate, lower=estimate - half, upper=estimate + half)
    variance_a, variance_c = active.var(ddof=1) / len(active), control.var(ddof=1) / len(control)
    estimate = float(active.mean() - control.mean())
    variance = variance_a + variance_c
    if variance == 0:
        raise ValueError("A degenerate observed sample cannot support the teaching interval")
    degrees = variance**2 / (variance_a**2 / (len(active) - 1) + variance_c**2 / (len(control) - 1))
    margin = float(t.ppf(0.975, degrees)) * math.sqrt(variance)
    return dict(estimate=estimate, lower=estimate - margin, upper=estimate + margin)


assert not Path("evaluation").exists(), "Evaluator data must not be participant-visible"
readouts = list(Path("evidence").glob("*/study.json"))
rows = []
if readouts:
    if len(readouts) != 1:
        raise ValueError("This teaching analysis supports exactly one acquired study")
    path = readouts[0]
    study = json.loads(path.read_text())
    root = Path("analysis_ready") / path.parent.name
    patients = pd.read_parquet(root / "participants.parquet")
    measurements = pd.read_parquet(root / "measurements.parquet")
    elapsed = measurements.occurrence_time - measurements.enrollment_day
    clinical = measurements.loc[np.isclose(elapsed, study["follow_up_days"]) & measurements.observed].copy()
    safety = pd.read_parquet(root / "safety_events.parquet")
    elapsed = safety.occurrence_time - safety.enrollment_day
    cases = safety.loc[elapsed.between(0, study["follow_up_days"]), "participant_id"]
    mature = patients.loc[study["readout_day"] - patients.enrollment_day >= study["follow_up_days"]].copy()
    mature["observed_outcome"] = mature.participant_id.isin(cases).astype(float)
    for target in study["targets"]:
        data = clinical if target["endpoint"] == "clinical_outcome" else mature
        active = data.loc[data.arm_id == target["arm_id"], "observed_outcome"]
        control = data.loc[data.arm_id == "control", "observed_outcome"]
        rows.append(dict(target_id=target["target_id"], **contrast(active, control)))
        print(f"{target['target_id']}: {len(active)} active, {len(control)} control observations")
elif Path("data/observational_cohort.parquet").is_file():
    frame = pd.read_parquet("data/observational_cohort.parquet")
    for target in targets():
        asset, endpoint = target.split(":")
        column = (
            "das28_score_end_of_follow_up" if endpoint == "clinical_outcome" else "infection_aesi_end_of_follow_up"
        )
        active = frame.loc[frame.asset_id == asset, column]
        control = frame.loc[frame.asset_id == "control", column]
        rows.append(dict(target_id=target, **contrast(active, control)))
else:
    # The plan's prespecified primary model: a Gaussian marginal GEE with an intercept, the randomized-arm
    # indicator, continuous occurrence_time in days and their interaction, an exchangeable working correlation and
    # participant-clustered sandwich uncertainty, evaluated at Day 365.
    frame = pd.read_parquet("data/analysis.parquet")
    frame = frame.loc[frame.observed & frame.urine_albumin_creatinine_ratio.notna()].copy()
    frame = frame.sort_values(["participant_id", "occurrence_time"])
    treatment = frame.assigned_arm.eq("active").astype(float)
    days = frame.occurrence_time.astype(float)
    design = np.column_stack([np.ones(len(frame)), treatment, days, treatment * days])
    fitted = sm.GEE(
        frame.urine_albumin_creatinine_ratio,
        design,
        groups=frame.participant_id,
        family=sm.families.Gaussian(),
        cov_struct=sm.cov_struct.Exchangeable(),
    ).fit()
    comparison = fitted.t_test([0.0, 1.0, 0.0, 365.0])
    lower, upper = comparison.conf_int(alpha=0.05)[0]
    for target in targets():
        rows.append(
            dict(target_id=target, estimate=float(comparison.effect.item()), lower=float(lower), upper=float(upper))
        )
    print(
        f"Day365 GEE treatment-by-time contrast (exchangeable, sandwich): {frame.participant_id.nunique()} "
        f"participants, {len(frame)} observations"
    )
report(rows)
print(json.dumps(rows, indent=2))
if not Path("data/analysis.parquet").is_file() or readouts:
    print(
        "Naive observed contrasts: no confounding adjustment, missingness correction, or population standardization."
    )
