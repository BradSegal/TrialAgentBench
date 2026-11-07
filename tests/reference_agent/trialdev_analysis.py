# mypy: ignore-errors
"""Reference TrialDev analysis written into scratch/ by the scripted reference agent.

Initial targets (``asset:endpoint``) use regression standardisation of the observational
cohort over every recorded pretreatment characteristic. Readout targets use the readout
folder whose study.json requests the current targets: complete-case mean differences at
the reporting follow-up day for the clinical endpoint and mature-participant risk
differences for safety. Lines starting with FRICTION record analysis judgements.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from interface.reporting import report, targets  # type: ignore[import-not-found]
from scipy.stats import norm

Z = float(norm.ppf(0.975))


def friction(text: str) -> None:
    print(f"FRICTION: {text}")


def interval(estimate: float, se: float) -> dict[str, float]:
    if not np.isfinite([estimate, se]).all() or se <= 0:
        raise ValueError("Estimate lacks finite sampling uncertainty")
    return dict(estimate=float(estimate), lower=float(estimate - Z * se), upper=float(estimate + Z * se))


def observational(requested: list[str]) -> list[dict[str, object]]:
    root = Path("data/initial")
    if root.is_dir():
        frame = pd.read_parquet(root / "participants.parquet")
        for name in ("treatments", "clinical_assessments", "safety_assessments"):
            frame = frame.merge(pd.read_parquet(root / f"{name}.parquet"), on="participant_id", how="left")
    else:
        frame = pd.read_parquet("data/observational_cohort.parquet")
    dictionary = pd.read_csv("documents/observational_data_dictionary.csv")
    roles = dictionary.set_index("variable").role
    outcomes = [v for v in roles.index if roles[v] == "outcome"]
    clinical = next(v for v in outcomes if v in frame and not v.endswith("_aesi_end_of_follow_up"))
    safety = next(v for v in outcomes if v in frame and v.endswith("_aesi_end_of_follow_up"))
    covariates = [v for v in roles.index if roles[v] == "covariate" and v in frame]
    design = pd.get_dummies(frame[covariates], drop_first=True, dtype=float)
    treatment = pd.get_dummies(frame.asset_id, dtype=float).drop(columns="control")
    rows = []
    for target in requested:
        asset, endpoint = target.split(":")
        column = clinical if endpoint == "clinical_outcome" else safety
        keep = frame[column].notna()
        x = sm.add_constant(pd.concat([treatment, design], axis=1).loc[keep])
        fit = sm.OLS(frame.loc[keep, column].astype(float), x).fit(cov_type="HC1")
        rows.append(dict(target_id=target, **interval(float(fit.params[asset]), float(fit.bse[asset]))))
    return rows


def readout(requested: list[str]) -> list[dict[str, object]]:
    # The current readout is the latest one whose study.json requests the targets in interface/targets.csv.
    studies = {p.parent: json.loads(p.read_text()) for p in Path("evidence").glob("*/study.json")}
    latest = max(
        (folder for folder, study in studies.items() if [t["target_id"] for t in study["targets"]] == requested),
        key=lambda folder: studies[folder]["readout_day"],
    )
    study = studies[latest]
    by_target = {t["target_id"]: t for t in study["targets"]}
    follow_up = float(study["follow_up_days"])

    def arm_data(arm: str, table: str) -> pd.DataFrame:
        base = pd.read_parquet(latest / "arms" / arm / "baseline.parquet")[["participant_id", "enrollment_day"]]
        path = latest / "arms" / arm / f"{table}.parquet"
        return base if table == "baseline" else pd.read_parquet(path).merge(base, on="participant_id")

    def clinical(arm: str) -> tuple[float, float, int]:
        root = latest / "arms" / arm
        if (root / "recurrent_events.parquet").is_file():
            data = arm_data(arm, "recurrent_events")
            data = data.assign(elapsed=data.occurrence_time - data.enrollment_day)
            data = data.loc[data.elapsed.between(0, follow_up)]
            events = data.loc[data.event == 1].groupby("participant_id").size()
            years = (data.groupby("participant_id").elapsed.max().clip(upper=follow_up) / 365.25).clip(lower=1e-9)
            y = events.reindex(years.index, fill_value=0).astype(float)
            rate = float(y.sum() / years.sum())
            variance = float(((y - rate * years) ** 2).sum() / years.sum() ** 2)
            return rate, variance, len(years)
        data = arm_data(arm, "measurements")
        data = data.assign(elapsed=(data.occurrence_time - data.enrollment_day).round())
        data = data.loc[data.observed.astype(bool) & (data.elapsed <= follow_up + 0.5)]
        day = data.elapsed.max()
        if day != round(follow_up):
            friction(f"no observed measurement at follow-up day {follow_up:g}; latest observed visit day {day:g} used")
        values = data.loc[data.elapsed == day].groupby("participant_id").observed_outcome.last().astype(float)
        return float(values.mean()), float(values.var(ddof=1) / len(values)), len(values)

    def safety(arm: str) -> tuple[float, float, int]:
        base = arm_data(arm, "baseline")
        mature = base.loc[study["readout_day"] - base.enrollment_day >= follow_up]
        events = arm_data(arm, "safety_events")
        elapsed = events.occurrence_time - events.enrollment_day
        cases = set(events.loc[elapsed.between(0, follow_up), "participant_id"])
        risk = mature.participant_id.isin(cases).astype(float)
        if risk.sum() in (0, len(risk)):
            friction("no or all safety events in an arm; Agresti-Caffo adjusted risk used for the interval")
            adjusted = (risk.sum() + 1) / (len(risk) + 2)
            return float(risk.mean()), float(adjusted * (1 - adjusted) / (len(risk) + 2)), len(risk)
        return float(risk.mean()), float(risk.var(ddof=1) / len(risk)), len(risk)

    rows = []
    for target in requested:
        spec = by_target[target]
        measure = clinical if spec["endpoint"] == "clinical_outcome" else safety
        (a, va, na), (c, vc, nc) = measure(spec["arm_id"]), measure("control")
        rows.append(dict(target_id=target, **interval(a - c, float(np.sqrt(va + vc)))))
        print(f"{target}: {na} active and {nc} control analysed")
    return rows


requested = list(targets())
rows = readout(requested) if requested and requested[0].count(":") > 1 else observational(requested)
report(rows)
print("ESTIMATES=" + json.dumps(rows))
