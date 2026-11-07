# mypy: ignore-errors
"""Reference TrialEval raw-view analysis written into scratch/ by the scripted reference agent.

It reads only participant documents and data. Every interpretation it has to hard-code
about the environment (rather than the clinical question) is printed as a FRICTION line,
so the agent transcript records what a scripted analyst needed to know. Estimators are
deliberately conventional: Kaplan-Meier risk and RMST, Wald proportions, Welch means,
Poisson rate ratios with robust errors, and cluster-robust linear models.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from interface.reporting import report, targets  # type: ignore[import-not-found]
from scipy.stats import norm

Z = float(norm.ppf(0.975))
UNIT_DAYS = {"day": 1.0, "days": 1.0, "hour": 1 / 24, "hours": 1 / 24, "week": 7.0, "weeks": 7.0}


def friction(text: str) -> None:
    print(f"FRICTION: {text}")


def document(name: str, *, optional: bool = False) -> str:
    for path in (Path(name), Path("documents") / name):
        if path.is_file():
            return path.read_text(encoding="utf-8")
    if optional:
        return ""
    raise FileNotFoundError(name)


REQUEST = document("analysis_request.md")
POLICY = document("data_integrity_policy.md", optional=True)
IDENTITY = {
    path: [field.strip("` ") for field in fields.split(",")]
    for path, fields in re.findall(r"^- `([^`]+)`: (.+)$", POLICY, flags=re.MULTILINE)
}
CROSSWALK = Path("data/raw/identifier_crosswalk.parquet")


def load(path: str) -> pd.DataFrame:
    """Read one raw table and apply the documented source-record rules."""
    frame = pd.read_parquet(path)
    if CROSSWALK.is_file() and "participant_id" in frame and frame.participant_id.dtype == object:
        mapping = pd.read_parquet(CROSSWALK).set_index("source_identifier").current_identifier
        legacy = frame.participant_id.astype(str).str.startswith("legacy-")
        friction(f"{path}: {int(legacy.sum())} legacy participant identifiers remapped via identifier_crosswalk")
        frame.loc[legacy, "participant_id"] = frame.loc[legacy, "participant_id"].map(mapping)
        frame["participant_id"] = frame.participant_id.astype("int64")
    for column in [c for c in frame.columns if c.endswith("_unit")]:
        base = column.removesuffix("_unit")
        if base in frame:
            factor = frame[column].map(UNIT_DAYS)
            if factor.isna().any():
                raise ValueError(f"Unknown time unit in {path}:{column}: {set(frame[column][factor.isna()])}")
            if (factor != 1).any():
                friction(f"{path}: {int((factor != 1).sum())} {base} values recorded in non-day units")
            if frame[base].dtype == object:
                friction(f"{path}: {base} stored as text alongside a unit column; parsed as numbers")
                frame[base] = pd.to_numeric(frame[base])
            frame[base] = frame[base] * factor
    keys = IDENTITY.get(path)
    if keys:
        order = ["record_version"] if "record_version" in frame else []
        before = len(frame)
        frame = frame.sort_values(order or keys, kind="stable").drop_duplicates(keys, keep="last")
        if len(frame) < before:
            friction(f"{path}: {before - len(frame)} superseded or duplicate source records removed")
    return frame


def measures() -> dict[str, str]:
    """Map each result identifier to the measure the analysis request names for it."""
    rows = re.findall(r"^\| `([^`]+)` \| ([^|]+) \|", REQUEST, flags=re.MULTILINE)
    if not rows:
        raise ValueError("The analysis request names no results")
    return {target: measure.strip().lower() for target, measure in rows}


def horizon() -> float:
    match = re.search(r"(?:by|at|through|to) (?:the final scheduled )?Day[- ](\d+)|during (\d+) days", REQUEST)
    if match is None:
        raise ValueError("The analysis request states no horizon")
    return float(match.group(1) or match.group(2))


def interval(estimate: float, se: float) -> dict[str, float]:
    if not np.isfinite([estimate, se]).all() or se <= 0:
        raise ValueError("Estimate lacks finite sampling uncertainty")
    return dict(estimate=float(estimate), lower=float(estimate - Z * se), upper=float(estimate + Z * se))


def kaplan_meier(time: np.ndarray, event: np.ndarray, tau: float) -> tuple[float, float, float, float]:
    """Return S(tau), Greenwood variance, RMST to tau and its Greenwood-type variance."""
    steps = np.unique(time[(event == 1) & (time <= tau)])
    at_risk = np.array([(time >= value).sum() for value in steps], dtype=float)
    deaths = np.array([((time == value) & (event == 1)).sum() for value in steps], dtype=float)
    survival = np.cumprod(1 - deaths / at_risk) if len(steps) else np.array([])
    increments = np.where(at_risk > deaths, deaths / (at_risk * np.maximum(at_risk - deaths, 1)), 0.0)
    final = float(survival[-1]) if len(steps) else 1.0
    edges = np.append(steps, tau)
    pieces = survival * np.diff(edges) if len(steps) else np.array([])
    area_after = np.cumsum(pieces[::-1])[::-1]
    first = float(steps[0]) if len(steps) else tau
    rmst = first + float(pieces.sum())
    return final, final**2 * float(increments.sum()), rmst, float((area_after**2 * increments).sum())


participants = load("data/raw/participant_records.parquet")
arms = set(participants.assigned_arm)
if arms != {"active", "control"}:
    raise ValueError(f"Expected active/control assignment, found {arms}")
active = participants.assigned_arm.eq("active")
tau = horizon()
declared = measures()
results = []


def time_to_event() -> pd.DataFrame:
    events = load("data/raw/event_history.parquet")
    endpoint = events.loc[events.event_kind.eq("endpoint")].groupby("participant_id").occurrence_time.min()
    follow = events.loc[events.event_kind.eq("follow_up")].groupby("participant_id").occurrence_time.max()
    data = participants[["participant_id", "assigned_arm"]].copy()
    data["event_time"] = data.participant_id.map(endpoint)
    end = data.participant_id.map(follow)
    if end.isna().any():
        friction("follow-up end missing for some participants; used completion_time")
        completion = participants.set_index("participant_id").completion_time
        end = end.fillna(data.participant_id.map(completion))
    data["event"] = data.event_time.notna().astype(int)
    data["time"] = data.event_time.fillna(end).astype(float)
    return data


def survival_contrast(data: pd.DataFrame, measure: str) -> dict[str, float]:
    fits = {
        arm: kaplan_meier(group.time.to_numpy(), group.event.to_numpy(), tau)
        for arm, group in data.groupby("assigned_arm")
    }
    (s1, v1, a1, w1), (s0, v0, a0, w0) = fits["active"], fits["control"]
    if "restricted mean" in measure:
        return interval(a1 - a0, float(np.sqrt(w1 + w0)))
    return interval((1 - s1) - (1 - s0), float(np.sqrt(v1 + v0)))


def proportion_contrast(y: pd.Series, treated: pd.Series) -> dict[str, float]:
    p1, p0 = y[treated].mean(), y[~treated].mean()
    se = np.sqrt(p1 * (1 - p1) / treated.sum() + p0 * (1 - p0) / (~treated).sum())
    return interval(p1 - p0, float(se))


def cluster_ols(y: pd.Series, design: pd.DataFrame, groups: pd.Series) -> dict[str, float]:
    fit = sm.OLS(y.astype(float), sm.add_constant(design.astype(float))).fit(
        cov_type="cluster", cov_kwds={"groups": groups}
    )
    return interval(float(fit.params["treated"]), float(fit.bse["treated"]))


for target in targets():
    measure = declared[target]
    question = REQUEST.lower()
    if "stepped-wedge" in question:
        data = time_to_event()
        frame = participants.set_index("participant_id").loc[data.participant_id]
        y = ((data.event == 1) & (data.time <= tau)).astype(float).to_numpy()
        friction("stepped wedge: period column and cluster column names inferred from data_specification")
        design = pd.get_dummies(frame.calendar_period.astype(str), prefix="period", drop_first=True)
        design["treated"] = frame.assigned_arm.eq("active").to_numpy()
        design.index = range(len(design))
        results.append(dict(target_id=target, **cluster_ols(pd.Series(y), design, frame.cluster_id.to_numpy())))
    elif "cluster_id" in participants and "resident" in target:
        data = time_to_event()
        y = ((data.event == 1) & (data.time <= tau)).astype(float)
        design = pd.DataFrame({"treated": data.assigned_arm.eq("active").to_numpy()})
        results.append(
            dict(target_id=target, **cluster_ols(y.reset_index(drop=True), design, participants.cluster_id))
        )
    elif Path("data/raw/endpoint_observations.parquet").is_file():
        observations = load("data/raw/endpoint_observations.parquet")
        # Select the primary endpoint column declared by the data dictionary.
        dictionary = pd.read_csv("data_dictionary.csv")
        (value,) = dictionary.loc[
            dictionary.table.eq("data/raw/endpoint_observations.parquet")
            & dictionary.role.str.contains("primary endpoint"),
            "variable",
        ]
        if "occurrence_time" in observations:
            final = observations.occurrence_time.max()
            observations = observations.loc[np.isclose(observations.occurrence_time, final)]
            if "observed" in observations:
                observations = observations.loc[observations.observed.astype(bool)]
            friction("longitudinal endpoint: final scheduled assessment selected by maximum occurrence_time")
        merged = participants[["participant_id", "assigned_arm"]].merge(
            observations[
                ["participant_id", value] + (["assessment_time"] if "assessment_time" in observations else [])
            ],
            on="participant_id",
            how="inner",
        )
        merged = merged.loc[merged[value].notna()]
        treated = merged.assigned_arm.eq("active")
        if "rate ratio" in measure:
            exposure = merged.assessment_time.clip(lower=1e-9)
            fit = sm.GLM(
                merged[value].astype(float),
                sm.add_constant(treated.astype(float).rename("treated")),
                family=sm.families.Poisson(),
                offset=np.log(exposure),
            ).fit(cov_type="HC0")
            low, high = fit.conf_int().loc["treated"]
            results.append(
                dict(
                    target_id=target,
                    estimate=float(np.exp(fit.params["treated"])),
                    lower=float(np.exp(low)),
                    upper=float(np.exp(high)),
                )
            )
        elif "risk difference" in measure:
            results.append(dict(target_id=target, **proportion_contrast(merged[value].astype(float), treated)))
        else:
            a, c = merged.loc[treated, value], merged.loc[~treated, value]
            se = float(np.sqrt(a.var(ddof=1) / len(a) + c.var(ddof=1) / len(c)))
            results.append(dict(target_id=target, **interval(float(a.mean() - c.mean()), se)))
    else:
        data = time_to_event()
        subset = re.search(r"baseline (\w+) at least ([\d.]+)", REQUEST)
        if subset is not None:
            column = next(c for c in participants.columns if subset.group(1).lower() in c.lower())
            friction(f"standardization population '{subset.group(0)}' mapped to column {column}")
            keep = participants.set_index("participant_id").loc[data.participant_id, column] >= float(subset.group(2))
            data = data.loc[keep.to_numpy()]
        results.append(dict(target_id=target, **survival_contrast(data, measure)))

report(results)
for row in results:
    print(f"{row['target_id']}: {row['estimate']:.6g} [{row['lower']:.6g}, {row['upper']:.6g}]")
