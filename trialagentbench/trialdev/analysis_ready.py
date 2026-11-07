"""Lossless participant joins for the paired TrialDev analysis-ready view."""

import csv
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from clinical_data_tables import elapsed_clinical_times
from pydantic import BaseModel, ConfigDict
from trialagentbench_verifier.execution_contracts import ReplayInputArtifactV1

from trialagentbench.clinical_tables import StudyBaselineFieldV1, write_observed_study_tables
from trialagentbench.io import read_json_model, sha256_file, write_json_model
from trialagentbench.io.directory import staged_directory
from trialagentbench.trialdev.materialization import TrialDevReadoutManifest


class TrialDevAnalysisReadyManifest(BaseModel):
    """Exact public sources and derived tables for one readout projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_id: Literal["trialdev.analysis-ready/v3"] = "trialdev.analysis-ready/v3"
    sources: tuple[ReplayInputArtifactV1, ...]
    outputs: tuple[ReplayInputArtifactV1, ...]
    baseline_fields: tuple[StudyBaselineFieldV1, ...] = ()


def _artifact(root: Path, relative: str) -> ReplayInputArtifactV1:
    path = root / relative
    path.resolve(strict=True).relative_to(root.resolve(strict=True))
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("Analysis-ready sources and outputs cannot use symbolic links")
    return ReplayInputArtifactV1(relative_path=relative, sha256=sha256_file(path), byte_length=path.stat().st_size)


def trialdev_baseline_fields_from_dictionary(path: Path) -> tuple[StudyBaselineFieldV1, ...]:
    """Read clinical baseline labels and units from the supplied initial dictionary."""
    with path.open(encoding="utf-8", newline="") as stream:
        return tuple(
            StudyBaselineFieldV1(field_id=row["variable"], label=row["label"], unit=row["unit_or_terminology"])
            for row in csv.DictReader(stream)
            if row["role"] == "covariate"
        )


def write_trialdev_analysis_ready(
    readout: Path, target: Path, *, baseline_fields: tuple[StudyBaselineFieldV1, ...] = ()
) -> Path:
    """Join public participant labels onto existing rows without deriving endpoints.

    Parameters
    ----------
    readout
        Published raw readout containing study.json and arm-specific domains.
    target
        Separate output directory; raw source files remain immutable.
    baseline_fields
        Labels and units from the initial public dictionary. These metadata are
        retained in the projection identity and never supply participant values.

    Returns
    -------
    Path
        Analysis-ready directory with combined participants and observation domains.

    Raises
    ------
    ValueError
        If keys are ambiguous, observations lack participants, or source bytes change.
    """
    readout = Path(readout)
    study_artifact = _artifact(readout, "study.json")
    study = read_json_model(TrialDevReadoutManifest, readout / "study.json")
    sources = [study_artifact]
    participants = []
    domains: dict[str, list[pd.DataFrame]] = {}
    schemas: dict[str, tuple[str, ...]] = {}
    clinical_domain: str | None = None

    def check_columns(domain: str, frame: pd.DataFrame) -> None:
        columns = tuple(frame.columns)
        if not frame.columns.is_unique or "participant_id" not in columns:
            raise ValueError("Released domains require unique columns and a participant key")
        if domain in schemas and schemas[domain] != columns:
            raise ValueError("Released domain columns differ across randomized arms")
        schemas[domain] = columns

    for arm in study.arms:
        base = f"arms/{arm.arm_id}"
        sources.append(_artifact(readout, f"{base}/baseline.parquet"))
        baseline = pd.read_parquet(readout / base / "baseline.parquet")
        check_columns("baseline", baseline)
        if not baseline.participant_id.is_unique or baseline.participant_id.isna().any():
            raise ValueError("Analysis-ready participant keys must be unique and nonmissing")
        if (
            "enrollment_day" not in baseline
            or not np.isfinite(baseline.enrollment_day).all()
            or baseline.enrollment_day.lt(0).any()
            or baseline.enrollment_day.gt(study.readout_day).any()
        ):
            raise ValueError("Readout participants must have enrolled by its cutoff")
        labels = {"study_id": study.study_id, "arm_id": arm.arm_id, "asset_id": arm.asset_id, "dose": arm.dose}
        if set(labels) & set(baseline.columns):
            raise ValueError("Participant fields collide with study treatment labels")
        baseline = baseline.assign(**labels)
        participants.append(baseline)
        endpoint = [
            name for name in ("measurements", "recurrent_events") if (readout / base / f"{name}.parquet").exists()
        ]
        if len(endpoint) != 1:
            raise ValueError("Each arm requires one released clinical observation domain")
        if clinical_domain is not None and clinical_domain != endpoint[0]:
            raise ValueError("Randomized arms require the same clinical observation domain")
        clinical_domain = endpoint[0]
        additional = tuple(
            name
            for name in ("intercurrent_events", "completion_records")
            if (readout / base / f"{name}.parquet").exists()
        )
        for domain in ("contacts", "exposure", "safety_events", *additional, *endpoint):
            sources.append(_artifact(readout, f"{base}/{domain}.parquet"))
            observed = pd.read_parquet(readout / base / f"{domain}.parquet")
            check_columns(domain, observed)
            enrollment = observed.participant_id.map(baseline.set_index("participant_id").enrollment_day)
            for clock in (
                "occurrence_time",
                "observation_time",
                "documentation_time",
                "availability_time",
                "resolution_time",
            ):
                if clock not in observed:
                    continue
                calendar_clock = observed[clock]
                times = calendar_clock.dropna() if clock == "resolution_time" else calendar_clock
                if not times.empty and (not pd.api.types.is_numeric_dtype(times) or not np.isfinite(times).all()):
                    raise ValueError("Released observation clocks must be finite; only resolution may be unknown")
                if times.gt(study.readout_day).any():
                    raise ValueError("Released observation contains evidence after the readout cutoff")
                if calendar_clock.lt(enrollment).any():
                    raise ValueError("Released observation precedes participant enrollment")
            if "occurrence_time" in observed:
                for clock in ("observation_time", "documentation_time", "availability_time", "resolution_time"):
                    if clock in observed and observed[clock].lt(observed.occurrence_time).any():
                        raise ValueError("Released event clocks precede their occurrence")
            if (set(observed.columns) & set(baseline.columns)) - {"participant_id"}:
                raise ValueError("Observation columns collide with participant fields")
            joined = observed.merge(baseline, on="participant_id", how="left", validate="many_to_one", indicator=True)
            if not joined._merge.eq("both").all():
                raise ValueError("Observation rows reference an unreleased participant")
            domains.setdefault(domain, []).append(joined.drop(columns="_merge"))
    if clinical_domain is None:
        raise ValueError("Readout requires randomized arms and a clinical observation domain")
    combined = pd.concat(participants, ignore_index=True)
    if not combined.participant_id.is_unique:
        raise ValueError("Participant identity is repeated across randomized arms")
    source_inventory = tuple(sorted(sources, key=lambda item: item.relative_path))
    target = Path(target)
    if target.resolve().is_relative_to(readout.resolve()) or readout.resolve().is_relative_to(target.resolve()):
        raise ValueError("Analysis-ready output must be separate from immutable raw readout")
    if target.exists():
        existing = read_json_model(TrialDevAnalysisReadyManifest, target / "manifest.json")
        expected_outputs = {"participants.parquet", "standards.zip", *(f"{name}.parquet" for name in domains)}
        if (
            {item.relative_path for item in existing.outputs} != expected_outputs
            or len(existing.outputs) != len(expected_outputs)
            or {path.name for path in target.iterdir()} != expected_outputs | {"manifest.json"}
        ):
            raise ValueError("Analysis-ready output inventory differs from the released domains")
        if (
            existing.sources != source_inventory
            or existing.baseline_fields != baseline_fields
            or any(_artifact(target, item.relative_path) != item for item in existing.outputs)
        ):
            raise ValueError("Analysis-ready projection differs from its bound public sources or outputs")
        return target
    with staged_directory(target) as staging:
        combined.to_parquet(staging / "participants.parquet", index=False)
        for name, frames in domains.items():
            pd.concat([frame for frame in frames if not frame.empty] or frames[:1], ignore_index=True).to_parquet(
                staging / f"{name}.parquet", index=False
            )
        # Project only already published rows; this route cannot read frozen pools.
        source = pd.concat(domains[clinical_domain], ignore_index=True)
        source = source.rename(columns={"arm_id": "assigned_arm"})
        population = combined.rename(columns={"arm_id": "assigned_arm"})
        recurrent = clinical_domain == "recurrent_events"
        if recurrent:
            source = source.rename(columns={"event": "clinical_event"})
            endpoint_id = "clinical_event"
        else:
            endpoint_id = "observed_outcome"
        declared_times = tuple(
            sorted(
                {
                    0.0,
                    float(study.follow_up_days),
                    *study.assessment_days,
                    *study.surveillance_days,
                    *(boundary for _, start, end in study.epochs for boundary in (start, end)),
                }
            )
        )

        def elapsed(frame: pd.DataFrame, field: str) -> pd.Series:
            return elapsed_clinical_times(frame[field], frame.enrollment_day, declared_times=declared_times)

        source["occurrence_time"] = elapsed(source, "occurrence_time")
        safety = pd.concat(
            [frame for frame in domains["safety_events"] if not frame.empty] or domains["safety_events"][:1],
            ignore_index=True,
        )
        for field in ("occurrence_time", "availability_time", "resolution_time"):
            safety[field] = elapsed(safety, field)
        exposure = pd.concat(domains["exposure"], ignore_index=True)
        exposure["occurrence_time"] = elapsed(exposure, "occurrence_time")
        intercurrent = None
        if "intercurrent_events" in domains:
            intercurrent = pd.concat(domains["intercurrent_events"], ignore_index=True)
            intercurrent["occurrence_time"] = elapsed(intercurrent, "occurrence_time")
        completion = None
        if "completion_records" in domains:
            completion = pd.concat(domains["completion_records"], ignore_index=True)
            for clock in ("occurrence_time", "observation_time", "documentation_time", "availability_time"):
                completion[clock] = elapsed(completion, clock)
        clinical_units = {target.unit for target in study.targets if target.endpoint == "clinical_outcome"}
        if len(clinical_units) != 1:
            raise ValueError("Readout requires one declared clinical outcome unit")
        write_observed_study_tables(
            staging / "standards.zip",
            study_id=study.study_id,
            study_start=study.calendar_origin,
            epochs=study.epochs,
            participants=population,
            analysis=source,
            endpoint_id=endpoint_id,
            baseline_fields=baseline_fields,
            endpoint_object=study.outcome_object,
            endpoint_unit="indicator" if recurrent else next(iter(clinical_units)),
            endpoint_kind="recurrent" if recurrent else "longitudinal",
            assessment_time=None,
            safety_events=safety,
            intercurrent_events=intercurrent,
            completion_records=completion,
            exposure_records=exposure,
        )
        outputs = tuple(_artifact(staging, path.name) for path in sorted(staging.iterdir()) if path.is_file())
        write_json_model(
            staging / "manifest.json",
            TrialDevAnalysisReadyManifest(sources=source_inventory, outputs=outputs, baseline_fields=baseline_fields),
        )
    return target
