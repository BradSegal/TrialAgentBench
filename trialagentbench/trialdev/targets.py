"""Exact study-horizon targets from frozen, evaluator-only patient potentials."""

from __future__ import annotations

import math
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from trialagentbench_verifier.authored_results import TrialDevNumericalRecovery as TrialDevNumericalRecovery
from trialagentbench_verifier.numerical import NumericalEstimate, NumericalReplay
from trialagentbench_verifier.numerical_scoring import finite_diagnostic, interval_score
from trialagentbench_verifier.units import compatible_units

from trialagentbench.contracts.trialdev.grading_key import TrialDevEvidenceTargetV1
from trialagentbench.contracts.trialdev.pools import TrialDevFrozenPool, TrialDevPoolContrast, TrialDevPoolPartition
from trialagentbench.contracts.trialdev.release import TrialDevDose
from trialagentbench.io import read_json_model, sha256_file
from trialagentbench.trialdev.materialization import (
    TrialDevReadoutCommitment,
    TrialDevReadoutManifest,
    read_pool_table,
)
from trialagentbench.trialdev.resources import (
    TRIALDEV_APPROVAL_CRITICAL_Z,
    TRIALDEV_APPROVAL_MINIMUM_COMPLETED_TREATED,
    TrialDevBudgetLedger,
    TrialDevRegulatoryReview,
)
from trialagentbench.trialdev.value import TrialDevSponsorPreferences


def _indexed(frame: pd.DataFrame) -> pd.DataFrame:
    indexed = frame.set_index("participant_id").sort_index()
    if not indexed.index.is_unique or indexed.index.hasnans:
        raise ValueError("pool truth requires unique nonmissing participant identities")
    return indexed


def pool_outcomes_at_horizon(
    *,
    participants: pd.DataFrame,
    complete_measurements: pd.DataFrame,
    safety_events: pd.DataFrame,
    recurrent_truth: pd.DataFrame | None,
    horizon: float,
    maximum_horizon: float,
    expected_arm: str,
) -> pd.DataFrame:
    """Extract complete potentials at a declared collection date.

    Parameters
    ----------
    participants
        Complete fixed reference population, including participant_id.
    complete_measurements
        Private complete recorded scores before visit missingness on the frozen collection grid.
    safety_events
        Private incident safety outcomes, including scheduled contact times.
    recurrent_truth
        Complete uncensored recurrent events, or None for continuous outcomes.
    horizon
        Requested study follow-up duration in days.
    maximum_horizon
        Final frozen collection date, also the complete recurrent censoring date.
    expected_arm
        Native intervention label identifying the partition.

    Returns
    -------
    pandas.DataFrame
        Participant-indexed outcome and cumulative safety indicator. Recurrent
        outcomes are annualized using 365 days; no observed sample is selected.

    Raises
    ------
    ValueError
        If the requested date, population coverage or private records are invalid.

    Examples
    --------
    The same frozen partition supplies Day-28 safety and Day-168 efficacy targets.
    """
    if (
        isinstance(horizon, bool)
        or isinstance(maximum_horizon, bool)
        or not math.isfinite(horizon)
        or not math.isfinite(maximum_horizon)
        or not 0 < horizon <= maximum_horizon
    ):
        raise ValueError("pool truth requires a supported positive horizon")
    population = _indexed(participants)
    if len(population.index) == 0:
        raise ValueError("pool truth requires a nonempty target population")
    if recurrent_truth is not None:
        frame = recurrent_truth
        if (
            not frame.participant_id.isin(population.index).all()
            or not frame.event.isin((0, 1)).all()
            or not frame.event_time.between(0, maximum_horizon).all()
            or not frame.assigned_arm.eq(expected_arm).all()
        ):
            raise ValueError("pool count truth contains invalid patient events")
        ends = _indexed(frame.loc[frame.event.eq(0)])
        if not ends.index.equals(population.index) or not ends.event_time.eq(maximum_horizon).all():
            raise ValueError("pool count truth must cover every target participant through the final horizon")
        counts = frame.loc[frame.event.eq(1) & frame.event_time.le(horizon)].groupby("participant_id").size()
        outcome = counts.reindex(population.index, fill_value=0).astype(float) * (365.0 / horizon)
    else:
        terminal = complete_measurements.loc[
            complete_measurements.occurrence_time.eq(horizon) & complete_measurements.outcome_id.eq("clinical_outcome")
        ]
        outcome = _indexed(terminal).complete_recorded_outcome
        if not outcome.index.equals(population.index):
            raise ValueError("pool outcome truth must cover every target participant")
    event_ids = safety_events.event_id.unique()
    final_safety = safety_events.loc[safety_events["_scheduled_contact_time"].eq(horizon)]
    expected_safety = pd.MultiIndex.from_product([population.index, event_ids])
    actual_safety = pd.MultiIndex.from_frame(final_safety[["participant_id", "event_id"]])
    if (
        not len(event_ids)
        or safety_events.event_id.isna().any()
        or not actual_safety.is_unique
        or len(actual_safety) != len(expected_safety)
        or not actual_safety.isin(expected_safety).all()
    ):
        raise ValueError(
            "pool safety truth must cover every target participant and event through the requested horizon"
        )
    incident = safety_events.loc[
        safety_events.event_at_contact & safety_events.occurrence_time.le(horizon), "participant_id"
    ]
    if not incident.isin(population.index).all():
        raise ValueError("pool safety truth contains an unknown participant")
    result = pd.DataFrame(
        {"outcome": outcome, "safety_event": population.index.isin(incident).astype(int)}, index=population.index
    )
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("pool truth must be finite for every target participant")
    return result


def trialdev_partition_outcomes(
    root: Path, manifest: TrialDevFrozenPool, partition: TrialDevPoolPartition, horizon: int
) -> pd.DataFrame:
    """Read one verified partition's complete potentials at a frozen collection date.

    Parameters
    ----------
    root, manifest
        Evaluator-only frozen pool and its verified manifest.
    partition
        Control or asset/regimen partition.
    horizon
        Frozen collection date in days.

    Returns
    -------
    pandas.DataFrame
        Participant-indexed complete outcome and cumulative safety indicator.

    Raises
    ------
    ValueError
        If units, population or private records are inconsistent.
    """
    root = root.resolve()
    target = next((row for row in manifest.targets if row.dose is not None), None)
    if target is None:
        raise ValueError("frozen pool has no active treatment target")
    baseline = read_pool_table(root, manifest, "baseline.parquet")
    if len(baseline) != manifest.participant_count:
        raise ValueError("frozen baseline differs from the declared target population")
    prefix = partition.directory
    recurrent_path = f"{prefix}/recurrent_truth.parquet"
    has_recurrent = any(row.relative_path == recurrent_path for row in manifest.artifacts)
    if has_recurrent != (target.outcome_unit == "events per participant-year"):
        raise ValueError("frozen target units and recurrent record support disagree")
    return pool_outcomes_at_horizon(
        participants=baseline,
        complete_measurements=read_pool_table(root, manifest, f"{prefix}/complete_measurements.parquet"),
        safety_events=read_pool_table(root, manifest, f"{prefix}/safety_events.parquet"),
        recurrent_truth=read_pool_table(root, manifest, recurrent_path) if has_recurrent else None,
        horizon=horizon,
        maximum_horizon=manifest.collection_days[-1],
        expected_arm="control" if partition.dose is None else "active",
    )


def trialdev_pool_contrast(
    pool_root: Path, *, asset_id: str, dose: TrialDevDose, follow_up_days: int
) -> TrialDevPoolContrast:
    """Calculate a fixed-population contrast at an authored study's follow-up.

    Parameters
    ----------
    pool_root
        Evaluator-only frozen pool with authenticated artifact inventory.
    asset_id
        Supported active treatment identifier.
    dose
        Supported treatment regimen.
    follow_up_days
        One declared frozen collection date.

    Returns
    -------
    TrialDevPoolContrast
        Exact outcome and safety contrasts over the complete development population.

    Raises
    ------
    ValueError
        If the treatment/date is unsupported or frozen records fail integrity checks.

    Examples
    --------
    A shorter study is compared against its own endpoint window, while terminal
    development value continues to use the predeclared final-horizon target.
    """
    root = pool_root.resolve()
    manifest = read_json_model(TrialDevFrozenPool, root / "manifest.json")
    if (
        isinstance(follow_up_days, bool)
        or not isinstance(follow_up_days, int)
        or follow_up_days not in manifest.collection_days
    ):
        raise ValueError("study target horizon is outside the frozen collection grid")
    target = next((row for row in manifest.targets if row.asset_id == asset_id and row.dose == dose), None)
    if target is None:
        raise ValueError("study target requires a supported active asset and regimen")
    control = next(row for row in manifest.partitions if row.dose is None)
    active = next(row for row in manifest.partitions if row.asset_id == asset_id and row.dose == dose)
    outcomes = [
        trialdev_partition_outcomes(root, manifest, partition, follow_up_days) for partition in (control, active)
    ]
    differences = outcomes[1] - outcomes[0]
    return TrialDevPoolContrast(
        asset_id=asset_id,
        dose=dose,
        participant_count=manifest.participant_count,
        reference_days=follow_up_days,
        outcome_unit=target.outcome_unit,
        outcome_mean_difference=float(differences.outcome.mean()),
        safety_risk_difference=float(differences.safety_event.mean()),
    )


def trialdev_regulatory_review(
    pool_root: Path,
    ledger: TrialDevBudgetLedger,
    *,
    asset_id: str,
    dose: TrialDevDose,
    submission_day: float,
) -> TrialDevRegulatoryReview:
    """Apply the regulator's primary analysis to completed confirmation patients.

    Parameters
    ----------
    pool_root
        Evaluator-only frozen pool with authenticated artifact inventory.
    ledger
        Programme state accrued to the submission day.
    asset_id, dose
        Advanced treatment.
    submission_day
        Programme day of the advancement decision.

    Returns
    -------
    TrialDevRegulatoryReview
        Completed counts and the inverse-variance pooled benefit z statistic over
        every confirmation study randomizing this treatment against control. Each
        study contributes a Welch difference in complete recorded outcomes at its
        committed follow-up, so its noise is the study's real sampling noise.

    Raises
    ------
    ValueError
        If the pool, ledger or treatment identities are inconsistent.

    Examples
    --------
    A study whose patients have not completed follow-up contributes nothing, so a
    sponsor cannot submit before its confirmation evidence matures. All current
    programme outcomes are lower-is-better, so benefit is a negative difference.
    """
    root = pool_root.resolve()
    manifest = read_json_model(TrialDevFrozenPool, root / "manifest.json")
    if ledger.pool_manifest_sha256 != sha256_file(root / "manifest.json"):
        raise ValueError("regulatory review requires the programme's frozen pool")
    control = next(row for row in manifest.partitions if row.dose is None)
    active = next((row for row in manifest.partitions if row.asset_id == asset_id and row.dose == dose), None)
    if active is None:
        raise ValueError("regulatory review requires a supported asset and regimen")
    enrolled = {row.study_id: row.patients for row in ledger.enrollments}
    estimates: list[tuple[float, float]] = []
    treated_total = control_total = 0
    for design in ledger.recruitment_commitments:
        arms = {arm.partition.directory: arm.arm_id for arm in design.arms}
        if design.phase != "confirmation" or active.directory not in arms:
            continue
        completed = [
            patient
            for patient in enrolled.get(design.study_id, ())
            if patient.enrollment_day + design.follow_up_days <= submission_day
        ]
        groups = []
        for partition in (active, control):
            ids = [patient.participant_id for patient in completed if patient.arm_id == arms[partition.directory]]
            outcome = trialdev_partition_outcomes(root, manifest, partition, design.follow_up_days).outcome
            groups.append(outcome.loc[ids].to_numpy(dtype=float))
        treated_total += len(groups[0])
        control_total += len(groups[1])
        if min(len(groups[0]), len(groups[1])) < 2:
            continue
        variance = float(np.var(groups[0], ddof=1) / len(groups[0]) + np.var(groups[1], ddof=1) / len(groups[1]))
        if variance > 0:
            estimates.append((float(groups[0].mean() - groups[1].mean()), variance))
    statistic = None
    if estimates:
        weights = [1 / variance for _, variance in estimates]
        pooled = sum(weight * difference for weight, (difference, _) in zip(weights, estimates, strict=True))
        statistic = -pooled / sum(weights) * math.sqrt(sum(weights))
    return TrialDevRegulatoryReview(
        completed_treated=treated_total,
        completed_control=control_total,
        benefit_z_statistic=statistic,
        approved=(
            statistic is not None
            and statistic >= TRIALDEV_APPROVAL_CRITICAL_Z
            and treated_total >= TRIALDEV_APPROVAL_MINIMUM_COMPLETED_TREATED
        ),
    )


def grade_trialdev_readout_numbers(
    *,
    workspace: Path,
    pool_root: Path,
    readout_commitment: TrialDevReadoutCommitment,
    replay: NumericalReplay,
    preferences: TrialDevSponsorPreferences,
) -> tuple[TrialDevNumericalRecovery, ...]:
    """Grade all declared targets against the fixed population at study follow-up.

    Parameters
    ----------
    workspace
        Participant evidence workspace containing the acquired readout.
    pool_root
        Evaluator-only frozen patient pool.
    readout_commitment
        Evaluator-owned acquisition receipt binding the study and frozen pool.
    replay
        Captured checkpoint replay with its authenticated evidence inventory.
    preferences
        Public outcome units and decision scale; safety uses one percentage point.

    Returns
    -------
    tuple of TrialDevNumericalRecovery
        One row per required target, including zero credit for missing or invalid
        replay. No average across self-selected studies is constructed.

    Raises
    ------
    ValueError
        If checkpoint, evidence, targets or scientific units are inconsistent.
    """
    if sha256_file(pool_root / "manifest.json") != readout_commitment.pool_manifest_sha256:
        raise ValueError("grading pool differs from the acquired study commitment")
    if replay.checkpoint_id is None:
        raise ValueError("study grading requires a captured readout checkpoint")
    relative = PurePosixPath(replay.checkpoint_id) / "study.json"
    if relative.is_absolute() or ".." in relative.parts or "\\" in replay.checkpoint_id:
        raise ValueError("study checkpoint must remain inside participant evidence")
    path = workspace.joinpath(*relative.parts)
    if any(workspace.joinpath(*relative.parts[:i]).is_symlink() for i in range(1, len(relative.parts) + 1)):
        raise ValueError("study checkpoint cannot use symbolic links")
    artifact = next((item for item in replay.evidence_artifacts if item.relative_path == relative.as_posix()), None)
    if artifact is None or not path.is_file() or path.stat().st_size != artifact.byte_length:
        raise ValueError("study manifest differs from the captured evidence inventory")
    readout = read_json_model(TrialDevReadoutManifest, path)
    if readout != readout_commitment.readout or readout.study_id != readout_commitment.study_id:
        raise ValueError("public readout differs from its private acquisition commitment")
    if replay.target_ids != tuple(item.target_id for item in readout.targets):
        raise ValueError("replay target inventory differs from the acquired study")
    contrasts = {}
    for arm in readout.arms:
        if arm.dose is not None:
            contrast = trialdev_pool_contrast(
                pool_root, asset_id=arm.asset_id, dose=arm.dose, follow_up_days=readout.follow_up_days
            )
            if contrast.outcome_unit != preferences.outcome_unit:
                raise ValueError("public outcome normalization uses different clinical units")
            contrasts[arm.arm_id] = contrast
    estimates = {item.target_id: item for item in replay.estimates or ()} if replay.reproducible else {}
    if not set(estimates).issubset(replay.target_ids):
        raise ValueError("replay contains undeclared numerical targets")
    rows = []
    for target in readout.targets:
        contrast = contrasts[target.arm_id]
        clinical = target.endpoint == "clinical_outcome"
        truth = contrast.outcome_mean_difference if clinical else contrast.safety_risk_difference
        scale = preferences.benefit_unit if clinical else 0.01
        if target.unit != (contrast.outcome_unit if clinical else "risk difference"):
            raise ValueError("public target units differ from frozen reference units")
        rows.append(
            _numerical_recovery(
                target_id=target.target_id,
                truth=truth,
                scale=scale,
                confidence=target.confidence_level,
                estimate=estimates.get(target.target_id),
                reproducible=replay.reproducible,
            )
        )
    return tuple(rows)


def _numerical_recovery(
    *,
    target_id: str,
    truth: float,
    scale: float,
    confidence: float,
    estimate: NumericalEstimate | None,
    reproducible: bool,
) -> TrialDevNumericalRecovery:
    """Apply the same numerical loss to observational and acquired-study targets."""
    return TrialDevNumericalRecovery(
        target_id=target_id,
        status="reported" if estimate is not None else "missing" if reproducible else "invalid_replay",
        reference_value=truth,
        normalization_scale=scale,
        normalized_absolute_error=(
            None if estimate is None else finite_diagnostic(abs(estimate.estimate - truth) / scale)
        ),
        normalized_interval_loss=(
            None
            if estimate is None
            else finite_diagnostic(interval_score(estimate.lower, estimate.upper, truth, confidence) / scale)
        ),
    )


def grade_trialdev_observational_numbers(
    *,
    targets: tuple[TrialDevEvidenceTargetV1, ...],
    replay: NumericalReplay,
    preferences: TrialDevSponsorPreferences,
) -> tuple[TrialDevNumericalRecovery, ...]:
    """Grade initial observed associations without judging an analysis method.

    Parameters
    ----------
    targets
        Release-bound observed-association reference values for the initial cohort.
        Causal-effect and nonidentified keys are rejected rather than silently reused.
    replay
        Verified numerical export, with asset:endpoint identifiers assigned by the runner.
    preferences
        Public clinical units and normalization shared with prospective study grading.

    Returns
    -------
    tuple of TrialDevNumericalRecovery
        One numerical score per assigned initial target, retaining missing targets.

    Raises
    ------
    ValueError
        If target meanings, identities or units do not match the initial task.
    """
    identities = tuple(f"{target.asset_id}:{target.endpoint}" for target in targets)
    if not identities or len(set(identities)) != len(identities) or replay.target_ids != identities:
        raise ValueError("initial replay target inventory differs from the observed-association key")
    if any(
        target.estimand != "standardized_observed_association" or not target.point_identified for target in targets
    ):
        raise ValueError("initial numerical grading requires identified observed-association targets")
    estimates = {item.target_id: item for item in replay.estimates or ()} if replay.reproducible else {}
    if not set(estimates).issubset(identities):
        raise ValueError("initial replay contains undeclared numerical targets")
    rows = []
    for target, identity in zip(targets, identities, strict=True):
        clinical = target.endpoint == "clinical_outcome"
        if not compatible_units(
            reported=target.unit, reference=preferences.outcome_unit if clinical else "risk difference"
        ):
            raise ValueError("initial target units differ from the public scientific units")
        rows.append(
            _numerical_recovery(
                target_id=identity,
                truth=target.oracle_value,
                scale=preferences.benefit_unit if clinical else 0.01,
                confidence=0.95,
                estimate=estimates.get(identity),
                reproducible=replay.reproducible,
            )
        )
    return tuple(rows)
