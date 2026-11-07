"""Grade TrialDev numerical replays using representable normalized losses.

Numerical grading uses the verifier's Decimal scorer. Acquisition provenance
binds the target-source file separately.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from trialagentbench_verifier.authored_results import TrialDevNumericalRecovery as TrialDevNumericalRecovery
from trialagentbench_verifier.authored_scoring import compute_numerical_recovery
from trialagentbench_verifier.numerical import NumericalReplay
from trialagentbench_verifier.units import compatible_units

from trialagentbench.contracts.trialdev.grading_key import TrialDevEvidenceTargetV1
from trialagentbench.io import read_json_model, sha256_file
from trialagentbench.trialdev.materialization import TrialDevReadoutCommitment, TrialDevReadoutManifest
from trialagentbench.trialdev.targets import trialdev_pool_contrast
from trialagentbench.trialdev.value import TrialDevSponsorPreferences


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
            compute_numerical_recovery(
                target_id=target.target_id,
                reference=truth,
                scale=scale,
                confidence=target.confidence_level,
                estimate=estimates.get(target.target_id),
                reproducible=replay.reproducible,
            )
        )
    return tuple(rows)


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
            compute_numerical_recovery(
                target_id=identity,
                reference=target.oracle_value,
                scale=preferences.benefit_unit if clinical else 0.01,
                confidence=0.95,
                estimate=estimates.get(identity),
                reproducible=replay.reproducible,
            )
        )
    return tuple(rows)
