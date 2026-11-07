"""Independent finite-population references and numerical recovery checks."""

import math
from decimal import Decimal, localcontext
from pathlib import Path

import pandas as pd
from pydantic import Field

from trialagentbench_verifier.authored_readouts import _Pool, _Projection, _Readout
from trialagentbench_verifier.authored_results import TrialDevNumericalRecovery, TrialDevProgrammeEvaluation
from trialagentbench_verifier.io import sha256_file
from trialagentbench_verifier.numerical import NumericalEstimate
from trialagentbench_verifier.resource_accounting import _Configuration, _Contrast


class _ReferencePool(_Pool):
    participant_count: int = Field(ge=1)
    collection_days: tuple[int, ...] = Field(min_length=1)
    targets: tuple[_Contrast, ...]


class TrialEvalTargetV1(_Projection):
    target_id: str
    arm_id: str
    endpoint: str
    unit: str
    confidence_level: float = Field(gt=0, lt=1)


class _Study(_Readout):
    targets: tuple[TrialEvalTargetV1, ...] = Field(min_length=1)


def compute_numerical_recovery(
    *,
    target_id: str,
    estimate: NumericalEstimate | None,
    reference: float,
    scale: float,
    confidence: float,
    reproducible: bool,
) -> TrialDevNumericalRecovery:
    """Calculate one target's recovery using independent high-precision arithmetic."""
    if not math.isfinite(reference) or not math.isfinite(scale) or scale <= 0 or not 0 < confidence < 1:
        raise ValueError("Numerical reference requires finite truth, positive scale and valid confidence")
    if estimate is not None and (not reproducible or estimate.target_id != target_id):
        raise ValueError("Numerical score differs from its reproduced target")
    status = "reported" if estimate is not None else "missing" if reproducible else "invalid_replay"
    absolute = interval = None
    if estimate is not None:
        with localcontext() as context:
            context.prec = 80
            point, lower, upper, truth, denominator = map(
                Decimal.from_float, (estimate.estimate, estimate.lower, estimate.upper, reference, scale)
            )
            alpha = 1 - Decimal.from_float(confidence)
            absolute_loss = abs(point - truth) / denominator
            interval_loss = (upper - lower + 2 * max(lower - truth, truth - upper, Decimal(0)) / alpha) / denominator
            absolute, interval = float(absolute_loss), float(interval_loss)
            absolute = absolute if math.isfinite(absolute) else None
            interval = interval if math.isfinite(interval) else None
    return TrialDevNumericalRecovery(
        target_id=target_id,
        status=status,
        reference_value=reference,
        normalization_scale=scale,
        normalized_absolute_error=absolute,
        normalized_interval_loss=interval,
    )


def verify_numerical_recovery(
    *,
    score: TrialDevNumericalRecovery,
    estimate: NumericalEstimate | None,
    reference: float,
    scale: float,
    confidence: float,
    reproducible: bool,
) -> float | None:
    """Check a retained score against independently reconstructed recovery."""
    expected = compute_numerical_recovery(
        target_id=score.target_id,
        estimate=estimate,
        reference=reference,
        scale=scale,
        confidence=confidence,
        reproducible=reproducible,
    )
    if score.status != expected.status:
        raise ValueError("Numerical score changes its reporting or replay status")
    pairs = (
        (score.reference_value, reference),
        (score.normalization_scale, scale),
        (score.normalized_absolute_error, expected.normalized_absolute_error),
        (score.normalized_interval_loss, expected.normalized_interval_loss),
    )
    if any(
        (left is None) != (right is None)
        or (left is not None and right is not None and not math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10))
        for left, right in pairs
    ):
        raise ValueError("Numerical analysis score differs from independent recovery")
    return expected.loss


def _contrast(pool_root: Path, pool: _ReferencePool, asset_id: str, dose: str, horizon: int) -> _Contrast:
    """Compute arm means over every frozen patient, including event-free patients."""
    if horizon not in pool.collection_days:
        raise ValueError("Study reference horizon is outside the frozen collection grid")
    references = [row for row in pool.targets if (row.asset_id, row.dose) == (asset_id, dose)]
    controls = [row for row in pool.partitions if row.dose is None]
    active = [row for row in pool.partitions if (row.asset_id, row.dose) == (asset_id, dose)]
    if len(references) != 1 or len(controls) != 1 or len(active) != 1:
        raise ValueError("Study reference has no unique released treatment pair")
    inventory = {row.relative_path: row for row in pool.artifacts}

    def read(relative: str) -> pd.DataFrame:
        entry = inventory.get(relative)
        path = pool_root / relative
        if (
            entry is None
            or path.is_symlink()
            or sha256_file(path) != entry.sha256
            or path.stat().st_size != entry.byte_length
        ):
            raise ValueError("Study reference source differs from its frozen inventory")
        return pd.read_parquet(path.open("rb"))

    population = read("baseline.parquet").participant_id
    identities = set(population)
    if len(population) != pool.participant_count or len(identities) != len(population) or population.isna().any():
        raise ValueError("Study reference changed its complete target population")
    means = []
    for partition in (controls[0], active[0]):
        recurrent_path = f"{partition.directory}/recurrent_truth.parquet"
        recurrent = recurrent_path in inventory
        if recurrent != (references[0].outcome_unit == "events per participant-year"):
            raise ValueError("Study reference units and recurrent support disagree")
        if recurrent:
            events = read(recurrent_path)
            ends = events.loc[events.event.eq(0)]
            if (
                len(ends) != len(population)
                or set(ends.participant_id) != identities
                or not ends.event_time.eq(pool.collection_days[-1]).all()
            ):
                raise ValueError("Recurrent reference omits event-free population follow-up")
            if (
                not events.participant_id.isin(identities).all()
                or not events.event.isin([0, 1]).all()
                or not events.event_time.between(0, pool.collection_days[-1]).all()
            ):
                raise ValueError("Recurrent reference includes invalid patient events")
            expected_arm = "control" if partition.dose is None else "active"
            if not events.assigned_arm.eq(expected_arm).all():
                raise ValueError("Recurrent reference contains a different intervention")
            clinical = (
                int((events.event.eq(1) & events.event_time.le(horizon)).sum()) * 365.0 / horizon / len(population)
            )
        else:
            outcomes = read(f"{partition.directory}/complete_measurements.parquet")
            selected = outcomes.loc[outcomes.occurrence_time.eq(horizon) & outcomes.outcome_id.eq("clinical_outcome")]
            if (
                len(selected) != len(population)
                or set(selected.participant_id) != identities
                or not selected.complete_recorded_outcome.map(math.isfinite).all()
            ):
                raise ValueError("Continuous reference omits or repeats target patients")
            clinical = math.fsum(selected.complete_recorded_outcome) / len(population)
        safety = read(f"{partition.directory}/safety_events.parquet")
        endpoints = safety.loc[safety["_scheduled_contact_time"].eq(horizon)]
        event_ids = set(safety.event_id)
        expected = {(patient, event) for patient in identities for event in event_ids}
        actual = list(zip(endpoints.participant_id, endpoints.event_id, strict=True))
        if not event_ids or safety.event_id.isna().any() or len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("Safety reference omits or repeats target patient/event rows at the study horizon")
        incidents = set(safety.loc[safety.event_at_contact & safety.occurrence_time.le(horizon), "participant_id"])
        if not incidents <= identities:
            raise ValueError("Safety reference includes an unknown patient")
        means.append((clinical, len(incidents) / len(population)))
    return _Contrast(
        asset_id=asset_id,
        dose=dose,
        outcome_unit=references[0].outcome_unit,
        outcome_mean_difference=means[1][0] - means[0][0],
        safety_risk_difference=means[1][1] - means[0][1],
    )


def verify_trialdev_prospective_scores(
    *, pool_root: Path, configuration_path: Path, workspace: Path, evaluation: TrialDevProgrammeEvaluation
) -> None:
    """Check acquired-study scores against independently reconstructed fixed truth.

    Parameters
    ----------
    pool_root, configuration_path
        Authenticated released pool and sponsor preferences.
    workspace
        Authenticated workspace containing the acquired study manifests.
    evaluation
        Retained programme result, with separate study targets and replays.

    Raises
    ------
    ValueError
        If targets, windows, units, numerical losses or complete-population references disagree.

    Notes
    -----
    Source execution and recruitment history are authenticated by the caller.
    """
    configuration = _Configuration.model_validate_json(configuration_path.read_bytes())
    pool_path = pool_root / "manifest.json"
    if sha256_file(pool_path) != configuration.pool_manifest_sha256:
        raise ValueError("Prospective scoring identifies a different released pool")
    pool = _ReferencePool.model_validate_json(pool_path.read_bytes())
    preferences = configuration.preferences
    for study in evaluation.studies:
        checkpoint = Path(study.report.checkpoint_id)
        if checkpoint.is_absolute() or ".." in checkpoint.parts or "\\" in str(checkpoint):
            raise ValueError("Prospective checkpoint must remain in its evidence workspace")
        manifest_path = workspace / checkpoint / "study.json"
        manifest = _Study.model_validate_json(manifest_path.read_bytes())
        if manifest.study_id != study.report.study_id or study.replay.checkpoint_id != study.report.checkpoint_id:
            raise ValueError("Prospective report changed its acquired study or cutoff")
        identities = tuple(row.target_id for row in manifest.targets)
        if identities != study.replay.target_ids or len(set(identities)) != len(identities):
            raise ValueError("Prospective report changed the scheduled target census")
        artifact = next(
            (
                row
                for row in study.replay.evidence_artifacts
                if row.relative_path == (checkpoint / "study.json").as_posix()
            ),
            None,
        )
        if (
            artifact is None
            or artifact.sha256 != sha256_file(manifest_path)
            or artifact.byte_length != manifest_path.stat().st_size
        ):
            raise ValueError("Prospective report changed its acquired manifest")
        contrasts = {
            arm.arm_id: _contrast(pool_root, pool, arm.asset_id, arm.dose, manifest.follow_up_days)
            for arm in manifest.arms
            if arm.dose is not None
        }
        scores = {row.target_id: row for row in study.scores}
        estimates = {row.target_id: row for row in study.replay.estimates or ()}
        for target in manifest.targets:
            reference = contrasts.get(target.arm_id)
            if (
                reference is None
                or target.endpoint not in {"clinical_outcome", "safety_risk"}
                or reference.outcome_unit != preferences.outcome_unit
            ):
                raise ValueError("Prospective report uses an undeclared treatment or endpoint")
            clinical = target.endpoint == "clinical_outcome"
            if target.unit != (reference.outcome_unit if clinical else "risk difference"):
                raise ValueError("Prospective target units differ from frozen truth")
            verify_numerical_recovery(
                score=scores[target.target_id],
                estimate=estimates.get(target.target_id),
                reference=reference.outcome_mean_difference if clinical else reference.safety_risk_difference,
                scale=preferences.benefit_unit if clinical else 0.01,
                confidence=target.confidence_level,
                reproducible=study.replay.reproducible,
            )
