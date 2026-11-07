"""Readout participant mapping is checked against frozen source records."""

import json
from pathlib import Path

import pandas as pd
import pytest

from trialagentbench_verifier.authored_readouts import _Patient, _Readout, verify_trialdev_readout_patients
from trialagentbench_verifier.io import sha256_file


def _case(root: Path, *, consent: bool = False) -> tuple[Path, Path, Path]:
    pool, readout = root / "pool", root / "readout"
    pool.mkdir()
    baseline = pd.DataFrame(
        {
            "participant_id": [10, 20],
            "severity": [3.2, 4.1],
            "sex": ["male", "female"],
            "site_id": ["site_001", "site_002"],
            "country_code": ["USA", "CAN"],
        }
    )
    if consent:
        baseline["consent_time"] = [-2.0, 0.0]
    baseline.to_parquet(pool / "baseline.parquet", index=False)
    partitions: list[dict[str, str | None]] = [
        {"asset_id": "control", "dose": None, "directory": "partitions/p000"},
        {"asset_id": "candidate", "dose": "standard", "directory": "partitions/p001"},
    ]
    manifest = {
        "schema_id": "trialdev.frozen-pool/v7",
        "outcome_object": "Protocol-defined clinical event",
        "baseline_fields": ["severity", "sex"],
        "partitions": partitions,
        "artifacts": [
            {
                "relative_path": "baseline.parquet",
                "sha256": sha256_file(pool / "baseline.parquet"),
                "byte_length": (pool / "baseline.parquet").stat().st_size,
            }
        ],
    }
    (pool / "manifest.json").write_text(json.dumps(manifest))
    arms = [
        {"arm_id": arm, "asset_id": partition["asset_id"], "dose": partition["dose"]}
        for arm, partition in zip(("control", "active"), partitions, strict=True)
    ]
    study = {
        "study_id": "study",
        "outcome_object": "Protocol-defined clinical event",
        "readout_day": 100,
        "follow_up_days": 28,
        "arms": arms,
        "assessment_days": [0, 28],
        "surveillance_days": [0, 28],
    }
    patient_rows = ((10, "p-a", "control"), (20, "p-b", "active"))
    patients = [
        {"participant_id": identity, "participant_label": label, "arm_id": arm, "enrollment_day": 80}
        for identity, label, arm in patient_rows
    ]
    commitment = root / "commitment.json"
    commitment.write_text(
        json.dumps(
            {
                "study_id": "study",
                "pool_manifest_sha256": sha256_file(pool / "manifest.json"),
                "readout": study,
                "assignments": [
                    {"arm_id": arm["arm_id"], "partition": partition}
                    for arm, partition in zip(arms, partitions, strict=True)
                ],
                "patients": patients,
            }
        )
    )
    readout.mkdir()
    (readout / "study.json").write_text(json.dumps(study))
    for identity, label, arm_id in patient_rows:
        arm_root = readout / "arms" / arm_id
        arm_root.mkdir(parents=True)
        selected = baseline.loc[baseline.participant_id.eq(identity)].copy()
        selected["participant_id"] = label
        selected["enrollment_day"] = 80
        selected.to_parquet(arm_root / "baseline.parquet", index=False)
    return pool, commitment, readout


@pytest.mark.parametrize("consent", (False, True))
def test_readout_baseline_reconstructs_source_values_and_anonymous_labels(tmp_path: Path, consent: bool) -> None:
    pool, commitment, readout = _case(tmp_path, consent=consent)
    verify_trialdev_readout_patients(pool_root=pool, commitment_path=commitment, readout_root=readout)


@pytest.mark.parametrize(
    "defect",
    [
        "clinical_value",
        "consent_value",
        "category_code",
        "wrong_category",
        "private_column",
        "missing_patient",
        "wrong_label",
        "future_enrollment",
        "duplicate_patient",
    ],
)
def test_readout_rejects_baseline_and_commitment_substitution(tmp_path: Path, defect: str) -> None:
    pool, commitment, readout = _case(tmp_path, consent=defect == "consent_value")
    path = readout / "arms/active/baseline.parquet"
    data = pd.read_parquet(path)
    if defect == "clinical_value":
        data["severity"] = 9.9
    elif defect == "consent_value":
        data["consent_time"] = -1.0
    elif defect == "category_code":
        data["sex"] = 1
    elif defect == "wrong_category":
        data["sex"] = "male"
    elif defect == "private_column":
        data["latent_prognosis"] = 0.7
    elif defect == "missing_patient":
        data = data.iloc[:0]
    elif defect == "wrong_label":
        data["participant_id"] = "p-a"
    else:
        payload = json.loads(commitment.read_text())
        if defect == "future_enrollment":
            payload["patients"][0]["enrollment_day"] = 101
        else:
            payload["patients"][1]["participant_id"] = 10
        commitment.write_text(json.dumps(payload))
    data.to_parquet(path, index=False)
    with pytest.raises(ValueError):
        verify_trialdev_readout_patients(pool_root=pool, commitment_path=commitment, readout_root=readout)


def _projection_context() -> tuple[tuple[_Patient, ...], _Readout]:

    return (
        (_Patient(participant_id=1, participant_label="p-1", arm_id="active", enrollment_day=80),),
        _Readout(
            study_id="study",
            outcome_object="Protocol-defined clinical event",
            readout_day=110,
            follow_up_days=28,
            arms=(),
            assessment_days=(0, 14, 28),
            surveillance_days=(0, 14, 28),
        ),
    )


def test_measurement_projection_respects_delay_schedule_and_missingness() -> None:
    from trialagentbench_verifier.authored_readouts import _record_projection

    patients, context = _projection_context()
    source = pd.DataFrame(
        {
            "participant_id": [1] * 4,
            "outcome_id": ["severity"] * 4,
            "occurrence_time": [0, 14, 28, 35],
            "availability_time": [0, 16, 35, 35],
            "observed": [True, False, True, True],
            "attended": [True] * 4,
            "observed_outcome": [2.5, 999.0, 1.2, 1.0],
            "private_effect": [42] * 4,
        }
    )
    result = _record_projection(source, patients, context, "measurements")
    assert result.participant_id.tolist() == ["p-1", "p-1"]
    assert result.occurrence_time.tolist() == [80, 94]
    assert result.availability_time.tolist() == [80, 96]
    assert result.observed_outcome.iloc[0] == 2.5
    assert pd.isna(result.observed_outcome.iloc[1])
    assert "private_effect" not in result


@pytest.mark.parametrize(
    "ascertainment, report_time, available", [("scheduled", float("nan"), 97), ("urgent", 16.0, 99)]
)
def test_safety_projection_cannot_reveal_future_reports_or_resolutions(ascertainment, report_time, available) -> None:
    from trialagentbench_verifier.authored_readouts import _record_projection

    patients, context = _projection_context()
    source = pd.DataFrame(
        {
            "participant_id": [1, 1],
            "event_id": ["first", "future"],
            "occurrence_time": [5.0, 25.0],
            "availability_time": [5.0, 25.0],
            "event_at_contact": [True, True],
            "reported": [True, True],
            "ascertainment": [ascertainment, ascertainment],
            "_spontaneous_observation_time": [report_time, 30.0 if ascertainment == "urgent" else float("nan")],
            "_recording_delay_days": [3.0, 3.0],
            "resolution_time": [20.0, 27.0],
            "medical_code": ["A", "B"],
            "coding_system": ["test", "test"],
            "severity": [2, 1],
            "serious": [True, False],
            "seriousness_criteria": ["AESHOSP|AESLIFE", ""],
        }
    )
    result = _record_projection(source, patients, context, "safety_events")
    assert result.event_id.tolist() == ["first"]
    assert result.occurrence_time.tolist() == [85]
    assert result.availability_time.tolist() == [available]
    assert result.seriousness_criteria.tolist() == ["AESHOSP|AESLIFE"]
    unknown = _record_projection(source.drop(columns="seriousness_criteria"), patients, context, "safety_events")
    assert pd.isna(unknown.seriousness_criteria.iloc[0])
    assert pd.isna(result.resolution_time.iloc[0])
    assert not any(str(column).startswith("_") for column in result)


def test_recurrent_projection_preserves_boundary_event_and_censors_future_end() -> None:
    from trialagentbench_verifier.authored_readouts import _record_projection

    patients, context = _projection_context()
    source = pd.DataFrame(
        {
            "participant_id": [1] * 4,
            "outcome_id": ["events"] * 4,
            "event_time": [5.0, 28.0, 29.0, 50.0],
            "event": [1, 1, 1, 0],
        }
    )
    result = _record_projection(source, patients, context, "recurrent_events")
    # Released recurrent times are programme days (enrolment on day 80), like every readout time.
    assert result.loc[result.event.eq(1), "occurrence_time"].tolist() == [85, 108]
    assert result.loc[result.event.eq(0), "occurrence_time"].tolist() == [108]


@pytest.mark.parametrize("assessment_days, terminal_observed", [((0, 28), True), ((0, 14, 28), False)])
def test_measurement_dropout_uses_committed_history_before_report_cutoff(assessment_days, terminal_observed):
    from trialagentbench_verifier.authored_readouts import _record_projection

    patients, readout = _projection_context()
    readout = readout.model_copy(update={"assessment_days": assessment_days})
    source = pd.DataFrame(
        {
            "participant_id": [1, 1, 1],
            "outcome_id": "score",
            "occurrence_time": [0, 14, 28],
            "availability_time": [0, 90, 28],
            "attended": True,
            "observed": [True, False, False],
            "observed_outcome": [5.0, float("nan"), float("nan")],
            "_measurement_attempt_succeeded": [True, False, True],
        }
    )
    keys = ["participant_id", "outcome_id", "occurrence_time"]
    potentials = source[keys].assign(complete_recorded_outcome=[5.0, 4.0, 3.0])
    result = _record_projection(source.iloc[::-1], patients, readout, "measurements", complete_measurements=potentials)
    assert result.occurrence_time.tolist() == [80, 108]
    assert result.observed.tolist() == [True, terminal_observed]
    if terminal_observed:
        assert result.observed_outcome.tolist() == [5.0, 3.0]
    else:
        assert pd.isna(result.observed_outcome.iloc[-1])
    assert not set(result.columns) & {"_measurement_attempt_succeeded", "complete_recorded_outcome"}
    with pytest.raises(ValueError, match="cover committed"):
        _record_projection(source, patients, readout, "measurements", complete_measurements=potentials.iloc[:-1])


def test_readout_cannot_substitute_programme_calendar(tmp_path: Path) -> None:
    """Agreement between public and private readouts cannot override pool metadata."""
    pool, commitment, readout = _case(tmp_path)
    payload = json.loads(commitment.read_text())
    payload["readout"]["calendar_origin"] = "2024-02-28"
    commitment.write_text(json.dumps(payload))
    (readout / "study.json").write_text(json.dumps(payload["readout"]))
    with pytest.raises(ValueError, match="calendar origin"):
        verify_trialdev_readout_patients(pool_root=pool, commitment_path=commitment, readout_root=readout)


@pytest.mark.parametrize("alter_commitment", [False, True])
def test_readout_rejects_substituted_clinical_object(tmp_path: Path, alter_commitment: bool) -> None:
    """Clinical meaning must match both the frozen source and public commitment."""
    pool, commitment, readout = _case(tmp_path)
    study_path = readout / "study.json"
    study = json.loads(study_path.read_text())
    study["outcome_object"] = "A different clinical condition"
    study_path.write_text(json.dumps(study))
    if alter_commitment:
        payload = json.loads(commitment.read_text())
        payload["readout"] = study
        commitment.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="clinical object|identity or cutoff"):
        verify_trialdev_readout_patients(pool_root=pool, commitment_path=commitment, readout_root=readout)
