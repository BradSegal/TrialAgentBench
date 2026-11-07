"""Public projection integrity and same-record reconstruction."""

import io
import json
import runpy
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import trialagentbench_verifier

from trialagentbench.io import write_json_model
from trialagentbench.trialdev.analysis_ready import write_trialdev_analysis_ready
from trialagentbench.trialdev.materialization import TrialDevReadoutArm, TrialDevReadoutManifest, TrialDevReadoutTarget
from trialagentbench.trialdev.resources import TrialDevStudyCost, trialdev_phase_tariff


def _readout(root: Path, endpoint: str = "measurements") -> Path:
    root.mkdir()
    write_json_model(
        root / "study.json",
        TrialDevReadoutManifest(
            study_id="study",
            outcome_object="Protocol-defined clinical event",
            readout_day=28,
            follow_up_days=28,
            assessment_days=(28,),
            surveillance_days=(28,),
            arms=tuple(TrialDevReadoutArm(arm_id=name, asset_id=name, dose=None) for name in ("a", "b")),
            targets=(
                TrialDevReadoutTarget(
                    target_id="a:clinical_outcome", arm_id="a", endpoint="clinical_outcome", unit="points"
                ),
            ),
            tariff=trialdev_phase_tariff("early_safety"),
            sites=2,
            screened=2,
            cost=TrialDevStudyCost(accrued_usd=100, remaining_liability_usd=0),
        ),
    )
    for name in ("a", "b"):
        arm = root / "arms" / name
        arm.mkdir(parents=True)
        pd.DataFrame({"participant_id": [name], "age": [55], "enrollment_day": [0.0]}).to_parquet(
            arm / "baseline.parquet"
        )
        for domain in ("contacts", "exposure"):
            frame = pd.DataFrame({"participant_id": [name, name], "occurrence_time": [0.0, 28.0]})
            if domain == "exposure":
                frame["received_arm"] = [name, "b"]
            frame.to_parquet(arm / f"{domain}.parquet")
        pd.DataFrame(
            {
                "participant_id": [name],
                "event_id": ["headache"],
                "occurrence_time": [12.5],
                "availability_time": [14.0],
                "resolution_time": [15.0],
                "serious": [False],
                "severity": ["mild"],
            }
        ).to_parquet(arm / "safety_events.parquet")
        clinical = (
            {"occurrence_time": [0.0, 28.0], "observed_outcome": [1.0, float("nan")], "observed": [True, False]}
            if endpoint == "measurements"
            else {"occurrence_time": [12.5, 28.0], "event": [1, 0]}
        )
        pd.DataFrame({"participant_id": [name, name], **clinical}).to_parquet(arm / f"{endpoint}.parquet")
    return root


@pytest.mark.parametrize("endpoint", ["measurements", "recurrent_events"])
def test_projection_preserves_observations_and_missingness(tmp_path: Path, endpoint: str) -> None:
    raw = _readout(tmp_path / "raw", endpoint)
    ready = write_trialdev_analysis_ready(raw, tmp_path / "ready")
    assert write_trialdev_analysis_ready(raw, ready) == ready
    combined = pd.read_parquet(ready / f"{endpoint}.parquet")
    for arm in ("a", "b"):
        original = pd.read_parquet(raw / "arms" / arm / f"{endpoint}.parquet")
        pd.testing.assert_frame_equal(
            combined.loc[combined.arm_id.eq(arm), original.columns].reset_index(drop=True), original
        )


def test_independent_reader_reconciles_programme_clock_recurrent_times(tmp_path: Path) -> None:
    """Recurrent event times are programme days like every other readout time (one time origin)."""
    raw = _readout(tmp_path / "raw", "recurrent_events")
    study = TrialDevReadoutManifest.model_validate_json((raw / "study.json").read_text())
    write_json_model(raw / "study.json", study.model_copy(update={"readout_day": 60.0}))
    for path in raw.glob("arms/b/*.parquet"):
        frame = pd.read_parquet(path)
        for field in ("enrollment_day", "occurrence_time", "availability_time", "resolution_time"):
            if field in frame:
                frame[field] += 30.0
        frame.to_parquet(path)
    ready = write_trialdev_analysis_ready(raw, tmp_path / "ready")
    reader = runpy.run_path(
        str(Path(trialagentbench_verifier.__file__).resolve().parents[2] / "examples/scientific/review_standards.py")
    )
    checked = reader["reconcile_readout"](raw, ready / "standards.zip")
    assert checked.status.eq("passed").all()
    # An event after the readout day is future evidence.
    path = raw / "arms" / "b" / "recurrent_events.parquet"
    pd.read_parquet(path).assign(occurrence_time=[42.5, 61.0]).to_parquet(path)
    with pytest.raises(ValueError, match="future record"):
        reader["reconcile_readout"](raw, ready / "standards.zip")


@pytest.mark.parametrize("origin", [12.2, 12.3])
def test_fractional_enrollment_preserves_final_day_and_epoch(tmp_path: Path, origin: float) -> None:
    """The public calendar round trip must not lose the inclusive final visit."""
    raw = _readout(tmp_path / "raw")
    study = TrialDevReadoutManifest.model_validate_json((raw / "study.json").read_text())
    write_json_model(
        raw / "study.json",
        study.model_copy(
            update={"readout_day": 50.0, "calendar_origin": date(2024, 1, 1), "epochs": (("TREATMENT", 0.0, 28.0),)}
        ),
    )
    for path in raw.glob("arms/*/*.parquet"):
        frame = pd.read_parquet(path)
        for field in ("enrollment_day", "occurrence_time", "availability_time", "resolution_time"):
            if field in frame:
                frame[field] += origin
        frame.to_parquet(path)
    ready = write_trialdev_analysis_ready(raw, tmp_path / "ready")
    with zipfile.ZipFile(ready / "standards.zip") as source:
        for domain, day in (("QS", "QSDY"), ("EX", "EXSTDY")):
            frame = pd.read_parquet(io.BytesIO(source.read(f"{domain}.parquet")))
            assert frame.EPOCH.eq("TREATMENT").all()
            assert frame[day].tolist() == [1, 29, 1, 29]
    joined = pd.read_parquet(ready / "measurements.parquet")
    assert joined.occurrence_time.tolist() == [origin, origin + 28, origin, origin + 28]


@pytest.mark.parametrize("domain", ["baseline", "contacts"])
def test_projection_rejects_inconsistent_columns(tmp_path: Path, domain: str) -> None:
    raw = _readout(tmp_path / "raw")
    path = raw / "arms" / "b" / f"{domain}.parquet"
    pd.read_parquet(path).assign(extra=1).to_parquet(path)
    with pytest.raises(ValueError, match="columns differ"):
        write_trialdev_analysis_ready(raw, tmp_path / "ready")
    assert not (tmp_path / "ready").exists()


def test_projection_rejects_mixed_endpoint_domains(tmp_path: Path) -> None:
    raw = _readout(tmp_path / "raw")
    arm = raw / "arms" / "b"
    (arm / "measurements.parquet").rename(arm / "recurrent_events.parquet")
    with pytest.raises(ValueError, match="same clinical"):
        write_trialdev_analysis_ready(raw, tmp_path / "ready")


def test_projection_rejects_unbound_extra_evidence(tmp_path: Path) -> None:
    raw = _readout(tmp_path / "raw")
    ready = write_trialdev_analysis_ready(raw, tmp_path / "ready")
    (ready / "extra.csv").write_text("unbound evidence")
    with pytest.raises(ValueError, match="inventory"):
        write_trialdev_analysis_ready(raw, ready)


@pytest.mark.parametrize("clock", ["occurrence_time", "availability_time", "resolution_time"])
def test_projection_rejects_future_safety_evidence(tmp_path: Path, clock: str) -> None:
    raw = _readout(tmp_path / "raw")
    path = raw / "arms" / "a" / "safety_events.parquet"
    pd.read_parquet(path).assign(**{clock: 29.0}).to_parquet(path)
    with pytest.raises(ValueError, match="after the readout cutoff"):
        write_trialdev_analysis_ready(raw, tmp_path / "ready")
    assert not (tmp_path / "ready").exists()


def test_projection_retains_a_readout_without_endpoint_observations(tmp_path: Path) -> None:
    raw = _readout(tmp_path / "raw")
    for path in raw.glob("arms/*/measurements.parquet"):
        pd.read_parquet(path).iloc[:0].to_parquet(path, index=False)
    ready = write_trialdev_analysis_ready(raw, tmp_path / "ready")
    assert pd.read_parquet(ready / "measurements.parquet").empty
    assert (ready / "standards.zip").is_file()


@pytest.mark.parametrize("damage", [None, "treatment", "time", "subject", "missing"])
def test_exposure_projection_reconciles_public_receipt_records(tmp_path: Path, damage: str | None) -> None:
    """The independent reader detects changed or omitted receipt records."""
    raw = _readout(tmp_path / "raw")
    ready = write_trialdev_analysis_ready(raw, tmp_path / "ready")
    archive = ready / "standards.zip"
    reader = runpy.run_path(
        str(Path(trialagentbench_verifier.__file__).resolve().parents[2] / "examples/scientific/review_standards.py")
    )
    if damage is None:
        checked = reader["reconcile_readout"](raw, archive)
        assert "recorded treatment receipt" in set(checked.quantity)
        with zipfile.ZipFile(archive) as source:
            exposure = pd.read_parquet(io.BytesIO(source.read("EX.parquet")))
        assert exposure.EXTRT.tolist() == ["a", "b", "b", "b"]
        assert not {"EXDOSE", "EXDOSU", "EXDOSFRQ"} & set(exposure)
        assert exposure.EXSTDTC.eq("").all()
        return
    with zipfile.ZipFile(archive) as source:
        payload = {name: source.read(name) for name in source.namelist()}
    name = "SUPPEX.parquet" if damage == "time" else "EX.parquet"
    if damage == "missing":
        del payload[name]
    else:
        frame = pd.read_parquet(io.BytesIO(payload[name]))
        frame.loc[0, {"treatment": "EXTRT", "time": "QVAL", "subject": "USUBJID"}[damage]] = {
            "treatment": "wrong",
            "time": "100",
            "subject": "absent",
        }[damage]
        payload[name] = frame.to_parquet(index=False)
    changed = tmp_path / "damaged.zip"
    with zipfile.ZipFile(changed, "w") as target:
        for name, body in payload.items():
            target.writestr(name, body)
    with pytest.raises((AssertionError, ValueError, KeyError)):
        reader["reconcile_readout"](raw, changed)


@pytest.mark.parametrize("damage", ["participant", "duplicate", "time", "treatment"])
def test_readout_rejects_invalid_exposure_sources(tmp_path: Path, damage: str) -> None:
    """Invalid receipt evidence cannot produce an apparently usable analysis view."""
    raw = _readout(tmp_path / "raw")
    path = raw / "arms" / "a" / "exposure.parquet"
    records = pd.read_parquet(path)
    if damage == "duplicate":
        records = pd.concat([records, records.iloc[[0]]], ignore_index=True)
    else:
        column, value = {
            "participant": ("participant_id", "absent"),
            "time": ("occurrence_time", float("nan")),
            "treatment": ("received_arm", None),
        }[damage]
        records.loc[0, column] = value
    records.to_parquet(path, index=False)
    with pytest.raises(ValueError):
        write_trialdev_analysis_ready(raw, tmp_path / "ready")
    assert not (tmp_path / "ready").exists()


def test_readout_rejects_a_previous_projection_contract(tmp_path: Path) -> None:
    """A cached export without the current receipt contract must be regenerated."""
    raw = _readout(tmp_path / "raw")
    ready = write_trialdev_analysis_ready(raw, tmp_path / "ready")
    manifest = ready / "manifest.json"
    payload = json.loads(manifest.read_text())
    payload["schema_id"] = "trialdev.analysis-ready/v2"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="trialdev.analysis-ready/v3"):
        write_trialdev_analysis_ready(raw, ready)


def test_authored_calendar_reaches_analysis_ready_dates(tmp_path: Path) -> None:
    """A public programme origin reaches projections without private pool access."""
    raw = _readout(tmp_path / "raw")
    path = raw / "study.json"
    payload = json.loads(path.read_text())
    payload["calendar_origin"] = "2024-02-28"
    path.write_text(json.dumps(payload))
    ready = write_trialdev_analysis_ready(raw, tmp_path / "ready")
    with zipfile.ZipFile(ready / "standards.zip") as archive:
        dm = pd.read_parquet(io.BytesIO(archive.read("DM.parquet")))
        ae = pd.read_parquet(io.BytesIO(archive.read("AE.parquet")))
    assert dm.RFSTDTC.eq("2024-02-28").all()
    assert ae.AESTDTC.eq("2024-03-11").all()
    reader = runpy.run_path(
        str(Path(trialagentbench_verifier.__file__).resolve().parents[2] / "examples/scientific/review_standards.py")
    )
    assert not reader["reconcile_readout"](raw, ready / "standards.zip").empty
    payload["calendar_origin"] = "2024-03-01"
    path.write_text(json.dumps(payload))
    with pytest.raises(AssertionError):
        reader["reconcile_readout"](raw, ready / "standards.zip")
    with pytest.raises(ValueError, match="bound public sources"):
        write_trialdev_analysis_ready(raw, ready)


@pytest.mark.parametrize(
    "domain,clock,value",
    [
        ("baseline", "enrollment_day", float("-inf")),
        ("baseline", "enrollment_day", -1.0),
        ("contacts", "occurrence_time", float("nan")),
        ("contacts", "documentation_time", float("inf")),
        ("contacts", "observation_time", 29.0),
        ("contacts", "occurrence_time", -1.0),
        ("safety_events", "availability_time", 11.0),
        ("safety_events", "resolution_time", 11.0),
    ],
)
def test_projection_rejects_invalid_public_clocks(tmp_path: Path, domain: str, clock: str, value: float) -> None:
    """All public clocks obey the participant and readout temporal boundaries."""
    raw = _readout(tmp_path / "raw")
    path = raw / "arms" / "a" / f"{domain}.parquet"
    records = pd.read_parquet(path)
    records.loc[0, clock] = value
    records.to_parquet(path, index=False)
    with pytest.raises(ValueError):
        write_trialdev_analysis_ready(raw, tmp_path / "ready")
    assert not (tmp_path / "ready").exists()


def test_projection_preserves_unresolved_safety_events(tmp_path: Path) -> None:
    """An unknown resolution remains missing instead of inventing an event end."""
    raw = _readout(tmp_path / "raw")
    for path in raw.glob("arms/*/safety_events.parquet"):
        pd.read_parquet(path).assign(resolution_time=float("nan")).to_parquet(path, index=False)
    ready = write_trialdev_analysis_ready(raw, tmp_path / "ready")
    assert pd.read_parquet(ready / "safety_events.parquet").resolution_time.isna().all()


@pytest.mark.parametrize("domain", ["QS", "AE", "EX"])
def test_authored_epochs_are_reconstructed_independently(tmp_path: Path, domain: str) -> None:
    """Period labels use elapsed follow-up and changes are detected by the reader."""
    raw = _readout(tmp_path / "raw")
    path = raw / "study.json"
    study = json.loads(path.read_text())
    study["epochs"] = [["TREATMENT", 0, 14], ["FOLLOW-UP", 14, 28]]
    path.write_text(json.dumps(study))
    ready = write_trialdev_analysis_ready(raw, tmp_path / "ready")
    reader = runpy.run_path(
        str(Path(trialagentbench_verifier.__file__).resolve().parents[2] / "examples/scientific/review_standards.py")
    )
    reader["reconcile_readout"](raw, ready / "standards.zip")
    with zipfile.ZipFile(ready / "standards.zip") as archive:
        payload = {name: archive.read(name) for name in archive.namelist()}
    name = f"{domain}.parquet"
    table = pd.read_parquet(io.BytesIO(payload[name]))
    assert set(table.EPOCH) <= {"TREATMENT", "FOLLOW-UP"}
    table.loc[0, "EPOCH"] = "WRONG"
    payload[name] = table.to_parquet(index=False)
    damaged = tmp_path / "damaged.zip"
    with zipfile.ZipFile(damaged, "w") as archive:
        for name, body in payload.items():
            archive.writestr(name, body)
    with pytest.raises(AssertionError):
        reader["reconcile_readout"](raw, damaged)


@pytest.mark.parametrize(
    "epochs",
    [
        [["TREATMENT", 0, 27]],
        [["TREATMENT", 1, 28]],
        [["TREATMENT", 0, 14], ["FOLLOW-UP", 15, 28]],
        [["TREATMENT", 0, 15], ["FOLLOW-UP", 14, 28]],
    ],
)
def test_readout_rejects_incomplete_epoch_declarations(tmp_path: Path, epochs: list) -> None:
    """A durable readout cannot publish inconsistent planned periods."""
    raw = _readout(tmp_path / "raw")
    path = raw / "study.json"
    study = json.loads(path.read_text())
    study["epochs"] = epochs
    path.write_text(json.dumps(study))
    with pytest.raises(ValueError, match="epoch"):
        write_trialdev_analysis_ready(raw, tmp_path / "ready")


def test_projection_rejects_text_clock_values(tmp_path: Path) -> None:
    """A populated nonnumeric clock fails at the public input boundary."""
    raw = _readout(tmp_path / "raw")
    path = raw / "arms" / "a" / "safety_events.parquet"
    pd.read_parquet(path).assign(occurrence_time="12.5").to_parquet(path, index=False)
    with pytest.raises(ValueError, match="clocks must be finite"):
        write_trialdev_analysis_ready(raw, tmp_path / "ready")


@pytest.mark.parametrize("damage", [None, "participation", "missing", "future"])
def test_discontinuation_is_treatment_disposition_from_public_records(tmp_path: Path, damage: str | None) -> None:
    """Readout-qualified discontinuations never imply withdrawal or completion."""
    raw = _readout(tmp_path / "raw")
    for arm in ("a", "b"):
        pd.DataFrame(
            {
                "participant_id": [arm, arm, arm],
                "event_id": ["stop_later", "rescue", "stop_first"],
                "event_kind": ["treatment_discontinuation", "rescue_medication", "treatment_discontinuation"],
                "occurrence_time": [15.0, 4.0, 7.0],
                "availability_time": [16.0, 5.0, 29.0 if damage == "future" else 8.0],
            }
        ).to_parquet(raw / "arms" / arm / "intercurrent_events.parquet", index=False)
    if damage == "future":
        with pytest.raises(ValueError, match="after the readout cutoff"):
            write_trialdev_analysis_ready(raw, tmp_path / "ready")
        return
    ready = write_trialdev_analysis_ready(raw, tmp_path / "ready")
    archive = ready / "standards.zip"
    reader = runpy.run_path(
        str(Path(trialagentbench_verifier.__file__).resolve().parents[2] / "examples/scientific/review_standards.py")
    )
    checked = reader["reconcile_readout"](raw, archive)
    assert "recorded treatment discontinuation" in set(checked.quantity)
    with zipfile.ZipFile(archive) as source:
        contents = {name: source.read(name) for name in source.namelist()}
    ds = pd.read_parquet(io.BytesIO(contents["DS.parquet"]))
    assert ds.USUBJID.tolist() == ["study-a", "study-b"]
    assert ds.DSSCAT.eq("STUDY TREATMENT").all()
    assert ds.DSDECOD.eq("DISCONTINUED").all()
    assert ds.DSSTDTC.eq("").all()
    if damage is None:
        return
    ds = ds.iloc[:1] if damage == "missing" else ds.assign(DSSCAT="STUDY PARTICIPATION")
    with zipfile.ZipFile(archive, "w") as destination:
        for member, data in contents.items():
            destination.writestr(member, ds.to_parquet(index=False) if member == "DS.parquet" else data)
    with pytest.raises((AssertionError, ValueError)):
        reader["reconcile_readout"](raw, archive)


@pytest.mark.parametrize("damage", [None, "future", "backwards", "missing"])
def test_completion_domain_reaches_shared_standards_and_reader(tmp_path, damage):
    """Administrative source forms retain clocks and independently checked membership."""
    raw = _readout(tmp_path / "raw")
    for arm in ("a", "b"):
        pd.DataFrame(
            {
                "participant_id": [arm],
                "occurrence_time": [20.0],
                "observation_time": [20.0],
                "documentation_time": [22.0],
                "availability_time": [29.0 if damage == "future" else 21.0 if damage == "backwards" else 24.0],
                "status": ["completed"],
            }
        ).to_parquet(raw / "arms" / arm / "completion_records.parquet", index=False)
    if damage in {"future", "backwards"}:
        with pytest.raises(ValueError):
            write_trialdev_analysis_ready(raw, tmp_path / "ready")
        return
    ready = write_trialdev_analysis_ready(raw, tmp_path / "ready")
    assert set(pd.read_parquet(ready / "completion_records.parquet").participant_id) == {"a", "b"}
    reader = runpy.run_path(
        str(Path(trialagentbench_verifier.__file__).resolve().parents[2] / "examples/scientific/review_standards.py")
    )
    archive = ready / "standards.zip"
    assert reader["reconcile_readout"](raw, archive).quantity.eq("recorded completion").any()
    if damage == "missing":
        with zipfile.ZipFile(archive) as source:
            contents = {name: source.read(name) for name in source.namelist()}
        ds = pd.read_parquet(io.BytesIO(contents["DS.parquet"]))
        with zipfile.ZipFile(archive, "w") as destination:
            for name, data in contents.items():
                destination.writestr(name, ds.iloc[:1].to_parquet(index=False) if name == "DS.parquet" else data)
        with pytest.raises(ValueError, match="completion"):
            reader["reconcile_readout"](raw, archive)


def test_delayed_clinical_forms_and_resolution_share_exact_raw_ready_cutoffs(tmp_path: Path) -> None:
    """Delayed forms and event resolution cross one exact cutoff without rewriting earlier evidence."""
    from clinical_data_tables import available_clinical_records

    from trialagentbench.io import sha256_file
    from trialagentbench.trialdev.observations import (
        trialdev_contact_records,
        trialdev_exposure_records,
        trialdev_intercurrent_records,
        trialdev_measurement_records,
        trialdev_safety_records,
    )

    enrollment = pd.DataFrame({"participant_id": ["a", "b"], "enrollment_day": 10.25})
    measurements = pd.DataFrame(
        {
            "participant_id": ["a", "b"],
            "outcome_id": "primary",
            "occurrence_time": 28.0,
            "availability_time": 30.0,
            "attended": True,
            "observed": [True, False],
            "observed_outcome": [5.0, 999.0],
        }
    )
    safety = pd.DataFrame(
        {
            "participant_id": ["a", "b"],
            "event_id": "infection",
            "occurrence_time": 12.0,
            "availability_time": [16.0, 15.0],
            "event_at_contact": True,
            "reported": True,
            "medical_code": "infection",
            "coding_system": "clinical",
            "dictionary_term": "Infection",
            "severity": "mild",
            "serious": False,
            "seriousness_criteria": "",
            "resolution_time": 27.0,
            "ascertainment": ["scheduled", "urgent"],
            "_spontaneous_observation_time": [15.0, 13.0],
            "_recording_delay_days": 2.0,
        }
    )
    intercurrent = pd.DataFrame(
        {
            "participant_id": ["a", "b"],
            "event_id": "rescue",
            "event_kind": "rescue_medication",
            "occurrence_time": 14.0,
            "availability_time": 30.0,
            "event_at_contact": True,
        }
    )
    completion = pd.DataFrame(
        {
            "participant_id": ["a", "b"],
            "occurrence_time": 28.0,
            "observation_time": 28.0,
            "documentation_time": 29.0,
            "availability_time": 30.0,
            "status": ["completed", "not_completed"],
        }
    )
    contacts = pd.DataFrame(
        {
            "participant_id": ["a", "a", "b", "b"],
            "occurrence_time": [14.0, 28.0, 14.0, 28.0],
            "availability_time": [14.0, 30.0, 14.0, 30.0],
        }
    )
    exposure = pd.DataFrame(
        {"participant_id": ["a", "b"], "occurrence_time": 0.0, "received_arm": ["control", "active"]}
    )
    reader = runpy.run_path(
        str(Path(trialagentbench_verifier.__file__).resolve().parents[2] / "examples/scientific/review_standards.py")
    )
    published = {}
    cutoff = 40.25
    for label, day, count in (
        ("before", np.nextafter(cutoff, -np.inf), 0),
        ("at", cutoff, 2),
        ("after", np.nextafter(cutoff, np.inf), 2),
    ):
        raw = _readout(tmp_path / f"raw-{label}")
        study = TrialDevReadoutManifest.model_validate_json((raw / "study.json").read_text())
        write_json_model(
            raw / "study.json", study.model_copy(update={"readout_day": day, "surveillance_days": (14, 28)})
        )
        args = dict(enrollment=enrollment, readout_day=day, follow_up_days=28)
        completed = available_clinical_records(completion, **args)
        clocks = ("occurrence_time", "observation_time", "documentation_time", "availability_time")
        for clock in clocks:
            completed[clock] += completed.enrollment_day
        frames = {
            "measurements": trialdev_measurement_records(measurements, assessment_days=(28,), **args),
            "contacts": trialdev_contact_records(contacts, surveillance_days=(14, 28), **args),
            "safety_events": trialdev_safety_records(safety, surveillance_days=(14, 28), **args),
            "exposure": trialdev_exposure_records(exposure, **args),
            "intercurrent_events": trialdev_intercurrent_records(intercurrent, **args),
            "completion_records": completed.drop(columns="enrollment_day"),
        }
        frames["exposure"]["received_arm"] = frames["exposure"].received_arm.map({"control": "a", "active": "b"})
        for arm in ("a", "b"):
            directory = raw / "arms" / arm
            baseline = pd.read_parquet(directory / "baseline.parquet").assign(enrollment_day=10.25)
            baseline.to_parquet(directory / "baseline.parquet", index=False)
            for domain, frame in frames.items():
                frame.loc[frame.participant_id.eq(arm)].to_parquet(directory / f"{domain}.parquet", index=False)
        source_hashes = {path: sha256_file(path) for path in raw.rglob("*") if path.is_file()}
        ready = write_trialdev_analysis_ready(raw, tmp_path / f"ready-{label}")
        assert write_trialdev_analysis_ready(raw, ready) == ready
        for domain, source in frames.items():
            actual = pd.read_parquet(ready / f"{domain}.parquet")
            pd.testing.assert_frame_equal(actual[source.columns], source.reset_index(drop=True))
            assert not any(column.startswith("_") for column in actual)
        for domain in ("measurements", "intercurrent_events", "completion_records"):
            assert len(frames[domain]) == count
        assert len(frames["contacts"]) == 2 + count
        assert frames["safety_events"].resolution_time.notna().sum() == count
        assert frames["safety_events"].availability_time.tolist() == [26.25, 25.25]
        if count:
            values = frames["measurements"].set_index("participant_id").observed_outcome
            assert values.loc["a"] == 5.0
            assert pd.isna(values.loc["b"])
            assert frames["completion_records"].status.tolist() == ["completed", "not_completed"]
        assert not reader["reconcile_readout"](raw, ready / "standards.zip").empty
        assert all(sha256_file(path) == digest for path, digest in source_hashes.items())
        published.update(source_hashes)
        published.update({path: sha256_file(path) for path in ready.iterdir()})
    assert all(sha256_file(path) == digest for path, digest in published.items())
    assert measurements.observed_outcome.tolist() == [5.0, 999.0]
    assert safety.resolution_time.eq(27.0).all()


def test_baseline_metadata_survives_standards_projection_and_binds_reuse(tmp_path):
    """The public dictionary supplies units; changed metadata cannot reuse old tables."""
    from trialagentbench.clinical_tables import StudyBaselineFieldV1
    from trialagentbench.trialdev.analysis_ready import trialdev_baseline_fields_from_dictionary

    raw = _readout(tmp_path / "raw")
    dictionary = tmp_path / "dictionary.csv"
    dictionary.write_text(
        "variable,label,role,timing,unit_or_terminology\nage,Recorded age,covariate,baseline,years\n"
    )
    fields = trialdev_baseline_fields_from_dictionary(dictionary)
    ready = write_trialdev_analysis_ready(raw, tmp_path / "ready", baseline_fields=fields)
    with zipfile.ZipFile(ready / "standards.zip") as archive:
        metadata = pd.read_csv(io.BytesIO(archive.read("variables.csv")), keep_default_na=False)
    age = metadata.loc[metadata.source_field.eq("age")]
    assert not age.empty and age.unit.eq("years").all()
    assert age.label.eq("Recorded age").all()
    assert write_trialdev_analysis_ready(raw, ready, baseline_fields=fields) == ready
    with pytest.raises(ValueError, match="bound public sources"):
        write_trialdev_analysis_ready(
            raw,
            ready,
            baseline_fields=(StudyBaselineFieldV1(field_id="age", label="Recorded age", unit="months"),),
        )


def test_bundled_trialdev_dictionary_supplies_readout_baseline_metadata(tmp_path: Path) -> None:
    """The offline demonstration's actual participant dictionary supports study readouts."""
    from trialagentbench.trialdev.analysis_ready import trialdev_baseline_fields_from_dictionary

    fixture = Path(__file__).resolve().parents[2] / "examples/clinical/demo-trialdev-evaluator.zip"
    with zipfile.ZipFile(fixture) as archive:
        names = [name for name in archive.namelist() if name.endswith("/documents/observational_data_dictionary.csv")]
        assert len(names) == 1
        dictionary = tmp_path / "dictionary.csv"
        dictionary.write_bytes(archive.read(names[0]))
    fields = trialdev_baseline_fields_from_dictionary(dictionary)
    assert fields
    assert next(field for field in fields if field.field_id == "age_years").unit == "years"
