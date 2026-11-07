"""No future or counterfactual data in timed participant observations."""

import numpy as np
import pandas as pd
import pytest

from trialagentbench.trialdev.observations import trialdev_measurement_records, trialdev_safety_records


def test_readout_uses_enrollment_and_availability_not_just_visit_time():
    records = pd.DataFrame(
        {
            "participant_id": [1, 1, 1, 2],
            "outcome_id": "score",
            "occurrence_time": [0, 28, 56, 28],
            "availability_time": [0, 35, 56, 35],
            "attended": True,
            "observed": [True, False, True, True],
            "observed_outcome": [5.0, 999.0, 4.0, 3.0],
            "latent_risk": 999,
        }
    )
    enrollment = pd.DataFrame({"participant_id": [1, 2], "enrollment_day": [10, 30]})
    early = trialdev_measurement_records(
        records, enrollment, readout_day=40, follow_up_days=56, assessment_days=(0, 28, 56)
    )
    assert early.participant_id.tolist() == [1]
    assert early.occurrence_time.tolist() == [10]
    later = trialdev_measurement_records(
        records, enrollment, readout_day=50, follow_up_days=56, assessment_days=(0, 28, 56)
    )
    assert later.occurrence_time.tolist() == [10, 38]
    assert later.availability_time.tolist() == [10, 45]
    assert np.isnan(later.observed_outcome.iloc[1])
    assert "latent_risk" not in later
    assert records.observed_outcome.iloc[1] == 999
    final = trialdev_measurement_records(
        records, enrollment, readout_day=100, follow_up_days=28, assessment_days=(0, 28)
    )
    assert len(final) == 3
    assert 66 not in final.occurrence_time.tolist()


def test_safety_hides_future_first_event_potentials_and_resolution():
    records = pd.DataFrame(
        {
            "participant_id": [1, 1],
            "event_id": "infection",
            "occurrence_time": [0, 20],
            "availability_time": [0, 25],
            "event_at_contact": [False, True],
            "reported": [None, True],
            "first_event_time": [20, 20],
            "potential_event__control_by_contact": True,
            "medical_code": [None, "infection"],
            "coding_system": [None, "clinical"],
            "severity": [None, "moderate"],
            "serious": [None, True],
            "resolution_time": [np.nan, 40],
        }
    )
    enrollment = pd.DataFrame({"participant_id": [1], "enrollment_day": [10]})
    assert trialdev_safety_records(records, enrollment, readout_day=30, follow_up_days=56).empty
    available = trialdev_safety_records(records, enrollment, readout_day=35, follow_up_days=56)
    assert len(available) == 1
    assert available.occurrence_time.item() == 30
    assert np.isnan(available.resolution_time.item())
    assert "first_event_time" not in available
    assert not any("potential" in column for column in available)
    resolved = trialdev_safety_records(records, enrollment, readout_day=50, follow_up_days=56)
    assert resolved.resolution_time.item() == 50
    short = trialdev_safety_records(records, enrollment, readout_day=100, follow_up_days=28)
    assert np.isnan(short.resolution_time.item())


def test_uncommitted_or_invalid_timing_is_rejected():
    records = pd.DataFrame({"participant_id": [1], "occurrence_time": [20], "availability_time": [10]})
    enrollment = pd.DataFrame({"participant_id": [1], "enrollment_day": [0]})
    with pytest.raises(ValueError, match="availability"):
        trialdev_measurement_records(records, enrollment, readout_day=30, follow_up_days=28, assessment_days=(28,))
    with pytest.raises(ValueError, match="assessment"):
        trialdev_measurement_records(records, enrollment, readout_day=30, follow_up_days=28, assessment_days=(56,))


def test_recurrent_readouts_hide_future_events_and_future_loss():
    from trialagentbench.trialdev.observations import trialdev_recurrent_records

    records = pd.DataFrame(
        {
            "participant_id": [1, 1, 1, 2],
            "outcome_id": "exacerbation",
            "event_time": [5, 30, 50, 12],
            "event": [1, 1, 0, 0],
            "private_frailty": 999,
        }
    )
    enrollment = pd.DataFrame({"participant_id": [1, 2], "enrollment_day": [10, 20]})
    early = trialdev_recurrent_records(records, enrollment, readout_day=25, follow_up_days=40)
    assert early.to_dict("records") == [
        {"participant_id": 1, "outcome_id": "exacerbation", "event_time": 5, "event": 1},
        {"participant_id": 1, "outcome_id": "exacerbation", "event_time": 15, "event": 0},
        {"participant_id": 2, "outcome_id": "exacerbation", "event_time": 5, "event": 0},
    ]
    changed_future = records.copy()
    changed_future.loc[1, "event_time"] = 40
    changed_future.loc[2, "event_time"] = 60
    pd.testing.assert_frame_equal(
        early, trialdev_recurrent_records(changed_future, enrollment, readout_day=25, follow_up_days=40)
    )
    late = trialdev_recurrent_records(records, enrollment, readout_day=100, follow_up_days=40)
    # Released times are days since enrolment (exposure accounting), not programme dates.
    assert late.loc[late.event.eq(0), "event_time"].tolist() == [40, 12]
    assert late.event.sum() == 2
    with pytest.raises(ValueError, match="one observed"):
        trialdev_recurrent_records(records.iloc[:-1], enrollment, readout_day=25, follow_up_days=40)
    invalid = records.copy()
    invalid.loc[1, "event_time"] = 51
    with pytest.raises(ValueError, match="outside"):
        trialdev_recurrent_records(invalid, enrollment, readout_day=25, follow_up_days=40)


def test_surveillance_contacts_do_not_assert_event_free_followup():
    from trialagentbench.trialdev.observations import trialdev_contact_records

    records = pd.DataFrame(
        {
            "participant_id": [1, 1, 1],
            "occurrence_time": [0, 28, 56],
            "availability_time": [0, 30, 56],
            "future_event": 50,
        }
    )
    enrollment = pd.DataFrame({"participant_id": [1], "enrollment_day": [10]})
    result = trialdev_contact_records(records, enrollment, readout_day=39, follow_up_days=56)
    assert result.to_dict("records") == [{"participant_id": 1, "occurrence_time": 10, "availability_time": 10}]
    complete = trialdev_contact_records(records, enrollment, readout_day=100, follow_up_days=28)
    assert complete.occurrence_time.tolist() == [10, 38]
    assert complete.columns.tolist() == ["participant_id", "occurrence_time", "availability_time"]
    with pytest.raises(ValueError, match="unique"):
        trialdev_contact_records(
            pd.concat([records, records.iloc[:1]]), enrollment, readout_day=100, follow_up_days=28
        )


def test_changed_surveillance_replays_detection_without_redrawing_events():
    records = pd.DataFrame(
        {
            "participant_id": [1],
            "event_id": ["infection"],
            "occurrence_time": [20.0],
            "availability_time": [30.0],
            "event_at_contact": [True],
            "reported": [True],
            "_spontaneous_observation_time": [40.0],
            "ascertainment": ["spontaneous"],
            "_recording_delay_days": [2.0],
            "medical_code": ["infection"],
            "coding_system": ["clinical"],
            "severity": ["moderate"],
            "serious": [True],
            "resolution_time": [45.0],
        }
    )
    enrollment = pd.DataFrame({"participant_id": [1], "enrollment_day": [0]})
    dense = trialdev_safety_records(records, enrollment, readout_day=35, follow_up_days=56, surveillance_days=(28, 56))
    sparse = trialdev_safety_records(records, enrollment, readout_day=35, follow_up_days=56, surveillance_days=(56,))
    assert dense.availability_time.item() == 30
    assert sparse.empty
    later = trialdev_safety_records(records, enrollment, readout_day=50, follow_up_days=56, surveillance_days=(56,))
    assert later.availability_time.item() == 42
    assert later.occurrence_time.item() == dense.occurrence_time.item() == 20
    assert np.isnan(later.resolution_time.item())
    resolved = trialdev_safety_records(records, enrollment, readout_day=58, follow_up_days=56, surveillance_days=(56,))
    assert resolved.resolution_time.item() == 45
    assert not any(column.startswith("_") for column in resolved)
    urgent = records.assign(ascertainment="urgent", availability_time=42.0)
    for grid in ((28, 56), (56,)):
        assert trialdev_safety_records(
            urgent, enrollment, readout_day=35, follow_up_days=56, surveillance_days=grid
        ).empty
        actual = trialdev_safety_records(urgent, enrollment, readout_day=50, follow_up_days=56, surveillance_days=grid)
        assert actual.availability_time.item() == 42.0
        assert actual.occurrence_time.item() == 20.0
    scheduled = records.assign(ascertainment="scheduled", _spontaneous_observation_time=np.nan)
    assert trialdev_safety_records(
        scheduled, enrollment, readout_day=57, follow_up_days=56, surveillance_days=(56,)
    ).empty
    with pytest.raises(ValueError, match="potentials"):
        trialdev_safety_records(
            records.drop(columns="_recording_delay_days"),
            enrollment,
            readout_day=35,
            follow_up_days=56,
            surveillance_days=(56,),
        )
    contacts = pd.DataFrame(
        {"participant_id": [1, 1, 1], "occurrence_time": [0, 28, 56], "availability_time": [0, 28, 56]}
    )
    from trialagentbench.trialdev.observations import trialdev_contact_records

    selected = trialdev_contact_records(
        contacts, enrollment, readout_day=56, follow_up_days=56, surveillance_days=(56,)
    )
    assert selected.occurrence_time.tolist() == [56]


@pytest.mark.parametrize("duration", [float("nan"), float("inf"), True, 0, -1])
def test_invalid_followup_cannot_silently_produce_an_empty_readout(duration):
    from trialagentbench.trialdev.observations import trialdev_contact_records

    records = pd.DataFrame({"participant_id": [1], "occurrence_time": [0], "availability_time": [0]})
    enrollment = pd.DataFrame({"participant_id": [1], "enrollment_day": [0]})
    with pytest.raises(ValueError, match="duration"):
        trialdev_contact_records(records, enrollment, readout_day=28, follow_up_days=duration)


@pytest.mark.parametrize("value,observed,attended", [(np.inf, True, True), (5.0, True, False), (5.0, "false", True)])
def test_invalid_measurement_cannot_be_released_as_observed(value, observed, attended):
    records = pd.DataFrame(
        {
            "participant_id": [1],
            "outcome_id": ["score"],
            "occurrence_time": [0],
            "availability_time": [0],
            "observed": [observed],
            "attended": [attended],
            "observed_outcome": [value],
        }
    )
    enrollment = pd.DataFrame({"participant_id": [1], "enrollment_day": [0]})
    with pytest.raises(ValueError, match="measurement"):
        trialdev_measurement_records(records, enrollment, readout_day=28, follow_up_days=28, assessment_days=(0, 28))


def test_exposure_history_excludes_future_receipt_and_private_adherence():
    from trialagentbench.trialdev.observations import trialdev_exposure_records

    records = pd.DataFrame(
        {
            "participant_id": [1, 1, 1],
            "occurrence_time": [0, 28, 56],
            "received_arm": ["active", "control", "active"],
            "initially_adherent": True,
        }
    )
    enrollment = pd.DataFrame({"participant_id": [1], "enrollment_day": [10]})
    result = trialdev_exposure_records(records, enrollment, readout_day=40, follow_up_days=56)
    assert result.occurrence_time.tolist() == [10, 38]
    assert result.received_arm.tolist() == ["active", "control"]
    assert result.columns.tolist() == ["participant_id", "occurrence_time", "received_arm"]
    with pytest.raises(ValueError, match="supported"):
        trialdev_exposure_records(
            records.assign(received_arm="unknown"), enrollment, readout_day=40, follow_up_days=56
        )


def test_information_look_counts_available_primary_measurements_not_enrollment_or_latent_values():
    from trialagentbench.trialdev.observations import trialdev_measurement_information_day

    records = pd.DataFrame(
        {
            "participant_id": [1, 2, 3, 4, 1],
            "outcome_id": ["primary"] * 4 + ["secondary"],
            "occurrence_time": 28,
            "availability_time": [28, 35, 28, 50, 28],
            "attended": True,
            "observed": [True, True, False, True, True],
            "observed_outcome": [1, 2, 999, 4, 5],
        }
    )
    enrollment = pd.DataFrame({"participant_id": [1, 2, 3, 4], "enrollment_day": [10, 20, 10, 10]})
    args = dict(enrollment=enrollment, outcome_id="primary", participant_count=2, follow_up_days=28, deadline_day=100)
    assert trialdev_measurement_information_day(records, **args) == 55
    assert trialdev_measurement_information_day(records, **{**args, "deadline_day": 54}) is None
    assert trialdev_measurement_information_day(records, **{**args, "participant_count": 4}) is None
    assert trialdev_measurement_information_day(records, **{**args, "participant_count": 3}) == 60
    changed = records.assign(observed_outcome=-123456)
    assert trialdev_measurement_information_day(changed, **args) == 55
    with pytest.raises(ValueError, match="one primary"):
        trialdev_measurement_information_day(pd.concat([records, records.iloc[:1]]), **args)
    with pytest.raises(ValueError, match="positive integer"):
        trialdev_measurement_information_day(records, **{**args, "participant_count": True})


def test_recurrent_information_counts_complete_windows_including_zero_events_but_not_loss():
    from trialagentbench.trialdev.observations import trialdev_recurrent_information_day

    records = pd.DataFrame(
        {
            "participant_id": [1, 2, 3, 4, 3],
            "outcome_id": "primary",
            "event_time": [100, 15, 100, 100, 5],
            "event": [0, 0, 0, 0, 1],
        }
    )
    enrollment = pd.DataFrame({"participant_id": [1, 2, 3, 4], "enrollment_day": [10, 10, 20, 30]})
    args = dict(enrollment=enrollment, outcome_id="primary", participant_count=2, follow_up_days=28, deadline_day=100)
    assert trialdev_recurrent_information_day(records, **args) == 48
    assert trialdev_recurrent_information_day(records, **{**args, "deadline_day": 47}) is None
    assert trialdev_recurrent_information_day(records, **{**args, "participant_count": 4}) is None
    assert trialdev_recurrent_information_day(records, **{**args, "participant_count": 3}) == 58
    # Removing the only event does not erase the information in completed zero-event windows.
    assert trialdev_recurrent_information_day(records.loc[records.event.eq(0)], **args) == 48
    with pytest.raises(ValueError, match="one observed follow-up end"):
        trialdev_recurrent_information_day(pd.concat([records, records.iloc[:1]]), **args)


def test_dropout_replays_only_committed_attempts_without_revealing_potentials():
    """An omitted visit cannot cause dropout; hidden earlier failures still affect follow-up."""
    records = pd.DataFrame(
        {
            "participant_id": [1, 1, 1, 2, 2, 2],
            "outcome_id": "score",
            "occurrence_time": [0, 28, 56, 0, 28, 56],
            "availability_time": [0, 100, 56, 0, 28, 56],
            "attended": True,
            "observed": [True, False, False, True, True, False],
            "observed_outcome": [5.0, np.nan, np.nan, 7.0, 6.0, np.nan],
            "_measurement_attempt_succeeded": [True, False, True, True, True, False],
        }
    )
    keys = ["participant_id", "outcome_id", "occurrence_time"]
    potentials = records[keys].assign(complete_recorded_outcome=[5.0, 4.0, 3.0, 7.0, 6.0, 5.0])
    enrollment = pd.DataFrame({"participant_id": [1, 2], "enrollment_day": 0})

    def read(days):
        return trialdev_measurement_records(
            records.sample(frac=1, random_state=17),
            enrollment,
            readout_day=60,
            follow_up_days=56,
            assessment_days=days,
            monotone_dropout=True,
            complete_measurements=potentials,
        )

    dense = read((0, 28, 56))
    sparse = read((0, 56))
    # The unavailable day-28 failure still causes dropout when that visit was committed.
    assert not dense.loc[dense.occurrence_time.eq(56), "observed"].any()
    # Removing that visit preserves patient 1's successful terminal attempt.
    terminal = sparse.loc[sparse.occurrence_time.eq(56)].set_index("participant_id")
    assert terminal.loc[1, "observed_outcome"] == 3.0
    assert not terminal.loc[2, "observed"]
    assert np.isnan(terminal.loc[2, "observed_outcome"])
    assert not any(column.startswith("_") or "potential" in column for column in sparse)
    assert records.observed_outcome.isna().sum() == 3
    with pytest.raises(ValueError, match="cover every"):
        trialdev_measurement_records(
            records,
            enrollment,
            readout_day=60,
            follow_up_days=56,
            assessment_days=(0, 56),
            monotone_dropout=True,
            complete_measurements=potentials.iloc[:-1],
        )


def test_intercurrent_records_use_enrollment_and_report_availability_without_private_flags():
    from trialagentbench_verifier.authored_readouts import _Patient, _Readout, _record_projection

    from trialagentbench.trialdev.observations import trialdev_intercurrent_records

    records = pd.DataFrame(
        {
            "participant_id": [1, 1, 1, 2],
            "event_id": "rescue",
            "event_kind": "rescue_medication",
            "occurrence_time": [10.0, 30.0, 8.0, 10.0],
            "availability_time": [12.0, 35.0, 8.0, 12.0],
            "event_at_contact": [True, True, False, True],
            "first_event_time": 10.0,
            "private_outcome_shift": -3.0,
        }
    )
    enrollment = pd.DataFrame({"participant_id": [1], "enrollment_day": [5.0]})
    patients = (_Patient(participant_id=1, participant_label="p1", arm_id="active", enrollment_day=5),)
    for day, count in ((16, 0), (17, 1)):
        actual = trialdev_intercurrent_records(records, enrollment, readout_day=day, follow_up_days=28)
        assert len(actual) == count
        assert set(actual) == {"participant_id", "event_id", "event_kind", "occurrence_time", "availability_time"}
        readout = _Readout(
            study_id="study",
            outcome_object="Recorded clinical outcome",
            readout_day=day,
            follow_up_days=28,
            arms=(),
            assessment_days=(28,),
            surveillance_days=(28,),
        )
        expected = _record_projection(records, patients, readout, "intercurrent_events")
        actual["participant_id"] = actual.participant_id.map({1: "p1"})
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
        if count:
            assert actual.occurrence_time.tolist() == [15.0]
            assert actual.availability_time.tolist() == [17.0]


def test_event_and_treatment_receipt_are_inclusive_at_fractional_followup_boundary():
    """A terminal event and treatment switch appear together, without future person-time."""
    from trialagentbench.trialdev.observations import trialdev_exposure_records, trialdev_recurrent_records

    enrollment = pd.DataFrame({"participant_id": [1], "enrollment_day": [10.25]})
    events = pd.DataFrame(
        {"participant_id": [1, 1], "outcome_id": "exacerbation", "event_time": [28.0, 56.0], "event": [1, 0]}
    )
    receipts = pd.DataFrame(
        {"participant_id": [1, 1], "occurrence_time": [0.0, 28.0], "received_arm": ["active", "control"]}
    )
    cutoff = 38.25
    for day, expected_events, expected_treatments in (
        (np.nextafter(cutoff, -np.inf), 0, ["active"]),
        (cutoff, 1, ["active", "control"]),
        (np.nextafter(cutoff, np.inf), 1, ["active", "control"]),
    ):
        recurrent = trialdev_recurrent_records(events, enrollment, readout_day=day, follow_up_days=28)
        exposure = trialdev_exposure_records(receipts, enrollment, readout_day=day, follow_up_days=28)
        assert recurrent.event.sum() == expected_events
        assert recurrent.loc[recurrent.event.eq(0), "event_time"].item() == min(day, cutoff) - 10.25
        assert exposure.received_arm.tolist() == expected_treatments
        assert (recurrent.event_time + 10.25).le(day).all()
        assert exposure.occurrence_time.le(day).all()


@pytest.mark.parametrize("recurrent", (False, True))
def test_information_deadline_admits_exactly_available_complete_participants(recurrent):
    """An information look counts the final complete record at, and never before, availability."""
    from trialagentbench.trialdev.observations import (
        trialdev_measurement_information_day,
        trialdev_recurrent_information_day,
    )

    enrollment = pd.DataFrame({"participant_id": [1, 2, 3], "enrollment_day": [10.25, 12.25, 0.0]})
    records = pd.DataFrame({"participant_id": [1, 2, 3], "outcome_id": "primary"})
    if recurrent:
        records = records.assign(event_time=[56.0, 56.0, 27.0], event=0)
        resolver = trialdev_recurrent_information_day
    else:
        records = records.assign(
            occurrence_time=28.0,
            availability_time=28.0,
            attended=True,
            observed=[True, True, False],
            observed_outcome=[5.0, 3.0, np.nan],
        )
        resolver = trialdev_measurement_information_day
    cutoff = 40.25
    for day, expected in (
        (np.nextafter(cutoff, -np.inf), None),
        (cutoff, cutoff),
        (np.nextafter(cutoff, np.inf), cutoff),
    ):
        assert (
            resolver(
                records, enrollment, outcome_id="primary", participant_count=2, follow_up_days=28, deadline_day=day
            )
            == expected
        )
    assert (
        resolver(records, enrollment, outcome_id="primary", participant_count=3, follow_up_days=28, deadline_day=100)
        is None
    )
