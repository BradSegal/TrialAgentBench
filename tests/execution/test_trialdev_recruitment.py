"""Finite screening, overlapping clinical preferences and randomized enrollment."""

import hashlib
import json

import pandas as pd
import pytest

from trialagentbench.trialdev.recruitment import (
    TrialDevEnrollmentPlan,
    TrialDevRecruitmentSelection,
    allocate_trialdev_patients,
)


def _plan(study_id="study-1", count=10, screen_failure_fraction=0.0, **recruitment):
    return TrialDevEnrollmentPlan.model_validate(
        {
            "study_id": study_id,
            "site_ids": ("site_001", "site_002"),
            "arms": [{"arm_id": arm, "participant_count": count} for arm in ("control", "active")],
            "recruitment": recruitment,
            "screen_failure_fraction": screen_failure_fraction,
        }
    )


def _frame():
    return pd.DataFrame(
        {
            "participant_id": [f"patient-{i:03}" for i in range(200)],
            "recruitment_prior": [i % 3 == 0 for i in range(200)],
            "recruitment_high": [i % 4 == 0 for i in range(200)],
            "hidden_potential": range(200),
            "site_id": ["site_001", "site_002"] * 100,
            "screening_day": range(1, 201),
        }
    )


def _digest(seed, namespace, identity):
    return hashlib.sha256(json.dumps([seed, namespace, identity], separators=(",", ":")).encode()).digest()


def _arrivals(frame, seed):
    """Independently assign each site's frozen screening days to its candidates in seeded order."""
    days = {}
    for _, rows in frame.groupby("site_id"):
        candidates = sorted(rows.participant_id, key=lambda identity: _digest(seed, "arrival", identity))
        days.update(zip(candidates, sorted(rows.screening_day), strict=True))
    return sorted(days, key=lambda identity: (days[identity], identity)), days


def test_recruitment_is_replayable_order_independent_and_outcome_blind():
    frame = _frame()
    before = frame.copy(deep=True)
    plan = _plan()
    actual = allocate_trialdev_patients(frame, plan, seed=17)
    pd.testing.assert_frame_equal(frame, before)
    assert actual == allocate_trialdev_patients(frame.iloc[::-1].assign(hidden_potential=-999), plan, seed=17)
    renamed = allocate_trialdev_patients(frame, _plan("other-name"), seed=17)
    assert actual.patients == renamed.patients and actual.screening_days == renamed.screening_days
    assert len({p.participant_id for p in actual.patients}) == 20
    assert sum(p.arm_id == "control" for p in actual.patients) == 10
    assert len(actual.screening_days) == 20
    # Another run recruits another sample of the population at the same frozen arrival times.
    other_run = allocate_trialdev_patients(frame, plan, seed=19)
    assert other_run.screening_days == actual.screening_days
    assert {p.participant_id for p in actual.patients} != {p.participant_id for p in other_run.patients}
    assert [p.participant_id for p in actual.patients] == _arrivals(frame, 17)[0][:20]


def test_overlap_uses_largest_selected_fraction_and_charges_rejected_candidates():
    frame = _frame()
    preferences = [
        {"group_id": "prior", "screening_fraction": 0.5},
        {"group_id": "high", "screening_fraction": 1},
    ]
    plan = _plan(background_fraction=0.1, preferences=preferences)
    result = allocate_trialdev_patients(frame, plan, seed=42)
    # Reconstruct membership, overlap precedence and screening independently.
    accepted = []
    queue, days = _arrivals(frame, 42)
    for identity in queue:
        i = int(identity.split("-")[1])
        fraction = 1 if i % 4 == 0 else 0.5 if i % 3 == 0 else 0.1
        if int.from_bytes(_digest(42, "screening", identity), "big") < int(fraction * 2**256):
            accepted.append(identity)
    assert [p.participant_id for p in result.patients] == accepted[:20]
    final = days[accepted[19]]
    assert result.screening_days == tuple(float(i) for i in range(1, final + 1))
    assert final > 20
    reversed_plan = _plan(background_fraction=0.1, preferences=list(reversed(preferences)))
    replay = allocate_trialdev_patients(frame, reversed_plan, seed=42)
    assert replay.patients == result.patients and replay.screening_days == result.screening_days
    # The background is used only outside selected groups, even if it is larger.
    selective = _plan(background_fraction=1, preferences=[{"group_id": "prior", "screening_fraction": 0.1}])
    assert (
        allocate_trialdev_patients(frame, selective, seed=42).patients
        != allocate_trialdev_patients(frame, _plan(), seed=42).patients
    )


def test_screen_failures_consume_screening_and_arrival_time_but_never_enroll():
    frame = _frame()
    result = allocate_trialdev_patients(frame, _plan(screen_failure_fraction=0.4), seed=42)
    queue, days = _arrivals(frame, 42)
    passing = [
        identity
        for identity in queue
        if int.from_bytes(_digest(42, "screen-failure", [0, identity]), "big") < int(0.6 * 2**256)
    ]
    assert [p.participant_id for p in result.patients] == passing[:20]
    assert len(result.screening_days) == days[passing[19]] > 20
    assert result.patients != allocate_trialdev_patients(frame, _plan(), seed=42).patients
    # A later study screens afresh: earlier failures are not deterministically failed again.
    later = allocate_trialdev_patients(frame, _plan(screen_failure_fraction=0.4), seed=42, screening_round=1)
    assert later.patients != result.patients


def test_studies_consume_available_patients_without_replacement():
    frame = _frame().assign(participant_id=range(200))
    first = allocate_trialdev_patients(frame, _plan(), seed=17)
    unavailable = [p.participant_id for p in first.patients]
    second = allocate_trialdev_patients(frame, _plan("second"), seed=17, unavailable_participant_ids=unavailable)
    assert all(isinstance(p.participant_id, int) for p in first.patients)
    assert set(unavailable).isdisjoint(p.participant_id for p in second.patients)
    assert second.patients[0].participant_id == _arrivals(frame, 17)[0][20]


def test_empty_and_rare_groups_do_not_force_patients_into_every_arm():
    frame = _frame().assign(recruitment_empty=False, recruitment_rare=False)
    rare = _arrivals(frame, 42)[0][0]
    frame.loc[frame.participant_id.eq(rare), "recruitment_rare"] = True
    plan = _plan(
        preferences=[{"group_id": "empty", "screening_fraction": 0.1}, {"group_id": "rare", "screening_fraction": 1}]
    )
    actual = allocate_trialdev_patients(frame, plan, seed=42)
    assert len(actual.patients) == 20
    assert sum(p.participant_id == rare for p in actual.patients) == 1


def test_exhaustion_and_invalid_frames_do_not_mutate_inputs():
    frame = _frame()
    before = frame.copy(deep=True)
    with pytest.raises(ValueError, match="unenrolled candidates remain"):
        allocate_trialdev_patients(frame, _plan(count=101), seed=17)
    pd.testing.assert_frame_equal(frame, before)
    with pytest.raises(ValueError, match="outside this pool"):
        allocate_trialdev_patients(frame, _plan(), seed=17, unavailable_participant_ids=["unknown"])
    with pytest.raises(ValueError, match="unique"):
        allocate_trialdev_patients(pd.concat([frame, frame.iloc[:1]]), _plan(), seed=17)
    plan = _plan(preferences=[{"group_id": "prior", "screening_fraction": 0.5}])
    with pytest.raises(ValueError, match="missing"):
        allocate_trialdev_patients(frame.drop(columns="recruitment_prior"), plan, seed=17)
    with pytest.raises(ValueError, match="complete"):
        allocate_trialdev_patients(frame.assign(recruitment_prior=None), plan, seed=17)
    with pytest.raises(ValueError, match="boolean"):
        allocate_trialdev_patients(frame.assign(recruitment_prior=1), plan, seed=17)


def test_activating_sites_changes_patients_and_clock_without_redrawing_candidates():
    frame = pd.DataFrame(
        [
            dict(participant_id=f"{site}:{i}", site_id=site, screening_day=opening + (i + 1) * interval)
            for site, opening, interval in (
                ("site_001", 30, 4),
                ("site_002", 20, 5),
                ("site_003", 0, 2),
                ("site_004", 0, 3),
            )
            for i in range(20)
        ]
    )
    regional = allocate_trialdev_patients(frame, _plan(), seed=17)
    broader = allocate_trialdev_patients(
        frame, _plan().model_copy(update={"site_ids": tuple(sorted(frame.site_id.unique()))}), seed=17
    )
    assert {p.participant_id for p in regional.patients} != {p.participant_id for p in broader.patients}
    assert broader.screening_days[-1] < regional.screening_days[-1]
    # Each patient presents at one of its own site's frozen arrival times.
    _, days = _arrivals(frame, 17)
    for reservation in (regional, broader):
        assert all(
            reservation.screening_days[p.screening_ordinal - 1] == days[p.participant_id] for p in reservation.patients
        )


@pytest.mark.parametrize("fraction", [0, 0.09, 1.1, float("nan"), float("inf")])
def test_invalid_screening_fractions_fail(fraction):
    with pytest.raises(ValueError):
        TrialDevRecruitmentSelection(background_fraction=fraction)


def test_duplicate_preferences_and_invalid_arm_caps_fail():
    with pytest.raises(ValueError, match="at most once"):
        _plan(preferences=[{"group_id": "prior", "screening_fraction": 0.5}] * 2)
    raw = _plan().model_dump()
    raw["arms"] = [raw["arms"][0]] * 2
    with pytest.raises(ValueError, match="unique"):
        TrialDevEnrollmentPlan.model_validate(raw)


def test_allocator_runs_without_native_generation(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    script = tmp_path / "standalone.py"
    script.write_text("""
import importlib.abc
import sys
sys.path[:0] = sys.argv[1:]
class NoGeneration(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("ariadne", "trialagentbench_construction")):
            raise ImportError("generation forbidden: " + fullname)
sys.meta_path.insert(0, NoGeneration())
import pandas as pd
from trialagentbench.trialdev.recruitment import TrialDevEnrollmentPlan, allocate_trialdev_patients
frame = pd.DataFrame({"participant_id":range(20), "site_id":"site_001", "screening_day":range(1,21)})
plan = TrialDevEnrollmentPlan.model_validate({"study_id":"new", "site_ids":["site_001","site_002"], "arms":[{"arm_id":a,"participant_count":7} for a in ("control","active")], "screen_failure_fraction":0.0})
assert len(allocate_trialdev_patients(frame, plan, seed=71).patients) == 14
""")
    subprocess.run(
        [
            sys.executable,
            "-I",
            str(script),
            str(Path(__file__).resolve().parents[2]),
            str(Path(__file__).resolve().parents[4] / "clinical-data-tables/src"),
        ],
        check=True,
        timeout=30,
    )
