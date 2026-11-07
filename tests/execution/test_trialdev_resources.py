"""Resource consequences of actual accrual and committed schedules."""

import pytest

from trialagentbench.trialdev.resources import TrialDevStudyTariff, trialdev_study_cost


def _candidates(frame):
    return frame.assign(
        site_id=[f"site_{i % 2 + 1:03d}" for i in range(len(frame))],
        screening_day=[(i + 1) * 30 / 16 for i in range(len(frame))],
    )


def test_stopping_enrollment_does_not_erase_followup_or_double_charge_visits():
    tariff = TrialDevStudyTariff(
        setup=100,
        per_site=10,
        screening=2,
        enrollment=5,
        followup_day=3,
        assessment=7,
        startup_days=0,
        closeout_days=0,
    )
    args = dict(
        tariff=tariff,
        sites=2,
        screened=3,
        enrollment_days=(0, 10),
        follow_up_days=20,
        assessment_days=(0, 20),
        surveillance_days=(0, 10, 20),
    )
    early = trialdev_study_cost(**args, programme_day=10)
    # 136 setup/screen/enrollment + 30 follow-up + 3 coincident-visit-union charges.
    assert early.accrued_usd == 187
    assert early.remaining_liability_usd == 111
    late = trialdev_study_cost(**args, programme_day=30)
    assert late.accrued_usd == 298
    assert late.remaining_liability_usd == 0
    assert late.committed_usd == early.committed_usd
    sparse = trialdev_study_cost(**{**args, "surveillance_days": (0, 20)}, programme_day=30)
    assert late.committed_usd - sparse.committed_usd == 14


def test_reserved_future_patients_cannot_be_charged_as_accrued():
    tariff = TrialDevStudyTariff(
        setup=0, per_site=0, screening=1, enrollment=1, followup_day=1, assessment=1, startup_days=0, closeout_days=0
    )
    with pytest.raises(ValueError, match="already enrolled"):
        trialdev_study_cost(
            tariff,
            sites=2,
            screened=1,
            enrollment_days=(20,),
            programme_day=10,
            follow_up_days=28,
            assessment_days=(0, 28),
            surveillance_days=(0, 28),
        )


def test_budget_receipts_reserve_liabilities_and_replay_without_double_charging():
    from trialagentbench.trialdev.resources import (
        TrialDevBudgetLedger,
        TrialDevCostReceipt,
        TrialDevStudyCost,
        reconcile_trialdev_cost,
    )

    initial = TrialDevBudgetLedger(budget_usd=400)
    first = TrialDevCostReceipt(
        study_id="study-1", readout_id="interim", cost=TrialDevStudyCost(accrued_usd=187, remaining_liability_usd=111)
    )
    interim = reconcile_trialdev_cost(initial, first)
    assert interim.spent_usd == 187 and interim.available_usd == 102
    final = reconcile_trialdev_cost(
        interim,
        TrialDevCostReceipt(
            study_id="study-1", readout_id="final", cost=TrialDevStudyCost(accrued_usd=298, remaining_liability_usd=0)
        ),
    )
    assert final.spent_usd == 298 and final.available_usd == 102
    assert reconcile_trialdev_cost(final, first) is final
    restored = TrialDevBudgetLedger.model_validate_json(final.model_dump_json())
    assert reconcile_trialdev_cost(restored, first) == final
    with pytest.raises(ValueError, match="budget"):
        reconcile_trialdev_cost(
            final,
            TrialDevCostReceipt(
                study_id="study-2",
                readout_id="second",
                cost=TrialDevStudyCost(accrued_usd=1, remaining_liability_usd=102),
            ),
        )
    assert final.available_usd == 102
    with pytest.raises(ValueError, match="already bound"):
        reconcile_trialdev_cost(final, first.model_copy(update={"study_id": "other-study"}))
    with pytest.raises(ValueError, match="erase"):
        reconcile_trialdev_cost(
            interim,
            TrialDevCostReceipt(
                study_id="study-1",
                readout_id="stop",
                cost=TrialDevStudyCost(accrued_usd=187, remaining_liability_usd=0),
            ),
        )

    # A live run's history: the interim parts sum to 10166000.000000002, the final readout
    # to exactly 10166000.0. The same obligation must not read as erased.
    rounded = reconcile_trialdev_cost(
        TrialDevBudgetLedger(budget_usd=150_000_000),
        TrialDevCostReceipt(
            study_id="study-1",
            readout_id="early",
            cost=TrialDevStudyCost(accrued_usd=10095262.20988816, remaining_liability_usd=70737.79011184082),
        ),
    )
    completed = reconcile_trialdev_cost(
        rounded,
        TrialDevCostReceipt(
            study_id="study-1",
            readout_id="complete",
            cost=TrialDevStudyCost(accrued_usd=10166000.0, remaining_liability_usd=0.0),
        ),
    )
    assert completed.spent_usd == 10166000.0
    with pytest.raises(ValueError, match="erase"):
        reconcile_trialdev_cost(
            rounded,
            TrialDevCostReceipt(
                study_id="study-1",
                readout_id="cent-short",
                cost=TrialDevStudyCost(accrued_usd=10165999.99, remaining_liability_usd=0.0),
            ),
        )

    invalid_history = final.model_dump(mode="json")
    invalid_history["budget_usd"] = 200
    with pytest.raises(ValueError, match="budget"):
        TrialDevBudgetLedger.model_validate(invalid_history)


def test_accrued_enrollment_is_immutable_and_cannot_be_reused_between_studies():
    from trialagentbench.trialdev.recruitment import TrialDevCommittedPatient, TrialDevStudyEnrollment
    from trialagentbench.trialdev.resources import TrialDevBudgetLedger, reconcile_trialdev_enrollment

    patient = TrialDevCommittedPatient(participant_id=1, participant_label="p1", arm_id="control", enrollment_day=10)
    study = TrialDevStudyEnrollment(study_id="first", design_sha256="a" * 64, patients=(patient,))
    state = reconcile_trialdev_enrollment(
        TrialDevBudgetLedger(budget_usd=1000), pool_manifest_sha256="b" * 64, enrollment=study
    )
    for changed in (
        study.model_copy(update={"patients": ()}),
        study.model_copy(update={"design_sha256": "c" * 64}),
        study.model_copy(update={"patients": (patient.model_copy(update={"arm_id": "active"}),)}),
    ):
        with pytest.raises(ValueError, match="erase/reassign"):
            reconcile_trialdev_enrollment(state, pool_manifest_sha256="b" * 64, enrollment=changed)
    with pytest.raises(ValueError, match="reuse"):
        reconcile_trialdev_enrollment(
            state, pool_manifest_sha256="b" * 64, enrollment=study.model_copy(update={"study_id": "second"})
        )
    with pytest.raises(ValueError, match="population"):
        reconcile_trialdev_enrollment(state, pool_manifest_sha256="c" * 64, enrollment=study)
    added = patient.model_copy(update={"participant_id": 2, "participant_label": "p2", "enrollment_day": 20})
    advanced = reconcile_trialdev_enrollment(
        state, pool_manifest_sha256="b" * 64, enrollment=study.model_copy(update={"patients": (patient, added)})
    )
    assert len(advanced.enrollments[0].patients) == 2
    assert len(state.enrollments[0].patients) == 1


def test_reservations_are_replayable_disjoint_and_not_enrollment_or_spending():
    import pandas as pd

    from trialagentbench.trialdev.recruitment import TrialDevEnrollmentPlan
    from trialagentbench.trialdev.resources import TrialDevBudgetLedger, reserve_trialdev_patients

    frame = _candidates(pd.DataFrame({"participant_id": range(40)}))
    plan = TrialDevEnrollmentPlan.model_validate(
        {
            "study_id": "first",
            "site_ids": ("site_001", "site_002"),
            "screen_failure_fraction": 0.0,
            "arms": [{"arm_id": arm, "participant_count": 10} for arm in ("control", "active")],
        }
    )
    first = reserve_trialdev_patients(
        TrialDevBudgetLedger(budget_usd=1000), frame, plan, pool_manifest_sha256="a" * 64, seed=42
    )
    second = reserve_trialdev_patients(
        first, frame, plan.model_copy(update={"study_id": "second"}), pool_manifest_sha256="a" * 64, seed=42
    )
    assert second.spent_usd == 0 and not second.enrollments
    ids = [{patient.participant_id for patient in reservation.patients} for reservation in second.reservations]
    assert ids[0].isdisjoint(ids[1]) and len(ids[0] | ids[1]) == 40
    assert reserve_trialdev_patients(second, frame, plan, pool_manifest_sha256="a" * 64, seed=42) is second
    restored = TrialDevBudgetLedger.model_validate_json(second.model_dump_json())
    assert restored == second
    with pytest.raises(ValueError, match="unenrolled candidates remain"):
        reserve_trialdev_patients(
            second, frame, plan.model_copy(update={"study_id": "third"}), pool_manifest_sha256="a" * 64, seed=42
        )
    with pytest.raises(ValueError, match="seed"):
        reserve_trialdev_patients(second, frame, plan, pool_manifest_sha256="a" * 64, seed=43)


def test_closure_releases_only_unaccrued_patients_and_cannot_restart_enrollment():
    import pandas as pd

    from trialagentbench.trialdev.recruitment import (
        TrialDevCommittedPatient,
        TrialDevEnrollmentPlan,
        TrialDevStudyEnrollment,
    )
    from trialagentbench.trialdev.resources import (
        TrialDevBudgetLedger,
        close_trialdev_reservation,
        reconcile_trialdev_enrollment,
        reserve_trialdev_patients,
    )

    frame = _candidates(pd.DataFrame({"participant_id": range(20)}))
    plan = TrialDevEnrollmentPlan.model_validate(
        {
            "study_id": "first",
            "site_ids": ("site_001", "site_002"),
            "screen_failure_fraction": 0.0,
            "arms": [{"arm_id": arm, "participant_count": 5} for arm in ("control", "active")],
        }
    )
    reserved = reserve_trialdev_patients(
        TrialDevBudgetLedger(budget_usd=1000), frame, plan, pool_manifest_sha256="a" * 64, seed=7
    )
    selected = reserved.reservations[0].patients[0]
    patient = TrialDevCommittedPatient(
        participant_id=selected.participant_id, arm_id=selected.arm_id, participant_label="p1", enrollment_day=10
    )
    accrued = TrialDevStudyEnrollment(study_id="first", design_sha256="b" * 64, patients=(patient,))
    state = reconcile_trialdev_enrollment(reserved, pool_manifest_sha256="a" * 64, enrollment=accrued)
    closed = close_trialdev_reservation(state, "first")
    assert closed.enrollments == state.enrollments and closed.receipts == state.receipts
    assert close_trialdev_reservation(closed, "first") is closed
    assert reserve_trialdev_patients(closed, frame, plan, pool_manifest_sha256="a" * 64, seed=7) is closed
    next_state = reserve_trialdev_patients(
        closed, frame, plan.model_copy(update={"study_id": "second"}), pool_manifest_sha256="a" * 64, seed=7
    )
    next_ids = {row.participant_id for row in next_state.reservations[1].patients}
    original_ids = {row.participant_id for row in state.reservations[0].patients}
    assert patient.participant_id not in next_ids
    assert next_ids & original_ids
    unused = reserved.reservations[0].patients[1]
    extra = TrialDevCommittedPatient(
        participant_id=unused.participant_id, arm_id=unused.arm_id, participant_label="p2", enrollment_day=20
    )
    with pytest.raises(ValueError, match="closed"):
        reconcile_trialdev_enrollment(
            closed, pool_manifest_sha256="a" * 64, enrollment=accrued.model_copy(update={"patients": (patient, extra)})
        )


def test_accrual_uses_real_screening_supply_and_preserves_calendar_identity():
    import math

    import pandas as pd

    from trialagentbench.trialdev.recruitment import TrialDevEnrollmentPlan
    from trialagentbench.trialdev.resources import (
        TrialDevBudgetLedger,
        reserve_trialdev_patients,
        trialdev_accrued_recruitment,
    )

    baseline = _candidates(pd.DataFrame({"participant_id": range(100)}))
    plan = TrialDevEnrollmentPlan.model_validate(
        {
            "study_id": "supply",
            "recruitment": {"background_fraction": 0.25},
            "site_ids": ("site_001", "site_002"),
            "screen_failure_fraction": 0.0,
            "arms": [{"arm_id": arm, "participant_count": 4} for arm in ("control", "active")],
        }
    )
    ledger = reserve_trialdev_patients(
        TrialDevBudgetLedger(budget_usd=1000), baseline, plan, pool_manifest_sha256="a" * 64, seed=42
    )
    reservation = ledger.reservations[0]
    tariff = TrialDevStudyTariff(
        setup=0, per_site=0, screening=1, enrollment=1, followup_day=1, assessment=1, startup_days=60, closeout_days=0
    )
    args = dict(reservation=reservation, tariff=tariff, start_day=10)
    before_open = trialdev_accrued_recruitment(**args, programme_day=69)
    assert before_open.screened == 0 and not before_open.patients
    full = trialdev_accrued_recruitment(**args, programme_day=1000)
    assert len(full.patients) == 8 and full.screened > 8
    assert full.screened == max(p.screening_ordinal for p in reservation.patients)
    for patient in full.patients:
        cutoff = trialdev_accrued_recruitment(**args, programme_day=patient.enrollment_day)
        previous = trialdev_accrued_recruitment(
            **args, programme_day=math.nextafter(patient.enrollment_day, -math.inf)
        )
        assert patient in cutoff.patients and patient not in previous.patients
    assert [row.enrollment_day for row in full.patients] == [
        70 + reservation.screening_days[row.screening_ordinal - 1]
        for row in sorted(reservation.patients, key=lambda row: row.screening_ordinal)
    ]
    restored = TrialDevBudgetLedger.model_validate_json(ledger.model_dump_json())
    assert (
        trialdev_accrued_recruitment(**{**args, "reservation": restored.reservations[0]}, programme_day=1000) == full
    )
    assert ledger.spent_usd == 0 and not ledger.enrollments


def test_interim_pause_preserves_followup_and_resumption_does_not_redraw(tmp_path):
    from trialagentbench.trialdev.recruitment import TrialDevEnrollmentPlan, TrialDevStudyReservation
    from trialagentbench.trialdev.resources import TrialDevRecruitmentPause, trialdev_accrued_recruitment

    plan = TrialDevEnrollmentPlan.model_validate(
        {
            "study_id": "interim",
            "site_ids": ("site_001", "site_002"),
            "screen_failure_fraction": 0.0,
            "arms": [{"arm_id": arm, "participant_count": 2} for arm in ("control", "active")],
        }
    )
    reservation = TrialDevStudyReservation.model_validate(
        {
            "plan": plan,
            "screening_days": [i * 30 / 16 for i in range(1, 5)],
            "patients": [
                {
                    "participant_id": i,
                    "arm_id": "control" if i % 2 else "active",
                    "screening_ordinal": i,
                }
                for i in range(1, 5)
            ],
        }
    )
    tariff = TrialDevStudyTariff(
        setup=0, per_site=0, screening=1, enrollment=1, followup_day=1, assessment=1, startup_days=60, closeout_days=0
    )
    args = dict(reservation=reservation, tariff=tariff, start_day=0)
    full = trialdev_accrued_recruitment(**args, programme_day=200)
    pause = TrialDevRecruitmentPause(after_enrolled=2)
    halt_day = full.patients[1].enrollment_day
    halted = trialdev_accrued_recruitment(**args, programme_day=halt_day, pause=pause)
    waiting = trialdev_accrued_recruitment(**args, programme_day=200, pause=pause)
    assert halted == waiting and halted.patients == full.patients[:2]
    assert halted.screened == 2
    costs = dict(
        tariff=tariff,
        sites=2,
        screened=halted.screened,
        enrollment_days=tuple(p.enrollment_day for p in halted.patients),
        follow_up_days=28,
        assessment_days=(0, 28),
        surveillance_days=(0, 28),
    )
    early = trialdev_study_cost(**costs, programme_day=halt_day)
    late = trialdev_study_cost(**costs, programme_day=200)
    assert early.remaining_liability_usd > 0 and late.remaining_liability_usd == 0
    assert early.committed_usd == late.committed_usd
    resumed_pause = pause.model_copy(update={"resume_day": 200})
    assert trialdev_accrued_recruitment(**args, programme_day=200, pause=resumed_pause) == waiting
    resumed = trialdev_accrued_recruitment(**args, programme_day=300, pause=resumed_pause)
    assert resumed.patients[:2] == full.patients[:2]
    assert [p.participant_id for p in resumed.patients] == [p.participant_id for p in full.patients]
    assert [p.arm_id for p in resumed.patients] == [p.arm_id for p in full.patients]
    assert resumed.patients[2].enrollment_day == 201.875
    with pytest.raises(ValueError, match="before"):
        trialdev_accrued_recruitment(**args, programme_day=200, pause=pause.model_copy(update={"resume_day": 60}))
    with pytest.raises(ValueError, match="precede"):
        trialdev_accrued_recruitment(**args, programme_day=200, pause=TrialDevRecruitmentPause(after_enrolled=4))

    from trialagentbench.contracts.trialdev.pools import TrialDevAssignedArm, TrialDevPoolPartition
    from trialagentbench.io import read_json_model, write_json_model
    from trialagentbench.trialdev.materialization import continue_trialdev_study_recruitment
    from trialagentbench.trialdev.recruitment import TrialDevStudyEnrollment
    from trialagentbench.trialdev.resources import (
        TrialDevBudgetLedger,
        TrialDevCostReceipt,
        TrialDevRecruitmentCommitment,
        TrialDevRecruitmentContinuation,
        trialdev_committed_accrual,
    )

    ledger = TrialDevBudgetLedger(
        budget_usd=1000,
        pool_manifest_sha256="b" * 64,
        recruitment_seed=1,
        reservations=(reservation,),
        recruitment_commitments=(
            TrialDevRecruitmentCommitment(
                phase="early_safety",
                looks=({"deadline_day": 200}, {"deadline_day": 300}),
                study_id=plan.study_id,
                tariff=tariff,
                sites=2,
                start_day=0,
                pause_after_enrolled=2,
                arms=(
                    TrialDevAssignedArm(
                        arm_id="control",
                        partition=TrialDevPoolPartition(asset_id="control", dose=None, directory="partitions/p000"),
                    ),
                    TrialDevAssignedArm(
                        arm_id="active",
                        partition=TrialDevPoolPartition(
                            asset_id="asset", dose="standard", directory="partitions/p001"
                        ),
                    ),
                ),
                follow_up_days=28,
                assessment_days=(0, 28),
                surveillance_days=(0, 28),
            ),
        ),
        enrollments=(
            TrialDevStudyEnrollment(
                study_id=plan.study_id,
                design_sha256="a" * 64,
                patients=halted.patients,
                screened=halted.screened,
            ),
        ),
        receipts=(TrialDevCostReceipt(study_id=plan.study_id, readout_id="interim", cost=late, programme_day=200),),
    )
    budget_path = tmp_path / "budget.json"
    write_json_model(budget_path, ledger)
    with pytest.raises(ValueError, match="backdated"):
        continue_trialdev_study_recruitment(
            budget_path, TrialDevRecruitmentContinuation(study_id=plan.study_id, resume_day=199)
        )
    assert read_json_model(TrialDevBudgetLedger, budget_path) == ledger
    decision = TrialDevRecruitmentContinuation(study_id=plan.study_id, resume_day=200)
    continued = continue_trialdev_study_recruitment(budget_path, decision)
    assert continued.enrollments == ledger.enrollments and continued.latest_study_costs == ledger.latest_study_costs
    assert trialdev_committed_accrual(continued, plan.study_id, programme_day=300) == resumed
    assert continue_trialdev_study_recruitment(budget_path, decision) == continued
    with pytest.raises(ValueError, match="different date"):
        continue_trialdev_study_recruitment(budget_path, decision.model_copy(update={"resume_day": 205}))
    assert read_json_model(TrialDevBudgetLedger, budget_path) == continued

    from trialagentbench.trialdev.resources import reconcile_trialdev_cost

    for day in (200, 250, 301):
        with pytest.raises(ValueError, match="committed look schedule"):
            reconcile_trialdev_cost(
                continued,
                TrialDevCostReceipt(study_id=plan.study_id, readout_id=f"extra-{day}", cost=late, programme_day=day),
            )
    resource_only = reconcile_trialdev_cost(
        continued,
        TrialDevCostReceipt(
            study_id=plan.study_id, readout_id="clock-250", kind="resource", cost=late, programme_day=250
        ),
    )
    assert len([item for item in resource_only.receipts if item.kind == "readout"]) == 1
    assert resource_only.latest_study_costs == continued.latest_study_costs
    from trialagentbench.trialdev.materialization import trialdev_next_readout_day
    from trialagentbench.trialdev.resources import close_trialdev_arm, continue_trialdev_recruitment

    assert trialdev_next_readout_day(tmp_path / "unused-pool", resource_only, plan.study_id) == 300
    unobserved = TrialDevBudgetLedger.model_validate(
        {**ledger.model_dump(), "receipts": (ledger.receipts[0].model_copy(update={"kind": "resource"}),)}
    )
    with pytest.raises(ValueError, match="published readout"):
        continue_trialdev_recruitment(unobserved, decision)
    with pytest.raises(ValueError, match="after its first readout"):
        close_trialdev_arm(unobserved, plan.study_id, "active")
    final = reconcile_trialdev_cost(
        resource_only,
        TrialDevCostReceipt(
            study_id=plan.study_id,
            readout_id="final",
            cost=late,
            programme_day=300,
            supported_arms=("control", "active"),
        ),
    )
    with pytest.raises(ValueError, match="committed look schedule"):
        reconcile_trialdev_cost(
            final, TrialDevCostReceipt(study_id=plan.study_id, readout_id="third", cost=late, programme_day=400)
        )
    assert reconcile_trialdev_cost(final, final.receipts[-1]) is final

    import pandas as pd

    from trialagentbench.trialdev.resources import (
        commit_trialdev_recruitment,
        reserve_trialdev_patients,
        trialdev_phase_tariff,
    )

    next_plan = plan.model_copy(update={"study_id": "exploration"})
    reserved_next = reserve_trialdev_patients(
        final,
        _candidates(pd.DataFrame({"participant_id": range(1, 101)})),
        next_plan,
        pool_manifest_sha256="b" * 64,
        seed=1,
    )
    next_design = final.recruitment_commitments[0].model_copy(
        update={
            "study_id": "exploration",
            "phase": "exploration",
            "start_day": 300,
            "tariff": trialdev_phase_tariff("exploration"),
            "pause_after_enrolled": None,
            "looks": (final.recruitment_commitments[0].looks[-1].model_copy(update={"deadline_day": 500}),),
        }
    )
    with pytest.raises(ValueError, match="backdated"):
        commit_trialdev_recruitment(reserved_next, next_design.model_copy(update={"start_day": 299}))
    with pytest.raises(ValueError, match="needs a readout from"):
        commit_trialdev_recruitment(reserved_next, next_design.model_copy(update={"phase": "confirmation"}))
    unseen_arm = next_design.arms[1].model_copy(
        update={"partition": next_design.arms[1].partition.model_copy(update={"asset_id": "unseen"})}
    )
    with pytest.raises(ValueError, match="needs a readout from"):
        commit_trialdev_recruitment(
            reserved_next, next_design.model_copy(update={"arms": (next_design.arms[0], unseen_arm)})
        )
    from trialagentbench.trialdev.resources import trialdev_observed_treatments

    assert trialdev_observed_treatments(final, "early_safety") == {("asset", "standard")}
    untested_regimen = unseen_arm.model_copy(
        update={"partition": next_design.arms[1].partition.model_copy(update={"dose": "intensified"})}
    )
    with pytest.raises(ValueError, match="needs a readout from"):
        commit_trialdev_recruitment(
            reserved_next, next_design.model_copy(update={"arms": (next_design.arms[0], untested_regimen)})
        )
    for supported in ((), ("control",), ("active",)):
        no_contrast = TrialDevBudgetLedger.model_validate(
            {
                **reserved_next.model_dump(),
                "receipts": tuple(
                    row.model_copy(update={"supported_arms": supported}) if row.readout_id == "final" else row
                    for row in reserved_next.receipts
                ),
            }
        )
        with pytest.raises(ValueError, match="needs a readout from"):
            commit_trialdev_recruitment(no_contrast, next_design)
    advanced = commit_trialdev_recruitment(reserved_next, next_design)
    assert advanced.enrollments == final.enrollments and advanced.receipts == final.receipts
    assert commit_trialdev_recruitment(advanced, next_design) is advanced

    from trialagentbench.trialdev.materialization import close_trialdev_study_arm

    # Dropping the only active arm closes future control recruitment as well.
    write_json_model(budget_path, continued)
    dropped = close_trialdev_study_arm(budget_path, plan.study_id, "active")
    assert dropped.reservations[0].closed and dropped.reservations[0].closed_arms == ("active",)
    assert dropped.enrollments == continued.enrollments and dropped.latest_study_costs == continued.latest_study_costs
    stopped = trialdev_committed_accrual(dropped, plan.study_id, programme_day=300)
    assert stopped == halted
    assert close_trialdev_study_arm(budget_path, plan.study_id, "active") == dropped
    assert read_json_model(TrialDevBudgetLedger, budget_path) == dropped
    with pytest.raises(ValueError, match="is not an active arm"):
        close_trialdev_study_arm(budget_path, plan.study_id, "control")
    assert {patient.participant_id for patient in stopped.patients} == {
        patient.participant_id for patient in halted.patients
    }


def test_dropped_arm_releases_unused_patients_while_other_arms_continue(tmp_path):
    import pandas as pd

    from trialagentbench.contracts.trialdev.pools import TrialDevAssignedArm, TrialDevPoolPartition
    from trialagentbench.trialdev.recruitment import TrialDevEnrollmentPlan, TrialDevStudyEnrollment
    from trialagentbench.trialdev.resources import (
        TrialDevBudgetLedger,
        TrialDevCostReceipt,
        TrialDevLook,
        TrialDevRecruitmentCommitment,
        close_trialdev_arm,
        commit_trialdev_recruitment,
        reconcile_trialdev_cost,
        reconcile_trialdev_enrollment,
        reserve_trialdev_patients,
        trialdev_committed_accrual,
        trialdev_phase_tariff,
    )

    frame = _candidates(pd.DataFrame({"participant_id": range(60)}))
    plan = TrialDevEnrollmentPlan.model_validate(
        {
            "study_id": "multi",
            "site_ids": ("site_001", "site_002"),
            "screen_failure_fraction": 0.0,
            "arms": [{"arm_id": arm, "participant_count": 4} for arm in ("control", "a", "b")],
        }
    )
    state = reserve_trialdev_patients(
        TrialDevBudgetLedger(budget_usd=10_000_000), frame, plan, pool_manifest_sha256="a" * 64, seed=7
    )
    tariff = trialdev_phase_tariff("early_safety")
    design = TrialDevRecruitmentCommitment(
        study_id="multi",
        phase="early_safety",
        tariff=tariff,
        sites=2,
        start_day=0,
        looks=(TrialDevLook(deadline_day=72), TrialDevLook(deadline_day=300)),
        arms=tuple(
            TrialDevAssignedArm(
                arm_id=arm,
                partition=TrialDevPoolPartition(
                    asset_id=arm, dose=None if arm == "control" else "standard", directory=f"partitions/p{i:03d}"
                ),
            )
            for i, arm in enumerate(("control", "a", "b"))
        ),
        follow_up_days=28,
        assessment_days=(0, 28),
        surveillance_days=(0, 28),
    )
    state = commit_trialdev_recruitment(state, design)
    first = trialdev_committed_accrual(state, "multi", programme_day=72)
    state = reconcile_trialdev_enrollment(
        state,
        pool_manifest_sha256="a" * 64,
        enrollment=TrialDevStudyEnrollment(
            study_id="multi",
            design_sha256="b" * 64,
            patients=first.patients,
            screened=first.screened,
        ),
    )
    cost = trialdev_study_cost(
        tariff,
        sites=2,
        screened=first.screened,
        enrollment_days=tuple(p.enrollment_day for p in first.patients),
        programme_day=72,
        follow_up_days=28,
        assessment_days=(0, 28),
        surveillance_days=(0, 28),
    )
    state = reconcile_trialdev_cost(
        state, TrialDevCostReceipt(study_id="multi", readout_id="first", cost=cost, programme_day=72)
    )
    from trialagentbench.io import read_json_model, write_json_model
    from trialagentbench.trialdev.materialization import advance_trialdev_programme_clock
    from trialagentbench.trialdev.value import TrialDevSponsorPreferences, trialdev_frozen_programme_value

    preferences = TrialDevSponsorPreferences(outcome_unit="clinical units", benefit_unit=1)
    with pytest.raises(ValueError, match="resource accounting"):
        trialdev_frozen_programme_value(
            decision="stop", target=None, preferences=preferences, ledger=state, elapsed_days=300
        )
    clock_path = tmp_path / "clock.json"
    write_json_model(clock_path, state)
    refreshed = advance_trialdev_programme_clock(clock_path, 300)
    assert refreshed.resource_day == 300
    assert len(refreshed.enrollments[0].patients) == 12
    assert refreshed.spent_usd + refreshed.remaining_liability_usd == 1_312_160
    assert len([row for row in refreshed.receipts if row.kind == "readout"]) == 1
    from trialagentbench.trialdev.materialization import finish_trialdev_run
    from trialagentbench.trialdev.resources import TrialDevTerminalDecision

    with pytest.raises(ValueError, match="recorded terminal"):
        trialdev_frozen_programme_value(
            decision="stop", target=None, preferences=preferences, ledger=refreshed, elapsed_days=300
        )
    terminal = finish_trialdev_run(clock_path, TrialDevTerminalDecision(decision="stop", programme_day=300))
    assert terminal.spent_usd == refreshed.spent_usd + refreshed.remaining_liability_usd
    assert terminal.remaining_liability_usd == 0
    assert all(reservation.closed for reservation in terminal.reservations)
    assert finish_trialdev_run(clock_path, terminal.terminal) == terminal
    with pytest.raises(ValueError, match="immutable"):
        finish_trialdev_run(clock_path, TrialDevTerminalDecision(decision="stop", programme_day=301))
    with pytest.raises(ValueError, match="immutable"):
        advance_trialdev_programme_clock(clock_path, 301)
    assert (
        trialdev_frozen_programme_value(
            decision="stop", target=None, preferences=preferences, ledger=terminal, elapsed_days=300
        ).net_value_usd
        == -5_812_160
    )
    assert advance_trialdev_programme_clock(clock_path, 300.0) == terminal
    with pytest.raises(ValueError, match="already at day"):
        advance_trialdev_programme_clock(clock_path, 299)
    assert read_json_model(TrialDevBudgetLedger, clock_path) == terminal
    dropped = close_trialdev_arm(state, "multi", "a")
    assert not dropped.reservations[0].closed
    with pytest.raises(ValueError, match="earlier cutoff"):
        trialdev_committed_accrual(dropped, "multi", programme_day=70)
    later = trialdev_committed_accrual(dropped, "multi", programme_day=300)
    assert tuple(p for p in later.patients if p.arm_id == "a") == tuple(p for p in first.patients if p.arm_id == "a")
    assert sum(p.arm_id == "b" for p in later.patients) == 4
    assert sum(p.arm_id == "control" for p in later.patients) == 4
    assert set(first.patients).issubset(set(later.patients))
    used = {p.participant_id for p in first.patients}
    unused_a = {p.participant_id for p in state.reservations[0].patients if p.arm_id == "a"} - used
    assert unused_a
    next_state = reserve_trialdev_patients(
        dropped, frame, plan.model_copy(update={"study_id": "next"}), pool_manifest_sha256="a" * 64, seed=7
    )
    new_ids = {p.participant_id for p in next_state.reservations[-1].patients}
    assert unused_a.issubset(new_ids) and not used.intersection(new_ids)
    assert next_state.receipts == state.receipts

    from trialagentbench.io import read_json_model, write_json_model
    from trialagentbench.trialdev.materialization import reduce_trialdev_study_control

    control_cap = max(2, sum(p.arm_id == "control" for p in first.patients))
    assert control_cap < 4
    budget_path = tmp_path / "reduced_control.json"
    write_json_model(budget_path, dropped)
    reduced = reduce_trialdev_study_control(budget_path, "multi", control_cap)
    after_reduction = trialdev_committed_accrual(reduced, "multi", programme_day=300)
    assert sum(p.arm_id == "control" for p in after_reduction.patients) == control_cap
    assert set(first.patients).issubset(set(after_reduction.patients))
    assert reduced.latest_study_costs == dropped.latest_study_costs and reduced.enrollments == dropped.enrollments
    assert reduce_trialdev_study_control(budget_path, "multi", control_cap) == reduced
    with pytest.raises(ValueError, match="at most the current cap"):
        reduce_trialdev_study_control(budget_path, "multi", 4)
    with pytest.raises(ValueError, match="already enrolled"):
        reduce_trialdev_study_control(budget_path, "multi", 1)
    assert read_json_model(TrialDevBudgetLedger, budget_path) == reduced


def test_terminal_decision_requires_confirmation_and_respects_horizon():
    from trialagentbench.trialdev.resources import (
        TrialDevBudgetLedger,
        TrialDevCostReceipt,
        TrialDevStudyCost,
        TrialDevTerminalDecision,
        advance_trialdev_clock,
        finish_trialdev_programme,
        reconcile_trialdev_cost,
    )

    ledger = TrialDevBudgetLedger(budget_usd=1000)
    with pytest.raises(ValueError, match="confirmation readout"):
        finish_trialdev_programme(
            ledger, TrialDevTerminalDecision(decision="advance", programme_day=0, asset_id="asset", dose="standard")
        )
    with pytest.raises(ValueError, match="horizon"):
        advance_trialdev_clock(ledger, 2191)
    stopped = finish_trialdev_programme(ledger, TrialDevTerminalDecision(decision="stop", programme_day=2190))
    assert stopped.spent_usd == 0
    with pytest.raises(ValueError, match="terminal programme"):
        reconcile_trialdev_cost(
            stopped,
            TrialDevCostReceipt(
                study_id="late",
                readout_id="late",
                programme_day=1460,
                cost=TrialDevStudyCost(accrued_usd=1, remaining_liability_usd=0),
            ),
        )
    with pytest.raises(ValueError, match="settlement"):
        TrialDevBudgetLedger(budget_usd=1000, settlement_usd=1)


def test_fixed_population_utility_counts_cumulative_costs_and_signed_safety_once() -> None:
    from functools import partial

    from trialagentbench.contracts.trialdev.pools import TrialDevPoolContrast
    from trialagentbench.trialdev.resources import (
        TrialDevBudgetLedger,
        TrialDevCostReceipt,
        TrialDevRegulatoryReview,
        TrialDevStudyCost,
        TrialDevTerminalDecision,
    )
    from trialagentbench.trialdev.value import (
        TrialDevSponsorPreferences,
        trialdev_frozen_programme_value,
        trialdev_market_years,
    )

    approved = TrialDevRegulatoryReview(
        completed_treated=300, completed_control=300, benefit_z_statistic=2.5, approved=True
    )
    rejected = TrialDevRegulatoryReview(
        completed_treated=300, completed_control=300, benefit_z_statistic=1.5, approved=False
    )

    def terminal_ledger(
        decision: str, review: TrialDevRegulatoryReview | None, extra: tuple = ()
    ) -> TrialDevBudgetLedger:
        # A second receipt updates the same study's cumulative bill, not a new charge.
        receipts = (
            TrialDevCostReceipt(
                study_id="one",
                readout_id="interim",
                programme_day=20,
                cost=TrialDevStudyCost(accrued_usd=100, remaining_liability_usd=100),
            ),
            TrialDevCostReceipt(
                study_id="one",
                readout_id="final",
                programme_day=40,
                cost=TrialDevStudyCost(accrued_usd=150, remaining_liability_usd=50),
            ),
            TrialDevCostReceipt(
                study_id="two",
                readout_id="second",
                programme_day=40,
                cost=TrialDevStudyCost(accrued_usd=30, remaining_liability_usd=20),
            ),
            *extra,
        )
        terminal = TrialDevTerminalDecision(
            decision=decision,
            programme_day=40,
            asset_id="candidate" if decision == "advance" else None,
            dose="standard" if decision == "advance" else None,
            regulatory_review=review,
        )
        return TrialDevBudgetLedger(
            budget_usd=1_000_000, resource_day=40, receipts=receipts, terminal=terminal, settlement_usd=70
        )

    target = TrialDevPoolContrast(
        asset_id="candidate",
        dose="standard",
        participant_count=64000,
        reference_days=336,
        outcome_unit="points",
        outcome_mean_difference=-0.5,
        safety_risk_difference=0.02,
    )
    preferences = TrialDevSponsorPreferences(outcome_unit="points", benefit_unit=2)
    years = trialdev_market_years(preferences, 40)
    calculate = partial(
        trialdev_frozen_programme_value,
        decision="advance",
        target=target,
        preferences=preferences,
        ledger=terminal_ledger("advance", approved),
        elapsed_days=40,
    )
    value = calculate()
    assert value.approved
    assert value.clinical_value_usd == pytest.approx(0.25 * 250_000_000 * years)
    assert value.safety_cost_usd == pytest.approx(2 * 20_000_000 * years)
    assert value.study_cost_usd == 250
    assert value.time_cost_usd == 40 * 15_000
    safer = calculate(target=target.model_copy(update={"safety_risk_difference": -0.02}))
    # Risk reduction adds value; no hidden safety veto.
    assert safer.net_value_usd == pytest.approx(value.net_value_usd + 4 * 20_000_000 * years)
    unapproved = calculate(ledger=terminal_ledger("advance", rejected))
    assert not unapproved.approved and unapproved.clinical_value_usd == unapproved.safety_cost_usd == 0
    assert unapproved.net_value_usd == -250 - 40 * 15_000
    stopped = calculate(decision="stop", target=None, ledger=terminal_ledger("stop", None))
    assert stopped.net_value_usd == unapproved.net_value_usd
    additional = TrialDevCostReceipt(
        study_id="unnecessary",
        readout_id="extra",
        programme_day=40,
        cost=TrialDevStudyCost(accrued_usd=500, remaining_liability_usd=0),
    )
    oversized = calculate(ledger=terminal_ledger("advance", approved, (additional,)))
    assert oversized.net_value_usd == pytest.approx(value.net_value_usd - 500)
    # Delay costs a share of value through discounting and lost exclusivity.
    assert trialdev_market_years(preferences, 405.25) / years == pytest.approx(0.88, abs=0.02)
    assert trialdev_market_years(preferences, preferences.exclusivity_end_day) == 0
    with pytest.raises(ValueError, match="regulatory review"):
        calculate(ledger=terminal_ledger("advance", None))


def test_resource_receipt_cannot_claim_phase_evidence():
    from trialagentbench.trialdev.resources import TrialDevCostReceipt, TrialDevStudyCost

    with pytest.raises(ValueError, match="resource receipts"):
        TrialDevCostReceipt(
            study_id="s",
            readout_id="wait",
            kind="resource",
            supported_arms=("control",),
            cost=TrialDevStudyCost(accrued_usd=0, remaining_liability_usd=0),
        )
