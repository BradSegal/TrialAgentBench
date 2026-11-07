"""Persisted authored scores retain scheduled targets and failure denominators."""

import pytest
from pydantic import ValidationError

from trialagentbench_verifier.authored_results import (
    TrialDevNumericalRecovery,
    TrialDevProgrammeEvaluation,
    TrialDevStudyEvaluation,
    TrialDevStudyReport,
)
from trialagentbench_verifier.numerical import NumericalReplay
from trialagentbench_verifier.response import TrialDevFrozenProgrammeValue

# An unfinished programme's value: a stop at its last day, charged its incurred costs.
_INCURRED = TrialDevFrozenProgrammeValue(
    decision="stop", clinical_value_usd=0, safety_cost_usd=0, study_cost_usd=9, time_cost_usd=1, net_value_usd=-10
)


def _study() -> TrialDevStudyEvaluation:
    return TrialDevStudyEvaluation(
        report=TrialDevStudyReport(
            study_id="study",
            checkpoint_id="cutoff",
            captured_relative_path=None,
            final_readout=True,
            programme_complete=False,
        ),
        replay=NumericalReplay(
            status="missing_result", entrypoint="analysis.py", target_ids=("effect",), evidence_artifacts=()
        ),
        scores=(
            TrialDevNumericalRecovery(
                target_id="effect",
                status="missing",
                reference_value=0,
                normalization_scale=1,
                normalized_absolute_error=None,
                normalized_interval_loss=None,
            ),
        ),
    )


@pytest.mark.parametrize("defect", ["omitted", "duplicated", "renamed"])
def test_study_cannot_change_target_denominator(defect: str) -> None:
    payload = _study().model_dump()
    if defect == "omitted":
        payload["scores"] = ()
    elif defect == "duplicated":
        payload["scores"] *= 2
    else:
        payload["scores"][0]["target_id"] = "other"
    with pytest.raises(ValidationError, match="target census"):
        TrialDevStudyEvaluation.model_validate(payload)


def test_programme_cannot_count_one_acquisition_twice() -> None:
    study = _study()
    with pytest.raises(ValidationError, match="repeats"):
        TrialDevProgrammeEvaluation(programme_complete=False, value=_INCURRED, studies=(study, study))


def test_unfinished_programme_is_valued_only_as_a_stop_at_its_incurred_costs() -> None:
    """Noncompletion carries its incurred costs, never no value or an advancement payoff."""
    assert TrialDevProgrammeEvaluation(programme_complete=False, value=_INCURRED, studies=()).value == _INCURRED
    with pytest.raises(ValidationError, match="valued only as a stop"):
        TrialDevProgrammeEvaluation(
            programme_complete=False, value=_INCURRED.model_copy(update={"decision": "advance"}), studies=()
        )
    with pytest.raises(ValidationError, match="value"):
        TrialDevProgrammeEvaluation(programme_complete=False, value=None, studies=())


@pytest.mark.parametrize(
    "change", [{"credit": 0.5}, {"normalization_scale": 0}, {"credit": 2}, {"normalized_interval_loss": -1}]
)
def test_missing_target_cannot_receive_credit_or_invalid_scale(change: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        TrialDevNumericalRecovery.model_validate(_study().scores[0].model_dump() | change)


@pytest.mark.parametrize("defect", ["missing_score", "duplicate_score", "false_completion", "missing_error"])
def test_standalone_analysis_preserves_outcome_and_denominator(defect: str) -> None:
    from trialagentbench_verifier.authored_results import TrialDevAnalysisRun

    study = _study()
    record = TrialDevAnalysisRun(
        execution_status="model_noncompletion", replay=study.replay, scores=study.scores, error="No analysis"
    )
    payload = record.model_dump()
    if defect == "missing_score":
        payload["scores"] = ()
    elif defect == "duplicate_score":
        payload["scores"] *= 2
    elif defect == "false_completion":
        payload["execution_status"] = "completed"
    else:
        payload["error"] = None
    with pytest.raises(ValidationError):
        TrialDevAnalysisRun.model_validate(payload)
