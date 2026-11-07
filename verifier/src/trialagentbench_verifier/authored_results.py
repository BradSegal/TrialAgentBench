"""Shared persisted authored-programme results, independent of construction and harness."""

from statistics import mean
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_verifier.numerical import NumericalReplay
from trialagentbench_verifier.response import TrialDevFrozenProgrammeValue, TrialDevTargetStatus


class TrialDevNumericalRecovery(BaseModel):
    """Per-target numerical performance; never an unconditional programme ranking."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    target_id: str = Field(min_length=1)
    status: Literal["reported", "missing", "invalid_replay"]
    reference_value: float
    normalization_scale: float = Field(gt=0)
    normalized_absolute_error: float | None = Field(ge=0)
    normalized_interval_loss: float | None = Field(ge=0)

    @model_validator(mode="after")
    def validate_missing_loss(self) -> Self:
        """Keep absent or unreproduced answers unavailable."""
        if self.status != "reported" and (
            self.normalized_absolute_error is not None or self.normalized_interval_loss is not None
        ):
            raise ValueError("Unreported numerical targets cannot carry loss diagnostics")
        return self

    @property
    def loss(self) -> float | None:
        """Return the untransformed mean point/interval loss, or an unavailable result."""
        point, interval = self.normalized_absolute_error, self.normalized_interval_loss
        if self.status != "reported" or point is None or interval is None:
            return None
        return point / 2 + interval / 2


def mean_numerical_loss(scores: tuple[TrialDevNumericalRecovery, ...]) -> float | None:
    """Average every scheduled target; any unavailable target withholds the mean."""
    losses = [score.loss for score in scores]
    if not losses or any(value is None for value in losses):
        return None
    return mean(value for value in losses if value is not None)


def numerical_target_statuses(scores: tuple[TrialDevNumericalRecovery, ...]) -> dict[str, TrialDevTargetStatus]:
    """Retain partial recovery without manufacturing a task mean or failure cause."""
    if len({score.target_id for score in scores}) != len(scores):
        raise ValueError("Numerical target status census repeats an identity")
    return {
        score.target_id: (
            "reported" if score.loss is not None else "unscored" if score.status == "reported" else score.status
        )
        for score in scores
    }


class TrialDevStudyReport(BaseModel):
    """Latest acquired study readout with explicit completion and missingness."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    study_id: str
    checkpoint_id: str
    captured_relative_path: str | None
    final_readout: bool
    programme_complete: bool


class TrialDevStudyEvaluation(BaseModel):
    """Acquired study's reporting status and target-wise numerical recovery."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    report: TrialDevStudyReport
    replay: NumericalReplay
    scores: tuple[TrialDevNumericalRecovery, ...]

    @model_validator(mode="after")
    def validate_target_census(self) -> Self:
        """Preserve exactly one score for every scheduled replay target."""
        identities = tuple(score.target_id for score in self.scores)
        if len(set(identities)) != len(identities) or set(identities) != set(self.replay.target_ids):
            raise ValueError("Study scores differ from the scheduled replay target census")
        return self


class TrialDevProgrammeEvaluation(BaseModel):
    """Development value and unaggregated study diagnostics for an authored programme.

    An unfinished programme is valued as a stop at its last programme day: no
    market value, with the study costs and overhead it incurred.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    programme_complete: bool
    value: TrialDevFrozenProgrammeValue
    studies: tuple[TrialDevStudyEvaluation, ...]
    initial_replay: NumericalReplay | None = None
    initial_scores: tuple[TrialDevNumericalRecovery, ...] = ()

    @model_validator(mode="after")
    def validate_terminal_value(self) -> Self:
        """Value an unfinished programme only as a stop."""
        if not self.programme_complete and self.value.decision != "stop":
            raise ValueError("An unfinished programme is valued only as a stop")
        study_ids = tuple(study.report.study_id for study in self.studies)
        checkpoints = tuple(study.report.checkpoint_id for study in self.studies)
        if len(set(study_ids)) != len(study_ids) or len(set(checkpoints)) != len(checkpoints):
            raise ValueError("Programme evaluation repeats an acquired study or checkpoint")
        initial_ids = tuple(score.target_id for score in self.initial_scores)
        expected = () if self.initial_replay is None else self.initial_replay.target_ids
        if len(set(initial_ids)) != len(initial_ids) or set(initial_ids) != set(expected):
            raise ValueError("Initial scores differ from the scheduled replay target census")
        return self


class TrialDevProgrammeRun(BaseModel):
    """Authored execution outcome, distinct from a voluntary terminal decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    execution_status: Literal["completed", "model_noncompletion"]
    error: str | None
    evaluation: TrialDevProgrammeEvaluation

    @model_validator(mode="after")
    def validate_execution_outcome(self) -> Self:
        """Keep model noncompletion distinct from an actual development decision."""
        completed = self.execution_status == "completed"
        if completed != self.evaluation.programme_complete or completed != (self.error is None):
            raise ValueError("Programme execution status, error and terminal evaluation disagree")
        return self


class TrialDevAnalysisRun(BaseModel):
    """One standalone numerical analysis and its separately scored recovery."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    execution_status: Literal["completed", "model_noncompletion"]
    replay: NumericalReplay
    scores: tuple[TrialDevNumericalRecovery, ...]
    report_relative_path: str | None = None
    error: str | None = None

    @model_validator(mode="after")
    def validate_analysis_outcome(self) -> Self:
        """Keep target counts and captured completion consistent across consumers."""
        completed = self.execution_status == "completed"
        if completed != (self.report_relative_path is not None) or completed != (self.error is None):
            raise ValueError("Analysis execution status, report and error disagree")
        identities = tuple(score.target_id for score in self.scores)
        if len(set(identities)) != len(identities) or set(identities) != set(self.replay.target_ids):
            raise ValueError("Analysis scores differ from the scheduled replay target census")
        return self
