"""Participant programme tools executed through the frozen programme lifecycle.

Agents call a small set of flat tools (``start_study``, ``request_readout``, ``adapt_study``, ``wait_until``, ``stop_development``,
``advance_candidate``). Deterministic code turns each call into one internal action:
it maps compound names to the frozen pool's assets, names the control arm
``control``, numbers studies by phase (``study-101``, ``study-201``, ...) and names
each readout folder after its study and look (``study-201-look-1``). The internal
action is what is persisted, replayed and audited.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from pydantic.types import JsonValue

from trialagentbench.contracts.trialdev.pools import TrialDevArmId, TrialDevFrozenPool
from trialagentbench.contracts.trialdev.release import TrialDevDose
from trialagentbench.io import read_json_model, write_json_model
from trialagentbench.ports.tool_input import provider_tool
from trialagentbench.trialdev.design import TrialDevRequestedArm, TrialDevStudyRequest
from trialagentbench.trialdev.errors import TrialDevActionRejectedError
from trialagentbench.trialdev.materialization import (
    advance_trialdev_programme_clock,
    close_trialdev_study_arm,
    close_trialdev_study_reservation,
    continue_trialdev_study_recruitment,
    finish_trialdev_run,
    reduce_trialdev_study_control,
    start_trialdev_study,
    write_trialdev_committed_readout,
)
from trialagentbench.trialdev.recruitment import TrialDevRecruitmentSelection
from trialagentbench.trialdev.resources import (
    TRIALDEV_HORIZON_DAYS,
    TrialDevBudgetLedger,
    TrialDevLook,
    TrialDevPhase,
    TrialDevRecruitmentContinuation,
    TrialDevTerminalDecision,
    advance_trialdev_clock,
    trialdev_observed_treatments,
)
from trialagentbench.trialdev.targets import trialdev_regulatory_review
from trialagentbench.util.run_ownership import exclusive_run_directory

_PHASE_NUMBER: dict[TrialDevPhase, int] = {"early_safety": 1, "exploration": 2, "confirmation": 3}
_STUDY_ID = r"^study-[1-3][0-9]{2}$"


# ----------------------------------------------------------------------------- internal actions
class TrialDevStartStudy(TrialDevStudyRequest):
    """Commit one resolved study design under its deterministic study identifier."""

    action: Literal["start_study"]
    study_id: str = Field(pattern=_STUDY_ID)


class TrialDevReadout(BaseModel):
    """Publish the next committed look of a study into its named evidence folder."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    action: Literal["readout"]
    study_id: str = Field(min_length=1)
    evidence_folder: str = Field(pattern=r"^study-[1-3][0-9]{2}-look-[1-2]$")


class TrialDevStudyAction(BaseModel):
    """Resume a paused cohort or close a study."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    action: Literal["continue_recruitment", "close_study"]
    study_id: str = Field(min_length=1)


class TrialDevDropArm(BaseModel):
    """Stop future recruitment to one active arm."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    action: Literal["drop_arm"]
    study_id: str = Field(min_length=1)
    arm_id: str = Field(min_length=1)


class TrialDevReduceControl(BaseModel):
    """Reduce the remaining control commitment after an active arm is dropped."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    action: Literal["reduce_control"]
    study_id: str = Field(min_length=1)
    participant_count: int = Field(ge=1, le=3000, strict=True)


class TrialDevWait(BaseModel):
    """Advance programme time without requesting an additional observation look."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    action: Literal["wait"]
    programme_day: float = Field(ge=0)


class TrialDevStop(BaseModel):
    """End development at the current programme clock."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    action: Literal["stop"]


class TrialDevAdvance(BaseModel):
    """Advance one confirmed asset and regimen at the current programme clock."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    action: Literal["advance"]
    asset_id: str = Field(min_length=1)
    dose: TrialDevDose


TrialDevAction = Annotated[
    TrialDevStartStudy
    | TrialDevReadout
    | TrialDevStudyAction
    | TrialDevDropArm
    | TrialDevReduceControl
    | TrialDevWait
    | TrialDevStop
    | TrialDevAdvance,
    Field(discriminator="action"),
]
TRIALDEV_ACTION: TypeAdapter[TrialDevAction] = TypeAdapter(TrialDevAction)


class TrialDevActionRequest(BaseModel):
    """One persisted internal programme operation."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    request: TrialDevAction


# ----------------------------------------------------------------------------- public tool arguments
class _ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class TrialDevActiveArm(_ToolArguments):
    """Validated active-arm tool arguments."""

    model_config = ConfigDict(json_schema_extra={"description": "One active arm: a compound at one dose."})

    compound: str = Field(min_length=1, description="Compound code from the committee brief, e.g. cnp-15967.")
    dose: TrialDevDose = Field(description="Dose of a compound.")
    participants: int = Field(ge=1, le=3000, strict=True, description="Enrollment cap for this arm.")
    name: TrialDevArmId | None = Field(
        default=None, description="Optional arm name used in readout folders and targets; default <compound>-<dose>."
    )


class TrialDevStudyProposal(_ToolArguments):
    """Validated study-proposal tool arguments."""

    model_config = ConfigDict(json_schema_extra={"description": "Study design choices of start_study."})

    preview: bool = Field(
        default=False,
        strict=True,
        description="Only check the design and show its study identifier and projected costs; start nothing.",
    )
    phase: TrialDevPhase
    arms: tuple[TrialDevActiveArm, ...] = Field(
        min_length=1, max_length=6, description="Active arms; control is added."
    )
    control_participants: int = Field(ge=1, le=3000, strict=True, description="Enrollment cap of the control arm.")
    sites: int = Field(ge=2, strict=True, description="Sites activated in network order, up to the network size.")
    follow_up_days: int = Field(gt=0, strict=True, description="Planned follow-up per participant; see the brief.")
    looks: tuple[TrialDevLook, ...] = Field(min_length=1, max_length=2, description="One or two readouts.")
    assessment_days: tuple[int, ...] = Field(
        default=(), description="Efficacy visit days after enrolment, from the visit grid; default every grid day."
    )
    surveillance_days: tuple[int, ...] = Field(
        default=(), description="Safety contact days after enrolment, from the visit grid; default every grid day."
    )
    recruitment: TrialDevRecruitmentSelection = Field(default_factory=TrialDevRecruitmentSelection)
    pause_after_enrolled: int | None = Field(
        default=None, ge=1, le=7999, strict=True, description="Optional interim recruitment pause; omit normally."
    )


class _AcceptIncomplete(_ToolArguments):
    accept_incomplete: bool = Field(
        default=False,
        strict=True,
        description="Proceed even if the saved analysis does not reproduce or omits targets at this cutoff.",
    )


class TrialDevReadoutArguments(_AcceptIncomplete):
    """Validated readout tool arguments."""

    model_config = ConfigDict(json_schema_extra={"description": "Request the next committed readout of a study."})

    study_id: str = Field(min_length=1, description="Study identifier, e.g. study-201.")


class TrialDevAdaptArguments(_ToolArguments):
    """Validated adaptation tool arguments."""

    model_config = ConfigDict(json_schema_extra={"description": "Change a running study."})

    study_id: str = Field(min_length=1)
    change: Literal["close_study", "resume_recruitment", "drop_arm", "reduce_control"]
    arm: str | None = Field(default=None, description="Arm name, for drop_arm.")
    control_participants: int | None = Field(
        default=None, ge=1, le=3000, strict=True, description="New total control cap, for reduce_control."
    )


class TrialDevWaitArguments(_ToolArguments):
    """Validated programme-clock tool arguments."""

    model_config = ConfigDict(json_schema_extra={"description": "Let programme time pass without a readout."})

    programme_day: float = Field(ge=0, description="Programme day to wait until.")


class TrialDevAdvanceArguments(_AcceptIncomplete):
    """Validated candidate-advancement tool arguments."""

    model_config = ConfigDict(
        json_schema_extra={"description": "Advance one compound and dose for regulatory submission."}
    )

    compound: str = Field(min_length=1)
    dose: TrialDevDose = Field(description="Dose of a compound.")


_PROGRAMME_TOOLS: dict[str, tuple[type[_ToolArguments], str]] = {
    "start_study": (
        TrialDevStudyProposal,
        "Start a study (or, with preview=true, only check and cost it). Control is added as the arm named control. "
        "Returns the study identifier, the resolved design and the programme status.",
    ),
    "request_readout": (
        TrialDevReadoutArguments,
        "Move the shared programme clock to the study's next planned look and release its data, which never "
        "changes afterwards. Each request uses one planned look, even if few or no participants have completed. "
        "Your saved analysis is run first and its output kept as your report for the current cutoff.",
    ),
    "adapt_study": (TrialDevAdaptArguments, "Close a study, resume a paused cohort, drop an arm or reduce control."),
    "wait_until": (
        TrialDevWaitArguments,
        "Move the shared programme clock to a later day without a readout; every study keeps recruiting and "
        "accruing costs. A look whose deadline passes meanwhile is kept: requesting it releases data as of that day.",
    ),
    "stop_development": (
        _AcceptIncomplete,
        "End development without advancing any compound. Your saved analysis is run first and kept.",
    ),
    "advance_candidate": (
        TrialDevAdvanceArguments,
        "End development by advancing one confirmed compound and dose under the simplified benchmark approval rule, not regulatory practice. Your saved "
        "analysis is run first and kept.",
    ),
}


def trialdev_action_names() -> tuple[str, ...]:
    """Return the public programme tool names in their offered order."""
    return tuple(_PROGRAMME_TOOLS)


def trialdev_programme_tools() -> list[dict[str, JsonValue]]:
    """Return the flat provider declarations of the programme tools."""
    return [provider_tool(name, description, model) for name, (model, description) in _PROGRAMME_TOOLS.items()]


def trialdev_terminal_tools() -> list[dict[str, JsonValue]]:
    """Return only the terminal decision tools offered on a programme's final turn."""
    terminal = ("stop_development", "advance_candidate")
    return [tool for name, tool in zip(_PROGRAMME_TOOLS, trialdev_programme_tools(), strict=True) if name in terminal]


@dataclass(frozen=True)
class TrialDevProgrammeCall:
    """One decoded programme tool call."""

    action: TrialDevAction
    preview: bool = False
    accept_incomplete: bool = False


# ----------------------------------------------------------------------------- public status
class TrialDevArmStatus(BaseModel):
    """Observable treatment, enrollment and remaining recruitment commitment."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    arm_id: str
    asset_id: str
    dose: TrialDevDose | None
    enrollment_cap: int = Field(ge=0)
    enrolled: int = Field(ge=0)
    recruitment_closed: bool
    phase_evidence_available: bool


class TrialDevStudyStatus(BaseModel):
    """Study progress without reserved identities or future observation values."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    study_id: str
    phase: TrialDevPhase
    start_day: float
    closed: bool
    recruitment_paused: bool
    arms: tuple[TrialDevArmStatus, ...]
    completed_looks: int = Field(ge=0)
    remaining_looks: tuple[TrialDevLook, ...]


class TrialDevProgrammeStatus(BaseModel):
    """Structured public resource accounting and study progress, rendered as letters."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    programme_day: float
    horizon_days: int
    budget_usd: float
    spent_usd: float
    committed_followup_usd: float
    available_usd: float
    studies: tuple[TrialDevStudyStatus, ...]
    terminal: TrialDevTerminalDecision | None
    readout_path: str | None = None
    preview: bool = False


def trialdev_programme_status(ledger: TrialDevBudgetLedger) -> TrialDevProgrammeStatus:
    """Project refreshed private state onto the participant-visible programme surface.

    Parameters
    ----------
    ledger
        Evaluator-owned state after the shared resource clock has run.

    Returns
    -------
    TrialDevProgrammeStatus
        Public costs, study progress and protocol look deadlines. No private pool
        identity, recruitment seed, patient queue or future record is included.

    Raises
    ------
    ValueError
        If programme resources have not been reconciled to current activity.

    Examples
    --------
    Remaining look deadlines describe the submitted protocol, not the hidden date
    at which a future information threshold will actually be met.
    """
    if ledger.resource_day != ledger.current_day:
        raise ValueError("public status requires current resource accounting")
    studies = []
    for study in ledger.recruitment_commitments:
        reservation = next(row for row in ledger.reservations if row.plan.study_id == study.study_id)
        enrollment = next((row for row in ledger.enrollments if row.study_id == study.study_id), None)
        patients = () if enrollment is None else enrollment.patients
        continued = any(row.study_id == study.study_id for row in ledger.recruitment_continuations)
        completed = sum(row.study_id == study.study_id and row.kind == "readout" for row in ledger.receipts)
        arms = []
        for arm in study.arms:
            enrolled = sum(row.arm_id == arm.arm_id for row in patients)
            retained = sum(row.arm_id == arm.arm_id for row in reservation.active_patients)
            arms.append(
                TrialDevArmStatus(
                    arm_id=arm.arm_id,
                    asset_id=arm.partition.asset_id,
                    dose=arm.partition.dose,
                    enrolled=enrolled,
                    phase_evidence_available=(arm.partition.asset_id, arm.partition.dose)
                    in trialdev_observed_treatments(ledger, study.phase),
                    enrollment_cap=max(enrolled, retained),
                    recruitment_closed=reservation.closed
                    or arm.arm_id in reservation.closed_arms
                    or retained <= enrolled,
                )
            )
        studies.append(
            TrialDevStudyStatus(
                study_id=study.study_id,
                phase=study.phase,
                start_day=study.start_day,
                closed=reservation.closed,
                recruitment_paused=(
                    not reservation.closed
                    and not continued
                    and study.pause_after_enrolled is not None
                    and len(patients) >= study.pause_after_enrolled
                    and any(not arm.recruitment_closed for arm in arms)
                ),
                arms=tuple(arms),
                completed_looks=completed,
                remaining_looks=() if ledger.terminal is not None else study.looks[completed:],
            )
        )
    return TrialDevProgrammeStatus(
        programme_day=ledger.current_day,
        horizon_days=TRIALDEV_HORIZON_DAYS,
        budget_usd=ledger.budget_usd,
        spent_usd=ledger.spent_usd,
        committed_followup_usd=ledger.remaining_liability_usd,
        available_usd=ledger.available_usd,
        studies=tuple(studies),
        # The regulator's analysis uses complete potential outcomes and stays private.
        terminal=None if ledger.terminal is None else ledger.terminal.model_copy(update={"regulatory_review": None}),
    )


def _rejected_arguments(name: str, error: ValidationError) -> TrialDevActionRejectedError:
    """Explain invalid tool arguments field by field, without echoing pydantic internals."""
    problems = "; ".join(
        f"{'.'.join(str(part) for part in item['loc']) or 'arguments'}: {item['msg']}"
        for item in error.errors(include_url=False, include_input=False)
    )
    return TrialDevActionRejectedError(f"{name} arguments are invalid ({problems})")


# ----------------------------------------------------------------------------- session
@dataclass(frozen=True)
class TrialDevActionSession:
    """Evaluator-owned paths and seed for one authored programme.

    These fields are runtime custody, never participant tool arguments. The caller
    serializes decisions within a programme and assigns a stable action identity
    before execution. Existing lifecycle transactions protect each mutation.
    """

    pool_root: Path
    budget_path: Path
    evidence_root: Path
    commitment_root: Path
    recruitment_seed: int

    def _manifest(self) -> TrialDevFrozenPool:
        return read_json_model(TrialDevFrozenPool, self.pool_root / "manifest.json")

    def _ledger(self) -> TrialDevBudgetLedger:
        return read_json_model(TrialDevBudgetLedger, self.budget_path)

    def _asset(self, compound: str, manifest: TrialDevFrozenPool) -> str:
        """Map a compound name, in any letter case, to its frozen-pool asset."""
        assets = sorted({row.asset_id for row in manifest.partitions if row.asset_id != "control"})
        matches = [asset for asset in assets if asset.casefold() == compound.strip().casefold()]
        if not matches:
            raise TrialDevActionRejectedError(
                f"unknown compound {compound!r}; this programme's compounds are {', '.join(assets)}"
            )
        return matches[0]

    def _study(self, study_id: str, ledger: TrialDevBudgetLedger) -> str:
        known = [row.study_id for row in ledger.recruitment_commitments]
        if study_id not in known:
            raise TrialDevActionRejectedError(
                f"unknown study {study_id!r}; "
                + (f"studies in this programme: {', '.join(known)}" if known else "no study has been started")
            )
        return study_id

    def resolve_study(self, proposal: TrialDevStudyProposal) -> TrialDevStartStudy:
        """Resolve public design choices into the committed design and its study identifier."""
        manifest = self._manifest()
        ledger = self._ledger()
        arms: list[dict[str, object]] = [
            {
                "arm_id": "control",
                "asset_id": "control",
                "dose": None,
                "participant_count": proposal.control_participants,
            }
        ]
        for arm in proposal.arms:
            asset = self._asset(arm.compound, manifest)
            name = arm.name or f"{asset}-{arm.dose.value}"
            if name == "control":
                raise TrialDevActionRejectedError("the name control is reserved for the control arm")
            arms.append({"arm_id": name, "asset_id": asset, "dose": arm.dose, "participant_count": arm.participants})
        grid = tuple(day for day in manifest.collection_days if day <= proposal.follow_up_days)
        number = 1 + sum(row.phase == proposal.phase for row in ledger.recruitment_commitments)
        if number > 99:
            raise TrialDevActionRejectedError("a programme can run at most 99 studies per phase")
        try:
            return TrialDevStartStudy(
                action="start_study",
                study_id=f"study-{_PHASE_NUMBER[proposal.phase]}{number:02d}",
                phase=proposal.phase,
                arms=tuple(TrialDevRequestedArm.model_validate(arm) for arm in arms),
                sites=proposal.sites,
                recruitment=proposal.recruitment,
                follow_up_days=proposal.follow_up_days,
                assessment_days=proposal.assessment_days or grid,
                surveillance_days=proposal.surveillance_days or grid,
                looks=proposal.looks,
                pause_after_enrolled=proposal.pause_after_enrolled,
            )
        except ValidationError as error:
            raise _rejected_arguments("study design", error) from error

    def decode(self, name: str, arguments: dict[str, object]) -> TrialDevProgrammeCall:
        """Validate one public tool call and resolve it into one internal action.

        Parameters
        ----------
        name
            Public programme tool name.
        arguments
            Parsed flat tool arguments.

        Returns
        -------
        TrialDevProgrammeCall
            The internal action, whether it is a preview, and the incomplete-report choice.

        Raises
        ------
        TrialDevActionRejectedError
            If the tool is unknown or its arguments break a stated rule; the message names
            the rule and the allowed values.
        """
        if name not in _PROGRAMME_TOOLS:
            raise TrialDevActionRejectedError(f"unknown programme tool {name!r}")
        model = _PROGRAMME_TOOLS[name][0]
        try:
            parsed = model.model_validate(arguments)
        except ValidationError as error:
            raise _rejected_arguments(name, error) from error
        ledger = self._ledger()
        if isinstance(parsed, TrialDevStudyProposal):
            return TrialDevProgrammeCall(action=self.resolve_study(parsed), preview=parsed.preview)
        if isinstance(parsed, TrialDevReadoutArguments):
            study_id = self._study(parsed.study_id, ledger)
            look = 1 + sum(row.study_id == study_id and row.kind == "readout" for row in ledger.receipts)
            if look > 2:
                raise TrialDevActionRejectedError(f"{study_id} has no remaining committed look")
            return TrialDevProgrammeCall(
                action=TrialDevReadout(action="readout", study_id=study_id, evidence_folder=f"{study_id}-look-{look}"),
                accept_incomplete=parsed.accept_incomplete,
            )
        if isinstance(parsed, TrialDevAdaptArguments):
            study_id = self._study(parsed.study_id, ledger)
            action: TrialDevAction
            if parsed.change == "drop_arm":
                if parsed.arm is None:
                    raise TrialDevActionRejectedError("drop_arm requires arm, the name of an active arm")
                action = TrialDevDropArm(action="drop_arm", study_id=study_id, arm_id=parsed.arm)
            elif parsed.change == "reduce_control":
                if parsed.control_participants is None:
                    raise TrialDevActionRejectedError(
                        "reduce_control requires control_participants, the new total cap"
                    )
                action = TrialDevReduceControl(
                    action="reduce_control", study_id=study_id, participant_count=parsed.control_participants
                )
            else:
                action = TrialDevStudyAction(
                    action="close_study" if parsed.change == "close_study" else "continue_recruitment",
                    study_id=study_id,
                )
            return TrialDevProgrammeCall(action=action)
        if isinstance(parsed, TrialDevWaitArguments):
            return TrialDevProgrammeCall(action=TrialDevWait(action="wait", programme_day=parsed.programme_day))
        if isinstance(parsed, TrialDevAdvanceArguments):
            return TrialDevProgrammeCall(
                action=TrialDevAdvance(
                    action="advance", asset_id=self._asset(parsed.compound, self._manifest()), dose=parsed.dose
                ),
                accept_incomplete=parsed.accept_incomplete,
            )
        if isinstance(parsed, _AcceptIncomplete):
            return TrialDevProgrammeCall(
                action=TrialDevStop(action="stop"), accept_incomplete=parsed.accept_incomplete
            )
        raise AssertionError(f"Unhandled programme tool {name}")

    def preview_study(self, action: TrialDevStartStudy) -> TrialDevBudgetLedger:
        """Apply real admission to a resolved design in memory, committing nothing.

        Parameters
        ----------
        action
            Resolved design and its would-be study identifier.

        Returns
        -------
        TrialDevBudgetLedger
            Hypothetical ledger after admission; it is never persisted.

        Raises
        ------
        TrialDevActionRejectedError
            If the design violates current admission rules or resource limits.
        """
        ledger = self._ledger()
        return start_trialdev_study(
            self.pool_root,
            self.budget_path,
            TrialDevStudyRequest.model_validate(action.model_dump(exclude={"action", "study_id"})),
            study_id=action.study_id,
            start_day=ledger.current_day,
            seed=self.recruitment_seed,
            validate_only=True,
        )

    def execute(self, action: TrialDevAction, *, action_id: str) -> TrialDevBudgetLedger:
        """Execute one internal action without accepting private allocation or cost inputs.

        Parameters
        ----------
        action
            Resolved participant design or decision.
        action_id
            Runner-assigned stable identity of the tool call, used for its persisted request.

        Returns
        -------
        TrialDevBudgetLedger
            Private updated state; it must not be returned verbatim to the agent.

        Raises
        ------
        ValueError
            If identity or action violates the committed programme.

        Examples
        --------
        A readout publishes only accrued raw records under evidence_root/<study>-look-<k>.
        A stop freezes the current clock and cannot choose arbitrary accounting dates.
        """
        if (
            not action_id
            or not action_id.isascii()
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in action_id)
        ):
            raise ValueError("action identity must be one safe ASCII filename component")
        self.commitment_root.mkdir(parents=True, exist_ok=True)
        request_path = self.commitment_root / f"{action_id}.request.json"
        request = TrialDevActionRequest(request=action)
        with exclusive_run_directory(self.commitment_root):
            if request_path.exists():
                if read_json_model(TrialDevActionRequest, request_path) != request:
                    raise ValueError("action identity is already committed to different inputs")
            else:
                write_json_model(request_path, request)
        ledger = self._ledger()
        if isinstance(action, TrialDevStartStudy):
            prior = next((row for row in ledger.recruitment_commitments if row.study_id == action.study_id), None)
            return start_trialdev_study(
                self.pool_root,
                self.budget_path,
                TrialDevStudyRequest.model_validate(action.model_dump(exclude={"action", "study_id"})),
                study_id=action.study_id,
                start_day=ledger.current_day if prior is None else prior.start_day,
                seed=self.recruitment_seed,
            )
        if isinstance(action, TrialDevReadout):
            return write_trialdev_committed_readout(
                self.pool_root,
                self.budget_path,
                action.study_id,
                self.evidence_root / action.evidence_folder,
                commitment_path=self.commitment_root / f"{action.evidence_folder}.json",
            )
        if isinstance(action, TrialDevStudyAction):
            if action.action == "close_study":
                return close_trialdev_study_reservation(self.budget_path, action.study_id)
            prior_continuation = next(
                (row for row in ledger.recruitment_continuations if row.study_id == action.study_id), None
            )
            return continue_trialdev_study_recruitment(
                self.budget_path,
                prior_continuation
                or TrialDevRecruitmentContinuation(study_id=action.study_id, resume_day=ledger.current_day),
            )
        if isinstance(action, TrialDevDropArm):
            return close_trialdev_study_arm(self.budget_path, action.study_id, action.arm_id)
        if isinstance(action, TrialDevReduceControl):
            return reduce_trialdev_study_control(self.budget_path, action.study_id, action.participant_count)
        if isinstance(action, TrialDevWait):
            return advance_trialdev_programme_clock(self.budget_path, action.programme_day)
        if isinstance(action, TrialDevAdvance):
            submission_day = ledger.current_day
            review = None
            if (action.asset_id, action.dose) in trialdev_observed_treatments(ledger, "confirmation"):
                # Unconfirmed treatments are rejected by the terminal lifecycle below.
                review = trialdev_regulatory_review(
                    self.pool_root,
                    advance_trialdev_clock(ledger, submission_day),
                    asset_id=action.asset_id,
                    dose=action.dose,
                    submission_day=submission_day,
                )
            decision = TrialDevTerminalDecision(
                decision="advance",
                programme_day=submission_day,
                asset_id=action.asset_id,
                dose=action.dose,
                regulatory_review=review,
            )
        elif isinstance(action, TrialDevStop):
            decision = TrialDevTerminalDecision(decision="stop", programme_day=ledger.current_day)
        else:
            raise TypeError("unsupported TrialDev action")
        return finish_trialdev_run(self.budget_path, decision)
