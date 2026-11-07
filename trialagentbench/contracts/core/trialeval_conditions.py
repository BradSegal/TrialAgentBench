"""Participant-visible TrialEval procedure-assistance conditions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from trialagentbench.contracts.core.config import ProcedureAssistanceV1

_TRIALEVAL_PROCEDURE_ASSISTANCE_LEVELS: tuple[ProcedureAssistanceV1, ...] = (
    "output_contract_only",
    "unordered_checklist",
)
_TRIALDEV_PROCEDURE_ASSISTANCE_LEVELS: tuple[ProcedureAssistanceV1, ...] = (
    "output_contract_only",
    "unordered_checklist",
)


class ProcedureAssistanceExposureV1(BaseModel):
    """Participant-visible components of one procedure-assistance intervention."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    suite: Literal["trialeval", "trialdev"]
    procedure_assistance: ProcedureAssistanceV1
    unordered_completeness_instruction: bool


def procedure_assistance_exposure_v1(
    *,
    suite: Literal["trialeval", "trialdev"],
    procedure_assistance: ProcedureAssistanceV1,
) -> ProcedureAssistanceExposureV1:
    """Return the exact procedure-assistance components exposed to a participant."""

    allowed = _TRIALEVAL_PROCEDURE_ASSISTANCE_LEVELS if suite == "trialeval" else _TRIALDEV_PROCEDURE_ASSISTANCE_LEVELS
    if procedure_assistance not in allowed:
        raise ValueError(f"{suite} does not define procedure assistance {procedure_assistance!r}.")
    return ProcedureAssistanceExposureV1(
        suite=suite,
        procedure_assistance=procedure_assistance,
        unordered_completeness_instruction=procedure_assistance == "unordered_checklist",
    )


__all__ = [
    "ProcedureAssistanceExposureV1",
    "ProcedureAssistanceV1",
    "procedure_assistance_exposure_v1",
]
