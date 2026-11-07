"""Participant-safe TrialEval diagnostic dictionary contract."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench.contracts.scoring.diagnostic_registry import (
    DiagnosticKeySpecV1,
    load_diagnostic_registry_v1,
)
from trialagentbench.io.checksums import canonical_payload_sha256


class TrialEvalParticipantDiagnosticV1(BaseModel):
    """One analyst-visible diagnostic operation and reporting contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    public_key: str = Field(min_length=1)
    diagnostic_scale: str = Field(min_length=1)
    decision_boundary: str = Field(min_length=1)
    interpretation: str = Field(min_length=1)
    participant_operation: str = Field(min_length=1)
    primary_credit_policy: Literal["always", "method_dependent", "design_modifier", "limitation"]
    metric_units: dict[str, str] = Field(default_factory=dict)
    required_metric_units: dict[str, str] = Field(default_factory=dict)


class TrialEvalParticipantDiagnosticDictionaryV1(BaseModel):
    """Task-general operations and measures for TrialEval diagnostics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["ariadne.trialeval.participant_diagnostic_dictionary/v1"] = (
        "ariadne.trialeval.participant_diagnostic_dictionary/v1"
    )
    purpose: Literal["define_diagnostic_operations_without_item_eligibility_or_results"] = (
        "define_diagnostic_operations_without_item_eligibility_or_results"
    )
    diagnostics: dict[str, TrialEvalParticipantDiagnosticV1] = Field(min_length=1)
    checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _canonical_dictionary(self) -> TrialEvalParticipantDiagnosticDictionaryV1:
        diagnostics = dict(sorted(self.diagnostics.items()))
        if any(identifier != definition.public_key for identifier, definition in diagnostics.items()):
            raise ValueError("participant diagnostic dictionary keys must match their public diagnostic IDs")
        object.__setattr__(self, "diagnostics", diagnostics)
        digest = canonical_payload_sha256(self.model_dump(mode="json", exclude={"checksum"}))
        if self.checksum is not None and self.checksum != digest:
            raise ValueError("participant diagnostic dictionary checksum does not match its payload")
        object.__setattr__(self, "checksum", digest)
        return self


def _participant_decision_boundary(definition: DiagnosticKeySpecV1) -> str:
    """Project a method boundary without evaluator severity calibration."""

    method_change = definition.primary_method_change_threshold
    thresholds = definition.severity_thresholds
    if method_change is not None:
        if thresholds is None:
            raise ValueError("a numeric method-change boundary requires a diagnostic metric")
        value = (
            f"log({math.exp(method_change):g}) = {method_change:g}"
            if thresholds.metric_unit.startswith("log_")
            else f"{method_change:g}"
        )
        return (
            "Prespecified method-change boundary: "
            f"{value} {thresholds.metric_unit}; report the continuous diagnostic and its uncertainty."
        )
    if thresholds is not None:
        return (
            "Report the continuous diagnostic and its 95% uncertainty. Use its magnitude, support, and "
            "direction to qualify the analysis assumptions; no categorical severity label is part of the "
            "participant analysis."
        )
    return definition.threshold


PARTICIPANT_PROSE_FIELDS: frozenset[str] = frozenset({"participant_operation"})
"""Dictionary fields owned by release construction rather than by the evaluator registry."""


def _installed_diagnostic_semantics() -> dict[str, dict[str, object]]:
    """Project every evaluator-owned dictionary field from the installed registry."""

    registry = load_diagnostic_registry_v1()
    return {
        definition.public_key: {
            "public_key": definition.public_key,
            "diagnostic_scale": definition.diagnostic_scale,
            "decision_boundary": _participant_decision_boundary(definition),
            "interpretation": definition.interpretation,
            "primary_credit_policy": definition.primary_credit_policy,
            "metric_units": dict(sorted(definition.metric_units.items())),
            "required_metric_units": (
                {
                    metric_id: definition.severity_thresholds.metric_unit
                    for metric_id in sorted(
                        {
                            definition.severity_thresholds.metric_name,
                            *definition.severity_thresholds.decision_metric_names.values(),
                        }
                    )
                }
                if definition.severity_thresholds is not None
                else {}
            ),
        }
        for definition in registry.diagnostic_keys.values()
    }


def participant_diagnostic_dictionary_v1(
    participant_operations: Mapping[str, str],
) -> TrialEvalParticipantDiagnosticDictionaryV1:
    """Combine installed evaluator semantics with construction-owned participant operations.

    Parameters
    ----------
    participant_operations
        Participant operation text for exactly every registered public diagnostic key.

    Returns
    -------
    TrialEvalParticipantDiagnosticDictionaryV1
        The participant dictionary without evaluator severity calibration.

    Raises
    ------
    ValueError
        If the operations do not cover exactly the registered diagnostics.
    """

    semantics = _installed_diagnostic_semantics()
    if set(participant_operations) != set(semantics):
        raise ValueError(
            "participant diagnostic operations must cover exactly the registered diagnostics: "
            f"missing={sorted(set(semantics) - set(participant_operations))!r} "
            f"unknown={sorted(set(participant_operations) - set(semantics))!r}"
        )
    return TrialEvalParticipantDiagnosticDictionaryV1(
        diagnostics={
            public_key: TrialEvalParticipantDiagnosticV1.model_validate(
                {**fields, "participant_operation": participant_operations[public_key]}
            )
            for public_key, fields in semantics.items()
        }
    )


def require_installed_diagnostic_semantics_v1(dictionary: TrialEvalParticipantDiagnosticDictionaryV1) -> None:
    """Refuse a shipped dictionary whose evaluator semantics differ from the installed registry.

    Participant prose (``PARTICIPANT_PROSE_FIELDS``) belongs to release construction and is not compared, so a
    wording change on either side never invalidates an existing release.

    Raises
    ------
    ValueError
        If any diagnostic key or evaluator-owned field differs from the installed registry.
    """

    shipped = {
        public_key: diagnostic.model_dump(mode="json", exclude=set(PARTICIPANT_PROSE_FIELDS))
        for public_key, diagnostic in dictionary.diagnostics.items()
    }
    if shipped != _installed_diagnostic_semantics():
        raise ValueError("TrialEval diagnostic dictionary differs from the installed canonical registry.")


__all__ = [
    "PARTICIPANT_PROSE_FIELDS",
    "TrialEvalParticipantDiagnosticV1",
    "TrialEvalParticipantDiagnosticDictionaryV1",
    "participant_diagnostic_dictionary_v1",
    "require_installed_diagnostic_semantics_v1",
]
