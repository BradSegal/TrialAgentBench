"""Strict public schema for simulation-calibrated TrialEval report likelihoods."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_verifier.decision_consequence import DiscreteEffectPriorV1, ReportCalibrationV1
from trialagentbench_verifier.layer_priors import LAYER_PRIOR_SHA256

LAYER_CALIBRATION_RESOURCE = "resources/scoring/trialeval-report-calibration.json"
LAYER_CALIBRATION_SHA256 = "4ad030d6137846c7b13e2f052783447b3f087fcd7010ca9dcf02d81b3c738910"
"""Exact bytes of the public, training-only calibration and common regulariser."""


class QuestionReportCalibrationV1(BaseModel):
    """One public estimand's density, shared by every cell asking that question."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    estimand_id: str = Field(min_length=1)
    density: ReportCalibrationV1


class LayerCalibrationFileV1(BaseModel):
    """Question-indexed calibration parameters without private construction selectors."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_id: Literal["trialagentbench.trialeval_report_calibration/v1"]
    prior_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    coordinates: tuple[Literal["comparison_effect"], Literal["reported_estimate"], Literal["log_interval_se"]]
    method: Literal["conditional_gaussian_mixture_divided_by_effect_marginal"]
    likelihood_ratio_weight: float = Field(default=1.0, ge=0, le=1, exclude_if=lambda value: value == 1)
    """One training-selected log-density regulariser, identical for every question."""
    calibrations: tuple[QuestionReportCalibrationV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_questions(self) -> Self:
        """Reject ambiguous question lookup."""
        if self.prior_sha256 != LAYER_PRIOR_SHA256:
            raise ValueError("report calibration binds a different atom prior")
        if len({entry.estimand_id for entry in self.calibrations}) != len(self.calibrations):
            raise ValueError("report calibration repeats a public estimand")
        return self

    def apply(self, estimand_id: str, prior: DiscreteEffectPriorV1) -> DiscreteEffectPriorV1:
        """Attach a question's density and the single common regulariser to its atom prior.

        Parameters
        ----------
        estimand_id
            Public primary estimand identifier used to select the density.
        prior
            Atom prior whose masses and physical effect support are preserved.

        Returns
        -------
        DiscreteEffectPriorV1
            Validated prior with the matching density and the common file-level weight.

        Raises
        ------
        ValueError
            If the question is absent or its fit probabilities do not align with the atoms.

        Examples
        --------
        Attach an already integrity-checked calibration to the matching public prior.

        >>> calibrated = calibration.apply(estimand_id, prior)  # doctest: +SKIP
        >>> calibrated.report_likelihood_weight == calibration.likelihood_ratio_weight  # doctest: +SKIP
        True
        """
        for entry in self.calibrations:
            if entry.estimand_id == estimand_id:
                return DiscreteEffectPriorV1.model_validate(
                    prior.model_dump()
                    | {"report_calibration": entry.density, "report_likelihood_weight": self.likelihood_ratio_weight}
                )
        raise ValueError(f"TrialEval report calibration has no density for estimand {estimand_id!r}")


def _checked(content: bytes, expected_sha256: str) -> LayerCalibrationFileV1:
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("TrialEval report-calibration SHA-256 mismatch")
    return LayerCalibrationFileV1.model_validate_json(content)


def load_layer_calibration_file(path: Path, *, expected_sha256: str) -> LayerCalibrationFileV1:
    """SHA-check and validate a public candidate before attaching it to layer priors."""
    return _checked(path.read_bytes(), expected_sha256)


@lru_cache(maxsize=1)
def load_public_layer_calibration() -> LayerCalibrationFileV1:
    """Return the packaged calibration, failing if its bytes or strict schema differ.

    Returns
    -------
    LayerCalibrationFileV1
        Question-indexed report densities with the single common training-selected weight.

    Raises
    ------
    FileNotFoundError
        If the package's required public calibration is absent.
    ValueError
        If its SHA-256 or schema is invalid.
    """
    return _checked(
        files("trialagentbench_verifier").joinpath(LAYER_CALIBRATION_RESOURCE).read_bytes(), LAYER_CALIBRATION_SHA256
    )


__all__ = [
    "LAYER_CALIBRATION_RESOURCE",
    "LAYER_CALIBRATION_SHA256",
    "LayerCalibrationFileV1",
    "QuestionReportCalibrationV1",
    "load_layer_calibration_file",
    "load_public_layer_calibration",
]
