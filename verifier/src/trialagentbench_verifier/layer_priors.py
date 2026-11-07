"""Load the integrity-bound public TrialEval layer prior in comparison-effect units.

The file defines the layer prior (discretised push-forward of the disclosed
mean-z prior). Its numerical fidelity is measured separately from scoring.
"""

from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_verifier.decision_consequence import DiscreteEffectPriorV1

LAYER_PRIOR_RESOURCE = "resources/scoring/trialeval-priors.json"
LAYER_PRIOR_SHA256 = "777b5419368f9b096b81944d6ab8e10f91ba0f337b3c009559c8095695038cbd"
"""Exact scoring-input SHA-256, included in the verifier source/scorer identity."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class PriorSourceRecordV1(_Model):
    """Integrity of a source record without its private path or construction labels."""

    role: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class MeanZPriorV1(_Model):
    """Generation-precision, globally truncated zero-centred normal-mixture source prior."""

    weights: tuple[float, ...] = Field(min_length=1)
    standard_deviations: tuple[float, ...] = Field(min_length=1)
    lower: float
    upper: float

    @model_validator(mode="after")
    def valid_mixture(self) -> Self:
        """Require a probability mixture and finite support around zero."""
        if (
            len(self.weights) != len(self.standard_deviations)
            or any(value <= 0 for value in (*self.weights, *self.standard_deviations))
            or not math.isclose(sum(self.weights), 1, abs_tol=1e-12, rel_tol=0)
            or not self.lower < 0 < self.upper
        ):
            raise ValueError("source mean-z prior requires positive mixture components and ordered truncation")
        return self


class MeanZMultiplierMapV1(_Model):
    """Stored piecewise-linear inverse map, with both coordinates named explicitly."""

    mean_z: tuple[float, ...] = Field(min_length=2)
    multiplier: tuple[float, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def aligned_monotone_coordinates(self) -> Self:
        """Reject mismatched coordinates or an unordered inverse map."""
        if len(self.mean_z) != len(self.multiplier) or any(
            right <= left
            for values in (self.mean_z, self.multiplier)
            for left, right in zip(values, values[1:], strict=False)
        ):
            raise ValueError("mean-z inverse map requires matching strictly increasing coordinates")
        return self


class PopulationEffectCurveV1(_Model):
    """Fitted physical curve at the scoring nodes and endpoints, with pointwise MC errors."""

    multiplier: tuple[float, ...] = Field(min_length=4)
    effect: tuple[float, ...] = Field(min_length=4)
    pointwise_mc_se: tuple[float, ...] = Field(min_length=4)
    independent_world_clusters: int = Field(ge=0)
    independent_control_calibration_worlds: int = Field(ge=0)
    representation: str = Field(min_length=1)
    population_curve_known_analytic: bool

    @model_validator(mode="after")
    def aligned_curve(self) -> Self:
        """Require aligned physical values and nonnegative pointwise errors."""
        if (
            len({len(self.multiplier), len(self.effect), len(self.pointwise_mc_se)}) != 1
            or any(right <= left for left, right in zip(self.multiplier, self.multiplier[1:], strict=False))
            or any(error < 0 for error in self.pointwise_mc_se)
        ):
            raise ValueError("population curve requires ordered, aligned coordinates and nonnegative MC SEs")
        return self


class QuestionLayerPriorV1(_Model):
    """One public question's layer prior, physical transformation and provenance."""

    estimand_id: str = Field(min_length=1)
    source_mean_z_prior: MeanZPriorV1
    mean_z_to_multiplier: MeanZMultiplierMapV1
    population_effect_curve: PopulationEffectCurveV1
    discrete_prior: DiscreteEffectPriorV1
    source_records: tuple[PriorSourceRecordV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def consistent_push_forward(self) -> Self:
        """Bind the disclosed coordinates and curve values to the actual scoring atoms."""
        mapping = self.mean_z_to_multiplier
        curve = self.population_effect_curve
        if (
            (mapping.mean_z[0], mapping.mean_z[-1]) != (self.source_mean_z_prior.lower, self.source_mean_z_prior.upper)
            or (curve.multiplier[0], curve.multiplier[-1]) != (mapping.multiplier[0], mapping.multiplier[-1])
            or curve.effect[1:-1] != self.discrete_prior.support
        ):
            raise ValueError("layer-prior source coordinates, curve and scoring atoms disagree")
        return self


class LayerPriorFileV1(_Model):
    """Public scoring file whose recorded bytes define every TrialEval layer prior."""

    schema_id: Literal["trialagentbench.trialeval_layer_priors/v1"]
    description: str = Field(min_length=1)
    coordinate: Literal["comparison_effect"]
    precision: str = Field(min_length=1)
    quadrature: str = Field(min_length=1)
    source_records: tuple[PriorSourceRecordV1, ...] = Field(min_length=1)
    priors: tuple[QuestionLayerPriorV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_questions(self) -> Self:
        """Reject ambiguous lookup by the task's public primary estimand identifier."""
        if len({entry.estimand_id for entry in self.priors}) != len(self.priors):
            raise ValueError("layer-prior file repeats a public question")
        return self


def _checked(content: bytes, expected_sha256: str) -> LayerPriorFileV1:
    observed = hashlib.sha256(content).hexdigest()
    if observed != expected_sha256:
        raise ValueError(f"TrialEval layer-prior SHA-256 mismatch: expected {expected_sha256}, observed {observed}")
    return LayerPriorFileV1.model_validate_json(content)


def load_layer_prior_file(path: Path, *, expected_sha256: str) -> LayerPriorFileV1:
    """Verify exact scoring-file bytes and parse the strict public schema.

    Parameters
    ----------
    path
        Public scoring file to read.
    expected_sha256
        SHA-256 recorded by the scoring-input owner.

    Returns
    -------
    LayerPriorFileV1
        Immutable validated scoring priors and provenance.

    Raises
    ------
    FileNotFoundError
        If the file is absent.
    ValueError
        If its integrity or schema is invalid.
    """
    return _checked(path.read_bytes(), expected_sha256)


@lru_cache(maxsize=1)
def _public_file() -> LayerPriorFileV1:
    return _checked(files("trialagentbench_verifier").joinpath(LAYER_PRIOR_RESOURCE).read_bytes(), LAYER_PRIOR_SHA256)


@lru_cache(maxsize=1)
def _scoring_priors() -> tuple[tuple[str, DiscreteEffectPriorV1], ...]:
    from trialagentbench_verifier.layer_calibration import load_public_layer_calibration

    public = _public_file()
    calibration = load_public_layer_calibration()
    if {entry.estimand_id for entry in public.priors} != {entry.estimand_id for entry in calibration.calibrations}:
        raise ValueError("TrialEval report calibration must cover exactly the public prior questions")
    return tuple(
        (entry.estimand_id, calibration.apply(entry.estimand_id, entry.discrete_prior)) for entry in public.priors
    )


def load_layer_prior(estimand_id: str) -> DiscreteEffectPriorV1:
    """Return the SHA-checked layer prior for a public task question, failing without a match.

    Parameters
    ----------
    estimand_id
        Public primary estimand identifier (``task.json``'s ``target_ids[0]``).

    Returns
    -------
    DiscreteEffectPriorV1
        Prior in comparison-effect units, including the stored truncation's push-forward
        and the SHA-checked simulation likelihood with its common training-selected weight.

    Raises
    ------
    FileNotFoundError
        If the package's required public scoring file is missing.
    ValueError
        If its SHA/schema is invalid or the question has no registered prior.
    """
    for question, prior in _scoring_priors():
        if question == estimand_id:
            return prior
    raise ValueError(f"TrialEval public scoring file has no layer prior for estimand {estimand_id!r}")


__all__ = [
    "LAYER_PRIOR_RESOURCE",
    "LAYER_PRIOR_SHA256",
    "LayerPriorFileV1",
    "load_layer_prior",
    "load_layer_prior_file",
]
