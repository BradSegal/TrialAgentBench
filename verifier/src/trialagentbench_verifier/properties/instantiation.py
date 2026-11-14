"""Data-only construction laws and exact released estimand attributes."""

from __future__ import annotations

import math
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ClusterSizeParameters(BaseModel):
    """Declared mean size and variation before facility recruitment."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    cluster_size: int = Field(gt=0)
    coefficient_of_variation: float = Field(ge=0, allow_inf_nan=False)


class ClusterCountForm(BaseModel):
    """Executable meaning of the exported deterministic count formula."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    zero_variation: Literal["ceil(n / cluster_size)"]
    positive_variation: Literal["max(minimum_clusters, min(n // minimum_members, round(n / cluster_size)))"]
    rounding: Literal["nearest_ties_to_even"]
    minimum_clusters: int = Field(gt=0)
    minimum_members: int = Field(gt=0)
    minimum_variable_cohort: int = Field(gt=0)


class ClusterSizeLaw(BaseModel):
    """Exported count law, evaluated independently of construction code."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    functional_form: ClusterCountForm
    series: dict[str, ClusterSizeParameters]

    def count(self, series_id: str, participants: int) -> int:
        """Return the declared cluster count for a positive completed cohort."""
        if participants < 1:
            raise ValueError("Cluster count requires a positive completed cohort")
        parameters = self.series[series_id]
        form = self.functional_form
        if parameters.coefficient_of_variation == 0:
            return math.ceil(participants / parameters.cluster_size)
        if participants < form.minimum_variable_cohort:
            raise ValueError("Completed cohort is too small for the variable cluster-size law")
        quotient, remainder = divmod(participants, parameters.cluster_size)
        rounded = quotient + int(
            2 * remainder > parameters.cluster_size or (2 * remainder == parameters.cluster_size and quotient % 2 == 1)
        )
        return max(form.minimum_clusters, min(participants // form.minimum_members, rounded))


class EstimandInstantiation(BaseModel):
    """Five instantiated attributes, excluding shared protocol boilerplate."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    estimand_id: str = Field(min_length=1)
    population: str = Field(min_length=1)
    treatment_conditions: str = Field(min_length=1)
    variable: str = Field(min_length=1)
    intercurrent_events: str = Field(min_length=1)
    summary_measure: str = Field(min_length=1)


def estimand_attributes(protocol: str) -> dict[str, str]:
    """Parse one primary instantiation; headings and boilerplate prove nothing."""
    lines = [line for line in protocol.splitlines() if line.startswith("Primary estimand attributes")]
    if len(lines) != 1:
        return {}
    match = re.fullmatch(
        r"Primary estimand attributes \(ICH E9\(R1\)\) \[`(?P<estimand_id>[^`]+)`\]: "
        r"Population: (?P<population>.+?)\. Treatment conditions: (?P<treatment_conditions>.+?)\. "
        r"Variable: (?P<variable>.+?)\. Intercurrent events and strategies: (?P<intercurrent_events>.+?)\. "
        r"(?:Treatment changes .*?\. Study withdrawal .*?\. )?"
        r"Population-level summary measure: (?P<summary_measure>.+)",
        lines[0],
    )
    return match.groupdict() if match else {}


def completed_cohort_size(protocol: str) -> int | None:
    """Bind facility n to the released completed-cohort protocol's sample size."""
    counts = re.findall(r"^Planned sample size: ([\d,]+) participants\.", protocol, re.MULTILINE)
    if len(counts) != 1:
        return None
    count = int(counts[0].replace(",", ""))
    return count if count > 0 else None
