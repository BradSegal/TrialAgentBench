"""Disclosed known-answer checks for an installed public verifier."""

from __future__ import annotations

from importlib.resources import as_file, files
from math import isclose
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from trialagentbench_verifier.contracts import CheckResultV1, PracticeVerificationReportV1
from trialagentbench_verifier.io import read_json_object, sha256_file
from trialagentbench_verifier.numerical_scoring import interval_score, numerical_loss
from trialagentbench_verifier.units import compatible_units


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class _IntervalCase(_Model):
    case_id: str
    operation: Literal["interval_score"]
    lower: float
    upper: float
    target: float
    decision_scale: float = Field(gt=0)
    expected: float = Field(ge=0)


class _UnitCase(_Model):
    case_id: str
    operation: Literal["unit"]
    reported: str
    reference: str
    expected: bool


class _LossCase(_Model):
    case_id: str
    operation: Literal["numerical_loss"]
    estimate: float
    lower: float
    upper: float
    target: float
    scale: float = Field(gt=0)
    confidence_level: float = Field(gt=0, lt=1)
    expected: float = Field(ge=0)


_Case = Annotated[_IntervalCase | _UnitCase | _LossCase, Field(discriminator="operation")]


class _Fixture(_Model):
    schema_id: Literal["trialagentbench.practice_cases/v3"]
    cases: tuple[_Case, ...] = Field(min_length=1)


def _observed(case: _Case) -> JsonValue:
    if isinstance(case, _LossCase):
        return numerical_loss(
            estimate=case.estimate,
            lower=case.lower,
            upper=case.upper,
            target=case.target,
            scale=case.scale,
            confidence_level=case.confidence_level,
        )
    if isinstance(case, _IntervalCase):
        return interval_score(case.lower, case.upper, case.target, 0.95) / case.decision_scale
    return compatible_units(case.reported, case.reference)


def verify_practice_suite() -> PracticeVerificationReportV1:
    """Run package-contained, disclosed-target checks without controlled truth."""

    resource = files("trialagentbench_verifier").joinpath("resources/practice_cases_v3.json")
    with as_file(resource) as path:
        fixture = _Fixture.model_validate(read_json_object(path))
        fixture_sha256 = sha256_file(path)
    checks = []
    for case in fixture.cases:
        observed = _observed(case)
        expected = case.expected
        matched = (
            isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12)
            if isinstance(observed, float) and isinstance(expected, float)
            else observed == expected
        )
        checks.append(
            CheckResultV1(
                check_id="practice.known_answer",
                scope_id=case.case_id,
                status="pass" if matched else "fail",
                observed=observed,
                expected=expected,
                artifact_paths=("resources/practice_cases_v3.json",),
                failure_code=None if matched else "known_answer_mismatch",
            )
        )
    failed = sum(check.status == "fail" for check in checks)
    return PracticeVerificationReportV1(
        status="pass" if failed == 0 else "fail",
        fixture_sha256=fixture_sha256,
        scheduled_check_count=len(checks),
        passed_check_count=len(checks) - failed,
        failed_check_count=failed,
        checks=tuple(checks),
    )


__all__ = ["verify_practice_suite"]
