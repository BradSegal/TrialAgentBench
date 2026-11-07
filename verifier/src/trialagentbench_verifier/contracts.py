"""Strict public wire contracts for release and score verification."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from trialagentbench_verifier.execution_contracts import ReplayInputArtifactV1
from trialagentbench_verifier.scorer import CurrentScorer

VerifierVersion = Literal["1.0.0", "1.0.1", "1.1.0"]
"""Package releases whose reports remain readable; every one shares these report schemas.

The version names the producing package only. Scores are current exactly when their
``scorer`` identity is, so retained execution proofs from any listed release still verify.
"""
VERIFIER_VERSION: Final = "1.1.0"
"""Version stamped on new reports: the installed package's."""


class _Model(BaseModel):
    """Strict immutable public contract."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class CheckResultV1(_Model):
    """One terminal verification result in the public report."""

    check_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    scope_id: str = Field(min_length=1)
    status: Literal["pass", "fail", "not_applicable"]
    observed: JsonValue
    expected: JsonValue
    tolerance_id: str | None = None
    artifact_paths: tuple[str, ...] = ()
    failure_code: str | None = None

    @model_validator(mode="after")
    def validate_failure(self) -> Self:
        """Require an explicit code exactly when a check fails."""

        if (self.status == "fail") != (self.failure_code is not None):
            raise ValueError("failure_code must be present exactly for failed checks")
        return self


class NormalizedSubmissionV1(_Model):
    """Participant result normalized by the released harness contract."""

    schema_id: Literal["trialagentbench.normalized_submission/v1"] = "trialagentbench.normalized_submission/v1"
    benchmark_stream: Literal["trialeval", "trialdev"]
    task_id: str = Field(min_length=1)
    submission_kind: Literal["analysis", "evidence", "portfolio"]
    submission_failure_code: Literal["missing_primary_submission", "missing_required_deliverable"] | None = None
    payload: dict[str, JsonValue]

    @model_validator(mode="after")
    def validate_kind(self) -> Self:
        """Bind submission kinds to the stream that defines them."""

        allowed = {
            "trialeval": {"analysis"},
            "trialdev": {"evidence", "portfolio"},
        }
        if self.submission_kind not in allowed[self.benchmark_stream]:
            raise ValueError("submission kind is incompatible with its benchmark stream")
        if self.submission_failure_code is not None and self.benchmark_stream != "trialeval":
            raise ValueError("submission failure codes apply only to TrialEval analysis submissions")
        return self


class QualificationReportIdentityV1(_Model):
    """Identity and terminal counts for one sealed qualification gate."""

    gate_id: Literal["G-EXT-01", "G-TAB-01", "X-19"]
    relative_path: str = Field(pattern=r"^evaluation/verification/reports/[A-Za-z0-9_.-]+\.json$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    accepted: bool
    scheduled_count: int = Field(ge=1)
    passed_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        """Require a complete terminal denominator and honest gate decision."""

        if self.passed_count + self.failed_count != self.scheduled_count:
            raise ValueError("qualification report counts do not reconcile")
        if self.accepted != (self.failed_count == 0):
            raise ValueError("qualification report decision differs from its failures")
        return self


class PortableComparisonV1(_Model):
    """One comparator difference that can be replayed from released numbers."""

    comparison_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]+$")
    scope_id: str = Field(min_length=1)
    quantity_id: str = Field(min_length=1)
    candidate_value: float
    reference_value: float
    absolute_difference: float = Field(ge=0)
    absolute_margin: float = Field(ge=0)
    matched: bool
    tolerance_id: str = Field(min_length=1)
    artifact_paths: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_difference(self) -> Self:
        """Bind the stored difference and disposition to its two values."""

        observed = abs(self.candidate_value - self.reference_value)
        epsilon = max(1e-15, 1e-12 * max(abs(self.candidate_value), abs(self.reference_value), 1.0))
        if abs(self.absolute_difference - observed) > epsilon:
            raise ValueError("portable comparison difference is not derived from its values")
        if self.matched != (observed <= self.absolute_margin):
            raise ValueError("portable comparison decision differs from its frozen margin")
        return self


class QualificationReportBundleV1(_Model):
    """Release-bound direct qualification identities and portable comparisons."""

    schema_id: Literal["trialagentbench.qualification_report_bundle/v1"] = (
        "trialagentbench.qualification_report_bundle/v1"
    )
    candidate_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    reports: tuple[QualificationReportIdentityV1, ...] = Field(min_length=3, max_length=3)
    comparisons: tuple[PortableComparisonV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_reports(self) -> Self:
        """Require one complete-trial gate and one coherent upstream source gate."""

        required = {"G-EXT-01", "G-TAB-01", "X-19"}
        if {report.gate_id for report in self.reports} != required:
            raise ValueError("qualification bundle must contain all required gate reports")
        by_gate = {report.gate_id: report for report in self.reports}
        if by_gate["G-TAB-01"].candidate_digest != self.candidate_digest:
            raise ValueError("complete-trial report identifies a different release candidate")
        if by_gate["G-EXT-01"].candidate_digest != by_gate["X-19"].candidate_digest:
            raise ValueError("external and standards reports identify different source candidates")
        return self


class ScoreVerificationBindingV1(_Model):
    """Immutable subject and execution identities for one score report."""

    task_id: str = Field(min_length=1)
    submission_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalized_submission_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    graded_item_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    execution_proof_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    program_artifacts: tuple[ReplayInputArtifactV1, ...] = ()
    program_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    program_result_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    normalized_program_result_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    replayed_result_sha256s: tuple[str, str] | None = None

    @model_validator(mode="after")
    def validate_execution_identity(self) -> Self:
        """Require the complete authenticated-execution identity or none of it."""

        if self.program_artifacts and self.execution_proof_sha256 is None:
            raise ValueError("source inventory requires authenticated execution")
        values = (
            self.execution_proof_sha256,
            self.program_sha256,
            self.program_result_sha256,
            self.replayed_result_sha256s,
        )
        if any(value is not None for value in values) and not all(value is not None for value in values):
            raise ValueError("authenticated score execution identity must be complete")
        if self.normalized_program_result_sha256 is not None and self.execution_proof_sha256 is None:
            raise ValueError("normalized program result requires authenticated execution")
        if self.replayed_result_sha256s is not None and any(
            re.fullmatch(r"[0-9a-f]{64}", value) is None for value in self.replayed_result_sha256s
        ):
            raise ValueError("replayed result identities must be lowercase SHA-256 values")
        return self


class VerificationReportV1(_Model):
    """Complete public release or score verification report."""

    scorer: CurrentScorer

    schema_id: Literal["trialagentbench.verification_report/v1"] = "trialagentbench.verification_report/v1"
    verifier_name: Literal["trialagentbench-verifier"] = "trialagentbench-verifier"
    verifier_version: VerifierVersion = VERIFIER_VERSION
    authorization_status: Literal["not_assessed"] = Field(
        default="not_assessed",
        description="Checks apply to supplied trust anchors; current publication/access authority must come from the release owner.",
    )
    input_kind: Literal["archive", "directory"]
    input_sha256: str | None = Field(default=None, exclude_if=lambda value: value is None, pattern=r"^[0-9a-f]{64}$")
    expected_input_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    release_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_stream: Literal["trialeval", "trialdev"]
    operation: Literal["release", "score", "analyse-trialdev"]
    score_binding: ScoreVerificationBindingV1 | None = None
    status: Literal["pass", "fail"]
    scheduled_check_count: int = Field(ge=1)
    passed_check_count: int = Field(ge=0)
    failed_check_count: int = Field(ge=0)
    checks: tuple[CheckResultV1, ...] = Field(min_length=1)
    qualification_report_digests: tuple[str, ...]
    development: bool = Field(default=False, exclude_if=lambda value: not value)
    standard_component_inventory_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        """Derive all counts and the terminal decision from emitted checks."""

        if self.development and self.qualification_report_digests:
            raise ValueError("Development verification cannot assert qualification report identities")
        passed = sum(check.status in {"pass", "not_applicable"} for check in self.checks)
        failed = sum(check.status == "fail" for check in self.checks)
        if self.scheduled_check_count != len(self.checks):
            raise ValueError("scheduled check count differs from emitted checks")
        if (self.passed_check_count, self.failed_check_count) != (passed, failed):
            raise ValueError("verification counts are not derived from checks")
        if self.status != ("pass" if failed == 0 else "fail"):
            raise ValueError("verification status is not derived from failed checks")
        if self.expected_input_sha256 is not None and self.expected_input_sha256 != self.input_sha256:
            raise ValueError("trusted expected input identity differs from the verified input")
        if (self.operation == "score") != (self.score_binding is not None):
            raise ValueError("score_binding must be present exactly for score verification")
        if self.score_binding is not None and (
            any(check.scope_id != self.score_binding.task_id for check in self.checks)
            or len({check.check_id for check in self.checks}) != len(self.checks)
        ):
            raise ValueError("score checks must be unique and scoped to the bound task")
        return self


class VerificationFailureReportV1(_Model):
    """Terminal public report when verification inputs cannot be evaluated."""

    schema_id: Literal["trialagentbench.verification_failure/v1"] = "trialagentbench.verification_failure/v1"
    verifier_name: Literal["trialagentbench-verifier"] = "trialagentbench-verifier"
    verifier_version: VerifierVersion = VERIFIER_VERSION
    operation: Literal[
        "release",
        "score",
        "execute",
        "toolkit",
        "release-set",
        "reproducibility",
        "analyse-trialeval",
        "analyse-trialdev",
        "attest-executor",
    ]
    status: Literal["fail"] = "fail"
    failure_code: Literal["invalid_input", "executor_unavailable"] = "invalid_input"
    detail: str = Field(min_length=1)


class DiagnosticArtifactExecutionV1(_Model):
    """One twice-replayed diagnostic artifact and its scientific verification."""

    asset_id: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    submitted_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    replayed_sha256s: tuple[str, str]
    verification_status: Literal[
        "verified",
        "missing_artifact",
        "invalid_artifact",
        "entity_mismatch",
        "value_mismatch",
    ]

    @model_validator(mode="after")
    def validate_artifact(self) -> Self:
        """Require a portable path and exact replay bytes for verified diagnostics."""

        path = PurePosixPath(self.relative_path)
        if (
            path.is_absolute()
            or path.parts[:1] != ("scratch",)
            or any(part in {"", ".", ".."} for part in path.parts)
            or "\\" in self.relative_path
            or path.as_posix() != self.relative_path
        ):
            raise ValueError("diagnostic proof path must remain below scratch")
        if self.verification_status == "verified" and any(
            digest != self.submitted_sha256 for digest in self.replayed_sha256s
        ):
            raise ValueError("verified diagnostic artifact differs across fresh replays")
        return self


class ExecutionProofV1(_Model):
    """Fresh offline execution proof, separate from release and score checks."""

    schema_id: Literal["trialagentbench.execution_proof/v1"] = "trialagentbench.execution_proof/v1"
    verifier_name: Literal["trialagentbench-verifier"] = "trialagentbench-verifier"
    verifier_version: VerifierVersion = VERIFIER_VERSION
    operation: Literal["execute"] = "execute"
    status: Literal["pass", "fail"]
    input_kind: Literal["archive", "directory"]
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    release_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_stream: Literal["trialeval", "trialdev"]
    task_id: str = Field(min_length=1)
    language: Literal["python", "r"] | None = None
    program_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    program_artifacts: tuple[ReplayInputArtifactV1, ...] = ()
    result_format: Literal["json", "csv"] = "json"
    target_ids: tuple[str, ...] = ()
    evidence_artifacts: tuple[ReplayInputArtifactV1, ...] = ()
    program_result_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    score_projection_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    replayed_result_sha256s: tuple[str, ...] = ()
    replay_status: Literal[
        "reproduced",
        "within_tolerance",
        "missing_program",
        "ambiguous_program",
        "invalid_program",
        "missing_result",
        "invalid_result",
        "execution_error",
        "result_mismatch",
        "artifact_mismatch",
    ]
    execution_status: Literal["success", "execution_error", "timeout", "session_terminated"] | None = None
    comparison_classification: Literal["exact", "within_tolerance", "irreproducible"] | None = None
    attempt_count: int = Field(ge=0, le=2)
    elapsed_seconds: float = Field(ge=0)
    cpu_seconds: float | None = Field(default=None, ge=0)
    session_peak_memory_bytes: int | None = Field(default=None, ge=0)
    session_peak_processes: int | None = Field(default=None, ge=0)
    executor_image_reference: str = Field(min_length=1)
    executor_image_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    expected_executor_image_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    executor_environment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_executor_environment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    diagnostic_artifacts: tuple[DiagnosticArtifactExecutionV1, ...] = ()
    analysis_data_filename: Literal["analysis_data.csv", "analysis_data.parquet"] | None = None
    analysis_data_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    analysis_data_reproduced: bool | None = None
    mismatch_paths: tuple[str, ...] = ()
    failure_code: (
        Literal[
            "missing_program",
            "ambiguous_program",
            "invalid_program",
            "missing_result",
            "invalid_result",
            "execution_error",
            "result_mismatch",
            "artifact_mismatch",
        ]
        | None
    ) = None

    @model_validator(mode="after")
    def validate_execution(self) -> Self:
        """Bind success to a completed two-attempt replay and authenticated input."""

        if self.result_format == "csv" and (
            not self.target_ids
            or len(set(self.target_ids)) != len(self.target_ids)
            or any(not target for target in self.target_ids)
        ):
            raise ValueError("CSV execution proof requires unique declared targets")
        reproduced = self.replay_status in {"reproduced", "within_tolerance"}
        if self.expected_input_sha256 != self.input_sha256:
            raise ValueError("execution input differs from its external trust anchor")
        if self.executor_image_id != self.expected_executor_image_id:
            raise ValueError("executor image differs from its external trust anchor")
        if self.executor_environment_sha256 != self.expected_executor_environment_sha256:
            raise ValueError("executor environment differs from its external trust anchor")
        if (self.status == "pass") != reproduced:
            raise ValueError("execution proof status differs from fresh replay")
        if (self.failure_code is None) != reproduced:
            raise ValueError("failure_code must be absent exactly for reproduced programs")
        if reproduced:
            if self.execution_status != "success" or self.attempt_count != 2:
                raise ValueError("successful execution proof requires two completed attempts")
            if len(self.replayed_result_sha256s) != 2:
                raise ValueError("successful execution proof requires two replayed result identities")
            if (
                self.program_result_sha256 is None
                or self.program_sha256 is None
                or self.score_projection_sha256 is None
            ):
                raise ValueError("successful execution proof requires program and result identities")
        if self.analysis_data_reproduced is True and (
            self.analysis_data_filename is None or self.analysis_data_sha256 is None or not reproduced
        ):
            raise ValueError("reproduced analysis data requires bound bytes and successful fresh replay")
        if (self.analysis_data_filename is None) != (self.analysis_data_sha256 is None):
            raise ValueError("analysis data filename and digest must be supplied together")
        artifact_keys = tuple((row.asset_id, row.relative_path) for row in self.diagnostic_artifacts)
        if artifact_keys != tuple(sorted(set(artifact_keys))):
            raise ValueError("diagnostic artifact proofs must be sorted and unique")
        if self.benchmark_stream != "trialdev" and self.diagnostic_artifacts:
            raise ValueError("TrialEval execution cannot verify TrialDev diagnostics")
        return self

    @property
    def verified_diagnostic_asset_ids(self) -> tuple[str, ...]:
        """Derive candidate support only from verified replay-bound artifacts."""

        return tuple(
            sorted({row.asset_id for row in self.diagnostic_artifacts if row.verification_status == "verified"})
        )


class PracticeVerificationReportV1(_Model):
    """Result of the installed disclosed-target practice suite."""

    schema_id: Literal["trialagentbench.practice_verification/v1"] = "trialagentbench.practice_verification/v1"
    verifier_name: Literal["trialagentbench-verifier"] = "trialagentbench-verifier"
    verifier_version: VerifierVersion = VERIFIER_VERSION
    operation: Literal["practice"] = "practice"
    status: Literal["pass", "fail"]
    fixture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scheduled_check_count: int = Field(ge=1)
    passed_check_count: int = Field(ge=0)
    failed_check_count: int = Field(ge=0)
    checks: tuple[CheckResultV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        """Derive the terminal result from the disclosed checks."""

        failed = sum(check.status == "fail" for check in self.checks)
        if self.scheduled_check_count != len(self.checks):
            raise ValueError("practice scheduled count differs from its checks")
        if (self.passed_check_count, self.failed_check_count) != (len(self.checks) - failed, failed):
            raise ValueError("practice counts differ from its checks")
        if self.status != ("pass" if failed == 0 else "fail"):
            raise ValueError("practice status differs from its failed checks")
        return self


__all__ = [
    "CheckResultV1",
    "DiagnosticArtifactExecutionV1",
    "NormalizedSubmissionV1",
    "ExecutionProofV1",
    "PracticeVerificationReportV1",
    "VerificationFailureReportV1",
    "VerificationReportV1",
]
