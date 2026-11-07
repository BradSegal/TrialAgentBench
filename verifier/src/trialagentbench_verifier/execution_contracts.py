"""Typed boundary for executing model-generated analysis code."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trialagentbench_verifier.io import canonical_sha256

AnalysisLanguage = Literal["python", "r"]
AnalysisMode = Literal["python_r", "python", "r"]
# TrialEval reporting interface: "v1" is the 1.0 wording and helpers; "clarified" states the helper types,
# accepts a data frame or row mappings in report(), and grants repair turns after a failed submission replay.
ReportingInterfaceV1 = Literal["v1", "clarified"]


def analysis_languages(mode: AnalysisMode) -> tuple[AnalysisLanguage, ...]:
    """Return the interpreters offered by an analysis condition."""
    if mode == "python_r":
        return ("python", "r")
    if mode in ("python", "r"):
        return (mode,)
    raise ValueError("Unknown analysis language condition")


# Release turn limits: per TrialEval item, and per TrialDev submission step (a whole programme).
TRIALEVAL_MAXIMUM_TURNS = 50
TRIALDEV_MAXIMUM_TURNS_PER_STEP = 100
DEFAULT_COMPLETION_TOKENS = 32_768
# Executor containers one running task may hold at once: its analysis session and an independent replay.
EXECUTOR_SLOTS_PER_TASK = 2


class ReplayInputArtifactV1(BaseModel):
    """One statistical input actually staged for fresh execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_length: int = Field(ge=0)


class CodeExecutionLimitsV1(BaseModel):
    """Resource limits applied to one isolated analysis session."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timeout_seconds: float = Field(default=120.0, gt=0.0, le=3600.0)
    memory_mb: int = Field(default=4096, ge=256, le=65536)
    cpu_count: float = Field(default=2.0, gt=0.0, le=64.0)
    process_limit: int = Field(default=128, ge=16, le=4096)
    output_bytes: int = Field(default=65536, ge=1024, le=16 * 1024 * 1024)
    workspace_mb: int = Field(default=2048, ge=128, le=32768)
    workspace_file_count: int = Field(default=10000, ge=1, le=1_000_000)
    evidence_mb: int = Field(default=8192, ge=1, le=131072)
    evidence_file_count: int = Field(default=10000, ge=1, le=1_000_000)


class CodeExecutionResultV1(BaseModel):
    """Result of one code block executed inside an isolated session.

    CPU is kernel-accounted session usage since the previous host snapshot,
    including accounting overhead and work between calls. Memory and process
    peaks cover the container lifetime. Unavailable measurements remain absent
    when a container terminates before it can be observed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["success", "execution_error", "timeout", "session_terminated"]
    language: AnalysisLanguage = "python"
    output: str
    output_truncated: bool = False
    elapsed_seconds: float = Field(ge=0.0)
    cpu_seconds: float | None = Field(default=None, ge=0.0)
    session_peak_memory_bytes: int | None = Field(default=None, ge=0)
    session_peak_processes: int | None = Field(default=None, ge=0)


class CodeExecutionSession(Protocol):
    """Stateful isolated code-execution session used by benchmark agents."""

    def execute_result(self, code: str, *, language: AnalysisLanguage = "python") -> CodeExecutionResultV1:
        """Execute one code block while preserving session state."""

    def execute(self, code: str, *, language: AnalysisLanguage = "python") -> str:
        """Execute one code block and return bounded agent-facing text."""

    def snapshot_scratch(self) -> Path:
        """Persist the current isolated scratch tree for checkpoint custody."""

    def close(self) -> None:
        """Terminate the session and all descendant processes."""


class ExecutorPackageV1(BaseModel):
    """One installed distribution in the isolated analysis environment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(..., min_length=1)
    version: str = Field(..., min_length=1)


class ExecutorSupportFileV1(BaseModel):
    """One executor-owned file that participant code reads, with its bytes and mode."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(..., pattern=r"^/")
    mode: str = Field(..., pattern=r"^0[0-7]{3}$")
    sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")


class ExecutorEnvironmentV1(BaseModel):
    """Immutable identity and package inventory for model-code execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["ariadne.executor_environment/v1"] = "ariadne.executor_environment/v1"
    image_reference: str = Field(..., min_length=1)
    image_id: str = Field(..., pattern=r"^sha256:[0-9a-f]{64}$")
    operating_system: str = Field(default="unknown", min_length=1)
    architecture: str = Field(default="unknown", min_length=1)
    python_version: str = Field(..., min_length=1)
    packages: tuple[ExecutorPackageV1, ...] = Field(..., min_length=1)
    r_version: str | None = Field(default=None, min_length=1)
    r_packages: tuple[ExecutorPackageV1, ...] = ()
    system_packages: tuple[ExecutorPackageV1, ...] = ()
    # Omitted when empty so environments recorded before file attestation keep their hash.
    support_files: tuple[ExecutorSupportFileV1, ...] = Field(default=(), exclude_if=lambda value: not value)
    limits: CodeExecutionLimitsV1

    @model_validator(mode="after")
    def validate_package_inventory(self) -> ExecutorEnvironmentV1:
        """Require a complete, uniquely named package inventory."""

        normalized = [package.name.casefold() for package in self.packages]
        if normalized != sorted(normalized):
            raise ValueError("Executor packages must be sorted by normalized name.")
        if len(normalized) != len(set(normalized)):
            raise ValueError("Executor package names must be unique.")
        normalized_r = [package.name.casefold() for package in self.r_packages]
        if normalized_r != sorted(normalized_r):
            raise ValueError("Executor R packages must be sorted by normalized name.")
        if len(normalized_r) != len(set(normalized_r)):
            raise ValueError("Executor R package names must be unique.")
        if (self.r_version is None) != (not self.r_packages):
            raise ValueError("Executor R version and package inventory must be present together.")
        normalized_system = [package.name.casefold() for package in self.system_packages]
        if normalized_system != sorted(normalized_system):
            raise ValueError("Executor system packages must be sorted by normalized name.")
        if len(normalized_system) != len(set(normalized_system)):
            raise ValueError("Executor system package names must be unique.")
        return self

    def semantic_sha256(self) -> str:
        """Hash runtime semantics while excluding host-local Docker identity."""

        return canonical_sha256(self.model_dump(mode="json", exclude={"image_reference", "image_id"}))


RELEASE_EXECUTION_LIMITS_V1 = CodeExecutionLimitsV1(timeout_seconds=300.0)

TRIALEVAL_REPLAY_ABS_TOLERANCE_V1 = 1e-10
TRIALEVAL_REPLAY_REL_TOLERANCE_V1 = 1e-8
