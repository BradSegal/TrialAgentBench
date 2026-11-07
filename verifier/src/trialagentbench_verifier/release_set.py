"""Installed verification for separated TrialAgentBench release sets."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import tomllib
import unicodedata
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Literal, Self, cast

from packaging.utils import canonicalize_name, parse_sdist_filename, parse_wheel_filename
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from trialagentbench_verifier.assets import ReleaseAssets, read_asset_catalog
from trialagentbench_verifier.campaign import (
    TrialEvalCampaignAnalysisV1,
    TrialEvalCampaignPlanV1,
    _ResultExportManifestV1,
    analyse_trial_eval_campaign_v1,
)
from trialagentbench_verifier.contracts import VerificationReportV1
from trialagentbench_verifier.execution_contracts import (
    RELEASE_EXECUTION_LIMITS_V1,
    CodeExecutionLimitsV1,
    ExecutorEnvironmentV1,
)
from trialagentbench_verifier.external_qualification import EXTERNAL_QUALIFICATION_NAME
from trialagentbench_verifier.io import (
    MAX_JSON_BYTES,
    MAX_RELEASE_ARCHIVE_BYTES,
    MAX_RELEASE_ARCHIVE_MEMBERS,
    MAX_RELEASE_MEMBER_BYTES,
    MAX_RELEASE_UNCOMPRESSED_BYTES,
    VerificationInputError,
    canonical_sha256,
    release_archive,
)
from trialagentbench_verifier.release import verify_release
from trialagentbench_verifier.run_contracts import (
    DecodingConfigV1,
    ExperimentConditionV1,
    ProviderRetryPolicyV1,
    RoutingConfigV1,
)

TOOLKIT_SCHEMA_VERSION: Literal["trialagentbench.public_toolkit/v1"] = "trialagentbench.public_toolkit/v1"
RELEASE_SET_SCHEMA_VERSION: Literal["trialagentbench.release_set/v2"] = "trialagentbench.release_set/v2"
TOOLKIT_ARCHIVE_ROOT = "trialagentbench-toolkit-v1"
TOOLKIT_ARCHIVE_NAME = "trialagentbench-toolkit-v1.zip"
RELEASE_SET_MANIFEST_NAME = "release-set-manifest.json"
RELEASE_SET_CHECKSUM_NAME = "SHA256SUMS"
RELEASE_SET_MANIFEST_HASH_NAME = f"{RELEASE_SET_MANIFEST_NAME}.sha256"
REPRODUCIBILITY_ARCHIVE_NAME = "trialagentbench-reproducibility.zip"
REPRODUCIBILITY_ARCHIVE_ROOT = "trialagentbench-reproducibility-v1"
PUBLIC_DISTRIBUTION_PROJECTS = frozenset(
    {
        canonicalize_name("trial-agent-bench"),
        canonicalize_name("trialagentbench-verifier"),
        canonicalize_name("clinical-data-tables"),
    }
)
ROLE_ARCHIVES = {
    ("trialeval", "participant"): "trialeval-participant.zip",
    ("trialeval", "evaluator"): "trialeval-evaluator.zip",
    ("trialdev", "participant"): "trialdev-participant.zip",
    ("trialdev", "evaluator"): "trialdev-evaluator.zip",
}
# An optional withheld evaluation split per benchmark. Both of its archives are
# private to the leaderboard operator and are never distributed publicly: the
# withheld participant archive's task set itself would expose cell roles.
WITHHELD_ROLE_ARCHIVES = {
    (benchmark, f"withheld-{role}"): f"{benchmark}-withheld-{role}.zip"
    for benchmark in ("trialeval", "trialdev")
    for role in ("participant", "evaluator")
}
ToolkitArtifactRole = Literal["distribution", "documentation", "execution", "practice", "provenance"]
ReleaseArtifactKind = Literal["public_toolkit", "controlled_role", "withheld_role", "reproducibility"]
AudienceRole = Literal[
    "public",
    "trialeval_participant",
    "trialeval_evaluator",
    "trialdev_participant",
    "trialdev_evaluator",
    "trialeval_withheld_participant",
    "trialeval_withheld_evaluator",
    "trialdev_withheld_participant",
    "trialdev_withheld_evaluator",
    "reproducer",
]
ReproducibilityArtifactRole = Literal[
    "campaign_plan",
    "model_condition_manifest",
    "result_export",
    "analysis_report",
    "result_data",
    "environment",
    "provenance",
]
_LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVIEWED_VENDORED_WHEELS = {
    ("autograd-gamma", "0.5.0"): {
        "source_sha256": "f27abb7b8bb9cffc8badcbf59f3fe44a9db39e124ecacf1992b6d952934ac9c4",
        "wheel_sha256": "70b354e1fd583051fcfc36aae5c5a295a9b340138384bf4b96ac5a78fa7b7699",
        "wheel_members": {
            "autograd_gamma/__init__.py",
            "autograd_gamma-0.5.0.dist-info/licenses/LICENSE",
            "autograd_gamma-0.5.0.dist-info/METADATA",
            "autograd_gamma-0.5.0.dist-info/WHEEL",
            "autograd_gamma-0.5.0.dist-info/top_level.txt",
            "autograd_gamma-0.5.0.dist-info/RECORD",
        },
    }
}


class VendoredWheelProvenance(BaseModel):
    """Reviewed equivalence between a shipped wheel and locked source."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["ariadne.vendored_wheel/v1"] = "ariadne.vendored_wheel/v1"
    name: str
    version: str
    license: str
    upstream_source: str
    wheel_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_code_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    wheel_code_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_license_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    wheel_license_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    wheel_members: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_reviewed_identity(self) -> Self:
        """Require reviewed archive identities, members, code, and licence."""

        identity = (str(canonicalize_name(self.name)), self.version)
        reviewed = _REVIEWED_VENDORED_WHEELS.get(identity)
        if (
            self.name != identity[0]
            or reviewed is None
            or self.source_sha256 != reviewed["source_sha256"]
            or self.wheel_sha256 != reviewed["wheel_sha256"]
            or set(self.wheel_members) != reviewed["wheel_members"]
        ):
            raise ValueError("vendored dependency provenance differs from its reviewed identity")
        if self.source_code_sha256 != self.wheel_code_sha256:
            raise ValueError("vendored wheel code differs from its source")
        if self.source_license_sha256 != self.wheel_license_sha256:
            raise ValueError("vendored wheel licence differs from its source")
        return self


class _Model(BaseModel):
    """Strict immutable release-record base."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SoftwareCandidateReference(_Model):
    """Non-disclosing identity of the reviewed software candidate."""

    candidate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    wheelhouse_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_distributions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PublicDistributionBuildV1(_Model):
    """Source-equivalent wheel/sdist identity for one public project."""

    project: str = Field(min_length=1)
    version: str = Field(min_length=1)
    import_package: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    wheel_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_distribution_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_member_count: int = Field(ge=1)


class PublicDistributionBuildProvenanceV1(_Model):
    """Exact public distribution pairs built from candidate-bound sources."""

    schema_version: Literal["trialagentbench.public_distribution_build/v1"] = (
        "trialagentbench.public_distribution_build/v1"
    )
    distributions: tuple[PublicDistributionBuildV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def current_public_projects(self) -> Self:
        """Require exactly one built pair for each public runtime distribution."""
        names = tuple(row.project for row in self.distributions)
        if len(names) != len(set(names)) or set(names) != set(PUBLIC_DISTRIBUTION_PROJECTS):
            raise ValueError("public build provenance must cover each public project exactly once")
        return self


class ExecutorTrustAnchorV1(_Model):
    """Release-owner observation and portable semantic executor identities."""

    image_reference: str = Field(min_length=1)
    image_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    environment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ExecutorReleaseProvenanceV1(ExecutorTrustAnchorV1):
    """Complete qualified executor inventory and its build-context identity."""

    schema_version: Literal["trialagentbench.executor_release/v1"] = "trialagentbench.executor_release/v1"
    environment: ExecutorEnvironmentV1

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Derive public trust anchors from the complete runtime inventory."""

        if self.image_reference != self.environment.image_reference or self.image_id != self.environment.image_id:
            raise ValueError("executor trust anchor differs from its environment inventory")
        if (
            self.environment.operating_system == "unknown"
            or self.environment.architecture == "unknown"
            or self.environment.r_version is None
            or not self.environment.r_packages
            or not self.environment.system_packages
        ):
            raise ValueError("release executor environment must include its system, Python, and R inventories")
        if self.environment.limits != RELEASE_EXECUTION_LIMITS_V1:
            raise ValueError("release executor limits differ from the public replay budget")
        observed = self.environment.semantic_sha256()
        if self.environment_sha256 != observed:
            raise ValueError("executor environment SHA-256 differs from its complete inventory")
        return self

    def trust_anchor(self) -> ExecutorTrustAnchorV1:
        """Project the minimal identities needed by a clean consumer."""

        return ExecutorTrustAnchorV1(
            image_reference=self.image_reference,
            image_id=self.image_id,
            environment_sha256=self.environment_sha256,
            context_sha256=self.context_sha256,
        )


class ExecutorAttestationV1(_Model):
    """Local immutable image identity derived from an authenticated toolkit."""

    schema_version: Literal["trialagentbench.executor_attestation/v1"] = "trialagentbench.executor_attestation/v1"
    toolkit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_reference: str = Field(min_length=1)
    image_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    environment_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    environment: ExecutorEnvironmentV1

    @model_validator(mode="after")
    def validate_observation(self) -> Self:
        """Bind the semantic digest and local image identity to the probe."""

        if (
            self.image_reference != self.environment.image_reference
            or self.image_id != self.environment.image_id
            or self.environment_sha256 != self.environment.semantic_sha256()
        ):
            raise ValueError("executor attestation differs from its observed environment")
        return self


def _catalog_name(archive_name: str) -> str:
    """Return the selective catalog that transports a role archive in a release set."""
    return archive_name.removesuffix(".zip") + ".assets.json"


def _portable_relative_path(value: str) -> PurePosixPath:
    """Parse one normalized non-empty relative POSIX path."""

    if not value or "\\" in value or ":" in value or "\x00" in value or unicodedata.normalize("NFC", value) != value:
        raise ValueError("release path must be a normalized relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("release path must be a normalized relative POSIX path")
    return path


class ToolkitArtifactV1(_Model):
    """Identity and public purpose of one toolkit member."""

    relative_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_length: int = Field(ge=0)
    role: ToolkitArtifactRole

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        """Reject non-portable member paths."""

        _portable_relative_path(self.relative_path)
        return self


class ToolkitManifestV1(_Model):
    """Complete content identity of the public software toolkit."""

    schema_version: Literal["trialagentbench.public_toolkit/v1"] = TOOLKIT_SCHEMA_VERSION
    toolkit_id: str = Field(pattern=r"^toolkit_[0-9a-f]{64}$")
    software_candidate: SoftwareCandidateReference
    executor: ExecutorTrustAnchorV1
    artifacts: tuple[ToolkitArtifactV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_inventory_and_identity(self) -> Self:
        """Require one sorted, public-only, content-derived inventory."""

        paths = tuple(artifact.relative_path for artifact in self.artifacts)
        if paths != tuple(sorted(set(paths))):
            raise ValueError("toolkit artifacts must be unique and sorted")
        by_path = {artifact.relative_path: artifact for artifact in self.artifacts}
        required = {
            "CITATION.cff",
            "LICENSE",
            "README.md",
            "execution/Dockerfile",
            "execution/README.md",
            "execution/parquet_to_csv.py",
            "execution/requirements.in",
            "execution/requirements.lock",
            "execution/r-package/DESCRIPTION",
            "execution/r-package/NAMESPACE",
            "execution/r-package/R/io.R",
            "execution/wheels/autograd_gamma-0.5.0-py3-none-any.whl",
            "execution/wheels/autograd-gamma-0.5.0.tar.gz",
            "practice/practice_cases_v3.json",
            "provenance/autograd-gamma-0.5.0.json",
            "provenance/executor-environment.json",
            "provenance/sbom.spdx.json",
            "requirements.lock",
        }
        if not required.issubset(by_path):
            raise ValueError("toolkit omits required execution, practice, or provenance material")
        distribution_paths = tuple(
            artifact.relative_path for artifact in self.artifacts if artifact.role == "distribution"
        )
        _validate_public_distribution_names(distribution_paths)
        if any(
            path.startswith("releases/") or path.endswith("-participant.zip") or path.endswith("-evaluator.zip")
            for path in paths
        ):
            raise ValueError("controlled role archives cannot enter the public toolkit")
        payload = self.model_dump(mode="json", exclude={"toolkit_id"})
        expected = f"toolkit_{hashlib.sha256(_canonical_json(payload)).hexdigest()}"
        if self.toolkit_id != expected:
            raise ValueError("toolkit identity does not match its content")
        return self


class ReleaseSetArtifactV1(_Model):
    """External identity and audience of one separately distributed archive."""

    filename: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_length: int = Field(ge=0)
    kind: ReleaseArtifactKind
    audience: AudienceRole
    benchmark_stream: Literal["trialeval", "trialdev"] | None = None

    @model_validator(mode="after")
    def validate_role(self) -> Self:
        """Bind fixed filenames to their only valid audience and stream."""

        path = _portable_relative_path(self.filename)
        if len(path.parts) != 1:
            raise ValueError("release-set artifacts must be top-level files")
        expected_roles = {
            TOOLKIT_ARCHIVE_NAME: ("public_toolkit", "public", None),
            REPRODUCIBILITY_ARCHIVE_NAME: ("reproducibility", "reproducer", None),
            **{
                _catalog_name(name): (
                    "withheld_role" if role.startswith("withheld-") else "controlled_role",
                    f"{benchmark}_{role.replace('-', '_')}",
                    benchmark,
                )
                for (benchmark, role), name in {**ROLE_ARCHIVES, **WITHHELD_ROLE_ARCHIVES}.items()
            },
        }
        expected = expected_roles.get(self.filename)
        if expected is None or (self.kind, self.audience, self.benchmark_stream) != expected:
            raise ValueError("release-set artifact filename and audience role disagree")
        return self


class ReleaseSetManifestV1(_Model):
    """External census for the toolkit and separately controlled archives."""

    schema_version: Literal["trialagentbench.release_set/v2"] = RELEASE_SET_SCHEMA_VERSION
    release_set_id: str = Field(pattern=r"^release_set_[0-9a-f]{64}$")
    software_candidate: SoftwareCandidateReference
    executor: ExecutorTrustAnchorV1
    base_release_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reproducibility_status: Literal["not_yet_available", "included"]
    artifacts: tuple[ReleaseSetArtifactV1, ...]

    @model_validator(mode="after")
    def validate_set_and_identity(self) -> Self:
        """Require the five base archives and at most one post-result archive."""

        filenames = tuple(artifact.filename for artifact in self.artifacts)
        if filenames != tuple(sorted(set(filenames))):
            raise ValueError("release-set artifacts must be unique and sorted")
        expected = {TOOLKIT_ARCHIVE_NAME, *(_catalog_name(name) for name in ROLE_ARCHIVES.values())}
        observed = set(filenames)
        for benchmark in ("trialeval", "trialdev"):
            withheld = {
                _catalog_name(WITHHELD_ROLE_ARCHIVES[(benchmark, f"withheld-{role}")])
                for role in ("participant", "evaluator")
            }
            if withheld & observed:
                expected |= withheld
        if self.reproducibility_status == "included":
            expected.add(REPRODUCIBILITY_ARCHIVE_NAME)
        elif REPRODUCIBILITY_ARCHIVE_NAME in observed:
            raise ValueError("reproducibility archive cannot precede result availability")
        if observed != expected:
            raise ValueError(
                "release set must contain one toolkit, four role archives and complete withheld pairs only"
            )
        expected_base = _base_release_sha256(self.software_candidate, self.executor, self.artifacts)
        if self.base_release_sha256 != expected_base:
            raise ValueError("base release identity does not match the toolkit and role archives")
        payload = self.model_dump(mode="json", exclude={"release_set_id"})
        expected_id = f"release_set_{hashlib.sha256(_canonical_json(payload)).hexdigest()}"
        if self.release_set_id != expected_id:
            raise ValueError("release-set identity does not match its artifacts")
        return self


def role_filename(filenames: Iterable[str], benchmark: str, role: str) -> str:
    """Resolve exactly one authenticated role archive or selective catalog.

    Release sets carry selective catalogs; development campaign inputs carry
    the construction role archives.

    Parameters
    ----------
    filenames
        Authenticated release-set artifact names.
    benchmark
        ``trialeval`` or ``trialdev``.
    role
        ``participant`` or ``evaluator``, or the ``withheld-`` prefixed roles
        of an optional withheld evaluation split.

    Returns
    -------
    str
        The single matching ZIP or ``.assets.json`` transport name.

    Raises
    ------
    ValueError
        If the role is unknown or the release does not contain exactly one match.
    """
    if (benchmark, role) not in ROLE_ARCHIVES and (benchmark, role) not in WITHHELD_ROLE_ARCHIVES:
        raise ValueError("Unknown benchmark release role")
    stem = f"{benchmark}-{role}"
    matches = [name for name in filenames if name in {f"{stem}.zip", f"{stem}.assets.json"}]
    if len(matches) != 1:
        raise ValueError(f"Release must identify exactly one {benchmark} {role} transport")
    return matches[0]


class ReproducibilityArtifactV1(_Model):
    """Identity and scientific role of one post-result artifact."""

    relative_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_length: int = Field(ge=0)
    role: ReproducibilityArtifactRole

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        """Reject non-portable artifact paths."""

        _portable_relative_path(self.relative_path)
        return self


class ModelConditionIdentityV1(_Model):
    """One prespecified model and experimental-condition pairing."""

    model_id: str = Field(min_length=1)
    condition_id: str = Field(min_length=1)


class ModelConditionRunIdentityV1(_Model):
    """One request replicate and the immutable run identities scheduled for it."""

    request_replicate_id: str = Field(min_length=1)
    run_identity_sha256s: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_runs(self) -> Self:
        """Require a sorted unique run-identity set."""

        if self.run_identity_sha256s != tuple(sorted(set(self.run_identity_sha256s))):
            raise ValueError("model-condition run identities must be unique and sorted")
        if any(not _LOWER_SHA256.fullmatch(value) for value in self.run_identity_sha256s):
            raise ValueError("model-condition run identities must be lowercase SHA-256 values")
        return self


class TrialEvalModelConditionV1(_Model):
    """One complete prospective TrialEval model condition using live run contracts."""

    model_id: str = Field(min_length=1)
    condition_id: str = Field(min_length=1)
    request_conditions: tuple[ExperimentConditionV1, ...] = Field(min_length=1)
    scheduled_runs: tuple[ModelConditionRunIdentityV1, ...] = Field(min_length=1)
    decoding: DecodingConfigV1
    routing: RoutingConfigV1
    retry: ProviderRetryPolicyV1
    budget_profile_id: Literal["trialeval_release_default_v1"] = "trialeval_release_default_v1"
    item_watchdog_seconds: int = Field(ge=1)
    maximum_context_characters: int = Field(ge=1)
    code_execution: CodeExecutionLimitsV1
    worker_count: int = Field(ge=1)
    reported_cost_stop_usd: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def validate_condition(self) -> Self:
        """Bind request replicates, route, reasoning evidence, and run identities."""

        request_ids = tuple(row.request_replicate_id for row in self.request_conditions)
        run_request_ids = tuple(row.request_replicate_id for row in self.scheduled_runs)
        if request_ids != tuple(sorted(set(request_ids))) or run_request_ids != request_ids:
            raise ValueError("model-condition request replicates must be unique, sorted, and fully scheduled")
        common = {
            json.dumps(row.model_dump(mode="json", exclude={"request_replicate_id"}), sort_keys=True)
            for row in self.request_conditions
        }
        if len(common) != 1 or any(row.condition_id != self.condition_id for row in self.request_conditions):
            raise ValueError("request replicates must share one exact experiment condition")
        for row in self.request_conditions:
            capability = row.provider_capability
            if capability is not None and (
                capability.model_id != self.model_id
                or capability.provider_transport != self.routing.provider
                or capability.upstream_provider != self.routing.openrouter_provider
            ):
                raise ValueError("reasoning capability differs from the model-condition route")
        return self


class TrialEvalModelConditionManifestV1(_Model):
    """Prospective census of every model condition and its exact execution policy."""

    schema_version: Literal["trialagentbench.trialeval_model_conditions/v1"] = (
        "trialagentbench.trialeval_model_conditions/v1"
    )
    benchmark_release_id: str = Field(min_length=1)
    benchmark_release_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prospective_analysis_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    conditions: tuple[TrialEvalModelConditionV1, ...] = Field(min_length=1)
    checksum: str = ""

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        """Require a canonical condition census and derived checksum."""

        keys = tuple((row.model_id, row.condition_id) for row in self.conditions)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("model-condition records must be unique and sorted")
        payload = self.model_dump(mode="json", exclude={"checksum"})
        expected = canonical_sha256(cast(JsonValue, payload))
        if self.checksum and self.checksum != expected:
            raise ValueError("model-condition manifest checksum mismatch")
        object.__setattr__(self, "checksum", expected)
        return self


class ReproducibilityManifestV1(_Model):
    """Typed binding from post-result evidence to the released benchmark bytes."""

    schema_version: Literal["trialagentbench.reproducibility/v1"] = "trialagentbench.reproducibility/v1"
    reproducibility_id: str = Field(pattern=r"^reproducibility_[0-9a-f]{64}$")
    base_release_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    benchmark_stream: Literal["trialeval"] = "trialeval"
    benchmark_release_id: str = Field(min_length=1)
    benchmark_release_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    campaign_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    campaign_plan_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_condition_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_condition_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_export_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_export_manifest_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    budget_censored_attempt_ids: tuple[str, ...] = Field(default=(), exclude_if=lambda value: not value)
    numeric_reproduction_tolerance: float = Field(default=1e-12, gt=0.0, le=1e-9)
    model_conditions: tuple[ModelConditionIdentityV1, ...] = Field(min_length=1)
    artifacts: tuple[ReproducibilityArtifactV1, ...] = Field(min_length=3)

    @model_validator(mode="after")
    def validate_membership_and_identity(self) -> Self:
        """Require result, plan and analysis membership with a derived identity."""

        paths = tuple(artifact.relative_path for artifact in self.artifacts)
        if paths != tuple(sorted(set(paths))):
            raise ValueError("reproducibility artifacts must be unique and sorted")
        condition_keys = tuple((row.model_id, row.condition_id) for row in self.model_conditions)
        if condition_keys != tuple(sorted(set(condition_keys))):
            raise ValueError("model conditions must be unique and sorted")
        if self.budget_censored_attempt_ids != tuple(sorted(set(self.budget_censored_attempt_ids))) or any(
            not attempt for attempt in self.budget_censored_attempt_ids
        ):
            raise ValueError("budget-censored attempt identities must be unique sorted nonempty strings")
        counts = {
            role: sum(artifact.role == role for artifact in self.artifacts)
            for role in ("campaign_plan", "model_condition_manifest", "result_export", "analysis_report")
        }
        if any(count != 1 for count in counts.values()):
            raise ValueError("reproducibility archive requires one campaign plan, result export, and analysis report")
        by_role = {artifact.role: artifact for artifact in self.artifacts}
        if self.campaign_plan_sha256 != by_role["campaign_plan"].sha256:
            raise ValueError("campaign plan artifact identity differs from the reproducibility manifest")
        if self.model_condition_manifest_sha256 != by_role["model_condition_manifest"].sha256:
            raise ValueError("model-condition artifact identity differs from the reproducibility manifest")
        if self.result_export_sha256 != by_role["result_export"].sha256:
            raise ValueError("result export artifact identity differs from the reproducibility manifest")
        if self.analysis_sha256 != by_role["analysis_report"].sha256:
            raise ValueError("analysis artifact identity differs from the reproducibility manifest")
        payload = self.model_dump(mode="json", exclude={"reproducibility_id"})
        expected = f"reproducibility_{hashlib.sha256(_canonical_json(payload)).hexdigest()}"
        if self.reproducibility_id != expected:
            raise ValueError("reproducibility identity does not match its artifacts")
        return self


def _canonical_json(payload: object) -> bytes:
    """Serialize a deterministic JSON record."""

    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _base_release_sha256(
    software: SoftwareCandidateReference,
    executor: ExecutorTrustAnchorV1,
    artifacts: tuple[ReleaseSetArtifactV1, ...],
) -> str:
    """Identify the toolkit and four controlled archives independently of results."""

    base = tuple(artifact for artifact in artifacts if artifact.kind != "reproducibility")
    payload = {
        "software_candidate": software.model_dump(mode="json"),
        "executor": executor.model_dump(mode="json"),
        "artifacts": [artifact.model_dump(mode="json") for artifact in base],
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _sha256(path: Path) -> str:
    """Hash one regular file without reading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_tree_sha256(records: dict[str, bytes]) -> str:
    identities = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(records.items())}
    return canonical_sha256(cast(JsonValue, identities))


def verify_public_distribution_pair(
    wheel: Path,
    source_distribution: Path,
    *,
    import_package: str,
    candidate_source_root: Path | None = None,
) -> tuple[str, int]:
    """Require wheel, sdist, and optional candidate source runtime bytes to agree."""

    wheel = _regular_file(wheel, label="public wheel")
    source_distribution = _regular_file(source_distribution, label="public source distribution")
    prefix = f"{import_package}/"
    with zipfile.ZipFile(wheel) as archive:
        validate_zip_members(archive)
        wheel_records = {
            name: archive.read(name)
            for name in archive.namelist()
            if name.startswith(prefix) and not name.endswith("/")
        }
    with tarfile.open(source_distribution, mode="r:gz") as archive:
        source_records: dict[str, bytes] = {}
        runtime_mappings: list[tuple[tuple[str, ...], str]] = []
        manifests = [
            member
            for member in archive.getmembers()
            if len(PurePosixPath(member.name).parts) == 2 and member.name.endswith("/pyproject.toml")
        ]
        if len(manifests) > 1:
            raise ValueError("source distribution repeats its build manifest")
        if manifests:
            manifest_stream = archive.extractfile(manifests[0])
            if manifest_stream is None or not manifests[0].isfile():
                raise ValueError("source build manifest must be a regular file")
            with manifest_stream:
                mapping: object = tomllib.loads(manifest_stream.read().decode("utf-8"))
            for key in ("tool", "hatch", "build", "targets", "wheel", "force-include"):
                if not isinstance(mapping, dict):
                    raise ValueError("source wheel inclusion declaration must be a table")
                mapping = mapping.get(key, {})
            if not isinstance(mapping, dict):
                raise ValueError("source wheel inclusion mapping must be a table")
            for source, target in mapping.items():
                if not isinstance(source, str) or not isinstance(target, str):
                    raise ValueError("source wheel inclusion paths must be strings")
                source_path, target_path = _portable_relative_path(source), _portable_relative_path(target)
                if target_path.as_posix().startswith(prefix):
                    runtime_mappings.append((source_path.parts, target_path.as_posix()))
        for member in archive.getmembers():
            member_path = _portable_relative_path(member.name)
            parts = member_path.parts
            direct_package = len(parts) >= 3 and parts[1] == import_package
            src_package = len(parts) >= 4 and parts[1:3] == ("src", import_package)
            relative: str | None = None
            if direct_package or src_package:
                start = 2 if parts[1] == "src" else 1
                relative = PurePosixPath(*parts[start:]).as_posix()
            else:
                for source_prefix, target_prefix in runtime_mappings:
                    if parts[1 : 1 + len(source_prefix)] == source_prefix:
                        tail = PurePosixPath(*parts[1 + len(source_prefix) :]).as_posix()
                        relative = target_prefix if tail == "." else f"{target_prefix}/{tail}"
                        break
            if relative is None or not relative.startswith(prefix):
                continue
            if not member.isfile():
                raise ValueError("public source distribution contains a non-regular runtime member")
            if relative in source_records:
                raise ValueError("public source distribution contains a duplicate runtime member")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("public source distribution runtime member cannot be read")
            source_records[relative] = stream.read()
    if not wheel_records or wheel_records != source_records:
        raise ValueError("public wheel runtime bytes differ from its source distribution")
    if candidate_source_root is not None:
        root = Path(candidate_source_root)
        candidate_records: dict[str, bytes] = {}
        for path in root.rglob("*"):
            candidate_relative = path.relative_to(root)
            if "__pycache__" in candidate_relative.parts or path.suffix == ".pyc":
                continue
            if path.is_symlink():
                raise ValueError("candidate package source tree contains a symbolic link")
            if path.is_dir():
                continue
            if not path.is_file() or not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
                raise ValueError("candidate package source tree contains a non-regular member")
            candidate_records[f"{import_package}/{candidate_relative.as_posix()}"] = path.read_bytes()
        if candidate_records != wheel_records:
            raise ValueError("public distributions differ from candidate-bound package sources")
    return _runtime_tree_sha256(wheel_records), len(wheel_records)


def _regular_file(path: Path, *, label: str) -> Path:
    """Require one non-symlinked regular file."""

    path = Path(path)
    if path.is_symlink() or not path.is_file() or not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
        raise ValueError(f"{label} must be one regular non-symlinked file: {path}")
    return path


def _read_bounded_file(path: Path, *, label: str, limit: int = MAX_JSON_BYTES) -> bytes:
    """Read one bounded regular metadata file."""

    path = _regular_file(path, label=label)
    if path.stat().st_size > limit:
        raise ValueError(f"{label} exceeds its byte limit: {path}")
    return path.read_bytes()


def _executor_context_sha256(artifacts: tuple[ToolkitArtifactV1, ...]) -> str:
    """Identify the exact archive-native Docker build context."""

    inventory = [
        {
            "relative_path": artifact.relative_path.removeprefix("execution/"),
            "sha256": artifact.sha256,
            "byte_length": artifact.byte_length,
        }
        for artifact in artifacts
        if artifact.relative_path.startswith("execution/")
    ]
    if not inventory:
        raise ValueError("executor build context is empty")
    return hashlib.sha256(_canonical_json(inventory)).hexdigest()


def _validate_public_distribution_names(relative_paths: tuple[str, ...]) -> None:
    """Require exactly the public harness and verifier wheel/source pairs."""

    names = tuple(PurePosixPath(path).name for path in relative_paths)
    wheels = tuple(name for name in names if name.endswith(".whl"))
    sdists = tuple(name for name in names if name.endswith(".tar.gz"))
    count = len(PUBLIC_DISTRIBUTION_PROJECTS)
    if len(wheels) != count or len(sdists) != count or len(names) != 2 * count:
        raise ValueError("toolkit requires one wheel/source pair per public project")
    wheel_projects = {
        (canonicalize_name(name), str(version)) for name, version, _, _ in map(parse_wheel_filename, wheels)
    }
    sdist_projects = {(canonicalize_name(name), str(version)) for name, version in map(parse_sdist_filename, sdists)}
    if wheel_projects != sdist_projects or {name for name, _ in wheel_projects} != PUBLIC_DISTRIBUTION_PROJECTS:
        raise ValueError("toolkit distributions differ from the public runtime project set")


def validate_zip_members(archive: zipfile.ZipFile, *, required_prefix: str | None = None) -> tuple[str, ...]:
    """Reject duplicate, unsafe, directory, encrypted, symlink and special members."""

    names: list[str] = []
    seen: set[str] = set()
    seen_casefolded: set[str] = set()
    total_bytes = 0
    prefix = f"{required_prefix}/" if required_prefix is not None else None
    infos = archive.infolist()
    if len(infos) > MAX_RELEASE_ARCHIVE_MEMBERS:
        raise ValueError("archive exceeds the member-count limit")
    for info in infos:
        name = info.filename
        try:
            path = _portable_relative_path(name)
        except ValueError as error:
            raise ValueError(f"archive contains a non-portable member: {name!r}") from error
        if prefix is not None and (not name.startswith(prefix) or len(path.parts) < 2):
            raise ValueError(f"archive member is outside {required_prefix}: {name}")
        if name in seen:
            raise ValueError(f"archive contains a duplicate member: {name}")
        seen.add(name)
        casefolded = unicodedata.normalize("NFC", name).casefold()
        if casefolded in seen_casefolded:
            raise ValueError(f"archive contains a case-insensitive path collision: {name}")
        seen_casefolded.add(casefolded)
        mode = (info.external_attr >> 16) & 0xFFFF
        kind = stat.S_IFMT(mode)
        if info.is_dir() or name.endswith("/") or kind not in {0, stat.S_IFREG}:
            raise ValueError(f"archive member is not a regular file: {name}")
        if info.flag_bits & 0x1:
            raise ValueError(f"archive member is encrypted: {name}")
        if info.file_size > MAX_RELEASE_MEMBER_BYTES:
            raise ValueError(f"archive member exceeds the per-member byte limit: {name}")
        total_bytes += info.file_size
        if total_bytes > MAX_RELEASE_UNCOMPRESSED_BYTES:
            raise ValueError("archive exceeds the aggregate uncompressed-byte limit")
        names.append(name)
    if not names:
        raise ValueError("archive contains no regular files")
    return tuple(names)


def _validate_archive_file(path: Path) -> tuple[str, ...]:
    """Validate one external ZIP before it enters the release set."""

    path = _regular_file(path, label="controlled archive")
    if path.stat().st_size > MAX_RELEASE_ARCHIVE_BYTES:
        raise ValueError(f"controlled archive exceeds the compressed-byte limit: {path.name}")
    if not zipfile.is_zipfile(path):
        raise ValueError(f"controlled archive is not a ZIP file: {path.name}")
    with zipfile.ZipFile(path) as archive:
        return validate_zip_members(archive)


def _read_bounded_member(archive: zipfile.ZipFile, name: str, *, limit: int = MAX_JSON_BYTES) -> bytes:
    """Read one bounded archive member after its central-directory limits pass."""

    info = archive.getinfo(name)
    if info.file_size > limit:
        raise ValueError(f"archive metadata member exceeds its byte limit: {name}")
    with archive.open(info) as source:
        payload = source.read(limit + 1)
    if len(payload) > limit:
        raise ValueError(f"archive metadata member exceeds its byte limit: {name}")
    return payload


def _copy_member_and_hash(archive: zipfile.ZipFile, name: str, target: Path | None = None) -> tuple[str, int]:
    """Stream one ZIP member to an optional regular file while hashing it."""

    digest = hashlib.sha256()
    observed = 0
    destination = target.open("xb") if target is not None else None
    try:
        with archive.open(name) as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                observed += len(chunk)
                if observed > MAX_RELEASE_ARCHIVE_BYTES:
                    raise ValueError(f"archive member exceeds the per-member byte limit: {name}")
                digest.update(chunk)
                if destination is not None:
                    destination.write(chunk)
    finally:
        if destination is not None:
            destination.close()
    return digest.hexdigest(), observed


def _read_checksum_census(data: bytes) -> dict[str, str]:
    """Parse a complete, normalized SHA-256 census."""

    try:
        lines = data.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("checksum census is not ASCII") from error
    records: dict[str, str] = {}
    for line in lines:
        digest, separator, relative = line.partition("  ")
        if separator != "  " or not _LOWER_SHA256.fullmatch(digest) or relative in records:
            raise ValueError("checksum census is malformed")
        _portable_relative_path(relative)
        records[relative] = digest
    if not records:
        raise ValueError("checksum census is empty")
    return records


def _verify_toolkit_archive_copy(archive_path: Path, *, expected_sha256: str) -> ToolkitManifestV1:
    """Verify public toolkit identity and complete member census."""

    archive_path = _regular_file(archive_path, label="public toolkit")
    if not _LOWER_SHA256.fullmatch(expected_sha256) or _sha256(archive_path) != expected_sha256:
        raise ValueError("public toolkit differs from its externally trusted SHA-256")
    with zipfile.ZipFile(archive_path) as archive:
        names = validate_zip_members(archive, required_prefix=TOOLKIT_ARCHIVE_ROOT)
        prefix = f"{TOOLKIT_ARCHIVE_ROOT}/"
        manifest_name = f"{prefix}toolkit_manifest.json"
        checksums_name = f"{prefix}SHA256SUMS"
        if manifest_name not in names or checksums_name not in names:
            raise ValueError("public toolkit is missing its manifest or checksum census")
        manifest_bytes = _read_bounded_member(archive, manifest_name)
        manifest = ToolkitManifestV1.model_validate_json(manifest_bytes)
        expected_names = {
            manifest_name,
            checksums_name,
            *(f"{prefix}{artifact.relative_path}" for artifact in manifest.artifacts),
        }
        if set(names) != expected_names:
            raise ValueError("public toolkit contains undeclared or stale bytes")
        checksums = _read_checksum_census(_read_bounded_member(archive, checksums_name))
        expected_checksums = {
            "toolkit_manifest.json": hashlib.sha256(manifest_bytes).hexdigest(),
            **{artifact.relative_path: artifact.sha256 for artifact in manifest.artifacts},
        }
        if checksums != expected_checksums:
            raise ValueError("public toolkit checksum census differs from its manifest")
        for artifact in manifest.artifacts:
            observed_sha256, observed_bytes = _copy_member_and_hash(archive, f"{prefix}{artifact.relative_path}")
            if observed_bytes != artifact.byte_length or observed_sha256 != artifact.sha256:
                raise ValueError(f"public toolkit artifact changed: {artifact.relative_path}")
        provenance = VendoredWheelProvenance.model_validate_json(
            _read_bounded_member(archive, f"{prefix}provenance/autograd-gamma-0.5.0.json")
        )
        distribution_provenance = PublicDistributionBuildProvenanceV1.model_validate_json(
            _read_bounded_member(archive, f"{prefix}provenance/public-distributions.json")
        )
        executor_provenance = ExecutorReleaseProvenanceV1.model_validate_json(
            _read_bounded_member(archive, f"{prefix}provenance/executor-environment.json")
        )
        by_path = {artifact.relative_path: artifact for artifact in manifest.artifacts}
        if (
            by_path["execution/wheels/autograd_gamma-0.5.0-py3-none-any.whl"].sha256 != provenance.wheel_sha256
            or by_path["execution/wheels/autograd-gamma-0.5.0.tar.gz"].sha256 != provenance.source_sha256
        ):
            raise ValueError("public toolkit vendored dependency identity mismatch")
        with tempfile.TemporaryDirectory(prefix="trialagentbench-distributions-") as temporary:
            distribution_root = Path(temporary)
            for record in distribution_provenance.distributions:
                wheel_artifact = next(
                    artifact
                    for artifact in manifest.artifacts
                    if artifact.role == "distribution"
                    and artifact.relative_path.endswith(".whl")
                    and str(canonicalize_name(parse_wheel_filename(Path(artifact.relative_path).name)[0]))
                    == record.project
                )
                source_artifact = next(
                    artifact
                    for artifact in manifest.artifacts
                    if artifact.role == "distribution"
                    and artifact.relative_path.endswith(".tar.gz")
                    and str(canonicalize_name(parse_sdist_filename(Path(artifact.relative_path).name)[0]))
                    == record.project
                )
                if (
                    wheel_artifact.sha256 != record.wheel_sha256
                    or source_artifact.sha256 != record.source_distribution_sha256
                ):
                    raise ValueError("public distribution provenance differs from toolkit artifacts")
                wheel_path = distribution_root / Path(wheel_artifact.relative_path).name
                source_path = distribution_root / Path(source_artifact.relative_path).name
                for artifact, target in ((wheel_artifact, wheel_path), (source_artifact, source_path)):
                    with archive.open(f"{prefix}{artifact.relative_path}") as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
                tree_sha256, member_count = verify_public_distribution_pair(
                    wheel_path,
                    source_path,
                    import_package=record.import_package,
                )
                if (tree_sha256, member_count) != (record.runtime_tree_sha256, record.runtime_member_count):
                    raise ValueError("public distribution runtime tree differs from its provenance")
        if (
            executor_provenance.trust_anchor() != manifest.executor
            or executor_provenance.context_sha256 != _executor_context_sha256(manifest.artifacts)
        ):
            raise ValueError("public toolkit executor provenance differs from its context or manifest")
        sbom = json.loads(_read_bounded_member(archive, f"{prefix}provenance/sbom.spdx.json"))
        if sbom.get("spdxVersion") != "SPDX-2.3" or not isinstance(sbom.get("packages"), list):
            raise ValueError("public toolkit SBOM is malformed")
        sbom_hashes = {
            checksum.get("checksumValue")
            for package in sbom["packages"]
            if isinstance(package, dict)
            for checksum in package.get("checksums", ())
            if isinstance(checksum, dict) and checksum.get("algorithm") == "SHA256"
        }
        expected_wheel_hashes = {
            artifact.sha256 for artifact in manifest.artifacts if artifact.relative_path.endswith(".whl")
        }
        if sbom_hashes != expected_wheel_hashes:
            raise ValueError("public toolkit SBOM does not bind every shipped wheel")
        return manifest


def verify_toolkit_archive(archive_path: Path, *, expected_sha256: str) -> ToolkitManifestV1:
    """Verify an authenticated private copy of one public toolkit archive."""

    with release_archive(archive_path, expected_sha256=expected_sha256) as authenticated:
        return _verify_toolkit_archive_copy(authenticated, expected_sha256=expected_sha256)


def attest_executor_from_toolkit(
    toolkit_archive: Path,
    *,
    expected_toolkit_sha256: str,
    image_reference: str,
) -> ExecutorAttestationV1:
    """Build and attest a local executor from authenticated toolkit bytes."""

    with tempfile.TemporaryDirectory(prefix="trialagentbench-executor-") as temporary:
        authenticated_toolkit = Path(temporary) / TOOLKIT_ARCHIVE_NAME
        with release_archive(toolkit_archive, expected_sha256=expected_toolkit_sha256) as source:
            shutil.copyfile(source, authenticated_toolkit)
        manifest = _verify_toolkit_archive_copy(authenticated_toolkit, expected_sha256=expected_toolkit_sha256)
        prefix = f"{TOOLKIT_ARCHIVE_ROOT}/"
        context = Path(temporary) / "execution"
        context.mkdir()
        with zipfile.ZipFile(authenticated_toolkit) as archive:
            provenance = ExecutorReleaseProvenanceV1.model_validate_json(
                _read_bounded_member(archive, f"{prefix}provenance/executor-environment.json")
            )
            execution_artifacts = tuple(
                artifact for artifact in manifest.artifacts if artifact.relative_path.startswith("execution/")
            )
            if not execution_artifacts:
                raise ValueError("public toolkit contains no executor build context")
            for artifact in execution_artifacts:
                relative = _portable_relative_path(artifact.relative_path).relative_to("execution")
                destination = context.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(f"{prefix}{artifact.relative_path}") as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
        build_reference = (
            f"trialagentbench/executor-build:{expected_toolkit_sha256[:20]}"
            if image_reference.startswith("sha256:") or "@" in image_reference
            else image_reference
        )
        image_identity = Path(temporary) / "built-image.id"
        build = subprocess.run(
            ["docker", "build", "--iidfile", str(image_identity), "--tag", build_reference, str(context)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
        )
        if build.returncode != 0:
            raise RuntimeError(f"executor image build failed: {build.stderr.strip() or 'docker build failed'}")
        built_image_id = image_identity.read_text(encoding="ascii").strip()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", built_image_id):
            raise ValueError("Executor build did not return an immutable image identity")
        from trialagentbench_verifier.executor import resolve_executor_environment

        environment = resolve_executor_environment(
            image=built_image_id,
            limits=provenance.environment.limits,
        )
        if environment.image_id != built_image_id:
            raise ValueError("Executor probe did not inspect the image produced by this build")
    environment_sha256 = environment.semantic_sha256()
    if environment_sha256 != manifest.executor.environment_sha256:
        raise ValueError("locally built executor differs from the published semantic environment")
    return ExecutorAttestationV1(
        toolkit_sha256=expected_toolkit_sha256,
        context_sha256=manifest.executor.context_sha256,
        image_reference=environment.image_reference,
        image_id=environment.image_id,
        environment_sha256=environment_sha256,
        environment=environment,
    )


def _verify_reproducibility_archive_copy(
    archive_path: Path,
    *,
    expected_sha256: str,
    expected_base_release_sha256: str,
) -> ReproducibilityManifestV1:
    """Verify a post-result archive and its binding to the five base artifacts."""

    archive_path = _regular_file(archive_path, label="reproducibility archive")
    if not _LOWER_SHA256.fullmatch(expected_sha256) or _sha256(archive_path) != expected_sha256:
        raise ValueError("reproducibility archive differs from its externally trusted SHA-256")
    with zipfile.ZipFile(archive_path) as archive:
        names = validate_zip_members(archive, required_prefix=REPRODUCIBILITY_ARCHIVE_ROOT)
        prefix = f"{REPRODUCIBILITY_ARCHIVE_ROOT}/"
        manifest_name = f"{prefix}reproducibility_manifest.json"
        checksums_name = f"{prefix}SHA256SUMS"
        if manifest_name not in names or checksums_name not in names:
            raise ValueError("reproducibility archive is missing its manifest or checksum census")
        manifest_bytes = _read_bounded_member(archive, manifest_name)
        manifest = ReproducibilityManifestV1.model_validate_json(manifest_bytes)
        if manifest.base_release_sha256 != expected_base_release_sha256:
            raise ValueError("reproducibility archive identifies a different base release")
        expected_names = {
            manifest_name,
            checksums_name,
            *(f"{prefix}{artifact.relative_path}" for artifact in manifest.artifacts),
        }
        if set(names) != expected_names:
            raise ValueError("reproducibility archive contains undeclared or stale bytes")
        checksums = _read_checksum_census(_read_bounded_member(archive, checksums_name))
        expected_checksums = {
            "reproducibility_manifest.json": hashlib.sha256(manifest_bytes).hexdigest(),
            **{artifact.relative_path: artifact.sha256 for artifact in manifest.artifacts},
        }
        if checksums != expected_checksums:
            raise ValueError("reproducibility checksum census differs from its manifest")
        artifact_bytes: dict[str, bytes] = {}
        with tempfile.TemporaryDirectory(prefix="trialagentbench-reproducibility-") as temporary:
            result_export_path = Path(temporary) / "result-export.zip"
            for artifact in manifest.artifacts:
                name = f"{prefix}{artifact.relative_path}"
                target = result_export_path if artifact.role == "result_export" else None
                observed_sha256, observed_bytes = _copy_member_and_hash(archive, name, target)
                if observed_bytes != artifact.byte_length or observed_sha256 != artifact.sha256:
                    raise ValueError(f"reproducibility artifact changed: {artifact.relative_path}")
                if artifact.role in {"campaign_plan", "model_condition_manifest", "analysis_report"}:
                    artifact_bytes[artifact.role] = _read_bounded_member(archive, name)
            plan = TrialEvalCampaignPlanV1.model_validate_json(artifact_bytes["campaign_plan"])
            condition_manifest = TrialEvalModelConditionManifestV1.model_validate_json(
                artifact_bytes["model_condition_manifest"]
            )
            result_export = verify_result_export(result_export_path)
            analysis = TrialEvalCampaignAnalysisV1.model_validate_json(artifact_bytes["analysis_report"])
            recomputed = analyse_trial_eval_campaign_v1(
                str(result_export_path),
                expected_release_id=manifest.benchmark_release_id,
                expected_release_manifest_sha256=manifest.benchmark_release_manifest_sha256,
                expected_archive_sha256=manifest.result_export_sha256,
                plan=plan,
                expected_campaign_plan_sha256=manifest.campaign_plan_checksum,
                bootstrap_resamples=analysis.bootstrap_resamples,
                confidence_level=analysis.confidence_level,
                bootstrap_seed=analysis.bootstrap_seed,
                budget_censored_attempt_ids=frozenset(manifest.budget_censored_attempt_ids),
            )
        model_conditions = tuple(
            ModelConditionIdentityV1(model_id=model_id, condition_id=condition_id)
            for model_id, condition_id in sorted(
                {(attempt.model_id, attempt.condition_id) for attempt in plan.attempts}
            )
        )
        if (
            plan.release_id != manifest.benchmark_release_id
            or result_export.release_id != manifest.benchmark_release_id
            or analysis.release_id != manifest.benchmark_release_id
            or plan.release_manifest_sha256 != manifest.benchmark_release_manifest_sha256
            or result_export.release_manifest_sha256 != manifest.benchmark_release_manifest_sha256
            or analysis.release_manifest_sha256 != manifest.benchmark_release_manifest_sha256
        ):
            raise ValueError("reproducibility artifacts identify different benchmark release bytes")
        if (
            plan.checksum != manifest.campaign_plan_checksum
            or result_export.checksum != manifest.result_export_manifest_checksum
            or analysis.checksum != manifest.analysis_checksum
            or analysis.campaign_plan_checksum != recomputed.campaign_plan_checksum
            or analysis.prospective_analysis_plan_sha256 != plan.prospective_analysis_plan_sha256
            or analysis.export_manifest_checksum != result_export.checksum
            or analysis.source_archive_sha256 != manifest.result_export_sha256
            or hashlib.sha256(artifact_bytes["model_condition_manifest"]).hexdigest()
            != manifest.model_condition_manifest_sha256
            or condition_manifest.checksum != manifest.model_condition_manifest_checksum
            or model_conditions != manifest.model_conditions
        ):
            raise ValueError("reproducibility campaign, export, analysis, or model identities disagree")
        _validate_model_condition_manifest(plan, condition_manifest)
        if not _json_equal_with_numeric_tolerance(
            analysis.model_dump(mode="json", exclude={"checksum"}),
            recomputed.model_dump(mode="json", exclude={"checksum"}),
            tolerance=manifest.numeric_reproduction_tolerance,
        ):
            raise ValueError("reproducibility analysis differs from a fresh derivation")
        return manifest


def verify_reproducibility_archive(
    archive_path: Path,
    *,
    expected_sha256: str,
    expected_base_release_sha256: str,
) -> ReproducibilityManifestV1:
    """Verify and recompute an authenticated private reproducibility archive."""

    with release_archive(archive_path, expected_sha256=expected_sha256) as authenticated:
        return _verify_reproducibility_archive_copy(
            authenticated,
            expected_sha256=expected_sha256,
            expected_base_release_sha256=expected_base_release_sha256,
        )


def _json_equal_with_numeric_tolerance(left: object, right: object, *, tolerance: float) -> bool:
    """Compare typed JSON exactly except for finite floating-point roundoff."""

    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if isinstance(left, int) and isinstance(right, int):
            return left == right
        return abs(float(left) - float(right)) <= tolerance * max(1.0, abs(float(left)), abs(float(right)))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _json_equal_with_numeric_tolerance(left[key], right[key], tolerance=tolerance) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _json_equal_with_numeric_tolerance(a, b, tolerance=tolerance) for a, b in zip(left, right, strict=True)
        )
    return type(left) is type(right) and left == right


class _GradeCensusV1(BaseModel):
    """Read the archived task denominator without interpreting historical scoring rules."""

    model_config = ConfigDict(extra="ignore")
    graded_task_count: int = Field(ge=0)


def verify_result_export(
    path: Path,
    *,
    required_profile: Literal["analysis_reproduction", "full_custody"] = "analysis_reproduction",
    expected_sha256: str | None = None,
) -> _ResultExportManifestV1:
    """Verify the complete internal identity of one archived result export.

    Members must be portable and unique ignoring case; SHA256SUMS, the manifest and every
    member's bytes must agree. An explicitly supplied digest pins the archive to externally trusted bytes.
    The exporter reads back each archive it writes with this check.
    """

    path = _regular_file(path, label="reproducibility result export")
    if path.stat().st_size > MAX_RELEASE_ARCHIVE_BYTES:
        raise ValueError("reproducibility result export exceeds the compressed-byte limit")
    if expected_sha256 is not None and _sha256(path) != expected_sha256:
        raise VerificationInputError("result export differs from its externally trusted SHA-256")
    with zipfile.ZipFile(path) as export:
        names = validate_zip_members(export)
        if "manifest.json" not in names or "SHA256SUMS" not in names:
            raise ValueError("reproducibility result export omits its manifest or checksum census")
        manifest_bytes = _read_bounded_member(export, "manifest.json")
        manifest = _ResultExportManifestV1.model_validate_json(manifest_bytes)
        if manifest.export_profile != required_profile:
            raise ValueError(f"result export requires the {required_profile} profile")
        prefixes = tuple(source.archive_prefix for source in manifest.sources)
        for prefix in prefixes:
            _portable_relative_path(prefix)
        if any(
            left.startswith(f"{right}/") or right.startswith(f"{left}/")
            for index, left in enumerate(prefixes)
            for right in prefixes[index + 1 :]
        ):
            raise ValueError("reproducibility result export source prefixes overlap")
        for source in manifest.sources:
            owned = tuple(member for member in manifest.members if member.path.startswith(f"{source.archive_prefix}/"))
            if len(owned) != source.file_count:
                raise ValueError("reproducibility result export source file count differs from its members")
        if any(
            sum(member.path.startswith(f"{source.archive_prefix}/") for source in manifest.sources) != 1
            for member in manifest.members
        ):
            raise ValueError("reproducibility result export member has no unique source")
        expected_names = {"manifest.json", "SHA256SUMS", *(member.path for member in manifest.members)}
        if set(names) != expected_names:
            raise ValueError("reproducibility result export contains undeclared or stale bytes")
        expected_checksums = {
            "manifest.json": hashlib.sha256(manifest_bytes).hexdigest(),
            **{member.path: member.sha256 for member in manifest.members},
        }
        if _read_checksum_census(_read_bounded_member(export, "SHA256SUMS")) != expected_checksums:
            raise ValueError("reproducibility result export checksum census differs from its manifest")
        for member in manifest.members:
            observed_sha256, observed_bytes = _copy_member_and_hash(export, member.path)
            if observed_bytes != member.size_bytes or observed_sha256 != member.sha256:
                raise ValueError(f"reproducibility result export member changed: {member.path}")
        for source in manifest.sources:
            grade_path = f"{source.archive_prefix}/GRADE_MANIFEST.json"
            if source.kind != "grade" or grade_path not in names:
                continue
            grade = _GradeCensusV1.model_validate_json(_read_bounded_member(export, grade_path))
            graded_count = sum(
                member.path.startswith(f"{source.archive_prefix}/items/") and member.path.endswith(".json")
                for member in manifest.members
            )
            if graded_count != grade.graded_task_count:
                raise ValueError("reproducibility result export graded count differs from its manifest")
    return manifest


def _campaign_model_conditions(plan: TrialEvalCampaignPlanV1) -> tuple[ModelConditionIdentityV1, ...]:
    """Project unique model and condition identities from the frozen attempt schedule."""

    return tuple(
        ModelConditionIdentityV1(model_id=model_id, condition_id=condition_id)
        for model_id, condition_id in sorted({(attempt.model_id, attempt.condition_id) for attempt in plan.attempts})
    )


def _validate_model_condition_manifest(
    plan: TrialEvalCampaignPlanV1,
    manifest: TrialEvalModelConditionManifestV1,
) -> None:
    """Bind the prospective condition policy to every scheduled campaign attempt."""

    if (
        manifest.benchmark_release_id != plan.release_id
        or manifest.benchmark_release_manifest_sha256 != plan.release_manifest_sha256
        or manifest.prospective_analysis_plan_sha256 != plan.prospective_analysis_plan_sha256
    ):
        raise ValueError("model-condition manifest identifies different prospective benchmark bytes")
    expected_conditions = _campaign_model_conditions(plan)
    observed_conditions = tuple(
        ModelConditionIdentityV1(model_id=row.model_id, condition_id=row.condition_id) for row in manifest.conditions
    )
    if observed_conditions != expected_conditions:
        raise ValueError("model-condition manifest differs from the campaign model-condition census")
    for condition in manifest.conditions:
        attempts = tuple(
            attempt
            for attempt in plan.attempts
            if (attempt.model_id, attempt.condition_id) == (condition.model_id, condition.condition_id)
        )
        expected_request_ids = tuple(sorted({attempt.request_replicate_id for attempt in attempts}))
        observed_request_ids = tuple(row.request_replicate_id for row in condition.request_conditions)
        if observed_request_ids != expected_request_ids:
            raise ValueError("model-condition request replicates differ from the campaign schedule")
        by_request = {row.request_replicate_id: row for row in condition.request_conditions}
        for attempt in attempts:
            request = by_request[attempt.request_replicate_id]
            if request.procedure_assistance != attempt.procedure_assistance:
                raise ValueError("model-condition assistance differs from the campaign schedule")
        expected_runs = tuple(
            ModelConditionRunIdentityV1(
                request_replicate_id=request_id,
                run_identity_sha256s=tuple(
                    sorted(
                        {
                            attempt.run_identity_sha256
                            for attempt in attempts
                            if attempt.request_replicate_id == request_id
                        }
                    )
                ),
            )
            for request_id in expected_request_ids
        )
        if condition.scheduled_runs != expected_runs:
            raise ValueError("model-condition run identities differ from the campaign schedule")


def authenticate_release_set_manifest(release_root: Path, *, expected_manifest_sha256: str) -> ReleaseSetManifestV1:
    """Load a release identity from its externally pinned manifest digest."""

    release_root = Path(release_root)
    if not release_root.is_dir() or release_root.is_symlink():
        raise ValueError("release-set root must be a non-symlinked directory")
    manifest_path = _regular_file(release_root / RELEASE_SET_MANIFEST_NAME, label="release-set manifest")
    if not _LOWER_SHA256.fullmatch(expected_manifest_sha256) or _sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("release-set manifest differs from its externally trusted SHA-256")
    manifest = ReleaseSetManifestV1.model_validate_json(
        _read_bounded_file(manifest_path, label="release-set manifest")
    )
    return manifest


def authenticate_release_set(release_root: Path, *, expected_manifest_sha256: str) -> ReleaseSetManifestV1:
    """Authenticate exact bytes of an externally anchored set without repeating archive qualification."""
    manifest = authenticate_release_set_manifest(release_root, expected_manifest_sha256=expected_manifest_sha256)
    for artifact in manifest.artifacts:
        if artifact.kind == "withheld_role" and not (release_root / artifact.filename).exists():
            continue  # Private to the leaderboard operator; never part of public downloads.
        path = _regular_file(release_root / artifact.filename, label="release-set artifact")
        if path.stat().st_size != artifact.byte_length or _sha256(path) != artifact.sha256:
            raise ValueError(f"release-set artifact changed: {artifact.filename}")
    return manifest


def verify_public_release_layout(release_root: Path, manifest: ReleaseSetManifestV1) -> None:
    """Refuse a public release-set directory that holds anything beyond the public census.

    Parameters
    ----------
    release_root
        Directory staged for public distribution.
    manifest
        Authenticated release-set manifest of that directory.

    Raises
    ------
    ValueError
        If a withheld archive or any file other than the public artifacts, the shards
        their catalogs name and the release-set manifest files is present.
    """
    release_root = Path(release_root)
    allowed = {RELEASE_SET_MANIFEST_NAME, RELEASE_SET_MANIFEST_HASH_NAME, RELEASE_SET_CHECKSUM_NAME}
    for artifact in manifest.artifacts:
        if artifact.kind == "withheld_role":
            continue
        allowed.add(artifact.filename)
        if artifact.filename.endswith(".assets.json"):
            catalog = read_asset_catalog(release_root / artifact.filename, expected_sha256=artifact.sha256)
            allowed.update(shard.filename for shard in catalog.shards)
    withheld = sorted(artifact.filename for artifact in manifest.artifacts if artifact.kind == "withheld_role")
    present = {path.relative_to(release_root).as_posix() for path in release_root.rglob("*") if not path.is_dir()}
    if exposed := sorted(present & set(withheld)):
        raise ValueError(f"public release set contains private withheld archives: {', '.join(exposed)}")
    if unexpected := sorted(present - allowed):
        raise ValueError(f"public release set contains files outside its public census: {', '.join(unexpected)}")


class ReleaseSetVerificationSummaryV1(_Model):
    """Coverage of a successful full-set verification; counts exclude absent private roles."""

    artifacts: int = Field(ge=0)
    artifact_bytes: int = Field(ge=0)
    payload_shards: int = Field(ge=0)
    payload_bytes: int = Field(ge=0)
    verified_roles: tuple[str, ...]
    unavailable_roles: tuple[str, ...]
    verified: tuple[str, ...]
    not_performed: tuple[str, ...]


def release_set_verification_summary(
    release_root: Path, manifest: ReleaseSetManifestV1
) -> ReleaseSetVerificationSummaryV1:
    """Describe a completed full-set verification without changing its authenticated manifest.

    Call only after ``verify_release_set`` succeeds without task selections.
    Shards are counted once by digest even when shared between role catalogs.

    Parameters
    ----------
    release_root
        Directory whose present artifacts have just passed full verification.
    manifest
        Authenticated manifest returned by that verification.

    Returns
    -------
    ReleaseSetVerificationSummaryV1
        Artifact and shard coverage, unavailable private roles and explicit limits.
    """
    present = tuple(item for item in manifest.artifacts if (release_root / item.filename).is_file())
    shards: dict[str, int] = {}
    external_trialdev = False
    for item in present:
        if item.kind in {"controlled_role", "withheld_role"}:
            catalog = read_asset_catalog(release_root / item.filename, expected_sha256=item.sha256)
            shards.update({shard.sha256: shard.byte_length for shard in catalog.shards})
            external_trialdev |= any(member.path == EXTERNAL_QUALIFICATION_NAME for member in catalog.members)
    return ReleaseSetVerificationSummaryV1(
        artifacts=len(present),
        artifact_bytes=sum(item.byte_length for item in present),
        payload_shards=len(shards),
        payload_bytes=sum(shards.values()),
        verified_roles=tuple(item.audience for item in present if item.kind in {"controlled_role", "withheld_role"}),
        unavailable_roles=tuple(item.audience for item in manifest.artifacts if item not in present),
        verified=(
            "Externally anchored manifest, checksum census, artifact byte lengths and SHA-256 digests",
            "Present role catalogs, payload integrity, release contracts and archived qualification evidence",
            "Participant/evaluator identity and catalog pairing where both roles are present",
            "Toolkit inventory, software candidate and executor context/environment trust-anchor agreement",
            *(
                ("Reproducibility archive integrity and aggregate score recomputation",)
                if any(item.kind == "reproducibility" for item in present)
                else ()
            ),
        ),
        not_performed=(
            *(
                (
                    "Public TrialDev recovery and qualification are external; no archive pass claimed; consult release-evidence",
                )
                if external_trialdev
                else ()
            ),
            "Local Docker image availability, executor rebuild or runtime attestation",
            "Fresh execution of participant analyses, model campaigns or scientific world simulations",
            "Independent clinical validation beyond the archived qualification evidence",
            *(
                ("Post-result reproducibility archive: not supplied",)
                if not any(item.kind == "reproducibility" for item in present)
                else ()
            ),
        ),
    )


def verify_release_set(
    release_root: Path,
    *,
    expected_manifest_sha256: str,
    task_selections: dict[str, tuple[str, ...]] | None = None,
    split: Literal["public", "withheld"] = "public",
) -> ReleaseSetManifestV1:
    """Verify the set, optionally limiting payload checks to selected tasks.

    The default qualifies every present role payload. Private withheld archives
    that are absent, as in every public copy, are skipped; the leaderboard
    operator's copy verifies them. An explicit selection verifies payloads only
    for its named benchmarks and the chosen split's role pair.
    """
    release_root = Path(release_root)
    manifest = authenticate_release_set_manifest(release_root, expected_manifest_sha256=expected_manifest_sha256)
    # Only manifest-named artifacts belong to this release. Local configuration,
    # result folders and other downloads never enter an agent workspace or archive.
    detached = _read_bounded_file(
        release_root / RELEASE_SET_MANIFEST_HASH_NAME,
        label="detached release-set manifest hash",
    ).decode("ascii")
    if detached != f"{expected_manifest_sha256}  {RELEASE_SET_MANIFEST_NAME}\n":
        raise ValueError("detached release-set manifest hash is inconsistent")
    checksums = _read_checksum_census(
        _read_bounded_file(release_root / RELEASE_SET_CHECKSUM_NAME, label="release-set checksum census")
    )
    expected_checksums = {
        RELEASE_SET_MANIFEST_NAME: expected_manifest_sha256,
        **{artifact.filename: artifact.sha256 for artifact in manifest.artifacts},
    }
    if checksums != expected_checksums:
        raise ValueError("release-set checksum census differs from its manifest")
    if task_selections is not None and (not task_selections or not set(task_selections) <= {"trialeval", "trialdev"}):
        raise ValueError("Selected release-set verification requires known benchmarks")
    reports: dict[tuple[str, str, str], VerificationReportV1] = {}
    for artifact in manifest.artifacts:
        path = release_root / artifact.filename
        if artifact.kind == "public_toolkit":
            toolkit = verify_toolkit_archive(path, expected_sha256=artifact.sha256)
            if toolkit.software_candidate != manifest.software_candidate:
                raise ValueError("toolkit and release set identify different software candidates")
            if toolkit.executor != manifest.executor:
                raise ValueError("toolkit and release set identify different executors")
        elif artifact.kind in {"controlled_role", "withheld_role"}:
            artifact_split = "withheld" if "_withheld_" in artifact.audience else "public"
            if task_selections is not None and (
                artifact.benchmark_stream not in task_selections or artifact_split != split
            ):
                continue
            if artifact.kind == "withheld_role" and not path.exists():
                continue
            catalog = ReleaseAssets.model_validate_json(_read_bounded_file(path, label="controlled asset catalog"))
            prefix = "withheld-" if artifact_split == "withheld" else ""
            if artifact.audience != f"{catalog.benchmark}_{prefix.replace('-', '_')}{catalog.audience}":
                raise ValueError("Asset catalog differs from its release-set audience")
            if catalog.audience == "evaluator":
                participant_name = role_filename(
                    (item.filename for item in manifest.artifacts), catalog.benchmark, f"{prefix}participant"
                )
                participant_digest = next(
                    item.sha256 for item in manifest.artifacts if item.filename == participant_name
                )
                if hashlib.sha256(catalog.participant_catalog().to_bytes()).hexdigest() != participant_digest:
                    raise ValueError("Evaluator catalog differs from its paired participant catalog")
            report = verify_release(
                path,
                expected_sha256=artifact.sha256,
                task_ids=None if task_selections is None else task_selections[artifact.benchmark_stream or ""],
            )
            if report.status != "pass" or report.benchmark_stream != artifact.benchmark_stream:
                raise ValueError(f"controlled archive failed independent verification: {artifact.filename}")
            reports[
                (artifact.benchmark_stream or "", artifact_split, artifact.audience.rsplit("_", maxsplit=1)[-1])
            ] = report
        else:
            verify_reproducibility_archive(
                path,
                expected_sha256=artifact.sha256,
                expected_base_release_sha256=manifest.base_release_sha256,
            )
    for stream, pair_split in {(key[0], key[1]) for key in reports}:
        if {(stream, pair_split, "participant"), (stream, pair_split, "evaluator")} - reports.keys():
            continue  # An absent private withheld archive has no pair to compare.
        participant = reports[(stream, pair_split, "participant")]
        evaluator = reports[(stream, pair_split, "evaluator")]
        if (
            participant.candidate_digest != evaluator.candidate_digest
            or participant.release_digest != evaluator.release_digest
        ):
            raise ValueError(f"{stream} participant and evaluator archives identify different releases")
    return manifest


__all__ = [
    "ExecutorAttestationV1",
    "PublicDistributionBuildProvenanceV1",
    "PublicDistributionBuildV1",
    "ReleaseSetManifestV1",
    "ReleaseSetVerificationSummaryV1",
    "release_set_verification_summary",
    "ReproducibilityManifestV1",
    "ToolkitManifestV1",
    "WITHHELD_ROLE_ARCHIVES",
    "TrialEvalModelConditionManifestV1",
    "attest_executor_from_toolkit",
    "authenticate_release_set",
    "authenticate_release_set_manifest",
    "verify_release_set",
    "verify_public_distribution_pair",
    "verify_public_release_layout",
    "verify_reproducibility_archive",
    "verify_result_export",
    "verify_toolkit_archive",
]
