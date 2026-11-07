"""Load and verify role-separated TrialDev releases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from trialagentbench_verifier.assets import ReleaseAssetScope
from trialagentbench_verifier.grading_key import (
    GRADING_KEY_MANIFEST,
    GradingKeyArtifactV1,
    GradingKeyManifestV1,
    TrialDevWorldStrataV1,
    TrialDevWorldStratumV1,
    read_trialdev_world_strata,
    verify_grading_key,
)
from trialagentbench_verifier.io import VerificationInputError, release_asset_scope
from trialagentbench_verifier.numerical import NumericalReportingContext

from trialagentbench.contracts.release.identity import BenchmarkReleaseIdentityV1
from trialagentbench.contracts.trialdev.grading_key import TrialDevEvidenceTargetV1
from trialagentbench.contracts.trialdev.pools import TrialDevFrozenPool
from trialagentbench.contracts.trialdev.release import (
    TrialDevReleaseArtifactV1,
    TrialDevReleaseManifestV1,
    TrialDevRoleManifestV1,
    TrialDevTaskV1,
)
from trialagentbench.io import canonical_payload_sha256, read_json, read_json_model, sha256_file
from trialagentbench.trialdev.value import TrialDevSponsorPreferences


class TrialDevReleaseError(ValueError):
    """Raised when released TrialDev bytes violate their declared contract."""


class TrialDevInitialEvidence(BaseModel):
    """Release-bound initial cohort context and observed-association references."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    context: NumericalReportingContext
    targets: tuple[TrialDevEvidenceTargetV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def bind_target_inventory(self) -> TrialDevInitialEvidence:
        """Reject a release whose public requests differ from its grading inventory."""
        identities = tuple(f"{target.asset_id}:{target.endpoint}" for target in self.targets)
        if identities != self.context.target_ids or len(set(identities)) != len(identities):
            raise ValueError("initial evidence context must match its exact grading target inventory")
        return self


class TrialDevAnalysisConfiguration(BaseModel):
    """Evaluator-only observed-association targets and numerical normalization."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_id: Literal["trialdev.numerical-analysis/v1"] = "trialdev.numerical-analysis/v1"
    preferences: TrialDevSponsorPreferences
    initial_evidence: TrialDevInitialEvidence


class TrialDevAuthoredProgrammeConfiguration(BaseModel):
    """Evaluator-only configuration for an authored development task."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_id: Literal["trialdev.authored-programme/v1"] = "trialdev.authored-programme/v1"
    pool_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    budget_usd: float = Field(gt=0)
    recruitment_seed: int = Field(ge=0, strict=True)
    preferences: TrialDevSponsorPreferences
    initial_evidence: TrialDevInitialEvidence


@dataclass(frozen=True)
class TrialDevParticipantRelease:
    """Verified participant release and its resolved task inventory."""

    root: Path
    manifest: TrialDevRoleManifestV1
    release_identity: BenchmarkReleaseIdentityV1
    release_manifest_sha256: str
    tasks: tuple[TrialDevTaskV1, ...]

    def world_root(self, task: TrialDevTaskV1) -> Path:
        """Return the verified participant directory for one task's case."""

        return self.root / "worlds" / task.case_id


@dataclass(frozen=True)
class TrialDevEvaluatorRelease:
    """Fully verified evaluator release reused across grading operations.

    A public key archive carries no evaluation role manifest: its grading key inventories the evaluator
    files it retains. An operator archive carries both; a construction release directory only the role.
    """

    root: Path
    manifest: TrialDevReleaseManifestV1
    participant: TrialDevParticipantRelease
    evaluation_manifest: TrialDevRoleManifestV1 | None
    grading_key: GradingKeyManifestV1 | None = None
    world_strata: TrialDevWorldStrataV1 | None = None

    @property
    def evaluation_artifacts(self) -> tuple[TrialDevReleaseArtifactV1 | GradingKeyArtifactV1, ...]:
        """Return every declared evaluator-side identity: the evaluation role's, else the grading key's."""
        if self.evaluation_manifest is not None:
            return tuple(self.evaluation_manifest.artifacts)
        if self.grading_key is None:
            raise TrialDevReleaseError("evaluator release declares neither its evaluation role nor a grading key")
        return self.grading_key.artifacts

    def world_stratum(self, world_id: str) -> TrialDevWorldStratumV1:
        """Return one world's ranking stratum from the grading key.

        Raises
        ------
        TrialDevReleaseError
            If the release carries no grading key or the world is not in it.
        """
        if self.world_strata is None:
            raise TrialDevReleaseError("ranking requires the grading key's world strata")
        try:
            return self.world_strata.stratum(world_id)
        except VerificationInputError as error:
            raise TrialDevReleaseError(str(error)) from error

    def _configuration_file(self, relative: Path) -> Path | None:
        """Resolve an optional evaluator file, rejecting undeclared or altered bytes."""
        path = self.root / "evaluation" / relative
        declared = {item.relative_path: item for item in self.evaluation_artifacts}
        artifact = declared.get((Path("evaluation") / relative).as_posix())
        if artifact is None and not path.exists() and not path.is_symlink():
            return None
        if (
            artifact is None
            or not path.is_file()
            or any(
                (self.root / "evaluation").joinpath(*relative.parts[:i]).is_symlink()
                for i in range(len(relative.parts) + 1)
            )
            or path.stat().st_size != artifact.byte_length
        ):
            raise TrialDevReleaseError("configuration and pool must match the evaluator inventory")
        return path

    def _verify_initial_context(self, task: TrialDevTaskV1, context: NumericalReportingContext) -> None:
        """Bind the reporting snapshot to the participant bytes before execution."""
        declared = {artifact.relative_path: artifact for artifact in self.participant.manifest.artifacts}
        for artifact in context.evidence_artifacts:
            identity = declared.get(f"participant/worlds/{task.case_id}/{artifact.relative_path}")
            if identity is None or (identity.sha256, identity.byte_length) != (artifact.sha256, artifact.byte_length):
                raise TrialDevReleaseError("Initial reporting context differs from the released evidence inventory")

    def analysis_configuration(self, task: TrialDevTaskV1) -> TrialDevAnalysisConfiguration | None:
        """Load the standalone numerical task configuration from verified bytes.

        Parameters
        ----------
        task
            Observational-analysis task whose identity owns the configuration.

        Returns
        -------
        TrialDevAnalysisConfiguration or None
            Verified configuration, or None for another stream or an absent file.

        Raises
        ------
        TrialDevReleaseError
            If a present or declared configuration differs from the inventory.
        """
        if task.stream.value != "evidence_analysis":
            return None
        path = self._configuration_file(Path("tasks") / task.task_id / "analysis.json")
        if path is None:
            return None
        configuration = read_json_model(TrialDevAnalysisConfiguration, path)
        self._verify_initial_context(task, configuration.initial_evidence.context)
        return configuration

    def authored_configuration(self, task: TrialDevTaskV1) -> TrialDevAuthoredProgrammeConfiguration | None:
        """Load an authored configuration only from the verified evaluator inventory.

        Parameters
        ----------
        task
            Participant task whose case owns the configuration and frozen pool.

        Returns
        -------
        TrialDevAuthoredProgrammeConfiguration or None
            Declared development configuration; None for an observational-analysis
            task or a release without authored development.

        Raises
        ------
        TrialDevReleaseError
            If configuration/pool bytes are undeclared, changed or cross-case.
        """
        if task.stream.value != "portfolio_management":
            return None
        relative = Path("tasks") / task.task_id / "authored_programme.json"
        configuration_path = self._configuration_file(relative)
        if configuration_path is None:
            return None
        pool_path = self._configuration_file(Path("worlds") / task.case_id / "pool" / "manifest.json")
        if pool_path is None:
            raise TrialDevReleaseError("authored pool is missing from the evaluator inventory")
        configuration = read_json_model(TrialDevAuthoredProgrammeConfiguration, configuration_path)
        if task.resources is None or task.resources.budget_usd != configuration.budget_usd:
            raise TrialDevReleaseError("Authored configuration differs from the task's declared resources")
        if configuration.pool_manifest_sha256 != sha256_file(pool_path):
            raise TrialDevReleaseError("authored configuration is bound to a different frozen pool")
        if read_json_model(TrialDevFrozenPool, pool_path).world_id != task.case_id:
            raise TrialDevReleaseError("authored pool belongs to a different task case")
        self._verify_initial_context(task, configuration.initial_evidence.context)
        return configuration


def _participant_root(root: Path) -> Path:
    selected = root.resolve(strict=True)
    if (selected / "participant" / "manifest.json").is_file():
        return selected / "participant"
    if (selected / "manifest.json").is_file():
        return selected
    raise TrialDevReleaseError(f"No TrialDev participant manifest found under {selected}")


def load_trialdev_participant_release(
    root: Path, *, scope: ReleaseAssetScope | None = None
) -> TrialDevParticipantRelease:
    """Verify every participant byte and return typed tasks in stable order."""

    scope = scope or release_asset_scope(root)
    participant = _participant_root(root)
    outer_path = participant / "release_manifest.json"
    if not outer_path.is_file() and participant.name == "participant":
        outer_path = participant.parent / "release_manifest.json"
    if not outer_path.is_file() or outer_path.is_symlink():
        raise TrialDevReleaseError("TrialDev participant release manifest must be a regular file")
    if scope is not None:
        scope.verify_release_identity(outer_path, "trialdev")
    release_identity = BenchmarkReleaseIdentityV1.model_validate(read_json(outer_path))
    if release_identity.schema_id != "ariadne.trialdev.release/v1":
        raise TrialDevReleaseError("TrialDev participant release has the wrong benchmark stream")
    manifest = read_json_model(TrialDevRoleManifestV1, participant / "manifest.json")
    if manifest.role != "participant":
        raise TrialDevReleaseError("TrialDev participant archive contains the wrong role manifest")
    if (
        manifest.release_id != release_identity.release_id
        or manifest.checksum != release_identity.participant_manifest_sha256
        or manifest.world_count != release_identity.world_count
        or manifest.task_count != release_identity.task_count
    ):
        raise TrialDevReleaseError("TrialDev participant role is not bound by its outer release manifest")
    declared = {"manifest.json"}
    if outer_path.parent == participant:
        declared.add("release_manifest.json")
    prefix = "participant/"
    for artifact in manifest.artifacts:
        if scope is not None and not scope.requires(artifact.relative_path, artifact.sha256, artifact.byte_length):
            continue
        if not artifact.relative_path.startswith(prefix):
            raise TrialDevReleaseError("TrialDev participant artifact has an invalid role prefix")
        relative = artifact.relative_path.removeprefix(prefix)
        path = participant / relative
        if not path.is_file() or path.is_symlink():
            raise TrialDevReleaseError(f"TrialDev participant artifact is missing: {relative}")
        if path.stat().st_size != artifact.byte_length:
            raise TrialDevReleaseError(f"TrialDev participant artifact changed: {relative}")
        declared.add(relative)
    actual = {
        path.relative_to(participant).as_posix()
        for path in participant.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual != declared:
        raise TrialDevReleaseError("TrialDev participant files differ from the declared manifest")
    tasks = tuple(
        read_json_model(TrialDevTaskV1, path)
        for path in sorted((participant / "tasks").glob("*/task.json"))
        if scope is None or path.parent.name in scope.selected_task_ids
    )
    expected_tasks = manifest.task_count if scope is None else len(scope.selected_task_ids)
    if scope is not None and (
        len(scope.catalog.tasks) != manifest.task_count or {task.task_id for task in tasks} != scope.selected_task_ids
    ):
        raise TrialDevReleaseError("TrialDev selected task census differs from its authenticated scope")
    if len(tasks) != expected_tasks or len({task.task_id for task in tasks}) != len(tasks):
        raise TrialDevReleaseError("TrialDev participant task census differs from its manifest")
    known_tasks = tasks if scope is None else scope.catalog.tasks
    if len({task.case_id for task in known_tasks}) != manifest.world_count:
        raise TrialDevReleaseError("TrialDev participant world census differs from its manifest")
    for task in tasks:
        world_root = participant / "worlds" / task.case_id
        for material in task.available_materials:
            path = world_root / material.relative_path
            if not path.is_file() or path.is_symlink():
                raise TrialDevReleaseError(
                    f"TrialDev task {task.task_id} names unavailable material: {material.relative_path}"
                )
    return TrialDevParticipantRelease(
        root=participant,
        manifest=manifest,
        release_identity=release_identity,
        release_manifest_sha256=sha256_file(outer_path),
        tasks=tasks,
    )


def load_trialdev_evaluator_release(root: Path, *, scope: ReleaseAssetScope | None = None) -> TrialDevEvaluatorRelease:
    """Verify an evaluator release once for repeated grading operations.

    A public key archive is verified through its grading key's inventory, an operator archive through
    its evaluation role and its key, and a construction release directory through its evaluation role.
    """

    scope = scope or release_asset_scope(root)
    selected = root.resolve(strict=True)
    path = selected / "release_manifest.json"
    if not path.is_file():
        raise TrialDevReleaseError("Evaluator-side grading requires the complete TrialDev release")
    manifest = read_json_model(TrialDevReleaseManifestV1, path)
    if scope is not None and scope.catalog.audience != "evaluator":
        raise TrialDevReleaseError("Evaluator loading requires evaluator asset scope")
    participant = load_trialdev_participant_release(selected, scope=scope)
    if participant.manifest.checksum != manifest.participant_manifest_sha256:
        raise TrialDevReleaseError("TrialDev participant identity differs from the release manifest")
    if participant.manifest.release_id != manifest.release_id:
        raise TrialDevReleaseError("TrialDev role release identities disagree")
    complete = (selected / "evaluation" / "manifest.json").is_file()
    keyed = (selected / GRADING_KEY_MANIFEST).is_file()
    if not complete and not keyed:
        raise TrialDevReleaseError("Evaluator-side grading requires the evaluation role or its grading key")
    declared = {"release_manifest.json", "participant/manifest.json"}
    declared.update(
        artifact.relative_path
        for artifact in participant.manifest.artifacts
        if scope is None or scope.requires(artifact.relative_path, artifact.sha256, artifact.byte_length)
    )
    evaluation: TrialDevRoleManifestV1 | None = None
    if complete:
        evaluation = read_json_model(TrialDevRoleManifestV1, selected / "evaluation" / "manifest.json")
        if evaluation.role != "evaluation" or evaluation.checksum != manifest.evaluation_manifest_sha256:
            raise TrialDevReleaseError("TrialDev evaluator identity differs from the release manifest")
        if evaluation.release_id != manifest.release_id:
            raise TrialDevReleaseError("TrialDev role release identities disagree")
        declared.add("evaluation/manifest.json")
        for artifact in evaluation.artifacts:
            if scope is not None and not scope.requires(artifact.relative_path, artifact.sha256, artifact.byte_length):
                continue
            artifact_path = selected / artifact.relative_path
            if not artifact_path.is_file() or artifact_path.is_symlink():
                raise TrialDevReleaseError(f"TrialDev evaluator artifact is missing: {artifact.relative_path}")
            if artifact_path.stat().st_size != artifact.byte_length:
                raise TrialDevReleaseError(f"TrialDev evaluator artifact changed: {artifact.relative_path}")
            declared.add(artifact.relative_path)
    key: GradingKeyManifestV1 | None = None
    strata: TrialDevWorldStrataV1 | None = None
    if keyed:
        try:
            key = verify_grading_key(selected, scope=scope, verify_files=False)
            strata = read_trialdev_world_strata(selected)
        except VerificationInputError as error:
            raise TrialDevReleaseError(f"TrialDev grading key is invalid: {error}") from error
        present = {
            artifact.relative_path
            for artifact in key.artifacts
            if scope is None or scope.requires(artifact.relative_path, artifact.sha256, artifact.byte_length)
        }
        if complete and not {path for path in present if path.startswith("evaluation/")} <= declared:
            raise TrialDevReleaseError("TrialDev grading key retains files outside the evaluation role")
        declared.update(present)
        declared.add(GRADING_KEY_MANIFEST)
    actual = {
        artifact.relative_to(selected).as_posix()
        for artifact in selected.rglob("*")
        if artifact.is_file() and not artifact.is_symlink()
    }
    if actual != declared:
        raise TrialDevReleaseError("TrialDev evaluator files differ from the declared manifests")
    qualification = read_json(selected / "evaluation" / "resource_qualification.json")
    if canonical_payload_sha256(qualification) != manifest.qualification_sha256:
        raise TrialDevReleaseError("TrialDev qualification identity differs from the release manifest")
    return TrialDevEvaluatorRelease(
        root=selected,
        manifest=manifest,
        participant=participant,
        evaluation_manifest=evaluation,
        grading_key=key,
        world_strata=strata,
    )


__all__ = [
    "TrialDevParticipantRelease",
    "TrialDevEvaluatorRelease",
    "TrialDevReleaseError",
    "load_trialdev_participant_release",
    "load_trialdev_evaluator_release",
]
