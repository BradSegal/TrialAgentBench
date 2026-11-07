"""Prepare and operate many-model campaigns through the native benchmark owners."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Literal
from zipfile import ZipFile

from pydantic import BaseModel, ConfigDict, Field, JsonValue
from trialagentbench_verifier.asset_cache import cache_footprint
from trialagentbench_verifier.assets import AssetTask, ReleaseAssets, read_asset_catalog
from trialagentbench_verifier.campaign import (
    TrialEvalCampaignAttemptV1,
    TrialEvalCampaignCellV1,
    TrialEvalCampaignPlanV1,
)
from trialagentbench_verifier.campaign_inputs import (
    DevelopmentCampaignInputsV1,
    authenticate_development_inputs,
    verify_development_evaluator,
)
from trialagentbench_verifier.execution_contracts import EXECUTOR_SLOTS_PER_TASK, RELEASE_EXECUTION_LIMITS_V1
from trialagentbench_verifier.executor import (
    DEFAULT_EXECUTOR_IMAGE,
    DOCKER_CONTROL_TIMEOUT_SECONDS,
    executor_session_capacity,
    resolve_executor_environment,
)
from trialagentbench_verifier.grading_key import TRIALEVAL_GRADING_KEY_RECORDS
from trialagentbench_verifier.io import read_json_object, release_archive, release_assets, sha256_file
from trialagentbench_verifier.release_set import (
    ExecutorAttestationV1,
    ReleaseSetManifestV1,
    attest_executor_from_toolkit,
    authenticate_release_set_manifest,
    role_filename,
    verify_release_set,
)
from trialagentbench_verifier.trialeval import TrialEvalGradingKeyRecordV1

from trialagentbench.contracts.campaign import (
    CampaignConfigurationV1,
    CampaignModelV1,
    CampaignPanelV1,
    CampaignPlanV1,
    CampaignRunV1,
)
from trialagentbench.contracts.core.config import ProviderModelCapabilityV1
from trialagentbench.contracts.core.runs import RunCoverageV1, TrialEvalItemResultV1, TrialEvalRunConfigV1
from trialagentbench.contracts.core.trialeval_factors import trialeval_evidence_factors_v1
from trialagentbench.contracts.release.archive_safety import materialized_release_archive
from trialagentbench.contracts.scoring.trialeval_scores import trialeval_infrastructure_outcome_v1
from trialagentbench.contracts.trialdev.release import TrialDevTaskV1
from trialagentbench.execution_policy import TRIALDEV_RELEASE_BUDGET_V1, TRIALEVAL_RELEASE_BUDGET_V1
from trialagentbench.io import canonical_payload_sha256, read_json_model, write_json_model
from trialagentbench.release import ParticipantTask, load_trialeval_grading_key
from trialagentbench.trialeval.data import canonical_task_factors
from trialagentbench.util.campaign_control import CampaignCapacityStop, CampaignRequestControl, print_spending_status
from trialagentbench.util.parallel import run_bounded_tasks
from trialagentbench.util.reported_cost import RunStopSignal
from trialagentbench.util.run_ownership import exclusive_run_directory as campaign_lock
from trialagentbench.util.source_identity import require_harness_scorer_v1

RETRY_REQUEST_REPLICATE_ID = "retry-1"


def _role(panel: CampaignPanelV1, role: Literal["participant", "evaluator"]) -> str:
    """Name the role of the selected public or privately scored withheld split."""
    return role if panel.split == "public" else f"withheld-{role}"


def _campaign_inputs(
    configuration: CampaignConfigurationV1, *, full_verification: bool = False
) -> ReleaseSetManifestV1 | DevelopmentCampaignInputsV1:
    """Authenticate the selected input class without weakening qualified release checks."""
    if configuration.input_mode == "development":
        if configuration.panel.split != "public":
            raise ValueError("Development inputs have no withheld evaluation split")
        manifest = authenticate_development_inputs(
            configuration.release_directory,
            expected_manifest_sha256=configuration.expected_release_manifest_sha256,
        )
        names = {artifact.filename for artifact in manifest.artifacts}
        for benchmark in configuration.panel.benchmarks:
            try:
                filename = role_filename(names, benchmark, "evaluator")
            except ValueError as error:
                raise ValueError("Development inputs omit a scheduled benchmark") from error
            if full_verification:
                artifact = next(item for item in manifest.artifacts if item.filename == filename)
                verify_development_evaluator(
                    configuration.release_directory / artifact.filename,
                    expected_sha256=artifact.sha256,
                    benchmark=benchmark,
                )
        return manifest
    qualified = authenticate_release_set_manifest(
        configuration.release_directory, expected_manifest_sha256=configuration.expected_release_manifest_sha256
    )
    if full_verification:
        selections = _campaign_asset_selections(configuration, qualified)
        by_benchmark: dict[str, tuple[str, ...]] | None = (
            {
                benchmark: selections[
                    configuration.release_directory
                    / role_filename(
                        (item.filename for item in qualified.artifacts),
                        benchmark,
                        _role(configuration.panel, "evaluator"),
                    )
                ]
                for benchmark in configuration.panel.benchmarks
            }
            if selections
            else None
        )
        return verify_release_set(
            configuration.release_directory,
            expected_manifest_sha256=configuration.expected_release_manifest_sha256,
            task_selections=by_benchmark,
            split=configuration.panel.split,
        )
    return qualified


def _catalog_task_ids(catalog: ReleaseAssets, selectors: Sequence[str]) -> tuple[str, ...]:
    """Resolve catalog metadata through the same selectors as native execution."""
    from trialagentbench.tools.run.canonical_trialdev import _selected_tasks
    from trialagentbench.tools.run.trialeval import _selected_task_ids

    if catalog.benchmark == "trialeval":
        if any(task.context_tier is None for task in catalog.tasks):
            raise ValueError("TrialEval catalog omits context tiers")
        return tuple(_selected_task_ids({task.task_id: str(task.context_tier) for task in catalog.tasks}, selectors))
    return tuple(task.task_id for task in _selected_tasks(catalog.tasks, selectors))


def _campaign_asset_selections(
    configuration: CampaignConfigurationV1,
    manifest: ReleaseSetManifestV1 | DevelopmentCampaignInputsV1,
) -> dict[Path, tuple[str, ...]]:
    """Bind exact selected views before any campaign payload materialization."""
    anchors = {item.filename: item.sha256 for item in manifest.artifacts}
    selections = {}
    for benchmark in configuration.panel.benchmarks:
        name = role_filename(anchors, benchmark, _role(configuration.panel, "evaluator"))
        if not name.endswith(".assets.json"):
            continue
        catalog = read_asset_catalog(configuration.release_directory / name, expected_sha256=anchors[name])
        selectors = (
            configuration.panel.trialeval_tasks if benchmark == "trialeval" else configuration.panel.trialdev_tasks
        )
        ids = _catalog_task_ids(catalog, selectors)
        for role in ("participant", "evaluator"):
            selections[
                configuration.release_directory / role_filename(anchors, benchmark, _role(configuration.panel, role))
            ] = ids
    return selections


def detached_manifest_digest(release: Path, manifest_name: str) -> str:
    """Return the digest a release's detached sidecar declares for its manifest."""
    detached = (release / f"{manifest_name}.sha256").read_text(encoding="ascii")
    match = re.fullmatch(r"([0-9a-f]{64})  " + re.escape(manifest_name) + r"\n?", detached)
    if match is None:
        raise ValueError("Release manifest sidecar must identify its exact manifest")
    return match.group(1)


def _configuration_payload(path: Path) -> tuple[Path, dict[str, JsonValue]]:
    """Resolve the editable input path and its manifest identity before route preparation."""
    parent = path.resolve(strict=True).parent
    payload = read_json_object(path)
    release_name = payload.get("release_directory")
    if not isinstance(release_name, str):
        raise ValueError("Campaign release_directory must be a path string")
    release = (parent / release_name).resolve(strict=True)
    manifest_name = (
        "development-inputs.json" if payload.get("input_mode") == "development" else "release-set-manifest.json"
    )
    if "expected_release_manifest_sha256" not in payload:
        payload["expected_release_manifest_sha256"] = detached_manifest_digest(release, manifest_name)
    if sha256_file(release / manifest_name) != payload["expected_release_manifest_sha256"]:
        raise ValueError("Campaign manifest differs from its pinned digest")
    payload["release_directory"] = str(release)
    return release, payload


def load_configuration(path: Path, *, prepare_executor: bool = False) -> CampaignConfigurationV1:
    """Resolve consumer paths once, relative to the editable configuration."""
    parent = path.resolve(strict=True).parent
    release, payload = _configuration_payload(path)
    manifest_name = (
        "development-inputs.json" if payload.get("input_mode") == "development" else "release-set-manifest.json"
    )
    if "executor_image_id" not in payload:
        manifest = (
            read_json_model(DevelopmentCampaignInputsV1, release / manifest_name)
            if payload.get("input_mode") == "development"
            else read_json_model(ReleaseSetManifestV1, release / manifest_name)
        )
        attestation_path = parent / "executor-attestation.json"
        attestation = read_json_model(ExecutorAttestationV1, attestation_path) if attestation_path.is_file() else None
        if attestation is not None and not _local_executor_image_exists(attestation.image_id):
            if not prepare_executor or not isinstance(manifest, ReleaseSetManifestV1):
                raise ValueError(
                    f"Stale executor attestation {attestation_path}: image {attestation.image_id} is unavailable. "
                    "Run campaign prepare with the authenticated release toolkit to re-attest, "
                    "or restore the saved Docker image."
                )
            print(f"Stale executor attestation: {attestation_path}; preparing a replacement image.")
            attestation = None
        if prepare_executor and attestation is None and isinstance(manifest, ReleaseSetManifestV1):
            authenticate_release_set_manifest(
                release, expected_manifest_sha256=str(payload["expected_release_manifest_sha256"])
            )
            toolkit = next(item for item in manifest.artifacts if item.kind == "public_toolkit")
            # Reuse the image the documentation builds; identity is the environment hash, not the tag.
            image = os.environ.get("TRIALAGENTBENCH_EXECUTOR_IMAGE", DEFAULT_EXECUTOR_IMAGE)
            if _local_executor_image_exists(image):
                environment = resolve_executor_environment(image=image, limits=RELEASE_EXECUTION_LIMITS_V1)
                if environment.semantic_sha256() != manifest.executor.environment_sha256:
                    raise ValueError(f"Local executor {image} differs from the release environment")
                attestation = ExecutorAttestationV1(
                    toolkit_sha256=toolkit.sha256,
                    context_sha256=manifest.executor.context_sha256,
                    image_reference=environment.image_reference,
                    image_id=environment.image_id,
                    environment_sha256=environment.semantic_sha256(),
                    environment=environment,
                )
            else:
                print("Preparing the analysis container from the release toolkit...")
                attestation = attest_executor_from_toolkit(
                    release / toolkit.filename,
                    expected_toolkit_sha256=toolkit.sha256,
                    image_reference=image,
                )
            if attestation_path.is_file():
                backup = attestation_path.with_name(f"executor-attestation.{sha256_file(attestation_path)[:12]}.json")
                if not backup.exists():
                    shutil.copyfile(attestation_path, backup)
                print(f"Retained previous executor attestation: {backup}")
            write_json_model(attestation_path, attestation)
        if attestation is not None:
            if attestation.environment_sha256 != manifest.executor.environment_sha256:
                raise ValueError("Local executor attestation differs from the campaign environment")
            if isinstance(manifest, ReleaseSetManifestV1) and not any(
                item.kind == "public_toolkit" and item.sha256 == attestation.toolkit_sha256
                for item in manifest.artifacts
            ):
                raise ValueError("Local executor attestation identifies a different software toolkit")
            payload["executor_image_id"] = attestation.image_id
        else:
            payload["executor_image_id"] = manifest.executor.image_id
    from trialagentbench_verifier.execution_contracts import DEFAULT_COMPLETION_TOKENS

    from trialagentbench.adapters.provider_capabilities import campaign_reservations, snapshot_openrouter_capability

    models = payload.get("models")
    if not isinstance(models, list):
        raise ValueError("Campaign models must be a list")
    for model in models:
        if isinstance(model, dict) and model.get("provider") == "offline_control":
            continue
        if (
            not isinstance(model, dict)
            or not isinstance(model.get("model"), str)
            or not isinstance(model.get("upstream_provider"), str)
        ):
            raise ValueError("Each campaign model needs its model ID and upstream provider")
        model_id, upstream_provider = model["model"], model["upstream_provider"]
        assert isinstance(model_id, str) and isinstance(upstream_provider, str)
        snapshot = model.get("capability_snapshot")
        if snapshot is not None and not isinstance(snapshot, str):
            raise ValueError("capability_snapshot must be a path string")
        capability_path = (
            (parent / snapshot).resolve(strict=True)
            if snapshot is not None
            else snapshot_openrouter_capability(
                model=model_id,
                upstream_provider=upstream_provider,
                directory=parent / ".trialagentbench-capabilities",
            )
        )
        model["capability_snapshot"] = str(capability_path)
        reservation_fields = {"reservation_micro_usd", "reservation_tokens", "pricing_premise"}
        supplied = reservation_fields.intersection(model)
        if supplied and supplied != reservation_fields:
            raise ValueError("Supply all three reservation fields together, or omit them for automatic route pricing")
        if not supplied:
            completion = model.get("maximum_completion_tokens", DEFAULT_COMPLETION_TOKENS)
            if not isinstance(completion, int) or isinstance(completion, bool) or completion < 1:
                raise ValueError("maximum_completion_tokens must be a positive integer")
            money, tokens, premise, pricing = campaign_reservations(capability_path, completion_tokens=completion)
            operations = payload.get("operations")
            limits = (
                model.get("spend_limit_micro_usd"),
                model.get("tokens_per_minute"),
                operations.get("spend_limit_micro_usd") if isinstance(operations, dict) else None,
            )
            if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in limits):
                raise ValueError("Campaign/model budgets and tokens_per_minute must be positive integers")
            model_budget, token_window, campaign_budget = limits
            assert isinstance(model_budget, int) and isinstance(token_window, int) and isinstance(campaign_budget, int)
            model.update(
                reservation_micro_usd=min(money, model_budget, campaign_budget),
                reservation_tokens=min(tokens, token_window),
                pricing_premise=premise,
                request_pricing=pricing.model_dump(mode="json"),
            )
    payload["release_directory"] = str(release)
    return CampaignConfigurationV1.model_validate(payload)


def _local_executor_image_exists(image: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=DOCKER_CONTROL_TIMEOUT_SECONDS,
    )
    if result.returncode and "No such image" not in result.stderr:
        raise RuntimeError(f"Docker image inspection failed: {result.stderr.strip()}")
    return result.returncode == 0


def _retain_campaign_executor(configuration: CampaignConfigurationV1, root: Path, plan: CampaignPlanV1) -> str:
    """Retain the pinned image under a unique campaign tag and show the portable backup command."""
    identity = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:12]
    tag = f"trialagentbench/executor-campaign:{identity}-{str(plan.checksum)[:12]}"
    subprocess.run(
        ["docker", "image", "tag", configuration.executor_image_id, tag],
        check=True,
        timeout=DOCKER_CONTROL_TIMEOUT_SECONDS,
    )
    print(f"Retained campaign executor: {tag}")
    print(
        "Save it before running or pruning Docker images: "
        + shlex.join(["docker", "image", "save", "--output", str(root.resolve() / "executor-image.tar"), tag])
    )
    return tag


def _native_arguments(
    configuration: CampaignConfigurationV1,
    manifest: ReleaseSetManifestV1 | DevelopmentCampaignInputsV1,
    *,
    model: CampaignModelV1,
    benchmark: str,
    repetition: int,
    output: Path,
    resume: Path | None = None,
    retry_task_ids: tuple[str, ...] = (),
    retry_replicate_id: str = RETRY_REQUEST_REPLICATE_ID,
) -> list[str]:
    """Translate one public configuration into the existing native command contracts.

    Nonempty ``retry_task_ids`` select the prespecified TrialEval retry run:
    only those tasks, under a separately labelled retry request replicate.
    """
    panel = configuration.panel
    archives = {artifact.filename: artifact.sha256 for artifact in manifest.artifacts}
    filename = role_filename(
        archives, benchmark, _role(panel, "participant" if benchmark == "trialeval" else "evaluator")
    )
    arguments = [
        "--participant-release" if benchmark == "trialeval" else "--release",
        str(configuration.release_directory / filename),
        "--expected-sha256",
        archives[filename],
        "--model",
        model.model,
        "--provider",
        model.provider,
        "--expected-executor-image-id",
        configuration.executor_image_id,
        "--executor-image",
        configuration.executor_image_id,
        "--expected-executor-environment-sha256",
        manifest.executor.environment_sha256,
        "--condition-id",
        panel.condition_id,
        "--analysis-language",
        panel.analysis_language,
        "--request-replicate-id",
        retry_replicate_id if retry_task_ids else f"request-{repetition}",
        "--max-tokens",
        str(model.maximum_completion_tokens),
        "--request-timeout-seconds",
        str(model.request_timeout_seconds),
        "--provider-max-attempts",
        str(model.retry.maximum_attempts),
        "--provider-retry-base-delay-seconds",
        str(model.retry.base_delay_seconds),
        "--provider-unavailability-policy",
        model.retry.unavailability_policy,
        "--output-dir" if benchmark == "trialeval" else "--output-root",
        str(output),
    ]
    if model.upstream_provider is not None:
        arguments.extend(["--openrouter-provider", model.upstream_provider])
    if benchmark == "trialeval":
        arguments.extend(
            [
                "--turns",
                str(panel.trialeval_turns),
                "--workers",
                str(configuration.operations.task_workers),
                "--item-watchdog-seconds",
                str(panel.watchdog_seconds or TRIALEVAL_RELEASE_BUDGET_V1.wall_time_limit_seconds),
                "--procedure-assistance",
                panel.procedure_assistance,
                "--quiet",
            ]
        )
        if panel.reporting_interface != "v1":
            arguments.extend(["--reporting-interface", panel.reporting_interface])
        if retry_task_ids or panel.trialeval_tasks:
            arguments.extend(["--task-id", *(retry_task_ids or panel.trialeval_tasks)])
    else:
        arguments.extend(
            [
                "--workers",
                str(configuration.operations.task_workers),
                "--max-turns-per-step",
                str(panel.trialdev_turns_per_step),
                "--watchdog-seconds",
                str(panel.watchdog_seconds or TRIALDEV_RELEASE_BUDGET_V1.wall_time_limit_seconds),
            ]
        )
        for task in panel.trialdev_tasks:
            arguments.extend(["--task", task])
    if not model.send_temperature:
        arguments.append("--omit-temperature")
    if model.decoding_seed is not None:
        arguments.extend(["--decoding-seed", str(model.decoding_seed)])
    if model.reasoning_effort is not None:
        arguments.extend(["--reasoning-effort", model.reasoning_effort])
    if model.exclude_reasoning:
        arguments.append("--exclude-returned-reasoning")
    if model.replay_reasoning:
        arguments.append("--replay-returned-reasoning")
    if model.capability_snapshot is not None:
        arguments.extend(["--provider-capability-snapshot", str(model.capability_snapshot)])
    arguments.extend(["--prepare-only"] if resume is None else ["--resume-run-dir", str(resume)])
    return arguments


def _invoke(benchmark: str, arguments: list[str]) -> None:
    # Keep typed route-local deferrals visible to the campaign scheduler. The
    # public CLI renders provider errors safely, but would erase their type here.
    if benchmark == "trialeval":
        from trialagentbench.tools.run.trialeval import main
    else:
        from trialagentbench.tools.run.canonical_trialdev import main

    if main(arguments) != 0:
        raise RuntimeError(f"Native {benchmark} run stopped; retain its checkpoints and resume the campaign")


def _trial_eval_plan(
    configuration: CampaignConfigurationV1, manifest: ReleaseSetManifestV1 | DevelopmentCampaignInputsV1, run: Path
) -> TrialEvalCampaignPlanV1:
    """Project analysis identities from authenticated metadata without copying oracle values."""
    native = read_json_model(TrialEvalRunConfigV1, run / "run_config.json")
    filename = role_filename(
        (row.filename for row in manifest.artifacts), "trialeval", _role(configuration.panel, "evaluator")
    )
    digest = next(row.sha256 for row in manifest.artifacts if row.filename == filename)
    with materialized_release_archive(
        configuration.release_directory / filename,
        expected_sha256=digest,
        task_ids=tuple(native.task_ids) if filename.endswith(".assets.json") else None,
    ) as materialized:
        grading_key = load_trialeval_grading_key(materialized.root, expected_task_ids=tuple(native.task_ids))
        selected = tuple(row for row in grading_key.records if row.task_id in native.task_ids)
        cells = {row.cell_ref: row.series_ref for row in selected}
        # The release allocation determines which questions use disjoint worlds across scenarios.
        release_worlds: dict[str, dict[str, set[str]]] = {}
        for row in grading_key.records:
            release_worlds.setdefault(row.series_ref, {}).setdefault(row.cell_ref, set()).add(row.base_world_id)
        nested = tuple(
            sorted(
                series
                for series in set(cells.values())
                if len(release_worlds[series]) > 1
                and sum(map(len, release_worlds[series].values()))
                == len(set().union(*release_worlds[series].values()))
            )
        )
        condition = native.experiment_condition
        attempts = tuple(
            TrialEvalCampaignAttemptV1(
                attempt_id=f"{native.model}:{condition.request_replicate_id}:{row.task_id}",
                scheduled_unit_id=f"{native.model}:{condition.request_replicate_id}:{row.task_id}",
                attempt_role="primary",
                model_id=native.model,
                run_identity_sha256=native.run_identity_sha256,
                task_id=row.task_id,
                trial_id=row.trial_id,
                base_world_id=row.base_world_id,
                series_ref=row.series_ref,
                cell_ref=row.cell_ref,
                effect_state=row.effect_state,
                context=row.context,
                design_family=row.design_family,
                reconstruction_family=None if row.data_integrity is None else row.data_integrity.resolution_action,
                condition_id=condition.condition_id,
                request_replicate_id=condition.request_replicate_id,
                procedure_assistance=condition.procedure_assistance,
                analysis_specification=trialeval_evidence_factors_v1(row.context).analysis_specification,
                target_ids=row.target_ids,
                target_kind=row.target.kind,
                effect_measure=row.target.binding.effect_measure,
                result_unit=row.target.binding.result_unit,
            )
            for row in selected
        )
        return TrialEvalCampaignPlanV1(
            release_id=grading_key.manifest.release_id,
            release_manifest_sha256=sha256_file(materialized.root / "release_manifest.json"),
            participant_manifest_checksum=native.participant_manifest_checksum,
            prospective_analysis_plan_sha256=canonical_payload_sha256(configuration.model_dump(mode="json")),
            cells=tuple(
                TrialEvalCampaignCellV1(cell_ref=cell, series_ref=series) for cell, series in sorted(cells.items())
            ),
            nested_series_refs=nested,
            attempts=attempts,
        )


def validate_model_capabilities(configuration: CampaignConfigurationV1) -> None:
    """Reject unsupported declared routes and limits before preparing or sending any request."""
    for model in configuration.models:
        if model.provider == "offline_control":
            continue
        if model.capability_snapshot is None:
            raise ValueError("Load campaign configuration to resolve provider capabilities before preparation")
        capability = read_json_model(ProviderModelCapabilityV1, model.capability_snapshot)
        capability.validate_route(
            provider=model.provider, model=model.model, upstream_provider=model.upstream_provider
        )
        if capability.context_window_tokens is None or capability.maximum_completion_tokens is None:
            raise ValueError("Campaign capability records require declared context and completion limits")
        if model.maximum_completion_tokens > capability.maximum_completion_tokens:
            raise ValueError("Configured completion tokens exceed the exact model route capacity")
        if model.reasoning_effort is not None and model.reasoning_effort not in capability.supported_efforts:
            raise ValueError("Configured reasoning effort is unsupported by the exact model route")


class _PreviewModel(BaseModel):
    """Read only the configured model count before model routes have been prepared."""

    model_config = ConfigDict(extra="ignore", frozen=True)
    name: str = Field(min_length=1)


class _PreviewConfiguration(BaseModel):
    """The selection fields of the editable campaign, without execution-readiness claims."""

    model_config = ConfigDict(extra="ignore", frozen=True)
    release_directory: Path
    expected_release_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_mode: Literal["qualified", "development"] = "qualified"
    panel: CampaignPanelV1 = Field(default_factory=CampaignPanelV1)
    models: tuple[_PreviewModel, ...] = Field(min_length=1)


def preview_configuration(path: Path, *, inventory: Path | None = None) -> None:
    """Preview selected participant tasks offline; leave provider and executor checks to prepare."""
    release, payload = _configuration_payload(path)
    configuration = _PreviewConfiguration.model_validate(payload)
    manifest = (
        authenticate_development_inputs(
            release, expected_manifest_sha256=configuration.expected_release_manifest_sha256
        )
        if configuration.input_mode == "development"
        else authenticate_release_set_manifest(
            release, expected_manifest_sha256=configuration.expected_release_manifest_sha256
        )
    )
    anchors = {artifact.filename: artifact.sha256 for artifact in manifest.artifacts}
    with ExitStack() as stack:
        sources: dict[str, Path | ReleaseAssets] = {}
        for benchmark in configuration.panel.benchmarks:
            role: Literal["participant", "evaluator"] = (
                "evaluator" if configuration.input_mode == "development" and benchmark == "trialdev" else "participant"
            )
            filename = role_filename(anchors, benchmark, _role(configuration.panel, role))
            sources[benchmark] = (
                read_asset_catalog(release / filename, expected_sha256=anchors[filename])
                if filename.endswith(".assets.json")
                else stack.enter_context(release_archive(release / filename, expected_sha256=anchors[filename]))
            )
        preview_panel(configuration, inventory=inventory, sources=sources)
    print(f"Analysis language: {configuration.panel.analysis_language}")
    print(
        "Task-selection preview only. Provider routes, prices, budgets and executor readiness are checked by prepare."
    )
    print("No model requests or provider metadata lookups were made. TrialEval world identities are evaluator-only.")


def _preview_rows(
    benchmark: str, source: Path | ReleaseAssets, selectors: Sequence[str]
) -> tuple[tuple[str, str, str, str], ...]:
    """Return (task, world or case, presentation, stream) rows for one selected benchmark."""
    from trialagentbench.tools.run.canonical_trialdev import _selected_tasks
    from trialagentbench.tools.run.trialeval import _selected_task_ids

    if isinstance(source, ReleaseAssets):
        if benchmark == "trialeval":
            task_ids = set(_catalog_task_ids(source, selectors))
            return tuple(
                (task.task_id, "", str(task.context_tier), "completed_trial")
                for task in source.tasks
                if task.task_id in task_ids
            )
        return tuple(
            (task.task_id, task.case_id, "raw" if task.main else "analysis_ready", str(task.stream))
            for task in _selected_tasks(source.tasks, selectors)
            if isinstance(task, AssetTask)
        )
    with ZipFile(source) as archive:
        names = archive.namelist()
        if benchmark == "trialeval":
            if TRIALEVAL_GRADING_KEY_RECORDS in names:
                records = tuple(
                    TrialEvalGradingKeyRecordV1.model_validate_json(line)
                    for line in archive.read(TRIALEVAL_GRADING_KEY_RECORDS).splitlines()
                )
                te_rows = tuple((row.task_id, row.base_world_id, row.context) for row in records)
            else:
                public_tasks = tuple(
                    ParticipantTask.model_validate_json(archive.read(name))
                    for name in sorted(names)
                    if name.startswith("tasks/") and name.endswith("/task.json")
                )
                te_rows = tuple(
                    (task.task_id, "", canonical_task_factors(task).context_configuration) for task in public_tasks
                )
            selected = set(_selected_task_ids({row[0]: row[2] for row in te_rows}, selectors))
            return tuple((*row, "completed_trial") for row in te_rows if row[0] in selected)
        tasks = tuple(
            TrialDevTaskV1.model_validate_json(archive.read(name))
            for name in names
            if name.startswith(("participant/tasks/", "tasks/")) and name.endswith("/task.json")
        )
        return tuple(
            (task.task_id, task.case_id, task.data_view, task.stream.value)
            for task in _selected_tasks(tasks, selectors)
        )


def preview_panel(
    configuration: CampaignConfigurationV1 | _PreviewConfiguration,
    *,
    inventory: Path | None = None,
    sources: Mapping[str, Path | ReleaseAssets] | None = None,
) -> None:
    """Describe and validate selectors against already authenticated release bytes.

    Without explicit sources, each benchmark's evaluator transport is resolved
    from the authenticated release manifest, as during preparation.
    """
    panel = configuration.panel
    anchors: dict[str, str] = {}
    if sources is None:
        if not isinstance(configuration, CampaignConfigurationV1):
            raise ValueError("A selection-only preview requires authenticated participant sources")
        anchors = {artifact.filename: artifact.sha256 for artifact in _campaign_inputs(configuration).artifacts}
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(
        ("benchmark", "task_id", "world_or_case", "presentation", "stream", "models", "repetitions", "attempts")
    )
    for benchmark in panel.benchmarks:
        selectors = panel.trialeval_tasks if benchmark == "trialeval" else panel.trialdev_tasks
        source: Path | ReleaseAssets
        if sources is not None:
            source = sources[benchmark]
        else:
            filename = role_filename(anchors, benchmark, _role(panel, "evaluator"))
            source = (
                read_asset_catalog(configuration.release_directory / filename, expected_sha256=anchors[filename])
                if filename.endswith(".assets.json")
                else configuration.release_directory / filename
            )
        rows = _preview_rows(benchmark, source, selectors)
        count = len(rows)
        if count == 0:
            raise ValueError(f"The selected {benchmark} release has no scheduled tasks")
        attempts = count * len(configuration.models) * panel.repetitions
        print(f"{benchmark}: {count} tasks/model/repetition; {attempts} scheduled attempts")
        for presentation in sorted({row[2] for row in rows}):
            print(f"  {presentation}: {sum(row[2] == presentation for row in rows)} tasks")
        for task_id, case_id, presentation, stream in rows[:10]:
            print(f"  {task_id} | {case_id} | {presentation} | {stream}")
        if count > 10:
            print(f"  ... {count - 10} additional tasks; use --inventory to save the complete selection.")
        for row in rows:
            writer.writerow(
                (
                    benchmark,
                    *row,
                    len(configuration.models),
                    panel.repetitions,
                    len(configuration.models) * panel.repetitions,
                )
            )
        cache_name = os.environ.get("TRIALAGENTBENCH_CACHE_DIR")
        if isinstance(source, ReleaseAssets) and isinstance(configuration, CampaignConfigurationV1):
            if not cache_name:
                print("Set TRIALAGENTBENCH_CACHE_DIR for reusable selective acquisition and its storage estimate.")
                continue
            for role in ("participant", "evaluator"):
                role_name = role_filename(anchors, benchmark, _role(panel, role))
                footprint = cache_footprint(
                    configuration.release_directory / role_name,
                    expected_sha256=anchors[role_name],
                    cache=Path(cache_name),
                    task_ids=tuple(row[0] for row in rows),
                    workers=configuration.operations.executor_capacity,
                )
                print(f"{benchmark} {role} storage (bytes): {footprint.model_dump_json()}")
    if inventory is not None:
        from trialagentbench.tools.finish_run import _write_once

        _write_once(inventory, output.getvalue())
        print(f"Exact task inventory: {inventory}")
    print(
        "Storage estimates exclude Docker images and host dependencies; retained run outputs grow with execution. "
        "Role bounds overlap; concurrent executor capacity is already included in each scratch bound."
    )
    print("Diagnostic subsets retain individual grades; insufficient linked panels withhold inferential comparisons.")


def prepare_campaign(configuration: CampaignConfigurationV1, root: Path) -> CampaignPlanV1:
    """Freeze every native invocation before permitting provider execution."""
    validate_model_capabilities(configuration)
    inputs = _campaign_inputs(configuration)
    preview_panel(configuration)
    with release_assets(
        configuration.release_directory, task_selections=_campaign_asset_selections(configuration, inputs)
    ):
        manifest = _campaign_inputs(configuration, full_verification=True)
        root.mkdir(parents=True, exist_ok=False)
        with campaign_lock(root):
            write_json_model(root / "configuration.json", configuration)
            runs = tuple(
                _prepare_native_run(
                    root,
                    configuration,
                    manifest,
                    model=model,
                    benchmark=benchmark,
                    repetition=repetition,
                    job_id=f"{model.name}-{benchmark}-{repetition}",
                )
                for repetition in range(1, configuration.panel.repetitions + 1)
                for benchmark in configuration.panel.benchmarks
                for model in configuration.models
            )
            plan = CampaignPlanV1(configuration=configuration, runs=runs)
            _retain_campaign_executor(configuration, root, plan)
            write_json_model(root / "campaign.json", plan)
        return plan


def _prepare_native_run(
    root: Path,
    configuration: CampaignConfigurationV1,
    manifest: ReleaseSetManifestV1 | DevelopmentCampaignInputsV1,
    *,
    model: CampaignModelV1,
    benchmark: Literal["trialeval", "trialdev"],
    repetition: int,
    job_id: str,
    retry_of_job_id: str | None = None,
    retry_task_ids: tuple[str, ...] = (),
    retry_replicate_id: str = RETRY_REQUEST_REPLICATE_ID,
    reclassification_audit_path: str | None = None,
) -> CampaignRunV1:
    """Freeze one native invocation and, for TrialEval, its hashed analysis plan."""
    output = root / "runs" / job_id
    _invoke(
        benchmark,
        _native_arguments(
            configuration,
            manifest,
            model=model,
            benchmark=benchmark,
            repetition=repetition,
            output=output,
            retry_task_ids=retry_task_ids,
            retry_replicate_id=retry_replicate_id,
        ),
    )
    custody = "run_config.json" if benchmark == "trialeval" else "invocation.json"
    candidates = tuple(output.rglob(custody))
    if len(candidates) != 1:
        raise ValueError("Native preparation must produce exactly one frozen invocation")
    run = candidates[0].parent
    plan_path = None
    plan_sha = None
    if benchmark == "trialeval":
        plan_path = f"plans/{job_id}.json"
        write_json_model(root / plan_path, _trial_eval_plan(configuration, manifest, run))
        plan_sha = sha256_file(root / plan_path)
    return CampaignRunV1(
        job_id=job_id,
        benchmark=benchmark,
        model_name=model.name,
        repetition=repetition,
        relative_run_root=run.relative_to(root).as_posix(),
        invocation_sha256=sha256_file(candidates[0]),
        analysis_plan_path=plan_path,
        analysis_plan_sha256=plan_sha,
        retry_of_job_id=retry_of_job_id,
        reclassification_audit_path=reclassification_audit_path,
        reclassification_audit_sha256=(
            None if reclassification_audit_path is None else sha256_file(root / reclassification_audit_path)
        ),
    )


def load_campaign(root: Path) -> CampaignPlanV1:
    """Authenticate saved configuration, native invocations and analysis plans."""
    plan = read_json_model(CampaignPlanV1, root / "campaign.json")
    if read_json_model(CampaignConfigurationV1, root / "configuration.json") != plan.configuration:
        raise ValueError("Editable configuration differs from the prepared campaign")
    retried_tasks: dict[str, set[str]] = {}
    for run in plan.runs:
        custody = "run_config.json" if run.benchmark == "trialeval" else "invocation.json"
        if sha256_file(root / run.relative_run_root / custody) != run.invocation_sha256:
            raise ValueError("Native invocation differs from the prepared campaign")
        if (
            run.analysis_plan_path is not None
            and sha256_file(root / run.analysis_plan_path) != run.analysis_plan_sha256
        ):
            raise ValueError("Analysis plan differs from the prepared campaign")
        if run.reclassification_audit_path is not None:
            if sha256_file(root / run.reclassification_audit_path) != run.reclassification_audit_sha256:
                raise ValueError("Reclassification audit differs from the prepared retry")
        if run.retry_of_job_id is not None:
            tasks = set(read_json_model(TrialEvalRunConfigV1, root / run.relative_run_root / custody).task_ids)
            prior = retried_tasks.setdefault(run.retry_of_job_id, set())
            if tasks & prior:
                raise ValueError("A TrialEval primary task has already been retried")
            prior.update(tasks)
    return plan


def check_host_capacity(capacity: int, *, host_memory_mb: int, host_cpus: int) -> None:
    """Refuse an executor capacity whose containers cannot all fit on this host.

    One task at a time (the two executor slots a task needs) is always allowed, so the default
    configuration runs on any host; above it, every container must fit the host's memory and CPUs.

    Parameters
    ----------
    capacity
        Configured simultaneous analysis containers (``operations.executor_capacity``).
    host_memory_mb
        Total physical memory of the host, in MiB.
    host_cpus
        Logical CPUs of the host.

    Raises
    ------
    ValueError
        If more than one task's capacity times the fixed per-container memory or CPU limit exceeds the host.

    Examples
    --------
    >>> check_host_capacity(4, host_memory_mb=65536, host_cpus=16)
    >>> check_host_capacity(2, host_memory_mb=4096, host_cpus=2)
    """
    if capacity <= EXECUTOR_SLOTS_PER_TASK:
        return
    limits = RELEASE_EXECUTION_LIMITS_V1
    setting = (
        "prepare a new campaign with a lower operations.executor_workers (or model_workers x task_workers when it is "
        "unset); a prepared campaign's operations are fixed"
    )
    if capacity * limits.memory_mb > host_memory_mb:
        raise ValueError(
            f"Executor capacity {capacity} x {limits.memory_mb} MB per container exceeds this host's "
            f"{host_memory_mb} MB of memory; {setting}"
        )
    if capacity * limits.cpu_count > host_cpus:
        raise ValueError(
            f"Executor capacity {capacity} x {limits.cpu_count:g} CPUs per container exceeds this host's "
            f"{host_cpus} CPUs; {setting}"
        )


def _host_resources() -> tuple[int, int]:
    """Return this host's total memory in MiB and its logical CPU count."""
    cpus = os.cpu_count()
    if cpus is None or not hasattr(os, "sysconf"):
        raise RuntimeError(
            "Cannot read this host's CPUs and memory for the executor capacity check; prepare a new campaign that "
            "runs one task at a time"
        )
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 2**20, cpus


def _require_host_capacity(configuration: CampaignConfigurationV1) -> None:
    """Refuse, before any container starts, a capacity this host cannot hold; one task at a time never reads it."""
    if configuration.operations.executor_capacity <= EXECUTOR_SLOTS_PER_TASK:
        return
    host_memory_mb, host_cpus = _host_resources()
    check_host_capacity(configuration.operations.executor_capacity, host_memory_mb=host_memory_mb, host_cpus=host_cpus)


def run_campaign(root: Path) -> None:
    """Resume the native schedulers while retaining every scientific primary."""
    with campaign_lock(root):
        plan = load_campaign(root)
        configuration = plan.configuration
        _require_host_capacity(configuration)
        originals = {run.job_id: run for run in plan.runs if run.retry_of_job_id is None}
        for retry in plan.runs:
            if retry.retry_of_job_id is None:
                continue
            primary_root = root / originals[retry.retry_of_job_id].relative_run_root
            primary_config = read_json_model(TrialEvalRunConfigV1, primary_root / "run_config.json")
            retry_config = read_json_model(TrialEvalRunConfigV1, root / retry.relative_run_root / "run_config.json")
            if any(
                not _retryable_trialeval_primary(primary_root, primary_config, task_id)
                for task_id in retry_config.task_ids
            ):
                raise ValueError("TrialEval retry is permitted only after an infrastructure failure of its primary")
        inputs = _campaign_inputs(configuration)
        with release_assets(
            configuration.release_directory, task_selections=_campaign_asset_selections(configuration, inputs)
        ):
            validate_model_capabilities(configuration)
            manifest = _campaign_inputs(configuration)
            models = {model.name: model for model in configuration.models}
            if any(model.provider == "openrouter" for model in models.values()) and not os.environ.get(
                "OPENROUTER_API_KEY"
            ):
                raise ValueError("Set OPENROUTER_API_KEY in the environment before running this campaign")

            def arguments(run: CampaignRunV1) -> list[str]:
                native = root / run.relative_run_root
                return _native_arguments(
                    configuration,
                    manifest,
                    model=models[run.model_name],
                    benchmark=run.benchmark,
                    repetition=run.repetition,
                    output=root / "runs" / run.job_id,
                    resume=native,
                    retry_task_ids=(
                        ()
                        if run.retry_of_job_id is None
                        else tuple(read_json_model(TrialEvalRunConfigV1, native / "run_config.json").task_ids)
                    ),
                    retry_replicate_id=(
                        RETRY_REQUEST_REPLICATE_ID
                        if run.retry_of_job_id is None
                        else read_json_model(
                            TrialEvalRunConfigV1, native / "run_config.json"
                        ).experiment_condition.request_replicate_id
                    ),
                )

            for run in plan.runs:
                _invoke(run.benchmark, [*arguments(run), "--prepare-only"])
            control = CampaignRequestControl(
                database=root / "requests.sqlite", configuration=configuration, model_name=configuration.models[0].name
            )
            control.set_cancelled(False)

            def execute(run: CampaignRunV1) -> None:
                try:
                    _invoke(
                        run.benchmark, [*arguments(run), "--campaign-root", str(root), "--campaign-job", run.job_id]
                    )
                except CampaignCapacityStop as error:
                    if error.campaign_wide:
                        raise
                    print(
                        f"Paused {run.job_id}: {error}. Other routes continue; retain pending checkpoints.", flush=True
                    )

            with executor_session_capacity(configuration.operations.executor_capacity):
                if configuration.dispatch is not None:
                    from trialagentbench.util.campaign_dispatch import dispatch_campaign

                    def execute_slice(run: CampaignRunV1, task_ids: tuple[str, ...]) -> None:
                        flags = [flag for task_id in task_ids for flag in ("--dispatch-task-id", task_id)]
                        _invoke(
                            run.benchmark,
                            [*arguments(run), "--campaign-root", str(root), "--campaign-job", run.job_id, *flags],
                        )

                    dispatch_campaign(
                        root=root, plan=plan, execute=execute_slice, cancelled=lambda: control.set_cancelled(True)
                    )
                    return
                run_bounded_tasks(
                    plan.runs,
                    execute,
                    workers=configuration.operations.model_workers,
                    consume=lambda _: None,
                    stop=RunStopSignal(),
                    request_control=control,
                )


def _finish_native_runs(
    root: Path,
    plan: CampaignPlanV1,
    manifest: ReleaseSetManifestV1 | DevelopmentCampaignInputsV1,
    runs: Sequence[CampaignRunV1],
) -> None:
    """Grade, verify and export native runs offline; a repeated call reuses completed stages.

    Tasks left pending by an exhausted spending limit are budget-censored:
    reported and unscored. Any other pending task refuses finishing.
    """
    from trialagentbench.tools.finish_campaign_analysis import campaign_censoring
    from trialagentbench.tools.finish_run import main as finish

    configuration = plan.configuration
    censoring = campaign_censoring(root, plan)
    censored = {} if censoring is None else {row.job_id: row.censored_task_ids for row in censoring.runs}
    unrun = set() if censoring is None else {row.job_id for row in censoring.runs if row.wholly_censored}

    def execute(run: CampaignRunV1) -> None:
        if run.benchmark == "trialeval" and run.job_id in unrun:
            return  # No TrialEval item ran, so there is nothing to grade, replay or export.
        arguments = [
            run.benchmark,
            "--release-dir",
            str(configuration.release_directory),
            "--expected-manifest-sha256",
            configuration.expected_release_manifest_sha256,
            "--run-root",
            str(root / run.relative_run_root),
            "--output",
            str(root / "results" / run.job_id),
            "--resume",
        ]
        if configuration.panel.split != "public":
            arguments.extend(["--split", configuration.panel.split])
        if run.benchmark == "trialeval":
            arguments.extend(
                [
                    "--executor-image-id",
                    configuration.executor_image_id,
                    "--campaign-plan",
                    str(root / str(run.analysis_plan_path)),
                    "--expected-plan-sha256",
                    str(run.analysis_plan_sha256),
                ]
            )
            for task_id in censored.get(run.job_id, ()):
                arguments.extend(["--budget-censored", task_id])
        else:
            arguments.extend(["--expected-invocation-sha256", run.invocation_sha256])
        if isinstance(manifest, DevelopmentCampaignInputsV1):
            finish([*arguments, "--development"])
        else:
            finish(arguments, qualified_manifest=manifest)

    with executor_session_capacity(configuration.operations.executor_capacity, queue=True):
        from trialagentbench.util.processes import process_map

        process_map(execute, runs, workers=configuration.operations.grading_workers)


def _retryable_trialeval_primary(native: Path, config: TrialEvalRunConfigV1, task_id: str) -> bool:
    """Bind retry admission to the original native terminal, including on resume."""
    record = read_json_model(TrialEvalItemResultV1, native / "items" / f"{task_id}.json")
    if record.item_id != task_id or record.run_config != config:
        raise ValueError("TrialEval retry checkpoint differs from its frozen native identity")
    return (
        record.agent_output.status == "failed"
        and trialeval_infrastructure_outcome_v1(record.agent_output.failure_type) is not None
    )


def retry_campaign(root: Path, *, reclassification_audit: Path | None = None) -> None:
    """Retry once, then run, every TrialEval primary that ended in an infrastructure failure.

    After all scheduled tasks reach an end state, each repetition-one TrialEval
    job's validated native checkpoints determine infrastructure eligibility.
    Numerical grading remains the offline finalizer's responsibility. Infrastructure-failed
    tasks not in any earlier retry run receive one separately labelled retry.
    A supplementary run requires a retained reclassification audit. Model
    outcomes and previously retried tasks are never scheduled again.

    Raises
    ------
    ValueError
        If comparisons were already finished, or work is pending without an exhausted spending limit.
    """
    from trialagentbench.tools.finish_campaign_analysis import campaign_censoring

    with campaign_lock(root):
        plan = load_campaign(root)
        audit_path = (
            None
            if reclassification_audit is None
            else reclassification_audit.resolve(strict=True).relative_to(root.resolve()).as_posix()
        )
        if (root / "comparisons").exists():
            raise ValueError("Infrastructure retries must precede campaign finish; comparisons already exist")
        campaign_censoring(root, plan)  # Refuse pending work without declared censoring.
        configuration = plan.configuration
        _require_host_capacity(configuration)
        models = {model.name: model for model in configuration.models}
        primaries = tuple(
            run
            for run in plan.runs
            if run.benchmark == "trialeval" and run.repetition == 1 and run.retry_of_job_id is None
        )
        inputs = _campaign_inputs(configuration)
        with release_assets(
            configuration.release_directory, task_selections=_campaign_asset_selections(configuration, inputs)
        ):
            manifest = _campaign_inputs(configuration)
            for run in primaries:
                previous = tuple(retry for retry in plan.runs if retry.retry_of_job_id == run.job_id)
                retried = {
                    task_id
                    for retry in previous
                    for task_id in read_json_model(
                        TrialEvalRunConfigV1, root / retry.relative_run_root / "run_config.json"
                    ).task_ids
                }
                native = root / run.relative_run_root
                native_config = read_json_model(TrialEvalRunConfigV1, native / "run_config.json")
                coverage = read_json_model(RunCoverageV1, native / "coverage.json")
                if coverage.run_identity_sha256 != native_config.run_identity_sha256 or coverage.unit_ids != tuple(
                    native_config.task_ids
                ):
                    raise ValueError("TrialEval retry coverage differs from its frozen native identity")
                failed_tasks = []
                for task_id in coverage.completed_unit_ids:
                    if task_id not in retried and _retryable_trialeval_primary(native, native_config, task_id):
                        failed_tasks.append(task_id)
                failed = tuple(failed_tasks)
                if not failed:
                    continue
                if previous and audit_path is None:
                    raise ValueError("Supplementary infrastructure retry runs require a reclassification audit")
                retry_number = len(previous) + 1
                retry = _prepare_native_run(
                    root,
                    configuration,
                    manifest,
                    model=models[run.model_name],
                    benchmark="trialeval",
                    repetition=1,
                    job_id=f"{run.job_id}-retry" + (f"-{retry_number}" if previous else ""),
                    retry_of_job_id=run.job_id,
                    retry_task_ids=failed,
                    retry_replicate_id=f"retry-{retry_number}",
                    reclassification_audit_path=audit_path,
                )
                plan = CampaignPlanV1(configuration=configuration, runs=(*plan.runs, retry))
                write_json_model(root / "campaign.json", plan)
                print(f"{run.job_id}: retrying {len(failed)} infrastructure-failed primaries as {retry.job_id}")
    if any(run.retry_of_job_id is not None for run in plan.runs):
        run_campaign(root)
    else:
        print("No TrialEval primary ended in an infrastructure failure; nothing to retry.")


def finish_campaign(
    root: Path,
    *,
    replay_bank: tuple[Path, str] | None = None,
    replay_workers: int | None = None,
    oracle_values: tuple[Path, str] | None = None,
) -> None:
    """Grade native runs offline, export once and compute campaign comparisons.

    Tasks left pending by an exhausted spending limit are budget-censored:
    reported and unscored. Any other pending task refuses finishing.
    ``replay_bank`` (directory, expected manifest SHA-256) scores TrialEval
    answers by replay at the true effect, ``replay_workers`` replays at once
    (default: the campaign's executor capacity).
    """
    from trialagentbench_verifier.behaviour import reproduce_behaviour_figures

    from trialagentbench.tools.analyse_calibration import main as analyse_resources
    from trialagentbench.tools.build.build_trace_analysis_bundle import build_trace_analysis_bundle
    from trialagentbench.tools.finish_run import _discard_incomplete_writes, _write_once

    require_harness_scorer_v1()
    with campaign_lock(root):
        _discard_incomplete_writes(root)
        plan = load_campaign(root)
        configuration = plan.configuration
        if (root / "regrade.json").is_file():
            from trialagentbench.tools.regrade_campaign import CampaignRegradeV1

            regrade = read_json_model(CampaignRegradeV1, root / "regrade.json")
            if regrade.primary_selection_sha256 is not None:
                from trialagentbench.tools.primary_selection import PRIMARY_SELECTION

                if sha256_file(root / PRIMARY_SELECTION) != regrade.primary_selection_sha256:
                    raise ValueError("Derived primary-selection manifest changed after regrade admission")
        _require_host_capacity(configuration)
        inputs = _campaign_inputs(configuration)
        with release_assets(
            configuration.release_directory, task_selections=_campaign_asset_selections(configuration, inputs)
        ):
            manifest = inputs
            _finish_native_runs(root, plan, manifest, plan.runs)
            from trialagentbench.tools.finish_campaign_analysis import finish_comparisons, write_campaign_index

            finish_comparisons(
                root,
                plan,
                manifest,
                replay_bank=replay_bank,
                replay_workers=replay_workers,
                oracle_values=oracle_values,
            )
            with tempfile.TemporaryDirectory(prefix="trialagentbench-behaviour-") as temporary:
                temporary_root = Path(temporary)
                bundle = build_trace_analysis_bundle(
                    out_dir=temporary_root / "behaviour",
                    trialeval_root=root / "results" if "trialeval" in configuration.panel.benchmarks else None,
                    trialdev_root=root / "results" if "trialdev" in configuration.panel.benchmarks else None,
                )
                reproduce_behaviour_figures(
                    bundle,
                    expected_manifest_sha256=sha256_file(bundle / "trace_bundle_manifest.json"),
                    output=temporary_root / "behaviour-summary",
                )
                analyse_resources(
                    [
                        *(str(root / run.relative_run_root) for run in plan.runs),
                        "--out-dir",
                        str(temporary_root / "resources"),
                    ]
                )
                for name in ("behaviour", "behaviour-summary", "resources"):
                    destination = root / name
                    destination.mkdir(exist_ok=True)
                    _discard_incomplete_writes(destination)
                    sources = tuple((temporary_root / name).iterdir())
                    if {path.name for path in destination.iterdir()} - {path.name for path in sources}:
                        raise ValueError("Retained behaviour output contains undeclared artifacts")
                    for source in sources:
                        _write_once(destination / source.name, source.read_bytes())
            # The resource summary above names its temporary directory; the retained copy is here.
            print(f"Resource calibration retained in {root / 'resources'}")
            index = write_campaign_index(root, plan)
            print(f"Campaign finished. Open {index}")
            print(
                "Share only the designated public-results.json summaries when grading references must remain private."
            )


def main(argv: Sequence[str] | None = None) -> int:
    """Operate one saved native campaign without reconstructing model commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("preview", "prepare", "run", "status", "resume", "retry", "finish", "regrade")
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--output", type=Path, help="New output root for retained-output regrading.")
    parser.add_argument(
        "--reclassification-audit",
        type=Path,
        help="Retained audit within this campaign, required for supplementary infrastructure retries.",
    )
    parser.add_argument(
        "--release",
        type=Path,
        help="Regrade against this release set; its campaign inputs and executor must equal the original's.",
    )
    parser.add_argument(
        "--inventory", type=Path, help="Save the complete resolved task selection as CSV (preview only)."
    )
    parser.add_argument(
        "--oracle-values", type=Path, help="Release-bound construction oracle values (finish/regrade)."
    )
    parser.add_argument("--oracle-values-sha256", help="Pinned construction oracle input SHA-256.")
    parser.add_argument("--replay-bank", type=Path, help="TrialEval replay bank directory (finish and regrade).")
    parser.add_argument("--primary-selection", type=Path, help="Audited earlier-completion manifest (regrade).")
    parser.add_argument("--replay-bank-sha256", help="Published SHA-256 of the replay bank's manifest.json.")
    parser.add_argument(
        "--replay-workers",
        type=int,
        help="Simultaneous replay containers (default: the campaign's executor capacity); throughput only.",
    )
    args = parser.parse_args(argv)
    if args.reclassification_audit is not None and args.action != "retry":
        parser.error("--reclassification-audit is available only for retry")
    if (args.oracle_values is None) != (args.oracle_values_sha256 is None):
        parser.error("--oracle-values and --oracle-values-sha256 go together")
    if args.oracle_values is not None and args.action not in {"finish", "regrade"}:
        parser.error("--oracle-values is available only for finish and regrade")
    oracle_values = (
        None if args.oracle_values is None else (args.oracle_values.resolve(strict=True), args.oracle_values_sha256)
    )
    if (args.replay_bank is None) != (args.replay_bank_sha256 is None):
        parser.error("--replay-bank and --replay-bank-sha256 go together")
    if args.replay_bank is not None and args.action not in {"finish", "regrade"}:
        parser.error("--replay-bank is available only for finish and regrade")
    if args.replay_workers is not None and (args.replay_bank is None or args.replay_workers < 1):
        parser.error("--replay-workers requires --replay-bank and must be positive")
    replay_bank = (
        None if args.replay_bank is None else (args.replay_bank.resolve(strict=True), args.replay_bank_sha256)
    )
    if (args.output is not None) != (args.action == "regrade"):
        parser.error("regrade requires --output; other actions do not accept it")
    if args.release is not None and args.action != "regrade":
        parser.error("--release is available only for regrade")
    if args.primary_selection is not None and args.action != "regrade":
        parser.error("--primary-selection is available only for regrade")
    if args.inventory is not None and args.action != "preview":
        parser.error("--inventory is available only for preview")
    if args.action in {"preview", "prepare"}:
        if args.config is None:
            parser.error("preview and prepare require --config")
        if args.action == "preview":
            preview_configuration(args.config, inventory=args.inventory)
            return 0
        configuration = load_configuration(args.config, prepare_executor=True)
        if args.campaign is None:
            parser.error("prepare requires a new --campaign directory")
        plan = prepare_campaign(configuration, args.campaign.resolve())
        print(f"Prepared {len(plan.runs)} native runs; campaign identity {plan.checksum}")
        return 0
    if args.config is not None or args.campaign is None:
        parser.error("Use only --campaign for a saved campaign lifecycle operation")
    root = args.campaign.resolve(strict=True)
    if args.action in {"run", "resume", "retry"}:
        try:
            if args.action == "retry":
                retry_campaign(root, reclassification_audit=args.reclassification_audit)
            else:
                run_campaign(root)
        except CampaignCapacityStop as stopped:
            print(f"Campaign paused: {stopped}")
            return 3
        plan = load_campaign(root)
        for run in plan.runs:
            coverage = read_json_model(RunCoverageV1, root / run.relative_run_root / "coverage.json")
            if coverage.completed_unit_ids != coverage.unit_ids:
                print(
                    "Campaign retains pending checkpoints; inspect status before resuming. "
                    "Pending tasks of a model whose spending limit is exhausted are budget-censored; finish reports them."
                )
                return 3
    elif args.action == "regrade":
        from trialagentbench.tools.regrade_campaign import regrade_campaign

        regrade_campaign(
            root,
            args.output,
            release=args.release,
            replay_bank=replay_bank,
            replay_workers=args.replay_workers,
            primary_selection=args.primary_selection,
            oracle_values=oracle_values,
        )
    elif args.action == "finish":
        finish_campaign(root, replay_bank=replay_bank, replay_workers=args.replay_workers, oracle_values=oracle_values)
    else:
        plan = load_campaign(root)
        for run in plan.runs:
            native_root = root / run.relative_run_root
            coverage = read_json_model(RunCoverageV1, native_root / "coverage.json")
            pending = len(coverage.unit_ids) - len(coverage.completed_unit_ids)
            print(
                f"{run.job_id}: terminal checkpoints={len(coverage.completed_unit_ids)}/{len(coverage.unit_ids)}; "
                f"pending checkpoints={pending}"
            )
        print_spending_status(root / "requests.sqlite", plan.configuration)
    return 0
