"""Independent verification and deterministic scoring of released benchmarks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Generator, Iterable
from contextlib import ExitStack, nullcontext
from pathlib import Path, PurePosixPath
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from trialagentbench_verifier.archive import opened_release
from trialagentbench_verifier.assets import ReleaseAssetScope
from trialagentbench_verifier.contracts import (
    CheckResultV1,
    ExecutionProofV1,
    NormalizedSubmissionV1,
    QualificationReportBundleV1,
    ScoreVerificationBindingV1,
    VerificationReportV1,
)
from trialagentbench_verifier.external_qualification import (
    EXTERNAL_QUALIFICATION_NAME,
    QUALIFICATION_PENDING,
    ExternalTrialDevQualificationV1,
    read_external_qualification,
)
from trialagentbench_verifier.grading_key import (
    GRADING_KEY_MANIFEST,
    TRIALEVAL_GRADING_KEY_RECORDS,
    verify_grading_key,
)
from trialagentbench_verifier.io import (
    VerificationInputError,
    canonical_sha256,
    materialized_release,
    read_json_object,
    release_archive,
    release_asset_scope,
    sha256_file,
)
from trialagentbench_verifier.scorer import current_scorer
from trialagentbench_verifier.units import compatible_units

EvaluationReleaseError = VerificationInputError

_PORTABLE_GRADED_ITEM_PATH_FIELDS = frozenset(
    {"bundle_dir", "input_run_dir", "output_run_dir", "participant_archive", "source_artifact_path", "suite_dir"}
)


def _portable_graded_item(value: JsonValue, *, field: str | None = None) -> JsonValue:
    """Apply the public result-export path projection to a graded item."""

    if isinstance(value, dict):
        return {key: _portable_graded_item(item, field=key) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_graded_item(item, field=field) for item in value]
    if isinstance(value, str) and field in _PORTABLE_GRADED_ITEM_PATH_FIELDS:
        label = Path(value).name
        if not label:
            raise EvaluationReleaseError(f"graded-item field {field!r} has no portable artifact label")
        return label
    return value


def _portable_graded_item_sha256(path: Path) -> str:
    """Hash the exact canonical graded-item bytes written by result export."""

    projected = _portable_graded_item(cast(JsonValue, read_json_object(path)))
    encoded = (json.dumps(projected, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def _release_identity(root: Path) -> tuple[Literal["trialeval", "trialdev"], str, str]:
    """Return stream, candidate digest, and release-manifest digest."""

    manifest_path = root / "release_manifest.json"
    manifest = read_json_object(manifest_path)
    schema_id = manifest.get("schema_id")
    streams: dict[str, Literal["trialeval", "trialdev"]] = {
        "ariadne.trialeval.release/v1": "trialeval",
        "ariadne.trialdev.release/v1": "trialdev",
    }
    if schema_id not in streams:
        raise EvaluationReleaseError("unsupported release manifest schema")
    candidate = manifest.get("construct_sha256")
    if not isinstance(candidate, str) or len(candidate) != 64:
        raise EvaluationReleaseError("release manifest lacks a candidate digest")
    if not isinstance(schema_id, str):
        raise EvaluationReleaseError("release manifest schema identity must be a string")
    scope = release_asset_scope(root)
    digest = scope.catalog.release_manifest_sha256 if scope is not None else sha256_file(manifest_path)
    return streams[schema_id], candidate, digest


def _object_checksum(payload: dict[str, JsonValue], *, exclude_checksum: bool) -> str:
    """Hash a released JSON object with the construction canonicalization."""

    body = dict(payload)
    if exclude_checksum:
        body.pop("checksum", None)
    return canonical_sha256(body)


def _verify_artifact_identity(
    root: Path,
    artifact: object,
    role: str,
    *,
    flattened: bool = False,
) -> str:
    """Verify one role-contained artifact and return its relative path."""

    if not isinstance(artifact, dict) or set(artifact) != {"relative_path", "sha256", "byte_length"}:
        raise EvaluationReleaseError("role manifest contains an invalid artifact identity")
    relative, expected_hash, expected_bytes = (
        artifact["relative_path"],
        artifact["sha256"],
        artifact["byte_length"],
    )
    if not isinstance(relative, str) or not isinstance(expected_hash, str) or not isinstance(expected_bytes, int):
        raise EvaluationReleaseError("role manifest artifact identity has invalid field types")
    portable = PurePosixPath(relative)
    if (
        portable.is_absolute()
        or ".." in portable.parts
        or portable.as_posix() != relative
        or not portable.parts
        or portable.parts[0] != role
    ):
        raise EvaluationReleaseError("role manifest artifact path is not portable or role-contained")
    path = root.joinpath(*portable.parts[1:]) if flattened else root / relative
    resolved_root = root.resolve(strict=True)
    if not path.is_file() or path.is_symlink() or not path.resolve(strict=True).is_relative_to(resolved_root):
        raise EvaluationReleaseError(f"release artifact is missing or unsafe: {relative}")
    if path.stat().st_size != expected_bytes or (
        release_asset_scope(root) is None and sha256_file(path) != expected_hash
    ):
        raise EvaluationReleaseError(f"release artifact changed: {relative}")
    return relative


def _participant_identity(root: Path) -> tuple[Literal["trialeval", "trialdev"], str, str]:
    """Return the stream, construct, and content identity of a participant release."""

    manifest_path = root / "manifest.json"
    manifest = read_json_object(manifest_path)
    schema_id = manifest.get("schema_id")
    streams: dict[str, Literal["trialeval", "trialdev"]] = {
        "ariadne.benchmark_role_manifest/v1": "trialeval",
        "ariadne.trialdev.role_manifest/v1": "trialdev",
    }
    if schema_id not in streams or manifest.get("role") != "participant":
        raise EvaluationReleaseError("unsupported participant release manifest")
    stream = streams[str(schema_id)]
    outer_path = root / "release_manifest.json"
    outer = read_json_object(outer_path)
    expected_outer_schema = {
        "trialeval": "ariadne.trialeval.release/v1",
        "trialdev": "ariadne.trialdev.release/v1",
    }[stream]
    if outer.get("schema_id") != expected_outer_schema:
        raise EvaluationReleaseError("participant release has an incompatible outer manifest")
    expected_outer_fields = {
        "schema_id",
        "data_provenance",
        "release_id",
        "construct_sha256",
        "qualification_sha256",
        "task_count",
        "participant_manifest_sha256",
        "evaluation_manifest_sha256",
        "checksum",
        "trial_count" if stream == "trialeval" else "world_count",
    }
    if set(outer) != expected_outer_fields:
        raise EvaluationReleaseError("participant outer manifest fields differ from the public contract")
    if outer.get("data_provenance") != "generated_under_declared_trial_mechanisms":
        raise EvaluationReleaseError("participant release data provenance differs from the public contract")
    release_id = outer.get("release_id")
    candidate = outer.get("construct_sha256")
    if (
        not isinstance(release_id, str)
        or not isinstance(candidate, str)
        or release_id != f"release_{candidate}"
        or len(candidate) != 64
        or any(character not in "0123456789abcdef" for character in candidate)
    ):
        raise EvaluationReleaseError("participant release lacks a valid construct identity")
    if manifest.get("release_id") != release_id:
        raise EvaluationReleaseError("participant role and outer release identities differ")
    participant_checksum = outer.get("participant_manifest_sha256")
    if not isinstance(participant_checksum, str) or manifest.get("checksum") != participant_checksum:
        raise EvaluationReleaseError("participant manifest is not bound by the outer release manifest")
    outer_checksum = outer.get("checksum")
    if not isinstance(outer_checksum, str) or outer_checksum != _object_checksum(outer, exclude_checksum=True):
        raise EvaluationReleaseError("participant outer release manifest checksum mismatch")
    census = "trial_count" if stream == "trialeval" else "world_count"
    if manifest.get(census) != outer.get(census) or manifest.get("task_count") != outer.get("task_count"):
        raise EvaluationReleaseError("participant role and outer release censuses differ")
    scope = release_asset_scope(root)
    digest = scope.catalog.release_manifest_sha256 if scope is not None else sha256_file(outer_path)
    return stream, candidate, digest


def _scoped_artifacts(artifacts: list[JsonValue], scope: ReleaseAssetScope | None) -> list[JsonValue]:
    """Filter materialization only after binding every original inventory record."""
    if scope is None:
        return artifacts
    selected: list[JsonValue] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) != {"relative_path", "sha256", "byte_length"}:
            raise EvaluationReleaseError("role inventory contains a malformed record")
        relative, digest, size = artifact["relative_path"], artifact["sha256"], artifact["byte_length"]
        if not isinstance(relative, str) or not isinstance(digest, str) or not isinstance(size, int):
            raise EvaluationReleaseError("role inventory contains malformed file identities")
        if scope.requires(relative, digest, size):
            selected.append(artifact)
    return selected


def _verify_participant_release(root: Path, *, stream: Literal["trialeval", "trialdev"], candidate: str) -> None:
    """Verify one flattened participant archive without product-package imports."""

    manifest = read_json_object(root / "manifest.json")
    expected_fields = {
        "schema_id",
        "release_id",
        "role",
        "task_count",
        "artifacts",
        "checksum",
        "trial_count" if stream == "trialeval" else "world_count",
    }
    if stream == "trialdev":
        expected_fields.add("streams")
    if set(manifest) != expected_fields:
        raise EvaluationReleaseError("participant manifest fields differ from the public contract")
    if manifest.get("release_id") != f"release_{candidate}" or manifest.get("role") != "participant":
        raise EvaluationReleaseError("participant manifest identity mismatch")
    checksum = manifest.get("checksum")
    if not isinstance(checksum, str) or checksum != _object_checksum(manifest, exclude_checksum=True):
        raise EvaluationReleaseError("participant manifest checksum mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise EvaluationReleaseError("participant manifest has no artifacts")
    scope = release_asset_scope(root)
    if scope is not None and scope.catalog.benchmark != stream:
        raise EvaluationReleaseError("Asset catalog identifies a different benchmark")
    paths = tuple(
        _verify_artifact_identity(root, artifact, "participant", flattened=True)
        for artifact in _scoped_artifacts(artifacts, scope)
    )
    if len(paths) != len(set(paths)):
        raise EvaluationReleaseError("participant manifest repeats an artifact")
    declared = {
        "manifest.json",
        "release_manifest.json",
        *(PurePosixPath(path).relative_to("participant").as_posix() for path in paths),
    }
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != declared:
        raise EvaluationReleaseError("participant files differ from the declared role inventory")


def _verify_role_manifest(
    root: Path,
    *,
    role: Literal["participant", "evaluation"],
    release_id: str,
    expected_checksum: str,
    stream: Literal["trialeval", "trialdev"],
    scope: ReleaseAssetScope | None = None,
) -> tuple[str, ...]:
    """Verify one complete role inventory without importing product packages."""

    path = root / role / "manifest.json"
    manifest = read_json_object(path)
    expected_schema = {
        "trialeval": "ariadne.benchmark_role_manifest/v1",
        "trialdev": "ariadne.trialdev.role_manifest/v1",
    }[stream]
    expected_fields = {
        "schema_id",
        "release_id",
        "role",
        "task_count",
        "artifacts",
        "checksum",
        "trial_count" if stream == "trialeval" else "world_count",
    }
    if stream == "trialdev":
        expected_fields.add("streams")
    if set(manifest) != expected_fields:
        raise EvaluationReleaseError(f"{role} manifest fields differ from the public contract")
    if (
        manifest.get("schema_id") != expected_schema
        or manifest.get("release_id") != release_id
        or manifest.get("role") != role
    ):
        raise EvaluationReleaseError(f"{role} manifest identity mismatch")
    checksum = manifest.get("checksum")
    if (
        not isinstance(checksum, str)
        or checksum != expected_checksum
        or checksum != _object_checksum(manifest, exclude_checksum=True)
    ):
        raise EvaluationReleaseError(f"{role} manifest checksum mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise EvaluationReleaseError(f"{role} manifest has no artifacts")
    paths = tuple(_verify_artifact_identity(root, artifact, role) for artifact in _scoped_artifacts(artifacts, scope))
    if len(paths) != len(set(paths)):
        raise EvaluationReleaseError(f"{role} manifest repeats an artifact")
    return paths


def _verify_paired_participant_archive_identity(
    root: Path,
    *,
    release_id: str,
    participant_manifest_checksum: str,
) -> None:
    """Verify the participant ZIP identity anchored by a TrialEval evaluator ZIP."""

    identity_path = root / "participant_archive_identity.json"
    if not identity_path.is_file() or identity_path.is_symlink():
        raise EvaluationReleaseError("paired participant archive identity is missing or unsafe")
    identity = read_json_object(identity_path)
    expected_fields = {
        "schema_id",
        "release_id",
        "participant_archive_sha256",
        "participant_manifest_checksum",
        "checksum",
    }
    if set(identity) != expected_fields:
        raise EvaluationReleaseError("paired participant archive identity fields differ from the public contract")
    archive_sha256 = identity.get("participant_archive_sha256")
    if (
        identity.get("schema_id") != "ariadne.trialeval.participant_archive_identity/v1"
        or identity.get("release_id") != release_id
        or identity.get("participant_manifest_checksum") != participant_manifest_checksum
        or not isinstance(archive_sha256, str)
        or len(archive_sha256) != 64
        or any(character not in "0123456789abcdef" for character in archive_sha256)
    ):
        raise EvaluationReleaseError("paired participant archive identity is inconsistent")
    checksum = identity.get("checksum")
    if not isinstance(checksum, str) or checksum != _object_checksum(identity, exclude_checksum=True):
        raise EvaluationReleaseError("paired participant archive identity checksum mismatch")


def _qualification_bundle(root: Path, candidate: str) -> QualificationReportBundleV1 | ExternalTrialDevQualificationV1:
    """Read and reconcile the mandatory public qualification bundle."""

    external = read_external_qualification(root)
    if external is not None:
        return external
    try:
        bundle = QualificationReportBundleV1.model_validate(
            read_json_object(root / "evaluation" / "verification" / "qualification_reports.json")
        )
    except ValidationError as error:
        raise EvaluationReleaseError("qualification bundle violates its public contract") from error
    if bundle.candidate_digest != candidate:
        raise EvaluationReleaseError("qualification bundle and release candidate identities differ")
    # A key archive omits the evaluation role; its cited files stay committed by the release manifest.
    committed_evaluation = not (root / "evaluation" / "manifest.json").is_file()
    inventory_path = root / ("evaluation/manifest.json" if not committed_evaluation else GRADING_KEY_MANIFEST)
    artifacts = cast(list[dict[str, JsonValue]], read_json_object(inventory_path)["artifacts"])
    declared = {str(item["relative_path"]): item["sha256"] for item in artifacts}
    for identity in bundle.reports:
        path = root / identity.relative_path
        if not path.is_file() or path.is_symlink() or declared.get(identity.relative_path) != identity.sha256:
            raise EvaluationReleaseError(
                f"qualification report changed: {_QUALIFICATION_NAMES[identity.gate_id].replace('_', ' ')}"
            )
        payload = read_json_object(path)
        expected = {
            "schema_id": "trialagentbench.qualification_gate_result/v1",
            "gate_id": identity.gate_id,
            "candidate_digest": identity.candidate_digest,
            "accepted": identity.accepted,
            "scheduled_count": identity.scheduled_count,
            "passed_count": identity.passed_count,
            "failed_count": identity.failed_count,
        }
        if payload != expected:
            raise EvaluationReleaseError(
                f"qualification report fields differ: {_QUALIFICATION_NAMES[identity.gate_id].replace('_', ' ')}"
            )
    release_root = root.resolve(strict=True)
    for comparison in bundle.comparisons:
        for relative in comparison.artifact_paths:
            portable = PurePosixPath(relative)
            path = root / relative
            if portable.is_absolute() or ".." in portable.parts or portable.as_posix() != relative:
                raise EvaluationReleaseError(f"portable comparison cites an unsafe artifact: {relative}")
            if committed_evaluation and portable.parts[0] == "evaluation" and not path.exists():
                continue
            if not path.is_file() or path.is_symlink() or not path.resolve(strict=True).is_relative_to(release_root):
                raise EvaluationReleaseError(f"portable comparison cites a missing or unsafe artifact: {relative}")
    return bundle


_QUALIFICATION_NAMES = {
    "G-EXT-01": "external_anchor",
    "G-TAB-01": "table_integrity",
    "X-19": "execution_replay",
}
"""Descriptive report names for the qualification wire schema's fixed gate identities."""


def _qualification_checks(
    bundle: QualificationReportBundleV1 | ExternalTrialDevQualificationV1,
) -> tuple[CheckResultV1, ...]:
    """Recompute report decisions and portable comparator differences."""

    if isinstance(bundle, ExternalTrialDevQualificationV1):
        return (
            CheckResultV1(
                check_id="release.qualification_external",
                scope_id=bundle.candidate_digest,
                status="not_applicable",
                observed={"qualification": "pending", "recovery": "pending", "notice": QUALIFICATION_PENDING},
                expected="Scientific qualification is checked in the separate release-evidence bundle.",
                artifact_paths=(EXTERNAL_QUALIFICATION_NAME,),
            ),
        )
    checks: list[CheckResultV1] = []
    for report in sorted(bundle.reports, key=lambda row: row.gate_id):
        checks.append(
            CheckResultV1(
                check_id=f"qualification.{_QUALIFICATION_NAMES[report.gate_id]}",
                scope_id=_QUALIFICATION_NAMES[report.gate_id].replace("_", " "),
                status="pass" if report.accepted else "fail",
                observed={
                    "passed": report.passed_count,
                    "failed": report.failed_count,
                    "scheduled": report.scheduled_count,
                },
                expected={"failed": 0, "scheduled": report.scheduled_count},
                artifact_paths=(report.relative_path,),
                failure_code=None if report.accepted else "qualification_gate_failed",
            )
        )
    for comparison in bundle.comparisons:
        difference = abs(comparison.candidate_value - comparison.reference_value)
        checks.append(
            CheckResultV1(
                check_id="qualification.portable_comparison",
                scope_id=comparison.comparison_id,
                status="pass" if difference <= comparison.absolute_margin else "fail",
                observed=difference,
                expected=comparison.absolute_margin,
                tolerance_id=comparison.tolerance_id,
                artifact_paths=comparison.artifact_paths,
                failure_code=(
                    None if difference <= comparison.absolute_margin else "portable_comparison_outside_margin"
                ),
            )
        )
    return tuple(checks)


def _standard_component_digest() -> str:
    """Bind the maintained numerical component inventory used by the verifier."""

    from importlib.metadata import version

    packages = ("lifelines", "numpy", "pandas", "pyarrow", "pydantic", "scipy", "statsmodels")
    inventory: JsonValue = {package: version(package) for package in packages}
    return canonical_sha256(inventory)


def _report(
    *,
    stream: Literal["trialeval", "trialdev"],
    operation: Literal["release", "score", "analyse-trialdev"],
    candidate_digest: str,
    release_digest: str,
    checks: tuple[CheckResultV1, ...],
    report_digests: tuple[str, ...],
    input_kind: Literal["archive", "directory"],
    input_sha256: str | None,
    expected_input_sha256: str | None,
    score_binding: ScoreVerificationBindingV1 | None = None,
    development: bool = False,
    asset_scope: ReleaseAssetScope | None = None,
) -> VerificationReportV1:
    """Construct one count-derived public report."""

    if asset_scope is not None:
        checks += (
            CheckResultV1(
                check_id="release.asset_scope",
                scope_id=release_digest,
                status="pass",
                observed={
                    "coverage": "full" if asset_scope.complete else "selected",
                    "task_ids": sorted(asset_scope.selected_task_ids),
                    "total_release_tasks": len(asset_scope.catalog.tasks),
                },
                expected="Integrity applies to the listed task assets; omitted payloads are not verified by a selected view.",
                artifact_paths=("release_manifest.json",),
            ),
        )
    failed = sum(check.status == "fail" for check in checks)
    return VerificationReportV1(
        scorer=current_scorer(),
        input_kind=input_kind,
        input_sha256=input_sha256,
        expected_input_sha256=expected_input_sha256,
        release_digest=release_digest,
        candidate_digest=candidate_digest,
        benchmark_stream=stream,
        operation=operation,
        score_binding=score_binding,
        status="pass" if failed == 0 else "fail",
        scheduled_check_count=len(checks),
        passed_check_count=len(checks) - failed,
        failed_check_count=failed,
        checks=checks,
        qualification_report_digests=report_digests,
        development=development,
        standard_component_inventory_digest=_standard_component_digest(),
    )


def _verify_materialized_release(
    root: Path,
    *,
    require_paired_participant_identity: bool = False,
    scope: ReleaseAssetScope | None = None,
) -> tuple[Literal["trialeval", "trialdev"], str, str, QualificationReportBundleV1 | ExternalTrialDevQualificationV1]:
    """Require both archive integrity and the exact scientific qualification bundle."""
    stream, candidate, release_digest = _verify_materialized_integrity(
        root, require_paired_participant_identity=require_paired_participant_identity, scope=scope
    )
    return stream, candidate, release_digest, _qualification_bundle(root, candidate)


def _verify_materialized_integrity(
    root: Path,
    *,
    require_paired_participant_identity: bool = False,
    scope: ReleaseAssetScope | None = None,
) -> tuple[Literal["trialeval", "trialdev"], str, str]:
    """Verify declared bytes and paired roles without asserting scientific qualification."""

    scope = scope or release_asset_scope(root)
    stream, candidate, release_digest = _release_identity(root)
    if scope is not None:
        if scope.catalog.audience != "evaluator":
            raise EvaluationReleaseError("paired verification requires evaluator asset scope")
        if (scope.catalog.benchmark, scope.catalog.release_manifest_sha256) != (stream, release_digest):
            raise EvaluationReleaseError("Asset catalog identifies a different scientific release")
    manifest = read_json_object(root / "release_manifest.json")
    expected_fields = {
        "schema_id",
        "data_provenance",
        "release_id",
        "construct_sha256",
        "qualification_sha256",
        "task_count",
        "participant_manifest_sha256",
        "evaluation_manifest_sha256",
        "checksum",
        "trial_count" if stream == "trialeval" else "world_count",
    }
    if set(manifest) != expected_fields:
        raise EvaluationReleaseError("release manifest fields differ from the public contract")
    if manifest.get("data_provenance") != "generated_under_declared_trial_mechanisms":
        raise EvaluationReleaseError("release data provenance differs from the public contract")
    release_id = manifest.get("release_id")
    checksum = manifest.get("checksum")
    if not isinstance(release_id, str) or release_id != f"release_{candidate}":
        raise EvaluationReleaseError("release identity differs from its construct")
    if not isinstance(checksum, str) or checksum != _object_checksum(manifest, exclude_checksum=True):
        raise EvaluationReleaseError("release manifest checksum mismatch")
    participant_checksum = manifest.get("participant_manifest_sha256")
    evaluation_checksum = manifest.get("evaluation_manifest_sha256")
    if not isinstance(participant_checksum, str) or not isinstance(evaluation_checksum, str):
        raise EvaluationReleaseError("release lacks role-manifest identities")
    # The operator archive carries the evaluation role, and both archives carry the grading key: a key
    # archive is verified through the key's inventory, the full archive through both.
    complete = (root / "evaluation" / "manifest.json").is_file()
    keyed = (root / GRADING_KEY_MANIFEST).is_file()
    if not complete and not keyed:
        raise EvaluationReleaseError("release has neither an evaluation role nor a grading key")
    declared = {"release_manifest.json", "participant/manifest.json"}
    role_identities: tuple[tuple[Literal["participant", "evaluation"], str], ...] = (
        ("participant", participant_checksum),
        *((("evaluation", evaluation_checksum),) if complete else ()),
    )
    for role, expected in role_identities:
        declared.update(
            _verify_role_manifest(
                root,
                role=role,
                release_id=release_id,
                expected_checksum=expected,
                stream=stream,
                scope=scope,
            )
        )
    if complete:
        declared.add("evaluation/manifest.json")
    if keyed:
        key = verify_grading_key(
            root, scope=scope, verified_paths=frozenset(declared), release_manifest_sha256=release_digest
        )
        present = {
            item.relative_path
            for item in key.artifacts
            if scope is None or scope.requires(item.relative_path, item.sha256, item.byte_length)
        }
        if complete and not {path for path in present if path.startswith("evaluation/")} <= declared:
            raise EvaluationReleaseError("grading key retains files outside the evaluation role")
        declared.update(present)
        declared.add(GRADING_KEY_MANIFEST)
    paired_identity_path = root / "participant_archive_identity.json"
    if stream == "trialeval" and (require_paired_participant_identity or paired_identity_path.exists()):
        _verify_paired_participant_archive_identity(
            root,
            release_id=release_id,
            participant_manifest_checksum=participant_checksum,
        )
        declared.add("participant_archive_identity.json")
    if read_external_qualification(root) is not None:
        declared.add(EXTERNAL_QUALIFICATION_NAME)
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != declared:
        raise EvaluationReleaseError("release files differ from the declared role inventories")
    qualification_path = (
        root / "evaluation" / ("qualification.json" if stream == "trialeval" else "resource_qualification.json")
    )
    if not complete and not qualification_path.exists():
        # A TrialEval key omits the cell-level qualification; the release manifest commits to it.
        return stream, candidate, release_digest
    qualification = read_json_object(qualification_path)
    expected_qualification = manifest.get("qualification_sha256")
    observed_qualification = (
        qualification.get("checksum")
        if stream == "trialeval"
        else _object_checksum(qualification, exclude_checksum=False)
    )
    if observed_qualification != expected_qualification:
        raise EvaluationReleaseError("qualification identity mismatch")
    return stream, candidate, release_digest


def _verify_expected_input(
    release: Path,
    expected_sha256: str | None,
) -> tuple[Literal["archive", "directory"], str]:
    """Verify an external trust anchor and return the actual input identity."""

    source = Path(release)
    input_kind: Literal["archive", "directory"] = "archive" if source.is_file() else "directory"
    identity_path = source if source.is_file() else source / "release_manifest.json"
    if not identity_path.is_file() or identity_path.is_symlink():
        raise EvaluationReleaseError("trusted release identity input is missing or unsafe")
    actual_sha256 = sha256_file(identity_path)
    if expected_sha256 is None:
        return input_kind, actual_sha256
    if len(expected_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_sha256):
        raise EvaluationReleaseError("expected SHA-256 must be 64 lowercase hexadecimal characters")
    if actual_sha256 != expected_sha256:
        raise EvaluationReleaseError("release differs from the trusted expected SHA-256")
    return input_kind, actual_sha256


def verify_release(
    release: Path, *, expected_sha256: str | None = None, task_ids: tuple[str, ...] | None = None
) -> VerificationReportV1:
    """Verify a release and, when supplied, its externally trusted SHA-256."""

    if task_ids is not None and not Path(release).name.endswith(".assets.json"):
        raise EvaluationReleaseError("task-selective verification requires an asset catalog")
    input_kind, input_sha256 = _verify_expected_input(release, expected_sha256)
    with ExitStack() as stack:
        if task_ids is None:
            root = stack.enter_context(opened_release(release, verify_files=True))
        else:
            source = stack.enter_context(release_archive(release, expected_sha256=input_sha256))
            root = stack.enter_context(materialized_release(source, task_ids=task_ids))
        participant_projection = (root / "manifest.json").is_file() and not (root / "participant").is_dir()
        if participant_projection:
            stream, candidate, release_digest = _participant_identity(root)
            try:
                _verify_participant_release(root, stream=stream, candidate=candidate)
            except EvaluationReleaseError as error:
                participant_checks: tuple[CheckResultV1, ...] = (
                    CheckResultV1(
                        check_id="participant.inventory",
                        scope_id=candidate,
                        status="fail",
                        observed=str(error),
                        expected="all participant files match the checksum-bound role inventory",
                        artifact_paths=("manifest.json",),
                        failure_code="participant_integrity_failure",
                    ),
                )
            else:
                identity_check = (
                    CheckResultV1(
                        check_id="participant.candidate_identity",
                        scope_id=candidate,
                        status="fail",
                        observed="no externally trusted participant archive SHA-256 was supplied",
                        expected="participant archive bytes match a supplied trusted SHA-256",
                        artifact_paths=("manifest.json",),
                        failure_code="participant_trust_anchor_required",
                    )
                    if expected_sha256 is None
                    else CheckResultV1(
                        check_id="participant.candidate_identity",
                        scope_id=candidate,
                        status="pass",
                        observed=input_sha256,
                        expected=expected_sha256,
                        artifact_paths=("manifest.json",),
                    )
                )
                participant_checks = (
                    CheckResultV1(
                        check_id="participant.inventory",
                        scope_id=candidate,
                        status="pass",
                        observed="all participant files match their byte identities",
                        expected="all participant files match their byte identities",
                        artifact_paths=("manifest.json",),
                    ),
                    identity_check,
                )
            return _report(
                stream=stream,
                operation="release",
                asset_scope=release_asset_scope(root),
                candidate_digest=candidate,
                release_digest=release_digest,
                checks=participant_checks,
                report_digests=(),
                input_kind=input_kind,
                input_sha256=input_sha256,
                expected_input_sha256=expected_sha256,
            )
        stream, candidate, release_digest = _release_identity(root)
        if input_kind == "archive" and expected_sha256 is None:
            return _report(
                stream=stream,
                operation="release",
                asset_scope=release_asset_scope(root),
                candidate_digest=candidate,
                release_digest=release_digest,
                checks=(
                    CheckResultV1(
                        check_id="release.archive_identity",
                        scope_id=candidate,
                        status="fail",
                        observed="no externally trusted evaluator archive SHA-256 was supplied",
                        expected="evaluator archive bytes match a supplied trusted SHA-256",
                        artifact_paths=("release_manifest.json",),
                        failure_code="evaluator_trust_anchor_required",
                    ),
                ),
                report_digests=(),
                input_kind=input_kind,
                input_sha256=input_sha256,
                expected_input_sha256=None,
            )
        bundle: QualificationReportBundleV1 | ExternalTrialDevQualificationV1 | None = None
        try:
            _, _, _, bundle = _verify_materialized_release(
                root,
                require_paired_participant_identity=input_kind == "archive",
            )
        except EvaluationReleaseError as error:
            release_checks: tuple[CheckResultV1, ...] = (
                CheckResultV1(
                    check_id="release.inventory",
                    scope_id=candidate,
                    status="fail",
                    observed=str(error),
                    expected="all declared files match their byte identities and release contracts",
                    artifact_paths=(
                        "release_manifest.json",
                        "participant/manifest.json",
                        "evaluation/manifest.json",
                    ),
                    failure_code="release_integrity_failure",
                ),
            )
        else:
            if input_kind == "archive":
                identity_check = CheckResultV1(
                    check_id="release.archive_identity",
                    scope_id=candidate,
                    status="pass",
                    observed=input_sha256,
                    expected=expected_sha256,
                    artifact_paths=("release_manifest.json",),
                )
            else:
                identity_check = CheckResultV1(
                    check_id="release.candidate_identity",
                    scope_id=candidate,
                    status="pass",
                    observed=candidate,
                    expected=candidate,
                    artifact_paths=("release_manifest.json",),
                )
            release_checks = (
                CheckResultV1(
                    check_id="release.inventory",
                    scope_id=candidate,
                    status="pass",
                    observed="all declared files match their byte identities",
                    expected="all declared files match their byte identities",
                    artifact_paths=(
                        "release_manifest.json",
                        "participant/manifest.json",
                        "evaluation/manifest.json",
                    ),
                ),
                identity_check,
            ) + _qualification_checks(bundle)
        return _report(
            stream=stream,
            operation="release",
            asset_scope=release_asset_scope(root),
            candidate_digest=candidate,
            release_digest=release_digest,
            checks=release_checks,
            report_digests=tuple(report.sha256 for report in bundle.reports) if bundle is not None else (),
            input_kind=input_kind,
            input_sha256=input_sha256,
            expected_input_sha256=expected_sha256,
        )


def _score_trialeval(
    root: Path,
    submission: NormalizedSubmissionV1,
    *,
    execution_authenticated: bool,
) -> tuple[CheckResultV1, ...]:
    """Independently reconstruct estimator-neutral TrialEval components."""

    from trialagentbench_verifier.trialeval import TrialEvalScoreInputV1, verify_trialeval_outcome_v1

    if submission.submission_failure_code is None:
        score_input = TrialEvalScoreInputV1.model_validate(submission.payload)
    else:
        score_input = TrialEvalScoreInputV1(
            schema_id="trialagentbench.trialeval_score_input/v1",
            task_id=submission.task_id,
            attempt_status=(
                "missing"
                if submission.submission_failure_code == "missing_primary_submission"
                else "missing_required_output"
            ),
        )
    if score_input.task_id != submission.task_id:
        raise EvaluationReleaseError("TrialEval score input and normalized submission identify different tasks")
    outcome = verify_trialeval_outcome_v1(
        release_root=root,
        score_input=score_input,
        authenticated_execution_reproduced=True if execution_authenticated else None,
    )
    checks: list[CheckResultV1] = []
    for component in (*outcome.components, *(() if outcome.decision is None else (outcome.decision,))):
        failed = component.status in {"not_satisfied", "not_evaluable"}
        observed: JsonValue = component.model_dump(mode="json")
        if component.component_id in {"estimand_performance", "uncertainty"} and (
            outcome.metrics is not None or outcome.target_outcomes
        ):
            observed = {
                "component": observed,
                "metrics": cast(JsonValue, outcome.metrics),
                "signal_loss": outcome.signal_loss,
                **(
                    {"targets": [item.model_dump(mode="json") for item in outcome.target_outcomes]}
                    if outcome.target_outcomes
                    else {}
                ),
            }
        checks.append(
            CheckResultV1(
                check_id=f"score.{component.component_id}",
                scope_id=submission.task_id,
                status=("fail" if failed else "not_applicable" if component.status == "not_applicable" else "pass"),
                observed=observed,
                expected="observed_or_satisfied",
                artifact_paths=(TRIALEVAL_GRADING_KEY_RECORDS,),
                failure_code=component.failure_origin if failed else None,
            )
        )
    if outcome.plan_execution is not None:
        # Secondary: the plan as executed, kept apart from the task's loss and decision.
        checks.append(
            CheckResultV1(
                check_id="score.plan_execution",
                scope_id=submission.task_id,
                status="pass" if outcome.plan_execution.accurate else "fail",
                observed=outcome.plan_execution.model_dump(mode="json"),
                expected="within_plan_tolerance",
                artifact_paths=(TRIALEVAL_GRADING_KEY_RECORDS,),
                failure_code=None if outcome.plan_execution.accurate else "plan_execution_mismatch",
            )
        )
    return tuple(checks)


def _score_trialdev(
    root: Path,
    submission_path: Path,
    submission: NormalizedSubmissionV1,
    *,
    execution_authenticated: bool,
) -> tuple[CheckResultV1, ...]:
    """Reconstruct one TrialDev evidence or portfolio grade."""

    if submission.payload.get("task_id") != submission.task_id:
        raise EvaluationReleaseError("TrialDev submission task identity mismatch")
    configuration_path = root / "evaluation/tasks" / submission.task_id / "analysis.json"
    if submission.submission_kind == "evidence" and configuration_path.is_file():
        from pydantic import TypeAdapter

        from trialagentbench_verifier.authored_results import mean_numerical_loss
        from trialagentbench_verifier.authored_scoring import compute_numerical_recovery
        from trialagentbench_verifier.development import _AnalysisConfiguration
        from trialagentbench_verifier.numerical import NumericalEstimate

        configuration = _AnalysisConfiguration.model_validate(read_json_object(configuration_path))
        initial = configuration.initial_evidence
        task = read_json_object(root / "participant/tasks" / submission.task_id / "task.json")
        identities = tuple(f"{row.asset_id}:{row.endpoint}" for row in initial.targets)
        if task.get("initial_target_ids") != list(identities) or initial.context.target_ids != identities:
            raise EvaluationReleaseError("Public numerical targets differ from the initial grading reference")
        estimates = TypeAdapter(tuple[NumericalEstimate, ...]).validate_python(submission.payload["estimates"])
        by_target = {row.target_id: row for row in estimates}
        if len(by_target) != len(estimates) or not by_target.keys() <= set(identities):
            raise EvaluationReleaseError("Numerical submission differs from the assigned target census")
        scores = []
        for target, target_id in zip(initial.targets, identities, strict=True):
            clinical = target.endpoint == "clinical_outcome"
            unit = configuration.preferences.outcome_unit if clinical else "risk difference"
            if (
                target.estimand != "standardized_observed_association"
                or not target.point_identified
                or not compatible_units(target.unit, unit)
            ):
                raise EvaluationReleaseError("Initial reference is not the declared observed association")
            scores.append(
                compute_numerical_recovery(
                    target_id=target_id,
                    estimate=by_target.get(target_id) if execution_authenticated else None,
                    reference=target.oracle_value,
                    scale=configuration.preferences.benefit_unit if clinical else 0.01,
                    confidence=0.95,
                    reproducible=execution_authenticated,
                )
            )
        return (
            CheckResultV1(
                check_id="score.trialdev.initial_loss",
                scope_id=submission.task_id,
                status="pass" if execution_authenticated else "fail",
                observed={
                    "scores": [row.model_dump(mode="json") for row in scores],
                    "mean_loss": mean_numerical_loss(tuple(scores)),
                },
                expected="authenticated numerical recovery for the declared initial targets",
                artifact_paths=(str(submission_path), configuration_path.relative_to(root).as_posix()),
                failure_code=None if execution_authenticated else "execution_proof_required",
            ),
        )
    raise EvaluationReleaseError(
        "Scalar TrialDev scoring requires a current numerical-analysis task; use whole-run grading for authored programmes"
    )


class ScoreSubmissionV1(BaseModel):
    """One submission and its original replay/custody inputs for batch scoring."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    submission: Path
    execution_proof: Path | None = None
    expected_execution_proof_sha256: str | None = None
    program: Path | None = None
    program_result: Path | None = None
    graded_item: Path | None = None


def score_submissions(
    release: Path,
    submissions: Iterable[ScoreSubmissionV1],
    *,
    expected_sha256: str | None = None,
    development: bool = False,
    task_ids: tuple[str, ...] | None = None,
) -> Generator[VerificationReportV1, None, None]:
    """Score a stream using one authenticated extraction, preserving each report.

    Consume or close the iterator to release its private temporary directory.
    Inputs and numerical scoring are identical to :func:`score_submission`.
    Release snapshots are shared only inside an explicit campaign action and are
    never exposed to submitted programs.
    """
    input_kind, input_sha256 = _verify_expected_input(release, expected_sha256)
    if input_kind == "archive" and expected_sha256 is None:
        raise EvaluationReleaseError("score replay from an evaluator archive requires a trusted expected SHA-256")
    with (
        (
            release_archive(release, expected_sha256=input_sha256) if input_kind == "archive" else nullcontext(release)
        ) as source,
        materialized_release(source, task_ids=task_ids) as root,
    ):
        if development:
            if expected_sha256 is None:
                raise EvaluationReleaseError("Development scoring requires an externally trusted archive digest")
            stream, candidate, release_digest = _verify_materialized_integrity(
                root, require_paired_participant_identity=input_kind == "archive"
            )
            report_digests: tuple[str, ...] = ()
        else:
            stream, candidate, release_digest, bundle = _verify_materialized_release(
                root, require_paired_participant_identity=input_kind == "archive"
            )
            report_digests = tuple(report.sha256 for report in bundle.reports)
        for request in submissions:
            yield _score_materialized_submission(
                root,
                request,
                stream=stream,
                candidate=candidate,
                release_digest=release_digest,
                report_digests=report_digests,
                development=development,
                input_kind=input_kind,
                input_sha256=input_sha256,
                expected_sha256=expected_sha256,
            )


def _score_materialized_submission(
    root: Path,
    request: ScoreSubmissionV1,
    *,
    stream: Literal["trialeval", "trialdev"],
    candidate: str,
    release_digest: str,
    report_digests: tuple[str, ...],
    development: bool,
    input_kind: Literal["archive", "directory"],
    input_sha256: str | None,
    expected_sha256: str | None,
) -> VerificationReportV1:
    """Apply the existing scorer within one authenticated release lifetime."""
    submission_path = request.submission
    execution_proof = request.execution_proof
    expected_execution_proof_sha256 = request.expected_execution_proof_sha256
    program = request.program
    program_result = request.program_result
    graded_item = request.graded_item
    if submission_path.suffix.lower() == ".csv":
        from trialagentbench_verifier.numerical import read_numerical_table

        if execution_proof is None or expected_execution_proof_sha256 is None:
            raise EvaluationReleaseError("Direct CSV scoring requires the trusted execution proof")
        if sha256_file(execution_proof) != expected_execution_proof_sha256:
            raise EvaluationReleaseError("Execution proof differs from its trusted SHA-256")
        proof = ExecutionProofV1.model_validate(read_json_object(execution_proof))
        if proof.task_id in {".", ".."} or "/" in proof.task_id or "\\" in proof.task_id:
            raise EvaluationReleaseError("CSV execution proof requires a task identifier, not a path")
        if proof.result_format != "csv":
            raise EvaluationReleaseError("CSV scoring requires a CSV execution proof")
        rows = read_numerical_table(submission_path, target_ids=proof.target_ids)
        if stream == "trialeval":
            from trialagentbench_verifier.execution import _task
            from trialagentbench_verifier.trialeval import (
                TrialEvalScoreInputV1,
                TrialEvalTargetScoreInputV1,
                _expected_conclusion,
                _load_record,
                _PointResult,
            )

            record = _load_record(root, proof.task_id)
            public_targets = _task(root / "participant", proof.task_id).target_ids
            if proof.target_ids != public_targets or public_targets != record.reported_target_ids:
                raise EvaluationReleaseError("CSV proof targets differ from the public TrialEval question")
            if {row.target_id for row in rows} != set(public_targets):
                raise EvaluationReleaseError("CSV execution proof cannot certify a missing TrialEval target")
            inputs: list[TrialEvalTargetScoreInputV1] = []
            # The plan as executed shares the primary question's scale.
            plan_reference = () if record.plan_replay is None else ((record.plan_replay.target_id, record),)
            for target_id, reference in (
                *((item.target.binding.estimand_id, item) for item in record.references),
                *plan_reference,
            ):
                question = reference.target.binding
                row = next(row for row in rows if row.target_id == target_id)
                scored_target_id = (
                    target_id if len(record.references) > 1 or target_id != record.target.binding.estimand_id else None
                )
                if question.effect_measure == "rate_ratio" and min(row.estimate, row.lower, row.upper) <= 0:
                    # As the harness grades it: the executed CSV is retained, but a nonpositive ratio is not an
                    # eligible answer on the logarithmic comparison scale, so it is a missing required output.
                    inputs.append(
                        TrialEvalTargetScoreInputV1(
                            target_id=scored_target_id,
                            attempt_status="missing_required_output",
                            evidence_source_artifacts=tuple(
                                sorted(item.relative_path for item in proof.evidence_artifacts)
                            ),
                        )
                    )
                    continue
                result = _PointResult.model_validate(
                    {
                        "kind": "point",
                        "value": row.estimate,
                        "effect_measure": question.effect_measure,
                        "unit": question.result_unit,
                        "interval": {
                            "lower": row.lower,
                            "upper": row.upper,
                            "confidence_level": question.confidence_level,
                            "interval_type": question.uncertainty_type,
                        },
                        "conclusion": "no_clear_difference",
                    }
                )
                result = result.model_copy(update={"conclusion": _expected_conclusion(result, question)})
                inputs.append(
                    TrialEvalTargetScoreInputV1(
                        target_id=scored_target_id,
                        attempt_status="complete",
                        submitted_binding=question,
                        submitted_result=result,
                        reproducible=True,
                        evidence_source_artifacts=tuple(
                            sorted(item.relative_path for item in proof.evidence_artifacts)
                        ),
                    )
                )
            scored = inputs[: len(record.references)]
            payload = TrialEvalScoreInputV1.model_validate(
                {
                    "task_id": proof.task_id,
                    **scored[0].model_dump(mode="json"),
                    "additional_inputs": [item.model_dump(mode="json") for item in scored[1:]],
                    **({"plan_input": inputs[-1].model_dump(mode="json")} if plan_reference else {}),
                }
            ).model_dump(mode="json")
        else:
            public_task = read_json_object(root / "participant/tasks" / proof.task_id / "task.json")
            if public_task.get("initial_target_ids") != list(proof.target_ids):
                raise EvaluationReleaseError("CSV proof targets differ from the public initial task")
            payload = NormalizedSubmissionV1(
                benchmark_stream="trialdev",
                task_id=proof.task_id,
                submission_kind="evidence",
                payload={"task_id": proof.task_id, "estimates": [row.model_dump(mode="json") for row in rows]},
            ).model_dump(mode="json")

    else:
        payload = read_json_object(submission_path)
    if payload.get("schema_id") == "trialagentbench.trialeval_score_input/v1":
        task_id = payload.get("task_id")
        if stream != "trialeval" or not isinstance(task_id, str):
            raise EvaluationReleaseError("portable TrialEval score input is incompatible with this release")
        normalized = NormalizedSubmissionV1(
            benchmark_stream="trialeval",
            task_id=task_id,
            submission_kind="analysis",
            payload=payload,
        )
    else:
        normalized = NormalizedSubmissionV1.model_validate(payload)
    if normalized.benchmark_stream != stream:
        raise EvaluationReleaseError("submission and release benchmark streams differ")
    proof_arguments = (execution_proof, expected_execution_proof_sha256, program, program_result)
    if any(argument is not None for argument in proof_arguments) and not all(
        argument is not None for argument in proof_arguments
    ):
        raise EvaluationReleaseError(
            "execution proof, its trusted SHA-256, the program, and the program result must be supplied together"
        )
    authenticated_facts = None
    if execution_proof is not None:
        from trialagentbench_verifier.proof import authenticate_execution_proof

        if expected_execution_proof_sha256 is None or program is None or program_result is None:
            raise EvaluationReleaseError("validated execution-proof inputs became incomplete")
        authenticated_facts = authenticate_execution_proof(
            execution_proof,
            expected_sha256=expected_execution_proof_sha256,
            program_path=program,
            program_result_path=program_result,
            root=root,
            stream=stream,
            candidate_digest=candidate,
            release_digest=release_digest,
            submission=normalized,
        )
    checks = (
        _score_trialeval(
            root,
            normalized,
            execution_authenticated=authenticated_facts is not None,
        )
        if stream == "trialeval"
        else _score_trialdev(
            root,
            submission_path,
            normalized,
            execution_authenticated=authenticated_facts is not None,
        )
    )
    if read_external_qualification(root) is not None:
        checks += (
            CheckResultV1(
                check_id="release.qualification_external",
                scope_id=normalized.task_id,
                status="not_applicable",
                observed={"qualification": "pending", "recovery": "pending", "notice": QUALIFICATION_PENDING},
                expected="This score does not assert scientific qualification; consult release-evidence.",
                artifact_paths=(EXTERNAL_QUALIFICATION_NAME,),
            ),
        )
    score_binding = ScoreVerificationBindingV1(
        task_id=normalized.task_id,
        submission_artifact_sha256=sha256_file(submission_path),
        normalized_submission_sha256=canonical_sha256(normalized.model_dump(mode="json")),
        graded_item_sha256=(None if graded_item is None else _portable_graded_item_sha256(Path(graded_item))),
        execution_proof_sha256=(expected_execution_proof_sha256 if authenticated_facts is not None else None),
        program_artifacts=(() if authenticated_facts is None else authenticated_facts.program_artifacts),
        program_sha256=(authenticated_facts.program_sha256 if authenticated_facts is not None else None),
        program_result_sha256=(authenticated_facts.program_result_sha256 if authenticated_facts is not None else None),
        normalized_program_result_sha256=(
            authenticated_facts.normalized_program_result_sha256 if authenticated_facts is not None else None
        ),
        replayed_result_sha256s=(
            authenticated_facts.replayed_result_sha256s if authenticated_facts is not None else None
        ),
    )
    return _report(
        stream=stream,
        operation="score",
        candidate_digest=candidate,
        release_digest=release_digest,
        checks=checks,
        report_digests=report_digests,
        development=development,
        input_kind=input_kind,
        input_sha256=input_sha256,
        expected_input_sha256=expected_sha256,
        score_binding=score_binding,
    )


def score_submission(
    release: Path,
    submission: Path,
    *,
    expected_sha256: str | None = None,
    execution_proof: Path | None = None,
    expected_execution_proof_sha256: str | None = None,
    program: Path | None = None,
    program_result: Path | None = None,
    graded_item: Path | None = None,
) -> VerificationReportV1:
    """Recompute a score from a release with a trusted archive identity."""

    requests = (
        ScoreSubmissionV1(
            submission=submission,
            execution_proof=execution_proof,
            expected_execution_proof_sha256=expected_execution_proof_sha256,
            program=program,
            program_result=program_result,
            graded_item=graded_item,
        ),
    )
    task_ids = None
    if release.name.endswith(".assets.json"):
        if submission.suffix.lower() == ".csv":
            if execution_proof is None or expected_execution_proof_sha256 is None:
                raise EvaluationReleaseError("Direct CSV scoring requires the trusted execution proof")
            if sha256_file(execution_proof) != expected_execution_proof_sha256:
                raise EvaluationReleaseError("Execution proof differs from its trusted SHA-256")
            task_id = ExecutionProofV1.model_validate(read_json_object(execution_proof)).task_id
        else:
            payload = read_json_object(submission)
            if payload.get("schema_id") == "trialagentbench.trialeval_score_input/v1":
                from trialagentbench_verifier.trialeval import TrialEvalScoreInputV1

                task_id = TrialEvalScoreInputV1.model_validate(payload).task_id
            else:
                task_id = NormalizedSubmissionV1.model_validate(payload).task_id
        task_ids = (task_id,)
    reports = score_submissions(release, requests, expected_sha256=expected_sha256, task_ids=task_ids)
    try:
        return next(reports)
    finally:
        reports.close()


__all__ = [
    "EvaluationReleaseError",
    "ScoreSubmissionV1",
    "score_submissions",
    "score_submission",
    "verify_release",
]
