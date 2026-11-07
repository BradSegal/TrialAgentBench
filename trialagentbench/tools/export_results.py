"""Build a deterministic, checksummed archive of caller-owned results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal, cast
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.types import JsonValue
from trialagentbench_verifier.io import trialdev_task_run_paths
from trialagentbench_verifier.response import TrialDevRunRankingV1

from trialagentbench.contracts.release.artifacts import TRIALEVAL_PARTICIPANT_ARCHIVE_IDENTITY_NAME
from trialagentbench.contracts.release.identity import PairedParticipantArchiveIdentityV1
from trialagentbench.contracts.release.result_export import (
    ResultArtifactKind,
    ResultExportBundleManifestV1,
    ResultExportMemberV1,
    ResultExportProfile,
    ResultExportSourceV1,
)
from trialagentbench.io import canonical_payload_sha256, loads_json, read_json, sha256_file
from trialagentbench.release import load_evaluation_release, load_participant_release, load_trialeval_grading_key
from trialagentbench.trialdev.release import load_trialdev_evaluator_release

_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
_FORBIDDEN_DIRECTORY_NAMES = frozenset({".git", ".mypy_cache", ".pytest_cache", "__pycache__"})
_FORBIDDEN_FILE_NAMES = frozenset({".env", "credentials.json", "provider.env"})
_FORBIDDEN_FILE_SUFFIXES = (".key", ".pem", ".pyc", ".pyo")
_PROVIDER_KEY_SUFFIX = "_API_KEY"
_MINIMUM_CREDENTIAL_CHARACTERS = 8
_RESULT_ARTIFACT_KINDS: tuple[ResultArtifactKind, ...] = (
    "run",
    "grade",
    "verification",
    "analysis",
)
_PORTABLE_PATH_FIELDS = frozenset(
    {
        "bundle_dir",
        "input_run_dir",
        "output_run_dir",
        "participant_archive",
        "source_artifact_path",
        "suite_dir",
    }
)
_MAX_PROJECTED_JSON_BYTES = 128 * 1024**2


class _ReleaseContractIdentityV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    release_id: str = Field(min_length=1)
    release_stage: Literal["benchmark_publication"]
    manifest_path: Path
    release_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    participant_manifest_checksum: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _VerificationReportIdentityV1(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    schema_id: Literal["trialagentbench.verification_report/v1"]
    release_digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class _BenchmarkReleaseManifestV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["ariadne.trialeval.release/v1", "ariadne.trialdev.release/v1"]
    data_provenance: Literal["generated_under_declared_trial_mechanisms"]
    release_id: str = Field(pattern=r"^release_[0-9a-f]{64}$")
    construct_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    qualification_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_count: int = Field(ge=1)
    participant_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trial_count: int | None = Field(default=None, ge=1)
    world_count: int | None = Field(default=None, ge=1)
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_stream(self) -> _BenchmarkReleaseManifestV1:
        """Require the stream-specific count and content-derived identities."""

        is_trialeval = self.schema_id == "ariadne.trialeval.release/v1"
        if is_trialeval != (self.trial_count is not None) or is_trialeval == (self.world_count is not None):
            raise ValueError("benchmark release has incompatible stream counts")
        if self.release_id != f"release_{self.construct_sha256}":
            raise ValueError("benchmark release identity differs from its construct")
        return self


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trialagentbench export results",
        description=(
            "Archive immutable run, grade, and independent-verification outputs "
            "against one released benchmark identity."
        ),
    )
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--grade-root", type=Path, action="append", required=True)
    parser.add_argument("--verification-root", type=Path, action="append", required=True)
    parser.add_argument("--analysis-root", type=Path, action="append", default=[])
    parser.add_argument(
        "--profile",
        choices=("full_custody", "analysis_reproduction"),
        default="full_custody",
        help="Export every retained byte or the bounded TrialEval analysis-reproduction projection.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _trialdev_release_contract(root: Path, manifest_path: Path) -> _ReleaseContractIdentityV1:
    release = load_trialdev_evaluator_release(root)
    if release.manifest.checksum is None:
        raise ValueError("TrialDev release manifest has no content checksum")
    return _ReleaseContractIdentityV1(
        release_id=release.manifest.release_id,
        release_stage="benchmark_publication",
        manifest_path=manifest_path,
        release_manifest_sha256=sha256_file(manifest_path),
        contract_sha256=release.manifest.checksum,
    )


def _trialeval_publication_contract(root: Path, manifest_path: Path) -> _ReleaseContractIdentityV1:
    """Verify a current TrialEval role release and return its export identity."""

    payload = read_json(manifest_path)
    manifest = _BenchmarkReleaseManifestV1.model_validate(payload)
    if manifest.schema_id != "ariadne.trialeval.release/v1":
        raise ValueError("TrialEval export received another benchmark stream")
    checksum_payload = dict(payload)
    checksum_payload.pop("checksum", None)
    if canonical_payload_sha256(checksum_payload) != manifest.checksum:
        raise ValueError("TrialEval release manifest checksum mismatch")
    participant = load_participant_release(root / "participant")
    grading_key = load_trialeval_grading_key(root)
    if (
        participant.manifest.checksum != manifest.participant_manifest_sha256
        or participant.manifest.release_id != manifest.release_id
        or grading_key.manifest.release_id != manifest.release_id
    ):
        raise ValueError("TrialEval role identities differ from the release manifest")
    if (root / "evaluation" / "manifest.json").is_file():
        # An operator release also carries the evaluation role its manifest commits to.
        evaluation = load_evaluation_release(root / "evaluation")
        if evaluation.manifest.checksum != manifest.evaluation_manifest_sha256:
            raise ValueError("TrialEval evaluation role differs from the release manifest")
        qualification = read_json(root / "evaluation" / "qualification.json")
        if qualification.get("checksum") != manifest.qualification_sha256:
            raise ValueError("TrialEval qualification identity differs from the release manifest")
    declared = {"release_manifest.json"}
    paired_path = root / TRIALEVAL_PARTICIPANT_ARCHIVE_IDENTITY_NAME
    if paired_path.exists():
        if paired_path.is_symlink() or not paired_path.is_file():
            raise ValueError("TrialEval paired participant identity must be a regular file")
        paired = PairedParticipantArchiveIdentityV1.model_validate_json(paired_path.read_bytes())
        if (
            paired.release_id != manifest.release_id
            or paired.participant_manifest_checksum != participant.manifest.checksum
        ):
            raise ValueError("TrialEval paired participant identity differs from the released roles")
        declared.add(TRIALEVAL_PARTICIPANT_ARCHIVE_IDENTITY_NAME)
    declared.update(
        f"participant/{path.relative_to(participant.root).as_posix()}"
        for path in participant.root.rglob("*")
        if path.is_file()
    )
    declared.update(
        path.relative_to(root).as_posix()
        for directory in ("evaluation", "grading_key")
        for path in (root / directory).rglob("*")
        if path.is_file()
    )
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != declared:
        raise ValueError("TrialEval release files differ from the role manifests")
    return _ReleaseContractIdentityV1(
        release_id=manifest.release_id,
        release_stage="benchmark_publication",
        manifest_path=manifest_path,
        release_manifest_sha256=sha256_file(manifest_path),
        participant_manifest_checksum=participant.manifest.checksum,
        contract_sha256=manifest.checksum,
    )


def _release_contract(root: Path) -> _ReleaseContractIdentityV1:
    manifest_path = root / "release_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("release root requires a regular release_manifest.json")
    schema_id = read_json(manifest_path).get("schema_id")
    if schema_id == "ariadne.trialeval.release/v1":
        return _trialeval_publication_contract(root, manifest_path)
    if schema_id == "ariadne.trialdev.release/v1":
        return _trialdev_release_contract(root, manifest_path)
    raise ValueError("release root contains an unsupported benchmark manifest")


def _safe_label(path: Path) -> str:
    label = re.sub(r"[^a-z0-9_-]+", "-", path.name.casefold()).strip("-_")
    if not label:
        raise ValueError(f"artifact root has no usable label: {path}")
    return label[:64]


def _provider_credentials() -> tuple[bytes, ...]:
    """Return the exporter's provider key values, which no exported record may contain."""
    return tuple(
        sorted(
            {
                value.encode("utf-8")
                for name, value in os.environ.items()
                if name.endswith(_PROVIDER_KEY_SUFFIX) and len(value) >= _MINIMUM_CREDENTIAL_CHARACTERS
            }
        )
    )


def _artifact_files(root: Path) -> tuple[Path, ...]:
    """List exportable files, refusing transients, links and any record holding a provider key.

    Names and directories under an agent's ``scratch`` are the participant's own and are
    exempt from the transient-name rule; their contents are still scanned.
    """
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"result artifact root must be a regular directory: {root}")
    credentials = _provider_credentials()
    files: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda candidate: candidate.relative_to(root).as_posix()):
        relative = path.relative_to(root)
        participant = "scratch" in relative.parts[:-1]
        if not participant and any(part in _FORBIDDEN_DIRECTORY_NAMES for part in relative.parts[:-1]):
            raise ValueError(f"result artifact contains a transient directory: {relative.as_posix()}")
        if path.is_symlink():
            raise ValueError(f"result artifact contains a symbolic link: {relative.as_posix()}")
        mode = path.stat().st_mode
        if path.is_dir():
            continue
        if not stat.S_ISREG(mode):
            raise ValueError(f"result artifact contains a non-regular file: {relative.as_posix()}")
        if not participant and (
            path.name.casefold() in _FORBIDDEN_FILE_NAMES or path.suffix.casefold() in _FORBIDDEN_FILE_SUFFIXES
        ):
            raise ValueError(
                f"result artifact contains a forbidden credential or transient file: {relative.as_posix()}"
            )
        if credentials and path.suffix.casefold() in {".json", ".jsonl"}:
            content = path.read_bytes()
            if any(credential in content for credential in credentials):
                raise ValueError(f"result artifact contains a provider credential: {relative.as_posix()}")
        files.append(path)
    if not files:
        raise ValueError(f"result artifact root contains no files: {root}")
    return tuple(files)


def _analysis_reproduction_files(
    kind: ResultArtifactKind,
    root: Path,
    files: tuple[Path, ...],
) -> tuple[Path, ...]:
    """Keep run summaries and grades for campaign re-analysis without raw workspaces."""

    if kind == "analysis":
        selected = files
    elif kind in {"run", "grade"}:
        selected = tuple(
            path
            for path in files
            if path.relative_to(root).as_posix()
            in {
                "run_config.json",
                "invocation.json",
                "coverage.json",
                "run_coverage.json",
                "run_index.json",
                "ranking.json",
            }
            or (
                path.name in {"run.json", "programme_run.json", "analysis_run.json"}
                and path.parent.name.startswith("td-")
                and (
                    len(path.relative_to(root).parts) == 2
                    or (len(path.relative_to(root).parts) == 3 and path.relative_to(root).parts[0] == "tasks")
                )
            )
            or (
                kind == "grade"
                and (
                    path.relative_to(root).as_posix() in {"GRADE_MANIFEST.json", "replay_executor.json"}
                    or (
                        path.suffix.casefold() == ".json"
                        and any(part in path.relative_to(root).parts for part in ("items", "verification_inputs"))
                    )
                )
            )
        )
    else:
        selected = tuple(path for path in files if path.suffix.casefold() == ".json")
    if not selected:
        raise ValueError(f"analysis-reproduction projection has no required {kind} artifacts: {root}")
    return selected


def _require_result_release_identity(
    *,
    kind: ResultArtifactKind,
    root: Path,
    release: _ReleaseContractIdentityV1,
) -> None:
    """Reject required result trees that do not identify the selected release bytes."""

    if kind == "analysis":
        return
    if kind == "run":
        invocation = root / "invocation.json"
        if invocation.is_file():
            payload = read_json(invocation)
            matched = (
                payload.get("release_id") == release.release_id
                and payload.get("release_manifest_sha256") == release.release_manifest_sha256
            )
        else:
            config = root / "run_config.json"
            if not config.is_file() or release.participant_manifest_checksum is None:
                raise ValueError("run result lacks a release-bound invocation or run configuration")
            payload = read_json(config)
            matched = payload.get("participant_manifest_checksum") == release.participant_manifest_checksum
        if not matched:
            raise ValueError("run result identifies different released bytes")
        return
    if kind == "grade":
        manifest = root / "GRADE_MANIFEST.json"
        if manifest.is_file():
            matched = read_json(manifest).get("release_manifest_sha256") == release.release_manifest_sha256
        else:
            task_runs = tuple(trialdev_task_run_paths(root).values())
            if not task_runs:
                ranking = TrialDevRunRankingV1.model_validate(read_json(root / "ranking.json"))
                invocation = read_json(root / "invocation.json")
                matched = (
                    ranking.release_id == release.release_id
                    and ranking.release_manifest_sha256 == release.release_manifest_sha256
                    and ranking.invocation_checksum == invocation.get("checksum")
                    and all(row.execution_status in {"not_started", "incomplete_run"} for row in ranking.items)
                )
            else:
                matched = all(
                    read_json(path).get("release_manifest_sha256") == release.release_manifest_sha256
                    for path in task_runs
                )
        if not matched:
            raise ValueError("grade result identifies different released bytes")
        return
    reports: list[_VerificationReportIdentityV1] = []
    for path in sorted(root.rglob("*.json")):
        payload = read_json(path)
        if isinstance(payload, dict) and payload.get("schema_id") == "trialagentbench.verification_report/v1":
            reports.append(_VerificationReportIdentityV1.model_validate(payload))
    if not reports:
        raise ValueError("verification result lacks an independent release report")
    if any(report.release_digest != release.release_manifest_sha256 for report in reports):
        raise ValueError("verification result identifies different released bytes")


def _source_specs(
    *,
    release_root: Path,
    output_root: Path,
    release_contract: _ReleaseContractIdentityV1,
    roots_by_kind: dict[ResultArtifactKind, Sequence[Path]],
    profile: ResultExportProfile,
) -> tuple[tuple[ResultExportSourceV1, Path, tuple[Path, ...]], ...]:
    specs: list[tuple[ResultExportSourceV1, Path, tuple[Path, ...]]] = []
    observed: set[tuple[ResultArtifactKind, str]] = set()
    for kind in _RESULT_ARTIFACT_KINDS:
        roots = roots_by_kind[kind]
        if len({root.resolve() for root in roots}) != len(roots):
            raise ValueError("duplicate result artifact label")
        label_counts = Counter(_safe_label(root.resolve()) for root in roots)
        for root_path in roots:
            if root_path.is_symlink():
                raise ValueError(f"result artifact root must not be a symbolic link: {root_path}")
            root = root_path.resolve()
            if root.is_relative_to(release_root):
                raise ValueError("result artifacts must remain outside the immutable release")
            if output_root.is_relative_to(root) or root.is_relative_to(output_root):
                raise ValueError("result artifacts and output directory must not contain one another")
            label = _safe_label(root)
            custody_files = _artifact_files(root)
            if label_counts[label] > 1:
                label = f"{label[:47]}-{len(specs):04d}"
            identity = (kind, label)
            if identity in observed:
                raise ValueError(f"duplicate result artifact label for {kind}: {label}")
            observed.add(identity)
            files = (
                custody_files if profile == "full_custody" else _analysis_reproduction_files(kind, root, custody_files)
            )
            _require_result_release_identity(kind=kind, root=root, release=release_contract)
            prefix = f"artifacts/{kind}/{label}"
            specs.append(
                (
                    ResultExportSourceV1(
                        kind=kind,
                        label=label,
                        archive_prefix=prefix,
                        file_count=len(files),
                    ),
                    root,
                    files,
                )
            )
    return tuple(specs)


def _members(
    specs: Iterable[tuple[ResultExportSourceV1, Path, tuple[Path, ...]]],
    *,
    profile: ResultExportProfile,
) -> tuple[ResultExportMemberV1, ...]:
    members: list[ResultExportMemberV1] = []
    for source, root, files in specs:
        for path in files:
            archive_path = f"{source.archive_prefix}/{path.relative_to(root).as_posix()}"
            projected = _projected_json_bytes(path) if profile == "analysis_reproduction" else None
            members.append(
                ResultExportMemberV1(
                    path=archive_path,
                    sha256=(hashlib.sha256(projected).hexdigest() if projected is not None else sha256_file(path)),
                    size_bytes=(len(projected) if projected is not None else path.stat().st_size),
                )
            )
    if len({member.path.casefold() for member in members}) != len(members):
        raise ValueError("result members contain a case-insensitive path collision")
    return tuple(sorted(members, key=lambda member: member.path))


def _json_bytes(model: BaseModel) -> bytes:
    return (json.dumps(model.model_dump(mode="json"), indent=2, sort_keys=True) + "\n").encode("utf-8")


def _zip_info(path: str) -> ZipInfo:
    info = ZipInfo(path, date_time=_FIXED_ZIP_TIME)
    info.compress_type = ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _portable_json(value: JsonValue, *, field: str | None = None) -> JsonValue:
    """Replace machine-local provenance paths with portable artifact labels."""

    if isinstance(value, dict):
        return {key: _portable_json(item, field=key) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_json(item, field=field) for item in value]
    if isinstance(value, str) and field in _PORTABLE_PATH_FIELDS:
        label = Path(value).name
        if not label:
            raise ValueError(f"result metadata field {field!r} has no portable artifact label")
        return label
    return value


def _projected_json_bytes(path: Path) -> bytes | None:
    """Return canonical portable JSON or JSONL bytes, or ``None`` for an opaque member."""

    suffix = path.suffix.casefold()
    if suffix not in {".json", ".jsonl"}:
        return None
    if path.stat().st_size > _MAX_PROJECTED_JSON_BYTES:
        raise ValueError(f"result JSON exceeds the portable-projection byte limit: {path}")
    if suffix == ".json":
        payload = cast(JsonValue, read_json(path))
        projected = _portable_json(payload)
        return (json.dumps(projected, indent=2, sort_keys=True) + "\n").encode("utf-8")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError(f"result JSONL must contain nonblank records: {path}")
    rows = [_portable_json(cast(JsonValue, loads_json(line))) for line in lines]
    return ("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)).encode("utf-8")


def _write_member(archive: ZipFile, *, member: str, source: Path, profile: ResultExportProfile) -> None:
    projected = _projected_json_bytes(source) if profile == "analysis_reproduction" else None
    if projected is not None:
        archive.writestr(_zip_info(member), projected)
        return
    with archive.open(_zip_info(member), "w", force_zip64=True) as destination, source.open("rb") as stream:
        shutil.copyfileobj(stream, destination, length=1024 * 1024)


def _write_archive(
    *,
    path: Path,
    manifest: ResultExportBundleManifestV1,
    specs: tuple[tuple[ResultExportSourceV1, Path, tuple[Path, ...]], ...],
) -> None:
    manifest_bytes = _json_bytes(manifest)
    source_by_prefix = {source.archive_prefix: (root, files) for source, root, files in specs}
    checksum_rows = [f"{hashlib.sha256(manifest_bytes).hexdigest()}  manifest.json"]
    checksum_rows.extend(f"{member.sha256}  {member.path}" for member in manifest.members)
    checksums = ("\n".join(checksum_rows) + "\n").encode("utf-8")
    with ZipFile(path, mode="w", compression=ZIP_DEFLATED, compresslevel=9, allowZip64=True) as archive:
        archive.writestr(_zip_info("manifest.json"), manifest_bytes)
        archive.writestr(_zip_info("SHA256SUMS"), checksums)
        for source in manifest.sources:
            root, files = source_by_prefix[source.archive_prefix]
            for source_file in files:
                member = f"{source.archive_prefix}/{source_file.relative_to(root).as_posix()}"
                _write_member(archive, member=member, source=source_file, profile=manifest.export_profile)


def export_results_v1(
    *,
    release_root: Path,
    output_dir: Path,
    run_roots: Sequence[Path],
    grade_roots: Sequence[Path],
    verification_roots: Sequence[Path],
    analysis_roots: Sequence[Path] = (),
    profile: ResultExportProfile = "full_custody",
) -> Path:
    """Build one deterministic result archive outside the immutable release."""

    supplied_release = release_root
    if supplied_release.is_symlink():
        raise ValueError("release root must not be a symbolic link")
    release = supplied_release.resolve()
    output = Path(output_dir).resolve()
    if not release.is_dir() or release.is_symlink():
        raise ValueError("release root must be a regular directory")
    unsupported_receipt = output / "result_export_receipt.json"
    if output.is_dir() and unsupported_receipt.exists():
        raise ValueError(
            "result_export_receipt.json is not a v1 result-export artifact; "
            "choose a new output directory to produce result_export_manifest.json"
        )
    if output.exists():
        raise FileExistsError(f"result export output already exists: {output}")
    if output.is_relative_to(release):
        raise ValueError("result export output must remain outside the immutable release")
    release_contract = _release_contract(release)
    specs = _source_specs(
        release_root=release,
        output_root=output,
        release_contract=release_contract,
        roots_by_kind={
            "run": run_roots,
            "grade": grade_roots,
            "verification": verification_roots,
            "analysis": analysis_roots,
        },
        profile=profile,
    )
    members = _members(specs, profile=profile)
    manifest = ResultExportBundleManifestV1(
        release_id=release_contract.release_id,
        release_stage=release_contract.release_stage,
        release_manifest_sha256=sha256_file(release_contract.manifest_path),
        release_contract_sha256=release_contract.contract_sha256,
        export_profile=profile,
        sources=tuple(source for source, _, _ in specs),
        members=members,
        member_count=len(members),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        staged.mkdir()
        archive_name = f"{release_contract.release_id}-results.zip"
        archive_path = staged / archive_name
        _write_archive(path=archive_path, manifest=manifest, specs=specs)
        # Read back with the verifier's rule: names unique ignoring case (a clash would overwrite
        # on macOS or Windows) and SHA256SUMS, manifest and member bytes in agreement.
        archive_sha256 = sha256_file(archive_path)
        (staged / f"{archive_name}.sha256").write_text(
            f"{archive_sha256}  {archive_name}\n",
            encoding="utf-8",
        )
        (staged / "result_export_manifest.json").write_bytes(_json_bytes(manifest))
        os.replace(staged, output)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    """Run the public result exporter."""

    args = _parser().parse_args(list(argv) if argv is not None else None)
    output = export_results_v1(
        release_root=args.release_root,
        output_dir=args.output_dir,
        run_roots=args.run_root,
        grade_roots=args.grade_root,
        verification_roots=args.verification_root,
        analysis_roots=args.analysis_root,
        profile=args.profile,
    )
    print(output)
    return 0


__all__ = ["export_results_v1", "main"]
