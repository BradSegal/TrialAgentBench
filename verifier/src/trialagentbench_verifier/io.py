"""Safe, bounded input and canonical byte operations."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import threading
import unicodedata
import zipfile
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from pydantic import JsonValue

if TYPE_CHECKING:
    from trialagentbench_verifier.assets import ReleaseAssets, ReleaseAssetScope

# Full-custody campaigns retain both original and graded analysis snapshots.
# Bound that inventory separately from the bytes of the 384-world frozen pool.
MAX_RELEASE_ARCHIVE_MEMBERS = 4_000_000
MAX_RELEASE_ARCHIVE_BYTES = 32 * 1024**3
MAX_RELEASE_UNCOMPRESSED_BYTES = 64 * 1024**3
MAX_RELEASE_MEMBER_BYTES = 2 * 1024**3
MAX_JSON_BYTES = 1024**3


@dataclass
class _AssetViews:
    """Own selected asset-catalog views for one consumer action."""

    stack: ExitStack
    lock: threading.RLock = field(default_factory=threading.RLock)
    asset_views: dict[Path, tuple[ReleaseAssetScope, int]] = field(default_factory=dict)
    asset_catalogs: dict[Path, tuple[str, ReleaseAssets]] = field(default_factory=dict)
    asset_roots: dict[tuple[Path, tuple[str, ...] | None], Path] = field(default_factory=dict)
    task_selections: dict[Path, tuple[str, ...]] = field(default_factory=dict)


_ACTIVE_ASSETS: _AssetViews | None = None
_ASSET_SCOPE_OWNER = threading.Lock()


@contextmanager
def release_assets(source_root: Path, *, task_selections: dict[Path, tuple[str, ...]] | None = None) -> Iterator[None]:
    """Reuse sparse asset-catalog views until all consumers finish."""
    global _ACTIVE_ASSETS
    if not _ASSET_SCOPE_OWNER.acquire(blocking=False):
        raise RuntimeError("This process already owns a release-archive scope")
    try:
        with ExitStack() as stack:
            _ACTIVE_ASSETS = _AssetViews(stack)
            _ACTIVE_ASSETS.task_selections = {
                path.absolute(): tasks for path, tasks in (task_selections or {}).items()
            }
            try:
                yield
            finally:
                _ACTIVE_ASSETS = None
    finally:
        _ASSET_SCOPE_OWNER.release()


@contextmanager
def registered_asset_scope(root: Path, selection: ReleaseAssetScope) -> Iterator[None]:
    """Bind an authenticated selected view for its drained consumer lifetime.

    Native consumers revisit the same root in worker threads. They may omit
    unselected files only while this owning context is active; a bare partial
    directory, local marker or environment variable cannot weaken verification.
    """
    context = release_assets(root) if _ACTIVE_ASSETS is None else nullcontext()
    with context:
        scope = _ACTIVE_ASSETS
        if scope is None:  # pragma: no cover - context ownership invariant
            raise RuntimeError("asset view has no owning archive scope")
        root = root.resolve(strict=True)
        with scope.lock:
            previous = scope.asset_views.get(root)
            if previous is not None and previous[0] != selection:
                raise RuntimeError("asset view is already bound to another selection")
            scope.asset_views[root] = selection, 1 if previous is None else previous[1] + 1
        try:
            yield
        finally:
            with scope.lock:
                _, references = scope.asset_views[root]
                if references == 1:
                    del scope.asset_views[root]
                else:
                    scope.asset_views[root] = selection, references - 1


def release_asset_scope(root: Path) -> ReleaseAssetScope | None:
    """Return only the live authenticated scope for a release or its role root."""
    from trialagentbench_verifier.archive import ArchivePath

    if isinstance(root, ArchivePath):
        return root.release_scope
    scope = _ACTIVE_ASSETS
    if scope is None:
        return None
    root = root.resolve(strict=True)
    with scope.lock:
        view = scope.asset_views.get(root)
        if view is None and root.name in {"participant", "evaluation"}:
            view = scope.asset_views.get(root.parent)
        return None if view is None else view[0]


class VerificationInputError(ValueError):
    """Raised when released inputs are invalid, unsafe, or inconsistent."""


def sha256_file(path: Path) -> str:
    """Return the streaming SHA-256 identity of one regular file."""

    if path.is_symlink():
        raise VerificationInputError(f"hash input must be one regular non-symlinked file: {path}")
    from trialagentbench_verifier.archive import ArchivePath

    if isinstance(path, ArchivePath):
        with path.open("rb") as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            return digest.hexdigest()
    digest = hashlib.sha256()
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise VerificationInputError(f"hash input must be one regular non-symlinked file: {path}") from error
    with os.fdopen(descriptor, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise VerificationInputError(f"hash input must be one regular non-symlinked file: {path}")
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bounded_archive_file(path: Path) -> Path:
    """Require a regular archive-sized input before any hashing or copying."""

    path = Path(path)
    if path.is_symlink() or not path.is_file() or not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
        raise VerificationInputError(f"archive input must be one regular non-symlinked file: {path}")
    if path.stat().st_size > MAX_RELEASE_ARCHIVE_BYTES:
        raise VerificationInputError(f"archive input exceeds the file size limit: {path}")
    return path


@contextmanager
def release_archive(path: Path, *, expected_sha256: str | None) -> Iterator[Path]:
    """Authenticate supplied transport digests before exposing released inputs."""
    source = bounded_archive_file(path).absolute()
    if source.name.endswith(".assets.json"):
        from trialagentbench_verifier.assets import read_asset_catalog

        if expected_sha256 is None:
            raise VerificationInputError("Asset catalog requires its released manifest digest")
        context = release_assets(source.parent) if _ACTIVE_ASSETS is None else nullcontext()
        with context:
            owner = _ACTIVE_ASSETS
            if owner is None:
                raise RuntimeError("catalog loading has no asset owner")
            with owner.lock:
                catalog = read_asset_catalog(source, expected_sha256=expected_sha256)
                owner.asset_catalogs[source] = expected_sha256, catalog
            yield source
    else:
        if expected_sha256 is not None and sha256_file(source) != expected_sha256:
            raise VerificationInputError("archive differs from the externally trusted SHA-256")
        yield source


def canonical_json_bytes(payload: JsonValue) -> bytes:
    """Encode one JSON value with sorted keys, no whitespace and ASCII escapes."""

    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(payload: JsonValue) -> str:
    """Hash one JSON value through the public canonical encoding."""

    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def read_json_object(path: Path) -> dict[str, JsonValue]:
    """Read one strict UTF-8 JSON object."""

    source = path
    if not source.is_file() or source.is_symlink():
        raise VerificationInputError(f"required JSON file is missing or unsafe: {source}")
    if source.stat().st_size > MAX_JSON_BYTES:
        raise VerificationInputError(f"JSON input exceeds the size limit: {source}")

    def reject_duplicate_fields(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {}
        for key, value in pairs:
            if key in result:
                raise VerificationInputError(f"JSON input repeats field {key!r}: {source}")
            result[key] = value
        return result

    def reject_non_finite(value: str) -> JsonValue:
        raise VerificationInputError(f"JSON input contains non-finite value {value!r}: {source}")

    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_fields,
            parse_constant=reject_non_finite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationInputError(f"invalid UTF-8 JSON: {source}") from error
    if not isinstance(payload, dict):
        raise VerificationInputError(f"expected one JSON object: {path}")
    return payload


def read_json_lines(path: Path) -> tuple[dict[str, JsonValue], ...]:
    """Read a bounded UTF-8 JSONL file as strict objects."""

    source = path
    if not source.is_file() or source.is_symlink():
        raise VerificationInputError(f"required JSONL file is missing or unsafe: {source}")
    if source.stat().st_size > MAX_JSON_BYTES:
        raise VerificationInputError(f"JSONL input exceeds the size limit: {source}")

    def reject_non_finite(value: str) -> JsonValue:
        raise VerificationInputError(f"JSONL input contains non-finite value {value!r}: {source}")

    records: list[dict[str, JsonValue]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise VerificationInputError(f"JSONL input contains a blank record: {source}:{line_number}")

        def reject_duplicate_fields(
            pairs: list[tuple[str, JsonValue]],
            record_number: int = line_number,
        ) -> dict[str, JsonValue]:
            result: dict[str, JsonValue] = {}
            for key, value in pairs:
                if key in result:
                    raise VerificationInputError(f"JSONL input repeats field {key!r}: {source}:{record_number}")
                result[key] = value
            return result

        try:
            record = json.loads(
                line,
                object_pairs_hook=reject_duplicate_fields,
                parse_constant=reject_non_finite,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise VerificationInputError(f"invalid UTF-8 JSONL: {source}:{line_number}") from error
        if not isinstance(record, dict):
            raise VerificationInputError(f"expected one JSON object: {source}:{line_number}")
        records.append(record)
    if not records:
        raise VerificationInputError(f"JSONL input is empty: {source}")
    return tuple(records)


def _safe_member(info: zipfile.ZipInfo) -> PurePosixPath:
    normalized = unicodedata.normalize("NFC", info.filename)
    member = PurePosixPath(info.filename)
    if (
        normalized != info.filename
        or member.is_absolute()
        or ".." in member.parts
        or "\\" in info.filename
        or ":" in info.filename
        or member.as_posix() != info.filename.rstrip("/")
    ):
        raise VerificationInputError("release archive contains a non-portable member")
    mode = info.external_attr >> 16
    kind = stat.S_IFMT(mode)
    if info.is_dir() or kind not in {0, stat.S_IFREG}:
        raise VerificationInputError("release archive contains a non-regular member")
    return member


@contextmanager
def materialized_release(release: Path, *, task_ids: tuple[str, ...] | None = None) -> Iterator[Path]:
    """Yield safe extraction for consumers requiring actual workspace files."""
    source = Path(release)
    scope = _ACTIVE_ASSETS
    if scope is not None and source in scope.asset_catalogs:
        from trialagentbench_verifier.assets import ReleaseAssetScope, materialize_asset_selection

        selected = task_ids if task_ids is not None else scope.task_selections.get(source)
        key = source, selected
        with scope.lock:
            if key not in scope.asset_roots:
                digest, catalog = scope.asset_catalogs[source]
                cache = os.environ.get("TRIALAGENTBENCH_CACHE_DIR")
                if cache:
                    from trialagentbench_verifier.asset_cache import cached_asset_selection

                    asset_root = scope.stack.enter_context(
                        cached_asset_selection(
                            source,
                            expected_sha256=digest,
                            # Resolved, so views compare equal to the resolved members inside them.
                            cache=Path(cache).resolve(),
                            task_ids=selected,
                            base_url=os.environ.get("TRIALAGENTBENCH_ASSET_BASE_URL"),
                        )
                    )
                else:
                    temporary = scope.stack.enter_context(
                        tempfile.TemporaryDirectory(prefix="trialagentbench-assets-")
                    )
                    asset_root = Path(temporary) / "release"
                    materialize_asset_selection(source, expected_sha256=digest, output=asset_root, task_ids=selected)
                scope.stack.enter_context(registered_asset_scope(asset_root, ReleaseAssetScope(catalog, selected)))
                scope.asset_roots[key] = asset_root
            asset_root = scope.asset_roots[key]
        yield asset_root
        return
    with _extract_release(source) as root:
        yield root


@contextmanager
def _extract_release(release: Path) -> Iterator[Path]:
    """Validate and extract one bounded release without caching untrusted inputs."""

    source = Path(release)
    if source.is_symlink():
        raise VerificationInputError("release root cannot be a symbolic link")
    if source.is_dir():
        yield source.resolve(strict=True)
        return
    source = bounded_archive_file(source)
    if not zipfile.is_zipfile(source):
        raise VerificationInputError("release must be a directory or ZIP archive")
    with zipfile.ZipFile(source) as archive:
        infos = archive.infolist()
        if len(infos) > MAX_RELEASE_ARCHIVE_MEMBERS:
            raise VerificationInputError("release archive contains too many members")
        if sum(info.file_size for info in infos) > MAX_RELEASE_UNCOMPRESSED_BYTES:
            raise VerificationInputError("release archive exceeds the uncompressed size limit")
        if any(info.file_size > MAX_RELEASE_MEMBER_BYTES for info in infos):
            raise VerificationInputError("release archive member exceeds the file size limit")
        members = tuple(_safe_member(info) for info in infos)
        if len(members) != len(set(members)):
            raise VerificationInputError("release archive repeats a member path")
        portable_keys = tuple(unicodedata.normalize("NFC", member.as_posix()).casefold() for member in members)
        if len(portable_keys) != len(set(portable_keys)):
            raise VerificationInputError("release archive contains a portable path collision")
        with tempfile.TemporaryDirectory(prefix="trialagentbench-verifier-") as temporary:
            root = Path(temporary)
            archive.extractall(root)
            if (root / "release_manifest.json").is_file():
                yield root
                return
            if (root / "manifest.json").is_file():
                yield root
                return
            roots = tuple(path.parent for path in root.glob("*/release_manifest.json"))
            if len(roots) != 1:
                raise VerificationInputError("release archive does not contain one release root")
            yield roots[0]


__all__ = [
    "release_assets",
    "bounded_archive_file",
    "release_archive",
    "canonical_json_bytes",
    "canonical_sha256",
    "materialized_release",
    "read_json_lines",
    "read_json_object",
    "sha256_file",
]


def trialdev_task_run_paths(root: Path) -> dict[str, Path]:
    """Locate task records directly under the run root or under its tasks directory.

    Parameters
    ----------
    root
        One retained TrialDev invocation directory.

    Returns
    -------
    dict of str to Path
        Task directory names mapped to their regular run records.

    Raises
    ------
    ValueError
        If a task appears twice or a record is reached through a symbolic link.
    """
    records: dict[str, Path] = {}
    for directory in (root, root / "tasks"):
        if directory.is_symlink():
            raise ValueError("TrialDev task directory cannot be a symbolic link")
        for path in sorted(directory.glob("*/run.json")):
            if path.is_symlink() or path.parent.is_symlink() or not path.is_file():
                raise ValueError("TrialDev task run must be a regular confined file")
            identity = path.parent.name
            if identity in records:
                raise ValueError("TrialDev run contains duplicate task directories")
            records[identity] = path
    return records
