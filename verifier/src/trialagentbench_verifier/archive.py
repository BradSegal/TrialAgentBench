"""Read-only pathlib access to released ZIP members without extracting payloads."""

from __future__ import annotations

import fnmatch
import io
import os
import stat
import zipfile
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path, PosixPath
from typing import IO, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trialagentbench_verifier.assets import ReleaseAssetScope

_OPEN: dict[tuple[str, int], zipfile.ZipFile] = {}


def _archive(filename: str) -> zipfile.ZipFile:
    key = filename, os.getpid()
    if key not in _OPEN:
        _OPEN[key] = zipfile.ZipFile(filename)
    return _OPEN[key]


@lru_cache(maxsize=8)
def _asset_scope(filename: str) -> ReleaseAssetScope:
    from trialagentbench_verifier.assets import ReleaseAssets, ReleaseAssetScope

    return ReleaseAssetScope(ReleaseAssets.model_validate_json(Path(filename).read_bytes()))


@lru_cache(maxsize=8)
def _members(filename: str) -> dict[str, tuple[str, str]]:
    if filename.endswith(".assets.json"):
        catalog = _asset_scope(filename).catalog
        shards = {shard.group: str(Path(filename).parent / shard.filename) for shard in catalog.shards}
        return {member.path: (shards[member.group], f"objects/{member.sha256}") for member in catalog.members}
    from trialagentbench_verifier.io import (
        MAX_RELEASE_ARCHIVE_MEMBERS,
        MAX_RELEASE_MEMBER_BYTES,
        MAX_RELEASE_UNCOMPRESSED_BYTES,
        _safe_member,
    )

    infos = _archive(filename).infolist()
    if (
        len(infos) > MAX_RELEASE_ARCHIVE_MEMBERS
        or sum(info.file_size for info in infos) > MAX_RELEASE_UNCOMPRESSED_BYTES
    ):
        raise ValueError("Release archive exceeds its member or uncompressed size limit")
    if any(info.file_size > MAX_RELEASE_MEMBER_BYTES for info in infos):
        raise ValueError("Release archive member exceeds its size limit")
    names = tuple(_safe_member(info).as_posix() for info in infos)
    if len(names) != len(set(name.casefold() for name in names)):
        raise ValueError("Release archive repeats a portable member path")
    return {name: (filename, name) for name in names}


@lru_cache(maxsize=8)
def _directories(filename: str) -> frozenset[str]:
    return frozenset(
        "/".join(name.split("/")[:index]) for name in _members(filename) for index in range(1, len(name.split("/")))
    )


class ArchivePath(PosixPath):
    """A read-only logical release member, backed by a ZIP or catalog shard."""

    def _logical(self) -> tuple[str, str]:
        filename, _, member = str(self).partition("!")
        return filename, member.lstrip("/")

    @property
    def release_scope(self) -> ReleaseAssetScope | None:
        """Keep a catalog's logical release identity available to native loaders."""
        filename, _ = self._logical()
        return _asset_scope(filename) if filename.endswith(".assets.json") else None

    def _location(self) -> tuple[zipfile.ZipFile, str]:
        filename, member = self._logical()
        try:
            shard, blob = _members(filename)[member]
        except KeyError:
            raise FileNotFoundError(str(self)) from None
        return _archive(shard), blob

    def is_symlink(self) -> bool:
        """Members have passed the release's portable regular-file validation."""
        return False

    def is_file(self) -> bool:
        """Return whether the logical release inventory contains this file."""
        filename, member = self._logical()
        return member in _members(filename)

    def is_dir(self) -> bool:
        """Return whether the logical release contains this directory."""
        filename, member = self._logical()
        return not member or member in _directories(filename)

    def exists(self) -> bool:
        """Return whether the release contains this file or directory."""
        return self.is_file() or self.is_dir()

    def stat(self, *, follow_symlinks: bool = True) -> os.stat_result:
        """Read uncompressed size from the ZIP directory without reading payloads."""
        if self.is_dir():
            return os.stat_result((stat.S_IFDIR | 0o555, 0, 0, 1, 0, 0, 0, 0, 0, 0))
        archive, member = self._location()
        try:
            size = archive.getinfo(member).file_size
        except KeyError:
            raise FileNotFoundError(str(self)) from None
        return os.stat_result((stat.S_IFREG | 0o444, 0, 0, 1, 0, 0, size, 0, 0, 0))

    def resolve(self, strict: bool = False) -> ArchivePath:
        """Return this absolute, validated logical member."""
        if strict and not self.exists():
            raise FileNotFoundError(str(self))
        return self

    def absolute(self) -> ArchivePath:
        """Return this already absolute logical member."""
        return self

    def open(  # type: ignore[override]  # ZIP streams implement IO without being BufferedReader instances.
        self,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> IO[Any]:
        """Read one member in place, supporting seekable Parquet readers."""
        if mode not in {"r", "rb", "rt"}:
            raise ValueError("Release archives are read-only")
        archive, member = self._location()
        handle = archive.open(member)
        return (
            handle
            if "b" in mode
            else io.TextIOWrapper(handle, encoding=encoding or "utf-8", errors=errors, newline=newline)
        )

    def glob(self, pattern: str) -> Generator[ArchivePath, None, None]:
        """List matching logical paths without reading file contents."""
        filename, member = self._logical()
        prefix = member + "/" if member else ""
        for name in sorted(set(_members(filename)) | _directories(filename)):
            if name.startswith(prefix):
                relative = name[len(prefix) :]
                if fnmatch.fnmatchcase(relative, pattern) and (
                    "**" in pattern or len(relative.split("/")) == len(pattern.split("/"))
                ):
                    yield self / relative

    def rglob(self, pattern: str) -> Generator[ArchivePath, None, None]:
        """Recursively list logical release members."""
        for path in self.glob("**"):
            if fnmatch.fnmatchcase(path.name, pattern):
                yield path

    def iterdir(self) -> Generator[ArchivePath, None, None]:
        """List the direct children of this logical directory."""
        yield from self.glob("*")


@contextmanager
def opened_release(
    path: Path,
    *,
    expected_sha256: str | None = None,
    task_ids: tuple[str, ...] | None = None,
    verify_files: bool = False,
) -> Iterator[Path]:
    """Expose a complete release's logical members directly from its transport."""
    from trialagentbench_verifier.io import bounded_archive_file, release_archive

    if path.is_dir():
        yield path
        return
    source = bounded_archive_file(path).resolve()
    catalog = None
    if source.name.endswith(".assets.json"):
        from trialagentbench_verifier.assets import read_asset_catalog

        catalog = (
            _asset_scope(str(source)).catalog
            if expected_sha256 is None
            else read_asset_catalog(source, expected_sha256=expected_sha256)
        )
        catalog.selected_members(task_ids)  # Fail fast on an unknown scheduled task.
    elif expected_sha256 is not None:
        with release_archive(source, expected_sha256=expected_sha256):
            pass
    root = ArchivePath(str(source) + "!/")
    try:
        _members(str(source))
        if verify_files and catalog is not None:
            from trialagentbench_verifier.assets import _verify_shard
            from trialagentbench_verifier.io import VerificationInputError, _safe_member, sha256_file

            checked_shards: set[tuple[str, int]] = set()
            for shard in catalog.shards:
                shard_path = source.parent / shard.filename
                identity = shard.sha256, shard.byte_length
                if identity not in checked_shards:
                    _verify_shard(shard_path, shard)
                    checked_shards.add(identity)
                infos = _archive(str(shard_path)).infolist()
                names = tuple(_safe_member(info).as_posix() for info in infos)
                expected = {f"objects/{member.sha256}" for member in catalog.members if member.group == shard.group}
                if len(names) != len(set(names)) or set(names) != expected:
                    raise VerificationInputError("Asset shard payload census differs from its catalog")
            verified: set[tuple[str, str]] = set()
            for member in catalog.members:
                physical = _members(str(source))[member.path]
                if physical not in verified:
                    path = root / member.path
                    if path.stat().st_size != member.byte_length or sha256_file(path) != member.sha256:
                        raise VerificationInputError(f"Asset payload differs from its release: {member.path}")
                    verified.add(physical)
        yield root
    finally:
        for archive in _OPEN.values():
            archive.close()
        _OPEN.clear()
        _members.cache_clear()
        _directories.cache_clear()
        _asset_scope.cache_clear()
