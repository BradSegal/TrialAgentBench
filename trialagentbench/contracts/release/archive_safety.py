"""Materialize native releases through the shared bounded archive implementation."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from trialagentbench_verifier.assets import ReleaseAssetScope
from trialagentbench_verifier.io import (
    bounded_archive_file,
    materialized_release,
    release_archive,
    release_asset_scope,
    sha256_file,
)


@dataclass(frozen=True)
class MaterializedReleaseArchive:
    """Read-only release root and its declared transport identity."""

    root: Path
    archive_sha256: str
    scope: ReleaseAssetScope | None = None


@contextmanager
def materialized_release_archive(
    path: Path,
    *,
    expected_sha256: str | None = None,
    task_ids: tuple[str, ...] | None = None,
    in_place: bool = False,
) -> Iterator[MaterializedReleaseArchive]:
    """Load a bounded release for workspace staging or reading members in place.

    Standards ZIPs remain ordinary files. Their consumers own their bounded
    decoding; this loader never recursively extracts nested archives.
    """
    path = bounded_archive_file(path)
    digest = sha256_file(path) if expected_sha256 is None else expected_sha256
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("expected SHA-256 must contain 64 lowercase hexadecimal characters")
    if in_place:
        from trialagentbench_verifier.archive import opened_release

        with opened_release(path, expected_sha256=digest, task_ids=task_ids) as root:
            yield MaterializedReleaseArchive(root=root, archive_sha256=digest)
        return
    with release_archive(path, expected_sha256=digest) as authenticated:
        with materialized_release(authenticated, task_ids=task_ids) as root:
            yield MaterializedReleaseArchive(root=root, archive_sha256=digest, scope=release_asset_scope(root))


__all__ = [
    "MaterializedReleaseArchive",
    "materialized_release_archive",
]
