"""Tests for externally anchored release archive materialization."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from trialagentbench.contracts.release.archive_safety import materialized_release_archive


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_zip(path: Path, members: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("release_manifest.json", "{}")
        for name, content in members.items():
            archive.writestr(name, content)


@pytest.mark.parametrize("in_place", [False, True])
@pytest.mark.parametrize("expected", [None, "correct", "f" * 64])
def test_materialized_release_archive_checks_digest(tmp_path: Path, in_place: bool, expected: str | None) -> None:
    """Staging and direct reads authenticate before returning a transport identity."""
    archive = tmp_path / "participant.zip"
    _write_zip(archive, {"participant/manifest.json": "{}"})
    actual = _sha256(archive)
    digest = actual if expected == "correct" else expected
    if expected == "f" * 64:
        with pytest.raises(ValueError, match="externally trusted SHA-256"):
            with materialized_release_archive(archive, expected_sha256=digest, in_place=in_place):
                pytest.fail("an untrusted archive must not be staged or opened")
    else:
        with materialized_release_archive(archive, expected_sha256=digest, in_place=in_place) as materialized:
            assert materialized.archive_sha256 == actual
            assert (materialized.root / "participant/manifest.json").read_text() == "{}"


def test_materialized_release_archive_validates_without_detached_digest(tmp_path: Path) -> None:
    """Ordinary participant use validates the archive without a detached digest."""

    archive = tmp_path / "participant.zip"
    _write_zip(archive, {"participant/manifest.json": '{"release_id":"release-a"}'})

    with materialized_release_archive(archive) as materialized:
        assert materialized.archive_sha256 == _sha256(archive)
        assert (materialized.root / "participant" / "manifest.json").is_file()


def test_materialized_release_archive_rejects_unsafe_anchored_members(tmp_path: Path) -> None:
    """A matching detached digest does not permit unsafe archive paths."""

    archive = tmp_path / "participant.zip"
    _write_zip(archive, {"../escaped.json": "{}"})

    with pytest.raises(ValueError, match="non-portable"):
        with materialized_release_archive(archive, expected_sha256=_sha256(archive)):
            pass


def test_materialized_release_archive_rejects_symlink_input(tmp_path: Path) -> None:
    """Resolving a path must not erase the caller-visible symlink boundary."""

    archive = tmp_path / "participant.zip"
    _write_zip(archive, {"participant/manifest.json": "{}"})
    link = tmp_path / "release.zip"
    link.symlink_to(archive)

    with pytest.raises(ValueError, match="non-symlink"):
        with materialized_release_archive(link, expected_sha256=_sha256(archive)):
            pass


def test_archive_size_is_bounded_before_implicit_hashing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unanchored oversized input must fail before reading its payload."""
    from trialagentbench_verifier import io as verifier_io

    from trialagentbench.contracts.release import archive_safety

    archive = tmp_path / "release.zip"
    _write_zip(archive, {})
    monkeypatch.setattr(verifier_io, "MAX_RELEASE_ARCHIVE_BYTES", 16)

    def unexpected_hash(path: Path) -> str:
        raise AssertionError("Oversized archive was read before its size check")

    monkeypatch.setattr(archive_safety, "sha256_file", unexpected_hash)
    with pytest.raises(ValueError, match="file size limit"):
        with materialized_release_archive(archive):
            pass


def test_standards_zip_is_retained_without_recursive_extraction(tmp_path: Path) -> None:
    """A standards data archive is an opaque member, not an extraction command."""
    nested = tmp_path / "standards.zip"
    _write_zip(nested, {"../must-not-escape.csv": "value\n1\n"})
    archive = tmp_path / "release.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("release_manifest.json", "{}")
        stream.write(nested, "participant/worlds/case/data/standards.zip")
    with materialized_release_archive(archive, expected_sha256=_sha256(archive)) as materialized:
        assert (materialized.root / "participant/worlds/case/data/standards.zip").read_bytes() == nested.read_bytes()
        assert not (materialized.root / "must-not-escape.csv").exists()
        assert not (materialized.root.parent / "must-not-escape.csv").exists()
