"""Regular-file boundaries for release archive inputs."""

import hashlib
import os
import zipfile
from pathlib import Path

import pytest

from trialagentbench_verifier import io as verifier_io


@pytest.mark.parametrize("expected", [None, "correct", "f" * 64, "invalid"])
def test_release_archive_checks_digest_before_yielding(tmp_path: Path, expected: str | None) -> None:
    """Authenticate supplied ZIP digests before any consumer can read members."""
    path = tmp_path / "release.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("release_manifest.json", "{}")
    actual = verifier_io.sha256_file(path)
    digest = actual if expected == "correct" else expected
    if digest is not None and digest != actual:
        with pytest.raises(verifier_io.VerificationInputError, match="externally trusted SHA-256"):
            with verifier_io.release_archive(path, expected_sha256=digest):
                pytest.fail("an archive with the wrong digest must not be yielded")
    else:
        with verifier_io.release_archive(path, expected_sha256=digest) as source:
            assert source == path
            assert verifier_io.sha256_file(source) == actual


def test_release_archive_rejects_symlinks_and_special_files(tmp_path: Path) -> None:
    """Reject filesystem indirection before attempting archive reads."""

    target = tmp_path / "target.zip"
    target.write_bytes(b"zip")
    linked = tmp_path / "linked.zip"
    linked.symlink_to(target)
    with pytest.raises(verifier_io.VerificationInputError, match="regular non-symlinked"):
        with verifier_io.release_archive(linked, expected_sha256=hashlib.sha256(b"zip").hexdigest()):
            pytest.fail("a symlinked archive must not be yielded")

    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "release.fifo"
        os.mkfifo(fifo)
        with pytest.raises(verifier_io.VerificationInputError, match="regular non-symlinked"):
            with verifier_io.release_archive(fifo, expected_sha256="0" * 64):
                pytest.fail("a FIFO must not be yielded")
