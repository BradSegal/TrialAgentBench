"""Checksum utilities for audit and provenance (SHA-256).

The harness uses checksums to make derived bundles reproducible and auditable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic.types import JsonValue


def canonical_json_bytes(payload: JsonValue) -> bytes:
    """Encode one value using the TrialAgentBench release protocol."""

    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_payload_sha256(payload: JsonValue) -> str:
    """Hash one JSON payload using the release protocol."""

    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 65536) -> str:
    """Compute SHA-256 over a file's bytes.

    Parameters
    ----------
    path:
        File to hash.
    chunk_size:
        Read size for streaming hashing.

    Returns
    -------
    str
        Lowercase hex digest.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dir_files(root: Path) -> dict[str, str]:
    """Compute SHA-256 for every file under `root` (relative-path keyed).

    Notes
    -----
    - Ordering is stable: paths are sorted lexicographically.
    - Symlinks are ignored; only regular files are included.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(root)
    out: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.is_symlink():
            rel = str(p.relative_to(root))
            out[rel] = sha256_file(p)
    return out


def sha256_dir_digest(root: Path) -> str:
    """Compute a single digest over a directory tree (path+hash manifest)."""
    files = sha256_dir_files(root)
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(rel.encode("utf-8"))
        h.update(b"\n")
        h.update(files[rel].encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def sha256_path(path: Path) -> str:
    """Compute a deterministic SHA-256 digest for a file or directory."""

    if path.is_symlink():
        raise ValueError(f"Refusing to hash a symbolic link: {path}")
    if path.is_file():
        return sha256_file(path)
    if path.is_dir():
        return sha256_dir_digest(path)
    raise FileNotFoundError(path)


__all__ = [
    "canonical_json_bytes",
    "canonical_payload_sha256",
    "sha256_file",
    "sha256_dir_files",
    "sha256_dir_digest",
    "sha256_path",
]
