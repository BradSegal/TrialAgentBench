"""IO utilities for the harness (JSON, text, checksums)."""

from __future__ import annotations

from trialagentbench.io.checksums import (
    canonical_payload_sha256,
    sha256_dir_digest,
    sha256_dir_files,
    sha256_file,
    sha256_path,
)
from trialagentbench.io.directory import staged_directory
from trialagentbench.io.json import (
    loads_json,
    read_json,
    read_json_model,
    read_jsonl_models,
    write_json,
    write_json_model,
)

__all__ = [
    "read_json",
    "loads_json",
    "read_json_model",
    "read_jsonl_models",
    "write_json",
    "write_json_model",
    "canonical_payload_sha256",
    "sha256_file",
    "sha256_dir_files",
    "sha256_dir_digest",
    "sha256_path",
    "staged_directory",
]
