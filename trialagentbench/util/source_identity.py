"""Stable source identities for benchmark execution and scoring behavior."""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable
from pathlib import Path

from pydantic.types import JsonValue
from trialagentbench_verifier.scorer import SCORER_HARNESS_SOURCE_SHA256, python_code_bytes_v1

from trialagentbench.io import canonical_payload_sha256


def source_file_sha256_v1(function: Callable[..., object]) -> str:
    """Hash executable syntax in the source file containing one callable."""

    source = inspect.getsourcefile(function)
    if source is None:
        raise RuntimeError(f"Cannot resolve source identity for {function!r}.")
    path = Path(source)
    if path.is_symlink():
        raise ValueError(f"Refusing to hash a symbolic link: {path}")
    return hashlib.sha256(python_code_bytes_v1(path.read_bytes())).hexdigest()


def _source_tree_sha256(*, include_resources: bool) -> str:
    source_root = Path(__file__).resolve().parents[1]
    source_hashes: dict[str, JsonValue] = {}
    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(source_root)
        if not include_resources and relative.parts[0] == "resources":
            continue
        if path.is_symlink():
            raise ValueError(f"Refusing to hash a symbolic link: {path}")
        source_hashes[relative.as_posix()] = hashlib.sha256(python_code_bytes_v1(path.read_bytes())).hexdigest()
    return canonical_payload_sha256(source_hashes)


def harness_source_sha256_v1() -> str:
    """Hash executable syntax in every installed harness Python module."""
    return _source_tree_sha256(include_resources=True)


def harness_scoring_source_sha256_v1() -> str:
    """Hash harness modules consistently across source and wheel installations.

    Returns
    -------
    str
        Canonical code digest excluding bundled example resources. This declaration is
        carried by the standalone verifier without a dependency on the harness.
    """
    return _source_tree_sha256(include_resources=False)


def require_harness_scorer_v1() -> None:
    """Refuse grading with harness sources outside the declared scorer.

    Raises
    ------
    RuntimeError
        If the installed harness and verifier sources do not belong together.
    """
    if harness_scoring_source_sha256_v1() != SCORER_HARNESS_SOURCE_SHA256:
        raise RuntimeError(
            "Harness sources differ from the declared scorer; install matching harness and verifier builds"
        )


__all__ = [
    "harness_source_sha256_v1",
    "harness_scoring_source_sha256_v1",
    "require_harness_scorer_v1",
    "source_file_sha256_v1",
]
