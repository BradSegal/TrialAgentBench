"""Authoritative identities for persisted benchmark run configurations."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel

from trialagentbench.contracts.core.runs import TrialEvalItemResultV1, TrialEvalRunConfigV1
from trialagentbench.io import canonical_payload_sha256, read_json_model


def _config_digest(config: BaseModel) -> str:
    return canonical_payload_sha256(config.model_dump(mode="json", exclude_none=True))


def trialeval_run_id(run_dir: Path) -> str:
    """Return the identity declared by a persisted TrialEval run configuration."""

    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        item_paths = sorted((run_dir / "items").glob("*.json"))
        if not item_paths:
            raise FileNotFoundError(f"TrialEval run is missing persisted configuration: {config_path}")
        configs = [read_json_model(TrialEvalItemResultV1, path).run_config for path in item_paths]
        digests = {_config_digest(config) for config in configs}
        if len(digests) != 1:
            raise ValueError(f"TrialEval item artifacts contain conflicting persisted configurations: {run_dir}")
        return f"trialeval:{next(iter(digests))}"
    config = read_json_model(TrialEvalRunConfigV1, config_path)
    return f"trialeval:{_config_digest(config)}"


def require_unique_run_ids(run_dirs: Iterable[Path], *, suite: str) -> dict[Path, str]:
    """Resolve TrialEval run IDs and reject duplicate persisted identities."""

    if suite != "trialeval":
        raise ValueError("Run identity resolution supports TrialEval runs only")
    identities: dict[Path, str] = {}
    paths_by_id: dict[str, list[Path]] = {}
    for run_dir in run_dirs:
        resolved = run_dir.resolve()
        run_id = trialeval_run_id(resolved)
        identities[resolved] = run_id
        paths_by_id.setdefault(run_id, []).append(resolved)
    collisions = {run_id: paths for run_id, paths in paths_by_id.items() if len(paths) > 1}
    if collisions:
        details = "; ".join(
            f"{run_id} -> {', '.join(path.as_posix() for path in paths)}"
            for run_id, paths in sorted(collisions.items())
        )
        raise ValueError(f"Duplicate persisted {suite} run identity: {details}")
    return identities


__all__ = ["require_unique_run_ids", "trialeval_run_id"]
