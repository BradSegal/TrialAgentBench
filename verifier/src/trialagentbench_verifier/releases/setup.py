"""Authenticate evaluator inputs and configure a new benchmark campaign."""

from __future__ import annotations

import json
from pathlib import Path

from trialagentbench_verifier.artifacts.anchor import ReleaseDatasetAnchorV1, ReleaseInputResolver
from trialagentbench_verifier.artifacts.asset_cache import acquire_asset_selection
from trialagentbench_verifier.artifacts.assets import materialize_asset_selection, read_asset_catalog
from trialagentbench_verifier.artifacts.io import VerificationInputError, read_json_object, sha256_file


def setup_campaigns(release: Path, anchor: Path, templates: Path, output: Path) -> tuple[Path, ...]:
    """Create smoke and campaign configurations from authenticated release bytes.

    Parameters
    ----------
    release
        Downloaded dataset root containing ``SHA256SUMS`` and ``full/``.
    anchor
        Source release's ``release-dataset.json`` declaration.
    templates
        Directory containing the smoke and campaign configuration templates.
    output
        Configuration, evaluator and cache parent. Existing outputs are refused.

    Returns
    -------
    tuple[Path, ...]
        Absolute paths to the created smoke and campaign configurations.

    Raises
    ------
    VerificationInputError
        If the release, evaluator catalog or grading key fails authentication.
    FileExistsError
        If a configuration or evaluator directory already exists.
    """
    release, output = release.resolve(), output.resolve()
    declaration = ReleaseDatasetAnchorV1.model_validate_json(anchor.read_bytes())
    if declaration.repository != "BradSegal/TrialAgentBench":
        raise VerificationInputError("dataset anchor names an unsupported repository")
    catalog = release / "full/trialeval-evaluator.assets.json"
    sums = release / "SHA256SUMS"
    resolver = ReleaseInputResolver(catalog, anchor_sha256=declaration.sha256sums_sha256, sums=sums)
    digest = resolver.require(catalog)
    authenticated = read_asset_catalog(catalog, expected_sha256=digest)
    key = next((member for member in authenticated.members if member.path == "grading_key/records.jsonl"), None)
    if key is None:
        raise VerificationInputError("evaluator catalog omits the grading key")
    configs = {name: read_json_object(templates / f"{name}.json") for name in ("smoke", "campaign")}
    paths = tuple(output / f"{name}.json" for name in configs)
    evaluator = output / "evaluator"
    for path in (*paths, evaluator):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    panel = configs["smoke"].get("panel")
    if not isinstance(panel, dict):
        raise VerificationInputError("smoke template requires a panel")
    tasks = panel.get("trialeval_tasks")
    if not isinstance(tasks, list) or not tasks or any(not isinstance(task, str) for task in tasks):
        raise VerificationInputError("smoke template requires TrialEval task identifiers")
    selected = tuple(task for task in tasks if isinstance(task, str))
    cached = acquire_asset_selection(
        catalog,
        expected_sha256=digest,
        cache=output / "cache",
        task_ids=selected,
        base_url=f"https://huggingface.co/datasets/{declaration.repository}/resolve/{declaration.revision}/full",
    )
    materialize_asset_selection(cached, expected_sha256=digest, output=evaluator, task_ids=selected)
    key_path = evaluator / key.path
    if sha256_file(key_path) != key.sha256:
        raise VerificationInputError("materialized grading key differs from its authenticated catalog")
    for config, path in zip(configs.values(), paths, strict=True):
        config.update(
            release_directory=str(release / "full"),
            anchor_sha256=declaration.sha256sums_sha256,
            sums=str(sums),
            grading_key=str(key_path),
            grading_key_sha256=key.sha256,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2)
            stream.write("\n")
    return paths
