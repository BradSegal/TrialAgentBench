"""Shared synthetic archive fixtures for transport and cache boundaries."""

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from zipfile import ZipFile

import pytest


def _release(path: Path) -> dict[str, bytes]:
    members = {
        "release_manifest.json": b'{"schema_id":"ariadne.trialdev.release/v1","release_id":"original"}',
        "participant/manifest.json": b'{"role":"participant"}',
        "evaluation/manifest.json": b'{"role":"evaluation"}',
        "evaluation/resource_qualification.json": b'{"qualification":"unchanged"}',
    }
    for index in (1, 2):
        task = f"td-{index:012x}"
        world = f"tdw-{index:012x}"
        members[f"participant/tasks/{task}/task.json"] = json.dumps(
            {"task_id": task, "case_id": world, "data_view": "raw" if index == 1 else "analysis_ready"}
        ).encode()
        members[f"participant/worlds/{world}/observed.csv"] = f"patient,value\n1,{index}\n".encode()
        for partition in (0, 1):
            members[f"evaluation/worlds/{world}/pool/partitions/p{partition:03}/completion.parquet"] = (
                b"identical completion records" * 100
            )
    members["grading_key/world_strata.json"] = b'{"worlds":2}'
    with ZipFile(path, "w") as archive:
        for name, body in members.items():
            archive.writestr(name, body)
    return members


@pytest.fixture
def asset_release_factory() -> Callable[[Path], dict[str, bytes]]:
    """Create a minimal paired release with two worlds and duplicate pool bytes."""
    return _release


@pytest.fixture
def require_records() -> Callable[..., None]:
    """Skip a test whose optional record archives are absent; source distributions omit them."""
    examples = Path(__file__).resolve().parents[1] / "examples" / "scientific"

    def require(*names: str) -> None:
        for name in names:
            if not (examples / name).is_file():
                pytest.skip(f"record archive {name} is not part of this release")

    return require


@pytest.fixture(autouse=True)
def _fresh_executor_probe() -> Iterator[None]:
    """Probe each test's executor image anew; the cache is per process in production."""
    from trialagentbench_verifier.executor import _probe_executor_image

    _probe_executor_image.cache_clear()
    yield
    _probe_executor_image.cache_clear()
