from __future__ import annotations

import os
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


def test_distributions_contain_public_runtime_and_reproducibility_assets(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "dist"
    subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(output), str(root)],
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = tuple(output.glob("*.whl"))
    sdists = tuple(output.glob("*.tar.gz"))
    assert len(wheels) == 1
    assert len(sdists) == 1

    with zipfile.ZipFile(wheels[0]) as archive:
        members = set(archive.namelist())

    prefix = "trialagentbench/resources"
    assert {
        f"{prefix}/examples/campaign.json",
        f"{prefix}/examples/smoke.json",
    } <= members
    assert not any(member.startswith(f"{prefix}/docs/") for member in members)
    assert all(member.startswith(("trialagentbench/", "trial_agent_bench-")) for member in members)
    assert not any(member.startswith("trialagentbench/contracts/experiments/") for member in members)
    assert not any(member.startswith("trialagentbench_qualification/") for member in members)
    assert not any(member in {f"{prefix}/agent_output_schema.json", f"{prefix}/eval_spec.json"} for member in members)

    with tarfile.open(sdists[0]) as archive:
        source_members = {Path(name).name for name in archive.getnames()}
        extracted = tmp_path / "source"
        archive.extractall(extracted, filter="data")
    assert {"README.md", "LICENSE", "Makefile"} <= source_members
    assert "uv.lock" not in source_members
    source_root = next(extracted.iterdir())
    for name in ("RUNNING.md", "METHODS.md"):
        assert (source_root / "docs" / name).read_bytes() == (root / "docs" / name).read_bytes()
    environment = dict(os.environ, PATH=f"{Path(sys.executable).parent}:/usr/bin:/bin")
    if environment.get("PYTHONPATH"):
        environment["PYTHONPATH"] = os.pathsep.join(
            str(Path(path).resolve()) for path in environment["PYTHONPATH"].split(os.pathsep) if path
        )
    subprocess.run(
        ["make", "lint", f"PY={sys.executable}"],
        cwd=source_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_package_metadata_points_to_canonical_public_repositories() -> None:
    root = Path(__file__).resolve().parents[2]
    with (root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]

    assert project["license"] == "Apache-2.0"
    assert project["authors"] == [{"name": "Bradley Segal"}]
    assert project["urls"] == {
        "Homepage": "https://github.com/BradSegal/TrialAgentBench",
        "Repository": "https://github.com/BradSegal/TrialAgentBench",
        "Data": "https://huggingface.co/datasets/BradSegal/TrialAgentBench/",
        "Documentation": "https://github.com/BradSegal/TrialAgentBench/tree/main/docs",
        "Issues": "https://github.com/BradSegal/TrialAgentBench/issues",
    }
