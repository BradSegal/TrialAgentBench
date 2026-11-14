"""Offline execution of the independently owned R property models."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from trialagentbench_verifier.artifacts.io import sha256_file


class RPins(BaseModel):
    """Immutable image, interpreter and R package inventory."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    image: str
    image_id: str
    R: str
    packages: dict[str, str]


class RRunner:
    """Run R only inside the pinned, disconnected reference image."""

    def __init__(
        self,
        package: Path,
        output: Path,
        declarations: Path,
        dependencies: Path,
        route_declarations: Path | None = None,
        input_roots: tuple[Path, ...] = (),
    ) -> None:
        self.package = output.resolve() / "r-package"
        shutil.copytree(package, self.package)
        self.output = output.resolve()
        self.declarations = declarations.resolve()
        self.dependencies = dependencies.resolve()
        self.input_roots = tuple(path.resolve() for path in input_roots)
        self.pins = RPins.model_validate_json((package / "package-versions.json").read_bytes())
        actual = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", self.pins.image],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if actual != self.pins.image_id or self.pins.image != "ariadne-reference-r:2025-10-30":
            raise ValueError("R reference image differs from the declared pin")
        (output / "library").mkdir()
        (output / "r-tmp").mkdir()
        if route_declarations is not None:
            self.execute(
                [
                    "Rscript",
                    "/pkg/tools/compile_routes.R",
                    str(route_declarations),
                    str(self.package / "inst"),
                    str(self.package / "inst/route_inputs.json"),
                ],
                output / "compile-routes.log",
            )
        self.execute(["R", "CMD", "INSTALL", "-l", str(output / "library"), "/pkg"], output / "install-r.log")

    def execute(self, command: list[str], log: Path) -> None:
        """Execute with one numerical thread, offline mounts and external scratch."""
        args = [
            "nice",
            "-n",
            "19",
            "ionice",
            "-c3",
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
        ]
        for key, value in {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_CORETYPE": "Haswell",
            "TMPDIR": str(self.output / "r-tmp"),
            "R_LIBS_USER": str(self.output / "library"),
            "R_LIBS": f"{self.output / 'library'}:/deps",
        }.items():
            args.extend(["-e", f"{key}={value}"])
        for source, target, mode in [
            (self.package, Path("/pkg"), "ro"),
            (self.output, self.output, "rw"),
            (self.declarations, self.declarations, "ro"),
            (self.dependencies, Path("/deps"), "ro"),
            *((path, path, "ro") for path in self.input_roots),
        ]:
            args.extend(["-v", f"{source}:{target}:{mode}"])
        args.extend(["-w", str(self.output), self.pins.image, "nice", "-n", "19", "ionice", "-c3", *command])
        with log.open("w") as stream:
            subprocess.run(args, check=True, stdout=stream, stderr=subprocess.STDOUT)

    def run_cell(self, job: Path, *, parameters_only: bool = False) -> None:
        """Execute one cell, preserving per-replication responses on fit failure."""
        command = ["Rscript", "/pkg/tools/statistical.R", str(job)]
        if parameters_only:
            command.append("parameters-only")
        self.execute(command, job.with_suffix(".log"))

    def source_identity(self) -> dict[str, str]:
        """Bind every R source, registry, bridge and package pin used by this run."""
        return {
            p.relative_to(self.package).as_posix(): sha256_file(p)
            for p in sorted(self.package.rglob("*"))
            if p.is_file()
        }
