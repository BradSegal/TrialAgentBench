"""The v1 reporting helpers are frozen; the clarified helpers accept tables and name what they need."""

from __future__ import annotations

import hashlib
import runpy
from pathlib import Path

import pytest

from trialagentbench_verifier.execution_contracts import RELEASE_EXECUTION_LIMITS_V1, ExecutorEnvironmentV1
from trialagentbench_verifier.numerical import (
    read_numerical_table,
    replay_numerical_program,
    stage_reporting_interface,
)

# SHA-256 of the 1.0 helper files staged for targets ("a", "b"); the v1 condition must not change them.
_V1_HELPERS = {
    "targets.csv": "8fbb784f7e9aa627e4f9e4857e006e586b8c7dc2337adb16b68cbde276af45f3",
    "reporting.py": "8ae4db3cd64750884700d841218e155c8cc12c5e4f2731b964aedd373e62a868",
    "reporting.R": "d70ddf46deb4a4064c2cc927e80f1c80d5997c58d30328b787d0906a806de328",
}


def _staged_hashes(root: Path, **kwargs: str) -> dict[str, str]:
    root.mkdir()
    stage_reporting_interface(root, ("a", "b"), **kwargs)  # type: ignore[arg-type]
    return {name: hashlib.sha256((root / "interface" / name).read_bytes()).hexdigest() for name in _V1_HELPERS}


def test_v1_helpers_are_the_1_0_helpers(tmp_path: Path) -> None:
    assert _staged_hashes(tmp_path / "default") == _V1_HELPERS
    assert _staged_hashes(tmp_path / "v1", interface="v1") == _V1_HELPERS
    clarified = _staged_hashes(tmp_path / "clarified", interface="clarified")
    assert clarified["targets.csv"] == _V1_HELPERS["targets.csv"]
    assert clarified["reporting.py"] != _V1_HELPERS["reporting.py"]
    assert clarified["reporting.R"] != _V1_HELPERS["reporting.R"]


def _clarified_python(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    stage_reporting_interface(tmp_path, ("a", "b"), interface="clarified")
    monkeypatch.chdir(tmp_path)
    return runpy.run_path(str(tmp_path / "interface" / "reporting.py"))


def test_clarified_python_report_accepts_a_data_frame_or_row_dicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pandas = pytest.importorskip("pandas")
    helper = _clarified_python(tmp_path, monkeypatch)
    assert helper["targets"]() == ["a", "b"]  # type: ignore[operator]
    frame = pandas.DataFrame(
        {
            "target_id": ["a", "b", "earlier"],
            "estimate": [1.0, 2.0, 9.0],
            "lower": [0.0, 1.0, 8.0],
            "upper": [2, 3, 10],
        }
    )
    helper["report"](frame)  # type: ignore[operator]
    from_frame = read_numerical_table(tmp_path / "scratch" / "estimates.csv", target_ids=("a", "b"))
    helper["report"](  # type: ignore[operator]
        [
            {"target_id": "a", "estimate": 1.0, "lower": 0.0, "upper": 2.0, "note": "extra keys are dropped"},
            {"target_id": "b", "estimate": 2.0, "lower": 1.0, "upper": 3.0},
        ]
    )
    from_rows = read_numerical_table(tmp_path / "scratch" / "estimates.csv", target_ids=("a", "b"))
    assert [(row.target_id, row.estimate, row.upper) for row in from_frame] == [("a", 1.0, 2.0), ("b", 2.0, 3.0)]
    assert from_rows == from_frame


def test_clarified_python_report_names_the_columns_and_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    helper = _clarified_python(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match=r"target_id.*estimate.*lower.*upper.*\['a', 'b'\]"):
        helper["report"]([{"target": "a", "estimate": 1.0, "lower": 0.0, "upper": 2.0}])  # type: ignore[operator]
    with pytest.raises(ValueError, match="the targets are"):
        helper["report"](["a", "b"])  # type: ignore[operator]


@pytest.fixture(scope="module")
def live_executor() -> ExecutorEnvironmentV1:
    """Bind container tests to the executor built on this host."""
    from trialagentbench_verifier.executor import resolve_executor_environment

    return resolve_executor_environment(limits=RELEASE_EXECUTION_LIMITS_V1)


def _replay(workspace: Path, entrypoint: str, interface: str, executor: ExecutorEnvironmentV1):  # type: ignore[no-untyped-def]
    return replay_numerical_program(
        workspace=workspace,
        entrypoint=entrypoint,  # type: ignore[arg-type]
        target_ids=("a", "b"),
        evidence_artifacts=(),
        executor_image=executor.image_id,
        executor_limits=RELEASE_EXECUTION_LIMITS_V1,
        capture_result=True,
        reporting_interface=interface,  # type: ignore[arg-type]
    )


@pytest.mark.executor
def test_replay_stages_the_run_conditions_helpers(tmp_path: Path, live_executor: ExecutorEnvironmentV1) -> None:
    """A data-frame report reproduces under clarified and fails under v1; R accepts a list of lists."""
    python = tmp_path / "python"
    (python / "scratch").mkdir(parents=True)
    (python / "scratch" / "analysis.py").write_text(
        "import pandas as pd\nfrom interface.reporting import targets, report\n"
        "report(pd.DataFrame({'target_id': targets(), 'estimate': [1.0, 2.0], 'lower': [0.0, 1.0], "
        "'upper': [2.0, 3.0]}))\n"
    )
    assert _replay(python, "analysis.py", "clarified", live_executor).reproducible
    assert _replay(python, "analysis.py", "v1", live_executor).status == "execution_error"
    r = tmp_path / "r"
    (r / "scratch").mkdir(parents=True)
    (r / "scratch" / "analysis.R").write_text(
        "source('interface/reporting.R')\n"
        "report(lapply(targets(), function(t) list(target_id=t, estimate=1, lower=0, upper=2)))\n"
    )
    clarified = _replay(r, "analysis.R", "clarified", live_executor)
    assert clarified.reproducible
    assert [row.target_id for row in clarified.estimates or ()] == ["a", "b"]
