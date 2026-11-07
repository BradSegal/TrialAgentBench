"""The staged reporting helper writes exactly the current checkpoint's targets."""

import runpy
from pathlib import Path

from trialagentbench_verifier.numerical import read_numerical_table, stage_reporting_interface


def test_report_keeps_only_current_targets(tmp_path: Path, monkeypatch) -> None:
    """Rows for an earlier checkpoint's targets are left out, so a growing script stays valid."""
    stage_reporting_interface(tmp_path, ("current",))
    monkeypatch.chdir(tmp_path)
    helper = runpy.run_path(str(tmp_path / "interface" / "reporting.py"))
    helper["report"](
        [
            {"target_id": "earlier", "estimate": 1.0, "lower": 0.0, "upper": 2.0},
            {"target_id": "current", "estimate": 3.0, "lower": 2.0, "upper": 4.0},
        ]
    )
    rows = read_numerical_table(tmp_path / "scratch" / "estimates.csv", target_ids=("current",))
    assert [(row.target_id, row.estimate) for row in rows] == [("current", 3.0)]
