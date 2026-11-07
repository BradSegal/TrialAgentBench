"""Summarize benchmark request denominators and resources.

Scientific loss and development value are reported by the grading and campaign commands.
independent_unit_count counts distinct release/task identities, not independent
worlds; linked tasks must not inflate scientific comparison denominators.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from trialagentbench.analysis.calibration import (
    CalibrationPanelV1,
    summarize_calibration_runs_v1,
    validate_calibration_panel_v1,
)
from trialagentbench.io import read_json_model, write_json_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="+", type=Path, help="Run directory or parent containing runs.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--panel",
        action="append",
        type=Path,
        default=[],
        help="Prospective matched-panel JSON to validate; repeat for multiple comparisons.",
    )
    return parser


def _write_rows(path: Path, rows: tuple[BaseModel, ...]) -> None:
    payloads = [row.model_dump(mode="json") for row in rows]
    fieldnames = list(payloads[0]) if payloads else ["schema_id"]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for payload in payloads:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                    for key, value in payload.items()
                }
            )


def main(argv: Sequence[str] | None = None) -> int:
    """Write denominator-complete calibration rows and cell summaries."""

    args = _parser().parse_args(list(argv) if argv is not None else None)
    output = args.out_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite calibration output: {output}")
    summary = summarize_calibration_runs_v1(tuple(args.source))
    for path in args.panel:
        validate_calibration_panel_v1(
            summary.units,
            panel=read_json_model(CalibrationPanelV1, path),
        )
    output.mkdir(parents=True)
    write_json_model(output / "calibration_summary.json", summary)
    _write_rows(output / "calibration_units.csv", summary.units)
    _write_rows(output / "calibration_cells.csv", summary.cells)
    print(
        f"summarized {summary.scheduled_request_count} scheduled requests for "
        f"{summary.independent_unit_count} distinct tasks across {summary.run_count} runs in {output}"
    )
    print(
        "Resource calibration reports execution and consumption. independent_unit_count counts tasks, not worlds. "
        "Use verified grades and campaign comparisons for scientific outcomes."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
