"""Score-blind validation for the shared numerical CSV export."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path

from trialagentbench_verifier.numerical import read_numerical_table


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trialagentbench verify submission",
        description="Validate estimates.csv against declared target IDs without loading evaluator answers.",
    )
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument(
        "--target-id", action="append", required=True, help="Declared target ID; repeat for each required target."
    )
    parser.add_argument("--output-format", choices=("human", "json"), default="human")
    return parser


def _message(*, valid: bool, error: str | None = None, missing_targets: tuple[str, ...] = ()) -> dict[str, object]:
    return {
        "schema_id": "trialagentbench.numerical_result_validation/v1",
        "missing_targets": list(missing_targets),
        "valid": valid,
        "error": error,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Validate numerical structure and report omitted targets without scoring."""

    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        path = Path(args.submission)
        if path.is_symlink() or not path.is_file():
            raise ValueError("submission must be a regular non-symlink file")
        targets = tuple(args.target_id)
        rows = read_numerical_table(path, target_ids=targets)
        reported = {row.target_id for row in rows}
        payload = _message(valid=True, missing_targets=tuple(target for target in targets if target not in reported))
    except (OSError, UnicodeError, ValueError, csv.Error) as exc:
        error = " ".join(str(exc).split())
        payload = _message(valid=False, error=error[:500])
    if args.output_format == "json":
        print(json.dumps(payload, sort_keys=True))
    elif payload["valid"]:
        print("valid numerical CSV")
        if payload["missing_targets"]:
            print(f"Unreported required targets: {payload['missing_targets']}")
    else:
        print(f"invalid numerical CSV: {payload['error']}")
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
