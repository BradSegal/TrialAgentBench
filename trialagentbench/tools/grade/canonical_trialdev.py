"""Offline grading command for canonical TrialDev task runs."""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from trialagentbench.contracts.release.archive_safety import materialized_release_archive
from trialagentbench.contracts.trialdev.run import TrialDevRunInvocationV1
from trialagentbench.io import read_json_model, staged_directory
from trialagentbench.trialdev.regrade import regrade_trialdev_run
from trialagentbench.trialdev.release import load_trialdev_evaluator_release


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True, help="Complete evaluator release ZIP.")
    parser.add_argument("--expected-sha256", required=True, help="Trusted SHA-256 of the evaluator ZIP.")
    parser.add_argument("--run-root", type=Path, required=True, help="Canonical TrialDev result directory.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Destination for the regraded result tree.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Copy and deterministically regrade a canonical TrialDev result tree."""

    args = _parser().parse_args(list(argv) if argv is not None else None)
    source = args.run_root.resolve()
    try:
        invocation = read_json_model(TrialDevRunInvocationV1, source / "invocation.json")
        with materialized_release_archive(
            args.release, expected_sha256=args.expected_sha256, in_place=True, task_ids=invocation.scheduled_task_ids
        ) as materialized:
            release = load_trialdev_evaluator_release(materialized.root)
            if (
                invocation.release_archive_sha256 != materialized.archive_sha256
                or invocation.release_manifest_sha256 != release.participant.release_manifest_sha256
            ):
                raise ValueError("TrialDev run and evaluator release identities differ")
            with staged_directory(args.out_dir.resolve()) as destination:
                shutil.copytree(source, destination, dirs_exist_ok=True)
                runs = regrade_trialdev_run(release=release, run_root=destination)
    except (OSError, ValueError, ValidationError) as error:
        sys.stderr.write(f"TrialDev grading failed: {error}\n")
        return 2
    sys.stdout.write(f"Graded {len(runs)} TrialDev task run(s) in {args.out_dir.resolve()}.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
