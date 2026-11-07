"""Grade, export and report one prepared native run offline.

All calculations belong to the installed harness or standalone verifier. This
delivered recipe connects those existing public operations without selecting
successful cases, replacing primary attempts, or calculating another score.
"""

from __future__ import annotations

import argparse
import os
import re
import uuid
from collections.abc import Sequence
from pathlib import Path

from trialagentbench_verifier.campaign import (
    TrialEvalCampaignPlanV1,
    TrialEvalCampaignUnavailableV1,
    UnsupportedTrialEvalPanelError,
    analyse_trial_eval_campaign_v1,
    render_trial_eval_campaign_analysis_v1,
)
from trialagentbench_verifier.campaign_inputs import authenticate_development_inputs
from trialagentbench_verifier.io import sha256_file
from trialagentbench_verifier.release_set import (
    ReleaseSetManifestV1,
    authenticate_release_set_manifest,
    role_filename,
)
from trialagentbench_verifier.response import TrialDevRunRankingV1
from trialagentbench_verifier.scorer import current_scorer

from trialagentbench.cli import main as harness_main
from trialagentbench.contracts.core.manifest import GradeManifestV1
from trialagentbench.contracts.release.archive_safety import materialized_release_archive
from trialagentbench.io.atomic import _sync_directory
from trialagentbench.tools.export_results import export_results_v1
from trialagentbench.util.source_identity import require_harness_scorer_v1


def _discard_incomplete_writes(directory: Path) -> None:
    """Recover unpublished writer files before resuming an exclusively owned output.

    Only direct children with the writer's exact reserved name are disposable.
    Canonical outputs, arbitrary files and participant workspaces are untouched.
    """
    for path in directory.iterdir():
        if re.fullmatch(r"\..+\.[0-9a-f]{32}\.tmp", path.name):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Incomplete stage output is not a regular file: {path}")
            path.unlink()


def _write_once(path: Path, content: str | bytes) -> None:
    """Retain a completed stage and refuse a conflicting replay of its output."""
    encoded = content.encode("utf-8") if isinstance(content, str) else content
    if path.is_symlink() or path.exists():
        if path.is_symlink() or path.read_bytes() != encoded:
            raise ValueError(f"Existing stage output differs from verified inputs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # Publish only a complete file; a concurrent first writer keeps ownership.
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != encoded:
                raise ValueError(f"Existing stage output differs from verified inputs: {path}") from None
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None, *, qualified_manifest: ReleaseSetManifestV1 | None = None) -> int:
    """Execute offline release operations with externally retained trust anchors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("trialeval", "trialdev"))
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--executor-image-id")
    parser.add_argument("--campaign-plan", type=Path)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--expected-invocation-sha256")
    parser.add_argument("--development", action="store_true", help="Finish explicitly unqualified rehearsal inputs.")
    parser.add_argument("--split", choices=("public", "withheld"), default="public")
    parser.add_argument("--resume", action="store_true", help="Verify and reuse completed offline stages.")
    parser.add_argument(
        "--budget-censored",
        action="append",
        default=[],
        metavar="TASK_ID",
        help="TrialEval task left unrun by an exhausted campaign spending limit; reported, never scored.",
    )
    args = parser.parse_args(argv)
    require_harness_scorer_v1()
    if args.development and args.split != "public":
        parser.error("Development inputs have no withheld evaluation split")
    if args.budget_censored and args.benchmark != "trialeval":
        parser.error("TrialDev rankings record unrun tasks natively; --budget-censored is TrialEval-only")
    if args.benchmark == "trialeval":
        if args.campaign_plan is None or args.expected_plan_sha256 is None or args.executor_image_id is None:
            parser.error("TrialEval requires its prospectively retained plan digest and local executor image ID")
        if sha256_file(args.campaign_plan) != args.expected_plan_sha256:
            raise ValueError("TrialEval plan differs from the prospectively retained identity")
        plan = TrialEvalCampaignPlanV1.model_validate_json(args.campaign_plan.read_bytes())
    elif args.expected_invocation_sha256 is None:
        parser.error("TrialDev requires its prospectively retained invocation digest")
    release_dir = args.release_dir.resolve(strict=True)
    run = args.run_root.resolve(strict=True)
    if args.output.is_symlink():
        raise ValueError("Offline output must not be a symbolic link")
    output = args.output.resolve()
    if args.development and qualified_manifest is not None:
        raise ValueError("Development finishing cannot receive a qualified release manifest")
    manifest = (
        authenticate_development_inputs(release_dir, expected_manifest_sha256=args.expected_manifest_sha256)
        if args.development
        else qualified_manifest
        or authenticate_release_set_manifest(release_dir, expected_manifest_sha256=args.expected_manifest_sha256)
    )
    anchors = {artifact.filename: artifact.sha256 for artifact in manifest.artifacts}
    evaluator_role = "withheld-evaluator" if args.split == "withheld" else "evaluator"
    evaluator = release_dir / role_filename(anchors, args.benchmark, evaluator_role)
    evaluator_sha = anchors[evaluator.name]
    if args.benchmark == "trialdev" and sha256_file(run / "invocation.json") != args.expected_invocation_sha256:
        raise ValueError("TrialDev invocation differs from the prospectively retained identity")
    output.mkdir(parents=True, exist_ok=args.resume)
    if args.resume:
        _discard_incomplete_writes(output)
    if args.development:
        _write_once(output / "development-inputs.json", manifest.model_dump_json(indent=2) + "\n")
    grade = output / "grade"
    grade_arguments = (
        ["grade", "trialeval", str(run), "--evaluator-release", str(evaluator)]
        if args.benchmark == "trialeval"
        else ["grade", "trialdev", "--run-root", str(run), "--release", str(evaluator)]
    )
    if args.benchmark == "trialeval":
        grade_arguments.extend(["--replay-executor-image-id", args.executor_image_id])
        for task_id in args.budget_censored:
            grade_arguments.extend(["--budget-censored", task_id])
    if grade.exists():
        if args.benchmark == "trialeval":
            GradeManifestV1.model_validate_json((grade / "GRADE_MANIFEST.json").read_bytes())
        else:
            TrialDevRunRankingV1.model_validate_json((grade / "ranking.json").read_bytes())
        custody = "run_config.json" if args.benchmark == "trialeval" else "invocation.json"
        complete = "GRADE_MANIFEST.json" if args.benchmark == "trialeval" else "ranking.json"
        if (
            not args.resume
            or grade.is_symlink()
            or not (grade / complete).is_file()
            or sha256_file(grade / custody) != sha256_file(run / custody)
        ):
            raise ValueError("Retained native grade is incomplete or belongs to different run custody")
    elif harness_main([*grade_arguments, "--expected-sha256", evaluator_sha, "--out-dir", str(grade)]) != 0:
        raise RuntimeError("Native grading failed; retain its outputs and resolve the evaluator fault")
    censored_attempts = (
        frozenset(attempt.attempt_id for attempt in plan.attempts if attempt.task_id in set(args.budget_censored))
        if args.benchmark == "trialeval"
        else frozenset()
    )
    exported = output / "export"
    if not exported.exists():
        with materialized_release_archive(evaluator, expected_sha256=evaluator_sha, in_place=True) as materialized:
            export_results_v1(
                release_root=materialized.root,
                output_dir=exported,
                run_roots=(run,),
                grade_roots=(grade,),
                verification_roots=(),
                profile="analysis_reproduction" if args.benchmark == "trialeval" else "full_custody",
            )
    if args.benchmark == "trialeval":
        try:
            analysis = analyse_trial_eval_campaign_v1(
                "",
                expected_release_id=plan.release_id,
                expected_release_manifest_sha256=plan.release_manifest_sha256,
                expected_archive_sha256=None,
                grade_roots=(grade,),
                plan=plan,
                expected_campaign_plan_sha256=plan.checksum,
                budget_censored_attempt_ids=censored_attempts,
            )
        except UnsupportedTrialEvalPanelError as error:
            unavailable = TrialEvalCampaignUnavailableV1(
                scorer=current_scorer(),
                reason=str(error),
                campaign_plan_sha256=plan.checksum,
            )
            _write_once(output / "analysis.json", unavailable.model_dump_json(indent=2) + "\n")
            _write_once(
                output / "analysis.md",
                f"Panel inference unavailable: {error}. Individual verified grades remain in the result export.\n",
            )
            return 0
        _write_once(output / "analysis.json", analysis.model_dump_json(indent=2) + "\n")
        _write_once(output / "analysis.md", render_trial_eval_campaign_analysis_v1(analysis))
    else:
        ranking = TrialDevRunRankingV1.model_validate_json((grade / "ranking.json").read_bytes())
        _write_once(output / "analysis.json", ranking.model_dump_json(indent=2) + "\n")
    print(f"Retain the complete output directory: {output}")
    return 0


if __name__ == "__main__":
    main()
