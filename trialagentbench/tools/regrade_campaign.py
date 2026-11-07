"""Regrade retained campaign submissions through the canonical offline finalizer."""

from __future__ import annotations

import errno
import os
import shutil
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from trialagentbench_verifier.release_set import ReleaseSetManifestV1, authenticate_release_set, role_filename
from trialagentbench_verifier.scorer import CurrentScorer, ScorerIdentityV1, current_scorer

from trialagentbench.contracts.campaign import CampaignPlanV1
from trialagentbench.io import read_json_model, sha256_file, write_json_model
from trialagentbench.util.campaign_control import rebind_ledger
from trialagentbench.util.run_ownership import exclusive_run_directory


class _PreviousGrade(BaseModel):
    """Read retained grade provenance without accepting its scores."""

    model_config = ConfigDict(extra="ignore", frozen=True)
    scorer: ScorerIdentityV1 | None = None
    harness_version: str | None = None
    harness_source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class RegradedRunV1(BaseModel):
    """Original grade and submission custody retained for one regraded job."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    job_id: str
    run_custody_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    previous_grade_path: str | None
    previous_grade_sha256: str | None = Field(pattern=r"^[0-9a-f]{64}$")
    previous_scorer: ScorerIdentityV1 | None
    previous_harness_version: str | None
    previous_harness_source_sha256: str | None = Field(pattern=r"^[0-9a-f]{64}$")


class CampaignRegradeV1(BaseModel):
    """Immutable link to original custody and the scorer used for new outputs.

    A missing scorer denotes a grade with unidentified scientific scoring rules.
    A package version does not identify those rules.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_id: Literal["trialagentbench.campaign_regrade/v1"] = "trialagentbench.campaign_regrade/v1"
    scorer: CurrentScorer
    source_campaign: str
    source_campaign_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    primary_selection_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    oracle_values_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$", exclude_if=lambda value: value is None
    )
    runs: tuple[RegradedRunV1, ...]


def _link_frozen(source: str, destination: str) -> str:
    """Share a frozen custody file's bytes instead of copying them (a copy across filesystems).

    Finishing only reads retained runs, plans and stored replays and publishes new files beside them, so a
    hard link keeps the regrade small without letting it alter the source campaign.
    """
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copy2(source, destination)
    return destination


_CAMPAIGN_ROLES: tuple[Literal["participant", "evaluator"], ...] = ("participant", "evaluator")


def _rebind_release(plan: CampaignPlanV1, release: Path) -> CampaignPlanV1:
    """Bind retained runs to another release that serves byte-identical campaign inputs.

    Agents read only the selected split's role catalogs and ran in the release
    executor, so a release differing elsewhere (for example in its toolkit and
    scorer) can grade them without any agent rerun.
    """
    from trialagentbench.tools.campaign import _role, detached_manifest_digest

    configuration = plan.configuration
    if configuration.input_mode != "qualified":
        raise ValueError("Only campaigns on a qualified release set can be rebound to another release")
    release = release.resolve(strict=True)
    digest = detached_manifest_digest(release, "release-set-manifest.json")
    original = authenticate_release_set(
        configuration.release_directory, expected_manifest_sha256=configuration.expected_release_manifest_sha256
    )
    target = authenticate_release_set(release, expected_manifest_sha256=digest)

    def inputs(manifest: ReleaseSetManifestV1) -> dict[str, str]:
        names = [artifact.filename for artifact in manifest.artifacts]
        used = {
            role_filename(names, benchmark, _role(configuration.panel, role))
            for benchmark in configuration.panel.benchmarks
            for role in _CAMPAIGN_ROLES
        }
        return {artifact.filename: artifact.sha256 for artifact in manifest.artifacts if artifact.filename in used}

    if inputs(original) != inputs(target):
        raise ValueError("Rebinding requires byte-identical campaign inputs in both releases")
    if original.executor.environment_sha256 != target.executor.environment_sha256:
        raise ValueError("Rebinding requires the original executor environment")
    rebound = configuration.model_copy(
        update={"release_directory": release, "expected_release_manifest_sha256": digest}
    )
    return CampaignPlanV1(configuration=rebound, runs=plan.runs)


def regrade_campaign(
    source: Path,
    output: Path,
    *,
    release: Path | None = None,
    replay_bank: tuple[Path, str] | None = None,
    replay_workers: int | None = None,
    oracle_values: tuple[Path, str] | None = None,
    primary_selection: Path | None = None,
) -> None:
    """Copy frozen inputs to a new root and finish them without agent execution.

    Parameters
    ----------
    source
        Retained campaign with original native runs and grading outputs.
    output
        New, disjoint output directory. Existing destinations are refused.
    release
        Optional release set to grade against instead of the original. Its role
        catalogs for the campaign's selection and its executor environment must
        equal the original release's, which must remain available.
    replay_bank
        Optional replay bank (directory, expected manifest SHA-256) for TrialEval
        replay scoring. Stored answer replays are carried over and reused.
    replay_workers
        Simultaneous replay containers; the campaign's executor capacity by default.
    primary_selection
        Optional source-bound manifest selecting earlier completed primaries of
        the same scientific condition. Kept only in the derived output.

    Returns
    -------
    None
        Results and their provenance are written under ``output``.

    Raises
    ------
    ValueError
        If custody is incomplete, contains symbolic links, changes during the
        snapshot, the destination overlaps the source, or ``release`` serves
        different campaign inputs or another executor environment.
    FileExistsError
        If the output already exists. Failed stages remain available for
        inspection and can be continued with ``campaign finish``.
    """
    from trialagentbench.tools.campaign import finish_campaign, load_campaign
    from trialagentbench.tools.finish_campaign_analysis import REPLAY_STORE, campaign_censoring
    from trialagentbench.tools.primary_selection import PRIMARY_SELECTION, TrialEvalPrimarySelectionV1

    source = source.resolve(strict=True)
    if output.is_symlink():
        raise ValueError("Regrade output cannot be a symbolic link")
    output = output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Regrade requires a new output root disjoint from the source campaign")
    if output.exists():
        raise FileExistsError(output)
    selection_source = primary_selection if primary_selection is not None else source / PRIMARY_SELECTION
    if selection_source.is_file():
        read_json_model(TrialEvalPrimarySelectionV1, selection_source)
    elif primary_selection is not None:
        raise FileNotFoundError(selection_source)
    with exclusive_run_directory(source):
        plan = load_campaign(source)
        paths = [
            source / name
            for name in (
                "campaign.json",
                "configuration.json",
                "runs",
                "plans",
                "requests.sqlite",
                "operator-exclusions.json",
                REPLAY_STORE,
            )
        ]
        # Supplementary retries retain their source-bound reclassification evidence.
        paths.extend(
            source / path
            for path in sorted(
                {run.reclassification_audit_path for run in plan.runs if run.reclassification_audit_path is not None}
            )
            if not any(parent in paths for parent in (source / path).parents)
        )
        existing = [path for path in paths if path.exists()]
        for path in existing:
            if path.is_symlink() or (path.is_dir() and any(child.is_symlink() for child in path.rglob("*"))):
                raise ValueError(f"Regrade custody cannot contain symbolic links: {path}")
        rebound = None if release is None else _rebind_release(plan, release)
        censoring = campaign_censoring(source, plan)
        wholly_censored = (
            set()
            if censoring is None
            else {run.job_id for run in censoring.runs if run.wholly_censored and run.benchmark == "trialeval"}
        )
        retained = []
        for run in plan.runs:
            grade = (
                source
                / "results"
                / run.job_id
                / "grade"
                / ("GRADE_MANIFEST.json" if run.benchmark == "trialeval" else "ranking.json")
            )
            if any(path.is_symlink() for path in (source / "results", grade.parent.parent, grade.parent, grade)):
                raise ValueError(f"Regrade grade custody cannot contain symbolic links: {grade}")
            if not grade.is_file() and run.job_id not in wholly_censored:
                raise ValueError(f"Regrade requires the retained original grade: {grade}")
            previous = read_json_model(_PreviousGrade, grade) if grade.is_file() else _PreviousGrade()
            retained.append(
                RegradedRunV1(
                    job_id=run.job_id,
                    previous_grade_path=grade.relative_to(source).as_posix() if grade.is_file() else None,
                    previous_grade_sha256=sha256_file(grade) if grade.is_file() else None,
                    previous_scorer=previous.scorer,
                    previous_harness_version=previous.harness_version,
                    previous_harness_source_sha256=previous.harness_source_sha256,
                )
            )
        # Take only frozen invocation/submission custody: directories as hard links, files (the request
        # ledger, which rebinding rewrites) as copies. Grades, analyses and proofs are produced anew by
        # finish; the original results stay in place.
        output.mkdir(parents=True, exist_ok=False)
        for path in existing:
            destination = output / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if path.is_dir():
                shutil.copytree(path, destination, copy_function=_link_frozen)
            else:
                shutil.copy2(path, destination)
        if rebound is not None:
            if (output / "requests.sqlite").exists():
                rebind_ledger(output / "requests.sqlite", plan.configuration, rebound.configuration)
            write_json_model(output / "configuration.json", rebound.configuration)
            write_json_model(output / "campaign.json", rebound)
        if selection_source.is_file():
            shutil.copy2(selection_source, output / PRIMARY_SELECTION)
        load_campaign(output)
        write_json_model(
            output / "regrade.json",
            CampaignRegradeV1(
                scorer=current_scorer(),
                source_campaign=str(source),
                source_campaign_sha256=sha256_file(source / "campaign.json"),
                primary_selection_sha256=(sha256_file(selection_source) if selection_source.is_file() else None),
                oracle_values_sha256=None if oracle_values is None else oracle_values[1],
                runs=tuple(retained),
            ),
        )
    finish_campaign(output, replay_bank=replay_bank, replay_workers=replay_workers, oracle_values=oracle_values)
