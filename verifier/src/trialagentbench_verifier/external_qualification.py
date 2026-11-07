"""Bind immutable TrialDev data to qualification published separately as evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from trialagentbench_verifier.contracts import QualificationReportIdentityV1
from trialagentbench_verifier.io import VerificationInputError, read_json_object, sha256_file

EXTERNAL_QUALIFICATION_NAME = "qualification_evidence.json"
QUALIFICATION_PENDING = "pending: published as an evidence update with unchanged data; no pass claimed"


class ExternalTrialDevQualificationV1(BaseModel):
    """Initial evidence disposition of one immutable public TrialDev archive."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_id: Literal["trialagentbench.external_trialdev_qualification/v1"] = (
        "trialagentbench.external_trialdev_qualification/v1"
    )
    benchmark_stream: Literal["trialdev"] = "trialdev"
    candidate_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    release_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    qualification_location: Literal["release-evidence"] = "release-evidence"
    initial_qualification_status: Literal["pending"] = "pending"
    initial_recovery_status: Literal["pending"] = "pending"

    @property
    def reports(self) -> tuple[QualificationReportIdentityV1, ...]:
        """Return no embedded qualification claims."""
        return ()


def read_external_qualification(root: Path) -> ExternalTrialDevQualificationV1 | None:
    """Read an optional external-evidence notice and authenticate its candidate.

    Parameters
    ----------
    root
        Materialized evaluator release.

    Returns
    -------
    ExternalTrialDevQualificationV1 or None
        Bound initial pending disposition, or no external-evidence notice.

    Raises
    ------
    VerificationInputError
        If the notice is linked, names other bytes, or shadows embedded qualification.
    """
    path = root / EXTERNAL_QUALIFICATION_NAME
    if not path.exists() and not path.is_symlink():
        return None
    if not path.is_file() or path.is_symlink():
        raise VerificationInputError("external qualification notice must be a regular file")
    try:
        notice = ExternalTrialDevQualificationV1.model_validate(read_json_object(path))
    except ValidationError as error:
        raise VerificationInputError("external qualification notice violates its contract") from error
    manifest = read_json_object(root / "release_manifest.json")
    if (
        manifest.get("schema_id") != "ariadne.trialdev.release/v1"
        or manifest.get("construct_sha256") != notice.candidate_digest
        or sha256_file(root / "release_manifest.json") != notice.release_manifest_sha256
    ):
        raise VerificationInputError("external qualification notice identifies different TrialDev data")
    if (root / "evaluation/verification/qualification_reports.json").exists():
        raise VerificationInputError("external qualification notice cannot shadow embedded qualification")
    return notice
