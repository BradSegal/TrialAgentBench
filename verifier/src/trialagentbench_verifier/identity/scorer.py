"""Content identity of the shared scientific scorer and reporting rules."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from trialagentbench_verifier.identity.python_code import SOURCE_CODE_FORMAT as SOURCE_CODE_FORMAT
from trialagentbench_verifier.identity.python_code import python_code_bytes_v1 as python_code_bytes_v1
from trialagentbench_verifier.identity.source_census import source_census_sha256

SCORER_VERSION = "decimal12-half-even-aggregate-recovery-v2-content"
# The standalone verifier cannot import the harness. Refresh this declaration
# with harness_scoring_source_sha256_v1() whenever harness executable syntax changes.
SCORER_HARNESS_SOURCE_SHA256 = "da781b4f27ebc4c4b56171a6d0879ab72e2dd9bcec8fa46d8578a95f154ddada"


class ScorerIdentityV1(BaseModel):
    """Declared scoring rule and SHA-256 of its installed executable content.

    The source census binds both distributions: logical verifier units with import routing and
    the declared harness source digest, verified by the harness when grading.
    Bundled example scripts are excluded so source and wheel builds agree.
    Harness grades additionally retain their complete harness source digest.
    Package and dataset versions do not identify scoring behavior.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    version: str = Field(min_length=1)
    sources_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


@lru_cache(maxsize=1)
def current_scorer() -> ScorerIdentityV1:
    """Return the installed scorer identity, independent of checkout location.

    Returns
    -------
    ScorerIdentityV1
        Declared rule and a deterministic source census, cached for this process.

    Raises
    ------
    OSError
        If an installed source file cannot be read.

    Examples
    --------
    >>> current_scorer().version == SCORER_VERSION
    True
    """
    root = Path(str(files("trialagentbench_verifier")))
    source_digest = source_census_sha256(root)
    digest = hashlib.sha256(b"trialagentbench-combined-scorer-content-v2\0")
    digest.update(bytes.fromhex(SCORER_HARNESS_SOURCE_SHA256))
    digest.update(bytes.fromhex(source_digest))
    return ScorerIdentityV1(version=SCORER_VERSION, sources_sha256=digest.hexdigest())


def require_current_scorer(identity: ScorerIdentityV1) -> ScorerIdentityV1:
    """Refuse stale scores before analysis or reporting.

    Parameters
    ----------
    identity
        Scorer recorded by the input artifact.

    Returns
    -------
    ScorerIdentityV1
        The validated, unchanged identity.

    Raises
    ------
    ValueError
        If retained outputs need regrading under the installed scorer.
    """
    if identity != current_scorer():
        raise ValueError("Scorer identity differs from the current scorer; regrade retained outputs before reporting")
    return identity


CurrentScorer = Annotated[ScorerIdentityV1, AfterValidator(require_current_scorer)]


def require_same_scorer(identities: Iterable[ScorerIdentityV1]) -> ScorerIdentityV1:
    """Refuse pooling across scorer versions or source hashes.

    Parameters
    ----------
    identities
        Identities of all inputs to a comparison or export, including copied models.

    Returns
    -------
    ScorerIdentityV1
        The common current identity.

    Raises
    ------
    ValueError
        If inputs are empty, mixed, or stale.
    """
    found = tuple(identities)
    if not found or any(identity != found[0] for identity in found):
        raise ValueError("Cannot compare or pool results across different scorer identities")
    return require_current_scorer(found[0])
