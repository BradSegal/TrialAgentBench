"""Independent verification and scoring for TrialAgentBench releases."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trialagentbench_verifier.contracts import (
        CheckResultV1,
        ExecutionProofV1,
        NormalizedSubmissionV1,
        PracticeVerificationReportV1,
        VerificationReportV1,
    )
    from trialagentbench_verifier.execution import ExecutionSubmissionV1, execute_submission, execute_submissions
    from trialagentbench_verifier.practice import verify_practice_suite
    from trialagentbench_verifier.release import (
        ScoreSubmissionV1,
        score_submission,
        score_submissions,
        verify_release,
    )

__all__ = [
    "CheckResultV1",
    "ExecutionProofV1",
    "PracticeVerificationReportV1",
    "NormalizedSubmissionV1",
    "VerificationReportV1",
    "ExecutionSubmissionV1",
    "ScoreSubmissionV1",
    "execute_submissions",
    "score_submissions",
    "execute_submission",
    "verify_practice_suite",
    "score_submission",
    "verify_release",
]

_EXPORTS = {
    "CheckResultV1": "trialagentbench_verifier.contracts",
    "ExecutionProofV1": "trialagentbench_verifier.contracts",
    "NormalizedSubmissionV1": "trialagentbench_verifier.contracts",
    "PracticeVerificationReportV1": "trialagentbench_verifier.contracts",
    "VerificationReportV1": "trialagentbench_verifier.contracts",
    "ExecutionSubmissionV1": "trialagentbench_verifier.execution",
    "execute_submission": "trialagentbench_verifier.execution",
    "execute_submissions": "trialagentbench_verifier.execution",
    "verify_practice_suite": "trialagentbench_verifier.practice",
    "ScoreSubmissionV1": "trialagentbench_verifier.release",
    "score_submission": "trialagentbench_verifier.release",
    "score_submissions": "trialagentbench_verifier.release",
    "verify_release": "trialagentbench_verifier.release",
}


def __getattr__(name: str) -> object:
    """Load a public entry only when a consumer requests it."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Keep public entries discoverable without importing their implementations."""
    return sorted(set(globals()) | set(_EXPORTS))
