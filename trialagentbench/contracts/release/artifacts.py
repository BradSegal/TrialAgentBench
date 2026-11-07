"""Canonical public release artifact identities."""

from __future__ import annotations

TRIALEVAL_PARTICIPANT_ARCHIVE_NAME = "TrialEvalBench_participant.zip"
TRIALEVAL_EVALUATOR_ARCHIVE_NAME = "TrialEvalBench_evaluator.zip"
TRIALEVAL_PARTICIPANT_ARCHIVE_IDENTITY_NAME = "participant_archive_identity.json"
TRIALEVAL_PARTICIPANT_ARCHIVE = f"TrialEvalBench/{TRIALEVAL_PARTICIPANT_ARCHIVE_NAME}"
TRIALEVAL_EVALUATOR_ARCHIVE = f"TrialEvalBench/{TRIALEVAL_EVALUATOR_ARCHIVE_NAME}"
TRIALEVAL_PARTICIPANT_ROOT_MEMBERS: frozenset[str] = frozenset(
    {
        "benchmark_charter.json",
        "benchmark_map.md",
        "diagnostic_dictionary.json",
        "diagnostic_methods.csv",
        "manifest.json",
    }
)
