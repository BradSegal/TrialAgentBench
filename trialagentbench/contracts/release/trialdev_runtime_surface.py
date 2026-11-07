"""Scientific roles of canonical TrialDev participant artifacts."""

from __future__ import annotations

from enum import StrEnum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Final


class TrialDevPublicMemberRoleV1(StrEnum):
    """Scientific role of one participant-visible TrialDev artifact."""

    SCENARIO_CONTEXT = "scenario_context"
    DECISION_POLICY = "decision_policy"
    INTERFACE_CONTRACT = "interface_contract"
    DATA_DICTIONARY = "data_dictionary"
    OBSERVATIONAL_DATA = "observational_data"
    PROSPECTIVE_DATA = "prospective_data"


TRIALDEV_PUBLIC_FILE_ROLES: Final = MappingProxyType(
    {
        "evidence_analysis_request.md": TrialDevPublicMemberRoleV1.INTERFACE_CONTRACT,
        "evidence_protocol.md": TrialDevPublicMemberRoleV1.SCENARIO_CONTEXT,
        "standards.zip": TrialDevPublicMemberRoleV1.OBSERVATIONAL_DATA,
        "observational_cohort.parquet": TrialDevPublicMemberRoleV1.OBSERVATIONAL_DATA,
        "observational_data_dictionary.csv": TrialDevPublicMemberRoleV1.DATA_DICTIONARY,
        "portfolio_management_request.md": TrialDevPublicMemberRoleV1.INTERFACE_CONTRACT,
        "prospective_data_dictionary.csv": TrialDevPublicMemberRoleV1.DATA_DICTIONARY,
        "committee_brief.md": TrialDevPublicMemberRoleV1.DECISION_POLICY,
        "site_network.csv": TrialDevPublicMemberRoleV1.DECISION_POLICY,
        "study.json": TrialDevPublicMemberRoleV1.SCENARIO_CONTEXT,
        "data_dictionary.csv": TrialDevPublicMemberRoleV1.DATA_DICTIONARY,
        "task.json": TrialDevPublicMemberRoleV1.INTERFACE_CONTRACT,
    }
)


def classify_trialdev_public_member(relative_path: str) -> TrialDevPublicMemberRoleV1:
    """Classify one canonical participant artifact by its normalized path."""

    path = PurePosixPath(relative_path)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != relative_path:
        raise ValueError("TrialDev participant paths must be normalized relative POSIX paths")
    if path.parts[-3:-1] == ("data", "initial") and path.name in {
        "participants.parquet",
        "treatments.parquet",
        "clinical_assessments.parquet",
        "safety_assessments.parquet",
    }:
        return TrialDevPublicMemberRoleV1.OBSERVATIONAL_DATA
    try:
        return TRIALDEV_PUBLIC_FILE_ROLES[path.name]
    except KeyError as error:
        if path.suffix == ".parquet":
            return TrialDevPublicMemberRoleV1.PROSPECTIVE_DATA
        raise ValueError(f"Unknown TrialDev participant artifact: {relative_path!r}") from error


__all__ = ["TrialDevPublicMemberRoleV1", "classify_trialdev_public_member"]
