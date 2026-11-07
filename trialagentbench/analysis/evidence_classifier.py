"""Origin-first evidence classification for action traces."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from trialagentbench.contracts.analysis.evidence_classification import (
    EvidenceClassificationBasisV1,
    EvidenceClassificationResultV1,
    EvidenceSourceRoleV1,
    ScratchArtifactKindV1,
)
from trialagentbench.contracts.release.trialdev_runtime_surface import (
    TrialDevPublicMemberRoleV1,
    classify_trialdev_public_member,
)
from trialagentbench.contracts.release.trialeval_runtime_surface import (
    TrialEvalItemMemberRoleV1,
    classify_trialeval_participant_member,
)
from trialagentbench.contracts.trace.observable import EvidenceCategoryV1

HIDDEN_EVIDENCE_PARTS = frozenset(
    {
        "grader",
        "hidden",
        "reference",
        "target_manifests",
        "grade_report.json",
    }
)
SHELL_PREFIX_RE = re.compile(r"^(cat|write|read|load|open|inspect|ls|head|tail|grep|sed|python|python3)\s+", re.I)
SCHEMA_KEYS = frozenset({"properties", "required", "fields", "columns", "variables", "schema", "data_dictionary"})
REQUEST_KEYS = frozenset(
    {
        "action",
        "asset_id",
        "study_id",
        "analysis_method",
        "estimates",
        "diagnostics",
    }
)
MODEL_KEYS = frozenset(
    {
        "coef",
        "coefficient",
        "coefficients",
        "prediction",
        "predictions",
        "propensity",
        "weights",
        "weight",
        "p_value",
        "pvalue",
        "model",
        "fit",
        "estimate",
    }
)
UNCERTAINTY_KEYS = frozenset(
    {"ci", "confidence_interval", "lower", "upper", "se", "stderr", "standard_error", "bootstrap", "interval"}
)
SURVIVAL_KEYS = frozenset({"time", "event", "hazard", "cox", "km", "kaplan", "rmst", "survival"})
SUMMARY_KEYS = frozenset({"rate", "rates", "count", "counts", "mean", "median", "group", "arm", "treatment"})
LISTING_KEYS = frozenset({"path", "paths", "file", "files", "directory", "directories", "cwd", "env"})
CODE_TOKENS = ("import ", "def ", "class ", "pd.", "np.", "statsmodels", "lifelines", "sklearn", "select ")


class EvidenceClassificationError(ValueError):
    """Raised when an evidence source cannot be classified deterministically."""


def is_hidden_or_grader_path(path: str) -> bool:
    """Return whether a path references hidden or grader-only evidence."""
    parts = {part.lower() for part in PurePosixPath(path.replace("\\", "/")).parts}
    lowered = path.lower()
    return bool(parts & HIDDEN_EVIDENCE_PARTS) or "/grader/" in lowered or "/hidden/" in lowered


def is_shell_literal_or_pseudo_path(path: str) -> bool:
    """Return whether a string is a shell command, glob, regex, or pseudo-path."""
    stripped = path.strip()
    lowered = stripped.lower().replace("\\", "/")
    return bool(
        "*" in stripped
        or "[" in stripped
        or "\\" in stripped
        or SHELL_PREFIX_RE.search(stripped)
        or (" " in stripped and not lowered.startswith(("/", "./", "../")))
    )


def _release_public_dir(trialdev_release_root: Path | None, scenario_id: str | None) -> Path | None:
    if trialdev_release_root is None or not scenario_id:
        return None
    participant = (
        trialdev_release_root / "participant"
        if (trialdev_release_root / "participant").is_dir()
        else trialdev_release_root
    )
    return participant / "worlds" / scenario_id


Resolution = tuple[Path | None, EvidenceSourceRoleV1, tuple[EvidenceClassificationBasisV1, ...]]


def _workspace_file(workspace: Path, path: Path) -> Resolution | None:
    """Resolve a task-workspace read as the agent's file tools do.

    A bare name absent from the workspace root means ``scratch/<name>``. The harness publishes
    readouts (``evidence/``, ``analysis_ready/``), saved long tool outputs (``scratch/outputs/``)
    and the compaction archive (``scratch/earlier_turns.json``); release materials are left to
    release resolution.
    """
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return None
    if path.parts[0] != "scratch" and not (workspace / path).exists():
        path = Path("scratch") / path
    located = workspace / path
    if path.parts[:2] == ("scratch", "outputs") or path.parts == ("scratch", "earlier_turns.json"):
        return (located if located.is_file() else None), "conversation_event", ("conversation_event_resolution",)
    if not located.is_file():
        return None
    if path.parts[0] == "scratch":
        return located, "agent_scratch_file", ("directory_role",)
    if path.parts[0] in {"evidence", "analysis_ready"}:
        return located, "run_internal_file", ("directory_role",)
    return None


def _resolve_path(
    observed_path: str,
    *,
    submission_paths: tuple[str, ...],
    program_dir: Path | None,
    workspace: Path | None,
    trialdev_release_root: Path | None,
    scenario_id: str | None,
    participant_release_relative: bool,
) -> Resolution:
    path = Path(observed_path)
    submission_set = {Path(value) for value in submission_paths}
    for submission_path in submission_set:
        if observed_path == submission_path.as_posix() or path == submission_path:
            return (
                submission_path if submission_path.exists() else None,
                "submitted_payload",
                ("submission_path_resolution",),
            )
    if workspace is not None and (published := _workspace_file(workspace, path)) is not None:
        return published
    if path.exists():
        public_dir = _release_public_dir(trialdev_release_root, scenario_id)
        if public_dir is not None and path.resolve().is_relative_to(public_dir.resolve()):
            return path.resolve(), "release_public_file", ("release_manifest_resolution",)
        return (
            path,
            "conversation_event" if path.name == "conversation.json" else "run_internal_file",
            ("conversation_event_resolution" if path.name == "conversation.json" else "directory_role",),
        )
    if participant_release_relative:
        if path.is_absolute() or ".." in path.parts or path.as_posix() != observed_path:
            raise EvidenceClassificationError(
                f"Participant release path must be a normalized relative path: {observed_path}"
            )
        if path.parts and path.parts[0] == "scratch":
            return None, "agent_scratch_file", ("transient_scratch_reference",)
        try:
            classify_trialeval_participant_member(observed_path)
        except ValueError:
            # Accessed paths are agent-chosen; one that names no participant member is an
            # unresolved agent reference, not grounds to abort trace export after grading.
            return None, "agent_scratch_file", ("transient_scratch_reference",)
        return path, "release_public_file", ("release_manifest_resolution",)
    if program_dir is not None and not path.is_absolute():
        candidate = program_dir / observed_path
        if candidate.exists():
            if any(part in {"workspace", "scratch", "reports"} for part in candidate.parts):
                return candidate, "agent_scratch_file", ("directory_role",)
            return candidate, "run_internal_file", ("directory_role",)
    public_dir = _release_public_dir(trialdev_release_root, scenario_id)
    if public_dir is not None:
        direct = public_dir / observed_path
        by_name = public_dir / path.name
        if direct.exists():
            return direct, "release_public_file", ("release_manifest_resolution",)
        if by_name.exists():
            return by_name, "release_public_file", ("release_manifest_resolution",)
    if path.is_absolute() or observed_path.startswith(("./", "../")) or "/" in observed_path:
        return None, "agent_scratch_file", ("transient_scratch_reference",)
    if program_dir is not None and path.suffix.lower() in {".json", ".csv", ".parquet", ".md", ".txt", ".tsv"}:
        return None, "agent_scratch_file", ("transient_scratch_reference",)
    raise EvidenceClassificationError(f"Cannot classify unresolved evidence source: {observed_path}")


def _read_bounded_text(path: Path, max_text_bytes: int) -> str:
    data = path.read_bytes()[:max_text_bytes]
    return data.decode("utf-8", errors="ignore")


def _flatten_json_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            keys.add(lowered)
            keys.update(_flatten_json_keys(item))
    elif isinstance(value, list):
        for item in value[:20]:
            keys.update(_flatten_json_keys(item))
    return keys


def _csv_header(path: Path, max_text_bytes: int) -> set[str]:
    text = _read_bounded_text(path, max_text_bytes)
    sample = text.splitlines()[:5]
    if not sample:
        return set()
    reader = csv.reader(sample)
    try:
        header = next(reader)
    except (csv.Error, StopIteration):
        return set()
    return {column.strip().lower() for column in header if column.strip()}


def _kind_from_tokens(tokens: Iterable[str]) -> tuple[ScratchArtifactKindV1, EvidenceCategoryV1]:
    token_set = {token.lower() for token in tokens if token}
    joined = " ".join(sorted(token_set))
    if token_set & SCHEMA_KEYS:
        return "schema_or_dictionary", "scratch_schema_dump"
    if token_set & REQUEST_KEYS:
        return "contract_or_request_copy", "scratch_required_fields_or_contract_copy"
    if token_set & UNCERTAINTY_KEYS:
        return "uncertainty_result", "scratch_ci_or_uncertainty_result"
    if token_set & SURVIVAL_KEYS:
        return "survival_result", "scratch_survival_result"
    if token_set & MODEL_KEYS:
        return "model_result", "scratch_model_result"
    if token_set & SUMMARY_KEYS:
        return "summary_table", "scratch_summary_table"
    if token_set & LISTING_KEYS:
        return "diagnostic_listing", "scratch_or_diagnostic_file"
    if any(token in joined for token in CODE_TOKENS):
        return "code_fragment", "scratch_or_diagnostic_file"
    return "diagnostic_listing", "scratch_or_diagnostic_file"


def classify_scratch_artifact_kind(
    path: Path | None,
    observed_path: str,
    *,
    max_text_bytes: int,
) -> tuple[ScratchArtifactKindV1, EvidenceCategoryV1, tuple[EvidenceClassificationBasisV1, ...]]:
    """Classify a scratch artifact from bounded content, not filename lists."""
    if path is None or not path.exists():
        return "transient_unresolved_workfile", "scratch_or_diagnostic_file", ("transient_scratch_reference",)
    if path.suffix.lower() in {".json", ".jsonl"}:
        text = _read_bounded_text(path, max_text_bytes)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            kind, category = _kind_from_tokens(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text.lower()))
            return kind, category, ("text_signature",)
        kind, category = _kind_from_tokens(_flatten_json_keys(payload))
        return kind, category, ("json_key_signature",)
    if path.suffix.lower() in {".csv", ".tsv"}:
        kind, category = _kind_from_tokens(_csv_header(path, max_text_bytes))
        return kind, category, ("table_column_signature",)
    if path.suffix.lower() in {".py", ".r", ".sql"}:
        return "code_fragment", "scratch_or_diagnostic_file", ("file_extension",)
    if path.suffix.lower() == ".parquet":
        return "summary_table", "scratch_summary_table", ("file_extension",)
    text = _read_bounded_text(path, max_text_bytes)
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text.lower())
    kind, category = _kind_from_tokens(tokens)
    return kind, category, ("text_signature",)


def _category_for_resolved_path(
    path: Path,
    role: EvidenceSourceRoleV1,
    *,
    participant_release_relative: bool,
) -> EvidenceCategoryV1:
    lowered = path.as_posix().lower()
    name = path.name.lower()
    if role == "hidden_or_grader_file":
        return "protected_reference_or_grader_artifact"
    if role == "submitted_payload":
        return "analysis_or_submission_workfile"
    if name == "conversation.json" or role == "conversation_event":
        return "analysis_or_submission_workfile"
    if role == "release_public_file" and participant_release_relative:
        participant_role = classify_trialeval_participant_member(path.as_posix())
        trialeval_category_by_role: dict[TrialEvalItemMemberRoleV1, EvidenceCategoryV1] = {
            TrialEvalItemMemberRoleV1.TASK: "protocol_or_program_contract",
            TrialEvalItemMemberRoleV1.PROTOCOL: "protocol_or_program_contract",
            TrialEvalItemMemberRoleV1.ANALYSIS_INSTRUCTIONS: "protocol_or_program_contract",
            TrialEvalItemMemberRoleV1.ENDPOINT: "protocol_or_program_contract",
            TrialEvalItemMemberRoleV1.INTERCURRENT_EVENT_STRATEGY: "protocol_or_program_contract",
            TrialEvalItemMemberRoleV1.RECONSTRUCTION_RECIPE: "protocol_or_program_contract",
            TrialEvalItemMemberRoleV1.DATA_DICTIONARY: "data_dictionary_or_schema",
            TrialEvalItemMemberRoleV1.PROVENANCE: "protocol_or_program_contract",
            TrialEvalItemMemberRoleV1.DOCUMENTATION: "protocol_or_program_contract",
            TrialEvalItemMemberRoleV1.OUTPUT_CONTRACT: "protocol_or_program_contract",
            TrialEvalItemMemberRoleV1.INTERFACE_TOOL: "analysis_or_submission_workfile",
            TrialEvalItemMemberRoleV1.PREPARED_ANALYSIS_DATA: "trial_population_table",
            TrialEvalItemMemberRoleV1.SOURCE_DOMAIN_DATA: "trial_population_table",
        }
        return trialeval_category_by_role[participant_role]
    if role == "release_public_file" and not participant_release_relative:
        relative = path.relative_to(path.anchor) if path.is_absolute() else path
        public_role = classify_trialdev_public_member(relative.as_posix())
        trialdev_category_by_role: dict[TrialDevPublicMemberRoleV1, EvidenceCategoryV1] = {
            TrialDevPublicMemberRoleV1.SCENARIO_CONTEXT: "protocol_or_program_contract",
            TrialDevPublicMemberRoleV1.DECISION_POLICY: "protocol_or_program_contract",
            TrialDevPublicMemberRoleV1.INTERFACE_CONTRACT: "protocol_or_program_contract",
            TrialDevPublicMemberRoleV1.DATA_DICTIONARY: "data_dictionary_or_schema",
            TrialDevPublicMemberRoleV1.OBSERVATIONAL_DATA: "observational_extract",
            TrialDevPublicMemberRoleV1.PROSPECTIVE_DATA: "trial_population_table",
        }
        return trialdev_category_by_role[public_role]
    if name in {"committee_brief.md", "study.json"}:
        return "protocol_or_program_contract"
    if any(token in lowered for token in ("observational_extract", "observational", "obs_extract")):
        return "observational_extract"
    if any(
        token in lowered
        for token in (
            "data_dictionary",
            "schema.json",
        )
    ):
        return "data_dictionary_or_schema"
    if any(
        token in lowered
        for token in ("evidence_protocol", "development_constraints", "portfolio_policy", "protocol", "sap")
    ):
        return "protocol_or_program_contract"
    if "submission" in lowered:
        return "analysis_or_submission_workfile"
    if name in {"portfolio_decision.json", "evidence_submission.json"}:
        return "trial_design_request"
    if "trial_output/participants.parquet" in lowered:
        return "trial_population_table"
    if any(
        token in lowered
        for token in (
            "participants.parquet",
            "subjects.parquet",
            "population.parquet",
            "baseline",
            "covariate",
            "sites",
        )
    ):
        return "baseline_covariates"
    if any(token in lowered for token in ("adtte", "tte", "survival", "endpoint", "endpoints.parquet")):
        return "time_to_event"
    if any(token in lowered for token in ("safety", "ae", "teae", "adverse")):
        return "safety_events"
    if any(token in lowered for token in ("longitudinal", "marker", "lab", "vital")):
        return "longitudinal_markers"
    if any(
        token in lowered
        for token in ("randomization", "randomisation", "treatment", "arm_mapping", "assignment", "exposure")
    ):
        return "randomization_or_treatment_assignment"
    if any(token in lowered for token in ("missing", "query", "data_quality", "defect", "disposition", "visits")):
        return "missingness_or_data_quality"
    if "estimates.json" in lowered:
        return "simulator_output_public_summary"
    if "budget" in lowered or "cost" in lowered:
        return "cost_or_budget"
    if any(token in lowered for token in ("portfolio_state", "prior_state", "transition")):
        return "phase_state_or_prior_decision"
    return "scratch_or_diagnostic_file" if role == "agent_scratch_file" else "analysis_or_submission_workfile"


def classify_evidence_source(
    observed_path: str,
    *,
    event_source_path: str | None = None,
    submission_paths: tuple[str, ...] = (),
    program_dir: Path | None = None,
    trialdev_release_root: Path | None = None,
    scenario_id: str | None = None,
    participant_release_relative: bool = False,
    workspace: Path | None = None,
    max_text_bytes: int = 8192,
) -> EvidenceClassificationResultV1:
    """Classify one observed source using origin first and bounded content.

    ``workspace`` is the task workspace the agent's paths are relative to; it defaults to
    ``program_dir/workspace`` (TrialDev).
    """
    raw = observed_path.strip()
    if not raw or raw == "not_available":
        return EvidenceClassificationResultV1(
            observed_path=raw or "not_available",
            canonical_source_path=None,
            source_role="not_available_by_design",
            evidence_category="scratch_or_diagnostic_file",
            scratch_artifact_kind="transient_unresolved_workfile",
            basis=("transient_scratch_reference",),
            participant_facing=True,
            hidden_or_grader=False,
            supports_positive_method_claim=False,
        )
    if is_shell_literal_or_pseudo_path(raw):
        return EvidenceClassificationResultV1(
            observed_path=raw,
            canonical_source_path=None,
            source_role="shell_literal_or_pseudo_path",
            evidence_category="shell_literal_or_pseudo_path",
            scratch_artifact_kind="not_scratch",
            basis=("shell_literal_rule",),
            participant_facing=True,
            hidden_or_grader=False,
            supports_positive_method_claim=False,
        )
    if is_hidden_or_grader_path(raw):
        path = Path(raw)
        return EvidenceClassificationResultV1(
            observed_path=raw,
            canonical_source_path=path.as_posix() if path.exists() else None,
            source_role="hidden_or_grader_file",
            evidence_category="protected_reference_or_grader_artifact",
            scratch_artifact_kind="not_scratch",
            basis=("hidden_path_boundary",),
            participant_facing=False,
            hidden_or_grader=True,
            supports_positive_method_claim=False,
        )
    if event_source_path and Path(event_source_path).name == "conversation.json":
        program_dir = program_dir or Path(event_source_path).parent
    resolved, role, basis = _resolve_path(
        raw,
        submission_paths=submission_paths,
        program_dir=program_dir,
        workspace=workspace or (program_dir / "workspace" if program_dir is not None else None),
        trialdev_release_root=trialdev_release_root,
        scenario_id=scenario_id,
        participant_release_relative=participant_release_relative,
    )
    if resolved is not None and is_hidden_or_grader_path(resolved.as_posix()):
        role = "hidden_or_grader_file"
        basis = ("hidden_path_boundary",)
    if role == "agent_scratch_file":
        kind, category, kind_basis = classify_scratch_artifact_kind(resolved, raw, max_text_bytes=max_text_bytes)
        basis = tuple(dict.fromkeys((*basis, *kind_basis)))
    else:
        kind = "not_scratch"
        category = _category_for_resolved_path(
            resolved or Path(raw),
            role,
            participant_release_relative=participant_release_relative,
        )
    participant_facing = role != "hidden_or_grader_file"
    supports_positive = role in {"release_public_file", "submitted_payload", "conversation_event", "run_internal_file"}
    return EvidenceClassificationResultV1(
        observed_path=raw,
        canonical_source_path=resolved.as_posix() if resolved is not None else None,
        source_role=role,
        evidence_category=category,
        scratch_artifact_kind=kind,
        basis=basis,
        participant_facing=participant_facing,
        hidden_or_grader=role == "hidden_or_grader_file",
        supports_positive_method_claim=supports_positive,
    )


__all__ = [
    "EvidenceClassificationError",
    "classify_evidence_source",
    "classify_scratch_artifact_kind",
    "is_hidden_or_grader_path",
    "is_shell_literal_or_pseudo_path",
]
