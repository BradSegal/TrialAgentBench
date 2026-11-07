from __future__ import annotations

import json
from pathlib import Path

import pytest

from trialagentbench.analysis.evidence_classifier import (
    EvidenceClassificationError,
    classify_evidence_source,
)


def _release_root(tmp_path: Path) -> Path:
    root = tmp_path / "release"
    public = root / "participant" / "worlds" / "world_s01"
    public.mkdir(parents=True)
    (public / "committee_brief.md").write_text("[]", encoding="utf-8")
    (public / "study.json").write_text("{}", encoding="utf-8")
    return root


def _program_dir(tmp_path: Path) -> Path:
    program = tmp_path / "run" / "tasks" / "td-000000000000"
    (program / "reports").mkdir(parents=True)
    (program / "conversation.json").write_text("[]", encoding="utf-8")
    (program / "reports" / "evidence_submission.json").write_text("{}", encoding="utf-8")
    return program


def test_bare_public_filename_resolves_to_release_public_file(tmp_path: Path) -> None:
    result = classify_evidence_source(
        "committee_brief.md",
        program_dir=_program_dir(tmp_path),
        trialdev_release_root=_release_root(tmp_path),
        scenario_id="world_s01",
    )

    assert result.source_role == "release_public_file"
    assert result.evidence_category == "protocol_or_program_contract"
    assert result.supports_positive_method_claim is True
    assert result.canonical_source_path and result.canonical_source_path.endswith("world_s01/committee_brief.md")


def test_submitted_payload_resolves_as_submitted_payload(tmp_path: Path) -> None:
    program = _program_dir(tmp_path)
    submission = program / "reports" / "evidence_submission.json"
    result = classify_evidence_source(
        submission.as_posix(),
        program_dir=program,
        submission_paths=(submission.as_posix(),),
    )

    assert result.source_role == "submitted_payload"
    assert result.evidence_category == "analysis_or_submission_workfile"
    assert result.supports_positive_method_claim is True


def test_conversation_path_resolves_as_conversation_event(tmp_path: Path) -> None:
    program = _program_dir(tmp_path)
    result = classify_evidence_source((program / "conversation.json").as_posix(), program_dir=program)

    assert result.source_role == "conversation_event"
    assert result.evidence_category == "analysis_or_submission_workfile"
    assert result.supports_positive_method_claim is True


def test_transient_tmp_reference_is_not_positive_method_support() -> None:
    result = classify_evidence_source(Path("/", "tmp", "model_result.csv").as_posix())

    assert result.source_role == "agent_scratch_file"
    assert result.scratch_artifact_kind == "transient_unresolved_workfile"
    assert result.evidence_category == "scratch_or_diagnostic_file"
    assert result.supports_positive_method_claim is False


def test_classified_scratch_file_is_not_positive_method_support(tmp_path: Path) -> None:
    program = _program_dir(tmp_path)
    scratch = program / "scratch"
    scratch.mkdir()
    (scratch / "model.csv").write_text("coef,p_value\n1,0.1\n", encoding="utf-8")

    result = classify_evidence_source("scratch/model.csv", program_dir=program)

    assert result.scratch_artifact_kind == "model_result"
    assert result.supports_positive_method_claim is False


@pytest.mark.parametrize("name", ["analysis.py", "analysis.R", "notes"])
@pytest.mark.parametrize("prefix", ["", "scratch/"])
def test_workspace_reader_scratch_paths_resolve_without_becoming_source_evidence(tmp_path, name, prefix):
    program = _program_dir(tmp_path)
    scratch = program / "workspace" / "scratch"
    scratch.mkdir(parents=True)
    source = scratch / name
    source.write_text("Participant-authored working file\n")
    result = classify_evidence_source(prefix + name, program_dir=program)
    assert result.canonical_source_path == str(source)
    assert result.source_role == "agent_scratch_file"
    assert result.supports_positive_method_claim is False


@pytest.mark.parametrize(
    ("path", "role", "category"),
    [
        ("evidence/study-201-look-1/data_dictionary.csv", "run_internal_file", "data_dictionary_or_schema"),
        ("evidence/study-201-look-1/study.json", "run_internal_file", "protocol_or_program_contract"),
        ("analysis_ready/study-201-look-1/safety.parquet", "run_internal_file", "safety_events"),
        ("scratch/outputs/turn-4-1.txt", "conversation_event", "analysis_or_submission_workfile"),
        ("scratch/earlier_turns.json", "conversation_event", "analysis_or_submission_workfile"),
    ],
)
def test_harness_published_trialdev_files_are_not_agent_scratch(tmp_path, path, role, category):
    program = _program_dir(tmp_path)
    published = program / "workspace" / path
    published.parent.mkdir(parents=True)
    published.write_text("{}")
    result = classify_evidence_source(path, program_dir=program)
    assert (result.source_role, result.evidence_category) == (role, category)
    assert result.canonical_source_path == str(published)


def test_scratch_shorthand_does_not_shadow_workspace_public_file(tmp_path):
    program = _program_dir(tmp_path)
    scratch = program / "workspace" / "scratch"
    scratch.mkdir(parents=True)
    (scratch / "committee_brief.md").write_text("{}")
    (program / "workspace" / "committee_brief.md").write_text("{}")
    result = classify_evidence_source(
        "committee_brief.md",
        program_dir=program,
        trialdev_release_root=_release_root(tmp_path),
        scenario_id="world_s01",
    )
    assert result.source_role == "release_public_file"


def test_shell_literal_is_pseudo_path_and_not_positive_support() -> None:
    result = classify_evidence_source("cat committee_brief.md")

    assert result.source_role == "shell_literal_or_pseudo_path"
    assert result.evidence_category == "shell_literal_or_pseudo_path"
    assert result.supports_positive_method_claim is False


def test_hidden_path_is_release_ineligible() -> None:
    result = classify_evidence_source("grader/domains/route_references.jsonl")

    assert result.source_role == "hidden_or_grader_file"
    assert result.evidence_category == "protected_reference_or_grader_artifact"
    assert result.participant_facing is False
    assert result.hidden_or_grader is True


def test_content_classifies_schema_model_survival_uncertainty_and_listing(tmp_path: Path) -> None:
    program = _program_dir(tmp_path)
    scratch = program / "scratch"
    scratch.mkdir()
    schema = scratch / "x.json"
    schema.write_text(json.dumps({"properties": {"age": {"type": "number"}}, "required": ["age"]}), encoding="utf-8")
    model = scratch / "model.csv"
    model.write_text("coef,p_value,propensity\n1,0.1,0.5\n", encoding="utf-8")
    survival = scratch / "survival.csv"
    survival.write_text("time,event,hazard\n1,0,0.1\n", encoding="utf-8")
    interval = scratch / "interval.csv"
    interval.write_text("estimate,lower,upper\n1,0,2\n", encoding="utf-8")
    listing = scratch / "listing.txt"
    listing.write_text("cwd=/tmp\nfiles: a.csv b.json\n", encoding="utf-8")

    assert classify_evidence_source("scratch/x.json", program_dir=program).scratch_artifact_kind == (
        "schema_or_dictionary"
    )
    assert classify_evidence_source("scratch/model.csv", program_dir=program).scratch_artifact_kind == ("model_result")
    assert classify_evidence_source("scratch/survival.csv", program_dir=program).scratch_artifact_kind == (
        "survival_result"
    )
    assert classify_evidence_source("scratch/interval.csv", program_dir=program).scratch_artifact_kind == (
        "uncertainty_result"
    )
    assert classify_evidence_source("scratch/listing.txt", program_dir=program).scratch_artifact_kind == (
        "diagnostic_listing"
    )


def test_unresolved_bare_filename_without_release_context_fails() -> None:
    try:
        classify_evidence_source("committee_brief.md")
    except EvidenceClassificationError as exc:
        assert "Cannot classify unresolved evidence source" in str(exc)
    else:
        raise AssertionError("unresolved bare filenames must fail without release context")


def test_trialeval_participant_members_use_canonical_roles() -> None:
    prepared = classify_evidence_source("data/ADSL.parquet", participant_release_relative=True)
    source = classify_evidence_source("data/raw/ADSL.parquet", participant_release_relative=True)
    protocol = classify_evidence_source("protocol_summary.json", participant_release_relative=True)
    result_guide = classify_evidence_source("analysis_result_guide.md", participant_release_relative=True)
    result_schema = classify_evidence_source(
        "interface/analysis_result.schema.json",
        participant_release_relative=True,
    )
    interface_tool = classify_evidence_source(
        "interface/data_integrity.py",
        participant_release_relative=True,
    )

    assert prepared.evidence_category == "trial_population_table"
    assert source.evidence_category == "trial_population_table"
    assert protocol.evidence_category == "protocol_or_program_contract"
    assert result_guide.evidence_category == "protocol_or_program_contract"
    assert result_schema.evidence_category == "protocol_or_program_contract"
    assert interface_tool.evidence_category == "analysis_or_submission_workfile"


def test_trialeval_instructed_interface_files_and_unknown_agent_paths_classify() -> None:
    """Every file the instructions name classifies; an agent's unknown path never aborts finish (F-4)."""
    for name in ("interface/targets.csv", "interface/reporting.py", "interface/reporting.R"):
        result = classify_evidence_source(name, participant_release_relative=True)
        assert result.source_role == "release_public_file"
    unknown = classify_evidence_source("data/derived/unknown.parquet", participant_release_relative=True)
    assert unknown.source_role == "agent_scratch_file"
    assert unknown.supports_positive_method_claim is False


def test_trialeval_scratch_path_is_not_misclassified_as_participant_evidence() -> None:
    result = classify_evidence_source("scratch/submission.json", participant_release_relative=True)

    assert result.source_role == "agent_scratch_file"
    assert result.evidence_category == "scratch_or_diagnostic_file"
    assert result.supports_positive_method_claim is False


def test_trialdev_public_documents_and_accrued_trials_use_canonical_roles(tmp_path: Path) -> None:
    """Classify every TrialDev evidence class used by the sequential runtime."""

    release = _release_root(tmp_path)
    public = release / "participant" / "worlds" / "world_s01"
    documents = public / "documents"
    data = public / "data"
    documents.mkdir(parents=True)
    data.mkdir()
    (documents / "committee_brief.md").write_text("Current evidence cutoff.\n", encoding="utf-8")
    (documents / "observational_data_dictionary.csv").write_text("name,role\n", encoding="utf-8")
    (data / "aw-12345_exploratory_parallel.parquet").write_bytes(b"PAR1")

    programme = _program_dir(tmp_path)
    state = classify_evidence_source(
        "documents/committee_brief.md",
        program_dir=programme,
        trialdev_release_root=release,
        scenario_id="world_s01",
    )
    dictionary = classify_evidence_source(
        "documents/observational_data_dictionary.csv",
        program_dir=programme,
        trialdev_release_root=release,
        scenario_id="world_s01",
    )
    trial = classify_evidence_source(
        "data/aw-12345_exploratory_parallel.parquet",
        program_dir=programme,
        trialdev_release_root=release,
        scenario_id="world_s01",
    )

    assert state.evidence_category == "protocol_or_program_contract"
    assert dictionary.evidence_category == "data_dictionary_or_schema"
    assert trial.evidence_category == "trial_population_table"


def test_raw_initial_cohort_is_distinct_from_acquired_trial_files(tmp_path: Path) -> None:
    release = _release_root(tmp_path)
    programme = _program_dir(tmp_path)
    public = release / "participant" / "worlds" / "world_s01"
    for directory, expected in (
        ("data/initial", "observational_extract"),
        ("data/acquired", "trial_population_table"),
    ):
        for name in ("participants", "treatments", "clinical_assessments", "safety_assessments"):
            relative = f"{directory}/{name}.parquet"
            path = public / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"PAR1")
            for source in (relative, path.as_posix()):
                result = classify_evidence_source(
                    source, program_dir=programme, trialdev_release_root=release, scenario_id="world_s01"
                )
                assert result.source_role == "release_public_file"
                assert result.evidence_category == expected


def test_authored_study_metadata_is_a_public_contract(tmp_path: Path) -> None:
    root = _release_root(tmp_path)
    program = _program_dir(tmp_path)
    result = classify_evidence_source(
        "study.json", program_dir=program, trialdev_release_root=root, scenario_id="world_s01"
    )
    assert result.source_role == "release_public_file"
    assert result.evidence_category == "protocol_or_program_contract"


@pytest.mark.parametrize("path", ["targets.csv", "interface/nested/targets.csv", "interface/unknown.csv"])
def test_unknown_runtime_interface_members_are_rejected(path: str) -> None:
    """Look-alike paths never join the TrialEval item surface (fail closed)."""
    from trialagentbench.analysis.evidence_classifier import classify_trialeval_participant_member

    with pytest.raises(ValueError, match="Unknown TrialEval item"):
        classify_trialeval_participant_member(path)
