"""Scientific artifacts cannot silently acquire a new scorer identity."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trialagentbench_verifier.campaign_contracts import CampaignAnalysisV1
from trialagentbench_verifier.development_campaign import compare_trialdev_rankings, trialdev_leaderboard_rows
from trialagentbench_verifier.response import TrialDevRunRankingV1
from trialagentbench_verifier.scorer import ScorerIdentityV1, current_scorer, python_code_bytes_v1, require_same_scorer


def test_code_serialization_has_a_pinned_format():
    assert python_code_bytes_v1(b"answer = 1\n") == b"trialagentbench-python-code-spans-v1\0answer = 1\n"


def test_code_serialization_ignores_nested_docstrings_and_comments():
    source = b'''"""Module description."""
class Example:
    """Class description."""
    def answer(self):
        """Function description."""
        async def nested():
            """Async description."""
            return "message"
        return nested
'''
    changed = source.replace(b"description", b"revised description") + b"# Comment.\n"
    assert python_code_bytes_v1(source) == python_code_bytes_v1(changed)
    assert python_code_bytes_v1(source) != python_code_bytes_v1(source.replace(b'"message"', b'"changed"'))
    assert python_code_bytes_v1(source) != python_code_bytes_v1(source.replace(b"return nested", b"return None"))


def test_code_serialization_keeps_non_docstring_string_expressions():
    assert python_code_bytes_v1(b'VALUE = 1\n"documentation"\n') != python_code_bytes_v1(b'VALUE = 1\n"changed"\n')


def test_code_serialization_refuses_version_dependent_fstring_comments():
    with pytest.raises(ValueError, match="syntax shared by Python 3.11 and 3.12"):
        python_code_bytes_v1(b'f"""{1 # version-dependent comment\n}"""\n')


def test_code_serialization_handles_utf8_offsets_and_overlapping_spans():
    source = 'class Exemple:\n    (  # Description.\n        "Descripci\u00f3n."\n    )\n    valeur = "caf\u00e9 # valeur"  # Comment.\n'
    expected = 'class Exemple:\n    valeur = "caf\u00e9 # valeur"\n'
    assert python_code_bytes_v1(source.encode()) == python_code_bytes_v1(expected.encode())
    source = 'def f(): "Descripci\u00f3n."; return "caf\u00e9" # Comment.\n'
    assert python_code_bytes_v1(source.encode()).endswith('def f(): ; return "caf\u00e9"\n'.encode())


def test_code_serialization_preserves_whitespace_inside_executable_strings():
    source = b'value = """first  \n\nlast"""\n'
    assert python_code_bytes_v1(source) != python_code_bytes_v1(source.replace(b"first  ", b"first"))
    assert python_code_bytes_v1(source) != python_code_bytes_v1(source.replace(b"\n\n", b"\n"))
    assert python_code_bytes_v1(b"\nvalue = 1 \t\n\n") == python_code_bytes_v1(b"value = 1\n")


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\x85", "\x0b", "\x0c"])
def test_code_serialization_preserves_non_newline_separators_in_strings(separator):
    source = f'value = """first{separator}last"""\n'.encode()
    assert python_code_bytes_v1(source) != python_code_bytes_v1(b'value = """first\nlast"""\n')
    documented = f'"""Description{separator}continued."""\nvalue = 1\n'.encode()
    assert python_code_bytes_v1(documented) == python_code_bytes_v1(b"value = 1\n")


def test_code_serialization_normalizes_python_source_newlines():
    source = b'"""Description.\nContinued."""\nvalue = """first\nlast""" # Comment.\n'
    assert python_code_bytes_v1(source) == python_code_bytes_v1(source.replace(b"\n", b"\r\n"))
    assert python_code_bytes_v1(source) == python_code_bytes_v1(source.replace(b"\n", b"\r"))


@pytest.mark.parametrize("prefix", [b"", b"f"])
def test_code_serialization_ignores_comments_between_concatenated_literals(prefix):
    source = b"value = (\n    " + prefix + b'"first"\n    "last"\n)\n'
    commented = source.replace(b'    "last"', b'# Comment.\n\n    "last"')
    assert python_code_bytes_v1(source) == python_code_bytes_v1(commented)


def test_verifier_identity_ignores_documentation_and_resources_but_binds_code(tmp_path, monkeypatch):
    from trialagentbench_verifier import scorer

    module = tmp_path / "scorer.py"
    module.write_text('"""Description."""\nVALUE = 1\n', encoding="utf-8")
    resources = tmp_path / "resources"
    resources.mkdir()
    example = resources / "example.py"
    example.write_text("EXAMPLE = 1\n", encoding="utf-8")
    monkeypatch.setattr(scorer, "__file__", str(module))
    current_scorer.cache_clear()
    try:
        original = current_scorer()
        module.write_text('# Comment.\n"""Revised description."""\n\nVALUE = 1 \t\n', encoding="utf-8")
        example.write_text("EXAMPLE = 2\n", encoding="utf-8")
        current_scorer.cache_clear()
        assert current_scorer() == original
        module.write_text("VALUE = 2\n", encoding="utf-8")
        current_scorer.cache_clear()
        assert current_scorer() != original
    finally:
        current_scorer.cache_clear()


def _ranking() -> TrialDevRunRankingV1:
    return TrialDevRunRankingV1(
        scorer=current_scorer(),
        release_id="release",
        release_manifest_sha256="a" * 64,
        invocation_checksum="b" * 64,
        model_id="model",
        scope="scheduled_panel",
        items=(),
        profile_means=(),
    )


@pytest.mark.parametrize("field,value", [("version", "earlier-scorer"), ("sources_sha256", "0" * 64)])
def test_reading_and_pooling_refuse_changed_scorer(field, value):
    ranking = _ranking()
    earlier = ScorerIdentityV1.model_validate(current_scorer().model_dump() | {field: value})
    changed = ranking.model_copy(update={"scorer": earlier})
    with pytest.raises(ValidationError, match="regrade retained outputs"):
        TrialDevRunRankingV1.model_validate_json(changed.model_dump_json())
    with pytest.raises(ValueError, match="different scorer identities"):
        require_same_scorer([ranking.scorer, changed.scorer])
    for operation in (compare_trialdev_rankings, trialdev_leaderboard_rows):
        with pytest.raises(ValueError, match="different scorer identities"):
            operation((ranking, changed), required_strata=(("clinical", "scenario", 3),), policy=CampaignAnalysisV1())


def test_missing_identity_is_refused_instead_of_assuming_current():
    payload = _ranking().model_dump(mode="json", exclude={"scorer"})
    with pytest.raises(ValidationError, match="scorer"):
        TrialDevRunRankingV1.model_validate(payload)


@pytest.mark.parametrize("operation", ["public_trialeval_campaign", "trialeval_leaderboard_rows"])
def test_trialeval_public_export_refuses_a_copied_stale_report(operation):
    from types import SimpleNamespace

    from trialagentbench_verifier import public_results

    report = SimpleNamespace(
        scorer=current_scorer().model_copy(update={"version": "earlier-scorer"}),
        between_model_contrasts=(),
    )
    kwargs = {"split": "public"} if operation == "public_trialeval_campaign" else {}
    with pytest.raises(ValueError, match="regrade retained outputs"):
        getattr(public_results, operation)(report, **kwargs)


def test_leaderboard_export_refuses_rows_from_different_scorers():
    from trialagentbench_verifier.campaign_contracts import LeaderboardRowV1
    from trialagentbench_verifier.public_results import CampaignLeaderboardV1

    row = LeaderboardRowV1(
        scorer=current_scorer(),
        benchmark="trialeval",
        track="point",
        model_id="model-a",
        metric="mean_loss",
        direction="lower_is_better",
        estimate=None,
        scheduled_tasks=1,
        scored_tasks=0,
        independent_worlds=1,
    )
    earlier = row.model_copy(update={"scorer": current_scorer().model_copy(update={"sources_sha256": "0" * 64})})
    with pytest.raises(ValueError, match="different scorer identities"):
        CampaignLeaderboardV1(
            scorer=current_scorer(),
            campaign_plan_sha256="a" * 64,
            release_set_manifest_sha256="b" * 64,
            split="public",
            confidence_level=0.95,
            bootstrap_resamples=100,
            bootstrap_seed=0,
            rows=(row, earlier),
        )
