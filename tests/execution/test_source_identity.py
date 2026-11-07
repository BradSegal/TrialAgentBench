"""Tests for the installed harness behavior identity."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from trialagentbench_verifier.scorer import python_code_bytes_v1

from trialagentbench.io import canonical_payload_sha256
from trialagentbench.util import source_identity
from trialagentbench.util.source_identity import harness_source_sha256_v1


def test_harness_source_identity_covers_every_installed_python_module() -> None:
    """Shared runtime moves must remain bound to persisted run identity."""

    source_root = Path(__file__).resolve().parents[2] / "trialagentbench"
    expected = canonical_payload_sha256(
        {
            path.relative_to(source_root)
            .as_posix(): hashlib.sha256(python_code_bytes_v1(path.read_bytes()))
            .hexdigest()
            for path in sorted(source_root.rglob("*.py"))
        }
    )

    assert harness_source_sha256_v1() == expected


def test_harness_identity_ignores_documentation_but_refuses_changed_code(tmp_path, monkeypatch) -> None:
    """Documentation edits preserve grading identity; executable edits do not."""
    root = tmp_path / "installed"
    (root / "util").mkdir(parents=True)
    module = root / "util" / "source_identity.py"
    module.write_text('"""Description."""\nVALUE = 1\n', encoding="utf-8")
    resources = root / "resources"
    resources.mkdir()
    example = resources / "example.py"
    example.write_text("EXAMPLE = 1\n", encoding="utf-8")
    monkeypatch.setattr(source_identity, "__file__", str(module))
    expected = source_identity.harness_scoring_source_sha256_v1()
    monkeypatch.setattr(source_identity, "SCORER_HARNESS_SOURCE_SHA256", expected)
    complete = source_identity.harness_source_sha256_v1()

    module.write_text('# Comment.\n"""Revised description."""\n\nVALUE = 1 \t\n', encoding="utf-8")
    source_identity.require_harness_scorer_v1()
    assert source_identity.harness_source_sha256_v1() == complete

    example.write_text("EXAMPLE = 2\n", encoding="utf-8")
    source_identity.require_harness_scorer_v1()
    assert source_identity.harness_source_sha256_v1() != complete

    module.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="install matching harness and verifier builds"):
        source_identity.require_harness_scorer_v1()


def test_harness_identity_preserves_symbolic_link_refusal(tmp_path, monkeypatch) -> None:
    """Included source links are refused, while excluded example links are ignored."""
    root = tmp_path / "installed"
    (root / "util").mkdir(parents=True)
    module = root / "util" / "source_identity.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(source_identity, "__file__", str(module))
    expected = source_identity.harness_scoring_source_sha256_v1()
    (root / "resources").mkdir()
    (root / "resources" / "example.py").symlink_to(module)
    assert source_identity.harness_scoring_source_sha256_v1() == expected
    with pytest.raises(ValueError, match="Refusing to hash a symbolic link"):
        source_identity.harness_source_sha256_v1()
    (root / "linked.py").symlink_to(module)
    with pytest.raises(ValueError, match="Refusing to hash a symbolic link"):
        source_identity.harness_scoring_source_sha256_v1()


def test_source_identity_matches_python311_and_python312() -> None:
    """Both supported interpreters compute the same complete installed identity."""
    interpreters = [
        os.environ.get(f"TRIALAGENTBENCH_IDENTITY_PYTHON{minor}")
        or (sys.executable if sys.version_info[:2] == (3, int(minor[1:])) else None)
        for minor in ("311", "312")
    ]
    if any(interpreter is None for interpreter in interpreters):
        pytest.skip("Set TRIALAGENTBENCH_IDENTITY_PYTHON311 and PYTHON312 to test both supported environments")
    script = """
import json, sys
from trialagentbench.util.source_identity import (
    harness_source_sha256_v1, harness_scoring_source_sha256_v1, require_harness_scorer_v1,
)
from trialagentbench_verifier.scorer import current_scorer
require_harness_scorer_v1()
print(json.dumps({"python": list(sys.version_info[:2]), "identity": [
    harness_source_sha256_v1(), harness_scoring_source_sha256_v1(), current_scorer().sources_sha256,
]}))
"""
    results = [
        json.loads(subprocess.run([interpreter, "-c", script], check=True, capture_output=True, text=True).stdout)
        for interpreter in interpreters
        if interpreter is not None
    ]
    assert [result["python"] for result in results] == [[3, 11], [3, 12]]
    assert results[0]["identity"] == results[1]["identity"]


@pytest.mark.parametrize(
    ("module_name", "model_name"),
    [
        ("actions", "TrialDevWaitArguments"),
        ("actions", "TrialDevReadoutArguments"),
        ("actions", "TrialDevAdaptArguments"),
        ("actions", "TrialDevStudyProposal"),
        ("actions", "TrialDevActiveArm"),
        ("actions", "TrialDevAdvanceArguments"),
        ("recruitment", "TrialDevRecruitmentSelection"),
        ("recruitment", "TrialDevRecruitmentPreference"),
        ("resources", "TrialDevLook"),
    ],
)
def test_agent_facing_model_descriptions_are_bound_as_code(module_name, model_name, tmp_path, monkeypatch) -> None:
    """Each model description is executable source; its internal docstring never changes the tools."""
    import ast
    import importlib

    from trialagentbench.trialdev.actions import trialdev_programme_tools, trialdev_terminal_tools

    module = importlib.import_module(f"trialagentbench.trialdev.{module_name}")
    model = getattr(module, model_name)
    before = json.dumps([trialdev_programme_tools(), trialdev_terminal_tools()])
    monkeypatch.setattr(model, "__doc__", "Revised internal validation documentation.")
    assert json.dumps([trialdev_programme_tools(), trialdev_terminal_tools()]) == before

    source = Path(module.__file__).read_text(encoding="utf-8")
    node = next(item for item in ast.parse(source).body if isinstance(item, ast.ClassDef) and item.name == model_name)
    descriptions = [
        value
        for item in ast.walk(node)
        if isinstance(item, ast.Dict)
        for key, value in zip(item.keys, item.values, strict=True)
        if isinstance(key, ast.Constant) and key.value == "description"
    ]
    assert len(descriptions) == 1
    description = descriptions[0]
    assert isinstance(description, ast.Constant) and isinstance(description.value, str)
    literal = ast.get_source_segment(source, description)
    assert literal is not None
    root = tmp_path / "installed"
    (root / "trialdev").mkdir(parents=True)
    target = root / "trialdev" / f"{module_name}.py"
    target.write_text(source, encoding="utf-8")
    monkeypatch.setattr(source_identity, "__file__", str(root / "util/source_identity.py"))
    original = harness_source_sha256_v1()
    monkeypatch.setattr(source_identity, "SCORER_HARNESS_SOURCE_SHA256", original)
    target.write_text(source.replace(literal, repr(description.value + " Changed instruction."), 1), encoding="utf-8")
    assert harness_source_sha256_v1() != original
    with pytest.raises(RuntimeError, match="install matching harness and verifier builds"):
        source_identity.require_harness_scorer_v1()


def test_callable_source_identity_ignores_documentation_and_preserves_refusal(tmp_path, monkeypatch) -> None:
    """Resume identity binds grading code while allowing internal documentation updates."""
    source = tmp_path / "outcome.py"
    monkeypatch.setattr(source_identity.inspect, "getsourcefile", lambda function: str(source))
    source.write_text('"""Grading documentation."""\nVALUE = 1\n', encoding="utf-8")
    original = source_identity.source_file_sha256_v1(harness_source_sha256_v1)
    source.write_text('# Revised comment.\n"""Revised documentation."""\nVALUE = 1 \t\n', encoding="utf-8")
    assert source_identity.source_file_sha256_v1(harness_source_sha256_v1) == original
    source.write_text("VALUE = 2\n", encoding="utf-8")
    assert source_identity.source_file_sha256_v1(harness_source_sha256_v1) != original
    link = tmp_path / "linked.py"
    link.symlink_to(source)
    monkeypatch.setattr(source_identity.inspect, "getsourcefile", lambda function: str(link))
    with pytest.raises(ValueError, match="Refusing to hash a symbolic link"):
        source_identity.source_file_sha256_v1(harness_source_sha256_v1)
    monkeypatch.setattr(source_identity.inspect, "getsourcefile", lambda function: None)
    with pytest.raises(RuntimeError, match="Cannot resolve source identity"):
        source_identity.source_file_sha256_v1(harness_source_sha256_v1)
