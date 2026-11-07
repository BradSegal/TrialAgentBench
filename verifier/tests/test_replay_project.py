"""Replay local source modules while excluding previously computed answers."""

import json
import subprocess
import sys

import pytest

from trialagentbench_verifier.executor import fresh_replay_workspace


def test_python_project_recomputes_from_evidence_without_scratch_outputs(tmp_path):
    source = tmp_path / "source"
    scratch = source / "scratch"
    (scratch / "methods").mkdir(parents=True)
    (source / "data").mkdir()
    (source / "data" / "values.json").write_text("[1, 2, 6]")
    (scratch / "analysis_result.json").write_text('{"estimate": 999}')
    (scratch / "cached.csv").write_text("estimate\n999\n")
    (scratch / "methods" / "effects.py").write_text("def mean(values):\n    return sum(values) / len(values)\n")
    (scratch / "analysis.py").write_text(
        "import json\nfrom pathlib import Path\nfrom methods.effects import mean\n"
        'assert not Path("scratch/analysis_result.json").exists()\n'
        'assert not Path("scratch/cached.csv").exists()\n'
        'print(json.dumps({"estimate": mean(json.loads(Path("data/values.json").read_text()))}))\n'
    )
    replay = fresh_replay_workspace(source=source, program=scratch / "analysis.py", destination=tmp_path / "replay")
    result = subprocess.run(
        [sys.executable, str(replay / "scratch" / "analysis.py")],
        cwd=replay,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == {"estimate": 3}
    assert (scratch / "analysis_result.json").read_text() == '{"estimate": 999}'


def test_r_sources_keep_relative_layout_and_exclude_generated_data(tmp_path):
    source = tmp_path / "source"
    scratch = source / "scratch"
    (scratch / "methods").mkdir(parents=True)
    (scratch / "analysis.R").write_text('source("scratch/methods/effect.R")\n')
    (scratch / "methods" / "effect.R").write_text("estimate <- function(x) mean(x)\n")
    (scratch / "methods" / "computed.rds").write_bytes(b"previous result")
    replay = fresh_replay_workspace(source=source, program=scratch / "analysis.R", destination=tmp_path / "replay")
    assert (replay / "scratch" / "methods" / "effect.R").read_text() == "estimate <- function(x) mean(x)\n"
    assert not (replay / "scratch" / "methods" / "computed.rds").exists()


def test_cross_run_entrypoints_and_source_symlinks_fail_before_staging(tmp_path):
    source = tmp_path / "source"
    scratch = source / "scratch"
    scratch.mkdir(parents=True)
    program = scratch / "analysis.py"
    program.write_text("print(1)\n")
    foreign = tmp_path / "other" / "analysis.py"
    foreign.parent.mkdir()
    foreign.write_text("print(999)\n")
    with pytest.raises(ValueError, match="selected workspace"):
        fresh_replay_workspace(source=source, program=foreign, destination=tmp_path / "foreign-replay")
    assert not (tmp_path / "foreign-replay").exists()
    (scratch / "helper.py").symlink_to(foreign)
    with pytest.raises(ValueError, match="symbolic links"):
        fresh_replay_workspace(source=source, program=program, destination=tmp_path / "linked-replay")
    assert not (tmp_path / "linked-replay").exists()


def test_symlinked_scratch_cannot_import_another_projects_sources(tmp_path):
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "analysis.py").write_text("print(999)\n")
    source = tmp_path / "source"
    source.mkdir()
    (source / "scratch").symlink_to(foreign, target_is_directory=True)
    with pytest.raises(ValueError, match="regular Python or R"):
        fresh_replay_workspace(
            source=source, program=source / "scratch" / "analysis.py", destination=tmp_path / "replay"
        )
    assert not (tmp_path / "replay").exists()


@pytest.mark.parametrize("relative", ["scratch/estimates.csv", "analysis_sources/answer.csv", "../answer.csv"])
def test_submitted_or_external_answers_cannot_be_declared_as_replay_evidence(tmp_path, relative):
    from trialagentbench_verifier.execution_contracts import ReplayInputArtifactV1

    source = tmp_path / "source"
    scratch = source / "scratch"
    scratch.mkdir(parents=True)
    (scratch / "analysis.py").write_text("print(1)\n")
    artifact = ReplayInputArtifactV1(relative_path=relative, sha256="0" * 64, byte_length=0)
    with pytest.raises(ValueError, match="canonical path outside generated"):
        fresh_replay_workspace(
            source=source,
            program=scratch / "analysis.py",
            destination=tmp_path / "replay",
            evidence_artifacts=(artifact,),
        )
