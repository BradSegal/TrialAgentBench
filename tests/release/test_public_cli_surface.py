from __future__ import annotations

import json
import tomllib
from collections.abc import Sequence
from pathlib import Path

import pytest

from trialagentbench import cli
from trialagentbench.tools.build import build_trace_analysis_bundle

PUBLIC_CONSOLE_COMMANDS = {"trialagentbench"}


def _harness_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "trialagentbench").is_dir():
            return parent
    raise RuntimeError("Could not locate TrialAgentBench harness root")


def test_source_pyproject_exposes_only_public_console_commands() -> None:
    """Installed source-package commands must stay aligned to the public harness surface."""

    pyproject = tomllib.loads((_harness_root() / "pyproject.toml").read_text(encoding="utf-8"))

    assert set(pyproject["project"]["scripts"]) == PUBLIC_CONSOLE_COMMANDS


def test_public_cli_exposes_public_workflows() -> None:
    """The installed CLI groups behavior by user workflow."""

    assert set(cli._COMMANDS) == {"run", "grade", "analyse", "verify", "export", "campaign", "demo"}
    assert set(cli._COMMANDS["demo"]) == {"run"}
    assert set(cli._COMMANDS["run"]) == {"trialeval", "trialdev"}
    assert cli._COMMANDS["run"]["trialdev"].handler is cli.run_trialdev
    assert set(cli._COMMANDS["grade"]) == {"trialeval", "trialdev"}
    assert cli._COMMANDS["grade"]["trialdev"].handler is cli.grade_trialdev
    assert set(cli._COMMANDS["verify"]) == {"submission"}
    assert set(cli._COMMANDS["export"]) == {"results"}
    assert set(cli._COMMANDS["analyse"]) == {"calibration", "trace"}
    assert "trialdev-worlds" not in cli._COMMANDS["analyse"]
    assert "trialdev-ablation" not in cli._COMMANDS["analyse"]
    assert "analysis-bundle" not in cli._COMMANDS["verify"]
    assert "analysis-claims" not in cli._COMMANDS["verify"]
    assert "analysis-source-data" not in cli._COMMANDS["export"]
    assert cli._COMMANDS["export"]["results"].handler is cli.export_results


def test_submission_validator_rejects_duplicate_csv_columns(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Match the live replay parser at the score-blind public boundary."""

    submission = tmp_path / "ambiguous.csv"
    submission.write_text(
        "target_id,estimate,estimate,lower,upper\neffect,1,999,0,2\n",
        encoding="utf-8",
    )

    status = cli.main(
        [
            "verify",
            "submission",
            "--submission",
            str(submission),
            "--target-id",
            "effect",
            "--output-format",
            "json",
        ]
    )

    assert status == 1
    assert json.loads(capsys.readouterr().out)["valid"] is False


def test_public_cli_forwards_leaf_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dispatcher must preserve leaf arguments without interpreting them."""

    captured: dict[str, object] = {}

    def fake_run(argv: Sequence[str] | None) -> int:
        captured["argv"] = argv
        return 7

    monkeypatch.setitem(
        cli._COMMANDS["run"],
        "trialdev",
        cli._Command(fake_run, "Run TrialDev."),
    )
    assert cli.main(["run", "trialdev", "--release", "participant.zip", "--max-turns-per-step", "90"]) == 7
    assert captured == {"argv": ["--release", "participant.zip", "--max-turns-per-step", "90"]}


def test_provider_failure_exits_without_the_wrapped_sdk_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A provider failure stops the run without printing the SDK error that may echo credentials."""
    from trialagentbench.ports.llm_provider import ProviderRequestError, RetryTelemetry

    def failing_run(argv: Sequence[str] | None) -> int:
        try:
            raise OSError("401 Unauthorized: Authorization: Bearer dummy-test-key-123")
        except OSError as sdk_error:
            raise ProviderRequestError(
                "Provider request failed after retries.",
                retry_telemetry=RetryTelemetry(request_attempts=3, transient_failure_count=3, backoff_seconds=1.0),
            ) from sdk_error

    monkeypatch.setitem(cli._COMMANDS["run"], "trialeval", cli._Command(failing_run, "Run TrialEval."))
    assert cli.main(["run", "trialeval"]) == 2
    captured = capsys.readouterr()
    assert "Provider request failed after retries." in captured.err
    assert "dummy-test-key-123" not in captured.out + captured.err


def test_public_cli_does_not_mask_missing_required_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing internal runtime modules remain packaging failures."""

    def missing_internal(_: Sequence[str] | None) -> int:
        raise ModuleNotFoundError("No module named 'broken_internal'", name="broken_internal")

    monkeypatch.setitem(
        cli._COMMANDS["verify"],
        "submission",
        cli._Command(missing_internal, "Validate submission."),
    )

    with pytest.raises(ModuleNotFoundError, match="broken_internal"):
        cli.main(["verify", "submission", "--help"])


def test_trace_analysis_cli_maps_to_bundle_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    """The public trace-analysis CLI must execute the shared bundle builder."""

    captured: dict[str, object] = {}

    def fake_builder(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(build_trace_analysis_bundle, "build_trace_analysis_bundle", fake_builder)

    result = build_trace_analysis_bundle.main(
        [
            "--out-dir",
            "trace_bundle",
            "--trialeval-root",
            "runs/trialeval",
            "--trialdev-root",
            "runs/trialdev",
            "--trialdev-release-root",
            "release/trialdev",
        ]
    )

    assert result == 0
    assert captured == {
        "out_dir": Path("trace_bundle"),
        "trialeval_root": Path("runs/trialeval"),
        "trialdev_root": Path("runs/trialdev"),
        "trialdev_release_root": Path("release/trialdev"),
    }
