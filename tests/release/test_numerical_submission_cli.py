"""The public validation command accepts the same numerical rows as grading."""

import json

import pytest

from trialagentbench.tools.validate.validate_submission import main


def test_csv_validation_reports_missing_targets_without_loading_answers(tmp_path, capsys):
    path = tmp_path / "estimates.csv"
    path.write_text("target_id,estimate,lower,upper\neffect,9,1,2\nsafety,nan,0,2\n")
    assert (
        main(["--submission", str(path), "--target-id", "effect", "--target-id", "safety", "--output-format", "json"])
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["valid"] and result["missing_targets"] == ["safety"]


@pytest.mark.parametrize("row", ["other,1,0,2", "effect,1,2,0", "effect,1,0,2\neffect,2,0,3"])
def test_csv_validation_rejects_wrong_target_or_invalid_numbers(tmp_path, row, capsys):
    path = tmp_path / "estimates.csv"
    path.write_text("target_id,estimate,lower,upper\n" + row + "\n")
    assert main(["--submission", str(path), "--target-id", "effect", "--output-format", "json"]) == 1
    assert not json.loads(capsys.readouterr().out)["valid"]
