"""Method-neutral numerical exports bind only the assigned scientific targets."""

import pytest

from trialagentbench_verifier.numerical import read_numerical_table


def test_numerical_export_accepts_reordered_columns_and_keeps_missing_targets(tmp_path):
    path = tmp_path / "estimates.csv"
    path.write_text("\ufeffupper,target_id,lower,estimate\n1,second,-1,2\n\n", encoding="utf-8")
    rows = read_numerical_table(path, target_ids=("first", "second"))
    assert len(rows) == 1
    assert rows[0].target_id == "second"
    assert rows[0].estimate == 2
    assert (rows[0].lower, rows[0].upper) == (-1, 1)


@pytest.mark.parametrize(
    "row",
    ["second,nan,0,1", "second,inf,0,1", "second,0,-inf,1", "second,0,0,NaN", "second,,0,1", "second,NA,NA,NA"],
)
def test_blank_na_or_nonfinite_row_loses_only_its_own_target(tmp_path, row):
    path = tmp_path / "estimates.csv"
    path.write_text("target_id,estimate,lower,upper\nfirst,1,0,2\n" + row + "\n")
    rows = read_numerical_table(path, target_ids=("first", "second"))
    assert [(row.target_id, row.estimate) for row in rows] == [("first", 1.0)]


def test_r_write_csv_with_na_keeps_reported_targets(tmp_path):
    path = tmp_path / "estimates.csv"
    path.write_text('"target_id","estimate","lower","upper"\n"first",1,0,2\n"second",NA,NA,NA\n')
    assert [row.target_id for row in read_numerical_table(path, target_ids=("first", "second"))] == ["first"]
    path.write_text("target_id,estimate,lower,upper\nsecond,NA,NA,NA\nsecond,1,0,2\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_numerical_table(path, target_ids=("first", "second"))


@pytest.mark.parametrize(
    "row",
    [
        "effect,0,2,1",
        "effect,True,0,1",
        "effect,0,0,1,extra",
        "unknown,0,0,1",
        "effect,0,0,1\neffect,1,0,2",
    ],
)
def test_invalid_numerical_rows_fail_without_method_adjudication(tmp_path, row):
    path = tmp_path / "estimates.csv"
    path.write_text("target_id,estimate,lower,upper\n" + row + "\n")
    with pytest.raises(ValueError):
        read_numerical_table(path, target_ids=("effect",))


def test_extra_metadata_and_duplicate_headers_are_rejected(tmp_path):
    path = tmp_path / "estimates.csv"
    for header in ("target_id,estimate,lower,lower", "target_id,estimate,lower,upper,method"):
        path.write_text(header + "\n")
        with pytest.raises(ValueError, match="columns"):
            read_numerical_table(path, target_ids=("effect",))


def test_empty_export_does_not_fabricate_unreported_targets(tmp_path):
    path = tmp_path / "estimates.csv"
    path.write_text("target_id,estimate,lower,upper\n")
    assert read_numerical_table(path, target_ids=("effect",)) == ()
    with pytest.raises(ValueError, match="assigned numerical targets"):
        read_numerical_table(path, target_ids=("effect", "effect"))
    link = tmp_path / "linked.csv"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="non-symlinked"):
        read_numerical_table(link, target_ids=("effect",))
