"""Tests for bounded-memory public CSV output."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from trialagentbench.io.csv import write_csv_models, write_csv_rows


class _Row(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int
    label: str


def test_csv_writers_consume_single_pass_iterables(tmp_path: Path) -> None:
    model_path = tmp_path / "models.csv"
    row_path = tmp_path / "rows.csv"

    write_csv_models(model_path, (_Row(index=index, label=f"v{index}") for index in range(3)))
    write_csv_rows(
        row_path,
        ({"index": index, "label": f"v{index}"} for index in range(3)),
        ("index", "label"),
    )

    expected = "index,label\n0,v0\n1,v1\n2,v2\n"
    assert model_path.read_text(encoding="utf-8") == expected
    assert row_path.read_text(encoding="utf-8") == expected


def test_empty_csv_models_preserve_an_explicit_schema(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"

    write_csv_models(path, iter(()), field_order=["index", "label"])

    assert path.read_text(encoding="utf-8") == "index,label\n"
