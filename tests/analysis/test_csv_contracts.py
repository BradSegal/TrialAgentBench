from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from trialagentbench.io.csv_contracts import (
    PublicationArtifactContractError,
    read_strict_csv_rows,
    write_contract_csv,
)


class _Row(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    value: int


def test_strict_csv_reader_rejects_row_width_drift(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("a,b\n1,2,3\n", encoding="utf-8")

    with pytest.raises(PublicationArtifactContractError, match="expected 2 fields"):
        read_strict_csv_rows(path, required_columns={"a", "b"})


def test_strict_csv_reader_rejects_duplicate_headers(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("a,a\n1,2\n", encoding="utf-8")

    with pytest.raises(PublicationArtifactContractError, match="duplicate CSV headers"):
        read_strict_csv_rows(path, required_columns={"a"})


def test_strict_csv_reader_rejects_missing_required_column(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(PublicationArtifactContractError, match="missing required CSV columns"):
        read_strict_csv_rows(path, required_columns={"a", "b", "c"}, allow_extra_columns=True)


def test_strict_csv_reader_rejects_unexpected_columns_by_default(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(PublicationArtifactContractError, match="unexpected CSV columns"):
        read_strict_csv_rows(path, required_columns={"a"})


def test_strict_csv_reader_allows_declared_extra_columns(tmp_path: Path) -> None:
    path = tmp_path / "good.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")

    assert read_strict_csv_rows(path, required_columns={"a"}, allow_extra_columns=True) == [{"a": "1", "b": "2"}]


def test_contract_writer_preserves_prior_artifact_when_a_row_is_invalid(tmp_path: Path) -> None:
    """Do not expose a partially replaced CSV when streaming validation fails."""

    path = tmp_path / "rows.csv"
    path.write_text("previous\n", encoding="utf-8")

    with pytest.raises(ValueError):
        write_contract_csv(path, (_Row(value=1), {"value": "invalid"}), model=_Row)

    assert path.read_text(encoding="utf-8") == "previous\n"
    assert not tuple(tmp_path.glob(".*.tmp"))
