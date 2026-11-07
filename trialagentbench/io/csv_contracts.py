"""Strict CSV contract helpers for publication artifacts."""

from __future__ import annotations

import csv
import json
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from trialagentbench.io.atomic import atomic_text_writer


class PublicationArtifactContractError(ValueError):
    """Raised when a publication artifact violates its tabular contract."""


TModel = TypeVar("TModel", bound=BaseModel)
CSV_FIELD_SIZE_LIMIT = min(sys.maxsize, 1024 * 1024 * 1024)


def _csv_value(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    if value is None:
        return ""
    return value


def _model_input(row: dict[str, str]) -> dict[str, object]:
    model_inputs: dict[str, object] = {}
    for key, value in row.items():
        if value == "":
            model_inputs[key] = None
        elif value in {"True", "true"}:
            model_inputs[key] = True
        elif value in {"False", "false"}:
            model_inputs[key] = False
        elif value.startswith(("[", "{")):
            try:
                model_inputs[key] = json.loads(value)
            except json.JSONDecodeError:
                model_inputs[key] = value
        else:
            model_inputs[key] = value
    return model_inputs


def write_contract_csv(path: Path, rows: Iterable[TModel | dict[str, object]], *, model: type[TModel]) -> None:
    """Validate and write Pydantic rows as a stable CSV.

    Parameters
    ----------
    path
        Destination CSV file.
    rows
        Schema-bearing row models.
    model
        Row model class used to revalidate rows and define field order.

    Raises
    ------
    pydantic.ValidationError
        If any row does not validate against ``model``.
    """

    fields = list(model.model_fields)
    with atomic_text_writer(path, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if isinstance(row, BaseModel):
                raw_payload = row.model_dump(mode="json")
            else:
                raw_payload = dict(row)
            validated = model.model_validate(
                {key: (None if value == "" else value) for key, value in raw_payload.items()}
            )
            payload = validated.model_dump(mode="json")
            writer.writerow({field: _csv_value(payload.get(field)) for field in fields})


def iter_contract_csv(path: Path, *, model: type[TModel]) -> Iterator[TModel]:  # noqa: UP047
    """Iterate over validated rows from a Pydantic-backed CSV.

    Parameters
    ----------
    path
        CSV artifact to read.
    model
        Row model defining the exact header and value contract.

    Yields
    ------
    TModel
        One validated row at a time without retaining the complete table.
    """

    for row in _iter_strict_csv_rows(
        Path(path),
        required_columns=set(model.model_fields),
        allow_extra_columns=False,
    ):
        yield model.model_validate(_model_input(row))


def read_strict_csv_rows(
    path: Path,
    *,
    required_columns: set[str],
    allow_extra_columns: bool = False,
) -> list[dict[str, str]]:
    """Read CSV rows while enforcing header and row-width integrity.

    Parameters
    ----------
    path:
        CSV file to read.
    required_columns:
        Columns that must be present in the header.
    allow_extra_columns:
        Whether columns outside ``required_columns`` are allowed.

    Returns
    -------
    list[dict[str, str]]
        Rows keyed by header field.

    Raises
    ------
    PublicationArtifactContractError
        If the CSV has duplicate headers, missing required columns, unexpected
        columns when disallowed, or any row whose width differs from the header.

    Examples
    --------
    >>> from pathlib import Path
    >>> rows = read_strict_csv_rows(Path("table.csv"), required_columns={"id"})
    >>> isinstance(rows, list)
    True
    """
    return list(
        _iter_strict_csv_rows(
            path,
            required_columns=required_columns,
            allow_extra_columns=allow_extra_columns,
        )
    )


def _iter_strict_csv_rows(
    path: Path,
    *,
    required_columns: set[str],
    allow_extra_columns: bool,
) -> Iterator[dict[str, str]]:
    if not path.is_file():
        raise PublicationArtifactContractError(f"CSV artifact not found: {path}")

    csv.field_size_limit(CSV_FIELD_SIZE_LIMIT)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            if required_columns:
                raise PublicationArtifactContractError(f"CSV artifact is empty: {path}") from None
            return

        duplicate_headers = sorted({name for name in header if header.count(name) > 1})
        if duplicate_headers:
            raise PublicationArtifactContractError(f"{path}: duplicate CSV headers: {', '.join(duplicate_headers)}")

        header_set = set(header)
        missing = sorted(required_columns - header_set)
        if missing:
            raise PublicationArtifactContractError(f"{path}: missing required CSV columns: {', '.join(missing)}")

        extra = sorted(header_set - required_columns)
        if extra and not allow_extra_columns:
            raise PublicationArtifactContractError(f"{path}: unexpected CSV columns: {', '.join(extra)}")

        expected_width = len(header)
        for line_number, row in enumerate(reader, start=2):
            if len(row) != expected_width:
                raise PublicationArtifactContractError(
                    f"{path}:{line_number}: expected {expected_width} fields, observed {len(row)}"
                )
            yield dict(zip(header, row, strict=True))


__all__ = [
    "PublicationArtifactContractError",
    "iter_contract_csv",
    "read_strict_csv_rows",
    "write_contract_csv",
]
