"""Internal bounded bridge from release Parquet to an R-readable CSV file."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict


class CSVSchema(BaseModel):
    """Scalar types and collision-free missing marker for the private R bridge."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    columns: tuple[str, ...]
    classes: tuple[Literal["character", "numeric", "logical"], ...]
    missing: str


def write_typed_csv(source: Path, destination: Path) -> None:
    """Preserve released scalar values instead of asking R to infer CSV types."""

    table = pq.read_table(source)
    if len(set(table.column_names)) != len(table.column_names):
        raise ValueError("R Parquet input must have unique column names.")
    frame = table.to_pandas()
    classes: list[Literal["character", "numeric", "logical"]] = []
    strings: set[str] = set()
    for field in table.schema:
        kind = field.type.value_type if pa.types.is_dictionary(field.type) else field.type
        values = frame[field.name]
        if pa.types.is_boolean(kind) or pa.types.is_null(kind):
            classes.append("logical")
            frame[field.name] = values.map({True: "TRUE", False: "FALSE"})
        elif pa.types.is_integer(kind) or pa.types.is_floating(kind):
            if pa.types.is_integer(kind) and any(
                abs(value) > 2**53 for value in table[field.name].drop_null().to_pylist()
            ):
                raise ValueError("R numeric conversion would lose integer precision.")
            classes.append("numeric")
        elif pa.types.is_string(kind) or pa.types.is_large_string(kind):
            classes.append("character")
            strings.update(values.dropna().astype(str))
        else:
            raise ValueError(f"Unsupported R Parquet scalar type for {field.name}: {kind}")
    missing = "__TRIALAGENTBENCH_MISSING__"
    while missing in strings:
        missing += "_"
    schema = CSVSchema(columns=tuple(table.column_names), classes=tuple(classes), missing=missing)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        stream.write(schema.model_dump_json() + "\n")
        frame.to_csv(stream, index=False, na_rep=missing, lineterminator="\n")


def main() -> None:
    """Convert one workspace Parquet file to a temporary CSV."""

    workspace = Path("/workspace").resolve(strict=True)
    source = Path(os.environ["TRIALAGENTBENCH_R_PARQUET_INPUT"]).resolve(strict=True)
    destination = Path(os.environ["TRIALAGENTBENCH_R_CSV_OUTPUT"]).resolve()
    if not source.is_relative_to(workspace) or source.suffix.casefold() != ".parquet":
        raise ValueError("R Parquet input must be a .parquet file inside /workspace.")
    if destination.parent != Path("/tmp") or destination.suffix.casefold() != ".csv":
        raise ValueError("R Parquet conversion output must be a .csv file directly under /tmp.")
    write_typed_csv(source, destination)


if __name__ == "__main__":
    main()
