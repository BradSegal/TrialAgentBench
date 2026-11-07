"""CSV IO helpers for schema-bearing harness exports."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from itertools import chain
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from trialagentbench.io.atomic import atomic_text_writer


def write_csv_rows(path: Path, rows: Iterable[dict[str, object]], field_order: Sequence[str]) -> None:
    """Write mapping rows to CSV using an explicit deterministic field order."""

    with atomic_text_writer(path, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(field_order), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in field_order})


def write_csv_models(path: Path, rows: Iterable[BaseModel], *, field_order: list[str] | None = None) -> None:
    """Write a list of Pydantic models to a CSV file deterministically.

    Parameters
    ----------
    path:
        Output CSV path.
    rows:
        Rows to write. Each row must be a Pydantic model instance.
    field_order:
        Optional explicit column order. If omitted, uses the first row's
        model field order.
    """
    iterator = iter(rows)
    first = next(iterator, None)
    if first is None:
        with atomic_text_writer(path, newline="") as handle:
            if field_order:
                csv.DictWriter(handle, fieldnames=field_order).writeheader()
        return
    fields = field_order or list(first.__class__.model_fields)
    with atomic_text_writer(path, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in chain((first,), iterator):
            payload = r.model_dump(mode="json")
            out: dict[str, Any] = {}
            for k in fields:
                v = payload.get(k)
                if isinstance(v, (list, dict)):
                    out[k] = json.dumps(v, sort_keys=True)
                else:
                    out[k] = v
            writer.writerow(out)
