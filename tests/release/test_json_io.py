"""Tests for strict and streaming public JSON input."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from trialagentbench.io import read_json, read_jsonl_models, write_json
from trialagentbench.io.exclusive_publication import atomic_text_writer


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    value: float


@pytest.mark.parametrize(
    "body",
    ('{"value": 1, "value": 2}', '{"value": NaN}'),
)
def test_read_json_rejects_nonportable_values(body: str, tmp_path: Path) -> None:
    """Reject duplicate keys and non-finite numbers at the shared input boundary."""

    path = tmp_path / "invalid.json"
    path.write_text(body, encoding="utf-8")

    with pytest.raises(ValueError):
        read_json(path)


def test_read_jsonl_models_is_lazy_and_reports_the_invalid_line(tmp_path: Path) -> None:
    """Validate JSONL one record at a time without reading past the consumer."""

    path = tmp_path / "records.jsonl"
    path.write_text('{"value": 1}\n{"value": 2, "value": 3}\n', encoding="utf-8")
    records = read_jsonl_models(_Record, path)

    assert next(records).value == 1.0
    with pytest.raises(ValueError, match=r"records\.jsonl:2"):
        next(records)


def test_write_json_rejects_nonfinite_numbers_without_replacing_the_target(tmp_path: Path) -> None:
    """Keep the previous artifact if canonical numerical serialization fails."""

    path = tmp_path / "result.json"
    path.write_text('{"value": 1}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="JSON compliant"):
        write_json(path, {"value": float("nan")})

    assert path.read_text(encoding="utf-8") == '{"value": 1}\n'


def test_killed_exclusive_writer_cannot_publish_a_partial_checkpoint(tmp_path: Path) -> None:
    """SIGKILL before publication leaves the canonical name free for a complete checkpoint."""
    path = tmp_path / "result.json"
    script = """
import sys
from pathlib import Path
from trialagentbench.io.exclusive_publication import atomic_text_writer
with atomic_text_writer(Path(sys.argv[1]), exclusive=True) as handle:
    handle.write('{"value": ')
    handle.flush()
    print('writing', flush=True)
    sys.stdin.read()
    handle.write('1}\\n')
"""
    with subprocess.Popen(
        [sys.executable, "-c", script, str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    ) as writer:
        try:
            assert writer.stdout is not None and writer.stdout.readline().strip() == "writing"
        finally:
            writer.kill()
            writer.communicate(timeout=10)
    assert not path.exists()
    with atomic_text_writer(path, exclusive=True) as handle:
        handle.write('{"value": 2}\n')
    with pytest.raises(FileExistsError):
        with atomic_text_writer(path, exclusive=True) as handle:
            handle.write('{"value": 3}\n')
    assert read_json(path) == {"value": 2}


@pytest.mark.parametrize("operation", ["append", "replace", "exclusive"])
def test_durable_writers_persist_new_campaign_directory_links_before_publication(tmp_path, monkeypatch, operation):
    from trialagentbench.io import exclusive_publication as atomic
    from trialagentbench.io.exclusive_publication import append_jsonl

    synchronized = []
    original = atomic._sync_directory

    def synchronize(directory):
        synchronized.append(directory)
        original(directory)

    monkeypatch.setattr(atomic, "_sync_directory", synchronize)
    target = tmp_path / "campaign" / "native" / "logs" / "record.jsonl"
    if operation == "append":
        append_jsonl(target, {"value": 1})
    else:
        with atomic_text_writer(target, exclusive=operation == "exclusive") as stream:
            stream.write('{"value":1}\n')
    assert synchronized[:3] == [tmp_path, tmp_path / "campaign", tmp_path / "campaign" / "native"]
    assert synchronized[-1] == target.parent
    assert read_json(target) == {"value": 1}
