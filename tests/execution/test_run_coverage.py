"""Tests for shared prospective schedule tracking."""

from __future__ import annotations

from pathlib import Path

import pytest

from trialagentbench.util.run_coverage import (
    load_or_create_run_coverage_v1,
    reconcile_run_coverage_v1,
)


def test_run_coverage_is_created_before_work_and_resumes_exactly(tmp_path: Path) -> None:
    path = tmp_path / "coverage.json"
    created = load_or_create_run_coverage_v1(
        path,
        run_identity_sha256="a" * 64,
        unit_ids=("one", "two"),
        resume=False,
    )
    updated = reconcile_run_coverage_v1(path, created, completed_unit_ids=("one",))

    resumed = load_or_create_run_coverage_v1(
        path,
        run_identity_sha256="a" * 64,
        unit_ids=("one", "two"),
        resume=True,
    )

    assert resumed == updated
    assert resumed.completed_unit_ids == ("one",)


def test_run_coverage_rejects_resume_schedule_drift(tmp_path: Path) -> None:
    path = tmp_path / "coverage.json"
    load_or_create_run_coverage_v1(
        path,
        run_identity_sha256="a" * 64,
        unit_ids=("one", "two"),
        resume=False,
    )

    with pytest.raises(ValueError, match="immutable schedule"):
        load_or_create_run_coverage_v1(
            path,
            run_identity_sha256="a" * 64,
            unit_ids=("one", "three"),
            resume=True,
        )
