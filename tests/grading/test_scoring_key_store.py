"""Release-integrity tests for portable scoring-key loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from trialagentbench.grading.key_store import ScoringKeyStoreV1


def _key_payload(*, item_id: str = "item-1") -> dict[str, object]:
    return {
        "schema_id": "ariadne.scoring_key/v1",
        "release_id": "release-1",
        "item_id": item_id,
        "question_id": "question-1",
        "context_tier": "C1",
        "prespecified_analysis_method_id": "fixture_decision_method",
        "credit_eligible_routes": [
            {
                "route_id": "route-1",
                "signature": {
                    "analysis_population_id": "itt",
                    "estimand_id": "estimand",
                    "intercurrent_event_strategy_ids": ["treatment_policy"],
                    "treatment_id": "active",
                    "comparator_id": "control",
                    "endpoint_id": "endpoint",
                    "effect_scale": "decision",
                    "analysis_method_id": "fixture_decision_method",
                },
                "method": {
                    "analysis_method_id": "fixture_decision_method",
                    "estimator_family": "estimator",
                    "result_kind": "decision",
                    "uncertainty_method": "not_applicable",
                    "design_modifiers": [],
                },
                "favorable_direction": "neither",
                "required_identification_assumptions": ["randomization"],
                "target": {
                    "kind": "categorical",
                    "credit_eligible_codes": ["advance"],
                },
            }
        ],
    }


def _write_release(root: Path, payloads: list[dict[str, object]]) -> None:
    grader = root / "grader"
    grader.mkdir(parents=True)
    body = "".join(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n" for payload in payloads).encode()
    (grader / "scoring_keys.jsonl").write_bytes(body)
    (grader / "scoring_key_manifest.json").write_text(
        json.dumps(
            {
                "schema_id": "ariadne.scoring_key_manifest/v1",
                "release_id": "release-1",
                "specification_sha256": "a" * 64,
                "scoring_keys_sha256": hashlib.sha256(body).hexdigest(),
                "item_ids": [payload["item_id"] for payload in payloads],
            }
        ),
        encoding="utf-8",
    )


def test_store_loads_checksum_bound_exact_denominator(tmp_path: Path) -> None:
    _write_release(tmp_path, [_key_payload()])

    store = ScoringKeyStoreV1.from_release(
        tmp_path,
        expected_item_ids=("item-1",),
    )

    assert store.for_item("item-1").release_id == "release-1"


def test_store_rejects_checksum_drift(tmp_path: Path) -> None:
    _write_release(tmp_path, [_key_payload()])
    with (tmp_path / "grader" / "scoring_keys.jsonl").open("ab") as handle:
        handle.write(b"\n")

    with pytest.raises(ValueError, match="checksum"):
        ScoringKeyStoreV1.from_release(tmp_path, expected_item_ids=("item-1",))


def test_store_rejects_partial_denominator(tmp_path: Path) -> None:
    _write_release(tmp_path, [_key_payload()])

    with pytest.raises(ValueError, match="coverage mismatch"):
        ScoringKeyStoreV1.from_release(
            tmp_path,
            expected_item_ids=("item-1", "item-2"),
        )


def test_store_allows_requested_subset_of_release(tmp_path: Path) -> None:
    _write_release(tmp_path, [_key_payload(), _key_payload(item_id="item-2")])

    store = ScoringKeyStoreV1.from_release(
        tmp_path,
        expected_item_ids=("item-1",),
    )

    assert store.for_item("item-1").item_id == "item-1"


def test_store_loads_identical_contract_from_evaluator_zip(tmp_path: Path) -> None:
    release = tmp_path / "release"
    _write_release(release, [_key_payload()])
    evaluator_zip = tmp_path / "evaluator.zip"
    with ZipFile(evaluator_zip, "w") as archive:
        archive.write(release / "grader" / "scoring_keys.jsonl", "grader/scoring_keys.jsonl")
        archive.write(
            release / "grader" / "scoring_key_manifest.json",
            "grader/scoring_key_manifest.json",
        )

    store = ScoringKeyStoreV1.from_evaluator_zip(
        evaluator_zip,
        expected_item_ids=("item-1",),
    )

    assert store.for_item("item-1").release_id == "release-1"
