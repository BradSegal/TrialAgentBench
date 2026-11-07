"""Released-byte protocol conformance vectors."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from trialagentbench_test_helpers import minimal_estimator_procedure

from trialagentbench.contracts.trialeval_methods import (
    TrialEvalParticipantMethodDictionaryV1,
    TrialEvalParticipantMethodV1,
)
from trialagentbench.io.checksums import canonical_json_bytes, canonical_payload_sha256


@pytest.mark.parametrize(
    ("payload", "encoded", "digest"),
    (
        (None, b"null", "74234e98afe7498fb5daf1f36ac2d78acc339464f950703b8c019892f982b90b"),
        (
            {"b": 2, "a": 1},
            b'{"a":1,"b":2}',
            "43258cff783fe7036d8a43033f830adfc60ec037382473548ac742b888292777",
        ),
        (
            {"label": "München", "items": [True, None, -2, 1.25]},
            b'{"items":[true,null,-2,1.25],"label":"M\\u00fcnchen"}',
            "7a3fe5f88f89e0d3660c1c7efb08199d3c6b2243fca07bba1011ad67dc077597",
        ),
    ),
)
def test_canonical_json_vectors(payload: object, encoded: bytes, digest: str) -> None:
    """Fix canonical bytes and digests across every release producer."""

    assert canonical_json_bytes(payload) == encoded  # type: ignore[arg-type]
    assert canonical_payload_sha256(payload) == digest  # type: ignore[arg-type]


def test_canonical_json_rejects_nonfinite_numbers() -> None:
    """Refuse values that JSON cannot represent portably."""

    with pytest.raises(ValueError, match="JSON compliant"):
        canonical_json_bytes(float("nan"))


def _method(method_id: str) -> TrialEvalParticipantMethodV1:
    return TrialEvalParticipantMethodV1(
        method_id=method_id,
        estimator_family="mean_difference",
        objective="Estimate the declared treatment-policy contrast.",
        result_kind="point_interval",
        effect_scale="mean_difference",
        uncertainty_method_id="welch_standard_error",
        procedure=minimal_estimator_procedure(),
        description="Difference in arm-specific endpoint means with a two-sided interval.",
    )


def test_construction_canonicalization_is_immutable_and_idempotent() -> None:
    """Construction-time sorting and checksums cannot enable later mutation or drift."""

    contract = TrialEvalParticipantMethodDictionaryV1(methods=(_method("z_method"), _method("a_method")))
    assert tuple(method.method_id for method in contract.methods) == ("a_method", "z_method")

    round_trip = TrialEvalParticipantMethodDictionaryV1.model_validate(contract.model_dump(mode="json"))
    assert round_trip == contract
    assert round_trip.checksum == contract.checksum
    with pytest.raises(ValidationError, match="frozen"):
        contract.checksum = "0" * 64
