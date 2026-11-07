"""Load and verify portable evaluator scoring keys."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal
from zipfile import ZipFile

from pydantic import BaseModel, ConfigDict, Field

from trialagentbench.grading.models import ValidatedScoringKeyV1


class ScoringKeyManifestV1(BaseModel):
    """Checksum-bound identity for a portable scoring-key collection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: Literal["ariadne.scoring_key_manifest/v1"]
    release_id: str = Field(min_length=1)
    specification_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scoring_keys_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    item_ids: tuple[str, ...] = Field(min_length=1)


class ScoringKeyStoreV1:
    """Exact item-indexed scoring keys from one verified evaluator release."""

    def __init__(
        self,
        *,
        manifest: ScoringKeyManifestV1,
        keys: tuple[ValidatedScoringKeyV1, ...],
    ) -> None:
        key_ids = tuple(key.item_id for key in keys)
        if key_ids != manifest.item_ids:
            raise ValueError("scoring-key records do not match the manifest item order")
        if len(key_ids) != len(set(key_ids)):
            raise ValueError("scoring-key item IDs must be unique")
        if {key.release_id for key in keys} != {manifest.release_id}:
            raise ValueError("scoring keys and manifest must identify one release")
        self.manifest = manifest
        self._keys = {key.item_id: key for key in keys}

    @classmethod
    def from_release(
        cls,
        release_root: Path,
        *,
        expected_item_ids: tuple[str, ...],
    ) -> ScoringKeyStoreV1:
        """Load keys and require coverage of every requested item."""

        grader = Path(release_root) / "grader"
        keys_path = grader / "scoring_keys.jsonl"
        manifest_path = grader / "scoring_key_manifest.json"
        return cls._from_serialized(
            body=keys_path.read_bytes(),
            manifest_body=manifest_path.read_bytes(),
            expected_item_ids=expected_item_ids,
        )

    @classmethod
    def from_evaluator_zip(
        cls,
        evaluator_zip: Path,
        *,
        expected_item_ids: tuple[str, ...],
    ) -> ScoringKeyStoreV1:
        """Load keys from a role-separated evaluator archive."""

        with ZipFile(evaluator_zip) as archive:
            try:
                body = archive.read("grader/scoring_keys.jsonl")
                manifest_body = archive.read("grader/scoring_key_manifest.json")
            except KeyError as exc:
                raise FileNotFoundError(f"Missing portable scoring-key artifact: {exc}") from exc
        return cls._from_serialized(
            body=body,
            manifest_body=manifest_body,
            expected_item_ids=expected_item_ids,
        )

    @classmethod
    def _from_serialized(
        cls,
        *,
        body: bytes,
        manifest_body: bytes,
        expected_item_ids: tuple[str, ...],
    ) -> ScoringKeyStoreV1:
        """Validate serialized keys against their manifest and requested items."""

        manifest = ScoringKeyManifestV1.model_validate_json(manifest_body)
        observed_sha256 = hashlib.sha256(body).hexdigest()
        if observed_sha256 != manifest.scoring_keys_sha256:
            raise ValueError("scoring-key checksum does not match its manifest")
        keys = tuple(
            ValidatedScoringKeyV1.model_validate(json.loads(line))
            for line in body.decode("utf-8").splitlines()
            if line.strip()
        )
        if not keys:
            raise ValueError("scoring-key release cannot be empty")
        store = cls(manifest=manifest, keys=keys)
        if len(expected_item_ids) != len(set(expected_item_ids)):
            raise ValueError("requested scoring-key item IDs must be unique")
        if not set(expected_item_ids) <= set(store._keys):
            missing = sorted(set(expected_item_ids) - set(store._keys))
            raise ValueError(f"scoring-key coverage mismatch: missing={missing}")
        return store

    def for_item(self, item_id: str) -> ValidatedScoringKeyV1:
        """Return one key or fail if the item is outside the release."""

        try:
            return self._keys[item_id]
        except KeyError as exc:
            raise KeyError(f"scoring key not found for item {item_id!r}") from exc

    def all(self) -> tuple[ValidatedScoringKeyV1, ...]:
        """Return all keys in manifest order."""

        return tuple(self._keys[item_id] for item_id in self.manifest.item_ids)


__all__ = ["ScoringKeyManifestV1", "ScoringKeyStoreV1"]
