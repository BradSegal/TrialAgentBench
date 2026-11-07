"""Scientific input reads from release ZIPs without materializing their payloads."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import pytest

from trialagentbench_verifier.archive import opened_release
from trialagentbench_verifier.io import VerificationInputError, read_json_object, sha256_file


@pytest.mark.parametrize("expected", [None, "correct", "f" * 64])
def test_opened_release_checks_zip_digest(tmp_path: Path, expected: str | None) -> None:
    """The direct member reader must enforce the same archive trust anchor."""
    path = tmp_path / "release.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("release_manifest.json", "{}")
    digest = sha256_file(path) if expected == "correct" else expected
    if expected == "f" * 64:
        with pytest.raises(VerificationInputError, match="externally trusted SHA-256"):
            with opened_release(path, expected_sha256=digest):
                pytest.fail("untrusted archive members must not be exposed")
    else:
        with opened_release(path, expected_sha256=digest) as root:
            assert read_json_object(root / "release_manifest.json") == {}


def test_release_members_support_filtered_parquet_without_extraction(tmp_path: Path, monkeypatch) -> None:
    table = pd.DataFrame({"participant_id": [1, 2, 3], "outcome": [2.5, 4.0, 5.5]})
    path = tmp_path / "release.zip"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("release_manifest.json", '{"release_id":"fixture"}')
        archive.writestr("evaluation/worlds/world/pool/baseline.parquet", table.to_parquet(index=False))

    def refuse(*args, **kwargs):
        raise AssertionError("Grading must read ZIP members in place")

    monkeypatch.setattr(ZipFile, "extractall", refuse)
    with opened_release(path) as root:
        assert read_json_object(root / "release_manifest.json") == {"release_id": "fixture"}
        members = tuple(root.glob("evaluation/worlds/*/pool/*.parquet"))
        assert len(members) == 1
        assert members[0].stat().st_size > 0
        with members[0].open("rb") as stream:
            result = pd.read_parquet(stream, filters=[("participant_id", "in", [1, 3])])
        pd.testing.assert_frame_equal(result.reset_index(drop=True), table.iloc[[0, 2]].reset_index(drop=True))
        assert not (root / "missing.json").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["release.zip"]


def test_release_member_json_retains_duplicate_field_validation(tmp_path: Path) -> None:
    path = tmp_path / "release.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("release_manifest.json", '{"release_id":"a","release_id":"b"}')
    with opened_release(path) as root:
        with pytest.raises(VerificationInputError, match="repeats field"):
            read_json_object(root / "release_manifest.json")


def test_catalog_members_read_their_shards_without_materializing_views(
    tmp_path: Path, monkeypatch, asset_release_factory
) -> None:
    from trialagentbench_verifier import assets
    from trialagentbench_verifier.io import release_asset_scope, sha256_file

    source = tmp_path / "source.zip"
    expected = asset_release_factory(source)
    _, catalog_path = assets.pack_release_assets(
        source, expected_sha256=sha256_file(source), output=tmp_path / "assets"
    )

    def refuse(*args, **kwargs):
        raise AssertionError("Grading must not materialize the evaluator")

    monkeypatch.setattr(assets, "materialize_asset_selection", refuse)
    with opened_release(catalog_path, expected_sha256=sha256_file(catalog_path)) as root:
        observed = {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}
        assert observed == expected
        scope = release_asset_scope(root / "participant")
        assert scope is not None
        assert scope.catalog.release_manifest_sha256 == sha256_file(root / "release_manifest.json")


def test_public_catalog_verification_hashes_each_physical_payload_once(
    tmp_path: Path, monkeypatch, asset_release_factory
) -> None:
    from trialagentbench_verifier import assets
    from trialagentbench_verifier import io as verifier_io
    from trialagentbench_verifier.archive import ArchivePath

    source = tmp_path / "source.zip"
    asset_release_factory(source)
    _, catalog_path = assets.pack_release_assets(
        source, expected_sha256=verifier_io.sha256_file(source), output=tmp_path / "assets"
    )
    digest = verifier_io.sha256_file(catalog_path)
    counts: dict[str, int] = {}
    original = verifier_io.sha256_file

    def counted(path):
        if isinstance(path, ArchivePath):
            archive, member = path._location()
            key = f"{archive.filename}:{member}"
        else:
            key = str(path)
        counts[key] = counts.get(key, 0) + 1
        return original(path)

    monkeypatch.setattr(verifier_io, "sha256_file", counted)
    monkeypatch.setattr(assets, "sha256_file", counted)
    with opened_release(catalog_path, expected_sha256=digest, verify_files=True) as root:
        assert (root / "release_manifest.json").is_file()
    assert counts and set(counts.values()) == {1}
    assert any(":objects/" in key for key in counts)
