"""Release transport preserves logical bytes and selects only required payloads."""

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from trialagentbench_verifier.assets import TRAINING_CANARY, ReleaseAssets, pack_release_assets
from trialagentbench_verifier.io import VerificationInputError, sha256_file


def test_pack_preserves_original_bytes_and_deduplicates_world_payloads(asset_release_factory, tmp_path: Path) -> None:
    source = tmp_path / "original.zip"
    original = asset_release_factory(source)
    participant, evaluator = pack_release_assets(
        source, expected_sha256=sha256_file(source), output=tmp_path / "assets"
    )
    catalog = ReleaseAssets.model_validate_json(evaluator.read_bytes())
    assert {member.path for member in catalog.members} == set(original)
    shards = {shard.group: shard for shard in catalog.shards}
    for member in catalog.members:
        shard = shards[member.group]
        assert sha256_file(evaluator.parent / shard.filename) == shard.sha256
        with ZipFile(evaluator.parent / shard.filename) as archive:
            assert archive.read(f"objects/{member.sha256}") == original[member.path]
    world = next(shard for shard in catalog.shards if shard.group == "evaluation-tdw-000000000001")
    with ZipFile(evaluator.parent / world.filename) as archive:
        assert len(archive.namelist()) == 1
    public = ReleaseAssets.model_validate_json(participant.read_bytes())
    assert not any(shard.group.startswith("evaluation-") for shard in public.shards)
    assert all(not member.path.startswith("participant/") for member in public.members)
    assert catalog.release_manifest_sha256 == hashlib.sha256(original["release_manifest.json"]).hexdigest()
    assert source.is_file()


def test_every_catalog_and_shard_carries_the_training_canary(asset_release_factory, tmp_path: Path) -> None:
    source = tmp_path / "original.zip"
    asset_release_factory(source)
    participant, evaluator = pack_release_assets(
        source, expected_sha256=sha256_file(source), output=tmp_path / "assets"
    )
    for path in (participant, evaluator):
        assert TRAINING_CANARY in path.read_text(encoding="utf-8")
    shards = sorted((tmp_path / "assets" / "shards").glob("*.zip"))
    assert shards
    for shard in shards:
        # The ZIP comment is stored uncompressed, so the canary is visible to byte scans.
        assert TRAINING_CANARY.encode("ascii") in shard.read_bytes()
    payload = json.loads(evaluator.read_text(encoding="utf-8"))
    payload["canary"] = "removed"
    with pytest.raises(ValueError, match="canary"):
        ReleaseAssets.model_validate(payload)


def test_selection_fetches_shared_metadata_and_only_selected_world(asset_release_factory, tmp_path: Path) -> None:
    source = tmp_path / "original.zip"
    asset_release_factory(source)
    _, path = pack_release_assets(source, expected_sha256=sha256_file(source), output=tmp_path / "assets")
    catalog = ReleaseAssets.model_validate_json(path.read_bytes())
    selected = catalog.selected_members(("td-000000000001",))
    assert any(member.path == "release_manifest.json" for member in selected)
    assert any("tdw-000000000001/pool" in member.path for member in selected)
    assert not any("worlds/tdw-000000000002/pool/" in member.path for member in selected)
    assert any(member.path == "grading_key/world_strata.json" for member in selected)
    assert [task.task_id for task in catalog.tasks if task.main] == ["td-000000000001"]
    for bad in [(), ("td-000000000003",), ("td-000000000001", "td-000000000001")]:
        with pytest.raises(ValueError):
            catalog.selected_members(bad)


def test_pack_rejects_corruption_and_portable_path_collisions(asset_release_factory, tmp_path: Path) -> None:
    source = tmp_path / "original.zip"
    asset_release_factory(source)
    with pytest.raises(VerificationInputError, match="trusted SHA"):
        pack_release_assets(source, expected_sha256="0" * 64, output=tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
    with ZipFile(source, "a") as archive:
        archive.writestr("../escaped", b"unsafe")
    with pytest.raises(VerificationInputError, match="non-portable"):
        pack_release_assets(source, expected_sha256=sha256_file(source), output=tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_catalog_rejects_cross_role_or_incomplete_groups(asset_release_factory, tmp_path: Path) -> None:
    source = tmp_path / "original.zip"
    asset_release_factory(source)
    _, path = pack_release_assets(source, expected_sha256=sha256_file(source), output=tmp_path / "assets")
    payload = json.loads(path.read_bytes())
    payload["audience"] = "participant"
    with pytest.raises(ValueError, match="evaluator boundary"):
        ReleaseAssets.model_validate(payload)
    payload["audience"] = "evaluator"
    payload["shards"].pop()
    with pytest.raises(ValueError, match="logical inventory"):
        ReleaseAssets.model_validate(payload)


def test_materialization_reconstructs_identical_files_and_shares_duplicate_storage(
    asset_release_factory, tmp_path: Path
) -> None:
    from trialagentbench_verifier.assets import asset_footprint, materialize_asset_selection

    source = tmp_path / "original.zip"
    original = asset_release_factory(source)
    _, path = pack_release_assets(source, expected_sha256=sha256_file(source), output=tmp_path / "assets")
    output = tmp_path / "view"
    task_ids = ("td-000000000001",)
    catalog = materialize_asset_selection(path, expected_sha256=sha256_file(path), output=output, task_ids=task_ids)
    expected = {member.path for member in catalog.selected_members(task_ids)}
    assert {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} == expected
    assert all((output / name).read_bytes() == original[name] for name in expected)
    duplicates = [output / name for name in expected if name.endswith("completion.parquet")]
    assert len(duplicates) == 2 and duplicates[0].stat().st_ino == duplicates[1].stat().st_ino
    footprint = asset_footprint(catalog, task_ids)
    assert footprint.materialized_unique_bytes < footprint.materialized_logical_bytes
    assert footprint.download_bytes < asset_footprint(catalog).download_bytes


def test_materialization_refuses_corrupt_shard_without_publishing_partial_view(
    asset_release_factory, tmp_path: Path
) -> None:
    from trialagentbench_verifier.assets import materialize_asset_selection

    source = tmp_path / "original.zip"
    asset_release_factory(source)
    _, path = pack_release_assets(source, expected_sha256=sha256_file(source), output=tmp_path / "assets")
    catalog = ReleaseAssets.model_validate_json(path.read_bytes())
    shard_path = path.parent / catalog.shards[0].filename
    body = bytearray(shard_path.read_bytes())
    body[len(body) // 2] ^= 1
    shard_path.write_bytes(body)
    with pytest.raises(VerificationInputError, match="trusted SHA"):
        materialize_asset_selection(path, expected_sha256=sha256_file(path), output=tmp_path / "view")
    assert not (tmp_path / "view").exists()


def test_repack_reuses_only_authenticated_groups_from_identical_source(asset_release_factory, tmp_path: Path) -> None:
    source = tmp_path / "original.zip"
    members = asset_release_factory(source)
    for role in ("participant", "evaluation"):
        members[f"{role}/manifest.json"] = json.dumps(
            {
                "role": role,
                "artifacts": [
                    {"relative_path": name, "sha256": hashlib.sha256(body).hexdigest(), "byte_length": len(body)}
                    for name, body in sorted(members.items())
                    if name.startswith(f"{role}/") and name != f"{role}/manifest.json"
                ],
            }
        ).encode()
    with ZipFile(source, "w") as archive:
        for name, body in members.items():
            archive.writestr(name, body)
    _, original_catalog = pack_release_assets(source, expected_sha256=sha256_file(source), output=tmp_path / "first")
    _, repacked_catalog = pack_release_assets(
        source,
        expected_sha256=sha256_file(source),
        output=tmp_path / "second",
        reuse=(original_catalog, sha256_file(original_catalog)),
    )
    assert repacked_catalog.read_bytes() == original_catalog.read_bytes()
    catalog = ReleaseAssets.model_validate_json(original_catalog.read_bytes())
    for shard in catalog.shards:
        assert (repacked_catalog.parent / shard.filename).read_bytes() == (
            original_catalog.parent / shard.filename
        ).read_bytes()
    shard_path = original_catalog.parent / catalog.shards[0].filename
    shard_path.write_bytes(b"corrupt cached payload")
    with pytest.raises(VerificationInputError):
        pack_release_assets(
            source,
            expected_sha256=sha256_file(source),
            output=tmp_path / "third",
            reuse=(original_catalog, sha256_file(original_catalog)),
        )
    assert not (tmp_path / "third").exists()


def test_selected_scope_is_bound_to_original_inventory_and_drained_context(
    asset_release_factory, tmp_path: Path
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from trialagentbench_verifier.assets import ReleaseAssetScope, materialize_asset_selection
    from trialagentbench_verifier.io import registered_asset_scope, release_asset_scope

    source = tmp_path / "original.zip"
    asset_release_factory(source)
    _, path = pack_release_assets(source, expected_sha256=sha256_file(source), output=tmp_path / "assets")
    root = tmp_path / "view"
    tasks = ("td-000000000001",)
    catalog = materialize_asset_selection(path, expected_sha256=sha256_file(path), output=root, task_ids=tasks)
    scope = ReleaseAssetScope(catalog, tasks)
    assert release_asset_scope(root) is None

    omitted = next(m for m in catalog.members if "worlds/tdw-000000000002/" in m.path)
    assert not scope.requires(omitted.path, omitted.sha256, omitted.byte_length)
    with pytest.raises(VerificationInputError, match="original role inventory"):
        scope.requires(omitted.path, "0" * 64, omitted.byte_length)
    with registered_asset_scope(root, scope):
        with ThreadPoolExecutor(max_workers=2) as pool:
            observed = list(pool.map(release_asset_scope, [root, root / "participant"]))
        assert observed == [scope, scope]
        with registered_asset_scope(root, scope):
            assert release_asset_scope(root) == scope
        assert release_asset_scope(root) == scope
    assert release_asset_scope(root) is None


def test_release_role_lookup_rejects_ambiguous_or_cross_role_transports() -> None:
    from trialagentbench_verifier.release_set import ReleaseSetArtifactV1, role_filename

    names = ("trialdev-participant.assets.json", "trialdev-evaluator.assets.json")
    assert role_filename(names, "trialdev", "participant") == names[0]
    for inventory in ((), names + ("trialdev-participant.zip",)):
        with pytest.raises(ValueError, match="exactly one"):
            role_filename(inventory, "trialdev", "participant")
    with pytest.raises(ValueError, match="Unknown"):
        role_filename(names, "unknown", "participant")
    with pytest.raises(ValueError, match="audience role"):
        ReleaseSetArtifactV1(
            filename=names[0],
            sha256="a" * 64,
            byte_length=100,
            kind="controlled_role",
            audience="trialdev_evaluator",
            benchmark_stream="trialdev",
        )


@pytest.mark.parametrize("benchmark", ["trialeval", "trialdev"])
@pytest.mark.parametrize("role", ["participant", "evaluator"])
def test_withheld_archives_are_operator_only_never_public_roles(benchmark: str, role: str) -> None:
    from trialagentbench_verifier.release_set import ReleaseSetArtifactV1

    artifact = {
        "filename": f"{benchmark}-withheld-{role}.assets.json",
        "sha256": "a" * 64,
        "byte_length": 100,
        "audience": f"{benchmark}_withheld_{role}",
        "benchmark_stream": benchmark,
    }
    assert ReleaseSetArtifactV1(**artifact, kind="withheld_role").kind == "withheld_role"
    with pytest.raises(ValueError, match="audience role"):
        ReleaseSetArtifactV1(**artifact, kind="controlled_role")
