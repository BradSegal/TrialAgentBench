"""Cache reuse must preserve byte authentication, isolation and recovery."""

import errno
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from trialagentbench_verifier import asset_cache
from trialagentbench_verifier.assets import pack_release_assets, read_asset_catalog
from trialagentbench_verifier.io import VerificationInputError, sha256_file


def _catalog(tmp_path: Path, asset_release_factory) -> Path:
    source = tmp_path / "original.zip"
    asset_release_factory(source)
    return pack_release_assets(source, expected_sha256=sha256_file(source), output=tmp_path / "assets")[1]


def test_warm_offline_view_reuses_inodes_without_acquisition(
    asset_release_factory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _catalog(tmp_path, asset_release_factory)
    digest = sha256_file(path)
    options = dict(expected_sha256=digest, cache=tmp_path / "cache", task_ids=("td-000000000001",))
    with asset_cache.cached_asset_selection(path, **options) as root:
        before = {p.relative_to(root): p.stat().st_ino for p in root.rglob("*") if p.is_file()}
        first = root

    def unexpected(*args, **kwargs):
        pytest.fail("warm view must not acquire or copy payloads")

    monkeypatch.setattr(asset_cache, "acquire_asset_selection", unexpected)
    with asset_cache.cached_asset_selection(path, **options) as root:
        assert root == first
        assert {p.relative_to(root): p.stat().st_ino for p in root.rglob("*") if p.is_file()} == before
        assert not any("worlds/tdw-000000000002/pool/" in str(p) for p in before)
        assert Path("grading_key/world_strata.json") in before


def test_cache_rejects_changed_bytes_and_detects_mutation_during_use(asset_release_factory, tmp_path: Path) -> None:
    path = _catalog(tmp_path, asset_release_factory)
    options = dict(expected_sha256=sha256_file(path), cache=tmp_path / "cache")
    with asset_cache.cached_asset_selection(path, **options) as root:
        changed = root / "release_manifest.json"
        changed.chmod(0o644)
        changed.write_bytes(b"corrupt")
    with pytest.raises(VerificationInputError, match="original digest"):
        with asset_cache.cached_asset_selection(path, **options):
            pytest.fail("corrupt cache cannot be borrowed")
    assert changed.read_bytes() == b"corrupt"


def test_concurrent_cold_requests_publish_one_view_and_allow_simultaneous_readers(
    asset_release_factory, tmp_path: Path
) -> None:
    path = _catalog(tmp_path, asset_release_factory)
    digest = sha256_file(path)
    barrier = Barrier(2)

    def borrow(_: int) -> Path:
        with asset_cache.cached_asset_selection(path, expected_sha256=digest, cache=tmp_path / "cache") as root:
            barrier.wait(timeout=5)
            return root

    with ThreadPoolExecutor(max_workers=2) as workers:
        roots = list(workers.map(borrow, range(2)))
    assert roots[0] == roots[1]


def test_failed_acquisition_preserves_completed_shards_and_retries_only_missing(
    asset_release_factory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _catalog(tmp_path, asset_release_factory)
    digest = sha256_file(path)
    catalog = read_asset_catalog(path, expected_sha256=digest)
    remote = tmp_path / "remote.assets.json"
    shutil.copyfile(path, remote)
    requests = []
    fail = True

    def download(shard, destination, base_url):
        requests.append(shard.sha256)
        if fail and len(requests) == 2:
            destination.write_bytes(b"partial")
            raise TimeoutError("injected transfer interruption")
        shutil.copyfile(path.parent / shard.filename, destination)

    monkeypatch.setattr(asset_cache, "_download", download)
    options = dict(
        expected_sha256=digest,
        cache=tmp_path / "cache",
        task_ids=("td-000000000001",),
        base_url="https://example.org/release",
    )
    with pytest.raises(TimeoutError, match="interruption"):
        asset_cache.acquire_asset_selection(remote, **options)
    first = requests[0]
    fail = False
    cached = asset_cache.acquire_asset_selection(remote, **options)
    assert requests.count(first) == 1
    expected = {shard.sha256 for shard in catalog.selected_shards(options["task_ids"])}
    assert {p.stem for p in (cached.parent / "shards").glob("*.zip")} == expected
    assert not list((cached.parent / "shards").glob(".acquire-*"))


def test_cache_refuses_symlinks_and_corrupt_completed_shards(asset_release_factory, tmp_path: Path) -> None:
    path = _catalog(tmp_path, asset_release_factory)
    digest = sha256_file(path)
    cache = tmp_path / "cache"
    cached = asset_cache.acquire_asset_selection(path, expected_sha256=digest, cache=cache)
    catalog = read_asset_catalog(cached, expected_sha256=digest)
    shard = cached.parent / catalog.shards[0].filename
    shard.chmod(0o644)
    shard.write_bytes(b"bad")
    with pytest.raises(VerificationInputError):
        asset_cache.acquire_asset_selection(path, expected_sha256=digest, cache=cache)
    assert shard.read_bytes() == b"bad"
    linked = tmp_path / "linked"
    linked.symlink_to(cache, target_is_directory=True)
    with pytest.raises(VerificationInputError, match="symbolic"):
        asset_cache.acquire_asset_selection(path, expected_sha256=digest, cache=linked)


def test_different_views_share_original_objects_and_link_fallback_preserves_bytes(
    asset_release_factory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trialagentbench_verifier import assets

    path = _catalog(tmp_path, asset_release_factory)
    digest = sha256_file(path)
    cache = tmp_path / "cache"
    with asset_cache.cached_asset_selection(
        path, expected_sha256=digest, cache=cache, task_ids=("td-000000000001",)
    ) as first:
        shared = first / "release_manifest.json"
        with asset_cache.cached_asset_selection(path, expected_sha256=digest, cache=cache) as second:
            assert first != second
            assert shared.stat().st_ino == (second / "release_manifest.json").stat().st_ino

    def unsupported(*args, **kwargs):
        raise OSError(errno.EOPNOTSUPP, "injected filesystem without hard links")

    monkeypatch.setattr(assets.os, "link", unsupported)
    with asset_cache.cached_asset_selection(path, expected_sha256=digest, cache=tmp_path / "copy-cache") as copied:
        assert (copied / "release_manifest.json").read_bytes() == shared.read_bytes()


def test_disk_failure_does_not_publish_partial_shard(
    asset_release_factory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _catalog(tmp_path, asset_release_factory)

    def full(*args, **kwargs):
        raise OSError(errno.ENOSPC, "injected full filesystem")

    monkeypatch.setattr(asset_cache.shutil, "copyfile", full)
    cache = tmp_path / "cache"
    with pytest.raises(OSError, match="full filesystem"):
        asset_cache.acquire_asset_selection(path, expected_sha256=sha256_file(path), cache=cache)
    assert not list((cache / "transport" / "shards").glob("*.zip"))


def test_network_retries_are_bounded_and_oversize_payloads_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io
    from urllib.error import URLError

    from trialagentbench_verifier.assets import AssetShard

    calls = []

    def unavailable(*args, **kwargs):
        calls.append(args)
        raise URLError("injected unavailable origin")

    monkeypatch.setattr(asset_cache, "urlopen", unavailable)
    monkeypatch.setattr(asset_cache.time, "sleep", lambda _: None)
    shard = AssetShard(group="release-shared", sha256="a" * 64, byte_length=1)
    with pytest.raises(URLError):
        asset_cache._download(shard, tmp_path / "partial", "https://example.org/release")
    assert len(calls) == 3

    class Response(io.BytesIO):
        def geturl(self):
            return "https://example.org/shard"

    monkeypatch.setattr(asset_cache, "urlopen", lambda *args, **kwargs: Response(b"too large"))
    with pytest.raises(VerificationInputError, match="byte length"):
        asset_cache._download(shard, tmp_path / "partial", "https://example.org/release")
    with pytest.raises(ValueError, match="HTTPS"):
        asset_cache._download(shard, tmp_path / "partial", "http://example.org/release")


def test_storage_preflight_and_action_cache_share_one_materialization(
    asset_release_factory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trialagentbench_verifier.io import materialized_release, release_archive, release_asset_scope

    path = _catalog(tmp_path, asset_release_factory)
    digest = sha256_file(path)
    cache = tmp_path / "cache"
    tasks = ("td-000000000001",)
    cold = asset_cache.cache_footprint(path, expected_sha256=digest, cache=cache, task_ids=tasks, workers=2)
    assert not cache.exists() and cold.payload_bytes_to_acquire > 0 and not cold.view_reusable
    # A cache named through ".." still yields resolved views, so members resolved inside a
    # view remain under its root (the participant isolation check compares both).
    (tmp_path / "work").mkdir()
    monkeypatch.setenv("TRIALAGENTBENCH_CACHE_DIR", str(tmp_path / "work" / ".." / "cache"))
    calls = []
    original = asset_cache.materialize_asset_selection

    def materialize(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(asset_cache, "materialize_asset_selection", materialize)
    roots = []
    for _ in range(3):
        with release_archive(path, expected_sha256=digest) as catalog:
            with materialized_release(catalog, task_ids=tasks) as root:
                assert release_asset_scope(root).selected_task_ids == frozenset(tasks)
                roots.append(root)
        assert release_asset_scope(root) is None
    assert len(calls) == 1 and len(set(roots)) == 1
    assert roots[0] == roots[0].resolve()
    warm = asset_cache.cache_footprint(path, expected_sha256=digest, cache=cache, task_ids=tasks, workers=2)
    assert warm.view_reusable and warm.payload_bytes_to_acquire == warm.additional_cache_bytes_bound == 0
    assert warm.retained_outputs_budget_bytes is None
    actual = sum(p.stat().st_blocks * 512 for p in cache.rglob("*") if p.is_file())
    assert actual < cold.additional_cache_bytes_bound


def test_busy_lock_is_bounded_and_restored_mtime_cannot_hide_tampering(asset_release_factory, tmp_path: Path) -> None:
    lock = tmp_path / "busy.lock"
    with asset_cache._lock(lock):
        with pytest.raises(TimeoutError, match="busy"):
            with asset_cache._lock(lock, timeout=0):
                pytest.fail("exclusive lock was bypassed")


def test_valid_view_cannot_hide_a_corrupt_colocated_transport(asset_release_factory, tmp_path: Path) -> None:
    path = _catalog(tmp_path, asset_release_factory)
    digest = sha256_file(path)
    cache = tmp_path / "cache"
    with asset_cache.cached_asset_selection(path, expected_sha256=digest, cache=cache):
        pass
    catalog = read_asset_catalog(path, expected_sha256=digest)
    (path.parent / catalog.shards[0].filename).write_bytes(b"corrupt distributed payload")
    with pytest.raises(VerificationInputError):
        with asset_cache.cached_asset_selection(path, expected_sha256=digest, cache=cache):
            pytest.fail("cached logical bytes hid a corrupt supplied shard")
