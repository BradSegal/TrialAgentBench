"""Verified selective acquisition and persistent release views on Linux hosts."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

from pydantic import BaseModel, ConfigDict, Field

from trialagentbench_verifier.assets import (
    AssetShard,
    ReleaseAssets,
    _verify_shard,
    materialize_asset_selection,
    read_asset_catalog,
)
from trialagentbench_verifier.io import VerificationInputError, sha256_file


class AssetCacheFootprint(BaseModel):
    """Selection-specific conservative storage bounds, separate from software."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    catalog_bytes: int = Field(ge=0)
    payload_bytes_to_acquire: int = Field(ge=0)
    verified_compressed_cache_bytes: int = Field(ge=0)
    additional_cache_bytes_bound: int = Field(ge=0)
    additional_scratch_bytes_bound: int = Field(ge=0)
    retained_outputs_budget_bytes: int | None = Field(default=None, ge=0)
    workers: int = Field(ge=1)
    view_reusable: bool


def _selection_key(catalog: ReleaseAssets, task_ids: tuple[str, ...] | None) -> str:
    catalog.selected_members(task_ids)
    selected = sorted(task_ids if task_ids is not None else (task.task_id for task in catalog.tasks))
    return hashlib.sha256(json.dumps(selected, separators=(",", ":")).encode()).hexdigest()


def cache_footprint(
    catalog_path: Path,
    *,
    expected_sha256: str,
    cache: Path,
    task_ids: tuple[str, ...] | None = None,
    workers: int = 1,
    retained_outputs_budget_bytes: int | None = None,
) -> AssetCacheFootprint:
    """Inspect reusable bytes without downloads, writes or assumed output sizes.

    Scratch includes one bounded executor workspace and all selected evidence
    per concurrent worker, plus one temporary transfer. Cache bounds assume
    logical copies when hard links are unavailable. Container images and host
    dependencies are separate; retained run outputs require an explicit budget.
    """
    from trialagentbench_verifier.execution_contracts import RELEASE_EXECUTION_LIMITS_V1

    if workers < 1:
        raise ValueError("Worker count must be positive")
    if any(parent.is_symlink() for parent in (cache, *cache.parents)):
        raise VerificationInputError("Cache paths cannot contain symbolic links")
    catalog = read_asset_catalog(catalog_path, expected_sha256=expected_sha256)
    shards = catalog.selected_shards(task_ids)
    reusable = 0
    missing = []
    for shard in shards:
        path = cache / "transport" / shard.filename
        if path.exists() or path.is_symlink():
            if path.parent.is_symlink():
                raise VerificationInputError("Cached shard directory cannot be a symbolic link")
            _verify_shard(path, shard)
            reusable += shard.byte_length
        else:
            missing.append(shard)
    view = cache / "views" / expected_sha256 / _selection_key(catalog, task_ids)
    warm = view.exists()
    if any(parent.is_symlink() for parent in (view, *view.parents)):
        raise VerificationInputError("Cached view path cannot contain symbolic links")
    if warm:
        _verify_view(view, catalog, task_ids, verify=True)
    members = catalog.selected_members(task_ids)
    logical = sum(member.byte_length for member in members)
    overhead = 4096 * (sum(len(Path(member.path).parts) + 1 for member in members) + len(shards) + 16)
    catalog_bytes = catalog_path.stat().st_size
    limits = RELEASE_EXECUTION_LIMITS_V1
    per_worker = limits.workspace_mb * 1024**2 + min(logical, limits.evidence_mb * 1024**2)
    return AssetCacheFootprint(
        catalog_bytes=catalog_bytes,
        payload_bytes_to_acquire=0 if warm else sum(shard.byte_length for shard in missing),
        verified_compressed_cache_bytes=reusable,
        additional_cache_bytes_bound=(
            0 if warm else sum(shard.byte_length for shard in missing) + catalog_bytes + logical + overhead
        ),
        additional_scratch_bytes_bound=workers * per_worker
        + (0 if warm else max((s.byte_length for s in missing), default=0)),
        retained_outputs_budget_bytes=retained_outputs_budget_bytes,
        workers=workers,
        view_reusable=warm,
    )


def _directory(path: Path) -> Path:
    """Refuse linked cache components before creating an owned directory."""
    path = path.absolute()
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise VerificationInputError(f"Cache paths cannot contain symbolic links: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise VerificationInputError(f"Cache location is not a directory: {path}")
    return path


@contextmanager
def _lock(path: Path, *, timeout: float = 30.0, shared: bool = False) -> Iterator[int]:
    """Hold a process-safe bounded lock; crashes release it without stale leases."""
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(descriptor, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Cache object is busy; retry after its current user finishes: {path}"
                    ) from None
                time.sleep(0.05)
        yield descriptor
    finally:
        os.close(descriptor)


def _download(shard: AssetShard, destination: Path, base_url: str) -> None:
    """Bound every HTTPS transfer by the trusted shard size and three attempts."""
    origin = urlsplit(base_url)
    if (
        origin.scheme != "https"
        or not origin.netloc
        or origin.username
        or origin.password
        or origin.query
        or origin.fragment
    ):
        raise ValueError("Asset base URL must be an HTTPS directory without credentials, query or fragment")
    url = base_url.rstrip("/") + "/" + shard.filename
    for attempt in range(3):
        try:
            with urlopen(url, timeout=30) as response, destination.open("wb") as target:
                if urlsplit(response.geturl()).scheme != "https":
                    raise VerificationInputError("Asset download redirected outside HTTPS")
                total = 0
                while body := response.read(1024 * 1024):
                    total += len(body)
                    if total > shard.byte_length:
                        raise VerificationInputError("Asset transfer exceeds its trusted byte length")
                    target.write(body)
                if total != shard.byte_length:
                    raise VerificationInputError("Asset transfer ended before its trusted byte length")
            return
        except (URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise
            time.sleep(0.25 * (attempt + 1))


def acquire_asset_selection(
    catalog_path: Path,
    *,
    expected_sha256: str,
    cache: Path,
    task_ids: tuple[str, ...] | None = None,
    base_url: str | None = None,
) -> Path:
    """Acquire only selected shards, reusing independently rehashed cached bytes.

    Local shards are copied when available. Network access occurs only with an
    explicit HTTPS base URL. Partial transfers remain private until their size
    and digest pass. Corrupt completed entries fail without deletion or eviction.
    """
    catalog = read_asset_catalog(catalog_path, expected_sha256=expected_sha256)
    transport = _directory(cache / "transport")
    shards = _directory(transport / "shards")
    locks = _directory(cache / "locks")
    for shard in catalog.selected_shards(task_ids):
        target = shards / f"{shard.sha256}.zip"
        with _lock(locks / f"{shard.sha256}.lock"):
            if target.exists() or target.is_symlink():
                _verify_shard(target, shard)
                continue
            with tempfile.TemporaryDirectory(prefix=".acquire-", dir=shards) as temporary:
                partial = Path(temporary) / "payload"
                source = catalog_path.parent / shard.filename
                if source.exists() or source.is_symlink():
                    if source.parent.is_symlink():
                        raise VerificationInputError("Local shard directory cannot be a symbolic link")
                    _verify_shard(source, shard)
                    shutil.copyfile(source, partial)
                elif base_url is not None:
                    _download(shard, partial, base_url)
                else:
                    raise FileNotFoundError(
                        f"Missing shard {shard.filename}; acquire it with assets fetch --base-url, then retry"
                    )
                _verify_shard(partial, shard)
                partial.chmod(0o444)
                os.rename(partial, target)
    cached_catalog = transport / f"{expected_sha256}.assets.json"
    with _lock(locks / f"{expected_sha256}.lock"):
        if cached_catalog.exists() or cached_catalog.is_symlink():
            read_asset_catalog(cached_catalog, expected_sha256=expected_sha256)
        else:
            # Preserve authenticated bytes rather than a reserialized approximation.
            with tempfile.TemporaryDirectory(prefix=".catalog-", dir=transport) as temporary:
                partial = Path(temporary) / "catalog"
                shutil.copyfile(catalog_path, partial)
                read_asset_catalog(partial, expected_sha256=expected_sha256)
                partial.chmod(0o444)
                os.rename(partial, cached_catalog)
    return cached_catalog


def _verify_view(root: Path, catalog: ReleaseAssets, task_ids: tuple[str, ...] | None, *, verify: bool) -> None:
    """Check the entire selected census, rejecting links and stale trust markers."""
    members = catalog.selected_members(task_ids)
    expected = {member.path for member in members}
    observed = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise VerificationInputError("Cached view contains a symbolic link")
        if not path.is_dir():
            observed.add(path.relative_to(root).as_posix())
    if observed != expected:
        raise VerificationInputError(
            "Cached view differs from its selected file census; retain it for diagnosis and choose a fresh cache"
        )
    for member in members:
        path = root / member.path
        if path.stat().st_size != member.byte_length or (verify and sha256_file(path) != member.sha256):
            raise VerificationInputError("Cached view content differs from its original digest; choose a fresh cache")


@contextmanager
def cached_asset_selection(
    catalog_path: Path,
    *,
    expected_sha256: str,
    cache: Path,
    task_ids: tuple[str, ...] | None = None,
    base_url: str | None = None,
) -> Iterator[Path]:
    """Lease a persistent verified view and detect mutation through the whole use.

    This context owns cache bytes only. The existing release action owns the
    explicit verification scope and stages participant records for the agent.
    """
    catalog = read_asset_catalog(catalog_path, expected_sha256=expected_sha256)
    selection = _selection_key(catalog, task_ids)
    for shard in catalog.selected_shards(task_ids):
        locations = {catalog_path.parent / shard.filename, cache / "transport" / shard.filename}
        for location in locations:
            if location.exists() or location.is_symlink():
                if any(parent.is_symlink() for parent in (location.parent, *location.parent.parents)):
                    raise VerificationInputError("Shard path cannot contain symbolic links")
                _verify_shard(location, shard)
    views = _directory(cache / "views" / expected_sha256)
    locks = _directory(cache / "locks")
    root = views / selection
    lock = locks / f"view-{expected_sha256}-{selection}.lock"
    if not root.exists():
        with _lock(lock.with_suffix(".build.lock")):
            if not root.exists():
                local = acquire_asset_selection(
                    catalog_path, expected_sha256=expected_sha256, cache=cache, task_ids=task_ids, base_url=base_url
                )
                materialize_asset_selection(
                    local,
                    expected_sha256=expected_sha256,
                    output=root,
                    task_ids=task_ids,
                    object_store=_directory(cache / "objects"),
                )
                for path in root.rglob("*"):
                    if path.is_file():
                        path.chmod(0o444)
    with _lock(lock, shared=True):
        if root.is_symlink():
            raise VerificationInputError("Cached view cannot be a symbolic link")
        _verify_view(root, catalog, task_ids, verify=True)
        yield root
