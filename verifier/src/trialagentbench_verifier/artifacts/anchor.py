"""Resolve required release inputs through the GitHub-pinned root checksum census."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.request import urlopen

from pydantic import BaseModel, ConfigDict, Field

from trialagentbench_verifier.artifacts.io import VerificationInputError, sha256_file

_SHA256 = re.compile(r"[0-9a-f]{64}")


class ReleaseDatasetAnchorV1(BaseModel):
    """The published GitHub root declaration, read at an immutable commit."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["trialagentbench.release_dataset/v1"]
    repository: str
    revision: str
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sha256sums_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _anchor_digest(anchor_sha256: str | None, anchor_url: str | None) -> str:
    if (anchor_sha256 is None) == (anchor_url is None):
        raise VerificationInputError("exactly one root anchor is required: anchor_sha256 or anchor_url")
    if anchor_url is not None:
        url, separator, commit = anchor_url.rpartition("@")
        if not separator or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise VerificationInputError("anchor URL requires @ followed by an immutable GitHub commit")
        match = re.fullmatch(r"https://github.com/([^/]+/[^/]+)/(?:blob/[^/]+/)?(.+)", url)
        if match is None:
            raise VerificationInputError("anchor URL must identify a GitHub repository file")
        repository, filename = match.groups()
        with urlopen(f"https://raw.githubusercontent.com/{repository}/{commit}/{filename}", timeout=30) as response:
            anchor = ReleaseDatasetAnchorV1.model_validate_json(response.read(64 * 1024))
        anchor_sha256 = anchor.sha256sums_sha256
    if anchor_sha256 is None or _SHA256.fullmatch(anchor_sha256) is None:
        raise VerificationInputError("root anchor SHA-256 must be lowercase hexadecimal")
    return anchor_sha256


class ReleaseInputResolver:
    """Authenticate the root census and resolve only inputs needed by a command.

    Parameters
    ----------
    source
        Required release input; its ancestors are searched for the root census.
    anchor_sha256
        Trusted ``sha256sums_sha256`` from GitHub's release-dataset declaration.
    anchor_url
        GitHub declaration URL suffixed with ``@`` and its full immutable commit.
    sums
        Optional location of the root ``SHA256SUMS`` bytes.
    """

    def __init__(
        self,
        source: Path,
        *,
        anchor_sha256: str | None = None,
        anchor_url: str | None = None,
        sums: Path | None = None,
    ) -> None:
        digest = _anchor_digest(anchor_sha256, anchor_url)
        source = Path(source).absolute()
        candidates = (source, *source.parents) if source.is_dir() else tuple(source.parents)
        if sums is None:
            sums = next(
                (
                    root / "SHA256SUMS"
                    for root in candidates
                    if (root / "SHA256SUMS").is_file() and sha256_file(root / "SHA256SUMS") == digest
                ),
                None,
            )
        if sums is None or not sums.is_file() or sums.is_symlink() or sha256_file(sums) != digest:
            raise VerificationInputError("root SHA256SUMS differs from the GitHub anchor or is missing")
        self.entries: dict[str, str] = {}
        for line in sums.read_text(encoding="utf-8").splitlines():
            checksum, separator, name = line.partition("  ")
            path = PurePosixPath(name)
            if (
                not separator
                or _SHA256.fullmatch(checksum) is None
                or not name
                or path.is_absolute()
                or ".." in path.parts
                or path.as_posix() != name
                or name in self.entries
            ):
                raise VerificationInputError("root SHA256SUMS has an invalid or repeated entry")
            self.entries[name] = checksum
        # The override can live separately from the download (for example retained anchor evidence).
        self.roots = tuple(dict.fromkeys((sums.absolute().parent, *candidates)))

    def require(self, path: Path, *, expected_sha256: str | None = None) -> str:
        """Return an input's digest only when it is reachable from the root anchor.

        Direct file pins and extracted release manifests may be reached through
        an authenticated asset catalog's source archive or member inventory.

        Parameters
        ----------
        path
            Required file or extracted release directory. Directory files must
            match the anchored census or authenticated catalog member inventory.
        expected_sha256
            Optional direct pin, checked in addition to root-anchor reachability.

        Returns
        -------
        str
            Authenticated SHA-256 of the file or directory's release manifest.

        Raises
        ------
        VerificationInputError
            If the required bytes are missing, changed or unreachable from the anchor.
        """
        path = Path(path)
        identity = path / "release_manifest.json" if path.is_dir() else path
        if not identity.is_file() or identity.is_symlink():
            raise VerificationInputError("anchored input is missing or unsafe")
        actual = sha256_file(identity)
        if expected_sha256 is not None and actual != expected_sha256:
            raise VerificationInputError("input differs from its trusted expected SHA-256")
        directory_files = tuple(file for file in path.rglob("*") if file.is_file()) if path.is_dir() else ()
        if actual in self.entries.values() and all(
            not file.is_symlink() and sha256_file(file) in self.entries.values() for file in directory_files
        ):
            return actual
        from trialagentbench_verifier.artifacts.assets import read_asset_catalog

        for name, digest in self.entries.items():
            if not name.endswith(".assets.json"):
                continue
            for root in self.roots:
                catalog_path = root / name
                if not catalog_path.is_file():
                    continue
                catalog = read_asset_catalog(catalog_path, expected_sha256=digest)
                if path.is_dir():
                    if actual != catalog.release_manifest_sha256:
                        continue
                    members = {member.path: member for member in catalog.members}
                    if all(
                        not file.is_symlink()
                        and (member := members.get(file.relative_to(path).as_posix())) is not None
                        and file.stat().st_size == member.byte_length
                        and sha256_file(file) == member.sha256
                        for file in directory_files
                    ):
                        return actual
                elif actual == catalog.source_archive_sha256 or any(
                    member.sha256 == actual for member in catalog.members
                ):
                    return actual
        raise VerificationInputError("required input is not reachable from the GitHub root anchor")
