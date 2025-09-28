"""Hashing helpers for deterministic manifests."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List


def sha256_bytes(payload: bytes) -> str:
    """Return the hex digest of the given bytes."""
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the sha256 hex digest of a file."""
    digest = hashlib.sha256()
    file_path = Path(path)
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ManifestEntry:
    relpath: str
    size: int
    sha256: str
    source_url: str


@dataclass(frozen=True)
class Manifest:
    version: int
    generated_at_utc: str
    files: List[ManifestEntry]


def write_manifest(entries: Iterable[ManifestEntry], out_path: str | Path) -> None:
    """Write a deterministic manifest sorted by relative path."""
    timestamp = datetime.now(timezone.utc).isoformat()
    sorted_entries = sorted(entries, key=lambda item: item.relpath)
    manifest = Manifest(version=1, generated_at_utc=timestamp, files=list(sorted_entries))
    payload = json.dumps(
        {
            "version": manifest.version,
            "generated_at_utc": manifest.generated_at_utc,
            "files": [asdict(entry) for entry in manifest.files],
        },
        indent=2,
    )
    Path(out_path).write_text(payload + "\n", encoding="utf-8")


__all__ = [
    "sha256_bytes",
    "sha256_file",
    "ManifestEntry",
    "write_manifest",
]
