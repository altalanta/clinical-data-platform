"""Data IO utilities and manifest validation."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List

from .hashing import ManifestEntry, sha256_file, write_manifest

PACKAGE_ROOT = Path(__file__).resolve().parent
STATIC_RAW = PACKAGE_ROOT / "static" / "raw"
DATA_ROOT = Path("data")
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
MANIFEST_PATH = Path("manifests/data_manifest.json")


def download_inputs() -> List[ManifestEntry]:
    """Copy bundled demo assets into the workspace and emit a manifest."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    entries: List[ManifestEntry] = []

    for asset in sorted(STATIC_RAW.iterdir()):
        target = RAW_DIR / asset.name
        shutil.copyfile(asset, target)
        sha = sha256_file(target)
        size = target.stat().st_size
        entries.append(
            ManifestEntry(
                relpath=str(target.relative_to(Path.cwd())),
                size=size,
                sha256=sha,
                source_url=f"local://{asset.relative_to(PACKAGE_ROOT)}",
            )
        )

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(entries, MANIFEST_PATH)
    return entries


def _load_manifest() -> Dict[str, Dict[str, str | int]]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("Manifest not found; run download command first")
    content = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if content.get("version") != 1:
        raise ValueError("Unsupported manifest version")
    files = content.get("files")
    if not isinstance(files, list):
        raise ValueError("Manifest is missing file entries")
    return {entry["relpath"]: entry for entry in files}


def validate_manifest() -> None:
    """Recompute hashes for raw inputs and assert they match the manifest."""
    entries = _load_manifest()
    mismatches: List[str] = []

    for relpath, entry in entries.items():
        path = Path(relpath)
        if not path.exists():
            mismatches.append(f"missing: {relpath}")
            continue
        recorded_sha = entry.get("sha256")
        recorded_size = entry.get("size")
        current_sha = sha256_file(path)
        current_size = path.stat().st_size
        if current_sha != recorded_sha:
            mismatches.append(f"sha mismatch for {relpath}")
        if current_size != recorded_size:
            mismatches.append(f"size mismatch for {relpath}")

    if mismatches:
        raise ValueError("Manifest validation failed: " + ", ".join(mismatches))


__all__ = [
    "download_inputs",
    "validate_manifest",
    "RAW_DIR",
    "PROCESSED_DIR",
    "MANIFEST_PATH",
]
