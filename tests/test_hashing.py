from __future__ import annotations

import json
from pathlib import Path

from cp_tox_mini.hashing import ManifestEntry, sha256_bytes, sha256_file, write_manifest


def test_sha256_bytes_and_file(tmp_path: Path) -> None:
    payload = b"deterministic"
    tmp_file = tmp_path / "payload.bin"
    tmp_file.write_bytes(payload)

    digest_bytes = sha256_bytes(payload)
    digest_file = sha256_file(tmp_file)
    assert digest_bytes == digest_file
    assert len(digest_bytes) == 64


def test_write_manifest(tmp_path: Path) -> None:
    entries = [
        ManifestEntry(relpath="data/raw/a.txt", size=1, sha256="abc", source_url="local://a"),
        ManifestEntry(relpath="data/raw/b.txt", size=2, sha256="def", source_url="local://b"),
    ]
    manifest_path = tmp_path / "manifest.json"
    write_manifest(entries, manifest_path)

    content = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert content["version"] == 1
    assert "generated_at_utc" in content
    file_entries = content["files"]
    assert [entry["relpath"] for entry in file_entries] == ["data/raw/a.txt", "data/raw/b.txt"]
