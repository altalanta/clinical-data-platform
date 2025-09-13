from __future__ import annotations

from pathlib import Path

from clinical_platform.config import get_config
from clinical_platform.ingestion.s3_client import S3Client


def main() -> None:
    cfg = get_config()
    s3 = S3Client.from_config(cfg)
    s3.ensure_buckets()
    raw_dir = Path(cfg.paths.raw_dir)
    for p in raw_dir.glob("*.csv"):
        key = f"{p.name}"
        s3.put_bytes(cfg.storage.raw_bucket, key, p.read_bytes())
        print(f"Uploaded {p} -> s3://{cfg.storage.raw_bucket}/{key}")


if __name__ == "__main__":
    main()

