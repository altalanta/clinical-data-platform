from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Iterable

import boto3

from clinical_platform.config import AppConfig, get_config


@dataclass
class S3Client:
    client: any
    cfg: AppConfig

    @classmethod
    def from_config(cls, cfg: AppConfig | None = None) -> "S3Client":
        cfg = cfg or get_config()
        session = boto3.session.Session()
        client = session.client(
            "s3",
            endpoint_url=cfg.storage.s3_endpoint,
            aws_access_key_id=cfg.storage.access_key,
            aws_secret_access_key=cfg.storage.secret_key,
            use_ssl=cfg.storage.use_ssl,
        )
        return cls(client=client, cfg=cfg)

    def ensure_buckets(self) -> None:
        for b in [
            self.cfg.storage.raw_bucket,
            self.cfg.storage.bronze_bucket,
            self.cfg.storage.silver_bucket,
        ]:
            try:
                self.client.head_bucket(Bucket=b)
            except Exception:
                self.client.create_bucket(Bucket=b)

    def put_bytes(self, bucket: str, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=bucket, Key=key, Body=BytesIO(data))

    def list_keys(self, bucket: str, prefix: str = "") -> Iterable[str]:
        resp = self.client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        for obj in resp.get("Contents", []) or []:
            yield obj["Key"]

    def get_bytes(self, bucket: str, key: str) -> bytes:
        obj = self.client.get_object(Bucket=bucket, Key=key)
        return obj["Body"].read()

