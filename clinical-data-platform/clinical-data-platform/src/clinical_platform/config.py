from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class StorageConfig(BaseModel):
    s3_endpoint: str
    access_key: str | None
    secret_key: str | None
    raw_bucket: str
    bronze_bucket: str
    silver_bucket: str
    use_ssl: bool = False


class WarehouseConfig(BaseModel):
    duckdb_path: str


class PathsConfig(BaseModel):
    data_root: str
    raw_dir: str
    standardized_dir: str


class MlflowConfig(BaseModel):
    tracking_uri: str


class AppConfig(BaseModel):
    env: Literal["local", "aws"] = "local"
    storage: StorageConfig
    warehouse: WarehouseConfig
    paths: PathsConfig
    mlflow: MlflowConfig

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)


def get_config() -> AppConfig:
    cfg_path = Path("configs/config.local.yaml")
    if Path(".env").exists():
        # environment switching could be enhanced here based on ENV
        pass
    return AppConfig.load(cfg_path)

