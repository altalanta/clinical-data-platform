from __future__ import annotations

import os
import json
import boto3
from pathlib import Path
from typing import Literal, Optional, Dict, Any
from functools import lru_cache

import yaml
from pydantic import BaseModel, SecretStr, Field
from pydantic_settings import BaseSettings


class StorageConfig(BaseModel):
    backend: Literal["minio", "s3"] = "minio"
    s3_endpoint: str = Field(default="http://localhost:9000")
    access_key: Optional[SecretStr] = None
    secret_key: Optional[SecretStr] = None
    raw_bucket: str = "clinical-raw"
    bronze_bucket: str = "clinical-bronze"
    silver_bucket: str = "clinical-silver"
    gold_bucket: str = "clinical-gold"
    use_ssl: bool = False


class WarehouseConfig(BaseModel):
    backend: Literal["duckdb", "postgres"] = "duckdb"
    duckdb_path: str = "clinical_warehouse.duckdb"
    postgres_host: Optional[str] = None
    postgres_port: int = 5432
    postgres_db: Optional[str] = None
    postgres_user: Optional[str] = None
    postgres_password: Optional[SecretStr] = None


class PathsConfig(BaseModel):
    data_root: str = "data"
    raw_dir: str = "raw"
    standardized_dir: str = "standardized"
    processed_dir: str = "processed"


class MlflowConfig(BaseModel):
    tracking_uri: str = "http://localhost:5000"
    auth_token: Optional[SecretStr] = None
    experiment_name: str = "clinical-platform"
    model_registry_stage: Literal["Staging", "Production"] = "Staging"


class SecurityConfig(BaseModel):
    api_key: Optional[SecretStr] = None
    jwt_secret: SecretStr = Field(default="change-in-production-please")
    enable_pii_redaction: bool = True
    read_only_mode: bool = False


class UnifiedConfig(BaseSettings):
    """Unified configuration system supporting both YAML and environment variables."""
    
    # Core application settings
    env: Literal["local", "dev", "staging", "prod"] = "local"
    app_name: str = "Clinical Data Platform"
    debug: bool = False
    log_level: str = "INFO"
    
    # Component configurations
    storage: StorageConfig = StorageConfig()
    warehouse: WarehouseConfig = WarehouseConfig()
    paths: PathsConfig = PathsConfig()
    mlflow: MlflowConfig = MlflowConfig()
    security: SecurityConfig = SecurityConfig()
    
    # AWS specific settings
    aws_region: str = "us-east-1"
    aws_account_id: Optional[str] = None
    
    model_config = {
        "env_file": ".env",
        "env_prefix": "CDP_",
        "case_sensitive": False,
        "extra": "ignore"
    }
        
    @classmethod
    def from_yaml(cls, path: str | Path) -> "UnifiedConfig":
        """Load configuration from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)
    
    @property
    def is_cloud_environment(self) -> bool:
        """Check if running in cloud environment."""
        return self.env in ["staging", "prod"] or bool(os.getenv("AWS_EXECUTION_ENV"))
    
    @property
    def storage_endpoint(self) -> str:
        """Get the appropriate storage endpoint based on backend."""
        if self.storage.backend == "s3":
            return f"https://s3.{self.aws_region}.amazonaws.com"
        return self.storage.s3_endpoint


class SecretsManager:
    """Manage secrets from AWS Secrets Manager for cloud deployments."""
    
    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.client = None
        if self._is_aws_environment():
            self.client = boto3.client('secretsmanager', region_name=region)
    
    def _is_aws_environment(self) -> bool:
        return bool(os.getenv("AWS_EXECUTION_ENV") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))
    
    @lru_cache(maxsize=32)
    def get_secret(self, secret_name: str) -> Dict[str, Any]:
        """Retrieve secret from AWS Secrets Manager."""
        if not self.client:
            return {}
        
        try:
            response = self.client.get_secret_value(SecretId=secret_name)
            return json.loads(response['SecretString'])
        except Exception as e:
            print(f"Warning: Could not retrieve secret {secret_name}: {e}")
            return {}
    
    def update_config_from_secrets(self, config: UnifiedConfig) -> UnifiedConfig:
        """Update configuration with secrets from AWS Secrets Manager."""
        if not self.client:
            return config
        
        project_name = f"clinical-platform-{config.env}"
        
        # Storage secrets
        storage_secrets = self.get_secret(f"{project_name}/storage")
        if storage_secrets:
            config.storage.access_key = storage_secrets.get("access_key")
            config.storage.secret_key = storage_secrets.get("secret_key")
        
        # Database secrets
        db_secrets = self.get_secret(f"{project_name}/database")
        if db_secrets:
            config.warehouse.postgres_password = db_secrets.get("password")
            config.warehouse.postgres_user = db_secrets.get("username")
            config.warehouse.postgres_host = db_secrets.get("host")
            config.warehouse.postgres_db = db_secrets.get("database")
        
        # Security secrets
        security_secrets = self.get_secret(f"{project_name}/security")
        if security_secrets:
            config.security.api_key = security_secrets.get("api_key")
            config.security.jwt_secret = security_secrets.get("jwt_secret", config.security.jwt_secret)
        
        # MLflow secrets
        mlflow_secrets = self.get_secret(f"{project_name}/mlflow")
        if mlflow_secrets:
            config.mlflow.auth_token = mlflow_secrets.get("auth_token")
            config.mlflow.tracking_uri = mlflow_secrets.get("tracking_uri", config.mlflow.tracking_uri)
        
        return config


@lru_cache()
def get_config() -> UnifiedConfig:
    """Get cached unified configuration with environment-based loading."""
    # First try environment variable loading
    config = UnifiedConfig()
    
    # Then try loading from YAML if available
    env = os.getenv("ENV", config.env).lower()
    cfg_path = Path(f"configs/config.{env}.yaml")
    
    if cfg_path.exists():
        config = UnifiedConfig.from_yaml(cfg_path)
        # Override with any environment variables
        env_config = UnifiedConfig()
        # Merge environment variables over YAML config
        config.env = env_config.env or config.env
        config.debug = env_config.debug if env_config.debug != UnifiedConfig().debug else config.debug
        config.log_level = env_config.log_level if env_config.log_level != UnifiedConfig().log_level else config.log_level
    
    # Update from AWS Secrets Manager if in cloud
    if config.is_cloud_environment:
        secrets_manager = SecretsManager(region=config.aws_region)
        config = secrets_manager.update_config_from_secrets(config)
    
    return config


# Legacy compatibility functions
def get_database_config() -> Dict[str, Any]:
    """Get database configuration for legacy compatibility."""
    config = get_config()
    if config.warehouse.backend == "postgres":
        return {
            "host": config.warehouse.postgres_host,
            "port": config.warehouse.postgres_port,
            "database": config.warehouse.postgres_db,
            "user": config.warehouse.postgres_user,
            "password": config.warehouse.postgres_password.get_secret_value() if config.warehouse.postgres_password else None,
        }
    else:
        return {"duckdb_path": config.warehouse.duckdb_path}


def get_storage_config() -> Dict[str, Any]:
    """Get storage configuration for legacy compatibility."""
    config = get_config()
    return {
        "backend": config.storage.backend,
        "endpoint": config.storage_endpoint,
        "raw_bucket": config.storage.raw_bucket,
        "bronze_bucket": config.storage.bronze_bucket,
        "silver_bucket": config.storage.silver_bucket,
        "gold_bucket": config.storage.gold_bucket,
        "access_key": config.storage.access_key.get_secret_value() if config.storage.access_key else None,
        "secret_key": config.storage.secret_key.get_secret_value() if config.storage.secret_key else None,
    }


# For backward compatibility
AppConfig = UnifiedConfig

