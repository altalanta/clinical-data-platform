"""
Configuration management for the API.
Reads from environment variables only - no .env files in cloud deployment.
"""

import os
import json
import boto3
from typing import Dict, Any, Optional
from functools import lru_cache
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings from environment variables."""
    
    # Application
    app_name: str = "Clinical Data Platform API"
    environment: str = os.getenv("ENVIRONMENT", "dev")
    debug: bool = os.getenv("DEBUG", "0") == "1"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Security
    jwt_secret: str = os.getenv("JWT_SECRET", "")  # No default in production
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
    
    # CORS Configuration
    cors_allow_origins: str = os.getenv("CORS_ALLOW_ORIGINS", "")
    cors_allow_credentials: bool = os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"
    cors_allow_methods: str = os.getenv("CORS_ALLOW_METHODS", "GET,POST,PUT,DELETE,OPTIONS")
    cors_allow_headers: str = os.getenv("CORS_ALLOW_HEADERS", "*")
    
    # Read-only mode and compliance
    read_only_mode: bool = os.getenv("READ_ONLY_MODE", "0") == "1"
    log_scrub_values: bool = os.getenv("LOG_SCRUB_VALUES", "1") == "1"
    enable_phi_redaction: bool = os.getenv("ENABLE_PHI_REDACTION", "1") == "1"
    
    # Database (from Secrets Manager in cloud)
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: int = int(os.getenv("DB_PORT", "5432"))
    db_name: str = os.getenv("DB_NAME", "clinical_platform")
    db_user: str = os.getenv("DB_USER", "postgres")
    db_password: str = os.getenv("DB_PASSWORD", "")
    
    # S3 Storage
    s3_bucket_raw: str = os.getenv("S3_BUCKET_RAW", "")
    s3_bucket_silver: str = os.getenv("S3_BUCKET_SILVER", "")
    s3_bucket_gold: str = os.getenv("S3_BUCKET_GOLD", "")
    s3_bucket_lineage: str = os.getenv("S3_BUCKET_LINEAGE", "")
    s3_bucket_artifacts: str = os.getenv("S3_BUCKET_ARTIFACTS", "")
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    
    # MLflow
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    
    # Observability
    prometheus_enabled: bool = os.getenv("PROMETHEUS_ENABLED", "1") == "1"
    openlineage_url: str = os.getenv("OPENLINEAGE_URL", "")
    openlineage_namespace: str = os.getenv("OPENLINEAGE_NAMESPACE", "clinical-platform")
    
    # API Configuration
    api_prefix: str = "/api/v1"
    max_request_size: int = int(os.getenv("MAX_REQUEST_SIZE", "10485760"))  # 10MB
    rate_limit_enabled: bool = os.getenv("RATE_LIMIT_ENABLED", "1") == "1"
    
    @property
    def database_url(self) -> str:
        """Construct database URL."""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    @property
    def is_cloud_environment(self) -> bool:
        """Check if running in cloud environment."""
        return self.environment in ["staging", "prod"] or bool(os.getenv("AWS_EXECUTION_ENV"))
    
    @property
    def is_production_environment(self) -> bool:
        """Check if running in production or staging."""
        return self.environment in ["prod", "production", "staging"]
    
    def get_cors_origins(self) -> list[str]:
        """Get CORS origins as a list, with security validation."""
        if not self.cors_allow_origins:
            if self.is_production_environment:
                raise ValueError(
                    "CORS_ALLOW_ORIGINS must be specified in production/staging. "
                    "Wildcard '*' is not allowed in production."
                )
            # Development default
            return ["http://localhost:3000", "http://localhost:8000", "http://localhost:8080"]
        
        origins = [origin.strip() for origin in self.cors_allow_origins.split(",")]
        
        # Validate production CORS settings
        if self.is_production_environment:
            if "*" in origins:
                raise ValueError(
                    "Wildcard '*' CORS origin is not allowed in production/staging. "
                    "Specify explicit allowed origins."
                )
            
            # Validate each origin is a proper URL
            for origin in origins:
                if not origin.startswith(("https://", "http://localhost", "http://127.0.0.1")):
                    raise ValueError(
                        f"Invalid CORS origin '{origin}' in production. "
                        "Must use HTTPS or localhost for testing."
                    )
        
        return origins
    
    def get_cors_methods(self) -> list[str]:
        """Get CORS methods as a list."""
        return [method.strip() for method in self.cors_allow_methods.split(",")]
    
    def get_cors_headers(self) -> list[str]:
        """Get CORS headers as a list."""
        if self.cors_allow_headers == "*":
            return ["*"]
        return [header.strip() for header in self.cors_allow_headers.split(",")]
    
    class Config:
        case_sensitive = False


class SecretsManager:
    """Helper class to retrieve secrets from AWS Secrets Manager."""
    
    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.client = None
        if self._is_aws_environment():
            self.client = boto3.client('secretsmanager', region_name=region)
    
    def _is_aws_environment(self) -> bool:
        """Check if running in AWS environment."""
        return bool(os.getenv("AWS_EXECUTION_ENV") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))
    
    @lru_cache(maxsize=32)
    def get_secret(self, secret_name: str) -> Dict[str, Any]:
        """Retrieve and cache secret from AWS Secrets Manager."""
        if not self.client:
            return {}
        
        try:
            response = self.client.get_secret_value(SecretId=secret_name)
            return json.loads(response['SecretString'])
        except Exception as e:
            print(f"Warning: Could not retrieve secret {secret_name}: {e}")
            return {}
    
    def update_settings_from_secrets(self, settings: Settings, project_name: str) -> Settings:
        """Update settings with values from Secrets Manager."""
        if not self.client:
            return settings
        
        # Database credentials
        db_secret = self.get_secret(f"{project_name}/database/credentials")
        if db_secret:
            settings.db_host = db_secret.get("DB_HOST", settings.db_host)
            settings.db_port = int(db_secret.get("DB_PORT", settings.db_port))
            settings.db_name = db_secret.get("DB_NAME", settings.db_name)
            settings.db_user = db_secret.get("DB_USER", settings.db_user)
            settings.db_password = db_secret.get("DB_PASSWORD", settings.db_password)
        
        # JWT secret
        jwt_secret = self.get_secret(f"{project_name}/auth/jwt-secret")
        if jwt_secret:
            settings.jwt_secret = jwt_secret.get("JWT_SECRET", settings.jwt_secret)
        
        # S3 configuration
        s3_config = self.get_secret(f"{project_name}/storage/s3-config")
        if s3_config:
            settings.s3_bucket_raw = s3_config.get("S3_BUCKET_RAW", settings.s3_bucket_raw)
            settings.s3_bucket_silver = s3_config.get("S3_BUCKET_SILVER", settings.s3_bucket_silver)
            settings.s3_bucket_gold = s3_config.get("S3_BUCKET_GOLD", settings.s3_bucket_gold)
            settings.s3_bucket_lineage = s3_config.get("S3_BUCKET_LINEAGE", settings.s3_bucket_lineage)
            settings.s3_bucket_artifacts = s3_config.get("S3_BUCKET_ARTIFACTS", settings.s3_bucket_artifacts)
        
        # Application configuration
        app_config = self.get_secret(f"{project_name}/app/config")
        if app_config:
            settings.environment = app_config.get("ENVIRONMENT", settings.environment)
            settings.log_level = app_config.get("LOG_LEVEL", settings.log_level)
            settings.read_only_mode = app_config.get("READ_ONLY_MODE", "0") == "1"
            settings.log_scrub_values = app_config.get("LOG_SCRUB_VALUES", "1") == "1"
            settings.enable_phi_redaction = app_config.get("ENABLE_PHI_REDACTION", "1") == "1"
            settings.prometheus_enabled = app_config.get("PROMETHEUS_ENABLED", "1") == "1"
            settings.openlineage_url = app_config.get("OPENLINEAGE_URL", settings.openlineage_url)
            settings.openlineage_namespace = app_config.get("OPENLINEAGE_NAMESPACE", settings.openlineage_namespace)
        
        # MLflow configuration
        mlflow_config = self.get_secret(f"{project_name}/mlflow/config")
        if mlflow_config:
            settings.mlflow_tracking_uri = mlflow_config.get("MLFLOW_TRACKING_URI", settings.mlflow_tracking_uri)
        
        return settings


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    settings = Settings()
    
    # Update from Secrets Manager if in cloud environment
    if settings.is_cloud_environment:
        secrets_manager = SecretsManager(region=settings.aws_region)
        project_name = f"clinical-platform-{settings.environment}"
        settings = secrets_manager.update_settings_from_secrets(settings, project_name)
    
    return settings


def get_database_config() -> Dict[str, Any]:
    """Get database configuration."""
    settings = get_settings()
    return {
        "host": settings.db_host,
        "port": settings.db_port,
        "database": settings.db_name,
        "user": settings.db_user,
        "password": settings.db_password,
        "url": settings.database_url
    }


def get_s3_config() -> Dict[str, Any]:
    """Get S3 configuration."""
    settings = get_settings()
    return {
        "bucket_raw": settings.s3_bucket_raw,
        "bucket_silver": settings.s3_bucket_silver,
        "bucket_gold": settings.s3_bucket_gold,
        "bucket_lineage": settings.s3_bucket_lineage,
        "bucket_artifacts": settings.s3_bucket_artifacts,
        "region": settings.aws_region
    }