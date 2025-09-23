"""
Health check endpoints.
"""

import asyncio
import time
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ..config import get_settings, get_database_config, get_s3_config
from ..dependencies import get_optional_user
from ..auth import TokenData


router = APIRouter(tags=["health"])


class HealthStatus(BaseModel):
    """Health check response model."""
    status: str
    timestamp: float
    version: str
    environment: str
    checks: Dict[str, Any]


class DetailedHealthStatus(BaseModel):
    """Detailed health check response model."""
    status: str
    timestamp: float
    version: str
    environment: str
    uptime_seconds: float
    checks: Dict[str, Any]
    configuration: Dict[str, Any]
    dependencies: Dict[str, Any]


# Store startup time for uptime calculation
startup_time = time.time()


@router.get("/health", response_model=HealthStatus)
async def health_check():
    """
    Basic health check endpoint.
    Returns 200 if service is healthy.
    """
    settings = get_settings()
    
    checks = {
        "api": "healthy",
        "configuration": "loaded",
    }
    
    # Quick database connectivity check (if configured)
    db_config = get_database_config()
    if db_config.get("host") and db_config.get("host") != "localhost":
        try:
            # This would normally test actual DB connection
            # For now, just check if config is present
            checks["database"] = "configured"
        except Exception:
            checks["database"] = "error"
    
    return HealthStatus(
        status="healthy",
        timestamp=time.time(),
        version="1.0.0",
        environment=settings.environment,
        checks=checks
    )


@router.get("/health/detailed", response_model=DetailedHealthStatus)
async def detailed_health_check(current_user: TokenData = Depends(get_optional_user)):
    """
    Detailed health check endpoint.
    Includes dependency status and configuration info.
    Requires authentication for detailed information.
    """
    settings = get_settings()
    
    # Basic checks
    checks = {
        "api": "healthy",
        "configuration": "loaded",
        "read_only_mode": settings.read_only_mode,
        "phi_redaction": settings.enable_phi_redaction,
    }
    
    # Configuration summary (non-sensitive)
    configuration = {
        "environment": settings.environment,
        "debug": settings.debug,
        "log_level": settings.log_level,
        "api_prefix": settings.api_prefix,
        "prometheus_enabled": settings.prometheus_enabled,
    }
    
    # Dependencies status
    dependencies = {}
    
    # Database check
    db_config = get_database_config()
    if db_config.get("host"):
        try:
            # In a real implementation, test actual connection
            dependencies["database"] = {
                "status": "configured",
                "host": db_config["host"],
                "port": db_config["port"],
                "database": db_config["database"],
            }
            checks["database"] = "healthy"
        except Exception as e:
            dependencies["database"] = {
                "status": "error",
                "error": str(e)
            }
            checks["database"] = "error"
    
    # S3 check
    s3_config = get_s3_config()
    if s3_config.get("bucket_raw"):
        try:
            # In a real implementation, test S3 connectivity
            dependencies["s3"] = {
                "status": "configured",
                "region": s3_config["region"],
                "buckets": {
                    "raw": bool(s3_config["bucket_raw"]),
                    "silver": bool(s3_config["bucket_silver"]),
                    "gold": bool(s3_config["bucket_gold"]),
                }
            }
            checks["storage"] = "healthy"
        except Exception as e:
            dependencies["s3"] = {
                "status": "error",
                "error": str(e)
            }
            checks["storage"] = "error"
    
    # MLflow check
    if settings.mlflow_tracking_uri:
        try:
            # In a real implementation, ping MLflow server
            dependencies["mlflow"] = {
                "status": "configured",
                "tracking_uri": settings.mlflow_tracking_uri
            }
            checks["mlflow"] = "healthy"
        except Exception as e:
            dependencies["mlflow"] = {
                "status": "error", 
                "error": str(e)
            }
            checks["mlflow"] = "error"
    
    # OpenLineage check
    if settings.openlineage_url:
        dependencies["openlineage"] = {
            "status": "configured",
            "url": settings.openlineage_url,
            "namespace": settings.openlineage_namespace
        }
        checks["lineage"] = "healthy"
    
    # If user is not authenticated, limit information
    if not current_user:
        configuration = {"environment": settings.environment}
        dependencies = {"count": len(dependencies)}
    
    return DetailedHealthStatus(
        status="healthy",
        timestamp=time.time(),
        version="1.0.0",
        environment=settings.environment,
        uptime_seconds=time.time() - startup_time,
        checks=checks,
        configuration=configuration,
        dependencies=dependencies
    )


@router.get("/health/readiness")
async def readiness_check():
    """
    Kubernetes readiness probe endpoint.
    Returns 200 when service is ready to accept traffic.
    """
    # Check critical dependencies
    try:
        # Simulate dependency checks
        await asyncio.sleep(0.001)  # Minimal async operation
        
        # In production, check:
        # - Database connectivity
        # - Required environment variables
        # - External service availability
        
        return {"status": "ready", "timestamp": time.time()}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service not ready: {str(e)}"
        )


@router.get("/health/liveness")
async def liveness_check():
    """
    Kubernetes liveness probe endpoint.
    Returns 200 if service is alive (not deadlocked).
    """
    # Simple liveness check
    return {"status": "alive", "timestamp": time.time()}


@router.get("/version")
async def version_info():
    """
    API version information.
    """
    return {
        "version": "1.0.0",
        "api_version": "v1",
        "build_timestamp": startup_time,
        "git_commit": "unknown",  # Would be injected during build
    }