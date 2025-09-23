"""
FastAPI dependencies for authentication and authorization.
"""

import os
from typing import List, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .auth import auth_service, UserRole, TokenData


# Security scheme
security = HTTPBearer()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenData:
    """
    Dependency to get the current authenticated user from JWT token.
    """
    return auth_service.verify_token(credentials.credentials)


def require_roles(*required_roles: UserRole):
    """
    Dependency factory to require specific roles.
    
    Usage:
        @app.get("/admin-only")
        async def admin_endpoint(user: TokenData = Depends(require_roles(UserRole.ADMIN))):
            return {"message": "Admin access granted"}
    """
    def check_roles(current_user: TokenData = Depends(get_current_user)) -> TokenData:
        if not auth_service.has_any_role(current_user.roles, list(required_roles)):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {[role.value for role in required_roles]}"
            )
        return current_user
    
    return check_roles


def require_permission(resource: str, action: str):
    """
    Dependency factory to require specific permissions.
    
    Usage:
        @app.post("/data")
        async def create_data(user: TokenData = Depends(require_permission("data", "write"))):
            return {"message": "Data creation allowed"}
    """
    def check_permission(current_user: TokenData = Depends(get_current_user)) -> TokenData:
        if not auth_service.check_permissions(current_user.roles, resource, action):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required permission: {action} on {resource}"
            )
        return current_user
    
    return check_permission


def get_optional_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[TokenData]:
    """
    Dependency to get the current user if authenticated, None otherwise.
    Useful for endpoints that have optional authentication.
    """
    if not credentials:
        return None
    
    try:
        return auth_service.verify_token(credentials.credentials)
    except HTTPException:
        return None


def check_read_only_mode():
    """
    Dependency to check if the application is in read-only mode.
    Blocks all write operations when READ_ONLY_MODE=1.
    """
    read_only = os.getenv("READ_ONLY_MODE", "0") == "1"
    if read_only:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Application is in read-only mode. Write operations are disabled."
        )


def get_database_config():
    """
    Dependency to get database configuration from environment or Secrets Manager.
    """
    # In cloud environment, this would read from Secrets Manager
    # For now, return environment variables
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "database": os.getenv("DB_NAME", "clinical_platform"),
        "username": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", ""),
    }


def get_s3_config():
    """
    Dependency to get S3 configuration from environment or Secrets Manager.
    """
    return {
        "bucket_raw": os.getenv("S3_BUCKET_RAW", "clinical-platform-raw"),
        "bucket_silver": os.getenv("S3_BUCKET_SILVER", "clinical-platform-silver"),
        "bucket_gold": os.getenv("S3_BUCKET_GOLD", "clinical-platform-gold"),
        "bucket_lineage": os.getenv("S3_BUCKET_LINEAGE", "clinical-platform-lineage"),
        "bucket_artifacts": os.getenv("S3_BUCKET_ARTIFACTS", "clinical-platform-artifacts"),
        "region": os.getenv("AWS_REGION", "us-east-1"),
    }


# Convenience dependencies for common role requirements
require_admin = require_roles(UserRole.ADMIN)
require_analyst = require_roles(UserRole.ANALYST, UserRole.ADMIN)
require_bi_readonly = require_roles(UserRole.BI_READONLY, UserRole.ANALYST, UserRole.ADMIN)
require_researcher = require_roles(UserRole.RESEARCHER, UserRole.ADMIN)

# Common permission dependencies
require_data_read = require_permission("data", "read")
require_data_write = require_permission("data", "write")
require_model_train = require_permission("models", "train")
require_report_read = require_permission("reports", "read")
require_report_write = require_permission("reports", "write")