"""
JWT Authentication and Authorization module for clinical data platform.
Implements RBAC with role-based claims with secure credential management.
"""

import os
import secrets
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from enum import Enum

import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status
from pydantic import BaseModel


class UserRole(str, Enum):
    """User roles for RBAC."""
    ADMIN = "admin"
    ANALYST = "analyst"
    BI_READONLY = "bi_readonly"
    RESEARCHER = "researcher"
    CLINICIAN = "clinician"


class TokenData(BaseModel):
    """Token payload data."""
    sub: str
    roles: List[UserRole]
    exp: datetime
    iat: datetime
    iss: str = "clinical-data-platform"


class UserCreate(BaseModel):
    """User creation model."""
    username: str
    email: str
    roles: List[UserRole]
    password: str


class UserLogin(BaseModel):
    """User login model."""
    username: str
    password: str


class Token(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    roles: List[UserRole]


class AuthService:
    """Authentication and authorization service with secure credential management."""
    
    def __init__(self):
        self.environment = os.getenv("ENVIRONMENT", "dev")
        self._validate_jwt_secret()
        
        self.secret_key = os.getenv("JWT_SECRET")
        self.algorithm = "HS256"
        self.access_token_expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # Anti-timing attack configuration
        self.auth_delay_enabled = os.getenv("AUTH_DELAY_ENABLED", "true").lower() == "true"
        self.auth_delay_min_ms = int(os.getenv("AUTH_DELAY_MIN_MS", "50"))
        self.auth_delay_max_ms = int(os.getenv("AUTH_DELAY_MAX_MS", "100"))
        
        # User database - load from environment or external store
        self.users_db = self._load_user_database()
    
    def _validate_jwt_secret(self) -> None:
        """Validate JWT secret is present and secure in production."""
        jwt_secret = os.getenv("JWT_SECRET")
        
        if self.environment in ["prod", "production", "staging"]:
            if not jwt_secret:
                raise ValueError(
                    "JWT_SECRET environment variable is required in production/staging"
                )
            if len(jwt_secret) < 32:
                raise ValueError(
                    "JWT_SECRET must be at least 32 characters in production/staging"
                )
        elif not jwt_secret:
            # Generate a random secret for development
            os.environ["JWT_SECRET"] = secrets.token_urlsafe(32)
    
    def _load_user_database(self) -> Dict[str, Dict[str, Any]]:
        """Load user database from environment variables or return empty dict."""
        # In production, users should be loaded from a secure database
        # For development, load from environment variables if present
        users_db = {}
        
        # Allow loading a single admin user from environment for bootstrapping
        admin_hash = os.getenv("ADMIN_PASSWORD_HASH")
        if admin_hash:
            users_db["admin"] = {
                "username": "admin",
                "email": os.getenv("ADMIN_EMAIL", "admin@clinical-platform.com"),
                "hashed_password": admin_hash,
                "roles": [UserRole.ADMIN]
            }
        
        return users_db

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return self.pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """Hash a password."""
        return self.pwd_context.hash(password)

    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate a user by username and password.
        Uses constant-time operations to prevent timing attacks.
        """
        start_time = time.time()
        
        # Always retrieve user and perform hash verification to normalize timing
        user = self.users_db.get(username)
        
        if user:
            # Verify against actual hash
            password_valid = self.verify_password(password, user["hashed_password"])
        else:
            # Perform dummy hash verification to normalize timing
            dummy_hash = "$2b$12$dummy.hash.to.prevent.timing.attacks.abcdefghijklmnopqr"
            self.verify_password(password, dummy_hash)
            password_valid = False
        
        # Add random delay to further prevent timing attacks
        if self.auth_delay_enabled:
            delay_ms = secrets.randbelow(
                self.auth_delay_max_ms - self.auth_delay_min_ms + 1
            ) + self.auth_delay_min_ms
            time.sleep(delay_ms / 1000.0)
        
        # Return user only if both user exists and password is valid
        return user if (user and password_valid) else None

    def create_access_token(self, username: str, roles: List[UserRole]) -> str:
        """Create a JWT access token."""
        now = datetime.utcnow()
        expire = now + timedelta(minutes=self.access_token_expire_minutes)
        
        payload = {
            "sub": username,
            "roles": [role.value for role in roles],
            "exp": expire,
            "iat": now,
            "iss": "clinical-data-platform"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> TokenData:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username = payload.get("sub")
            if username is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token",
                    headers={"WWW-Authenticate": "Bearer"}
                )
            
            roles = [UserRole(role) for role in payload.get("roles", [])]
            
            return TokenData(
                sub=username,
                roles=roles,
                exp=datetime.fromtimestamp(payload.get("exp")),
                iat=datetime.fromtimestamp(payload.get("iat"))
            )
            
        except jwt.PyJWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"}
            )

    def login(self, credentials: UserLogin) -> Token:
        """Login and return JWT token."""
        user = self.authenticate_user(credentials.username, credentials.password)
        if not user:
            # Always return the same error message to prevent user enumeration
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        access_token = self.create_access_token(
            username=user["username"],
            roles=user["roles"]
        )
        
        return Token(
            access_token=access_token,
            expires_in=self.access_token_expire_minutes * 60,
            roles=user["roles"]
        )

    def has_role(self, user_roles: List[UserRole], required_role: UserRole) -> bool:
        """Check if user has a specific role."""
        return required_role in user_roles

    def has_any_role(self, user_roles: List[UserRole], required_roles: List[UserRole]) -> bool:
        """Check if user has any of the required roles."""
        return any(role in user_roles for role in required_roles)

    def check_permissions(self, user_roles: List[UserRole], resource: str, action: str) -> bool:
        """Check if user has permission for a specific resource and action."""
        # Define role-based permissions
        permissions = {
            UserRole.ADMIN: {
                "*": ["*"]  # Admin has all permissions
            },
            UserRole.ANALYST: {
                "data": ["read", "write", "analyze"],
                "models": ["read", "write", "train"],
                "reports": ["read", "write"]
            },
            UserRole.BI_READONLY: {
                "data": ["read"],
                "reports": ["read"],
                "dashboard": ["read"]
            },
            UserRole.RESEARCHER: {
                "data": ["read", "analyze"],
                "models": ["read", "train"],
                "reports": ["read", "write"]
            },
            UserRole.CLINICIAN: {
                "data": ["read"],
                "reports": ["read"],
                "patients": ["read"]
            }
        }
        
        # Check admin permissions
        if UserRole.ADMIN in user_roles:
            return True
            
        # Check specific role permissions
        for role in user_roles:
            role_perms = permissions.get(role, {})
            
            # Check wildcard permissions
            if "*" in role_perms and "*" in role_perms["*"]:
                return True
                
            # Check resource-specific permissions
            if resource in role_perms:
                allowed_actions = role_perms[resource]
                if "*" in allowed_actions or action in allowed_actions:
                    return True
                    
        return False


# Global auth service instance
auth_service = AuthService()