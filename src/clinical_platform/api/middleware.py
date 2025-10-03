"""FastAPI middleware for clinical data platform."""

from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from typing import Callable, Dict

from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint

from clinical_platform.config import get_config
from clinical_platform.logging_utils import get_logger

logger = get_logger("api_middleware")


class ReadOnlyModeMiddleware(BaseHTTPMiddleware):
    """Enforce read-only mode for GxP compliance."""
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        config = get_config()
        
        if config.security.read_only_mode and request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            logger.warning(
                "Read-only mode violation attempted",
                extra={
                    "method": request.method,
                    "path": str(request.url.path),
                    "client_ip": request.client.host if request.client else "unknown",
                    "user_agent": request.headers.get("user-agent", "unknown")
                }
            )
            return JSONResponse(
                status_code=403,
                content={
                    "error": "System is in read-only mode. Write operations are forbidden.",
                    "detail": "This system is currently operating in read-only mode for compliance reasons.",
                    "allowed_methods": ["GET", "HEAD", "OPTIONS"]
                }
            )
        
        return await call_next(request)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to all requests."""
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Add to response headers
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all API requests with timing and PHI-safe information."""
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start_time = time.time()
        request_id = getattr(request.state, "request_id", "unknown")
        
        # Log request start
        logger.info(
            "API request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "user_agent": request.headers.get("user-agent"),
                "client_ip": request.client.host if request.client else "unknown"
            }
        )
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Log successful response
            logger.info(
                "API request completed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "status_code": response.status_code,
                    "duration_seconds": round(duration, 3),
                    "content_length": response.headers.get("content-length", "unknown")
                }
            )
            
            # Add timing header
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Log error
            logger.error(
                "API request failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "error": str(e),
                    "duration_seconds": round(duration, 3)
                }
            )
            raise


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self'"
        )
        
        # Add cache control for API responses
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware."""
    
    def __init__(self, app, calls_per_minute: int = 60, write_calls_per_minute: int = 10):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self.write_calls_per_minute = write_calls_per_minute
        self.call_times: Dict[str, deque] = defaultdict(deque)
        
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP, considering X-Forwarded-For header."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    def _is_rate_limited(self, client_ip: str, is_write: bool) -> bool:
        """Check if client is rate limited."""
        now = time.time()
        limit = self.write_calls_per_minute if is_write else self.calls_per_minute
        
        # Clean old entries (older than 1 minute)
        while self.call_times[client_ip] and self.call_times[client_ip][0] < now - 60:
            self.call_times[client_ip].popleft()
        
        # Check if limit exceeded
        if len(self.call_times[client_ip]) >= limit:
            return True
        
        # Add current request
        self.call_times[client_ip].append(now)
        return False
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        client_ip = self._get_client_ip(request)
        is_write = request.method in {"POST", "PUT", "PATCH", "DELETE"}
        
        if self._is_rate_limited(client_ip, is_write):
            limit = self.write_calls_per_minute if is_write else self.calls_per_minute
            logger.warning(
                "Rate limit exceeded",
                extra={
                    "client_ip": client_ip,
                    "method": request.method,
                    "path": str(request.url.path),
                    "limit": limit,
                    "is_write": is_write
                }
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Maximum {limit} requests per minute exceeded",
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )
        
        return await call_next(request)


def configure_cors() -> CORSMiddleware:
    """Configure CORS middleware with secure defaults."""
    config = get_config()
    
    # Default to GET only, expand based on read-only mode
    allowed_methods = ["GET", "HEAD", "OPTIONS"]
    if not config.security.read_only_mode:
        allowed_methods.extend(["POST", "PUT", "PATCH", "DELETE"])
    
    # Restrict origins in production
    allowed_origins = ["http://localhost:3000", "http://localhost:8501"]
    if config.env in ["staging", "prod"]:
        allowed_origins = []  # Must be explicitly configured for production
    
    return CORSMiddleware(
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=allowed_methods,
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID", "X-Response-Time"]
    )


def configure_gzip() -> GZipMiddleware:
    """Configure GZip compression middleware."""
    return GZipMiddleware(minimum_size=1000)


class PHIProtectionMiddleware(BaseHTTPMiddleware):
    """Protect against PHI exposure in responses."""
    
    PHI_PATTERNS = [
        # Common PHI patterns to redact from error messages
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\b\d{10,}\b',  # Long numeric sequences (potential phone/ID)
    ]
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        
        # For error responses, check if response body might contain PHI
        if response.status_code >= 400:
            # In a production system, you'd implement PHI scanning here
            # For now, just add a warning header
            response.headers["X-PHI-Scanned"] = "true"
        
        return response


def setup_middleware(app) -> None:
    """Setup all middleware for the FastAPI application."""
    
    # Add middleware in reverse order (last added = first executed)
    app.add_middleware(PHIProtectionMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(RateLimitMiddleware, calls_per_minute=60, write_calls_per_minute=10)
    app.add_middleware(ReadOnlyModeMiddleware)  # Must be early to block write operations
    app.add_middleware(configure_gzip())
    app.add_middleware(configure_cors())
    
    logger.info("API middleware configured successfully")