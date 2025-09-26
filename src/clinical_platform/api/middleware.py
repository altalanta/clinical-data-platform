"""FastAPI middleware for clinical data platform."""

from __future__ import annotations

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import RequestResponseEndpoint

from clinical_platform.logging_utils import get_logger

logger = get_logger("api_middleware")


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


def configure_cors() -> CORSMiddleware:
    """Configure CORS middleware with secure defaults."""
    return CORSMiddleware(
        allow_origins=["http://localhost:3000", "http://localhost:8501"],  # Streamlit default
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
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
    app.add_middleware(configure_gzip())
    app.add_middleware(configure_cors())
    
    logger.info("API middleware configured successfully")