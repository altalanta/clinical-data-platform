"""
Telemetry and monitoring instrumentation for FastAPI.
Includes Prometheus metrics, request tracking, and performance monitoring.
"""

import time
import logging
from typing import Callable, Dict, Any
from contextvars import ContextVar

from fastapi import FastAPI, Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response as StarletteResponse

# Context variables for request tracking
request_id_var: ContextVar[str] = ContextVar('request_id', default='')
user_id_var: ContextVar[str] = ContextVar('user_id', default='')

# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status_code', 'user_role']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint', 'status_code'],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
)

REQUEST_SIZE = Histogram(
    'http_request_size_bytes',
    'HTTP request size in bytes',
    ['method', 'endpoint']
)

RESPONSE_SIZE = Histogram(
    'http_response_size_bytes',
    'HTTP response size in bytes',
    ['method', 'endpoint', 'status_code']
)

ACTIVE_CONNECTIONS = Gauge(
    'http_active_connections',
    'Number of active HTTP connections'
)

DB_CONNECTIONS = Gauge(
    'database_connections_active',
    'Number of active database connections'
)

DATA_PROCESSING_TIME = Histogram(
    'data_processing_duration_seconds',
    'Time spent processing data',
    ['operation_type', 'data_source']
)

PHI_REDACTION_COUNT = Counter(
    'phi_redaction_total',
    'Total number of PHI redaction operations',
    ['redaction_type', 'field_type']
)

VALIDATION_ERRORS = Counter(
    'data_validation_errors_total',
    'Total number of data validation errors',
    ['validation_type', 'error_type']
)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect HTTP metrics."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip metrics endpoint to avoid recursion
        if request.url.path == "/metrics":
            return await call_next(request)
        
        # Increment active connections
        ACTIVE_CONNECTIONS.inc()
        
        start_time = time.time()
        
        # Get request size
        request_size = int(request.headers.get("content-length", 0))
        
        # Extract endpoint pattern (remove query params and path params)
        endpoint = self._get_endpoint_pattern(request)
        method = request.method
        
        # Get user role from request if available
        user_role = "anonymous"
        if hasattr(request.state, "user") and request.state.user:
            user_roles = getattr(request.state.user, "roles", [])
            user_role = user_roles[0].value if user_roles else "authenticated"
        
        try:
            response = await call_next(request)
            status_code = str(response.status_code)
            
            # Record metrics
            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code,
                user_role=user_role
            ).inc()
            
        except Exception as e:
            status_code = "500"
            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code,
                user_role=user_role
            ).inc()
            raise
            
        finally:
            # Record duration
            duration = time.time() - start_time
            REQUEST_DURATION.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code
            ).observe(duration)
            
            # Record request size
            if request_size > 0:
                REQUEST_SIZE.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(request_size)
            
            # Record response size
            if hasattr(response, "headers") and "content-length" in response.headers:
                response_size = int(response.headers["content-length"])
                RESPONSE_SIZE.labels(
                    method=method,
                    endpoint=endpoint,
                    status_code=status_code
                ).observe(response_size)
            
            # Decrement active connections
            ACTIVE_CONNECTIONS.dec()
        
        return response
    
    def _get_endpoint_pattern(self, request: Request) -> str:
        """Extract endpoint pattern from request."""
        path = request.url.path
        
        # Remove query parameters
        if "?" in path:
            path = path.split("?")[0]
        
        # Generalize path parameters (simple heuristic)
        path_parts = path.split("/")
        for i, part in enumerate(path_parts):
            # Replace UUIDs and numeric IDs with placeholders
            if len(part) == 36 and part.count("-") == 4:  # UUID
                path_parts[i] = "{uuid}"
            elif part.isdigit():  # Numeric ID
                path_parts[i] = "{id}"
        
        return "/".join(path_parts)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request logging with PHI scrubbing."""
    
    def __init__(self, app, scrub_values: bool = True):
        super().__init__(app)
        self.scrub_values = scrub_values
        self.logger = logging.getLogger("api.requests")
        
        # Fields that should be scrubbed
        self.sensitive_fields = {
            "password", "token", "secret", "key", "ssn", "social_security",
            "date_of_birth", "dob", "phone", "email", "address", "zip",
            "medical_record_number", "mrn", "patient_id"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Generate request ID
        import uuid
        request_id = str(uuid.uuid4())
        request_id_var.set(request_id)
        
        # Log request start
        self._log_request_start(request, request_id)
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Log successful response
            self._log_request_end(request, response, duration, request_id)
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            self._log_request_error(request, e, duration, request_id)
            raise
    
    def _log_request_start(self, request: Request, request_id: str):
        """Log request start."""
        self.logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query_params": self._scrub_dict(dict(request.query_params)) if self.scrub_values else dict(request.query_params),
                "user_agent": request.headers.get("user-agent"),
                "client_ip": request.client.host if request.client else None
            }
        )
    
    def _log_request_end(self, request: Request, response: Response, duration: float, request_id: str):
        """Log successful request completion."""
        self.logger.info(
            "Request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration * 1000, 2),
                "response_size": response.headers.get("content-length")
            }
        )
    
    def _log_request_error(self, request: Request, error: Exception, duration: float, request_id: str):
        """Log request error."""
        self.logger.error(
            "Request failed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "duration_ms": round(duration * 1000, 2),
                "error": str(error),
                "error_type": type(error).__name__
            }
        )
    
    def _scrub_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Scrub sensitive values from dictionary."""
        if not self.scrub_values:
            return data
        
        scrubbed = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in self.sensitive_fields):
                scrubbed[key] = "[REDACTED]"
            elif isinstance(value, dict):
                scrubbed[key] = self._scrub_dict(value)
            elif isinstance(value, list):
                scrubbed[key] = [self._scrub_dict(item) if isinstance(item, dict) else "[REDACTED]" if any(sensitive in str(item).lower() for sensitive in self.sensitive_fields) else item for item in value]
            else:
                scrubbed[key] = value
        
        return scrubbed


def setup_telemetry(app: FastAPI, enable_metrics: bool = True, scrub_logs: bool = True):
    """Set up telemetry middleware and endpoints."""
    
    if enable_metrics:
        # Add metrics middleware
        app.add_middleware(MetricsMiddleware)
        
        # Add metrics endpoint
        @app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint."""
            return StarletteResponse(
                generate_latest(),
                media_type=CONTENT_TYPE_LATEST
            )
    
    # Add logging middleware
    app.add_middleware(LoggingMiddleware, scrub_values=scrub_logs)


def record_data_processing_time(operation_type: str, data_source: str, duration: float):
    """Record data processing time metric."""
    DATA_PROCESSING_TIME.labels(
        operation_type=operation_type,
        data_source=data_source
    ).observe(duration)


def record_phi_redaction(redaction_type: str, field_type: str):
    """Record PHI redaction metric."""
    PHI_REDACTION_COUNT.labels(
        redaction_type=redaction_type,
        field_type=field_type
    ).inc()


def record_validation_error(validation_type: str, error_type: str):
    """Record validation error metric."""
    VALIDATION_ERRORS.labels(
        validation_type=validation_type,
        error_type=error_type
    ).inc()


def update_db_connections(count: int):
    """Update database connections gauge."""
    DB_CONNECTIONS.set(count)