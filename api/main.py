"""
Main FastAPI application with RBAC, monitoring, and cloud-ready configuration.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .telemetry import setup_telemetry
from .auth import auth_service, UserLogin, Token
from .routers import health, data

# Initialize FastAPI app
app = FastAPI(
    title="Clinical Data Platform API",
    description="Production-ready clinical data platform with RBAC and cloud deployment",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Get settings
settings = get_settings()

# Setup telemetry and monitoring
setup_telemetry(
    app, 
    enable_metrics=settings.prometheus_enabled,
    scrub_logs=settings.log_scrub_values
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.environment == "dev" else ["https://*.clinical-platform.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Trusted host middleware for production
if settings.environment in ["staging", "prod"]:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*.clinical-platform.com", "*.amazonaws.com"]
    )

# Include routers
app.include_router(health.router)
app.include_router(data.router, prefix=settings.api_prefix)

# Authentication endpoints
@app.post("/auth/login", response_model=Token, tags=["authentication"])
async def login(credentials: UserLogin):
    """Login and receive JWT token."""
    return auth_service.login(credentials)

@app.post("/auth/logout", tags=["authentication"])
async def logout():
    """Logout (client should discard token)."""
    return {"message": "Logout successful"}

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic info."""
    return {
        "name": "Clinical Data Platform API",
        "version": "1.0.0",
        "environment": settings.environment,
        "status": "operational",
        "docs": "/docs"
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler with PHI scrubbing."""
    # In production, log the full error but return sanitized response
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "request_id": getattr(request.state, "request_id", "unknown")
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )