from __future__ import annotations

import re
from typing import List, Optional

import duckdb
from fastapi import FastAPI, HTTPException, Depends, status, Path
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, validator

from clinical_platform.config import get_config
from clinical_platform.logging_utils import get_logger, PHIFilter
from .middleware import setup_middleware

logger = get_logger("api")
security = HTTPBearer(auto_error=False)

app = FastAPI(
    title="Clinical Data Platform API",
    description="Secure API for clinical data operations with HIPAA compliance",
    version="1.0.0",
    openapi_tags=[
        {"name": "health", "description": "Health check endpoints"},
        {"name": "studies", "description": "Study management"},
        {"name": "subjects", "description": "Subject data operations"},
        {"name": "ml", "description": "Machine learning predictions"},
    ]
)

# Setup all security middleware
setup_middleware(app)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["localhost", "127.0.0.1", "*.clinical-platform.com"]
)


def safe_http_exception(status_code: int, detail: str, original_error: Exception = None) -> HTTPException:
    """Create HTTP exception with PHI-safe error messages."""
    phi_filter = PHIFilter()
    safe_detail = phi_filter._redact_phi_from_message(detail)
    
    # Log original error for debugging (will be redacted by logging system)
    if original_error:
        logger.error(f"API error occurred: {original_error}")
    
    # Return sanitized message to client
    return HTTPException(status_code=status_code, detail=safe_detail)


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify API key from Authorization header - REQUIRED for all protected endpoints."""
    config = get_config()
    expected_key = config.security.api_key
    
    # API key is MANDATORY - no bypass allowed
    if not expected_key:
        logger.error("API key not configured on server")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server misconfigured: API key missing",
        )
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if credentials.credentials != expected_key.get_secret_value():
        logger.warning("Invalid API key attempt", extra={"ip": "redacted"})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return credentials.credentials


class Health(BaseModel):
    status: str


class ScoreRequest(BaseModel):
    """Request model for ML scoring endpoint with validation."""
    AGE: float = Field(..., ge=0, le=120, description="Patient age in years")
    AE_COUNT: float = Field(..., ge=0, le=100, description="Total adverse event count")
    SEVERE_AE_COUNT: float = Field(..., ge=0, le=50, description="Severe adverse event count")
    
    @validator('SEVERE_AE_COUNT')
    def severe_count_must_be_less_than_total(cls, v, values):
        if 'AE_COUNT' in values and v > values['AE_COUNT']:
            raise ValueError('Severe AE count cannot exceed total AE count')
        return v


class ScoreResponse(BaseModel):
    """Response model for ML scoring with metadata."""
    risk: float = Field(..., ge=0.0, le=1.0, description="Risk probability [0-1]")
    model_version: str = Field(default="1.0.0", description="Model version used")
    confidence: Optional[str] = Field(default=None, description="Prediction confidence level")


class SubjectResponse(BaseModel):
    """Response model for subject data with PII redaction."""
    subject_id: str
    study_id: str
    arm: Optional[str] = None
    # Note: PII fields like age, sex are redacted in response
    enrollment_status: Optional[str] = None


@app.get("/health", response_model=Health, tags=["health"])
def health():
    """Health check endpoint - no authentication required."""
    return Health(status="ok")


@app.get("/studies", response_model=List[str], tags=["studies"])
def list_studies(api_key: str = Depends(verify_api_key)):
    """List available studies - requires authentication."""
    try:
        config = get_config()
        con = duckdb.connect(config.warehouse.duckdb_path)
        studies = [r[0] for r in con.execute("SELECT DISTINCT study_id FROM stg_subjects").fetchall()]
        logger.info("Studies listed", extra={"study_count": len(studies)})
        return studies
    except Exception as e:
        raise safe_http_exception(500, "Database error occurred", e)


@app.get("/subjects/{subject_id}", response_model=SubjectResponse, tags=["subjects"])
def get_subject(
    subject_id: str = Path(..., regex="^[A-Za-z0-9_-]+$", description="Subject ID (alphanumeric, underscore, hyphen only)"), 
    api_key: str = Depends(verify_api_key)
):
    """Get subject information with PII redaction - requires authentication."""
    # Additional validation beyond Path regex
    if not subject_id.strip() or len(subject_id) > 50:
        raise safe_http_exception(400, "Invalid subject ID format", None)
    
    try:
        config = get_config()
        con = duckdb.connect(config.warehouse.duckdb_path)
        # Parameterized query - already safe
        df = con.execute(
            "SELECT subject_id, study_id, arm FROM stg_subjects WHERE subject_id = ?", 
            [subject_id]
        ).fetch_df()
        
        if df.empty:
            logger.warning("Subject not found", extra={"subject_id": "***REDACTED***"})
            raise safe_http_exception(404, "Subject not found", None)
        
        result = df.to_dict(orient="records")[0]
        logger.info("Subject retrieved", extra={"subject_id": "***REDACTED***"})
        return SubjectResponse(**result, enrollment_status="active")
        
    except HTTPException:
        raise
    except Exception as e:
        raise safe_http_exception(500, "Database error occurred", e)


@app.post("/score", response_model=ScoreResponse, tags=["ml"])
def score(req: ScoreRequest, api_key: str = Depends(verify_api_key)) -> ScoreResponse:
    """Generate ML risk score - requires authentication."""
    try:
        # Simple logistic function as a stand-in for a model; deterministic and safe for demo
        z = 0.02 * req.AGE + 0.3 * req.AE_COUNT + 0.6 * req.SEVERE_AE_COUNT - 2.0
        import math
        
        risk = 1 / (1 + math.exp(-z))
        confidence = "high" if abs(z) > 1.0 else "medium" if abs(z) > 0.5 else "low"
        
        logger.info("Score generated", extra={
            "risk": risk, 
            "confidence": confidence,
            "age": "***REDACTED***"  # PII redaction
        })
        
        return ScoreResponse(
            risk=float(risk), 
            model_version="1.0.0", 
            confidence=confidence
        )
        
    except Exception as e:
        raise safe_http_exception(500, "Model computation error", e)

