"""Pydantic models for the Clinical Data Platform API."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict


class ModelStageEnum(str, Enum):
    """Model stage enumeration for API responses."""
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class ValidationStatusEnum(str, Enum):
    """Validation status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    EXPIRED = "expired"


class SafetyRiskCategoryEnum(str, Enum):
    """Safety risk category enumeration."""
    LOW_RISK = "LOW_RISK"
    MODERATE_RISK = "MODERATE_RISK"
    HIGH_RISK = "HIGH_RISK"
    UNKNOWN = "UNKNOWN"


class ParticipationQualityEnum(str, Enum):
    """Data participation quality enumeration."""
    HIGH_QUALITY = "HIGH_QUALITY"
    MEDIUM_QUALITY = "MEDIUM_QUALITY"
    LOW_QUALITY = "LOW_QUALITY"


# Base response models
class APIResponse(BaseModel):
    """Base API response model."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    success: bool
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(APIResponse):
    """Error response model."""
    success: bool = False
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# Data ingestion models
class IngestionRequest(BaseModel):
    """Data ingestion request model."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    data_dir: str = Field(..., description="Path to SDTM data directory")
    validate_data: bool = Field(default=True, description="Run validation after ingestion")
    force_reload: bool = Field(default=False, description="Force reload existing data")


class IngestionResponse(APIResponse):
    """Data ingestion response model."""
    files_processed: List[str] = Field(default_factory=list)
    validation_results: Optional[Dict[str, Any]] = None
    processing_time_seconds: Optional[float] = None


# Validation models
class ValidationRequest(BaseModel):
    """Data validation request model."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    data_dir: str = Field(..., description="Path to data directory")
    domains: Optional[List[str]] = Field(default=None, description="Specific domains to validate")
    validation_type: str = Field(default="both", description="Validation type: pandera, ge, or both")


class ValidationResult(BaseModel):
    """Single validation result."""
    domain: str
    validation_type: str
    status: str
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    record_count: Optional[int] = None


class ValidationResponse(APIResponse):
    """Data validation response model."""
    results: List[ValidationResult] = Field(default_factory=list)
    overall_status: str
    total_errors: int = 0
    total_warnings: int = 0


# Analytics models
class SubjectOutcome(BaseModel):
    """Subject outcome model."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    subject_key: str
    studyid: str
    subjid: str
    treatment_arm: str
    sex: str
    sex_desc: str
    age: int
    age_group: str
    
    # Safety metrics
    total_adverse_events: int = 0
    serious_adverse_events: int = 0
    severe_adverse_events: int = 0
    ongoing_adverse_events: int = 0
    has_serious_adverse_event: bool = False
    max_ae_severity_rank: int = 0
    safety_risk_category: SafetyRiskCategoryEnum
    
    # Laboratory metrics
    total_lab_tests: int = 0
    unique_lab_tests: int = 0
    abnormal_lab_results: int = 0
    abnormal_lab_rate: float = 0.0
    lab_categories_tested: int = 0
    
    # Vital signs metrics
    total_vital_measurements: int = 0
    unique_vital_tests: int = 0
    abnormal_vitals: int = 0
    abnormal_vital_rate: float = 0.0
    
    # Treatment exposure
    total_exposures: int = 0
    max_treatment_duration: int = 0
    ongoing_treatments: int = 0
    treatment_categories: Optional[str] = None
    
    # Data quality
    has_ae_data: bool = False
    has_lab_data: bool = False
    has_vital_data: bool = False
    has_exposure_data: bool = False
    data_completeness_score: float = 0.0
    
    # Clinical profile
    clinical_profile: str
    participation_quality: ParticipationQualityEnum


class StudyOverview(BaseModel):
    """Study overview model."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    study_id: str
    study_name: str
    
    # Enrollment metrics
    total_subjects: int
    treatment_arms_count: int
    
    # Demographics
    male_subjects: int = 0
    female_subjects: int = 0
    pediatric_subjects: int = 0
    adult_subjects: int = 0
    elderly_subjects: int = 0
    
    # Age statistics
    mean_age: Optional[float] = None
    median_age: Optional[float] = None
    min_age: Optional[int] = None
    max_age: Optional[int] = None
    
    # Safety metrics
    total_adverse_events_study: int = 0
    total_serious_aes_study: int = 0
    subjects_with_serious_aes: int = 0
    serious_ae_rate_percent: float = 0.0
    safety_concern_subjects: int = 0
    safety_concern_rate_percent: float = 0.0
    
    # Data quality
    avg_data_completeness: float = 0.0
    avg_data_completeness_percent: float = 0.0
    high_quality_subjects: int = 0
    medium_quality_subjects: int = 0
    low_quality_subjects: int = 0
    high_quality_rate_percent: float = 0.0
    
    # Clinical profiles
    normal_profile_subjects: int = 0
    multiple_abnormalities_subjects: int = 0
    
    # Status assessments
    study_status: str = "UNKNOWN"
    overall_safety_profile: str = "UNKNOWN"
    data_quality_assessment: str = "UNKNOWN"


class AnalyticsResponse(APIResponse):
    """Analytics response model."""
    study_overview: Optional[StudyOverview] = None
    subject_outcomes: List[SubjectOutcome] = Field(default_factory=list)
    summary_statistics: Optional[Dict[str, Any]] = None


# ML models
class ModelMetadata(BaseModel):
    """Model metadata model."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    name: str
    version: str
    stage: ModelStageEnum
    status: str
    description: str
    tags: Dict[str, str] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)
    created_timestamp: str
    last_updated_timestamp: str
    
    is_production_ready: bool = False
    performance_gate_passed: bool = False
    validation_approved: bool = False


class TrainingRequest(BaseModel):
    """Model training request."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    data_dir: Optional[str] = Field(default=None, description="Path to training data")
    model_name: str = Field(default="cdp_logreg", description="Model name for registry")
    experiment_name: str = Field(default="clinical_ml", description="MLflow experiment name")
    cv_folds: int = Field(default=5, description="Cross-validation folds")
    test_size: float = Field(default=0.2, description="Test set proportion")
    random_state: int = Field(default=42, description="Random seed")
    

class TrainingResponse(APIResponse):
    """Model training response."""
    run_id: Optional[str] = None
    model_uri: Optional[str] = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    artifacts: List[str] = Field(default_factory=list)
    training_time_seconds: Optional[float] = None


class PredictionRequest(BaseModel):
    """Batch prediction request."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    data_dir: Optional[str] = Field(default=None, description="Path to input data")
    model_uri: Optional[str] = Field(default=None, description="Model URI or 'models:/name/stage'")
    output_path: str = Field(default="predictions.parquet", description="Output file path")


class PredictionResponse(APIResponse):
    """Batch prediction response."""
    predictions_count: int = 0
    output_path: str
    prediction_summary: Optional[Dict[str, Any]] = None


class ModelGovernanceRequest(BaseModel):
    """Model governance request."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    model_name: str
    version: str
    clinical_use: str
    validation_status: ValidationStatusEnum
    regulatory_approval: str
    data_version: str
    performance_gate: str


class ModelPromotionRequest(BaseModel):
    """Model promotion request."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    model_name: str
    version: str
    target_stage: ModelStageEnum
    force: bool = Field(default=False, description="Force promotion without governance checks")


class GovernanceReport(BaseModel):
    """Model governance report."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    generated_at: str
    total_models: int
    governance_summary: Dict[str, int]
    model_details: List[Dict[str, Any]]


class ModelRegistryResponse(APIResponse):
    """Model registry response."""
    models: List[ModelMetadata] = Field(default_factory=list)
    governance_report: Optional[GovernanceReport] = None


# Health check models
class HealthCheck(BaseModel):
    """Health check response model."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    status: str
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str
    environment: str
    services: Dict[str, str] = Field(default_factory=dict)
    
    # Service health
    database_status: str = "unknown"
    mlflow_status: str = "unknown"
    storage_status: str = "unknown"
    
    # System metrics
    uptime_seconds: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None


# Query parameters
class QueryParams(BaseModel):
    """Base query parameters."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    limit: Optional[int] = Field(default=100, ge=1, le=1000)
    offset: Optional[int] = Field(default=0, ge=0)
    sort_by: Optional[str] = None
    order: Optional[str] = Field(default="asc", pattern="^(asc|desc)$")


class SubjectQueryParams(QueryParams):
    """Subject query parameters."""
    treatment_arm: Optional[str] = None
    safety_risk_category: Optional[SafetyRiskCategoryEnum] = None
    age_group: Optional[str] = None
    participation_quality: Optional[ParticipationQualityEnum] = None
    has_serious_ae: Optional[bool] = None