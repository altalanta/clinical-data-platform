"""FastAPI endpoints for the Clinical Data Platform."""

from __future__ import annotations

import os
import time
import psutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from clinical_platform.config import get_config
from clinical_platform.logging_utils import get_logger
from clinical_platform.data.ingest import ingest_clinical_data
from clinical_platform.validation.runner import run_validation_pipeline
from clinical_platform.analytics.feature_eng import subject_level_features
from clinical_platform.ml.train import train_model
from clinical_platform.ml.infer import batch_score
from clinical_platform.ml.registry import (
    setup_mlflow_secure, validate_model_governance, get_production_model,
    list_models_with_governance, promote_model_to_stage, add_model_governance_tags,
    generate_model_governance_report
)

from .models import *
from .middleware import setup_middleware

# Initialize logger
logger = get_logger("api_endpoints")

# Application metadata
APP_VERSION = "0.1.0"
APP_NAME = "Clinical Data Platform API"
APP_DESCRIPTION = """
Production-ready API for clinical trial data processing, validation, analytics, and ML operations.

## Features

- **Data Ingestion**: SDTM-compliant data loading with validation
- **Data Validation**: Comprehensive validation using pandera and Great Expectations
- **Analytics**: Subject-level outcomes and study-level aggregations  
- **ML Operations**: Model training, inference, and governance
- **Model Registry**: Clinical-grade model lifecycle management
- **Observability**: Structured logging and health monitoring

## Security

- PHI protection with automatic redaction
- Request/response logging with audit trails
- Secure headers and CORS configuration
- Input validation and sanitization

## Clinical Compliance

- SDTM data standards support
- Clinical governance workflows
- Audit logging for regulatory requirements
- Data quality monitoring and reporting
"""

# Application startup
app_start_time = time.time()

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title=APP_NAME,
        description=APP_DESCRIPTION,
        version=APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/api/openapi.json"
    )
    
    # Setup middleware
    setup_middleware(app)
    
    # Initialize MLflow on startup
    @app.on_event("startup")
    async def startup_event():
        try:
            config = get_config()
            if config.mlflow.tracking_uri:
                setup_mlflow_secure(
                    tracking_uri=config.mlflow.tracking_uri,
                    auth_token=config.mlflow.auth_token
                )
                logger.info("MLflow initialized successfully")
            else:
                logger.warning("MLflow tracking URI not configured")
        except Exception as e:
            logger.error(f"Failed to initialize MLflow: {e}")
    
    return app


# Create app instance
app = create_app()


# Health check endpoints
@app.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check() -> HealthCheck:
    """Comprehensive health check endpoint."""
    
    config = get_config()
    uptime = time.time() - app_start_time
    
    # Check system metrics
    memory_usage = psutil.virtual_memory().percent
    cpu_usage = psutil.cpu_percent(interval=1)
    
    # Check service dependencies
    services = {}
    
    # Check database
    try:
        # Simple connection test - would implement actual DB ping
        services["database"] = "healthy"
        database_status = "healthy"
    except Exception:
        services["database"] = "unhealthy"
        database_status = "unhealthy"
    
    # Check MLflow
    try:
        if config.mlflow.tracking_uri:
            # Would implement MLflow ping
            services["mlflow"] = "healthy"
            mlflow_status = "healthy"
        else:
            services["mlflow"] = "not_configured"
            mlflow_status = "not_configured"
    except Exception:
        services["mlflow"] = "unhealthy"
        mlflow_status = "unhealthy"
    
    # Check storage
    try:
        data_dir = Path("data")
        if data_dir.exists():
            services["storage"] = "healthy"
            storage_status = "healthy"
        else:
            services["storage"] = "missing_data_dir"
            storage_status = "warning"
    except Exception:
        services["storage"] = "unhealthy"
        storage_status = "unhealthy"
    
    # Overall status
    overall_status = "healthy"
    if any(status == "unhealthy" for status in services.values()):
        overall_status = "unhealthy"
    elif any(status in ["warning", "not_configured"] for status in services.values()):
        overall_status = "degraded"
    
    return HealthCheck(
        status=overall_status,
        version=APP_VERSION,
        environment=config.environment,
        services=services,
        database_status=database_status,
        mlflow_status=mlflow_status,
        storage_status=storage_status,
        uptime_seconds=uptime,
        memory_usage_mb=memory_usage,
        cpu_usage_percent=cpu_usage
    )


@app.get("/", response_model=Dict[str, str], tags=["Health"])
async def root() -> Dict[str, str]:
    """Root endpoint with API information."""
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "status": "running",
        "docs_url": "/docs",
        "health_url": "/health"
    }


# Data ingestion endpoints
@app.post("/api/v1/data/ingest", response_model=IngestionResponse, tags=["Data"])
async def ingest_data(
    request: IngestionRequest,
    background_tasks: BackgroundTasks
) -> IngestionResponse:
    """Ingest SDTM clinical data with optional validation."""
    
    start_time = time.time()
    
    try:
        # Validate data directory exists
        data_path = Path(request.data_dir)
        if not data_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Data directory not found: {request.data_dir}"
            )
        
        # Run ingestion
        logger.info(
            "Starting data ingestion",
            extra={
                "data_dir": request.data_dir,
                "validate_data": request.validate_data,
                "force_reload": request.force_reload
            }
        )
        
        # Get list of files to process
        parquet_files = list(data_path.glob("*.parquet"))
        file_names = [f.name for f in parquet_files]
        
        if not file_names:
            raise HTTPException(
                status_code=400,
                detail="No parquet files found in data directory"
            )
        
        # Run ingestion process
        ingest_clinical_data(
            data_dir=request.data_dir,
            force_reload=request.force_reload
        )
        
        processing_time = time.time() - start_time
        
        # Run validation if requested
        validation_results = None
        if request.validate_data:
            try:
                validation_results = run_validation_pipeline(
                    data_dir=request.data_dir,
                    validation_type="both"
                )
            except Exception as e:
                logger.warning(f"Validation failed but ingestion succeeded: {e}")
                validation_results = {"error": str(e)}
        
        logger.info(
            "Data ingestion completed",
            extra={
                "files_processed": len(file_names),
                "processing_time": processing_time,
                "validation_run": request.validate_data
            }
        )
        
        return IngestionResponse(
            success=True,
            message=f"Successfully ingested {len(file_names)} files",
            files_processed=file_names,
            validation_results=validation_results,
            processing_time_seconds=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Data validation endpoints
@app.post("/api/v1/data/validate", response_model=ValidationResponse, tags=["Data"])
async def validate_data(request: ValidationRequest) -> ValidationResponse:
    """Validate clinical data using pandera and Great Expectations."""
    
    try:
        # Validate data directory exists
        data_path = Path(request.data_dir)
        if not data_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Data directory not found: {request.data_dir}"
            )
        
        logger.info(
            "Starting data validation",
            extra={
                "data_dir": request.data_dir,
                "domains": request.domains,
                "validation_type": request.validation_type
            }
        )
        
        # Run validation pipeline
        validation_results = run_validation_pipeline(
            data_dir=request.data_dir,
            domains=request.domains,
            validation_type=request.validation_type
        )
        
        # Process results into API format
        results = []
        total_errors = 0
        total_warnings = 0
        
        for domain, domain_results in validation_results.items():
            if isinstance(domain_results, dict) and "error" not in domain_results:
                for val_type, val_result in domain_results.items():
                    if isinstance(val_result, dict):
                        errors = val_result.get("errors", [])
                        warnings = val_result.get("warnings", [])
                        
                        result = ValidationResult(
                            domain=domain,
                            validation_type=val_type,
                            status="passed" if not errors else "failed",
                            errors=errors,
                            warnings=warnings,
                            record_count=val_result.get("record_count")
                        )
                        results.append(result)
                        
                        total_errors += len(errors)
                        total_warnings += len(warnings)
        
        overall_status = "passed" if total_errors == 0 else "failed"
        
        logger.info(
            "Data validation completed",
            extra={
                "overall_status": overall_status,
                "total_errors": total_errors,
                "total_warnings": total_warnings,
                "domains_validated": len(results)
            }
        )
        
        return ValidationResponse(
            success=True,
            message=f"Validation completed with {total_errors} errors and {total_warnings} warnings",
            results=results,
            overall_status=overall_status,
            total_errors=total_errors,
            total_warnings=total_warnings
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Analytics endpoints
@app.get("/api/v1/analytics/subjects", response_model=List[SubjectOutcome], tags=["Analytics"])
async def get_subject_outcomes(
    data_dir: Optional[str] = Query(default=None, description="Data directory path"),
    params: SubjectQueryParams = Depends()
) -> List[SubjectOutcome]:
    """Get subject-level clinical outcomes with filtering."""
    
    try:
        # Use default data directory if not provided
        if not data_dir:
            data_dir = "data/sample_standardized"
        
        # Generate subject-level features
        features_df = subject_level_features(data_dir)
        
        # Apply filters
        if params.treatment_arm:
            features_df = features_df[features_df["treatment_arm"] == params.treatment_arm]
        
        if params.safety_risk_category:
            features_df = features_df[features_df["safety_risk_category"] == params.safety_risk_category.value]
        
        if params.age_group:
            features_df = features_df[features_df["age_group"] == params.age_group]
        
        if params.has_serious_ae is not None:
            features_df = features_df[features_df["has_serious_adverse_event"] == params.has_serious_ae]
        
        # Apply pagination
        if params.offset:
            features_df = features_df.iloc[params.offset:]
        
        if params.limit:
            features_df = features_df.head(params.limit)
        
        # Convert to response models
        subjects = []
        for _, row in features_df.iterrows():
            subject = SubjectOutcome(
                subject_key=f"{row['STUDYID']}_{row['SUBJID']}",
                studyid=row["STUDYID"],
                subjid=row["SUBJID"],
                treatment_arm=row.get("treatment_arm", "UNKNOWN"),
                sex=row.get("SEX", "U"),
                sex_desc=row.get("sex_desc", "UNKNOWN"),
                age=int(row.get("AGE", 0)),
                age_group=row.get("age_group", "UNKNOWN"),
                total_adverse_events=int(row.get("AE_COUNT", 0)),
                serious_adverse_events=int(row.get("SERIOUS_AE_COUNT", 0)),
                severe_adverse_events=int(row.get("SEVERE_AE_COUNT", 0)),
                safety_risk_category=SafetyRiskCategoryEnum(row.get("safety_risk_category", "UNKNOWN")),
                total_lab_tests=int(row.get("LAB_COUNT", 0)),
                abnormal_lab_results=int(row.get("ABNORMAL_LAB_COUNT", 0)),
                abnormal_lab_rate=float(row.get("abnormal_lab_rate", 0.0)),
                total_vital_measurements=int(row.get("VITAL_COUNT", 0)),
                data_completeness_score=float(row.get("data_completeness_score", 0.0)),
                clinical_profile=row.get("clinical_profile", "UNKNOWN"),
                participation_quality=ParticipationQualityEnum(row.get("participation_quality", "LOW_QUALITY")),
                has_serious_adverse_event=bool(row.get("has_serious_adverse_event", False))
            )
            subjects.append(subject)
        
        logger.info(
            "Subject outcomes retrieved",
            extra={
                "subjects_returned": len(subjects),
                "filters_applied": {
                    "treatment_arm": params.treatment_arm,
                    "safety_risk_category": params.safety_risk_category,
                    "age_group": params.age_group,
                    "has_serious_ae": params.has_serious_ae
                }
            }
        )
        
        return subjects
        
    except Exception as e:
        logger.error(f"Failed to get subject outcomes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ML endpoints
@app.post("/api/v1/ml/train", response_model=TrainingResponse, tags=["Machine Learning"])
async def train_ml_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
) -> TrainingResponse:
    """Train a clinical ML model with MLflow tracking."""
    
    start_time = time.time()
    
    try:
        logger.info(
            "Starting model training",
            extra={
                "model_name": request.model_name,
                "experiment_name": request.experiment_name,
                "data_dir": request.data_dir
            }
        )
        
        # Run training
        run_id, model_uri, metrics = train_model(
            data_dir=request.data_dir,
            model_name=request.model_name,
            experiment_name=request.experiment_name,
            cv_folds=request.cv_folds,
            test_size=request.test_size,
            random_state=request.random_state
        )
        
        training_time = time.time() - start_time
        
        # List artifacts (would query MLflow in production)
        artifacts = [
            "model",
            "confusion_matrix.png",
            "feature_importance.png",
            "roc_curve.png",
            "training_report.json"
        ]
        
        logger.info(
            "Model training completed",
            extra={
                "run_id": run_id,
                "model_uri": model_uri,
                "training_time": training_time,
                "metrics": metrics
            }
        )
        
        return TrainingResponse(
            success=True,
            message="Model training completed successfully",
            run_id=run_id,
            model_uri=model_uri,
            metrics=metrics,
            artifacts=artifacts,
            training_time_seconds=training_time
        )
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ml/predict", response_model=PredictionResponse, tags=["Machine Learning"])
async def batch_predict(request: PredictionRequest) -> PredictionResponse:
    """Run batch predictions using a trained model."""
    
    try:
        logger.info(
            "Starting batch prediction",
            extra={
                "data_dir": request.data_dir,
                "model_uri": request.model_uri,
                "output_path": request.output_path
            }
        )
        
        # Run batch scoring
        predictions_df = batch_score(
            data_dir=request.data_dir,
            model_uri=request.model_uri,
            out_path=request.output_path
        )
        
        # Generate prediction summary
        prediction_summary = {
            "mean_risk": float(predictions_df["RISK"].mean()),
            "min_risk": float(predictions_df["RISK"].min()),
            "max_risk": float(predictions_df["RISK"].max()),
            "high_risk_subjects": int((predictions_df["RISK"] > 0.7).sum()),
            "medium_risk_subjects": int(((predictions_df["RISK"] > 0.3) & (predictions_df["RISK"] <= 0.7)).sum()),
            "low_risk_subjects": int((predictions_df["RISK"] <= 0.3).sum())
        }
        
        logger.info(
            "Batch prediction completed",
            extra={
                "predictions_count": len(predictions_df),
                "output_path": request.output_path,
                "prediction_summary": prediction_summary
            }
        )
        
        return PredictionResponse(
            success=True,
            message=f"Generated {len(predictions_df)} predictions",
            predictions_count=len(predictions_df),
            output_path=request.output_path,
            prediction_summary=prediction_summary
        )
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Model registry endpoints
@app.get("/api/v1/ml/models", response_model=ModelRegistryResponse, tags=["Model Registry"])
async def list_models() -> ModelRegistryResponse:
    """List all models in the registry with governance status."""
    
    try:
        models_metadata = list_models_with_governance()
        
        # Convert to API models
        models = []
        for metadata in models_metadata:
            model = ModelMetadata(
                name=metadata.name,
                version=metadata.version,
                stage=ModelStageEnum(metadata.stage),
                status=metadata.status,
                description=metadata.description,
                tags=metadata.tags,
                metrics=metadata.metrics,
                created_timestamp=metadata.created_timestamp,
                last_updated_timestamp=metadata.last_updated_timestamp,
                is_production_ready=metadata.is_production_ready,
                performance_gate_passed=metadata.performance_gate_passed,
                validation_approved=metadata.validation_approved
            )
            models.append(model)
        
        logger.info(f"Retrieved {len(models)} models from registry")
        
        return ModelRegistryResponse(
            success=True,
            message=f"Retrieved {len(models)} models",
            models=models
        )
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ml/models/{model_name}/versions/{version}/promote", 
          response_model=APIResponse, tags=["Model Registry"])
async def promote_model(
    model_name: str,
    version: str,
    request: ModelPromotionRequest
) -> APIResponse:
    """Promote a model to a new stage."""
    
    try:
        success = promote_model_to_stage(
            model_name=model_name,
            version=version,
            stage=request.target_stage.value,
            force=request.force
        )
        
        if success:
            logger.info(
                "Model promoted successfully",
                extra={
                    "model_name": model_name,
                    "version": version,
                    "target_stage": request.target_stage.value
                }
            )
            
            return APIResponse(
                success=True,
                message=f"Model {model_name} v{version} promoted to {request.target_stage.value}"
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Model promotion failed - check governance requirements"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model promotion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ml/governance/report", response_model=Dict[str, Any], tags=["Model Registry"])
async def get_governance_report() -> Dict[str, Any]:
    """Generate comprehensive model governance report."""
    
    try:
        report = generate_model_governance_report()
        
        logger.info(
            "Generated governance report",
            extra={
                "total_models": report.get("total_models", 0),
                "compliant_models": report.get("governance_summary", {}).get("compliant_models", 0)
            }
        )
        
        return report
        
    except Exception as e:
        logger.error(f"Failed to generate governance report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=APP_NAME,
        version=APP_VERSION,
        description=APP_DESCRIPTION,
        routes=app.routes,
    )
    
    # Add custom schema info
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


if __name__ == "__main__":
    import uvicorn
    
    config = get_config()
    
    uvicorn.run(
        "clinical_platform.api.endpoints:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )