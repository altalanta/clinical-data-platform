from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from enum import Enum
from dataclasses import dataclass
from pydantic import SecretStr

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion

from clinical_platform.logging_utils import get_logger

logger = get_logger("ml_registry")


class ModelStage(Enum):
    """Enumeration of model registry stages."""
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class ValidationStatus(Enum):
    """Enumeration of model validation statuses."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class ModelMetadata:
    """Model metadata for clinical governance."""
    name: str
    version: str
    stage: str
    status: str
    description: str
    tags: Dict[str, str]
    metrics: Dict[str, float]
    created_timestamp: str
    last_updated_timestamp: str
    
    @property
    def is_production_ready(self) -> bool:
        """Check if model meets production readiness criteria."""
        required_tags = ["clinical_use", "validation_status", "performance_gate"]
        return all(tag in self.tags for tag in required_tags)
    
    @property
    def performance_gate_passed(self) -> bool:
        """Check if model passed performance gates."""
        return self.tags.get("performance_gate") == "passed"
    
    @property
    def validation_approved(self) -> bool:
        """Check if model passed clinical validation."""
        return self.tags.get("validation_status") == "passed"


def setup_mlflow_secure(tracking_uri: str, auth_token: Optional[SecretStr] = None) -> None:
    """Setup MLflow with secure authentication for clinical compliance."""
    
    if not tracking_uri:
        raise ValueError("MLflow tracking URI is required for clinical deployments")
    
    # Set tracking URI
    mlflow.set_tracking_uri(tracking_uri)
    logger.info("MLflow tracking URI configured", extra={"uri": tracking_uri})
    
    # Configure authentication if provided
    if auth_token:
        os.environ["MLFLOW_TRACKING_TOKEN"] = auth_token.get_secret_value()
        logger.info("MLflow authentication configured")
    
    # Verify connection
    try:
        client = MlflowClient(tracking_uri=tracking_uri)
        experiments = client.search_experiments()
        logger.info("MLflow connection verified", extra={"experiment_count": len(experiments)})
    except Exception as e:
        logger.error("Failed to connect to MLflow", extra={"error": str(e)})
        raise ConnectionError(f"Cannot connect to MLflow at {tracking_uri}: {e}")


def setup_mlflow(tracking_uri: str | None = None) -> None:
    """Legacy setup function - deprecated, use setup_mlflow_secure instead."""
    logger.warning("Using deprecated setup_mlflow function - upgrade to setup_mlflow_secure")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def validate_model_governance(model_name: str, version: str) -> dict:
    """Validate model governance requirements for clinical use."""
    client = MlflowClient()
    
    try:
        model_version = client.get_model_version(model_name, version)
        
        governance_status = {
            "model_name": model_name,
            "version": version,
            "stage": model_version.current_stage,
            "governance_checks": {}
        }
        
        # Check required tags for clinical models
        required_tags = [
            "clinical_use", 
            "validation_status", 
            "regulatory_approval", 
            "data_version",
            "performance_gate"
        ]
        
        for tag in required_tags:
            if tag in model_version.tags:
                governance_status["governance_checks"][tag] = model_version.tags[tag]
            else:
                governance_status["governance_checks"][tag] = "missing"
        
        # Overall governance status
        missing_tags = [tag for tag in required_tags if governance_status["governance_checks"].get(tag) == "missing"]
        governance_status["compliant"] = len(missing_tags) == 0
        governance_status["missing_requirements"] = missing_tags
        
        return governance_status
        
    except Exception as e:
        logger.error("Failed to validate model governance", extra={"model": model_name, "version": version, "error": str(e)})
        return {
            "model_name": model_name,
            "version": version,
            "compliant": False,
            "error": str(e)
        }


def get_production_model(model_name: str) -> Optional[str]:
    """Get the production version of a model if it exists and passes governance."""
    client = MlflowClient()
    
    try:
        # Get production versions
        production_versions = client.get_latest_versions(model_name, stages=["Production"])
        
        if not production_versions:
            logger.warning("No production model found", extra={"model_name": model_name})
            return None
        
        latest_prod = production_versions[0]
        
        # Validate governance
        governance = validate_model_governance(model_name, latest_prod.version)
        
        if not governance["compliant"]:
            logger.warning(
                "Production model fails governance checks", 
                extra={
                    "model_name": model_name, 
                    "version": latest_prod.version,
                    "missing": governance["missing_requirements"]
                }
            )
            return None
        
        logger.info(
            "Production model validated", 
            extra={"model_name": model_name, "version": latest_prod.version}
        )
        return latest_prod.version
        
    except Exception as e:
        logger.error("Failed to get production model", extra={"model_name": model_name, "error": str(e)})
        return None


def list_models_with_governance() -> List[ModelMetadata]:
    """List all models with their governance status."""
    client = MlflowClient()
    models = []
    
    try:
        registered_models = client.search_registered_models()
        
        for model in registered_models:
            latest_versions = client.get_latest_versions(model.name)
            
            for version in latest_versions:
                metadata = ModelMetadata(
                    name=model.name,
                    version=version.version,
                    stage=version.current_stage,
                    status=version.status,
                    description=version.description or "",
                    tags=version.tags,
                    metrics={},  # Would need to fetch from run
                    created_timestamp=version.creation_timestamp,
                    last_updated_timestamp=version.last_updated_timestamp
                )
                models.append(metadata)
        
        logger.info("Retrieved model registry overview", extra={"model_count": len(models)})
        return models
        
    except Exception as e:
        logger.error("Failed to list models", extra={"error": str(e)})
        return []


def promote_model_to_stage(model_name: str, version: str, stage: str, 
                          force: bool = False) -> bool:
    """Promote a model to a new stage with governance validation."""
    client = MlflowClient()
    
    try:
        # Validate stage
        if stage not in [s.value for s in ModelStage]:
            raise ValueError(f"Invalid stage: {stage}. Must be one of {[s.value for s in ModelStage]}")
        
        # For production, require governance validation unless forced
        if stage == ModelStage.PRODUCTION.value and not force:
            governance = validate_model_governance(model_name, version)
            if not governance["compliant"]:
                logger.error(
                    "Model promotion blocked - governance requirements not met",
                    extra={
                        "model_name": model_name,
                        "version": version,
                        "missing": governance["missing_requirements"]
                    }
                )
                return False
        
        # Archive existing production models if promoting to production
        if stage == ModelStage.PRODUCTION.value:
            existing_prod = client.get_latest_versions(model_name, stages=["Production"])
            for existing in existing_prod:
                client.transition_model_version_stage(
                    name=model_name,
                    version=existing.version,
                    stage=ModelStage.ARCHIVED.value
                )
                logger.info(
                    "Archived previous production model",
                    extra={"model_name": model_name, "archived_version": existing.version}
                )
        
        # Promote the model
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        
        logger.info(
            "Model promoted successfully",
            extra={"model_name": model_name, "version": version, "stage": stage}
        )
        return True
        
    except Exception as e:
        logger.error(
            "Failed to promote model",
            extra={"model_name": model_name, "version": version, "stage": stage, "error": str(e)}
        )
        return False


def add_model_governance_tags(model_name: str, version: str, 
                             clinical_use: str, validation_status: str,
                             regulatory_approval: str, data_version: str,
                             performance_gate: str) -> bool:
    """Add required governance tags to a model version."""
    client = MlflowClient()
    
    try:
        governance_tags = {
            "clinical_use": clinical_use,
            "validation_status": validation_status,
            "regulatory_approval": regulatory_approval,
            "data_version": data_version,
            "performance_gate": performance_gate,
            "governance_updated": datetime.now().isoformat(),
            "compliance_reviewer": "clinical_platform_system"
        }
        
        for key, value in governance_tags.items():
            client.set_model_version_tag(model_name, version, key, value)
        
        logger.info(
            "Governance tags added successfully",
            extra={"model_name": model_name, "version": version, "tags": list(governance_tags.keys())}
        )
        return True
        
    except Exception as e:
        logger.error(
            "Failed to add governance tags",
            extra={"model_name": model_name, "version": version, "error": str(e)}
        )
        return False


def get_model_approval_workflow_status(model_name: str, version: str) -> Dict[str, Any]:
    """Get the approval workflow status for a model version."""
    client = MlflowClient()
    
    try:
        model_version = client.get_model_version(model_name, version)
        governance = validate_model_governance(model_name, version)
        
        workflow_status = {
            "model_name": model_name,
            "version": version,
            "current_stage": model_version.current_stage,
            "governance_compliant": governance["compliant"],
            "approval_gates": {
                "performance_validated": model_version.tags.get("performance_gate") == "passed",
                "clinical_validation": model_version.tags.get("validation_status") == "passed",
                "regulatory_approved": model_version.tags.get("regulatory_approval") == "approved",
                "data_lineage_verified": "data_version" in model_version.tags
            },
            "next_available_actions": []
        }
        
        # Determine next available actions based on current state
        current_stage = model_version.current_stage
        
        if current_stage == ModelStage.NONE.value:
            if all(workflow_status["approval_gates"].values()):
                workflow_status["next_available_actions"].append("promote_to_staging")
            else:
                workflow_status["next_available_actions"].append("complete_governance_requirements")
        
        elif current_stage == ModelStage.STAGING.value:
            if governance["compliant"]:
                workflow_status["next_available_actions"].extend(["promote_to_production", "archive"])
            else:
                workflow_status["next_available_actions"].append("complete_governance_requirements")
        
        elif current_stage == ModelStage.PRODUCTION.value:
            workflow_status["next_available_actions"].extend(["archive", "demote_to_staging"])
        
        return workflow_status
        
    except Exception as e:
        logger.error(
            "Failed to get approval workflow status",
            extra={"model_name": model_name, "version": version, "error": str(e)}
        )
        return {"error": str(e)}


def validate_model_expiry(model_name: str, version: str, max_age_days: int = 90) -> Dict[str, Any]:
    """Validate if a model version has expired based on age."""
    client = MlflowClient()
    
    try:
        model_version = client.get_model_version(model_name, version)
        created_date = datetime.fromtimestamp(int(model_version.creation_timestamp) / 1000)
        current_date = datetime.now()
        age_days = (current_date - created_date).days
        
        expiry_status = {
            "model_name": model_name,
            "version": version,
            "creation_date": created_date.isoformat(),
            "age_days": age_days,
            "max_age_days": max_age_days,
            "is_expired": age_days > max_age_days,
            "days_until_expiry": max(0, max_age_days - age_days),
            "expiry_date": (created_date + timedelta(days=max_age_days)).isoformat()
        }
        
        if expiry_status["is_expired"]:
            logger.warning(
                "Model version has expired",
                extra={
                    "model_name": model_name,
                    "version": version,
                    "age_days": age_days,
                    "max_age_days": max_age_days
                }
            )
        
        return expiry_status
        
    except Exception as e:
        logger.error(
            "Failed to validate model expiry",
            extra={"model_name": model_name, "version": version, "error": str(e)}
        )
        return {"error": str(e)}


def generate_model_governance_report() -> Dict[str, Any]:
    """Generate a comprehensive governance report for all models."""
    client = MlflowClient()
    
    try:
        registered_models = client.search_registered_models()
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_models": len(registered_models),
            "governance_summary": {
                "compliant_models": 0,
                "non_compliant_models": 0,
                "production_models": 0,
                "staging_models": 0,
                "expired_models": 0
            },
            "model_details": []
        }
        
        for model in registered_models:
            latest_versions = client.get_latest_versions(model.name)
            
            for version in latest_versions:
                governance = validate_model_governance(model.name, version.version)
                expiry = validate_model_expiry(model.name, version.version)
                
                model_detail = {
                    "name": model.name,
                    "version": version.version,
                    "stage": version.current_stage,
                    "compliant": governance["compliant"],
                    "missing_requirements": governance.get("missing_requirements", []),
                    "is_expired": expiry.get("is_expired", False),
                    "age_days": expiry.get("age_days", 0)
                }
                
                report["model_details"].append(model_detail)
                
                # Update summary counts
                if governance["compliant"]:
                    report["governance_summary"]["compliant_models"] += 1
                else:
                    report["governance_summary"]["non_compliant_models"] += 1
                
                if version.current_stage == ModelStage.PRODUCTION.value:
                    report["governance_summary"]["production_models"] += 1
                elif version.current_stage == ModelStage.STAGING.value:
                    report["governance_summary"]["staging_models"] += 1
                
                if expiry.get("is_expired", False):
                    report["governance_summary"]["expired_models"] += 1
        
        logger.info(
            "Generated governance report",
            extra={
                "total_models": report["total_models"],
                "compliant_rate": report["governance_summary"]["compliant_models"] / max(1, report["total_models"])
            }
        )
        
        return report
        
    except Exception as e:
        logger.error("Failed to generate governance report", extra={"error": str(e)})
        return {"error": str(e)}


__all__ = [
    "mlflow",
    "infer_signature",
    "ModelStage",
    "ValidationStatus", 
    "ModelMetadata",
    "setup_mlflow",
    "setup_mlflow_secure",
    "validate_model_governance",
    "get_production_model",
    "list_models_with_governance",
    "promote_model_to_stage",
    "add_model_governance_tags",
    "get_model_approval_workflow_status",
    "validate_model_expiry",
    "generate_model_governance_report"
]

