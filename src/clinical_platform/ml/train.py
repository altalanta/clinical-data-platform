from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, 
    roc_auc_score, 
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from clinical_platform.analytics.feature_eng import subject_level_features
from clinical_platform.config import get_config
from clinical_platform.logging_utils import get_logger, set_request_id, set_data_lineage_id
from clinical_platform.ml.registry import infer_signature, mlflow, setup_mlflow_secure


def create_model_artifacts(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray = None) -> Dict[str, str]:
    """Create comprehensive model artifacts for MLflow logging."""
    artifacts = {}
    
    # Set style for plots
    plt.style.use('default')
    sns.set_palette("husl")
    
    if y_pred is None:
        y_pred = (y_prob > 0.5).astype(int)
    
    # 1. ROC Curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc_score(y_true, y_prob):.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - Clinical Risk Prediction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    roc_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close()
    artifacts['roc_curve'] = roc_path
    
    # 2. Precision-Recall Curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(recall, precision, linewidth=2, label=f'PR Curve (AP = {average_precision_score(y_true, y_prob):.3f})')
    ax.axhline(y=np.mean(y_true), color='k', linestyle='--', alpha=0.5, label='Baseline')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve - Clinical Risk Prediction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    pr_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
    plt.savefig(pr_path, dpi=150, bbox_inches='tight')
    plt.close()
    artifacts['precision_recall_curve'] = pr_path
    
    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Low Risk', 'High Risk'],
                yticklabels=['Low Risk', 'High Risk'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix - Clinical Risk Prediction')
    
    cm_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    artifacts['confusion_matrix'] = cm_path
    
    # 4. Risk Score Distribution
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot histograms for each class
    low_risk_scores = y_prob[y_true == 0]
    high_risk_scores = y_prob[y_true == 1]
    
    ax.hist(low_risk_scores, bins=30, alpha=0.7, label='Actual Low Risk', color='skyblue', density=True)
    ax.hist(high_risk_scores, bins=30, alpha=0.7, label='Actual High Risk', color='salmon', density=True)
    
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.8, label='Decision Threshold (0.5)')
    ax.set_xlabel('Risk Score')
    ax.set_ylabel('Density')
    ax.set_title('Risk Score Distribution by Actual Class')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    dist_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    plt.close()
    artifacts['risk_distribution'] = dist_path
    
    return artifacts


def generate_model_report(X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray, 
                         y_val: np.ndarray, y_prob: np.ndarray, pipe: Pipeline, 
                         feature_names: list) -> Dict[str, Any]:
    """Generate comprehensive model performance report."""
    
    y_pred = (y_prob > 0.5).astype(int)
    
    # Basic metrics
    auc = roc_auc_score(y_val, y_prob)
    ap = average_precision_score(y_val, y_prob)
    
    # Classification report
    class_report = classification_report(y_val, y_pred, output_dict=True, 
                                       target_names=['Low Risk', 'High Risk'])
    
    # Cross-validation scores on training data
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='roc_auc')
    
    # Feature importance (for logistic regression)
    feature_importance = {}
    if hasattr(pipe.named_steps['clf'], 'coef_'):
        coefficients = pipe.named_steps['clf'].coef_[0]
        for i, coef in enumerate(coefficients):
            feature_importance[feature_names[i]] = float(coef)
    
    # Model performance summary
    performance_summary = {
        'validation_metrics': {
            'auc': float(auc),
            'average_precision': float(ap),
            'accuracy': float(class_report['accuracy']),
            'precision_high_risk': float(class_report['High Risk']['precision']),
            'recall_high_risk': float(class_report['High Risk']['recall']),
            'f1_high_risk': float(class_report['High Risk']['f1-score']),
            'precision_low_risk': float(class_report['Low Risk']['precision']),
            'recall_low_risk': float(class_report['Low Risk']['recall']),
            'f1_low_risk': float(class_report['Low Risk']['f1-score'])
        },
        'cross_validation': {
            'mean_cv_auc': float(cv_scores.mean()),
            'std_cv_auc': float(cv_scores.std()),
            'cv_scores': cv_scores.tolist()
        },
        'feature_importance': feature_importance,
        'data_summary': {
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_features': X_train.shape[1],
            'positive_rate_train': float(y_train.mean()),
            'positive_rate_val': float(y_val.mean())
        },
        'model_info': {
            'algorithm': 'LogisticRegression',
            'regularization': pipe.named_steps['clf'].get_params().get('C', 1.0),
            'max_iter': pipe.named_steps['clf'].get_params().get('max_iter', 100),
            'solver': pipe.named_steps['clf'].get_params().get('solver', 'lbfgs')
        }
    }
    
    return performance_summary


def train(data_dir: str | Path | None = None, out_dir: str | Path | None = None, 
          seed: int = 42, model_name: str = "clinical_risk_model") -> Tuple[float, float]:
    """
    Train clinical risk prediction model with comprehensive MLflow tracking.
    
    Args:
        data_dir: Directory containing training data
        out_dir: Output directory for artifacts
        seed: Random seed for reproducibility
        model_name: Name for model registry
        
    Returns:
        Tuple of (validation_auc, validation_ap)
    """
    
    # Set up tracking and logging
    cfg = get_config()
    request_id = set_request_id()
    lineage_id = set_data_lineage_id()
    logger = get_logger("ml_training", request_id=request_id, data_lineage_id=lineage_id)
    
    logger.info("Starting ML model training", 
                model_name=model_name, 
                data_dir=str(data_dir),
                seed=seed)
    
    # Setup MLflow
    setup_mlflow_secure(cfg.mlflow.tracking_uri, cfg.mlflow.auth_token)
    
    # Load and prepare data
    try:
        feats = subject_level_features(data_dir)
        logger.info("Features loaded", shape=feats.shape, columns=list(feats.columns))
    except Exception as e:
        logger.error("Failed to load features", error=str(e))
        raise
    
    # Define feature set and target
    feature_columns = ["AGE", "AE_COUNT", "SEVERE_AE_COUNT"]
    X = feats[feature_columns].fillna(0).astype(float).values
    y = (feats["SEVERE_AE_COUNT"] > 0).astype(int).values
    
    logger.info("Training data prepared", 
                n_samples=len(X), 
                n_features=X.shape[1],
                positive_rate=float(y.mean()))
    
    # Train-validation split
    test_size = 0.3
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, 
                                                      random_state=seed, stratify=y)
    
    # Model configuration
    max_iter = 200
    C = 1.0
    pipe = Pipeline([
        ("scaler", StandardScaler()), 
        ("clf", LogisticRegression(max_iter=max_iter, C=C, random_state=seed))
    ])
    
    # Set up MLflow experiment
    experiment_name = cfg.mlflow.experiment_name
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info("Created new MLflow experiment", experiment_id=experiment_id)
    except Exception:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        logger.info("Using existing MLflow experiment", experiment_id=experiment_id)
    
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run with comprehensive tracking
    run_name = f"clinical_risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name) as run:
        logger.info("Started MLflow run", run_id=run.info.run_id)
        
        # Log parameters
        params = {
            "random_seed": seed,
            "test_size": test_size,
            "max_iter": max_iter,
            "regularization_C": C,
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "algorithm": "logistic_regression",
            "feature_columns": ",".join(feature_columns),
            "target_definition": "severe_ae_count > 0",
            "data_lineage_id": lineage_id,
            "request_id": request_id
        }
        
        for key, value in params.items():
            mlflow.log_param(key, value)
        
        # Log model metadata and governance tags
        tags = {
            "model_type": "binary_classification",
            "framework": "sklearn",
            "target": "severe_ae_risk",
            "data_version": "v1.0",
            "clinical_use": "adverse_event_risk_prediction",
            "validation_status": "pending",
            "regulatory_approval": "pending",
            "created_by": "clinical_ml_pipeline",
            "training_date": datetime.now().isoformat(),
            "data_lineage_id": lineage_id,
            "request_id": request_id,
            "model_purpose": "clinical_decision_support",
            "risk_category": "patient_safety"
        }
        
        for key, value in tags.items():
            mlflow.set_tag(key, value)
        
        # Train model
        logger.info("Training model")
        pipe.fit(X_train, y_train)
        
        # Generate predictions
        y_prob = pipe.predict_proba(X_val)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)
        
        # Generate comprehensive performance report
        performance_report = generate_model_report(
            X_train, X_val, y_train, y_val, y_prob, pipe, feature_columns
        )
        
        # Log all metrics
        for metric_name, metric_value in performance_report['validation_metrics'].items():
            mlflow.log_metric(f"val_{metric_name}", metric_value)
        
        for metric_name, metric_value in performance_report['cross_validation'].items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(f"cv_{metric_name}", metric_value)
        
        # Log data summary metrics
        for metric_name, metric_value in performance_report['data_summary'].items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Log model artifacts (plots)
        artifacts = create_model_artifacts(y_val, y_prob, y_pred)
        for artifact_name, artifact_path in artifacts.items():
            mlflow.log_artifact(artifact_path, f"plots/{artifact_name}.png")
        
        # Log performance report as JSON
        report_path = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False).name
        with open(report_path, 'w') as f:
            json.dump(performance_report, f, indent=2)
        mlflow.log_artifact(report_path, "reports/performance_report.json")
        
        # Performance gate evaluation
        auc = performance_report['validation_metrics']['auc']
        ap = performance_report['validation_metrics']['average_precision']
        min_auc_threshold = 0.65  # Adjusted for demo data
        min_ap_threshold = 0.6
        
        performance_gate_passed = auc >= min_auc_threshold and ap >= min_ap_threshold
        
        mlflow.set_tag("performance_gate", "passed" if performance_gate_passed else "failed")
        mlflow.log_metric("performance_gate_passed", 1 if performance_gate_passed else 0)
        
        if not performance_gate_passed:
            logger.warning("Model performance below clinical thresholds", 
                          auc=auc, ap=ap, 
                          min_auc=min_auc_threshold, min_ap=min_ap_threshold)
        
        # Log model with signature and register
        sig = infer_signature(X_train, y_prob)
        
        model_info = mlflow.sklearn.log_model(
            pipe, 
            "model", 
            signature=sig,
            registered_model_name=model_name,
            pip_requirements=["scikit-learn", "pandas", "numpy"],
            input_example=X_train[:5]
        )
        
        # Model registry management
        client = mlflow.tracking.MlflowClient()
        
        # Get the latest model version
        try:
            latest_versions = client.get_latest_versions(model_name, stages=["None"])
            if latest_versions:
                model_version = latest_versions[0].version
            else:
                # Fallback - get all versions and find the latest
                all_versions = client.search_model_versions(f"name='{model_name}'")
                model_version = max([int(v.version) for v in all_versions])
        except Exception as e:
            logger.error("Failed to get model version", error=str(e))
            model_version = "1"  # Fallback
        
        # Update model version with detailed description
        description = f"""Clinical Risk Prediction Model
        
Performance Metrics:
- Validation AUC: {auc:.3f}
- Average Precision: {ap:.3f}
- Accuracy: {performance_report['validation_metrics']['accuracy']:.3f}
- High Risk Precision: {performance_report['validation_metrics']['precision_high_risk']:.3f}
- High Risk Recall: {performance_report['validation_metrics']['recall_high_risk']:.3f}

Training Info:
- Training Samples: {len(X_train)}
- Validation Samples: {len(X_val)}
- Features: {', '.join(feature_columns)}
- Cross-val AUC: {performance_report['cross_validation']['mean_cv_auc']:.3f} ± {performance_report['cross_validation']['std_cv_auc']:.3f}

Data Lineage: {lineage_id}
Request ID: {request_id}
"""
        
        try:
            client.update_model_version(
                name=model_name,
                version=model_version,
                description=description
            )
        except Exception as e:
            logger.warning("Failed to update model description", error=str(e))
        
        # Stage transition based on performance
        if performance_gate_passed:
            try:
                client.transition_model_version_stage(
                    name=model_name,
                    version=model_version,
                    stage=cfg.mlflow.model_registry_stage,
                    archive_existing_versions=False
                )
                logger.info("Model transitioned to staging", 
                           model_name=model_name, 
                           version=model_version,
                           stage=cfg.mlflow.model_registry_stage)
            except Exception as e:
                logger.error("Failed to transition model stage", error=str(e))
        else:
            logger.info("Model performance below threshold - remaining in None stage")
        
        # Log final summary
        logger.info("ML training completed",
                    run_id=run.info.run_id,
                    model_version=model_version,
                    auc=auc,
                    ap=ap,
                    performance_gate_passed=performance_gate_passed)
        
        print(f"✅ Training completed successfully!")
        print(f"📊 Validation AUC: {auc:.3f}")
        print(f"📊 Average Precision: {ap:.3f}")
        print(f"🔬 MLflow Run ID: {run.info.run_id}")
        print(f"📦 Model Version: {model_version}")
        print(f"🏆 Performance Gate: {'PASSED' if performance_gate_passed else 'FAILED'}")
    
    # Save local artifacts
    out = Path(out_dir or "models")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "val_probs.npy", y_prob)
    
    # Save performance summary
    with open(out / "performance_summary.json", 'w') as f:
        json.dump(performance_report, f, indent=2)
    
    return float(auc), float(ap)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--out", type=str, default="models")
    args = parser.parse_args()
    auc, ap = train(args.data, args.out)
    print({"val_auc": auc, "val_ap": ap})

