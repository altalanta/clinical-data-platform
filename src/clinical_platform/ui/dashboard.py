"""Streamlit dashboard for the Clinical Data Platform."""

from __future__ import annotations

import os
import sys
import time
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from clinical_platform.config import get_config
from clinical_platform.analytics.feature_eng import subject_level_features

# Page configuration
st.set_page_config(
    page_title="Clinical Data Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_BASE_URL = "http://localhost:8000"
DEFAULT_DATA_DIR = "data/sample_standardized"

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin: 0.5rem 0;
    }
    
    .status-healthy { color: #10b981; }
    .status-warning { color: #f59e0b; }
    .status-error { color: #ef4444; }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health() -> Dict[str, Any]:
    """Check API health status."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"status": "unreachable", "error": str(e)}


def call_api(endpoint: str, method: str = "GET", data: Optional[Dict] = None) -> Dict[str, Any]:
    """Make API calls with error handling."""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=30)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
    
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}


def render_header():
    """Render the main header."""
    st.title("🏥 Clinical Data Platform")
    st.markdown("**Production-ready clinical trial data processing, validation, and analytics**")
    
    # API Health Status
    health = check_api_health()
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        if health.get("status") == "healthy":
            st.success("🟢 API Online")
        elif health.get("status") == "degraded":
            st.warning("🟡 API Degraded")
        else:
            st.error("🔴 API Offline")
    
    with col2:
        if health.get("database_status") == "healthy":
            st.success("🟢 Database")
        else:
            st.error("🔴 Database")
    
    with col3:
        if health.get("mlflow_status") == "healthy":
            st.success("🟢 MLflow")
        else:
            st.warning("🟡 MLflow")
    
    with col4:
        st.info(f"v{health.get('version', 'unknown')}")
    
    st.divider()


def render_sidebar():
    """Render the sidebar with navigation and controls."""
    st.sidebar.title("Navigation")
    
    # Page selection
    page = st.sidebar.selectbox(
        "Select Page",
        ["Dashboard", "Data Management", "Analytics", "ML Operations", "Model Registry", "System Health"],
        index=0
    )
    
    st.sidebar.divider()
    
    # Data directory configuration
    st.sidebar.subheader("Data Configuration")
    data_dir = st.sidebar.text_input(
        "Data Directory",
        value=DEFAULT_DATA_DIR,
        help="Path to SDTM standardized data files"
    )
    
    # Quick actions
    st.sidebar.subheader("Quick Actions")
    
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()
    
    if st.sidebar.button("📊 Load Sample Data", use_container_width=True):
        with st.spinner("Loading sample data..."):
            # Simulate data loading
            time.sleep(1)
            st.sidebar.success("Sample data loaded!")
    
    return page, data_dir


def render_dashboard_page(data_dir: str):
    """Render the main dashboard page."""
    st.header("📊 Study Overview Dashboard")
    
    try:
        # Load subject-level features
        with st.spinner("Loading clinical data..."):
            features_df = subject_level_features(data_dir)
        
        if features_df.empty:
            st.warning("No data found in the specified directory.")
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_subjects = len(features_df)
            st.metric("Total Subjects", total_subjects)
        
        with col2:
            total_aes = features_df["AE_COUNT"].sum()
            st.metric("Total Adverse Events", int(total_aes))
        
        with col3:
            serious_aes = features_df["SERIOUS_AE_COUNT"].sum()
            st.metric("Serious AEs", int(serious_aes))
        
        with col4:
            avg_age = features_df["AGE"].mean()
            st.metric("Average Age", f"{avg_age:.1f}")
        
        # Charts row 1
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Demographics Distribution")
            
            # Age distribution
            fig_age = px.histogram(
                features_df,
                x="AGE",
                nbins=20,
                title="Age Distribution",
                color_discrete_sequence=['#667eea']
            )
            fig_age.update_layout(height=400)
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            st.subheader("Safety Profile")
            
            # Safety risk distribution
            safety_counts = features_df["safety_risk_category"].value_counts()
            fig_safety = px.pie(
                values=safety_counts.values,
                names=safety_counts.index,
                title="Safety Risk Categories",
                color_discrete_map={
                    'LOW_RISK': '#10b981',
                    'MODERATE_RISK': '#f59e0b',
                    'HIGH_RISK': '#ef4444'
                }
            )
            fig_safety.update_layout(height=400)
            st.plotly_chart(fig_safety, use_container_width=True)
        
        # Charts row 2
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Treatment Arms")
            
            # Treatment arm distribution
            if "treatment_arm" in features_df.columns:
                arm_counts = features_df["treatment_arm"].value_counts()
                fig_arms = px.bar(
                    x=arm_counts.index,
                    y=arm_counts.values,
                    title="Subjects by Treatment Arm",
                    color_discrete_sequence=['#764ba2']
                )
                fig_arms.update_layout(height=400)
                st.plotly_chart(fig_arms, use_container_width=True)
            else:
                st.info("Treatment arm data not available")
        
        with col2:
            st.subheader("Data Quality")
            
            # Data completeness
            completeness_df = features_df[["data_completeness_score"]].copy()
            completeness_df["Quality"] = pd.cut(
                completeness_df["data_completeness_score"],
                bins=[0, 0.5, 0.8, 1.0],
                labels=["Low", "Medium", "High"]
            )
            
            quality_counts = completeness_df["Quality"].value_counts()
            fig_quality = px.bar(
                x=quality_counts.index,
                y=quality_counts.values,
                title="Data Completeness Quality",
                color_discrete_sequence=['#10b981', '#f59e0b', '#ef4444']
            )
            fig_quality.update_layout(height=400)
            st.plotly_chart(fig_quality, use_container_width=True)
        
        # Recent activity table
        st.subheader("Subject Data Summary")
        
        # Prepare summary data
        summary_df = features_df[[
            "STUDYID", "SUBJID", "AGE", "SEX",
            "AE_COUNT", "SERIOUS_AE_COUNT", "safety_risk_category", "data_completeness_score"
        ]].copy()
        
        summary_df.columns = [
            "Study ID", "Subject ID", "Age", "Sex",
            "AE Count", "Serious AEs", "Risk Category", "Data Quality"
        ]
        
        # Format data quality as percentage
        summary_df["Data Quality"] = (summary_df["Data Quality"] * 100).round(1).astype(str) + "%"
        
        st.dataframe(
            summary_df.head(20),
            use_container_width=True,
            hide_index=True
        )
        
    except Exception as e:
        st.error(f"Error loading dashboard data: {str(e)}")
        st.info("Please check if the data directory exists and contains valid SDTM files.")


def render_data_management_page(data_dir: str):
    """Render the data management page."""
    st.header("📁 Data Management")
    
    # Data ingestion section
    st.subheader("Data Ingestion")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.text_input("Data Directory Path", value=data_dir, disabled=True)
    
    with col2:
        validate_data = st.checkbox("Run Validation", value=True)
        force_reload = st.checkbox("Force Reload", value=False)
    
    if st.button("🚀 Ingest Data", type="primary", use_container_width=True):
        with st.spinner("Ingesting data..."):
            payload = {
                "data_dir": data_dir,
                "validate_data": validate_data,
                "force_reload": force_reload
            }
            
            result = call_api("/api/v1/data/ingest", "POST", payload)
            
            if "error" not in result:
                st.success(f"✅ {result.get('message', 'Data ingested successfully')}")
                
                if result.get("files_processed"):
                    st.info(f"Processed files: {', '.join(result['files_processed'])}")
                
                if result.get("validation_results"):
                    st.json(result["validation_results"])
            else:
                st.error(f"❌ Ingestion failed: {result['error']}")
    
    st.divider()
    
    # Data validation section
    st.subheader("Data Validation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        validation_type = st.selectbox(
            "Validation Type",
            ["both", "pandera", "great_expectations"],
            index=0
        )
    
    with col2:
        domains = st.multiselect(
            "Domains (leave empty for all)",
            ["DM", "AE", "LB", "VS", "EX"],
            default=[]
        )
    
    if st.button("🔍 Validate Data", use_container_width=True):
        with st.spinner("Validating data..."):
            payload = {
                "data_dir": data_dir,
                "validation_type": validation_type,
                "domains": domains if domains else None
            }
            
            result = call_api("/api/v1/data/validate", "POST", payload)
            
            if "error" not in result:
                st.success(f"✅ Validation completed: {result.get('overall_status', 'unknown')}")
                
                if result.get("results"):
                    # Display validation results
                    results_df = pd.DataFrame(result["results"])
                    
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Errors", result.get("total_errors", 0))
                    with col2:
                        st.metric("Total Warnings", result.get("total_warnings", 0))
                    with col3:
                        passed_count = len(results_df[results_df["status"] == "passed"])
                        st.metric("Passed Validations", passed_count)
                    
                    # Results table
                    st.dataframe(results_df, use_container_width=True)
            else:
                st.error(f"❌ Validation failed: {result['error']}")
    
    # File browser section
    st.divider()
    st.subheader("📂 Data Files")
    
    try:
        data_path = Path(data_dir)
        if data_path.exists():
            files = list(data_path.glob("*.parquet"))
            if files:
                files_data = []
                for file in files:
                    stat = file.stat()
                    files_data.append({
                        "File": file.name,
                        "Size (MB)": round(stat.st_size / 1024 / 1024, 2),
                        "Modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                    })
                
                files_df = pd.DataFrame(files_data)
                st.dataframe(files_df, use_container_width=True, hide_index=True)
            else:
                st.info("No parquet files found in the data directory.")
        else:
            st.warning(f"Data directory does not exist: {data_dir}")
    except Exception as e:
        st.error(f"Error browsing files: {str(e)}")


def render_analytics_page(data_dir: str):
    """Render the analytics page."""
    st.header("📊 Clinical Analytics")
    
    try:
        # Load data
        with st.spinner("Loading analytics data..."):
            features_df = subject_level_features(data_dir)
        
        if features_df.empty:
            st.warning("No data available for analytics.")
            return
        
        # Analytics tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Safety Analysis", "Demographics", "Data Quality", "Treatment Outcomes"])
        
        with tab1:
            st.subheader("🚨 Safety Analysis")
            
            # Safety metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_aes = features_df["AE_COUNT"].sum()
                st.metric("Total AEs", int(total_aes))
            
            with col2:
                serious_aes = features_df["SERIOUS_AE_COUNT"].sum()
                serious_rate = (serious_aes / total_aes * 100) if total_aes > 0 else 0
                st.metric("Serious AEs", int(serious_aes), f"{serious_rate:.1f}%")
            
            with col3:
                subjects_with_serious = (features_df["SERIOUS_AE_COUNT"] > 0).sum()
                serious_subject_rate = subjects_with_serious / len(features_df) * 100
                st.metric("Subjects w/ Serious AEs", subjects_with_serious, f"{serious_subject_rate:.1f}%")
            
            with col4:
                high_risk_subjects = (features_df["safety_risk_category"] == "HIGH_RISK").sum()
                st.metric("High Risk Subjects", high_risk_subjects)
            
            # Safety charts
            col1, col2 = st.columns(2)
            
            with col1:
                # AE distribution by treatment arm
                if "treatment_arm" in features_df.columns:
                    ae_by_arm = features_df.groupby("treatment_arm")["AE_COUNT"].agg(["sum", "mean", "count"]).reset_index()
                    
                    fig = px.bar(
                        ae_by_arm,
                        x="treatment_arm",
                        y="sum",
                        title="Total AEs by Treatment Arm",
                        color_discrete_sequence=['#ef4444']
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Serious AE severity distribution
                severity_data = features_df[features_df["SERIOUS_AE_COUNT"] > 0]
                if not severity_data.empty:
                    fig = px.scatter(
                        severity_data,
                        x="AGE",
                        y="SERIOUS_AE_COUNT",
                        color="safety_risk_category",
                        title="Age vs Serious AEs",
                        color_discrete_map={
                            'LOW_RISK': '#10b981',
                            'MODERATE_RISK': '#f59e0b',
                            'HIGH_RISK': '#ef4444'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("👥 Demographics Analysis")
            
            # Demographics summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                male_count = (features_df["SEX"] == "M").sum()
                male_pct = male_count / len(features_df) * 100
                st.metric("Male Subjects", male_count, f"{male_pct:.1f}%")
            
            with col2:
                female_count = (features_df["SEX"] == "F").sum()
                female_pct = female_count / len(features_df) * 100
                st.metric("Female Subjects", female_count, f"{female_pct:.1f}%")
            
            with col3:
                mean_age = features_df["AGE"].mean()
                median_age = features_df["AGE"].median()
                st.metric("Mean Age", f"{mean_age:.1f}", f"Median: {median_age:.1f}")
            
            with col4:
                age_range = features_df["AGE"].max() - features_df["AGE"].min()
                st.metric("Age Range", f"{features_df['AGE'].min()}-{features_df['AGE'].max()}", f"Span: {age_range}")
            
            # Demographics visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Age distribution by sex
                fig = px.histogram(
                    features_df,
                    x="AGE",
                    color="SEX",
                    title="Age Distribution by Sex",
                    nbins=20,
                    opacity=0.7
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Treatment arm demographics
                if "treatment_arm" in features_df.columns:
                    demo_summary = features_df.groupby(["treatment_arm", "SEX"]).size().reset_index(name="count")
                    
                    fig = px.bar(
                        demo_summary,
                        x="treatment_arm",
                        y="count",
                        color="SEX",
                        title="Enrollment by Treatment Arm and Sex",
                        barmode="group"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("📋 Data Quality Analysis")
            
            # Data quality metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_completeness = features_df["data_completeness_score"].mean()
                st.metric("Avg Completeness", f"{avg_completeness:.1%}")
            
            with col2:
                high_quality = (features_df["data_completeness_score"] > 0.8).sum()
                st.metric("High Quality Subjects", high_quality)
            
            with col3:
                has_ae_data = features_df["AE_COUNT"].gt(0).sum()
                st.metric("Subjects w/ AE Data", has_ae_data)
            
            with col4:
                has_lab_data = features_df["LAB_COUNT"].gt(0).sum()
                st.metric("Subjects w/ Lab Data", has_lab_data)
            
            # Data quality visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Completeness distribution
                fig = px.histogram(
                    features_df,
                    x="data_completeness_score",
                    title="Data Completeness Distribution",
                    nbins=20,
                    color_discrete_sequence=['#10b981']
                )
                fig.update_xaxis(title="Completeness Score")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Data availability heatmap
                data_availability = pd.DataFrame({
                    'AE Data': features_df["AE_COUNT"] > 0,
                    'Lab Data': features_df["LAB_COUNT"] > 0,
                    'Vital Data': features_df["VITAL_COUNT"] > 0
                }).astype(int)
                
                correlation_matrix = data_availability.corr()
                
                fig = px.imshow(
                    correlation_matrix,
                    title="Data Availability Correlation",
                    color_continuous_scale="RdYlGn",
                    aspect="auto"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("💊 Treatment Outcomes")
            
            if "treatment_arm" in features_df.columns:
                # Treatment arm analysis
                treatment_summary = features_df.groupby("treatment_arm").agg({
                    "AE_COUNT": ["mean", "sum"],
                    "SERIOUS_AE_COUNT": ["mean", "sum"],
                    "LAB_COUNT": "mean",
                    "data_completeness_score": "mean"
                }).round(2)
                
                treatment_summary.columns = [
                    "Avg AEs", "Total AEs", "Avg Serious AEs", "Total Serious AEs",
                    "Avg Lab Tests", "Avg Completeness"
                ]
                
                st.subheader("Treatment Arm Summary")
                st.dataframe(treatment_summary, use_container_width=True)
                
                # Treatment effectiveness visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.box(
                        features_df,
                        x="treatment_arm",
                        y="AE_COUNT",
                        title="AE Distribution by Treatment Arm"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Safety profile by treatment
                    safety_by_treatment = pd.crosstab(
                        features_df["treatment_arm"],
                        features_df["safety_risk_category"],
                        normalize="index"
                    ) * 100
                    
                    fig = px.bar(
                        safety_by_treatment,
                        title="Safety Risk Profile by Treatment (%)",
                        color_discrete_map={
                            'LOW_RISK': '#10b981',
                            'MODERATE_RISK': '#f59e0b',
                            'HIGH_RISK': '#ef4444'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Treatment arm data not available for outcomes analysis.")
    
    except Exception as e:
        st.error(f"Error in analytics: {str(e)}")


def render_ml_operations_page():
    """Render the ML operations page."""
    st.header("🤖 ML Operations")
    
    # Model training section
    st.subheader("Model Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_name = st.text_input("Model Name", value="cdp_logreg")
        experiment_name = st.text_input("Experiment Name", value="clinical_ml")
        data_dir_ml = st.text_input("Training Data Directory", value=DEFAULT_DATA_DIR)
    
    with col2:
        cv_folds = st.number_input("CV Folds", min_value=2, max_value=10, value=5)
        test_size = st.slider("Test Size", min_value=0.1, max_value=0.5, value=0.2, step=0.1)
        random_state = st.number_input("Random Seed", min_value=1, max_value=9999, value=42)
    
    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        with st.spinner("Training model... This may take a few minutes."):
            payload = {
                "model_name": model_name,
                "experiment_name": experiment_name,
                "data_dir": data_dir_ml,
                "cv_folds": cv_folds,
                "test_size": test_size,
                "random_state": random_state
            }
            
            result = call_api("/api/v1/ml/train", "POST", payload)
            
            if "error" not in result:
                st.success(f"✅ Model trained successfully!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Run ID:** {result.get('run_id', 'unknown')}")
                    st.info(f"**Model URI:** {result.get('model_uri', 'unknown')}")
                
                with col2:
                    training_time = result.get('training_time_seconds', 0)
                    st.info(f"**Training Time:** {training_time:.2f} seconds")
                    st.info(f"**Artifacts:** {len(result.get('artifacts', []))}")
                
                # Display metrics
                if result.get("metrics"):
                    st.subheader("Training Metrics")
                    metrics_df = pd.DataFrame(list(result["metrics"].items()), columns=["Metric", "Value"])
                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            else:
                st.error(f"❌ Training failed: {result['error']}")
    
    st.divider()
    
    # Batch prediction section
    st.subheader("Batch Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        prediction_data_dir = st.text_input("Prediction Data Directory", value=DEFAULT_DATA_DIR)
        model_uri = st.text_input("Model URI", value="models:/cdp_logreg/latest")
    
    with col2:
        output_path = st.text_input("Output Path", value="predictions.parquet")
    
    if st.button("🔮 Generate Predictions", use_container_width=True):
        with st.spinner("Generating predictions..."):
            payload = {
                "data_dir": prediction_data_dir,
                "model_uri": model_uri,
                "output_path": output_path
            }
            
            result = call_api("/api/v1/ml/predict", "POST", payload)
            
            if "error" not in result:
                st.success(f"✅ Generated {result.get('predictions_count', 0)} predictions")
                
                # Display prediction summary
                if result.get("prediction_summary"):
                    summary = result["prediction_summary"]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Mean Risk", f"{summary.get('mean_risk', 0):.3f}")
                    with col2:
                        st.metric("High Risk", summary.get('high_risk_subjects', 0))
                    with col3:
                        st.metric("Medium Risk", summary.get('medium_risk_subjects', 0))
                    with col4:
                        st.metric("Low Risk", summary.get('low_risk_subjects', 0))
            else:
                st.error(f"❌ Prediction failed: {result['error']}")


def render_model_registry_page():
    """Render the model registry page."""
    st.header("📚 Model Registry")
    
    # Models list
    st.subheader("Registered Models")
    
    if st.button("🔄 Refresh Models"):
        st.rerun()
    
    # Fetch models from API
    result = call_api("/api/v1/ml/models")
    
    if "error" not in result:
        models = result.get("models", [])
        
        if models:
            # Convert to DataFrame for display
            models_data = []
            for model in models:
                models_data.append({
                    "Name": model["name"],
                    "Version": model["version"],
                    "Stage": model["stage"],
                    "Status": model["status"],
                    "Production Ready": "✅" if model["is_production_ready"] else "❌",
                    "Performance Gate": "✅" if model["performance_gate_passed"] else "❌",
                    "Validation Approved": "✅" if model["validation_approved"] else "❌",
                    "Created": model["created_timestamp"][:10] if model["created_timestamp"] else "unknown"
                })
            
            models_df = pd.DataFrame(models_data)
            st.dataframe(models_df, use_container_width=True, hide_index=True)
            
            # Model promotion section
            st.divider()
            st.subheader("Model Promotion")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                model_names = [m["name"] for m in models]
                selected_model = st.selectbox("Model Name", model_names if model_names else ["No models"])
            
            with col2:
                if selected_model and selected_model != "No models":
                    model_versions = [m["version"] for m in models if m["name"] == selected_model]
                    selected_version = st.selectbox("Version", model_versions)
                else:
                    selected_version = st.selectbox("Version", ["No versions"])
            
            with col3:
                target_stage = st.selectbox("Target Stage", ["Staging", "Production", "Archived"])
            
            with col4:
                force_promotion = st.checkbox("Force Promotion")
            
            if st.button("🚀 Promote Model", type="primary", use_container_width=True):
                if selected_model and selected_model != "No models" and selected_version != "No versions":
                    payload = {
                        "model_name": selected_model,
                        "version": selected_version,
                        "target_stage": target_stage,
                        "force": force_promotion
                    }
                    
                    result = call_api(f"/api/v1/ml/models/{selected_model}/versions/{selected_version}/promote", "POST", payload)
                    
                    if "error" not in result:
                        st.success(f"✅ Model promoted to {target_stage}")
                        st.rerun()
                    else:
                        st.error(f"❌ Promotion failed: {result['error']}")
                else:
                    st.error("Please select a valid model and version")
        else:
            st.info("No models found in the registry. Train a model first.")
    else:
        st.error(f"Failed to fetch models: {result['error']}")
    
    # Governance report
    st.divider()
    st.subheader("Governance Report")
    
    if st.button("📊 Generate Report"):
        result = call_api("/api/v1/ml/governance/report")
        
        if "error" not in result:
            report = result
            
            # Summary metrics
            summary = report.get("governance_summary", {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Models", report.get("total_models", 0))
            with col2:
                st.metric("Compliant", summary.get("compliant_models", 0))
            with col3:
                st.metric("Production", summary.get("production_models", 0))
            with col4:
                st.metric("Expired", summary.get("expired_models", 0))
            
            # Detailed report
            if report.get("model_details"):
                details_df = pd.DataFrame(report["model_details"])
                st.dataframe(details_df, use_container_width=True, hide_index=True)
        else:
            st.error(f"Failed to generate report: {result['error']}")


def render_system_health_page():
    """Render the system health page."""
    st.header("🏥 System Health")
    
    # Get health data
    health = check_api_health()
    
    if health.get("status") != "unreachable":
        # Overall status
        status = health.get("status", "unknown")
        if status == "healthy":
            st.success(f"🟢 System Status: {status.upper()}")
        elif status == "degraded":
            st.warning(f"🟡 System Status: {status.upper()}")
        else:
            st.error(f"🔴 System Status: {status.upper()}")
        
        # System metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            uptime = health.get("uptime_seconds", 0)
            uptime_hours = uptime / 3600
            st.metric("Uptime", f"{uptime_hours:.1f} hours")
        
        with col2:
            memory = health.get("memory_usage_mb", 0)
            st.metric("Memory Usage", f"{memory:.1f}%")
        
        with col3:
            cpu = health.get("cpu_usage_percent", 0)
            st.metric("CPU Usage", f"{cpu:.1f}%")
        
        with col4:
            version = health.get("version", "unknown")
            st.metric("Version", version)
        
        # Service status
        st.subheader("Service Health")
        
        services = health.get("services", {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            db_status = health.get("database_status", "unknown")
            if db_status == "healthy":
                st.success("🟢 Database: Healthy")
            else:
                st.error(f"🔴 Database: {db_status}")
        
        with col2:
            mlflow_status = health.get("mlflow_status", "unknown")
            if mlflow_status == "healthy":
                st.success("🟢 MLflow: Healthy")
            elif mlflow_status == "not_configured":
                st.warning("🟡 MLflow: Not Configured")
            else:
                st.error(f"🔴 MLflow: {mlflow_status}")
        
        with col3:
            storage_status = health.get("storage_status", "unknown")
            if storage_status == "healthy":
                st.success("🟢 Storage: Healthy")
            else:
                st.warning(f"🟡 Storage: {storage_status}")
        
        # Environment info
        st.subheader("Environment Information")
        
        env_info = {
            "Environment": health.get("environment", "unknown"),
            "Timestamp": health.get("timestamp", "unknown"),
            "API Base URL": API_BASE_URL
        }
        
        for key, value in env_info.items():
            st.text(f"{key}: {value}")
    
    else:
        st.error("🔴 Cannot connect to API")
        st.error(f"Error: {health.get('error', 'Unknown error')}")
        
        st.subheader("Troubleshooting")
        st.markdown("""
        **API Connection Failed**
        
        1. Ensure the API server is running: `uvicorn clinical_platform.api.endpoints:app --reload`
        2. Check the API URL is correct: `http://localhost:8000`
        3. Verify no firewall is blocking the connection
        4. Check the server logs for errors
        """)


def main():
    """Main Streamlit application."""
    
    # Render header
    render_header()
    
    # Render sidebar and get navigation
    page, data_dir = render_sidebar()
    
    # Render selected page
    if page == "Dashboard":
        render_dashboard_page(data_dir)
    elif page == "Data Management":
        render_data_management_page(data_dir)
    elif page == "Analytics":
        render_analytics_page(data_dir)
    elif page == "ML Operations":
        render_ml_operations_page()
    elif page == "Model Registry":
        render_model_registry_page()
    elif page == "System Health":
        render_system_health_page()
    
    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 1rem;'>
            Clinical Data Platform v0.1.0 | Built with Streamlit & FastAPI
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

