# =============================================================================
# Clinical Data Platform - Production-Ready Makefile
# One-command demo: make demo (≤10 minutes on laptop)
# =============================================================================

# Configuration
POETRY := poetry
PY := $(POETRY) run python
DBT := $(POETRY) run dbt
DOCKER_COMPOSE := docker-compose
TF := terraform

# Port configuration (matching README)
API_PORT := 8000
UI_PORT := 8501
MINIO_PORT := 9000
MINIO_CONSOLE_PORT := 9001
MLFLOW_PORT := 5000

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
BOLD := \033[1m
NC := \033[0m

# =============================================================================
# PHONY TARGETS
# =============================================================================
.PHONY: help setup clean test lint typecheck security docs demo
.PHONY: data minio ingest dbt analytics train api ui
.PHONY: infra.init infra.plan infra.apply infra.destroy
.PHONY: validate.good validate.bad api.readonly

# =============================================================================
# HELP & SETUP
# =============================================================================

help: ## Show this help message
	@echo "$(BLUE)$(BOLD)Clinical Data Platform - Available Commands$(NC)"
	@echo ""
	@echo "$(GREEN)$(BOLD)🚀 ONE-COMMAND DEMO:$(NC)"
	@echo "  $(YELLOW)make demo$(NC)          Complete end-to-end local demo (≤10 min)"
	@echo ""
	@echo "$(GREEN)$(BOLD)📋 QUICK START:$(NC)"
	@grep -E '^[a-zA-Z_.-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-18s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)$(BOLD)🌐 ENDPOINTS (after demo):$(NC)"
	@echo "  API:       http://localhost:$(API_PORT)"
	@echo "  Dashboard: http://localhost:$(UI_PORT)"
	@echo "  MinIO:     http://localhost:$(MINIO_CONSOLE_PORT)"
	@echo "  MLflow:    http://localhost:$(MLFLOW_PORT)"

setup: ## Set up development environment
	@echo "$(BLUE)Setting up development environment...$(NC)"
	@command -v poetry >/dev/null 2>&1 || { echo "$(RED)Poetry not found. Install: https://python-poetry.org/docs/#installation$(NC)"; exit 1; }
	$(POETRY) install --with dev
	$(POETRY) run pre-commit install
	@mkdir -p logs data/sample_raw data/sample_standardized data/analytics mlruns
	@echo "$(GREEN)✅ Development environment ready!$(NC)"

# =============================================================================
# ONE-COMMAND DEMO (≤10 minutes)
# =============================================================================

demo: ## 🚀 Run complete end-to-end demo (data→MinIO→ingest→dbt→ML→API+UI)
	@echo "$(BLUE)$(BOLD)🚀 Starting Clinical Data Platform Demo...$(NC)"
	@echo "$(YELLOW)⏱️  Expected completion: ≤10 minutes$(NC)"
	@echo ""
	@$(MAKE) _demo_step_1_data
	@$(MAKE) _demo_step_2_infrastructure  
	@$(MAKE) _demo_step_3_validation
	@$(MAKE) _demo_step_4_ingestion
	@$(MAKE) _demo_step_5_warehouse
	@$(MAKE) _demo_step_6_analytics
	@$(MAKE) _demo_step_7_ml
	@$(MAKE) _demo_step_8_services
	@echo ""
	@echo "$(GREEN)$(BOLD)🎉 Demo Complete! Services running:$(NC)"
	@echo "$(GREEN)  ✅ API:       http://localhost:$(API_PORT)$(NC)"
	@echo "$(GREEN)  ✅ Dashboard: http://localhost:$(UI_PORT)$(NC)"
	@echo "$(GREEN)  ✅ MinIO:     http://localhost:$(MINIO_CONSOLE_PORT) (admin/password)$(NC)"
	@echo "$(GREEN)  ✅ MLflow:    http://localhost:$(MLFLOW_PORT)$(NC)"
	@echo ""
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  • Test API: curl http://localhost:$(API_PORT)/health"
	@echo "  • View logs: tail -f logs/clinical_platform.log"
	@echo "  • Stop services: make clean"

_demo_step_1_data:
	@echo "$(BLUE)📊 Step 1/8: Generating synthetic SDTM-like data...$(NC)"
	@$(MAKE) data
	@echo "$(GREEN)✅ Synthetic data generated$(NC)"

_demo_step_2_infrastructure:
	@echo "$(BLUE)🐳 Step 2/8: Starting infrastructure (MinIO + MLflow)...$(NC)"
	@$(MAKE) minio
	@$(MAKE) mlflow
	@echo "$(GREEN)✅ Infrastructure ready$(NC)"

_demo_step_3_validation:
	@echo "$(BLUE)🔍 Step 3/8: Validating data quality...$(NC)"
	@$(MAKE) validate.good
	@echo "$(GREEN)✅ Data validation passed$(NC)"

_demo_step_4_ingestion:
	@echo "$(BLUE)📥 Step 4/8: Running ingestion (land→bronze→silver)...$(NC)"
	@$(MAKE) ingest
	@echo "$(GREEN)✅ Data ingested$(NC)"

_demo_step_5_warehouse:
	@echo "$(BLUE)🏗️  Step 5/8: Building warehouse (dbt models + tests)...$(NC)"
	@$(MAKE) dbt
	@echo "$(GREEN)✅ Warehouse built$(NC)"

_demo_step_6_analytics:
	@echo "$(BLUE)📈 Step 6/8: Running analytics queries...$(NC)"
	@$(MAKE) analytics
	@echo "$(GREEN)✅ Analytics complete$(NC)"

_demo_step_7_ml:
	@echo "$(BLUE)🤖 Step 7/8: Training ML model...$(NC)"
	@$(MAKE) train
	@echo "$(GREEN)✅ Model trained$(NC)"

_demo_step_8_services:
	@echo "$(BLUE)🌐 Step 8/8: Starting API + Dashboard...$(NC)"
	@$(MAKE) api &
	@sleep 3
	@$(MAKE) ui &
	@sleep 2
	@echo "$(GREEN)✅ Services started$(NC)"

# =============================================================================
# DATA PIPELINE
# =============================================================================

data: ## Generate synthetic SDTM-like CSV data
	@echo "$(BLUE)Generating synthetic data...$(NC)"
	$(PY) scripts/generate_synthetic_data.py --out data/sample_raw --rows 500 --seed 42
	@echo "$(GREEN)✅ Data generated in data/sample_raw/$(NC)"

minio: ## Start MinIO S3 mock and seed buckets
	@echo "$(BLUE)Starting MinIO...$(NC)"
	$(DOCKER_COMPOSE) up -d minio
	@sleep 3
	$(PY) scripts/seed_minio.py
	@echo "$(GREEN)✅ MinIO ready at http://localhost:$(MINIO_CONSOLE_PORT)$(NC)"

mlflow: ## Start MLflow tracking server
	@echo "$(BLUE)Starting MLflow...$(NC)"
	$(DOCKER_COMPOSE) up -d mlflow
	@sleep 2
	@echo "$(GREEN)✅ MLflow ready at http://localhost:$(MLFLOW_PORT)$(NC)"

ingest: ## Run ingestion pipeline (land→bronze→silver with lineage)
	@echo "$(BLUE)Running ingestion pipeline...$(NC)"
	$(PY) -m clinical_platform.ingestion.flows run-local
	@echo "$(GREEN)✅ Ingestion complete$(NC)"

dbt: ## Run dbt models and tests (DuckDB backend)
	@echo "$(BLUE)Running dbt transformations...$(NC)"
	DBT_PROFILES_DIR=dbt $(DBT) deps --project-dir dbt/clinical_dbt
	DBT_PROFILES_DIR=dbt $(DBT) build --project-dir dbt/clinical_dbt
	@echo "$(GREEN)✅ dbt build complete$(NC)"

analytics: ## Run curated analytics queries
	@echo "$(BLUE)Running analytics...$(NC)"
	$(PY) -m clinical_platform.analytics.queries --out data/analytics
	@echo "$(GREEN)✅ Analytics complete$(NC)"

train: ## Train ML model and log to MLflow
	@echo "$(BLUE)Training ML model...$(NC)"
	$(PY) -m clinical_platform.ml.train --data data/sample_standardized --out models
	@echo "$(GREEN)✅ Model training complete$(NC)"

# =============================================================================
# DATA VALIDATION
# =============================================================================

validate.good: ## Test validation with good data (should pass)
	@echo "$(BLUE)Testing validation with good data...$(NC)"
	$(PY) -m clinical_platform.validation.runner --data data/sample_raw --output data/validation_good.json
	@echo "$(GREEN)✅ Validation passed$(NC)"

validate.bad: ## Test validation with bad data (should fail)
	@echo "$(BLUE)Testing validation with bad data (expected to fail)...$(NC)"
	$(PY) -m clinical_platform.validation.runner --data tests/fixtures/bad_data --output data/validation_bad.json || echo "$(YELLOW)⚠️  Validation failed as expected$(NC)"

# =============================================================================
# SERVICES
# =============================================================================

api: ## Run FastAPI server (port 8000)
	@echo "$(BLUE)Starting API server...$(NC)"
	$(POETRY) run uvicorn clinical_platform.api.main:app --host 0.0.0.0 --port $(API_PORT) --reload

api.readonly: ## Run API in read-only mode (GxP/HIPAA compliance)
	@echo "$(BLUE)Starting API in read-only mode...$(NC)"
	READ_ONLY_MODE=1 LOG_SCRUB_VALUES=1 $(POETRY) run uvicorn clinical_platform.api.main:app --host 0.0.0.0 --port $(API_PORT)

ui: ## Run Streamlit dashboard (port 8501)
	@echo "$(BLUE)Starting Streamlit dashboard...$(NC)"
	$(POETRY) run streamlit run src/clinical_platform/ui/dashboard.py --server.port $(UI_PORT)

# =============================================================================
# QUALITY ASSURANCE
# =============================================================================

test: ## Run test suite with 85%+ coverage requirement
	@echo "$(BLUE)Running test suite...$(NC)"
	$(POETRY) run pytest --cov-fail-under=85 -v
	@echo "$(GREEN)✅ Tests passed$(NC)"

test.fast: ## Run fast tests only (skip slow/integration tests)
	$(POETRY) run pytest -m "not slow" -v

test.compliance: ## Run compliance and PHI redaction tests
	$(POETRY) run pytest tests/test_security.py tests/test_compliance.py -v

lint: ## Run linting (ruff + black)
	@echo "$(BLUE)Running linting...$(NC)"
	$(POETRY) run ruff check src/ tests/
	$(POETRY) run black --check src/ tests/
	@echo "$(GREEN)✅ Linting passed$(NC)"

lint.fix: ## Fix linting issues automatically
	$(POETRY) run ruff check --fix src/ tests/
	$(POETRY) run black src/ tests/

typecheck: ## Run type checking (mypy)
	@echo "$(BLUE)Running type checks...$(NC)"
	$(POETRY) run mypy src/
	@echo "$(GREEN)✅ Type checking passed$(NC)"

security: ## Run security analysis (bandit + safety)
	@echo "$(BLUE)Running security analysis...$(NC)"
	$(POETRY) run bandit -r src/ -f json -o security-report.json
	$(POETRY) run safety check --json --output safety-report.json || true
	@echo "$(GREEN)✅ Security analysis complete$(NC)"

# =============================================================================
# DOCUMENTATION
# =============================================================================

docs: ## Generate and serve documentation (MkDocs)
	@echo "$(BLUE)Starting documentation server...$(NC)"
	$(POETRY) run mkdocs serve --dev-addr 0.0.0.0:8002

docs.build: ## Build documentation for deployment
	$(POETRY) run mkdocs build

# =============================================================================
# DOCKER & CONTAINERS
# =============================================================================

docker.build: ## Build Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	docker build -t clinical-platform:latest .
	@echo "$(GREEN)✅ Docker images built$(NC)"

docker.up: ## Start all services with Docker Compose
	$(DOCKER_COMPOSE) up -d

docker.down: ## Stop all services
	$(DOCKER_COMPOSE) down

# =============================================================================
# INFRASTRUCTURE (AWS)
# =============================================================================

infra.init: ## Initialize Terraform
	cd infra/terraform && $(TF) init

infra.plan: ## Plan Terraform changes
	cd infra/terraform && $(TF) plan

infra.apply: ## Apply Terraform changes
	cd infra/terraform && $(TF) apply

infra.destroy: ## Destroy Terraform infrastructure
	cd infra/terraform && $(TF) destroy

# =============================================================================
# MAINTENANCE
# =============================================================================

clean: ## Clean up containers, logs, and temporary files
	@echo "$(BLUE)Cleaning up...$(NC)"
	$(DOCKER_COMPOSE) down -v --remove-orphans
	docker system prune -f
	rm -rf logs/*.log data/warehouse.duckdb mlruns/ .pytest_cache/ .coverage htmlcov/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✅ Cleanup complete$(NC)"

status: ## Check system status and health
	@echo "$(BLUE)$(BOLD)System Status$(NC)"
	@echo ""
	@echo "$(YELLOW)Dependencies:$(NC)"
	@command -v poetry >/dev/null 2>&1 && echo "  ✅ Poetry" || echo "  ❌ Poetry"
	@command -v docker >/dev/null 2>&1 && echo "  ✅ Docker" || echo "  ❌ Docker"
	@echo ""
	@echo "$(YELLOW)Services:$(NC)"
	@curl -s http://localhost:$(API_PORT)/health >/dev/null 2>&1 && echo "  ✅ API ($(API_PORT))" || echo "  ❌ API ($(API_PORT))"
	@curl -s http://localhost:$(UI_PORT) >/dev/null 2>&1 && echo "  ✅ UI ($(UI_PORT))" || echo "  ❌ UI ($(UI_PORT))"
	@curl -s http://localhost:$(MINIO_PORT)/minio/health/live >/dev/null 2>&1 && echo "  ✅ MinIO ($(MINIO_PORT))" || echo "  ❌ MinIO ($(MINIO_PORT))"
	@curl -s http://localhost:$(MLFLOW_PORT) >/dev/null 2>&1 && echo "  ✅ MLflow ($(MLFLOW_PORT))" || echo "  ❌ MLflow ($(MLFLOW_PORT))"

# Default target
.DEFAULT_GOAL := help

