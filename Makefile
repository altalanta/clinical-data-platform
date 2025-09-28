# Clinical Data Platform - Makefile
# Production-ready makefile with cloud and local deployment options

.PHONY: help setup clean test lint security deploy demo-local demo-cloud

# Default environment
ENVIRONMENT ?= dev
AWS_REGION ?= us-east-1
PROJECT_NAME ?= clinical-platform

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Clinical Data Platform - Available Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Environment Variables:$(NC)"
	@echo "  ENVIRONMENT=$(ENVIRONMENT)"
	@echo "  AWS_REGION=$(AWS_REGION)"
	@echo "  PROJECT_NAME=$(PROJECT_NAME)"

# =============================================================================
# SETUP AND DEPENDENCIES
# =============================================================================

setup: ## Install pinned dependencies with pip
	@echo "$(BLUE)Installing dependencies with pip...$(NC)"
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	@echo "$(GREEN)Dependencies installed. Consider \"make setup.poetry\" for the legacy workflow.$(NC)"

setup.poetry: ## Legacy Poetry-based setup
	@echo "$(BLUE)Setting up development environment with Poetry...$(NC)"
	@command -v poetry >/dev/null 2>&1 || { echo "$(RED)Poetry not found. Please install: https://python-poetry.org/docs/#installation$(NC)"; exit 1; }
	poetry install --with dev
	poetry run pre-commit install
	@echo "$(GREEN)Development environment ready!$(NC)"

clean: ## Clean up generated files and caches
	@echo "$(BLUE)Cleaning up...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/ dist/ build/ data/processed reports docs/figures
	rm -f docs/index.html docs/cp-tox-mini_report.html docs/model_card.md manifests/data_manifest.json
	rm -f reports/index.html reports/cp-tox-mini_report.html reports/cp-tox-mini_report.md reports/model_card.md reports/model_metrics.json reports/leakage.json reports/ic50_summary.json
	@echo "$(GREEN)Cleanup complete!$(NC)"

# =============================================================================
# TESTING AND VALIDATION
# =============================================================================

test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	poetry run pytest tests/ -v --cov=src/clinical_data_platform --cov-report=html --cov-report=term

test.fast: ## Run fast tests only
	@echo "$(BLUE)Running fast tests...$(NC)"
	poetry run pytest tests/ -v -m "not slow"

test.compliance: ## Run compliance and PHI redaction tests
	@echo "$(BLUE)Running compliance tests...$(NC)"
	poetry run pytest tests/test_compliance.py tests/test_readonly.py -v

lint: ## Run code quality checks
	@echo "$(BLUE)Running linting...$(NC)"
	poetry run ruff check src/ tests/
	poetry run black --check src/ tests/
	poetry run mypy src/

lint.fix: ## Fix linting issues automatically
	@echo "$(BLUE)Fixing linting issues...$(NC)"
	poetry run ruff check --fix src/ tests/
	poetry run black src/ tests/

security: ## Run security analysis
	@echo "$(BLUE)Running security analysis...$(NC)"
	poetry run bandit -r src/ -f json -o bandit-report.json
	poetry run safety check --json --output safety-report.json || true
	@echo "$(GREEN)Security reports generated: bandit-report.json, safety-report.json$(NC)"

# =============================================================================
# DATA PIPELINE
# =============================================================================

data.generate: ## Generate synthetic data
	@echo "$(BLUE)Generating synthetic data...$(NC)"
	poetry run python scripts/generate_synthetic_data.py
	@echo "$(GREEN)Synthetic data generated in data/$(NC)"

data.validate: ## Run data validation with Great Expectations
	@echo "$(BLUE)Running data validation...$(NC)"
	poetry run great_expectations checkpoint run validate_clinical_data

validate.bad: ## Test validation with bad data (should fail)
	@echo "$(BLUE)Testing validation with bad data...$(NC)"
	python validation/run_validation.py --data analytics/dbt/seeds/visits.csv

validate.good: ## Test validation with good data (should pass)
	@echo "$(BLUE)Testing validation with good data...$(NC)"
	python validation/run_validation.py --data analytics/dbt/seeds/visits_good.csv

dbt.run: ## Run dbt models
	@echo "$(BLUE)Running dbt models...$(NC)"
	poetry run dbt run --project-dir dbt_project/ --target local

dbt.test: ## Run dbt tests
	@echo "$(BLUE)Running dbt tests...$(NC)"
	poetry run dbt test --project-dir dbt_project/ --target local

dbt.docs: ## Generate dbt documentation
	@echo "$(BLUE)Generating dbt documentation...$(NC)"
	poetry run dbt docs generate --project-dir dbt_project/ --target local
	poetry run dbt docs serve --project-dir dbt_project/ --port 8080

pipeline.local: data.generate dbt.run dbt.test data.validate ## Run complete data pipeline locally
	@echo "$(GREEN)Local data pipeline completed successfully!$(NC)"

# =============================================================================
# MACHINE LEARNING
# =============================================================================

ml.train: ## Train machine learning models
	@echo "$(BLUE)Training ML models...$(NC)"
	poetry run python -m clinical_data_platform.ml.train

ml.evaluate: ## Evaluate trained models
	@echo "$(BLUE)Evaluating ML models...$(NC)"
	poetry run python -m clinical_data_platform.ml.evaluate

ml.register: ## Register model with MLflow
	@echo "$(BLUE)Registering model with MLflow...$(NC)"
	poetry run python -m clinical_data_platform.ml.registry

# =============================================================================
# API AND SERVICES
# =============================================================================

api.dev: ## Run API in development mode
	@echo "$(BLUE)Starting API in development mode...$(NC)"
	poetry run uvicorn clinical_data_platform.api.main:app --reload --host 0.0.0.0 --port 8000

api.readonly: ## Run API in read-only mode
	@echo "$(BLUE)Starting API in read-only mode...$(NC)"
	READ_ONLY_MODE=1 LOG_SCRUB_VALUES=1 poetry run uvicorn clinical_data_platform.api.main:app --host 0.0.0.0 --port 8000

ui: ## Run Streamlit dashboard
	@echo "$(BLUE)Starting Streamlit dashboard...$(NC)"
	poetry run streamlit run ui/dashboard.py --server.port 8501

# =============================================================================
# DOCKER AND CONTAINERS
# =============================================================================

docker.build: ## Build all Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	docker build -f docker/api.Dockerfile -t clinical-platform-api:latest .
	docker build -f docker/dbt.Dockerfile -t clinical-platform-dbt:latest .
	docker build -f docker/worker.Dockerfile -t clinical-platform-worker:latest .
	@echo "$(GREEN)Docker images built successfully!$(NC)"

docker.run.api: ## Run API in Docker container
	@echo "$(BLUE)Running API in Docker...$(NC)"
	docker run -p 8000:8000 --env-file .env clinical-platform-api:latest

# =============================================================================
# CLOUD INFRASTRUCTURE
# =============================================================================

infra.init: ## Initialize Terraform
	@echo "$(BLUE)Initializing Terraform...$(NC)"
	cd infra/terraform && terraform init

infra.plan: ## Plan Terraform changes
	@echo "$(BLUE)Planning Terraform changes...$(NC)"
	cd infra/terraform && terraform plan \
		-var="environment=$(ENVIRONMENT)" \
		-var="project_name=$(PROJECT_NAME)" \
		-var="aws_region=$(AWS_REGION)"

infra.apply: ## Apply Terraform changes
	@echo "$(BLUE)Applying Terraform changes...$(NC)"
	cd infra/terraform && terraform apply \
		-var="environment=$(ENVIRONMENT)" \
		-var="project_name=$(PROJECT_NAME)" \
		-var="aws_region=$(AWS_REGION)"

infra.destroy: ## Destroy Terraform infrastructure
	@echo "$(RED)Destroying Terraform infrastructure...$(NC)"
	@read -p "Are you sure you want to destroy the infrastructure? (yes/no): " confirm && [ "$$confirm" = "yes" ]
	cd infra/terraform && terraform destroy \
		-var="environment=$(ENVIRONMENT)" \
		-var="project_name=$(PROJECT_NAME)" \
		-var="aws_region=$(AWS_REGION)"

secrets.bootstrap: ## Bootstrap cloud secrets
	@echo "$(BLUE)Bootstrapping cloud secrets...$(NC)"
	@command -v aws >/dev/null 2>&1 || { echo "$(RED)AWS CLI not found. Please install and configure.$(NC)"; exit 1; }
	chmod +x scripts/bootstrap_cloud.sh
	./scripts/bootstrap_cloud.sh $(ENVIRONMENT)

# =============================================================================
# DEMO WORKFLOWS (Legacy compatibility)
# =============================================================================

demo: demo.dbt demo.schema demo.ml demo.api demo.gif ## Run complete demo (legacy)
	@echo "Demo artifacts in docs/assets/demo/"

demo.dbt: ## Generate dbt demo artifacts
	python scripts/run_dbt_and_capture.py

demo.schema: ## Generate schema diagram
	python scripts/generate_star_schema_diagram.py

demo.ml: ## Run ML demo
	python -m pip install 'mlflow>=2.15' 'scikit-learn>=1.5' 'matplotlib>=3.8'
	python scripts/run_demo_mlflow.py

demo.api: ## Exercise API and capture outputs
	python -m pip install 'fastapi>=0.111' 'uvicorn>=0.30' 'httpx>=0.27'
	python scripts/exercise_api_and_capture.py

demo.gif: ## Create demo GIF
	python -m pip install pillow imageio
	python scripts/make_demo_gif.py

demo.clean: ## Clean demo artifacts
	rm -rf data/demo.duckdb mlruns
	rm -rf docs/assets/demo/*

# =============================================================================
# DEMO WORKFLOWS (New cloud-ready)
# =============================================================================

demo-local: setup data.generate dbt.run ml.train ## Run complete local demo
	@echo "$(GREEN)Starting local demo...$(NC)"
	@echo "$(BLUE)1. Generated synthetic data$(NC)"
	@echo "$(BLUE)2. Ran dbt transformations$(NC)"
	@echo "$(BLUE)3. Trained ML models$(NC)"
	@echo ""
	@echo "$(GREEN)Demo ready! Start services:$(NC)"
	@echo "  make api.dev    # API on http://localhost:8000"
	@echo "  make ui         # Dashboard on http://localhost:8501"
	@echo ""

demo-cloud: secrets.bootstrap infra.apply docker.build ## Deploy complete cloud demo
	@echo "$(GREEN)Cloud demo deployment initiated!$(NC)"
	@echo ""
	@echo "$(BLUE)Next steps:$(NC)"
	@echo "1. Push code to main branch to trigger GitHub Actions"
	@echo "2. Monitor deployment in GitHub Actions"
	@echo "3. Access API at: $$(cd infra/terraform && terraform output -raw api_url 2>/dev/null || echo 'Run terraform apply first')"
	@echo ""

# =============================================================================
# MAINTENANCE AND UTILITIES
# =============================================================================

logs.api: ## View API logs (cloud)
	@echo "$(BLUE)Viewing API logs...$(NC)"
	aws logs tail /ecs/$(PROJECT_NAME)-$(ENVIRONMENT) --follow --filter-pattern="api"

logs.dbt: ## View dbt logs (cloud)
	@echo "$(BLUE)Viewing dbt logs...$(NC)"
	aws logs tail /ecs/$(PROJECT_NAME)-$(ENVIRONMENT) --follow --filter-pattern="dbt"

status: ## Check system status
	@echo "$(BLUE)System Status Check$(NC)"
	@echo ""
	@echo "$(YELLOW)Local Environment:$(NC)"
	@command -v poetry >/dev/null 2>&1 && echo "✅ Poetry installed" || echo "❌ Poetry not found"
	@command -v docker >/dev/null 2>&1 && echo "✅ Docker installed" || echo "❌ Docker not found"
	@command -v aws >/dev/null 2>&1 && echo "✅ AWS CLI installed" || echo "❌ AWS CLI not found"
	@command -v terraform >/dev/null 2>&1 && echo "✅ Terraform installed" || echo "❌ Terraform not found"
	@echo ""
	@echo "$(YELLOW)Data:$(NC)"
	@[ -d "data/raw" ] && echo "✅ Raw data exists" || echo "❌ No raw data found"
	@[ -d "data/silver" ] && echo "✅ Silver data exists" || echo "❌ No silver data found"
	@[ -d "data/gold" ] && echo "✅ Gold data exists" || echo "❌ No gold data found"
	@echo ""
	@echo "$(YELLOW)Infrastructure:$(NC)"
	@[ -f "infra/terraform/terraform.tfstate" ] && echo "✅ Terraform state exists" || echo "❌ No Terraform state found"
	@echo ""

docs: ## Generate and serve documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	poetry run mkdocs serve --dev-addr 0.0.0.0:8002


# =============================================================================
# CP-TOX MINI PIPELINE
# =============================================================================

.PHONY: data features fuse train eval diagnostics ic50 report

data: ## Download deterministic CP tox inputs
	python -m cp_tox_mini.cli download

features: ## Build CP + chem features
	python -m cp_tox_mini.cli features

fuse: ## Fuse modalities and create train/test splits
	python -m cp_tox_mini.cli fuse

train: ## Train the CP tox baseline model
	python -m cp_tox_mini.cli train

# Retain eval verb to align with CLI naming conventions
eval: ## Evaluate the CP tox model and emit metrics/figures
	python -m cp_tox_mini.cli eval

diagnostics: ## Run leakage probes and permutation diagnostics
	python -m cp_tox_mini.cli diagnostics

ic50: ## Estimate IC50 on the example dose-response curve
	python -m cp_tox_mini.cli ic50

report: ## Build Markdown/HTML reports and publish to docs/
	python -m cp_tox_mini.cli report

# =============================================================================
# SHORTCUTS AND ALIASES
# =============================================================================

dev: api.dev ## Alias for api.dev
build: docker.build ## Alias for docker.build
deploy: demo-cloud ## Alias for demo-cloud
local: demo-local ## Alias for demo-local

# =============================================================================
# VALIDATION TARGETS
# =============================================================================

check: lint test security ## Run all checks (lint, test, security)
	@echo "$(GREEN)All checks passed!$(NC)"

ci: check data.validate ## Run CI pipeline locally
	@echo "$(GREEN)CI pipeline completed successfully!$(NC)"

# Default target
all: ## Run the deterministic CP tox mini pipeline
	python -m cp_tox_mini.cli all

.PHONY: obsv.up obsv.down obsv.status obsv.demo

obsv.up: ## Start local observability stack (Grafana, Loki, Tempo, OTel collector)
	docker compose -f observability/docker-compose.obsv.yml up -d

obsv.down: ## Stop observability stack and remove volumes
	docker compose -f observability/docker-compose.obsv.yml down -v

obsv.status: ## Show observability stack status
	docker compose -f observability/docker-compose.obsv.yml ps

obsv.demo: ## Run instrumented demo pipelines and emit freshness SLI
	python -m src.pipelines.ingest_demo
	python -m src.pipelines.dbt_demo
	python -m src.pipelines.train_demo
	python -m src.common.freshness --path data/silver/_last_update.txt --slo-minutes 120
