POETRY?=poetry
PY?=$(POETRY) run python
PIP?=$(POETRY) run pip

.PHONY: setup data minio ingest dbt analytics train api ui demo test lint typecheck security docs clean

setup:
	$(POETRY) install --no-root
	$(POETRY) run pre-commit install

data:
	$(PY) scripts/generate_synthetic_data.py --out data/sample_raw --rows 200 --seed 42

minio:
	docker compose up -d minio
	$(PY) scripts/seed_minio.py

ingest:
	$(PY) -m clinical_platform.ingestion.flows run-local

dbt:
	DBT_PROFILES_DIR=dbt $(POETRY) run dbt --project-dir dbt/clinical_dbt deps
	DBT_PROFILES_DIR=dbt $(POETRY) run dbt --project-dir dbt/clinical_dbt build

analytics:
	$(PY) -m clinical_platform.analytics.queries --out data/analytics

train:
	$(PY) -m clinical_platform.ml.train --data data/sample_standardized --out models

api:
	$(POETRY) run uvicorn clinical_platform.api.main:app --host 0.0.0.0 --port 8000 --reload

ui:
	$(POETRY) run streamlit run src/clinical_platform/ui/dashboard.py --server.port 8501

demo: data minio ingest dbt analytics train
	@echo "Opening dashboard at http://localhost:8501"

test:
	$(POETRY) run pytest --cov-fail-under=85

lint:
	$(POETRY) run ruff check .
	$(POETRY) run black --check .

typecheck:
	$(POETRY) run mypy src

security:
	$(POETRY) run bandit -r src -ll

docs:
	$(POETRY) run mkdocs build --strict

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache htmlcov .coverage* mlruns

