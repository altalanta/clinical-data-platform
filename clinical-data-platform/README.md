# Clinical Data Platform

Production-quality sample project demonstrating an end-to-end clinical data platform: ingest → validate/standardize (SDTM-like) → warehouse (star schema) → dbt transforms → analytics + ML → dashboard + API. Cloud-ready (AWS-first) while runnable locally without external credentials using DuckDB + MinIO + local MLflow.

- Language: Python 3.11
- Storage/Compute: DuckDB, Parquet, (mocked) S3 via MinIO
- Transform: pandas/polars, dbt-duckdb
- Validation: pandera + Great Expectations
- ML: scikit-learn + MLflow
- API/UI: FastAPI + Streamlit
- Orchestration (optional, local only): Prefect
- CI/CD: GitHub Actions (lint/type/test/security/coverage, build Docker, docs)

## Quickstart

Prereqs: Docker, Docker Compose, Python 3.11, Poetry.

- make setup — create virtualenv, install deps, install pre-commit
- make data — generate tiny synthetic SDTM-like CSVs (DM/AE/LB/VS/EX)
- make minio — start MinIO + seed buckets
- make ingest — land→bronze→silver (Parquet) with logging + lineage
- make dbt — run dbt models + tests against DuckDB
- make analytics — run curated analytics queries into `data/analytics/`
- make train — train a simple ML model and log to MLflow
- make api — run FastAPI on http://localhost:8000
- make ui — run Streamlit on http://localhost:8501
- make demo — end-to-end local pipeline, then open dashboard

Everything runs locally by default; no external credentials required. To switch to AWS later, adjust `configs/config.aws.yaml` and env vars (see `.env.example`).

## Mapping to the Job Description

- Design and implement data models (SQL + dbt)
  - `sql/warehouse_ddl.sql`, `dbt/clinical_dbt/`, `src/clinical_platform/warehouse/`
- Data engineering + analytics for clinical trials
  - `src/clinical_platform/ingestion/`, `standards/`, `validation/`, `analytics/`
- Python, SQL, data modeling
  - Python code across `src/`, SQL in `sql/`, dbt models/tests in `dbt/`
- AWS (mocked locally) + CI/CD
  - MinIO S3 mocks (`docker-compose.yml`), `s3_client.py`, GitHub Actions in `.github/workflows/`
- Clean code + tests
  - `ruff`, `black`, `mypy`, `pytest` (+ coverage ≥85%), `bandit`, pre-commit hooks
- Clinical standards (CDISC SDTM/ADaM)
  - `standards/cdisc_sdtm_mapping.py`, `standards/sdtm_schemas/*.json`, `standards/adam/build_adsl.py`
- Dashboards and ML
  - `ui/dashboard.py` (Streamlit), `ml/train.py`, `ml/infer.py`, `ml/registry.py`
- Security/compliance awareness (GxP/HIPAA/GDPR)
  - `security/phi_redaction.py`, `configs/logging.yaml`, `docs/compliance.md`, IAM doc

## Local Configuration

- Configs: `configs/config.local.yaml` (default) and `configs/config.aws.yaml` (cloud)
- Logging: `configs/logging.yaml` (JSON logs; avoids data values by default)
- Env vars: `.env.example` lists toggles for local vs cloud and secrets for MinIO

## One-command Demo

`make demo` runs: data generation → start MinIO → ingest bronze→silver → dbt → analytics → train ML → launch API + dashboard.

## Troubleshooting

- DuckDB/Parquet: Ensure read/write permissions in `data/`
- MinIO fails to start: Ports 9000/9001 in use. Stop other services or change ports.
- dbt errors: Verify `dbt/profiles.yml.example` copied to `~/.dbt/profiles.yml` or set `DBT_PROFILES_DIR`.
- MLflow UI empty: Set `MLFLOW_TRACKING_URI` to `http://localhost:5000` (docker) or use local file `mlruns/` default.

## Contributing & SDLC

- Conventional commits; pre-commit hooks enforce format/lint
- CI: lint + typecheck + tests + coverage gate (>=85%) + bandit + docker build
- Docs: MkDocs Material + Mermaid, published via GitHub Pages

## License

MIT

