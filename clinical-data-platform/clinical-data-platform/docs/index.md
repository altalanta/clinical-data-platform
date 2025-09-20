# Clinical Data Platform

This project demonstrates an end-to-end clinical data platform architecture and implementation with a local-first experience (DuckDB + MinIO) and AWS-ready design.

Highlights:
- Deterministic synthetic SDTM-like data (DM/AE/LB/VS/EX)
- Ingestion to S3-like object store, Parquet bronze→silver
- SDTM mapping and validation (pandera + Great Expectations)
- Warehouse modeling (star schema) and dbt marts/tests
- Analytics queries + basic statistical methods
- ML slice with MLflow tracking
- API (FastAPI) and Dashboard (Streamlit)

