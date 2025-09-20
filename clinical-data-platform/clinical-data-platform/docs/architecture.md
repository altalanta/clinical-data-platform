# Architecture

```mermaid
flowchart LR
  A[Raw CSV (SDTM-like)] -->|ingest_csv.py| B[(MinIO S3 - raw)]
  B -->|parquet-ize| C[(MinIO S3 - bronze)]
  C -->|cdisc_sdtm_mapping| D[(MinIO S3 - silver)]
  D -->|loaders.py| E[(DuckDB Warehouse)]
  E -->|dbt| F[(Marts)]
  F -->|queries.py| G[Analytics Outputs]
  F --> H[API]
  F --> I[Dashboard]
  D --> J[ML Features]
  J --> K[ML Training/Scoring]
```

## C4 (Container Level)

```mermaid
C4Context
title Clinical Data Platform (Container Diagram)
Person(user, "Analyst/Scientist")
Container_Boundary(b1, "Local/AWS"){
  Container(s3, "Object Store", "MinIO/AWS S3")
  Container(db, "Warehouse", "DuckDB")
  Container(app, "Application", "FastAPI/Streamlit")
  Container(dbt, "Transformation", "dbt-duckdb")
  Container(ml, "ML Tracking", "MLflow")
}
Rel(user, app, "use")
Rel(app, db, "query")
Rel(dbt, db, "build models")
Rel(ml, app, "log / query runs")
Rel(app, s3, "read/write Parquet")
```

