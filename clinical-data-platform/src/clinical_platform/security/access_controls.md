## IAM Role Model (Conceptual)

- Data Engineer: read/write raw, bronze, silver; manage dbt; limited prod write
- Analyst/Scientist: read silver, marts; no raw access
- Service/API: read marts; write to logging/monitoring buckets
- ML Training: read features; write models/artifacts to registry

Principles: least privilege, scoped KMS keys, separation of duties, break-glass access.

