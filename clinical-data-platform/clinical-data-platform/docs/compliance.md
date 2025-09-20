# Compliance Considerations (Non-legal)

This project is for demonstration only. For HIPAA/GDPR/GxP environments, you would:

- Access Controls: IAM roles, least privilege, S3 bucket policies, scoped KMS keys
- Audit & Traceability: structured JSON logs, request/lineage IDs, immutable audit trails
- Data Protection: PHI masking/tokenization, irreversible hashing, date shifting
- Validation & Change Control: SDLC with documented validation, PR reviews, approvals, tagged releases
- SOPs & Training: define/execute SOPs for data changes, model updates, deployments
- Vendor Qualification: assess cloud tools and validate processes (e.g., dbt/MLflow usage)

This repo demonstrates patterns only; it does not provide compliance guarantees.

