# Model Card: Survival Modeling Features

## Intended Use
- Downstream models predicting 365-day mortality for cardio-metabolic patients.
- Benchmarking of feature engineering pipelines for survival analysis on synthetic data.

## Users
- Analytics engineers exploring dbt/DuckDB patterns.
- Data scientists prototyping survival models before applying to sensitive PHI environments.

## Data Sources
- Deterministic synthetic CSVs generated via `scripts/gen_synth_data.py`.
- Seed mappings for ICD→CCS and LOINC→lab categories.

## Feature Overview
- Demographics from `dim_patient` (age buckets, sex, race, ethnicity).
- Utilization metrics from `fct_features` (lookback encounter counts, procedure intensity).
- Lab recency and flag indicators from `int_lab_latest`.
- Diagnosis/procedure indicators capturing comorbidity burden.

## Performance Considerations
- Designed to run locally (<5 minutes, <2 GB RAM) on DuckDB.
- Determinism ensures reproducible row counts and event rates across runs with identical seeds.

## Limitations
- Synthetic mappings simplify coding systems; not exhaustive.
- Continuous enrollment proxy may not capture insurance churn.
- Event definition limited to all-cause mortality; no competing risks modeled.

## Fairness & Ethics
- Synthetic data avoids PHI leakage, but demographic distributions may not align with real populations and should not be used to infer bias metrics.
- Downstream fairness analysis should be re-run on real-world cohorts before deployment.

## Maintenance
- Update threshold checks in `scripts/validate_metrics.py` alongside major data distribution changes.
- Document schema evolution using dbt docs and versioned changelogs.
