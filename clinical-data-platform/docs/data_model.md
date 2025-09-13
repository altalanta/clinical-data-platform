# Data Model

## SDTM-like → Star Schema

- Dimensions: `dim_subject`, `dim_study`, `dim_visit`, `dim_measurement`
- Facts: `fact_adverse_events`, `fact_labs`, `fact_vitals`, `fact_exposure`

Key mapping examples:
- DM (Demographics) → `dim_subject` (subject SK, study FK, arm, sex, age)
- AE → `fact_adverse_events` (subject FK, dates, severity, seriousness)
- LB → `fact_labs` (subject FK, test code, value, unit, normal range)
- VS → `fact_vitals` (subject FK, vital measure, value, unit)
- EX → `fact_exposure` (subject FK, treatment, dose, start/end)

ADaM ADSL (minimal) derived per-subject from SDTM to support ML features.

