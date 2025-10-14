# Cohort Card

## Purpose
- Assemble an adult cardio-metabolic population with sufficient utilization history for survival modeling.
- Provide deterministic inclusion/exclusion logic for benchmarking feature pipelines.

## Inclusion Criteria
- Age ≥ 18 at the derived index date.
- ≥1 encounter in the 365-day lookback window.
- ≤120 days since the most recent visit before index to proxy continuous engagement.

## Exclusion Criteria
- None; synthetic population is fully retained for transparency.

## Risks & Biases
- Synthetic race / ethnicity distributions loosely mimic U.S. population; they may not represent local site demographics.
- Encounter-driven continuity proxy may undercount patients with virtual visits or transferred care.
- Deterministic lab mappings limit clinical nuance (e.g., missing pediatrics, obstetrics).

## Leakage Mitigation
- Survival labels censor at 365 days and use only lookback features.
- Labs and diagnoses restricted to the pre-index window.
- Post-index encounters excluded from feature aggregation except for follow-up QA metrics.

## Operational Notes
- Regenerate data with `make data` (seeded at 1337) to maintain determinism.
- Review `scripts/validate_metrics.py` output after each build for guardrail thresholds.
