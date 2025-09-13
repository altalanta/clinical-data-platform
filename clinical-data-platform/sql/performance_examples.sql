-- Demonstrates window functions, filtered aggregations, and predicate pushdown.

-- Window function: running count of AEs by subject over time
WITH ae AS (
  SELECT subject_sk, ae_start, severity
  FROM fact_adverse_events
)
SELECT subject_sk,
       ae_start,
       COUNT(*) OVER (PARTITION BY subject_sk ORDER BY ae_start ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_ae_count
FROM ae
ORDER BY subject_sk, ae_start;

-- Filtered aggregation: AE rates by arm
SELECT s.arm,
       AVG(CASE WHEN f.severity IN ('SEVERE','SERIOUS') THEN 1 ELSE 0 END)::DOUBLE AS severe_rate
FROM fact_adverse_events f
JOIN dim_subject s USING(subject_sk)
GROUP BY s.arm
ORDER BY severe_rate DESC;

-- Predicate pushdown: restrict to a single study for partition pruning
SELECT *
FROM fact_labs
JOIN dim_study USING(study_sk)
WHERE dim_study.study_id = 'STUDY001';

