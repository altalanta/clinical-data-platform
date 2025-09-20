-- DuckDB Star Schema for Clinical Warehouse
-- Dimensions
CREATE TABLE IF NOT EXISTS dim_study (
  study_sk INTEGER PRIMARY KEY,
  study_id VARCHAR UNIQUE NOT NULL,
  title VARCHAR,
  phase VARCHAR,
  start_date DATE,
  end_date DATE,
  COMMENT 'Study metadata'
);

CREATE TABLE IF NOT EXISTS dim_subject (
  subject_sk INTEGER PRIMARY KEY,
  subject_id VARCHAR UNIQUE NOT NULL,
  study_sk INTEGER NOT NULL REFERENCES dim_study(study_sk),
  arm VARCHAR,
  sex VARCHAR,
  age INTEGER,
  COMMENT 'Subject demographics'
);

CREATE TABLE IF NOT EXISTS dim_visit (
  visit_sk INTEGER PRIMARY KEY,
  visit_id VARCHAR UNIQUE NOT NULL,
  visit_day INTEGER,
  visit_name VARCHAR,
  COMMENT 'Visit/timepoint'
);

CREATE TABLE IF NOT EXISTS dim_measurement (
  measurement_sk INTEGER PRIMARY KEY,
  code VARCHAR NOT NULL,
  name VARCHAR,
  category VARCHAR,
  unit VARCHAR,
  COMMENT 'Lab/Vital measurement dictionary'
);

-- Facts
CREATE TABLE IF NOT EXISTS fact_adverse_events (
  subject_sk INTEGER REFERENCES dim_subject(subject_sk),
  study_sk INTEGER REFERENCES dim_study(study_sk),
  ae_start DATE,
  ae_end DATE,
  severity VARCHAR,
  seriousness BOOLEAN,
  outcome VARCHAR,
  PRIMARY KEY (subject_sk, ae_start)
);

CREATE TABLE IF NOT EXISTS fact_labs (
  subject_sk INTEGER REFERENCES dim_subject(subject_sk),
  study_sk INTEGER REFERENCES dim_study(study_sk),
  visit_sk INTEGER REFERENCES dim_visit(visit_sk),
  measurement_sk INTEGER REFERENCES dim_measurement(measurement_sk),
  value DOUBLE,
  low_norm DOUBLE,
  high_norm DOUBLE,
  abnormal_flag VARCHAR,
  PRIMARY KEY (subject_sk, visit_sk, measurement_sk)
);

CREATE TABLE IF NOT EXISTS fact_vitals (
  subject_sk INTEGER REFERENCES dim_subject(subject_sk),
  study_sk INTEGER REFERENCES dim_study(study_sk),
  visit_sk INTEGER REFERENCES dim_visit(visit_sk),
  measurement_sk INTEGER REFERENCES dim_measurement(measurement_sk),
  value DOUBLE,
  PRIMARY KEY (subject_sk, visit_sk, measurement_sk)
);

CREATE TABLE IF NOT EXISTS fact_exposure (
  subject_sk INTEGER REFERENCES dim_subject(subject_sk),
  study_sk INTEGER REFERENCES dim_study(study_sk),
  drug VARCHAR,
  dose DOUBLE,
  start_date DATE,
  end_date DATE,
  PRIMARY KEY (subject_sk, drug, start_date)
);

-- Indexing hints
-- DuckDB supports zone maps and efficient Parquet scanning; cluster by common filters
-- Example: CREATE INDEX IF NOT EXISTS idx_ae_study ON fact_adverse_events(study_sk);

