-- =============================================================================
-- Clinical Data Platform - Warehouse DDL for DuckDB
-- Production-ready schema aligned with dbt models
-- =============================================================================

-- Create schemas
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS staging; 
CREATE SCHEMA IF NOT EXISTS intermediate;
CREATE SCHEMA IF NOT EXISTS marts;
CREATE SCHEMA IF NOT EXISTS clinical_marts;
CREATE SCHEMA IF NOT EXISTS analytics_marts;
CREATE SCHEMA IF NOT EXISTS seeds;
CREATE SCHEMA IF NOT EXISTS snapshots;

-- =============================================================================
-- RAW DATA VIEWS (External tables reading from Parquet)
-- =============================================================================

-- Demographics domain
CREATE OR REPLACE VIEW raw.dm AS
SELECT * FROM read_parquet('data/sample_standardized/DM.parquet');

-- Adverse Events domain
CREATE OR REPLACE VIEW raw.ae AS  
SELECT * FROM read_parquet('data/sample_standardized/AE.parquet');

-- Laboratory domain
CREATE OR REPLACE VIEW raw.lb AS
SELECT * FROM read_parquet('data/sample_standardized/LB.parquet');

-- Vital Signs domain
CREATE OR REPLACE VIEW raw.vs AS
SELECT * FROM read_parquet('data/sample_standardized/VS.parquet');

-- Exposure domain
CREATE OR REPLACE VIEW raw.ex AS
SELECT * FROM read_parquet('data/sample_standardized/EX.parquet');

-- =============================================================================
-- STAGING LAYER (Cleaned and standardized)
-- =============================================================================

-- These views will be created by dbt, but we define the expected structure

CREATE OR REPLACE VIEW staging._temp_demographics_structure AS
SELECT
    studyid::VARCHAR as studyid,
    subjid::VARCHAR as subjid,
    treatment_arm::VARCHAR as treatment_arm,
    sex::VARCHAR as sex,
    age::INTEGER as age,
    age_group::VARCHAR as age_group,
    sex_desc::VARCHAR as sex_desc,
    dbt_updated_at::TIMESTAMP as dbt_updated_at,
    dbt_study_id::VARCHAR as dbt_study_id
FROM (VALUES ('STUDY001', 'SUBJ001', 'ARM1', 'M', 45, 'ADULT', 'MALE', NOW(), 'STUDY001')) 
AS t(studyid, subjid, treatment_arm, sex, age, age_group, sex_desc, dbt_updated_at, dbt_study_id)
WHERE 1=0; -- Empty structure template

-- =============================================================================
-- CLINICAL MARTS (Final business tables)
-- =============================================================================

-- Subject outcomes fact table
CREATE TABLE IF NOT EXISTS clinical_marts.fact_subject_outcomes (
    subject_key VARCHAR PRIMARY KEY,
    studyid VARCHAR NOT NULL,
    subjid VARCHAR UNIQUE NOT NULL,
    treatment_arm VARCHAR NOT NULL,
    sex VARCHAR NOT NULL,
    sex_desc VARCHAR NOT NULL,
    age INTEGER NOT NULL,
    age_group VARCHAR NOT NULL,
    
    -- Adverse events metrics
    total_adverse_events INTEGER DEFAULT 0,
    serious_adverse_events INTEGER DEFAULT 0,
    severe_adverse_events INTEGER DEFAULT 0,
    ongoing_adverse_events INTEGER DEFAULT 0,
    has_serious_adverse_event BOOLEAN DEFAULT FALSE,
    max_ae_severity_rank INTEGER DEFAULT 0,
    safety_risk_category VARCHAR NOT NULL,
    
    -- Laboratory metrics
    total_lab_tests INTEGER DEFAULT 0,
    unique_lab_tests INTEGER DEFAULT 0,
    abnormal_lab_results INTEGER DEFAULT 0,
    abnormal_lab_rate DECIMAL(5,3) DEFAULT 0,
    lab_categories_tested INTEGER DEFAULT 0,
    
    -- Vital signs metrics  
    total_vital_measurements INTEGER DEFAULT 0,
    unique_vital_tests INTEGER DEFAULT 0,
    abnormal_vitals INTEGER DEFAULT 0,
    abnormal_vital_rate DECIMAL(5,3) DEFAULT 0,
    
    -- Treatment exposure
    total_exposures INTEGER DEFAULT 0,
    max_treatment_duration INTEGER DEFAULT 0,
    ongoing_treatments INTEGER DEFAULT 0,
    treatment_categories VARCHAR,
    
    -- Data quality indicators
    has_ae_data BOOLEAN DEFAULT FALSE,
    has_lab_data BOOLEAN DEFAULT FALSE,
    has_vital_data BOOLEAN DEFAULT FALSE,
    has_exposure_data BOOLEAN DEFAULT FALSE,
    data_completeness_score DECIMAL(3,2) DEFAULT 0,
    
    -- Clinical profile
    clinical_profile VARCHAR NOT NULL,
    participation_quality VARCHAR NOT NULL,
    
    -- Audit fields
    record_created_at TIMESTAMP DEFAULT NOW(),
    dbt_updated_at TIMESTAMP,
    dbt_study_id VARCHAR
);

-- =============================================================================
-- ANALYTICS MARTS (Aggregated metrics)
-- =============================================================================

-- Study overview dimension
CREATE TABLE IF NOT EXISTS analytics_marts.dim_study_overview (
    study_id VARCHAR PRIMARY KEY,
    study_name VARCHAR NOT NULL,
    
    -- Enrollment metrics
    total_subjects INTEGER NOT NULL,
    treatment_arms_count INTEGER NOT NULL,
    
    -- Demographics distribution
    male_subjects INTEGER DEFAULT 0,
    female_subjects INTEGER DEFAULT 0,
    pediatric_subjects INTEGER DEFAULT 0,
    adult_subjects INTEGER DEFAULT 0,
    elderly_subjects INTEGER DEFAULT 0,
    
    -- Age statistics
    mean_age DECIMAL(5,2),
    median_age DECIMAL(5,2),
    min_age INTEGER,
    max_age INTEGER,
    
    -- Safety metrics
    total_adverse_events_study INTEGER DEFAULT 0,
    total_serious_aes_study INTEGER DEFAULT 0,
    subjects_with_serious_aes INTEGER DEFAULT 0,
    serious_ae_rate_percent DECIMAL(5,2) DEFAULT 0,
    safety_concern_subjects INTEGER DEFAULT 0,
    safety_concern_rate_percent DECIMAL(5,2) DEFAULT 0,
    
    -- Data quality metrics
    avg_data_completeness DECIMAL(5,4) DEFAULT 0,
    avg_data_completeness_percent DECIMAL(5,2) DEFAULT 0,
    high_quality_subjects INTEGER DEFAULT 0,
    medium_quality_subjects INTEGER DEFAULT 0,
    low_quality_subjects INTEGER DEFAULT 0,
    high_quality_rate_percent DECIMAL(5,2) DEFAULT 0,
    
    -- Clinical profiles
    normal_profile_subjects INTEGER DEFAULT 0,
    multiple_abnormalities_subjects INTEGER DEFAULT 0,
    
    -- Treatment metrics
    avg_treatment_duration DECIMAL(8,2) DEFAULT 0,
    max_treatment_duration_overall INTEGER DEFAULT 0,
    subjects_with_ongoing_treatment INTEGER DEFAULT 0,
    
    -- Laboratory metrics
    total_lab_tests_study INTEGER DEFAULT 0,
    avg_abnormal_lab_rate DECIMAL(5,3) DEFAULT 0,
    
    -- Vital signs metrics
    total_vital_measurements_study INTEGER DEFAULT 0,
    avg_abnormal_vital_rate DECIMAL(5,3) DEFAULT 0,
    
    -- Calculated percentages
    male_percentage DECIMAL(5,2) DEFAULT 0,
    female_percentage DECIMAL(5,2) DEFAULT 0,
    
    -- Status assessments
    study_status VARCHAR DEFAULT 'UNKNOWN',
    overall_safety_profile VARCHAR DEFAULT 'UNKNOWN',
    data_quality_assessment VARCHAR DEFAULT 'UNKNOWN',
    
    -- Audit fields
    analysis_timestamp TIMESTAMP DEFAULT NOW()
);

-- =============================================================================
-- PERFORMANCE INDEXES
-- =============================================================================

-- Primary indexes for fact table
CREATE UNIQUE INDEX IF NOT EXISTS idx_fact_subject_outcomes_subjid 
ON clinical_marts.fact_subject_outcomes(subjid);

CREATE INDEX IF NOT EXISTS idx_fact_subject_outcomes_treatment_arm 
ON clinical_marts.fact_subject_outcomes(treatment_arm);

CREATE INDEX IF NOT EXISTS idx_fact_subject_outcomes_safety_category 
ON clinical_marts.fact_subject_outcomes(safety_risk_category);

CREATE INDEX IF NOT EXISTS idx_fact_subject_outcomes_age_group 
ON clinical_marts.fact_subject_outcomes(age_group);

CREATE INDEX IF NOT EXISTS idx_fact_subject_outcomes_clinical_profile
ON clinical_marts.fact_subject_outcomes(clinical_profile);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_fact_subject_outcomes_treatment_safety
ON clinical_marts.fact_subject_outcomes(treatment_arm, safety_risk_category);

CREATE INDEX IF NOT EXISTS idx_fact_subject_outcomes_age_sex
ON clinical_marts.fact_subject_outcomes(age_group, sex);

-- =============================================================================
-- UTILITY VIEWS FOR ANALYTICS
-- =============================================================================

CREATE OR REPLACE VIEW analytics_marts.v_safety_summary AS
SELECT
    treatment_arm,
    count(*) as subject_count,
    sum(total_adverse_events) as total_aes,
    sum(serious_adverse_events) as total_serious_aes,
    round(avg(total_adverse_events::DECIMAL), 2) as avg_aes_per_subject,
    round(sum(serious_adverse_events)::DECIMAL / count(*) * 100, 1) as serious_ae_rate_percent,
    count(case when safety_risk_category = 'HIGH_RISK' then 1 end) as high_risk_subjects,
    count(case when has_serious_adverse_event then 1 end) as subjects_with_serious_aes
FROM clinical_marts.fact_subject_outcomes
GROUP BY treatment_arm
ORDER BY treatment_arm;

CREATE OR REPLACE VIEW analytics_marts.v_demographics_summary AS
SELECT
    treatment_arm,
    sex_desc,
    age_group,
    count(*) as subject_count,
    round(avg(age::DECIMAL), 1) as mean_age,
    min(age) as min_age,
    max(age) as max_age,
    round(count(*)::DECIMAL / sum(count(*)) OVER (PARTITION BY treatment_arm) * 100, 1) as percentage_in_arm
FROM clinical_marts.fact_subject_outcomes
GROUP BY treatment_arm, sex_desc, age_group
ORDER BY treatment_arm, sex_desc, age_group;

CREATE OR REPLACE VIEW analytics_marts.v_data_quality_summary AS
SELECT
    participation_quality,
    count(*) as subject_count,
    round(avg(data_completeness_score) * 100, 1) as avg_completeness_percent,
    round(avg(total_lab_tests::DECIMAL), 1) as avg_lab_tests,
    round(avg(total_vital_measurements::DECIMAL), 1) as avg_vital_measurements,
    round(avg(total_adverse_events::DECIMAL), 1) as avg_adverse_events
FROM clinical_marts.fact_subject_outcomes
GROUP BY participation_quality
ORDER BY 
    CASE participation_quality 
        WHEN 'HIGH_QUALITY' THEN 1
        WHEN 'MEDIUM_QUALITY' THEN 2 
        WHEN 'LOW_QUALITY' THEN 3
        ELSE 4
    END;

-- =============================================================================
-- DATA QUALITY CONSTRAINTS (Examples)
-- =============================================================================

-- Note: DuckDB has limited constraint support, these serve as documentation

-- Example business rules that should be enforced:
-- - serious_adverse_events <= total_adverse_events
-- - severe_adverse_events <= total_adverse_events  
-- - abnormal_lab_results <= total_lab_tests
-- - abnormal_vitals <= total_vital_measurements
-- - age between 0 and 150
-- - data_completeness_score between 0 and 1
-- - abnormal_lab_rate between 0 and 1
-- - abnormal_vital_rate between 0 and 1

-- =============================================================================
-- SCHEMA DOCUMENTATION
-- =============================================================================

COMMENT ON SCHEMA clinical_marts IS 'Production clinical data marts containing subject-level facts and dimensions';
COMMENT ON SCHEMA analytics_marts IS 'Analytics-focused aggregations and summary tables for reporting';
COMMENT ON SCHEMA staging IS 'Cleaned and standardized data layer (managed by dbt)';
COMMENT ON SCHEMA raw IS 'External table views reading directly from parquet files';

COMMENT ON TABLE clinical_marts.fact_subject_outcomes IS 'Core fact table with comprehensive subject-level clinical outcomes, safety metrics, and data quality indicators';
COMMENT ON TABLE analytics_marts.dim_study_overview IS 'Study-level aggregations and key performance indicators for high-level study analysis';

-- =============================================================================
-- INITIALIZATION COMPLETE
-- =============================================================================

SELECT 
    'Clinical Data Warehouse DDL initialized successfully' as status,
    current_timestamp as executed_at,
    version() as duckdb_version;

