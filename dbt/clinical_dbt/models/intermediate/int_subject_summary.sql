{{ config(
    materialized='view',
    docs={'node_color': '#FFE4B5'}
) }}

with demographics as (
    select * from {{ ref('stg_demographics') }}
),

adverse_events as (
    select 
        subjid,
        count(*) as total_aes,
        sum(case when is_serious then 1 else 0 end) as serious_aes,
        sum(case when severity = 'SEVERE' then 1 else 0 end) as severe_aes,
        sum(case when is_ongoing then 1 else 0 end) as ongoing_aes,
        max(severity_rank) as max_severity_rank,
        count(distinct case when is_serious then subjid end) as has_serious_ae
    from {{ ref('stg_adverse_events') }}
    group by subjid
),

laboratory as (
    select 
        subjid,
        count(*) as total_lab_tests,
        count(distinct test_code) as unique_lab_tests,
        sum(case when result_flag = 'ABNORMAL' then 1 else 0 end) as abnormal_lab_results,
        count(distinct test_category) as lab_categories_tested
    from {{ ref('stg_laboratory') }}
    group by subjid
),

vital_signs as (
    select 
        subjid,
        count(*) as total_vital_measurements,
        count(distinct test_code) as unique_vital_tests,
        sum(case when clinical_flag not in ('NORMAL', 'N/A') then 1 else 0 end) as abnormal_vitals
    from {{ ref('stg_vital_signs') }}
    group by subjid
),

exposure as (
    select 
        subjid,
        count(*) as total_exposures,
        max(treatment_duration_days) as max_treatment_duration,
        count(case when is_ongoing_treatment then 1 end) as ongoing_treatments,
        string_agg(distinct treatment_category, ', ') as treatment_categories
    from {{ ref('stg_exposure') }}
    group by subjid
),

combined as (
    select
        d.studyid,
        d.subjid,
        d.treatment_arm,
        d.sex,
        d.sex_desc,
        d.age,
        d.age_group,
        
        -- Adverse events summary
        coalesce(ae.total_aes, 0) as total_adverse_events,
        coalesce(ae.serious_aes, 0) as serious_adverse_events,
        coalesce(ae.severe_aes, 0) as severe_adverse_events,
        coalesce(ae.ongoing_aes, 0) as ongoing_adverse_events,
        coalesce(ae.max_severity_rank, 0) as max_ae_severity_rank,
        case when ae.has_serious_ae > 0 then true else false end as has_serious_adverse_event,
        
        -- Laboratory summary
        coalesce(lb.total_lab_tests, 0) as total_lab_tests,
        coalesce(lb.unique_lab_tests, 0) as unique_lab_tests,
        coalesce(lb.abnormal_lab_results, 0) as abnormal_lab_results,
        coalesce(lb.lab_categories_tested, 0) as lab_categories_tested,
        case 
            when lb.total_lab_tests > 0 
            then round(lb.abnormal_lab_results::float / lb.total_lab_tests::float, 3)
            else 0
        end as abnormal_lab_rate,
        
        -- Vital signs summary
        coalesce(vs.total_vital_measurements, 0) as total_vital_measurements,
        coalesce(vs.unique_vital_tests, 0) as unique_vital_tests,
        coalesce(vs.abnormal_vitals, 0) as abnormal_vitals,
        case 
            when vs.total_vital_measurements > 0 
            then round(vs.abnormal_vitals::float / vs.total_vital_measurements::float, 3)
            else 0
        end as abnormal_vital_rate,
        
        -- Exposure summary
        coalesce(ex.total_exposures, 0) as total_exposures,
        coalesce(ex.max_treatment_duration, 0) as max_treatment_duration,
        coalesce(ex.ongoing_treatments, 0) as ongoing_treatments,
        coalesce(ex.treatment_categories, 'NONE') as treatment_categories,
        
        -- Overall safety flags
        case 
            when ae.serious_aes > 0 or ae.severe_aes > 0 then 'HIGH_RISK'
            when ae.total_aes > 5 then 'MEDIUM_RISK'
            when ae.total_aes > 0 then 'LOW_RISK'
            else 'NO_EVENTS'
        end as safety_risk_category,
        
        -- Data completeness indicators
        case when ae.total_aes > 0 then 1 else 0 end as has_ae_data,
        case when lb.total_lab_tests > 0 then 1 else 0 end as has_lab_data,
        case when vs.total_vital_measurements > 0 then 1 else 0 end as has_vital_data,
        case when ex.total_exposures > 0 then 1 else 0 end as has_exposure_data,
        
        -- Audit fields
        current_timestamp as dbt_updated_at,
        d.dbt_study_id
        
    from demographics d
    left join adverse_events ae on d.subjid = ae.subjid
    left join laboratory lb on d.subjid = lb.subjid
    left join vital_signs vs on d.subjid = vs.subjid
    left join exposure ex on d.subjid = ex.subjid
)

select * from combined