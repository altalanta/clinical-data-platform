{{ config(
    materialized='table',
    docs={'node_color': '#87CEEB'}
) }}

with subject_outcomes as (
    select * from {{ ref('fact_subject_outcomes') }}
),

study_metrics as (
    select
        dbt_study_id as study_id,
        '{{ var("study_name") }}' as study_name,
        
        -- Enrollment metrics
        count(*) as total_subjects,
        count(distinct treatment_arm) as treatment_arms_count,
        
        -- Demographics distribution
        count(case when sex = 'M' then 1 end) as male_subjects,
        count(case when sex = 'F' then 1 end) as female_subjects,
        count(case when age_group = 'PEDIATRIC' then 1 end) as pediatric_subjects,
        count(case when age_group = 'ADULT' then 1 end) as adult_subjects,
        count(case when age_group = 'ELDERLY' then 1 end) as elderly_subjects,
        
        round(avg(age), 1) as mean_age,
        approx_quantile(age, 0.5) as median_age,
        min(age) as min_age,
        max(age) as max_age,
        
        -- Safety metrics
        sum(total_adverse_events) as total_adverse_events_study,
        sum(serious_adverse_events) as total_serious_aes_study,
        count(case when has_serious_adverse_event then 1 end) as subjects_with_serious_aes,
        
        -- Data quality metrics
        avg(data_completeness_score) as avg_data_completeness,
        count(case when participation_quality = 'HIGH_QUALITY' then 1 end) as high_quality_subjects,
        count(case when participation_quality = 'MEDIUM_QUALITY' then 1 end) as medium_quality_subjects,
        count(case when participation_quality = 'LOW_QUALITY' then 1 end) as low_quality_subjects,
        
        -- Clinical profile distribution
        count(case when clinical_profile = 'NORMAL_PROFILE' then 1 end) as normal_profile_subjects,
        count(case when clinical_profile = 'SERIOUS_SAFETY_CONCERN' then 1 end) as safety_concern_subjects,
        count(case when clinical_profile = 'MULTIPLE_ABNORMALITIES' then 1 end) as multiple_abnormalities_subjects,
        
        -- Treatment exposure metrics
        avg(max_treatment_duration) as avg_treatment_duration,
        max(max_treatment_duration) as max_treatment_duration_overall,
        count(case when ongoing_treatments > 0 then 1 end) as subjects_with_ongoing_treatment,
        
        -- Laboratory metrics
        sum(total_lab_tests) as total_lab_tests_study,
        avg(abnormal_lab_rate) as avg_abnormal_lab_rate,
        
        -- Vital signs metrics
        sum(total_vital_measurements) as total_vital_measurements_study,
        avg(abnormal_vital_rate) as avg_abnormal_vital_rate,
        
        current_timestamp as analysis_timestamp
        
    from subject_outcomes
    group by dbt_study_id
),

calculated_rates as (
    select
        *,
        
        -- Calculate rates and percentages
        round(male_subjects::float / total_subjects::float * 100, 1) as male_percentage,
        round(female_subjects::float / total_subjects::float * 100, 1) as female_percentage,
        
        round(subjects_with_serious_aes::float / total_subjects::float * 100, 1) as serious_ae_rate_percent,
        round(safety_concern_subjects::float / total_subjects::float * 100, 1) as safety_concern_rate_percent,
        
        round(high_quality_subjects::float / total_subjects::float * 100, 1) as high_quality_rate_percent,
        
        round(avg_data_completeness * 100, 1) as avg_data_completeness_percent,
        
        -- Study status indicators
        case 
            when subjects_with_ongoing_treatment > 0 then 'ONGOING'
            else 'COMPLETED'
        end as study_status,
        
        case 
            when serious_ae_rate_percent > 20 then 'HIGH_SAFETY_CONCERN'
            when serious_ae_rate_percent > 10 then 'MODERATE_SAFETY_CONCERN'
            else 'LOW_SAFETY_CONCERN'
        end as overall_safety_profile,
        
        case 
            when avg_data_completeness_percent >= 80 then 'HIGH_QUALITY_DATA'
            when avg_data_completeness_percent >= 60 then 'MEDIUM_QUALITY_DATA'
            else 'LOW_QUALITY_DATA'
        end as data_quality_assessment
        
    from study_metrics
)

select * from calculated_rates