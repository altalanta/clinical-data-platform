{{ config(
    materialized='table',
    docs={'node_color': '#87CEEB'},
    indexes=[
        {'columns': ['subjid'], 'unique': True},
        {'columns': ['treatment_arm']},
        {'columns': ['safety_risk_category']},
        {'columns': ['age_group']}
    ]
) }}

with subject_summary as (
    select * from {{ ref('int_subject_summary') }}
),

final as (
    select
        -- Primary keys
        {{ dbt_utils.generate_surrogate_key(['subjid']) }} as subject_key,
        studyid,
        subjid,
        
        -- Demographics
        treatment_arm,
        sex,
        sex_desc,
        age,
        age_group,
        
        -- Safety outcomes
        total_adverse_events,
        serious_adverse_events,
        severe_adverse_events,
        ongoing_adverse_events,
        has_serious_adverse_event,
        safety_risk_category,
        
        -- Laboratory outcomes
        total_lab_tests,
        unique_lab_tests,
        abnormal_lab_results,
        abnormal_lab_rate,
        lab_categories_tested,
        
        -- Vital signs outcomes
        total_vital_measurements,
        unique_vital_tests,
        abnormal_vitals,
        abnormal_vital_rate,
        
        -- Treatment exposure
        total_exposures,
        max_treatment_duration,
        ongoing_treatments,
        treatment_categories,
        
        -- Data quality indicators
        has_ae_data,
        has_lab_data,
        has_vital_data,
        has_exposure_data,
        (has_ae_data + has_lab_data + has_vital_data + has_exposure_data)::float / 4.0 as data_completeness_score,
        
        -- Derived clinical indicators
        case 
            when abnormal_lab_rate > 0.3 and abnormal_vital_rate > 0.2 then 'MULTIPLE_ABNORMALITIES'
            when abnormal_lab_rate > 0.5 then 'HIGH_LAB_ABNORMALITIES'
            when abnormal_vital_rate > 0.3 then 'HIGH_VITAL_ABNORMALITIES'
            when has_serious_adverse_event then 'SERIOUS_SAFETY_CONCERN'
            else 'NORMAL_PROFILE'
        end as clinical_profile,
        
        -- Study participation quality
        case 
            when data_completeness_score >= 0.8 then 'HIGH_QUALITY'
            when data_completeness_score >= 0.5 then 'MEDIUM_QUALITY'
            else 'LOW_QUALITY'
        end as participation_quality,
        
        -- Audit fields
        dbt_updated_at,
        dbt_study_id,
        current_timestamp as record_created_at
        
    from subject_summary
)

select * from final