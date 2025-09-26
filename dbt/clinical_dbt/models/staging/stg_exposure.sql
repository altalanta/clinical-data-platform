{{ config(
    materialized='view',
    docs={'node_color': '#90EE90'}
) }}

with source_data as (
    select * from {{ ref('ex') }}
),

cleaned as (
    select
        studyid,
        subjid,
        upper(coalesce(extrt, 'UNKNOWN')) as treatment_name,
        exdose as dose_amount,
        exstdtc as start_date,
        exendtc as end_date,
        
        -- Derived fields
        case 
            when exstdtc is not null and exendtc is not null 
            then date_diff('day', exstdtc::date, exendtc::date) + 1
            else null
        end as treatment_duration_days,
        
        case 
            when exendtc is null then true
            else false
        end as is_ongoing_treatment,
        
        -- Treatment categorization (example for demo)
        case 
            when lower(extrt) like '%placebo%' then 'PLACEBO'
            when lower(extrt) like '%active%' or lower(extrt) like '%drug%' then 'ACTIVE_TREATMENT'
            when extrt is null then 'UNKNOWN'
            else 'OTHER'
        end as treatment_category,
        
        -- Dose categorization (example ranges)
        case 
            when exdose is null then 'UNKNOWN'
            when exdose = 0 then 'ZERO_DOSE'
            when exdose > 0 and exdose <= 10 then 'LOW_DOSE'
            when exdose > 10 and exdose <= 100 then 'MEDIUM_DOSE'
            when exdose > 100 then 'HIGH_DOSE'
            else 'UNKNOWN'
        end as dose_category,
        
        -- Treatment compliance indicator
        case 
            when exstdtc is not null and exendtc is not null then 'COMPLETED'
            when exstdtc is not null and exendtc is null then 'ONGOING'
            when exstdtc is null then 'NOT_STARTED'
            else 'UNKNOWN'
        end as treatment_status,
        
        -- Audit fields
        current_timestamp as dbt_updated_at,
        '{{ var("study_id") }}' as dbt_study_id
        
    from source_data
    where studyid = '{{ var("study_id") }}'
)

select * from cleaned