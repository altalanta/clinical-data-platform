{{ config(
    materialized='view',
    docs={'node_color': '#90EE90'}
) }}

with source_data as (
    select * from {{ ref('ae') }}
),

cleaned as (
    select
        studyid,
        subjid,
        aestdtc,
        aeendtc,
        upper(coalesce(aesev, 'UNKNOWN')) as severity,
        coalesce(aeser, false) as is_serious,
        upper(coalesce(aeout, 'UNKNOWN')) as outcome,
        
        -- Derived fields
        case 
            when aestdtc is not null and aeendtc is not null 
            then date_diff('day', aestdtc::date, aeendtc::date)
            else null
        end as duration_days,
        
        case 
            when aeendtc is null and aeout != 'FATAL' then true
            else false
        end as is_ongoing,
        
        case 
            when aesev = 'MILD' then 1
            when aesev = 'MODERATE' then 2
            when aesev = 'SEVERE' then 3
            else 0
        end as severity_rank,
        
        -- Audit fields
        current_timestamp as dbt_updated_at,
        '{{ var("study_id") }}' as dbt_study_id
        
    from source_data
    where studyid = '{{ var("study_id") }}'
)

select * from cleaned