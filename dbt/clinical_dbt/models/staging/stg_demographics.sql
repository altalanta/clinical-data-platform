{{ config(
    materialized='view',
    docs={'node_color': '#90EE90'}
) }}

with source_data as (
    select * from {{ ref('dm') }}
),

cleaned as (
    select
        studyid,
        subjid,
        coalesce(arm, 'UNKNOWN') as treatment_arm,
        upper(coalesce(sex, 'U')) as sex,
        age,
        
        -- Derived fields
        case 
            when age < 18 then 'PEDIATRIC'
            when age >= 18 and age < 65 then 'ADULT'
            when age >= 65 then 'ELDERLY'
            else 'UNKNOWN'
        end as age_group,
        
        case 
            when sex = 'M' then 'MALE'
            when sex = 'F' then 'FEMALE'
            else 'UNKNOWN'
        end as sex_desc,
        
        -- Audit fields
        current_timestamp as dbt_updated_at,
        '{{ var("study_id") }}' as dbt_study_id
        
    from source_data
    where studyid = '{{ var("study_id") }}'
)

select * from cleaned