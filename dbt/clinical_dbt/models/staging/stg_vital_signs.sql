{{ config(
    materialized='view',
    docs={'node_color': '#90EE90'}
) }}

with source_data as (
    select * from {{ ref('vs') }}
),

cleaned as (
    select
        studyid,
        subjid,
        upper(vstestcd) as test_code,
        vsorres as result_value,
        upper(coalesce(vsorresu, 'UNKNOWN')) as result_unit,
        
        -- Derived fields and standardization
        case 
            when vstestcd = 'SYSBP' and vsorres is not null then
                case 
                    when vsorres < 90 then 'HYPOTENSIVE'
                    when vsorres >= 90 and vsorres < 140 then 'NORMAL'
                    when vsorres >= 140 and vsorres < 180 then 'HYPERTENSIVE'
                    when vsorres >= 180 then 'SEVERE_HYPERTENSIVE'
                    else 'UNKNOWN'
                end
            when vstestcd = 'DIABP' and vsorres is not null then
                case 
                    when vsorres < 60 then 'HYPOTENSIVE'
                    when vsorres >= 60 and vsorres < 90 then 'NORMAL'
                    when vsorres >= 90 and vsorres < 110 then 'HYPERTENSIVE'
                    when vsorres >= 110 then 'SEVERE_HYPERTENSIVE'
                    else 'UNKNOWN'
                end
            when vstestcd = 'HR' and vsorres is not null then
                case 
                    when vsorres < 60 then 'BRADYCARDIA'
                    when vsorres >= 60 and vsorres <= 100 then 'NORMAL'
                    when vsorres > 100 then 'TACHYCARDIA'
                    else 'UNKNOWN'
                end
            when vstestcd = 'TEMP' and vsorres is not null then
                case 
                    when vsorres < 36.1 then 'HYPOTHERMIA'
                    when vsorres >= 36.1 and vsorres <= 37.2 then 'NORMAL'
                    when vsorres > 37.2 and vsorres <= 38.0 then 'LOW_FEVER'
                    when vsorres > 38.0 then 'HIGH_FEVER'
                    else 'UNKNOWN'
                end
            else 'N/A'
        end as clinical_flag,
        
        -- Test categorization
        case 
            when vstestcd in ('SYSBP', 'DIABP') then 'BLOOD_PRESSURE'
            when vstestcd = 'HR' then 'HEART_RATE'
            when vstestcd = 'TEMP' then 'TEMPERATURE'
            when vstestcd = 'WEIGHT' then 'WEIGHT'
            when vstestcd = 'HEIGHT' then 'HEIGHT'
            when vstestcd = 'RESP' then 'RESPIRATORY_RATE'
            else 'OTHER'
        end as vital_category,
        
        -- Standard vital signs description
        case 
            when vstestcd = 'SYSBP' then 'Systolic Blood Pressure'
            when vstestcd = 'DIABP' then 'Diastolic Blood Pressure'
            when vstestcd = 'HR' then 'Heart Rate'
            when vstestcd = 'TEMP' then 'Temperature'
            when vstestcd = 'WEIGHT' then 'Weight'
            when vstestcd = 'HEIGHT' then 'Height'
            when vstestcd = 'RESP' then 'Respiratory Rate'
            else vstestcd
        end as test_description,
        
        -- Audit fields
        current_timestamp as dbt_updated_at,
        '{{ var("study_id") }}' as dbt_study_id
        
    from source_data
    where studyid = '{{ var("study_id") }}'
)

select * from cleaned