{{ config(
    materialized='view',
    docs={'node_color': '#90EE90'}
) }}

with source_data as (
    select * from {{ ref('lb') }}
),

cleaned as (
    select
        studyid,
        subjid,
        upper(lbtestcd) as test_code,
        lborres as result_value,
        upper(coalesce(lborresu, 'UNKNOWN')) as result_unit,
        lblnor as normal_low,
        lbhnor as normal_high,
        
        -- Derived fields
        case 
            when lborres is null then 'MISSING'
            when lblnor is not null and lborres < lblnor then 'LOW'
            when lbhnor is not null and lborres > lbhnor then 'HIGH'
            when lblnor is not null and lbhnor is not null 
                 and lborres >= lblnor and lborres <= lbhnor then 'NORMAL'
            else 'UNKNOWN'
        end as result_flag,
        
        case 
            when lblnor is not null and lbhnor is not null and lborres is not null
            then (lborres - lblnor) / (lbhnor - lblnor)
            else null
        end as normalized_result,
        
        -- Common lab test categorization
        case 
            when lbtestcd in ('ALT', 'AST', 'BILI', 'ALP') then 'LIVER_FUNCTION'
            when lbtestcd in ('CREAT', 'BUN', 'UREA') then 'KIDNEY_FUNCTION'
            when lbtestcd in ('HGB', 'HCT', 'RBC', 'WBC', 'PLT') then 'HEMATOLOGY'
            when lbtestcd in ('GLUC', 'HBA1C') then 'GLUCOSE_METABOLISM'
            when lbtestcd in ('CHOL', 'LDL', 'HDL', 'TRIG') then 'LIPID_PANEL'
            else 'OTHER'
        end as test_category,
        
        -- Audit fields
        current_timestamp as dbt_updated_at,
        '{{ var("study_id") }}' as dbt_study_id
        
    from source_data
    where studyid = '{{ var("study_id") }}'
)

select * from cleaned