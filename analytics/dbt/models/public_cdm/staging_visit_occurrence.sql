{{
  config(
    materialized='view',
    schema='public_cdm_staging' if var('adapter', 'default') == 'public_cdm' else 'staging'
  )
}}

-- Staging layer for synthetic visit occurrence data
-- Provides visit-level aggregations and quality checks

select
    visit_occurrence_id,
    person_id,
    visit_concept_id,
    visit_start_date,
    visit_start_datetime,
    visit_end_date,
    visit_end_datetime,
    visit_type_concept_id,
    provider_id,
    care_site_id,
    visit_source_value,
    visit_source_concept_id,
    admitted_from_concept_id,
    admitted_from_source_value,
    discharged_to_concept_id,
    discharged_to_source_value,
    preceding_visit_occurrence_id,
    
    -- Computed fields
    visit_end_date - visit_start_date as length_of_stay_days,
    
    -- Visit type categorization
    case 
        when visit_concept_id = 9201 then 'Inpatient Visit'
        when visit_concept_id = 9202 then 'Outpatient Visit'
        when visit_concept_id = 9203 then 'Emergency Room Visit'
        else 'Other'
    end as visit_type_name,
    
    -- Data quality flags
    case 
        when visit_start_date is null then 'Missing start date'
        when visit_end_date is null then 'Missing end date'
        when visit_end_date < visit_start_date then 'End before start'
        when visit_end_date - visit_start_date > 365 then 'Excessive length of stay'
        else null
    end as visit_quality_flag

from {{ source('public_cdm', 'visit_occurrence') }}

{% if var('adapter', 'default') == 'public_cdm' %}
where visit_occurrence_id is not null
  and person_id is not null
{% endif %}