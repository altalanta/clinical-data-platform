{{
  config(
    materialized='view',
    schema='public_cdm_staging' if var('adapter', 'default') == 'public_cdm' else 'staging'
  )
}}

-- Staging layer for synthetic person data
-- Provides basic cleansing and type casting for person demographics

select
    person_id,
    gender_concept_id,
    year_of_birth,
    month_of_birth,
    day_of_birth,
    birth_datetime,
    race_concept_id,
    ethnicity_concept_id,
    location_id,
    provider_id,
    care_site_id,
    person_source_value,
    gender_source_value,
    gender_source_concept_id,
    race_source_value,
    race_source_concept_id,
    ethnicity_source_value,
    ethnicity_source_concept_id,
    
    -- Computed fields
    current_date() - birth_datetime as age_days,
    extract(year from current_date()) - year_of_birth as age_years,
    
    -- Data quality flags
    case 
        when year_of_birth is null then 'Missing birth year'
        when year_of_birth < 1900 or year_of_birth > extract(year from current_date()) then 'Invalid birth year'
        else null
    end as birth_year_quality_flag

from {{ source('public_cdm', 'person') }}

{% if var('adapter', 'default') == 'public_cdm' %}
where person_id is not null
{% endif %}