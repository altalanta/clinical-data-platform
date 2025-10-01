{{
  config(
    materialized='table',
    schema='public_cdm_marts' if var('adapter', 'default') == 'public_cdm' else 'marts'
  )
}}

-- Patient summary mart combining demographics with visit and condition patterns
-- This provides a comprehensive view of each patient for analytics

with person_base as (
    select * from {{ ref('staging_person') }}
),

visit_summary as (
    select 
        person_id,
        count(*) as total_visits,
        count(distinct visit_concept_id) as unique_visit_types,
        min(visit_start_date) as first_visit_date,
        max(visit_end_date) as last_visit_date,
        sum(length_of_stay_days) as total_los_days,
        avg(length_of_stay_days) as avg_los_days,
        
        -- Visit type breakdown
        sum(case when visit_type_name = 'Inpatient Visit' then 1 else 0 end) as inpatient_visits,
        sum(case when visit_type_name = 'Outpatient Visit' then 1 else 0 end) as outpatient_visits,
        sum(case when visit_type_name = 'Emergency Room Visit' then 1 else 0 end) as er_visits
        
    from {{ ref('staging_visit_occurrence') }}
    group by person_id
),

condition_summary as (
    select 
        person_id,
        count(*) as total_conditions,
        count(distinct condition_concept_id) as unique_conditions
    from {{ source('public_cdm', 'condition_occurrence') }}
    group by person_id
),

measurement_summary as (
    select 
        person_id,
        count(*) as total_measurements,
        count(distinct measurement_concept_id) as unique_measurement_types
    from {{ source('public_cdm', 'measurement') }}
    group by person_id
)

select 
    p.person_id,
    p.gender_concept_id,
    p.year_of_birth,
    p.race_concept_id,
    p.ethnicity_concept_id,
    p.age_years,
    
    -- Visit patterns
    coalesce(v.total_visits, 0) as total_visits,
    coalesce(v.unique_visit_types, 0) as unique_visit_types,
    v.first_visit_date,
    v.last_visit_date,
    coalesce(v.total_los_days, 0) as total_los_days,
    coalesce(v.avg_los_days, 0) as avg_los_days,
    coalesce(v.inpatient_visits, 0) as inpatient_visits,
    coalesce(v.outpatient_visits, 0) as outpatient_visits,
    coalesce(v.er_visits, 0) as er_visits,
    
    -- Clinical patterns
    coalesce(c.total_conditions, 0) as total_conditions,
    coalesce(c.unique_conditions, 0) as unique_conditions,
    coalesce(m.total_measurements, 0) as total_measurements,
    coalesce(m.unique_measurement_types, 0) as unique_measurement_types,
    
    -- Patient complexity score (simple heuristic)
    coalesce(v.total_visits, 0) * 0.1 +
    coalesce(c.unique_conditions, 0) * 0.5 +
    coalesce(v.inpatient_visits, 0) * 2.0 as complexity_score,
    
    -- Data quality assessment
    case 
        when p.birth_year_quality_flag is not null then p.birth_year_quality_flag
        when coalesce(v.total_visits, 0) = 0 then 'No visits recorded'
        else 'Good'
    end as data_quality_status

from person_base p
left join visit_summary v on p.person_id = v.person_id
left join condition_summary c on p.person_id = c.person_id  
left join measurement_summary m on p.person_id = m.person_id