with inpatient_candidates as (
    select
        enc.patient_id,
        date_trunc('day', enc.admit_ts) as candidate_date,
        'inpatient_encounter' as index_source
    from {{ ref('stg_encounters') }} as enc
    where enc.encounter_type = 'INPT'
),
high_risk_dx as (
    select
        dx.patient_id,
        date_trunc('day', dx.diagnosis_ts) as candidate_date,
        'high_risk_diagnosis' as index_source
    from {{ ref('stg_diagnoses') }} as dx
    inner join {{ ref('icd10_to_ccs') }} as map
        on dx.diagnosis_code = map.icd10_code
    where lower(map.risk_tier) = 'high'
),
all_candidates as (
    select * from inpatient_candidates
    union all
    select * from high_risk_dx
),
ranked as (
    select
        patient_id,
        candidate_date,
        index_source,
        row_number() over (
            partition by patient_id
            order by candidate_date
        ) as rn
    from all_candidates
),
selected as (
    select
        patient_id,
        candidate_date,
        index_source
    from ranked
    where rn = 1
),
fallback_encounter as (
    select
        patient_id,
        date_trunc('day', min(admit_ts)) as candidate_date
    from {{ ref('stg_encounters') }}
    group by patient_id
),
combined as (
    select
        patients.patient_id,
        coalesce(sel.candidate_date, fallback.candidate_date) as index_date,
        coalesce(sel.index_source, 'fallback_first_encounter') as index_source
    from {{ ref('stg_patients') }} as patients
    left join selected as sel
        on patients.patient_id = sel.patient_id
    left join fallback_encounter as fallback
        on patients.patient_id = fallback.patient_id
)

select
    combined.patient_id,
    combined.index_date,
    combined.index_source,
    datediff('year', patients.birth_date, combined.index_date) as age_at_index
from combined
inner join {{ ref('stg_patients') }} as patients
    on combined.patient_id = patients.patient_id
