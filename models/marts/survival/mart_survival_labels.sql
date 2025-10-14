with cohort as (
    select patient_id, index_date
    from {{ ref('fct_cohort') }}
),
death as (
    select patient_id, death_ts
    from {{ ref('stg_death') }}
),
last_encounter as (
    select
        patient_id,
        max(discharge_ts) as last_encounter_ts
    from {{ ref('stg_encounters') }}
    group by patient_id
),
joined as (
    select
        cohort.patient_id,
        cohort.index_date,
        death.death_ts,
        last_encounter.last_encounter_ts,
        cohort.index_date + interval '365' day as max_followup_ts
    from cohort
    left join death
        on cohort.patient_id = death.patient_id
    left join last_encounter
        on cohort.patient_id = last_encounter.patient_id
),
prepared as (
    select
        patient_id,
        index_date,
        death_ts,
        coalesce(last_encounter_ts, index_date) as last_encounter_ts,
        max_followup_ts,
        case
            when death_ts is not null and death_ts <= max_followup_ts then 1
            else 0
        end as event
    from joined
),
timings as (
    select
        *,
        case
            when event = 1 then death_ts
            else least(last_encounter_ts, max_followup_ts)
        end as censor_ts
    from prepared
)

select
    patient_id,
    index_date,
    event,
    censor_ts as censor_date,
    case
        when event = 1 then datediff('day', index_date, death_ts)
        else datediff('day', index_date, censor_ts)
    end as time_to_event,
    death_ts
from timings
