with encounters as (
    select
        enc.patient_id,
        enc.encounter_id,
        enc.encounter_type,
        date_trunc('day', enc.admit_ts) as admit_date,
        date_trunc('day', enc.discharge_ts) as discharge_date,
        idx.index_date
    from {{ ref('stg_encounters') }} as enc
    inner join {{ ref('int_index_dates') }} as idx
        on enc.patient_id = idx.patient_id
),
scored as (
    select
        *,
        case
            when admit_date between index_date - interval '365' day and index_date then 1
            else 0
        end as in_lookback,
        case
            when admit_date > index_date
             and admit_date <= index_date + interval '30' day then 1
            else 0
        end as in_followup
    from encounters
),
aggregated as (
    select
        patient_id,
        sum(in_lookback) as lookback_encounters,
        sum(case when in_lookback = 1 and encounter_type = 'INPT' then 1 else 0 end)
            as lookback_inpt_encounters,
        sum(case when in_lookback = 1 and encounter_type = 'ED' then 1 else 0 end)
            as lookback_ed_encounters,
        sum(in_followup) as followup_30d_encounters,
        sum(case when in_followup = 1 and encounter_type = 'INPT' then 1 else 0 end)
            as followup_30d_inpt_encounters
    from scored
    group by patient_id
),
last_pre_index as (
    select
        patient_id,
        max(admit_date) as last_encounter_date
    from scored
    where admit_date <= index_date
    group by patient_id
)

select
    idx.patient_id,
    coalesce(agg.lookback_encounters, 0) as lookback_encounters,
    coalesce(agg.lookback_inpt_encounters, 0) as lookback_inpt_encounters,
    coalesce(agg.lookback_ed_encounters, 0) as lookback_ed_encounters,
    coalesce(agg.followup_30d_encounters, 0) as followup_30d_encounters,
    coalesce(agg.followup_30d_inpt_encounters, 0) as followup_30d_inpt_encounters,
    datediff(
        'day',
        coalesce(last_pre_index.last_encounter_date, idx.index_date),
        idx.index_date
    ) as days_since_last_encounter
from {{ ref('int_index_dates') }} as idx
left join aggregated as agg
    on idx.patient_id = agg.patient_id
left join last_pre_index
    on idx.patient_id = last_pre_index.patient_id
