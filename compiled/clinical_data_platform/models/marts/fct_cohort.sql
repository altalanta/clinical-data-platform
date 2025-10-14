with base as (
    select
        idx.patient_id,
        idx.index_date,
        idx.age_at_index,
        idx.index_source,
        coalesce(vis.lookback_encounters, 0) as lookback_encounters,
        coalesce(vis.lookback_inpt_encounters, 0) as lookback_inpt_encounters,
        coalesce(vis.lookback_ed_encounters, 0) as lookback_ed_encounters,
        coalesce(vis.followup_30d_encounters, 0) as followup_30d_encounters,
        coalesce(vis.days_since_last_encounter, 9999) as days_since_last_encounter
    from "clinical"."main_intermediate"."int_index_dates" as idx
    left join "clinical"."main_intermediate"."int_visit_windows" as vis
        on idx.patient_id = vis.patient_id
),
scored as (
    select
        base.*,
        case when age_at_index >= 18 then 1 else 0 end as meets_age,
        case when lookback_encounters >= 1 then 1 else 0 end as meets_utilization,
        case when days_since_last_encounter <= 120 then 1 else 0 end as meets_continuity
    from base
)

select
    *,
    case
        when meets_age = 1
         and meets_utilization = 1
         and meets_continuity = 1 then 1
        else 0
    end as include_flag
from scored
where meets_age = 1