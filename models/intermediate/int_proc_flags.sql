with proc_window as (
    select
        proc.patient_id,
        idx.index_date,
        upper(proc.procedure_code) as procedure_code,
        date_trunc('day', proc.procedure_ts) as procedure_date
    from {{ ref('stg_procedures') }} as proc
    inner join {{ ref('int_index_dates') }} as idx
        on proc.patient_id = idx.patient_id
    where proc.procedure_ts between idx.index_date - interval '365' day and idx.index_date
),
aggregated as (
    select
        patient_id,
        count(*) as procedures_in_lookback,
        max(case when procedure_code in ('92950') then 1 else 0 end) as flag_resuscitation,
        max(case when procedure_code in ('93000', '93010') then 1 else 0 end) as flag_ecg,
        max(case when procedure_code in ('71275') then 1 else 0 end) as flag_advanced_imaging,
        max(case when procedure_code in ('71045') then 1 else 0 end) as flag_chest_imaging,
        max(case when procedure_code in ('99223') then 1 else 0 end) as flag_inpatient_visit_code,
        max(case when procedure_code in ('99214') then 1 else 0 end) as flag_outpatient_visit_code,
        max(case when procedure_code in ('36591') then 1 else 0 end) as flag_iv_access
    from proc_window
    group by patient_id
)

select
    idx.patient_id,
    coalesce(agg.procedures_in_lookback, 0) as procedures_in_lookback,
    coalesce(agg.flag_resuscitation, 0) as flag_resuscitation,
    coalesce(agg.flag_ecg, 0) as flag_ecg,
    coalesce(agg.flag_advanced_imaging, 0) as flag_advanced_imaging,
    coalesce(agg.flag_chest_imaging, 0) as flag_chest_imaging,
    coalesce(agg.flag_inpatient_visit_code, 0) as flag_inpatient_visit_code,
    coalesce(agg.flag_outpatient_visit_code, 0) as flag_outpatient_visit_code,
    coalesce(agg.flag_iv_access, 0) as flag_iv_access
from {{ ref('int_index_dates') }} as idx
left join aggregated as agg
    on idx.patient_id = agg.patient_id
