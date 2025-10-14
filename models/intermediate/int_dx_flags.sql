with dx_window as (
    select
        dx.patient_id,
        idx.index_date,
        upper(dx.diagnosis_code) as diagnosis_code,
        map.ccs_label,
        lower(map.risk_tier) as risk_tier,
        date_trunc('day', dx.diagnosis_ts) as diagnosis_date
    from {{ ref('stg_diagnoses') }} as dx
    inner join {{ ref('int_index_dates') }} as idx
        on dx.patient_id = idx.patient_id
    left join {{ ref('icd10_to_ccs') }} as map
        on dx.diagnosis_code = map.icd10_code
    where dx.diagnosis_ts between idx.index_date - interval '365' day and idx.index_date
),
aggregated as (
    select
        patient_id,
        max(case when ccs_label = 'Acute myocardial infarction' then 1 else 0 end) as flag_prior_mi,
        max(case when ccs_label = 'Type 2 diabetes mellitus' then 1 else 0 end) as flag_diabetes,
        max(case when ccs_label = 'Chronic kidney disease stage 3' then 1 else 0 end) as flag_ckd,
        max(case when ccs_label = 'Lung cancer' then 1 else 0 end) as flag_lung_cancer,
        max(case when ccs_label = 'Breast cancer' then 1 else 0 end) as flag_breast_cancer,
        max(case when ccs_label = 'Hyperlipidemia' then 1 else 0 end) as flag_hyperlipidemia,
        max(case when ccs_label = 'Nicotine dependence' then 1 else 0 end) as flag_tobacco_use,
        count(distinct case when risk_tier = 'high' then diagnosis_code end) as distinct_high_risk_dx
    from dx_window
    group by patient_id
)

select
    idx.patient_id,
    coalesce(agg.flag_prior_mi, 0) as flag_prior_mi,
    coalesce(agg.flag_diabetes, 0) as flag_diabetes,
    coalesce(agg.flag_ckd, 0) as flag_ckd,
    coalesce(agg.flag_lung_cancer, 0) as flag_lung_cancer,
    coalesce(agg.flag_breast_cancer, 0) as flag_breast_cancer,
    coalesce(agg.flag_hyperlipidemia, 0) as flag_hyperlipidemia,
    coalesce(agg.flag_tobacco_use, 0) as flag_tobacco_use,
    coalesce(agg.distinct_high_risk_dx, 0) as distinct_high_risk_dx
from {{ ref('int_index_dates') }} as idx
left join aggregated as agg
    on idx.patient_id = agg.patient_id
