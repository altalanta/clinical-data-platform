with labs as (
    select
        patient_id,
        lab_category,
        meas_value,
        high_flag,
        low_flag
    from {{ ref('int_lab_latest') }}
),
lab_pivot as (
    select
        patient_id,
        max(case when lab_category = 'ldl_cholesterol' then meas_value end) as lab_ldl_value,
        max(case when lab_category = 'ldl_cholesterol' then high_flag end) as lab_ldl_high_flag,
        max(case when lab_category = 'ldl_cholesterol' then low_flag end) as lab_ldl_low_flag,
        max(case when lab_category = 'glucose' then meas_value end) as lab_glucose_value,
        max(case when lab_category = 'glucose' then high_flag end) as lab_glucose_high_flag,
        max(case when lab_category = 'glucose' then low_flag end) as lab_glucose_low_flag,
        max(case when lab_category = 'hematocrit' then meas_value end) as lab_hematocrit_value,
        max(case when lab_category = 'platelets' then meas_value end) as lab_platelets_value,
        max(case when lab_category = 'leukocytes' then meas_value end) as lab_leukocytes_value,
        max(case when lab_category = 'creatinine' then meas_value end) as lab_creatinine_value,
        max(case when lab_category = 'creatinine' then high_flag end) as lab_creatinine_high_flag,
        max(case when lab_category = 'hba1c' then meas_value end) as lab_hba1c_value,
        max(case when lab_category = 'hba1c' then high_flag end) as lab_hba1c_high_flag
    from labs
    group by patient_id
),
features as (
    select
        cohort.patient_id,
        cohort.index_date,
        cohort.age_at_index,
        cohort.lookback_encounters,
        cohort.lookback_inpt_encounters,
        cohort.lookback_ed_encounters,
        cohort.followup_30d_encounters,
        cohort.days_since_last_encounter,
        dx.flag_prior_mi,
        dx.flag_diabetes,
        dx.flag_ckd,
        dx.flag_lung_cancer,
        dx.flag_breast_cancer,
        dx.flag_hyperlipidemia,
        dx.flag_tobacco_use,
        dx.distinct_high_risk_dx,
        proc.procedures_in_lookback,
        proc.flag_resuscitation,
        proc.flag_ecg,
        proc.flag_advanced_imaging,
        proc.flag_chest_imaging,
        proc.flag_inpatient_visit_code,
        proc.flag_outpatient_visit_code,
        proc.flag_iv_access,
        labs.lab_ldl_value,
        coalesce(labs.lab_ldl_high_flag, 0) as lab_ldl_high_flag,
        coalesce(labs.lab_ldl_low_flag, 0) as lab_ldl_low_flag,
        labs.lab_glucose_value,
        coalesce(labs.lab_glucose_high_flag, 0) as lab_glucose_high_flag,
        coalesce(labs.lab_glucose_low_flag, 0) as lab_glucose_low_flag,
        labs.lab_hematocrit_value,
        labs.lab_platelets_value,
        labs.lab_leukocytes_value,
        labs.lab_creatinine_value,
        coalesce(labs.lab_creatinine_high_flag, 0) as lab_creatinine_high_flag,
        labs.lab_hba1c_value,
        coalesce(labs.lab_hba1c_high_flag, 0) as lab_hba1c_high_flag
    from {{ ref('fct_cohort') }} as cohort
    left join {{ ref('int_dx_flags') }} as dx
        on cohort.patient_id = dx.patient_id
    left join {{ ref('int_proc_flags') }} as proc
        on cohort.patient_id = proc.patient_id
    left join lab_pivot as labs
        on cohort.patient_id = labs.patient_id
)

select
    *,
    coalesce(
        flag_prior_mi +
        flag_diabetes +
        flag_ckd +
        flag_lung_cancer +
        flag_breast_cancer +
        flag_hyperlipidemia +
        flag_tobacco_use, 0
    ) as comorbidity_score,
    coalesce(lookback_encounters, 0) + coalesce(procedures_in_lookback, 0) as utilization_score
from features
