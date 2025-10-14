with labs as (
    select
        labs.patient_id,
        labs.loinc_code,
        labs.meas_value,
        labs.meas_unit,
        labs.result_ts,
        idx.index_date
    from {{ ref('stg_labs') }} as labs
    inner join {{ ref('int_index_dates') }} as idx
        on labs.patient_id = idx.patient_id
    where labs.result_ts between idx.index_date - interval '365' day and idx.index_date
),
categorized as (
    select
        labs.patient_id,
        map.lab_category,
        labs.loinc_code,
        labs.meas_value,
        labs.meas_unit,
        labs.result_ts,
        map.normal_low,
        map.normal_high
    from labs
    inner join {{ ref('loinc_to_lab_category') }} as map
        on labs.loinc_code = map.loinc_code
),
ranked as (
    select
        *,
        row_number() over (
            partition by patient_id, lab_category
            order by result_ts desc
        ) as rn
    from categorized
)

select
    patient_id,
    lab_category,
    loinc_code,
    result_ts,
    meas_value,
    meas_unit,
    case
        when meas_value is null or try_cast(normal_high as double) is null then 0
        when meas_value > try_cast(normal_high as double) then 1
        else 0
    end as high_flag,
    case
        when meas_value is null or try_cast(normal_low as double) is null then 0
        when meas_value < try_cast(normal_low as double) then 1
        else 0
    end as low_flag
from ranked
where rn = 1
