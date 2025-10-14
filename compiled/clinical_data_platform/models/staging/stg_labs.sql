with source as (
    select
        trim(lab_id) as lab_id,
        trim(patient_id) as patient_id,
        trim(loinc_code) as loinc_code,
        try_cast(meas_value as double) as meas_value,
        trim(meas_unit) as meas_unit,
        try_cast(result_ts as timestamp) as result_ts
    from read_csv_auto('/Users/artemisfolle/Documents/Coding_Projects/clinical-data-platform/clinical-data-platform/data/raw/labs.csv', header=True)
)

select
    lab_id,
    patient_id,
    loinc_code,
    meas_value,
    meas_unit,
    result_ts
from source