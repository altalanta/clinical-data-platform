with source as (
    select
        trim(encounter_id) as encounter_id,
        trim(patient_id) as patient_id,
        upper(trim(diagnosis_code)) as diagnosis_code,
        try_cast(diagnosis_ts as timestamp) as diagnosis_ts
    from read_csv_auto('/Users/artemisfolle/Documents/Coding_Projects/clinical-data-platform/clinical-data-platform/data/raw/diagnoses.csv', header=True)
)

select
    encounter_id,
    patient_id,
    diagnosis_code,
    diagnosis_ts
from source