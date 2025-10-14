with source as (
    select
        trim(encounter_id) as encounter_id,
        trim(patient_id) as patient_id,
        try_cast(admit_ts as timestamp) as admit_ts,
        try_cast(discharge_ts as timestamp) as discharge_ts,
        upper(trim(encounter_type)) as encounter_type
    from {{ raw_csv('encounters.csv') }}
)

select
    encounter_id,
    patient_id,
    admit_ts,
    discharge_ts,
    encounter_type
from source
