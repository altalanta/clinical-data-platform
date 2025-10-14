with source as (
    select
        trim(patient_id) as patient_id,
        try_cast(death_ts as timestamp) as death_ts
    from read_csv_auto('/Users/artemisfolle/Documents/Coding_Projects/clinical-data-platform/clinical-data-platform/data/raw/death.csv', header=True)
)

select
    patient_id,
    death_ts
from source