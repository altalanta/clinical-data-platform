with source as (
    select
        trim(patient_id) as patient_id,
        try_cast(death_ts as timestamp) as death_ts
    from {{ raw_csv('death.csv') }}
)

select
    patient_id,
    death_ts
from source
