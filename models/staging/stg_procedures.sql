with source as (
    select
        trim(encounter_id) as encounter_id,
        trim(patient_id) as patient_id,
        trim(cast(procedure_code as varchar)) as procedure_code,
        try_cast(procedure_ts as timestamp) as procedure_ts
    from {{ raw_csv('procedures.csv') }}
)

select
    encounter_id,
    patient_id,
    procedure_code,
    procedure_ts
from source
