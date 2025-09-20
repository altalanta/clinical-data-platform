select patient_id, sex, age from {{ ref('stg_patients') }}
