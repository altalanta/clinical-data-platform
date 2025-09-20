select v.visit_id, v.patient_id, v.provider_id, v.cost from {{ ref('stg_visits') }} v
