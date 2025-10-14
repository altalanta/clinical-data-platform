with features as (
    select *
    from {{ ref('fct_features') }}
),
labels as (
    select *
    from {{ ref('mart_survival_labels') }}
)

select
    features.*,
    labels.event,
    labels.time_to_event,
    labels.censor_date,
    labels.death_ts
from features
inner join labels
    on features.patient_id = labels.patient_id
