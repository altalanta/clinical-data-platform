with cohort as (
    select count(*) as cnt from {{ ref('fct_cohort') }}
),
features as (
    select count(*) as cnt from {{ ref('mart_survival_features') }}
)

select
    cohort.cnt as cohort_count,
    features.cnt as feature_count
from cohort, features
where cohort.cnt <> features.cnt
