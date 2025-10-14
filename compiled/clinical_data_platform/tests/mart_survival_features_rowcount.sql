with cohort as (
    select count(*) as cnt from "clinical"."main_marts"."fct_cohort"
),
features as (
    select count(*) as cnt from "clinical"."main_survival"."mart_survival_features"
)

select
    cohort.cnt as cohort_count,
    features.cnt as feature_count
from cohort, features
where cohort.cnt <> features.cnt