
    
    

with all_values as (

    select
        race as value_field,
        count(*) as n_records

    from "clinical"."main_staging"."stg_patients"
    group by race

)

select *
from all_values
where value_field not in (
    'White','Black','Asian','American Indian/Alaska Native','Native Hawaiian/Pacific Islander'
)


