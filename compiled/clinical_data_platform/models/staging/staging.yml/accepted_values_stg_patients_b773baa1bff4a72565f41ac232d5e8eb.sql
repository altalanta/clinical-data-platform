
    
    

with all_values as (

    select
        ethnicity as value_field,
        count(*) as n_records

    from "clinical"."main_staging"."stg_patients"
    group by ethnicity

)

select *
from all_values
where value_field not in (
    'Hispanic or Latino','Not Hispanic or Latino'
)


