
    
    

with all_values as (

    select
        sex as value_field,
        count(*) as n_records

    from "clinical"."main_staging"."stg_patients"
    group by sex

)

select *
from all_values
where value_field not in (
    'F','M'
)


