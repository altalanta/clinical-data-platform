
    
    

with all_values as (

    select
        encounter_type as value_field,
        count(*) as n_records

    from "clinical"."main_staging"."stg_encounters"
    group by encounter_type

)

select *
from all_values
where value_field not in (
    'INPT','OUTPT','ED'
)


