
    
    

with all_values as (

    select
        include_flag as value_field,
        count(*) as n_records

    from "clinical"."main_marts"."fct_cohort"
    group by include_flag

)

select *
from all_values
where value_field not in (
    '0','1'
)


