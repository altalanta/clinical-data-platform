
    
    

with all_values as (

    select
        event as value_field,
        count(*) as n_records

    from "clinical"."main_survival"."mart_survival_labels"
    group by event

)

select *
from all_values
where value_field not in (
    '0','1'
)


