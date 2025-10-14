
    
    

with all_values as (

    select
        index_site as value_field,
        count(*) as n_records

    from "clinical"."main_staging"."stg_patients"
    group by index_site

)

select *
from all_values
where value_field not in (
    'A','B','C'
)


