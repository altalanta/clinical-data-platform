
    
    

with child as (
    select patient_id as from_field
    from "clinical"."main_staging"."stg_death"
    where patient_id is not null
),

parent as (
    select patient_id as to_field
    from "clinical"."main_staging"."stg_patients"
)

select
    from_field

from child
left join parent
    on child.from_field = parent.to_field

where parent.to_field is null


