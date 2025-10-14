
    
    

select
    patient_id as unique_field,
    count(*) as n_records

from "clinical"."main_staging"."stg_death"
where patient_id is not null
group by patient_id
having count(*) > 1


