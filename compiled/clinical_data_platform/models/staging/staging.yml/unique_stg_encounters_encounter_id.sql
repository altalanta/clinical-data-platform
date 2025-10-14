
    
    

select
    encounter_id as unique_field,
    count(*) as n_records

from "clinical"."main_staging"."stg_encounters"
where encounter_id is not null
group by encounter_id
having count(*) > 1


