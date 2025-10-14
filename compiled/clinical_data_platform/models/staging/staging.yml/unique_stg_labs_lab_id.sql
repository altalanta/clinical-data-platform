
    
    

select
    lab_id as unique_field,
    count(*) as n_records

from "clinical"."main_staging"."stg_labs"
where lab_id is not null
group by lab_id
having count(*) > 1


