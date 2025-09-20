
  
    
    

    create  table
      "demo"."main"."dim_patients__dbt_tmp"
  
    as (
      select patient_id, sex, age from "demo"."main"."stg_patients"
    );
  
  