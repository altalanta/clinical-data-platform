
  
    
    

    create  table
      "demo"."main"."stg_patients__dbt_tmp"
  
    as (
      select * from "demo"."main"."patients"
    );
  
  