
  
    
    

    create  table
      "demo"."main"."stg_providers__dbt_tmp"
  
    as (
      select * from "demo"."main"."providers"
    );
  
  