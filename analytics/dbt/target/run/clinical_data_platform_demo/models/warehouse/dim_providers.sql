
  
    
    

    create  table
      "demo"."main"."dim_providers__dbt_tmp"
  
    as (
      select provider_id, specialty from "demo"."main"."stg_providers"
    );
  
  