
  
    
    

    create  table
      "demo"."main"."fact_visits__dbt_tmp"
  
    as (
      select v.visit_id, v.patient_id, v.provider_id, v.cost from "demo"."main"."stg_visits" v
    );
  
  