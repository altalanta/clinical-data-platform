
  
    
    

    create  table
      "demo"."main"."stg_visits__dbt_tmp"
  
    as (
      select * from "demo"."main"."visits"
    );
  
  