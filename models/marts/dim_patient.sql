select
    pat.patient_id,
    pat.sex,
    pat.race,
    pat.ethnicity,
    pat.index_site,
    pat.birth_date,
    idx.index_date,
    idx.index_source,
    idx.age_at_index,
    case
        when idx.age_at_index < 40 then '18-39'
        when idx.age_at_index between 40 and 64 then '40-64'
        when idx.age_at_index between 65 and 79 then '65-79'
        else '80+'
    end as age_bucket
from {{ ref('stg_patients') }} as pat
inner join {{ ref('int_index_dates') }} as idx
    on pat.patient_id = idx.patient_id
