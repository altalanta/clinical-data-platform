with source as (
    select
        trim(patient_id) as patient_id,
        upper(trim(sex)) as sex_raw,
        try_cast(birth_date as date) as birth_date,
        trim(race) as race_raw,
        trim(ethnicity) as ethnicity_raw,
        upper(trim(index_site)) as index_site
    from read_csv_auto('/Users/artemisfolle/Documents/Coding_Projects/clinical-data-platform/clinical-data-platform/data/raw/patients.csv', header=True)
)

select
    patient_id,
    case
        when sex_raw in ('F', 'M') then sex_raw
        else null
    end as sex,
    birth_date,
    race_raw as race,
    ethnicity_raw as ethnicity,
    index_site
from source