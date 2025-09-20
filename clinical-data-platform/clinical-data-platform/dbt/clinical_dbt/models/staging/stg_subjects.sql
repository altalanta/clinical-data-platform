-- Stage SDTM-like DM to subject dim-like table
WITH dm AS (
  SELECT * FROM read_parquet('../../data/sample_standardized/DM/*.parquet')
)
SELECT DISTINCT
  dm.SUBJID AS subject_id,
  dm.STUDYID AS study_id,
  dm.ARM AS arm,
  dm.SEX AS sex,
  dm.AGE AS age
FROM dm;

