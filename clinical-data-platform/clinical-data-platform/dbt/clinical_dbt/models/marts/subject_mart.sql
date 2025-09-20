WITH subjects AS (
  SELECT * FROM {{ ref('stg_subjects') }}
)
SELECT
  subjects.*
FROM subjects;

