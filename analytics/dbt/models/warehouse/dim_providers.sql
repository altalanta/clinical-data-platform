select provider_id, specialty from {{ ref('stg_providers') }}
