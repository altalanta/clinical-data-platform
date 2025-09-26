-- Clinical data utility macros

{% macro categorize_age(age_column) %}
    case 
        when {{ age_column }} < 18 then 'PEDIATRIC'
        when {{ age_column }} >= 18 and {{ age_column }} < 65 then 'ADULT'
        when {{ age_column }} >= 65 then 'ELDERLY'
        else 'UNKNOWN'
    end
{% endmacro %}

{% macro flag_abnormal_vital(test_code, result_value) %}
    case 
        when {{ test_code }} = 'SYSBP' and {{ result_value }} is not null then
            case 
                when {{ result_value }} < 90 then 'HYPOTENSIVE'
                when {{ result_value }} >= 90 and {{ result_value }} < 140 then 'NORMAL'
                when {{ result_value }} >= 140 and {{ result_value }} < 180 then 'HYPERTENSIVE'
                when {{ result_value }} >= 180 then 'SEVERE_HYPERTENSIVE'
                else 'UNKNOWN'
            end
        when {{ test_code }} = 'DIABP' and {{ result_value }} is not null then
            case 
                when {{ result_value }} < 60 then 'HYPOTENSIVE'
                when {{ result_value }} >= 60 and {{ result_value }} < 90 then 'NORMAL'
                when {{ result_value }} >= 90 and {{ result_value }} < 110 then 'HYPERTENSIVE'
                when {{ result_value }} >= 110 then 'SEVERE_HYPERTENSIVE'
                else 'UNKNOWN'
            end
        when {{ test_code }} = 'HR' and {{ result_value }} is not null then
            case 
                when {{ result_value }} < 60 then 'BRADYCARDIA'
                when {{ result_value }} >= 60 and {{ result_value }} <= 100 then 'NORMAL'
                when {{ result_value }} > 100 then 'TACHYCARDIA'
                else 'UNKNOWN'
            end
        else 'N/A'
    end
{% endmacro %}

{% macro calculate_days_between(start_date, end_date) %}
    case 
        when {{ start_date }} is not null and {{ end_date }} is not null 
        then date_diff('day', {{ start_date }}::date, {{ end_date }}::date)
        else null
    end
{% endmacro %}

{% macro safety_risk_category(total_aes, serious_aes, severe_aes) %}
    case 
        when {{ serious_aes }} > 0 or {{ severe_aes }} > 0 then 'HIGH_RISK'
        when {{ total_aes }} > 5 then 'MEDIUM_RISK'
        when {{ total_aes }} > 0 then 'LOW_RISK'
        else 'NO_EVENTS'
    end
{% endmacro %}

{% macro normalize_lab_result(result_value, normal_low, normal_high) %}
    case 
        when {{ normal_low }} is not null and {{ normal_high }} is not null and {{ result_value }} is not null
        then ({{ result_value }} - {{ normal_low }}) / ({{ normal_high }} - {{ normal_low }})
        else null
    end
{% endmacro %}

{% macro lab_result_flag(result_value, normal_low, normal_high) %}
    case 
        when {{ result_value }} is null then 'MISSING'
        when {{ normal_low }} is not null and {{ result_value }} < {{ normal_low }} then 'LOW'
        when {{ normal_high }} is not null and {{ result_value }} > {{ normal_high }} then 'HIGH'
        when {{ normal_low }} is not null and {{ normal_high }} is not null 
             and {{ result_value }} >= {{ normal_low }} and {{ result_value }} <= {{ normal_high }} then 'NORMAL'
        else 'UNKNOWN'
    end
{% endmacro %}

{% macro phi_safe_concat(field_list, separator=', ') %}
    {# Safely concatenate fields while avoiding PHI exposure in logs #}
    {% set safe_fields = [] %}
    {% for field in field_list %}
        {% set safe_field = "coalesce(" + field + ", 'NULL')" %}
        {{ safe_fields.append(safe_field) or "" }}
    {% endfor %}
    string_agg(distinct {{ safe_fields | join(', ' + separator + ' ') }}, '{{ separator }}')
{% endmacro %}

{% macro audit_fields(study_id_var='study_id') %}
    {# Standard audit fields for all clinical tables #}
    current_timestamp as dbt_updated_at,
    '{{ var(study_id_var) }}' as dbt_study_id
{% endmacro %}

{% macro data_completeness_score(has_fields) %}
    {# Calculate completeness score from list of has_* boolean fields #}
    ({% for field in has_fields -%}
        {{ field }}{{ " + " if not loop.last else "" }}
    {%- endfor %})::float / {{ has_fields | length }}.0
{% endmacro %}

{% macro clinical_profile_category(abnormal_lab_rate, abnormal_vital_rate, has_serious_ae) %}
    case 
        when {{ abnormal_lab_rate }} > 0.3 and {{ abnormal_vital_rate }} > 0.2 then 'MULTIPLE_ABNORMALITIES'
        when {{ abnormal_lab_rate }} > 0.5 then 'HIGH_LAB_ABNORMALITIES'
        when {{ abnormal_vital_rate }} > 0.3 then 'HIGH_VITAL_ABNORMALITIES'
        when {{ has_serious_ae }} then 'SERIOUS_SAFETY_CONCERN'
        else 'NORMAL_PROFILE'
    end
{% endmacro %}

{% macro generate_clinical_key(field_list) %}
    {# Generate surrogate key for clinical tables #}
    {{ dbt_utils.generate_surrogate_key(field_list) }}
{% endmacro %}