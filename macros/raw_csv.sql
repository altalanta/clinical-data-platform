{% macro raw_csv(filename) -%}
    read_csv_auto('{{ var("raw_data_root") }}/{{ filename }}', header=True)
{%- endmacro %}
