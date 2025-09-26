from __future__ import annotations

import pandas as pd


def infer_schema(df: pd.DataFrame) -> dict[str, str]:
    schema: dict[str, str] = {}
    for col, dtype in df.dtypes.items():
        if pd.api.types.is_integer_dtype(dtype):
            schema[col] = "integer"
        elif pd.api.types.is_float_dtype(dtype):
            schema[col] = "number"
        elif pd.api.types.is_bool_dtype(dtype):
            schema[col] = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            schema[col] = "datetime"
        else:
            schema[col] = "string"
    return schema

