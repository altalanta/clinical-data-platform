from __future__ import annotations

import pandas as pd
import pandera as pa
from pandera.typing import Series


class DMModel(pa.SchemaModel):
    STUDYID: Series[str]
    SUBJID: Series[str]
    ARM: Series[str | None]
    SEX: Series[str | None]
    AGE: Series[float | int | None]

    class Config:
        strict = True


class AEModel(pa.SchemaModel):
    STUDYID: Series[str]
    SUBJID: Series[str]
    AESTDTC: Series[pd.Timestamp] = pa.Field(coerce=True, nullable=True)
    AEENDTC: Series[pd.Timestamp] = pa.Field(coerce=True, nullable=True)
    AESEV: Series[str | None] = pa.Field(isin=['MILD', 'MODERATE', 'SEVERE'], nullable=True)
    AESER: Series[bool | None]
    AEOUT: Series[str | None] = pa.Field(isin=['RECOVERED', 'ONGOING', 'FATAL'], nullable=True)

    class Config:
        strict = True

