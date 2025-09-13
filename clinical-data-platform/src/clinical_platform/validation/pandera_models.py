from __future__ import annotations

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
    AESTDTC: Series[object]
    AEENDTC: Series[object]
    AESEV: Series[str | None]
    AESER: Series[bool | None]
    AEOUT: Series[str | None]

    class Config:
        strict = True

