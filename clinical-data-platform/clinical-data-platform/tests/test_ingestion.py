from pathlib import Path

import pandas as pd

from clinical_platform.ingestion.ingest_csv import infer_dtypes


def test_infer_dtypes_handles_various_types(tmp_path: Path):
    df = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [1.1, 2.2, 3.3],
        'c': [True, False, True],
        'd': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
        'e': ['x', 'y', 'z'],
    })
    dtypes = infer_dtypes(df)
    assert dtypes == {
        'a': 'int64', 'b': 'float64', 'c': 'bool', 'd': 'datetime64[ns]', 'e': 'string'
    }

