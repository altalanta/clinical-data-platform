from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from clinical_platform.config import get_config
from clinical_platform.ingestion.s3_client import S3Client
from clinical_platform.logging_utils import get_logger


def infer_dtypes(df: pd.DataFrame) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for col, dtype in df.dtypes.items():
        if pd.api.types.is_integer_dtype(dtype):
            mapping[col] = "int64"
        elif pd.api.types.is_float_dtype(dtype):
            mapping[col] = "float64"
        elif pd.api.types.is_bool_dtype(dtype):
            mapping[col] = "bool"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            mapping[col] = "datetime64[ns]"
        else:
            mapping[col] = "string"
    return mapping


def chunk_read_csv(path: Path, chunksize: int = 10000) -> Iterable[pd.DataFrame]:
    for chunk in pd.read_csv(path, chunksize=chunksize):
        yield chunk


def ingest_csv_to_parquet() -> None:
    cfg = get_config()
    log = get_logger("ingest", request_id="local")
    s3 = S3Client.from_config(cfg)
    s3.ensure_buckets()

    raw_dir = Path(cfg.paths.raw_dir)
    for csv_path in sorted(raw_dir.glob("*.csv")):
        domain = csv_path.stem.upper()
        study_id = "STUDY001"
        part_key = f"study_id={study_id}/domain={domain}/{csv_path.stem}.parquet"
        log.info("ingest_start", domain=domain, csv=str(csv_path))
        frames = []
        for df in chunk_read_csv(csv_path, chunksize=5000):
            frames.append(df)
        if not frames:
            continue
        df_all = pd.concat(frames, ignore_index=True)
        dtypes = infer_dtypes(df_all)
        df_all = df_all.astype(dtypes)
        # Write to Parquet bytes
        buf = BytesIO()
        df_all.to_parquet(buf, index=False)
        s3.put_bytes(cfg.storage.bronze_bucket, part_key, buf.getvalue())
        log.info("ingest_complete", domain=domain, key=part_key, rows=len(df_all))


if __name__ == "__main__":
    ingest_csv_to_parquet()

