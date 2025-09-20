from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd

from clinical_platform.config import get_config
from clinical_platform.ingestion.s3_client import S3Client
from clinical_platform.logging_utils import get_logger


def map_dm(df: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "STUDYID": "STUDYID",
        "SUBJID": "SUBJID",
        "ARM": "ARM",
        "SEX": "SEX",
        "AGE": "AGE",
    }
    out = df.rename(columns=cols)[list(cols.values())]
    out["AGE"] = pd.to_numeric(out["AGE"], errors="coerce")
    return out


def map_ae(df: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "STUDYID": "STUDYID",
        "SUBJID": "SUBJID",
        "AESTDTC": "AESTDTC",
        "AEENDTC": "AEENDTC",
        "AESEV": "AESEV",
        "AESER": "AESER",
        "AEOUT": "AEOUT",
    }
    out = df.rename(columns=cols)[list(cols.values())]
    out["AESTDTC"] = pd.to_datetime(out["AESTDTC"], errors="coerce")
    out["AEENDTC"] = pd.to_datetime(out["AEENDTC"], errors="coerce")
    out["AESER"] = out["AESER"].astype("boolean")
    return out


def map_lb(df: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "STUDYID": "STUDYID",
        "SUBJID": "SUBJID",
        "LBTESTCD": "LBTESTCD",
        "LBORRES": "LBORRES",
        "LBORRESU": "LBORRESU",
        "LBLNOR": "LBLNOR",
        "LBHNOR": "LBHNOR",
    }
    out = df.rename(columns=cols)[list(cols.values())]
    for c in ["LBORRES", "LBLNOR", "LBHNOR"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def map_vs(df: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "STUDYID": "STUDYID",
        "SUBJID": "SUBJID",
        "VSTESTCD": "VSTESTCD",
        "VSORRES": "VSORRES",
        "VSORRESU": "VSORRESU",
    }
    out = df.rename(columns=cols)[list(cols.values())]
    out["VSORRES"] = pd.to_numeric(out["VSORRES"], errors="coerce")
    return out


def map_ex(df: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "STUDYID": "STUDYID",
        "SUBJID": "SUBJID",
        "EXTRT": "EXTRT",
        "EXDOSE": "EXDOSE",
        "EXSTDTC": "EXSTDTC",
        "EXENDTC": "EXENDTC",
    }
    out = df.rename(columns=cols)[list(cols.values())]
    out["EXDOSE"] = pd.to_numeric(out["EXDOSE"], errors="coerce")
    out["EXSTDTC"] = pd.to_datetime(out["EXSTDTC"], errors="coerce")
    out["EXENDTC"] = pd.to_datetime(out["EXENDTC"], errors="coerce")
    return out


DOMAIN_MAPPERS = {
    "DM": map_dm,
    "AE": map_ae,
    "LB": map_lb,
    "VS": map_vs,
    "EX": map_ex,
}


def standardize_bronze_to_sdtm() -> None:
    cfg = get_config()
    s3 = S3Client.from_config(cfg)
    log = get_logger("standardize", request_id="local")
    out_dir = Path(cfg.paths.standardized_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for key in s3.list_keys(cfg.storage.bronze_bucket, prefix="study_id="):
        domain = key.split("domain=")[-1].split("/")[0]
        if domain not in DOMAIN_MAPPERS:
            continue
        raw_bytes = s3.get_bytes(cfg.storage.bronze_bucket, key)
        df = pd.read_parquet(BytesIO(raw_bytes))
        mapped = DOMAIN_MAPPERS[domain](df)
        out_path = out_dir / f"{domain}.parquet"
        mapped.to_parquet(out_path, index=False)
        log.info("standardized", domain=domain, rows=len(mapped), out=str(out_path))

