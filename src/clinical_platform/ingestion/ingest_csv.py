from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable
import uuid

import pandas as pd

from clinical_platform.config import get_config
from clinical_platform.ingestion.s3_client import S3Client
from clinical_platform.logging_utils import get_logger, set_data_lineage_id, set_request_id
from clinical_platform.validation.validator import create_validator


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


def ingest_csv_to_parquet(validate_data: bool = True, fail_on_validation_error: bool = True) -> Dict[str, bool]:
    """
    Ingest CSV files to Parquet with optional validation.
    
    Args:
        validate_data: Whether to validate data during ingestion
        fail_on_validation_error: Whether to stop ingestion on validation failure
        
    Returns:
        Dict mapping domain names to success status
    """
    cfg = get_config()
    
    # Set up tracking IDs
    request_id = set_request_id()
    lineage_id = set_data_lineage_id()
    
    log = get_logger("ingest", request_id=request_id, data_lineage_id=lineage_id)
    s3 = S3Client.from_config(cfg)
    s3.ensure_buckets()
    
    # Initialize validator if validation is enabled
    validator = None
    if validate_data:
        validator = create_validator(fail_fast=fail_on_validation_error)
        log.info("Validation enabled for ingestion", validator_config={
            "pandera_enabled": validator.enable_pandera,
            "ge_enabled": validator.enable_great_expectations,
            "fail_fast": validator.fail_fast
        })
    
    ingestion_results = {}
    validation_reports = []
    
    raw_dir = Path(cfg.paths.raw_dir)
    if not raw_dir.exists():
        log.error("Raw data directory does not exist", path=str(raw_dir))
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
    
    csv_files = list(raw_dir.glob("*.csv"))
    if not csv_files:
        log.warning("No CSV files found in raw directory", path=str(raw_dir))
        return {}
    
    log.info("Starting batch ingestion", 
             file_count=len(csv_files), 
             raw_dir=str(raw_dir),
             request_id=request_id,
             lineage_id=lineage_id)
    
    for csv_path in sorted(csv_files):
        domain = csv_path.stem.upper()
        study_id = "STUDY001"
        part_key = f"study_id={study_id}/domain={domain}/{csv_path.stem}.parquet"
        
        try:
            log.info("Starting domain ingestion", 
                    domain=domain, 
                    csv_file=str(csv_path),
                    target_key=part_key)
            
            # Read CSV in chunks
            frames = []
            for chunk_df in chunk_read_csv(csv_path, chunksize=5000):
                frames.append(chunk_df)
            
            if not frames:
                log.warning("No data found in CSV", domain=domain, csv_file=str(csv_path))
                ingestion_results[domain] = False
                continue
            
            # Combine chunks
            df_all = pd.concat(frames, ignore_index=True)
            log.info("Data loaded from CSV", 
                    domain=domain, 
                    rows=len(df_all), 
                    columns=len(df_all.columns),
                    memory_mb=df_all.memory_usage(deep=True).sum() / 1024 / 1024)
            
            # Validate data if enabled
            if validator and domain in ["DM", "AE", "LB", "VS", "EX"]:  # Only validate known SDTM domains
                log.info("Validating data", domain=domain)
                validation_result = validator.validate_domain_data(df_all, domain, lineage_id)
                validation_reports.append(validation_result)
                
                if not validation_result["overall_valid"]:
                    log.error("Data validation failed", 
                             domain=domain,
                             error_count=len(validation_result["errors"]),
                             errors=validation_result["errors"][:3])
                    
                    if fail_on_validation_error:
                        ingestion_results[domain] = False
                        log.error("Stopping ingestion due to validation failure", domain=domain)
                        break
                    else:
                        log.warning("Continuing ingestion despite validation failure", domain=domain)
                else:
                    log.info("Data validation passed", domain=domain)
            
            # Infer and apply data types
            dtypes = infer_dtypes(df_all)
            df_all = df_all.astype(dtypes)
            
            # Write to Parquet
            buf = BytesIO()
            df_all.to_parquet(buf, index=False, compression='snappy')
            parquet_bytes = buf.getvalue()
            
            s3.put_bytes(cfg.storage.bronze_bucket, part_key, parquet_bytes)
            
            log.info("Domain ingestion complete", 
                    domain=domain, 
                    key=part_key, 
                    rows=len(df_all),
                    parquet_size_mb=len(parquet_bytes) / 1024 / 1024)
            
            ingestion_results[domain] = True
            
        except Exception as e:
            log.error("Domain ingestion failed", 
                     domain=domain, 
                     error=str(e),
                     csv_file=str(csv_path))
            ingestion_results[domain] = False
            
            if fail_on_validation_error:
                raise
    
    # Save validation reports if any were generated
    if validation_reports:
        reports_dir = Path("data/validation_reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = reports_dir / f"ingestion_validation_{request_id}.json"
        
        combined_report = {
            "request_id": request_id,
            "lineage_id": lineage_id,
            "ingestion_summary": {
                "total_domains": len(ingestion_results),
                "successful_domains": sum(1 for success in ingestion_results.values() if success),
                "failed_domains": sum(1 for success in ingestion_results.values() if not success),
                "validation_enabled": validate_data
            },
            "domain_validations": validation_reports,
            "ingestion_results": ingestion_results
        }
        
        import json
        with open(report_path, 'w') as f:
            json.dump(combined_report, f, indent=2, default=str)
        
        log.info("Validation report saved", report_path=str(report_path))
    
    # Log final summary
    successful_domains = [domain for domain, success in ingestion_results.items() if success]
    failed_domains = [domain for domain, success in ingestion_results.items() if not success]
    
    log.info("Batch ingestion complete",
             successful_domains=successful_domains,
             failed_domains=failed_domains,
             total_processed=len(ingestion_results))
    
    if failed_domains:
        log.error("Some domains failed ingestion", failed_domains=failed_domains)
    
    return ingestion_results


if __name__ == "__main__":
    ingest_csv_to_parquet()

