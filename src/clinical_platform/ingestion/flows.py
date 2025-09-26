from __future__ import annotations

from prefect import flow, task

from clinical_platform.ingestion.ingest_csv import ingest_csv_to_parquet
from clinical_platform.logging_utils import get_logger
from clinical_platform.standards.cdisc_sdtm_mapping import standardize_bronze_to_sdtm


@task
def land_to_bronze():
    ingest_csv_to_parquet()


@task
def bronze_to_silver():
    standardize_bronze_to_sdtm()


@flow
def land_bronze_silver_flow():
    log = get_logger("flow", flow="land_bronze_silver")
    log.info("flow_start")
    land_to_bronze()
    bronze_to_silver()
    log.info("flow_complete")


def main(cmd: str = "run-local") -> None:
    if cmd == "run-local":
        land_bronze_silver_flow()
    else:
        raise SystemExit(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()

