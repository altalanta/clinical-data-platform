from __future__ import annotations

from pathlib import Path

import duckdb

from clinical_platform.config import get_config


def init_warehouse() -> duckdb.DuckDBPyConnection:
    cfg = get_config()
    db_path = cfg.warehouse.duckdb_path
    con = duckdb.connect(db_path)
    with open("sql/warehouse_ddl.sql", "r", encoding="utf-8") as f:
        con.execute(f.read())
    return con


def load_sdtm_to_star(std_dir: str | Path | None = None) -> None:
    std = Path(std_dir or get_config().paths.standardized_dir)
    con = init_warehouse()

    # Minimal dim + fact population for demo
    con.execute("DELETE FROM dim_study;")
    con.execute("INSERT INTO dim_study SELECT 1 AS study_sk, 'STUDY001' AS study_id, NULL, NULL, NULL, NULL;")

    dm = std / "DM.parquet"
    con.execute(
        """
        INSERT OR REPLACE INTO dim_subject
        SELECT ROW_NUMBER() OVER () AS subject_sk,
               SUBJID AS subject_id,
               1 AS study_sk,
               ARM AS arm,
               SEX AS sex,
               CAST(AGE AS INTEGER) AS age
        FROM read_parquet(?);
        """,
        [str(dm)],
    )

    ae = std / "AE.parquet"
    con.execute(
        """
        INSERT OR REPLACE INTO fact_adverse_events
        SELECT s.subject_sk,
               1 AS study_sk,
               CAST(AESTDTC AS DATE) AS ae_start,
               CAST(AEENDTC AS DATE) AS ae_end,
               AESEV AS severity,
               AESER AS seriousness,
               AEOUT AS outcome
        FROM read_parquet(?) a
        JOIN dim_subject s ON s.subject_id = a.SUBJID;
        """,
        [str(ae)],
    )

