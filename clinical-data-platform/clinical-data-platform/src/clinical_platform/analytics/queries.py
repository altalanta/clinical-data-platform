from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from clinical_platform.config import get_config


def ae_rates_by_arm(con: duckdb.DuckDBPyConnection):
    """Return AE rates by treatment arm and visit.

    Clinical question: Which arm shows higher severe/serious AE burden over time?
    """
    return con.execute(
        """
        WITH ae AS (
          SELECT s.arm,
                 severity,
                 DATE_PART('day', ae_start) AS visit_day
          FROM fact_adverse_events f JOIN dim_subject s USING(subject_sk)
        )
        SELECT arm,
               visit_day,
               AVG(CASE WHEN severity IN ('SEVERE','SERIOUS') THEN 1 ELSE 0 END) AS severe_rate
        FROM ae
        GROUP BY arm, visit_day
        ORDER BY arm, visit_day;
        """
    ).fetch_df()


def lab_abnormality_rates(con: duckdb.DuckDBPyConnection):
    """Approximate abnormal rates with CTCAE-like thresholds.

    Clinical question: How often do lab values exceed typical ranges by arm?
    """
    return con.execute(
        """
        SELECT s.arm,
               COUNT(*) AS n,
               AVG(CASE WHEN value > high_norm OR value < low_norm THEN 1 ELSE 0 END) AS abn_rate
        FROM fact_labs f JOIN dim_subject s USING(subject_sk)
        GROUP BY s.arm
        ORDER BY abn_rate DESC;
        """
    ).fetch_df()


def vital_trend_summaries(con: duckdb.DuckDBPyConnection):
    """Vital sign trends across visits.

    Clinical question: Are there consistent increases/decreases by arm?
    """
    return con.execute(
        """
        SELECT s.arm, m.code AS vs_code,
               AVG(value) AS mean_value,
               STDDEV(value) AS sd_value,
               COUNT(*) AS n
        FROM fact_vitals v
        JOIN dim_subject s USING(subject_sk)
        JOIN dim_measurement m USING(measurement_sk)
        GROUP BY s.arm, m.code
        ORDER BY s.arm, m.code;
        """
    ).fetch_df()


def main(out: str) -> None:
    cfg = get_config()
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(cfg.warehouse.duckdb_path)
    ae_rates_by_arm(con).to_csv(out_dir / "ae_rates_by_arm.csv", index=False)
    lab_abnormality_rates(con).to_csv(out_dir / "lab_abnormality_rates.csv", index=False)
    vital_trend_summaries(con).to_csv(out_dir / "vital_trend_summaries.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/analytics")
    args = parser.parse_args()
    main(args.out)

