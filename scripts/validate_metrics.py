"""Sanity checks for the dbt/DuckDB clinical platform outputs."""

from __future__ import annotations

import math
import os
import pathlib
import subprocess
import sys
from typing import Iterable


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "clinical.duckdb"

def ensure_duckdb_available() -> None:
    try:
        import duckdb  # noqa: F401
    except ImportError as exc:  # pragma: no cover - fallback path
        if os.environ.get("VALIDATE_METRICS_CHILD") == "1":
            raise
        dbt_path = shutil_which("dbt")
        if dbt_path is None:
            raise SystemExit("duckdb module unavailable and `dbt` not found on PATH.") from exc
        python_path = pathlib.Path(dbt_path).with_name("python")
        if not python_path.exists():
            raise SystemExit(
                "duckdb module unavailable and associated dbt python interpreter missing."
            ) from exc
        env = os.environ.copy()
        env["VALIDATE_METRICS_CHILD"] = "1"
        result = subprocess.run(
            [str(python_path), __file__],
            env=env,
            check=False,
            capture_output=False,
        )
        sys.exit(result.returncode)


def shutil_which(cmd: str) -> str | None:
    from shutil import which
    return which(cmd)


ensure_duckdb_available()


def get_connection():
    import duckdb

    if not DB_PATH.exists():
        sys.exit(f"DuckDB database not found at {DB_PATH}. Run `make build` first.")
    return duckdb.connect(database=str(DB_PATH))


def assert_cohort_size(conn: duckdb.DuckDBPyConnection) -> int:
    cohort_size = conn.execute(
        """
        select count(*) from main_marts.fct_cohort where include_flag = 1
        """
    ).fetchone()[0]
    if not (3000 <= cohort_size <= 8000):
        sys.exit(
            f"Cohort size {cohort_size} outside expected bounds [3000, 8000]."
        )
    return cohort_size


def assert_event_rate(conn: duckdb.DuckDBPyConnection) -> float:
    event_rate = conn.execute(
        """
        select
            avg(event::double) as event_rate
        from main_survival.mart_survival_features
        """
    ).fetchone()[0]
    if event_rate is None or not (0.02 <= event_rate <= 0.15):
        sys.exit(
            f"Event rate {event_rate:.3f} outside expected bounds [0.02, 0.15]."
        )
    return event_rate


def check_missingness(
    conn: duckdb.DuckDBPyConnection,
    columns: Iterable[str],
) -> list[tuple[str, float]]:
    problems: list[tuple[str, float]] = []
    for column in columns:
        missing = conn.execute(
            f"""
            select
                sum(case when {column} is null then 1 else 0 end)::double
                    / nullif(count(*), 0) as missing_rate
            from main_survival.mart_survival_features
            """
        ).fetchone()[0]
        missing = float(missing or 0.0)
        if missing > 0.30 + 1e-6:
            problems.append((column, missing))
    if problems:
        formatted = ", ".join(f"{col}={rate:.1%}" for col, rate in problems)
        sys.exit(f"Missingness too high (>30%) for columns: {formatted}")
    return [(column, conn.execute(
        f"select coalesce(avg({column}), 0) from main_survival.mart_survival_features"
    ).fetchone()[0]) for column in columns]


def summarize_conditions(conn: duckdb.DuckDBPyConnection) -> list[tuple[str, int]]:
    rows = conn.execute(
        """
        select metric, total
        from (
            select 'prior_mi' as metric, sum(flag_prior_mi) as total from main_intermediate.int_dx_flags
            union all
            select 'diabetes', sum(flag_diabetes) from main_intermediate.int_dx_flags
            union all
            select 'ckd', sum(flag_ckd) from main_intermediate.int_dx_flags
            union all
            select 'lung_cancer', sum(flag_lung_cancer) from main_intermediate.int_dx_flags
            union all
            select 'hba1c_high', sum(lab_hba1c_high_flag) from main_marts.fct_features
            union all
            select 'ldl_high', sum(lab_ldl_high_flag) from main_marts.fct_features
        )
        order by total desc
        limit 5
        """
    ).fetchall()
    return [(row[0], int(row[1])) for row in rows]


def main() -> None:
    conn = get_connection()

    cohort_size = assert_cohort_size(conn)
    event_rate = assert_event_rate(conn)
    stats = check_missingness(
        conn,
        [
            "comorbidity_score",
            "lab_ldl_value",
            "lab_glucose_value",
            "lab_creatinine_value",
        ],
    )
    top_conditions = summarize_conditions(conn)

    print("=== Validation Snapshot ===")
    print(f"Cohort size (include_flag=1): {cohort_size:,}")
    print(f"Event rate (365d mortality): {event_rate:.2%}")
    print("Average feature values:")
    for column, value in stats:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        print(f"  - {column}: {value:.2f}")
    print("Top risk indicators:")
    for metric, total in top_conditions:
        print(f"  - {metric}: {total}")

    conn.close()


if __name__ == "__main__":
    main()
