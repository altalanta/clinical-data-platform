"""Simulated dbt transform pipeline instrumented for observability."""
from __future__ import annotations

import json
import time
from pathlib import Path

from src.pipelines.base import bootstrap, pipeline_span, stage_event, utc_now_iso

BRONZE_ROOT = Path("data/bronze")
SILVER_ROOT = Path("data/silver")
LAST_UPDATE_FILE = SILVER_ROOT / "_last_update.txt"


def run() -> None:
    pipeline = "dbt"
    logger, run_id = bootstrap(pipeline)

    stage_event(
        logger,
        message="pipeline_start",
        pipeline=pipeline,
        stage="pipeline",
        run_id=run_id,
        status="started",
    )

    SILVER_ROOT.mkdir(parents=True, exist_ok=True)

    bronze_payloads = sorted(BRONZE_ROOT.glob("bronze_run_*.json"))

    with pipeline_span(
        "transform",
        pipeline=pipeline,
        stage="transform",
        dataset="silver",
        run_id=run_id,
        attributes={"inputs": len(bronze_payloads)},
    ):
        start = time.perf_counter()
        stage_event(
            logger,
            message="transform_start",
            pipeline=pipeline,
            stage="transform",
            dataset="silver",
            run_id=run_id,
            status="started",
            extra_fields={"inputs": len(bronze_payloads)},
        )

        total_rows = 0
        for payload in bronze_payloads:
            content = json.loads(payload.read_text(encoding="utf-8"))
            total_rows += int(content.get("rows", 0))

        silver_output = SILVER_ROOT / f"silver_run_{run_id}.json"
        silver_output.write_text(
            json.dumps({"run_id": run_id, "rows": total_rows, "transformed_at": utc_now_iso()}, indent=2),
            encoding="utf-8",
        )
        LAST_UPDATE_FILE.write_text(utc_now_iso(), encoding="utf-8")
        time.sleep(0.1)

        stage_event(
            logger,
            message="transform_complete",
            pipeline=pipeline,
            stage="transform",
            dataset="silver",
            run_id=run_id,
            status="success",
            rows=total_rows,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )

    stage_event(
        logger,
        message="pipeline_complete",
        pipeline=pipeline,
        stage="pipeline",
        run_id=run_id,
        status="success",
        dataset="silver",
        rows=total_rows,
    )


def main() -> None:
    run()


if __name__ == "__main__":  # pragma: no cover
    main()
