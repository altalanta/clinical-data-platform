"""Simulated ingest pipeline instrumented with logging + tracing."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

from src.pipelines.base import bootstrap, pipeline_span, stage_event, utc_now_iso

DATA_ROOT = Path("data/bronze")


def run() -> None:
    pipeline = "ingest"
    logger, run_id = bootstrap(pipeline)

    stage_event(
        logger,
        message="pipeline_start",
        pipeline=pipeline,
        stage="pipeline",
        run_id=run_id,
        status="started",
    )

    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    # Extract stage
    with pipeline_span(
        "extract",
        pipeline=pipeline,
        stage="extract",
        dataset="landing",
        run_id=run_id,
    ):
        start = time.perf_counter()
        stage_event(
            logger,
            message="extract_start",
            pipeline=pipeline,
            stage="extract",
            dataset="landing",
            run_id=run_id,
            status="started",
        )
        time.sleep(random.uniform(0.05, 0.2))
        extracted_rows = random.randint(500, 1500)
        stage_event(
            logger,
            message="extract_complete",
            pipeline=pipeline,
            stage="extract",
            dataset="landing",
            run_id=run_id,
            status="success",
            rows=extracted_rows,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )

    # Load stage
    output_path = DATA_ROOT / f"bronze_run_{run_id}.json"
    with pipeline_span(
        "load",
        pipeline=pipeline,
        stage="load",
        dataset="bronze",
        run_id=run_id,
        attributes={"output_path": str(output_path)},
    ):
        start = time.perf_counter()
        stage_event(
            logger,
            message="load_start",
            pipeline=pipeline,
            stage="load",
            dataset="bronze",
            run_id=run_id,
            status="started",
        )
        sample = {
            "run_id": run_id,
            "rows": extracted_rows,
            "generated_at": utc_now_iso(),
        }
        output_path.write_text(json.dumps(sample, indent=2), encoding="utf-8")
        time.sleep(random.uniform(0.05, 0.15))
        stage_event(
            logger,
            message="load_complete",
            pipeline=pipeline,
            stage="load",
            dataset="bronze",
            run_id=run_id,
            status="success",
            rows=extracted_rows,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )

    stage_event(
        logger,
        message="pipeline_complete",
        pipeline=pipeline,
        stage="pipeline",
        run_id=run_id,
        status="success",
        rows=extracted_rows,
    )


def main() -> None:
    run()


if __name__ == "__main__":  # pragma: no cover
    main()
