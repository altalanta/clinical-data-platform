"""Simulated training pipeline with observability hooks."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

from src.pipelines.base import bootstrap, pipeline_span, stage_event, utc_now_iso

SILVER_ROOT = Path("data/silver")
MODEL_ROOT = Path("artifacts/models")


def run() -> None:
    pipeline = "train"
    logger, run_id = bootstrap(pipeline)

    stage_event(
        logger,
        message="pipeline_start",
        pipeline=pipeline,
        stage="pipeline",
        run_id=run_id,
        status="started",
    )

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    silver_payloads = sorted(SILVER_ROOT.glob("silver_run_*.json"))
    total_rows = 0
    for payload in silver_payloads:
        content = json.loads(payload.read_text(encoding="utf-8"))
        total_rows += int(content.get("rows", 0))

    with pipeline_span(
        "train_model",
        pipeline=pipeline,
        stage="train",
        dataset="silver",
        run_id=run_id,
        attributes={"input_rows": total_rows},
    ):
        start = time.perf_counter()
        stage_event(
            logger,
            message="train_start",
            pipeline=pipeline,
            stage="train",
            dataset="silver",
            run_id=run_id,
            status="started",
            rows=total_rows,
        )
        time.sleep(random.uniform(0.05, 0.2))
        metrics = {
            "accuracy": round(random.uniform(0.7, 0.95), 3),
            "f1": round(random.uniform(0.6, 0.9), 3),
            "trained_at": utc_now_iso(),
            "rows": total_rows,
        }
        model_path = MODEL_ROOT / f"model_{run_id}.json"
        model_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        stage_event(
            logger,
            message="train_complete",
            pipeline=pipeline,
            stage="train",
            dataset="silver",
            run_id=run_id,
            status="success",
            rows=total_rows,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            extra_fields=metrics,
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
