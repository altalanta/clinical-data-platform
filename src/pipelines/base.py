"""Shared helpers for demo pipeline instrumentation."""
from __future__ import annotations

import logging
import random
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.common.logging_config import event_context, get_logger
from src.common.otel import init_tracer, pipeline_span


def bootstrap(pipeline_name: str) -> tuple[logging.Logger, str]:
    """Initialise logging/tracing and return logger plus run_id."""
    init_tracer(service_name="clinical-data-pipeline", resource_attributes={"pipeline": pipeline_name})
    logger = get_logger(pipeline_name)
    run_id = uuid.uuid4().hex[:12]
    return logger, run_id


def stage_event(
    logger: logging.Logger,
    *,
    message: str,
    pipeline: str,
    stage: str,
    run_id: str,
    status: str,
    dataset: Optional[str] = None,
    rows: Optional[int] = None,
    latency_ms: Optional[float] = None,
    level: int = logging.INFO,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> None:
    record_extra = event_context(
        pipeline=pipeline,
        stage=stage,
        dataset=dataset,
        run_id=run_id,
        status=status,
        rows=rows,
        latency_ms=latency_ms,
    )
    if extra_fields:
        record_extra.update(extra_fields)

    logger.log(level, message, extra=record_extra)


def simulate_latency(min_ms: int = 50, max_ms: int = 450) -> float:
    return round(random.uniform(min_ms, max_ms), 2)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "bootstrap",
    "stage_event",
    "pipeline_span",
    "simulate_latency",
    "utc_now_iso",
]
