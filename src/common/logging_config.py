"""Shared logging setup for pipeline observability."""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict

from pythonjsonlogger import jsonlogger

try:
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
except ImportError:  # pragma: no cover - otel optional at import time
    LoggingInstrumentor = None  # type: ignore[assignment]


_DEFAULT_EVENT_FIELDS = (
    "pipeline",
    "stage",
    "dataset",
    "run_id",
    "latency_ms",
    "rows",
    "status",
    "slo_minutes",
    "freshness_minutes",
)

_LOGGING_CONFIGURED = False


class PipelineJsonFormatter(jsonlogger.JsonFormatter):
    """Formatter that emits JSON logs with pipeline context."""

    def process_log_record(self, log_record: Dict[str, Any]) -> Dict[str, Any]:
        log_record["ts"] = log_record.get("ts") or datetime.now(timezone.utc).isoformat()

        level = log_record.get("level") or log_record.get("levelname")
        if level:
            log_record["level"] = str(level).lower()
        log_record.pop("levelname", None)

        message = log_record.pop("message", None)
        if message is not None:
            log_record["msg"] = message
        log_record.setdefault("msg", "")

        for field in _DEFAULT_EVENT_FIELDS:
            log_record.setdefault(field, None)

        # python-json-logger reserves "exc_info"/"stack_info"; let parent handle.
        return super().process_log_record(log_record)


def configure_logging(level: str | None = None) -> None:
    """Configure root logger for structured JSON output once."""
    global _LOGGING_CONFIGURED

    if _LOGGING_CONFIGURED:
        return

    log_level = (level or os.getenv("LOG_LEVEL") or "INFO").upper()

    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(PipelineJsonFormatter())
    root.addHandler(handler)

    if LoggingInstrumentor is not None:
        LoggingInstrumentor().instrument(set_logging_format=False)

    _LOGGING_CONFIGURED = True


def event_context(**fields: Any) -> Dict[str, Any]:
    """Build a safe context dict for JSON logging."""
    context: Dict[str, Any] = {field: fields.get(field) for field in _DEFAULT_EVENT_FIELDS}

    for key, value in fields.items():
        if key not in context or context[key] is None:
            context[key] = value
    return context


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with configured handlers."""
    configure_logging()
    return logging.getLogger(name)
