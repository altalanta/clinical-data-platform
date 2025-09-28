from __future__ import annotations

import logging
import logging.config
import os

from pythonjsonlogger import jsonlogger

from .phi_redaction import PHIFilter


def _normalize_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value in {"1", "true", "TRUE", "yes", "on"}


def configure_logging_from_env() -> bool:
    scrub = _normalize_bool(os.getenv("LOG_SCRUB_VALUES")) or _normalize_bool(
        os.getenv("READ_ONLY_MODE")
    )

    level = os.getenv("LOG_LEVEL", "INFO")

    handlers: dict[str, dict[str, object]] = {
        "console": {
            "class": "logging.StreamHandler",
            "level": level,
            "formatter": "json" if scrub else "plain",
            "filters": ["phi"] if scrub else [],
        }
    }

    filters = {"phi": {"()": PHIFilter}} if scrub else {}

    formatters = {
        "plain": {"format": "%(levelname)s %(name)s %(message)s"},
        "json": {
            "()": jsonlogger.JsonFormatter,
            "fmt": "%(levelname)s %(name)s %(message)s %(patient)s %(payload)s",
        },
    }

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": filters,
        "formatters": formatters,
        "handlers": handlers,
        "root": {"level": level, "handlers": ["console"]},
    }

    logging.config.dictConfig(config)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(level)
    return scrub
