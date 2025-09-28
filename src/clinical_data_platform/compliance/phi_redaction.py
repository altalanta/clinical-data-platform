from __future__ import annotations

import os
import re
from typing import Any, Dict

REDACTION_TOKEN = "[REDACTED]"

_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b\d{9}\b"),
    re.compile(r"(?i)\b[a-z0-9]{6,12}\b(?=.*\bmrn\b)"),
    re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(r"(?i)(name|patient|first_name|last_name)\s*=\s*[^,;]+"),
]


def redact_text(value: str | Any) -> str | Any:
    if not isinstance(value, str):
        return value
    redacted = value
    for pattern in _PATTERNS:
        redacted = pattern.sub(REDACTION_TOKEN, redacted)
    return redacted


def scrub_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
    sensitive_keys = {
        "name",
        "patient",
        "first_name",
        "last_name",
        "mrn",
        "email",
        "phone",
        "dob",
        "address",
        "ssn",
    }

    def _scrub(item: Any) -> Any:
        if isinstance(item, dict):
            scrubbed: Dict[str, Any] = {}
            for key, value in item.items():
                lowered = key.lower()
                if lowered in sensitive_keys and not isinstance(value, (dict, list)):
                    scrubbed[key] = REDACTION_TOKEN
                else:
                    scrubbed[key] = _scrub(value)
            return scrubbed
        if isinstance(item, list):
            return [_scrub(value) for value in item]
        if isinstance(item, str):
            return redact_text(item)
        return item

    return _scrub(payload)


class PHIFilter:
    def filter(self, record: Any) -> bool:  # pragma: no cover - defensive logging
        try:
            if hasattr(record, "msg") and isinstance(record.msg, str):
                record.msg = redact_text(record.msg)
            for attr in ("patient", "payload", "body", "query", "params", "extra", "data"):
                if hasattr(record, attr):
                    value = getattr(record, attr)
                    if isinstance(value, dict):
                        setattr(record, attr, scrub_dict(value))
                    elif isinstance(value, str):
                        setattr(record, attr, redact_text(value))
        except Exception:
            pass
        return True


def read_only_enabled() -> bool:
    flag = os.getenv("READ_ONLY_MODE", "0")
    return flag in {"1", "true", "TRUE", "yes", "on"}
