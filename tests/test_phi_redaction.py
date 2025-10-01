from __future__ import annotations

import logging
import sys

from clinical_data_platform.compliance.phi_redaction import (
    REDACTION_TOKEN,
    PHIFilter,
    redact_text,
    scrub_dict,
)


def test_redact_text_basic() -> None:
    sample = "SSN 123-45-6789, email a@b.com, phone (212) 555-1212, dob 1985-05-07"
    result = redact_text(sample)
    assert REDACTION_TOKEN in result
    for snippet in ("123-45-6789", "a@b.com", "555-1212", "1985-05-07"):
        assert snippet not in result


def test_scrub_dict_removes_sensitive_values() -> None:
    payload = {
        "patient": {"name": "Alice Smith", "mrn": "A1B2C3D4"},
        "payload": {"email": "z@y.org", "note": "called 03/01/2020"},
        "ok": True,
    }
    scrubbed = scrub_dict(payload)
    assert scrubbed["patient"]["name"] == REDACTION_TOKEN
    assert scrubbed["patient"]["mrn"] == REDACTION_TOKEN
    assert scrubbed["payload"]["email"] == REDACTION_TOKEN
    assert REDACTION_TOKEN in scrubbed["payload"]["note"]


def test_logging_filter_redacts_message(capsys) -> None:
    logger = logging.getLogger("test.phi")
    logger.handlers = []
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.addFilter(PHIFilter())
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    logger.info("patient=Alice Smith SSN 123-45-6789 phone 212-555-1212")
    captured = capsys.readouterr().out
    assert "Alice Smith" not in captured
    assert "123-45-6789" not in captured
    assert "212-555-1212" not in captured
    assert REDACTION_TOKEN in captured
