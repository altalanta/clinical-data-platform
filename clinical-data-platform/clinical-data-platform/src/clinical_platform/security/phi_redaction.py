from __future__ import annotations

import base64
import hashlib
import hmac
from datetime import date, timedelta
from typing import Optional


def irreversible_hash(value: str, salt: str) -> str:
    """Return a salted SHA-256 hash hex digest."""
    digest = hashlib.sha256()
    digest.update(salt.encode("utf-8"))
    digest.update(value.encode("utf-8"))
    return digest.hexdigest()


def tokenize(value: str, key: str) -> str:
    """Deterministic reversible tokenization via HMAC + value.

    Returns a base64 token combining HMAC and value.
    """
    mac = hmac.new(key.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).digest()
    token = base64.urlsafe_b64encode(mac + b"::" + value.encode("utf-8")).decode("ascii")
    return token


def detokenize(token: str, key: str) -> Optional[str]:
    try:
        raw = base64.urlsafe_b64decode(token.encode("ascii"))
        mac, _, value = raw.partition(b"::")
        expected = hmac.new(key.encode("utf-8"), value, hashlib.sha256).digest()
        if hmac.compare_digest(mac, expected):
            return value.decode("utf-8")
        return None
    except Exception:
        return None


def shift_date(d: date, days: int) -> date:
    return d + timedelta(days=days)

