from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Optional, Union

REDACTION_TOKEN = "[REDACTED]"
REDACTION_FAILURE_TOKEN = "[REDACTION_FAILED]"

# Configure logger for redaction failures
_logger = logging.getLogger(__name__)

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


def redact_text(value: Union[str, Any], context: Optional[str] = None) -> Union[str, Any]:
    """
    Redact PHI patterns from text with safe error handling.
    
    Args:
        value: The value to redact (only processes strings)
        context: Optional context for logging failures
    
    Returns:
        Redacted text or original value if not a string
    """
    if not isinstance(value, str):
        return value
    
    try:
        redacted = value
        for pattern in _PATTERNS:
            redacted = pattern.sub(REDACTION_TOKEN, redacted)
        return redacted
    except (re.error, AttributeError, MemoryError) as e:
        # Log specific redaction failures but return safe fallback
        _logger.warning(
            "PHI redaction failed",
            extra={
                "error_type": type(e).__name__,
                "context": context or "unknown",
                "original_length": len(value) if isinstance(value, str) else None,
                "phi_redaction": "failed"
            }
        )
        # Return safe fallback - completely redact on failure
        return REDACTION_FAILURE_TOKEN
    except Exception as e:
        # Catch any other unexpected errors
        _logger.error(
            "Unexpected error during PHI redaction",
            extra={
                "error_type": type(e).__name__,
                "context": context or "unknown",
                "phi_redaction": "failed"
            }
        )
        # Return safe fallback
        return REDACTION_FAILURE_TOKEN


def scrub_dict(payload: Dict[str, Any], max_depth: int = 10, context: Optional[str] = None) -> Dict[str, Any]:
    """
    Scrub sensitive data from dictionary structures with safe error handling.
    
    Args:
        payload: Dictionary to scrub
        max_depth: Maximum recursion depth to prevent infinite loops
        context: Optional context for logging failures
    
    Returns:
        Scrubbed dictionary with PHI removed or redacted
    """
    sensitive_keys = {
        "name", "patient", "first_name", "last_name", "mrn", "email", 
        "phone", "dob", "date_of_birth", "address", "ssn", "social_security",
        "patient_id", "medical_record_number", "phone_number", "zip_code",
        "postal_code", "credit_card", "account_number"
    }

    def _scrub(item: Any, depth: int = 0) -> Any:
        # Prevent excessive recursion
        if depth > max_depth:
            _logger.warning(
                "PHI scrubbing depth limit exceeded",
                extra={"max_depth": max_depth, "context": context, "phi_redaction": "truncated"}
            )
            return REDACTION_TOKEN
        
        try:
            if isinstance(item, dict):
                scrubbed: Dict[str, Any] = {}
                for key, value in item.items():
                    try:
                        lowered = str(key).lower()  # Safely convert key to string
                        if lowered in sensitive_keys and not isinstance(value, (dict, list)):
                            scrubbed[key] = REDACTION_TOKEN
                        else:
                            scrubbed[key] = _scrub(value, depth + 1)
                    except (AttributeError, TypeError) as e:
                        # Handle problematic keys/values safely
                        _logger.warning(
                            "Error processing dict key during PHI scrubbing",
                            extra={
                                "error_type": type(e).__name__,
                                "key_type": type(key).__name__,
                                "context": context,
                                "phi_redaction": "failed"
                            }
                        )
                        scrubbed[key] = REDACTION_FAILURE_TOKEN
                return scrubbed
            
            elif isinstance(item, list):
                try:
                    return [_scrub(value, depth + 1) for value in item]
                except (TypeError, ValueError) as e:
                    _logger.warning(
                        "Error processing list during PHI scrubbing",
                        extra={
                            "error_type": type(e).__name__,
                            "list_length": len(item) if hasattr(item, '__len__') else None,
                            "context": context,
                            "phi_redaction": "failed"
                        }
                    )
                    return [REDACTION_FAILURE_TOKEN]
            
            elif isinstance(item, str):
                return redact_text(item, context)
            
            else:
                return item
                
        except Exception as e:
            # Catch any other unexpected errors during recursion
            _logger.error(
                "Unexpected error during PHI scrubbing",
                extra={
                    "error_type": type(e).__name__,
                    "item_type": type(item).__name__,
                    "depth": depth,
                    "context": context,
                    "phi_redaction": "failed"
                }
            )
            return REDACTION_FAILURE_TOKEN

    try:
        return _scrub(payload)
    except Exception as e:
        # Final safety net - if entire scrubbing fails, return empty dict
        _logger.error(
            "Complete PHI scrubbing failure",
            extra={
                "error_type": type(e).__name__,
                "context": context,
                "phi_redaction": "failed"
            }
        )
        return {"error": REDACTION_FAILURE_TOKEN}


class PHIFilter:
    """
    Logging filter that redacts PHI from log records with comprehensive error handling.
    """
    
    def __init__(self, request_id_extractor=None):
        """
        Initialize PHI filter.
        
        Args:
            request_id_extractor: Optional function to extract request ID from record
        """
        self.request_id_extractor = request_id_extractor
        self._failure_count = 0
        self._max_failures = 100  # Prevent log spam
    
    def filter(self, record: Any) -> bool:
        """
        Filter log record to redact PHI.
        
        Args:
            record: Log record to filter
            
        Returns:
            True to allow the record, False to block it
        """
        try:
            # Extract context for better error tracking
            request_id = None
            if self.request_id_extractor:
                try:
                    request_id = self.request_id_extractor(record)
                except Exception:
                    pass  # Don't fail on request ID extraction
            
            context = f"log_filter_{request_id}" if request_id else "log_filter"
            
            # Redact message
            if hasattr(record, "msg") and isinstance(record.msg, str):
                try:
                    record.msg = redact_text(record.msg, context)
                except Exception as e:
                    if self._failure_count < self._max_failures:
                        _logger.warning(
                            "Failed to redact log message",
                            extra={
                                "error_type": type(e).__name__,
                                "request_id": request_id,
                                "phi_redaction": "failed"
                            }
                        )
                        self._failure_count += 1
                    record.msg = REDACTION_FAILURE_TOKEN
            
            # Redact common PHI attributes
            phi_attrs = ("patient", "payload", "body", "query", "params", "extra", "data", "args")
            for attr in phi_attrs:
                if hasattr(record, attr):
                    try:
                        value = getattr(record, attr)
                        if isinstance(value, dict):
                            setattr(record, attr, scrub_dict(value, context=context))
                        elif isinstance(value, str):
                            setattr(record, attr, redact_text(value, context))
                        elif isinstance(value, (list, tuple)):
                            # Handle lists/tuples that might contain PHI
                            redacted_items = []
                            for item in value:
                                if isinstance(item, dict):
                                    redacted_items.append(scrub_dict(item, context=context))
                                elif isinstance(item, str):
                                    redacted_items.append(redact_text(item, context))
                                else:
                                    redacted_items.append(item)
                            setattr(record, attr, type(value)(redacted_items))
                    except Exception as e:
                        if self._failure_count < self._max_failures:
                            _logger.warning(
                                f"Failed to redact log attribute '{attr}'",
                                extra={
                                    "error_type": type(e).__name__,
                                    "attribute": attr,
                                    "request_id": request_id,
                                    "phi_redaction": "failed"
                                }
                            )
                            self._failure_count += 1
                        # Set safe fallback value
                        setattr(record, attr, REDACTION_FAILURE_TOKEN)
            
            return True
            
        except Exception as e:
            # Final safety net - if filtering completely fails, still allow the log
            # but try to log the failure (without creating infinite recursion)
            if self._failure_count < self._max_failures:
                try:
                    _logger.critical(
                        "Complete PHI filter failure - log may contain unredacted PHI",
                        extra={
                            "error_type": type(e).__name__,
                            "phi_redaction": "completely_failed"
                        }
                    )
                    self._failure_count += 1
                except Exception:
                    pass  # Prevent infinite recursion
            
            # Still return True to allow logging - better to have unredacted logs 
            # than no logs in critical situations, but we've warned about it
            return True


def read_only_enabled() -> bool:
    flag = os.getenv("READ_ONLY_MODE", "0")
    return flag in {"1", "true", "TRUE", "yes", "on"}
