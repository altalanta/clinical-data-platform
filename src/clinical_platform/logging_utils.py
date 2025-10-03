from __future__ import annotations

import logging
import os
import re
import uuid
from contextvars import ContextVar
from typing import Any, Dict, List, Optional
from pathlib import Path

import structlog
from pythonjsonlogger import jsonlogger


# Context variables for request tracking
request_id_var: ContextVar[str] = ContextVar('request_id', default='')
data_lineage_id_var: ContextVar[str] = ContextVar('data_lineage_id', default='')


class PHIFilter(logging.Filter):
    """Filter to redact PHI/PII from log records."""
    
    def __init__(self, scrub_values: bool = None, read_only_mode: bool = None):
        super().__init__()
        self.scrub_values = scrub_values if scrub_values is not None else bool(os.getenv('LOG_SCRUB_VALUES', '1') == '1')
        self.read_only_mode = read_only_mode if read_only_mode is not None else bool(os.getenv('READ_ONLY_MODE', '0') == '1')
        
        # Enhanced PHI/PII patterns to redact
        self.phi_patterns = [
            re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),  # SSN
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),  # Email
            re.compile(r'\b\d{3}-\d{3}-\d{4}\b'),  # Phone
            re.compile(r'\b\d{4}-\d{4}-\d{4}-\d{4}\b'),  # Credit card
            re.compile(r'\b\d{4}-\d{2}-\d{2}\b'),  # ISO date YYYY-MM-DD
            re.compile(r'\b\d{2}/\d{2}/\d{4}\b'),  # US date MM/DD/YYYY
            re.compile(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'),  # Various date formats
            re.compile(r'\b[A-Z]{2,3}-\d{4,10}\b'),  # MRN-like (e.g., MRN-123456)
            re.compile(r'\bSUBJ\d{3,8}\b'),  # Subject IDs
            re.compile(r'\bPT\d{3,8}\b'),  # Patient IDs  
            re.compile(r'\bSTUDY\d{3,8}\b'),  # Study IDs
            re.compile(r'\b\d{8,12}\b'),  # Long numeric IDs
            re.compile(r'\b[A-Z]{1,2}\d{6,10}\b'),  # Alpha-numeric IDs
            re.compile(r'\b\d{3,4}\s?\d{3,4}\s?\d{4}\b'),  # Phone variations
            re.compile(r'\b\d{5}(-\d{4})?\b'),  # ZIP codes
        ]
        
        # Sensitive field names
        self.sensitive_fields = {
            'SUBJID', 'USUBJID', 'AGE', 'SEX', 'RACE', 'ETHNIC', 
            'ssn', 'social_security', 'dob', 'date_of_birth', 'subject_id',
            'patient_id', 'first_name', 'last_name', 'full_name',
            'address', 'phone', 'email', 'zip_code', 'postal_code'
        }
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter and redact PHI from log records."""
        if self.scrub_values or self.read_only_mode:
            record.msg = self._redact_phi_from_message(str(record.msg))
            
            # Redact from args if present
            if hasattr(record, 'args') and record.args:
                record.args = tuple(self._redact_phi_from_message(str(arg)) for arg in record.args)
        
        return True
    
    def _redact_phi_from_message(self, message: str) -> str:
        """Redact PHI patterns from message text."""
        redacted = message
        
        # Apply compiled regex patterns
        for pattern in self.phi_patterns:
            redacted = pattern.sub('[REDACTED]', redacted)
        
        # Redact sensitive field values in key=value patterns
        for field in self.sensitive_fields:
            pattern = rf'\b{field}\s*[=:]\s*[^\s,\]}}]+'
            redacted = re.sub(pattern, f'{field}=[REDACTED]', redacted, flags=re.IGNORECASE)
        
        # Additional redaction for common patterns in error messages
        # File paths that might contain PHI
        redacted = re.sub(r'/[^/\s]*(?:patient|subject|phi|pii)[^/\s]*[^\s]*', '/[REDACTED_PATH]', redacted, flags=re.IGNORECASE)
        
        # Database connection strings
        redacted = re.sub(r'://[^@]+@[^/]+/', '://[REDACTED]@[REDACTED]/', redacted)
        
        # Potential sensitive strings in quotes
        redacted = re.sub(r'"[^"]*(?:ssn|social|mrn|patient|subject)[^"]*"', '"[REDACTED]"', redacted, flags=re.IGNORECASE)
        
        return redacted


class PHIFilterFormatter(logging.Formatter):
    """Custom formatter that applies PHI filtering."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phi_filter = PHIFilter()
    
    def format(self, record: logging.LogRecord) -> str:
        # Apply PHI filtering
        self.phi_filter.filter(record)
        
        # Add context variables if available
        record.request_id = request_id_var.get('')
        record.data_lineage_id = data_lineage_id_var.get('')
        
        return super().format(record)


def redact_pii(logger, name, event_dict):
    """Redact PII fields from log entries for clinical data compliance."""
    phi_filter = PHIFilter()
    
    # Redact sensitive keys
    for key in list(event_dict.keys()):
        if key.lower() in phi_filter.sensitive_fields:
            event_dict[key] = '***REDACTED***'
        elif isinstance(event_dict[key], str):
            event_dict[key] = phi_filter._redact_phi_from_message(event_dict[key])
    
    return event_dict


def add_context_vars(logger, name, event_dict):
    """Add context variables to event dict."""
    request_id = request_id_var.get('')
    data_lineage_id = data_lineage_id_var.get('')
    
    if request_id:
        event_dict['request_id'] = request_id
    if data_lineage_id:
        event_dict['data_lineage_id'] = data_lineage_id
    
    return event_dict


def configure_logging(config_path: Optional[str] = None) -> None:
    """Configure logging from YAML config file."""
    import logging.config
    import yaml
    
    if config_path is None:
        config_path = os.getenv('LOG_CONFIG_PATH', 'configs/logging.yaml')
    
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Ensure log directory exists
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        logging.config.dictConfig(config)
    else:
        # Fallback to basic configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)8s] %(name)s: %(message)s (%(filename)s:%(lineno)d)'
        )
    
    # Configure structlog
    structlog.configure(
        processors=[
            add_context_vars,
            redact_pii,
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str, **context: Dict[str, Any]):
    """Get a structured logger with optional context."""
    log = structlog.get_logger(name)
    return log.bind(**context)


def set_request_id(request_id: str = None) -> str:
    """Set request ID for current context."""
    if request_id is None:
        request_id = str(uuid.uuid4())
    request_id_var.set(request_id)
    return request_id


def set_data_lineage_id(lineage_id: str = None) -> str:
    """Set data lineage ID for current context."""
    if lineage_id is None:
        lineage_id = str(uuid.uuid4())
    data_lineage_id_var.set(lineage_id)
    return lineage_id


def get_request_id() -> str:
    """Get current request ID."""
    return request_id_var.get('')


def get_data_lineage_id() -> str:
    """Get current data lineage ID."""
    return data_lineage_id_var.get('')

