import pandas as pd
import pytest

from clinical_platform.validation.pandera_models import AEModel, DMModel
from clinical_platform.logging_utils import redact_pii


def test_dm_schema_strict():
    df = pd.DataFrame({
        'STUDYID': ['S'], 'SUBJID': ['SUBJ1'], 'ARM': ['ACTIVE'], 'SEX': ['M'], 'AGE': [55]
    })
    validated = DMModel.validate(df)
    assert len(validated) == 1


def test_ae_schema_basic():
    df = pd.DataFrame({
        'STUDYID': ['S'], 'SUBJID': ['SUBJ1'], 'AESTDTC': ['2024-01-01'], 'AEENDTC': ['2024-01-02'],
        'AESEV': ['MILD'], 'AESER': [True], 'AEOUT': ['RECOVERED']
    })
    validated = AEModel.validate(df)
    assert len(validated) == 1


def test_ae_date_coercion():
    """Test that date fields are properly coerced to timestamps."""
    df = pd.DataFrame({
        'STUDYID': ['S'], 'SUBJID': ['SUBJ1'], 'AESTDTC': ['2024-01-01T10:30:00'], 
        'AEENDTC': ['2024-01-02T15:45:00'], 'AESEV': ['MILD'], 'AESER': [True], 'AEOUT': ['RECOVERED']
    })
    validated = AEModel.validate(df)
    assert pd.api.types.is_datetime64_any_dtype(validated['AESTDTC'])
    assert pd.api.types.is_datetime64_any_dtype(validated['AEENDTC'])


def test_ae_schema_invalid_severity():
    """Test that invalid severity values are rejected."""
    df = pd.DataFrame({
        'STUDYID': ['S'], 'SUBJID': ['SUBJ1'], 'AESTDTC': ['2024-01-01'], 'AEENDTC': ['2024-01-02'],
        'AESEV': ['INVALID'], 'AESER': [True], 'AEOUT': ['RECOVERED']
    })
    with pytest.raises(Exception):  # pandera.errors.SchemaError
        AEModel.validate(df)


def test_pii_redaction():
    """Test that PII fields are properly redacted from log entries."""
    event_dict = {
        'message': 'Processing subject data',
        'SUBJID': 'SUBJ-001',
        'AGE': 45,
        'SEX': 'M',
        'study': 'STUDY001'
    }
    
    redacted = redact_pii(None, None, event_dict.copy())
    
    assert redacted['SUBJID'] == '***REDACTED***'
    assert redacted['AGE'] == '***REDACTED***'
    assert redacted['SEX'] == '***REDACTED***'
    assert redacted['study'] == 'STUDY001'  # Should not be redacted
    assert redacted['message'] == 'Processing subject data'  # Should not be redacted

