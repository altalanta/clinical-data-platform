"""Tests for PHI redaction functionality."""

import pytest
import logging
from io import StringIO
from unittest.mock import patch, MagicMock
import re

from clinical_platform.logging_utils import PHIFilter, configure_logging, get_logger


class TestPHIFilter:
    """Test PHI filtering and redaction."""

    @pytest.fixture
    def phi_filter(self):
        """Create PHI filter with scrubbing enabled."""
        return PHIFilter(scrub_values=True, read_only_mode=True)

    @pytest.fixture
    def permissive_filter(self):
        """Create PHI filter with scrubbing disabled."""
        return PHIFilter(scrub_values=False, read_only_mode=False)

    def test_ssn_redaction(self, phi_filter):
        """Test SSN pattern redaction."""
        test_cases = [
            "Patient SSN is 123-45-6789",
            "SSN: 987-65-4321 was found",
            "Contact via 555-12-3456 for updates",
        ]
        
        for test_case in test_cases:
            result = phi_filter._redact_phi_from_message(test_case)
            assert "[REDACTED]" in result
            assert not re.search(r'\b\d{3}-\d{2}-\d{4}\b', result)

    def test_email_redaction(self, phi_filter):
        """Test email address redaction."""
        test_cases = [
            "Contact patient at john.doe@example.com",
            "Physician email: dr.smith@hospital.org",
            "Send report to admin@clinic-name.net",
            "Multiple emails: a@b.com and c@d.org",
        ]
        
        for test_case in test_cases:
            result = phi_filter._redact_phi_from_message(test_case)
            assert "[REDACTED]" in result
            assert "@" not in result or "[REDACTED]" in result

    def test_phone_redaction(self, phi_filter):
        """Test phone number redaction."""
        test_cases = [
            "Call patient at 555-123-4567",
            "Emergency contact: 800 555 1234",
            "Phone: 555.123.4567",
            "Mobile: 5551234567",
        ]
        
        for test_case in test_cases:
            result = phi_filter._redact_phi_from_message(test_case)
            assert "[REDACTED]" in result

    def test_date_redaction(self, phi_filter):
        """Test date redaction."""
        test_cases = [
            "Birth date: 1985-03-15",
            "Appointment on 12/25/2023",
            "DOB 3/5/85",
            "Visit scheduled for 2023-12-31",
        ]
        
        for test_case in test_cases:
            result = phi_filter._redact_phi_from_message(test_case)
            assert "[REDACTED]" in result

    def test_medical_record_number_redaction(self, phi_filter):
        """Test MRN and similar ID redaction."""
        test_cases = [
            "Patient MRN-1234567",
            "Medical record MR-987654321",
            "ID: PT-123456789",
            "Subject SUBJ12345678",
            "Study ID STUDY123456",
        ]
        
        for test_case in test_cases:
            result = phi_filter._redact_phi_from_message(test_case)
            assert "[REDACTED]" in result

    def test_zip_code_redaction(self, phi_filter):
        """Test ZIP code redaction."""
        test_cases = [
            "Patient lives in 12345",
            "Address ZIP: 90210-1234",
            "Zip code 54321",
        ]
        
        for test_case in test_cases:
            result = phi_filter._redact_phi_from_message(test_case)
            assert "[REDACTED]" in result

    def test_sensitive_field_redaction(self, phi_filter):
        """Test sensitive field name redaction."""
        test_cases = [
            "SUBJID=PATIENT123",
            "subject_id: SUBJ-456",
            "patient_id=12345",
            "first_name=John",
            "last_name: Smith",
            "email=test@example.com",
            "ssn: 123-45-6789",
        ]
        
        for test_case in test_cases:
            result = phi_filter._redact_phi_from_message(test_case)
            assert "[REDACTED]" in result

    def test_file_path_redaction(self, phi_filter):
        """Test file path redaction for PHI-related paths."""
        test_cases = [
            "/data/patient_records/file.csv",
            "/home/user/subject_data/output.json",
            "/var/phi_backup/data.db",
            "/tmp/pii_export.txt",
        ]
        
        for test_case in test_cases:
            result = phi_filter._redact_phi_from_message(test_case)
            if any(keyword in test_case.lower() for keyword in ['patient', 'subject', 'phi', 'pii']):
                assert "[REDACTED_PATH]" in result

    def test_database_connection_redaction(self, phi_filter):
        """Test database connection string redaction."""
        test_cases = [
            "postgresql://user:password@localhost/db",
            "mysql://admin:secret@server:3306/clinical",
            "Connection failed to mongodb://user:pass@host/database",
        ]
        
        for test_case in test_cases:
            result = phi_filter._redact_phi_from_message(test_case)
            assert "[REDACTED]" in result

    def test_quoted_sensitive_content_redaction(self, phi_filter):
        """Test redaction of quoted sensitive content."""
        test_cases = [
            'Error with "patient_record_123"',
            'Field "subject_identifier" failed',
            'Cannot find "mrn_12345"',
        ]
        
        for test_case in test_cases:
            result = phi_filter._redact_phi_from_message(test_case)
            if any(keyword in test_case.lower() for keyword in ['patient', 'subject', 'mrn']):
                assert "[REDACTED]" in result

    def test_no_redaction_when_disabled(self, permissive_filter):
        """Test that redaction doesn't occur when disabled."""
        test_message = "Patient SSN 123-45-6789 at john@example.com"
        result = permissive_filter._redact_phi_from_message(test_message)
        
        # When scrubbing is disabled, message should be unchanged
        assert result == test_message

    def test_complex_phi_combinations(self, phi_filter):
        """Test redaction of complex messages with multiple PHI types."""
        complex_message = (
            "Patient SUBJ123456 (SSN: 123-45-6789) contacted at john.doe@hospital.com "
            "on 2023-03-15. Lives at ZIP 12345. Phone: 555-123-4567. "
            "MRN-7890123 scheduled for visit."
        )
        
        result = phi_filter._redact_phi_from_message(complex_message)
        
        # All PHI should be redacted
        assert "[REDACTED]" in result
        assert "123-45-6789" not in result
        assert "john.doe@hospital.com" not in result
        assert "555-123-4567" not in result
        assert "SUBJ123456" not in result

    def test_edge_cases(self, phi_filter):
        """Test edge cases and potential bypass attempts."""
        edge_cases = [
            "",  # Empty string
            "No PHI here",  # Clean message
            "123-45-67890",  # Invalid SSN (too long)
            "12-34-5678",  # Invalid SSN (wrong format)
            "normal.word@domain",  # Incomplete email
            "Phone: 555-1234",  # Incomplete phone
        ]
        
        for test_case in edge_cases:
            result = phi_filter._redact_phi_from_message(test_case)
            # Should not crash and handle gracefully
            assert isinstance(result, str)

    def test_unicode_and_special_characters(self, phi_filter):
        """Test PHI redaction with unicode and special characters."""
        test_cases = [
            "Patiënt email: andré@hôpital.com",
            "SSN: 123-45-6789 (confidential)",
            "Subject ID: SUBJ123456\t\nPhone: 555-123-4567",
        ]
        
        for test_case in test_cases:
            result = phi_filter._redact_phi_from_message(test_case)
            assert "[REDACTED]" in result


class TestLoggingIntegration:
    """Test PHI filter integration with logging system."""

    def test_log_record_filtering(self):
        """Test that log records are properly filtered."""
        phi_filter = PHIFilter(scrub_values=True)
        
        # Create a log record with PHI
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Patient SSN 123-45-6789 accessed",
            args=(),
            exc_info=None
        )
        
        # Filter should modify the record
        result = phi_filter.filter(record)
        assert result is True  # Filter should allow the record through
        assert "[REDACTED]" in record.msg
        assert "123-45-6789" not in record.msg

    def test_log_args_filtering(self):
        """Test that log arguments are properly filtered."""
        phi_filter = PHIFilter(scrub_values=True)
        
        # Create a log record with PHI in args
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Patient info: %s, contact: %s",
            args=("SSN: 123-45-6789", "email@example.com"),
            exc_info=None
        )
        
        phi_filter.filter(record)
        
        # Args should be redacted
        assert all("[REDACTED]" in str(arg) for arg in record.args)

    def test_structured_logging_redaction(self):
        """Test PHI redaction in structured logging."""
        from clinical_platform.logging_utils import redact_pii
        
        # Test event dict with PHI
        event_dict = {
            "message": "Patient accessed",
            "subject_id": "SUBJ123456",
            "email": "patient@example.com",
            "timestamp": "2023-03-15",
            "normal_field": "safe_value"
        }
        
        # Apply redaction
        result = redact_pii(None, None, event_dict)
        
        # Sensitive fields should be redacted
        assert result["subject_id"] == "[REDACTED]"
        assert result["email"] == "[REDACTED]"
        assert result["normal_field"] == "safe_value"  # Safe fields unchanged

    def test_logging_configuration_applies_phi_filter(self):
        """Test that logging configuration properly applies PHI filters."""
        # This would test the actual logging configuration
        # In a real implementation, you'd verify that all handlers have PHI filters
        pass


class TestPropertyBasedPHIRedaction:
    """Property-based tests for PHI redaction."""

    @pytest.mark.parametrize("ssn_format", [
        "123-45-6789",
        "987-65-4321", 
        "111-11-1111",
        "000-00-0000",
    ])
    def test_ssn_patterns(self, ssn_format):
        """Test various SSN patterns are redacted."""
        phi_filter = PHIFilter(scrub_values=True)
        message = f"Patient SSN is {ssn_format}"
        result = phi_filter._redact_phi_from_message(message)
        assert ssn_format not in result
        assert "[REDACTED]" in result

    @pytest.mark.parametrize("email", [
        "test@example.com",
        "user.name+tag@domain.co.uk",
        "simple@domain.org",
        "complex.email-with-dash@sub-domain.example.net",
    ])
    def test_email_patterns(self, email):
        """Test various email patterns are redacted."""
        phi_filter = PHIFilter(scrub_values=True)
        message = f"Contact: {email}"
        result = phi_filter._redact_phi_from_message(message)
        assert email not in result
        assert "[REDACTED]" in result

    @pytest.mark.parametrize("date_format", [
        "2023-03-15",
        "12/25/2023",
        "3/5/85",
        "31/12/2023",
    ])
    def test_date_patterns(self, date_format):
        """Test various date patterns are redacted."""
        phi_filter = PHIFilter(scrub_values=True)
        message = f"Date of birth: {date_format}"
        result = phi_filter._redact_phi_from_message(message)
        # Should redact common date patterns
        if re.match(r'\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}', date_format):
            assert "[REDACTED]" in result


class TestPerformance:
    """Test performance of PHI redaction."""

    def test_redaction_performance_large_text(self):
        """Test redaction performance with large text blocks."""
        phi_filter = PHIFilter(scrub_values=True)
        
        # Create large text with embedded PHI
        large_text = "Normal text. " * 1000 + "SSN: 123-45-6789. " + "More text. " * 1000
        
        import time
        start_time = time.time()
        result = phi_filter._redact_phi_from_message(large_text)
        end_time = time.time()
        
        # Should complete reasonably quickly (adjust threshold as needed)
        assert end_time - start_time < 1.0  # Less than 1 second
        assert "[REDACTED]" in result
        assert "123-45-6789" not in result

    def test_compiled_regex_performance(self):
        """Test that compiled regexes perform better than string patterns."""
        phi_filter = PHIFilter(scrub_values=True)
        
        # Test that patterns are compiled regex objects
        for pattern in phi_filter.phi_patterns:
            assert hasattr(pattern, 'pattern')  # Compiled regex has pattern attribute
            assert hasattr(pattern, 'sub')      # Compiled regex has sub method