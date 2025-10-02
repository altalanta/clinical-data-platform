"""
Tests for PHI redaction security improvements.
Validates that redaction failures don't leak PHI and proper error handling.
"""

import logging
import re
from unittest.mock import Mock, patch
import pytest

from src.clinical_data_platform.compliance.phi_redaction import (
    redact_text,
    scrub_dict,
    PHIFilter,
    REDACTION_TOKEN,
    REDACTION_FAILURE_TOKEN
)


class TestRedactionErrorHandling:
    """Test error handling in PHI redaction functions."""
    
    def test_redact_text_handles_regex_errors(self):
        """Test that regex errors are handled safely."""
        # Mock a regex pattern that raises an error
        with patch('src.clinical_data_platform.compliance.phi_redaction._PATTERNS') as mock_patterns:
            mock_pattern = Mock()
            mock_pattern.sub.side_effect = re.error("Regex error")
            mock_patterns.__iter__ = Mock(return_value=iter([mock_pattern]))
            
            result = redact_text("John Doe SSN: 123-45-6789", "test_context")
            
            # Should return failure token on regex error
            assert result == REDACTION_FAILURE_TOKEN
    
    def test_redact_text_handles_memory_errors(self):
        """Test that memory errors are handled safely."""
        with patch('src.clinical_data_platform.compliance.phi_redaction._PATTERNS') as mock_patterns:
            mock_pattern = Mock()
            mock_pattern.sub.side_effect = MemoryError("Out of memory")
            mock_patterns.__iter__ = Mock(return_value=iter([mock_pattern]))
            
            result = redact_text("test data", "test_context")
            
            assert result == REDACTION_FAILURE_TOKEN
    
    def test_redact_text_handles_unexpected_errors(self):
        """Test that unexpected errors are handled safely."""
        with patch('src.clinical_data_platform.compliance.phi_redaction._PATTERNS') as mock_patterns:
            mock_pattern = Mock()
            mock_pattern.sub.side_effect = RuntimeError("Unexpected error")
            mock_patterns.__iter__ = Mock(return_value=iter([mock_pattern]))
            
            result = redact_text("test data", "test_context")
            
            assert result == REDACTION_FAILURE_TOKEN
    
    def test_redact_text_logs_failures(self, caplog):
        """Test that redaction failures are logged properly."""
        with patch('src.clinical_data_platform.compliance.phi_redaction._PATTERNS') as mock_patterns:
            mock_pattern = Mock()
            mock_pattern.sub.side_effect = re.error("Test error")
            mock_patterns.__iter__ = Mock(return_value=iter([mock_pattern]))
            
            with caplog.at_level(logging.WARNING):
                redact_text("test data", "test_context")
            
            # Check that failure was logged
            assert "PHI redaction failed" in caplog.text
            assert "test_context" in caplog.text
    
    def test_redact_text_preserves_non_strings(self):
        """Test that non-string values are passed through unchanged."""
        assert redact_text(123) == 123
        assert redact_text(None) is None
        assert redact_text([1, 2, 3]) == [1, 2, 3]


class TestDictScrubbing:
    """Test dictionary scrubbing with error handling."""
    
    def test_scrub_dict_handles_depth_limit(self):
        """Test that excessive recursion is prevented."""
        # Create deeply nested dict
        deep_dict = {"level": "data"}
        for i in range(15):  # Exceed max_depth of 10
            deep_dict = {"nested": deep_dict}
        
        result = scrub_dict(deep_dict, max_depth=5)
        
        # Should contain redaction token due to depth limit
        assert REDACTION_TOKEN in str(result)
    
    def test_scrub_dict_handles_circular_references(self):
        """Test handling of circular references."""
        circular_dict = {"name": "John"}
        circular_dict["self"] = circular_dict
        
        # Should not hang or crash
        result = scrub_dict(circular_dict, max_depth=3)
        
        # Name should be redacted
        assert result["name"] == REDACTION_TOKEN
    
    def test_scrub_dict_handles_problematic_keys(self):
        """Test handling of non-string keys and other edge cases."""
        problematic_dict = {
            123: "numeric key",
            None: "none key",
            ("tuple", "key"): "tuple key",
            "name": "John Doe"  # This should be redacted
        }
        
        result = scrub_dict(problematic_dict)
        
        # Normal redaction should still work
        assert result["name"] == REDACTION_TOKEN
        # Other keys should be handled gracefully
        assert 123 in result
        assert None in result
    
    def test_scrub_dict_handles_list_errors(self):
        """Test handling of errors when processing lists."""
        data_with_list = {
            "patients": ["John Doe", "Jane Smith"],
            "name": "Test"
        }
        
        # Mock list processing to raise an error
        with patch('builtins.len', side_effect=TypeError("Length error")):
            result = scrub_dict(data_with_list)
            
            # Should handle the error and redact sensitive data
            assert result["name"] == REDACTION_TOKEN
    
    def test_scrub_dict_complete_failure_fallback(self):
        """Test complete failure fallback returns safe result."""
        with patch('src.clinical_data_platform.compliance.phi_redaction._logger'):
            # Make the main function raise an error
            with patch('builtins.isinstance', side_effect=RuntimeError("Complete failure")):
                result = scrub_dict({"name": "John"})
                
                # Should return safe fallback
                assert result == {"error": REDACTION_FAILURE_TOKEN}


class TestPHIFilter:
    """Test PHI filter error handling."""
    
    def setup_method(self):
        """Set up test environment."""
        self.phi_filter = PHIFilter()
    
    def test_phi_filter_handles_message_redaction_errors(self):
        """Test that message redaction errors are handled."""
        record = Mock()
        record.msg = "Patient John Doe"
        
        with patch('src.clinical_data_platform.compliance.phi_redaction.redact_text', 
                  side_effect=RuntimeError("Redaction error")):
            result = self.phi_filter.filter(record)
            
            # Should still allow the record
            assert result is True
            # Message should be safely redacted
            assert record.msg == REDACTION_FAILURE_TOKEN
    
    def test_phi_filter_handles_attribute_errors(self):
        """Test that attribute redaction errors are handled."""
        record = Mock()
        record.msg = "Test message"
        record.patient = {"name": "John Doe"}
        
        with patch('src.clinical_data_platform.compliance.phi_redaction.scrub_dict',
                  side_effect=RuntimeError("Scrub error")):
            result = self.phi_filter.filter(record)
            
            # Should still allow the record
            assert result is True
            # Attribute should be safely redacted
            assert record.patient == REDACTION_FAILURE_TOKEN
    
    def test_phi_filter_handles_complete_failure(self):
        """Test that complete filter failure is handled safely."""
        record = Mock()
        
        # Make hasattr raise an error
        with patch('builtins.hasattr', side_effect=RuntimeError("Complete failure")):
            result = self.phi_filter.filter(record)
            
            # Should still allow logging (better than no logs)
            assert result is True
    
    def test_phi_filter_handles_list_attributes(self):
        """Test that list attributes are properly redacted."""
        record = Mock()
        record.args = ["Patient: John Doe", {"name": "Jane Smith"}]
        
        result = self.phi_filter.filter(record)
        
        assert result is True
        # Should handle both string and dict items in list
        assert len(record.args) == 2
    
    def test_phi_filter_limits_failure_logging(self):
        """Test that failure logging is limited to prevent spam."""
        record = Mock()
        record.msg = "test"
        
        phi_filter = PHIFilter()
        phi_filter._max_failures = 2  # Set low limit for testing
        
        with patch('src.clinical_data_platform.compliance.phi_redaction.redact_text',
                  side_effect=RuntimeError("Error")):
            # Should log first two failures
            phi_filter.filter(record)
            phi_filter.filter(record)
            phi_filter.filter(record)  # This should not log
            
            assert phi_filter._failure_count == 2
    
    def test_phi_filter_with_request_id_extractor(self):
        """Test PHI filter with request ID extraction."""
        def extract_request_id(record):
            return getattr(record, 'request_id', 'unknown')
        
        phi_filter = PHIFilter(request_id_extractor=extract_request_id)
        record = Mock()
        record.request_id = "test-123"
        record.msg = "Test message"
        
        result = phi_filter.filter(record)
        assert result is True


class TestPHIPatterns:
    """Test PHI pattern detection and redaction."""
    
    def test_ssn_patterns_redacted(self):
        """Test that SSN patterns are redacted."""
        text = "SSN: 123-45-6789 and 987654321"
        result = redact_text(text)
        
        assert "123-45-6789" not in result
        assert "987654321" not in result
        assert REDACTION_TOKEN in result
    
    def test_phone_patterns_redacted(self):
        """Test that phone number patterns are redacted."""
        text = "Call (555) 123-4567 or 555.123.4567"
        result = redact_text(text)
        
        assert "(555) 123-4567" not in result
        assert "555.123.4567" not in result
        assert REDACTION_TOKEN in result
    
    def test_email_patterns_redacted(self):
        """Test that email patterns are redacted."""
        text = "Contact john.doe@example.com for info"
        result = redact_text(text)
        
        assert "john.doe@example.com" not in result
        assert REDACTION_TOKEN in result
    
    def test_date_patterns_redacted(self):
        """Test that date patterns are redacted."""
        text = "Born on 1990-01-15 or 01/15/1990"
        result = redact_text(text)
        
        assert "1990-01-15" not in result
        assert "01/15/1990" not in result
        assert REDACTION_TOKEN in result
    
    def test_sensitive_key_redaction(self):
        """Test that sensitive dictionary keys are redacted."""
        data = {
            "name": "John Doe",
            "patient_id": "12345",
            "age": 30,
            "medical_record_number": "MRN123",
            "treatment": "Therapy A"
        }
        
        result = scrub_dict(data)
        
        # Sensitive fields should be redacted
        assert result["name"] == REDACTION_TOKEN
        assert result["patient_id"] == REDACTION_TOKEN
        assert result["medical_record_number"] == REDACTION_TOKEN
        
        # Non-sensitive fields should remain
        assert result["age"] == 30
        assert result["treatment"] == "Therapy A"
    
    def test_nested_structure_redaction(self):
        """Test redaction in nested structures."""
        data = {
            "patient": {
                "name": "John Doe",
                "contact": {
                    "email": "john@example.com",
                    "phone": "555-1234"
                }
            },
            "visits": [
                {"date": "2023-01-15", "doctor": "Dr. Smith"},
                {"patient_name": "Jane Doe", "notes": "Routine checkup"}
            ]
        }
        
        result = scrub_dict(data)
        
        # Should redact nested sensitive data
        assert result["patient"]["name"] == REDACTION_TOKEN
        assert REDACTION_TOKEN in str(result["patient"]["contact"]["email"])
        assert result["visits"][1]["patient_name"] == REDACTION_TOKEN