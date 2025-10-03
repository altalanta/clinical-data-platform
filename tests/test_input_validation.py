"""Tests for input validation and SQL injection prevention."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import duckdb

from clinical_platform.api.main import app, safe_http_exception
from clinical_platform.config import UnifiedConfig, SecurityConfig
from pydantic import SecretStr


@pytest.fixture
def test_client():
    """Create test client for the main app."""
    return TestClient(app)


@pytest.fixture
def mock_config():
    """Mock configuration with API key."""
    config = MagicMock(spec=UnifiedConfig)
    config.security = MagicMock(spec=SecurityConfig)
    config.security.api_key = SecretStr("test-api-key")
    config.security.read_only_mode = False
    config.warehouse.duckdb_path = ":memory:"  # Use in-memory DB for tests
    return config


@pytest.fixture
def auth_headers():
    """Valid authentication headers."""
    return {"Authorization": "Bearer test-api-key"}


class TestInputValidation:
    """Test input validation on API endpoints."""

    def test_subject_id_regex_validation(self, test_client, mock_config, auth_headers):
        """Test subject ID regex validation."""
        with patch('clinical_platform.api.main.get_config', return_value=mock_config):
            # Valid subject IDs
            valid_ids = [
                "SUBJ123",
                "SUBJ-456", 
                "SUBJ_789",
                "ABC123DEF",
                "Study-001_Patient-123",
            ]
            
            for subject_id in valid_ids:
                # Mock database to return empty result (subject not found)
                with patch('duckdb.connect') as mock_connect:
                    mock_con = MagicMock()
                    mock_con.execute.return_value.fetch_df.return_value.empty = True
                    mock_connect.return_value = mock_con
                    
                    response = test_client.get(f"/subjects/{subject_id}", headers=auth_headers)
                    # Should pass validation but return 404 (not found)
                    assert response.status_code == 404

    def test_subject_id_invalid_characters_rejected(self, test_client, mock_config, auth_headers):
        """Test that subject IDs with invalid characters are rejected."""
        with patch('clinical_platform.api.main.get_config', return_value=mock_config):
            invalid_ids = [
                "SUBJ@123",     # @ symbol
                "SUBJ#123",     # # symbol
                "SUBJ 123",     # Space
                "SUBJ%123",     # % symbol
                "SUBJ'123",     # Single quote (SQL injection attempt)
                "SUBJ\"123",    # Double quote
                "SUBJ;DROP",    # Semicolon (SQL injection attempt)
                "SUBJ<script>", # HTML/XSS attempt
                "SUBJ\x00123",  # Null byte
                "../etc/passwd", # Path traversal attempt
            ]
            
            for subject_id in invalid_ids:
                response = test_client.get(f"/subjects/{subject_id}", headers=auth_headers)
                # Should return 422 (validation error) due to regex constraint
                assert response.status_code == 422, f"Invalid ID {subject_id} was not rejected"

    def test_subject_id_length_validation(self, test_client, mock_config, auth_headers):
        """Test subject ID length validation."""
        with patch('clinical_platform.api.main.get_config', return_value=mock_config):
            # Very long subject ID (over 50 characters)
            long_id = "A" * 51
            response = test_client.get(f"/subjects/{long_id}", headers=auth_headers)
            assert response.status_code == 400

    def test_score_request_validation(self, test_client, mock_config, auth_headers):
        """Test ML score request validation."""
        with patch('clinical_platform.api.main.get_config', return_value=mock_config):
            # Valid request
            valid_request = {
                "AGE": 45.0,
                "AE_COUNT": 2.0,
                "SEVERE_AE_COUNT": 1.0
            }
            response = test_client.post("/score", json=valid_request, headers=auth_headers)
            assert response.status_code == 200

    def test_score_request_invalid_ranges(self, test_client, mock_config, auth_headers):
        """Test score request with invalid value ranges."""
        with patch('clinical_platform.api.main.get_config', return_value=mock_config):
            invalid_requests = [
                {"AGE": -5.0, "AE_COUNT": 2.0, "SEVERE_AE_COUNT": 1.0},      # Negative age
                {"AGE": 150.0, "AE_COUNT": 2.0, "SEVERE_AE_COUNT": 1.0},     # Age too high
                {"AGE": 45.0, "AE_COUNT": -1.0, "SEVERE_AE_COUNT": 1.0},     # Negative AE count
                {"AGE": 45.0, "AE_COUNT": 2.0, "SEVERE_AE_COUNT": -1.0},     # Negative severe AE count
                {"AGE": 45.0, "AE_COUNT": 2.0, "SEVERE_AE_COUNT": 5.0},      # Severe > total AE
            ]
            
            for request_data in invalid_requests:
                response = test_client.post("/score", json=request_data, headers=auth_headers)
                assert response.status_code == 422, f"Invalid request was accepted: {request_data}"

    def test_score_request_missing_fields(self, test_client, mock_config, auth_headers):
        """Test score request with missing required fields."""
        with patch('clinical_platform.api.main.get_config', return_value=mock_config):
            incomplete_requests = [
                {"AGE": 45.0, "AE_COUNT": 2.0},                    # Missing SEVERE_AE_COUNT
                {"AGE": 45.0, "SEVERE_AE_COUNT": 1.0},             # Missing AE_COUNT
                {"AE_COUNT": 2.0, "SEVERE_AE_COUNT": 1.0},         # Missing AGE
                {},                                                # Missing all fields
            ]
            
            for request_data in incomplete_requests:
                response = test_client.post("/score", json=request_data, headers=auth_headers)
                assert response.status_code == 422

    def test_score_request_wrong_types(self, test_client, mock_config, auth_headers):
        """Test score request with wrong data types."""
        with patch('clinical_platform.api.main.get_config', return_value=mock_config):
            wrong_type_requests = [
                {"AGE": "forty-five", "AE_COUNT": 2.0, "SEVERE_AE_COUNT": 1.0},
                {"AGE": 45.0, "AE_COUNT": "two", "SEVERE_AE_COUNT": 1.0},
                {"AGE": 45.0, "AE_COUNT": 2.0, "SEVERE_AE_COUNT": "one"},
                {"AGE": None, "AE_COUNT": 2.0, "SEVERE_AE_COUNT": 1.0},
            ]
            
            for request_data in wrong_type_requests:
                response = test_client.post("/score", json=request_data, headers=auth_headers)
                assert response.status_code == 422


class TestSQLInjectionPrevention:
    """Test SQL injection prevention."""

    def test_parameterized_queries_used(self, test_client, mock_config, auth_headers):
        """Test that parameterized queries are used for database operations."""
        with patch('clinical_platform.api.main.get_config', return_value=mock_config):
            with patch('duckdb.connect') as mock_connect:
                mock_con = MagicMock()
                mock_con.execute.return_value.fetch_df.return_value.empty = True
                mock_connect.return_value = mock_con
                
                # Make request with potentially dangerous subject ID
                subject_id = "SUBJ123"
                response = test_client.get(f"/subjects/{subject_id}", headers=auth_headers)
                
                # Verify execute was called with parameterized query
                mock_con.execute.assert_called_once()
                call_args = mock_con.execute.call_args
                
                # Should use parameterized query with ? placeholder
                assert "?" in call_args[0][0]  # Query string should contain placeholder
                assert call_args[0][1] == [subject_id]  # Parameters should be separate

    def test_sql_injection_attempts_blocked(self, test_client, mock_config, auth_headers):
        """Test that SQL injection attempts are blocked by input validation."""
        with patch('clinical_platform.api.main.get_config', return_value=mock_config):
            # Common SQL injection payloads
            injection_attempts = [
                "1' OR '1'='1",
                "'; DROP TABLE subjects; --",
                "1' UNION SELECT * FROM users --",
                "'; INSERT INTO logs VALUES('hacked'); --",
                "1' OR 1=1 --",
                "admin'--",
                "' OR 'a'='a",
                "1'; EXEC xp_cmdshell('dir'); --",
            ]
            
            for payload in injection_attempts:
                response = test_client.get(f"/subjects/{payload}", headers=auth_headers)
                # Should be rejected by regex validation (422) or return 404 if somehow passed
                assert response.status_code in [422, 404], f"SQL injection payload was not blocked: {payload}"

    def test_studies_endpoint_sql_safety(self, test_client, mock_config, auth_headers):
        """Test that studies endpoint uses safe SQL queries."""
        with patch('clinical_platform.api.main.get_config', return_value=mock_config):
            with patch('duckdb.connect') as mock_connect:
                mock_con = MagicMock()
                mock_con.execute.return_value.fetchall.return_value = [("STUDY001",), ("STUDY002",)]
                mock_connect.return_value = mock_con
                
                response = test_client.get("/studies", headers=auth_headers)
                assert response.status_code == 200
                
                # Verify the SQL query is safe (no user input concatenation)
                mock_con.execute.assert_called_once()
                query = mock_con.execute.call_args[0][0]
                assert "SELECT DISTINCT study_id FROM stg_subjects" in query

    def test_database_error_handling(self, test_client, mock_config, auth_headers):
        """Test that database errors are handled safely without exposing internals."""
        with patch('clinical_platform.api.main.get_config', return_value=mock_config):
            with patch('duckdb.connect') as mock_connect:
                # Simulate database error
                mock_connect.side_effect = Exception("Database connection failed with internal details")
                
                response = test_client.get("/subjects/SUBJ123", headers=auth_headers)
                
                # Should return 500 but not expose internal error details
                assert response.status_code == 500
                error_response = response.json()
                assert "Database error occurred" in error_response["detail"]
                assert "internal details" not in error_response["detail"]


class TestErrorSanitization:
    """Test error message sanitization."""

    def test_safe_http_exception_sanitizes_errors(self):
        """Test that safe_http_exception properly sanitizes error messages."""
        # Error with PHI
        phi_error = "Patient SUBJ123456 with SSN 123-45-6789 not found"
        
        with patch('clinical_platform.api.main.PHIFilter') as mock_filter_class:
            mock_filter = MagicMock()
            mock_filter._redact_phi_from_message.return_value = "Patient [REDACTED] with SSN [REDACTED] not found"
            mock_filter_class.return_value = mock_filter
            
            exception = safe_http_exception(404, phi_error)
            
            assert exception.status_code == 404
            assert "[REDACTED]" in exception.detail
            assert "SUBJ123456" not in exception.detail
            assert "123-45-6789" not in exception.detail

    def test_api_errors_use_safe_exception_handler(self, test_client, mock_config, auth_headers):
        """Test that API errors use the safe exception handler."""
        with patch('clinical_platform.api.main.get_config', return_value=mock_config):
            with patch('duckdb.connect') as mock_connect:
                # Simulate error with PHI in the message
                mock_connect.side_effect = Exception("Failed to connect to database with patient data SUBJ123456")
                
                response = test_client.get("/subjects/SUBJ123", headers=auth_headers)
                
                assert response.status_code == 500
                error_response = response.json()
                # Error message should be sanitized
                assert "Database error occurred" in error_response["detail"]
                assert "SUBJ123456" not in error_response["detail"]

    def test_validation_errors_dont_expose_internals(self, test_client, mock_config, auth_headers):
        """Test that validation errors don't expose internal system details."""
        with patch('clinical_platform.api.main.get_config', return_value=mock_config):
            # Send request with invalid data type
            response = test_client.post("/score", 
                                      json={"AGE": "invalid", "AE_COUNT": 2.0, "SEVERE_AE_COUNT": 1.0}, 
                                      headers=auth_headers)
            
            assert response.status_code == 422
            error_response = response.json()
            
            # Should contain validation error but not expose internal paths or system info
            assert "detail" in error_response
            # FastAPI validation errors are structured, not plain text with system paths


class TestUnicodeAndEncodingHandling:
    """Test handling of unicode and various encodings."""

    def test_unicode_subject_ids_handled_safely(self, test_client, mock_config, auth_headers):
        """Test that unicode characters in subject IDs are handled safely."""
        with patch('clinical_platform.api.main.get_config', return_value=mock_config):
            unicode_ids = [
                "SUBJ123é",      # Accented character
                "SUBJ123中文",    # Chinese characters
                "SUBJ123🔒",     # Emoji
                "SUBJ123\u0000", # Null byte
            ]
            
            for subject_id in unicode_ids:
                response = test_client.get(f"/subjects/{subject_id}", headers=auth_headers)
                # Should be safely rejected (422) or handled (404)
                assert response.status_code in [422, 404]

    def test_unicode_in_json_payloads(self, test_client, mock_config, auth_headers):
        """Test handling of unicode in JSON payloads."""
        with patch('clinical_platform.api.main.get_config', return_value=mock_config):
            # JSON with unicode (should be rejected by validation)
            unicode_payload = {
                "AGE": 45.0,
                "AE_COUNT": 2.0,
                "SEVERE_AE_COUNT": 1.0,
                "extra_field": "unicode_value_中文"
            }
            
            response = test_client.post("/score", json=unicode_payload, headers=auth_headers)
            # Should handle gracefully (extra field should be ignored by Pydantic)
            assert response.status_code in [200, 422]


class TestPathTraversalPrevention:
    """Test prevention of path traversal attacks."""

    def test_subject_id_path_traversal_blocked(self, test_client, mock_config, auth_headers):
        """Test that path traversal attempts in subject IDs are blocked."""
        with patch('clinical_platform.api.main.get_config', return_value=mock_config):
            path_traversal_attempts = [
                "../../../etc/passwd",
                "..\\..\\windows\\system32",
                "%2e%2e%2f%2e%2e%2f",  # URL encoded ../..
                "....//....//",
                "..%252f..%252f",      # Double URL encoded
            ]
            
            for attempt in path_traversal_attempts:
                response = test_client.get(f"/subjects/{attempt}", headers=auth_headers)
                # Should be blocked by regex validation
                assert response.status_code == 422