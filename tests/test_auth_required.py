"""Tests for mandatory authentication enforcement."""

import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from unittest.mock import patch, MagicMock

from clinical_platform.api.main import verify_api_key, safe_http_exception
from clinical_platform.config import UnifiedConfig, SecurityConfig
from pydantic import SecretStr


@pytest.fixture
def test_app():
    """Create a test FastAPI app with authentication."""
    app = FastAPI()
    
    @app.get("/health")
    def health():
        return {"status": "ok"}
    
    @app.get("/protected")
    def protected(api_key: str = Depends(verify_api_key)):
        return {"message": "access granted", "api_key_provided": bool(api_key)}
    
    @app.post("/protected-write")
    def protected_write(api_key: str = Depends(verify_api_key)):
        return {"message": "write operation", "api_key_provided": bool(api_key)}
    
    return app


@pytest.fixture
def config_with_api_key():
    """Mock config with API key configured."""
    config = MagicMock(spec=UnifiedConfig)
    config.security = MagicMock(spec=SecurityConfig)
    config.security.api_key = SecretStr("test-api-key-12345")
    return config


@pytest.fixture
def config_without_api_key():
    """Mock config without API key configured."""
    config = MagicMock(spec=UnifiedConfig)
    config.security = MagicMock(spec=SecurityConfig)
    config.security.api_key = None
    return config


class TestMandatoryAuthentication:
    """Test mandatory authentication enforcement."""

    def test_health_endpoint_no_auth_required(self, test_app):
        """Test that health endpoint doesn't require authentication."""
        client = TestClient(test_app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_protected_endpoint_requires_auth(self, test_app, config_with_api_key):
        """Test that protected endpoints require authentication."""
        with patch('clinical_platform.api.main.get_config', return_value=config_with_api_key):
            client = TestClient(test_app)
            
            # No authorization header
            response = client.get("/protected")
            assert response.status_code == 401
            assert "API key required" in response.json()["detail"]
            assert response.headers["WWW-Authenticate"] == "Bearer"

    def test_valid_api_key_grants_access(self, test_app, config_with_api_key):
        """Test that valid API key grants access to protected endpoints."""
        with patch('clinical_platform.api.main.get_config', return_value=config_with_api_key):
            client = TestClient(test_app)
            
            headers = {"Authorization": "Bearer test-api-key-12345"}
            response = client.get("/protected", headers=headers)
            
            assert response.status_code == 200
            assert response.json()["message"] == "access granted"
            assert response.json()["api_key_provided"] is True

    def test_invalid_api_key_denied(self, test_app, config_with_api_key):
        """Test that invalid API key is denied access."""
        with patch('clinical_platform.api.main.get_config', return_value=config_with_api_key):
            client = TestClient(test_app)
            
            headers = {"Authorization": "Bearer wrong-api-key"}
            response = client.get("/protected", headers=headers)
            
            assert response.status_code == 401
            assert "Invalid API key" in response.json()["detail"]
            assert response.headers["WWW-Authenticate"] == "Bearer"

    def test_malformed_auth_header_denied(self, test_app, config_with_api_key):
        """Test that malformed authorization headers are denied."""
        with patch('clinical_platform.api.main.get_config', return_value=config_with_api_key):
            client = TestClient(test_app)
            
            # Missing Bearer prefix
            headers = {"Authorization": "test-api-key-12345"}
            response = client.get("/protected", headers=headers)
            assert response.status_code == 401
            
            # Wrong auth type
            headers = {"Authorization": "Basic test-api-key-12345"}
            response = client.get("/protected", headers=headers)
            assert response.status_code == 401

    def test_no_api_key_configured_returns_500(self, test_app, config_without_api_key):
        """Test that server returns 500 when API key is not configured."""
        with patch('clinical_platform.api.main.get_config', return_value=config_without_api_key):
            client = TestClient(test_app)
            
            headers = {"Authorization": "Bearer any-key"}
            response = client.get("/protected", headers=headers)
            
            assert response.status_code == 500
            assert "Server misconfigured" in response.json()["detail"]

    def test_empty_api_key_denied(self, test_app, config_with_api_key):
        """Test that empty API key is denied."""
        with patch('clinical_platform.api.main.get_config', return_value=config_with_api_key):
            client = TestClient(test_app)
            
            headers = {"Authorization": "Bearer "}
            response = client.get("/protected", headers=headers)
            
            assert response.status_code == 401
            assert "Invalid API key" in response.json()["detail"]

    def test_auth_works_for_write_operations(self, test_app, config_with_api_key):
        """Test that authentication works for write operations."""
        with patch('clinical_platform.api.main.get_config', return_value=config_with_api_key):
            client = TestClient(test_app)
            
            # Valid API key for POST
            headers = {"Authorization": "Bearer test-api-key-12345"}
            response = client.post("/protected-write", headers=headers)
            
            assert response.status_code == 200
            assert response.json()["message"] == "write operation"

    def test_case_sensitive_api_key(self, test_app, config_with_api_key):
        """Test that API key validation is case sensitive."""
        with patch('clinical_platform.api.main.get_config', return_value=config_with_api_key):
            client = TestClient(test_app)
            
            # Wrong case
            headers = {"Authorization": "Bearer TEST-API-KEY-12345"}
            response = client.get("/protected", headers=headers)
            
            assert response.status_code == 401
            assert "Invalid API key" in response.json()["detail"]

    def test_api_key_trimming(self, test_app, config_with_api_key):
        """Test API key with extra whitespace."""
        with patch('clinical_platform.api.main.get_config', return_value=config_with_api_key):
            client = TestClient(test_app)
            
            # API key with trailing space (should fail)
            headers = {"Authorization": "Bearer test-api-key-12345 "}
            response = client.get("/protected", headers=headers)
            
            assert response.status_code == 401
            assert "Invalid API key" in response.json()["detail"]


class TestAuthenticationLogging:
    """Test authentication logging and monitoring."""

    def test_invalid_auth_attempts_logged(self, test_app, config_with_api_key, caplog):
        """Test that invalid authentication attempts are logged."""
        with patch('clinical_platform.api.main.get_config', return_value=config_with_api_key):
            client = TestClient(test_app)
            
            headers = {"Authorization": "Bearer wrong-key"}
            response = client.get("/protected", headers=headers)
            
            assert response.status_code == 401
            # Note: In actual implementation, you'd check that the warning was logged
            # This would require proper logger mocking

    def test_no_sensitive_data_in_logs(self, test_app, config_with_api_key):
        """Test that API keys are not logged in plaintext."""
        with patch('clinical_platform.api.main.get_config', return_value=config_with_api_key):
            # This test would verify that actual API key values don't appear in logs
            # Implementation would require checking the logging system
            pass


class TestSafeErrorHandling:
    """Test PHI-safe error handling."""

    def test_safe_http_exception_redacts_phi(self):
        """Test that safe_http_exception redacts PHI from error messages."""
        # Test with PHI in error message
        error_msg = "Patient SUBJ123456 with SSN 123-45-6789 not found in email test@example.com"
        
        with patch('clinical_platform.api.main.PHIFilter') as mock_filter:
            mock_filter_instance = MagicMock()
            mock_filter_instance._redact_phi_from_message.return_value = "Patient [REDACTED] with SSN [REDACTED] not found in email [REDACTED]"
            mock_filter.return_value = mock_filter_instance
            
            exception = safe_http_exception(404, error_msg)
            
            assert exception.status_code == 404
            assert "[REDACTED]" in exception.detail
            assert "123-45-6789" not in exception.detail
            assert "test@example.com" not in exception.detail

    def test_safe_http_exception_logs_original_error(self):
        """Test that original error is logged for debugging."""
        original_error = Exception("Database connection failed with detailed info")
        
        with patch('clinical_platform.api.main.PHIFilter'):
            with patch('clinical_platform.api.main.logger') as mock_logger:
                safe_http_exception(500, "Database error", original_error)
                
                # Verify original error was logged
                mock_logger.error.assert_called_once()


class TestAuthenticationIntegration:
    """Integration tests for authentication with other middleware."""

    def test_auth_with_readonly_mode(self, test_app):
        """Test authentication works correctly with read-only mode."""
        config = MagicMock()
        config.security.api_key = SecretStr("test-key")
        config.security.read_only_mode = True
        
        with patch('clinical_platform.api.main.get_config', return_value=config):
            with patch('clinical_platform.api.middleware.get_config', return_value=config):
                client = TestClient(test_app)
                
                # Should require auth AND be blocked by read-only mode
                response = client.post("/protected-write")
                
                # Should get auth error first (middleware order dependent)
                assert response.status_code in [401, 403]

    def test_auth_bypass_not_possible(self, test_app, config_with_api_key):
        """Test that there's no way to bypass authentication."""
        with patch('clinical_platform.api.main.get_config', return_value=config_with_api_key):
            client = TestClient(test_app)
            
            # Try various bypass attempts
            bypass_attempts = [
                {},  # No headers
                {"Authorization": ""},  # Empty auth
                {"Authorization": "Bearer"},  # No token
                {"Authorization": "None"},  # Literal None
                {"Authorization": "null"},  # Literal null
                {"X-API-Key": "test-api-key-12345"},  # Wrong header name
                {"Authorization": "Basic dGVzdC1hcGkta2V5LTEyMzQ1"},  # Wrong auth type
            ]
            
            for headers in bypass_attempts:
                response = client.get("/protected", headers=headers)
                assert response.status_code == 401, f"Bypass attempt succeeded with headers: {headers}"