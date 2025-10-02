"""
Tests for secure CORS configuration.
Validates production-safe CORS settings and prevents wildcard origins in production.
"""

import os
import pytest
from unittest.mock import patch
from api.config import Settings


class TestCORSConfiguration:
    """Test CORS configuration security."""
    
    def test_development_cors_defaults(self):
        """Test that development environment has sensible CORS defaults."""
        with patch.dict(os.environ, {"ENVIRONMENT": "dev"}, clear=True):
            settings = Settings()
            origins = settings.get_cors_origins()
            
            assert "http://localhost:3000" in origins
            assert "http://localhost:8000" in origins
            assert "http://localhost:8080" in origins
            assert "*" not in origins  # Even in dev, we don't default to wildcard
    
    def test_production_requires_cors_origins(self):
        """Test that production requires explicit CORS origins."""
        with patch.dict(os.environ, {"ENVIRONMENT": "prod"}, clear=True):
            settings = Settings()
            
            with pytest.raises(ValueError, match="CORS_ALLOW_ORIGINS must be specified"):
                settings.get_cors_origins()
    
    def test_production_rejects_wildcard_cors(self):
        """Test that production rejects wildcard CORS origins."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "prod",
            "CORS_ALLOW_ORIGINS": "*"
        }, clear=True):
            settings = Settings()
            
            with pytest.raises(ValueError, match="Wildcard '\\*' CORS origin is not allowed"):
                settings.get_cors_origins()
    
    def test_production_rejects_mixed_wildcard_cors(self):
        """Test that production rejects wildcard mixed with other origins."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "prod",
            "CORS_ALLOW_ORIGINS": "https://app.example.com,*"
        }, clear=True):
            settings = Settings()
            
            with pytest.raises(ValueError, match="Wildcard '\\*' CORS origin is not allowed"):
                settings.get_cors_origins()
    
    def test_production_requires_https_origins(self):
        """Test that production requires HTTPS origins (except localhost)."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "prod",
            "CORS_ALLOW_ORIGINS": "http://app.example.com"
        }, clear=True):
            settings = Settings()
            
            with pytest.raises(ValueError, match="Invalid CORS origin.*Must use HTTPS"):
                settings.get_cors_origins()
    
    def test_production_allows_https_origins(self):
        """Test that production allows valid HTTPS origins."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "prod",
            "CORS_ALLOW_ORIGINS": "https://app.clinical-platform.com,https://admin.clinical-platform.com"
        }, clear=True):
            settings = Settings()
            origins = settings.get_cors_origins()
            
            assert origins == [
                "https://app.clinical-platform.com",
                "https://admin.clinical-platform.com"
            ]
    
    def test_production_allows_localhost_for_testing(self):
        """Test that production allows localhost for testing."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "prod",
            "CORS_ALLOW_ORIGINS": "https://app.example.com,http://localhost:3000,http://127.0.0.1:8080"
        }, clear=True):
            settings = Settings()
            origins = settings.get_cors_origins()
            
            assert "https://app.example.com" in origins
            assert "http://localhost:3000" in origins
            assert "http://127.0.0.1:8080" in origins
    
    def test_staging_has_same_restrictions_as_prod(self):
        """Test that staging environment has same CORS restrictions as production."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "staging",
            "CORS_ALLOW_ORIGINS": "*"
        }, clear=True):
            settings = Settings()
            
            with pytest.raises(ValueError, match="Wildcard '\\*' CORS origin is not allowed"):
                settings.get_cors_origins()
    
    def test_cors_methods_parsing(self):
        """Test CORS methods parsing."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "dev",
            "CORS_ALLOW_METHODS": "GET, POST, PUT, DELETE"
        }, clear=True):
            settings = Settings()
            methods = settings.get_cors_methods()
            
            assert methods == ["GET", "POST", "PUT", "DELETE"]
    
    def test_cors_headers_parsing(self):
        """Test CORS headers parsing."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "dev",
            "CORS_ALLOW_HEADERS": "Content-Type, Authorization, X-Requested-With"
        }, clear=True):
            settings = Settings()
            headers = settings.get_cors_headers()
            
            assert headers == ["Content-Type", "Authorization", "X-Requested-With"]
    
    def test_cors_headers_wildcard(self):
        """Test CORS headers wildcard handling."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "dev",
            "CORS_ALLOW_HEADERS": "*"
        }, clear=True):
            settings = Settings()
            headers = settings.get_cors_headers()
            
            assert headers == ["*"]
    
    def test_cors_credentials_configuration(self):
        """Test CORS credentials configuration."""
        # Test enabled (default)
        with patch.dict(os.environ, {"ENVIRONMENT": "dev"}, clear=True):
            settings = Settings()
            assert settings.cors_allow_credentials is True
        
        # Test disabled
        with patch.dict(os.environ, {
            "ENVIRONMENT": "dev",
            "CORS_ALLOW_CREDENTIALS": "false"
        }, clear=True):
            settings = Settings()
            assert settings.cors_allow_credentials is False


class TestApplicationStartupSecurity:
    """Test application startup security validations."""
    
    def test_app_fails_on_invalid_cors_config(self):
        """Test that FastAPI app fails to start with invalid CORS config."""
        # This would be an integration test that imports and initializes the FastAPI app
        # with invalid CORS configuration and verifies it raises RuntimeError
        pass  # Placeholder for integration test
    
    def test_production_environment_detection(self):
        """Test production environment detection."""
        test_cases = [
            ("prod", True),
            ("production", True),
            ("staging", True),
            ("dev", False),
            ("development", False),
            ("test", False),
        ]
        
        for env, expected in test_cases:
            with patch.dict(os.environ, {"ENVIRONMENT": env}, clear=True):
                settings = Settings()
                assert settings.is_production_environment == expected