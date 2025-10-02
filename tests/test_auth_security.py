"""
Security tests for authentication system.
Tests for hardcoded credential removal and timing attack prevention.
"""

import os
import time
import pytest
from unittest.mock import patch, MagicMock
from api.auth import AuthService, UserLogin
from fastapi import HTTPException


class TestJWTSecretValidation:
    """Test JWT secret validation in different environments."""
    
    def test_prod_requires_jwt_secret(self):
        """Test that production environment requires JWT_SECRET."""
        with patch.dict(os.environ, {"ENVIRONMENT": "prod"}, clear=True):
            with patch.dict(os.environ, {}, clear=True):  # Clear JWT_SECRET
                with pytest.raises(ValueError, match="JWT_SECRET environment variable is required"):
                    AuthService()
    
    def test_prod_requires_long_jwt_secret(self):
        """Test that production requires minimum JWT secret length."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "prod", 
            "JWT_SECRET": "short"
        }, clear=True):
            with pytest.raises(ValueError, match="JWT_SECRET must be at least 32 characters"):
                AuthService()
    
    def test_dev_generates_jwt_secret(self):
        """Test that development generates JWT secret if missing."""
        with patch.dict(os.environ, {"ENVIRONMENT": "dev"}, clear=True):
            with patch.dict(os.environ, {}, clear=True):  # Clear JWT_SECRET
                auth_service = AuthService()
                assert len(os.environ["JWT_SECRET"]) >= 32
    
    def test_staging_requires_jwt_secret(self):
        """Test that staging environment requires JWT_SECRET."""
        with patch.dict(os.environ, {"ENVIRONMENT": "staging"}, clear=True):
            with patch.dict(os.environ, {}, clear=True):  # Clear JWT_SECRET
                with pytest.raises(ValueError, match="JWT_SECRET environment variable is required"):
                    AuthService()


class TestTimingAttackPrevention:
    """Test timing attack prevention in authentication."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_env = {
            "ENVIRONMENT": "dev",
            "JWT_SECRET": "test-secret-key-32-chars-minimum",
            "AUTH_DELAY_ENABLED": "true",
            "AUTH_DELAY_MIN_MS": "10",
            "AUTH_DELAY_MAX_MS": "20",
            "ADMIN_PASSWORD_HASH": "$2b$12$test.hash.for.testing.purposes.only"
        }
        
    def test_timing_consistency_valid_vs_invalid_user(self):
        """Test that authentication timing is consistent for valid vs invalid users."""
        with patch.dict(os.environ, self.test_env, clear=True):
            auth_service = AuthService()
            
            # Measure timing for invalid user
            start_time = time.time()
            result1 = auth_service.authenticate_user("nonexistent", "password")
            invalid_user_time = time.time() - start_time
            
            # Measure timing for valid user with wrong password
            start_time = time.time()
            result2 = auth_service.authenticate_user("admin", "wrongpassword")
            valid_user_time = time.time() - start_time
            
            assert result1 is None
            assert result2 is None
            
            # Timing should be similar (within reasonable variance)
            time_diff = abs(invalid_user_time - valid_user_time)
            assert time_diff < 0.05  # 50ms tolerance
    
    def test_dummy_hash_verification_occurs(self):
        """Test that dummy hash verification occurs for non-existent users."""
        with patch.dict(os.environ, self.test_env, clear=True):
            auth_service = AuthService()
            
            # Mock the verify_password method to track calls
            with patch.object(auth_service, 'verify_password') as mock_verify:
                mock_verify.return_value = False
                
                # Authenticate non-existent user
                result = auth_service.authenticate_user("nonexistent", "password")
                
                assert result is None
                # verify_password should be called even for non-existent users
                mock_verify.assert_called_once()
                # Should be called with dummy hash
                call_args = mock_verify.call_args[0]
                assert call_args[0] == "password"
                assert "dummy.hash.to.prevent.timing.attacks" in call_args[1]
    
    def test_consistent_error_messages(self):
        """Test that login returns consistent error messages."""
        with patch.dict(os.environ, self.test_env, clear=True):
            auth_service = AuthService()
            
            # Test invalid user
            with pytest.raises(HTTPException) as exc_info1:
                auth_service.login(UserLogin(username="nonexistent", password="password"))
            
            # Test valid user with wrong password
            with pytest.raises(HTTPException) as exc_info2:
                auth_service.login(UserLogin(username="admin", password="wrongpassword"))
            
            # Both should return identical error messages
            assert exc_info1.value.detail == exc_info2.value.detail
            assert exc_info1.value.detail == "Invalid credentials"
            assert exc_info1.value.status_code == 401
            assert exc_info2.value.status_code == 401


class TestUserDatabaseSecurity:
    """Test user database loading security."""
    
    def test_no_hardcoded_users_by_default(self):
        """Test that no users are hardcoded by default."""
        with patch.dict(os.environ, {
            "ENVIRONMENT": "dev",
            "JWT_SECRET": "test-secret-key-32-chars-minimum"
        }, clear=True):
            auth_service = AuthService()
            assert len(auth_service.users_db) == 0
    
    def test_admin_user_loaded_from_env(self):
        """Test that admin user is loaded from environment variables."""
        test_env = {
            "ENVIRONMENT": "dev",
            "JWT_SECRET": "test-secret-key-32-chars-minimum",
            "ADMIN_PASSWORD_HASH": "$2b$12$test.hash.for.admin.user",
            "ADMIN_EMAIL": "admin@test.com"
        }
        
        with patch.dict(os.environ, test_env, clear=True):
            auth_service = AuthService()
            
            assert "admin" in auth_service.users_db
            admin_user = auth_service.users_db["admin"]
            assert admin_user["email"] == "admin@test.com"
            assert admin_user["hashed_password"] == "$2b$12$test.hash.for.admin.user"
    
    def test_admin_user_default_email(self):
        """Test that admin user gets default email if not specified."""
        test_env = {
            "ENVIRONMENT": "dev",
            "JWT_SECRET": "test-secret-key-32-chars-minimum",
            "ADMIN_PASSWORD_HASH": "$2b$12$test.hash.for.admin.user"
        }
        
        with patch.dict(os.environ, test_env, clear=True):
            auth_service = AuthService()
            
            admin_user = auth_service.users_db["admin"]
            assert admin_user["email"] == "admin@clinical-platform.com"


class TestAuthDelayConfiguration:
    """Test authentication delay configuration."""
    
    def test_auth_delay_can_be_disabled(self):
        """Test that auth delay can be disabled."""
        test_env = {
            "ENVIRONMENT": "dev",
            "JWT_SECRET": "test-secret-key-32-chars-minimum",
            "AUTH_DELAY_ENABLED": "false"
        }
        
        with patch.dict(os.environ, test_env, clear=True):
            auth_service = AuthService()
            assert auth_service.auth_delay_enabled is False
    
    def test_auth_delay_timing_configuration(self):
        """Test auth delay timing configuration."""
        test_env = {
            "ENVIRONMENT": "dev",
            "JWT_SECRET": "test-secret-key-32-chars-minimum",
            "AUTH_DELAY_MIN_MS": "100",
            "AUTH_DELAY_MAX_MS": "200"
        }
        
        with patch.dict(os.environ, test_env, clear=True):
            auth_service = AuthService()
            assert auth_service.auth_delay_min_ms == 100
            assert auth_service.auth_delay_max_ms == 200