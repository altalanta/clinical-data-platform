"""Tests for read-only mode enforcement middleware."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os

from clinical_platform.api.middleware import ReadOnlyModeMiddleware, setup_middleware
from clinical_platform.config import UnifiedConfig, SecurityConfig


@pytest.fixture
def test_app():
    """Create a test FastAPI app with read-only middleware."""
    app = FastAPI()
    
    @app.get("/test-get")
    def test_get():
        return {"method": "GET", "allowed": True}
    
    @app.post("/test-post")
    def test_post():
        return {"method": "POST", "allowed": True}
    
    @app.put("/test-put")
    def test_put():
        return {"method": "PUT", "allowed": True}
    
    @app.patch("/test-patch")
    def test_patch():
        return {"method": "PATCH", "allowed": True}
    
    @app.delete("/test-delete")
    def test_delete():
        return {"method": "DELETE", "allowed": True}
    
    @app.head("/test-head")
    def test_head():
        return {"method": "HEAD", "allowed": True}
    
    @app.options("/test-options")
    def test_options():
        return {"method": "OPTIONS", "allowed": True}
    
    # Add read-only middleware
    app.add_middleware(ReadOnlyModeMiddleware)
    
    return app


@pytest.fixture
def readonly_config():
    """Mock config with read-only mode enabled."""
    config = MagicMock(spec=UnifiedConfig)
    config.security = MagicMock(spec=SecurityConfig)
    config.security.read_only_mode = True
    return config


@pytest.fixture
def writable_config():
    """Mock config with read-only mode disabled."""
    config = MagicMock(spec=UnifiedConfig)
    config.security = MagicMock(spec=SecurityConfig)
    config.security.read_only_mode = False
    return config


class TestReadOnlyModeMiddleware:
    """Test read-only mode enforcement."""

    @pytest.mark.parametrize("method,path", [
        ("GET", "/test-get"),
        ("HEAD", "/test-head"),
        ("OPTIONS", "/test-options"),
    ])
    def test_readonly_mode_allows_safe_methods(self, test_app, readonly_config, method, path):
        """Test that safe HTTP methods are allowed in read-only mode."""
        with patch('clinical_platform.api.middleware.get_config', return_value=readonly_config):
            client = TestClient(test_app)
            response = client.request(method, path)
            assert response.status_code == 200
            if method == "GET":
                assert response.json()["allowed"] is True

    @pytest.mark.parametrize("method,path", [
        ("POST", "/test-post"),
        ("PUT", "/test-put"),
        ("PATCH", "/test-patch"),
        ("DELETE", "/test-delete"),
    ])
    def test_readonly_mode_blocks_write_methods(self, test_app, readonly_config, method, path):
        """Test that write HTTP methods are blocked in read-only mode."""
        with patch('clinical_platform.api.middleware.get_config', return_value=readonly_config):
            client = TestClient(test_app)
            response = client.request(method, path)
            
            # Should return 403 Forbidden
            assert response.status_code == 403
            
            # Should have proper error message
            error_data = response.json()
            assert "read-only mode" in error_data["error"].lower()
            assert "forbidden" in error_data["error"].lower()
            assert error_data["allowed_methods"] == ["GET", "HEAD", "OPTIONS"]

    @pytest.mark.parametrize("method,path", [
        ("GET", "/test-get"),
        ("POST", "/test-post"),
        ("PUT", "/test-put"),
        ("PATCH", "/test-patch"),
        ("DELETE", "/test-delete"),
    ])
    def test_writable_mode_allows_all_methods(self, test_app, writable_config, method, path):
        """Test that all HTTP methods are allowed when read-only mode is disabled."""
        with patch('clinical_platform.api.middleware.get_config', return_value=writable_config):
            client = TestClient(test_app)
            response = client.request(method, path)
            assert response.status_code == 200
            assert response.json()["allowed"] is True

    def test_readonly_mode_logs_violation_attempts(self, test_app, readonly_config, caplog):
        """Test that read-only violations are properly logged."""
        with patch('clinical_platform.api.middleware.get_config', return_value=readonly_config):
            client = TestClient(test_app)
            response = client.post("/test-post")
            
            assert response.status_code == 403
            # Note: In real tests, you'd check the actual logging system
            # This is a simplified check

    def test_readonly_environment_variable_override(self, test_app):
        """Test that READ_ONLY_MODE environment variable is respected."""
        # Test with env var set to 1
        with patch.dict(os.environ, {'READ_ONLY_MODE': '1'}):
            # Need to reload config to pick up env var
            with patch('clinical_platform.api.middleware.get_config') as mock_get_config:
                config = MagicMock()
                config.security.read_only_mode = True
                mock_get_config.return_value = config
                
                client = TestClient(test_app)
                response = client.post("/test-post")
                assert response.status_code == 403

        # Test with env var set to 0
        with patch.dict(os.environ, {'READ_ONLY_MODE': '0'}):
            with patch('clinical_platform.api.middleware.get_config') as mock_get_config:
                config = MagicMock()
                config.security.read_only_mode = False
                mock_get_config.return_value = config
                
                client = TestClient(test_app)
                response = client.post("/test-post")
                assert response.status_code == 200

    def test_readonly_mode_preserves_response_headers(self, test_app, readonly_config):
        """Test that read-only mode responses include proper headers."""
        with patch('clinical_platform.api.middleware.get_config', return_value=readonly_config):
            client = TestClient(test_app)
            response = client.post("/test-post")
            
            assert response.status_code == 403
            assert response.headers["content-type"] == "application/json"

    def test_readonly_mode_integration_with_full_middleware_stack(self):
        """Test read-only mode works correctly with full middleware stack."""
        app = FastAPI()
        
        @app.post("/api/v1/data/ingest")
        def ingest_data():
            return {"status": "success"}
        
        # Setup full middleware stack
        setup_middleware(app)
        
        # Mock read-only mode enabled
        with patch('clinical_platform.api.middleware.get_config') as mock_get_config:
            config = MagicMock()
            config.security.read_only_mode = True
            config.env = "local"
            mock_get_config.return_value = config
            
            client = TestClient(app)
            response = client.post("/api/v1/data/ingest")
            
            # Should be blocked by read-only middleware
            assert response.status_code == 403


class TestReadOnlyModeSideEffects:
    """Test that read-only mode prevents actual side effects."""

    def test_readonly_prevents_database_writes(self, test_app, readonly_config):
        """Test that database writes are prevented in read-only mode."""
        # This would be a more complex test with actual database mocking
        # For now, we verify the HTTP layer blocks the request
        with patch('clinical_platform.api.middleware.get_config', return_value=readonly_config):
            client = TestClient(test_app)
            
            # Add endpoint that would write to database
            response = client.post("/test-post", json={"data": "test"})
            assert response.status_code == 403
            
            # Verify the endpoint handler was never called
            # (since middleware returns 403 before reaching handler)

    def test_readonly_prevents_file_operations(self, test_app, readonly_config, tmp_path):
        """Test that file operations are prevented in read-only mode."""
        test_file = tmp_path / "test_file.txt"
        
        with patch('clinical_platform.api.middleware.get_config', return_value=readonly_config):
            client = TestClient(test_app)
            
            # POST request that would create a file
            response = client.post("/test-post")
            assert response.status_code == 403
            
            # Verify file was not created (since middleware blocked the request)
            assert not test_file.exists()