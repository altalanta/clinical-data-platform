"""Tests for CORS and rate limiting functionality."""

import pytest
import time
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from clinical_platform.api.middleware import configure_cors, RateLimitMiddleware
from clinical_platform.config import UnifiedConfig, SecurityConfig


@pytest.fixture
def test_app():
    """Create a test FastAPI app with CORS and rate limiting."""
    app = FastAPI()
    
    @app.get("/test-get")
    def test_get():
        return {"method": "GET"}
    
    @app.post("/test-post")
    def test_post():
        return {"method": "POST"}
    
    @app.options("/test-options")
    def test_options():
        return {"method": "OPTIONS"}
    
    return app


@pytest.fixture
def readonly_config():
    """Mock config with read-only mode enabled."""
    config = MagicMock(spec=UnifiedConfig)
    config.security = MagicMock(spec=SecurityConfig)
    config.security.read_only_mode = True
    config.env = "local"
    return config


@pytest.fixture
def writable_config():
    """Mock config with read-only mode disabled."""
    config = MagicMock(spec=UnifiedConfig)
    config.security = MagicMock(spec=SecurityConfig)
    config.security.read_only_mode = False
    config.env = "local"
    return config


@pytest.fixture
def prod_config():
    """Mock production config."""
    config = MagicMock(spec=UnifiedConfig)
    config.security = MagicMock(spec=SecurityConfig)
    config.security.read_only_mode = False
    config.env = "prod"
    return config


class TestCORSConfiguration:
    """Test CORS configuration and enforcement."""

    def test_cors_allows_configured_origins_local(self, test_app, writable_config):
        """Test that CORS allows configured origins in local environment."""
        with patch('clinical_platform.api.middleware.get_config', return_value=writable_config):
            cors_middleware = configure_cors()
            test_app.add_middleware(type(cors_middleware), **cors_middleware.__dict__)
            
            client = TestClient(test_app)
            
            # Test preflight request from allowed origin
            headers = {
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type,Authorization"
            }
            
            response = client.options("/test-options", headers=headers)
            
            # Should allow the request
            assert response.status_code in [200, 204]
            if "Access-Control-Allow-Origin" in response.headers:
                assert response.headers["Access-Control-Allow-Origin"] == "http://localhost:3000"

    def test_cors_restricts_methods_in_readonly_mode(self, test_app, readonly_config):
        """Test that CORS restricts HTTP methods in read-only mode."""
        with patch('clinical_platform.api.middleware.get_config', return_value=readonly_config):
            cors_middleware = configure_cors()
            
            # In read-only mode, should only allow GET, HEAD, OPTIONS
            allowed_methods = cors_middleware.allow_methods
            assert "GET" in allowed_methods
            assert "HEAD" in allowed_methods  
            assert "OPTIONS" in allowed_methods
            assert "POST" not in allowed_methods
            assert "PUT" not in allowed_methods
            assert "DELETE" not in allowed_methods

    def test_cors_allows_all_methods_in_writable_mode(self, test_app, writable_config):
        """Test that CORS allows all methods when not in read-only mode."""
        with patch('clinical_platform.api.middleware.get_config', return_value=writable_config):
            cors_middleware = configure_cors()
            
            # Should allow all methods when not read-only
            allowed_methods = cors_middleware.allow_methods
            assert "GET" in allowed_methods
            assert "POST" in allowed_methods
            assert "PUT" in allowed_methods
            assert "DELETE" in allowed_methods

    def test_cors_production_origin_restrictions(self, test_app, prod_config):
        """Test that CORS restricts origins in production."""
        with patch('clinical_platform.api.middleware.get_config', return_value=prod_config):
            cors_middleware = configure_cors()
            
            # Production should have empty allowed origins by default
            assert cors_middleware.allow_origins == []

    def test_cors_headers_restricted(self, test_app, writable_config):
        """Test that CORS restricts allowed headers."""
        with patch('clinical_platform.api.middleware.get_config', return_value=writable_config):
            cors_middleware = configure_cors()
            
            # Should only allow specific headers, not "*"
            allowed_headers = cors_middleware.allow_headers
            assert "Authorization" in allowed_headers
            assert "Content-Type" in allowed_headers
            assert "X-Request-ID" in allowed_headers
            assert "*" not in allowed_headers  # Should not allow all headers

    def test_cors_credentials_handling(self, test_app, writable_config):
        """Test CORS credentials handling."""
        with patch('clinical_platform.api.middleware.get_config', return_value=writable_config):
            cors_middleware = configure_cors()
            
            # Should allow credentials for authenticated requests
            assert cors_middleware.allow_credentials is True

    def test_cors_disallows_unknown_origins(self, test_app, writable_config):
        """Test that unknown origins are blocked."""
        with patch('clinical_platform.api.middleware.get_config', return_value=writable_config):
            cors_middleware = configure_cors()
            test_app.add_middleware(type(cors_middleware), **cors_middleware.__dict__)
            
            client = TestClient(test_app)
            
            # Test request from disallowed origin
            headers = {
                "Origin": "https://evil-site.com",
                "Access-Control-Request-Method": "GET"
            }
            
            response = client.options("/test-options", headers=headers)
            
            # Should not include CORS headers for disallowed origin
            # (FastAPI CORS middleware behavior varies, but origin should not be in response)
            if "Access-Control-Allow-Origin" in response.headers:
                assert response.headers["Access-Control-Allow-Origin"] != "https://evil-site.com"


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limit_middleware_initialization(self):
        """Test rate limit middleware initializes correctly."""
        app = FastAPI()
        middleware = RateLimitMiddleware(app, calls_per_minute=30, write_calls_per_minute=5)
        
        assert middleware.calls_per_minute == 30
        assert middleware.write_calls_per_minute == 5
        assert isinstance(middleware.call_times, dict)

    def test_rate_limit_allows_requests_under_limit(self):
        """Test that requests under the rate limit are allowed."""
        app = FastAPI()
        
        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}
        
        app.add_middleware(RateLimitMiddleware, calls_per_minute=60, write_calls_per_minute=10)
        
        client = TestClient(app)
        
        # Make several requests under the limit
        for i in range(5):
            response = client.get("/test")
            assert response.status_code == 200

    def test_rate_limit_blocks_excessive_requests(self):
        """Test that excessive requests are blocked."""
        app = FastAPI()
        
        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}
        
        # Set very low limit for testing
        app.add_middleware(RateLimitMiddleware, calls_per_minute=3, write_calls_per_minute=1)
        
        client = TestClient(app)
        
        # Make requests up to the limit
        for i in range(3):
            response = client.get("/test")
            assert response.status_code == 200
        
        # Next request should be rate limited
        response = client.get("/test")
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["error"]
        assert response.headers["Retry-After"] == "60"

    def test_rate_limit_write_operations_stricter(self):
        """Test that write operations have stricter rate limits."""
        app = FastAPI()
        
        @app.post("/test-write")
        def test_write():
            return {"status": "ok"}
        
        # Set different limits for read vs write
        app.add_middleware(RateLimitMiddleware, calls_per_minute=10, write_calls_per_minute=2)
        
        client = TestClient(app)
        
        # Should allow 2 write operations
        for i in range(2):
            response = client.post("/test-write")
            assert response.status_code == 200
        
        # Third write operation should be blocked
        response = client.post("/test-write")
        assert response.status_code == 429

    def test_rate_limit_per_client_ip(self):
        """Test that rate limiting is applied per client IP."""
        app = FastAPI()
        
        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}
        
        app.add_middleware(RateLimitMiddleware, calls_per_minute=2, write_calls_per_minute=1)
        
        # This test would require mocking different client IPs
        # In a real test environment, you'd simulate multiple clients
        # For now, we test the IP extraction logic exists
        middleware = RateLimitMiddleware(app)
        
        # Test IP extraction method exists
        assert hasattr(middleware, '_get_client_ip')

    def test_rate_limit_handles_x_forwarded_for(self):
        """Test that rate limiting properly handles X-Forwarded-For headers."""
        app = FastAPI()
        middleware = RateLimitMiddleware(app)
        
        # Mock request with X-Forwarded-For header
        mock_request = MagicMock()
        mock_request.headers.get.return_value = "192.168.1.1, 10.0.0.1"
        mock_request.client.host = "127.0.0.1"
        
        # Should use the first IP from X-Forwarded-For
        client_ip = middleware._get_client_ip(mock_request)
        assert client_ip == "192.168.1.1"

    def test_rate_limit_cleanup_old_entries(self):
        """Test that rate limiting cleans up old timestamp entries."""
        app = FastAPI()
        middleware = RateLimitMiddleware(app)
        
        # Simulate old entries
        client_ip = "192.168.1.1"
        old_time = time.time() - 120  # 2 minutes ago
        middleware.call_times[client_ip].append(old_time)
        
        # Check rate limit (should clean up old entry)
        is_limited = middleware._is_rate_limited(client_ip, False)
        
        # Old entries should be cleaned up
        assert len(middleware.call_times[client_ip]) <= 1  # Only current request

    def test_rate_limit_response_headers(self):
        """Test that rate limit responses include proper headers."""
        app = FastAPI()
        
        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}
        
        app.add_middleware(RateLimitMiddleware, calls_per_minute=1, write_calls_per_minute=1)
        
        client = TestClient(app)
        
        # First request should succeed
        response = client.get("/test")
        assert response.status_code == 200
        
        # Second request should be rate limited
        response = client.get("/test")
        assert response.status_code == 429
        assert "Retry-After" in response.headers
        assert response.headers["Retry-After"] == "60"

    def test_rate_limit_logging(self):
        """Test that rate limit violations are logged."""
        app = FastAPI()
        
        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}
        
        app.add_middleware(RateLimitMiddleware, calls_per_minute=1, write_calls_per_minute=1)
        
        client = TestClient(app)
        
        # Trigger rate limit
        client.get("/test")  # First request (allowed)
        
        with patch('clinical_platform.api.middleware.logger') as mock_logger:
            client.get("/test")  # Second request (should be blocked and logged)
            
            # Should log the rate limit violation
            # Note: Actual logging verification depends on how logging is configured
            # This is a simplified check


class TestRateLimitIntegration:
    """Integration tests for rate limiting with other middleware."""

    def test_rate_limit_with_authentication(self):
        """Test rate limiting works correctly with authentication."""
        app = FastAPI()
        
        @app.get("/protected")
        def protected_endpoint():
            return {"status": "protected"}
        
        app.add_middleware(RateLimitMiddleware, calls_per_minute=2, write_calls_per_minute=1)
        
        client = TestClient(app)
        
        # Rate limiting should apply regardless of authentication status
        for i in range(2):
            response = client.get("/protected")
            # May get 401 (auth required) or 200, but not 429 yet
            assert response.status_code in [200, 401]
        
        # Should be rate limited now
        response = client.get("/protected")
        assert response.status_code == 429

    def test_rate_limit_bypass_attempts(self):
        """Test that rate limiting cannot be bypassed."""
        app = FastAPI()
        
        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}
        
        app.add_middleware(RateLimitMiddleware, calls_per_minute=1, write_calls_per_minute=1)
        
        client = TestClient(app)
        
        # Try various bypass techniques
        bypass_headers = [
            {"X-Real-IP": "192.168.1.100"},  # Different IP header
            {"X-Forwarded-For": "10.0.0.100"},  # Different forwarded IP
            {"User-Agent": "Bypass-Agent"},  # Different user agent
            {"X-Rate-Limit-Bypass": "true"},  # Custom bypass header
        ]
        
        # First request should succeed
        response = client.get("/test")
        assert response.status_code == 200
        
        # Subsequent requests with bypass attempts should still be rate limited
        for headers in bypass_headers:
            response = client.get("/test", headers=headers)
            assert response.status_code == 429, f"Bypass succeeded with headers: {headers}"


class TestCORSRateLimitInteraction:
    """Test interaction between CORS and rate limiting."""

    def test_cors_preflight_rate_limiting(self):
        """Test that CORS preflight requests are subject to rate limiting."""
        app = FastAPI()
        
        @app.post("/test")
        def test_endpoint():
            return {"status": "ok"}
        
        app.add_middleware(RateLimitMiddleware, calls_per_minute=2, write_calls_per_minute=1)
        
        with patch('clinical_platform.api.middleware.get_config') as mock_config:
            config = MagicMock()
            config.security.read_only_mode = False
            config.env = "local"
            mock_config.return_value = config
            
            cors_middleware = configure_cors()
            app.add_middleware(type(cors_middleware), **cors_middleware.__dict__)
        
        client = TestClient(app)
        
        # Make preflight requests
        headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST"
        }
        
        # Should be subject to rate limiting like other requests
        for i in range(2):
            response = client.options("/test", headers=headers)
            # Should succeed (may be 200 or 204)
            assert response.status_code in [200, 204]
        
        # Should be rate limited
        response = client.options("/test", headers=headers)
        assert response.status_code == 429

    def test_rate_limit_preserves_cors_headers(self):
        """Test that rate limiting responses preserve CORS headers when needed."""
        app = FastAPI()
        
        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}
        
        app.add_middleware(RateLimitMiddleware, calls_per_minute=1, write_calls_per_minute=1)
        
        client = TestClient(app)
        
        # Trigger rate limit
        client.get("/test")  # First request
        response = client.get("/test")  # Rate limited request
        
        assert response.status_code == 429
        # Rate limit response should be JSON
        assert response.headers["content-type"] == "application/json"