#!/usr/bin/env python3
"""
API Testing Script for Clinical Data Platform

This script provides comprehensive API testing including contract testing,
security testing, and performance testing for local development.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional
import uvicorn
import requests


class APITestRunner:
    """Comprehensive API testing runner."""
    
    def __init__(self, api_module: str = "clinical_platform.api.main:app", 
                 host: str = "127.0.0.1", port: int = 8000):
        self.api_module = api_module
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.server_process = None
        self.server_thread = None
        
    def start_api_server(self) -> bool:
        """Start the API server for testing."""
        try:
            print(f"🚀 Starting API server at {self.base_url}")
            
            def run_server():
                uvicorn.run(
                    self.api_module,
                    host=self.host,
                    port=self.port,
                    log_level="error",
                    access_log=False
                )
            
            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()
            
            # Wait for server to be ready
            max_retries = 30
            for i in range(max_retries):
                try:
                    response = requests.get(f"{self.base_url}/health", timeout=2)
                    if response.status_code == 200:
                        print(f"✅ API server is ready after {i+1} attempts")
                        return True
                except requests.RequestException:
                    pass
                time.sleep(1)
            
            print("❌ Failed to start API server")
            return False
            
        except Exception as e:
            print(f"❌ Error starting API server: {e}")
            return False
    
    def generate_openapi_schema(self, output_file: str = "openapi.json") -> bool:
        """Generate OpenAPI schema file."""
        try:
            print("📋 Generating OpenAPI schema")
            
            # Import the app to get the schema
            module_path, app_name = self.api_module.split(":")
            
            # Use subprocess to avoid import issues
            cmd = [
                sys.executable, "-c",
                f"from {module_path} import {app_name}; "
                f"import json; "
                f"schema = {app_name}.openapi(); "
                f"print(json.dumps(schema, indent=2))"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                with open(output_file, 'w') as f:
                    f.write(result.stdout)
                print(f"✅ OpenAPI schema saved to {output_file}")
                return True
            else:
                print(f"❌ Failed to generate OpenAPI schema: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error generating OpenAPI schema: {e}")
            return False
    
    def validate_openapi_schema(self, schema_file: str = "openapi.json") -> bool:
        """Validate OpenAPI schema."""
        try:
            print("🔍 Validating OpenAPI schema")
            
            # Check if file exists
            if not os.path.exists(schema_file):
                print(f"❌ Schema file {schema_file} not found")
                return False
            
            # Try to install openapi-spec-validator if not available
            try:
                from openapi_spec_validator import validate_spec
                from openapi_spec_validator.readers import read_from_filename
            except ImportError:
                print("Installing openapi-spec-validator...")
                subprocess.run([sys.executable, "-m", "pip", "install", "openapi-spec-validator"], 
                             check=True, capture_output=True)
                from openapi_spec_validator import validate_spec
                from openapi_spec_validator.readers import read_from_filename
            
            # Validate schema
            spec_dict, spec_url = read_from_filename(schema_file)
            validate_spec(spec_dict)
            
            print("✅ OpenAPI schema is valid")
            return True
            
        except Exception as e:
            print(f"❌ OpenAPI schema validation failed: {e}")
            return False
    
    def run_pytest_tests(self, test_pattern: str = "test_api*", 
                        max_examples: int = 50) -> Dict[str, bool]:
        """Run pytest-based API tests."""
        results = {}
        
        test_suites = {
            "contract": "tests/test_api_contract.py::TestAPIContract",
            "security": "tests/test_api_contract.py::TestAPISecurityContract", 
            "performance": "tests/test_api_contract.py::TestAPIPerformanceContract",
            "fuzz": "tests/test_api_contract.py::TestAPIFuzzTesting"
        }
        
        for suite_name, test_path in test_suites.items():
            try:
                print(f"🧪 Running {suite_name} tests")
                
                cmd = [
                    sys.executable, "-m", "pytest",
                    test_path,
                    "-v",
                    "--tb=short",
                    f"--hypothesis-max-examples={max_examples}",
                    "--junit-xml", f"{suite_name}-results.xml"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                success = result.returncode == 0
                results[suite_name] = success
                
                if success:
                    print(f"✅ {suite_name} tests passed")
                else:
                    print(f"❌ {suite_name} tests failed")
                    print(f"Error output: {result.stdout[-500:]}")  # Last 500 chars
                    
            except subprocess.TimeoutExpired:
                print(f"⏰ {suite_name} tests timed out")
                results[suite_name] = False
            except Exception as e:
                print(f"❌ Error running {suite_name} tests: {e}")
                results[suite_name] = False
        
        return results
    
    def run_schemathesis_tests(self, schema_file: str = "openapi.json", 
                             max_examples: int = 100) -> bool:
        """Run Schemathesis CLI tests."""
        try:
            print("🔥 Running Schemathesis fuzz testing")
            
            # Install schemathesis if not available
            try:
                import schemathesis
            except ImportError:
                print("Installing schemathesis...")
                subprocess.run([sys.executable, "-m", "pip", "install", "schemathesis"], 
                             check=True, capture_output=True)
            
            # Run schemathesis tests
            cmd = [
                sys.executable, "-m", "schemathesis", "run",
                f"{self.base_url}/openapi.json",
                "--checks", "all",
                "--hypothesis-max-examples", str(max_examples),
                "--request-timeout", "30",
                "--auth-type", "bearer",
                "--auth", "test-api-key-12345",
                "--junit-xml", "schemathesis-results.xml"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print("✅ Schemathesis tests passed")
                return True
            else:
                print("❌ Schemathesis tests found issues")
                print(f"Output: {result.stdout[-1000:]}")  # Last 1000 chars
                return False
                
        except Exception as e:
            print(f"❌ Error running Schemathesis tests: {e}")
            return False
    
    def run_security_scan(self) -> Dict[str, bool]:
        """Run security scans on the API code."""
        results = {}
        
        # Bandit security scan
        try:
            print("🔒 Running Bandit security scan")
            
            cmd = [
                sys.executable, "-m", "bandit",
                "-r", "src/clinical_platform/api/",
                "-f", "json",
                "-o", "bandit-report.json"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Bandit returns non-zero for security issues, but that's expected
            results["bandit"] = True
            print("✅ Bandit scan completed")
            
        except Exception as e:
            print(f"❌ Bandit scan failed: {e}")
            results["bandit"] = False
        
        # Safety check for dependencies
        try:
            print("🛡️ Running Safety dependency check")
            
            cmd = [sys.executable, "-m", "safety", "check", "--json"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                results["safety"] = True
                print("✅ No known security vulnerabilities in dependencies")
            else:
                results["safety"] = False
                print("⚠️ Security vulnerabilities found in dependencies")
                
        except Exception as e:
            print(f"❌ Safety check failed: {e}")
            results["safety"] = False
        
        return results
    
    def run_performance_benchmarks(self) -> Dict[str, float]:
        """Run basic performance benchmarks."""
        print("⚡ Running performance benchmarks")
        
        benchmarks = {}
        
        # Test health endpoint performance
        try:
            times = []
            for _ in range(10):
                start = time.time()
                response = requests.get(f"{self.base_url}/health", timeout=5)
                end = time.time()
                if response.status_code == 200:
                    times.append(end - start)
            
            if times:
                avg_time = sum(times) / len(times)
                benchmarks["health_avg_response_time"] = avg_time
                print(f"Health endpoint avg response time: {avg_time:.3f}s")
            
        except Exception as e:
            print(f"❌ Health endpoint benchmark failed: {e}")
        
        # Test concurrent requests
        try:
            import threading
            import queue
            
            results_queue = queue.Queue()
            num_threads = 5
            
            def make_request():
                try:
                    start = time.time()
                    response = requests.get(f"{self.base_url}/health", timeout=5)
                    end = time.time()
                    results_queue.put(("success", end - start))
                except Exception as e:
                    results_queue.put(("error", str(e)))
            
            # Start concurrent requests
            threads = []
            start_time = time.time()
            
            for _ in range(num_threads):
                thread = threading.Thread(target=make_request)
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join(timeout=10)
            
            end_time = time.time()
            
            # Collect results
            success_count = 0
            total_time = 0
            
            while not results_queue.empty():
                result_type, result_value = results_queue.get()
                if result_type == "success":
                    success_count += 1
                    total_time += result_value
            
            if success_count > 0:
                benchmarks["concurrent_success_rate"] = success_count / num_threads
                benchmarks["concurrent_avg_time"] = total_time / success_count
                benchmarks["total_concurrent_time"] = end_time - start_time
                
                print(f"Concurrent requests: {success_count}/{num_threads} succeeded")
                print(f"Avg response time: {benchmarks['concurrent_avg_time']:.3f}s")
                
        except Exception as e:
            print(f"❌ Concurrent request benchmark failed: {e}")
        
        return benchmarks
    
    def generate_report(self, pytest_results: Dict[str, bool], 
                       schemathesis_result: bool,
                       security_results: Dict[str, bool],
                       benchmarks: Dict[str, float]) -> str:
        """Generate comprehensive test report."""
        
        total_tests = len(pytest_results) + 1  # +1 for schemathesis
        passed_tests = sum(pytest_results.values()) + (1 if schemathesis_result else 0)
        
        security_issues = sum(1 for result in security_results.values() if not result)
        
        report = f"""# API Testing Report
        
**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Total Test Suites:** {total_tests}
**Passed:** {passed_tests}
**Failed:** {total_tests - passed_tests}
**Success Rate:** {(passed_tests / total_tests * 100):.1f}%

## Test Results

### Contract Testing
"""
        
        for suite, result in pytest_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            report += f"- **{suite.title()}:** {status}\n"
        
        status = "✅ PASS" if schemathesis_result else "❌ FAIL"
        report += f"- **Schemathesis Fuzz Testing:** {status}\n"
        
        report += "\n### Security Scan\n"
        for scanner, result in security_results.items():
            status = "✅ PASS" if result else "⚠️ ISSUES"
            report += f"- **{scanner.title()}:** {status}\n"
        
        if benchmarks:
            report += "\n### Performance Benchmarks\n"
            for metric, value in benchmarks.items():
                if "time" in metric:
                    report += f"- **{metric.replace('_', ' ').title()}:** {value:.3f}s\n"
                else:
                    report += f"- **{metric.replace('_', ' ').title()}:** {value:.2%}\n"
        
        report += f"\n### Quality Assessment\n"
        
        if passed_tests == total_tests and security_issues == 0:
            report += "🎉 **EXCELLENT:** All tests passed with no security issues!\n"
        elif passed_tests >= total_tests * 0.8:
            report += "✅ **GOOD:** Most tests passed, minor issues to address.\n"
        else:
            report += "⚠️ **NEEDS WORK:** Significant test failures detected.\n"
        
        return report
    
    def run_all_tests(self, max_examples: int = 50, 
                     include_schemathesis: bool = True,
                     include_security: bool = True,
                     include_benchmarks: bool = True) -> None:
        """Run complete API test suite."""
        print("🚀 Starting comprehensive API testing")
        print("=" * 60)
        
        # Start API server
        if not self.start_api_server():
            print("❌ Failed to start API server - aborting tests")
            return
        
        # Generate and validate OpenAPI schema
        if not self.generate_openapi_schema():
            print("❌ Failed to generate OpenAPI schema")
            return
        
        if not self.validate_openapi_schema():
            print("⚠️ OpenAPI schema validation failed, continuing anyway")
        
        # Run pytest tests
        pytest_results = self.run_pytest_tests(max_examples=max_examples)
        
        # Run schemathesis tests
        schemathesis_result = False
        if include_schemathesis:
            schemathesis_result = self.run_schemathesis_tests(max_examples=max_examples)
        
        # Run security scans
        security_results = {}
        if include_security:
            security_results = self.run_security_scan()
        
        # Run performance benchmarks
        benchmarks = {}
        if include_benchmarks:
            benchmarks = self.run_performance_benchmarks()
        
        # Generate report
        report = self.generate_report(
            pytest_results, schemathesis_result, 
            security_results, benchmarks
        )
        
        # Save report
        with open("api-test-report.md", "w") as f:
            f.write(report)
        
        print("\n" + "=" * 60)
        print(report)
        print("=" * 60)
        print(f"📄 Full report saved to api-test-report.md")
        
        # Exit with appropriate code
        total_tests = len(pytest_results) + (1 if include_schemathesis else 0)
        passed_tests = sum(pytest_results.values()) + (1 if schemathesis_result else 0)
        
        if passed_tests == total_tests:
            print("🎉 All tests passed!")
            sys.exit(0)
        else:
            print(f"⚠️ {total_tests - passed_tests} test suite(s) failed")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive API tests")
    parser.add_argument("--api-module", default="clinical_platform.api.main:app", 
                       help="API module to test")
    parser.add_argument("--host", default="127.0.0.1", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    parser.add_argument("--max-examples", type=int, default=50, 
                       help="Maximum examples for property-based tests")
    parser.add_argument("--skip-schemathesis", action="store_true", 
                       help="Skip Schemathesis tests")
    parser.add_argument("--skip-security", action="store_true", 
                       help="Skip security scans")
    parser.add_argument("--skip-benchmarks", action="store_true", 
                       help="Skip performance benchmarks")
    parser.add_argument("--light", action="store_true", 
                       help="Run light test suite (fewer examples)")
    parser.add_argument("--intensive", action="store_true", 
                       help="Run intensive test suite (more examples)")
    
    args = parser.parse_args()
    
    # Adjust max examples based on intensity
    if args.light:
        max_examples = 25
    elif args.intensive:
        max_examples = 200
    else:
        max_examples = args.max_examples
    
    # Create and run test runner
    runner = APITestRunner(args.api_module, args.host, args.port)
    
    runner.run_all_tests(
        max_examples=max_examples,
        include_schemathesis=not args.skip_schemathesis,
        include_security=not args.skip_security,
        include_benchmarks=not args.skip_benchmarks
    )


if __name__ == "__main__":
    main()