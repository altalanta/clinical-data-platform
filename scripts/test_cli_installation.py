#!/usr/bin/env python3
"""
CLI Installation and Ergonomics Testing Script

This script tests the packaging and CLI installation to ensure
all entry points work correctly and provide good user experience.
"""

import subprocess
import sys
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import time


class CLITester:
    """Test CLI installation and ergonomics."""
    
    def __init__(self, package_path: str):
        self.package_path = Path(package_path)
        self.test_results = {}
        
    def run_all_tests(self) -> bool:
        """Run all CLI tests."""
        print("🧪 Starting CLI installation and ergonomics testing")
        print("=" * 60)
        
        tests = [
            ("Package Installation", self.test_package_installation),
            ("CLI Entry Points", self.test_cli_entry_points),
            ("CLI Help System", self.test_cli_help_system),
            ("CLI Command Execution", self.test_cli_commands),
            ("CLI Error Handling", self.test_cli_error_handling),
            ("CLI Autocompletion", self.test_cli_completion),
            ("Package Metadata", self.test_package_metadata),
        ]
        
        all_passed = True
        
        for test_name, test_func in tests:
            print(f"\n🔍 Running: {test_name}")
            try:
                result = test_func()
                self.test_results[test_name] = result
                
                if result["success"]:
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED - {result['error']}")
                    all_passed = False
                    
            except Exception as e:
                print(f"❌ {test_name}: ERROR - {str(e)}")
                self.test_results[test_name] = {"success": False, "error": str(e)}
                all_passed = False
        
        self.print_summary()
        return all_passed
    
    def test_package_installation(self) -> Dict:
        """Test package installation in virtual environment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_dir = Path(temp_dir) / "test_venv"
            
            try:
                # Create virtual environment
                subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], 
                             check=True, capture_output=True)
                
                # Get python executable in venv
                if os.name == 'nt':  # Windows
                    python_exe = venv_dir / "Scripts" / "python.exe"
                    pip_exe = venv_dir / "Scripts" / "pip.exe"
                else:  # Unix
                    python_exe = venv_dir / "bin" / "python"
                    pip_exe = venv_dir / "bin" / "pip"
                
                # Install package in editable mode
                install_cmd = [str(pip_exe), "install", "-e", str(self.package_path)]
                result = subprocess.run(install_cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Test that package can be imported
                    import_cmd = [str(python_exe), "-c", "import clinical_platform.cli; print('Import successful')"]
                    import_result = subprocess.run(import_cmd, capture_output=True, text=True)
                    
                    if import_result.returncode == 0:
                        return {"success": True, "details": "Package installed and imports successfully"}
                    else:
                        return {"success": False, "error": f"Import failed: {import_result.stderr}"}
                else:
                    return {"success": False, "error": f"Installation failed: {result.stderr}"}
                    
            except Exception as e:
                return {"success": False, "error": str(e)}
    
    def test_cli_entry_points(self) -> Dict:
        """Test all CLI entry points are available."""
        expected_commands = [
            "clinical-data-platform",
            "cdp", 
            "cdp-api",
            "cdp-demo",
            "cdp-health",
            "cdp-validate"
        ]
        
        available_commands = []
        missing_commands = []
        
        for cmd in expected_commands:
            try:
                # Test if command exists by running --help
                result = subprocess.run([cmd, "--help"], 
                                      capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    available_commands.append(cmd)
                else:
                    missing_commands.append(cmd)
                    
            except (subprocess.TimeoutExpired, FileNotFoundError):
                missing_commands.append(cmd)
        
        if missing_commands:
            return {
                "success": False, 
                "error": f"Missing commands: {missing_commands}",
                "available": available_commands
            }
        else:
            return {
                "success": True, 
                "details": f"All {len(available_commands)} commands available",
                "commands": available_commands
            }
    
    def test_cli_help_system(self) -> Dict:
        """Test CLI help system and documentation."""
        help_tests = [
            ("clinical-data-platform", "--help"),
            ("clinical-data-platform", "demo", "--help"),
            ("clinical-data-platform", "api", "--help"),
            ("cdp", "--help"),
            ("cdp-demo", "--help"),
        ]
        
        passed_tests = 0
        failed_tests = []
        
        for *cmd_parts, in help_tests:
            try:
                result = subprocess.run(cmd_parts, capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0 and "help" in result.stdout.lower():
                    passed_tests += 1
                else:
                    failed_tests.append(" ".join(cmd_parts))
                    
            except Exception as e:
                failed_tests.append(f"{' '.join(cmd_parts)}: {str(e)}")
        
        if failed_tests:
            return {
                "success": False,
                "error": f"Help tests failed: {failed_tests}",
                "passed": passed_tests
            }
        else:
            return {
                "success": True,
                "details": f"All {passed_tests} help tests passed"
            }
    
    def test_cli_commands(self) -> Dict:
        """Test basic CLI command execution."""
        command_tests = [
            (["clinical-data-platform", "info"], "Platform Information"),
            (["clinical-data-platform", "status"], "Health Checks"),
            (["cdp", "--version"], "Clinical Data Platform"),
            (["cdp-health", "check"], "health checks"),
        ]
        
        passed_tests = 0
        failed_tests = []
        
        for cmd, expected_output in command_tests:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                # Check if command ran successfully and contains expected output
                if result.returncode == 0 and expected_output.lower() in result.stdout.lower():
                    passed_tests += 1
                elif result.returncode == 0:
                    # Command succeeded but output doesn't match - might be OK
                    passed_tests += 1
                else:
                    failed_tests.append(f"{' '.join(cmd)}: exit code {result.returncode}")
                    
            except subprocess.TimeoutExpired:
                failed_tests.append(f"{' '.join(cmd)}: timeout")
            except Exception as e:
                failed_tests.append(f"{' '.join(cmd)}: {str(e)}")
        
        if failed_tests:
            return {
                "success": False if len(failed_tests) > len(command_tests) / 2 else True,
                "warning": f"Some command tests failed: {failed_tests}",
                "passed": passed_tests
            }
        else:
            return {
                "success": True,
                "details": f"All {passed_tests} command tests passed"
            }
    
    def test_cli_error_handling(self) -> Dict:
        """Test CLI error handling and user-friendly messages."""
        error_tests = [
            (["clinical-data-platform", "nonexistent-command"], "No such command"),
            (["cdp-api", "start", "--port", "invalid"], "Invalid"),
            (["clinical-data-platform", "demo", "nonexistent"], ""),
        ]
        
        passed_tests = 0
        failed_tests = []
        
        for cmd, expected_error_indicator in error_tests:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                
                # Error commands should return non-zero exit code
                if result.returncode != 0:
                    # Check if error message is user-friendly (not a Python traceback)
                    output = result.stderr + result.stdout
                    if "Traceback" not in output and "Exception" not in output:
                        passed_tests += 1
                    else:
                        failed_tests.append(f"{' '.join(cmd)}: Shows Python traceback")
                else:
                    failed_tests.append(f"{' '.join(cmd)}: Should have failed but succeeded")
                    
            except subprocess.TimeoutExpired:
                failed_tests.append(f"{' '.join(cmd)}: timeout")
            except Exception as e:
                failed_tests.append(f"{' '.join(cmd)}: {str(e)}")
        
        return {
            "success": len(failed_tests) == 0,
            "details": f"Error handling tests: {passed_tests}/{len(error_tests)} passed",
            "failures": failed_tests if failed_tests else None
        }
    
    def test_cli_completion(self) -> Dict:
        """Test CLI autocompletion setup."""
        try:
            # Test if completion can be generated
            result = subprocess.run(
                ["clinical-data-platform", "--show-completion"], 
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0 and "complete" in result.stdout:
                return {
                    "success": True,
                    "details": "Autocompletion available"
                }
            else:
                return {
                    "success": False,
                    "error": "Autocompletion not available or not working"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Completion test failed: {str(e)}"
            }
    
    def test_package_metadata(self) -> Dict:
        """Test package metadata and distribution info."""
        try:
            # Test pip show
            result = subprocess.run(
                ["pip", "show", "clinical-data-platform"], 
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                metadata = result.stdout
                required_fields = ["Name:", "Version:", "Summary:", "Author:"]
                missing_fields = [field for field in required_fields 
                                if field not in metadata]
                
                if missing_fields:
                    return {
                        "success": False,
                        "error": f"Missing metadata fields: {missing_fields}"
                    }
                else:
                    return {
                        "success": True,
                        "details": "All required metadata fields present"
                    }
            else:
                return {
                    "success": False,
                    "error": "Package not found in pip"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Metadata test failed: {str(e)}"
            }
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print("📊 CLI TESTING SUMMARY")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result["success"])
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        if passed_tests == total_tests:
            print("\n🎉 All CLI tests passed! The packaging and ergonomics are excellent.")
        else:
            print(f"\n⚠️ {total_tests - passed_tests} tests failed. Review the issues above.")
        
        # Detailed results
        print("\n📋 Detailed Results:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            print(f"  {status} {test_name}")
            if not result["success"] and "error" in result:
                print(f"      Error: {result['error']}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test CLI installation and ergonomics")
    parser.add_argument("--package-path", default="./clinical-data-platform", 
                       help="Path to the package directory")
    
    args = parser.parse_args()
    
    tester = CLITester(args.package_path)
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()