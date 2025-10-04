#!/usr/bin/env python3
"""
dbt Quality Gates Validation Script

This script validates dbt quality gates and generates comprehensive reports
for the Clinical Data Platform.
"""

import json
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import subprocess


class DBTQualityValidator:
    """Validates dbt quality gates and generates reports."""
    
    def __init__(self, dbt_project_dir: str = "analytics/dbt"):
        self.dbt_project_dir = Path(dbt_project_dir)
        self.target_dir = self.dbt_project_dir / "target"
        self.quality_thresholds = {
            "critical_test_success_rate": 1.0,  # 100% for critical tests
            "overall_test_success_rate": 0.95,  # 95% for all tests
            "freshness_pass_rate": 0.90,       # 90% for freshness checks
            "max_critical_failures": 0,         # 0 critical failures allowed
        }
        
    def run_dbt_command(self, command: List[str]) -> Dict[str, Any]:
        """Execute dbt command and return results."""
        try:
            cmd = ["dbt"] + command + ["--profiles-dir", str(self.dbt_project_dir)]
            result = subprocess.run(
                cmd, 
                cwd=self.dbt_project_dir,
                capture_output=True, 
                text=True,
                check=False
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "returncode": -1
            }
    
    def load_run_results(self) -> Optional[Dict[str, Any]]:
        """Load dbt run results from target directory."""
        run_results_path = self.target_dir / "run_results.json"
        
        if not run_results_path.exists():
            print(f"Warning: {run_results_path} not found")
            return None
            
        try:
            with open(run_results_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading run results: {e}")
            return None
    
    def load_sources_results(self) -> Optional[Dict[str, Any]]:
        """Load dbt source freshness results."""
        sources_path = self.target_dir / "sources.json"
        
        if not sources_path.exists():
            print(f"Warning: {sources_path} not found")
            return None
            
        try:
            with open(sources_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading sources results: {e}")
            return None
    
    def analyze_test_results(self, run_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dbt test results."""
        if not run_results or 'results' not in run_results:
            return {"error": "No test results found"}
        
        test_results = [
            r for r in run_results['results'] 
            if r.get('resource_type') == 'test'
        ]
        
        total_tests = len(test_results)
        passed_tests = len([r for r in test_results if r.get('status') == 'success'])
        failed_tests = total_tests - passed_tests
        
        # Categorize tests by criticality
        critical_patterns = [
            'unique', 'not_null', 'referential_integrity', 
            'freshness_slo', 'assert_'
        ]
        
        critical_tests = []
        warning_tests = []
        
        for test in test_results:
            test_name = test.get('unique_id', '').lower()
            if any(pattern in test_name for pattern in critical_patterns):
                critical_tests.append(test)
            else:
                warning_tests.append(test)
        
        critical_failures = [
            t for t in critical_tests 
            if t.get('status') != 'success'
        ]
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "critical_tests": len(critical_tests),
            "critical_failures": len(critical_failures),
            "critical_failure_details": critical_failures,
            "warning_tests": len(warning_tests),
            "warning_failures": len([t for t in warning_tests if t.get('status') != 'success'])
        }
    
    def analyze_freshness_results(self, sources_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dbt source freshness results."""
        if not sources_results or 'results' not in sources_results:
            return {"error": "No freshness results found"}
        
        freshness_results = sources_results['results']
        total_sources = len(freshness_results)
        
        passed_freshness = len([
            r for r in freshness_results 
            if r.get('status') in ['pass', 'warn']
        ])
        
        failed_freshness = total_sources - passed_freshness
        
        critical_freshness_failures = [
            r for r in freshness_results 
            if r.get('status') == 'error'
        ]
        
        return {
            "total_sources": total_sources,
            "passed_freshness": passed_freshness,
            "failed_freshness": failed_freshness,
            "freshness_pass_rate": passed_freshness / total_sources if total_sources > 0 else 0,
            "critical_freshness_failures": len(critical_freshness_failures),
            "critical_freshness_details": critical_freshness_failures
        }
    
    def validate_quality_gates(self, test_analysis: Dict[str, Any], 
                             freshness_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Validate quality gates against thresholds."""
        gates_status = {}
        overall_status = "PASS"
        
        # Critical test success rate
        critical_success_rate = (
            (test_analysis.get('critical_tests', 0) - test_analysis.get('critical_failures', 0)) /
            max(test_analysis.get('critical_tests', 1), 1)
        )
        
        gates_status["critical_test_success_rate"] = {
            "value": critical_success_rate,
            "threshold": self.quality_thresholds["critical_test_success_rate"],
            "status": "PASS" if critical_success_rate >= self.quality_thresholds["critical_test_success_rate"] else "FAIL"
        }
        
        # Overall test success rate
        gates_status["overall_test_success_rate"] = {
            "value": test_analysis.get('success_rate', 0),
            "threshold": self.quality_thresholds["overall_test_success_rate"],
            "status": "PASS" if test_analysis.get('success_rate', 0) >= self.quality_thresholds["overall_test_success_rate"] else "FAIL"
        }
        
        # Freshness pass rate
        freshness_pass_rate = freshness_analysis.get('freshness_pass_rate', 0)
        gates_status["freshness_pass_rate"] = {
            "value": freshness_pass_rate,
            "threshold": self.quality_thresholds["freshness_pass_rate"],
            "status": "PASS" if freshness_pass_rate >= self.quality_thresholds["freshness_pass_rate"] else "FAIL"
        }
        
        # Critical failures
        critical_failures = test_analysis.get('critical_failures', 0)
        gates_status["max_critical_failures"] = {
            "value": critical_failures,
            "threshold": self.quality_thresholds["max_critical_failures"],
            "status": "PASS" if critical_failures <= self.quality_thresholds["max_critical_failures"] else "FAIL"
        }
        
        # Determine overall status
        if any(gate["status"] == "FAIL" for gate in gates_status.values()):
            overall_status = "FAIL"
        
        return {
            "overall_status": overall_status,
            "gates": gates_status,
            "summary": {
                "total_gates": len(gates_status),
                "passed_gates": len([g for g in gates_status.values() if g["status"] == "PASS"]),
                "failed_gates": len([g for g in gates_status.values() if g["status"] == "FAIL"])
            }
        }
    
    def generate_report(self, test_analysis: Dict[str, Any], 
                       freshness_analysis: Dict[str, Any],
                       gates_validation: Dict[str, Any]) -> str:
        """Generate comprehensive quality report."""
        timestamp = datetime.now().isoformat()
        
        report = f"""# dbt Quality Gates Report

**Generated:** {timestamp}  
**Overall Status:** {gates_validation['overall_status']}

## Summary

- **Total Tests:** {test_analysis.get('total_tests', 0)}
- **Passed Tests:** {test_analysis.get('passed_tests', 0)}
- **Failed Tests:** {test_analysis.get('failed_tests', 0)}
- **Success Rate:** {test_analysis.get('success_rate', 0):.1%}

- **Critical Tests:** {test_analysis.get('critical_tests', 0)}
- **Critical Failures:** {test_analysis.get('critical_failures', 0)}

- **Total Sources:** {freshness_analysis.get('total_sources', 0)}
- **Freshness Pass Rate:** {freshness_analysis.get('freshness_pass_rate', 0):.1%}

## Quality Gates Status

"""
        
        for gate_name, gate_info in gates_validation.get('gates', {}).items():
            status_icon = "✅" if gate_info['status'] == "PASS" else "❌"
            report += f"- {status_icon} **{gate_name.replace('_', ' ').title()}:** {gate_info['value']:.1%} (threshold: {gate_info['threshold']:.1%})\n"
        
        # Add critical failures details
        critical_failures = test_analysis.get('critical_failure_details', [])
        if critical_failures:
            report += "\n## Critical Test Failures\n\n"
            for failure in critical_failures:
                report += f"- **{failure.get('unique_id', 'Unknown')}:** {failure.get('status', 'Unknown')}\n"
                if failure.get('message'):
                    report += f"  - Message: {failure['message']}\n"
        
        # Add freshness failures
        freshness_failures = freshness_analysis.get('critical_freshness_details', [])
        if freshness_failures:
            report += "\n## Critical Freshness Failures\n\n"
            for failure in freshness_failures:
                report += f"- **{failure.get('unique_id', 'Unknown')}:** {failure.get('status', 'Unknown')}\n"
        
        report += f"""
## Recommendations

"""
        
        if gates_validation['overall_status'] == "FAIL":
            report += "🚨 **IMMEDIATE ACTION REQUIRED:** Critical quality gates have failed.\n\n"
            
        if test_analysis.get('critical_failures', 0) > 0:
            report += "- Fix critical test failures before deploying to production\n"
            
        if test_analysis.get('success_rate', 0) < 0.95:
            report += "- Investigate and resolve test failures to improve overall quality\n"
            
        if freshness_analysis.get('freshness_pass_rate', 0) < 0.90:
            report += "- Review data pipeline freshness and update SLOs if necessary\n"
        
        return report
    
    def run_full_validation(self, skip_freshness: bool = False) -> Dict[str, Any]:
        """Run complete dbt quality validation."""
        print("🔍 Running dbt quality validation...")
        
        # Run dbt tests
        print("Running dbt tests...")
        test_result = self.run_dbt_command(["test", "--store-failures"])
        
        # Run freshness checks (unless skipped)
        freshness_result = None
        if not skip_freshness:
            print("Running freshness checks...")
            freshness_result = self.run_dbt_command(["source", "freshness"])
        
        # Load and analyze results
        run_results = self.load_run_results()
        sources_results = self.load_sources_results() if not skip_freshness else {}
        
        test_analysis = self.analyze_test_results(run_results or {})
        freshness_analysis = self.analyze_freshness_results(sources_results or {})
        gates_validation = self.validate_quality_gates(test_analysis, freshness_analysis)
        
        # Generate report
        report = self.generate_report(test_analysis, freshness_analysis, gates_validation)
        
        return {
            "test_analysis": test_analysis,
            "freshness_analysis": freshness_analysis,
            "gates_validation": gates_validation,
            "report": report,
            "dbt_results": {
                "test_success": test_result.get("success", False),
                "freshness_success": freshness_result.get("success", True) if freshness_result else True
            }
        }


def main():
    parser = argparse.ArgumentParser(description="Validate dbt quality gates")
    parser.add_argument("--dbt-dir", default="analytics/dbt", help="dbt project directory")
    parser.add_argument("--skip-freshness", action="store_true", help="Skip freshness checks")
    parser.add_argument("--output", help="Output file for report")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit with error code if quality gates fail")
    
    args = parser.parse_args()
    
    validator = DBTQualityValidator(args.dbt_dir)
    results = validator.run_full_validation(skip_freshness=args.skip_freshness)
    
    # Print report
    print("\n" + "="*80)
    print(results["report"])
    print("="*80)
    
    # Save report if output specified
    if args.output:
        with open(args.output, 'w') as f:
            f.write(results["report"])
        print(f"\n📄 Report saved to {args.output}")
    
    # Exit with appropriate code
    overall_status = results["gates_validation"]["overall_status"]
    if args.fail_on_error and overall_status == "FAIL":
        print("\n❌ Quality gates failed - exiting with error code")
        sys.exit(1)
    elif overall_status == "PASS":
        print("\n✅ All quality gates passed!")
        sys.exit(0)
    else:
        print("\n⚠️  Quality gates failed but continuing...")
        sys.exit(0)


if __name__ == "__main__":
    main()