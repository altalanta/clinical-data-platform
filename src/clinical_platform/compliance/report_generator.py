#!/usr/bin/env python3
"""
Compliance Report Generator for Clinical Data Platform.

This module generates comprehensive compliance reports by aggregating
results from security scans, PHI validation, and HIPAA audits.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import html

import structlog

logger = structlog.get_logger()


class ComplianceReportGenerator:
    """Generates comprehensive compliance reports."""
    
    def __init__(self):
        self.report_data = {}
        self.timestamp = datetime.now(timezone.utc)
        
    def load_security_scan_results(self, results_dir: Path) -> None:
        """Load security scan results from directory."""
        
        # Load Bandit results
        bandit_file = results_dir / "bandit-results.sarif"
        if bandit_file.exists():
            try:
                with open(bandit_file, 'r') as f:
                    bandit_data = json.load(f)
                self.report_data['bandit_scan'] = self._process_bandit_results(bandit_data)
            except Exception as e:
                logger.warning(f"Failed to load Bandit results: {e}")
                
        # Load Safety results
        safety_file = results_dir / "safety-results.json"
        if safety_file.exists():
            try:
                with open(safety_file, 'r') as f:
                    safety_data = json.load(f)
                self.report_data['safety_scan'] = self._process_safety_results(safety_data)
            except Exception as e:
                logger.warning(f"Failed to load Safety results: {e}")
    
    def load_phi_compliance_results(self, results_dir: Path) -> None:
        """Load PHI compliance results."""
        phi_file = results_dir / "phi-compliance-results.json"
        if phi_file.exists():
            try:
                with open(phi_file, 'r') as f:
                    phi_data = json.load(f)
                self.report_data['phi_compliance'] = phi_data
            except Exception as e:
                logger.warning(f"Failed to load PHI compliance results: {e}")
    
    def load_hipaa_audit_results(self, results_dir: Path) -> None:
        """Load HIPAA audit results."""
        hipaa_file = results_dir / "hipaa-audit-results.json"
        if hipaa_file.exists():
            try:
                with open(hipaa_file, 'r') as f:
                    hipaa_data = json.load(f)
                self.report_data['hipaa_audit'] = hipaa_data
            except Exception as e:
                logger.warning(f"Failed to load HIPAA audit results: {e}")
    
    def load_secrets_scan_results(self, results_dir: Path) -> None:
        """Load secrets scan results."""
        secrets_file = results_dir / "trufflehog-results.json"
        if secrets_file.exists():
            try:
                with open(secrets_file, 'r') as f:
                    # TruffleHog results can be JSONL format
                    secrets_data = []
                    for line in f:
                        if line.strip():
                            secrets_data.append(json.loads(line))
                self.report_data['secrets_scan'] = self._process_secrets_results(secrets_data)
            except Exception as e:
                logger.warning(f"Failed to load secrets scan results: {e}")
    
    def _process_bandit_results(self, bandit_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Bandit SARIF results."""
        processed = {
            'total_issues': 0,
            'high_severity': 0,
            'medium_severity': 0,
            'low_severity': 0,
            'issues': []
        }
        
        if 'runs' in bandit_data:
            for run in bandit_data['runs']:
                results = run.get('results', [])
                processed['total_issues'] = len(results)
                
                for result in results:
                    level = result.get('level', 'unknown')
                    
                    if level == 'error':
                        processed['high_severity'] += 1
                    elif level == 'warning':
                        processed['medium_severity'] += 1
                    elif level == 'note':
                        processed['low_severity'] += 1
                    
                    processed['issues'].append({
                        'rule_id': result.get('ruleId', ''),
                        'message': result.get('message', {}).get('text', ''),
                        'severity': level,
                        'locations': result.get('locations', [])
                    })
        
        return processed
    
    def _process_safety_results(self, safety_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process Safety vulnerability results."""
        processed = {
            'total_vulnerabilities': len(safety_data),
            'critical_vulnerabilities': 0,
            'high_vulnerabilities': 0,
            'medium_vulnerabilities': 0,
            'low_vulnerabilities': 0,
            'vulnerabilities': []
        }
        
        for vuln in safety_data:
            severity = vuln.get('severity', 'unknown').lower()
            
            if severity == 'critical':
                processed['critical_vulnerabilities'] += 1
            elif severity == 'high':
                processed['high_vulnerabilities'] += 1
            elif severity == 'medium':
                processed['medium_vulnerabilities'] += 1
            elif severity == 'low':
                processed['low_vulnerabilities'] += 1
            
            processed['vulnerabilities'].append({
                'package': vuln.get('package_name', ''),
                'vulnerability_id': vuln.get('vulnerability_id', ''),
                'severity': severity,
                'advisory': vuln.get('advisory', ''),
                'affected_versions': vuln.get('affected_versions', '')
            })
        
        return processed
    
    def _process_secrets_results(self, secrets_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process secrets scan results."""
        processed = {
            'total_secrets': len(secrets_data),
            'verified_secrets': 0,
            'unverified_secrets': 0,
            'secrets_by_type': {},
            'secrets': []
        }
        
        for secret in secrets_data:
            verified = secret.get('verified', False)
            if verified:
                processed['verified_secrets'] += 1
            else:
                processed['unverified_secrets'] += 1
            
            detector_name = secret.get('DetectorName', 'unknown')
            if detector_name not in processed['secrets_by_type']:
                processed['secrets_by_type'][detector_name] = 0
            processed['secrets_by_type'][detector_name] += 1
            
            processed['secrets'].append({
                'detector': detector_name,
                'verified': verified,
                'file': secret.get('file', ''),
                'line': secret.get('line', 0)
            })
        
        return processed
    
    def generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary of compliance status."""
        summary = {
            'overall_status': 'UNKNOWN',
            'risk_level': 'UNKNOWN',
            'compliance_score': 0,
            'critical_issues': 0,
            'recommendations': []
        }
        
        critical_issues = 0
        total_checks = 0
        passed_checks = 0
        
        # Analyze security scan results
        if 'bandit_scan' in self.report_data:
            bandit = self.report_data['bandit_scan']
            critical_issues += bandit.get('high_severity', 0)
            
        if 'safety_scan' in self.report_data:
            safety = self.report_data['safety_scan']
            critical_issues += safety.get('critical_vulnerabilities', 0)
            critical_issues += safety.get('high_vulnerabilities', 0)
            
        if 'secrets_scan' in self.report_data:
            secrets = self.report_data['secrets_scan']
            critical_issues += secrets.get('verified_secrets', 0)
        
        # Analyze PHI compliance
        if 'phi_compliance' in self.report_data:
            phi = self.report_data['phi_compliance']
            if 'summary' in phi:
                phi_summary = phi['summary']
                total_checks += phi_summary.get('total_checks', 0)
                passed_checks += phi_summary.get('passed', 0)
                critical_issues += phi_summary.get('failed', 0)
        
        # Analyze HIPAA audit
        if 'hipaa_audit' in self.report_data:
            hipaa = self.report_data['hipaa_audit']
            if 'executive_summary' in hipaa:
                hipaa_summary = hipaa['executive_summary']
                total_checks += hipaa_summary.get('total_requirements_checked', 0)
                passed_checks += hipaa_summary.get('compliant_requirements', 0)
                critical_issues += hipaa_summary.get('non_compliant_requirements', 0)
        
        # Calculate compliance score
        if total_checks > 0:
            summary['compliance_score'] = round((passed_checks / total_checks) * 100, 2)
        
        # Determine overall status
        summary['critical_issues'] = critical_issues
        
        if critical_issues == 0 and summary['compliance_score'] >= 95:
            summary['overall_status'] = 'COMPLIANT'
            summary['risk_level'] = 'LOW'
        elif critical_issues <= 2 and summary['compliance_score'] >= 80:
            summary['overall_status'] = 'MOSTLY_COMPLIANT'
            summary['risk_level'] = 'MEDIUM'
        elif critical_issues <= 5 and summary['compliance_score'] >= 60:
            summary['overall_status'] = 'PARTIALLY_COMPLIANT'
            summary['risk_level'] = 'HIGH'
        else:
            summary['overall_status'] = 'NON_COMPLIANT'
            summary['risk_level'] = 'CRITICAL'
        
        # Generate recommendations
        summary['recommendations'] = self._generate_recommendations()
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate priority recommendations based on findings."""
        recommendations = []
        
        # Security scan recommendations
        if 'bandit_scan' in self.report_data:
            bandit = self.report_data['bandit_scan']
            if bandit.get('high_severity', 0) > 0:
                recommendations.append("Address high-severity security issues identified by Bandit scan")
        
        if 'safety_scan' in self.report_data:
            safety = self.report_data['safety_scan']
            if safety.get('critical_vulnerabilities', 0) > 0:
                recommendations.append("Update dependencies with critical security vulnerabilities")
        
        if 'secrets_scan' in self.report_data:
            secrets = self.report_data['secrets_scan']
            if secrets.get('verified_secrets', 0) > 0:
                recommendations.append("Remove or rotate verified secrets found in code")
        
        # PHI compliance recommendations
        if 'phi_compliance' in self.report_data:
            phi = self.report_data['phi_compliance']
            if phi.get('summary', {}).get('failed', 0) > 0:
                recommendations.append("Implement missing PHI protection controls")
        
        # HIPAA compliance recommendations
        if 'hipaa_audit' in self.report_data:
            hipaa = self.report_data['hipaa_audit']
            priority_recs = hipaa.get('priority_recommendations', [])
            recommendations.extend(priority_recs[:3])  # Top 3 HIPAA recommendations
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def generate_html_report(self) -> str:
        """Generate HTML compliance report."""
        executive_summary = self.generate_executive_summary()
        
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clinical Data Platform - Compliance Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        .header {{
            text-align: center;
            border-bottom: 3px solid #2c3e50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #2c3e50;
            margin: 0;
            font-size: 2.5em;
        }}
        .header .timestamp {{
            color: #7f8c8d;
            margin-top: 10px;
        }}
        .executive-summary {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .summary-card {{
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            font-size: 2em;
        }}
        .status-badge {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            text-transform: uppercase;
        }}
        .status-compliant {{ background-color: #27ae60; }}
        .status-mostly-compliant {{ background-color: #f39c12; }}
        .status-partially-compliant {{ background-color: #e74c3c; }}
        .status-non-compliant {{ background-color: #c0392b; }}
        .risk-low {{ background-color: #27ae60; }}
        .risk-medium {{ background-color: #f39c12; }}
        .risk-high {{ background-color: #e67e22; }}
        .risk-critical {{ background-color: #e74c3c; }}
        .section {{
            margin-bottom: 40px;
        }}
        .section h2 {{
            color: #2c3e50;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-bottom: 20px;
        }}
        .findings-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        .finding-card {{
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 20px;
            background-color: #fafafa;
        }}
        .finding-card h3 {{
            color: #2c3e50;
            margin-top: 0;
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }}
        .metric:last-child {{
            border-bottom: none;
        }}
        .recommendations {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 6px;
            border-left: 4px solid #e74c3c;
        }}
        .recommendations ul {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        .recommendations li {{
            margin-bottom: 8px;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Compliance Report</h1>
            <p class="timestamp">Generated on {timestamp}</p>
        </div>
        
        <div class="executive-summary">
            <h2>Executive Summary</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <h3>{compliance_score}%</h3>
                    <p>Compliance Score</p>
                </div>
                <div class="summary-card">
                    <h3>{critical_issues}</h3>
                    <p>Critical Issues</p>
                </div>
                <div class="summary-card">
                    <h3><span class="status-badge status-{overall_status_class}">{overall_status}</span></h3>
                    <p>Overall Status</p>
                </div>
                <div class="summary-card">
                    <h3><span class="status-badge risk-{risk_level_class}">{risk_level}</span></h3>
                    <p>Risk Level</p>
                </div>
            </div>
        </div>
        
        {findings_section}
        
        <div class="section">
            <h2>Priority Recommendations</h2>
            <div class="recommendations">
                <p><strong>Immediate actions required to improve compliance:</strong></p>
                <ul>
                    {recommendations_list}
                </ul>
            </div>
        </div>
        
        <div class="footer">
            <p>This report was automatically generated by the Clinical Data Platform compliance monitoring system.</p>
            <p>For questions or support, contact the security team.</p>
        </div>
    </div>
</body>
</html>
        """
        
        # Format variables
        timestamp = self.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
        overall_status = executive_summary['overall_status'].replace('_', ' ')
        overall_status_class = executive_summary['overall_status'].lower().replace('_', '-')
        risk_level = executive_summary['risk_level']
        risk_level_class = risk_level.lower()
        
        recommendations_list = '\n'.join([
            f"<li>{html.escape(rec)}</li>" 
            for rec in executive_summary['recommendations']
        ])
        
        findings_section = self._generate_findings_html()
        
        return html_template.format(
            timestamp=timestamp,
            compliance_score=executive_summary['compliance_score'],
            critical_issues=executive_summary['critical_issues'],
            overall_status=overall_status,
            overall_status_class=overall_status_class,
            risk_level=risk_level,
            risk_level_class=risk_level_class,
            findings_section=findings_section,
            recommendations_list=recommendations_list
        )
    
    def _generate_findings_html(self) -> str:
        """Generate HTML for detailed findings."""
        findings_html = '<div class="section"><h2>Detailed Findings</h2><div class="findings-grid">'
        
        # Bandit findings
        if 'bandit_scan' in self.report_data:
            bandit = self.report_data['bandit_scan']
            findings_html += f'''
            <div class="finding-card">
                <h3>Security Code Analysis (Bandit)</h3>
                <div class="metric">
                    <span>Total Issues:</span>
                    <strong>{bandit.get('total_issues', 0)}</strong>
                </div>
                <div class="metric">
                    <span>High Severity:</span>
                    <strong>{bandit.get('high_severity', 0)}</strong>
                </div>
                <div class="metric">
                    <span>Medium Severity:</span>
                    <strong>{bandit.get('medium_severity', 0)}</strong>
                </div>
                <div class="metric">
                    <span>Low Severity:</span>
                    <strong>{bandit.get('low_severity', 0)}</strong>
                </div>
            </div>
            '''
        
        # Safety findings
        if 'safety_scan' in self.report_data:
            safety = self.report_data['safety_scan']
            findings_html += f'''
            <div class="finding-card">
                <h3>Dependency Vulnerabilities (Safety)</h3>
                <div class="metric">
                    <span>Total Vulnerabilities:</span>
                    <strong>{safety.get('total_vulnerabilities', 0)}</strong>
                </div>
                <div class="metric">
                    <span>Critical:</span>
                    <strong>{safety.get('critical_vulnerabilities', 0)}</strong>
                </div>
                <div class="metric">
                    <span>High:</span>
                    <strong>{safety.get('high_vulnerabilities', 0)}</strong>
                </div>
                <div class="metric">
                    <span>Medium/Low:</span>
                    <strong>{safety.get('medium_vulnerabilities', 0) + safety.get('low_vulnerabilities', 0)}</strong>
                </div>
            </div>
            '''
        
        # PHI compliance findings
        if 'phi_compliance' in self.report_data:
            phi = self.report_data['phi_compliance']
            phi_summary = phi.get('summary', {})
            findings_html += f'''
            <div class="finding-card">
                <h3>PHI Compliance</h3>
                <div class="metric">
                    <span>Total Checks:</span>
                    <strong>{phi_summary.get('total_checks', 0)}</strong>
                </div>
                <div class="metric">
                    <span>Passed:</span>
                    <strong>{phi_summary.get('passed', 0)}</strong>
                </div>
                <div class="metric">
                    <span>Failed:</span>
                    <strong>{phi_summary.get('failed', 0)}</strong>
                </div>
                <div class="metric">
                    <span>Warnings:</span>
                    <strong>{phi_summary.get('warnings', 0)}</strong>
                </div>
            </div>
            '''
        
        # HIPAA audit findings
        if 'hipaa_audit' in self.report_data:
            hipaa = self.report_data['hipaa_audit']
            hipaa_summary = hipaa.get('executive_summary', {})
            findings_html += f'''
            <div class="finding-card">
                <h3>HIPAA Compliance Audit</h3>
                <div class="metric">
                    <span>Compliance Percentage:</span>
                    <strong>{hipaa_summary.get('overall_compliance_percentage', 0)}%</strong>
                </div>
                <div class="metric">
                    <span>Compliant Requirements:</span>
                    <strong>{hipaa_summary.get('compliant_requirements', 0)}</strong>
                </div>
                <div class="metric">
                    <span>Non-Compliant:</span>
                    <strong>{hipaa_summary.get('non_compliant_requirements', 0)}</strong>
                </div>
                <div class="metric">
                    <span>Status:</span>
                    <strong>{hipaa_summary.get('overall_status', 'Unknown')}</strong>
                </div>
            </div>
            '''
        
        findings_html += '</div></div>'
        return findings_html


def main():
    """CLI entry point for compliance report generator."""
    parser = argparse.ArgumentParser(description="Generate comprehensive compliance report")
    parser.add_argument("--input-dir", action="append", help="Input directory with scan results")
    parser.add_argument("--output-file", required=True, help="Output file path")
    parser.add_argument("--format", choices=['html', 'json'], default='html')
    
    args = parser.parse_args()
    
    # Initialize report generator
    generator = ComplianceReportGenerator()
    
    # Load results from input directories
    if args.input_dir:
        for input_dir in args.input_dir:
            input_path = Path(input_dir)
            if input_path.exists():
                generator.load_security_scan_results(input_path)
                generator.load_phi_compliance_results(input_path)
                generator.load_hipaa_audit_results(input_path)
                generator.load_secrets_scan_results(input_path)
    
    # Generate report
    if args.format == 'html':
        report_content = generator.generate_html_report()
    else:
        report_data = {
            'executive_summary': generator.generate_executive_summary(),
            'detailed_findings': generator.report_data,
            'generated_at': generator.timestamp.isoformat()
        }
        report_content = json.dumps(report_data, indent=2)
    
    # Write report
    with open(args.output_file, 'w') as f:
        f.write(report_content)
    
    print(f"Compliance report generated: {args.output_file}")


if __name__ == "__main__":
    main()