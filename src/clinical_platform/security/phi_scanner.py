#!/usr/bin/env python3
"""
PHI (Protected Health Information) Scanner for Clinical Data Platform.

This module scans code, data, and configuration files for potential PHI exposure
and ensures compliance with HIPAA privacy rules.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import structlog

logger = structlog.get_logger()


class PHISeverity(Enum):
    """Severity levels for PHI findings."""
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class PHIFinding:
    """Represents a potential PHI exposure finding."""
    file_path: str
    line_number: int
    column: int
    phi_type: str
    severity: PHISeverity
    description: str
    matched_text: str
    context: str
    remediation: str


class PHIPatterns:
    """Common PHI patterns for detection."""
    
    # Social Security Numbers
    SSN_PATTERNS = [
        r'\b\d{3}-\d{2}-\d{4}\b',
        r'\b\d{3}\s\d{2}\s\d{4}\b',
        r'\b\d{9}\b',
    ]
    
    # Medical Record Numbers
    MRN_PATTERNS = [
        r'\bMRN\s*:?\s*\d+\b',
        r'\bMedical\s+Record\s+Number\s*:?\s*\d+\b',
        r'\bPatient\s+ID\s*:?\s*\d+\b',
    ]
    
    # Phone Numbers
    PHONE_PATTERNS = [
        r'\b\d{3}-\d{3}-\d{4}\b',
        r'\(\d{3}\)\s*\d{3}-\d{4}',
        r'\b\d{10}\b',
    ]
    
    # Email Addresses
    EMAIL_PATTERNS = [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    ]
    
    # Dates (potential birth dates)
    DATE_PATTERNS = [
        r'\b\d{1,2}/\d{1,2}/\d{4}\b',
        r'\b\d{4}-\d{2}-\d{2}\b',
        r'\b\d{1,2}-\d{1,2}-\d{4}\b',
    ]
    
    # Credit Card Numbers
    CREDIT_CARD_PATTERNS = [
        r'\b4\d{3}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Visa
        r'\b5[1-5]\d{2}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Mastercard
        r'\b3[47]\d{2}[\s-]?\d{6}[\s-]?\d{5}\b',  # American Express
    ]
    
    # Names (common patterns)
    NAME_PATTERNS = [
        r'\bDr\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',
        r'\bPatient\s+Name\s*:?\s*[A-Z][a-z]+\s+[A-Z][a-z]+\b',
    ]
    
    # IP Addresses (potentially identifying)
    IP_PATTERNS = [
        r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
    ]


class PHIScanner:
    """Scanner for detecting potential PHI in code and data files."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.findings: List[PHIFinding] = []
        self.excluded_paths: Set[str] = set(self.config.get('excluded_paths', []))
        self.included_extensions: Set[str] = set(self.config.get('included_extensions', [
            '.py', '.sql', '.json', '.yaml', '.yml', '.csv', '.txt', '.md'
        ]))
        
        # Compile patterns for performance
        self.compiled_patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[str, List[Tuple[re.Pattern, PHISeverity, str]]]:
        """Compile regex patterns for PHI detection."""
        patterns = {
            'ssn': [(re.compile(p), PHISeverity.CRITICAL, "Social Security Number") 
                   for p in PHIPatterns.SSN_PATTERNS],
            'mrn': [(re.compile(p), PHISeverity.HIGH, "Medical Record Number") 
                   for p in PHIPatterns.MRN_PATTERNS],
            'phone': [(re.compile(p), PHISeverity.MEDIUM, "Phone Number") 
                     for p in PHIPatterns.PHONE_PATTERNS],
            'email': [(re.compile(p), PHISeverity.MEDIUM, "Email Address") 
                     for p in PHIPatterns.EMAIL_PATTERNS],
            'date': [(re.compile(p), PHISeverity.LOW, "Date (Potential Birth Date)") 
                    for p in PHIPatterns.DATE_PATTERNS],
            'credit_card': [(re.compile(p), PHISeverity.HIGH, "Credit Card Number") 
                           for p in PHIPatterns.CREDIT_CARD_PATTERNS],
            'name': [(re.compile(p), PHISeverity.MEDIUM, "Personal Name") 
                    for p in PHIPatterns.NAME_PATTERNS],
            'ip': [(re.compile(p), PHISeverity.LOW, "IP Address") 
                  for p in PHIPatterns.IP_PATTERNS],
        }
        return patterns
    
    def scan_file(self, file_path: Path) -> List[PHIFinding]:
        """Scan a single file for PHI patterns."""
        if not self._should_scan_file(file_path):
            return []
        
        findings = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
            for line_num, line in enumerate(lines, 1):
                line_findings = self._scan_line(
                    line, str(file_path), line_num
                )
                findings.extend(line_findings)
                
        except Exception as e:
            logger.warning(f"Error scanning file {file_path}: {e}")
            
        return findings
    
    def _should_scan_file(self, file_path: Path) -> bool:
        """Determine if file should be scanned."""
        # Skip excluded paths
        if any(excluded in str(file_path) for excluded in self.excluded_paths):
            return False
            
        # Check file extension
        if file_path.suffix not in self.included_extensions:
            return False
            
        # Skip binary files
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                if b'\x00' in chunk:
                    return False
        except Exception:
            return False
            
        return True
    
    def _scan_line(self, line: str, file_path: str, line_num: int) -> List[PHIFinding]:
        """Scan a single line for PHI patterns."""
        findings = []
        
        for phi_type, patterns in self.compiled_patterns.items():
            for pattern, severity, description in patterns:
                matches = pattern.finditer(line)
                for match in matches:
                    # Skip obvious test data or examples
                    if self._is_likely_test_data(line, match.group()):
                        continue
                        
                    finding = PHIFinding(
                        file_path=file_path,
                        line_number=line_num,
                        column=match.start(),
                        phi_type=phi_type,
                        severity=severity,
                        description=description,
                        matched_text=match.group(),
                        context=line.strip(),
                        remediation=self._get_remediation(phi_type)
                    )
                    findings.append(finding)
                    
        return findings
    
    def _is_likely_test_data(self, line: str, matched_text: str) -> bool:
        """Check if the matched text is likely test/sample data."""
        line_lower = line.lower()
        
        # Common test indicators
        test_indicators = [
            'test', 'example', 'sample', 'demo', 'fake', 'mock',
            'dummy', 'placeholder', 'xxx', '000', '123-45-6789'
        ]
        
        # Check if line contains test indicators
        if any(indicator in line_lower for indicator in test_indicators):
            return True
            
        # Check for obviously fake SSNs
        if matched_text in ['123-45-6789', '000-00-0000', '999-99-9999']:
            return True
            
        return False
    
    def _get_remediation(self, phi_type: str) -> str:
        """Get remediation advice for PHI type."""
        remediation_map = {
            'ssn': "Remove or anonymize SSN. Use randomized IDs for testing.",
            'mrn': "Replace with synthetic MRNs. Use consistent mapping for testing.",
            'phone': "Use fake phone numbers (555-0100 to 555-0199) for testing.",
            'email': "Use example.com domain for test emails.",
            'date': "Replace with shifted dates maintaining relative relationships.",
            'credit_card': "Use test credit card numbers provided by payment processors.",
            'name': "Replace with synthetic names or anonymize.",
            'ip': "Use private IP ranges (192.168.x.x, 10.x.x.x) for testing."
        }
        return remediation_map.get(phi_type, "Review and anonymize if contains real PHI.")
    
    def scan_directory(self, directory_path: Path) -> List[PHIFinding]:
        """Scan all applicable files in a directory."""
        all_findings = []
        
        for file_path in directory_path.rglob('*'):
            if file_path.is_file():
                file_findings = self.scan_file(file_path)
                all_findings.extend(file_findings)
                
        self.findings = all_findings
        return all_findings
    
    def generate_report(self, output_format: str = 'json') -> str:
        """Generate a report of PHI findings."""
        if output_format == 'json':
            return self._generate_json_report()
        elif output_format == 'text':
            return self._generate_text_report()
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _generate_json_report(self) -> str:
        """Generate JSON report."""
        report = {
            'scan_summary': {
                'total_findings': len(self.findings),
                'critical_findings': len([f for f in self.findings if f.severity == PHISeverity.CRITICAL]),
                'high_findings': len([f for f in self.findings if f.severity == PHISeverity.HIGH]),
                'medium_findings': len([f for f in self.findings if f.severity == PHISeverity.MEDIUM]),
                'low_findings': len([f for f in self.findings if f.severity == PHISeverity.LOW]),
            },
            'findings': [
                {
                    **asdict(finding),
                    'severity': finding.severity.value
                }
                for finding in self.findings
            ]
        }
        return json.dumps(report, indent=2)
    
    def _generate_text_report(self) -> str:
        """Generate text report."""
        lines = [
            "PHI Scanner Report",
            "=" * 50,
            "",
            f"Total findings: {len(self.findings)}",
            f"Critical: {len([f for f in self.findings if f.severity == PHISeverity.CRITICAL])}",
            f"High: {len([f for f in self.findings if f.severity == PHISeverity.HIGH])}",
            f"Medium: {len([f for f in self.findings if f.severity == PHISeverity.MEDIUM])}",
            f"Low: {len([f for f in self.findings if f.severity == PHISeverity.LOW])}",
            "",
            "Findings:",
            "-" * 50,
        ]
        
        for finding in sorted(self.findings, key=lambda x: (x.severity.value, x.file_path)):
            lines.extend([
                f"",
                f"File: {finding.file_path}:{finding.line_number}",
                f"Type: {finding.phi_type} ({finding.severity.value})",
                f"Description: {finding.description}",
                f"Matched: {finding.matched_text}",
                f"Context: {finding.context}",
                f"Remediation: {finding.remediation}",
            ])
            
        return "\n".join(lines)


def main():
    """CLI entry point for PHI scanner."""
    parser = argparse.ArgumentParser(description="Scan for PHI in code and data files")
    parser.add_argument("--scan-path", required=True, help="Path to scan")
    parser.add_argument("--output-format", choices=['json', 'text'], default='json')
    parser.add_argument("--output-file", help="Output file path")
    parser.add_argument("--config", help="Configuration file path")
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Initialize scanner
    scanner = PHIScanner(config)
    
    # Scan directory
    scan_path = Path(args.scan_path)
    if scan_path.is_file():
        findings = scanner.scan_file(scan_path)
    else:
        findings = scanner.scan_directory(scan_path)
    
    # Generate report
    report = scanner.generate_report(args.output_format)
    
    # Output report
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(report)
        print(f"Report written to {args.output_file}")
    else:
        print(report)
    
    # Exit with error code if critical findings
    critical_count = len([f for f in findings if f.severity == PHISeverity.CRITICAL])
    if critical_count > 0:
        print(f"\nERROR: {critical_count} critical PHI findings detected!", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()