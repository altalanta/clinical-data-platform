#!/usr/bin/env python3
"""
HIPAA Compliance Auditor for Clinical Data Platform.

This module performs comprehensive HIPAA compliance audits including
administrative, physical, and technical safeguards.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

import structlog

logger = structlog.get_logger()


class ComplianceLevel(Enum):
    """HIPAA compliance levels."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class ComplianceCheck:
    """Represents a HIPAA compliance check."""
    safeguard_type: str
    requirement_id: str
    requirement_name: str
    description: str
    compliance_level: ComplianceLevel
    findings: List[str]
    recommendations: List[str]
    evidence: Optional[Dict[str, Any]] = None


class HIPAASafeguards:
    """HIPAA Administrative, Physical, and Technical Safeguards."""
    
    ADMINISTRATIVE_SAFEGUARDS = {
        "164.308(a)(1)": {
            "name": "Security Officer",
            "description": "Assign security responsibilities to an individual",
            "required": True
        },
        "164.308(a)(2)": {
            "name": "Assigned Security Responsibilities", 
            "description": "Identify security officer and assign security responsibilities",
            "required": True
        },
        "164.308(a)(3)": {
            "name": "Workforce Training",
            "description": "Implement procedures for workforce training and access management",
            "required": True
        },
        "164.308(a)(4)": {
            "name": "Information Access Management",
            "description": "Implement procedures for granting access to PHI",
            "required": True
        },
        "164.308(a)(5)": {
            "name": "Security Awareness and Training",
            "description": "Implement security awareness and training program",
            "required": True
        },
        "164.308(a)(6)": {
            "name": "Security Incident Procedures",
            "description": "Implement procedures to address security incidents",
            "required": True
        },
        "164.308(a)(7)": {
            "name": "Contingency Plan",
            "description": "Establish and implement procedures for responding to emergencies",
            "required": True
        },
        "164.308(a)(8)": {
            "name": "Evaluation",
            "description": "Perform periodic technical and non-technical evaluation",
            "required": True
        },
    }
    
    PHYSICAL_SAFEGUARDS = {
        "164.310(a)(1)": {
            "name": "Facility Access Controls",
            "description": "Limit physical access to systems containing PHI",
            "required": True
        },
        "164.310(a)(2)": {
            "name": "Workstation Use",
            "description": "Implement policies for workstation use",
            "required": True
        },
        "164.310(b)": {
            "name": "Workstation Security",
            "description": "Implement physical safeguards for workstations",
            "required": True
        },
        "164.310(c)": {
            "name": "Device and Media Controls",
            "description": "Implement policies for electronic media",
            "required": True
        },
    }
    
    TECHNICAL_SAFEGUARDS = {
        "164.312(a)(1)": {
            "name": "Access Control",
            "description": "Implement technical policies for accessing PHI",
            "required": True
        },
        "164.312(b)": {
            "name": "Audit Controls",
            "description": "Implement hardware, software, and procedural mechanisms for audit trails",
            "required": True
        },
        "164.312(c)(1)": {
            "name": "Integrity",
            "description": "Protect PHI from improper alteration or destruction",
            "required": True
        },
        "164.312(d)": {
            "name": "Person or Entity Authentication",
            "description": "Verify identity before access to PHI",
            "required": True
        },
        "164.312(e)(1)": {
            "name": "Transmission Security",
            "description": "Implement technical safeguards for PHI transmission",
            "required": True
        },
    }


class HIPAAAuditor:
    """Performs HIPAA compliance audits."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.compliance_checks: List[ComplianceCheck] = []
        self.audit_timestamp = datetime.utcnow()
        
    def audit_administrative_safeguards(self) -> List[ComplianceCheck]:
        """Audit administrative safeguards compliance."""
        checks = []
        admin_config = self.config.get('administrative_safeguards', {})
        
        for req_id, requirement in HIPAASafeguards.ADMINISTRATIVE_SAFEGUARDS.items():
            check = self._evaluate_safeguard(
                safeguard_type="Administrative",
                req_id=req_id,
                requirement=requirement,
                config=admin_config.get(req_id, {})
            )
            checks.append(check)
            
        return checks
    
    def audit_physical_safeguards(self) -> List[ComplianceCheck]:
        """Audit physical safeguards compliance."""
        checks = []
        physical_config = self.config.get('physical_safeguards', {})
        
        for req_id, requirement in HIPAASafeguards.PHYSICAL_SAFEGUARDS.items():
            check = self._evaluate_safeguard(
                safeguard_type="Physical",
                req_id=req_id,
                requirement=requirement,
                config=physical_config.get(req_id, {})
            )
            checks.append(check)
            
        return checks
    
    def audit_technical_safeguards(self) -> List[ComplianceCheck]:
        """Audit technical safeguards compliance."""
        checks = []
        technical_config = self.config.get('technical_safeguards', {})
        
        for req_id, requirement in HIPAASafeguards.TECHNICAL_SAFEGUARDS.items():
            check = self._evaluate_safeguard(
                safeguard_type="Technical",
                req_id=req_id,
                requirement=requirement,
                config=technical_config.get(req_id, {})
            )
            checks.append(check)
            
        return checks
    
    def _evaluate_safeguard(
        self,
        safeguard_type: str,
        req_id: str,
        requirement: Dict[str, Any],
        config: Dict[str, Any]
    ) -> ComplianceCheck:
        """Evaluate a specific safeguard requirement."""
        
        findings = []
        recommendations = []
        compliance_level = ComplianceLevel.NON_COMPLIANT
        
        # Check if safeguard is implemented
        implemented = config.get('implemented', False)
        
        if not implemented:
            findings.append(f"Safeguard {req_id} is not implemented")
            recommendations.append(f"Implement {requirement['name']} controls")
        else:
            # Check specific implementation details
            if safeguard_type == "Administrative":
                compliance_level = self._evaluate_administrative_implementation(
                    req_id, config, findings, recommendations
                )
            elif safeguard_type == "Physical":
                compliance_level = self._evaluate_physical_implementation(
                    req_id, config, findings, recommendations
                )
            elif safeguard_type == "Technical":
                compliance_level = self._evaluate_technical_implementation(
                    req_id, config, findings, recommendations
                )
        
        return ComplianceCheck(
            safeguard_type=safeguard_type,
            requirement_id=req_id,
            requirement_name=requirement['name'],
            description=requirement['description'],
            compliance_level=compliance_level,
            findings=findings,
            recommendations=recommendations,
            evidence=config
        )
    
    def _evaluate_administrative_implementation(
        self,
        req_id: str,
        config: Dict[str, Any],
        findings: List[str],
        recommendations: List[str]
    ) -> ComplianceLevel:
        """Evaluate administrative safeguard implementation."""
        
        if req_id == "164.308(a)(1)":  # Security Officer
            security_officer = config.get('security_officer')
            if not security_officer:
                findings.append("No security officer assigned")
                recommendations.append("Assign a dedicated security officer")
                return ComplianceLevel.NON_COMPLIANT
            
        elif req_id == "164.308(a)(3)":  # Workforce Training
            training_program = config.get('training_program', {})
            if not training_program.get('implemented'):
                findings.append("No workforce training program")
                recommendations.append("Implement comprehensive workforce training")
                return ComplianceLevel.NON_COMPLIANT
            
            if not training_program.get('documented'):
                findings.append("Training program not properly documented")
                recommendations.append("Document training procedures and completion")
                return ComplianceLevel.PARTIALLY_COMPLIANT
        
        elif req_id == "164.308(a)(4)":  # Information Access Management
            access_mgmt = config.get('access_management', {})
            if not access_mgmt.get('procedures_documented'):
                findings.append("Access management procedures not documented")
                recommendations.append("Document access granting and removal procedures")
                return ComplianceLevel.PARTIALLY_COMPLIANT
        
        elif req_id == "164.308(a)(6)":  # Security Incident Procedures
            incident_procedures = config.get('incident_procedures', {})
            if not incident_procedures.get('documented'):
                findings.append("Security incident procedures not documented")
                recommendations.append("Document incident response procedures")
                return ComplianceLevel.NON_COMPLIANT
        
        elif req_id == "164.308(a)(7)":  # Contingency Plan
            contingency = config.get('contingency_plan', {})
            if not contingency.get('plan_exists'):
                findings.append("No contingency plan exists")
                recommendations.append("Develop and document contingency plan")
                return ComplianceLevel.NON_COMPLIANT
            
            if not contingency.get('tested'):
                findings.append("Contingency plan not tested")
                recommendations.append("Regularly test contingency plan")
                return ComplianceLevel.PARTIALLY_COMPLIANT
        
        return ComplianceLevel.COMPLIANT
    
    def _evaluate_physical_implementation(
        self,
        req_id: str,
        config: Dict[str, Any],
        findings: List[str],
        recommendations: List[str]
    ) -> ComplianceLevel:
        """Evaluate physical safeguard implementation."""
        
        if req_id == "164.310(a)(1)":  # Facility Access Controls
            facility_controls = config.get('facility_controls', {})
            if not facility_controls.get('access_controls'):
                findings.append("No facility access controls implemented")
                recommendations.append("Implement physical access controls")
                return ComplianceLevel.NON_COMPLIANT
        
        elif req_id == "164.310(a)(2)":  # Workstation Use
            workstation_policy = config.get('workstation_policy', {})
            if not workstation_policy.get('documented'):
                findings.append("Workstation use policies not documented")
                recommendations.append("Document workstation use policies")
                return ComplianceLevel.NON_COMPLIANT
        
        elif req_id == "164.310(c)":  # Device and Media Controls
            media_controls = config.get('media_controls', {})
            if not media_controls.get('disposal_procedures'):
                findings.append("No secure media disposal procedures")
                recommendations.append("Implement secure media disposal procedures")
                return ComplianceLevel.PARTIALLY_COMPLIANT
        
        return ComplianceLevel.COMPLIANT
    
    def _evaluate_technical_implementation(
        self,
        req_id: str,
        config: Dict[str, Any],
        findings: List[str],
        recommendations: List[str]
    ) -> ComplianceLevel:
        """Evaluate technical safeguard implementation."""
        
        if req_id == "164.312(a)(1)":  # Access Control
            access_control = config.get('access_control', {})
            if not access_control.get('unique_user_identification'):
                findings.append("No unique user identification system")
                recommendations.append("Implement unique user identification")
                return ComplianceLevel.NON_COMPLIANT
            
            if not access_control.get('emergency_access'):
                findings.append("No emergency access procedures")
                recommendations.append("Implement emergency access procedures")
                return ComplianceLevel.PARTIALLY_COMPLIANT
        
        elif req_id == "164.312(b)":  # Audit Controls
            audit_controls = config.get('audit_controls', {})
            if not audit_controls.get('audit_logs_enabled'):
                findings.append("Audit logging not enabled")
                recommendations.append("Enable comprehensive audit logging")
                return ComplianceLevel.NON_COMPLIANT
            
            if not audit_controls.get('log_monitoring'):
                findings.append("Audit logs not monitored")
                recommendations.append("Implement audit log monitoring")
                return ComplianceLevel.PARTIALLY_COMPLIANT
        
        elif req_id == "164.312(c)(1)":  # Integrity
            integrity_controls = config.get('integrity_controls', {})
            if not integrity_controls.get('checksums_enabled'):
                findings.append("No integrity verification mechanisms")
                recommendations.append("Implement data integrity checks")
                return ComplianceLevel.PARTIALLY_COMPLIANT
        
        elif req_id == "164.312(d)":  # Person or Entity Authentication
            authentication = config.get('authentication', {})
            if not authentication.get('strong_authentication'):
                findings.append("No strong authentication implemented")
                recommendations.append("Implement multi-factor authentication")
                return ComplianceLevel.NON_COMPLIANT
        
        elif req_id == "164.312(e)(1)":  # Transmission Security
            transmission = config.get('transmission_security', {})
            if not transmission.get('encryption_in_transit'):
                findings.append("No encryption in transit")
                recommendations.append("Implement encryption for data transmission")
                return ComplianceLevel.NON_COMPLIANT
        
        return ComplianceLevel.COMPLIANT
    
    def run_comprehensive_audit(self) -> List[ComplianceCheck]:
        """Run comprehensive HIPAA compliance audit."""
        all_checks = []
        
        # Audit all safeguard types
        all_checks.extend(self.audit_administrative_safeguards())
        all_checks.extend(self.audit_physical_safeguards()) 
        all_checks.extend(self.audit_technical_safeguards())
        
        self.compliance_checks = all_checks
        return all_checks
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        
        # Calculate compliance statistics
        total_checks = len(self.compliance_checks)
        compliant = len([c for c in self.compliance_checks if c.compliance_level == ComplianceLevel.COMPLIANT])
        non_compliant = len([c for c in self.compliance_checks if c.compliance_level == ComplianceLevel.NON_COMPLIANT])
        partially_compliant = len([c for c in self.compliance_checks if c.compliance_level == ComplianceLevel.PARTIALLY_COMPLIANT])
        
        compliance_percentage = (compliant / total_checks * 100) if total_checks > 0 else 0
        
        # Group checks by safeguard type
        admin_checks = [c for c in self.compliance_checks if c.safeguard_type == "Administrative"]
        physical_checks = [c for c in self.compliance_checks if c.safeguard_type == "Physical"]
        technical_checks = [c for c in self.compliance_checks if c.safeguard_type == "Technical"]
        
        report = {
            'audit_metadata': {
                'timestamp': self.audit_timestamp.isoformat(),
                'auditor': 'Clinical Data Platform HIPAA Auditor',
                'scope': 'Comprehensive HIPAA Security Rule Audit'
            },
            'executive_summary': {
                'overall_compliance_percentage': round(compliance_percentage, 2),
                'total_requirements_checked': total_checks,
                'compliant_requirements': compliant,
                'non_compliant_requirements': non_compliant,
                'partially_compliant_requirements': partially_compliant,
                'overall_status': self._determine_overall_status(compliance_percentage)
            },
            'safeguard_breakdown': {
                'administrative_safeguards': {
                    'total': len(admin_checks),
                    'compliant': len([c for c in admin_checks if c.compliance_level == ComplianceLevel.COMPLIANT]),
                    'non_compliant': len([c for c in admin_checks if c.compliance_level == ComplianceLevel.NON_COMPLIANT]),
                    'checks': [asdict(check) for check in admin_checks]
                },
                'physical_safeguards': {
                    'total': len(physical_checks),
                    'compliant': len([c for c in physical_checks if c.compliance_level == ComplianceLevel.COMPLIANT]),
                    'non_compliant': len([c for c in physical_checks if c.compliance_level == ComplianceLevel.NON_COMPLIANT]),
                    'checks': [asdict(check) for check in physical_checks]
                },
                'technical_safeguards': {
                    'total': len(technical_checks),
                    'compliant': len([c for c in technical_checks if c.compliance_level == ComplianceLevel.COMPLIANT]),
                    'non_compliant': len([c for c in technical_checks if c.compliance_level == ComplianceLevel.NON_COMPLIANT]),
                    'checks': [asdict(check) for check in technical_checks]
                }
            },
            'priority_recommendations': self._get_priority_recommendations(),
            'detailed_findings': [
                {
                    **asdict(check),
                    'compliance_level': check.compliance_level.value
                }
                for check in self.compliance_checks
            ]
        }
        
        return report
    
    def _determine_overall_status(self, compliance_percentage: float) -> str:
        """Determine overall compliance status."""
        if compliance_percentage >= 95:
            return "Highly Compliant"
        elif compliance_percentage >= 80:
            return "Mostly Compliant"
        elif compliance_percentage >= 60:
            return "Partially Compliant"
        else:
            return "Non-Compliant"
    
    def _get_priority_recommendations(self) -> List[str]:
        """Get priority recommendations for compliance improvement."""
        non_compliant_checks = [
            c for c in self.compliance_checks 
            if c.compliance_level == ComplianceLevel.NON_COMPLIANT
        ]
        
        priority_recommendations = []
        for check in non_compliant_checks[:5]:  # Top 5 priority items
            priority_recommendations.extend(check.recommendations)
        
        return list(set(priority_recommendations))  # Remove duplicates


def main():
    """CLI entry point for HIPAA auditor."""
    parser = argparse.ArgumentParser(description="Perform HIPAA compliance audit")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--audit-level", choices=['basic', 'comprehensive'], default='comprehensive')
    parser.add_argument("--output-format", choices=['json', 'html'], default='json')
    parser.add_argument("--output-file", help="Output file path")
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Initialize auditor
    auditor = HIPAAAuditor(config)
    
    # Run audit
    if args.audit_level == 'comprehensive':
        auditor.run_comprehensive_audit()
    else:
        # Run basic audit (subset of checks)
        auditor.audit_technical_safeguards()
    
    # Generate report
    report = auditor.generate_compliance_report()
    
    # Output report
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"HIPAA audit report written to {args.output_file}")
    else:
        print(json.dumps(report, indent=2))
    
    # Exit with error code if non-compliant
    non_compliant_count = report['executive_summary']['non_compliant_requirements']
    if non_compliant_count > 0:
        print(f"\nWARNING: {non_compliant_count} HIPAA requirements are non-compliant!", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()