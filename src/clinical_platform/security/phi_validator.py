#!/usr/bin/env python3
"""
PHI Validator for Clinical Data Platform.

This module validates PHI handling, anonymization processes, and read-only mode
compliance for protected health information.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import structlog

logger = structlog.get_logger()


class ValidationResult(Enum):
    """Validation result status."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"


@dataclass
class ValidationCheck:
    """Represents a validation check result."""
    check_name: str
    status: ValidationResult
    message: str
    details: Optional[Dict[str, Any]] = None


class PHIValidator:
    """Validates PHI handling and compliance."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.checks: List[ValidationCheck] = []
        
    def validate_readonly_mode(self) -> ValidationCheck:
        """Validate that PHI data is accessed in read-only mode."""
        try:
            # Check for read-only configuration
            readonly_config = self.config.get('phi_readonly_mode', False)
            
            if not readonly_config:
                return ValidationCheck(
                    check_name="phi_readonly_mode",
                    status=ValidationResult.FAIL,
                    message="PHI read-only mode is not enabled in configuration",
                    details={"required": True, "current": readonly_config}
                )
            
            # Check database connections for read-only settings
            db_config = self.config.get('database', {})
            readonly_connections = db_config.get('readonly_replicas', [])
            
            if not readonly_connections:
                return ValidationCheck(
                    check_name="phi_readonly_mode",
                    status=ValidationResult.WARNING,
                    message="No read-only database replicas configured for PHI access",
                    details={"readonly_replicas": readonly_connections}
                )
            
            return ValidationCheck(
                check_name="phi_readonly_mode",
                status=ValidationResult.PASS,
                message="PHI read-only mode is properly configured"
            )
            
        except Exception as e:
            return ValidationCheck(
                check_name="phi_readonly_mode",
                status=ValidationResult.FAIL,
                message=f"Error validating read-only mode: {e}"
            )
    
    def validate_anonymization(self) -> ValidationCheck:
        """Validate anonymization processes and configuration."""
        try:
            anonymization_config = self.config.get('anonymization', {})
            
            required_settings = [
                'enabled',
                'methods',
                'identifier_fields',
                'quasi_identifier_fields'
            ]
            
            missing_settings = [
                setting for setting in required_settings
                if setting not in anonymization_config
            ]
            
            if missing_settings:
                return ValidationCheck(
                    check_name="anonymization_config",
                    status=ValidationResult.FAIL,
                    message=f"Missing anonymization settings: {missing_settings}",
                    details={"missing": missing_settings, "required": required_settings}
                )
            
            # Validate anonymization methods
            methods = anonymization_config.get('methods', {})
            supported_methods = ['k_anonymity', 'l_diversity', 'differential_privacy', 'masking']
            
            configured_methods = list(methods.keys())
            unsupported_methods = [m for m in configured_methods if m not in supported_methods]
            
            if unsupported_methods:
                return ValidationCheck(
                    check_name="anonymization_config",
                    status=ValidationResult.WARNING,
                    message=f"Unsupported anonymization methods: {unsupported_methods}",
                    details={"unsupported": unsupported_methods, "supported": supported_methods}
                )
            
            return ValidationCheck(
                check_name="anonymization_config",
                status=ValidationResult.PASS,
                message="Anonymization configuration is valid"
            )
            
        except Exception as e:
            return ValidationCheck(
                check_name="anonymization_config",
                status=ValidationResult.FAIL,
                message=f"Error validating anonymization: {e}"
            )
    
    def validate_data_access_controls(self) -> ValidationCheck:
        """Validate data access controls and permissions."""
        try:
            access_config = self.config.get('access_control', {})
            
            # Check role-based access control
            rbac_enabled = access_config.get('rbac_enabled', False)
            if not rbac_enabled:
                return ValidationCheck(
                    check_name="data_access_controls",
                    status=ValidationResult.FAIL,
                    message="Role-based access control (RBAC) is not enabled",
                    details={"rbac_enabled": rbac_enabled}
                )
            
            # Check for defined roles and permissions
            roles = access_config.get('roles', {})
            if not roles:
                return ValidationCheck(
                    check_name="data_access_controls",
                    status=ValidationResult.FAIL,
                    message="No access control roles defined",
                    details={"roles": roles}
                )
            
            # Validate minimum required roles
            required_roles = ['admin', 'researcher', 'viewer']
            defined_roles = list(roles.keys())
            missing_roles = [role for role in required_roles if role not in defined_roles]
            
            if missing_roles:
                return ValidationCheck(
                    check_name="data_access_controls", 
                    status=ValidationResult.WARNING,
                    message=f"Recommended roles not defined: {missing_roles}",
                    details={"missing_roles": missing_roles, "defined_roles": defined_roles}
                )
            
            return ValidationCheck(
                check_name="data_access_controls",
                status=ValidationResult.PASS,
                message="Data access controls are properly configured"
            )
            
        except Exception as e:
            return ValidationCheck(
                check_name="data_access_controls",
                status=ValidationResult.FAIL,
                message=f"Error validating access controls: {e}"
            )
    
    def validate_audit_logging(self) -> ValidationCheck:
        """Validate audit logging configuration."""
        try:
            logging_config = self.config.get('audit_logging', {})
            
            # Check if audit logging is enabled
            audit_enabled = logging_config.get('enabled', False)
            if not audit_enabled:
                return ValidationCheck(
                    check_name="audit_logging",
                    status=ValidationResult.FAIL,
                    message="Audit logging is not enabled",
                    details={"enabled": audit_enabled}
                )
            
            # Check for required audit events
            required_events = [
                'data_access',
                'data_modification',
                'login_attempts',
                'permission_changes',
                'export_requests'
            ]
            
            logged_events = logging_config.get('events', [])
            missing_events = [event for event in required_events if event not in logged_events]
            
            if missing_events:
                return ValidationCheck(
                    check_name="audit_logging",
                    status=ValidationResult.WARNING,
                    message=f"Some audit events not configured: {missing_events}",
                    details={"missing_events": missing_events, "logged_events": logged_events}
                )
            
            # Check log retention policy
            retention_days = logging_config.get('retention_days', 0)
            if retention_days < 30:
                return ValidationCheck(
                    check_name="audit_logging",
                    status=ValidationResult.WARNING,
                    message=f"Audit log retention period is too short: {retention_days} days",
                    details={"retention_days": retention_days, "recommended_minimum": 30}
                )
            
            return ValidationCheck(
                check_name="audit_logging",
                status=ValidationResult.PASS,
                message="Audit logging is properly configured"
            )
            
        except Exception as e:
            return ValidationCheck(
                check_name="audit_logging",
                status=ValidationResult.FAIL,
                message=f"Error validating audit logging: {e}"
            )
    
    def validate_encryption(self) -> ValidationCheck:
        """Validate encryption configuration."""
        try:
            encryption_config = self.config.get('encryption', {})
            
            # Check encryption at rest
            at_rest_enabled = encryption_config.get('at_rest', {}).get('enabled', False)
            if not at_rest_enabled:
                return ValidationCheck(
                    check_name="encryption",
                    status=ValidationResult.FAIL,
                    message="Encryption at rest is not enabled",
                    details={"at_rest_enabled": at_rest_enabled}
                )
            
            # Check encryption in transit
            in_transit_enabled = encryption_config.get('in_transit', {}).get('enabled', False)
            if not in_transit_enabled:
                return ValidationCheck(
                    check_name="encryption",
                    status=ValidationResult.FAIL,
                    message="Encryption in transit is not enabled",
                    details={"in_transit_enabled": in_transit_enabled}
                )
            
            # Check encryption algorithms
            at_rest_algorithm = encryption_config.get('at_rest', {}).get('algorithm', '')
            approved_algorithms = ['AES-256', 'AES-256-GCM', 'ChaCha20-Poly1305']
            
            if at_rest_algorithm not in approved_algorithms:
                return ValidationCheck(
                    check_name="encryption",
                    status=ValidationResult.WARNING,
                    message=f"Encryption algorithm not in approved list: {at_rest_algorithm}",
                    details={"current_algorithm": at_rest_algorithm, "approved": approved_algorithms}
                )
            
            return ValidationCheck(
                check_name="encryption",
                status=ValidationResult.PASS,
                message="Encryption is properly configured"
            )
            
        except Exception as e:
            return ValidationCheck(
                check_name="encryption",
                status=ValidationResult.FAIL,
                message=f"Error validating encryption: {e}"
            )
    
    def validate_data_minimization(self) -> ValidationCheck:
        """Validate data minimization practices."""
        try:
            minimization_config = self.config.get('data_minimization', {})
            
            # Check if data minimization is configured
            if not minimization_config:
                return ValidationCheck(
                    check_name="data_minimization",
                    status=ValidationResult.WARNING,
                    message="Data minimization policies not configured",
                    details={"config_present": False}
                )
            
            # Check for field-level controls
            field_controls = minimization_config.get('field_controls', {})
            if not field_controls:
                return ValidationCheck(
                    check_name="data_minimization",
                    status=ValidationResult.WARNING,
                    message="No field-level data minimization controls configured",
                    details={"field_controls": field_controls}
                )
            
            # Check for retention policies
            retention_policy = minimization_config.get('retention_policy', {})
            if not retention_policy:
                return ValidationCheck(
                    check_name="data_minimization",
                    status=ValidationResult.WARNING,
                    message="No data retention policy configured",
                    details={"retention_policy": retention_policy}
                )
            
            return ValidationCheck(
                check_name="data_minimization",
                status=ValidationResult.PASS,
                message="Data minimization practices are configured"
            )
            
        except Exception as e:
            return ValidationCheck(
                check_name="data_minimization",
                status=ValidationResult.FAIL,
                message=f"Error validating data minimization: {e}"
            )
    
    def run_all_validations(self) -> List[ValidationCheck]:
        """Run all PHI validation checks."""
        validation_methods = [
            self.validate_readonly_mode,
            self.validate_anonymization,
            self.validate_data_access_controls,
            self.validate_audit_logging,
            self.validate_encryption,
            self.validate_data_minimization,
        ]
        
        self.checks = []
        for validation_method in validation_methods:
            try:
                check = validation_method()
                self.checks.append(check)
                logger.info(f"Validation check completed: {check.check_name} - {check.status.value}")
            except Exception as e:
                error_check = ValidationCheck(
                    check_name=validation_method.__name__,
                    status=ValidationResult.FAIL,
                    message=f"Exception during validation: {e}"
                )
                self.checks.append(error_check)
                logger.error(f"Validation check failed: {validation_method.__name__}: {e}")
        
        return self.checks
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate validation report."""
        passed = len([c for c in self.checks if c.status == ValidationResult.PASS])
        failed = len([c for c in self.checks if c.status == ValidationResult.FAIL])
        warnings = len([c for c in self.checks if c.status == ValidationResult.WARNING])
        
        report = {
            'summary': {
                'total_checks': len(self.checks),
                'passed': passed,
                'failed': failed,
                'warnings': warnings,
                'overall_status': 'FAIL' if failed > 0 else ('WARNING' if warnings > 0 else 'PASS')
            },
            'checks': [
                {
                    'name': check.check_name,
                    'status': check.status.value,
                    'message': check.message,
                    'details': check.details
                }
                for check in self.checks
            ]
        }
        
        return report


def main():
    """CLI entry point for PHI validator."""
    parser = argparse.ArgumentParser(description="Validate PHI handling and compliance")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--check-readonly-mode", action="store_true", help="Check read-only mode")
    parser.add_argument("--validate-anonymization", action="store_true", help="Validate anonymization")
    parser.add_argument("--output-file", help="Output file path")
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Initialize validator
    validator = PHIValidator(config)
    
    # Run specific validations or all
    if args.check_readonly_mode:
        check = validator.validate_readonly_mode()
        validator.checks = [check]
    elif args.validate_anonymization:
        check = validator.validate_anonymization()
        validator.checks = [check]
    else:
        validator.run_all_validations()
    
    # Generate report
    report = validator.generate_report()
    
    # Output report
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report written to {args.output_file}")
    else:
        print(json.dumps(report, indent=2))
    
    # Exit with error code if any checks failed
    if report['summary']['failed'] > 0:
        print(f"\nERROR: {report['summary']['failed']} validation checks failed!", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()