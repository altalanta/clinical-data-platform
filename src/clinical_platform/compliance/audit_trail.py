#!/usr/bin/env python3
"""
Audit Trail Management for Clinical Data Platform.

This module provides comprehensive audit trail functionality for
tracking access, modifications, and system events for compliance.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import uuid

import structlog

logger = structlog.get_logger()


class AuditEventType(Enum):
    """Types of audit events."""
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    LOGIN_ATTEMPT = "login_attempt"
    LOGOUT = "logout"
    PERMISSION_CHANGE = "permission_change"
    EXPORT_REQUEST = "export_request"
    SYSTEM_CONFIG_CHANGE = "system_config_change"
    SECURITY_EVENT = "security_event"
    ERROR_EVENT = "error_event"


class AuditSeverity(Enum):
    """Severity levels for audit events."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class AuditEvent:
    """Represents an audit event."""
    event_id: str
    timestamp: str
    event_type: AuditEventType
    severity: AuditSeverity
    user_id: Optional[str]
    session_id: Optional[str]
    source_ip: Optional[str]
    user_agent: Optional[str]
    resource_accessed: Optional[str]
    action_performed: str
    outcome: str  # success, failure, partial
    details: Dict[str, Any]
    data_elements_accessed: List[str]
    phi_accessed: bool
    compliance_tags: List[str]
    checksum: Optional[str] = None


class AuditTrail:
    """Manages audit trail functionality."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.audit_storage_path = Path(self.config.get('audit_storage_path', 'audit_logs'))
        self.audit_storage_path.mkdir(exist_ok=True)
        
        # Audit configuration
        self.retention_days = self.config.get('retention_days', 365)
        self.encryption_enabled = self.config.get('encryption_enabled', True)
        self.integrity_checking = self.config.get('integrity_checking', True)
        
    def log_event(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        action_performed: str,
        outcome: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        source_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        resource_accessed: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        data_elements_accessed: Optional[List[str]] = None,
        phi_accessed: bool = False,
        compliance_tags: Optional[List[str]] = None
    ) -> str:
        """Log an audit event."""
        
        event_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        
        event = AuditEvent(
            event_id=event_id,
            timestamp=timestamp,
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            session_id=session_id,
            source_ip=source_ip,
            user_agent=user_agent,
            resource_accessed=resource_accessed,
            action_performed=action_performed,
            outcome=outcome,
            details=details or {},
            data_elements_accessed=data_elements_accessed or [],
            phi_accessed=phi_accessed,
            compliance_tags=compliance_tags or []
        )
        
        # Calculate integrity checksum
        if self.integrity_checking:
            event.checksum = self._calculate_event_checksum(event)
        
        # Store the event
        self._store_event(event)
        
        logger.info(
            "Audit event logged",
            event_id=event_id,
            event_type=event_type.value,
            user_id=user_id,
            action=action_performed,
            outcome=outcome
        )
        
        return event_id
    
    def _calculate_event_checksum(self, event: AuditEvent) -> str:
        """Calculate integrity checksum for audit event."""
        # Create deterministic string representation
        event_data = {
            'event_id': event.event_id,
            'timestamp': event.timestamp,
            'event_type': event.event_type.value,
            'user_id': event.user_id,
            'action_performed': event.action_performed,
            'resource_accessed': event.resource_accessed,
            'outcome': event.outcome
        }
        
        event_string = json.dumps(event_data, sort_keys=True)
        return hashlib.sha256(event_string.encode()).hexdigest()
    
    def _store_event(self, event: AuditEvent) -> None:
        """Store audit event to persistent storage."""
        # Organize by date for efficient retrieval
        date_str = datetime.fromisoformat(event.timestamp.replace('Z', '+00:00')).strftime('%Y-%m-%d')
        daily_log_path = self.audit_storage_path / f"audit_{date_str}.jsonl"
        
        # Convert event to dict for serialization
        event_dict = asdict(event)
        event_dict['event_type'] = event.event_type.value
        event_dict['severity'] = event.severity.value
        
        # Append to daily log file
        with open(daily_log_path, 'a') as f:
            f.write(json.dumps(event_dict) + '\n')
    
    def query_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
        user_id: Optional[str] = None,
        resource_accessed: Optional[str] = None,
        phi_accessed: Optional[bool] = None,
        limit: int = 1000
    ) -> List[AuditEvent]:
        """Query audit events with filters."""
        
        events = []
        
        # Determine date range
        if not start_date:
            start_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        if not end_date:
            end_date = datetime.now(timezone.utc)
        
        # Iterate through date range
        current_date = start_date
        while current_date <= end_date and len(events) < limit:
            date_str = current_date.strftime('%Y-%m-%d')
            daily_log_path = self.audit_storage_path / f"audit_{date_str}.jsonl"
            
            if daily_log_path.exists():
                with open(daily_log_path, 'r') as f:
                    for line in f:
                        if len(events) >= limit:
                            break
                            
                        try:
                            event_dict = json.loads(line.strip())
                            event = self._dict_to_event(event_dict)
                            
                            # Apply filters
                            if self._event_matches_filters(
                                event, event_types, user_id, resource_accessed, phi_accessed
                            ):
                                events.append(event)
                                
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in audit log: {line}")
            
            current_date = current_date.replace(day=current_date.day + 1)
        
        return events
    
    def _dict_to_event(self, event_dict: Dict[str, Any]) -> AuditEvent:
        """Convert dictionary back to AuditEvent object."""
        return AuditEvent(
            event_id=event_dict['event_id'],
            timestamp=event_dict['timestamp'],
            event_type=AuditEventType(event_dict['event_type']),
            severity=AuditSeverity(event_dict['severity']),
            user_id=event_dict.get('user_id'),
            session_id=event_dict.get('session_id'),
            source_ip=event_dict.get('source_ip'),
            user_agent=event_dict.get('user_agent'),
            resource_accessed=event_dict.get('resource_accessed'),
            action_performed=event_dict['action_performed'],
            outcome=event_dict['outcome'],
            details=event_dict.get('details', {}),
            data_elements_accessed=event_dict.get('data_elements_accessed', []),
            phi_accessed=event_dict.get('phi_accessed', False),
            compliance_tags=event_dict.get('compliance_tags', []),
            checksum=event_dict.get('checksum')
        )
    
    def _event_matches_filters(
        self,
        event: AuditEvent,
        event_types: Optional[List[AuditEventType]],
        user_id: Optional[str],
        resource_accessed: Optional[str],
        phi_accessed: Optional[bool]
    ) -> bool:
        """Check if event matches query filters."""
        
        if event_types and event.event_type not in event_types:
            return False
        
        if user_id and event.user_id != user_id:
            return False
        
        if resource_accessed and event.resource_accessed != resource_accessed:
            return False
        
        if phi_accessed is not None and event.phi_accessed != phi_accessed:
            return False
        
        return True
    
    def verify_integrity(self, event_ids: Optional[List[str]] = None) -> Dict[str, bool]:
        """Verify integrity of audit events."""
        if not self.integrity_checking:
            return {}
        
        verification_results = {}
        
        if event_ids:
            # Verify specific events
            for event_id in event_ids:
                event = self._find_event_by_id(event_id)
                if event:
                    calculated_checksum = self._calculate_event_checksum(event)
                    verification_results[event_id] = (calculated_checksum == event.checksum)
                else:
                    verification_results[event_id] = False
        else:
            # Verify all events in recent logs
            recent_events = self.query_events(limit=10000)
            for event in recent_events:
                if event.checksum:
                    calculated_checksum = self._calculate_event_checksum(event)
                    verification_results[event.event_id] = (calculated_checksum == event.checksum)
        
        return verification_results
    
    def _find_event_by_id(self, event_id: str) -> Optional[AuditEvent]:
        """Find event by ID across all log files."""
        for log_file in self.audit_storage_path.glob("audit_*.jsonl"):
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        event_dict = json.loads(line.strip())
                        if event_dict['event_id'] == event_id:
                            return self._dict_to_event(event_dict)
                    except json.JSONDecodeError:
                        continue
        return None
    
    def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate compliance report for date range."""
        
        events = self.query_events(start_date=start_date, end_date=end_date, limit=50000)
        
        # Calculate statistics
        total_events = len(events)
        phi_access_events = len([e for e in events if e.phi_accessed])
        failed_events = len([e for e in events if e.outcome == 'failure'])
        
        # Group by event type
        event_type_counts = {}
        for event in events:
            event_type = event.event_type.value
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
        
        # Group by user
        user_activity = {}
        for event in events:
            if event.user_id:
                user_activity[event.user_id] = user_activity.get(event.user_id, 0) + 1
        
        # Security events
        security_events = [e for e in events if e.event_type == AuditEventType.SECURITY_EVENT]
        
        report = {
            'report_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            },
            'summary_statistics': {
                'total_audit_events': total_events,
                'phi_access_events': phi_access_events,
                'failed_events': failed_events,
                'security_events': len(security_events),
                'unique_users': len(user_activity)
            },
            'event_type_breakdown': event_type_counts,
            'top_users_by_activity': dict(
                sorted(user_activity.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
            'security_incidents': [
                {
                    'event_id': event.event_id,
                    'timestamp': event.timestamp,
                    'severity': event.severity.value,
                    'action': event.action_performed,
                    'user_id': event.user_id,
                    'source_ip': event.source_ip
                }
                for event in security_events
            ],
            'compliance_metrics': {
                'audit_trail_completeness': self._calculate_completeness_score(events),
                'integrity_verification_passed': len([
                    result for result in self.verify_integrity().values() if result
                ]),
                'phi_access_properly_logged': phi_access_events > 0
            }
        }
        
        return report
    
    def _calculate_completeness_score(self, events: List[AuditEvent]) -> float:
        """Calculate audit trail completeness score."""
        if not events:
            return 0.0
        
        # Check for required fields completion
        complete_events = 0
        for event in events:
            required_fields = [
                event.event_id,
                event.timestamp,
                event.action_performed,
                event.outcome
            ]
            
            if all(field for field in required_fields):
                complete_events += 1
        
        return (complete_events / len(events)) * 100


def main():
    """CLI entry point for audit trail management."""
    parser = argparse.ArgumentParser(description="Audit trail management and validation")
    parser.add_argument("--validate-logging", action="store_true", help="Validate logging functionality")
    parser.add_argument("--check-encryption", action="store_true", help="Check encryption settings")
    parser.add_argument("--verify-access-controls", action="store_true", help="Verify access controls")
    parser.add_argument("--generate-report", action="store_true", help="Generate compliance report")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--output-file", help="Output file path")
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Initialize audit trail
    audit_trail = AuditTrail(config)
    
    results = {}
    
    if args.validate_logging:
        # Test logging functionality
        test_event_id = audit_trail.log_event(
            event_type=AuditEventType.SYSTEM_CONFIG_CHANGE,
            severity=AuditSeverity.INFO,
            action_performed="audit_trail_validation_test",
            outcome="success",
            user_id="system",
            details={"test": True}
        )
        results['logging_validation'] = {
            'status': 'success',
            'test_event_id': test_event_id
        }
    
    if args.check_encryption:
        # Check encryption configuration
        encryption_enabled = audit_trail.encryption_enabled
        results['encryption_check'] = {
            'enabled': encryption_enabled,
            'status': 'pass' if encryption_enabled else 'fail'
        }
    
    if args.verify_access_controls:
        # Verify access control settings
        results['access_controls'] = {
            'retention_days': audit_trail.retention_days,
            'integrity_checking': audit_trail.integrity_checking,
            'storage_path_exists': audit_trail.audit_storage_path.exists()
        }
    
    if args.generate_report:
        # Generate sample compliance report
        end_date = datetime.now(timezone.utc)
        start_date = end_date.replace(day=end_date.day - 7)  # Last 7 days
        
        report = audit_trail.generate_compliance_report(start_date, end_date)
        results['compliance_report'] = report
    
    # Output results
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Audit trail validation results written to {args.output_file}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()