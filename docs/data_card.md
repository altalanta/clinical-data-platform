# Clinical Data Platform - Data Card

## Overview
This document describes the data handling, privacy, and consent policies for the clinical data platform.

## De-identification Policy (HIPAA Safe Harbor)

### HIPAA Safe Harbor Method
The platform implements the HIPAA Safe Harbor method for de-identification, removing the following 18 identifiers:

1. **Names** - All personal names and names of relatives, employers, or household members
2. **Geographic Subdivisions** - ZIP codes, except first 3 digits if area > 20,000 people
3. **Dates** - Birth dates, admission/discharge dates (except year), death dates
4. **Telephone Numbers** - All phone and fax numbers
5. **Vehicle Identifiers** - License plate numbers, VINs, serial numbers
6. **Device Identifiers** - Serial numbers of medical devices
7. **Web URLs** - Any web-based identifiers
8. **IP Addresses** - Internet protocol addresses  
9. **Biometric Identifiers** - Fingerprints, voiceprints, iris scans
10. **Facial Photographs** - Full face photos and comparable images
11. **SSN** - Social Security Numbers
12. **Medical Record Numbers** - Institution-assigned medical record numbers
13. **Health Plan Numbers** - Health insurance beneficiary numbers
14. **Account Numbers** - Medical and financial account numbers
15. **Certificate/License Numbers** - Professional license numbers
16. **Email Addresses** - Electronic mail addresses
17. **Other Unique Identifiers** - Any other unique identifying numbers
18. **Any Other Identifiers** - Additional identifiers not listed above

### De-identification Process
```python
# Example de-identification pipeline
def deidentify_record(record):
    """Apply Safe Harbor de-identification."""
    # Remove direct identifiers
    record.pop('ssn', None)
    record.pop('phone', None)
    record.pop('email', None)
    
    # Generalize dates (keep year only)
    if 'date_of_birth' in record:
        record['birth_year'] = record['date_of_birth'].year
        del record['date_of_birth']
    
    # Generalize geography (3-digit ZIP)
    if 'zip_code' in record and len(record['zip_code']) == 5:
        record['zip_3digit'] = record['zip_code'][:3] + "XX"
        del record['zip_code']
    
    # Replace ID with research ID
    record['study_id'] = generate_study_id(record['medical_record_number'])
    del record['medical_record_number']
    
    return record
```

## Consent Management

### Consent Types
The platform recognizes the following consent categories:

```python
from enum import Enum

class ConsentType(str, Enum):
    """Patient consent categories."""
    RESEARCH_GENERAL = "research_general"     # General research participation
    RESEARCH_GENETIC = "research_genetic"     # Genetic/genomic research
    DATA_SHARING = "data_sharing"            # Sharing with external researchers
    COMMERCIAL_USE = "commercial_use"        # Commercial/industry research
    CONTACT_FUTURE = "contact_future"        # Future study contact
    WITHDRAWN = "withdrawn"                  # Consent withdrawn
```

### Consent Schema
```json
{
  "patient_id": "string",
  "consent_date": "2024-01-15T10:30:00Z",
  "consent_version": "2.1",
  "consents": {
    "research_general": {
      "granted": true,
      "date": "2024-01-15T10:30:00Z",
      "expiry": "2029-01-15T10:30:00Z"
    },
    "research_genetic": {
      "granted": false,
      "date": "2024-01-15T10:30:00Z"
    },
    "data_sharing": {
      "granted": true,
      "date": "2024-01-15T10:30:00Z",
      "restrictions": ["no_international_transfer"]
    }
  },
  "withdrawal_date": null,
  "data_retention_years": 10
}
```

### Consent Enforcement
- **Query-time filtering**: All data queries automatically filter by consent status
- **Audit logging**: All consent-related actions are logged immutably
- **Withdrawal processing**: Patient data is anonymized within 30 days of withdrawal
- **Expiry handling**: Expired consents trigger data review and potential deletion

## Data Provenance

### Data Sources
1. **Electronic Health Records (EHR)**
   - Source: Epic/Cerner systems
   - De-identification: Automated Safe Harbor pipeline
   - Update frequency: Daily batch processing
   - Retention: 10 years post-study completion

2. **Clinical Trials Data**
   - Source: REDCap/EDC systems  
   - De-identification: Manual review + automated
   - Update frequency: Real-time via API
   - Retention: Per protocol requirements (typically 15+ years)

3. **Laboratory Results**
   - Source: Laboratory Information Systems (LIS)
   - De-identification: Automated with manual QC
   - Update frequency: Near real-time
   - Retention: 7 years minimum

4. **Imaging Data**
   - Source: PACS systems
   - De-identification: DICOM tag removal + face anonymization
   - Update frequency: On-demand
   - Retention: Per institutional policy

### Data Lineage Tracking
```yaml
# OpenLineage metadata example
dataset:
  namespace: clinical-platform
  name: patients_deidentified
  facets:
    schema:
      fields:
        - name: study_id
          type: string
          description: De-identified patient identifier
        - name: birth_year
          type: integer
          description: Year of birth (HIPAA Safe Harbor)
    dataSource:
      name: epic_ehr
      uri: jdbc:oracle:thin:@//epic-prod:1521/EPIC
    transformation:
      type: deidentification
      method: hipaa_safe_harbor
      version: "2.1"
      applied_at: "2024-01-15T14:30:00Z"
```

## Security and Access Controls

### Role-Based Access
- **Researchers**: De-identified data only, query access
- **Clinical Staff**: Identified data with audit logging
- **Data Scientists**: Aggregated/statistical data only
- **External Partners**: Specific datasets per data use agreement

### Data Classification
```python
class DataClassification(str, Enum):
    """Data sensitivity levels."""
    PUBLIC = "public"                    # No restrictions
    INTERNAL = "internal"               # Organization only
    CONFIDENTIAL = "confidential"       # Need-to-know basis
    RESTRICTED = "restricted"           # PHI/highly sensitive
```

### Audit Requirements
- All data access logged with user, timestamp, query details
- Quarterly access reviews for role appropriateness
- Annual consent status audits
- Breach notification procedures per HIPAA/GDPR

## Data Retention and Deletion

### Retention Schedule
| Data Type | Retention Period | Trigger |
|-----------|------------------|---------|
| Clinical Trial Data | 15+ years | Protocol completion |
| EHR Data | 10 years | Last patient contact |
| Research Data | 7 years | Publication + 7 years |
| Consent Records | Permanent | N/A |
| Audit Logs | 7 years | Creation date |

### Deletion Process
1. **Automated screening**: Monthly job identifies expired data
2. **Manual review**: Data steward approves deletion list
3. **Secure deletion**: Multi-pass overwrite of storage media
4. **Certificate of destruction**: Documented proof of deletion
5. **Audit trail**: Permanent record of what was deleted when

## Quality Assurance

### Data Validation
- **Schema validation**: All incoming data validated against defined schemas
- **PHI detection**: Automated scanning for residual identifiers
- **Consistency checks**: Cross-reference validation between datasets
- **Manual QC**: Statistical review of de-identification quality

### Monitoring and Alerts
- Real-time alerts for potential PHI exposure
- Data quality dashboards for researchers
- Consent status monitoring
- Access pattern anomaly detection

## Compliance Framework

### Applicable Regulations
- **HIPAA**: Health Insurance Portability and Accountability Act
- **GDPR**: General Data Protection Regulation (EU patients)
- **PIPEDA**: Personal Information Protection (Canadian patients)
- **21 CFR Part 11**: FDA Electronic Records requirements
- **ICH GCP**: Good Clinical Practice guidelines

### Certification and Audits
- Annual HIPAA compliance audit
- SOC 2 Type II certification
- Regular penetration testing
- Third-party security assessments

## Contact Information

**Data Protection Officer**: privacy@clinical-platform.com  
**Research Data Office**: research-data@clinical-platform.com  
**Security Team**: security@clinical-platform.com  
**Compliance Hotline**: +1-800-COMPLY-1

---
*Last Updated: January 2024*  
*Version: 2.1*  
*Next Review: July 2024*