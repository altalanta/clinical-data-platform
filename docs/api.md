# API Reference

The Clinical Data Platform exposes a RESTful API built with FastAPI, providing programmatic access to data, models, and analytics.

## Authentication

The API uses JWT token-based authentication:

```bash
# Get access token
curl -X POST /auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "user", "password": "pass"}'

# Use token in requests
curl -H "Authorization: Bearer $TOKEN" /api/v1/patients
```

## Base URLs

- **Production**: `https://api.clinical-platform.com`
- **Staging**: `https://staging-api.clinical-platform.com`
- **Local**: `http://localhost:8000`

## Endpoints

### Health & Status

#### `GET /health`

System health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2023-10-01T12:00:00Z",
  "version": "1.0.0",
  "components": {
    "database": "healthy",
    "ml_model": "healthy", 
    "data_validation": "healthy"
  }
}
```

#### `GET /metrics`

Prometheus-compatible metrics endpoint.

**Response:**
```
# HELP api_requests_total Total API requests
# TYPE api_requests_total counter
api_requests_total{method="GET",endpoint="/health"} 1247

# HELP data_quality_score Current data quality score
# TYPE data_quality_score gauge  
data_quality_score 0.95
```

### Data Access

#### `GET /api/v1/patients`

Retrieve patient records with optional filtering.

**Parameters:**
- `limit` (int): Maximum records to return (default: 100, max: 1000)
- `offset` (int): Number of records to skip (default: 0)
- `gender` (str): Filter by gender ("male", "female", "unknown")
- `age_min` (int): Minimum age filter
- `age_max` (int): Maximum age filter

**Example Request:**
```bash
curl "/api/v1/patients?limit=10&gender=female&age_min=18&age_max=65" \
  -H "Authorization: Bearer $TOKEN"
```

**Response:**
```json
{
  "data": [
    {
      "person_id": 12345,
      "gender_concept_id": 8532,
      "gender": "female",
      "year_of_birth": 1985,
      "race_concept_id": 8527,
      "ethnicity_concept_id": 38003564
    }
  ],
  "pagination": {
    "limit": 10,
    "offset": 0,
    "total": 15670,
    "has_next": true
  }
}
```

#### `GET /api/v1/patients/{patient_id}`

Get details for a specific patient.

**Response:**
```json
{
  "person_id": 12345,
  "gender": "female",
  "age": 38,
  "race": "White",
  "ethnicity": "Not Hispanic",
  "visits_count": 23,
  "last_visit_date": "2023-09-15",
  "conditions": [
    {
      "condition_concept_id": 201826,
      "condition_name": "Type 2 diabetes mellitus",
      "condition_start_date": "2020-03-15"
    }
  ]
}
```

#### `POST /api/v1/patients`

Create a new patient record.

!!! warning "Read-Only Mode"
    This endpoint returns 403 Forbidden when `READ_ONLY_MODE=1`

**Request Body:**
```json
{
  "person_id": 98765,
  "gender_concept_id": 8507,
  "year_of_birth": 1975,
  "race_concept_id": 8527,
  "ethnicity_concept_id": 38003564
}
```

**Response:**
```json
{
  "status": "created",
  "person_id": 98765,
  "created_at": "2023-10-01T12:00:00Z"
}
```

### Machine Learning

#### `POST /api/v1/predict/risk`

Generate risk predictions for a patient.

**Request Body:**
```json
{
  "patient_id": 12345,
  "features": {
    "age": 45,
    "gender": "female", 
    "bmi": 28.5,
    "lab_values": {
      "glucose": 120,
      "hemoglobin": 12.5,
      "creatinine": 1.1
    },
    "conditions": ["diabetes", "hypertension"],
    "medications": ["metformin", "lisinopril"]
  }
}
```

**Response:**
```json
{
  "patient_id": 12345,
  "prediction": {
    "risk_score": 0.73,
    "risk_category": "high",
    "confidence": 0.89
  },
  "model": {
    "name": "clinical_risk_v2",
    "version": "2.1.0",
    "training_date": "2023-09-01"
  },
  "feature_importance": {
    "age": 0.25,
    "glucose": 0.20,
    "bmi": 0.18,
    "conditions": 0.15,
    "medications": 0.12,
    "other": 0.10
  },
  "timestamp": "2023-10-01T12:00:00Z"
}
```

#### `GET /api/v1/models`

List available ML models.

**Response:**
```json
{
  "models": [
    {
      "name": "clinical_risk_v2",
      "version": "2.1.0", 
      "description": "Clinical risk prediction model",
      "accuracy": 0.87,
      "training_date": "2023-09-01",
      "status": "active"
    },
    {
      "name": "readmission_predictor",
      "version": "1.3.2",
      "description": "30-day readmission risk",
      "accuracy": 0.82,
      "training_date": "2023-08-15", 
      "status": "active"
    }
  ]
}
```

### Data Quality

#### `GET /api/v1/quality/summary`

Get overall data quality summary.

**Response:**
```json
{
  "overall_score": 0.94,
  "last_updated": "2023-10-01T06:00:00Z",
  "datasets": {
    "patients": {
      "score": 0.96,
      "total_tests": 25,
      "passed_tests": 24,
      "failed_tests": 1
    },
    "visits": {
      "score": 0.92,
      "total_tests": 18,
      "passed_tests": 16,
      "failed_tests": 2  
    }
  },
  "recent_issues": [
    {
      "dataset": "visits",
      "test": "visit_end_after_start",
      "severity": "warning",
      "count": 5,
      "last_seen": "2023-10-01T05:30:00Z"
    }
  ]
}
```

#### `GET /api/v1/quality/reports/{report_id}`

Get detailed quality validation report.

**Response:**
```json
{
  "report_id": "20231001_patients_validation",
  "dataset": "patients",
  "timestamp": "2023-10-01T06:00:00Z",
  "summary": {
    "total_expectations": 25,
    "successful_expectations": 24,
    "overall_success": false
  },
  "results": [
    {
      "expectation_type": "expect_column_values_to_not_be_null",
      "column": "person_id",
      "success": true,
      "result": {
        "observed_value": 0,
        "expected_value": 0
      }
    },
    {
      "expectation_type": "expect_column_values_to_be_between", 
      "column": "year_of_birth",
      "success": false,
      "result": {
        "observed_value": "5 values outside range",
        "expected_range": [1900, 2023],
        "unexpected_values": [1899, 1885, 2024]
      }
    }
  ]
}
```

### Analytics

#### `GET /api/v1/analytics/dashboard`

Get dashboard analytics data.

**Parameters:**
- `start_date` (date): Start date for analysis
- `end_date` (date): End date for analysis
- `granularity` (str): "day", "week", "month"

**Response:**
```json
{
  "period": {
    "start_date": "2023-09-01",
    "end_date": "2023-09-30",
    "granularity": "day"
  },
  "metrics": {
    "total_patients": 15670,
    "new_patients": 234,
    "total_visits": 45230,
    "avg_visits_per_patient": 2.9
  },
  "time_series": [
    {
      "date": "2023-09-01",
      "new_patients": 8,
      "total_visits": 145,
      "data_quality_score": 0.95
    }
  ],
  "top_conditions": [
    {
      "condition": "Hypertension",
      "patient_count": 3456,
      "percentage": 22.1
    }
  ]
}
```

## Error Handling

The API uses standard HTTP status codes and provides detailed error messages:

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid patient data provided",
    "details": {
      "field": "year_of_birth",
      "value": 2025,
      "constraint": "must be between 1900 and 2023"
    },
    "timestamp": "2023-10-01T12:00:00Z",
    "request_id": "req_abc123"
  }
}
```

### Status Codes

- **200 OK**: Successful request
- **201 Created**: Resource created successfully
- **400 Bad Request**: Invalid request parameters
- **401 Unauthorized**: Missing or invalid authentication
- **403 Forbidden**: Insufficient permissions (or read-only mode)
- **404 Not Found**: Resource not found
- **422 Unprocessable Entity**: Validation errors
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Server error
- **503 Service Unavailable**: System maintenance

## Rate Limiting

API requests are rate limited to ensure fair usage:

- **Authenticated users**: 1000 requests/hour
- **Anonymous users**: 100 requests/hour
- **Burst limit**: 50 requests/minute

Rate limit headers are included in responses:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1696176000
```

## SDKs & Examples

### Python SDK

```python
from clinical_platform import ClinicalAPI

# Initialize client
client = ClinicalAPI(
    base_url="https://api.clinical-platform.com",
    token="your_jwt_token"
)

# Get patients
patients = client.patients.list(limit=10, gender="female")

# Get predictions
prediction = client.predict.risk(
    patient_id=12345,
    features={
        "age": 45,
        "bmi": 28.5,
        "lab_values": {"glucose": 120}
    }
)

# Check data quality
quality = client.quality.summary()
print(f"Overall quality score: {quality.overall_score}")
```

### JavaScript/Node.js

```javascript
const { ClinicalAPI } = require('@clinical-platform/js-sdk');

const client = new ClinicalAPI({
  baseURL: 'https://api.clinical-platform.com',
  token: 'your_jwt_token'
});

// Get patients
const patients = await client.patients.list({
  limit: 10,
  gender: 'female'
});

// Generate prediction
const prediction = await client.predict.risk({
  patient_id: 12345,
  features: {
    age: 45,
    bmi: 28.5,
    lab_values: { glucose: 120 }
  }
});
```

### cURL Examples

```bash
# List patients
curl -H "Authorization: Bearer $TOKEN" \
  "https://api.clinical-platform.com/api/v1/patients?limit=10"

# Get prediction
curl -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"patient_id": 12345, "features": {...}}' \
  "https://api.clinical-platform.com/api/v1/predict/risk"

# Check health
curl "https://api.clinical-platform.com/health"
```

## OpenAPI Schema

The complete API schema is available at `/openapi.json` and interactive docs at `/docs`:

- **Swagger UI**: `/docs` 
- **ReDoc**: `/redoc`
- **OpenAPI JSON**: `/openapi.json`

You can also generate client SDKs using the OpenAPI schema:

```bash
# Generate Python client
openapi-generator generate \
  -i https://api.clinical-platform.com/openapi.json \
  -g python \
  -o ./python-client

# Generate TypeScript client  
openapi-generator generate \
  -i https://api.clinical-platform.com/openapi.json \
  -g typescript-axios \
  -o ./ts-client
```