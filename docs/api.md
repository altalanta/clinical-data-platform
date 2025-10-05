# API Reference

The Clinical Data Platform provides a secure, HIPAA-compliant REST API for clinical data operations and ML predictions.

## OpenAPI Specification

**[📋 Interactive API Documentation](http://localhost:8000/docs)** *(when API is running)*

**[📄 OpenAPI JSON Schema](assets/api/openapi.json)**

## Authentication

All protected endpoints require API key authentication via Bearer token:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" http://localhost:8000/studies
```

## Endpoints Overview

### Health Check

#### `GET /health`

Public endpoint for health monitoring.

**Response:**
```json
{
  "status": "ok"
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

### Studies

#### `GET /studies`

List all available clinical studies.

**Authentication:** Required  
**Response:** Array of study IDs

**Example:**
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
  http://localhost:8000/studies
```

**Response:**
```json
["STUDY001", "STUDY002", "ONCOLOGY_TRIAL_2023"]
```

### Subjects

#### `GET /subjects/{subject_id}`

Retrieve subject information with PII redaction.

**Authentication:** Required  
**Parameters:**
- `subject_id` (path): Subject identifier (alphanumeric, underscore, hyphen only)

**Example:**
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
  http://localhost:8000/subjects/SUBJ001
```

**Response:**
```json
{
  "subject_id": "SUBJ001",
  "study_id": "STUDY001", 
  "arm": "treatment",
  "enrollment_status": "active"
}
```

### ML Predictions

#### `POST /score`

Generate ML risk predictions based on clinical parameters.

**Authentication:** Required  
**Content-Type:** `application/json`

**Request Body:**
```json
{
  "AGE": 65.0,
  "AE_COUNT": 3.0,
  "SEVERE_AE_COUNT": 1.0
}
```

**Response:**
```json
{
  "risk": 0.456,
  "model_version": "1.0.0",
  "confidence": "medium"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/score \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "AGE": 65.0,
    "AE_COUNT": 3.0, 
    "SEVERE_AE_COUNT": 1.0
  }'
```

## Request/Response Models

### ScoreRequest

Pydantic model for ML scoring with built-in validation:

```python
class ScoreRequest(BaseModel):
    """Request model for ML scoring endpoint with validation."""
    AGE: float = Field(..., ge=0, le=120, description="Patient age in years")
    AE_COUNT: float = Field(..., ge=0, le=100, description="Total adverse event count")
    SEVERE_AE_COUNT: float = Field(..., ge=0, le=50, description="Severe adverse event count")
    
    @validator('SEVERE_AE_COUNT')
    def severe_count_must_be_less_than_total(cls, v, values):
        if 'AE_COUNT' in values and v > values['AE_COUNT']:
            raise ValueError('Severe AE count cannot exceed total AE count')
        return v
```

**Validation Rules:**
- `AGE`: Must be between 0 and 120 years
- `AE_COUNT`: Must be between 0 and 100 events
- `SEVERE_AE_COUNT`: Must be ≤ total AE count and ≤ 50 events

### ScoreResponse

```python
class ScoreResponse(BaseModel):
    """Response model for ML scoring with metadata."""
    risk: float = Field(..., ge=0.0, le=1.0, description="Risk probability [0-1]")
    model_version: str = Field(default="1.0.0", description="Model version used")
    confidence: Optional[str] = Field(default=None, description="Prediction confidence level")
```

### SubjectResponse

```python
class SubjectResponse(BaseModel):
    """Response model for subject data with PII redaction."""
    subject_id: str
    study_id: str
    arm: Optional[str] = None
    enrollment_status: Optional[str] = None
    # Note: PII fields like age, sex are redacted from response
```

## Security Features

### HIPAA Compliance

- **PHI Redaction**: Automatic PII/PHI redaction in logs and error messages
- **API Key Authentication**: Mandatory for all protected endpoints
- **Input Validation**: Strict validation on all inputs with sanitization
- **Parameterized Queries**: SQL injection prevention
- **Error Sanitization**: PHI-safe error messages

### Security Middleware

```python
# Trusted host validation
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["localhost", "127.0.0.1", "*.clinical-platform.com"]
)

# CORS configuration (restrictive in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://clinical-platform.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)
```

## Error Handling

### Standard Error Responses

All errors follow a consistent format with PHI-safe messages:

```json
{
  "detail": "Invalid subject ID format"
}
```

### HTTP Status Codes

| Code | Description | Example |
|------|-------------|---------|
| 200 | Success | Successful data retrieval |
| 400 | Bad Request | Invalid input parameters |
| 401 | Unauthorized | Missing or invalid API key |
| 404 | Not Found | Subject/study not found |
| 500 | Internal Server Error | Database or model errors |

## Client Examples

### Python Client

```python
import requests

# Configure client
base_url = "http://localhost:8000"
headers = {"Authorization": "Bearer YOUR_API_KEY"}

# Health check
response = requests.get(f"{base_url}/health")
print(response.json())

# List studies
response = requests.get(f"{base_url}/studies", headers=headers)
studies = response.json()

# Get subject
response = requests.get(f"{base_url}/subjects/SUBJ001", headers=headers)
subject = response.json()

# ML prediction
payload = {
    "AGE": 65.0,
    "AE_COUNT": 3.0,
    "SEVERE_AE_COUNT": 1.0
}
response = requests.post(f"{base_url}/score", json=payload, headers=headers)
prediction = response.json()
print(f"Risk score: {prediction['risk']:.3f}")
```

## Development

### Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start API server
uvicorn clinical_platform.api.main:app --reload --host 0.0.0.0 --port 8000

# View interactive docs
open http://localhost:8000/docs

# View OpenAPI spec
curl http://localhost:8000/openapi.json
```