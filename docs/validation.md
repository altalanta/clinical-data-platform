# Data Validation

The Clinical Data Platform uses a multi-layered approach to data validation, combining Great Expectations for data profiling and Pandera for schema validation.

## Overview

Data validation occurs at multiple stages:

1. **Ingestion**: Great Expectations validates raw data quality
2. **Transformation**: Pandera enforces schema compliance
3. **Output**: dbt tests ensure analytical data integrity
4. **Runtime**: API validation prevents invalid requests

## Great Expectations

Great Expectations provides data profiling and validation capabilities.

### Configuration

```yaml
# great_expectations/great_expectations.yml
config_version: 3.0

datasources:
  clinical_data:
    class_name: Datasource
    module_name: great_expectations.datasource
    execution_engine:
      class_name: PandasExecutionEngine
      module_name: great_expectations.execution_engine
    data_connectors:
      default_inferred_data_connector_name:
        class_name: InferredAssetFilesystemDataConnector
        base_directory: data/
        default_regex:
          group_names:
            - data_asset_name
          pattern: (.*)\.csv
```

### Example Expectation Suite

```python
# great_expectations/expectations/patients_suite.json
{
    "expectation_suite_name": "patients_suite",
    "expectations": [
        {
            "expectation_type": "expect_table_row_count_to_be_between",
            "kwargs": {
                "min_value": 1000,
                "max_value": 1000000
            },
            "meta": {
                "notes": "Patient table should have reasonable number of records"
            }
        },
        {
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {
                "column": "person_id"
            },
            "meta": {
                "notes": "Every patient must have a unique identifier"
            }
        },
        {
            "expectation_type": "expect_column_values_to_be_unique",
            "kwargs": {
                "column": "person_id"
            }
        },
        {
            "expectation_type": "expect_column_values_to_be_in_set",
            "kwargs": {
                "column": "gender_concept_id",
                "value_set": [8507, 8532, 0]
            },
            "meta": {
                "notes": "Gender must be Male (8507), Female (8532), or Unknown (0)"
            }
        },
        {
            "expectation_type": "expect_column_values_to_be_between",
            "kwargs": {
                "column": "year_of_birth", 
                "min_value": 1900,
                "max_value": 2023
            }
        }
    ]
}
```

### Validation Results

Great Expectations generates detailed validation reports:

```python
# Run validation
context = DataContext()
batch = context.get_batch(
    datasource_name="clinical_data",
    data_asset_name="patients", 
    expectation_suite_name="patients_suite"
)

validation_result = context.run_validation_operator(
    "action_list_operator",
    assets_to_validate=[batch]
)

# Check results
if validation_result.success:
    print("✅ All expectations passed!")
else:
    print("❌ Some expectations failed:")
    for result in validation_result.run_results:
        for expectation in result.validation_result.results:
            if not expectation.success:
                print(f"Failed: {expectation.expectation_config.expectation_type}")
```

### Data Docs

Automatically generated documentation:

- **Expectation Suites**: What we expect from each dataset
- **Validation Results**: Historical validation outcomes  
- **Data Profiling**: Statistical summaries and distributions
- **Data Lineage**: How datasets relate to each other

## Pandera

Pandera provides runtime schema validation for pandas DataFrames.

### Schema Definition

```python
import pandera as pa
from pandera.typing import DataFrame, Series
from typing import Optional

class PatientSchema(pa.SchemaModel):
    """Schema for patient data"""
    
    person_id: Series[int] = pa.Field(gt=0, unique=True)
    gender_concept_id: Series[int] = pa.Field(isin=[8507, 8532, 0])
    year_of_birth: Series[int] = pa.Field(ge=1900, le=2023)
    race_concept_id: Optional[Series[int]] = pa.Field(nullable=True)
    ethnicity_concept_id: Optional[Series[int]] = pa.Field(nullable=True)
    
    class Config:
        name = "patient_schema"
        description = "Schema for OMOP patient dimension"
        strict = True  # Disallow columns not in schema

class VisitSchema(pa.SchemaModel):
    """Schema for visit/encounter data"""
    
    visit_occurrence_id: Series[int] = pa.Field(gt=0, unique=True)
    person_id: Series[int] = pa.Field(gt=0)
    visit_concept_id: Series[int] = pa.Field(gt=0)
    visit_start_date: Series[pa.typing.pandas.Timestamp]
    visit_end_date: Optional[Series[pa.typing.pandas.Timestamp]] = pa.Field(nullable=True)
    
    @pa.check("visit_end_date")
    def visit_end_after_start(cls, series: Series, df: DataFrame) -> Series[bool]:
        """Visit end date must be after start date"""
        return (df["visit_end_date"].isna()) | (df["visit_end_date"] >= df["visit_start_date"])
```

### Validation Decorators

```python
@pa.check_types
def load_patients(file_path: str) -> DataFrame[PatientSchema]:
    """Load and validate patient data"""
    df = pd.read_csv(file_path)
    return df

@pa.check_types  
def process_visits(
    visits_df: DataFrame[VisitSchema],
    patients_df: DataFrame[PatientSchema]
) -> DataFrame[VisitSchema]:
    """Process visit data with patient validation"""
    
    # Join with patients to validate foreign keys
    validated_visits = visits_df.merge(
        patients_df[['person_id']], 
        on='person_id',
        how='inner'
    )
    
    return validated_visits

# Usage
try:
    patients = load_patients("data/patients.csv")
    visits = load_patients("data/visits.csv") 
    processed_visits = process_visits(visits, patients)
    print("✅ All data validated successfully")
except pa.errors.SchemaError as e:
    print(f"❌ Schema validation failed: {e}")
```

### Custom Checks

```python
# Custom validation functions
@pa.check("person_id")
def check_person_id_format(person_ids: Series) -> Series[bool]:
    """Person IDs should be positive integers"""
    return person_ids > 0

@pa.check("year_of_birth") 
def check_reasonable_age(years: Series) -> Series[bool]:
    """Patients should be reasonable age (0-120 years old)"""
    current_year = pd.Timestamp.now().year
    ages = current_year - years
    return (ages >= 0) & (ages <= 120)

# Register checks in schema
class EnhancedPatientSchema(PatientSchema):
    person_id: Series[int] = pa.Field(checks=[check_person_id_format])
    year_of_birth: Series[int] = pa.Field(checks=[check_reasonable_age])
```

## dbt Tests

dbt provides SQL-based testing for analytical models.

### Built-in Tests

```yaml
# models/schema.yml
models:
  - name: dim_patients
    description: "Patient dimension table"
    columns:
      - name: person_id
        description: "Unique patient identifier"
        tests:
          - unique
          - not_null
      - name: year_of_birth
        description: "Year patient was born"
        tests:
          - not_null
          - relationships:
              to: ref('valid_years')
              field: year
              
  - name: fact_visits
    description: "Visit fact table"
    tests:
      - dbt_utils.unique_combination_of_columns:
          combination_of_columns:
            - visit_occurrence_id
            - person_id
```

### Custom Tests

```sql
-- tests/assert_visit_dates_logical.sql
-- Ensure visit end dates are after start dates

select 
    visit_occurrence_id,
    visit_start_date,
    visit_end_date
from {{ ref('fact_visits') }}
where visit_end_date < visit_start_date
```

### dbt Expectations

Advanced testing with dbt-expectations package:

```yaml
models:
  - name: dim_patients
    tests:
      # Row count tests
      - dbt_expectations.expect_table_row_count_to_be_between:
          min_value: 1000
          max_value: 100000
          
      # Column distribution tests  
      - dbt_expectations.expect_column_values_to_be_in_set:
          column_name: gender_concept_id
          value_set: [8507, 8532, 0]
          
      # Statistical tests
      - dbt_expectations.expect_column_mean_to_be_between:
          column_name: year_of_birth
          min_value: 1950
          max_value: 2010
```

## API Validation

FastAPI provides automatic request/response validation.

### Pydantic Models

```python
from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import date

class PatientRequest(BaseModel):
    """Request model for patient data"""
    
    person_id: int = Field(..., gt=0, description="Unique patient ID")
    gender_concept_id: int = Field(..., description="Gender concept ID")
    year_of_birth: int = Field(..., ge=1900, le=2023, description="Birth year")
    race_concept_id: Optional[int] = Field(None, description="Race concept ID")
    
    @validator('gender_concept_id')
    def validate_gender(cls, v):
        if v not in [8507, 8532, 0]:
            raise ValueError('Gender must be Male (8507), Female (8532), or Unknown (0)')
        return v

class PredictionRequest(BaseModel):
    """Request model for ML predictions"""
    
    patient_id: int = Field(..., gt=0)
    age: int = Field(..., ge=0, le=120)
    gender: str = Field(..., regex=r'^(male|female|unknown)$')
    lab_values: dict = Field(..., description="Laboratory test results")
    
    @validator('lab_values')
    def validate_lab_values(cls, v):
        required_labs = {'glucose', 'hemoglobin', 'creatinine'}
        if not required_labs.issubset(v.keys()):
            raise ValueError(f'Missing required lab values: {required_labs - v.keys()}')
        return v

class PredictionResponse(BaseModel):
    """Response model for ML predictions"""
    
    patient_id: int
    prediction: float = Field(..., ge=0, le=1, description="Risk score 0-1")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence")
    model_version: str
    timestamp: date
```

### API Endpoints with Validation

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="Clinical Data Platform API")

@app.post("/patients", response_model=dict)
async def create_patient(patient: PatientRequest):
    """Create a new patient record"""
    try:
        # Additional business logic validation
        if await patient_exists(patient.person_id):
            raise HTTPException(400, "Patient already exists")
            
        # Save patient
        result = await save_patient(patient)
        return {"status": "created", "patient_id": patient.person_id}
        
    except ValueError as e:
        raise HTTPException(422, f"Validation error: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)  
async def predict_risk(request: PredictionRequest):
    """Generate risk prediction for patient"""
    try:
        # Load model and generate prediction
        model = await load_model()
        prediction = model.predict(request.dict())
        
        return PredictionResponse(
            patient_id=request.patient_id,
            prediction=prediction['risk_score'],
            confidence=prediction['confidence'],
            model_version="v1.2.3",
            timestamp=date.today()
        )
        
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")
```

## Quality Gates

Automated quality gates ensure data meets standards before promotion.

### CI/CD Integration

```yaml
# .github/workflows/data-quality.yml
name: Data Quality

on:
  push:
    paths:
      - 'data/**'
      - 'models/**'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Run Great Expectations
        run: |
          great_expectations checkpoint run patients_checkpoint
          great_expectations checkpoint run visits_checkpoint
          
      - name: Run dbt tests
        run: |
          dbt test --store-failures
          
      - name: Check test coverage
        run: |
          python scripts/check_test_coverage.py --threshold 0.8
```

### Quality Metrics

Track validation results over time:

```python
# Quality metrics calculation
def calculate_quality_score(validation_results):
    """Calculate overall data quality score"""
    
    total_expectations = len(validation_results.expectations)
    passed_expectations = sum(1 for exp in validation_results.expectations if exp.success)
    
    quality_score = passed_expectations / total_expectations
    
    return {
        'quality_score': quality_score,
        'total_tests': total_expectations,
        'passed_tests': passed_expectations,
        'failed_tests': total_expectations - passed_expectations,
        'timestamp': datetime.now()
    }
```

## Monitoring & Alerting

Set up alerts for data quality issues:

```python
# Example alerting logic
def check_data_quality_alerts(quality_score):
    """Send alerts if data quality drops below threshold"""
    
    if quality_score < 0.95:
        send_alert(
            level="WARNING",
            message=f"Data quality score dropped to {quality_score:.2%}",
            channels=["#data-engineering", "#alerts"]
        )
        
    if quality_score < 0.8:
        send_alert(
            level="CRITICAL", 
            message=f"Data quality score critically low: {quality_score:.2%}",
            channels=["#incidents", "#on-call"]
        )
```

## Best Practices

### 1. Test Early and Often
- Validate data at ingestion
- Use schema evolution to handle changes
- Monitor data drift over time

### 2. Meaningful Error Messages
- Provide clear descriptions of what failed
- Include context about expected vs actual values
- Suggest remediation steps

### 3. Performance Considerations
- Sample large datasets for faster validation
- Use incremental validation for streaming data
- Cache validation results where appropriate

### 4. Documentation
- Document all expectations and their business rationale
- Maintain schema documentation
- Track validation history and trends