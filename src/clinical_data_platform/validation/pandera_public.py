"""
Pandera schemas for public CDM synthetic data validation.

These schemas define the expected structure and constraints for the synthetic
OMOP CDM-like data used in the public demo. They provide runtime validation
to ensure data quality and adherence to expected patterns.
"""

from datetime import datetime
from typing import Optional

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema


class PublicCDMSchemas:
    """Collection of pandera schemas for public CDM validation."""
    
    @staticmethod
    def person_schema() -> DataFrameSchema:
        """Schema for the person table."""
        return DataFrameSchema(
            {
                "person_id": Column(
                    pa.Int64,
                    checks=[
                        pa.Check.greater_than(0),
                        pa.Check(lambda s: s.nunique() == len(s), error="person_id must be unique")
                    ],
                    nullable=False,
                    description="Unique identifier for each person"
                ),
                "gender_concept_id": Column(
                    pa.Int64,
                    checks=[
                        pa.Check.isin([8507, 8532, 8551, 0])  # OMOP gender concepts
                    ],
                    nullable=False,
                    description="OMOP concept for gender"
                ),
                "year_of_birth": Column(
                    pa.Int64,
                    checks=[
                        pa.Check.between(1900, 2024)
                    ],
                    nullable=True,
                    description="Year of birth"
                ),
                "month_of_birth": Column(
                    pa.Int64,
                    checks=[
                        pa.Check.between(1, 12)
                    ],
                    nullable=True,
                    description="Month of birth"
                ),
                "day_of_birth": Column(
                    pa.Int64,
                    checks=[
                        pa.Check.between(1, 31)
                    ],
                    nullable=True,
                    description="Day of birth"
                ),
                "birth_datetime": Column(
                    pa.DateTime,
                    nullable=True,
                    description="Birth date and time"
                ),
                "race_concept_id": Column(
                    pa.Int64,
                    nullable=True,
                    description="OMOP concept for race"
                ),
                "ethnicity_concept_id": Column(
                    pa.Int64,
                    nullable=True,
                    description="OMOP concept for ethnicity"
                ),
                "location_id": Column(
                    pa.Int64,
                    nullable=True,
                    description="Foreign key to location"
                ),
                "provider_id": Column(
                    pa.Int64,
                    nullable=True,
                    description="Foreign key to provider"
                ),
                "care_site_id": Column(
                    pa.Int64,
                    nullable=True,
                    description="Foreign key to care site"
                ),
                "person_source_value": Column(
                    pa.String,
                    nullable=True,
                    description="Source value for person"
                ),
                "gender_source_value": Column(
                    pa.String,
                    nullable=True,
                    description="Source value for gender"
                ),
                "gender_source_concept_id": Column(
                    pa.Int64,
                    nullable=True,
                    description="Source concept for gender"
                ),
                "race_source_value": Column(
                    pa.String,
                    nullable=True,
                    description="Source value for race"
                ),
                "race_source_concept_id": Column(
                    pa.Int64,
                    nullable=True,
                    description="Source concept for race"
                ),
                "ethnicity_source_value": Column(
                    pa.String,
                    nullable=True,
                    description="Source value for ethnicity"
                ),
                "ethnicity_source_concept_id": Column(
                    pa.Int64,
                    nullable=True,
                    description="Source concept for ethnicity"
                ),
            },
            checks=[
                pa.Check(lambda df: len(df) >= 100, error="Must have at least 100 persons"),
                pa.Check(lambda df: len(df) <= 10000, error="Must have at most 10000 persons for demo"),
            ],
            description="OMOP CDM Person table schema for synthetic data"
        )
    
    @staticmethod 
    def visit_occurrence_schema() -> DataFrameSchema:
        """Schema for the visit_occurrence table."""
        return DataFrameSchema(
            {
                "visit_occurrence_id": Column(
                    pa.Int64,
                    checks=[
                        pa.Check.greater_than(0),
                        pa.Check(lambda s: s.nunique() == len(s), error="visit_occurrence_id must be unique")
                    ],
                    nullable=False,
                    description="Unique identifier for each visit"
                ),
                "person_id": Column(
                    pa.Int64,
                    checks=[pa.Check.greater_than(0)],
                    nullable=False,
                    description="Foreign key to person"
                ),
                "visit_concept_id": Column(
                    pa.Int64,
                    checks=[
                        pa.Check.isin([9201, 9202, 9203])  # Common visit types
                    ],
                    nullable=False,
                    description="OMOP concept for visit type"
                ),
                "visit_start_date": Column(
                    pa.DateTime,
                    nullable=False,
                    description="Start date of visit"
                ),
                "visit_start_datetime": Column(
                    pa.DateTime,
                    nullable=True,
                    description="Start datetime of visit"
                ),
                "visit_end_date": Column(
                    pa.DateTime,
                    nullable=False,
                    description="End date of visit"
                ),
                "visit_end_datetime": Column(
                    pa.DateTime,
                    nullable=True,
                    description="End datetime of visit"
                ),
                "visit_type_concept_id": Column(
                    pa.Int64,
                    nullable=False,
                    description="OMOP concept for visit type"
                ),
                "provider_id": Column(
                    pa.Int64,
                    nullable=True,
                    description="Foreign key to provider"
                ),
                "care_site_id": Column(
                    pa.Int64,
                    nullable=True,
                    description="Foreign key to care site"
                ),
                "visit_source_value": Column(
                    pa.String,
                    nullable=True,
                    description="Source value for visit"
                ),
                "visit_source_concept_id": Column(
                    pa.Int64,
                    nullable=True,
                    description="Source concept for visit"
                ),
                "admitted_from_concept_id": Column(
                    pa.Int64,
                    nullable=True,
                    description="Concept for admission source"
                ),
                "admitted_from_source_value": Column(
                    pa.String,
                    nullable=True,
                    description="Source value for admission"
                ),
                "discharged_to_concept_id": Column(
                    pa.Int64,
                    nullable=True,
                    description="Concept for discharge destination"
                ),
                "discharged_to_source_value": Column(
                    pa.String,
                    nullable=True,
                    description="Source value for discharge"
                ),
                "preceding_visit_occurrence_id": Column(
                    pa.Int64,
                    nullable=True,
                    description="Foreign key to preceding visit"
                ),
            },
            checks=[
                pa.Check(lambda df: (df["visit_end_date"] >= df["visit_start_date"]).all(),
                        error="Visit end date must be >= start date"),
                pa.Check(lambda df: ((df["visit_end_date"] - df["visit_start_date"]).dt.days <= 365).all(),
                        error="Visit length must be <= 365 days")
            ],
            description="OMOP CDM Visit Occurrence table schema for synthetic data"
        )

    @staticmethod
    def condition_occurrence_schema() -> DataFrameSchema:
        """Schema for the condition_occurrence table."""
        return DataFrameSchema(
            {
                "condition_occurrence_id": Column(
                    pa.Int64,
                    checks=[
                        pa.Check.greater_than(0),
                        pa.Check(lambda s: s.nunique() == len(s), error="condition_occurrence_id must be unique")
                    ],
                    nullable=False,
                    description="Unique identifier for each condition"
                ),
                "person_id": Column(
                    pa.Int64,
                    checks=[pa.Check.greater_than(0)],
                    nullable=False,
                    description="Foreign key to person"
                ),
                "condition_concept_id": Column(
                    pa.Int64,
                    checks=[pa.Check.greater_than_or_equal_to(0)],
                    nullable=False,
                    description="OMOP concept for condition"
                ),
                "condition_start_date": Column(
                    pa.DateTime,
                    nullable=False,
                    description="Start date of condition"
                ),
                "condition_start_datetime": Column(
                    pa.DateTime,
                    nullable=True,
                    description="Start datetime of condition"
                ),
                "condition_end_date": Column(
                    pa.DateTime,
                    nullable=True,
                    description="End date of condition"
                ),
                "condition_end_datetime": Column(
                    pa.DateTime,
                    nullable=True,
                    description="End datetime of condition"
                ),
                "condition_type_concept_id": Column(
                    pa.Int64,
                    nullable=False,
                    description="OMOP concept for condition type"
                ),
                "condition_status_concept_id": Column(
                    pa.Int64,
                    nullable=True,
                    description="OMOP concept for condition status"
                ),
                "stop_reason": Column(
                    pa.String,
                    nullable=True,
                    description="Reason condition ended"
                ),
                "provider_id": Column(
                    pa.Int64,
                    nullable=True,
                    description="Foreign key to provider"
                ),
                "visit_occurrence_id": Column(
                    pa.Int64,
                    nullable=True,
                    description="Foreign key to visit"
                ),
                "visit_detail_id": Column(
                    pa.Int64,
                    nullable=True,
                    description="Foreign key to visit detail"
                ),
                "condition_source_value": Column(
                    pa.String,
                    nullable=True,
                    description="Source value for condition"
                ),
                "condition_source_concept_id": Column(
                    pa.Int64,
                    nullable=True,
                    description="Source concept for condition"
                ),
                "condition_status_source_value": Column(
                    pa.String,
                    nullable=True,
                    description="Source value for condition status"
                ),
            },
            description="OMOP CDM Condition Occurrence table schema for synthetic data"
        )

    @staticmethod
    def measurement_schema() -> DataFrameSchema:
        """Schema for the measurement table."""
        return DataFrameSchema(
            {
                "measurement_id": Column(
                    pa.Int64,
                    checks=[
                        pa.Check.greater_than(0),
                        pa.Check(lambda s: s.nunique() == len(s), error="measurement_id must be unique")
                    ],
                    nullable=False,
                    description="Unique identifier for each measurement"
                ),
                "person_id": Column(
                    pa.Int64,
                    checks=[pa.Check.greater_than(0)],
                    nullable=False,
                    description="Foreign key to person"
                ),
                "measurement_concept_id": Column(
                    pa.Int64,
                    checks=[pa.Check.greater_than_or_equal_to(0)],
                    nullable=False,
                    description="OMOP concept for measurement"
                ),
                "measurement_date": Column(
                    pa.DateTime,
                    nullable=False,
                    description="Date of measurement"
                ),
                "measurement_datetime": Column(
                    pa.DateTime,
                    nullable=True,
                    description="Datetime of measurement"
                ),
                "measurement_time": Column(
                    pa.String,
                    nullable=True,
                    description="Time of measurement"
                ),
                "measurement_type_concept_id": Column(
                    pa.Int64,
                    nullable=False,
                    description="OMOP concept for measurement type"
                ),
                "operator_concept_id": Column(
                    pa.Int64,
                    nullable=True,
                    description="OMOP concept for operator"
                ),
                "value_as_number": Column(
                    pa.Float64,
                    nullable=True,
                    description="Numeric value of measurement"
                ),
                "value_as_concept_id": Column(
                    pa.Int64,
                    nullable=True,
                    description="OMOP concept for value"
                ),
                "unit_concept_id": Column(
                    pa.Int64,
                    nullable=True,
                    description="OMOP concept for unit"
                ),
                "range_low": Column(
                    pa.Float64,
                    nullable=True,
                    description="Lower range value"
                ),
                "range_high": Column(
                    pa.Float64,
                    nullable=True,
                    description="Upper range value"
                ),
                "provider_id": Column(
                    pa.Int64,
                    nullable=True,
                    description="Foreign key to provider"
                ),
                "visit_occurrence_id": Column(
                    pa.Int64,
                    nullable=True,
                    description="Foreign key to visit"
                ),
                "visit_detail_id": Column(
                    pa.Int64,
                    nullable=True,
                    description="Foreign key to visit detail"
                ),
                "measurement_source_value": Column(
                    pa.String,
                    nullable=True,
                    description="Source value for measurement"
                ),
                "measurement_source_concept_id": Column(
                    pa.Int64,
                    nullable=True,
                    description="Source concept for measurement"
                ),
                "unit_source_value": Column(
                    pa.String,
                    nullable=True,
                    description="Source value for unit"
                ),
                "unit_source_concept_id": Column(
                    pa.Int64,
                    nullable=True,
                    description="Source concept for unit"
                ),
                "value_source_value": Column(
                    pa.String,
                    nullable=True,
                    description="Source value for measurement value"
                ),
                "measurement_event_id": Column(
                    pa.Int64,
                    nullable=True,
                    description="Foreign key to measurement event"
                ),
                "meas_event_field_concept_id": Column(
                    pa.Int64,
                    nullable=True,
                    description="OMOP concept for measurement event field"
                ),
            },
            description="OMOP CDM Measurement table schema for synthetic data"
        )


def validate_public_cdm_data(
    person_df: pd.DataFrame,
    visit_df: pd.DataFrame, 
    condition_df: pd.DataFrame,
    measurement_df: pd.DataFrame
) -> dict:
    """
    Validate all public CDM tables using pandera schemas.
    
    Args:
        person_df: Person table dataframe
        visit_df: Visit occurrence table dataframe
        condition_df: Condition occurrence table dataframe
        measurement_df: Measurement table dataframe
        
    Returns:
        Dictionary with validation results for each table
    """
    schemas = PublicCDMSchemas()
    results = {}
    
    tables = {
        "person": (person_df, schemas.person_schema()),
        "visit_occurrence": (visit_df, schemas.visit_occurrence_schema()),
        "condition_occurrence": (condition_df, schemas.condition_occurrence_schema()),
        "measurement": (measurement_df, schemas.measurement_schema())
    }
    
    for table_name, (df, schema) in tables.items():
        try:
            validated_df = schema.validate(df, lazy=True)
            results[table_name] = {
                "status": "passed",
                "row_count": len(validated_df),
                "errors": []
            }
        except pa.errors.SchemaErrors as e:
            results[table_name] = {
                "status": "failed", 
                "row_count": len(df),
                "errors": [str(error) for error in e.schema_errors]
            }
        except Exception as e:
            results[table_name] = {
                "status": "error",
                "row_count": len(df),
                "errors": [f"Unexpected error: {str(e)}"]
            }
    
    return results