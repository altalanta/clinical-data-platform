"""
Pandera schemas for SDTM domain validation.
"""

from typing import Dict, Any
import pandera as pa
from pandera import Column, DataFrameSchema, Check
import pandas as pd


# =============================================================================
# SDTM Domain Schemas
# =============================================================================

DM_SCHEMA = DataFrameSchema(
    {
        "STUDYID": Column(str, checks=[
            Check.str_length(min_value=1, max_value=50),
            Check(lambda x: x.notna().all(), error="STUDYID cannot be null")
        ]),
        "SUBJID": Column(str, checks=[
            Check.str_length(min_value=1, max_value=50),
            Check(lambda x: x.notna().all(), error="SUBJID cannot be null"),
            Check(lambda x: x.str.match(r'^[A-Z0-9\-]+$').all(), 
                  error="SUBJID must contain only alphanumeric characters and hyphens")
        ]),
        "ARM": Column(str, nullable=True, checks=[
            Check.str_length(max_value=200)
        ]),
        "SEX": Column(str, nullable=True, checks=[
            Check.isin(["M", "F", "U", None], error="SEX must be M, F, U, or null")
        ]),
        "AGE": Column(float, nullable=True, checks=[
            Check.in_range(0, 150, include_min=True, include_max=True,
                          error="AGE must be between 0 and 150")
        ]),
    },
    strict=True,
    coerce=True,
    name="Demographics (DM)"
)

AE_SCHEMA = DataFrameSchema(
    {
        "STUDYID": Column(str, checks=[Check.str_length(min_value=1, max_value=50)]),
        "SUBJID": Column(str, checks=[Check.str_length(min_value=1, max_value=50)]),
        "AESTDTC": Column("datetime64[ns]", nullable=True),
        "AEENDTC": Column("datetime64[ns]", nullable=True),
        "AESEV": Column(str, nullable=True, checks=[
            Check.isin(["MILD", "MODERATE", "SEVERE", None])
        ]),
        "AESER": Column("boolean", nullable=True),
        "AEOUT": Column(str, nullable=True, checks=[
            Check.isin(["RECOVERED/RESOLVED", "RECOVERING/RESOLVING", 
                       "NOT RECOVERED/NOT RESOLVED", "RECOVERED/RESOLVED WITH SEQUELAE",
                       "FATAL", "UNKNOWN", None])
        ]),
    },
    strict=True,
    coerce=True,
    name="Adverse Events (AE)"
)

LB_SCHEMA = DataFrameSchema(
    {
        "STUDYID": Column(str, checks=[Check.str_length(min_value=1, max_value=50)]),
        "SUBJID": Column(str, checks=[Check.str_length(min_value=1, max_value=50)]),
        "LBTESTCD": Column(str, checks=[
            Check.str_length(min_value=1, max_value=8),
            Check(lambda x: x.str.isupper().all(), error="LBTESTCD must be uppercase")
        ]),
        "LBORRES": Column(float, nullable=True, checks=[
            Check(lambda x: x.ge(0).all() | x.isna(), error="LBORRES must be non-negative")
        ]),
        "LBORRESU": Column(str, nullable=True, checks=[
            Check.str_length(max_value=20)
        ]),
        "LBLNOR": Column(float, nullable=True),
        "LBHNOR": Column(float, nullable=True),
    },
    checks=[
        Check(lambda df: (df["LBLNOR"] <= df["LBHNOR"]).all() | 
              df["LBLNOR"].isna() | df["LBHNOR"].isna(),
              error="LBLNOR must be <= LBHNOR when both are present")
    ],
    strict=True,
    coerce=True,
    name="Laboratory (LB)"
)

VS_SCHEMA = DataFrameSchema(
    {
        "STUDYID": Column(str, checks=[Check.str_length(min_value=1, max_value=50)]),
        "SUBJID": Column(str, checks=[Check.str_length(min_value=1, max_value=50)]),
        "VSTESTCD": Column(str, checks=[
            Check.str_length(min_value=1, max_value=8),
            Check(lambda x: x.str.isupper().all(), error="VSTESTCD must be uppercase")
        ]),
        "VSORRES": Column(float, nullable=True, checks=[
            Check(lambda x: x.ge(0).all() | x.isna(), error="VSORRES must be non-negative")
        ]),
        "VSORRESU": Column(str, nullable=True, checks=[
            Check.str_length(max_value=20)
        ]),
    },
    strict=True,
    coerce=True,
    name="Vital Signs (VS)"
)

EX_SCHEMA = DataFrameSchema(
    {
        "STUDYID": Column(str, checks=[Check.str_length(min_value=1, max_value=50)]),
        "SUBJID": Column(str, checks=[Check.str_length(min_value=1, max_value=50)]),
        "EXTRT": Column(str, nullable=True, checks=[
            Check.str_length(max_value=200)
        ]),
        "EXDOSE": Column(float, nullable=True, checks=[
            Check(lambda x: x.gt(0).all() | x.isna(), error="EXDOSE must be positive")
        ]),
        "EXSTDTC": Column("datetime64[ns]", nullable=True),
        "EXENDTC": Column("datetime64[ns]", nullable=True),
    },
    checks=[
        Check(lambda df: (df["EXSTDTC"] <= df["EXENDTC"]).all() | 
              df["EXSTDTC"].isna() | df["EXENDTC"].isna(),
              error="EXSTDTC must be <= EXENDTC when both are present")
    ],
    strict=True,
    coerce=True,
    name="Exposure (EX)"
)

# =============================================================================
# Schema Registry
# =============================================================================

SDTM_SCHEMAS: Dict[str, DataFrameSchema] = {
    "DM": DM_SCHEMA,
    "AE": AE_SCHEMA,
    "LB": LB_SCHEMA,
    "VS": VS_SCHEMA,
    "EX": EX_SCHEMA,
}


def get_schema_for_domain(domain: str) -> DataFrameSchema:
    """Get pandera schema for SDTM domain."""
    domain_upper = domain.upper()
    if domain_upper not in SDTM_SCHEMAS:
        raise ValueError(f"No schema available for domain: {domain}")
    return SDTM_SCHEMAS[domain_upper]


def validate_dataframe(df: pd.DataFrame, domain: str) -> Dict[str, Any]:
    """
    Validate dataframe against SDTM schema using pandera.
    
    Returns:
        Dict with validation results including errors and warnings.
    """
    schema = get_schema_for_domain(domain)
    
    try:
        validated_df = schema(df)
        return {
            "valid": True,
            "errors": [],
            "warnings": [],
            "row_count": len(validated_df),
            "domain": domain
        }
    except pa.errors.SchemaError as e:
        return {
            "valid": False,
            "errors": [str(e)],
            "warnings": [],
            "row_count": len(df),
            "domain": domain,
            "error_details": {
                "failure_cases": e.failure_cases.to_dict() if hasattr(e, 'failure_cases') else {},
                "schema_errors": str(e.schema_errors) if hasattr(e, 'schema_errors') else str(e)
            }
        }
    except Exception as e:
        return {
            "valid": False,
            "errors": [f"Unexpected validation error: {str(e)}"],
            "warnings": [],
            "row_count": len(df),
            "domain": domain
        }