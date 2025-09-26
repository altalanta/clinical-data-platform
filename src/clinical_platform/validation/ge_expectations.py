"""
Great Expectations configuration for clinical data validation.
"""

import os
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import great_expectations as ge
from great_expectations.core.batch import BatchRequest
from great_expectations.data_context import BaseDataContext
from great_expectations.exceptions import DataContextError


class ClinicalDataContext:
    """
    Wrapper for Great Expectations DataContext with clinical-specific configurations.
    """
    
    def __init__(self, context_root_dir: str = None):
        if context_root_dir is None:
            context_root_dir = os.getenv('GE_DATA_CONTEXT_ROOT_DIR', 'configs/great_expectations')
        
        self.context_root_dir = Path(context_root_dir)
        self._context = None
        self._initialize_context()
    
    def _initialize_context(self):
        """Initialize Great Expectations context."""
        try:
            if not self.context_root_dir.exists():
                self.context_root_dir.mkdir(parents=True, exist_ok=True)
                
            # Try to get existing context
            try:
                self._context = ge.data_context.DataContext(context_root_dir=str(self.context_root_dir))
            except DataContextError:
                # Initialize new context if none exists
                self._context = ge.data_context.DataContext.create(context_root_dir=str(self.context_root_dir))
                self._create_clinical_expectations()
                
        except Exception as e:
            # Fallback to ephemeral context for testing
            self._context = ge.data_context.EphemeralDataContext()
            self._create_clinical_expectations()
    
    def _create_clinical_expectations(self):
        """Create clinical data expectations suites."""
        expectations_dir = self.context_root_dir / "expectations"
        expectations_dir.mkdir(exist_ok=True)
        
        # Create expectation suites for each SDTM domain
        domains = ["DM", "AE", "LB", "VS", "EX"]
        
        for domain in domains:
            suite_name = f"clinical_data_{domain.lower()}_suite"
            try:
                self._context.get_expectation_suite(suite_name)
            except ge.exceptions.DataContextError:
                # Create new suite
                suite = self._context.create_expectation_suite(suite_name)
                self._add_domain_expectations(suite, domain)
    
    def _add_domain_expectations(self, suite, domain: str):
        """Add domain-specific expectations to suite."""
        if domain == "DM":
            suite.add_expectation(
                ge.expectations.ExpectColumnToExist(column="STUDYID")
            )
            suite.add_expectation(
                ge.expectations.ExpectColumnToExist(column="SUBJID")
            )
            suite.add_expectation(
                ge.expectations.ExpectColumnValuesToNotBeNull(column="STUDYID")
            )
            suite.add_expectation(
                ge.expectations.ExpectColumnValuesToNotBeNull(column="SUBJID")
            )
            suite.add_expectation(
                ge.expectations.ExpectColumnValuesToBeInSet(
                    column="SEX", 
                    value_set=["M", "F", "U"]
                )
            )
            suite.add_expectation(
                ge.expectations.ExpectColumnValuesToBeBetween(
                    column="AGE",
                    min_value=0,
                    max_value=150
                )
            )
            
        elif domain == "AE":
            suite.add_expectation(
                ge.expectations.ExpectColumnToExist(column="STUDYID")
            )
            suite.add_expectation(
                ge.expectations.ExpectColumnToExist(column="SUBJID")
            )
            suite.add_expectation(
                ge.expectations.ExpectColumnValuesToBeInSet(
                    column="AESEV",
                    value_set=["MILD", "MODERATE", "SEVERE"]
                )
            )
            suite.add_expectation(
                ge.expectations.ExpectColumnValuesToBeInSet(
                    column="AESER",
                    value_set=[True, False]
                )
            )
            
        elif domain == "LB":
            suite.add_expectation(
                ge.expectations.ExpectColumnToExist(column="STUDYID")
            )
            suite.add_expectation(
                ge.expectations.ExpectColumnToExist(column="SUBJID")
            )
            suite.add_expectation(
                ge.expectations.ExpectColumnToExist(column="LBTESTCD")
            )
            suite.add_expectation(
                ge.expectations.ExpectColumnValuesToBeOfType(
                    column="LBORRES",
                    type_="float"
                )
            )
            
        elif domain == "VS":
            suite.add_expectation(
                ge.expectations.ExpectColumnToExist(column="STUDYID")
            )
            suite.add_expectation(
                ge.expectations.ExpectColumnToExist(column="SUBJID")
            )
            suite.add_expectation(
                ge.expectations.ExpectColumnToExist(column="VSTESTCD")
            )
            
        elif domain == "EX":
            suite.add_expectation(
                ge.expectations.ExpectColumnToExist(column="STUDYID")
            )
            suite.add_expectation(
                ge.expectations.ExpectColumnToExist(column="SUBJID")
            )
            suite.add_expectation(
                ge.expectations.ExpectColumnValuesToBeOfType(
                    column="EXDOSE",
                    type_="float"
                )
            )
    
    def validate_dataframe(self, df: pd.DataFrame, domain: str) -> Dict[str, Any]:
        """
        Validate dataframe using Great Expectations.
        
        Args:
            df: DataFrame to validate
            domain: SDTM domain (DM, AE, LB, VS, EX)
            
        Returns:
            Dict with validation results
        """
        suite_name = f"clinical_data_{domain.lower()}_suite"
        
        try:
            # Create validator from dataframe
            validator = self._context.get_validator(
                batch_request=BatchRequest(
                    datasource_name="default_pandas_datasource",
                    data_connector_name="default_runtime_data_connector",
                    data_asset_name="clinical_data",
                    runtime_parameters={"batch_data": df},
                    batch_identifiers={"default_identifier_name": f"{domain}_data"}
                ),
                expectation_suite_name=suite_name
            )
            
            # Run validation
            results = validator.validate()
            
            # Process results
            validation_results = {
                "valid": results.success,
                "domain": domain,
                "total_expectations": len(results.results),
                "successful_expectations": sum(1 for r in results.results if r.success),
                "failed_expectations": [
                    {
                        "expectation_type": r.expectation_config.expectation_type,
                        "column": r.expectation_config.kwargs.get("column"),
                        "result": r.result
                    }
                    for r in results.results if not r.success
                ],
                "statistics": results.statistics if hasattr(results, 'statistics') else {},
                "meta": {
                    "great_expectations_version": ge.__version__,
                    "row_count": len(df),
                    "column_count": len(df.columns) if not df.empty else 0
                }
            }
            
            return validation_results
            
        except Exception as e:
            return {
                "valid": False,
                "domain": domain,
                "error": f"Great Expectations validation failed: {str(e)}",
                "total_expectations": 0,
                "successful_expectations": 0,
                "failed_expectations": [],
                "meta": {
                    "row_count": len(df),
                    "column_count": len(df.columns) if not df.empty else 0
                }
            }
    
    @property
    def context(self) -> BaseDataContext:
        """Get the underlying Great Expectations context."""
        return self._context


def validate_with_great_expectations(df: pd.DataFrame, domain: str) -> Dict[str, Any]:
    """
    Convenience function to validate DataFrame with Great Expectations.
    
    Args:
        df: DataFrame to validate
        domain: SDTM domain
        
    Returns:
        Validation results
    """
    clinical_context = ClinicalDataContext()
    return clinical_context.validate_dataframe(df, domain)


def create_validation_report(validation_results: List[Dict[str, Any]], 
                           output_path: str = None) -> Dict[str, Any]:
    """
    Create a comprehensive validation report from multiple validation results.
    
    Args:
        validation_results: List of validation result dictionaries
        output_path: Optional path to save report as JSON
        
    Returns:
        Comprehensive validation report
    """
    if not validation_results:
        return {"error": "No validation results provided"}
    
    report = {
        "validation_summary": {
            "total_domains": len(validation_results),
            "successful_validations": sum(1 for r in validation_results if r.get("valid", False)),
            "failed_validations": sum(1 for r in validation_results if not r.get("valid", False)),
            "total_rows_validated": sum(r.get("meta", {}).get("row_count", 0) for r in validation_results)
        },
        "domain_results": validation_results,
        "recommendations": []
    }
    
    # Add recommendations based on failures
    for result in validation_results:
        if not result.get("valid", False):
            domain = result.get("domain", "Unknown")
            if "failed_expectations" in result:
                for failure in result["failed_expectations"]:
                    report["recommendations"].append({
                        "domain": domain,
                        "issue": failure.get("expectation_type", "Unknown"),
                        "column": failure.get("column"),
                        "recommendation": f"Review {failure.get('column', 'data')} in {domain} domain"
                    })
    
    # Save report if path provided
    if output_path:
        import json
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    return report