"""
Unified data validator combining pandera and Great Expectations for robust validation.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd

from ..logging_utils import get_logger, set_data_lineage_id
from .pandera_schemas import validate_dataframe as pandera_validate
from .ge_expectations import validate_with_great_expectations


class DataValidator:
    """
    Unified data validator that combines pandera and Great Expectations
    for comprehensive clinical data validation.
    """
    
    def __init__(self, 
                 enable_pandera: bool = True,
                 enable_great_expectations: bool = True,
                 fail_fast: bool = False):
        """
        Initialize the data validator.
        
        Args:
            enable_pandera: Whether to run pandera validation
            enable_great_expectations: Whether to run Great Expectations validation
            fail_fast: Whether to stop on first validation failure
        """
        self.enable_pandera = enable_pandera
        self.enable_great_expectations = enable_great_expectations
        self.fail_fast = fail_fast
        self.logger = get_logger("data_validator")
        
    def validate_domain_data(self, 
                           df: pd.DataFrame, 
                           domain: str,
                           data_lineage_id: str = None) -> Dict[str, Any]:
        """
        Validate a single domain dataset using both validation frameworks.
        
        Args:
            df: DataFrame to validate
            domain: SDTM domain (DM, AE, LB, VS, EX)
            data_lineage_id: Optional lineage tracking ID
            
        Returns:
            Comprehensive validation results
        """
        if data_lineage_id:
            set_data_lineage_id(data_lineage_id)
        
        self.logger.info(
            "Starting validation",
            domain=domain,
            row_count=len(df),
            column_count=len(df.columns) if not df.empty else 0
        )
        
        validation_results = {
            "domain": domain,
            "input_summary": {
                "row_count": len(df),
                "column_count": len(df.columns) if not df.empty else 0,
                "columns": list(df.columns) if not df.empty else [],
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024 if not df.empty else 0
            },
            "pandera_results": None,
            "great_expectations_results": None,
            "overall_valid": True,
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Run pandera validation
        if self.enable_pandera:
            try:
                self.logger.info("Running pandera validation", domain=domain)
                pandera_results = pandera_validate(df, domain)
                validation_results["pandera_results"] = pandera_results
                
                if not pandera_results["valid"]:
                    validation_results["overall_valid"] = False
                    validation_results["errors"].extend(pandera_results["errors"])
                    
                    if self.fail_fast:
                        self.logger.error("Pandera validation failed, stopping", domain=domain)
                        return validation_results
                        
            except Exception as e:
                error_msg = f"Pandera validation error: {str(e)}"
                validation_results["errors"].append(error_msg)
                validation_results["overall_valid"] = False
                self.logger.error("Pandera validation exception", domain=domain, error=str(e))
                
                if self.fail_fast:
                    return validation_results
        
        # Run Great Expectations validation
        if self.enable_great_expectations:
            try:
                self.logger.info("Running Great Expectations validation", domain=domain)
                ge_results = validate_with_great_expectations(df, domain)
                validation_results["great_expectations_results"] = ge_results
                
                if not ge_results["valid"]:
                    validation_results["overall_valid"] = False
                    if "error" in ge_results:
                        validation_results["errors"].append(ge_results["error"])
                    
                    # Add failed expectations to errors
                    for failure in ge_results.get("failed_expectations", []):
                        error_msg = f"GE: {failure['expectation_type']} failed for column {failure.get('column', 'N/A')}"
                        validation_results["errors"].append(error_msg)
                        
            except Exception as e:
                error_msg = f"Great Expectations validation error: {str(e)}"
                validation_results["errors"].append(error_msg)
                validation_results["overall_valid"] = False
                self.logger.error("Great Expectations validation exception", domain=domain, error=str(e))
        
        # Generate recommendations
        validation_results["recommendations"] = self._generate_recommendations(validation_results)
        
        # Log final results
        if validation_results["overall_valid"]:
            self.logger.info(
                "Validation completed successfully",
                domain=domain,
                pandera_valid=validation_results["pandera_results"]["valid"] if validation_results["pandera_results"] else "N/A",
                ge_valid=validation_results["great_expectations_results"]["valid"] if validation_results["great_expectations_results"] else "N/A"
            )
        else:
            self.logger.error(
                "Validation failed",
                domain=domain,
                error_count=len(validation_results["errors"]),
                errors=validation_results["errors"][:3]  # Log first 3 errors
            )
        
        return validation_results
    
    def validate_multiple_domains(self, 
                                 datasets: Dict[str, pd.DataFrame],
                                 data_lineage_id: str = None) -> Dict[str, Any]:
        """
        Validate multiple domain datasets.
        
        Args:
            datasets: Dict mapping domain names to DataFrames
            data_lineage_id: Optional lineage tracking ID
            
        Returns:
            Comprehensive validation results for all domains
        """
        if data_lineage_id:
            set_data_lineage_id(data_lineage_id)
        
        self.logger.info("Starting multi-domain validation", domains=list(datasets.keys()))
        
        all_results = {
            "validation_summary": {
                "total_domains": len(datasets),
                "domains_validated": [],
                "successful_validations": 0,
                "failed_validations": 0,
                "total_rows": sum(len(df) for df in datasets.values()),
                "overall_valid": True
            },
            "domain_results": {},
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
        
        for domain, df in datasets.items():
            try:
                domain_results = self.validate_domain_data(df, domain, data_lineage_id)
                all_results["domain_results"][domain] = domain_results
                all_results["validation_summary"]["domains_validated"].append(domain)
                
                if domain_results["overall_valid"]:
                    all_results["validation_summary"]["successful_validations"] += 1
                else:
                    all_results["validation_summary"]["failed_validations"] += 1
                    all_results["validation_summary"]["overall_valid"] = False
                    all_results["errors"].extend(
                        [f"{domain}: {error}" for error in domain_results["errors"]]
                    )
                
                all_results["recommendations"].extend(domain_results["recommendations"])
                
                if self.fail_fast and not domain_results["overall_valid"]:
                    self.logger.error("Stopping validation due to failure", domain=domain)
                    break
                    
            except Exception as e:
                error_msg = f"Failed to validate domain {domain}: {str(e)}"
                all_results["errors"].append(error_msg)
                all_results["validation_summary"]["failed_validations"] += 1
                all_results["validation_summary"]["overall_valid"] = False
                self.logger.error("Domain validation exception", domain=domain, error=str(e))
                
                if self.fail_fast:
                    break
        
        self.logger.info(
            "Multi-domain validation completed",
            successful=all_results["validation_summary"]["successful_validations"],
            failed=all_results["validation_summary"]["failed_validations"],
            overall_valid=all_results["validation_summary"]["overall_valid"]
        )
        
        return all_results
    
    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        if not validation_results["overall_valid"]:
            recommendations.append(
                f"Review and fix validation errors in {validation_results['domain']} domain before proceeding"
            )
        
        # Pandera-specific recommendations
        pandera_results = validation_results.get("pandera_results")
        if pandera_results and not pandera_results["valid"]:
            recommendations.append(
                "Check data types and format constraints - pandera validation identified schema violations"
            )
            
            if "error_details" in pandera_results and "failure_cases" in pandera_results["error_details"]:
                recommendations.append(
                    "Review specific data values that failed validation checks"
                )
        
        # Great Expectations-specific recommendations
        ge_results = validation_results.get("great_expectations_results")
        if ge_results and not ge_results["valid"]:
            failed_count = len(ge_results.get("failed_expectations", []))
            if failed_count > 0:
                recommendations.append(
                    f"Address {failed_count} failed data quality expectations"
                )
        
        return recommendations
    
    def save_validation_report(self, 
                             validation_results: Dict[str, Any], 
                             output_path: Union[str, Path]) -> None:
        """
        Save validation results to a JSON report file.
        
        Args:
            validation_results: Results from validation
            output_path: Path to save the report
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        enhanced_results = {
            **validation_results,
            "report_metadata": {
                "validator_config": {
                    "pandera_enabled": self.enable_pandera,
                    "great_expectations_enabled": self.enable_great_expectations,
                    "fail_fast": self.fail_fast
                },
                "generated_at": pd.Timestamp.now().isoformat()
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(enhanced_results, f, indent=2, default=str)
        
        self.logger.info("Validation report saved", output_path=str(output_file))


def create_validator(pandera_enabled: bool = None,
                    ge_enabled: bool = None,
                    fail_fast: bool = False) -> DataValidator:
    """
    Factory function to create DataValidator with environment-based defaults.
    
    Args:
        pandera_enabled: Override for pandera validation (defaults to env var)
        ge_enabled: Override for GE validation (defaults to env var)
        fail_fast: Whether to stop on first failure
        
    Returns:
        Configured DataValidator instance
    """
    import os
    
    if pandera_enabled is None:
        pandera_enabled = os.getenv('PANDERA_VALIDATION_ENABLED', 'true').lower() == 'true'
    
    if ge_enabled is None:
        ge_enabled = os.getenv('GE_VALIDATION_ENABLED', 'true').lower() == 'true'
    
    return DataValidator(
        enable_pandera=pandera_enabled,
        enable_great_expectations=ge_enabled,
        fail_fast=fail_fast
    )