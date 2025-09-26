"""
Command-line runner for data validation.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

from ..logging_utils import get_logger, configure_logging, set_data_lineage_id
from .validator import create_validator


def load_data_files(data_path: Path) -> Dict[str, pd.DataFrame]:
    """
    Load SDTM data files from directory.
    
    Args:
        data_path: Path to directory containing SDTM CSV or Parquet files
        
    Returns:
        Dict mapping domain names to DataFrames
    """
    logger = get_logger("data_loader")
    datasets = {}
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    # Look for CSV and Parquet files
    supported_extensions = ['.csv', '.parquet']
    data_files = []
    
    for ext in supported_extensions:
        data_files.extend(data_path.glob(f"*{ext}"))
    
    if not data_files:
        raise ValueError(f"No CSV or Parquet files found in {data_path}")
    
    logger.info(f"Found {len(data_files)} data files", path=str(data_path))
    
    for file_path in data_files:
        # Extract domain from filename (e.g., DM.csv -> DM)
        domain = file_path.stem.upper()
        
        # Only process known SDTM domains
        known_domains = ["DM", "AE", "LB", "VS", "EX"]
        if domain not in known_domains:
            logger.warning(f"Skipping unknown domain: {domain}", file=str(file_path))
            continue
        
        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                continue
            
            datasets[domain] = df
            logger.info(f"Loaded {domain}", rows=len(df), columns=len(df.columns), file=str(file_path))
            
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {str(e)}")
            raise
    
    if not datasets:
        raise ValueError("No valid SDTM domain files could be loaded")
    
    return datasets


def run_validation(data_path: str,
                  output_path: str = None,
                  pandera_enabled: bool = True,
                  ge_enabled: bool = True,
                  fail_fast: bool = False,
                  data_lineage_id: str = None) -> Dict:
    """
    Run comprehensive data validation on clinical datasets.
    
    Args:
        data_path: Path to directory containing data files
        output_path: Optional path to save validation report
        pandera_enabled: Whether to run pandera validation
        ge_enabled: Whether to run Great Expectations validation
        fail_fast: Whether to stop on first validation failure
        data_lineage_id: Optional data lineage tracking ID
        
    Returns:
        Validation results dictionary
    """
    # Configure logging
    configure_logging()
    logger = get_logger("validation_runner")
    
    if data_lineage_id:
        set_data_lineage_id(data_lineage_id)
    
    logger.info(
        "Starting data validation",
        data_path=data_path,
        output_path=output_path,
        pandera_enabled=pandera_enabled,
        ge_enabled=ge_enabled,
        fail_fast=fail_fast
    )
    
    try:
        # Load data files
        data_path_obj = Path(data_path)
        datasets = load_data_files(data_path_obj)
        
        # Create validator
        validator = create_validator(
            pandera_enabled=pandera_enabled,
            ge_enabled=ge_enabled,
            fail_fast=fail_fast
        )
        
        # Run validation
        validation_results = validator.validate_multiple_domains(datasets, data_lineage_id)
        
        # Save report if output path specified
        if output_path:
            validator.save_validation_report(validation_results, output_path)
        
        # Log summary
        summary = validation_results["validation_summary"]
        if summary["overall_valid"]:
            logger.info(
                "Validation completed successfully",
                successful_domains=summary["successful_validations"],
                total_domains=summary["total_domains"],
                total_rows=summary["total_rows"]
            )
        else:
            logger.error(
                "Validation failed",
                failed_domains=summary["failed_validations"],
                successful_domains=summary["successful_validations"],
                total_errors=len(validation_results["errors"])
            )
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Validation runner failed: {str(e)}")
        raise


def main():
    """Command-line interface for data validation."""
    parser = argparse.ArgumentParser(
        description="Clinical Data Platform - Data Validation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate data with both validators
  python -m clinical_platform.validation.runner --data data/sample_raw --output validation_report.json

  # Validate with only pandera (fast validation)
  python -m clinical_platform.validation.runner --data data/sample_raw --no-ge

  # Fail fast mode (stop on first error)
  python -m clinical_platform.validation.runner --data data/sample_raw --fail-fast

  # Validate specific directory with custom lineage ID
  python -m clinical_platform.validation.runner --data /path/to/data --lineage-id batch_001
        """
    )
    
    parser.add_argument(
        "--data",
        required=True,
        help="Path to directory containing SDTM data files (CSV or Parquet)"
    )
    
    parser.add_argument(
        "--output",
        help="Path to save validation report (JSON format)"
    )
    
    parser.add_argument(
        "--no-pandera",
        action="store_true",
        help="Disable pandera validation"
    )
    
    parser.add_argument(
        "--no-ge",
        action="store_true", 
        help="Disable Great Expectations validation"
    )
    
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop validation on first failure"
    )
    
    parser.add_argument(
        "--lineage-id",
        help="Data lineage tracking ID"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        import os
        os.environ['LOG_LEVEL'] = 'DEBUG'
    
    try:
        # Run validation
        results = run_validation(
            data_path=args.data,
            output_path=args.output,
            pandera_enabled=not args.no_pandera,
            ge_enabled=not args.no_ge,
            fail_fast=args.fail_fast,
            data_lineage_id=args.lineage_id
        )
        
        # Exit with appropriate code
        if results["validation_summary"]["overall_valid"]:
            print("✅ All validations passed!")
            sys.exit(0)
        else:
            print("❌ Validation failed!")
            print(f"Errors: {len(results['errors'])}")
            for error in results['errors'][:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(results['errors']) > 5:
                print(f"  ... and {len(results['errors']) - 5} more errors")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Validation runner failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()