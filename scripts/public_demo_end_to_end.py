#!/usr/bin/env python3
"""
Public demo end-to-end orchestration script.

Runs the complete public CDM demo pipeline:
fetch → materialize_duckdb → run_transform → run_quality_checks → report

This script demonstrates the full clinical data platform capabilities
using synthetic data that requires no PHI or external credentials.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from clinical_platform.data_adapters.public_cdm import PublicCDMAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main orchestration function."""
    parser = argparse.ArgumentParser(
        description="Run public CDM demo end-to-end pipeline"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run in fast mode with smaller datasets (for CI)"
    )
    parser.add_argument(
        "--step",
        choices=["fetch", "materialize", "transform", "validate", "report", "all"],
        default="all",
        help="Run specific step or all steps"
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/public_demo",
        help="Output directory for artifacts"
    )
    parser.add_argument(
        "--data-dir", 
        default="data/public_cdm",
        help="Directory for raw data files"
    )
    parser.add_argument(
        "--db-path",
        default="public_demo.duckdb",
        help="Path to DuckDB database file"
    )
    
    args = parser.parse_args()
    
    # Initialize adapter
    logger.info(f"Initializing PublicCDMAdapter (fast_mode={args.fast})")
    adapter = PublicCDMAdapter(seed=42, fast_mode=args.fast)
    
    try:
        if args.step in ["fetch", "all"]:
            logger.info("Step 1: Fetching/generating synthetic data...")
            adapter.fetch(target_dir=args.data_dir)
            logger.info("✓ Data fetched successfully")
        
        if args.step in ["materialize", "all"]:
            logger.info("Step 2: Materializing data in DuckDB...")
            adapter.materialize_duckdb(db_path=args.db_path)
            logger.info("✓ Data materialized successfully")
        
        if args.step in ["transform", "all"]:
            logger.info("Step 3: Running dbt transformations...")
            adapter.run_transform()
            logger.info("✓ Transformations completed successfully")
        
        if args.step in ["validate", "all"]:
            logger.info("Step 4: Running quality checks...")
            adapter.run_quality_checks()
            logger.info("✓ Quality checks completed successfully")
        
        if args.step in ["report", "all"]:
            logger.info("Step 5: Generating reports...")
            adapter.report(output_dir=args.output_dir)
            logger.info("✓ Reports generated successfully")
        
        logger.info("🎉 Public demo pipeline completed successfully!")
        logger.info(f"📁 Check artifacts in: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()