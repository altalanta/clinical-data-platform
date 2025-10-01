"""Public synthetic CDM adapter for offline demo purposes.

This module provides a synthetic OMOP/CDM-like dataset that mirrors the structure
and relationships of real clinical data without containing any PHI. It generates
deterministic synthetic data for persons, visits, conditions, and measurements
that can be used to run the complete data platform pipeline offline.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PublicCDMAdapter:
    """Synthetic OMOP CDM adapter for public demo.
    
    Generates realistic but synthetic clinical data following OMOP CDM patterns.
    Designed to be small (<50MB), deterministic, and representative of real
    clinical data relationships and distributions.
    """

    def __init__(self, seed: int = 42, fast_mode: bool = False):
        """Initialize the adapter.
        
        Args:
            seed: Random seed for deterministic generation
            fast_mode: If True, generate smaller datasets for CI/testing
        """
        self.seed = seed
        self.fast_mode = fast_mode
        self.rng = np.random.RandomState(seed)
        
        # Scale factors for fast mode
        if fast_mode:
            self.n_persons = 1000
            self.n_visits_per_person = (1, 5)
            self.n_conditions_per_visit = (0, 3)
            self.n_measurements_per_visit = (1, 8)
        else:
            self.n_persons = 5000
            self.n_visits_per_person = (1, 10)
            self.n_conditions_per_visit = (0, 5)
            self.n_measurements_per_visit = (2, 15)

    def fetch(self, target_dir: str | Path) -> Path:
        """Generate synthetic CDM data and save to CSV files.
        
        Args:
            target_dir: Directory to save CSV files
            
        Returns:
            Path to the created directory
        """
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating synthetic CDM data (seed={self.seed}, fast={self.fast_mode})")
        
        # Generate core tables
        persons_df = self._generate_persons()
        visits_df = self._generate_visits(persons_df)
        conditions_df = self._generate_conditions(visits_df)
        measurements_df = self._generate_measurements(visits_df)
        
        # Save as CSV
        persons_df.to_csv(target_dir / "person.csv", index=False)
        visits_df.to_csv(target_dir / "visit_occurrence.csv", index=False)
        conditions_df.to_csv(target_dir / "condition_occurrence.csv", index=False)
        measurements_df.to_csv(target_dir / "measurement.csv", index=False)
        
        # Generate manifest
        manifest = {
            "generator": "PublicCDMAdapter",
            "version": "1.0",
            "seed": self.seed,
            "fast_mode": self.fast_mode,
            "generated_at": datetime.now().isoformat(),
            "tables": {
                "person": {"rows": len(persons_df), "file": "person.csv"},
                "visit_occurrence": {"rows": len(visits_df), "file": "visit_occurrence.csv"},
                "condition_occurrence": {"rows": len(conditions_df), "file": "condition_occurrence.csv"},
                "measurement": {"rows": len(measurements_df), "file": "measurement.csv"},
            },
            "total_size_mb": sum(
                (target_dir / table["file"]).stat().st_size for table in manifest["tables"].values()
            ) / (1024 * 1024),
        }
        
        with open(target_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Generated {sum(t['rows'] for t in manifest['tables'].values())} total rows")
        logger.info(f"Dataset size: {manifest['total_size_mb']:.1f} MB")
        
        return target_dir

    def materialize_duckdb(self, db_path: str | Path, data_dir: str | Path) -> None:
        """Load CSV data into DuckDB and apply proper typing.
        
        Args:
            db_path: Path to DuckDB database file
            data_dir: Directory containing CSV files
        """
        try:
            import duckdb
        except ImportError:
            raise ImportError("duckdb is required. Install with: pip install duckdb")
        
        db_path = Path(db_path)
        data_dir = Path(data_dir)
        
        logger.info(f"Creating DuckDB at {db_path}")
        
        # Remove existing database
        if db_path.exists():
            db_path.unlink()
        
        # Connect and create tables
        conn = duckdb.connect(str(db_path))
        
        try:
            # Create person table
            conn.execute("""
                CREATE TABLE person AS 
                SELECT 
                    person_id::INTEGER as person_id,
                    gender_concept_id::INTEGER as gender_concept_id,
                    year_of_birth::INTEGER as year_of_birth,
                    month_of_birth::INTEGER as month_of_birth,
                    day_of_birth::INTEGER as day_of_birth,
                    birth_datetime::TIMESTAMP as birth_datetime,
                    race_concept_id::INTEGER as race_concept_id,
                    ethnicity_concept_id::INTEGER as ethnicity_concept_id
                FROM read_csv_auto(?)
            """, [str(data_dir / "person.csv")])
            
            # Create visit_occurrence table
            conn.execute("""
                CREATE TABLE visit_occurrence AS
                SELECT 
                    visit_occurrence_id::INTEGER as visit_occurrence_id,
                    person_id::INTEGER as person_id,
                    visit_concept_id::INTEGER as visit_concept_id,
                    visit_start_date::DATE as visit_start_date,
                    visit_start_datetime::TIMESTAMP as visit_start_datetime,
                    visit_end_date::DATE as visit_end_date,
                    visit_end_datetime::TIMESTAMP as visit_end_datetime,
                    visit_type_concept_id::INTEGER as visit_type_concept_id
                FROM read_csv_auto(?)
            """, [str(data_dir / "visit_occurrence.csv")])
            
            # Create condition_occurrence table
            conn.execute("""
                CREATE TABLE condition_occurrence AS
                SELECT 
                    condition_occurrence_id::INTEGER as condition_occurrence_id,
                    person_id::INTEGER as person_id,
                    condition_concept_id::INTEGER as condition_concept_id,
                    condition_start_date::DATE as condition_start_date,
                    condition_start_datetime::TIMESTAMP as condition_start_datetime,
                    condition_end_date::DATE as condition_end_date,
                    condition_end_datetime::TIMESTAMP as condition_end_datetime,
                    condition_type_concept_id::INTEGER as condition_type_concept_id,
                    visit_occurrence_id::INTEGER as visit_occurrence_id
                FROM read_csv_auto(?)
            """, [str(data_dir / "condition_occurrence.csv")])
            
            # Create measurement table
            conn.execute("""
                CREATE TABLE measurement AS
                SELECT 
                    measurement_id::INTEGER as measurement_id,
                    person_id::INTEGER as person_id,
                    measurement_concept_id::INTEGER as measurement_concept_id,
                    measurement_date::DATE as measurement_date,
                    measurement_datetime::TIMESTAMP as measurement_datetime,
                    measurement_type_concept_id::INTEGER as measurement_type_concept_id,
                    value_as_number::DOUBLE as value_as_number,
                    value_as_concept_id::INTEGER as value_as_concept_id,
                    unit_concept_id::INTEGER as unit_concept_id,
                    visit_occurrence_id::INTEGER as visit_occurrence_id
                FROM read_csv_auto(?)
            """, [str(data_dir / "measurement.csv")])
            
            # Create indexes for performance
            conn.execute("CREATE INDEX idx_person_id ON person(person_id)")
            conn.execute("CREATE INDEX idx_visit_person_id ON visit_occurrence(person_id)")
            conn.execute("CREATE INDEX idx_visit_id ON visit_occurrence(visit_occurrence_id)")
            conn.execute("CREATE INDEX idx_condition_person_id ON condition_occurrence(person_id)")
            conn.execute("CREATE INDEX idx_condition_visit_id ON condition_occurrence(visit_occurrence_id)")
            conn.execute("CREATE INDEX idx_measurement_person_id ON measurement(person_id)")
            conn.execute("CREATE INDEX idx_measurement_visit_id ON measurement(visit_occurrence_id)")
            
            logger.info("DuckDB tables created successfully")
            
        finally:
            conn.close()

    def run_transform(self, db_path: str | Path, profiles_dir: str | Path | None = None) -> Dict[str, Any]:
        """Run dbt transformations for public CDM.
        
        Args:
            db_path: Path to DuckDB database
            profiles_dir: Directory containing dbt profiles
            
        Returns:
            Dictionary with transformation results
        """
        logger.info("Running dbt transformations for public CDM")
        
        # Change to dbt directory
        dbt_dir = Path("dbt/clinical_dbt")
        if not dbt_dir.exists():
            raise FileNotFoundError(f"dbt directory not found: {dbt_dir}")
        
        # Prepare dbt command
        cmd = [
            sys.executable, "-m", "dbt", "run",
            "--vars", json.dumps({"adapter": "public_cdm"}),
            "--target", "public_cdm",
        ]
        
        if profiles_dir:
            cmd.extend(["--profiles-dir", str(profiles_dir)])
        
        # Set environment variables
        env = {
            "DBT_PUBLIC_CDM_PATH": str(Path(db_path).absolute()),
            **dict(os.environ)
        }
        
        try:
            result = subprocess.run(
                cmd,
                cwd=dbt_dir,
                capture_output=True,
                text=True,
                env=env,
                timeout=300,  # 5 minute timeout
            )
            
            if result.returncode == 0:
                logger.info("dbt transformations completed successfully")
                return {
                    "success": True,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            else:
                logger.error(f"dbt transformations failed: {result.stderr}")
                return {
                    "success": False,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                }
                
        except subprocess.TimeoutExpired:
            logger.error("dbt transformations timed out")
            return {
                "success": False,
                "error": "Timeout after 5 minutes",
            }
        except Exception as e:
            logger.error(f"Error running dbt: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def run_quality_checks(self, db_path: str | Path, data_dir: str | Path) -> Dict[str, Any]:
        """Run Great Expectations and pandera quality checks.
        
        Args:
            db_path: Path to DuckDB database
            data_dir: Directory containing source data
            
        Returns:
            Dictionary with quality check results
        """
        logger.info("Running quality checks")
        
        results = {
            "great_expectations": None,
            "pandera": None,
            "overall_success": False,
        }
        
        # Run Great Expectations
        try:
            ge_results = self._run_great_expectations(data_dir)
            results["great_expectations"] = ge_results
        except Exception as e:
            logger.error(f"Great Expectations failed: {e}")
            results["great_expectations"] = {"success": False, "error": str(e)}
        
        # Run pandera checks
        try:
            pandera_results = self._run_pandera_checks(data_dir)
            results["pandera"] = pandera_results
        except Exception as e:
            logger.error(f"pandera checks failed: {e}")
            results["pandera"] = {"success": False, "error": str(e)}
        
        # Overall success
        results["overall_success"] = (
            results["great_expectations"] and results["great_expectations"].get("success", False) and
            results["pandera"] and results["pandera"].get("success", False)
        )
        
        return results

    def report(self, output_dir: str | Path, data_dir: str | Path, 
              transform_results: Dict[str, Any], quality_results: Dict[str, Any]) -> None:
        """Generate artifacts and report for public demo.
        
        Args:
            output_dir: Directory to save artifacts
            data_dir: Directory containing source data
            transform_results: Results from dbt transformations
            quality_results: Results from quality checks
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating report in {output_dir}")
        
        # Load manifest
        with open(Path(data_dir) / "manifest.json") as f:
            manifest = json.load(f)
        
        # Generate metrics
        metrics = {
            "dataset": {
                "tables": manifest["tables"],
                "total_rows": sum(t["rows"] for t in manifest["tables"].values()),
                "size_mb": manifest["total_size_mb"],
                "generated_at": manifest["generated_at"],
            },
            "transformations": transform_results,
            "quality_checks": quality_results,
            "summary": {
                "data_generated": True,
                "transforms_successful": transform_results.get("success", False),
                "quality_checks_passed": quality_results.get("overall_success", False),
                "overall_success": (
                    transform_results.get("success", False) and 
                    quality_results.get("overall_success", False)
                ),
            },
        }
        
        # Save metrics
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Generate README
        readme_content = self._generate_readme(metrics)
        with open(output_dir / "README.md", "w") as f:
            f.write(readme_content)
        
        logger.info("Report generated successfully")

    def _generate_persons(self) -> pd.DataFrame:
        """Generate synthetic person records."""
        persons = []
        
        for person_id in range(1, self.n_persons + 1):
            # Generate birth date (ages 18-90)
            age = self.rng.randint(18, 91)
            birth_year = datetime.now().year - age
            birth_month = self.rng.randint(1, 13)
            birth_day = self.rng.randint(1, 29)  # Avoid month-end issues
            birth_date = datetime(birth_year, birth_month, birth_day)
            
            person = {
                "person_id": person_id,
                "gender_concept_id": self.rng.choice([8507, 8532], p=[0.51, 0.49]),  # F/M
                "year_of_birth": birth_year,
                "month_of_birth": birth_month,
                "day_of_birth": birth_day,
                "birth_datetime": birth_date.isoformat(),
                "race_concept_id": self.rng.choice([8527, 8516, 8515, 8557], p=[0.7, 0.15, 0.1, 0.05]),
                "ethnicity_concept_id": self.rng.choice([38003563, 38003564], p=[0.85, 0.15]),
            }
            persons.append(person)
        
        return pd.DataFrame(persons)

    def _generate_visits(self, persons_df: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic visit records."""
        visits = []
        visit_id = 1
        
        for _, person in persons_df.iterrows():
            person_id = person["person_id"]
            n_visits = self.rng.randint(*self.n_visits_per_person)
            
            # Generate visits over the past 2 years
            start_date = datetime.now() - timedelta(days=730)
            
            for _ in range(n_visits):
                # Random visit date
                days_offset = self.rng.randint(0, 730)
                visit_date = start_date + timedelta(days=days_offset)
                
                # Visit duration (1-5 days for inpatient, same day for outpatient)
                visit_concept_id = self.rng.choice([9201, 9202], p=[0.8, 0.2])  # Outpatient/Inpatient
                if visit_concept_id == 9202:  # Inpatient
                    duration = self.rng.randint(1, 6)
                else:  # Outpatient
                    duration = 0
                
                visit_end = visit_date + timedelta(days=duration)
                
                visit = {
                    "visit_occurrence_id": visit_id,
                    "person_id": person_id,
                    "visit_concept_id": visit_concept_id,
                    "visit_start_date": visit_date.date().isoformat(),
                    "visit_start_datetime": visit_date.isoformat(),
                    "visit_end_date": visit_end.date().isoformat(),
                    "visit_end_datetime": visit_end.isoformat(),
                    "visit_type_concept_id": 44818517,  # EHR
                }
                visits.append(visit)
                visit_id += 1
        
        return pd.DataFrame(visits)

    def _generate_conditions(self, visits_df: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic condition records."""
        conditions = []
        condition_id = 1
        
        # Common condition concept IDs (ICD-10/SNOMED)
        common_conditions = [
            401267002,  # Hypertension
            44054006,   # Diabetes
            13645005,   # COPD
            22298006,   # Myocardial infarction
            40481000119105,  # Depression
            56717001,   # Hypothyroidism
            195967001,  # Asthma
        ]
        
        for _, visit in visits_df.iterrows():
            n_conditions = self.rng.randint(*self.n_conditions_per_visit)
            
            for _ in range(n_conditions):
                condition_concept_id = self.rng.choice(common_conditions)
                
                condition = {
                    "condition_occurrence_id": condition_id,
                    "person_id": visit["person_id"],
                    "condition_concept_id": condition_concept_id,
                    "condition_start_date": visit["visit_start_date"],
                    "condition_start_datetime": visit["visit_start_datetime"],
                    "condition_end_date": visit["visit_end_date"],
                    "condition_end_datetime": visit["visit_end_datetime"],
                    "condition_type_concept_id": 32020,  # EHR
                    "visit_occurrence_id": visit["visit_occurrence_id"],
                }
                conditions.append(condition)
                condition_id += 1
        
        return pd.DataFrame(conditions)

    def _generate_measurements(self, visits_df: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic measurement records."""
        measurements = []
        measurement_id = 1
        
        # Common lab measurements with realistic ranges
        lab_measurements = [
            {"concept_id": 3000963, "name": "Hemoglobin", "mean": 13.5, "std": 2.0, "unit": 8713},
            {"concept_id": 3013682, "name": "WBC", "mean": 7.5, "std": 2.5, "unit": 8784},
            {"concept_id": 3004249, "name": "Glucose", "mean": 95, "std": 15, "unit": 8840},
            {"concept_id": 3027018, "name": "Creatinine", "mean": 1.0, "std": 0.3, "unit": 8840},
            {"concept_id": 3028437, "name": "Total Cholesterol", "mean": 190, "std": 40, "unit": 8840},
            {"concept_id": 3027114, "name": "Sodium", "mean": 140, "std": 3, "unit": 8753},
            {"concept_id": 3019550, "name": "Potassium", "mean": 4.0, "std": 0.5, "unit": 8753},
        ]
        
        for _, visit in visits_df.iterrows():
            n_measurements = self.rng.randint(*self.n_measurements_per_visit)
            
            selected_labs = self.rng.choice(lab_measurements, size=min(n_measurements, len(lab_measurements)), replace=False)
            
            for lab in selected_labs:
                # Generate realistic value with some outliers
                if self.rng.random() < 0.05:  # 5% outliers
                    value = self.rng.normal(lab["mean"], lab["std"] * 3)
                else:
                    value = self.rng.normal(lab["mean"], lab["std"])
                
                value = max(0, value)  # Ensure non-negative
                
                measurement = {
                    "measurement_id": measurement_id,
                    "person_id": visit["person_id"],
                    "measurement_concept_id": lab["concept_id"],
                    "measurement_date": visit["visit_start_date"],
                    "measurement_datetime": visit["visit_start_datetime"],
                    "measurement_type_concept_id": 44818701,  # Lab
                    "value_as_number": round(value, 2),
                    "value_as_concept_id": None,
                    "unit_concept_id": lab["unit"],
                    "visit_occurrence_id": visit["visit_occurrence_id"],
                }
                measurements.append(measurement)
                measurement_id += 1
        
        return pd.DataFrame(measurements)

    def _run_great_expectations(self, data_dir: Path) -> Dict[str, Any]:
        """Run Great Expectations validation."""
        try:
            import great_expectations as ge
        except ImportError:
            return {"success": False, "error": "great_expectations not installed"}
        
        # Simple validation suite
        results = {"success": True, "expectations": []}
        
        # Check person table
        person_df = pd.read_csv(data_dir / "person.csv")
        
        # Basic expectations
        expectations = [
            ("person_id", "not_null", len(person_df) == person_df["person_id"].notna().sum()),
            ("person_id", "unique", len(person_df) == person_df["person_id"].nunique()),
            ("year_of_birth", "range", person_df["year_of_birth"].between(1920, 2010).all()),
        ]
        
        for column, expectation, passed in expectations:
            results["expectations"].append({
                "column": column,
                "expectation": expectation,
                "passed": passed,
            })
            if not passed:
                results["success"] = False
        
        return results

    def _run_pandera_checks(self, data_dir: Path) -> Dict[str, Any]:
        """Run pandera schema validation."""
        try:
            import pandera as pa
        except ImportError:
            return {"success": False, "error": "pandera not installed"}
        
        # Define schemas
        person_schema = pa.DataFrameSchema({
            "person_id": pa.Column(int, checks=pa.Check.greater_than(0)),
            "gender_concept_id": pa.Column(int, checks=pa.Check.isin([8507, 8532])),
            "year_of_birth": pa.Column(int, checks=pa.Check.between(1920, 2010)),
        })
        
        try:
            # Validate person table
            person_df = pd.read_csv(data_dir / "person.csv")
            person_schema.validate(person_df, lazy=True)
            
            return {"success": True, "validated_tables": ["person"]}
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _generate_readme(self, metrics: Dict[str, Any]) -> str:
        """Generate README for artifacts."""
        return f"""# Public CDM Demo Artifacts

Generated on: {datetime.now().isoformat()}

## Dataset Summary

- **Total Tables**: {len(metrics['dataset']['tables'])}
- **Total Rows**: {metrics['dataset']['total_rows']:,}
- **Size**: {metrics['dataset']['size_mb']:.1f} MB
- **Status**: {'✅ Success' if metrics['summary']['overall_success'] else '❌ Failed'}

## Tables

| Table | Rows | File |
|-------|------|------|
{chr(10).join(f"| {name} | {info['rows']:,} | {info['file']} |" for name, info in metrics['dataset']['tables'].items())}

## Transformation Results

- **dbt Run**: {'✅ Success' if metrics['transformations'].get('success', False) else '❌ Failed'}

## Quality Checks

- **Great Expectations**: {'✅ Pass' if metrics['quality_checks'].get('great_expectations', {}).get('success', False) else '❌ Fail'}
- **pandera**: {'✅ Pass' if metrics['quality_checks'].get('pandera', {}).get('success', False) else '❌ Fail'}

## Files

- `metrics.json` - Detailed metrics and results
- `README.md` - This summary file

## Usage

This synthetic dataset can be used to:
1. Test the clinical data platform pipeline
2. Develop and validate transformations
3. Run quality checks and validations
4. Train and test ML models

**Note**: This data is completely synthetic and contains no PHI.
"""


# Add missing import
import os