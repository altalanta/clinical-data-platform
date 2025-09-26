from __future__ import annotations

from pathlib import Path

import great_expectations as ge
import pandas as pd


def dm_suite(df: pd.DataFrame) -> ge.core.ExpectationSuiteValidationResult:
    ge_df = ge.from_pandas(df)
    ge_df.expect_column_values_to_not_be_null("STUDYID")
    ge_df.expect_column_values_to_not_be_null("SUBJID")
    ge_df.expect_column_values_to_be_of_type("AGE", "float64", result_format="SUMMARY")
    ge_df.expect_column_values_to_be_in_set("SEX", ["M", "F", None])
    ge_df.expect_column_values_to_be_in_set("ARM", ["PLACEBO", "ACTIVE", None])
    return ge_df.validate()


def ae_suite(df: pd.DataFrame) -> ge.core.ExpectationSuiteValidationResult:
    ge_df = ge.from_pandas(df)
    ge_df.expect_column_values_to_not_be_null("STUDYID")
    ge_df.expect_column_values_to_not_be_null("SUBJID")
    ge_df.expect_column_values_to_be_in_set("AESEV", ["MILD", "MODERATE", "SEVERE", "SERIOUS", None])
    ge_df.expect_column_values_to_be_between("AESTDTC", min_value=None, max_value=None)
    ge_df.expect_column_values_to_match_regex("AEOUT", ".*|^$")
    return ge_df.validate()


def run_all(standardized_dir: str | Path) -> dict[str, bool]:
    std = Path(standardized_dir)
    dm = pd.read_parquet(std / "DM.parquet")
    ae = pd.read_parquet(std / "AE.parquet")
    r1 = dm_suite(dm)
    r2 = ae_suite(ae)
    return {"DM": r1.success, "AE": r2.success}

