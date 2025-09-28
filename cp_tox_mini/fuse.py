"""Fuse modalities into a modeling table with deterministic splits."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split

from .features import CP_FEATURES, CHEM_FEATURES
from .io import PROCESSED_DIR

FUSED_DATASET = PROCESSED_DIR / "fused.parquet"
FEATURE_META = PROCESSED_DIR / "feature_columns.json"

NUMERIC_FEATURES = [
    "dose_nM",
    "readout",
    "well_row_index",
    "well_col",
] + CP_FEATURES + CHEM_FEATURES


def fuse_modalities(test_size: float = 0.25) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features_path = PROCESSED_DIR / "features.parquet"
    if not features_path.exists():
        raise FileNotFoundError("Features not found; run features step first")

    df = pd.read_parquet(features_path)
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        stratify=df["label"],
        random_state=42,
    )

    df.loc[:, "split"] = "train"
    df.loc[test_idx, "split"] = "test"
    df = df.sort_values(["split", "sample_id"]).reset_index(drop=True)
    df.to_parquet(FUSED_DATASET, index=False)

    metadata = {
        "feature_columns": NUMERIC_FEATURES,
        "label_column": "label",
        "split_column": "split",
    }
    FEATURE_META.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return FUSED_DATASET


__all__ = ["fuse_modalities", "FUSED_DATASET", "FEATURE_META", "NUMERIC_FEATURES"]
