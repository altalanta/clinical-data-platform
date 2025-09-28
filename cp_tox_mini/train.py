"""Training routines for the CP tox mini demo."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .fuse import FEATURE_META, FUSED_DATASET, NUMERIC_FEATURES
from .io import PROCESSED_DIR

MODEL_PATH = PROCESSED_DIR / "model.joblib"


def _load_training_data() -> Tuple[pd.DataFrame, pd.Series, list[str]]:
    if not FUSED_DATASET.exists():
        raise FileNotFoundError("Fused dataset missing; run fuse step first")
    df = pd.read_parquet(FUSED_DATASET)

    if FEATURE_META.exists():
        metadata = json.loads(FEATURE_META.read_text(encoding="utf-8"))
        feature_columns = metadata.get("feature_columns", NUMERIC_FEATURES)
    else:
        feature_columns = NUMERIC_FEATURES

    X_train = df.loc[df["split"] == "train", feature_columns]
    y_train = df.loc[df["split"] == "train", "label"].astype(int)
    return X_train, y_train, feature_columns


def train_model() -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train, y_train, feature_columns = _load_training_data()

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=500,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    joblib.dump({"model": pipeline, "features": feature_columns}, MODEL_PATH)
    return MODEL_PATH


__all__ = ["train_model", "MODEL_PATH"]
