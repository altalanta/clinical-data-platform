from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from clinical_platform.analytics.feature_eng import subject_level_features
from clinical_platform.config import get_config
from clinical_platform.ml.registry import setup_mlflow


def batch_score(data_dir: str | Path | None = None, model_uri: str | None = None, out_path: str = "predictions.parquet"):
    cfg = get_config()
    setup_mlflow(cfg.mlflow.tracking_uri)
    feats = subject_level_features(data_dir)
    X = feats[["AGE", "AE_COUNT", "SEVERE_AE_COUNT"]].fillna(0).astype(float).values
    model = mlflow.sklearn.load_model(model_uri or "models:/cdp_logreg/latest")
    prob = model.predict_proba(X)[:, 1]
    out_df = feats[["STUDYID", "SUBJID"]].copy()
    out_df["RISK"] = prob
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--model_uri", type=str, default=None)
    parser.add_argument("--out", type=str, default="data/predictions.parquet")
    args = parser.parse_args()
    df = batch_score(args.data, args.model_uri, args.out)
    print(df.head().to_dict(orient="records"))

