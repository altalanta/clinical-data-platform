from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from clinical_platform.analytics.feature_eng import subject_level_features
from clinical_platform.config import get_config
from clinical_platform.ml.registry import infer_signature, mlflow, setup_mlflow


def train(data_dir: str | Path | None = None, out_dir: str | Path | None = None, seed: int = 42):
    cfg = get_config()
    setup_mlflow(cfg.mlflow.tracking_uri)
    feats = subject_level_features(data_dir)

    # Target: simple proxy for dropout/severe AE risk
    y = (feats["SEVERE_AE_COUNT"] > 0).astype(int).values
    X = feats[["AGE", "AE_COUNT", "SEVERE_AE_COUNT"]].fillna(0).astype(float).values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=seed)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200))])

    with mlflow.start_run(run_name="cdp_logreg"):
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, prob)
        ap = average_precision_score(y_val, prob)
        mlflow.log_metric("val_auc", float(auc))
        mlflow.log_metric("val_ap", float(ap))
        sig = infer_signature(X_train, prob)
        mlflow.sklearn.log_model(pipe, "model", signature=sig)

    out = Path(out_dir or "models")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "val_probs.npy", prob)
    return float(auc), float(ap)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--out", type=str, default="models")
    args = parser.parse_args()
    auc, ap = train(args.data, args.out)
    print({"val_auc": auc, "val_ap": ap})

