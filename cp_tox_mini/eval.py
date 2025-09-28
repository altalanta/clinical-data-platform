"""Model evaluation and figure generation."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.calibration import calibration_curve

from .fuse import FUSED_DATASET
from .io import PROCESSED_DIR
from .train import MODEL_PATH

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_JSON = REPORTS_DIR / "model_metrics.json"


def _expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.digitize(probs, bin_edges, right=True)
    total = len(probs)
    ece = 0.0
    for idx in range(1, bins + 1):
        mask = indices == idx
        if not np.any(mask):
            continue
        bin_prob = probs[mask].mean()
        bin_true = labels[mask].mean()
        weight = mask.sum() / total
        ece += weight * abs(bin_prob - bin_true)
    return float(ece)


def _write_figure(path: Path, fig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def evaluate_model() -> Dict[str, float]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model artifact missing; run train step first")
    if not FUSED_DATASET.exists():
        raise FileNotFoundError("Fused dataset missing; run fuse step first")

    payload = joblib.load(MODEL_PATH)
    model = payload["model"]
    feature_columns = payload["features"]

    df = pd.read_parquet(FUSED_DATASET)
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    X_test = test_df[feature_columns]
    y_test = test_df["label"].astype(int).to_numpy()
    proba = model.predict_proba(X_test)[:, 1]

    auroc = metrics.roc_auc_score(y_test, proba)
    ap = metrics.average_precision_score(y_test, proba)

    fpr, tpr, _ = metrics.roc_curve(y_test, proba)
    precision, recall, _ = metrics.precision_recall_curve(y_test, proba)

    fig_roc, ax_roc = plt.subplots(figsize=(4, 3))
    ax_roc.plot(fpr, tpr, label=f"AUROC={auroc:.2f}")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend(loc="lower right")
    _write_figure(FIGURES_DIR / "roc.png", fig_roc)

    fig_pr, ax_pr = plt.subplots(figsize=(4, 3))
    ax_pr.plot(recall, precision, label=f"AP={ap:.2f}")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall")
    ax_pr.legend(loc="lower left")
    _write_figure(FIGURES_DIR / "pr.png", fig_pr)

    prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10, strategy="uniform")
    fig_cal, ax_cal = plt.subplots(figsize=(4, 3))
    ax_cal.plot(prob_pred, prob_true, marker="o", label="Calibration")
    ax_cal.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect")
    ax_cal.set_xlabel("Predicted probability")
    ax_cal.set_ylabel("Observed frequency")
    ax_cal.set_title("Calibration")
    ax_cal.legend(loc="upper left")
    _write_figure(FIGURES_DIR / "calibration.png", fig_cal)

    brier = metrics.brier_score_loss(y_test, proba)
    ece = _expected_calibration_error(proba, y_test)

    metrics_payload = {
        "auroc": round(float(auroc), 4),
        "ap": round(float(ap), 4),
        "brier": round(float(brier), 4),
        "ece": round(float(ece), 4),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")
    return metrics_payload


__all__ = ["evaluate_model", "REPORTS_DIR", "FIGURES_DIR", "METRICS_JSON"]
