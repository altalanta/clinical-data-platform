"""Leakage and batch diagnostics for the CP tox mini demo."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .fuse import FUSED_DATASET, NUMERIC_FEATURES
from .train import MODEL_PATH

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
LEAKAGE_JSON = REPORTS_DIR / "leakage.json"


@dataclass
class ProbeResult:
    auroc: float
    ap: float
    figure_path: Path


def _cross_validated_probe(features: pd.DataFrame, target: pd.Series, figure_name: str) -> ProbeResult:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=500, random_state=42, multi_class="auto"),
            ),
        ]
    )
    skf = StratifiedKFold(n_splits=min(4, len(np.unique(target))), shuffle=True, random_state=42)
    y_true = []
    y_proba = []

    for train_idx, test_idx in skf.split(features, target):
        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)
        y_true.append(y_test)
        y_proba.append(proba)

    y_true_arr = np.concatenate([y.to_numpy() for y in y_true])
    proba_arr = np.concatenate(y_proba, axis=0)

    encoder = LabelEncoder().fit(y_true_arr)
    y_encoded = encoder.transform(y_true_arr)

    if proba_arr.shape[1] == 2:
        auroc = metrics.roc_auc_score(y_encoded, proba_arr[:, 1])
        ap = metrics.average_precision_score(y_encoded, proba_arr[:, 1])
        fpr, tpr, _ = metrics.roc_curve(y_encoded, proba_arr[:, 1])
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(fpr, tpr, label=f"AUROC={auroc:.2f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(figure_name)
        ax.legend(loc="lower right")
    else:
        y_binarized = np.eye(len(encoder.classes_))[y_encoded]
        auroc = metrics.roc_auc_score(y_binarized, proba_arr, multi_class="ovr")
        ap = metrics.average_precision_score(y_binarized, proba_arr)
        fpr = dict()
        tpr = dict()
        for idx, label in enumerate(encoder.classes_):
            fpr[idx], tpr[idx], _ = metrics.roc_curve(y_binarized[:, idx], proba_arr[:, idx])
        fig, ax = plt.subplots(figsize=(4, 3))
        for idx, label in enumerate(encoder.classes_):
            ax.plot(fpr[idx], tpr[idx], label=f"{label}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(figure_name)
        ax.legend(loc="lower right")

    figure_path = FIGURES_DIR / f"{figure_name.lower().replace(' ', '_')}.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return ProbeResult(auroc=float(auroc), ap=float(ap), figure_path=figure_path)


def _permutation_test(df: pd.DataFrame, model_metrics: Dict[str, float], iterations: int = 100) -> Tuple[float, float]:
    payload = joblib.load(MODEL_PATH)
    model = payload["model"]
    features = payload["features"]

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    X_train = train_df[features]
    y_train = train_df["label"].astype(int).to_numpy()
    X_test = test_df[features]
    y_test = test_df["label"].astype(int).to_numpy()

    baseline_proba = model.predict_proba(X_test)[:, 1]
    observed_auroc = float(model_metrics.get("auroc", metrics.roc_auc_score(y_test, baseline_proba)))

    rng = np.random.default_rng(42)

    def _fit_and_score(labels: np.ndarray) -> float:
        if len(np.unique(labels)) < 2:
            return 0.5
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(max_iter=500, random_state=42),
                ),
            ]
        )
        pipeline.fit(X_train, labels)
        proba = pipeline.predict_proba(X_test)[:, 1]
        return float(metrics.roc_auc_score(y_test, proba))

    within_plate_scores = []
    for _ in range(iterations):
        permuted = y_train.copy()
        for plate in train_df["plate_id"].unique():
            mask = train_df["plate_id"] == plate
            permuted[mask] = rng.permutation(permuted[mask])
        within_plate_scores.append(_fit_and_score(permuted))

    full_shuffle_scores = []
    for _ in range(iterations):
        permuted = rng.permutation(y_train)
        full_shuffle_scores.append(_fit_and_score(permuted))

    p_value = (sum(score >= observed_auroc for score in within_plate_scores) + 1) / (len(within_plate_scores) + 1)
    baseline = float(np.mean(full_shuffle_scores)) if full_shuffle_scores else float("nan")
    return float(p_value), baseline


def run_diagnostics(iterations: int = 100) -> Dict[str, object]:
    if not FUSED_DATASET.exists():
        raise FileNotFoundError("Fused dataset missing; run fuse step first")
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model artifact missing; run train step first")

    df = pd.read_parquet(FUSED_DATASET)
    features = df[NUMERIC_FEATURES]

    plate_probe = _cross_validated_probe(features, df["plate_id"], "Leakage Probe Plate")
    layout_probe = _cross_validated_probe(features, df["well_row"], "Leakage Probe Layout")

    metrics_path = Path("reports/model_metrics.json")
    if metrics_path.exists():
        model_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        model_metrics = {}

    p_value, baseline = _permutation_test(df, model_metrics, iterations=iterations)

    if plate_probe.auroc >= 0.80 or p_value <= 0.05:
        risk = "high"
    elif plate_probe.auroc >= 0.65 or layout_probe.auroc >= 0.65:
        risk = "medium"
    else:
        risk = "low"

    RESULTS = {
        "plate_probe_auroc": round(plate_probe.auroc, 4),
        "plate_probe_ap": round(plate_probe.ap, 4),
        "layout_probe_auroc": round(layout_probe.auroc, 4),
        "layout_probe_ap": round(layout_probe.ap, 4),
        "perm_p_value": round(p_value, 4),
        "null_auroc_mean": round(baseline, 4),
        "risk_of_leakage": risk,
        "notes": f"Permutation N={iterations}, within-plate shuffle; null mean AUROC={baseline:.3f}",
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    LEAKAGE_JSON.write_text(json.dumps(RESULTS, indent=2) + "\n", encoding="utf-8")
    return RESULTS


__all__ = ["run_diagnostics", "LEAKAGE_JSON"]
