import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets" / "demo" / "mlflow"
ASSETS.mkdir(parents=True, exist_ok=True)
MLRUNS = ROOT / "mlruns"
MLRUNS.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(MLRUNS.as_uri())
mlflow.set_experiment("demo")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=400)
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)
predictions = probabilities.argmax(axis=1)

metrics = {
    "accuracy": float(accuracy_score(y_test, predictions)),
    "log_loss": float(log_loss(y_test, probabilities)),
}
conf_matrix = confusion_matrix(y_test, predictions)

with mlflow.start_run(run_name="demo_run"):
    mlflow.log_params({"model": "log_reg", "dataset": "iris"})
    for key, value in metrics.items():
        mlflow.log_metric(key, value)

    model_dir = ASSETS / "model"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    mlflow.sklearn.save_model(model, path=str(model_dir))

    plt.figure(figsize=(4, 3))
    plt.imshow(conf_matrix, cmap="Blues")
    plt.title("Confusion Matrix (iris)")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(ASSETS / "confusion_matrix.png", bbox_inches="tight")
    plt.close()

with open(ASSETS / "metrics.json", "w", encoding="utf-8") as fh:
    json.dump({"metrics": metrics, "mlruns_path": str(MLRUNS)}, fh, indent=2)

print("MLflow demo artifacts ->", ASSETS)
