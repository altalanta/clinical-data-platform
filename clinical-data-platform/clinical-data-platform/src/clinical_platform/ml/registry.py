from __future__ import annotations

import mlflow
from mlflow.models.signature import infer_signature


def setup_mlflow(tracking_uri: str | None = None) -> None:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


__all__ = ["mlflow", "infer_signature", "setup_mlflow"]

