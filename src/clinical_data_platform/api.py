from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from .compliance.logging_config import configure_logging_from_env
from .compliance.read_only import ReadOnlyMiddleware

configure_logging_from_env()

app = FastAPI(title="Clinical Data Platform API")
app.add_middleware(ReadOnlyMiddleware)


class PredictRequest(BaseModel):
    features: List[float]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, list[float]]:
    root = Path(__file__).resolve().parents[2]
    model_dir = root / "docs" / "assets" / "demo" / "mlflow" / "model"

    if (model_dir / "MLmodel").exists():
        try:
            import mlflow.pyfunc

            model = mlflow.pyfunc.load_model(model_dir)
            prediction = model.predict(np.array([request.features])).tolist()
        except Exception:  # fall back to deterministic default
            prediction = [0.0]
    else:
        prediction = [0.0]

    return {"prediction": prediction}
