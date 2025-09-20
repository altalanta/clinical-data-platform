from __future__ import annotations

from typing import List, Optional

import duckdb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from clinical_platform.config import get_config


app = FastAPI(title="Clinical Data Platform API")


class Health(BaseModel):
    status: str


class ScoreRequest(BaseModel):
    AGE: float
    AE_COUNT: float
    SEVERE_AE_COUNT: float


class ScoreResponse(BaseModel):
    risk: float


@app.get("/health", response_model=Health)
def health():
    return Health(status="ok")


@app.get("/studies", response_model=List[str])
def list_studies():
    con = duckdb.connect(get_config().warehouse.duckdb_path)
    studies = [r[0] for r in con.execute("SELECT study_id FROM dim_study").fetchall()]
    return studies


@app.get("/subjects/{subject_id}")
def get_subject(subject_id: str):
    con = duckdb.connect(get_config().warehouse.duckdb_path)
    df = con.execute(
        "SELECT * FROM dim_subject WHERE subject_id = ?", [subject_id]
    ).fetch_df()
    if df.empty:
        raise HTTPException(status_code=404, detail="Subject not found")
    return df.to_dict(orient="records")[0]


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    # Simple logistic function as a stand-in for a model; deterministic and safe for demo
    z = 0.02 * req.AGE + 0.3 * req.AE_COUNT + 0.6 * req.SEVERE_AE_COUNT - 2.0
    import math

    risk = 1 / (1 + math.exp(-z))
    return ScoreResponse(risk=float(risk))

