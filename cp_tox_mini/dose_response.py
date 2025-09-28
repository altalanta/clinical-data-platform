"""Dose-response utilities including an LL4 IC50 estimator."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import curve_fit

REPORTS_DIR = Path("reports")
IC50_JSON = REPORTS_DIR / "ic50_summary.json"


@dataclass(frozen=True)
class IC50Result:
    ic50: float
    bottom: float
    top: float
    hill_slope: float
    r_squared: float


def _ll4(dose: np.ndarray, bottom: float, top: float, hill: float, log_ic50: float) -> np.ndarray:
    return bottom + (top - bottom) / (1 + np.exp(hill * (np.log10(dose) - log_ic50)))


def estimate_ic50(doses: Sequence[float], responses: Sequence[float]) -> IC50Result:
    x = np.asarray(doses, dtype=np.float64)
    y = np.asarray(responses, dtype=np.float64)
    if np.any(x <= 0):
        raise ValueError("Doses must be positive for log-logistic fitting")

    bottom_guess = float(np.min(y))
    top_guess = float(np.max(y))
    hill_guess = 1.0
    log_ic50_guess = float(np.mean(np.log10(x)))

    popt, _ = curve_fit(
        _ll4,
        x,
        y,
        p0=[bottom_guess, top_guess, hill_guess, log_ic50_guess],
        bounds=(
            [bottom_guess - 1.0, top_guess - 1.0, 0.01, np.log10(np.min(x)) - 1],
            [top_guess + 1.0, top_guess + 1.0, 5.0, np.log10(np.max(x)) + 1],
        ),
        maxfev=10000,
    )

    fitted = _ll4(x, *popt)
    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot else 1.0

    ic50_value = 10 ** popt[3]
    return IC50Result(
        ic50=float(ic50_value),
        bottom=float(popt[0]),
        top=float(popt[1]),
        hill_slope=float(popt[2]),
        r_squared=float(r_squared),
    )


def summarize_ic50_table(csv_path: Path) -> IC50Result:
    import pandas as pd

    df = pd.read_csv(csv_path)
    result = estimate_ic50(df["dose_nM"].to_numpy(), df["response"].to_numpy())

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "ic50": round(result.ic50, 4),
        "bottom": round(result.bottom, 4),
        "top": round(result.top, 4),
        "hill_slope": round(result.hill_slope, 4),
        "r_squared": round(result.r_squared, 4),
        "n_points": int(len(df)),
        "source": str(csv_path),
    }
    IC50_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return result


__all__ = ["estimate_ic50", "summarize_ic50_table", "IC50Result", "IC50_JSON"]
