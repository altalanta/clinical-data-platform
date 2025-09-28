"""Feature engineering for the CP tox mini demo."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from PIL import Image

from .io import RAW_DIR, PROCESSED_DIR

CP_FEATURES = ["cp_intensity_mean", "cp_intensity_std", "cp_entropy"]
CHEM_FEATURES = ["chem_length", "chem_hetero", "chem_halogen"]


def _load_plate_features() -> Dict[str, Dict[str, float]]:
    plate_features: Dict[str, Dict[str, float]] = {}
    for plate_id, file_name in {"plate_A": "plate_A1.tiff", "plate_B": "plate_B1.tiff"}.items():
        image_path = RAW_DIR / file_name
        with Image.open(image_path) as img:
            arr = np.array(img, dtype=np.float32)
        arr = arr / 255.0
        plate_features[plate_id] = {
            "cp_intensity_mean": float(arr.mean()),
            "cp_intensity_std": float(arr.std()),
            "cp_entropy": float(-np.sum(arr * np.log(arr + 1e-8))),
        }
    return plate_features


def _chem_features(smiles: str) -> Dict[str, int]:
    smile = smiles.upper()
    hetero = sum(smile.count(atom) for atom in ["N", "O", "S"])
    halogen = sum(smile.count(atom) for atom in ["CL", "BR", "F"])
    return {
        "chem_length": len(smiles),
        "chem_hetero": hetero,
        "chem_halogen": halogen,
    }


def build_features() -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw_csv = RAW_DIR / "tox21_mini.csv"
    df = pd.read_csv(raw_csv)

    plate_lookup = _load_plate_features()

    feature_rows = []
    for _, row in df.iterrows():
        base = {
            "sample_id": row["sample_id"],
            "compound_id": row["compound_id"],
            "plate_id": row["plate_id"],
            "well_row": row["well_row"],
            "well_col": int(row["well_col"]),
            "label": int(row["label"]),
            "dose_nM": float(row["dose_nM"]),
            "readout": float(row["readout"]),
        }
        base["well_row_index"] = ord(str(row["well_row"]).upper()[0]) - ord("A")
        base.update(plate_lookup[row["plate_id"]])
        base.update(_chem_features(row["smiles"]))
        feature_rows.append(base)

    feature_df = pd.DataFrame(feature_rows)
    feature_df = feature_df.sort_values("sample_id").reset_index(drop=True)

    output = PROCESSED_DIR / "features.parquet"
    feature_df.to_parquet(output, index=False)
    return output


__all__ = ["build_features", "CP_FEATURES", "CHEM_FEATURES"]
