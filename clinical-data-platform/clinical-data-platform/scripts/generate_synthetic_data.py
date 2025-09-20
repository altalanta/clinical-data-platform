from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
from random import Random

import pandas as pd


def gen_dm(rng: Random, n: int) -> pd.DataFrame:
    rows = []
    for i in range(n):
        subj = f"SUBJ{i+1:04d}"
        arm = rng.choice(["PLACEBO", "ACTIVE", None])
        sex = rng.choice(["M", "F", None])
        age = rng.choice([rng.randint(18, 85), None])
        rows.append({"STUDYID": "STUDY001", "SUBJID": subj, "ARM": arm, "SEX": sex, "AGE": age})
    # Edge case: invalid code
    rows[0]["SEX"] = "X"
    return pd.DataFrame(rows)


def gen_ae(rng: Random, dm: pd.DataFrame) -> pd.DataFrame:
    severities = ["MILD", "MODERATE", "SEVERE", "SERIOUS", None]
    outcomes = ["RECOVERED", "RECOVERING", "NOT RECOVERED", None]
    rows = []
    base = date(2024, 1, 1)
    for _, r in dm.iterrows():
        for _ in range(rng.randint(0, 3)):
            start = base + timedelta(days=rng.randint(0, 60))
            end = start + timedelta(days=rng.randint(0, 10))
            rows.append(
                {
                    "STUDYID": r["STUDYID"],
                    "SUBJID": r["SUBJID"],
                    "AESTDTC": start.isoformat(),
                    "AEENDTC": rng.choice([end.isoformat(), None]),
                    "AESEV": rng.choice(severities),
                    "AESER": rng.choice([True, False, None]),
                    "AEOUT": rng.choice(outcomes),
                }
            )
    return pd.DataFrame(rows)


def gen_lb(rng: Random, dm: pd.DataFrame) -> pd.DataFrame:
    tests = [
        ("ALT", 10, 40, "U/L"),
        ("AST", 10, 40, "U/L"),
        ("CREAT", 0.6, 1.3, "mg/dL"),
    ]
    rows = []
    for _, r in dm.iterrows():
        for t, lo, hi, unit in tests:
            val = rng.uniform(lo * 0.5, hi * 1.8)
            rows.append(
                {
                    "STUDYID": r["STUDYID"],
                    "SUBJID": r["SUBJID"],
                    "LBTESTCD": t,
                    "LBORRES": rng.choice([val, None]),
                    "LBORRESU": unit,
                    "LBLNOR": lo,
                    "LBHNOR": hi,
                }
            )
    return pd.DataFrame(rows)


def gen_vs(rng: Random, dm: pd.DataFrame) -> pd.DataFrame:
    tests = [("HR", 50, 110, "bpm"), ("SBP", 90, 160, "mmHg")]
    rows = []
    for _, r in dm.iterrows():
        for t, lo, hi, unit in tests:
            val = rng.uniform(lo, hi)
            rows.append(
                {
                    "STUDYID": r["STUDYID"],
                    "SUBJID": r["SUBJID"],
                    "VSTESTCD": t,
                    "VSORRES": rng.choice([val, None]),
                    "VSORRESU": unit,
                }
            )
    return pd.DataFrame(rows)


def gen_ex(rng: Random, dm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = date(2024, 1, 1)
    for _, r in dm.iterrows():
        if r["ARM"] is None:
            continue
        start = base + timedelta(days=rng.randint(0, 14))
        end = start + timedelta(days=rng.randint(14, 60))
        rows.append(
            {
                "STUDYID": r["STUDYID"],
                "SUBJID": r["SUBJID"],
                "EXTRT": r["ARM"],
                "EXDOSE": rng.choice([rng.uniform(10, 100), None]),
                "EXSTDTC": start.isoformat(),
                "EXENDTC": end.isoformat(),
            }
        )
    return pd.DataFrame(rows)


def main(out: str, rows: int, seed: int) -> None:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = Random(seed)
    dm = gen_dm(rng, rows)
    ae = gen_ae(rng, dm)
    lb = gen_lb(rng, dm)
    vs = gen_vs(rng, dm)
    ex = gen_ex(rng, dm)
    dm.to_csv(out_dir / "DM.csv", index=False)
    ae.to_csv(out_dir / "AE.csv", index=False)
    lb.to_csv(out_dir / "LB.csv", index=False)
    vs.to_csv(out_dir / "VS.csv", index=False)
    ex.to_csv(out_dir / "EX.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="data/sample_raw")
    p.add_argument("--rows", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args.out, args.rows, args.seed)

