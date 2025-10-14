"""Generate deterministic synthetic clinical data for the dbt/DuckDB demo."""

from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class GenerationConfig:
    seed: int = 1337
    num_patients: int = 6000
    min_encounters: int = 2
    max_encounters: int = 12
    death_rate: float = 0.08
    dx_per_encounter_mean: float = 2.4
    proc_per_encounter_mean: float = 1.2
    labs_per_patient_mean: float = 8.0


RACES = [
    "White",
    "Black",
    "Asian",
    "American Indian/Alaska Native",
    "Native Hawaiian/Pacific Islander",
]

ETHNICITIES = ["Hispanic or Latino", "Not Hispanic or Latino"]

INDEX_SITES = ["A", "B", "C"]

ENCOUNTER_TYPES = ["INPT", "OUTPT", "ED"]

DX_POOL = [
    ("I21.9", "Acute myocardial infarction"),
    ("I10", "Essential (primary) hypertension"),
    ("E11.9", "Type 2 diabetes mellitus"),
    ("C34.90", "Malignant neoplasm of unspecified part of bronchus or lung"),
    ("C50.919", "Malignant neoplasm of unspecified female breast"),
    ("N18.3", "Chronic kidney disease, stage 3"),
    ("F17.210", "Nicotine dependence, unspecified, uncomplicated"),
    ("E78.5", "Hyperlipidemia, unspecified"),
]

PROC_POOL = [
    ("92950", "Cardiopulmonary resuscitation"),
    ("99223", "Initial hospital care"),
    ("93000", "Electrocardiogram"),
    ("93010", "ECG interpretation and report"),
    ("36591", "Collection of blood specimen"),
    ("71045", "Chest x-ray"),
    ("71275", "CTA chest"),
    ("99214", "Established patient office visit"),
]

LAB_POOL = [
    ("2085-9", "cholesterol", "mg/dL"),
    ("2345-7", "glucose", "mg/dL"),
    ("718-7", "hematocrit", "%"),
    ("777-3", "platelets", "10^3/uL"),
    ("6690-2", "leukocytes", "10^3/uL"),
    ("718-7", "hematocrit", "%"),
    ("2160-0", "creatinine", "mg/dL"),
    ("4548-4", "hemoglobin A1c", "%"),
]

CORE_LABS = [
    ("2085-9", "cholesterol", "mg/dL"),
    ("2345-7", "glucose", "mg/dL"),
    ("2160-0", "creatinine", "mg/dL"),
]


def parse_args() -> GenerationConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument(
        "--num-patients",
        type=int,
        default=6000,
        help="Number of synthetic patients to generate",
    )
    args = parser.parse_args()
    return GenerationConfig(seed=args.seed, num_patients=args.num_patients)


def ensure_directories(base_dir: pathlib.Path) -> None:
    (base_dir / "data" / "raw").mkdir(parents=True, exist_ok=True)


def generate_patients(cfg: GenerationConfig, rng: np.random.Generator) -> pd.DataFrame:
    patient_ids = np.array([f"P{idx:05d}" for idx in range(1, cfg.num_patients + 1)])
    sexes = rng.choice(["F", "M"], size=cfg.num_patients, p=[0.52, 0.48])
    birth_years = rng.integers(1940, 2003, size=cfg.num_patients)
    birth_days = rng.integers(1, 366, size=cfg.num_patients)
    birth_dates = pd.to_datetime(birth_years, format="%Y") + pd.to_timedelta(
        birth_days, unit="D"
    )

    races = rng.choice(RACES, size=cfg.num_patients)
    ethnicities = rng.choice(ETHNICITIES, size=cfg.num_patients, p=[0.18, 0.82])
    index_sites = rng.choice(INDEX_SITES, size=cfg.num_patients, p=[0.4, 0.35, 0.25])

    return pd.DataFrame(
        {
            "patient_id": patient_ids,
            "sex": sexes,
            "birth_date": birth_dates.date,
            "race": races,
            "ethnicity": ethnicities,
            "index_site": index_sites,
        }
    )


def generate_encounters(
    patients: pd.DataFrame, cfg: GenerationConfig, rng: np.random.Generator
) -> pd.DataFrame:
    encounter_rows = []
    encounter_id = 1

    base_start = pd.Timestamp("2016-01-01")
    window_days = 365 * 7  # seven-year window

    for patient_id in patients["patient_id"]:
        num_encounters = rng.integers(cfg.min_encounters, cfg.max_encounters + 1)
        encounter_times = np.sort(rng.integers(0, window_days, size=num_encounters))

        for offset in encounter_times:
            admit_ts = base_start + pd.to_timedelta(int(offset), unit="D")
            los_days = max(1, int(rng.integers(1, 6)))
            discharge_ts = admit_ts + pd.to_timedelta(los_days, unit="D")
            encounter_type = rng.choice(ENCOUNTER_TYPES, p=[0.22, 0.58, 0.20])

            encounter_rows.append(
                {
                    "encounter_id": f"E{encounter_id:07d}",
                    "patient_id": patient_id,
                    "admit_ts": admit_ts,
                    "discharge_ts": discharge_ts,
                    "encounter_type": encounter_type,
                }
            )
            encounter_id += 1

    encounters = pd.DataFrame(encounter_rows)
    encounters.sort_values(["patient_id", "admit_ts"], inplace=True)
    encounters.reset_index(drop=True, inplace=True)
    return encounters


def compute_index_dates(
    encounters: pd.DataFrame, diagnoses: pd.DataFrame
) -> dict[str, pd.Timestamp]:
    inpatient = (
        encounters.loc[encounters["encounter_type"] == "INPT", ["patient_id", "admit_ts"]]
        .groupby("patient_id")["admit_ts"]
        .min()
    )
    high_risk_codes = {"I21.9", "E11.9", "N18.3", "C34.90"}
    if diagnoses.empty:
        high_risk = pd.Series(dtype="datetime64[ns]")
    else:
        high_risk = (
            diagnoses.loc[
                diagnoses["diagnosis_code"].isin(high_risk_codes),
                ["patient_id", "diagnosis_ts"],
            ]
            .groupby("patient_id")["diagnosis_ts"]
            .min()
        )
    any_encounter = encounters.groupby("patient_id")["admit_ts"].min()

    index_dates: dict[str, pd.Timestamp] = {}
    for patient_id, first_enc in any_encounter.items():
        candidates = [first_enc]
        if patient_id in inpatient.index:
            candidates.append(inpatient.loc[patient_id])
        if patient_id in high_risk.index:
            candidates.append(high_risk.loc[patient_id])
        index_dates[patient_id] = min(candidates)
    return index_dates


def sample_codes(
    pool: list[tuple[str, str]], count: int, rng: np.random.Generator
) -> list[str]:
    codes = [code for code, _ in pool]
    probs = np.linspace(1.5, 0.5, num=len(codes))
    probs = probs / probs.sum()
    return rng.choice(codes, size=count, p=probs).tolist()


def generate_diagnoses(
    encounters: pd.DataFrame, cfg: GenerationConfig, rng: np.random.Generator
) -> pd.DataFrame:
    dx_rows = []
    for _, encounter in encounters.iterrows():
        count = rng.poisson(cfg.dx_per_encounter_mean)
        if count == 0:
            continue
        codes = sample_codes(DX_POOL, count, rng)
        for code in codes:
            dx_rows.append(
                {
                    "encounter_id": encounter["encounter_id"],
                    "patient_id": encounter["patient_id"],
                    "diagnosis_code": code,
                    "diagnosis_ts": encounter["admit_ts"]
                    + pd.to_timedelta(int(rng.integers(0, 2)), unit="D"),
                }
            )
    diagnoses = pd.DataFrame(dx_rows)
    diagnoses.sort_values(["patient_id", "diagnosis_ts"], inplace=True)
    diagnoses.reset_index(drop=True, inplace=True)
    return diagnoses


def generate_procedures(
    encounters: pd.DataFrame, cfg: GenerationConfig, rng: np.random.Generator
) -> pd.DataFrame:
    proc_rows = []
    for _, encounter in encounters.iterrows():
        count = rng.poisson(cfg.proc_per_encounter_mean)
        if count == 0:
            continue
        codes = sample_codes(PROC_POOL, count, rng)
        for code in codes:
            proc_rows.append(
                {
                    "encounter_id": encounter["encounter_id"],
                    "patient_id": encounter["patient_id"],
                    "procedure_code": code,
                    "procedure_ts": encounter["admit_ts"]
                    + pd.to_timedelta(int(rng.integers(0, 2)), unit="D"),
                }
            )
    procedures = pd.DataFrame(proc_rows)
    procedures.sort_values(["patient_id", "procedure_ts"], inplace=True)
    procedures.reset_index(drop=True, inplace=True)
    return procedures


def generate_labs(
    encounters: pd.DataFrame, cfg: GenerationConfig, rng: np.random.Generator
) -> pd.DataFrame:
    lab_rows = []
    lab_id = 1

    patient_encounters = encounters.groupby("patient_id")

    for patient_id, enc_df in patient_encounters:
        # ensure at least one lab per patient
        num_labs = max(1, int(rng.poisson(cfg.labs_per_patient_mean)))
        encounter_indices = rng.choice(enc_df.index, size=num_labs, replace=True)
        for idx in encounter_indices:
            encounter = enc_df.loc[idx]
            loinc_code, category, unit = LAB_POOL[rng.integers(0, len(LAB_POOL))]

            base_ts = encounter["admit_ts"] + pd.to_timedelta(
                int(rng.integers(-3, 4)), unit="D"
            )
            meas_value = generate_lab_value(category, rng)

            lab_rows.append(
                {
                    "lab_id": f"L{lab_id:07d}",
                    "patient_id": patient_id,
                    "loinc_code": loinc_code,
                    "meas_value": round(meas_value, 2),
                    "meas_unit": unit,
                    "result_ts": base_ts,
                }
            )
            lab_id += 1

    labs = pd.DataFrame(lab_rows)
    labs.sort_values(["patient_id", "result_ts"], inplace=True)
    labs.reset_index(drop=True, inplace=True)
    return labs


def generate_lab_value(category: str, rng: np.random.Generator) -> float:
    if category == "cholesterol":
        return rng.normal(180, 35)
    if category == "glucose":
        return rng.normal(115, 30)
    if category == "hematocrit":
        return rng.normal(40, 4)
    if category == "platelets":
        return rng.normal(250, 60)
    if category == "leukocytes":
        return rng.normal(7.0, 1.5)
    if category == "creatinine":
        return rng.normal(1.0, 0.3)
    if category == "hemoglobin A1c":
        return rng.normal(6.2, 1.1)
    return rng.normal(50, 5)


def ensure_core_labs(
    labs: pd.DataFrame,
    index_dates: dict[str, pd.Timestamp],
    rng: np.random.Generator,
) -> pd.DataFrame:
    if labs.empty:
        lab_id_counter = 1
    else:
        lab_id_counter = labs["lab_id"].str[1:].astype(int).max() + 1
    supplemental_rows = []
    for patient_id, index_ts in index_dates.items():
        if pd.isna(index_ts):
            continue
        for loinc_code, category, unit in CORE_LABS:
            existing = labs[
                (labs["patient_id"] == patient_id)
                & (labs["loinc_code"] == loinc_code)
                & (labs["result_ts"] >= index_ts - pd.Timedelta(days=365))
                & (labs["result_ts"] <= index_ts)
            ]
            if existing.empty:
                offset = int(rng.integers(30, 300))
                result_ts = index_ts - pd.to_timedelta(offset, unit="D")
                supplemental_rows.append(
                    {
                        "lab_id": f"L{lab_id_counter:07d}",
                        "patient_id": patient_id,
                        "loinc_code": loinc_code,
                        "meas_value": round(generate_lab_value(category, rng), 2),
                        "meas_unit": unit,
                        "result_ts": result_ts,
                    }
                )
                lab_id_counter += 1
    if supplemental_rows:
        labs = pd.concat([labs, pd.DataFrame(supplemental_rows)], ignore_index=True)
        labs.sort_values(["patient_id", "result_ts"], inplace=True)
        labs.reset_index(drop=True, inplace=True)
    return labs


def generate_deaths(
    encounters: pd.DataFrame,
    index_dates: dict[str, pd.Timestamp],
    cfg: GenerationConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    last_encounter = (
        encounters.sort_values("admit_ts")
        .groupby("patient_id")["discharge_ts"]
        .max()
        .reset_index()
    )
    death_flags = rng.random(len(last_encounter)) < cfg.death_rate

    death_rows = []
    for flagged, (_, row) in zip(death_flags, last_encounter.iterrows()):
        if not flagged:
            continue
        index_ts = index_dates.get(row["patient_id"], row["discharge_ts"])
        if pd.isna(index_ts):
            index_ts = row["discharge_ts"]
        death_offset = int(rng.integers(30, 365))
        death_rows.append(
            {
                "patient_id": row["patient_id"],
                "death_ts": index_ts + pd.to_timedelta(death_offset, unit="D"),
            }
        )

    deaths = pd.DataFrame(death_rows)
    deaths.sort_values(["patient_id", "death_ts"], inplace=True)
    deaths.reset_index(drop=True, inplace=True)
    return deaths


def filter_after_death(
    df: pd.DataFrame,
    death_lookup: dict[str, pd.Timestamp],
    timestamp_col: str,
) -> pd.DataFrame:
    if df.empty or not death_lookup:
        return df
    death_series = df["patient_id"].map(death_lookup)
    mask = death_series.isna() | (df[timestamp_col] <= death_series)
    filtered = df.loc[mask].copy()
    sort_cols = ["patient_id", timestamp_col]
    filtered.sort_values(sort_cols, inplace=True)
    filtered.reset_index(drop=True, inplace=True)
    return filtered


def write_csv(df: pd.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    cfg = parse_args()
    rng = np.random.default_rng(cfg.seed)

    project_root = pathlib.Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    ensure_directories(project_root)

    patients = generate_patients(cfg, rng)
    encounters = generate_encounters(patients, cfg, rng)
    diagnoses = generate_diagnoses(encounters, cfg, rng)
    procedures = generate_procedures(encounters, cfg, rng)
    labs = generate_labs(encounters, cfg, rng)
    index_dates = compute_index_dates(encounters, diagnoses)
    labs = ensure_core_labs(labs, index_dates, rng)
    deaths = generate_deaths(encounters, index_dates, cfg, rng)

    death_lookup = dict(zip(deaths["patient_id"], deaths["death_ts"]))
    encounters = filter_after_death(encounters, death_lookup, "admit_ts")
    diagnoses = filter_after_death(diagnoses, death_lookup, "diagnosis_ts")
    procedures = filter_after_death(procedures, death_lookup, "procedure_ts")
    labs = filter_after_death(labs, death_lookup, "result_ts")

    write_csv(patients, raw_dir / "patients.csv")
    write_csv(encounters, raw_dir / "encounters.csv")
    write_csv(diagnoses, raw_dir / "diagnoses.csv")
    write_csv(procedures, raw_dir / "procedures.csv")
    write_csv(labs, raw_dir / "labs.csv")
    write_csv(deaths, raw_dir / "death.csv")

    print(f"Generated data for {len(patients)} patients at seed {cfg.seed}.")
    print(f"Encounters: {len(encounters)} | Diagnoses: {len(diagnoses)}")
    print(f"Procedures: {len(procedures)} | Labs: {len(labs)} | Deaths: {len(deaths)}")


if __name__ == "__main__":
    main()
