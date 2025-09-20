from __future__ import annotations

from pathlib import Path

import pandas as pd

from clinical_platform.config import get_config


def build_adsl(standardized_dir: str | Path | None = None) -> pd.DataFrame:
    cfg = get_config()
    std_dir = Path(standardized_dir or cfg.paths.standardized_dir)
    dm = pd.read_parquet(std_dir / "DM.parquet")
    ae = pd.read_parquet(std_dir / "AE.parquet")

    # Derive: dropout flag (no AE end date or severe AE as proxy)
    ae_severe = ae[ae["AESEV"].isin(["SEVERE", "SERIOUS"]).fillna(False)]
    ae_any = ae.groupby(["STUDYID", "SUBJID"]).size().reset_index(name="AE_COUNT")
    ae_sev = ae_severe.groupby(["STUDYID", "SUBJID"]).size().reset_index(name="SEVERE_AE_COUNT")
    adsl = (
        dm.merge(ae_any, on=["STUDYID", "SUBJID"], how="left")
        .merge(ae_sev, on=["STUDYID", "SUBJID"], how="left")
        .fillna({"AE_COUNT": 0, "SEVERE_AE_COUNT": 0})
    )
    adsl["DROPOUT_RISK"] = (adsl["SEVERE_AE_COUNT"] > 0).astype(int)
    return adsl


if __name__ == "__main__":
    df = build_adsl()
    out = Path(get_config().paths.standardized_dir) / "ADSL.parquet"
    df.to_parquet(out, index=False)
    print(f"Wrote {out} ({len(df)} rows)")

