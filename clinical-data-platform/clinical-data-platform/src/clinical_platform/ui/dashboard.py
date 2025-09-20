from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import streamlit as st

from clinical_platform.config import get_config


st.set_page_config(page_title="Clinical Data Platform", layout="wide")
st.title("Clinical Data Platform Dashboard")

cfg = get_config()
con = duckdb.connect(cfg.warehouse.duckdb_path)

studies = [r[0] for r in con.execute("SELECT study_id FROM dim_study").fetchall()] or ["STUDY001"]
study = st.sidebar.selectbox("Study", studies)
arms = [r[0] for r in con.execute("SELECT DISTINCT arm FROM dim_subject").fetchall()]
arm = st.sidebar.selectbox("Arm", ["ALL"] + [a for a in arms if a])

subject_df = con.execute(
    "SELECT subject_id, arm, sex, age FROM dim_subject"
).fetch_df()
if arm != "ALL":
    subject_df = subject_df[subject_df["arm"] == arm]

st.subheader("Subjects")
st.dataframe(subject_df, use_container_width=True, hide_index=True)

ae_df = con.execute(
    "SELECT s.arm, f.severity, f.ae_start FROM fact_adverse_events f JOIN dim_subject s USING(subject_sk)"
).fetch_df()
ae_counts = (
    ae_df.groupby(["arm", pd.to_datetime(ae_df["ae_start"]).dt.to_period("W").astype(str)])
    .size()
    .reset_index(name="count")
    .rename(columns={"ae_start": "week"})
)

st.subheader("AE Trend by Arm")
st.line_chart(
    ae_counts.pivot_table(index="week", columns="arm", values="count", fill_value=0)
)

st.subheader("Model Score Histogram (Toy)")
toy = subject_df[["age"]].fillna(0).astype(float)
scores = 1 / (1 + np.exp(-0.03 * toy["age"].values + -1.0))
st.bar_chart(pd.DataFrame({"score": scores}).value_counts(bins=10, sort=False))

st.subheader("Subject Drill-down")
sel = st.selectbox("Subject", subject_df["subject_id"].tolist())
detail = con.execute(
    "SELECT * FROM fact_adverse_events f JOIN dim_subject s USING(subject_sk) WHERE s.subject_id = ?",
    [sel],
).fetch_df()
st.write(detail)

