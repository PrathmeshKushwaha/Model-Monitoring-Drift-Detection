import streamlit as st
import mlflow
import pandas as pd

# ---------------- Page setup ----------------
st.set_page_config(layout="wide")
st.title("📊 Bank Marketing – Model Monitoring Dashboard")

# ---------------- MLflow setup ----------------
mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("bank-marketing-monitoring")

if experiment is None:
    st.error("MLflow experiment not found.")
    st.stop()

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time ASC"]
)

if runs.empty:
    st.warning("No monitoring runs found.")
    st.stop()

# ---------------- Latest status ----------------
latest_run = runs.iloc[-1]

st.subheader("🔴 Current Monitoring Status")
status = latest_run.get("params.final_status", "UNKNOWN")

if status == "PASS":
    st.success("PASS")
elif status == "FAIL":
    st.error("FAIL")
else:
    st.warning("UNKNOWN")

# ---------------- Status over time ----------------
st.subheader("Monitoring Status Over Time")

status_df = runs[
    ["tags.mlflow.runName", "params.final_status", "start_time"]
].copy()

# Rename ONCE and use consistently
status_df.rename(
    columns={"tags.mlflow.runName": "run_name"},
    inplace=True
)

status_df["status_numeric"] = status_df["params.final_status"].map(
    {"PASS": 1, "FAIL": 0}
)

status_df.set_index("start_time", inplace=True)

st.line_chart(status_df["status_numeric"])

# ---------------- Performance trends ----------------
st.subheader("Production Performance Over Time")

perf_cols = [c for c in runs.columns if c.startswith("metrics.prod_")]

if perf_cols:
    perf_df = runs[["start_time"] + perf_cols].set_index("start_time")
    st.line_chart(perf_df)
else:
    st.info("No performance metrics found.")

# ---------------- Feature drift trends ----------------
st.subheader("Feature Drift Trends")

drift_cols = [c for c in runs.columns if c.startswith("metrics.feature_drift_")]

if drift_cols:
    drift_df = runs[["start_time"] + drift_cols].set_index("start_time")
    st.line_chart(drift_df)
else:
    st.info("No feature drift metrics found.")

# ---------------- Raw run table ----------------
with st.expander("View raw monitoring runs"):
    st.dataframe(
        runs[
            ["tags.mlflow.runName", "params.final_status", "start_time"]
        ].rename(columns={"tags.mlflow.runName": "run_name"})
    )
