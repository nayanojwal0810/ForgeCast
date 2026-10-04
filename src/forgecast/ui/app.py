"""Streamlit demonstration UI for ForgeCast 15-minute industrial energy forecasting."""

import os
from pathlib import Path
import sys

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import streamlit as st

from forgecast.ui.service import run_bounded_demonstration


def main() -> None:
    st.set_page_config(
        page_title="ForgeCast - 15-Minute Industrial Energy Forecasting",
        page_icon="⚡",
        layout="wide",
    )

    # 1. Header
    st.title("⚡ ForgeCast")
    st.subheader("15-Minute-Ahead Industrial Energy Forecasting")
    st.markdown(
        "A lightweight forecasting system that predicts the next 15-minute "
        "energy consumption interval and compares it with a simple persistence baseline."
    )

    # 2. Demo controls (Sidebar)
    st.sidebar.header("Demo Controls")
    record_count = st.sidebar.slider(
        "Demo length (telemetry records)",
        min_value=100,
        max_value=300,
        value=150,
        step=10,
        help="Number of 15-minute telemetry intervals to replay. Initial warm-up requires 96 observations.",
    )
    run_btn = st.sidebar.button("Run Demo", use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "This is an educational portfolio demonstration of 15-minute energy forecasting "
        "over historical telemetry. Frozen model artifacts and raw data remain unchanged."
    )

    # Replay execution
    cache_key = f"{record_count}"
    if "last_cache_key" not in st.session_state or st.session_state["last_cache_key"] != cache_key or run_btn:
        with st.spinner("Executing forecasting demonstration..."):
            st.session_state["snapshot"] = run_bounded_demonstration(
                record_count=record_count,
                inject_gap=False,
            )
            st.session_state["last_cache_key"] = cache_key

    snapshot = st.session_state["snapshot"]

    st.divider()

    # 3. Main Forecast Result
    st.markdown("### Latest Forecast")
    if snapshot.latest_prediction is not None and snapshot.latest_feedback is not None:
        pred = snapshot.latest_prediction
        fb = snapshot.latest_feedback

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ML Forecast", f"{pred.predicted_usage_kwh:.2f} kWh")
        with col2:
            st.metric("Persistence Baseline", f"{pred.baseline_persistence_kwh:.2f} kWh")
        with col3:
            st.metric("Actual Usage", f"{fb.actual_usage_kwh:.2f} kWh")

        # Interval comparison verdict
        if fb.ml_absolute_error < fb.persistence_absolute_error:
            st.success(
                f"✅ **ML performed better on this interval** "
                f"(Error: {fb.ml_absolute_error:.2f} kWh vs Baseline Error: {fb.persistence_absolute_error:.2f} kWh)"
            )
        else:
            st.info(
                f"⚠️ **Persistence baseline performed better on this interval** "
                f"(Baseline Error: {fb.persistence_absolute_error:.2f} kWh vs ML Error: {fb.ml_absolute_error:.2f} kWh)"
            )
    else:
        st.info("Run the demo to view the latest forecast.")

    st.divider()

    # 4. Forecast vs Actual Chart
    st.markdown("### Forecast vs. Actual Usage")
    if snapshot.recent_history:
        df_chart = pd.DataFrame(snapshot.recent_history)
        df_chart["target_timestamp"] = pd.to_datetime(df_chart["target_timestamp"])
        df_chart = df_chart.set_index("target_timestamp")[
            ["actual_usage_kwh", "predicted_usage_kwh", "baseline_persistence_kwh"]
        ]
        df_chart.columns = ["Actual Usage (kWh)", "ML Forecast (kWh)", "Persistence Baseline (kWh)"]
        st.line_chart(df_chart)
        st.caption("Recent forecast performance compared with actual energy usage and a persistence baseline.")
    else:
        st.info("Chart will populate once completed feedback observations are accumulated.")

    st.divider()

    # 5. Model Performance
    st.markdown("### Model Performance")
    p1, p2 = st.columns(2)
    with p1:
        st.metric(
            "Chronological Evaluation MAE",
            "3.8626 kWh",
            delta="-28.05% vs baseline",
            delta_color="normal",
        )
        st.caption("Based on chronological evaluation across 10,512 held-out observations.")
    with p2:
        st.metric(
            "Post-Training Replay MAE",
            "3.2133 kWh",
            delta="-24.45% vs baseline",
            delta_color="normal",
        )
        st.caption("Based on 3,504 held-out replay intervals following model training.")

    st.divider()

    # 6. What this demo demonstrates
    st.markdown("### What this demo demonstrates")
    st.markdown(
        "- **15-minute-ahead energy forecasting** for industrial operations\n"
        "- **Causal time-series features** using only data known prior to forecast cutoff\n"
        "- **Comparison against a persistence baseline** (`ŷ(t+1) = Usage(t)`)\n"
        "- **Delayed actual-vs-prediction evaluation** as ground truth meter readings arrive"
    )

    st.markdown("---")

    # 7. Data note
    st.caption(
        "Demo uses the [UCI Steel Industry Energy Consumption dataset]"
        "(https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption). "
        "This is a demonstration of the forecasting lifecycle over historical telemetry."
    )


if __name__ == "__main__":
    main()
