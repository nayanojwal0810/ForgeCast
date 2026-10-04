"""Streamlit operational demonstration UI for ForgeCast."""

from datetime import datetime
import os
from pathlib import Path
import sys

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import streamlit as st

from forgecast.ui.service import (
    load_chronological_evaluation_summary,
    load_retraining_lifecycle_summary,
    run_bounded_demonstration,
)


def format_mae(val: float | None) -> str:
    return f"{val:.4f} kWh" if val is not None else "N/A"


def format_pct(val: float | None) -> str:
    if val is None:
        return "N/A"
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.2f}%"


def main() -> None:
    st.set_page_config(
        page_title="ForgeCast - 15-Minute Industrial Energy Forecasting",
        page_icon="⚡",
        layout="wide",
    )

    st.title("⚡ FORGECAST")
    st.subheader("15-Minute Industrial Energy Forecasting & MLOps Demo")
    st.caption(
        "Historical Replay & Operational Demonstration | "
        "Deterministic Telemetry Replay, Inference, Delayed Feedback & Rolling Monitoring"
    )

    # Sidebar: Replay Controls
    st.sidebar.header("🕹️ Replay Controls")
    record_count = st.sidebar.slider(
        "Bounded Telemetry Records",
        min_value=50,
        max_value=500,
        value=150,
        step=10,
        help="Replay a bounded slice of historical telemetry. Initial warm-up requires exactly 96 observations.",
    )
    inject_gap = st.sidebar.checkbox(
        "Simulate Missing Telemetry (Inject Gap)",
        value=False,
        help="Simulates missing telemetry by omitting record 110 in memory. Triggers gap detection, prediction invalidation, and re-warming.",
    )
    run_btn = st.sidebar.button("Run Replay Demo", use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Portfolio Demonstration Notice**\n\n"
        "This application is an educational and portfolio demonstration of a robust 15-minute time-series MLOps architecture. "
        "It performs a bounded in-memory replay of historical steel industry telemetry. "
        "Raw data and frozen model artifacts remain completely unchanged."
    )

    # Cache bounded replay execution per parameters
    cache_key = f"{record_count}_{inject_gap}"
    if "last_cache_key" not in st.session_state or st.session_state["last_cache_key"] != cache_key or run_btn:
        with st.spinner("Executing bounded telemetry replay..."):
            st.session_state["snapshot"] = run_bounded_demonstration(
                record_count=record_count,
                inject_gap=inject_gap,
                gap_record_index=110,
            )
            st.session_state["last_cache_key"] = cache_key

    snapshot = st.session_state["snapshot"]

    # Top Status Bar
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    with stat_col1:
        st.metric("Active Model", snapshot.model_version)
        st.caption(f"Estimator: {snapshot.model_class}")
    with stat_col2:
        st.metric("System State", snapshot.system_state)
        warmup_desc = "Yes" if snapshot.is_warmed_up else f"No ({min(snapshot.records_consumed, 96)}/96)"
        st.caption(f"Warmed Up: {warmup_desc}")
    with stat_col3:
        st.metric("Continuous Segment", f"#{snapshot.current_segment_id}")
        ts_str = (
            snapshot.latest_logical_timestamp.strftime("%Y-%m-%d %H:%M")
            if snapshot.latest_logical_timestamp
            else "N/A"
        )
        st.caption(f"Latest Telemetry: {ts_str}")
    with stat_col4:
        st.metric("Gap Incidents", snapshot.gap_incident_count)
        st.caption(f"Invalidated Predictions: {snapshot.invalidated_prediction_count}")

    st.divider()

    # Section 1 & 2: Current Prediction & Latest Feedback
    pred_col, fb_col = st.columns(2)

    with pred_col:
        st.markdown("### Current Prediction")
        if snapshot.latest_prediction is not None:
            pred = snapshot.latest_prediction
            p1, p2 = st.columns(2)
            with p1:
                st.metric("ML Forecast", f"{pred.predicted_usage_kwh:.3f} kWh")
                st.caption(f"Persistence Baseline: {pred.baseline_persistence_kwh:.3f} kWh")
            with p2:
                st.metric("Target Interval", pred.target_timestamp.strftime("%H:%M"))
                st.caption(f"Origin Interval: {pred.origin_timestamp.strftime('%H:%M')}")
            st.text(
                f"Model: {pred.model_version} | "
                f"Predicted at: {pred.prediction_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}"
            )
        else:
            st.info("ℹ️ No active prediction. System is currently in warm-up (<96 valid observations).")

    with fb_col:
        st.markdown("### Latest Delayed Feedback")
        if snapshot.latest_feedback is not None:
            fb = snapshot.latest_feedback
            f1, f2 = st.columns(2)
            with f1:
                st.metric("Actual Usage", f"{fb.actual_usage_kwh:.3f} kWh")
                st.caption(f"Target Timestamp: {fb.target_timestamp.strftime('%H:%M')}")
            with f2:
                st.metric("ML Absolute Error", f"{fb.ml_absolute_error:.3f} kWh")
                st.caption(f"Persistence Error: {fb.persistence_absolute_error:.3f} kWh")

            if fb.ml_absolute_error < fb.persistence_absolute_error:
                verdict = "✅ ML outperformed persistence baseline"
            else:
                verdict = "⚠️ Persistence baseline outperformed ML"
            st.text(f"Resolved Target: {fb.target_timestamp.strftime('%Y-%m-%d %H:%M')} | {verdict}")
        else:
            st.info("ℹ️ No delayed feedback paired yet. Delayed feedback pairs upon receiving target interval actuals.")

    st.divider()

    # Section 3: Model Monitoring
    st.markdown("### Rolling Operational Monitoring")
    m1, m2, m3 = st.columns(3)

    with m1:
        st.markdown("#### Recent 24-Step Window (~24h)")
        r24 = snapshot.rolling_24h_snapshot
        if r24 is not None and r24.sample_count > 0:
            st.write(f"Sample Count: **{r24.sample_count}**")
            st.write(f"ML MAE: **{format_mae(r24.ml_mae)}**")
            st.write(f"Persistence MAE: **{format_mae(r24.baseline_mae)}**")
            st.write(f"MAE Improvement: **{format_pct(r24.mae_improvement_pct)}**")
        else:
            st.caption("Accumulating observations for full 24-step rolling window.")

    with m2:
        st.markdown("#### Recent 672-Step Window (~7d)")
        r7d = snapshot.rolling_7d_snapshot
        if r7d is not None and r7d.sample_count > 0:
            st.write(f"Sample Count: **{r7d.sample_count}**")
            st.write(f"ML MAE: **{format_mae(r7d.ml_mae)}**")
            st.write(f"Persistence MAE: **{format_mae(r7d.baseline_mae)}**")
            st.write(f"MAE Improvement: **{format_pct(r7d.mae_improvement_pct)}**")
        else:
            st.caption(f"Accumulating observations ({snapshot.records_consumed}/672).")

    with m3:
        st.markdown("#### Cumulative Replay Performance")
        cum = snapshot.cumulative_snapshot
        if cum is not None and cum.sample_count > 0:
            st.write(f"Completed Feedback: **{cum.sample_count}**")
            st.write(f"Cumulative ML MAE: **{format_mae(cum.ml_mae)}**")
            st.write(f"Cumulative Persist MAE: **{format_mae(cum.baseline_mae)}**")
            win_txt = f"{cum.ml_win_rate * 100:.1f}%" if cum.ml_win_rate is not None else "N/A"
            st.write(f"ML Win Rate: **{win_txt}**")
        else:
            st.caption("No completed feedback records in current demonstration session.")

    st.divider()

    # Section 4: Prediction vs Actual Chart
    st.markdown("### Prediction vs. Actual Telemetry")
    if snapshot.recent_history:
        df_chart = pd.DataFrame(snapshot.recent_history)
        df_chart["target_timestamp"] = pd.to_datetime(df_chart["target_timestamp"])
        df_chart = df_chart.set_index("target_timestamp")[
            ["actual_usage_kwh", "predicted_usage_kwh", "baseline_persistence_kwh"]
        ]
        df_chart.columns = ["Actual Usage (kWh)", "ML Forecast (kWh)", "Persistence Baseline (kWh)"]
        st.line_chart(df_chart)
    else:
        st.info("Chart will populate once completed feedback observations are accumulated.")

    st.divider()

    # Section 5 & 6: Historical Evidence & Retraining Lifecycle Summary
    h_col, r_col = st.columns(2)

    with h_col:
        st.markdown("### Historical Evaluation Evidence")
        eval_summary = load_chronological_evaluation_summary()
        if eval_summary is not None:
            st.markdown(
                f"**3-Fold Chronological TimeSeriesSplit Validation** (Dataset: 35,040 rows)\n\n"
                f"- Pooled Test Samples: **{eval_summary['total_samples']}**\n"
                f"- Pooled ML MAE: **{eval_summary['ml_mae']:.4f} kWh**\n"
                f"- Pooled Persistence MAE: **{eval_summary['persistence_mae']:.4f} kWh**\n"
                f"- Pooled Improvement: **+{eval_summary['improvement_pct']:.2f}%**"
            )
        else:
            st.caption("Evaluation artifact not found.")

    with r_col:
        st.markdown("### Model Retraining Lifecycle")
        retrain_summary = load_retraining_lifecycle_summary()
        if retrain_summary is not None:
            st.markdown(
                f"**Latest Candidate Workflow** ({retrain_summary['checkpoint_name']} Checkpoint)\n\n"
                f"- Candidate Version: **{retrain_summary['candidate_version']}**\n"
                f"- Training Cutoff: **{retrain_summary['training_cutoff']}**\n"
                f"- Candidate MAE: **{retrain_summary['candidate_mae']:.4f} kWh** vs Baseline: **{retrain_summary['persistence_mae']:.4f} kWh**\n"
                f"- Promotion Decision: **{retrain_summary['decision_status']}**"
            )
        else:
            st.caption("No historical retraining artifacts available.")

    st.divider()

    # Section 7: Operational Health & Events
    st.markdown("### Operational Health & Continuity")
    ev1, ev2, ev3 = st.columns(3)

    with ev1:
        st.markdown("**Telemetry Missingness**")
        if snapshot.latest_gap_incident is not None:
            gap = snapshot.latest_gap_incident
            st.warning(
                f"⚠️ Gap Incident at {gap.current_timestamp.strftime('%H:%M')}\n\n"
                f"- Missing Intervals: {gap.missing_interval_count}\n"
                f"- State Reset: {gap.state_reset} (Re-warm initiated)"
            )
        else:
            st.success("✅ 0 Gap Incidents (Continuous Cadence)")

    with ev2:
        st.markdown("**Quarantined Predictions**")
        if snapshot.latest_invalidated_prediction is not None:
            inv = snapshot.latest_invalidated_prediction
            st.warning(
                f"⚠️ Prediction Quarantined for {inv.target_timestamp.strftime('%H:%M')}\n\n"
                f"- Reason: {inv.reason}\n"
                f"- Excluded from feedback & error metrics"
            )
        else:
            st.success("✅ 0 Invalidated Predictions")

    with ev3:
        st.markdown("**Temporal Sequence Invariants**")
        if snapshot.sequence_failure_count == 0:
            st.success("✅ 0 Sequence Failures (Zero out-of-order or duplicate records)")
        else:
            st.error(f"❌ {snapshot.sequence_failure_count} Sequence Failures")


if __name__ == "__main__":
    main()
