"""Presentation views for the ForgeCast Streamlit application."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from forgecast.ui.service import (
    UIDataSnapshot,
    load_active_model_summary,
    load_chronological_evaluation_summary,
    load_retraining_lifecycle_summary,
    run_bounded_demonstration,
)


def configure_page(page_name: str) -> None:
    st.set_page_config(
        page_title=f"ForgeCast | {page_name}",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1180px;
            padding-top: 1.6rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 1.5rem 1.6rem;
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 16px;
            margin: 0.75rem 0 1.4rem 0;
        }

        .hero-kicker {
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            opacity: 0.65;
            margin-bottom: 0.35rem;
        }

        .hero-value {
            font-size: 2.25rem;
            font-weight: 760;
            line-height: 1.08;
            margin-bottom: 0.35rem;
        }

        .hero-copy {
            font-size: 0.96rem;
            opacity: 0.75;
            max-width: 820px;
            line-height: 1.55;
        }

        .section-copy {
            opacity: 0.72;
            line-height: 1.55;
            max-width: 900px;
        }

        .fc-card {
            padding: 1.15rem 1.2rem;
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 12px;
            height: 100%;
        }

        .fc-card-title {
            font-weight: 720;
            font-size: 1rem;
            margin-bottom: 0.35rem;
        }

        .fc-card-text {
            opacity: 0.75;
            line-height: 1.5;
            font-size: 0.92rem;
        }

        .fc-step {
            padding: 0.95rem 1rem;
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 10px;
            min-height: 104px;
        }

        .fc-step-no {
            font-size: 0.72rem;
            font-weight: 700;
            opacity: 0.55;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.2rem;
        }

        .fc-step-title {
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .fc-step-text {
            font-size: 0.88rem;
            opacity: 0.72;
            line-height: 1.4;
        }

        .status-ok {
            display: inline-block;
            padding: 0.35rem 0.62rem;
            border: 1px solid rgba(46, 125, 50, 0.35);
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
        }

        .status-neutral {
            display: inline-block;
            padding: 0.35rem 0.62rem;
            border: 1px solid rgba(128, 128, 128, 0.28);
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
        }

        .small-note {
            font-size: 0.84rem;
            opacity: 0.68;
            line-height: 1.5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_ts(value: datetime | None) -> str:
    if value is None:
        return "—"
    return value.strftime("%Y-%m-%d %H:%M")


def _get_snapshot(
    session_key: str,
    record_count: int,
    inject_gap: bool,
    gap_record_index: int = 110,
) -> UIDataSnapshot:
    state_key = f"forgecast_snapshot_{session_key}"
    config_key = f"forgecast_config_{session_key}"
    config = (record_count, inject_gap, gap_record_index)

    if (
        state_key not in st.session_state
        or st.session_state.get(config_key) != config
    ):
        with st.spinner("Replaying historical telemetry..."):
            st.session_state[state_key] = run_bounded_demonstration(
                record_count=record_count,
                inject_gap=inject_gap,
                gap_record_index=gap_record_index,
            )
            st.session_state[config_key] = config

    return st.session_state[state_key]


def _render_flow(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for idx, ((title, body), col) in enumerate(zip(items, cols), start=1):
        with col:
            st.markdown(
                f"""
                <div class="fc-step">
                    <div class="fc-step-no">Step {idx}</div>
                    <div class="fc-step-title">{title}</div>
                    <div class="fc-step-text">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_overview() -> None:
    st.title("⚡ ForgeCast")
    st.subheader("15-minute-ahead industrial energy forecasting")
    st.markdown(
        '<div class="section-copy">'
        "Forecast the next 15-minute energy-use interval from historical telemetry, "
        "then score each prediction only after the corresponding observation arrives."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Historical evaluation</div>
            <div class="hero-value">28.1% lower MAE than persistence</div>
            <div class="hero-copy">
                Three expanding chronological holdout windows covering 10,512 observations.
                The result compares the forecasting model with a simple persistence baseline
                available at prediction time.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, value in [
        (c1, "Forecast horizon", "15 min"),
        (c2, "History used", "96 intervals"),
        (c3, "ML MAE", "3.86 kWh"),
        (c4, "Persistence MAE", "5.37 kWh"),
        (c5, "MAE improvement", "28.1%"),
    ]:
        with col:
            st.metric(label, value)

    st.divider()
    st.markdown("### From telemetry to a measured result")
    st.markdown(
        '<div class="section-copy">'
        "The application treats forecasting as a sequence of time-aware events rather than "
        "a single model call."
        "</div>",
        unsafe_allow_html=True,
    )
    _render_flow(
        [
            ("Telemetry", "Receive the next validated 15-minute observation."),
            ("Validation", "Check the source contract, values, timestamp, and cadence."),
            ("Features", "Build a causal feature vector from the latest 96 observations."),
            ("Forecast", "Predict the next interval and record the persistence baseline."),
            ("Feedback", "Wait for the target interval and pair it by timestamp."),
            ("Monitoring", "Update recent and cumulative performance from completed results."),
            ("Candidate review", "Evaluate retraining candidates against an explicit baseline gate."),
        ]
    )

    st.divider()
    st.markdown("### Why the result is meaningful")
    a, b, c = st.columns(3)
    for col, title, body in [
        (
            a,
            "Chronological evaluation",
            "Later observations are held out in three expanding windows instead of being randomly shuffled.",
        ),
        (
            b,
            "Causal feature design",
            "The model uses only information available at the forecast cutoff; target-interval measurements are excluded.",
        ),
        (
            c,
            "A strong reference point",
            "Every forecast is compared with Usage(t), a simple persistence forecast that is available at inference time.",
        ),
    ]:
        with col:
            st.markdown(
                f"""
                <div class="fc-card">
                    <div class="fc-card-title">{title}</div>
                    <div class="fc-card-text">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()
    st.info(
        "This application replays historical UCI Steel Industry Energy Consumption data. "
        "It is a portfolio demonstration, not a live plant connection."
    )


def render_forecast_demo() -> None:
    st.title("Forecast Demo")
    st.markdown(
        '<div class="section-copy">'
        "Step through a bounded historical replay and watch a forecast move from creation "
        "to delayed measurement."
        "</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Replay")
        record_count = st.slider(
            "Historical intervals",
            min_value=100,
            max_value=300,
            value=180,
            step=10,
            help="Number of historical 15-minute observations to replay.",
        )
        st.caption("The replay uses the same operational engine as the project runtime.")

    snapshot = _get_snapshot("forecast", record_count, False)

    st.divider()
    st.markdown("### Replay status")
    s1, s2, s3, s4 = st.columns(4)
    for col, label, value in [
        (s1, "Observations consumed", str(snapshot.records_consumed)),
        (s2, "Forecasts generated", str(max(0, snapshot.records_consumed - 95))),
        (s3, "Feedback completed", str(snapshot.completed_feedback_count)),
        (s4, "Forecasts still pending", str(snapshot.pending_prediction_count)),
    ]:
        with col:
            st.metric(label, value)

    st.caption(
        f"Last telemetry timestamp: {_format_ts(snapshot.latest_logical_timestamp)}"
    )

    st.divider()
    st.markdown("### The latest forecast waiting for its target")
    if snapshot.latest_prediction is not None:
        p = snapshot.latest_prediction
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("ML forecast", f"{p.predicted_usage_kwh:.2f} kWh")
        with c2:
            st.metric("Persistence forecast", f"{p.baseline_persistence_kwh:.2f} kWh")
        with c3:
            st.metric("Target interval", _format_ts(p.target_timestamp))
        st.markdown(
            '<div class="small-note">'
            "This prediction is stored before its target value exists. It becomes measurable only "
            "when telemetry for the target timestamp arrives."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("The replay has not accumulated enough history to produce a forecast.")

    st.divider()
    st.markdown("### The most recent measured result")
    if snapshot.latest_feedback is not None:
        fb = snapshot.latest_feedback
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("ML forecast", f"{fb.predicted_usage_kwh:.2f} kWh")
        with c2:
            st.metric("Actual usage", f"{fb.actual_usage_kwh:.2f} kWh")
        with c3:
            st.metric("Persistence forecast", f"{fb.baseline_persistence_kwh:.2f} kWh")

        e1, e2 = st.columns(2)
        with e1:
            st.metric("ML absolute error", f"{fb.ml_absolute_error:.2f} kWh")
        with e2:
            st.metric(
                "Persistence absolute error",
                f"{fb.persistence_absolute_error:.2f} kWh",
            )
        st.caption(f"Target timestamp: {_format_ts(fb.target_timestamp)}")
    else:
        st.info("A measured result will appear after the first delayed feedback event is matched.")

    st.divider()
    st.markdown("### One forecast, end to end")
    _render_flow(
        [
            ("1 · Receive", "Telemetry for time t enters the validated runtime."),
            ("2 · Forecast", "The model predicts Usage(t + 15 min)."),
            ("3 · Hold", "The prediction remains pending by target timestamp."),
            ("4 · Match", "The target observation arrives and closes the prediction."),
            ("5 · Measure", "ML and persistence errors are calculated from the same feedback record."),
            ("6 · Monitor", "The completed result updates rolling and cumulative metrics."),
        ]
    )

    st.divider()
    st.markdown("### Recent forecast performance")
    if snapshot.recent_history:
        df = pd.DataFrame(snapshot.recent_history)
        df["target_timestamp"] = pd.to_datetime(df["target_timestamp"])
        df = df.sort_values("target_timestamp").tail(48).set_index("target_timestamp")
        chart = df[
            [
                "actual_usage_kwh",
                "predicted_usage_kwh",
                "baseline_persistence_kwh",
            ]
        ].copy()
        chart.columns = [
            "Actual usage",
            "ML forecast",
            "Persistence",
        ]
        st.line_chart(chart, use_container_width=True)
        st.caption("Most recent completed intervals in the replay.")
    else:
        st.info("The chart will populate once completed feedback is available.")


def render_data_reliability() -> None:
    st.title("Data & Reliability")
    st.markdown(
        '<div class="section-copy">'
        "A forecast is only as trustworthy as the sequence behind it. ForgeCast validates "
        "incoming telemetry and resets forecasting state when continuity is broken."
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### What is checked before forecasting")
    a, b = st.columns(2)
    with a:
        st.markdown(
            """
            <div class="fc-card">
                <div class="fc-card-title">Source contract</div>
                <div class="fc-card-text">
                    Required fields, numeric values, categorical values, null handling,
                    and timestamp formatting are validated before a record enters the runtime.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            """
            <div class="fc-card">
                <div class="fc-card-title">Temporal continuity</div>
                <div class="fc-card-text">
                    Records must follow the 15-minute cadence. A forecast is withheld until
                    96 contiguous Usage observations are available.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("### Operational safeguards")
    a, b, c = st.columns(3)
    for col, title, body in [
        (
            a,
            "Exact feedback matching",
            "Predictions are paired to actuals by logical target timestamp, not by row position.",
        ),
        (
            b,
            "Fail closed",
            "Duplicate and out-of-order events are rejected rather than silently scored.",
        ),
        (
            c,
            "Gap recovery",
            "A detected gap invalidates affected predictions, resets feature history, and requires re-warm-up.",
        ),
    ]:
        with col:
            st.markdown(
                f"""
                <div class="fc-card">
                    <div class="fc-card-title">{title}</div>
                    <div class="fc-card-text">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("### Simulate a missing interval")
    st.markdown(
        '<div class="section-copy">'
        "The scenario below intentionally skips one 15-minute telemetry observation. "
        "The runtime must detect the gap, quarantine the affected prediction, reset its "
        "history, and rebuild 96 contiguous observations before forecasting resumes."
        "</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Reliability scenario")
        record_count = st.slider(
            "Historical intervals",
            min_value=130,
            max_value=300,
            value=220,
            step=10,
            help="Use at least 130 intervals so the injected gap is observable.",
        )

    snapshot = _get_snapshot("gap", record_count, True)

    st.divider()
    g1, g2, g3, g4 = st.columns(4)
    for col, label, value in [
        (g1, "Intervals consumed", str(snapshot.records_consumed)),
        (g2, "Gap incidents", str(snapshot.gap_incident_count)),
        (g3, "Invalidated forecasts", str(snapshot.invalidated_prediction_count)),
        (g4, "Sequence failures", str(snapshot.sequence_failure_count)),
    ]:
        with col:
            st.metric(label, value)

    state_label = "Forecasting ready" if snapshot.is_warmed_up else "Re-warming"
    badge_class = "status-ok" if snapshot.is_warmed_up else "status-neutral"
    st.markdown(
        f'<span class="{badge_class}">{state_label}</span>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Current history: {snapshot.current_history_length}/96 contiguous observations · "
        f"Segment {snapshot.current_segment_id}"
    )

    if snapshot.latest_gap_incident is not None:
        gap = snapshot.latest_gap_incident
        st.markdown("#### What the runtime detected")
        t1, t2, t3 = st.columns(3)
        with t1:
            st.metric("Missing intervals", str(gap.missing_interval_count))
        with t2:
            st.metric("Invalidated forecasts", str(gap.invalidated_prediction_count))
        with t3:
            st.metric("History reset", "Yes" if gap.state_reset else "No")

        st.caption(
            f"Gap from {_format_ts(gap.previous_timestamp)} to "
            f"{_format_ts(gap.current_timestamp)} · "
            f"Re-warm required: {'Yes' if gap.rewarm_required else 'No'}"
        )

        if snapshot.latest_invalidated_prediction is not None:
            inv = snapshot.latest_invalidated_prediction
            st.info(
                f"Prediction for {_format_ts(inv.target_timestamp)} was not scored because "
                "its actual observation fell inside the missing interval."
            )
    else:
        st.info("The reliability scenario has not produced a gap incident.")

    st.divider()
    st.markdown("### Replay integrity")
    r1, r2, r3 = st.columns(3)
    r1.metric("Completed feedback", snapshot.completed_feedback_count)
    r2.metric("Unmatched actuals", snapshot.unmatched_feedback_count)
    r3.metric("Pending forecasts", snapshot.pending_prediction_count)
    st.caption(
        "The replay reads and validates source records without modifying the dataset or model artifact."
    )


def render_model_lifecycle() -> None:
    st.title("Model Lifecycle")
    st.markdown(
        '<div class="section-copy">'
        "See how ForgeCast keeps the currently served artifact separate from a retraining "
        "candidate and makes the next model a measured decision."
        "</div>",
        unsafe_allow_html=True,
    )

    active = load_active_model_summary()
    retrain = load_retraining_lifecycle_summary()
    evaluation = load_chronological_evaluation_summary()

    if active is None or retrain is None:
        st.warning("Lifecycle artifacts are not available in this environment.")
        return

    st.divider()
    st.markdown("### Current model and next candidate")
    a, b = st.columns(2)
    with a:
        st.markdown(
            f"""
            <div class="fc-card">
                <div class="fc-card-title">Current replay model</div>
                <div class="fc-card-text">
                    <strong>{active['model_version']}</strong> · {active['model_class']}<br>
                    {active['training_sample_count']:,} training observations ·
                    {active['feature_count']} model features
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            f"""
            <div class="fc-card">
                <div class="fc-card-title">Retraining candidate</div>
                <div class="fc-card-text">
                    <strong>{retrain['candidate_version']}</strong> · checkpoint {retrain['checkpoint_name']}<br>
                    Candidate MAE {retrain['candidate_mae']:.2f} kWh versus
                    {retrain['persistence_mae']:.2f} kWh for persistence
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("### Evaluation gate")
    d1, d2, d3 = st.columns(3)
    d1.metric("Candidate MAE", f"{retrain['candidate_mae']:.2f} kWh")
    d2.metric("Persistence MAE", f"{retrain['persistence_mae']:.2f} kWh")
    d3.metric("Improvement over baseline", f"{retrain['improvement_pct']:.2f}%")

    st.success(
        "The candidate passes the implemented 5% improvement requirement over persistence."
    )
    st.caption(
        "The candidate is stored separately. The hosted replay continues to use the existing v1 artifact."
    )

    st.divider()
    st.markdown("### Temporal boundaries")
    b1, b2 = st.columns(2)
    with b1:
        st.markdown("**Current model training window**")
        st.write(
            f"{_format_ts(datetime.fromisoformat(active['training_start_target']))} "
            f"→ {_format_ts(datetime.fromisoformat(active['training_end_target']))}"
        )
    with b2:
        st.markdown("**Candidate validation window**")
        st.write(
            f"{_format_ts(datetime.fromisoformat(retrain['eval_start']))} "
            f"→ {_format_ts(datetime.fromisoformat(retrain['eval_end']))}"
        )

    st.divider()
    st.markdown("### Historical evaluation record")
    if evaluation and evaluation.get("folds"):
        rows = []
        for fold in evaluation["folds"]:
            rows.append(
                {
                    "Window": f"Fold {fold['fold_index']}",
                    "ML MAE": round(fold["ml_mae"], 3),
                    "Persistence MAE": round(fold["persistence_mae"], 3),
                    "Improvement": f"{fold['ml_improvement_vs_persistence_pct']:.2f}%",
                    "Held-out observations": fold["eval_sample_count"],
                }
            )
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"Pooled result: {evaluation['ml_mae']:.2f} kWh MAE versus "
            f"{evaluation['persistence_mae']:.2f} kWh for persistence across "
            f"{evaluation['total_samples']:,} observations."
        )

    st.divider()
    st.markdown("### Monitoring after replay")
    with st.sidebar:
        st.header("Monitoring")
        monitor_count = st.slider(
            "Replay intervals",
            min_value=200,
            max_value=900,
            value=720,
            step=40,
            help="Longer replays populate the recent monitoring windows.",
        )

    snap = _get_snapshot("monitoring", monitor_count, False)
    m1, m2, m3 = st.columns(3)
    c24 = snap.rolling_24h_snapshot
    c7d = snap.rolling_7d_snapshot
    cumulative = snap.cumulative_snapshot

    m1.metric(
        "24-hour MAE",
        f"{c24.ml_mae:.2f} kWh" if c24 and c24.ml_mae is not None else "—",
    )
    m2.metric(
        "7-day MAE",
        f"{c7d.ml_mae:.2f} kWh" if c7d and c7d.ml_mae is not None else "—",
    )
    m3.metric(
        "Cumulative ML win rate",
        f"{cumulative.ml_win_rate * 100:.1f}%"
        if cumulative and cumulative.ml_win_rate is not None
        else "—",
    )
    st.caption(
        "These monitoring values are recomputed from completed feedback generated by the selected historical replay."
    )

    st.divider()
    st.info(
        "The lifecycle shown here is a controlled historical workflow. It demonstrates "
        "candidate evaluation, explicit performance gates, artifact separation, and traceable model versions."
    )
