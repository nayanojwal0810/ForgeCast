"""Streamlit demonstration UI for ForgeCast 15-minute industrial energy forecasting."""

import os
from pathlib import Path
import sys

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Keep hosted inference lightweight.
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import streamlit as st

from forgecast.ui.service import run_bounded_demonstration


def _inject_styles() -> None:
    """Apply lightweight presentation styling for the portfolio demo."""
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 1.4rem 1.5rem;
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 14px;
            margin: 0.5rem 0 1.5rem 0;
        }

        .hero-label {
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            opacity: 0.7;
            margin-bottom: 0.35rem;
        }

        .hero-value {
            font-size: 2.15rem;
            font-weight: 750;
            line-height: 1.1;
            margin-bottom: 0.3rem;
        }

        .hero-caption {
            font-size: 0.95rem;
            opacity: 0.75;
        }

        .section-note {
            opacity: 0.72;
            font-size: 0.92rem;
        }

        .highlight-card {
            padding: 1.15rem 1.2rem;
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 12px;
            height: 100%;
        }

        .highlight-title {
            font-weight: 700;
            font-size: 1rem;
            margin-bottom: 0.4rem;
        }

        .highlight-text {
            opacity: 0.78;
            line-height: 1.5;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="ForgeCast | Industrial Energy Forecasting",
        page_icon="⚡",
        layout="wide",
    )

    _inject_styles()

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    st.title("⚡ ForgeCast")
    st.subheader("15-Minute-Ahead Industrial Energy Forecasting")
    st.markdown(
        "An end-to-end forecasting system for predicting near-term industrial "
        "energy demand from historical telemetry."
    )

    # ------------------------------------------------------------------
    # Primary model evidence
    # ------------------------------------------------------------------
    st.markdown(
        """
        <div class="hero">
            <div class="hero-label">Model Result</div>
            <div class="hero-value">28.05% lower MAE than persistence</div>
            <div class="hero-caption">
                Chronological evaluation across 10,512 held-out observations
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # Demo controls
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("Demo")
        record_count = st.slider(
            "Replay window",
            min_value=100,
            max_value=300,
            value=150,
            step=10,
            help="Number of historical 15-minute intervals replayed through the forecasting pipeline.",
        )
        run_btn = st.button("Run Demo", use_container_width=True)

    cache_key = str(record_count)

    if (
        "last_cache_key" not in st.session_state
        or st.session_state["last_cache_key"] != cache_key
        or run_btn
    ):
        with st.spinner("Running forecasting demonstration..."):
            st.session_state["snapshot"] = run_bounded_demonstration(
                record_count=record_count,
                inject_gap=False,
            )
            st.session_state["last_cache_key"] = cache_key

    snapshot = st.session_state["snapshot"]

    # ------------------------------------------------------------------
    # Latest completed forecast
    # ------------------------------------------------------------------
    st.divider()
    st.markdown("### Latest Completed Forecast")

    if snapshot.latest_feedback is not None:
        fb = snapshot.latest_feedback

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "ML Forecast",
                f"{fb.predicted_usage_kwh:.2f} kWh",
            )

        with c2:
            st.metric(
                "Persistence Baseline",
                f"{fb.baseline_persistence_kwh:.2f} kWh",
            )

        with c3:
            st.metric(
                "Actual Usage",
                f"{fb.actual_usage_kwh:.2f} kWh",
            )

        e1, e2 = st.columns(2)

        with e1:
            st.metric(
                "ML Forecast Error",
                f"{fb.ml_absolute_error:.2f} kWh",
            )

        with e2:
            st.metric(
                "Persistence Error",
                f"{fb.persistence_absolute_error:.2f} kWh",
            )

        st.caption(
            f"Completed target interval: "
            f"{fb.target_timestamp.strftime('%Y-%m-%d %H:%M')}"
        )

    else:
        st.info("Run the demo to view a completed forecast.")

    # ------------------------------------------------------------------
    # Model performance
    # ------------------------------------------------------------------
    st.divider()
    st.markdown("### Model Performance")

    p1, p2, p3 = st.columns(3)

    with p1:
        st.metric(
            "ML MAE",
            "3.8626 kWh",
        )
        st.caption("Chronological held-out evaluation")

    with p2:
        st.metric(
            "Persistence MAE",
            "5.3688 kWh",
        )
        st.caption("Same 10,512 held-out observations")

    with p3:
        st.metric(
            "MAE Reduction",
            "28.05%",
        )
        st.caption("ML vs. persistence")

    st.markdown(
        '<div class="section-note">'
        "A separate post-training replay achieved 3.2133 kWh MAE, "
        "24.45% lower than persistence across 3,504 held-out intervals."
        "</div>",
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # Forecast chart
    # ------------------------------------------------------------------
    st.divider()
    st.markdown("### Forecast vs. Actual Usage")

    if snapshot.recent_history:
        df_chart = pd.DataFrame(snapshot.recent_history)
        df_chart["target_timestamp"] = pd.to_datetime(
            df_chart["target_timestamp"]
        )

        # Keep the visual focused on the most recent completed intervals.
        df_chart = (
            df_chart
            .sort_values("target_timestamp")
            .tail(48)
            .set_index("target_timestamp")
        )

        df_chart = df_chart[
            [
                "actual_usage_kwh",
                "predicted_usage_kwh",
                "baseline_persistence_kwh",
            ]
        ]

        df_chart.columns = [
            "Actual Usage (kWh)",
            "ML Forecast (kWh)",
            "Persistence Baseline (kWh)",
        ]

        st.line_chart(df_chart, use_container_width=True)

        st.caption(
            "Recent completed forecast intervals compared with actual usage "
            "and the persistence baseline."
        )
    else:
        st.info("Chart will populate after completed forecast feedback is available.")

    # ------------------------------------------------------------------
    # Engineering highlights
    # ------------------------------------------------------------------
    st.divider()
    st.markdown("### Engineering Highlights")

    h1, h2 = st.columns(2)

    with h1:
        st.markdown(
            """
            <div class="highlight-card">
                <div class="highlight-title">End-to-End ML Lifecycle</div>
                <div class="highlight-text">
                    Data validation, feature engineering, forecasting,
                    feedback, monitoring, and model retraining.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with h2:
        st.markdown(
            """
            <div class="highlight-card">
                <div class="highlight-title">Leakage-Safe Time-Series Modeling</div>
                <div class="highlight-text">
                    Causal feature generation with chronological evaluation
                    and persistence-baseline benchmarking.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    h3, h4 = st.columns(2)

    with h3:
        st.markdown(
            """
            <div class="highlight-card">
                <div class="highlight-title">Operational ML Reliability</div>
                <div class="highlight-text">
                    Stateful inference, delayed ground-truth handling,
                    and robust recovery from missing telemetry.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with h4:
        st.markdown(
            """
            <div class="highlight-card">
                <div class="highlight-title">Reproducible ML Delivery</div>
                <div class="highlight-text">
                    Versioned model artifacts, automated test coverage,
                    and a hosted demonstration workflow.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------------
    # Data attribution
    # ------------------------------------------------------------------
    st.divider()
    st.caption(
        "Demo uses the "
        "[UCI Steel Industry Energy Consumption dataset]"
        "(https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption). "
        "The application demonstrates the forecasting lifecycle over historical telemetry."
    )


if __name__ == "__main__":
    main()
