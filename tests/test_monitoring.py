"""Comprehensive tests for the operational monitoring layer: metrics aggregator and baseline comparison tracker."""

from datetime import datetime, timedelta
import math

import pytest

from forgecast.monitoring.metrics import (
    BaselineComparisonSummary,
    BaselineComparisonTracker,
    MetricSnapshot,
    MonitoringContinuityError,
    MonitoringError,
    RollingMetricsAggregator,
)
from forgecast.operational.events import FeedbackRecord


def _make_feedback(
    origin_dt: datetime,
    actual: float,
    pred: float,
    baseline: float,
    version: str = "v1",
) -> FeedbackRecord:
    """Helper creating a valid FeedbackRecord with derived absolute errors."""
    target_dt = origin_dt + timedelta(minutes=15)
    return FeedbackRecord(
        origin_timestamp=origin_dt,
        target_timestamp=target_dt,
        actual_usage_kwh=actual,
        predicted_usage_kwh=pred,
        baseline_persistence_kwh=baseline,
        ml_absolute_error=abs(actual - pred),
        persistence_absolute_error=abs(actual - baseline),
        model_version=version,
        origin_index=0,
        target_index=1,
    )


def test_empty_snapshot_null_semantics() -> None:
    """Verify that an empty window or tracker returns null semantics without misleading zeros."""
    aggregator = RollingMetricsAggregator(window_sizes=(96, 672))
    snap_cum = aggregator.get_snapshot(model_version="v1", window_size=None)
    snap_roll = aggregator.get_snapshot(model_version="v1", window_size=96)

    for snap in [snap_cum, snap_roll]:
        assert snap.sample_count == 0
        assert snap.first_target_timestamp is None
        assert snap.last_target_timestamp is None
        assert snap.ml_mae is None
        assert snap.ml_rmse is None
        assert snap.baseline_mae is None
        assert snap.baseline_rmse is None
        assert snap.mae_improvement_pct is None
        assert snap.rmse_improvement_pct is None
        assert snap.ml_win_rate is None


def test_hand_calculated_math_and_win_rate() -> None:
    """Verify exact mathematical correctness of MAE, RMSE, relative improvement, and win rate.

    Records:
    1: actual=10.0, pred=9.0 (err=1.0), base=8.0 (err=2.0) -> ML wins
    2: actual=20.0, pred=24.0 (err=4.0), base=22.0 (err=2.0) -> Base wins
    3: actual=30.0, pred=33.0 (err=3.0), base=27.0 (err=3.0) -> Tie (ML does not win)
    4: actual=40.0, pred=40.0 (err=0.0), base=45.0 (err=5.0) -> ML wins

    Totals:
    ML abs errors: [1.0, 4.0, 3.0, 0.0] -> sum = 8.0, MAE = 2.0
    Base abs errors: [2.0, 2.0, 3.0, 5.0] -> sum = 12.0, MAE = 3.0
    MAE improvement: (3.0 - 2.0) / 3.0 * 100 = 33.33333333%

    ML sq errors: [1.0, 16.0, 9.0, 0.0] -> sum = 26.0, RMSE = sqrt(6.5) = 2.5495097567963922
    Base sq errors: [4.0, 4.0, 9.0, 25.0] -> sum = 42.0, RMSE = sqrt(10.5) = 3.24037034920393
    RMSE improvement: (3.24037034920393 - 2.5495097567963922) / 3.24037034920393 * 100 = 21.32042168%

    Wins: records 1 and 4 out of 4 (record 3 is a tie) -> win_rate = 2/4 = 0.5 (50%)
    """
    aggregator = RollingMetricsAggregator(window_sizes=(4,))
    t0 = datetime(2018, 1, 1, 0, 0)

    records = [
        _make_feedback(t0, actual=10.0, pred=9.0, baseline=8.0),
        _make_feedback(t0 + timedelta(minutes=15), actual=20.0, pred=24.0, baseline=22.0),
        _make_feedback(t0 + timedelta(minutes=30), actual=30.0, pred=33.0, baseline=27.0),
        _make_feedback(t0 + timedelta(minutes=45), actual=40.0, pred=40.0, baseline=45.0),
    ]

    for r in records:
        aggregator.process_feedback(r)

    snap = aggregator.get_snapshot("v1", window_size=4)
    assert snap.sample_count == 4
    assert snap.ml_mae == pytest.approx(2.0, rel=1e-6)
    assert snap.baseline_mae == pytest.approx(3.0, rel=1e-6)
    assert snap.mae_improvement_pct == pytest.approx(100.0 / 3.0, rel=1e-6)

    expected_ml_rmse = math.sqrt(6.5)
    expected_base_rmse = math.sqrt(10.5)
    assert snap.ml_rmse == pytest.approx(expected_ml_rmse, rel=1e-6)
    assert snap.baseline_rmse == pytest.approx(expected_base_rmse, rel=1e-6)
    assert snap.rmse_improvement_pct == pytest.approx(
        (expected_base_rmse - expected_ml_rmse) / expected_base_rmse * 100.0, rel=1e-6
    )
    assert snap.ml_win_rate == pytest.approx(0.5, rel=1e-6)


def test_rolling_window_capacity_and_eviction() -> None:
    """Verify FIFO eviction when window reaches capacity and correct incremental sample_count."""
    aggregator = RollingMetricsAggregator(window_sizes=(2,))
    t0 = datetime(2018, 1, 1, 0, 0)

    # Record 1 (err=1.0)
    aggregator.process_feedback(_make_feedback(t0, actual=10.0, pred=9.0, baseline=10.0))
    s1 = aggregator.get_snapshot("v1", window_size=2)
    assert s1.sample_count == 1
    assert s1.ml_mae == pytest.approx(1.0)

    # Record 2 (err=3.0) -> window has [1.0, 3.0]
    aggregator.process_feedback(
        _make_feedback(t0 + timedelta(minutes=15), actual=10.0, pred=7.0, baseline=10.0)
    )
    s2 = aggregator.get_snapshot("v1", window_size=2)
    assert s2.sample_count == 2
    assert s2.ml_mae == pytest.approx(2.0)  # (1.0 + 3.0)/2

    # Record 3 (err=5.0) -> record 1 evicted; window has [3.0, 5.0]
    aggregator.process_feedback(
        _make_feedback(t0 + timedelta(minutes=30), actual=10.0, pred=5.0, baseline=10.0)
    )
    s3 = aggregator.get_snapshot("v1", window_size=2)
    assert s3.sample_count == 2
    assert s3.ml_mae == pytest.approx(4.0)  # (3.0 + 5.0)/2
    assert s3.first_target_timestamp == t0 + timedelta(minutes=30)
    assert s3.last_target_timestamp == t0 + timedelta(minutes=45)

    # Cumulative should still have all 3 records: (1.0 + 3.0 + 5.0)/3 = 3.0
    cum = aggregator.get_snapshot("v1", window_size=None)
    assert cum.sample_count == 3
    assert cum.ml_mae == pytest.approx(3.0)


def test_zero_baseline_safe_handling() -> None:
    """Verify safe zero division handling when baseline error is zero."""
    aggregator = RollingMetricsAggregator(window_sizes=(2,))
    t0 = datetime(2018, 1, 1, 0, 0)

    # actual == baseline -> baseline error = 0.0
    aggregator.process_feedback(_make_feedback(t0, actual=10.0, pred=12.0, baseline=10.0))
    snap = aggregator.get_snapshot("v1", window_size=2)
    assert snap.baseline_mae == 0.0
    assert snap.mae_improvement_pct == 0.0
    assert snap.rmse_improvement_pct == 0.0


def test_temporal_out_of_order_and_duplicate_rejection() -> None:
    """Verify fail-closed behavior for backward or duplicate feedback timestamps."""
    aggregator = RollingMetricsAggregator(window_sizes=(96,))
    t0 = datetime(2018, 1, 1, 0, 0)

    aggregator.process_feedback(_make_feedback(t0, actual=10.0, pred=10.0, baseline=10.0))

    # Duplicate target timestamp
    with pytest.raises(MonitoringContinuityError, match="Duplicate feedback target timestamp"):
        aggregator.process_feedback(_make_feedback(t0, actual=12.0, pred=11.0, baseline=10.0))

    # Out of order target timestamp
    with pytest.raises(MonitoringContinuityError, match="Out-of-order feedback target timestamp"):
        aggregator.process_feedback(
            _make_feedback(t0 - timedelta(minutes=15), actual=10.0, pred=10.0, baseline=10.0)
        )


def test_model_version_isolation() -> None:
    """Verify that distinct model versions maintain isolated statistics and do not contaminate each other."""
    aggregator = RollingMetricsAggregator(window_sizes=(2,))
    t0 = datetime(2018, 1, 1, 0, 0)

    # Feed version v1: ML error 1.0
    aggregator.process_feedback(
        _make_feedback(t0, actual=10.0, pred=9.0, baseline=8.0, version="v1")
    )
    # Feed version v2: ML error 10.0
    aggregator.process_feedback(
        _make_feedback(t0, actual=10.0, pred=20.0, baseline=8.0, version="v2")
    )

    snap_v1 = aggregator.get_snapshot("v1", window_size=None)
    snap_v2 = aggregator.get_snapshot("v2", window_size=None)

    assert snap_v1.model_version == "v1"
    assert snap_v1.sample_count == 1
    assert snap_v1.ml_mae == pytest.approx(1.0)

    assert snap_v2.model_version == "v2"
    assert snap_v2.sample_count == 1
    assert snap_v2.ml_mae == pytest.approx(10.0)


def test_baseline_comparison_tracker_summary() -> None:
    """Verify BaselineComparisonTracker coordinates cumulative and rolling windows into BaselineComparisonSummary."""
    tracker = BaselineComparisonTracker(window_sizes=(2, 4))
    t0 = datetime(2018, 1, 1, 0, 0)

    for i in range(5):
        dt = t0 + timedelta(minutes=15 * i)
        tracker.process_feedback(_make_feedback(dt, actual=10.0, pred=9.0, baseline=8.0, version="v1"))

    summary: BaselineComparisonSummary = tracker.get_comparison_summary("v1")
    assert summary.model_version == "v1"
    assert summary.cumulative.sample_count == 5
    assert summary.rolling_windows[2].sample_count == 2
    assert summary.rolling_windows[4].sample_count == 4

    summary_dict = summary.to_dict()
    assert summary_dict["model_version"] == "v1"
    assert summary_dict["cumulative"]["sample_count"] == 5
    assert summary_dict["rolling_windows"][2]["sample_count"] == 2


def test_operational_engine_monitoring_integration() -> None:
    """Verify OperationalEngine automatically forwards resolved feedback to its monitoring tracker."""
    from forgecast.features.generator import FEATURE_NAMES
    from forgecast.ingestion.contract import TelemetryRecord
    from forgecast.models.config import FrozenModelConfig
    from forgecast.models.forecaster import EnergyForecaster
    from forgecast.operational.engine import OperationalEngine
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    df_x = pd.DataFrame({col: rng.uniform(0.0, 50.0, size=20) for col in FEATURE_NAMES})
    y = rng.uniform(1.0, 50.0, size=20)
    forecaster = EnergyForecaster(FrozenModelConfig())
    forecaster.fit(df_x, y)

    meta = {
        "model_version": "v1",
        "model_class": "HistGradientBoostingRegressor",
        "feature_names": list(FEATURE_NAMES),
        "hyperparameters": FrozenModelConfig().to_dict(),
    }
    tracker = BaselineComparisonTracker(window_sizes=(96, 672))
    engine = OperationalEngine(forecaster=forecaster, metadata=meta, monitoring_tracker=tracker)

    def _rec(dt: datetime, usage: float = 10.0, idx: int = 0) -> TelemetryRecord:
        return TelemetryRecord(
            logical_timestamp=dt,
            raw_date=dt.strftime("%d/%m/%Y %H:%M"),
            usage_kwh=usage,
            lagging_reactive_power_kvarh=1.0,
            leading_reactive_power_kvarh=0.0,
            co2_tco2=0.0,
            lagging_power_factor=80.0,
            leading_power_factor=100.0,
            nsm=dt.hour * 3600 + dt.minute * 60,
            week_status="Weekday",
            day_of_week=dt.strftime("%A"),
            load_type="Light_Load",
            row_index=idx,
        )

    t0 = datetime(2018, 1, 1, 0, 0)
    # Feed 96 records to warm up
    for i in range(96):
        engine.process_telemetry(_rec(t0 + timedelta(minutes=15 * i), idx=i))

    # At 96th record, prediction for 97th is emitted.
    # At 97th record, feedback for 97th is resolved and fed to tracker.
    step97 = engine.process_telemetry(_rec(t0 + timedelta(minutes=15 * 96), idx=96))
    assert step97.feedback is not None

    snap = tracker.get_cumulative_snapshot("v1")
    assert snap.sample_count == 1
    assert snap.ml_mae is not None
    assert snap.baseline_mae is not None


def test_deployment_boundary_classification() -> None:
    """Verify that feedback before/at deployment boundary is pre-deployment only, and feedback after is post-deployment."""
    boundary = datetime(2018, 11, 25, 12, 0)
    tracker = BaselineComparisonTracker(window_sizes=(2,), deployment_boundary=boundary)

    # 1. Target timestamp before boundary (2018-11-25 11:45:00)
    t_before = datetime(2018, 11, 25, 11, 30)
    tracker.process_feedback(_make_feedback(t_before, actual=10.0, pred=9.0, baseline=8.0))

    # 2. Target timestamp exactly at boundary (2018-11-25 12:00:00)
    t_at = datetime(2018, 11, 25, 11, 45)
    tracker.process_feedback(_make_feedback(t_at, actual=12.0, pred=11.0, baseline=10.0))

    # Before and at boundary: full replay has 2, post-deployment has 0
    full_snap = tracker.get_full_replay_snapshot("v1")
    post_snap = tracker.get_post_deployment_snapshot("v1")
    assert full_snap.sample_count == 2
    assert post_snap.sample_count == 0
    assert post_snap.ml_mae is None

    # 3. Target timestamp strictly after boundary (2018-11-25 12:15:00)
    t_after1 = datetime(2018, 11, 25, 12, 0)
    tracker.process_feedback(_make_feedback(t_after1, actual=20.0, pred=18.0, baseline=15.0))

    # 4. Target timestamp strictly after boundary (2018-11-25 12:30:00)
    t_after2 = datetime(2018, 11, 25, 12, 15)
    tracker.process_feedback(_make_feedback(t_after2, actual=30.0, pred=25.0, baseline=20.0))

    full_snap2 = tracker.get_full_replay_snapshot("v1")
    post_snap2 = tracker.get_post_deployment_snapshot("v1")

    # Full replay has all 4
    assert full_snap2.sample_count == 4
    # Post deployment has exactly 2
    assert post_snap2.sample_count == 2
    assert post_snap2.first_target_timestamp == datetime(2018, 11, 25, 12, 15)
    assert post_snap2.last_target_timestamp == datetime(2018, 11, 25, 12, 30)
    # Post deployment ML error: (abs(20-18) + abs(30-25)) / 2 = (2.0 + 5.0) / 2 = 3.5
    assert post_snap2.ml_mae == pytest.approx(3.5)
    # Post deployment Persistence error: (abs(20-15) + abs(30-20)) / 2 = (5.0 + 10.0) / 2 = 7.5
    assert post_snap2.baseline_mae == pytest.approx(7.5)


def test_metric_labeling_and_summary_distinction() -> None:
    """Verify that full replay and post-deployment summaries are cleanly separated in BaselineComparisonSummary."""
    boundary = datetime(2018, 1, 1, 12, 0)
    tracker = BaselineComparisonTracker(window_sizes=(2,), deployment_boundary=boundary)

    t0 = datetime(2018, 1, 1, 11, 45)  # Target 12:00 (at boundary)
    t1 = datetime(2018, 1, 1, 12, 0)   # Target 12:15 (post boundary)

    tracker.process_feedback(_make_feedback(t0, actual=10.0, pred=9.0, baseline=8.0))
    tracker.process_feedback(_make_feedback(t1, actual=20.0, pred=19.0, baseline=18.0))

    summary = tracker.get_comparison_summary("v1")
    assert summary.deployment_boundary == boundary
    assert summary.full_replay_cumulative.sample_count == 2
    assert summary.post_deployment_cumulative is not None
    assert summary.post_deployment_cumulative.sample_count == 1
    assert summary.post_deployment_cumulative.first_target_timestamp == datetime(2018, 1, 1, 12, 15)


def test_window_diagnostic_correctness() -> None:
    """Verify hand-calculable WindowDiagnostic statistics, load regimes, and error concentrations."""
    from forgecast.monitoring.metrics import diagnose_recent_window

    t0 = datetime(2018, 1, 1, 0, 0)
    # 4 synthetic records with flat baseload actuals (around 3.5 kWh)
    records = [
        _make_feedback(t0, actual=3.5, pred=4.5, baseline=3.5),                           # ml_err=1.0, base_err=0.0 (Base wins)
        _make_feedback(t0 + timedelta(minutes=15), actual=3.6, pred=4.6, baseline=3.5),   # ml_err=1.0, base_err=0.1 (Base wins)
        _make_feedback(t0 + timedelta(minutes=30), actual=3.4, pred=4.4, baseline=3.6),   # ml_err=1.0, base_err=0.2 (Base wins)
        _make_feedback(t0 + timedelta(minutes=45), actual=3.5, pred=3.5, baseline=3.0),   # ml_err=0.0, base_err=0.5 (ML wins)
    ]

    diag = diagnose_recent_window(records, window_name="test_window")
    assert diag.sample_count == 4
    # ML errors: [1.0, 1.0, 1.0, 0.0] -> MAE = 0.75
    assert diag.ml_mae == pytest.approx(0.75)
    # Base errors: [0.0, 0.1, 0.2, 0.5] -> MAE = 0.8/4 = 0.2
    assert diag.persistence_mae == pytest.approx(0.2)
    # Difference: 0.75 - 0.2 = 0.55
    assert diag.mae_difference == pytest.approx(0.55)
    # Wins: ML=1, Base=3 -> win rates: ML=0.25, Base=0.75
    assert diag.ml_win_rate == pytest.approx(0.25)
    assert diag.persistence_win_rate == pytest.approx(0.75)
    assert diag.tie_count == 0

    # Usages: [3.5, 3.6, 3.4, 3.5] -> mean=3.5, std ~ 0.0707
    assert diag.usage_mean == pytest.approx(3.5)
    assert diag.usage_std < 0.1
    assert diag.mean_prediction == pytest.approx(4.25)
    assert diag.mean_actual == pytest.approx(3.5)
    assert diag.mean_signed_error == pytest.approx(0.75)
    assert diag.top_10pct_sample_count == 1
    assert diag.top_10pct_error_sum == pytest.approx(1.0)
    assert diag.total_ml_error_sum == pytest.approx(3.0)
    assert diag.top_10pct_error_fraction == pytest.approx(1.0 / 3.0)
    assert "Flat baseload regime" in diag.error_concentration_summary
    assert "positive signed bias" in diag.error_concentration_summary


def test_signed_bias_directional_correctness() -> None:
    """Verify signed error bias correctly distinguishes positive, zero, and negative mean residuals."""
    from forgecast.monitoring.metrics import diagnose_recent_window

    t0 = datetime(2018, 1, 1, 0, 0)

    # 1. Negative bias (model underpredicts)
    neg_records = [
        _make_feedback(t0, actual=10.0, pred=8.0, baseline=10.0),
        _make_feedback(t0 + timedelta(minutes=15), actual=20.0, pred=18.0, baseline=20.0),
    ]
    diag_neg = diagnose_recent_window(neg_records, window_name="neg_test")
    assert diag_neg.mean_signed_error == pytest.approx(-2.0)
    assert diag_neg.mean_prediction == pytest.approx(13.0)
    assert diag_neg.mean_actual == pytest.approx(15.0)

    # 2. Zero bias (balanced errors)
    zero_records = [
        _make_feedback(t0, actual=10.0, pred=12.0, baseline=10.0),
        _make_feedback(t0 + timedelta(minutes=15), actual=20.0, pred=18.0, baseline=20.0),
    ]
    diag_zero = diagnose_recent_window(zero_records, window_name="zero_test")
    assert diag_zero.mean_signed_error == pytest.approx(0.0)


def test_top10pct_concentration_deterministic_ceil() -> None:
    """Verify deterministic ceil behavior for top-10% selection on 96 observations."""
    from forgecast.monitoring.metrics import diagnose_recent_window

    t0 = datetime(2018, 1, 1, 0, 0)
    # Construct 96 records: 10 large errors (10.0 each), 86 small errors (1.0 each)
    records = []
    for i in range(10):
        records.append(_make_feedback(t0 + timedelta(minutes=15 * i), actual=10.0, pred=20.0, baseline=10.0))
    for i in range(10, 96):
        records.append(_make_feedback(t0 + timedelta(minutes=15 * i), actual=10.0, pred=11.0, baseline=10.0))

    diag = diagnose_recent_window(records, window_name="96_test")
    # 96 * 0.10 = 9.6 -> ceil is 10
    assert diag.top_10pct_sample_count == 10
    assert diag.top_10pct_error_sum == pytest.approx(10 * 10.0)
    assert diag.total_ml_error_sum == pytest.approx(10 * 10.0 + 86 * 1.0)
    expected_frac = 100.0 / 186.0
    assert diag.top_10pct_error_fraction == pytest.approx(expected_frac)



