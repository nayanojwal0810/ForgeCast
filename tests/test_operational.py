"""Tests for OperationalEngine, PredictionEvent, DelayedFeedbackTracker, and lifecycle ordering."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forgecast.features.generator import FEATURE_NAMES
from forgecast.ingestion.contract import TelemetryRecord
from forgecast.models.config import FrozenModelConfig
from forgecast.models.forecaster import EnergyForecaster
from forgecast.operational.engine import OperationalEngine, validate_artifact_compatibility
from forgecast.operational.events import (
    ArtifactValidationError,
    FeedbackError,
    OperationalContinuityError,
    PredictionEvent,
)
from forgecast.operational.feedback import DelayedFeedbackTracker


def _make_record(logical_dt: datetime, usage: float = 5.0, row_idx: int = 0) -> TelemetryRecord:
    """Helper creating a minimal valid TelemetryRecord."""
    return TelemetryRecord(
        logical_timestamp=logical_dt,
        raw_date=logical_dt.strftime("%d/%m/%Y %H:%M"),
        usage_kwh=usage,
        lagging_reactive_power_kvarh=1.0,
        leading_reactive_power_kvarh=0.0,
        co2_tco2=0.0,
        lagging_power_factor=80.0,
        leading_power_factor=100.0,
        nsm=logical_dt.hour * 3600 + logical_dt.minute * 60,
        week_status="Weekday" if logical_dt.weekday() < 5 else "Weekend",
        day_of_week=logical_dt.strftime("%A"),
        load_type="Light_Load",
        row_index=row_idx,
    )


@pytest.fixture
def mock_engine() -> OperationalEngine:
    """Create an OperationalEngine with a fitted mock EnergyForecaster."""
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
    return OperationalEngine(forecaster=forecaster, metadata=meta)


def test_prediction_event_temporal_invariants() -> None:
    """Verify PredictionEvent enforces origin < target and target == origin + 15m."""
    origin = datetime(2018, 1, 1, 10, 0)
    target = datetime(2018, 1, 1, 10, 15)

    ev = PredictionEvent(
        model_version="v1",
        origin_timestamp=origin,
        target_timestamp=target,
        prediction_timestamp=datetime.now(timezone.utc),
        predicted_usage_kwh=25.0,
        baseline_persistence_kwh=22.0,
        origin_index=0,
    )
    assert ev.origin_timestamp == origin
    assert ev.target_timestamp == target

    # Invalid: origin >= target
    with pytest.raises(ValueError, match="strictly earlier"):
        PredictionEvent(
            model_version="v1",
            origin_timestamp=target,
            target_timestamp=origin,
            prediction_timestamp=datetime.now(timezone.utc),
            predicted_usage_kwh=25.0,
            baseline_persistence_kwh=22.0,
        )

    # Invalid: target != origin + 15m
    with pytest.raises(ValueError, match="Temporal cadence violation"):
        PredictionEvent(
            model_version="v1",
            origin_timestamp=origin,
            target_timestamp=origin + timedelta(minutes=30),
            prediction_timestamp=datetime.now(timezone.utc),
            predicted_usage_kwh=25.0,
            baseline_persistence_kwh=22.0,
        )


def test_artifact_compatibility_validation() -> None:
    """Confirm validate_artifact_compatibility fails closed on any mismatch."""
    forecaster = EnergyForecaster(FrozenModelConfig())
    rng = np.random.default_rng(42)
    df_x = pd.DataFrame({col: rng.uniform(0.0, 50.0, size=10) for col in FEATURE_NAMES})
    y = rng.uniform(1.0, 50.0, size=10)
    forecaster.fit(df_x, y)

    valid_meta = {
        "model_version": "v1",
        "model_class": "HistGradientBoostingRegressor",
        "feature_names": list(FEATURE_NAMES),
        "hyperparameters": FrozenModelConfig().to_dict(),
    }
    # Valid passes
    validate_artifact_compatibility(forecaster, valid_meta)

    # Mismatched model class
    bad_class_meta = dict(valid_meta, model_class="RandomForestRegressor")
    with pytest.raises(ArtifactValidationError, match="Model class mismatch"):
        validate_artifact_compatibility(forecaster, bad_class_meta)

    # Missing/reordered feature names
    bad_features_meta = dict(valid_meta, feature_names=list(reversed(list(FEATURE_NAMES))))
    with pytest.raises(ArtifactValidationError, match="Feature names mismatch"):
        validate_artifact_compatibility(forecaster, bad_features_meta)

    # Mismatched hyperparameter
    bad_hypers = FrozenModelConfig().to_dict()
    bad_hypers["learning_rate"] = 0.1
    bad_hypers_meta = dict(valid_meta, hyperparameters=bad_hypers)
    with pytest.raises(ArtifactValidationError, match="Hyperparameter mismatch"):
        validate_artifact_compatibility(forecaster, bad_hypers_meta)


def test_operational_warmup_and_first_prediction(mock_engine: OperationalEngine) -> None:
    """Verify exactly 95 observations produce zero predictions, and the 96th produces the first prediction."""
    start_dt = datetime(2018, 1, 1, 0, 15)

    # First 95 records: no prediction
    for i in range(95):
        dt = start_dt + timedelta(minutes=15 * i)
        res = mock_engine.process_telemetry(_make_record(dt, usage=float(i), row_idx=i))
        assert res.prediction is None
        assert res.is_warmup is True
        assert res.feedback is None

    # 96th record (origin: 2018-01-02 00:00:00, index 95)
    dt_96 = start_dt + timedelta(minutes=15 * 95)  # 2018-01-02 00:00:00
    res_96 = mock_engine.process_telemetry(_make_record(dt_96, usage=95.0, row_idx=95))

    assert res_96.prediction is not None
    assert res_96.is_warmup is False
    assert res_96.prediction.origin_timestamp == datetime(2018, 1, 2, 0, 0)
    assert res_96.prediction.target_timestamp == datetime(2018, 1, 2, 0, 15)
    assert res_96.prediction.baseline_persistence_kwh == 95.0
    assert res_96.prediction.model_version == "v1"


def test_one_interval_delayed_feedback_lifecycle(mock_engine: OperationalEngine) -> None:
    """Test the complete one-interval operational feedback cycle:

    telemetry(t) -> prediction(t+1)
    telemetry(t+1) -> feedback(prediction(t), actual(t+1)) -> prediction(t+2)
    """
    start_dt = datetime(2018, 1, 1, 0, 15)

    # Warm-up 96 records
    for i in range(96):
        dt = start_dt + timedelta(minutes=15 * i)
        res = mock_engine.process_telemetry(_make_record(dt, usage=float(i + 1), row_idx=i))

    # Observation 95 (2018-01-02 00:00) generated prediction for target 2018-01-02 00:15
    pred_for_97 = res.prediction
    assert pred_for_97 is not None
    assert pred_for_97.target_timestamp == datetime(2018, 1, 2, 0, 15)

    # Observation 96 (2018-01-02 00:15) arrives with actual usage 20.0
    dt_97 = datetime(2018, 1, 2, 0, 15)
    res_97 = mock_engine.process_telemetry(_make_record(dt_97, usage=20.0, row_idx=96))

    # Delayed feedback for 2018-01-02 00:15 must be resolved immediately!
    fb = res_97.feedback
    assert fb is not None
    assert fb.target_timestamp == dt_97
    assert fb.origin_timestamp == datetime(2018, 1, 2, 0, 0)
    assert fb.actual_usage_kwh == 20.0
    assert fb.predicted_usage_kwh == pred_for_97.predicted_usage_kwh
    assert fb.baseline_persistence_kwh == pred_for_97.baseline_persistence_kwh
    assert fb.ml_absolute_error == pytest.approx(abs(20.0 - pred_for_97.predicted_usage_kwh))
    assert fb.persistence_absolute_error == pytest.approx(abs(20.0 - pred_for_97.baseline_persistence_kwh))
    assert fb.model_version == "v1"

    # And a new prediction for 2018-01-02 00:30 is generated
    assert res_97.prediction is not None
    assert res_97.prediction.target_timestamp == datetime(2018, 1, 2, 0, 30)


def test_feedback_tracker_duplicate_actual_rejected() -> None:
    """Confirm delivering a duplicate actual for an already resolved target raises FeedbackError."""
    tracker = DelayedFeedbackTracker()
    origin = datetime(2018, 1, 1, 12, 0)
    target = datetime(2018, 1, 1, 12, 15)

    pred = PredictionEvent(
        model_version="v1",
        origin_timestamp=origin,
        target_timestamp=target,
        prediction_timestamp=datetime.now(timezone.utc),
        predicted_usage_kwh=10.0,
        baseline_persistence_kwh=8.0,
    )
    tracker.register_prediction(pred)

    rec = _make_record(target, usage=12.0)
    fb, _ = tracker.resolve_feedback(rec)
    assert fb is not None
    assert tracker.completed_count == 1

    # Second actual arrives for the exact same target timestamp
    with pytest.raises(FeedbackError, match="Duplicate feedback"):
        tracker.resolve_feedback(rec)
    assert tracker.duplicate_feedback_count == 1


def test_feedback_tracker_pending_collision_rejected() -> None:
    """Confirm registering two predictions for the same target timestamp raises FeedbackError."""
    tracker = DelayedFeedbackTracker()
    origin = datetime(2018, 1, 1, 12, 0)
    target = datetime(2018, 1, 1, 12, 15)

    pred1 = PredictionEvent(
        model_version="v1",
        origin_timestamp=origin,
        target_timestamp=target,
        prediction_timestamp=datetime.now(timezone.utc),
        predicted_usage_kwh=10.0,
        baseline_persistence_kwh=8.0,
    )
    tracker.register_prediction(pred1)

    pred2 = PredictionEvent(
        model_version="v1",
        origin_timestamp=origin,
        target_timestamp=target,
        prediction_timestamp=datetime.now(timezone.utc),
        predicted_usage_kwh=15.0,
        baseline_persistence_kwh=8.0,
    )
    with pytest.raises(FeedbackError, match="Pending prediction collision"):
        tracker.register_prediction(pred2)


def test_sequence_failure_stops_pipeline(mock_engine: OperationalEngine) -> None:
    """Confirm duplicate, out-of-order, or disabled gap recovery raises OperationalContinuityError and preserves state."""
    start_dt = datetime(2018, 1, 1, 0, 15)

    # Ingest 10 valid records
    for i in range(10):
        dt = start_dt + timedelta(minutes=15 * i)
        mock_engine.process_telemetry(_make_record(dt, usage=float(i), row_idx=i))

    assert mock_engine.records_consumed == 10
    last_valid_ts = mock_engine.last_logical_timestamp

    # 1. Out-of-order timestamp fails closed
    ooo_dt = last_valid_ts - timedelta(minutes=15)  # type: ignore[operator]
    with pytest.raises(OperationalContinuityError, match="out-of-order"):
        mock_engine.process_telemetry(_make_record(ooo_dt, usage=99.0, row_idx=10))
    assert mock_engine.records_consumed == 10
    assert mock_engine.last_logical_timestamp == last_valid_ts
    assert mock_engine.sequence_failure_count == 1

    # 2. Duplicate timestamp fails closed
    with pytest.raises(OperationalContinuityError, match="duplicate"):
        mock_engine.process_telemetry(_make_record(last_valid_ts, usage=99.0, row_idx=11))  # type: ignore[arg-type]
    assert mock_engine.records_consumed == 10
    assert mock_engine.last_logical_timestamp == last_valid_ts
    assert mock_engine.sequence_failure_count == 2

    # 3. Positive gap with enable_gap_recovery=False fails closed
    mock_engine.enable_gap_recovery = False
    gap_dt = last_valid_ts + timedelta(minutes=30)  # type: ignore[operator]
    with pytest.raises(OperationalContinuityError, match="continuity violation"):
        mock_engine.process_telemetry(_make_record(gap_dt, usage=99.0, row_idx=12))
    assert mock_engine.records_consumed == 10
    assert mock_engine.last_logical_timestamp == last_valid_ts
    assert mock_engine.sequence_failure_count == 3


def test_gap_detection_single_and_multi_interval(mock_engine: OperationalEngine) -> None:
    """Verify exact gap detection, duration, and missing timestamp calculation."""
    start_dt = datetime(2018, 1, 1, 10, 0)
    mock_engine.process_telemetry(_make_record(start_dt, usage=10.0, row_idx=0))

    # Single missing interval: skip 10:15, receive 10:30
    res = mock_engine.process_telemetry(_make_record(datetime(2018, 1, 1, 10, 30), usage=12.0, row_idx=1))
    assert res.gap_incident is not None
    gap = res.gap_incident
    assert gap.previous_timestamp == datetime(2018, 1, 1, 10, 0)
    assert gap.current_timestamp == datetime(2018, 1, 1, 10, 30)
    assert gap.gap_duration == timedelta(minutes=30)
    assert gap.missing_interval_count == 1
    assert gap.first_missing_timestamp == datetime(2018, 1, 1, 10, 15)
    assert gap.last_missing_timestamp == datetime(2018, 1, 1, 10, 15)
    assert gap.missing_timestamps == (datetime(2018, 1, 1, 10, 15),)
    assert gap.state_reset is True
    assert gap.rewarm_required is True

    # Multi-interval gap: skip 10:45, 11:00, receive 11:15
    res2 = mock_engine.process_telemetry(_make_record(datetime(2018, 1, 1, 11, 15), usage=15.0, row_idx=2))
    assert res2.gap_incident is not None
    gap2 = res2.gap_incident
    assert gap2.previous_timestamp == datetime(2018, 1, 1, 10, 30)
    assert gap2.current_timestamp == datetime(2018, 1, 1, 11, 15)
    assert gap2.gap_duration == timedelta(minutes=45)
    assert gap2.missing_interval_count == 2
    assert gap2.first_missing_timestamp == datetime(2018, 1, 1, 10, 45)
    assert gap2.last_missing_timestamp == datetime(2018, 1, 1, 11, 0)
    assert gap2.missing_timestamps == (datetime(2018, 1, 1, 10, 45), datetime(2018, 1, 1, 11, 0))


def test_gap_state_reset_and_rewarm_isolation(mock_engine: OperationalEngine) -> None:
    """Verify state reset purges pre-gap history and requires exactly 96 post-gap observations."""
    start_dt = datetime(2018, 1, 1, 0, 15)

    # Ingest 96 records with usage=100.0 (completing warm-up)
    for i in range(96):
        dt = start_dt + timedelta(minutes=15 * i)
        res = mock_engine.process_telemetry(_make_record(dt, usage=100.0, row_idx=i))

    assert res.prediction is not None
    assert mock_engine.feedback_tracker.pending_count == 1
    assert len(mock_engine.feature_generator.usage_buffer) == 96

    # Inject positive gap: skip 1 interval, send usage=10.0
    gap_dt = dt + timedelta(minutes=30)
    gap_res = mock_engine.process_telemetry(_make_record(gap_dt, usage=10.0, row_idx=96))

    # Gap incident occurred
    assert gap_res.gap_incident is not None
    assert mock_engine.gap_incident_count == 1
    assert len(gap_res.invalidated_predictions) == 1

    # Stateful feature generator was reset: buffer now contains ONLY the single new record
    assert len(mock_engine.feature_generator.usage_buffer) == 1
    assert list(mock_engine.feature_generator.usage_buffer) == [10.0]
    assert gap_res.is_warmup is True
    assert gap_res.prediction is None

    # Pending prediction for the missed target was quarantined, NOT kept pending
    assert mock_engine.feedback_tracker.pending_count == 0
    assert mock_engine.feedback_tracker.invalidated_count == 1
    inv = gap_res.invalidated_predictions[0]
    assert inv.reason == "missing_actual_due_to_gap"
    assert inv.target_timestamp == dt + timedelta(minutes=15)

    # Feed 94 more records with usage=10.0 (reaching 95 post-gap records total)
    for i in range(1, 95):
        step_dt = gap_dt + timedelta(minutes=15 * i)
        step_res = mock_engine.process_telemetry(_make_record(step_dt, usage=10.0, row_idx=96 + i))
        assert step_res.is_warmup is True
        assert step_res.prediction is None

    # Post-gap observation 96: warm-up completes, prediction resumes!
    obs_96_dt = gap_dt + timedelta(minutes=15 * 95)
    resume_res = mock_engine.process_telemetry(_make_record(obs_96_dt, usage=10.0, row_idx=96 + 95))
    assert resume_res.is_warmup is False
    assert resume_res.prediction is not None
    assert resume_res.prediction.origin_timestamp == obs_96_dt
    assert resume_res.prediction.target_timestamp == obs_96_dt + timedelta(minutes=15)

    # Confirm roll mean 96 is exactly 10.0, proving ZERO contamination from pre-gap 100.0 values!
    assert resume_res.prediction.features["usage_roll_mean_96"] == 10.0


def test_monitoring_isolation_across_gaps(mock_engine: OperationalEngine) -> None:
    """Verify invalidated predictions do not affect MAE/RMSE or mutate feedback metrics."""
    from forgecast.monitoring.metrics import BaselineComparisonTracker

    tracker = BaselineComparisonTracker()
    mock_engine.monitoring_tracker = tracker

    start_dt = datetime(2018, 1, 1, 0, 15)
    # Warm up 96 records
    for i in range(96):
        dt = start_dt + timedelta(minutes=15 * i)
        mock_engine.process_telemetry(_make_record(dt, usage=10.0, row_idx=i))

    # Record 97 arrives normally -> resolves feedback for 96's prediction
    dt_97 = start_dt + timedelta(minutes=15 * 96)
    res_97 = mock_engine.process_telemetry(_make_record(dt_97, usage=12.0, row_idx=96))
    assert res_97.feedback is not None
    snap_before = tracker.get_full_replay_snapshot("v1")
    assert snap_before.sample_count == 1
    pre_gap_mae = snap_before.ml_mae

    # Gap skips interval 98, receives 99
    dt_99 = dt_97 + timedelta(minutes=30)
    res_99 = mock_engine.process_telemetry(_make_record(dt_99, usage=15.0, row_idx=97))
    assert res_99.gap_incident is not None
    assert len(res_99.invalidated_predictions) == 1

    # Monitoring tracker must remain untouched: sample count still 1, MAE unchanged
    snap_after = tracker.get_full_replay_snapshot("v1")
    assert snap_after.sample_count == 1
    assert snap_after.ml_mae == pre_gap_mae
    assert mock_engine.feedback_tracker.completed_count == 1
    assert mock_engine.feedback_tracker.invalidated_count == 1


def test_midnight_gap_recovery(mock_engine: OperationalEngine) -> None:
    """Verify gap recovery when the missing intervals straddle the midnight boundary."""
    # Feed records up to 23:45
    start_dt = datetime(2018, 1, 1, 23, 0)
    for i in range(4):
        dt = start_dt + timedelta(minutes=15 * i)
        mock_engine.process_telemetry(_make_record(dt, usage=5.0, row_idx=i))

    # Last accepted: 2018-01-01 23:45.
    # Positive gap: missing 2018-01-02 00:00 (midnight) and 00:15, arrives at 00:30.
    res = mock_engine.process_telemetry(_make_record(datetime(2018, 1, 2, 0, 30), usage=6.0, row_idx=4))
    assert res.gap_incident is not None
    gap = res.gap_incident
    assert gap.previous_timestamp == datetime(2018, 1, 1, 23, 45)
    assert gap.current_timestamp == datetime(2018, 1, 2, 0, 30)
    assert gap.gap_duration == timedelta(minutes=45)
    assert gap.missing_interval_count == 2
    assert gap.first_missing_timestamp == datetime(2018, 1, 2, 0, 0)
    assert gap.last_missing_timestamp == datetime(2018, 1, 2, 0, 15)
    assert gap.missing_timestamps == (datetime(2018, 1, 2, 0, 0), datetime(2018, 1, 2, 0, 15))
    assert mock_engine.records_consumed == 5


def test_midnight_lifecycle_continuity(mock_engine: OperationalEngine) -> None:
    """Verify operational prediction and delayed feedback pairing across the midnight boundary."""
    start_dt = datetime(2018, 1, 1, 0, 15)

    # Fast forward through 95 records (up to 2018-01-01 23:45)
    for i in range(95):
        dt = start_dt + timedelta(minutes=15 * i)
        mock_engine.process_telemetry(_make_record(dt, usage=float(i + 1), row_idx=i))

    # Observation 95 (index 94): 2018-01-01 23:45. Still in warm-up (95 records)
    # Observation 96 (index 95): 2018-01-02 00:00 (the midnight closing interval)
    res_0000 = mock_engine.process_telemetry(
        _make_record(datetime(2018, 1, 2, 0, 0), usage=3.42, row_idx=95)
    )
    assert res_0000.prediction is not None
    assert res_0000.prediction.origin_timestamp == datetime(2018, 1, 2, 0, 0)
    assert res_0000.prediction.target_timestamp == datetime(2018, 1, 2, 0, 15)

    # Observation 97 (index 96): 2018-01-02 00:15
    res_0015 = mock_engine.process_telemetry(
        _make_record(datetime(2018, 1, 2, 0, 15), usage=3.20, row_idx=96)
    )
    # Delayed feedback for 2018-01-02 00:15 must pair perfectly with the midnight prediction
    assert res_0015.feedback is not None
    assert res_0015.feedback.origin_timestamp == datetime(2018, 1, 2, 0, 0)
    assert res_0015.feedback.target_timestamp == datetime(2018, 1, 2, 0, 15)
    assert res_0015.feedback.actual_usage_kwh == 3.20

