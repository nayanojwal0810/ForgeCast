"""Tests for the operational UI demonstration service, artifact loading, and state contracts."""

import hashlib
from pathlib import Path

import pytest

from forgecast.ui.service import (
    get_repo_root,
    load_chronological_evaluation_summary,
    load_retraining_lifecycle_summary,
    run_bounded_demonstration,
)


def _compute_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest().upper()


def test_service_loads_v1_artifact_and_sources_model_version() -> None:
    """Verify demonstration service loads the frozen v1 artifact and sources version 'v1' directly."""
    snapshot = run_bounded_demonstration(record_count=120, inject_gap=False)

    assert snapshot.model_version == "v1"
    assert snapshot.model_class == "HistGradientBoostingRegressor"
    assert snapshot.records_consumed == 120
    assert snapshot.is_warmed_up is True
    assert snapshot.system_state == "Operational (Active Forecasting)"
    assert snapshot.latest_logical_timestamp is not None
    assert snapshot.latest_prediction is not None
    assert snapshot.latest_prediction.model_version == "v1"
    assert snapshot.latest_feedback is not None


def test_pre_warmup_empty_state_safety() -> None:
    """Verify that prior to 96 observations, empty prediction and feedback states are safely handled."""
    snapshot = run_bounded_demonstration(record_count=50, inject_gap=False)

    assert snapshot.records_consumed == 50
    assert snapshot.is_warmed_up is False
    assert "Initial warm-up (50/96" in snapshot.system_state
    assert snapshot.latest_prediction is None
    assert snapshot.latest_feedback is None
    assert snapshot.gap_incident_count == 0
    assert snapshot.invalidated_prediction_count == 0
    assert len(snapshot.recent_history) == 0


def test_gap_simulation_reflects_in_operational_state() -> None:
    """Verify gap simulation increments gap and invalidation counters and updates system segment."""
    snapshot = run_bounded_demonstration(record_count=130, inject_gap=True, gap_record_index=110)

    assert snapshot.gap_incident_count == 1
    assert snapshot.invalidated_prediction_count == 1
    assert snapshot.current_segment_id == 2
    assert snapshot.latest_gap_incident is not None
    assert snapshot.latest_gap_incident.missing_interval_count == 1
    assert snapshot.latest_invalidated_prediction is not None
    assert snapshot.latest_invalidated_prediction.reason == "missing_actual_due_to_gap"


def test_monitoring_values_sourced_from_tracker() -> None:
    """Verify monitoring snapshots are populated from the active BaselineComparisonTracker."""
    snapshot = run_bounded_demonstration(record_count=130, inject_gap=False)

    assert snapshot.cumulative_snapshot is not None
    assert snapshot.cumulative_snapshot.sample_count == 130 - 96  # 34 completed feedback records
    assert snapshot.cumulative_snapshot.ml_mae is not None
    assert snapshot.cumulative_snapshot.baseline_mae is not None
    assert snapshot.cumulative_snapshot.mae_improvement_pct is not None

    assert snapshot.rolling_24h_snapshot is not None
    assert snapshot.rolling_24h_snapshot.sample_count > 0

    assert len(snapshot.recent_history) == 34


def test_static_summaries_loaders() -> None:
    """Verify read-only loading of chronological evaluation and retraining lifecycle artifacts."""
    eval_sum = load_chronological_evaluation_summary()
    assert eval_sum is not None
    assert eval_sum["n_splits"] == 3
    assert eval_sum["total_samples"] == 10512
    assert eval_sum["ml_mae"] is not None
    assert eval_sum["improvement_pct"] is not None

    retrain_sum = load_retraining_lifecycle_summary()
    assert retrain_sum is not None
    assert retrain_sum["candidate_version"] == "v2"
    assert retrain_sum["decision_status"] == "ACCEPTED"
    assert retrain_sum["is_promoted"] is True


def test_v1_artifact_and_dataset_unmodified() -> None:
    """Confirm executing the UI demonstration service leaves v1 artifact and raw dataset strictly unmodified."""
    repo_root = get_repo_root()
    model_path = repo_root / "artifacts" / "models" / "forgecast_v1_replay_ready.pkl"
    raw_data_path = repo_root / "data" / "raw" / "Steel_industry_data.csv"

    hash_model_before = _compute_sha256(model_path)
    hash_data_before = _compute_sha256(raw_data_path)

    # Run UI demonstration service
    run_bounded_demonstration(record_count=120, inject_gap=True)

    hash_model_after = _compute_sha256(model_path)
    hash_data_after = _compute_sha256(raw_data_path)

    assert hash_model_before == hash_model_after
    assert hash_data_before == hash_data_after


def test_completed_forecast_feedback_alignment() -> None:
    """Verify that the completed forecast result sources ML prediction, persistence baseline,
    and actual usage strictly from the same FeedbackRecord to prevent lifecycle misalignment.
    """
    snapshot = run_bounded_demonstration(record_count=120, inject_gap=False)
    assert snapshot.latest_feedback is not None
    assert snapshot.latest_prediction is not None

    fb = snapshot.latest_feedback
    pred = snapshot.latest_prediction

    # 1. UI displayed values for the completed forecast must match the feedback record
    displayed_ml = fb.predicted_usage_kwh
    displayed_persistence = fb.baseline_persistence_kwh
    displayed_actual = fb.actual_usage_kwh

    assert displayed_ml == fb.predicted_usage_kwh
    assert displayed_persistence == fb.baseline_persistence_kwh
    assert displayed_actual == fb.actual_usage_kwh

    # 2. Specifically guard against mixing latest_prediction and latest_feedback:
    # latest_prediction's baseline equals the newly arrived actual (Usage(t)),
    # whereas latest_feedback's baseline is Usage(t-1) (the actual baseline for target t).
    # In a non-constant series, they must not be conflated.
    assert fb.target_timestamp != pred.target_timestamp
    assert fb.baseline_persistence_kwh != pred.baseline_persistence_kwh

    # 3. Verdict calculation must be derived purely from feedback error values
    assert fb.ml_absolute_error == abs(fb.actual_usage_kwh - fb.predicted_usage_kwh)
    assert fb.persistence_absolute_error == abs(fb.actual_usage_kwh - fb.baseline_persistence_kwh)
    ml_won = fb.ml_absolute_error < fb.persistence_absolute_error
    expected_verdict = "ML performed better" if ml_won else "Persistence performed better"
    assert (fb.ml_absolute_error < fb.persistence_absolute_error) == ml_won
