"""Backend demonstration service powering the operational UI without duplicating forecasting logic."""

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from forgecast.monitoring.metrics import BaselineComparisonTracker, MetricSnapshot
from forgecast.operational.engine import OperationalEngine
from forgecast.operational.events import (
    FeedbackRecord,
    GapIncident,
    InvalidatedPrediction,
    PredictionEvent,
)
from forgecast.replay.engine import TelemetryReplay


def get_repo_root() -> Path:
    """Resolve repository root directory robustly across local and cloud environments."""
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class UIDataSnapshot:
    """Consolidated operational snapshot consumed directly by the demonstration UI."""

    model_version: str
    model_class: str
    records_consumed: int
    is_warmed_up: bool
    system_state: str
    latest_logical_timestamp: datetime | None
    latest_prediction: PredictionEvent | None
    latest_feedback: FeedbackRecord | None
    current_segment_id: int
    gap_incident_count: int
    invalidated_prediction_count: int
    sequence_failure_count: int
    latest_gap_incident: GapIncident | None
    latest_invalidated_prediction: InvalidatedPrediction | None
    rolling_24h_snapshot: MetricSnapshot | None
    rolling_7d_snapshot: MetricSnapshot | None
    cumulative_snapshot: MetricSnapshot | None
    recent_history: list[dict[str, Any]] = field(default_factory=list)


def load_chronological_evaluation_summary(
    artifacts_dir: Path | str | None = None,
) -> dict[str, Any] | None:
    """Read historical 3-fold chronological evaluation evidence from disk."""
    base = Path(artifacts_dir) if artifacts_dir else get_repo_root() / "artifacts"
    eval_file = base / "experiments" / "chronological_evaluation_v1.json"
    if not eval_file.exists():
        return None
    try:
        with open(eval_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        pooled = data.get("pooled_results", {})
        return {
            "n_splits": data.get("n_splits", 3),
            "total_samples": pooled.get("pooled_evaluation_sample_count", data.get("total_samples", 0)),
            "ml_mae": pooled.get("pooled_ml_mae"),
            "ml_rmse": pooled.get("pooled_ml_rmse"),
            "persistence_mae": pooled.get("pooled_persistence_mae"),
            "improvement_pct": pooled.get("pooled_ml_relative_improvement_pct"),
        }
    except Exception:
        return None


def load_retraining_lifecycle_summary(
    artifacts_dir: Path | str | None = None,
) -> dict[str, Any] | None:
    """Read latest candidate retraining evidence from disk (read-only portfolio inspection)."""
    base = Path(artifacts_dir) if artifacts_dir else get_repo_root() / "artifacts"
    exp_file = base / "experiments" / "retraining_v2.json"
    if not exp_file.exists():
        return None
    try:
        with open(exp_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        cp = data.get("checkpoint", {})
        dec = data.get("decision", {})
        cand_m = dec.get("candidate_metrics", {})
        base_m = dec.get("baseline_metrics", {})
        return {
            "checkpoint_name": cp.get("name", "Q3"),
            "candidate_version": cp.get("candidate_version", "v2"),
            "training_cutoff": cp.get("training_cutoff"),
            "eval_start": cp.get("eval_start"),
            "eval_end": cp.get("eval_end"),
            "decision_status": dec.get("status", "UNKNOWN"),
            "is_promoted": dec.get("is_promoted", False),
            "candidate_mae": cand_m.get("mae"),
            "persistence_mae": base_m.get("mae"),
            "improvement_pct": dec.get("baseline_improvement_pct"),
            "created_at_utc": data.get("created_at_utc"),
        }
    except Exception:
        return None


def run_bounded_demonstration(
    record_count: int = 150,
    inject_gap: bool = False,
    gap_record_index: int = 110,
    csv_path: Path | str | None = None,
    model_path: Path | str | None = None,
) -> UIDataSnapshot:
    """Run a bounded telemetry replay through the operational engine and capture UI state.

    Args:
        record_count: Number of telemetry records to process.
        inject_gap: If True, simulates missingness by dropping a single record in memory.
        gap_record_index: Index of record to drop when inject_gap is True.
        csv_path: Source CSV dataset path (read-only stream).
        model_path: Trained model artifact path.

    Returns:
        UIDataSnapshot populated with live engine and monitoring state.
    """
    repo_root = get_repo_root()
    c_path = Path(csv_path) if csv_path else repo_root / "data" / "raw" / "Steel_industry_data.csv"
    m_path = (
        Path(model_path)
        if model_path
        else repo_root / "artifacts" / "models" / "forgecast_v1_replay_ready.pkl"
    )

    tracker = BaselineComparisonTracker(window_sizes=(96, 672))
    engine = OperationalEngine(
        model_path=m_path,
        monitoring_tracker=tracker,
        enable_gap_recovery=True,
    )

    replay = TelemetryReplay(csv_path=c_path, delay_seconds=0.0)

    latest_prediction: PredictionEvent | None = None
    latest_feedback: FeedbackRecord | None = None
    latest_gap_incident: GapIncident | None = None
    latest_invalidated: InvalidatedPrediction | None = None
    recent_history: list[dict[str, Any]] = []

    for idx, record in enumerate(replay.run()):
        if idx >= record_count:
            break

        if inject_gap and idx == gap_record_index:
            # Simulate a 1-interval missing telemetry packet in memory
            continue

        step_res = engine.process_telemetry(record)

        if step_res.prediction is not None:
            latest_prediction = step_res.prediction

        if step_res.feedback is not None:
            latest_feedback = step_res.feedback
            recent_history.append(
                {
                    "target_timestamp": step_res.feedback.target_timestamp.isoformat(),
                    "actual_usage_kwh": step_res.feedback.actual_usage_kwh,
                    "predicted_usage_kwh": step_res.feedback.predicted_usage_kwh,
                    "baseline_persistence_kwh": step_res.feedback.baseline_persistence_kwh,
                    "ml_absolute_error": step_res.feedback.ml_absolute_error,
                    "persistence_absolute_error": step_res.feedback.persistence_absolute_error,
                }
            )

        if step_res.gap_incident is not None:
            latest_gap_incident = step_res.gap_incident

        if step_res.invalidated_predictions:
            latest_invalidated = step_res.invalidated_predictions[-1]

    # Evaluate system state descriptor
    buffer_len = len(engine.feature_generator.usage_buffer)
    is_warmed_up = buffer_len >= engine.feature_generator.history_size

    if buffer_len < engine.feature_generator.history_size:
        if engine.gap_incident_count > 0:
            system_state = f"Re-warming post-gap ({buffer_len}/96 observations)"
        else:
            system_state = f"Initial warm-up ({buffer_len}/96 observations)"
    else:
        system_state = "Operational (Active Forecasting)"

    summary = tracker.get_comparison_summary(engine.model_version)

    return UIDataSnapshot(
        model_version=engine.model_version,
        model_class=engine.metadata.get("model_class", "HistGradientBoostingRegressor"),
        records_consumed=engine.records_consumed,
        is_warmed_up=is_warmed_up,
        system_state=system_state,
        latest_logical_timestamp=engine.last_logical_timestamp,
        latest_prediction=latest_prediction,
        latest_feedback=latest_feedback,
        current_segment_id=engine.current_segment_id,
        gap_incident_count=engine.gap_incident_count,
        invalidated_prediction_count=engine.invalidated_prediction_count,
        sequence_failure_count=engine.sequence_failure_count,
        latest_gap_incident=latest_gap_incident,
        latest_invalidated_prediction=latest_invalidated,
        rolling_24h_snapshot=summary.rolling_24h,
        rolling_7d_snapshot=summary.rolling_7d,
        cumulative_snapshot=summary.full_replay_cumulative,
        recent_history=recent_history[-96:],  # Keep last 96 completed steps for chart
    )
