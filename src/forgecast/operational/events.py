"""Operational prediction events, feedback records, and lifecycle exceptions."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from forgecast.ingestion.contract import TelemetryRecord


class ArtifactValidationError(ValueError):
    """Raised when loaded model artifact or metadata violates the operational contract."""


class OperationalContinuityError(ValueError):
    """Raised when incoming telemetry breaks temporal sequence continuity."""


class FeedbackError(ValueError):
    """Raised when delayed feedback encounters a duplicate, collision, or invalid state."""


@dataclass(frozen=True)
class PredictionEvent:
    """One-step-ahead prediction event carrying temporal metadata, predictions, and baselines."""

    model_version: str
    origin_timestamp: datetime
    target_timestamp: datetime
    prediction_timestamp: datetime
    predicted_usage_kwh: float
    baseline_persistence_kwh: float
    origin_index: int | None = None
    features: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (self.origin_timestamp < self.target_timestamp):
            raise ValueError(
                f"Temporal violation: origin ({self.origin_timestamp}) must be strictly earlier "
                f"than target ({self.target_timestamp})"
            )
        expected_target = self.origin_timestamp + timedelta(minutes=15)
        if self.target_timestamp != expected_target:
            raise ValueError(
                f"Temporal cadence violation: expected target {expected_target}, got {self.target_timestamp}"
            )


@dataclass(frozen=True)
class FeedbackRecord:
    """Completed feedback record pairing a previous prediction with observed ground truth actuals."""

    origin_timestamp: datetime
    target_timestamp: datetime
    actual_usage_kwh: float
    predicted_usage_kwh: float
    baseline_persistence_kwh: float
    ml_absolute_error: float
    persistence_absolute_error: float
    model_version: str
    origin_index: int | None = None
    target_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert feedback record to flat dictionary."""
        return {
            "origin_timestamp": self.origin_timestamp.isoformat(),
            "target_timestamp": self.target_timestamp.isoformat(),
            "actual_usage_kwh": self.actual_usage_kwh,
            "predicted_usage_kwh": self.predicted_usage_kwh,
            "baseline_persistence_kwh": self.baseline_persistence_kwh,
            "ml_absolute_error": self.ml_absolute_error,
            "persistence_absolute_error": self.persistence_absolute_error,
            "model_version": self.model_version,
            "origin_index": self.origin_index,
            "target_index": self.target_index,
        }


@dataclass(frozen=True)
class InvalidatedPrediction:
    """A prediction that could not be evaluated because actual telemetry was missing in a gap."""

    model_version: str
    origin_timestamp: datetime
    target_timestamp: datetime
    predicted_usage_kwh: float
    baseline_persistence_kwh: float
    reason: str
    gap_previous_timestamp: datetime
    gap_current_timestamp: datetime
    origin_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert invalidated prediction to flat dictionary."""
        return {
            "model_version": self.model_version,
            "origin_timestamp": self.origin_timestamp.isoformat(),
            "target_timestamp": self.target_timestamp.isoformat(),
            "predicted_usage_kwh": self.predicted_usage_kwh,
            "baseline_persistence_kwh": self.baseline_persistence_kwh,
            "reason": self.reason,
            "gap_previous_timestamp": self.gap_previous_timestamp.isoformat(),
            "gap_current_timestamp": self.gap_current_timestamp.isoformat(),
            "origin_index": self.origin_index,
        }


@dataclass(frozen=True)
class GapIncident:
    """Explicit typed representation of a detected temporal missingness gap."""

    previous_timestamp: datetime
    current_timestamp: datetime
    gap_duration: timedelta
    missing_interval_count: int
    first_missing_timestamp: datetime
    last_missing_timestamp: datetime
    missing_timestamps: tuple[datetime, ...]
    state_reset: bool
    rewarm_required: bool
    invalidated_prediction_count: int
    segment_id: int

    def to_dict(self) -> dict[str, Any]:
        """Convert gap incident to flat dictionary."""
        return {
            "previous_timestamp": self.previous_timestamp.isoformat(),
            "current_timestamp": self.current_timestamp.isoformat(),
            "gap_duration_seconds": self.gap_duration.total_seconds(),
            "missing_interval_count": self.missing_interval_count,
            "first_missing_timestamp": self.first_missing_timestamp.isoformat(),
            "last_missing_timestamp": self.last_missing_timestamp.isoformat(),
            "missing_timestamps": [t.isoformat() for t in self.missing_timestamps],
            "state_reset": self.state_reset,
            "rewarm_required": self.rewarm_required,
            "invalidated_prediction_count": self.invalidated_prediction_count,
            "segment_id": self.segment_id,
        }


@dataclass(frozen=True)
class OperationalStepResult:
    """Consolidated outcome of one operational event processing cycle."""

    telemetry_record: TelemetryRecord
    feedback: FeedbackRecord | None = None
    prediction: PredictionEvent | None = None
    is_warmup: bool = False
    unmatched_feedback: bool = False
    gap_incident: GapIncident | None = None
    invalidated_predictions: tuple[InvalidatedPrediction, ...] = ()
