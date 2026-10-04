from collections.abc import Sequence
from datetime import datetime

from forgecast.ingestion.contract import TelemetryRecord
from forgecast.operational.events import (
    FeedbackError,
    FeedbackRecord,
    InvalidatedPrediction,
    PredictionEvent,
)


class DelayedFeedbackTracker:
    """Tracks pending predictions and pairs them with delayed ground-truth actual telemetry."""

    def __init__(self) -> None:
        self.pending_predictions: dict[datetime, PredictionEvent] = {}
        self.completed_feedback: dict[datetime, FeedbackRecord] = {}
        self.invalidated_predictions: list[InvalidatedPrediction] = []
        self.unmatched_feedback_timestamps: list[datetime] = []
        self.duplicate_feedback_count: int = 0

    def register_prediction(self, prediction: PredictionEvent) -> None:
        """Register a pending prediction keyed strictly by its logical target timestamp.

        Raises:
            FeedbackError: If a prediction for the same target timestamp is already pending or completed.
        """
        target_ts = prediction.target_timestamp

        if target_ts in self.pending_predictions:
            raise FeedbackError(
                f"Pending prediction collision: a prediction for target timestamp '{target_ts.isoformat()}' "
                "is already pending"
            )

        if target_ts in self.completed_feedback:
            raise FeedbackError(
                f"Prediction registration rejected: target timestamp '{target_ts.isoformat()}' "
                "has already completed feedback"
            )

        self.pending_predictions[target_ts] = prediction

    def resolve_feedback(
        self, telemetry: TelemetryRecord, is_warmup: bool = False
    ) -> tuple[FeedbackRecord | None, bool]:
        """Pair newly arrived actual telemetry with any matching pending prediction.

        Args:
            telemetry: Telemetry record for the closed interval.
            is_warmup: Whether system is in pre-prediction warm-up phase.

        Returns:
            Tuple of (FeedbackRecord if paired or None, was_unmatched_boolean).

        Raises:
            FeedbackError: If actual telemetry arrives for an already completed target timestamp.
        """
        target_ts = telemetry.logical_timestamp

        if target_ts in self.completed_feedback:
            self.duplicate_feedback_count += 1
            raise FeedbackError(
                f"Duplicate feedback error: actual observation for target timestamp '{target_ts.isoformat()}' "
                "was already paired and resolved"
            )

        if target_ts in self.pending_predictions:
            pred = self.pending_predictions.pop(target_ts)
            ml_err = abs(telemetry.usage_kwh - pred.predicted_usage_kwh)
            persist_err = abs(telemetry.usage_kwh - pred.baseline_persistence_kwh)

            record = FeedbackRecord(
                origin_timestamp=pred.origin_timestamp,
                target_timestamp=pred.target_timestamp,
                actual_usage_kwh=telemetry.usage_kwh,
                predicted_usage_kwh=pred.predicted_usage_kwh,
                baseline_persistence_kwh=pred.baseline_persistence_kwh,
                ml_absolute_error=ml_err,
                persistence_absolute_error=persist_err,
                model_version=pred.model_version,
                origin_index=pred.origin_index,
                target_index=telemetry.row_index,
            )
            self.completed_feedback[target_ts] = record
            return record, False

        # No pending prediction exists for this timestamp
        if not is_warmup:
            self.unmatched_feedback_timestamps.append(target_ts)
            return None, True

        return None, False

    def quarantine_pending_predictions(
        self,
        missing_timestamps: Sequence[datetime],
        previous_ts: datetime,
        current_ts: datetime,
        reason: str = "missing_actual_due_to_gap",
    ) -> list[InvalidatedPrediction]:
        """Quarantine and remove pending predictions whose actual observation was missed in a gap."""
        invalidated: list[InvalidatedPrediction] = []
        for ts in missing_timestamps:
            if ts in self.pending_predictions:
                pred = self.pending_predictions.pop(ts)
                inv = InvalidatedPrediction(
                    model_version=pred.model_version,
                    origin_timestamp=pred.origin_timestamp,
                    target_timestamp=pred.target_timestamp,
                    predicted_usage_kwh=pred.predicted_usage_kwh,
                    baseline_persistence_kwh=pred.baseline_persistence_kwh,
                    reason=reason,
                    gap_previous_timestamp=previous_ts,
                    gap_current_timestamp=current_ts,
                    origin_index=pred.origin_index,
                )
                invalidated.append(inv)
                self.invalidated_predictions.append(inv)
        return invalidated

    def reset(self) -> None:
        """Reset tracker state."""
        self.pending_predictions.clear()
        self.completed_feedback.clear()
        self.invalidated_predictions.clear()
        self.unmatched_feedback_timestamps.clear()
        self.duplicate_feedback_count = 0

    @property
    def pending_count(self) -> int:
        return len(self.pending_predictions)

    @property
    def completed_count(self) -> int:
        return len(self.completed_feedback)

    @property
    def invalidated_count(self) -> int:
        return len(self.invalidated_predictions)

    @property
    def unmatched_count(self) -> int:
        return len(self.unmatched_feedback_timestamps)
