"""Operational prediction engine orchestrating feedback pairing, stateful features, and inference."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from forgecast.features.generator import FEATURE_NAMES, StatefulFeatureGenerator
from forgecast.ingestion.contract import TelemetryRecord
from forgecast.models.config import FrozenModelConfig
from forgecast.models.forecaster import EnergyForecaster
from forgecast.models.persistence import load_model_artifact

if TYPE_CHECKING:
    from forgecast.monitoring.metrics import BaselineComparisonTracker
from forgecast.operational.events import (
    ArtifactValidationError,
    GapIncident,
    InvalidatedPrediction,
    OperationalContinuityError,
    OperationalStepResult,
    PredictionEvent,
)
from forgecast.operational.feedback import DelayedFeedbackTracker
from forgecast.replay.engine import TelemetryReplay


def validate_artifact_compatibility(
    forecaster: EnergyForecaster, metadata: dict[str, Any] | None
) -> None:
    """Validate that the loaded model artifact and metadata adhere to the strict application contract."""
    if not isinstance(forecaster, EnergyForecaster):
        raise ArtifactValidationError(
            f"Expected EnergyForecaster instance, got {type(forecaster).__name__}"
        )

    if not getattr(forecaster, "is_fitted", False):
        raise ArtifactValidationError("Loaded model artifact is not fitted")

    if metadata is None:
        raise ArtifactValidationError("Missing companion metadata for model artifact")

    model_version = metadata.get("model_version")
    if not model_version or not isinstance(model_version, str):
        raise ArtifactValidationError("Model metadata does not specify a valid 'model_version'")

    model_class = metadata.get("model_class")
    if model_class != "HistGradientBoostingRegressor":
        raise ArtifactValidationError(
            f"Model class mismatch: expected 'HistGradientBoostingRegressor', got '{model_class}'"
        )

    meta_features = metadata.get("feature_names", [])
    if list(meta_features) != list(FEATURE_NAMES):
        raise ArtifactValidationError(
            "Feature names mismatch: metadata features do not match canonical FEATURE_NAMES"
        )

    recorded_hypers = metadata.get("hyperparameters", {})
    frozen_hypers = FrozenModelConfig().to_dict()
    for param, expected_val in frozen_hypers.items():
        if recorded_hypers.get(param) != expected_val:
            raise ArtifactValidationError(
                f"Hyperparameter mismatch on '{param}': expected {expected_val}, got {recorded_hypers.get(param)}"
            )


class OperationalEngine:
    """Sequential operational forecasting engine executing the prediction-to-feedback lifecycle."""

    def __init__(
        self,
        model_path: Path | str = "artifacts/models/forgecast_v1_replay_ready.pkl",
        forecaster: EnergyForecaster | None = None,
        metadata: dict[str, Any] | None = None,
        monitoring_tracker: BaselineComparisonTracker | None = None,
        enable_gap_recovery: bool = True,
    ) -> None:
        if forecaster is not None and metadata is not None:
            self.forecaster = forecaster
            self.metadata = metadata
        else:
            self.forecaster, self.metadata = load_model_artifact(model_path)

        validate_artifact_compatibility(self.forecaster, self.metadata)
        self.model_version: str = self.metadata["model_version"]

        self.feature_generator = StatefulFeatureGenerator()
        self.feedback_tracker = DelayedFeedbackTracker()
        self.monitoring_tracker = monitoring_tracker
        self.enable_gap_recovery = enable_gap_recovery
        self.gap_incidents: list[GapIncident] = []
        self.current_segment_id: int = 1

        if self.monitoring_tracker is not None:
            training_end_str = self.metadata.get("training_end_target")
            if training_end_str and self.monitoring_tracker.deployment_boundary is None:
                self.monitoring_tracker.set_deployment_boundary(
                    datetime.fromisoformat(training_end_str)
                )

        self.last_logical_timestamp: datetime | None = None
        self.records_consumed: int = 0
        self.sequence_failure_count: int = 0

    def process_telemetry(self, record: TelemetryRecord) -> OperationalStepResult:
        """Execute one complete operational cycle for an arriving telemetry event.

        Order of operations:
        1. Sequence continuity verification (fail closed on duplicate/out-of-order, recover on positive gap)
        2. Resolve delayed feedback for target timestamp t
        3. Feed completed feedback to monitoring layer (if configured)
        4. Ingest telemetry t into stateful feature buffer
        5. Generate features for t+1 if warm-up permits
        6. Run ML inference for t+1
        7. Register PredictionEvent for t+1
        """
        gap_incident: GapIncident | None = None
        invalidated_predictions: tuple[InvalidatedPrediction, ...] = ()

        # 1. Sequence continuity verification
        if self.last_logical_timestamp is not None:
            prev_ts = self.last_logical_timestamp
            curr_ts = record.logical_timestamp
            expected_ts = prev_ts + timedelta(minutes=15)

            if curr_ts < prev_ts:
                # Case D: Out-of-order timestamp -> fail closed
                self.sequence_failure_count += 1
                raise OperationalContinuityError(
                    f"Operational sequence out-of-order violation: received '{curr_ts.isoformat()}' "
                    f"earlier than previous accepted '{prev_ts.isoformat()}'"
                )

            if curr_ts == prev_ts:
                # Case C: Duplicate timestamp -> fail closed
                self.sequence_failure_count += 1
                raise OperationalContinuityError(
                    f"Operational sequence duplicate violation: received duplicate interval '{curr_ts.isoformat()}'"
                )

            if curr_ts > expected_ts:
                # Case B: Positive gap
                if not self.enable_gap_recovery:
                    self.sequence_failure_count += 1
                    raise OperationalContinuityError(
                        f"Operational sequence continuity violation: expected interval at "
                        f"'{expected_ts.isoformat()}', received '{curr_ts.isoformat()}'"
                    )

                # Calculate exact missing logical timestamps
                missing_ts_list: list[datetime] = []
                cur_missing = expected_ts
                while cur_missing < curr_ts:
                    missing_ts_list.append(cur_missing)
                    cur_missing += timedelta(minutes=15)

                missing_count = len(missing_ts_list)
                first_missing = missing_ts_list[0]
                last_missing = missing_ts_list[-1]
                gap_duration = curr_ts - prev_ts

                # Invalidate/quarantine pending predictions whose targets fell into the gap
                quarantined = self.feedback_tracker.quarantine_pending_predictions(
                    missing_timestamps=missing_ts_list,
                    previous_ts=prev_ts,
                    current_ts=curr_ts,
                    reason="missing_actual_due_to_gap",
                )
                invalidated_predictions = tuple(quarantined)

                # Reset stateful feature history
                self.feature_generator.reset()

                # Increment continuous segment identifier
                self.current_segment_id += 1

                # Record deterministic gap incident
                gap_incident = GapIncident(
                    previous_timestamp=prev_ts,
                    current_timestamp=curr_ts,
                    gap_duration=gap_duration,
                    missing_interval_count=missing_count,
                    first_missing_timestamp=first_missing,
                    last_missing_timestamp=last_missing,
                    missing_timestamps=tuple(missing_ts_list),
                    state_reset=True,
                    rewarm_required=True,
                    invalidated_prediction_count=len(invalidated_predictions),
                    segment_id=self.current_segment_id,
                )
                self.gap_incidents.append(gap_incident)

        # 2. Resolve delayed feedback for arriving actual
        is_warmup = len(self.feature_generator.usage_buffer) < self.feature_generator.history_size
        feedback_record, was_unmatched = self.feedback_tracker.resolve_feedback(
            record, is_warmup=is_warmup
        )

        # 3. Feed completed feedback into downstream monitoring layer
        if feedback_record is not None and self.monitoring_tracker is not None:
            self.monitoring_tracker.process_feedback(feedback_record)

        # 4. Ingest into stateful feature generator
        feature_vector = self.feature_generator.process_record(record)
        self.last_logical_timestamp = record.logical_timestamp
        self.records_consumed += 1

        prediction_event: PredictionEvent | None = None

        # 5, 6, 7. Generate prediction and register if warm-up is satisfied
        if feature_vector is not None:
            arr_features = np.asarray([feature_vector.to_list()], dtype=np.float64)
            preds = self.forecaster.predict(arr_features)
            predicted_kwh = float(preds[0])
            baseline_kwh = float(record.usage_kwh)

            prediction_event = PredictionEvent(
                model_version=self.model_version,
                origin_timestamp=record.logical_timestamp,
                target_timestamp=feature_vector.target_timestamp,
                prediction_timestamp=datetime.now(timezone.utc),
                predicted_usage_kwh=predicted_kwh,
                baseline_persistence_kwh=baseline_kwh,
                origin_index=record.row_index,
                features=feature_vector.features,
            )
            self.feedback_tracker.register_prediction(prediction_event)

        return OperationalStepResult(
            telemetry_record=record,
            feedback=feedback_record,
            prediction=prediction_event,
            is_warmup=(feature_vector is None),
            unmatched_feedback=was_unmatched,
            gap_incident=gap_incident,
            invalidated_predictions=invalidated_predictions,
        )

    def run_replay(self, replay: TelemetryReplay) -> Iterator[OperationalStepResult]:
        """Stream replay records through the operational engine."""
        self.reset()
        for record in replay.run():
            yield self.process_telemetry(record)

    def reset(self) -> None:
        """Reset operational engine state."""
        self.feature_generator.reset()
        self.feedback_tracker.reset()
        if self.monitoring_tracker is not None:
            self.monitoring_tracker.reset()
        self.last_logical_timestamp = None
        self.records_consumed = 0
        self.sequence_failure_count = 0
        self.gap_incidents.clear()
        self.current_segment_id = 1

    @property
    def gap_incident_count(self) -> int:
        """Total count of detected positive gap incidents."""
        return len(self.gap_incidents)

    @property
    def invalidated_prediction_count(self) -> int:
        """Total count of predictions invalidated due to missing actual observations."""
        return self.feedback_tracker.invalidated_count

