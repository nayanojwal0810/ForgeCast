"""Offline dataset builder pairing stateful features with delayed target observations."""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from forgecast.features.generator import (
    FEATURE_NAMES,
    FeatureContinuityError,
    FeatureVector,
    StatefulFeatureGenerator,
)
from forgecast.replay.engine import TelemetryReplay


@dataclass(frozen=True)
class SupervisedSample:
    """A supervised training/evaluation sample pairing features with ground truth target."""

    origin_timestamp: datetime
    target_timestamp: datetime
    features: dict[str, float]
    target_usage_kwh: float
    origin_index: int | None = None
    target_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert sample to flat dictionary suitable for tabular and DataFrame representations."""
        row: dict[str, Any] = {
            "origin_timestamp": self.origin_timestamp,
            "target_timestamp": self.target_timestamp,
            "origin_index": self.origin_index,
            "target_index": self.target_index,
        }
        for name in FEATURE_NAMES:
            row[name] = self.features[name]
        row["target_usage_kwh"] = self.target_usage_kwh
        return row


class OfflineFeatureDatasetBuilder:
    """Offline adapter building supervised (features, target) samples using StatefulFeatureGenerator."""

    def __init__(self, generator: StatefulFeatureGenerator | None = None) -> None:
        self.generator = generator or StatefulFeatureGenerator()

    def build_from_replay(self, replay: TelemetryReplay) -> Iterator[SupervisedSample]:
        """Stream replay records through the stateful generator and pair with delayed actual targets.

        Yields:
            SupervisedSample: Paired (X_features, y_target) observation.

        Raises:
            FeatureContinuityError: If sequence continuity is broken or expected actual does not arrive.
        """
        self.generator.reset()
        pending_feature_vector: FeatureVector | None = None

        for record in replay.run():
            # If a feature vector was emitted previously expecting this interval, pair it as target
            if pending_feature_vector is not None:
                if record.logical_timestamp == pending_feature_vector.target_timestamp:
                    yield SupervisedSample(
                        origin_timestamp=pending_feature_vector.origin_timestamp,
                        target_timestamp=pending_feature_vector.target_timestamp,
                        features=pending_feature_vector.features,
                        target_usage_kwh=record.usage_kwh,
                        origin_index=pending_feature_vector.origin_index,
                        target_index=record.row_index,
                    )
                    pending_feature_vector = None
                else:
                    raise FeatureContinuityError(
                        f"Expected delayed actual at '{pending_feature_vector.target_timestamp.isoformat()}', "
                        f"received '{record.logical_timestamp.isoformat()}'"
                    )

            # Advance state and generate feature vector predicting next interval
            feature_vector = self.generator.process_record(record)
            if feature_vector is not None:
                pending_feature_vector = feature_vector

    def build_dataset(self, replay: TelemetryReplay) -> list[SupervisedSample]:
        """Convenience method collecting all supervised samples into a list."""
        return list(self.build_from_replay(replay))

    def build_dataframe(self, replay: TelemetryReplay) -> Any:
        """Convert replay into a pandas DataFrame of features and targets."""
        import pandas as pd

        samples = [s.to_dict() for s in self.build_from_replay(replay)]
        return pd.DataFrame(samples)
