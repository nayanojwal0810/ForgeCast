"""Feature generation and dataset building components."""

from forgecast.features.builder import OfflineFeatureDatasetBuilder, SupervisedSample
from forgecast.features.generator import (
    FEATURE_NAMES,
    FeatureContinuityError,
    FeatureVector,
    StatefulFeatureGenerator,
)

__all__ = [
    "FEATURE_NAMES",
    "FeatureContinuityError",
    "FeatureVector",
    "StatefulFeatureGenerator",
    "SupervisedSample",
    "OfflineFeatureDatasetBuilder",
]
