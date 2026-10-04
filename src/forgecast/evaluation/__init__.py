"""Evaluation harness, baselines, and metrics."""

from forgecast.evaluation.baselines import PersistenceBaseline, SeasonalPersistenceBaseline
from forgecast.evaluation.harness import ChronologicalEvaluationHarness, FoldResult
from forgecast.evaluation.metrics import calculate_metrics, calculate_relative_improvement

__all__ = [
    "PersistenceBaseline",
    "SeasonalPersistenceBaseline",
    "calculate_metrics",
    "calculate_relative_improvement",
    "FoldResult",
    "ChronologicalEvaluationHarness",
]
