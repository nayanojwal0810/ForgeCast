"""Operational monitoring metrics: rolling health aggregator, deployment boundary tracking, and baseline comparison."""

from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
import math
from typing import Any

from forgecast.operational.events import FeedbackRecord


class MonitoringError(ValueError):
    """Base exception for operational monitoring layer errors."""


class MonitoringContinuityError(MonitoringError):
    """Raised when feedback arrives out of order or with duplicate target timestamp."""


@dataclass(frozen=True)
class MetricSnapshot:
    """Immutable snapshot of rolling or cumulative forecasting performance metrics."""

    model_version: str
    window_size: int | None
    sample_count: int
    first_target_timestamp: datetime | None
    last_target_timestamp: datetime | None
    ml_mae: float | None
    ml_rmse: float | None
    baseline_mae: float | None
    baseline_rmse: float | None
    mae_improvement_pct: float | None
    rmse_improvement_pct: float | None
    ml_win_rate: float | None

    def to_dict(self) -> dict[str, Any]:
        """Convert snapshot to dictionary representation."""
        return {
            "model_version": self.model_version,
            "window_size": self.window_size,
            "sample_count": self.sample_count,
            "first_target_timestamp": (
                self.first_target_timestamp.isoformat()
                if self.first_target_timestamp
                else None
            ),
            "last_target_timestamp": (
                self.last_target_timestamp.isoformat()
                if self.last_target_timestamp
                else None
            ),
            "ml_mae": self.ml_mae,
            "ml_rmse": self.ml_rmse,
            "baseline_mae": self.baseline_mae,
            "baseline_rmse": self.baseline_rmse,
            "mae_improvement_pct": self.mae_improvement_pct,
            "rmse_improvement_pct": self.rmse_improvement_pct,
            "ml_win_rate": self.ml_win_rate,
        }


def _calculate_snapshot(
    records: list[FeedbackRecord] | deque[FeedbackRecord],
    model_version: str,
    window_size: int | None,
) -> MetricSnapshot:
    """Compute typed MetricSnapshot over an explicit collection of FeedbackRecord objects."""
    n = len(records)
    if n == 0:
        return MetricSnapshot(
            model_version=model_version,
            window_size=window_size,
            sample_count=0,
            first_target_timestamp=None,
            last_target_timestamp=None,
            ml_mae=None,
            ml_rmse=None,
            baseline_mae=None,
            baseline_rmse=None,
            mae_improvement_pct=None,
            rmse_improvement_pct=None,
            ml_win_rate=None,
        )

    sum_ml_abs = 0.0
    sum_ml_sq = 0.0
    sum_base_abs = 0.0
    sum_base_sq = 0.0
    wins = 0

    for r in records:
        sum_ml_abs += r.ml_absolute_error
        sum_ml_sq += r.ml_absolute_error * r.ml_absolute_error
        sum_base_abs += r.persistence_absolute_error
        sum_base_sq += r.persistence_absolute_error * r.persistence_absolute_error
        if r.ml_absolute_error < r.persistence_absolute_error:
            wins += 1

    ml_mae = sum_ml_abs / n
    ml_rmse = math.sqrt(sum_ml_sq / n)
    base_mae = sum_base_abs / n
    base_rmse = math.sqrt(sum_base_sq / n)

    mae_imp = (
        ((base_mae - ml_mae) / base_mae * 100.0)
        if base_mae > 0.0
        else 0.0
    )
    rmse_imp = (
        ((base_rmse - ml_rmse) / base_rmse * 100.0)
        if base_rmse > 0.0
        else 0.0
    )
    win_rate = wins / n

    return MetricSnapshot(
        model_version=model_version,
        window_size=window_size,
        sample_count=n,
        first_target_timestamp=records[0].target_timestamp,
        last_target_timestamp=records[-1].target_timestamp,
        ml_mae=ml_mae,
        ml_rmse=ml_rmse,
        baseline_mae=base_mae,
        baseline_rmse=base_rmse,
        mae_improvement_pct=mae_imp,
        rmse_improvement_pct=rmse_imp,
        ml_win_rate=win_rate,
    )


class CumulativeMetricsAccumulator:
    """O(1) exact running accumulator for completed feedback records."""

    def __init__(self, model_version: str) -> None:
        self.model_version = model_version
        self.count = 0
        self.first_target_timestamp: datetime | None = None
        self.last_target_timestamp: datetime | None = None
        self.sum_ml_abs = 0.0
        self.sum_ml_sq = 0.0
        self.sum_base_abs = 0.0
        self.sum_base_sq = 0.0
        self.wins = 0

    def add(self, r: FeedbackRecord) -> None:
        if self.count == 0:
            self.first_target_timestamp = r.target_timestamp
        self.last_target_timestamp = r.target_timestamp
        self.count += 1
        self.sum_ml_abs += r.ml_absolute_error
        self.sum_ml_sq += r.ml_absolute_error * r.ml_absolute_error
        self.sum_base_abs += r.persistence_absolute_error
        self.sum_base_sq += r.persistence_absolute_error * r.persistence_absolute_error
        if r.ml_absolute_error < r.persistence_absolute_error:
            self.wins += 1

    def get_snapshot(self) -> MetricSnapshot:
        if self.count == 0:
            return MetricSnapshot(
                model_version=self.model_version,
                window_size=None,
                sample_count=0,
                first_target_timestamp=None,
                last_target_timestamp=None,
                ml_mae=None,
                ml_rmse=None,
                baseline_mae=None,
                baseline_rmse=None,
                mae_improvement_pct=None,
                rmse_improvement_pct=None,
                ml_win_rate=None,
            )
        ml_mae = self.sum_ml_abs / self.count
        ml_rmse = math.sqrt(self.sum_ml_sq / self.count)
        base_mae = self.sum_base_abs / self.count
        base_rmse = math.sqrt(self.sum_base_sq / self.count)
        mae_imp = (base_mae - ml_mae) / base_mae * 100.0 if base_mae > 0.0 else 0.0
        rmse_imp = (base_rmse - ml_rmse) / base_rmse * 100.0 if base_rmse > 0.0 else 0.0
        win_rate = self.wins / self.count

        return MetricSnapshot(
            model_version=self.model_version,
            window_size=None,
            sample_count=self.count,
            first_target_timestamp=self.first_target_timestamp,
            last_target_timestamp=self.last_target_timestamp,
            ml_mae=ml_mae,
            ml_rmse=ml_rmse,
            baseline_mae=base_mae,
            baseline_rmse=base_rmse,
            mae_improvement_pct=mae_imp,
            rmse_improvement_pct=rmse_imp,
            ml_win_rate=win_rate,
        )


class RollingWindow:
    """Maintains a bounded historical deque of recent feedback records with automatic FIFO eviction."""

    def __init__(self, window_size: int, model_version: str) -> None:
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        self.window_size = window_size
        self.model_version = model_version
        self.buffer: deque[FeedbackRecord] = deque(maxlen=window_size)

    def add(self, r: FeedbackRecord) -> None:
        self.buffer.append(r)

    def get_snapshot(self) -> MetricSnapshot:
        return _calculate_snapshot(self.buffer, self.model_version, self.window_size)

    def get_records(self) -> list[FeedbackRecord]:
        return list(self.buffer)


class VersionMetricsTracker:
    """Manages cumulative and rolling windows for a single, strictly isolated model version."""

    def __init__(
        self,
        model_version: str,
        window_sizes: tuple[int, ...] = (96, 672),
        deployment_boundary: datetime | None = None,
    ) -> None:
        self.model_version = model_version
        self.window_sizes = window_sizes
        self.deployment_boundary = deployment_boundary
        self.last_target_timestamp: datetime | None = None
        self.full_replay_cumulative = CumulativeMetricsAccumulator(model_version=model_version)
        self.post_deployment_cumulative = CumulativeMetricsAccumulator(model_version=model_version)
        self.rolling_windows: dict[int, RollingWindow] = {
            w: RollingWindow(window_size=w, model_version=model_version) for w in window_sizes
        }

    def set_deployment_boundary(self, boundary: datetime) -> None:
        self.deployment_boundary = boundary

    def process_feedback(self, record: FeedbackRecord) -> None:
        if record.model_version != self.model_version:
            raise MonitoringError(
                f"Model version mismatch: expected '{self.model_version}', got '{record.model_version}'"
            )

        if self.last_target_timestamp is not None:
            if record.target_timestamp < self.last_target_timestamp:
                raise MonitoringContinuityError(
                    f"Out-of-order feedback target timestamp: incoming '{record.target_timestamp.isoformat()}', "
                    f"latest accepted was '{self.last_target_timestamp.isoformat()}'"
                )
            if record.target_timestamp == self.last_target_timestamp:
                raise MonitoringContinuityError(
                    f"Duplicate feedback target timestamp: incoming '{record.target_timestamp.isoformat()}' "
                    "has already been processed"
                )

        self.last_target_timestamp = record.target_timestamp
        self.full_replay_cumulative.add(record)

        if self.deployment_boundary is not None and record.target_timestamp > self.deployment_boundary:
            self.post_deployment_cumulative.add(record)

        for window in self.rolling_windows.values():
            window.add(record)


class RollingMetricsAggregator:
    """Manages rolling observation windows and cumulative metrics with strict model version isolation."""

    def __init__(
        self,
        window_sizes: tuple[int, ...] = (96, 672),
        deployment_boundary: datetime | None = None,
    ) -> None:
        self.window_sizes = window_sizes
        self.deployment_boundary = deployment_boundary
        self._trackers: dict[str, VersionMetricsTracker] = {}

    def set_deployment_boundary(self, boundary: datetime) -> None:
        self.deployment_boundary = boundary
        for tracker in self._trackers.values():
            tracker.set_deployment_boundary(boundary)

    def process_feedback(self, record: FeedbackRecord) -> None:
        """Ingest feedback record into the appropriate model version tracker."""
        version = record.model_version
        if version not in self._trackers:
            self._trackers[version] = VersionMetricsTracker(
                model_version=version,
                window_sizes=self.window_sizes,
                deployment_boundary=self.deployment_boundary,
            )
        self._trackers[version].process_feedback(record)

    def get_snapshot(
        self,
        model_version: str,
        window_size: int | None = None,
        post_deployment_only: bool = False,
    ) -> MetricSnapshot:
        """Retrieve metric snapshot for model_version.

        Args:
            model_version: Model identifier string.
            window_size: Bounded window capacity (e.g. 96, 672) or None for cumulative.
            post_deployment_only: If True and window_size is None, returns post-deployment cumulative.
        """
        if model_version not in self._trackers:
            return MetricSnapshot(
                model_version=model_version,
                window_size=window_size,
                sample_count=0,
                first_target_timestamp=None,
                last_target_timestamp=None,
                ml_mae=None,
                ml_rmse=None,
                baseline_mae=None,
                baseline_rmse=None,
                mae_improvement_pct=None,
                rmse_improvement_pct=None,
                ml_win_rate=None,
            )
        tracker = self._trackers[model_version]
        if window_size is None:
            if post_deployment_only:
                return tracker.post_deployment_cumulative.get_snapshot()
            return tracker.full_replay_cumulative.get_snapshot()

        if window_size not in tracker.rolling_windows:
            raise KeyError(f"Window size {window_size} not configured for model version '{model_version}'")
        return tracker.rolling_windows[window_size].get_snapshot()

    def get_records(self, model_version: str, window_size: int) -> list[FeedbackRecord]:
        """Return list of FeedbackRecords currently held in rolling window."""
        if model_version not in self._trackers:
            return []
        tracker = self._trackers[model_version]
        if window_size not in tracker.rolling_windows:
            raise KeyError(f"Window size {window_size} not configured for model version '{model_version}'")
        return tracker.rolling_windows[window_size].get_records()

    def get_tracked_versions(self) -> list[str]:
        """Return list of distinct model versions with recorded feedback."""
        return list(self._trackers.keys())

    def reset(self) -> None:
        """Clear all version trackers."""
        self._trackers.clear()


@dataclass(frozen=True)
class BaselineComparisonSummary:
    """Consolidated operational comparison between ML forecasts and persistence baseline."""

    model_version: str
    deployment_boundary: datetime | None
    full_replay_cumulative: MetricSnapshot
    post_deployment_cumulative: MetricSnapshot | None
    rolling_24h: MetricSnapshot | None
    rolling_7d: MetricSnapshot | None
    rolling_windows: dict[int, MetricSnapshot]

    @property
    def cumulative(self) -> MetricSnapshot:
        """Backward compatibility alias for full_replay_cumulative."""
        return self.full_replay_cumulative

    def to_dict(self) -> dict[str, Any]:
        """Convert comparison summary to dictionary."""
        return {
            "model_version": self.model_version,
            "deployment_boundary": (
                self.deployment_boundary.isoformat() if self.deployment_boundary else None
            ),
            "cumulative": self.full_replay_cumulative.to_dict(),
            "full_replay_cumulative": self.full_replay_cumulative.to_dict(),
            "post_deployment_cumulative": (
                self.post_deployment_cumulative.to_dict()
                if self.post_deployment_cumulative
                else None
            ),
            "rolling_24h": self.rolling_24h.to_dict() if self.rolling_24h else None,
            "rolling_7d": self.rolling_7d.to_dict() if self.rolling_7d else None,
            "rolling_windows": {w: s.to_dict() for w, s in self.rolling_windows.items()},
        }


@dataclass(frozen=True)
class WindowDiagnostic:
    """Factual descriptive profile of a rolling observation window."""

    window_name: str
    sample_count: int
    ml_mae: float
    persistence_mae: float
    mae_difference: float
    ml_win_rate: float
    persistence_win_rate: float
    tie_count: int
    usage_mean: float
    usage_std: float
    usage_min: float
    usage_max: float
    mean_prediction: float
    mean_actual: float
    mean_signed_error: float
    top_10pct_sample_count: int
    top_10pct_error_sum: float
    total_ml_error_sum: float
    top_10pct_error_fraction: float
    top_5_error_indices: list[int]
    error_concentration_summary: str

    def to_dict(self) -> dict[str, Any]:
        """Convert diagnostic to dictionary representation."""
        return asdict(self)


def diagnose_recent_window(
    records: Sequence[FeedbackRecord], window_name: str = "24h"
) -> WindowDiagnostic:
    """Produce factual descriptive diagnostic on recent feedback records."""
    n = len(records)
    if n == 0:
        raise ValueError("Cannot diagnose an empty sequence of records")

    sum_ml = sum(r.ml_absolute_error for r in records)
    sum_base = sum(r.persistence_absolute_error for r in records)
    ml_mae = sum_ml / n
    base_mae = sum_base / n
    mae_diff = ml_mae - base_mae

    ml_wins = sum(1 for r in records if r.ml_absolute_error < r.persistence_absolute_error)
    base_wins = sum(1 for r in records if r.persistence_absolute_error < r.ml_absolute_error)
    ties = n - ml_wins - base_wins

    predictions = [r.predicted_usage_kwh for r in records]
    usages = [r.actual_usage_kwh for r in records]
    signed_errors = [p - a for p, a in zip(predictions, usages)]
    mean_pred = sum(predictions) / n
    mean_act = sum(usages) / n
    mean_signed_err = sum(signed_errors) / n

    u_mean = mean_act
    u_std = math.sqrt(sum((u - u_mean) ** 2 for u in usages) / n)
    u_min = min(usages)
    u_max = max(usages)

    indexed_errors = sorted(
        enumerate(records), key=lambda x: x[1].ml_absolute_error, reverse=True
    )
    k_10pct = max(1, math.ceil(n * 0.10))
    top_10pct_err = sum(r.ml_absolute_error for _, r in indexed_errors[:k_10pct])
    top_10pct_fraction = top_10pct_err / sum_ml if sum_ml > 0 else 0.0
    top_5_indices = [idx for idx, _ in indexed_errors[:5]]

    if u_std < 1.0 and u_max < 5.0:
        summary = (
            f"Flat baseload regime (usage std={u_std:.3f} kWh, range [{u_min:.2f}, {u_max:.2f}]). "
            f"Zero-order hold persistence baseline achieves near-zero error ({base_mae:.4f} kWh), "
            f"while ML produces positive signed bias (mean signed error = {mean_signed_err:+.4f} kWh, "
            f"mean pred = {mean_pred:.4f} vs mean actual = {mean_act:.4f})."
        )
    else:
        summary = (
            f"Dynamic operating regime (usage std={u_std:.3f} kWh, range [{u_min:.2f}, {u_max:.2f}]). "
            f"ML MAE={ml_mae:.4f} vs Persistence MAE={base_mae:.4f} "
            f"(mean signed error = {mean_signed_err:+.4f} kWh)."
        )

    return WindowDiagnostic(
        window_name=window_name,
        sample_count=n,
        ml_mae=ml_mae,
        persistence_mae=base_mae,
        mae_difference=mae_diff,
        ml_win_rate=ml_wins / n,
        persistence_win_rate=base_wins / n,
        tie_count=ties,
        usage_mean=u_mean,
        usage_std=u_std,
        usage_min=u_min,
        usage_max=u_max,
        mean_prediction=mean_pred,
        mean_actual=mean_act,
        mean_signed_error=mean_signed_err,
        top_10pct_sample_count=k_10pct,
        top_10pct_error_sum=top_10pct_err,
        total_ml_error_sum=sum_ml,
        top_10pct_error_fraction=top_10pct_fraction,
        top_5_error_indices=top_5_indices,
        error_concentration_summary=summary,
    )


class BaselineComparisonTracker:
    """Operational tracker coordinating ML vs persistence baseline comparisons."""

    def __init__(
        self,
        window_sizes: tuple[int, ...] = (96, 672),
        deployment_boundary: datetime | None = None,
        aggregator: RollingMetricsAggregator | None = None,
    ) -> None:
        self.window_sizes = window_sizes
        self.deployment_boundary = deployment_boundary
        self.aggregator = aggregator or RollingMetricsAggregator(
            window_sizes=window_sizes,
            deployment_boundary=deployment_boundary,
        )

    def set_deployment_boundary(self, boundary: datetime) -> None:
        """Register model deployment / training cutoff boundary."""
        self.deployment_boundary = boundary
        self.aggregator.set_deployment_boundary(boundary)

    def process_feedback(self, record: FeedbackRecord) -> None:
        """Ingest completed feedback record into monitoring aggregator."""
        self.aggregator.process_feedback(record)

    def get_full_replay_snapshot(self, model_version: str) -> MetricSnapshot:
        """Return full replay cumulative metrics snapshot for model_version."""
        return self.aggregator.get_snapshot(
            model_version=model_version, window_size=None, post_deployment_only=False
        )

    def get_post_deployment_snapshot(self, model_version: str) -> MetricSnapshot:
        """Return post-deployment cumulative metrics snapshot for model_version."""
        return self.aggregator.get_snapshot(
            model_version=model_version, window_size=None, post_deployment_only=True
        )

    def get_cumulative_snapshot(self, model_version: str) -> MetricSnapshot:
        """Backward compatibility alias for full replay cumulative snapshot."""
        return self.get_full_replay_snapshot(model_version)

    def get_rolling_snapshot(self, model_version: str, window_size: int) -> MetricSnapshot:
        """Return rolling window metrics snapshot for model_version."""
        return self.aggregator.get_snapshot(model_version=model_version, window_size=window_size)

    def get_rolling_records(self, model_version: str, window_size: int) -> list[FeedbackRecord]:
        """Return underlying FeedbackRecords in specified rolling window."""
        return self.aggregator.get_records(model_version=model_version, window_size=window_size)

    def get_comparison_summary(self, model_version: str) -> BaselineComparisonSummary:
        """Return consolidated cumulative and rolling baseline comparison summary."""
        full_replay = self.get_full_replay_snapshot(model_version)
        post_deploy = (
            self.get_post_deployment_snapshot(model_version)
            if self.deployment_boundary is not None
            else None
        )
        rolling_snaps = {
            w: self.get_rolling_snapshot(model_version, w) for w in self.window_sizes
        }
        return BaselineComparisonSummary(
            model_version=model_version,
            deployment_boundary=self.deployment_boundary,
            full_replay_cumulative=full_replay,
            post_deployment_cumulative=post_deploy,
            rolling_24h=rolling_snaps.get(96),
            rolling_7d=rolling_snaps.get(672),
            rolling_windows=rolling_snaps,
        )

    def diagnose_window(self, model_version: str, window_size: int) -> WindowDiagnostic:
        """Generate WindowDiagnostic for specified rolling window."""
        records = self.get_rolling_records(model_version, window_size)
        return diagnose_recent_window(records, window_name=f"{window_size}-observation")

    def reset(self) -> None:
        """Reset monitoring tracker state."""
        self.aggregator.reset()
