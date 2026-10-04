"""Stateful feature generator for one-step-ahead causal energy forecasting."""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from typing import Any

from forgecast.ingestion.contract import TelemetryRecord

FEATURE_NAMES: tuple[str, ...] = (
    "usage_lag_0",
    "usage_lag_1",
    "usage_lag_3",
    "usage_lag_7",
    "usage_lag_95",
    "usage_roll_mean_4",
    "usage_roll_mean_96",
    "cal_hour",
    "cal_quarter_slot",
    "cal_dayofweek",
    "cal_is_weekend",
    "cal_month",
    "cal_nsm",
    "cal_nsm_sin",
    "cal_nsm_cos",
    "cal_dow_sin",
    "cal_dow_cos",
    "cal_month_sin",
    "cal_month_cos",
)

WARMUP_RECORDS: int = 96
STEP_CADENCE: timedelta = timedelta(minutes=15)


class FeatureContinuityError(ValueError):
    """Raised when sequence continuity is violated before state ingestion."""

    def __init__(
        self,
        message: str,
        expected_timestamp: datetime | None = None,
        received_timestamp: datetime | None = None,
        record_index: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.expected_timestamp = expected_timestamp
        self.received_timestamp = received_timestamp
        self.record_index = record_index

    def __str__(self) -> str:
        idx_prefix = f"[Record #{self.record_index}] " if self.record_index is not None else ""
        return f"{idx_prefix}{self.message}"


@dataclass(frozen=True)
class FeatureVector:
    """Immutable prediction feature vector carrying explicit temporal provenance."""

    origin_timestamp: datetime
    target_timestamp: datetime
    features: dict[str, float]
    origin_index: int | None = None

    def to_list(self) -> list[float]:
        """Return feature values in canonical FEATURE_NAMES order."""
        return [self.features[name] for name in FEATURE_NAMES]


class StatefulFeatureGenerator:
    """Stateful generator producing Model C feature vectors for next-interval forecasting.

    Maintains a strictly bounded 96-step in-memory history of Usage_kWh.
    At origin interval t, uses historical Usage up to t and deterministic target
    calendar properties to produce features predicting target interval t + 15m.
    """

    def __init__(self, history_size: int = WARMUP_RECORDS, cadence: timedelta = STEP_CADENCE) -> None:
        self.history_size = history_size
        self.cadence = cadence
        self.usage_buffer: deque[float] = deque(maxlen=history_size)
        self.timestamp_buffer: deque[datetime] = deque(maxlen=history_size)
        self.last_logical_timestamp: datetime | None = None
        self.processed_count: int = 0

    def process_record(self, record: TelemetryRecord) -> FeatureVector | None:
        """Ingest a validated telemetry record and generate a feature vector if warm-up is satisfied.

        Args:
            record: Validated TelemetryRecord from the ingestion/replay pipeline.

        Returns:
            FeatureVector if buffer has accumulated >= 96 observations, otherwise None.

        Raises:
            FeatureContinuityError: If record does not arrive strictly cadence after previous record.
        """
        # Temporal continuity protection: verify before modifying state
        if self.last_logical_timestamp is not None:
            expected_ts = self.last_logical_timestamp + self.cadence
            if record.logical_timestamp != expected_ts:
                raise FeatureContinuityError(
                    message=(
                        f"Temporal sequence gap: expected interval at '{expected_ts.isoformat()}', "
                        f"received '{record.logical_timestamp.isoformat()}'"
                    ),
                    expected_timestamp=expected_ts,
                    received_timestamp=record.logical_timestamp,
                    record_index=record.row_index,
                )

        # Ingest into bounded history
        self.usage_buffer.append(record.usage_kwh)
        self.timestamp_buffer.append(record.logical_timestamp)
        self.last_logical_timestamp = record.logical_timestamp
        self.processed_count += 1

        # Warm-up check: 96 contiguous observations required
        if len(self.usage_buffer) < self.history_size:
            return None

        origin_ts = record.logical_timestamp
        target_ts = origin_ts + self.cadence

        # 1. Historical Usage Features (relative to origin i)
        # buffer[-1] is i (usage_lag_0)
        # buffer[-2] is i-1 (usage_lag_1)
        # buffer[-4] is i-3 (usage_lag_3)
        # buffer[-8] is i-7 (usage_lag_7)
        # buffer[0] is i-95 (usage_lag_95)
        u_lag_0 = self.usage_buffer[-1]
        u_lag_1 = self.usage_buffer[-2]
        u_lag_3 = self.usage_buffer[-4]
        u_lag_7 = self.usage_buffer[-8]
        u_lag_95 = self.usage_buffer[0]

        # 2. Rolling Usage Features (all <= origin i)
        # usage_roll_mean_4 = mean(Usage(i-3)...Usage(i))
        roll_4 = (
            self.usage_buffer[-1]
            + self.usage_buffer[-2]
            + self.usage_buffer[-3]
            + self.usage_buffer[-4]
        ) / 4.0

        # usage_roll_mean_96 = mean(Usage(i-95)...Usage(i))
        roll_96 = sum(self.usage_buffer) / 96.0

        # 3. Deterministic Target Calendar Features
        cal_hour = target_ts.hour
        cal_quarter_slot = target_ts.minute // 15
        cal_dayofweek = target_ts.weekday()
        cal_is_weekend = 1 if cal_dayofweek >= 5 else 0
        cal_month = target_ts.month
        cal_nsm = cal_hour * 3600 + target_ts.minute * 60

        # 4. Cyclical Calendar Features
        cal_nsm_sin = math.sin(2.0 * math.pi * cal_nsm / 86400.0)
        cal_nsm_cos = math.cos(2.0 * math.pi * cal_nsm / 86400.0)

        cal_dow_sin = math.sin(2.0 * math.pi * cal_dayofweek / 7.0)
        cal_dow_cos = math.cos(2.0 * math.pi * cal_dayofweek / 7.0)

        cal_month_sin = math.sin(2.0 * math.pi * (cal_month - 1) / 12.0)
        cal_month_cos = math.cos(2.0 * math.pi * (cal_month - 1) / 12.0)

        features: dict[str, float] = {
            "usage_lag_0": u_lag_0,
            "usage_lag_1": u_lag_1,
            "usage_lag_3": u_lag_3,
            "usage_lag_7": u_lag_7,
            "usage_lag_95": u_lag_95,
            "usage_roll_mean_4": roll_4,
            "usage_roll_mean_96": roll_96,
            "cal_hour": float(cal_hour),
            "cal_quarter_slot": float(cal_quarter_slot),
            "cal_dayofweek": float(cal_dayofweek),
            "cal_is_weekend": float(cal_is_weekend),
            "cal_month": float(cal_month),
            "cal_nsm": float(cal_nsm),
            "cal_nsm_sin": cal_nsm_sin,
            "cal_nsm_cos": cal_nsm_cos,
            "cal_dow_sin": cal_dow_sin,
            "cal_dow_cos": cal_dow_cos,
            "cal_month_sin": cal_month_sin,
            "cal_month_cos": cal_month_cos,
        }

        return FeatureVector(
            origin_timestamp=origin_ts,
            target_timestamp=target_ts,
            features=features,
            origin_index=record.row_index,
        )

    def reset(self) -> None:
        """Reset internal history buffers and sequence tracking state."""
        self.usage_buffer.clear()
        self.timestamp_buffer.clear()
        self.last_logical_timestamp = None
        self.processed_count = 0
