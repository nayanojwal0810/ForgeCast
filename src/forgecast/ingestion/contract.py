"""Data contract definitions, schema constants, and structured record types."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

REQUIRED_FIELDS: tuple[str, ...] = (
    "date",
    "Usage_kWh",
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM",
    "WeekStatus",
    "Day_of_week",
    "Load_Type",
)

ALLOWED_WEEK_STATUS: frozenset[str] = frozenset({"Weekday", "Weekend"})

ALLOWED_DAYS_OF_WEEK: frozenset[str] = frozenset(
    {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
)

ALLOWED_LOAD_TYPES: frozenset[str] = frozenset(
    {"Light_Load", "Medium_Load", "Maximum_Load"}
)

INTERVAL_CADENCE: timedelta = timedelta(minutes=15)


class TelemetryValidationError(ValueError):
    """Raised when an incoming telemetry record violates schema, domain, or temporal rules."""

    def __init__(
        self,
        message: str,
        rule: str,
        field: str | None = None,
        value: Any = None,
        expected: str | None = None,
        record_index: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.rule = rule
        self.field = field
        self.value = value
        self.expected = expected
        self.record_index = record_index

    def __str__(self) -> str:
        idx_prefix = f"[Record #{self.record_index}] " if self.record_index is not None else ""
        details = []
        if self.rule:
            details.append(f"rule='{self.rule}'")
        if self.field:
            details.append(f"field='{self.field}'")
        if self.value is not None:
            details.append(f"received={self.value!r}")
        if self.expected:
            details.append(f"expected={self.expected}")
        suffix = f" ({', '.join(details)})" if details else ""
        return f"{idx_prefix}{self.message}{suffix}"


@dataclass(frozen=True)
class TelemetryRecord:
    """Validated, typed, and timestamp-normalized telemetry observation."""

    logical_timestamp: datetime
    raw_date: str
    usage_kwh: float
    lagging_reactive_power_kvarh: float
    leading_reactive_power_kvarh: float
    co2_tco2: float
    lagging_power_factor: float
    leading_power_factor: float
    nsm: int
    week_status: str
    day_of_week: str
    load_type: str
    row_index: int | None = None
    raw_data: dict[str, Any] = field(default_factory=dict)
