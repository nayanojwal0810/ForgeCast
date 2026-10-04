"""Ingestion validation functions and stateful temporal sequence validator."""

import math
from datetime import datetime, timedelta
from typing import Any

from forgecast.ingestion.contract import (
    ALLOWED_DAYS_OF_WEEK,
    ALLOWED_LOAD_TYPES,
    ALLOWED_WEEK_STATUS,
    INTERVAL_CADENCE,
    REQUIRED_FIELDS,
    TelemetryRecord,
    TelemetryValidationError,
)

DATE_FORMAT: str = "%d/%m/%Y %H:%M"


def parse_logical_timestamp(raw_date_str: str) -> datetime:
    """Parse raw dataset timestamp string into continuous logical interval closing time.

    The dataset uses format DD/MM/YYYY HH:MM.
    Each daily block ends with '00:00' with NSM=0 following '23:45'.
    This record represents the interval closing at midnight (23:45-24:00)
    for that day, which logically corresponds to 00:00:00 of the following day.

    Raises:
        TelemetryValidationError: If raw_date_str cannot be parsed as expected.
    """
    if not isinstance(raw_date_str, str) or not raw_date_str.strip():
        raise TelemetryValidationError(
            message="Timestamp string is empty or invalid type",
            rule="malformed_timestamp",
            field="date",
            value=raw_date_str,
            expected=f"Non-empty string matching format '{DATE_FORMAT}'",
        )

    clean_str = raw_date_str.strip()
    try:
        dt = datetime.strptime(clean_str, DATE_FORMAT)
    except ValueError as exc:
        raise TelemetryValidationError(
            message=f"Failed to parse timestamp string '{clean_str}': {exc}",
            rule="malformed_timestamp",
            field="date",
            value=clean_str,
            expected=f"Date string matching '{DATE_FORMAT}'",
        ) from exc

    # Midnight convention: 00:00 marks the end of the 23:45-24:00 interval
    if dt.hour == 0 and dt.minute == 0:
        dt += timedelta(days=1)

    return dt


def _parse_float(
    field_name: str,
    value: Any,
    min_val: float | None = None,
    max_val: float | None = None,
    record_index: int | None = None,
) -> float:
    """Validate and convert numeric float value."""
    if value is None or (isinstance(value, str) and value.strip() == ""):
        raise TelemetryValidationError(
            message=f"Missing or null value for numeric field '{field_name}'",
            rule="null_value",
            field=field_name,
            value=value,
            expected="Valid numeric value",
            record_index=record_index,
        )

    try:
        val = float(value)
    except (ValueError, TypeError) as exc:
        raise TelemetryValidationError(
            message=f"Cannot coerce value '{value}' in field '{field_name}' to float: {exc}",
            rule="type_error",
            field=field_name,
            value=value,
            expected="Numeric float",
            record_index=record_index,
        ) from exc

    if math.isnan(val):
        raise TelemetryValidationError(
            message=f"NaN encountered in numeric field '{field_name}'",
            rule="null_or_nan",
            field=field_name,
            value=value,
            expected="Finite real number",
            record_index=record_index,
        )

    if math.isinf(val):
        raise TelemetryValidationError(
            message=f"Infinite value encountered in numeric field '{field_name}'",
            rule="infinite_value",
            field=field_name,
            value=value,
            expected="Finite real number",
            record_index=record_index,
        )

    if min_val is not None and val < min_val:
        raise TelemetryValidationError(
            message=f"Domain range violation on '{field_name}': {val} is below minimum {min_val}",
            rule="domain_range",
            field=field_name,
            value=val,
            expected=f">= {min_val}",
            record_index=record_index,
        )

    if max_val is not None and val > max_val:
        raise TelemetryValidationError(
            message=f"Domain range violation on '{field_name}': {val} exceeds maximum {max_val}",
            rule="domain_range",
            field=field_name,
            value=val,
            expected=f"<= {max_val}",
            record_index=record_index,
        )

    return val


def _parse_int(
    field_name: str,
    value: Any,
    min_val: int | None = None,
    max_val: int | None = None,
    record_index: int | None = None,
) -> int:
    """Validate and convert numeric integer value."""
    if value is None or (isinstance(value, str) and value.strip() == ""):
        raise TelemetryValidationError(
            message=f"Missing or null value for integer field '{field_name}'",
            rule="null_value",
            field=field_name,
            value=value,
            expected="Valid integer value",
            record_index=record_index,
        )

    try:
        f_val = float(value)
        if not f_val.is_integer():
            raise ValueError(f"Value {value} is not an integer")
        val = int(f_val)
    except (ValueError, TypeError) as exc:
        raise TelemetryValidationError(
            message=f"Cannot coerce value '{value}' in field '{field_name}' to integer: {exc}",
            rule="type_error",
            field=field_name,
            value=value,
            expected="Integer",
            record_index=record_index,
        ) from exc

    if min_val is not None and val < min_val:
        raise TelemetryValidationError(
            message=f"Domain range violation on '{field_name}': {val} is below minimum {min_val}",
            rule="domain_range",
            field=field_name,
            value=val,
            expected=f">= {min_val}",
            record_index=record_index,
        )

    if max_val is not None and val > max_val:
        raise TelemetryValidationError(
            message=f"Domain range violation on '{field_name}': {val} exceeds maximum {max_val}",
            rule="domain_range",
            field=field_name,
            value=val,
            expected=f"<= {max_val}",
            record_index=record_index,
        )

    return val


def validate_record_schema_and_domain(
    raw_record: dict[str, Any], record_index: int | None = None
) -> TelemetryRecord:
    """Validate a single raw telemetry record against schema and domain rules.

    Checks:
    - Required fields existence
    - Unexpected field detection
    - Null / empty checks
    - Numeric type coercion and domain constraints
    - Categorical value validity
    - Logical timestamp generation

    Returns:
        TelemetryRecord: Fully validated, typed record instance.

    Raises:
        TelemetryValidationError: If any schema, type, or domain constraint fails.
    """
    if not isinstance(raw_record, dict):
        raise TelemetryValidationError(
            message=f"Expected telemetry record as dictionary, got {type(raw_record).__name__}",
            rule="type_error",
            value=raw_record,
            expected="dict",
            record_index=record_index,
        )

    # 1. Missing fields check
    missing = [f for f in REQUIRED_FIELDS if f not in raw_record]
    if missing:
        raise TelemetryValidationError(
            message=f"Missing required field(s): {', '.join(missing)}",
            rule="missing_field",
            field=missing[0],
            expected="All required schema fields must be present",
            record_index=record_index,
        )

    # 2. Unexpected fields check
    unexpected = [f for f in raw_record if f not in REQUIRED_FIELDS]
    if unexpected:
        raise TelemetryValidationError(
            message=f"Unexpected field(s) encountered: {', '.join(unexpected)}",
            rule="unexpected_field",
            field=unexpected[0],
            expected="Only declared schema fields allowed",
            record_index=record_index,
        )

    # 3. Timestamp validation and logical time conversion
    raw_date = raw_record["date"]
    try:
        logical_ts = parse_logical_timestamp(raw_date)
    except TelemetryValidationError as err:
        err.record_index = record_index
        raise err

    # 4. Numeric fields validation
    usage_kwh = _parse_float("Usage_kWh", raw_record["Usage_kWh"], min_val=0.0, record_index=record_index)
    lagging_reactive = _parse_float(
        "Lagging_Current_Reactive.Power_kVarh",
        raw_record["Lagging_Current_Reactive.Power_kVarh"],
        min_val=0.0,
        record_index=record_index,
    )
    leading_reactive = _parse_float(
        "Leading_Current_Reactive_Power_kVarh",
        raw_record["Leading_Current_Reactive_Power_kVarh"],
        min_val=0.0,
        record_index=record_index,
    )
    co2 = _parse_float("CO2(tCO2)", raw_record["CO2(tCO2)"], min_val=0.0, record_index=record_index)
    lagging_pf = _parse_float(
        "Lagging_Current_Power_Factor",
        raw_record["Lagging_Current_Power_Factor"],
        min_val=0.0,
        max_val=100.0,
        record_index=record_index,
    )
    leading_pf = _parse_float(
        "Leading_Current_Power_Factor",
        raw_record["Leading_Current_Power_Factor"],
        min_val=0.0,
        max_val=100.0,
        record_index=record_index,
    )
    nsm = _parse_int("NSM", raw_record["NSM"], min_val=0, max_val=86399, record_index=record_index)

    # 5. Categorical fields validation
    week_status = str(raw_record["WeekStatus"]).strip()
    if week_status not in ALLOWED_WEEK_STATUS:
        raise TelemetryValidationError(
            message=f"Invalid value '{week_status}' for WeekStatus",
            rule="invalid_category",
            field="WeekStatus",
            value=week_status,
            expected=f"One of {sorted(ALLOWED_WEEK_STATUS)}",
            record_index=record_index,
        )

    day_of_week = str(raw_record["Day_of_week"]).strip()
    if day_of_week not in ALLOWED_DAYS_OF_WEEK:
        raise TelemetryValidationError(
            message=f"Invalid value '{day_of_week}' for Day_of_week",
            rule="invalid_category",
            field="Day_of_week",
            value=day_of_week,
            expected=f"One of {sorted(ALLOWED_DAYS_OF_WEEK)}",
            record_index=record_index,
        )

    load_type = str(raw_record["Load_Type"]).strip()
    if load_type not in ALLOWED_LOAD_TYPES:
        raise TelemetryValidationError(
            message=f"Invalid value '{load_type}' for Load_Type",
            rule="invalid_category",
            field="Load_Type",
            value=load_type,
            expected=f"One of {sorted(ALLOWED_LOAD_TYPES)}",
            record_index=record_index,
        )

    return TelemetryRecord(
        logical_timestamp=logical_ts,
        raw_date=str(raw_date),
        usage_kwh=usage_kwh,
        lagging_reactive_power_kvarh=lagging_reactive,
        leading_reactive_power_kvarh=leading_reactive,
        co2_tco2=co2,
        lagging_power_factor=lagging_pf,
        leading_power_factor=leading_pf,
        nsm=nsm,
        week_status=week_status,
        day_of_week=day_of_week,
        load_type=load_type,
        row_index=record_index,
        raw_data=dict(raw_record),
    )


class TemporalValidator:
    """Stateful validator enforcing per-record schema/domain contracts and temporal sequence continuity."""

    def __init__(self, cadence: timedelta = INTERVAL_CADENCE) -> None:
        self.cadence = cadence
        self.last_logical_timestamp: datetime | None = None
        self.record_count: int = 0

    def validate_record(
        self, raw_record: dict[str, Any], record_index: int | None = None
    ) -> TelemetryRecord:
        """Validate an incoming raw telemetry record and verify sequence continuity."""
        current_index = record_index if record_index is not None else self.record_count
        record = validate_record_schema_and_domain(raw_record, record_index=current_index)

        if self.last_logical_timestamp is not None:
            delta = record.logical_timestamp - self.last_logical_timestamp
            if delta == timedelta(0):
                raise TelemetryValidationError(
                    message=f"Duplicate logical interval timestamp '{record.logical_timestamp.isoformat()}'",
                    rule="duplicate_interval",
                    field="date",
                    value=record.raw_date,
                    expected=f"Interval strictly {self.cadence} after {self.last_logical_timestamp.isoformat()}",
                    record_index=current_index,
                )
            if delta < timedelta(0):
                raise TelemetryValidationError(
                    message=(
                        f"Out-of-order logical timestamp '{record.logical_timestamp.isoformat()}' "
                        f"arrived after '{self.last_logical_timestamp.isoformat()}'"
                    ),
                    rule="out_of_order",
                    field="date",
                    value=record.raw_date,
                    expected="Monotonically advancing timestamp",
                    record_index=current_index,
                )
            if delta != self.cadence:
                raise TelemetryValidationError(
                    message=(
                        f"Temporal continuity gap / invalid cadence: interval delta is {delta}, "
                        f"expected {self.cadence} (last: '{self.last_logical_timestamp.isoformat()}', "
                        f"current: '{record.logical_timestamp.isoformat()}')"
                    ),
                    rule="temporal_gap",
                    field="date",
                    value=record.raw_date,
                    expected=f"Gap exactly equal to {self.cadence}",
                    record_index=current_index,
                )

        self.last_logical_timestamp = record.logical_timestamp
        self.record_count += 1
        return record

    def reset(self) -> None:
        """Reset temporal state buffer."""
        self.last_logical_timestamp = None
        self.record_count = 0
