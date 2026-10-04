"""Unit tests for telemetry data contract and schema validation."""

import pytest

from forgecast.ingestion.contract import TelemetryValidationError
from forgecast.ingestion.validator import validate_record_schema_and_domain


@pytest.fixture
def valid_record() -> dict[str, str]:
    """Provide a standard valid raw telemetry record dictionary."""
    return {
        "date": "01/01/2018 00:15",
        "Usage_kWh": "3.17",
        "Lagging_Current_Reactive.Power_kVarh": "2.95",
        "Leading_Current_Reactive_Power_kVarh": "0",
        "CO2(tCO2)": "0",
        "Lagging_Current_Power_Factor": "73.21",
        "Leading_Current_Power_Factor": "100",
        "NSM": "900",
        "WeekStatus": "Weekday",
        "Day_of_week": "Monday",
        "Load_Type": "Light_Load",
    }


def test_valid_record_passes(valid_record: dict[str, str]) -> None:
    """Confirm a valid raw record is validated and coerced into typed TelemetryRecord."""
    rec = validate_record_schema_and_domain(valid_record, record_index=0)
    assert rec.row_index == 0
    assert rec.usage_kwh == 3.17
    assert rec.lagging_reactive_power_kvarh == 2.95
    assert rec.leading_reactive_power_kvarh == 0.0
    assert rec.co2_tco2 == 0.0
    assert rec.lagging_power_factor == 73.21
    assert rec.leading_power_factor == 100.0
    assert rec.nsm == 900
    assert rec.week_status == "Weekday"
    assert rec.day_of_week == "Monday"
    assert rec.load_type == "Light_Load"
    assert rec.logical_timestamp.year == 2018
    assert rec.logical_timestamp.month == 1
    assert rec.logical_timestamp.day == 1
    assert rec.logical_timestamp.hour == 0
    assert rec.logical_timestamp.minute == 15


@pytest.mark.parametrize(
    "missing_field",
    [
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
    ],
)
def test_missing_required_field_fails(
    valid_record: dict[str, str], missing_field: str
) -> None:
    """Confirm omission of any required schema field raises TelemetryValidationError."""
    record = dict(valid_record)
    del record[missing_field]

    with pytest.raises(TelemetryValidationError) as exc_info:
        validate_record_schema_and_domain(record)
    assert exc_info.value.rule == "missing_field"
    assert exc_info.value.field == missing_field


def test_unexpected_field_fails(valid_record: dict[str, str]) -> None:
    """Confirm that extraneous, undeclared fields are rejected."""
    record = dict(valid_record)
    record["Unapproved_Field"] = "unexpected_data"

    with pytest.raises(TelemetryValidationError) as exc_info:
        validate_record_schema_and_domain(record)
    assert exc_info.value.rule == "unexpected_field"
    assert exc_info.value.field == "Unapproved_Field"


@pytest.mark.parametrize("null_val", [None, "", "   "])
def test_null_numeric_value_fails(valid_record: dict[str, str], null_val: str | None) -> None:
    """Confirm null or empty strings in numeric fields raise validation errors."""
    record = dict(valid_record)
    record["Usage_kWh"] = null_val  # type: ignore[assignment]

    with pytest.raises(TelemetryValidationError) as exc_info:
        validate_record_schema_and_domain(record)
    assert exc_info.value.rule == "null_value"
    assert exc_info.value.field == "Usage_kWh"


def test_nan_numeric_value_fails(valid_record: dict[str, str]) -> None:
    """Confirm NaN values in numeric fields are rejected."""
    record = dict(valid_record)
    record["Usage_kWh"] = "NaN"

    with pytest.raises(TelemetryValidationError) as exc_info:
        validate_record_schema_and_domain(record)
    assert exc_info.value.rule == "null_or_nan"
    assert exc_info.value.field == "Usage_kWh"


def test_infinite_numeric_value_fails(valid_record: dict[str, str]) -> None:
    """Confirm Inf values in numeric fields are rejected."""
    record = dict(valid_record)
    record["Usage_kWh"] = "inf"

    with pytest.raises(TelemetryValidationError) as exc_info:
        validate_record_schema_and_domain(record)
    assert exc_info.value.rule == "infinite_value"
    assert exc_info.value.field == "Usage_kWh"


def test_non_numeric_type_error(valid_record: dict[str, str]) -> None:
    """Confirm uncoercible non-numeric strings raise type_error."""
    record = dict(valid_record)
    record["Usage_kWh"] = "not_a_float"

    with pytest.raises(TelemetryValidationError) as exc_info:
        validate_record_schema_and_domain(record)
    assert exc_info.value.rule == "type_error"
    assert exc_info.value.field == "Usage_kWh"


@pytest.mark.parametrize(
    ("field", "bad_val"),
    [
        ("Usage_kWh", "-0.01"),
        ("Usage_kWh", "-10.5"),
        ("Lagging_Current_Reactive.Power_kVarh", "-0.001"),
        ("Leading_Current_Reactive_Power_kVarh", "-1.0"),
        ("CO2(tCO2)", "-0.0001"),
        ("Lagging_Current_Power_Factor", "-0.1"),
        ("Lagging_Current_Power_Factor", "100.01"),
        ("Leading_Current_Power_Factor", "-1.0"),
        ("Leading_Current_Power_Factor", "105.0"),
        ("NSM", "-1"),
        ("NSM", "86400"),
    ],
)
def test_domain_range_violations(
    valid_record: dict[str, str], field: str, bad_val: str
) -> None:
    """Confirm out-of-domain numeric quantities fail domain constraints."""
    record = dict(valid_record)
    record[field] = bad_val

    with pytest.raises(TelemetryValidationError) as exc_info:
        validate_record_schema_and_domain(record)
    assert exc_info.value.rule == "domain_range"
    assert exc_info.value.field == field


@pytest.mark.parametrize(
    ("field", "bad_cat"),
    [
        ("WeekStatus", "Holiday"),
        ("WeekStatus", "weekday"),
        ("Day_of_week", "Funday"),
        ("Day_of_week", "monday"),
        ("Load_Type", "Extreme_Load"),
        ("Load_Type", "light_load"),
    ],
)
def test_invalid_categorical_values(
    valid_record: dict[str, str], field: str, bad_cat: str
) -> None:
    """Confirm unrecognized categorical values raise invalid_category error."""
    record = dict(valid_record)
    record[field] = bad_cat

    with pytest.raises(TelemetryValidationError) as exc_info:
        validate_record_schema_and_domain(record)
    assert exc_info.value.rule == "invalid_category"
    assert exc_info.value.field == field


@pytest.mark.parametrize(
    "malformed_date",
    [
        "2018-01-01 00:15",
        "01-01-2018 00:15",
        "01/01/2018",
        "invalid_timestamp",
        "32/01/2018 00:15",
        "01/13/2018 00:15",
    ],
)
def test_malformed_timestamp_fails(
    valid_record: dict[str, str], malformed_date: str
) -> None:
    """Confirm unparseable or incorrect date format strings raise malformed_timestamp."""
    record = dict(valid_record)
    record["date"] = malformed_date

    with pytest.raises(TelemetryValidationError) as exc_info:
        validate_record_schema_and_domain(record)
    assert exc_info.value.rule == "malformed_timestamp"
    assert exc_info.value.field == "date"
