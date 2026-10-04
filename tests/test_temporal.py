"""Unit and integration tests for temporal continuity, sequence integrity, and midnight convention."""

from datetime import datetime, timedelta

import pytest

from forgecast.ingestion.contract import TelemetryValidationError
from forgecast.ingestion.validator import TemporalValidator, parse_logical_timestamp


def _make_record(date_str: str, usage: float = 3.0, nsm: int = 900) -> dict[str, str]:
    """Helper to generate a minimally valid raw telemetry row dictionary."""
    return {
        "date": date_str,
        "Usage_kWh": str(usage),
        "Lagging_Current_Reactive.Power_kVarh": "2.0",
        "Leading_Current_Reactive_Power_kVarh": "0.0",
        "CO2(tCO2)": "0.0",
        "Lagging_Current_Power_Factor": "70.0",
        "Leading_Current_Power_Factor": "100.0",
        "NSM": str(nsm),
        "WeekStatus": "Weekday",
        "Day_of_week": "Monday",
        "Load_Type": "Light_Load",
    }


def test_correctly_ordered_records_pass() -> None:
    """Confirm strictly contiguous 15-minute records advance state smoothly."""
    validator = TemporalValidator()
    rec1 = validator.validate_record(_make_record("01/01/2018 00:15", nsm=900), record_index=0)
    rec2 = validator.validate_record(_make_record("01/01/2018 00:30", nsm=1800), record_index=1)
    rec3 = validator.validate_record(_make_record("01/01/2018 00:45", nsm=2700), record_index=2)

    assert rec1.logical_timestamp == datetime(2018, 1, 1, 0, 15)
    assert rec2.logical_timestamp == datetime(2018, 1, 1, 0, 30)
    assert rec3.logical_timestamp == datetime(2018, 1, 1, 0, 45)
    assert validator.record_count == 3


def test_duplicate_interval_fails() -> None:
    """Confirm that receiving the same logical interval consecutively raises duplicate_interval error."""
    validator = TemporalValidator()
    validator.validate_record(_make_record("01/01/2018 01:00", nsm=3600), record_index=0)

    with pytest.raises(TelemetryValidationError) as exc_info:
        validator.validate_record(_make_record("01/01/2018 01:00", nsm=3600), record_index=1)
    assert exc_info.value.rule == "duplicate_interval"
    assert exc_info.value.record_index == 1


def test_skipped_interval_fails() -> None:
    """Confirm a dropped interval (e.g. 30-minute jump instead of 15 minutes) raises temporal_gap error."""
    validator = TemporalValidator()
    validator.validate_record(_make_record("01/01/2018 01:00", nsm=3600), record_index=0)

    # 01:30 arrives instead of 01:15 (a 30-minute delta)
    with pytest.raises(TelemetryValidationError) as exc_info:
        validator.validate_record(_make_record("01/01/2018 01:30", nsm=5400), record_index=1)
    assert exc_info.value.rule == "temporal_gap"
    assert exc_info.value.record_index == 1


def test_out_of_order_interval_fails() -> None:
    """Confirm a backwards-jumping record raises out_of_order error."""
    validator = TemporalValidator()
    validator.validate_record(_make_record("01/01/2018 02:00", nsm=7200), record_index=0)

    # 01:45 arrives after 02:00
    with pytest.raises(TelemetryValidationError) as exc_info:
        validator.validate_record(_make_record("01/01/2018 01:45", nsm=6300), record_index=1)
    assert exc_info.value.rule == "out_of_order"
    assert exc_info.value.record_index == 1


def test_midnight_boundary_interpreted_correctly() -> None:
    """Confirm the 00:00 record at the end of each day resolves to the following day's 00:00:00."""
    # Day 1 final observation (23:45-24:00 closing interval)
    ts_midnight_day1 = parse_logical_timestamp("01/01/2018 00:00")
    assert ts_midnight_day1 == datetime(2018, 1, 2, 0, 0)

    # Final record of 2018 (closing 2018-12-31 23:45-24:00)
    ts_midnight_yearend = parse_logical_timestamp("31/12/2018 00:00")
    assert ts_midnight_yearend == datetime(2019, 1, 1, 0, 0)


def test_midnight_transition_continuity_permanent_guard() -> None:
    """PERMANENT REGRESSION GUARD: Verify the exact 96th-row midnight boundary transition.

    The UCI Steel Industry dataset records the final 15-minute interval of each day
    (23:45 - 24:00) with string 'DD/MM/YYYY 00:00' and NSM=0, placed at the end of the daily block.
    
    If naively parsed without the midnight convention, '01/01/2018 00:00' would become 
    2018-01-01 00:00:00, creating an apparent 23h45m jump BACKWARDS in time from 23:45:00.
    
    If naively sorted by string or standard datetime, row 96 would jump to row 1, 
    destroying the true temporal sequence.
    
    This test verifies that under the validator:
    1. '01/01/2018 23:45' -> 2018-01-01 23:45:00
    2. '01/01/2018 00:00' -> 2018-01-02 00:00:00 (+15 min delta)
    3. '02/01/2018 00:15' -> 2018-01-02 00:15:00 (+15 min delta)
    """
    validator = TemporalValidator()

    # Observation 95: 23:45 of Day 1
    rec_2345 = validator.validate_record(
        _make_record("01/01/2018 23:45", usage=3.67, nsm=85500), record_index=94
    )
    assert rec_2345.logical_timestamp == datetime(2018, 1, 1, 23, 45)

    # Observation 96: 00:00 of Day 1 closing interval
    rec_0000 = validator.validate_record(
        _make_record("01/01/2018 00:00", usage=3.42, nsm=0), record_index=95
    )
    assert rec_0000.logical_timestamp == datetime(2018, 1, 2, 0, 0)
    assert (rec_0000.logical_timestamp - rec_2345.logical_timestamp) == timedelta(minutes=15)

    # Observation 97: 00:15 of Day 2
    rec_0015 = validator.validate_record(
        _make_record("02/01/2018 00:15", usage=3.20, nsm=900), record_index=96
    )
    assert rec_0015.logical_timestamp == datetime(2018, 1, 2, 0, 15)
    assert (rec_0015.logical_timestamp - rec_0000.logical_timestamp) == timedelta(minutes=15)


def test_naive_datetime_sorting_corruption_demonstration() -> None:
    """Verify that naive sorting by standard datetime inverts the chronological sequence.

    Demonstrates why the source row order must be preserved and sorting forbidden.
    """
    row_95_str = "01/01/2018 23:45"
    row_96_str = "01/01/2018 00:00"

    # Naive parse without midnight closing convention:
    naive_dt_95 = datetime.strptime(row_95_str, "%d/%m/%Y %H:%M")  # 2018-01-01 23:45:00
    naive_dt_96 = datetime.strptime(row_96_str, "%d/%m/%Y %H:%M")  # 2018-01-01 00:00:00

    # Naively, row 96 is smaller than row 95 (backwards in time!)
    assert naive_dt_96 < naive_dt_95

    # But with the approved logical timestamp conversion:
    logical_dt_95 = parse_logical_timestamp(row_95_str)
    logical_dt_96 = parse_logical_timestamp(row_96_str)

    # The physical chronological order is strictly preserved:
    assert logical_dt_95 < logical_dt_96
    assert (logical_dt_96 - logical_dt_95) == timedelta(minutes=15)
