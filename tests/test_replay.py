"""Tests for the deterministic historical telemetry replay engine."""

from datetime import datetime, timedelta
import time
from pathlib import Path
import pytest

from forgecast.replay.engine import TelemetryReplay

DATASET_PATH = Path("data/raw/Steel_industry_data.csv")


def test_replay_file_not_found() -> None:
    """Confirm requesting a non-existent CSV path raises FileNotFoundError."""
    replay = TelemetryReplay(csv_path="data/raw/non_existent.csv")
    with pytest.raises(FileNotFoundError):
        next(replay.run())


def test_replay_negative_delay_rejected() -> None:
    """Confirm negative demo delay raises ValueError."""
    with pytest.raises(ValueError, match="cannot be negative"):
        TelemetryReplay(delay_seconds=-0.5)


def test_replay_slice_order_and_cadence() -> None:
    """Confirm replay emits records in sequential order with strictly 15-minute cadence."""
    replay = TelemetryReplay(csv_path=DATASET_PATH, limit=192)  # exactly 2 days (192 intervals)
    records = list(replay.run())

    assert len(records) == 192
    assert replay.emitted_count == 192

    # Check first record
    assert records[0].row_index == 0
    assert records[0].raw_date == "01/01/2018 00:15"
    assert records[0].logical_timestamp == datetime(2018, 1, 1, 0, 15)

    # Check day 1 boundary (observation 96: 00:00)
    assert records[95].row_index == 95
    assert records[95].raw_date == "01/01/2018 00:00"
    assert records[95].logical_timestamp == datetime(2018, 1, 2, 0, 0)

    # Check day 2 first record (observation 97: 00:15)
    assert records[96].row_index == 96
    assert records[96].raw_date == "02/01/2018 00:15"
    assert records[96].logical_timestamp == datetime(2018, 1, 2, 0, 15)

    # Check day 2 boundary (observation 192: 00:00)
    assert records[191].row_index == 191
    assert records[191].raw_date == "02/01/2018 00:00"
    assert records[191].logical_timestamp == datetime(2018, 1, 3, 0, 0)

    # Verify all consecutive pairs are strictly 15 minutes apart
    for i in range(1, len(records)):
        delta = records[i].logical_timestamp - records[i - 1].logical_timestamp
        assert delta == timedelta(minutes=15)
        assert records[i].row_index == i


def test_replay_full_dataset_verification() -> None:
    """Full smoke validation: sequentially replay all 35,040 rows from the raw CSV.

    Confirms:
    1. Total row count matches 35,040
    2. First logical timestamp is 2018-01-01 00:15:00
    3. Final logical timestamp is 2019-01-01 00:00:00
    4. Complete temporal continuity across all 35,040 records (zero gaps or duplicates)
    """
    replay = TelemetryReplay(csv_path=DATASET_PATH, delay_seconds=0.0)

    first_record = None
    last_record = None
    count = 0

    for rec in replay.run():
        if first_record is None:
            first_record = rec
        last_record = rec
        count += 1

    assert count == 35040
    assert replay.emitted_count == 35040
    assert first_record is not None
    assert first_record.logical_timestamp == datetime(2018, 1, 1, 0, 15)
    assert first_record.raw_date == "01/01/2018 00:15"
    assert last_record is not None
    assert last_record.logical_timestamp == datetime(2019, 1, 1, 0, 0)
    assert last_record.raw_date == "31/12/2018 00:00"


def test_replay_demo_mode_delay_configurable() -> None:
    """Confirm demo delay configures wall-clock pause between intervals."""
    demo_delay = 0.02  # 20ms per record
    limit = 5

    replay = TelemetryReplay(csv_path=DATASET_PATH, delay_seconds=demo_delay, limit=limit)
    start_time = time.perf_counter()
    records = list(replay.run())
    elapsed = time.perf_counter() - start_time

    assert len(records) == limit
    # For N items with sleep between emissions, expected sleep count is (N - 1)
    min_expected_sleep = (limit - 1) * demo_delay
    assert elapsed >= min_expected_sleep
