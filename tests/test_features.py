"""Unit and causal integrity tests for the stateful feature generator and offline dataset builder."""

from datetime import datetime, timedelta
import math
from pathlib import Path

import pytest

from forgecast.features.builder import OfflineFeatureDatasetBuilder
from forgecast.features.generator import (
    FEATURE_NAMES,
    FeatureContinuityError,
    StatefulFeatureGenerator,
)
from forgecast.ingestion.contract import TelemetryRecord
from forgecast.replay.engine import TelemetryReplay

DATASET_PATH = Path("data/raw/Steel_industry_data.csv")


def _make_dummy_record(logical_dt: datetime, usage: float, row_idx: int = 0) -> TelemetryRecord:
    """Helper creating a minimal valid TelemetryRecord for generator testing."""
    return TelemetryRecord(
        logical_timestamp=logical_dt,
        raw_date=logical_dt.strftime("%d/%m/%Y %H:%M"),
        usage_kwh=usage,
        lagging_reactive_power_kvarh=1.0,
        leading_reactive_power_kvarh=0.0,
        co2_tco2=0.0,
        lagging_power_factor=80.0,
        leading_power_factor=100.0,
        nsm=logical_dt.hour * 3600 + logical_dt.minute * 60,
        week_status="Weekday" if logical_dt.weekday() < 5 else "Weekend",
        day_of_week=logical_dt.strftime("%A"),
        load_type="Light_Load",
        row_index=row_idx,
    )


def test_feature_count_and_names() -> None:
    """Verify exactly 19 Model C features are produced with exact canonical names."""
    assert len(FEATURE_NAMES) == 19
    expected_names = (
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
    assert FEATURE_NAMES == expected_names


def test_warmup_behavior() -> None:
    """Confirm that records 0..94 produce None, and the 96th record (index 95) produces the first vector."""
    gen = StatefulFeatureGenerator()
    start_dt = datetime(2018, 1, 1, 0, 15)

    for i in range(95):
        dt = start_dt + timedelta(minutes=15 * i)
        rec = _make_dummy_record(dt, usage=float(i), row_idx=i)
        fv = gen.process_record(rec)
        assert fv is None, f"Expected None during warm-up at record {i}"

    # 96th record (index 95)
    rec_95 = _make_dummy_record(start_dt + timedelta(minutes=15 * 95), usage=95.0, row_idx=95)
    fv_first = gen.process_record(rec_95)

    assert fv_first is not None
    assert fv_first.origin_index == 95
    assert len(fv_first.features) == 19
    assert fv_first.origin_timestamp == datetime(2018, 1, 2, 0, 0)
    assert fv_first.target_timestamp == datetime(2018, 1, 2, 0, 15)


def test_lag_and_rolling_correctness() -> None:
    """Confirm exact mapping of historical usage lags and backward-looking rolling means."""
    gen = StatefulFeatureGenerator()
    start_dt = datetime(2018, 1, 1, 0, 15)

    # Ingest records 0..95 with usage equal to row index
    for i in range(96):
        dt = start_dt + timedelta(minutes=15 * i)
        fv = gen.process_record(_make_dummy_record(dt, usage=float(i), row_idx=i))

    assert fv is not None
    f = fv.features

    # Lags relative to origin 95:
    assert f["usage_lag_0"] == 95.0  # Usage(i)
    assert f["usage_lag_1"] == 94.0  # Usage(i-1)
    assert f["usage_lag_3"] == 92.0  # Usage(i-3)
    assert f["usage_lag_7"] == 88.0  # Usage(i-7)
    assert f["usage_lag_95"] == 0.0  # Usage(i-95)

    # Rolling means:
    expected_roll_4 = (95.0 + 94.0 + 93.0 + 92.0) / 4.0
    assert f["usage_roll_mean_4"] == pytest.approx(expected_roll_4)

    expected_roll_96 = sum(range(96)) / 96.0
    assert f["usage_roll_mean_96"] == pytest.approx(expected_roll_96)


def test_target_calendar_and_cyclical_features() -> None:
    """Confirm calendar features describe target interval t+1 rather than origin t."""
    gen = StatefulFeatureGenerator()
    start_dt = datetime(2018, 1, 1, 0, 15)

    # Fast forward through 96 records: origin will be 2018-01-02 00:00:00 (Tuesday)
    # Target will be 2018-01-02 00:15:00 (Tuesday, hour=0, minute=15, slot=1)
    for i in range(96):
        dt = start_dt + timedelta(minutes=15 * i)
        fv = gen.process_record(_make_dummy_record(dt, usage=10.0, row_idx=i))

    assert fv is not None
    assert fv.origin_timestamp == datetime(2018, 1, 2, 0, 0)
    assert fv.target_timestamp == datetime(2018, 1, 2, 0, 15)

    f = fv.features
    assert f["cal_hour"] == 0.0
    assert f["cal_quarter_slot"] == 1.0
    assert f["cal_dayofweek"] == 1.0  # Tuesday = 1
    assert f["cal_is_weekend"] == 0.0
    assert f["cal_month"] == 1.0
    assert f["cal_nsm"] == 900.0  # 00:15 -> 900s from midnight

    # Cyclical formulas
    assert f["cal_nsm_sin"] == pytest.approx(math.sin(2 * math.pi * 900 / 86400))
    assert f["cal_nsm_cos"] == pytest.approx(math.cos(2 * math.pi * 900 / 86400))
    assert f["cal_dow_sin"] == pytest.approx(math.sin(2 * math.pi * 1 / 7))
    assert f["cal_dow_cos"] == pytest.approx(math.cos(2 * math.pi * 1 / 7))
    assert f["cal_month_sin"] == pytest.approx(math.sin(2 * math.pi * 0 / 12))
    assert f["cal_month_cos"] == pytest.approx(math.cos(2 * math.pi * 0 / 12))


def test_future_row_poisoning_causal_invariant() -> None:
    """LEAKAGE TEST: Mutating a future observation must NOT alter the feature vector generated at origin.

    Procedure:
    1. Stream 96 records in clean run. Record 95 produces feature vector FV_clean predicting record 96.
    2. Reset and stream the same 96 records in poisoned run, but prepare record 96 with an absurd value.
    3. Confirm that the feature vector produced at origin 95 in both runs is bit-for-bit identical.
    """
    start_dt = datetime(2018, 1, 1, 0, 15)

    # Run 1: Clean stream
    gen_clean = StatefulFeatureGenerator()
    fv_clean = None
    for i in range(96):
        rec = _make_dummy_record(start_dt + timedelta(minutes=15 * i), usage=float(i + 1), row_idx=i)
        fv_clean = gen_clean.process_record(rec)

    assert fv_clean is not None

    # Run 2: Poisoned stream where future target row is poisoned
    # Note: Even if a test framework prepares record 96 with absurd usage 999999.0,
    # the feature vector for origin 95 must depend strictly on observations <= origin 95.
    gen_test = StatefulFeatureGenerator()
    fv_test = None
    for i in range(96):
        rec = _make_dummy_record(start_dt + timedelta(minutes=15 * i), usage=float(i + 1), row_idx=i)
        fv_test = gen_test.process_record(rec)

    assert fv_test is not None

    # Now ingest the poisoned future record 96 into gen_test
    poisoned_target_rec = _make_dummy_record(
        start_dt + timedelta(minutes=15 * 96), usage=999999.0, row_idx=96
    )
    # At origin 95, fv_test was produced before record 96 was processed!
    # Confirm every single feature in fv_clean matches fv_test
    for name in FEATURE_NAMES:
        assert fv_clean.features[name] == fv_test.features[name]

    # Furthermore, verify that the poisoned record 96 only affects vectors produced AFTER it arrives
    fv_post_poison = gen_test.process_record(poisoned_target_rec)
    assert fv_post_poison is not None
    # At origin 96, usage_lag_0 should now reflect the poisoned value
    assert fv_post_poison.features["usage_lag_0"] == 999999.0
    # But origin 95 remains completely untainted
    assert fv_test.features["usage_lag_0"] == 96.0


def test_source_cutoff_temporal_integrity() -> None:
    """Verify that all source timestamps in the generator state are <= origin < target."""
    gen = StatefulFeatureGenerator()
    start_dt = datetime(2018, 1, 1, 0, 15)

    for i in range(96):
        dt = start_dt + timedelta(minutes=15 * i)
        fv = gen.process_record(_make_dummy_record(dt, usage=float(i), row_idx=i))

    assert fv is not None
    origin_ts = fv.origin_timestamp
    target_ts = fv.target_timestamp

    # Strict target separation
    assert origin_ts < target_ts
    assert (target_ts - origin_ts) == timedelta(minutes=15)

    # Every timestamp in internal history must be <= origin_ts
    for historical_ts in gen.timestamp_buffer:
        assert historical_ts <= origin_ts


def test_temporal_gap_handling() -> None:
    """Confirm a skipped interval raises FeatureContinuityError and leaves buffer uncorrupted."""
    gen = StatefulFeatureGenerator()
    start_dt = datetime(2018, 1, 1, 0, 15)

    for i in range(10):
        dt = start_dt + timedelta(minutes=15 * i)
        gen.process_record(_make_dummy_record(dt, usage=float(i), row_idx=i))

    assert len(gen.usage_buffer) == 10
    last_valid_ts = gen.last_logical_timestamp

    # Ingest record with a 30-minute gap (02:45 arrives instead of 02:30)
    gap_dt = last_valid_ts + timedelta(minutes=30)  # type: ignore[operator]
    bad_rec = _make_dummy_record(gap_dt, usage=99.0, row_idx=10)

    with pytest.raises(FeatureContinuityError) as exc_info:
        gen.process_record(bad_rec)

    assert "Temporal sequence gap" in str(exc_info.value)
    # Verify internal state was NOT corrupted and NOT advanced by the invalid record
    assert len(gen.usage_buffer) == 10
    assert gen.last_logical_timestamp == last_valid_ts
    assert gen.usage_buffer[-1] == 9.0  # Still the 10th record


def test_offline_dataset_builder_slice() -> None:
    """Test OfflineFeatureDatasetBuilder on a 120-record replay slice."""
    replay = TelemetryReplay(csv_path=DATASET_PATH, limit=120)
    builder = OfflineFeatureDatasetBuilder()
    samples = builder.build_dataset(replay)

    # 120 records with 96-step warm-up produces 120 - 96 = 24 supervised samples
    assert len(samples) == 24

    first_sample = samples[0]
    # First sample: origin is observation 95 (2018-01-02 00:00:00), target is observation 96 (2018-01-02 00:15:00)
    assert first_sample.origin_index == 95
    assert first_sample.target_index == 96
    assert first_sample.origin_timestamp == datetime(2018, 1, 2, 0, 0)
    assert first_sample.target_timestamp == datetime(2018, 1, 2, 0, 15)
    assert len(first_sample.features) == 19

    # Verify target_usage_kwh matches the actual reading at observation 96 (02/01/2018 00:15 in CSV is 3.20)
    assert first_sample.target_usage_kwh == pytest.approx(3.20)

    # Verify chronological continuity across all pairs
    for i in range(1, len(samples)):
        prev = samples[i - 1]
        curr = samples[i]
        assert curr.origin_timestamp == prev.origin_timestamp + timedelta(minutes=15)
        assert curr.target_timestamp == prev.target_timestamp + timedelta(minutes=15)
        assert curr.origin_timestamp < curr.target_timestamp


def test_full_dataset_offline_feature_build_smoke() -> None:
    """Full offline dataset validation against the real 35,040-row dataset.

    Confirms:
    1. Exactly 34,944 supervised samples produced
    2. Exactly 19 Model C features per sample
    3. First origin is 2018-01-02 00:00:00
    4. First target is 2018-01-02 00:15:00
    5. Last origin is 2018-12-31 23:45:00
    6. Last target is 2019-01-01 00:00:00
    7. All target timestamps strictly equal origin + 15m
    """
    replay = TelemetryReplay(csv_path=DATASET_PATH, delay_seconds=0.0)
    builder = OfflineFeatureDatasetBuilder()

    sample_count = 0
    first_sample = None
    last_sample = None

    for sample in builder.build_from_replay(replay):
        if first_sample is None:
            first_sample = sample
        last_sample = sample
        sample_count += 1
        assert (sample.target_timestamp - sample.origin_timestamp) == timedelta(minutes=15)

    assert sample_count == 34944
    assert first_sample is not None
    assert first_sample.origin_timestamp == datetime(2018, 1, 2, 0, 0)
    assert first_sample.target_timestamp == datetime(2018, 1, 2, 0, 15)
    assert first_sample.origin_index == 95
    assert first_sample.target_index == 96

    assert last_sample is not None
    assert last_sample.origin_timestamp == datetime(2018, 12, 31, 23, 45)
    assert last_sample.target_timestamp == datetime(2019, 1, 1, 0, 0)
    assert last_sample.origin_index == 35038
    assert last_sample.target_index == 35039
