"""Fault-injection smoke test for operational missingness and cascade handling."""

import os
import sys
from pathlib import Path

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

os.environ["OMP_NUM_THREADS"] = "1"

from datetime import datetime, timedelta

from forgecast.ingestion.contract import TelemetryRecord
from forgecast.monitoring.metrics import BaselineComparisonTracker
from forgecast.operational.engine import OperationalEngine
from forgecast.operational.events import OperationalContinuityError
from forgecast.replay.engine import TelemetryReplay


def load_telemetry_slice(n_records: int = 250) -> list[TelemetryRecord]:
    """Read a bounded slice of raw telemetry records in memory without altering raw CSV."""
    replay = TelemetryReplay(csv_path="data/raw/Steel_industry_data.csv", delay_seconds=0.0)
    records: list[TelemetryRecord] = []
    for rec in replay.run():
        records.append(rec)
        if len(records) >= n_records:
            break
    return records


def run_scenario(
    scenario_name: str,
    all_records: list[TelemetryRecord],
    drop_indices: set[int],
) -> None:
    """Execute operational engine on a sequence with injected missingness."""
    tracker = BaselineComparisonTracker(window_sizes=(96, 672))
    engine = OperationalEngine(
        model_path="artifacts/models/forgecast_v1_replay_ready.pkl",
        monitoring_tracker=tracker,
        enable_gap_recovery=True,
    )

    duplicate_failures = 0
    out_of_order_failures = 0
    predictions_generated = 0
    completed_feedback = 0
    predictions_resumed_count = 0
    post_gap_predictions = 0

    first_gap = None

    for idx, rec in enumerate(all_records):
        if idx in drop_indices:
            # Skip record to simulate missing telemetry interval
            continue

        step_res = engine.process_telemetry(rec)

        if step_res.gap_incident is not None and first_gap is None:
            first_gap = step_res.gap_incident

        if step_res.prediction is not None:
            predictions_generated += 1
            if first_gap is not None:
                post_gap_predictions += 1

        if step_res.feedback is not None:
            completed_feedback += 1

    # Verify duplicate failure fail-closed invariant
    if engine.last_logical_timestamp is not None:
        dup_rec = TelemetryRecord(
            logical_timestamp=engine.last_logical_timestamp,
            raw_date=engine.last_logical_timestamp.strftime("%d/%m/%Y %H:%M"),
            usage_kwh=10.0,
            lagging_reactive_power_kvarh=1.0,
            leading_reactive_power_kvarh=0.0,
            co2_tco2=0.0,
            lagging_power_factor=80.0,
            leading_power_factor=100.0,
            nsm=0,
            week_status="Weekday",
            day_of_week="Monday",
            load_type="Light_Load",
            row_index=999999,
        )
        try:
            engine.process_telemetry(dup_rec)
        except OperationalContinuityError:
            duplicate_failures += 1

        # Verify out-of-order failure fail-closed invariant
        ooo_ts = engine.last_logical_timestamp - timedelta(minutes=15)
        ooo_rec = TelemetryRecord(
            logical_timestamp=ooo_ts,
            raw_date=ooo_ts.strftime("%d/%m/%Y %H:%M"),
            usage_kwh=10.0,
            lagging_reactive_power_kvarh=1.0,
            leading_reactive_power_kvarh=0.0,
            co2_tco2=0.0,
            lagging_power_factor=80.0,
            leading_power_factor=100.0,
            nsm=0,
            week_status="Weekday",
            day_of_week="Monday",
            load_type="Light_Load",
            row_index=999998,
        )
        try:
            engine.process_telemetry(ooo_rec)
        except OperationalContinuityError:
            out_of_order_failures += 1

    print(f"Scenario: {scenario_name}")
    if first_gap is not None:
        print(f"Previous timestamp:        {first_gap.previous_timestamp.isoformat()}")
        print(f"Current timestamp:         {first_gap.current_timestamp.isoformat()}")
        print(f"Missing interval count:    {first_gap.missing_interval_count}")
        print(f"First missing timestamp:   {first_gap.first_missing_timestamp.isoformat()}")
        print(f"Last missing timestamp:    {first_gap.last_missing_timestamp.isoformat()}")
        print()
        print(f"Gap incidents:             {engine.gap_incident_count}")
        print(f"Invalidated predictions:   {engine.invalidated_prediction_count}")
        print(f"Completed feedback:        {completed_feedback}")
        print()
        print(f"State reset:               {first_gap.state_reset}")
        print(f"Rewarm required:           {first_gap.rewarm_required}")
        print(f"Predictions resumed:       {post_gap_predictions > 0} ({post_gap_predictions} emitted)")
        print()
        print(f"Duplicate failures:        {duplicate_failures}")
        print(f"Out-of-order failures:     {out_of_order_failures}")
    print()


def main() -> None:
    print("--- MISSINGNESS SMOKE TEST ---\n")
    records = load_telemetry_slice(250)

    # Scenario A: single missing interval at index 110
    run_scenario(
        scenario_name="single missing interval",
        all_records=records,
        drop_indices={110},
    )

    # Scenario B: multi-interval consecutive gap at indices 110, 111, 112 (3 missing intervals)
    run_scenario(
        scenario_name="consecutive missing intervals (3 intervals)",
        all_records=records,
        drop_indices={110, 111, 112},
    )


if __name__ == "__main__":
    main()
