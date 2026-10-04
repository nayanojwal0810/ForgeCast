import os
import sys
from pathlib import Path

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

os.environ["OMP_NUM_THREADS"] = "1"

import time

from forgecast.monitoring.metrics import BaselineComparisonTracker
from forgecast.operational.engine import OperationalEngine
from forgecast.replay.engine import TelemetryReplay


def main() -> None:
    print("Starting operational replay smoke test on full 35,040-row dataset...", flush=True)
    start_time = time.perf_counter()

    tracker = BaselineComparisonTracker(window_sizes=(96, 672))
    replay = TelemetryReplay(csv_path="data/raw/Steel_industry_data.csv", delay_seconds=0.0)
    engine = OperationalEngine(
        model_path="artifacts/models/forgecast_v1_replay_ready.pkl",
        monitoring_tracker=tracker,
    )

    predictions_generated = 0
    completed_feedback_records = 0
    first_prediction = None
    last_prediction = None
    first_feedback = None
    last_feedback = None

    warmup_events = 0
    unmatched_feedback_events = 0

    for step in engine.run_replay(replay):
        if engine.records_consumed % 5000 == 0:
            print(f"Processed {engine.records_consumed} records ({predictions_generated} predictions)...", flush=True)

        if step.is_warmup:
            warmup_events += 1

        if step.unmatched_feedback:
            unmatched_feedback_events += 1

        if step.prediction is not None:
            predictions_generated += 1
            if first_prediction is None:
                first_prediction = step.prediction
            last_prediction = step.prediction

        if step.feedback is not None:
            completed_feedback_records += 1
            if first_feedback is None:
                first_feedback = step.feedback
            last_feedback = step.feedback

    elapsed = time.perf_counter() - start_time
    print(f"Operational smoke test completed in {elapsed:.2f}s.\n")

    summary = tracker.get_comparison_summary(engine.model_version)
    full_cum = summary.full_replay_cumulative
    post_cum = summary.post_deployment_cumulative
    r24 = summary.rolling_24h
    r7d = summary.rolling_7d

    print("--- FULL REPLAY ---")
    print(f"Telemetry records consumed:         {engine.records_consumed}")
    print(f"Predictions generated:              {predictions_generated}")
    print(f"Completed feedback:                 {completed_feedback_records}")
    print(f"Pending predictions remaining:      {engine.feedback_tracker.pending_count}")
    print(f"Gap incidents:                      {engine.gap_incident_count}")
    print(f"Invalidated predictions:            {engine.invalidated_prediction_count}")
    print(f"Sequence failures:                  {engine.sequence_failure_count}")
    print(f"Full-replay cumulative ML MAE:      {full_cum.ml_mae:.4f}")
    print(f"Full-replay cumulative persist MAE: {full_cum.baseline_mae:.4f}")
    print(f"Full-replay ML MAE improvement %:   {full_cum.mae_improvement_pct:.2f}%")
    print(f"Full-replay ML win rate:            {full_cum.ml_win_rate * 100:.2f}%\n")

    print("--- DEPLOYMENT BOUNDARY ---")
    print(f"Model version:                      {summary.model_version}")
    print(f"Deployment boundary:                {summary.deployment_boundary.isoformat() if summary.deployment_boundary else 'None'}")
    if post_cum and post_cum.sample_count > 0:
        print(f"Post-deployment first eligible fb:  {post_cum.first_target_timestamp.isoformat() if post_cum.first_target_timestamp else 'None'}")
        print(f"Post-deployment feedback count:     {post_cum.sample_count}\n")
    else:
        print("Post-deployment feedback count:     0\n")

    if post_cum and post_cum.sample_count > 0:
        print("--- POST-DEPLOYMENT PERFORMANCE ---")
        print(f"ML MAE:                             {post_cum.ml_mae:.4f}")
        print(f"Persistence MAE:                    {post_cum.baseline_mae:.4f}")
        print(f"MAE improvement %:                  {post_cum.mae_improvement_pct:.2f}%")
        print(f"ML RMSE:                            {post_cum.ml_rmse:.4f}")
        print(f"Persistence RMSE:                   {post_cum.baseline_rmse:.4f}")
        print(f"RMSE improvement %:                 {post_cum.rmse_improvement_pct:.2f}%")
        print(f"ML win rate:                        {post_cum.ml_win_rate * 100:.2f}%\n")

    if r24 is not None:
        diff_24 = r24.ml_mae - r24.baseline_mae if (r24.ml_mae is not None and r24.baseline_mae is not None) else 0.0
        print("--- RECENT 24H ---")
        print(f"Sample count:                       {r24.sample_count}")
        print(f"ML MAE:                             {r24.ml_mae:.4f}")
        print(f"Persistence MAE:                    {r24.baseline_mae:.4f}")
        print(f"MAE difference:                     {diff_24:+.4f}")
        print(f"ML win rate:                        {r24.ml_win_rate * 100:.2f}%\n")

    if r7d is not None:
        diff_7d = r7d.ml_mae - r7d.baseline_mae if (r7d.ml_mae is not None and r7d.baseline_mae is not None) else 0.0
        print("--- RECENT 7D ---")
        print(f"Sample count:                       {r7d.sample_count}")
        print(f"ML MAE:                             {r7d.ml_mae:.4f}")
        print(f"Persistence MAE:                    {r7d.baseline_mae:.4f}")
        print(f"MAE difference:                     {diff_7d:+.4f}")
        print(f"ML win rate:                        {r7d.ml_win_rate * 100:.2f}%\n")

    # Diagnostic
    diag_24 = tracker.diagnose_window(engine.model_version, 96)
    diag_7d = tracker.diagnose_window(engine.model_version, 672)

    print("--- RECENT PERFORMANCE DIAGNOSTIC ---")
    print(f"24h load std:                       {diag_24.usage_std:.4f} kWh (range: [{diag_24.usage_min:.2f}, {diag_24.usage_max:.2f}])")
    print(f"7d load std:                        {diag_7d.usage_std:.4f} kWh (range: [{diag_7d.usage_min:.2f}, {diag_7d.usage_max:.2f}])")
    print(f"24h top 10% error fraction:         {diag_24.top_10pct_error_fraction * 100:.2f}% of total ML error")
    print(f"24h diagnostic finding:             {diag_24.error_concentration_summary}\n")

    print("--- METRIC RECONCILIATION ---")
    print(f"Full replay win rate:               {full_cum.ml_win_rate * 100:.2f}%")
    if post_cum and post_cum.sample_count > 0 and post_cum.ml_win_rate is not None:
        print(f"Post-deployment win rate:           {post_cum.ml_win_rate * 100:.2f}%")
    if r24 and r24.ml_win_rate is not None:
        print(f"24h win rate:                       {r24.ml_win_rate * 100:.2f}%")
    if r7d and r7d.ml_win_rate is not None:
        print(f"7d win rate:                        {r7d.ml_win_rate * 100:.2f}%\n")

    print(f"24h top-10% error fraction:         {diag_24.top_10pct_error_fraction * 100:.2f}%\n")

    print(f"24h mean prediction:                {diag_24.mean_prediction:.4f} kWh")
    print(f"24h mean actual:                    {diag_24.mean_actual:.4f} kWh")
    print(f"24h mean signed error:              {diag_24.mean_signed_error:+.4f} kWh")


if __name__ == "__main__":
    main()
