"""Script to run the full chronological evaluation on the pristine dataset."""

import json
from pathlib import Path
import time

from forgecast.evaluation.harness import ChronologicalEvaluationHarness
from forgecast.features.builder import OfflineFeatureDatasetBuilder
from forgecast.replay.engine import TelemetryReplay


def main() -> None:
    print("Starting full chronological evaluation pipeline...")
    start_time = time.perf_counter()

    # 1. Build supervised dataset from replay stream
    print("1. Streaming telemetry and building supervised dataset...")
    replay = TelemetryReplay(csv_path="data/raw/Steel_industry_data.csv", delay_seconds=0.0)
    builder = OfflineFeatureDatasetBuilder()
    df_supervised = builder.build_dataframe(replay)
    print(f"   Supervised dataset built: {len(df_supervised)} samples.")

    # 2. Run chronological evaluation harness
    print("2. Running 3-fold chronological evaluation harness...")
    harness = ChronologicalEvaluationHarness(
        n_splits=3,
        test_size=3504,
        gap=0,
    )
    results = harness.run(
        df_supervised=df_supervised,
        dataset_path="data/raw/Steel_industry_data.csv",
        artifacts_dir="artifacts",
        save_replay_model=True,
    )

    elapsed = time.perf_counter() - start_time
    print(f"3. Evaluation completed in {elapsed:.2f}s.")
    print("\n--- FOLD RESULTS ---")
    for f in results["folds"]:
        print(
            f"Fold {f['fold_index']}: Train={f['train_sample_count']}, Test={f['eval_sample_count']} | "
            f"ML MAE={f['ml_mae']:.4f} (RMSE={f['ml_rmse']:.4f}) | "
            f"Persist MAE={f['persistence_mae']:.4f} | "
            f"Seasonal MAE={f['seasonal_naive_mae']:.4f} | "
            f"Improvement={f['ml_improvement_vs_persistence_pct']:.2f}%"
        )

    p = results["pooled_results"]
    print("\n--- POOLED RESULTS ---")
    print(f"Pooled Evaluation Samples: {p['pooled_evaluation_sample_count']}")
    print(f"Pooled ML MAE:             {p['pooled_ml_mae']:.4f}")
    print(f"Pooled ML RMSE:            {p['pooled_ml_rmse']:.4f}")
    print(f"Pooled Persistence MAE:    {p['pooled_persistence_mae']:.4f}")
    print(f"Pooled Persistence RMSE:   {p['pooled_persistence_rmse']:.4f}")
    print(f"Pooled Seasonal Naive MAE: {p['pooled_seasonal_naive_mae']:.4f}")
    print(f"Pooled Seasonal RMSE:      {p['pooled_seasonal_naive_rmse']:.4f}")
    print(f"Pooled ML Rel Improvement: {p['pooled_ml_relative_improvement_pct']:.2f}%")

    print("\n--- ARTIFACTS ---")
    print(f"Replay Model:    {results['replay_model_artifact']}")
    print(f"Replay Metadata: {results['replay_metadata_artifact']}")
    print(f"Evaluation JSON: {results['evaluation_json_artifact']}")
    print(f"Predictions CSV: {results['evaluation_predictions_csv']}")


if __name__ == "__main__":
    main()
