"""Chronological evaluation harness executing TimeSeriesSplit protocol on Model C."""

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from forgecast.evaluation.baselines import PersistenceBaseline, SeasonalPersistenceBaseline
from forgecast.evaluation.metrics import calculate_metrics, calculate_relative_improvement
from forgecast.features.generator import FEATURE_NAMES
from forgecast.models.config import FrozenModelConfig
from forgecast.models.forecaster import EnergyForecaster
from forgecast.models.persistence import create_model_metadata, save_model_artifact


def compute_file_sha256(filepath: Path | str) -> str:
    """Compute SHA256 hex digest of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest().upper()


@dataclass(frozen=True)
class FoldResult:
    """Evaluation metrics for a single chronological fold."""

    fold_index: int
    train_sample_count: int
    eval_sample_count: int
    train_start_origin: str
    train_end_origin: str
    train_start_target: str
    train_end_target: str
    eval_start_origin: str
    eval_end_origin: str
    eval_start_target: str
    eval_end_target: str
    ml_mae: float
    ml_rmse: float
    persistence_mae: float
    persistence_rmse: float
    seasonal_naive_mae: float
    seasonal_naive_rmse: float
    ml_improvement_vs_persistence_pct: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ChronologicalEvaluationHarness:
    """Executes chronological evaluation using expanding-window TimeSeriesSplit without lookahead."""

    def __init__(
        self,
        n_splits: int = 3,
        test_size: int = 3504,
        gap: int = 0,
        model_config: FrozenModelConfig | None = None,
    ) -> None:
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.model_config = model_config or FrozenModelConfig()

    def run(
        self,
        df_supervised: pd.DataFrame,
        dataset_path: Path | str = "data/raw/Steel_industry_data.csv",
        artifacts_dir: Path | str = "artifacts",
        save_replay_model: bool = True,
    ) -> dict[str, Any]:
        """Run the complete chronological evaluation across all folds.

        Args:
            df_supervised: DataFrame containing origin_timestamp, target_timestamp,
                           the 19 FEATURE_NAMES, and target_usage_kwh.
            dataset_path: Path to raw source CSV for hash verification.
            artifacts_dir: Base directory for persisting models and evaluation metrics.
            save_replay_model: If True, saves the final fold's trained model as the replay artifact.

        Returns:
            Dictionary containing fold metrics, pooled metrics, and execution metadata.
        """
        EnergyForecaster.validate_features(df_supervised[list(FEATURE_NAMES)])

        dataset_file = Path(dataset_path)
        dataset_hash = compute_file_sha256(dataset_file) if dataset_file.exists() else "UNKNOWN"

        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size, gap=self.gap)

        fold_results: list[FoldResult] = []
        pooled_records: list[dict[str, Any]] = []

        replay_model_path: str | None = None
        replay_metadata_path: str | None = None

        total_samples = len(df_supervised)

        for fold_idx, (train_idx, eval_idx) in enumerate(tscv.split(df_supervised), start=1):
            train_df = df_supervised.iloc[train_idx]
            eval_df = df_supervised.iloc[eval_idx]

            # 1. Chronological integrity assertions
            max_train_target = train_df["target_timestamp"].max()
            min_eval_target = eval_df["target_timestamp"].min()
            if max_train_target >= min_eval_target:
                raise ValueError(
                    f"Temporal leakage in fold {fold_idx}: max train target {max_train_target} >= min eval target {min_eval_target}"
                )

            max_train_origin = train_df["origin_timestamp"].max()
            min_eval_origin = eval_df["origin_timestamp"].min()
            if max_train_origin >= min_eval_origin:
                raise ValueError(
                    f"Temporal leakage in fold {fold_idx}: max train origin {max_train_origin} >= min eval origin {min_eval_origin}"
                )

            # 2. Extract features and targets
            X_train = train_df[list(FEATURE_NAMES)]
            y_train = train_df["target_usage_kwh"].to_numpy()

            X_eval = eval_df[list(FEATURE_NAMES)]
            y_eval = eval_df["target_usage_kwh"].to_numpy()

            # 3. Fit fresh frozen model
            forecaster = EnergyForecaster(config=self.model_config)
            forecaster.fit(X_train, y_train)

            # 4. Predict
            y_pred_ml = forecaster.predict(X_eval)
            y_pred_persist = PersistenceBaseline.predict(X_eval)
            y_pred_seasonal = SeasonalPersistenceBaseline.predict(X_eval)

            # 5. Compute fold metrics
            ml_metrics = calculate_metrics(y_eval, y_pred_ml)
            persist_metrics = calculate_metrics(y_eval, y_pred_persist)
            seasonal_metrics = calculate_metrics(y_eval, y_pred_seasonal)

            ml_improvement = calculate_relative_improvement(persist_metrics["mae"], ml_metrics["mae"])

            fold_res = FoldResult(
                fold_index=fold_idx,
                train_sample_count=len(train_df),
                eval_sample_count=len(eval_df),
                train_start_origin=train_df["origin_timestamp"].min().isoformat(),
                train_end_origin=train_df["origin_timestamp"].max().isoformat(),
                train_start_target=train_df["target_timestamp"].min().isoformat(),
                train_end_target=train_df["target_timestamp"].max().isoformat(),
                eval_start_origin=eval_df["origin_timestamp"].min().isoformat(),
                eval_end_origin=eval_df["origin_timestamp"].max().isoformat(),
                eval_start_target=eval_df["target_timestamp"].min().isoformat(),
                eval_end_target=eval_df["target_timestamp"].max().isoformat(),
                ml_mae=ml_metrics["mae"],
                ml_rmse=ml_metrics["rmse"],
                persistence_mae=persist_metrics["mae"],
                persistence_rmse=persist_metrics["rmse"],
                seasonal_naive_mae=seasonal_metrics["mae"],
                seasonal_naive_rmse=seasonal_metrics["rmse"],
                ml_improvement_vs_persistence_pct=ml_improvement,
            )
            fold_results.append(fold_res)

            # Accumulate predictions for pooled evaluation
            for i, idx in enumerate(eval_df.index):
                pooled_records.append(
                    {
                        "fold": fold_idx,
                        "origin_timestamp": eval_df.at[idx, "origin_timestamp"],
                        "target_timestamp": eval_df.at[idx, "target_timestamp"],
                        "actual_usage_kwh": float(y_eval[i]),
                        "pred_ml_kwh": float(y_pred_ml[i]),
                        "pred_persistence_kwh": float(y_pred_persist[i]),
                        "pred_seasonal_kwh": float(y_pred_seasonal[i]),
                    }
                )

            # 6. Save final fold model as the replay-ready model
            if fold_idx == self.n_splits and save_replay_model:
                models_dir = Path(artifacts_dir) / "models"
                meta = create_model_metadata(
                    forecaster=forecaster,
                    model_version="v1",
                    training_sample_count=len(train_df),
                    training_start_origin=train_df["origin_timestamp"].min(),
                    training_end_origin=train_df["origin_timestamp"].max(),
                    training_start_target=train_df["target_timestamp"].min(),
                    training_end_target=train_df["target_timestamp"].max(),
                    dataset_sha256=dataset_hash,
                )
                m_file, meta_file = save_model_artifact(
                    forecaster=forecaster,
                    metadata=meta,
                    output_dir=models_dir,
                    prefix="forgecast_v1_replay_ready",
                )
                replay_model_path = str(m_file)
                replay_metadata_path = str(meta_file)

        # 7. Pooled aggregate metrics across all evaluation folds
        df_pooled = pd.DataFrame(pooled_records)
        all_y_true = df_pooled["actual_usage_kwh"].to_numpy()
        all_y_ml = df_pooled["pred_ml_kwh"].to_numpy()
        all_y_persist = df_pooled["pred_persistence_kwh"].to_numpy()
        all_y_seasonal = df_pooled["pred_seasonal_kwh"].to_numpy()

        pooled_ml = calculate_metrics(all_y_true, all_y_ml)
        pooled_persist = calculate_metrics(all_y_true, all_y_persist)
        pooled_seasonal = calculate_metrics(all_y_true, all_y_seasonal)
        pooled_rel_imp = calculate_relative_improvement(pooled_persist["mae"], pooled_ml["mae"])

        pooled_results = {
            "pooled_evaluation_sample_count": len(all_y_true),
            "pooled_ml_mae": pooled_ml["mae"],
            "pooled_ml_rmse": pooled_ml["rmse"],
            "pooled_persistence_mae": pooled_persist["mae"],
            "pooled_persistence_rmse": pooled_persist["rmse"],
            "pooled_seasonal_naive_mae": pooled_seasonal["mae"],
            "pooled_seasonal_naive_rmse": pooled_seasonal["rmse"],
            "pooled_ml_relative_improvement_pct": pooled_rel_imp,
        }

        evaluation_output = {
            "dataset_sha256": dataset_hash,
            "total_samples": total_samples,
            "n_splits": self.n_splits,
            "test_size": self.test_size,
            "model_config": self.model_config.to_dict(),
            "features": list(FEATURE_NAMES),
            "folds": [f.to_dict() for f in fold_results],
            "pooled_results": pooled_results,
            "replay_model_artifact": replay_model_path,
            "replay_metadata_artifact": replay_metadata_path,
        }

        # Save machine-readable evaluation results
        experiments_dir = Path(artifacts_dir) / "experiments"
        experiments_dir.mkdir(parents=True, exist_ok=True)
        eval_json_file = experiments_dir / "chronological_evaluation_v1.json"
        with open(eval_json_file, "w", encoding="utf-8") as f:
            json.dump(evaluation_output, f, indent=2)

        # Also save evaluation predictions for monitoring/diagnostic use
        predictions_csv = experiments_dir / "evaluation_predictions_v1.csv"
        df_pooled.to_csv(predictions_csv, index=False)

        evaluation_output["evaluation_json_artifact"] = str(eval_json_file)
        evaluation_output["evaluation_predictions_csv"] = str(predictions_csv)

        return evaluation_output
