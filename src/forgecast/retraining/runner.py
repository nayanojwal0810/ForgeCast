"""Deterministic quarterly retraining workflow and candidate promotion runner."""

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import pickle
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from forgecast.evaluation.baselines import PersistenceBaseline
from forgecast.evaluation.harness import compute_file_sha256
from forgecast.evaluation.metrics import calculate_metrics, calculate_relative_improvement
from forgecast.features.builder import OfflineFeatureDatasetBuilder
from forgecast.features.generator import FEATURE_NAMES
from forgecast.models.config import FrozenModelConfig
from forgecast.models.forecaster import EnergyForecaster
from forgecast.models.persistence import load_model_artifact
from forgecast.replay.engine import TelemetryReplay
from forgecast.retraining.config import (
    PromotionDecision,
    PromotionGateConfig,
    RetrainingCheckpoint,
)


class RetrainingRunner:
    """Coordinates leakage-safe training, chronological validation, and promotion evaluation."""

    def __init__(
        self,
        model_config: FrozenModelConfig | None = None,
        dataset_path: Path | str = "data/raw/Steel_industry_data.csv",
        artifacts_dir: Path | str = "artifacts",
    ) -> None:
        self.model_config = model_config or FrozenModelConfig()
        self.dataset_path = Path(dataset_path)
        self.artifacts_dir = Path(artifacts_dir)
        self._cached_df_supervised: pd.DataFrame | None = None

    def load_data(self, force_reload: bool = False) -> pd.DataFrame:
        """Construct or return cached supervised feature dataset from the raw telemetry stream."""
        if self._cached_df_supervised is not None and not force_reload:
            return self._cached_df_supervised

        replay = TelemetryReplay(csv_path=self.dataset_path, delay_seconds=0.0)
        builder = OfflineFeatureDatasetBuilder()
        df = builder.build_dataframe(replay)
        self._cached_df_supervised = df
        return df

    @staticmethod
    def validate_checkpoint(
        checkpoint: RetrainingCheckpoint, df_supervised: pd.DataFrame
    ) -> None:
        """Validate chronological integrity, boundary ordering, and data sufficiency using target timestamps."""
        if checkpoint.training_cutoff >= checkpoint.eval_start:
            raise ValueError(
                f"Temporal violation: training target cutoff ({checkpoint.training_cutoff.isoformat()}) "
                f"must be strictly earlier than evaluation target start ({checkpoint.eval_start.isoformat()})"
            )

        if checkpoint.eval_start > checkpoint.eval_end:
            raise ValueError(
                f"Temporal violation: evaluation target start ({checkpoint.eval_start.isoformat()}) "
                f"must be earlier than or equal to evaluation target end ({checkpoint.eval_end.isoformat()})"
            )

        min_target = df_supervised["target_timestamp"].min()

        if checkpoint.training_cutoff < min_target:
            raise ValueError(
                f"Training target cutoff ({checkpoint.training_cutoff.isoformat()}) precedes first available "
                f"target observation ({min_target.isoformat()})"
            )

        train_mask = df_supervised["target_timestamp"] <= checkpoint.training_cutoff
        train_count = int(train_mask.sum())
        if train_count < 96:
            raise ValueError(
                f"Insufficient training observations: got {train_count}, minimum required is 96"
            )

        val_mask = (df_supervised["target_timestamp"] >= checkpoint.eval_start) & (
            df_supervised["target_timestamp"] <= checkpoint.eval_end
        )
        val_count = int(val_mask.sum())
        if val_count == 0:
            raise ValueError(
                f"Evaluation target window [{checkpoint.eval_start.isoformat()} to {checkpoint.eval_end.isoformat()}] "
                "contains zero observations in dataset"
            )

    @staticmethod
    def prepare_splits(
        checkpoint: RetrainingCheckpoint, df_supervised: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split supervised dataset into strictly separated training and validation partitions by target timestamp.

        A sample belongs to training if and only if:
            sample.target_timestamp <= checkpoint.training_cutoff

        A sample belongs to validation if and only if:
            checkpoint.eval_start <= sample.target_timestamp <= checkpoint.eval_end
        """
        df_train = df_supervised[
            df_supervised["target_timestamp"] <= checkpoint.training_cutoff
        ].copy()

        df_val = df_supervised[
            (df_supervised["target_timestamp"] >= checkpoint.eval_start)
            & (df_supervised["target_timestamp"] <= checkpoint.eval_end)
        ].copy()

        # Enforce target-based temporal separation invariants
        max_train_target = df_train["target_timestamp"].max()
        min_val_target = df_val["target_timestamp"].min()
        max_val_target = df_val["target_timestamp"].max()

        if max_train_target >= min_val_target:
            raise ValueError(
                f"Leakage violation: max training target ({max_train_target.isoformat()}) "
                f"is not strictly before min validation target ({min_val_target.isoformat()})"
            )

        if max_train_target > checkpoint.training_cutoff:
            raise ValueError(
                f"Leakage violation: max training target ({max_train_target.isoformat()}) "
                f"exceeds training cutoff ({checkpoint.training_cutoff.isoformat()})"
            )

        if min_val_target < checkpoint.eval_start or max_val_target > checkpoint.eval_end:
            raise ValueError(
                f"Boundary violation: validation targets [{min_val_target.isoformat()}, {max_val_target.isoformat()}] "
                f"fall outside evaluation window [{checkpoint.eval_start.isoformat()}, {checkpoint.eval_end.isoformat()}]"
            )

        overlap = set(df_train["target_timestamp"]).intersection(set(df_val["target_timestamp"]))
        if overlap:
            raise ValueError(
                f"Leakage violation: {len(overlap)} target timestamps overlap between train and validation"
            )

        return df_train, df_val

    def train_candidate(self, df_train: pd.DataFrame) -> EnergyForecaster:
        """Fit candidate model using frozen hyperparameter configuration."""
        forecaster = EnergyForecaster(self.model_config)
        X_train = df_train[list(FEATURE_NAMES)]
        y_train = np.asarray(df_train["target_usage_kwh"], dtype=np.float64)
        forecaster.fit(X_train, y_train)
        return forecaster

    @staticmethod
    def evaluate_candidate(
        forecaster: EnergyForecaster, df_val: pd.DataFrame
    ) -> dict[str, float]:
        """Evaluate candidate forecaster on the chronological validation partition."""
        X_val = df_val[list(FEATURE_NAMES)]
        y_val = np.asarray(df_val["target_usage_kwh"], dtype=np.float64)
        preds = forecaster.predict(X_val)
        return calculate_metrics(y_val, preds)

    @staticmethod
    def evaluate_baseline(df_val: pd.DataFrame) -> dict[str, float]:
        """Evaluate operational persistence baseline on the validation partition."""
        X_val = df_val[list(FEATURE_NAMES)]
        y_val = np.asarray(df_val["target_usage_kwh"], dtype=np.float64)
        preds_persist = PersistenceBaseline.predict(X_val)
        return calculate_metrics(y_val, preds_persist)

    @staticmethod
    def evaluate_reference(
        reference_model_path: Path | str | None,
        checkpoint: RetrainingCheckpoint,
        df_val: pd.DataFrame,
    ) -> tuple[dict[str, float] | None, str | None]:
        """Evaluate reference model if available and logically valid under information constraints.

        Returns:
            Tuple of (metrics_dict or None, status_explanation_string)
        """
        if reference_model_path is None:
            return None, "No reference model provided"

        ref_path = Path(reference_model_path)
        if not ref_path.exists():
            return None, f"Reference model file not found at '{ref_path}'"

        ref_forecaster, ref_metadata = load_model_artifact(ref_path)

        # Verify information constraints using target-time boundaries
        ref_train_end_str = ref_metadata.get("training_end_target") or ref_metadata.get(
            "training_end_origin"
        )
        if ref_train_end_str:
            ref_train_end = datetime.fromisoformat(ref_train_end_str)
            if ref_train_end >= checkpoint.eval_start:
                return (
                    None,
                    f"Reference model training target cutoff ({ref_train_end.isoformat()}) overlaps evaluation "
                    f"target start ({checkpoint.eval_start.isoformat()}); in-sample comparison excluded",
                )

        X_val = df_val[list(FEATURE_NAMES)]
        y_val = np.asarray(df_val["target_usage_kwh"], dtype=np.float64)
        preds_ref = ref_forecaster.predict(X_val)
        metrics = calculate_metrics(y_val, preds_ref)
        return metrics, None

    @staticmethod
    def apply_promotion_gate(
        candidate_metrics: dict[str, float],
        baseline_metrics: dict[str, float],
        reference_metrics: dict[str, float] | None,
        gate_config: PromotionGateConfig,
    ) -> PromotionDecision:
        """Apply objective criteria to determine candidate promotion eligibility."""
        cand_mae = candidate_metrics["mae"]
        base_mae = baseline_metrics["mae"]

        baseline_imp = calculate_relative_improvement(base_mae, cand_mae)
        crit_baseline = baseline_imp >= gate_config.minimum_baseline_mae_improvement_pct

        criteria_results: dict[str, bool] = {
            "beats_persistence_threshold": crit_baseline,
        }
        reasons: list[str] = []

        if crit_baseline:
            reasons.append(
                f"Candidate MAE ({cand_mae:.4f}) improved {baseline_imp:.2f}% vs persistence ({base_mae:.4f}), "
                f"exceeding required {gate_config.minimum_baseline_mae_improvement_pct:.2f}% threshold"
            )
        else:
            reasons.append(
                f"Candidate MAE ({cand_mae:.4f}) improved only {baseline_imp:.2f}% vs persistence ({base_mae:.4f}), "
                f"failing required {gate_config.minimum_baseline_mae_improvement_pct:.2f}% threshold"
            )

        ref_imp: float | None = None
        if reference_metrics is not None:
            ref_mae = reference_metrics["mae"]
            ref_imp = calculate_relative_improvement(ref_mae, cand_mae)
            allowed_max_mae = ref_mae * (
                1.0 + gate_config.maximum_allowed_regression_vs_reference_pct / 100.0
            )
            crit_reference = cand_mae <= allowed_max_mae
            criteria_results["reference_comparison"] = crit_reference

            if crit_reference:
                reasons.append(
                    f"Candidate MAE ({cand_mae:.4f}) did not exceed reference limit ({allowed_max_mae:.4f})"
                )
            else:
                reasons.append(
                    f"Candidate MAE ({cand_mae:.4f}) regressed beyond reference limit ({allowed_max_mae:.4f})"
                )
        else:
            if gate_config.require_reference_comparison:
                criteria_results["reference_comparison"] = False
                reasons.append("Reference model comparison was required by policy but not available")
            else:
                criteria_results["reference_comparison"] = True

        all_passed = all(criteria_results.values())
        status = "ACCEPTED" if all_passed else "REJECTED"

        return PromotionDecision(
            status=status,
            is_promoted=all_passed,
            criteria_results=criteria_results,
            reasons=reasons,
            candidate_metrics=candidate_metrics,
            baseline_metrics=baseline_metrics,
            reference_metrics=reference_metrics,
            baseline_improvement_pct=baseline_imp,
            reference_improvement_pct=ref_imp,
        )

    def save_candidate_artifacts(
        self,
        forecaster: EnergyForecaster,
        checkpoint: RetrainingCheckpoint,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        decision: PromotionDecision,
        gate_config: PromotionGateConfig,
        output_dir: Path | str,
    ) -> tuple[Path, Path, Path]:
        """Persist candidate model artifact, metadata, and retraining experiment log."""
        out_path = Path(output_dir)
        models_dir = out_path / "models"
        exp_dir = out_path / "experiments"
        models_dir.mkdir(parents=True, exist_ok=True)
        exp_dir.mkdir(parents=True, exist_ok=True)

        version = checkpoint.candidate_version
        model_file = models_dir / f"forgecast_{version}_candidate.pkl"
        metadata_file = models_dir / f"forgecast_{version}_candidate_metadata.json"
        exp_file = exp_dir / f"retraining_{version}.json"

        dataset_hash = (
            compute_file_sha256(self.dataset_path)
            if self.dataset_path.exists()
            else "UNKNOWN"
        )

        metadata_dict: dict[str, Any] = {
            "model_version": version,
            "artifact_role": "candidate",
            "model_class": forecaster.estimator.__class__.__name__,
            "estimator_class": forecaster.estimator.__class__.__name__,
            "hyperparameters": forecaster.config.to_dict(),
            "feature_names": list(FEATURE_NAMES),
            "target_name": "target_usage_kwh",
            "training_cutoff": checkpoint.training_cutoff.isoformat(),
            "training_cutoff_type": "target_timestamp",
            "training_sample_count": len(df_train),
            "training_start_origin": df_train["origin_timestamp"].min().isoformat(),
            "training_end_origin": df_train["origin_timestamp"].max().isoformat(),
            "training_start_target": df_train["target_timestamp"].min().isoformat(),
            "training_end_target": df_train["target_timestamp"].max().isoformat(),
            "evaluation_start": checkpoint.eval_start.isoformat(),
            "evaluation_end": checkpoint.eval_end.isoformat(),
            "evaluation_window_type": "target_timestamp",
            "evaluation_start_origin": df_val["origin_timestamp"].min().isoformat(),
            "evaluation_end_origin": df_val["origin_timestamp"].max().isoformat(),
            "evaluation_start_target": df_val["target_timestamp"].min().isoformat(),
            "evaluation_end_target": df_val["target_timestamp"].max().isoformat(),
            "evaluation_sample_count": len(df_val),
            "candidate_metrics": decision.candidate_metrics,
            "baseline_metrics": decision.baseline_metrics,
            "reference_metrics": decision.reference_metrics,
            "promotion_policy": gate_config.to_dict(),
            "promotion_decision": decision.to_dict(),
            "dataset_sha256": dataset_hash,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scikit_learn_version": sklearn.__version__,
            "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

        # Save candidate model binary
        with open(model_file, "wb") as f:
            pickle.dump(forecaster, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save candidate companion metadata
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=2)

        # Save experiment audit record
        exp_record = {
            "checkpoint": checkpoint.to_dict(),
            "decision": decision.to_dict(),
            "metadata_path": str(metadata_file),
            "model_path": str(model_file),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        with open(exp_file, "w", encoding="utf-8") as f:
            json.dump(exp_record, f, indent=2)

        return model_file, metadata_file, exp_file

    def run(
        self,
        checkpoint: RetrainingCheckpoint,
        gate_config: PromotionGateConfig | None = None,
        reference_model_path: Path | str | None = None,
        df_supervised: pd.DataFrame | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Execute complete historical retraining cycle."""
        gate = gate_config or PromotionGateConfig()

        if df_supervised is None:
            df_supervised = self.load_data()

        self.validate_checkpoint(checkpoint, df_supervised)
        df_train, df_val = self.prepare_splits(checkpoint, df_supervised)

        candidate = self.train_candidate(df_train)
        cand_metrics = self.evaluate_candidate(candidate, df_val)
        base_metrics = self.evaluate_baseline(df_val)
        ref_metrics, ref_status = self.evaluate_reference(
            reference_model_path, checkpoint, df_val
        )

        decision = self.apply_promotion_gate(cand_metrics, base_metrics, ref_metrics, gate)

        artifacts: dict[str, str | None] = {
            "model_file": None,
            "metadata_file": None,
            "experiment_file": None,
        }

        if decision.is_promoted and not dry_run:
            m_path, meta_path, exp_path = self.save_candidate_artifacts(
                forecaster=candidate,
                checkpoint=checkpoint,
                df_train=df_train,
                df_val=df_val,
                decision=decision,
                gate_config=gate,
                output_dir=self.artifacts_dir,
            )
            artifacts["model_file"] = str(m_path)
            artifacts["metadata_file"] = str(meta_path)
            artifacts["experiment_file"] = str(exp_path)
        elif not dry_run:
            # Preserve experiment audit record even when rejected
            exp_dir = self.artifacts_dir / "experiments"
            exp_dir.mkdir(parents=True, exist_ok=True)
            exp_file = exp_dir / f"retraining_{checkpoint.candidate_version}.json"
            exp_record = {
                "checkpoint": checkpoint.to_dict(),
                "decision": decision.to_dict(),
                "rejected": True,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            with open(exp_file, "w", encoding="utf-8") as f:
                json.dump(exp_record, f, indent=2)
            artifacts["experiment_file"] = str(exp_file)

        return {
            "checkpoint": checkpoint,
            "training_sample_count": len(df_train),
            "validation_sample_count": len(df_val),
            "candidate_metrics": cand_metrics,
            "baseline_metrics": base_metrics,
            "reference_metrics": ref_metrics,
            "reference_status": ref_status,
            "decision": decision,
            "artifacts": artifacts,
            "dry_run": dry_run,
        }
