"""Tests for EnergyForecaster, input contracts, persistence, and chronological evaluation."""

from datetime import datetime, timedelta
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forgecast.evaluation.baselines import PersistenceBaseline, SeasonalPersistenceBaseline
from forgecast.evaluation.harness import ChronologicalEvaluationHarness
from forgecast.features.generator import FEATURE_NAMES
from forgecast.models.config import FrozenModelConfig
from forgecast.models.forecaster import EnergyForecaster
from forgecast.models.persistence import create_model_metadata, load_model_artifact, save_model_artifact


@pytest.fixture
def dummy_features() -> pd.DataFrame:
    """Create a small DataFrame with exact FEATURE_NAMES columns."""
    rng = np.random.default_rng(42)
    data = {name: rng.uniform(0.0, 50.0, size=50) for name in FEATURE_NAMES}
    return pd.DataFrame(data)


@pytest.fixture
def dummy_targets() -> np.ndarray:
    """Create a 1D target array."""
    rng = np.random.default_rng(42)
    return rng.uniform(1.0, 50.0, size=50)


def test_frozen_model_hyperparameters() -> None:
    """Verify exact frozen hyperparameters are enforced without deviation."""
    config = FrozenModelConfig()
    assert config.loss == "squared_error"
    assert config.learning_rate == 0.05
    assert config.max_iter == 200
    assert config.max_leaf_nodes == 31
    assert config.l2_regularization == 1.0
    assert config.random_state == 42
    assert config.early_stopping is False

    forecaster = EnergyForecaster(config)
    params = forecaster.get_params()
    assert params["loss"] == "squared_error"
    assert params["learning_rate"] == 0.05
    assert params["max_iter"] == 200
    assert params["max_leaf_nodes"] == 31
    assert params["l2_regularization"] == 1.0
    assert params["random_state"] == 42
    assert params["early_stopping"] is False


def test_missing_feature_column_fails(dummy_features: pd.DataFrame, dummy_targets: np.ndarray) -> None:
    """Confirm fitting fails if any canonical feature column is missing."""
    df_missing = dummy_features.drop(columns=["usage_lag_0"])
    forecaster = EnergyForecaster()

    with pytest.raises(ValueError, match="missing required columns"):
        forecaster.fit(df_missing, dummy_targets)


def test_extra_feature_column_fails(dummy_features: pd.DataFrame, dummy_targets: np.ndarray) -> None:
    """Confirm fitting fails if unexpected extra columns are present."""
    df_extra = dummy_features.copy()
    df_extra["extra_sensor"] = 1.0
    forecaster = EnergyForecaster()

    with pytest.raises(ValueError, match="unexpected extra columns"):
        forecaster.fit(df_extra, dummy_targets)


def test_feature_reordering_fails(dummy_features: pd.DataFrame, dummy_targets: np.ndarray) -> None:
    """Confirm feature order is strictly enforced."""
    reversed_cols = list(reversed(list(dummy_features.columns)))
    df_reordered = dummy_features[reversed_cols]
    forecaster = EnergyForecaster()

    with pytest.raises(ValueError, match="column order does not match"):
        forecaster.fit(df_reordered, dummy_targets)


def test_target_column_rejected_in_feature_input(
    dummy_features: pd.DataFrame, dummy_targets: np.ndarray
) -> None:
    """Confirm that passing target column inside feature DataFrame is explicitly rejected."""
    df_with_target = dummy_features.copy()
    df_with_target["target_usage_kwh"] = dummy_targets
    forecaster = EnergyForecaster()

    with pytest.raises(ValueError, match="Target variable 'target_usage_kwh' must not be included"):
        forecaster.fit(df_with_target, dummy_targets)


def test_forecaster_train_and_predict(dummy_features: pd.DataFrame, dummy_targets: np.ndarray) -> None:
    """Confirm forecaster fits successfully and generates finite predictions of matching shape."""
    forecaster = EnergyForecaster()
    assert not forecaster.is_fitted

    with pytest.raises(RuntimeError, match="Cannot predict with unfitted forecaster"):
        forecaster.predict(dummy_features)

    forecaster.fit(dummy_features, dummy_targets)
    assert forecaster.is_fitted

    preds = forecaster.predict(dummy_features)
    assert len(preds) == len(dummy_features)
    assert np.all(np.isfinite(preds))


def test_model_persistence_equality(dummy_features: pd.DataFrame, dummy_targets: np.ndarray) -> None:
    """Confirm trained forecaster can be saved, loaded, and produces bit-identical predictions."""
    forecaster = EnergyForecaster()
    forecaster.fit(dummy_features, dummy_targets)
    preds_orig = forecaster.predict(dummy_features)

    with tempfile.TemporaryDirectory() as tmp_dir:
        meta = create_model_metadata(
            forecaster=forecaster,
            model_version="v1_test",
            training_sample_count=len(dummy_features),
            training_start_origin=datetime(2018, 1, 1, 0, 0),
            training_end_origin=datetime(2018, 1, 1, 12, 0),
            training_start_target=datetime(2018, 1, 1, 0, 15),
            training_end_target=datetime(2018, 1, 1, 12, 15),
            dataset_sha256="TEST_HASH",
        )
        model_file, meta_file = save_model_artifact(
            forecaster=forecaster,
            metadata=meta,
            output_dir=tmp_dir,
            prefix="test_model",
        )
        assert model_file.exists()
        assert meta_file.exists()

        # Load back
        loaded_forecaster, loaded_meta = load_model_artifact(model_file)
        assert loaded_forecaster.is_fitted
        assert loaded_meta is not None
        assert loaded_meta["model_version"] == "v1_test"
        assert loaded_meta["dataset_sha256"] == "TEST_HASH"

        preds_loaded = loaded_forecaster.predict(dummy_features)
        np.testing.assert_array_equal(preds_orig, preds_loaded)


def test_chronological_evaluation_harness_small_mock() -> None:
    """Test ChronologicalEvaluationHarness temporal split integrity on synthetic contiguous data."""
    n_samples = 400
    test_size = 50
    n_splits = 3

    start_origin = datetime(2018, 1, 1, 0, 0)
    records = []
    rng = np.random.default_rng(42)

    for i in range(n_samples):
        orig_dt = start_origin + timedelta(minutes=15 * i)
        target_dt = orig_dt + timedelta(minutes=15)
        row = {
            "origin_timestamp": orig_dt,
            "target_timestamp": target_dt,
            "origin_index": i,
            "target_index": i + 1,
            "target_usage_kwh": float(rng.uniform(5.0, 30.0)),
        }
        for name in FEATURE_NAMES:
            row[name] = float(rng.uniform(1.0, 25.0))
        records.append(row)

    df_synth = pd.DataFrame(records)

    with tempfile.TemporaryDirectory() as tmp_dir:
        harness = ChronologicalEvaluationHarness(n_splits=n_splits, test_size=test_size, gap=0)
        results = harness.run(
            df_supervised=df_synth,
            dataset_path="data/raw/Steel_industry_data.csv",
            artifacts_dir=tmp_dir,
            save_replay_model=True,
        )

        assert results["n_splits"] == 3
        assert len(results["folds"]) == 3
        assert results["pooled_results"]["pooled_evaluation_sample_count"] == n_splits * test_size

        # Check fold 1, 2, 3 progression
        for fold in results["folds"]:
            assert fold["eval_sample_count"] == test_size
            assert fold["ml_mae"] > 0.0
            assert fold["persistence_mae"] > 0.0

        # Check replay model artifact was created
        assert results["replay_model_artifact"] is not None
        assert Path(results["replay_model_artifact"]).exists()
