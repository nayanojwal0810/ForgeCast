"""Primary industrial energy forecasting model wrapper enforcing feature contract."""

from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from forgecast.features.generator import FEATURE_NAMES
from forgecast.models.config import FrozenModelConfig


class EnergyForecaster:
    """Forecasting model wrapping HistGradientBoostingRegressor with strict input contract validation."""

    def __init__(self, config: FrozenModelConfig | None = None) -> None:
        self.config = config or FrozenModelConfig()
        self.estimator = HistGradientBoostingRegressor(**self.config.to_dict())
        self.is_fitted: bool = False

    @staticmethod
    def validate_features(X: pd.DataFrame) -> None:
        """Enforce strict feature contract on input DataFrame.

        Asserts:
        - Input is a pandas DataFrame with named columns.
        - Target variable is not present in feature matrix.
        - No required features are missing.
        - No unexpected or extra features are present.
        - Column order strictly matches canonical FEATURE_NAMES.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame for feature input, got {type(X).__name__}")

        if "target_usage_kwh" in X.columns:
            raise ValueError("Target variable 'target_usage_kwh' must not be included in feature input X")

        actual_cols = list(X.columns)
        expected_cols = list(FEATURE_NAMES)

        missing = [c for c in expected_cols if c not in X.columns]
        if missing:
            raise ValueError(f"Feature contract violation: missing required columns {missing}")

        extra = [c for c in actual_cols if c not in expected_cols]
        if extra:
            raise ValueError(f"Feature contract violation: unexpected extra columns {extra}")

        if actual_cols != expected_cols:
            raise ValueError(
                "Feature contract violation: column order does not match canonical FEATURE_NAMES ordering"
            )

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> "EnergyForecaster":
        """Fit the frozen estimator on validated training features and ground truth targets."""
        self.validate_features(X)

        if len(X) != len(y):
            raise ValueError(f"Mismatched sample counts: X has {len(X)} rows, y has {len(y)} elements")

        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim != 1:
            raise ValueError(f"Expected 1D target array, got shape {y_arr.shape}")

        if np.isnan(y_arr).any() or np.isinf(y_arr).any():
            raise ValueError("Target array contains NaN or infinite values")

        self.estimator.fit(X, y_arr)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Generate one-step-ahead consumption forecasts."""
        if not self.is_fitted:
            raise RuntimeError("Cannot predict with unfitted forecaster; call fit() first")

        if isinstance(X, pd.DataFrame):
            self.validate_features(X)
            preds = self.estimator.predict(X)
        elif isinstance(X, np.ndarray):
            if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES):
                raise ValueError(
                    f"Expected 2D array with {len(FEATURE_NAMES)} features, got shape {X.shape}"
                )
            df_x = pd.DataFrame(X, columns=list(FEATURE_NAMES))
            preds = self.estimator.predict(df_x)
        else:
            raise TypeError(f"Expected pandas DataFrame or 2D numpy array, got {type(X).__name__}")

        return np.asarray(preds, dtype=np.float64)

    def get_params(self) -> dict[str, Any]:
        """Return the estimator hyperparameters."""
        return self.estimator.get_params()
