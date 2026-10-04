"""Baseline forecasting methods: persistence and seasonal persistence."""

import numpy as np
import pandas as pd


class PersistenceBaseline:
    """Predicts target(i+1) using latest observed consumption Usage(i) via usage_lag_0."""

    @staticmethod
    def predict(X: pd.DataFrame) -> np.ndarray:
        if "usage_lag_0" not in X.columns:
            raise KeyError("Feature 'usage_lag_0' required for persistence baseline")
        return np.asarray(X["usage_lag_0"], dtype=np.float64)


class SeasonalPersistenceBaseline:
    """Predicts target(i+1) using 24h seasonal persistence Usage(i+1-96) via usage_lag_95."""

    @staticmethod
    def predict(X: pd.DataFrame) -> np.ndarray:
        if "usage_lag_95" not in X.columns:
            raise KeyError("Feature 'usage_lag_95' required for seasonal persistence baseline")
        return np.asarray(X["usage_lag_95"], dtype=np.float64)
