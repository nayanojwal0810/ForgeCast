"""Standard evaluation metrics for time-series forecasting."""

import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE and RMSE using scikit-learn standard implementations."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(root_mean_squared_error(y_t, y_p))
    return {"mae": mae, "rmse": rmse}


def calculate_relative_improvement(baseline_mae: float, model_mae: float) -> float:
    """Calculate percentage MAE reduction relative to baseline: (baseline - model) / baseline * 100."""
    if baseline_mae <= 0.0:
        return 0.0
    return float((baseline_mae - model_mae) / baseline_mae * 100.0)
