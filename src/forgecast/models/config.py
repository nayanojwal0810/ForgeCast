"""Frozen hyperparameter configuration for the primary industrial energy forecaster."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FrozenModelConfig:
    """Exact frozen hyperparameters for HistGradientBoostingRegressor.

    Hyperparameter tuning and automated sweeps are prohibited by project specification.
    early_stopping=False is strictly enforced to prevent internal non-chronological splits.
    """

    loss: str = "squared_error"
    learning_rate: float = 0.05
    max_iter: int = 200
    max_leaf_nodes: int = 31
    l2_regularization: float = 1.0
    random_state: int = 42
    early_stopping: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
