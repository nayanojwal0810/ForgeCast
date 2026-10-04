"""Artifact persistence and metadata serialization for trained forecasters."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import platform
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from forgecast.features.generator import FEATURE_NAMES
from forgecast.models.forecaster import EnergyForecaster


@dataclass(frozen=True)
class ModelMetadata:
    """Complete provenance and reproducibility metadata for a trained model artifact."""

    model_version: str
    model_class: str
    hyperparameters: dict[str, Any]
    feature_names: list[str]
    target_name: str
    training_sample_count: int
    training_start_origin: str
    training_end_origin: str
    training_start_target: str
    training_end_target: str
    dataset_sha256: str
    python_version: str
    numpy_version: str
    pandas_version: str
    scikit_learn_version: str
    training_timestamp_utc: str

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)


def create_model_metadata(
    forecaster: EnergyForecaster,
    model_version: str,
    training_sample_count: int,
    training_start_origin: datetime,
    training_end_origin: datetime,
    training_start_target: datetime,
    training_end_target: datetime,
    dataset_sha256: str,
    target_name: str = "target_usage_kwh",
) -> ModelMetadata:
    """Construct a ModelMetadata instance capturing the exact active runtime environment."""
    return ModelMetadata(
        model_version=model_version,
        model_class=forecaster.estimator.__class__.__name__,
        hyperparameters=forecaster.config.to_dict(),
        feature_names=list(FEATURE_NAMES),
        target_name=target_name,
        training_sample_count=training_sample_count,
        training_start_origin=training_start_origin.isoformat(),
        training_end_origin=training_end_origin.isoformat(),
        training_start_target=training_start_target.isoformat(),
        training_end_target=training_end_target.isoformat(),
        dataset_sha256=dataset_sha256,
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        pandas_version=pd.__version__,
        scikit_learn_version=sklearn.__version__,
        training_timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )


def save_model_artifact(
    forecaster: EnergyForecaster,
    metadata: ModelMetadata,
    output_dir: Path | str,
    prefix: str = "forgecast_v1",
) -> tuple[Path, Path]:
    """Serialize model artifact and metadata to disk.

    Returns:
        (model_path, metadata_path)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model_file = out_path / f"{prefix}.pkl"
    metadata_file = out_path / f"{prefix}_metadata.json"

    if model_file.exists():
        raise FileExistsError(f"Immutable model artifact already exists at '{model_file}'")
    if metadata_file.exists():
        raise FileExistsError(f"Immutable metadata file already exists at '{metadata_file}'")

    with open(model_file, "wb") as f:
        pickle.dump(forecaster, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata.to_dict(), f, indent=2)

    return model_file, metadata_file


def load_model_artifact(model_path: Path | str) -> tuple[EnergyForecaster, dict[str, Any] | None]:
    """Load pickled EnergyForecaster and optional companion metadata file."""
    m_path = Path(model_path)
    if not m_path.exists():
        raise FileNotFoundError(f"Model artifact not found at '{m_path}'")

    with open(m_path, "rb") as f:
        forecaster = pickle.load(f)

    meta_file = m_path.parent / f"{m_path.stem}_metadata.json"
    metadata = None
    if meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return forecaster, metadata
