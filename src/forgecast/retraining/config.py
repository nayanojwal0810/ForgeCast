"""Configuration contracts and promotion gates for historical model retraining."""

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class RetrainingCheckpoint:
    """Historical retraining checkpoint defining target-time training cutoff and chronological validation window.

    Attributes:
        name: Identifier for the checkpoint (e.g. 'Q3').
        training_cutoff: Latest allowable target_timestamp in the training partition.
        eval_start: Earliest allowable target_timestamp in the validation partition.
        eval_end: Latest allowable target_timestamp in the validation partition.
        candidate_version: Version identifier for the candidate model (e.g. 'v2').
    """

    name: str
    training_cutoff: datetime
    eval_start: datetime
    eval_end: datetime
    candidate_version: str

    def to_dict(self) -> dict[str, Any]:
        """Convert checkpoint to serializable dictionary."""
        return {
            "name": self.name,
            "training_cutoff": self.training_cutoff.isoformat(),
            "eval_start": self.eval_start.isoformat(),
            "eval_end": self.eval_end.isoformat(),
            "candidate_version": self.candidate_version,
        }


# Predefined quarterly boundaries based on the 2018 dataset
# The dataset covers 2018-01-01 to 2018-12-31.
# Q1: Jan 1 - Mar 31
# Q2: Apr 1 - Jun 30
# Q3: Jul 1 - Sep 30
# Q4: Oct 1 - Dec 31 (evaluation strictly bounded prior to the held-out final test set at Nov 25 12:00)
QUARTERLY_CHECKPOINTS: dict[str, RetrainingCheckpoint] = {
    "Q2": RetrainingCheckpoint(
        name="Q2",
        training_cutoff=datetime(2018, 3, 31, 23, 45),
        eval_start=datetime(2018, 4, 1, 0, 0),
        eval_end=datetime(2018, 6, 30, 23, 45),
        candidate_version="v2_q2",
    ),
    "Q3": RetrainingCheckpoint(
        name="Q3",
        training_cutoff=datetime(2018, 6, 30, 23, 45),
        eval_start=datetime(2018, 7, 1, 0, 0),
        eval_end=datetime(2018, 9, 30, 23, 45),
        candidate_version="v2",
    ),
    "Q4": RetrainingCheckpoint(
        name="Q4",
        training_cutoff=datetime(2018, 9, 30, 23, 45),
        eval_start=datetime(2018, 10, 1, 0, 0),
        eval_end=datetime(2018, 11, 25, 11, 45),
        candidate_version="v2_q4",
    ),
}


@dataclass(frozen=True)
class PromotionGateConfig:
    """Objective criteria required for candidate model promotion."""

    minimum_baseline_mae_improvement_pct: float = 5.0
    maximum_allowed_regression_vs_reference_pct: float = 0.0
    require_reference_comparison: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert gate configuration to dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class PromotionDecision:
    """Explicit verdict of the promotion gate."""

    status: str  # "ACCEPTED" or "REJECTED"
    is_promoted: bool
    criteria_results: dict[str, bool]
    reasons: list[str]
    candidate_metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    reference_metrics: dict[str, float] | None
    baseline_improvement_pct: float
    reference_improvement_pct: float | None

    def to_dict(self) -> dict[str, Any]:
        """Convert decision to dictionary."""
        return asdict(self)
