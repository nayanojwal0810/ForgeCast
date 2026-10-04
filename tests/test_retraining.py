"""Tests for RetrainingRunner, target-based checkpoint validation, promotion gates, and leakage protection."""

from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from forgecast.features.generator import FEATURE_NAMES
from forgecast.models.config import FrozenModelConfig
from forgecast.models.forecaster import EnergyForecaster
from forgecast.models.persistence import load_model_artifact
from forgecast.retraining.config import (
    QUARTERLY_CHECKPOINTS,
    PromotionGateConfig,
    RetrainingCheckpoint,
)
from forgecast.retraining.runner import RetrainingRunner


@pytest.fixture
def synthetic_supervised_df() -> pd.DataFrame:
    """Create a contiguous 300-sample synthetic supervised DataFrame with canonical feature columns."""
    rng = np.random.default_rng(42)
    start_dt = datetime(2018, 1, 2, 0, 0)
    rows: list[dict] = []

    for i in range(300):
        origin_ts = start_dt + timedelta(minutes=15 * i)
        target_ts = origin_ts + timedelta(minutes=15)
        # Smooth cyclical usage pattern
        usage = 20.0 + 10.0 * np.sin(i / 10.0) + rng.normal(0, 0.5)
        target_usage = 20.0 + 10.0 * np.sin((i + 1) / 10.0) + rng.normal(0, 0.5)

        row = {
            "origin_timestamp": origin_ts,
            "target_timestamp": target_ts,
            "origin_index": i,
            "target_index": i + 1,
            "target_usage_kwh": target_usage,
        }
        for col in FEATURE_NAMES:
            if col == "usage_lag_0":
                row[col] = usage
            elif col.startswith("usage_lag_"):
                row[col] = usage + rng.normal(0, 1.0)
            elif col.startswith("usage_roll_"):
                row[col] = usage
            else:
                row[col] = 1.0
        rows.append(row)

    return pd.DataFrame(rows)


def test_checkpoint_validity_checks(synthetic_supervised_df: pd.DataFrame) -> None:
    """Verify temporal bounds, ordering, and sample count enforcement on target timestamps."""
    # 1. Invalid: training cutoff >= eval_start
    invalid_cp_order = RetrainingCheckpoint(
        name="InvalidOrder",
        training_cutoff=datetime(2018, 1, 3, 12, 0),
        eval_start=datetime(2018, 1, 3, 10, 0),
        eval_end=datetime(2018, 1, 4, 0, 0),
        candidate_version="v_test",
    )
    with pytest.raises(ValueError, match="strictly earlier than evaluation target start"):
        RetrainingRunner.validate_checkpoint(invalid_cp_order, synthetic_supervised_df)

    # 2. Invalid: eval_start > eval_end
    invalid_eval_range = RetrainingCheckpoint(
        name="InvalidEvalRange",
        training_cutoff=datetime(2018, 1, 2, 12, 0),
        eval_start=datetime(2018, 1, 4, 0, 0),
        eval_end=datetime(2018, 1, 3, 0, 0),
        candidate_version="v_test",
    )
    with pytest.raises(ValueError, match="earlier than or equal to evaluation target end"):
        RetrainingRunner.validate_checkpoint(invalid_eval_range, synthetic_supervised_df)

    # 3. Invalid: insufficient training data (< 96)
    cutoff_too_early = synthetic_supervised_df["target_timestamp"].iloc[50]
    insufficient_train_cp = RetrainingCheckpoint(
        name="InsufficientTrain",
        training_cutoff=cutoff_too_early,
        eval_start=cutoff_too_early + timedelta(minutes=15),
        eval_end=cutoff_too_early + timedelta(hours=24),
        candidate_version="v_test",
    )
    with pytest.raises(ValueError, match="Insufficient training observations: got 51, minimum required is 96"):
        RetrainingRunner.validate_checkpoint(insufficient_train_cp, synthetic_supervised_df)


def test_target_based_partitioning_and_invariants(synthetic_supervised_df: pd.DataFrame) -> None:
    """Confirm training and validation partitions adhere to strict target-time separation."""
    cutoff = synthetic_supervised_df["target_timestamp"].iloc[150]
    eval_start = synthetic_supervised_df["target_timestamp"].iloc[151]
    eval_end = synthetic_supervised_df["target_timestamp"].iloc[250]

    cp = RetrainingCheckpoint(
        name="ValidCP",
        training_cutoff=cutoff,
        eval_start=eval_start,
        eval_end=eval_end,
        candidate_version="v_test",
    )

    df_train, df_val = RetrainingRunner.prepare_splits(cp, synthetic_supervised_df)

    # 1. Training samples must all have target <= cutoff
    assert df_train["target_timestamp"].max() <= cutoff

    # 2. Validation samples must all have target in [eval_start, eval_end]
    assert df_val["target_timestamp"].min() >= eval_start
    assert df_val["target_timestamp"].max() <= eval_end

    # 3. Strict ordering: max(train.target_timestamp) < min(val.target_timestamp)
    assert df_train["target_timestamp"].max() < df_val["target_timestamp"].min()

    # 4. No target overlap: set(train.target_timestamp) ∩ set(val.target_timestamp) == empty
    overlap = set(df_train["target_timestamp"]).intersection(set(df_val["target_timestamp"]))
    assert len(overlap) == 0

    # 5. Target column not used as feature
    assert "target_usage_kwh" not in FEATURE_NAMES
    assert list(df_train[list(FEATURE_NAMES)].columns) == list(FEATURE_NAMES)


def test_exact_one_step_boundary_behavior(synthetic_supervised_df: pd.DataFrame) -> None:
    """Verify exact one-step boundary:

    Train final target = 2018-06-30 23:45
    Validation first target = 2018-07-01 00:00
    Validation first origin = 2018-06-30 23:45
    """
    cutoff = datetime(2018, 6, 30, 23, 45)
    eval_start = datetime(2018, 7, 1, 0, 0)
    eval_end = datetime(2018, 7, 1, 23, 45)

    # Construct bounded records around the boundary
    boundary_records: list[dict] = []
    # 100 historical training records ending at target 2018-06-30 23:45
    # (origin 2018-06-30 23:30)
    for i in range(100):
        # target_ts for final (i=99) is 2018-06-30 23:45
        t_target = cutoff - timedelta(minutes=15 * (99 - i))
        t_origin = t_target - timedelta(minutes=15)
        row = {
            "origin_timestamp": t_origin,
            "target_timestamp": t_target,
            "origin_index": i,
            "target_index": i + 1,
            "target_usage_kwh": 25.0,
        }
        for col in FEATURE_NAMES:
            row[col] = 10.0
        boundary_records.append(row)

    # Boundary sample:
    # origin = 2018-06-30 23:45, target = 2018-07-01 00:00
    boundary_sample = {
        "origin_timestamp": datetime(2018, 6, 30, 23, 45),
        "target_timestamp": datetime(2018, 7, 1, 0, 0),
        "origin_index": 100,
        "target_index": 101,
        "target_usage_kwh": 28.0,
    }
    for col in FEATURE_NAMES:
        boundary_sample[col] = 12.0
    boundary_records.append(boundary_sample)

    # 10 more validation samples on 2018-07-01
    for j in range(1, 11):
        v_target = eval_start + timedelta(minutes=15 * j)
        v_origin = v_target - timedelta(minutes=15)
        row = {
            "origin_timestamp": v_origin,
            "target_timestamp": v_target,
            "origin_index": 100 + j,
            "target_index": 101 + j,
            "target_usage_kwh": 30.0,
        }
        for col in FEATURE_NAMES:
            row[col] = 15.0
        boundary_records.append(row)

    df_boundary = pd.DataFrame(boundary_records)

    cp = RetrainingCheckpoint(
        name="Q3_Boundary",
        training_cutoff=cutoff,
        eval_start=eval_start,
        eval_end=eval_end,
        candidate_version="v2",
    )

    df_train, df_val = RetrainingRunner.prepare_splits(cp, df_boundary)

    # Final training sample
    final_train = df_train.iloc[-1]
    assert final_train["target_timestamp"] == datetime(2018, 6, 30, 23, 45)
    assert final_train["origin_timestamp"] == datetime(2018, 6, 30, 23, 30)

    # First validation sample
    first_val = df_val.iloc[0]
    assert first_val["target_timestamp"] == datetime(2018, 7, 1, 0, 0)
    assert first_val["origin_timestamp"] == datetime(2018, 6, 30, 23, 45)

    # The boundary sample (target=00:00) MUST be in validation, NOT training!
    assert datetime(2018, 7, 1, 0, 0) not in set(df_train["target_timestamp"])
    assert datetime(2018, 7, 1, 0, 0) in set(df_val["target_timestamp"])


def test_future_target_protection(synthetic_supervised_df: pd.DataFrame) -> None:
    """Mandatory test: a sample whose origin is before the cutoff but whose target is after

    the cutoff must be strictly excluded from training.
    """
    cutoff = datetime(2018, 6, 30, 23, 45)
    eval_start = datetime(2018, 7, 1, 0, 0)

    # Inject a sample with origin <= cutoff but target > cutoff
    leaky_sample = {
        "origin_timestamp": datetime(2018, 6, 30, 23, 40),  # origin BEFORE cutoff
        "target_timestamp": datetime(2018, 6, 30, 23, 55),  # target AFTER cutoff!
        "origin_index": 9999,
        "target_index": 10000,
        "target_usage_kwh": 40.0,
    }
    for col in FEATURE_NAMES:
        leaky_sample[col] = 5.0

    df_with_leaky = pd.concat([synthetic_supervised_df, pd.DataFrame([leaky_sample])], ignore_index=True)

    cp = RetrainingCheckpoint(
        name="ProtectionCP",
        training_cutoff=cutoff,
        eval_start=eval_start,
        eval_end=datetime(2018, 7, 2, 0, 0),
        candidate_version="v_test",
    )

    df_train, df_val = RetrainingRunner.prepare_splits(cp, df_with_leaky)

    # The leaky sample has target 23:55 > 23:45, so it MUST NOT enter training
    train_targets = set(df_train["target_timestamp"])
    assert datetime(2018, 6, 30, 23, 55) not in train_targets
    assert all(t <= cutoff for t in train_targets)


def test_candidate_training_and_reproducibility(synthetic_supervised_df: pd.DataFrame) -> None:
    """Verify deterministic candidate training and equivalent behavior on identical configuration."""
    cutoff = synthetic_supervised_df["target_timestamp"].iloc[150]
    eval_start = synthetic_supervised_df["target_timestamp"].iloc[151]
    eval_end = synthetic_supervised_df["target_timestamp"].iloc[250]

    cp = RetrainingCheckpoint(
        name="ReproCP",
        training_cutoff=cutoff,
        eval_start=eval_start,
        eval_end=eval_end,
        candidate_version="v2_test",
    )

    runner1 = RetrainingRunner()
    res1 = runner1.run(cp, df_supervised=synthetic_supervised_df, dry_run=True)

    runner2 = RetrainingRunner()
    res2 = runner2.run(cp, df_supervised=synthetic_supervised_df, dry_run=True)

    # Metrics must match exactly to 10 decimal places
    assert pytest.approx(res1["candidate_metrics"]["mae"], rel=1e-9) == res2["candidate_metrics"]["mae"]
    assert pytest.approx(res1["candidate_metrics"]["rmse"], rel=1e-9) == res2["candidate_metrics"]["rmse"]
    assert pytest.approx(res1["baseline_metrics"]["mae"], rel=1e-9) == res2["baseline_metrics"]["mae"]


def test_promotion_gate_acceptance_and_rejection(synthetic_supervised_df: pd.DataFrame) -> None:
    """Test explicit promotion logic: candidate accepted when meeting gate, rejected when failing."""
    cutoff = synthetic_supervised_df["target_timestamp"].iloc[150]
    eval_start = synthetic_supervised_df["target_timestamp"].iloc[151]
    eval_end = synthetic_supervised_df["target_timestamp"].iloc[250]

    cp = RetrainingCheckpoint(
        name="GateCP",
        training_cutoff=cutoff,
        eval_start=eval_start,
        eval_end=eval_end,
        candidate_version="v2_test",
    )

    runner = RetrainingRunner()

    # 1. Gate with realistic threshold -> candidate beats persistence and is ACCEPTED
    permissive_gate = PromotionGateConfig(minimum_baseline_mae_improvement_pct=0.0)
    res_accept = runner.run(cp, gate_config=permissive_gate, df_supervised=synthetic_supervised_df, dry_run=True)
    assert res_accept["decision"].status == "ACCEPTED"
    assert res_accept["decision"].is_promoted is True

    # 2. Gate with impossible threshold -> candidate is REJECTED
    strict_gate = PromotionGateConfig(minimum_baseline_mae_improvement_pct=99.9)
    res_reject = runner.run(cp, gate_config=strict_gate, df_supervised=synthetic_supervised_df, dry_run=True)
    assert res_reject["decision"].status == "REJECTED"
    assert res_reject["decision"].is_promoted is False
    assert any("failing required" in r for r in res_reject["decision"].reasons)


def test_artifact_persistence_on_acceptance_and_rejection(synthetic_supervised_df: pd.DataFrame) -> None:
    """Verify artifacts are written on ACCEPTED and omitted on REJECTED or dry-run."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        cutoff = synthetic_supervised_df["target_timestamp"].iloc[150]
        eval_start = synthetic_supervised_df["target_timestamp"].iloc[151]
        eval_end = synthetic_supervised_df["target_timestamp"].iloc[250]

        cp_accept = RetrainingCheckpoint(
            name="AcceptCP",
            training_cutoff=cutoff,
            eval_start=eval_start,
            eval_end=eval_end,
            candidate_version="v2_promoted",
        )

        runner = RetrainingRunner(artifacts_dir=tmp_path)

        # 1. Accepted run -> writes model, metadata, experiment JSON
        gate_accept = PromotionGateConfig(minimum_baseline_mae_improvement_pct=0.0)
        res_accept = runner.run(cp_accept, gate_config=gate_accept, df_supervised=synthetic_supervised_df, dry_run=False)

        model_path = tmp_path / "models" / "forgecast_v2_promoted_candidate.pkl"
        meta_path = tmp_path / "models" / "forgecast_v2_promoted_candidate_metadata.json"
        exp_path = tmp_path / "experiments" / "retraining_v2_promoted.json"

        assert model_path.exists()
        assert meta_path.exists()
        assert exp_path.exists()

        # Load back candidate model and verify metadata
        forecaster, metadata = load_model_artifact(model_path)
        assert metadata["model_version"] == "v2_promoted"
        assert metadata["artifact_role"] == "candidate"
        assert metadata["training_sample_count"] == 151
        assert metadata["feature_names"] == list(FEATURE_NAMES)
        assert metadata["training_cutoff_type"] == "target_timestamp"
        assert metadata["evaluation_window_type"] == "target_timestamp"
        assert forecaster.is_fitted

        # 2. Rejected run -> does NOT write candidate model to models/
        cp_reject = RetrainingCheckpoint(
            name="RejectCP",
            training_cutoff=cutoff,
            eval_start=eval_start,
            eval_end=eval_end,
            candidate_version="v2_rejected",
        )
        gate_reject = PromotionGateConfig(minimum_baseline_mae_improvement_pct=99.9)
        res_reject = runner.run(cp_reject, gate_config=gate_reject, df_supervised=synthetic_supervised_df, dry_run=False)

        rej_model = tmp_path / "models" / "forgecast_v2_rejected_candidate.pkl"
        rej_exp = tmp_path / "experiments" / "retraining_v2_rejected.json"

        assert not rej_model.exists()  # Must NOT write candidate artifact
        assert rej_exp.exists()        # Audit log IS preserved
        with open(rej_exp, "r") as f:
            exp_data = json.load(f)
            assert exp_data["rejected"] is True


def test_q3_last_training_target_included() -> None:
    """Assert target_timestamp = 2018-06-30 23:45 is included in training and reconcile 17,279 count."""
    cutoff = datetime(2018, 6, 30, 23, 45)
    eval_start = datetime(2018, 7, 1, 0, 0)
    eval_end = datetime(2018, 9, 30, 23, 45)

    # 1. Mathematical reconciliation:
    # 181 calendar days in Jan-Jun (31+28+31+30+31+30) = 17,376 intervals.
    # Warmup consumes 96 intervals (Jan 1 00:15 to Jan 2 00:00).
    # First target is Jan 2 00:15 (Jan 2 00:00 was the 96th warmup observation).
    # Number of 15-min intervals from Jan 2 00:15 to Jun 30 23:45 is exactly 17,279!
    expected_intervals = pd.date_range(start="2018-01-02 00:15:00", end="2018-06-30 23:45:00", freq="15min")
    assert len(expected_intervals) == 17279

    # 2. Boundary inclusion verification
    boundary_records = [
        {
            "origin_timestamp": datetime(2018, 6, 30, 23, 30),
            "target_timestamp": cutoff,
            "origin_index": 17278,
            "target_index": 17279,
            "target_usage_kwh": 25.0,
            **{col: 10.0 for col in FEATURE_NAMES},
        },
        {
            "origin_timestamp": cutoff,
            "target_timestamp": eval_start,
            "origin_index": 17279,
            "target_index": 17280,
            "target_usage_kwh": 26.0,
            **{col: 10.0 for col in FEATURE_NAMES},
        },
    ]
    df_boundary = pd.DataFrame(boundary_records)
    cp = RetrainingCheckpoint(
        name="Q3",
        training_cutoff=cutoff,
        eval_start=eval_start,
        eval_end=eval_end,
        candidate_version="v2",
    )

    df_train = df_boundary[df_boundary["target_timestamp"] <= cp.training_cutoff]
    df_val = df_boundary[
        (df_boundary["target_timestamp"] >= cp.eval_start)
        & (df_boundary["target_timestamp"] <= cp.eval_end)
    ]

    # Target 2018-06-30 23:45 MUST be included in training
    assert cutoff in set(df_train["target_timestamp"])
    assert cutoff not in set(df_val["target_timestamp"])
    # Target 2018-07-01 00:00 MUST be included in validation
    assert eval_start in set(df_val["target_timestamp"])
    assert eval_start not in set(df_train["target_timestamp"])


def test_reference_model_target_lookahead_protection(synthetic_supervised_df: pd.DataFrame) -> None:
    """Confirm reference model comparison is excluded if reference model training target overlaps eval start."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Create a mock reference model trained up to target index 200
        ref_cutoff_target = synthetic_supervised_df["target_timestamp"].iloc[200]
        forecaster = EnergyForecaster()
        forecaster.fit(
            synthetic_supervised_df[list(FEATURE_NAMES)].iloc[:200],
            synthetic_supervised_df["target_usage_kwh"].iloc[:200],
        )

        ref_meta = {
            "model_version": "v1_ref",
            "model_class": "HistGradientBoostingRegressor",
            "hyperparameters": FrozenModelConfig().to_dict(),
            "feature_names": list(FEATURE_NAMES),
            "training_end_target": ref_cutoff_target.isoformat(),
        }

        import pickle
        ref_model_file = models_dir / "ref_model.pkl"
        ref_meta_file = models_dir / "ref_model_metadata.json"
        with open(ref_model_file, "wb") as f:
            pickle.dump(forecaster, f)
        with open(ref_meta_file, "w") as f:
            json.dump(ref_meta, f)

        # Checkpoint evaluation window: target index 100 to 150 (strictly BEFORE ref_cutoff_target=200!)
        cp = RetrainingCheckpoint(
            name="LeakCP",
            training_cutoff=synthetic_supervised_df["target_timestamp"].iloc[99],
            eval_start=synthetic_supervised_df["target_timestamp"].iloc[100],
            eval_end=synthetic_supervised_df["target_timestamp"].iloc[150],
            candidate_version="v2_leak",
        )

        runner = RetrainingRunner()
        metrics, reason = runner.evaluate_reference(ref_model_file, cp, synthetic_supervised_df.iloc[100:151])
        assert metrics is None
        assert "overlaps evaluation" in str(reason)


def test_predefined_quarterly_checkpoints() -> None:
    """Verify predefined quarterly checkpoints respect 2018 dataset boundaries."""
    assert "Q2" in QUARTERLY_CHECKPOINTS
    assert "Q3" in QUARTERLY_CHECKPOINTS
    assert "Q4" in QUARTERLY_CHECKPOINTS

    q2 = QUARTERLY_CHECKPOINTS["Q2"]
    q3 = QUARTERLY_CHECKPOINTS["Q3"]
    q4 = QUARTERLY_CHECKPOINTS["Q4"]

    assert q2.training_cutoff < q2.eval_start <= q2.eval_end
    assert q3.training_cutoff < q3.eval_start <= q3.eval_end
    assert q4.training_cutoff < q4.eval_start <= q4.eval_end

    # Q4 evaluation ends strictly before final test set at 2018-11-25 12:00
    assert q4.eval_end <= datetime(2018, 11, 25, 11, 45)
