"""Command-line runner for historical model retraining and candidate promotion."""

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys

# Ensure src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

os.environ["OMP_NUM_THREADS"] = "1"

from forgecast.retraining.config import (
    QUARTERLY_CHECKPOINTS,
    PromotionGateConfig,
    RetrainingCheckpoint,
)
from forgecast.retraining.runner import RetrainingRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ForgeCast Historical Retraining & Candidate Promotion Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="Q3",
        choices=["Q2", "Q3", "Q4"],
        help="Predefined historical quarterly retraining checkpoint",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Override candidate version identifier (e.g. v2)",
    )
    parser.add_argument(
        "--cutoff",
        type=str,
        default=None,
        help="Override training cutoff ISO timestamp (e.g. 2018-06-30T23:45:00)",
    )
    parser.add_argument(
        "--eval-start",
        type=str,
        default=None,
        help="Override evaluation window start ISO timestamp",
    )
    parser.add_argument(
        "--eval-end",
        type=str,
        default=None,
        help="Override evaluation window end ISO timestamp",
    )
    parser.add_argument(
        "--reference-model",
        type=str,
        default="artifacts/models/forgecast_v1_replay_ready.pkl",
        help="Path to currently deployed reference model artifact",
    )
    parser.add_argument(
        "--min-baseline-improvement",
        type=float,
        default=5.0,
        help="Minimum required percentage MAE improvement over persistence baseline",
    )
    parser.add_argument(
        "--max-reference-regression",
        type=float,
        default=0.0,
        help="Maximum allowed percentage MAE regression relative to reference model",
    )
    parser.add_argument(
        "--require-reference",
        action="store_true",
        help="Require valid reference model comparison for promotion",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate candidate promotion decision without persisting promoted artifact",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Directory where models and experiment logs are stored",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/raw/Steel_industry_data.csv",
        help="Path to raw telemetry CSV dataset",
    )
    return parser.parse_args()


def resolve_checkpoint(args: argparse.Namespace) -> RetrainingCheckpoint:
    base_cp = QUARTERLY_CHECKPOINTS.get(args.checkpoint)
    if base_cp is None:
        raise ValueError(f"Unknown predefined checkpoint '{args.checkpoint}'")

    cutoff = (
        datetime.fromisoformat(args.cutoff) if args.cutoff else base_cp.training_cutoff
    )
    e_start = (
        datetime.fromisoformat(args.eval_start)
        if args.eval_start
        else base_cp.eval_start
    )
    e_end = (
        datetime.fromisoformat(args.eval_end) if args.eval_end else base_cp.eval_end
    )
    version = args.version or base_cp.candidate_version

    return RetrainingCheckpoint(
        name=args.checkpoint if not args.cutoff else f"Custom_{version}",
        training_cutoff=cutoff,
        eval_start=e_start,
        eval_end=e_end,
        candidate_version=version,
    )


def main() -> None:
    args = parse_args()
    checkpoint = resolve_checkpoint(args)

    gate_config = PromotionGateConfig(
        minimum_baseline_mae_improvement_pct=args.min_baseline_improvement,
        maximum_allowed_regression_vs_reference_pct=args.max_reference_regression,
        require_reference_comparison=args.require_reference,
    )

    runner = RetrainingRunner(
        dataset_path=args.dataset,
        artifacts_dir=args.artifacts_dir,
    )

    result = runner.run(
        checkpoint=checkpoint,
        gate_config=gate_config,
        reference_model_path=args.reference_model,
        dry_run=args.dry_run,
    )

    dec = result["decision"]
    cand_m = result["candidate_metrics"]
    base_m = result["baseline_metrics"]
    ref_m = result["reference_metrics"]
    ref_status = result["reference_status"]

    print("--- RETRAINING RUN ---")
    print(f"Candidate version:                       {checkpoint.candidate_version}")
    print(f"Training cutoff:                         {checkpoint.training_cutoff.isoformat()}\n")

    print(f"Training samples:                        {result['training_sample_count']}")
    print(f"Validation samples:                      {result['validation_sample_count']}\n")

    print(f"Candidate MAE:                           {cand_m['mae']:.4f} kWh")
    print(f"Persistence MAE:                         {base_m['mae']:.4f} kWh")
    print(f"Candidate MAE improvement vs persistence:{dec.baseline_improvement_pct:+.2f}%\n")

    if ref_m is not None:
        print(f"Reference v1 MAE:                        {ref_m['mae']:.4f} kWh")
        if dec.reference_improvement_pct is not None:
            print(f"Candidate vs reference:                  {dec.reference_improvement_pct:+.2f}%\n")
        else:
            print("Candidate vs reference:                  N/A\n")
    else:
        print(f"Reference v1 MAE:                        EXCLUDED ({ref_status})")
        print("Candidate vs reference:                  N/A\n")

    gate_str = (
        f"min_baseline_improvement={gate_config.minimum_baseline_mae_improvement_pct:.1f}%, "
        f"max_regression={gate_config.maximum_allowed_regression_vs_reference_pct:.1f}%"
    )
    print(f"Promotion gate:                          {gate_str}")
    print(f"Promotion decision:                      {dec.status}\n")

    arts = result["artifacts"]
    if args.dry_run:
        print("Artifact:                                [DRY RUN - Not Written]")
        print("Metadata:                                [DRY RUN - Not Written]")
    elif dec.is_promoted:
        print(f"Artifact:                                {arts['model_file']}")
        print(f"Metadata:                                {arts['metadata_file']}")
    else:
        print("Artifact:                                [REJECTED - Not Promoted]")
        print("Metadata:                                [REJECTED - Not Promoted]")
        if arts.get("experiment_file"):
            print(f"Audit log:                               {arts['experiment_file']}")


if __name__ == "__main__":
    main()
