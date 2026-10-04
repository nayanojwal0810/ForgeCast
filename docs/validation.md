# Validation

ForgeCast is validated as a temporal system: the model score matters, but so do causality, state integrity, feedback pairing, and lifecycle safety.

## Temporal correctness

The raw dataset contains 35,040 rows at a nominal 15-minute cadence. The source stores each daily `00:00` record as the interval closing at midnight, so the implementation preserves source row order instead of naively sorting literal timestamps.

For every prediction:

```text
feature_source_time <= forecast_cutoff < target_time
```

The test suite covers future-row poisoning, lag/rolling boundaries, timestamp continuity, duplicate delivery, skipped intervals, out-of-order records, and midnight transitions.

## Forecast evaluation

The frozen v1 model was evaluated with three chronological windows.

| Fold | ML MAE | Persistence MAE | MAE reduction |
|---|---:|---:|---:|
| 1 | 3.7735 | 5.2173 | 27.67% |
| 2 | 4.6011 | 6.6360 | 30.67% |
| 3 | 3.2133 | 4.2532 | 24.45% |
| **Pooled** | **3.8626** | **5.3688** | **28.05%** |

The pooled evaluation contains **10,512 held-out observations**.

The active feature contract was kept deliberately compact. The historical comparison showed that adding 16 historical electrical features changed aggregate MAE by only about **0.69%** and degraded one chronological window, so the 19-feature Usage + Calendar contract remained the primary formulation.

## Operational replay

The full replay exercises the prediction-to-feedback lifecycle over all 35,040 telemetry rows.

Observed results:

| Metric | Result |
|---|---:|
| Telemetry consumed | 35,040 |
| Predictions generated | 34,945 |
| Completed feedback records | 34,944 |
| Pending predictions at end | 1 |
| Unmatched feedback events | 0 |
| Duplicate feedback events | 0 |
| Sequence failures | 0 |
| Gap incidents in clean replay | 0 |
| Invalidated predictions in clean replay | 0 |

The post-training out-of-sample period contains **3,504** completed feedback records and matches the third chronological evaluation window:

- ML MAE: `3.2133 kWh`
- persistence MAE: `4.2532 kWh`
- ML MAE reduction: `24.45%`

This is historical replay evidence, not measured production performance.

## Missingness and recovery

Injected-gap tests verify the explicit no-imputation policy.

When a positive gap occurs:

1. missing timestamps are identified;
2. affected pending predictions are quarantined as unscorable;
3. state is reset so pre-gap history cannot leak forward;
4. the runtime starts a new segment;
5. forecasting resumes only after 96 contiguous observations have re-warmed the feature state.

Duplicate and out-of-order timestamps remain fail-closed.

## Retraining validation

The Q3 retraining workflow was validated with target-time partitioning and reproducibility checks.

- training samples: `17,279`
- validation samples: `8,832`
- candidate MAE: `4.0639 kWh`
- persistence MAE: `5.4154 kWh`
- improvement: `24.96%`
- decision: **ACCEPTED**

The reference v1 model was excluded from that comparison because its training target cutoff overlapped the Q3 validation period. This avoids an invalid in-sample reference comparison.

## Monitoring behavior

The monitoring layer is intentionally honest about regime variation. In the final historical 24-hour window, for example:

- ML MAE: `0.9815 kWh`
- persistence MAE: `0.2030 kWh`
- ML win rate: `29.17%`

The 7-day window was different:

- ML MAE: `1.8664 kWh`
- persistence MAE: `2.0194 kWh`
- ML win rate: `34.23%`

These windows are descriptive diagnostics, not tuning targets. They show why the baseline and rolling monitoring remain part of the system.

## Test suite

The latest full suite reports:

```text
106 passed
0 failed
```

Coverage spans ingestion, replay, feature generation, model evaluation, operational feedback, monitoring, missingness recovery, retraining, and the Streamlit UI.

## Limits of the evidence

The project demonstrates a reproducible historical forecasting and ML lifecycle workflow. It does not establish:

- live SCADA integration
- plant-level deployment performance
- network or sensor failure behavior beyond the tested simulations
- distributed-system scalability
- universal superiority over simple baselines across all operating regimes

