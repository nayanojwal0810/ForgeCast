# Validation

ForgeCast is validated as a temporal ML system. The evaluation covers data contracts, causal feature construction, chronological model comparison, prediction/feedback pairing, missingness recovery, monitoring behavior, retraining boundaries, and the hosted demonstration path.

## Validation Scope

Validation is designed to catch both model-quality errors and system-level temporal errors. The most important invariant is:

```text
feature_source_time <= forecast_cutoff < target_time
```

A prediction uses only information logically available by its forecast cutoff. Delayed labels are handled separately after the target interval closes.

## Dataset

The project uses the UCI Steel Industry Energy Consumption dataset. The source file contains 35,040 telemetry rows at a nominal 15-minute cadence and 11 source fields.

The implementation preserves source row order and applies a logical timestamp rule for the daily `00:00` closing record so that the 15-minute sequence remains continuous across midnight. This avoids corrupting the sequence by naively sorting the raw clock values.

## Leakage Controls

Several controls are enforced in code and tested explicitly:

- Contemporaneous target-interval electrical measurements and target-derived CO2 are excluded from the primary 15-minute feature contract.
- The primary feature set uses historical Usage lags/rollups and calendar information known for the target interval.
- The stateful generator requires 96 contiguous observations before the first forecast.
- The model wrapper rejects missing, extra, or misordered features and rejects target leakage through `target_usage_kwh`.
- Retraining partitions are defined by `target_timestamp`, with explicit checks for temporal separation and disjoint train/validation target sets.
- The model configuration disables internal `early_stopping` so model fitting does not introduce a hidden non-chronological split.

These controls are structural. They do not depend on test-set performance to choose features or hyperparameters.

## Chronological Evaluation

The frozen v1 model was evaluated over three expanding chronological windows of 3,504 observations each.

| Fold | ML MAE | Persistence MAE | ML MAE reduction |
| --- | ---: | ---: | ---: |
| 1 | 3.7735 kWh | 5.2173 kWh | 27.67% |
| 2 | 4.6011 kWh | 6.6360 kWh | 30.67% |
| 3 | 3.2133 kWh | 4.2532 kWh | 24.45% |
| **Pooled** | **3.8626 kWh** | **5.3688 kWh** | **28.05%** |

The pooled result covers 10,512 held-out observations. The third fold is also the post-training replay window for the active v1 artifact.

## Operational Validation

A full historical replay exercised the prediction-to-feedback lifecycle over all 35,040 telemetry records.

| Operational result | Count |
| --- | ---: |
| Telemetry consumed | 35,040 |
| Predictions generated | 34,945 |
| Completed feedback records | 34,944 |
| Pending prediction at end | 1 |
| Unmatched feedback events | 0 |
| Duplicate feedback events | 0 |
| Sequence failures | 0 |
| Gap incidents in clean replay | 0 |
| Invalidated predictions in clean replay | 0 |

The single pending prediction is expected because the last source observation creates one forecast whose 15-minute target lies beyond the available dataset.

The post-training replay window contains 3,504 completed feedback records and reproduces the third chronological evaluation result: 3.2133 kWh ML MAE versus 4.2532 kWh persistence MAE.

This is historical replay evidence. It is not measured plant production performance.

## Missing Telemetry

Injected-gap tests exercise the recovery policy. A positive gap is detected before stale state is reused. Affected pending predictions are quarantined as unscorable, feature history is reset, and the next segment must re-warm on 96 contiguous observations.

There is no imputation step in this recovery path. Duplicate and out-of-order timestamps remain fail-closed.

Tests also cover midnight gap handling, state-reset isolation, feedback isolation, and the requirement that pre-gap values do not contaminate post-gap rolling features.

## Retraining Validation

The Q3 candidate was trained and validated using target-time partitions:

| Quantity | Result |
| --- | ---: |
| Training samples | 17,279 |
| Validation samples | 8,832 |
| Candidate MAE | 4.0639 kWh |
| Persistence MAE | 5.4154 kWh |
| Improvement vs persistence | 24.96% |
| Promotion gate | ACCEPTED |

The v1 reference model was excluded for this checkpoint because its training target boundary extends into the Q3 evaluation period. Presenting that comparison would make the result in-sample.

## Test Evidence

A full local test run recorded **106 passed** across ingestion, replay, feature generation, temporal checks, model evaluation, operational feedback, monitoring, missingness recovery, retraining, and Streamlit UI tests.

The suite includes dedicated checks for the midnight timestamp convention, one-step target alignment, future-row protection, duplicate delivery, skipped intervals, gap recovery, target-based retraining boundaries, artifact validation, and the UI's completed-forecast feedback alignment.

## Limitations

The evidence is strong for the implemented historical workflow, but it does not establish live plant deployment performance. The project uses a single site/dataset, one forecast horizon, a compact feature universe, local file artifacts, and controlled historical replay. It does not measure business impact, provide probabilistic uncertainty, or implement continuous production retraining and distributed serving.

The recent 24-hour diagnostic also shows why point-in-time metrics should be interpreted carefully: some short windows favor persistence even when pooled chronological evidence favors ML.