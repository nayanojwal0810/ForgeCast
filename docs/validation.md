# Validation

The question this document answers is: **How do we know the forecasting system is behaving correctly, and how do we know the reported model improvement is honest?**

## Validation Scope

ForgeCast is checked at three levels:

1. **Model performance** — chronological holdout performance and comparison against persistence.
2. **System correctness** — tests for feature contracts, timestamps, state, feedback, and temporal boundaries.
3. **Operational behavior** — full historical replay and fault-injection tests for missing telemetry.

Good holdout metrics support the model claim; passing lifecycle tests support the system claim; neither alone proves production readiness.

## Model Performance

The active v1 configuration was evaluated on three expanding chronological holdout windows of 3,504 observations.

| Measure | ML | Persistence |
| --- | ---: | ---: |
| Pooled MAE | **3.8626 kWh** | 5.3688 kWh |
| Pooled RMSE | **8.2464 kWh** | 12.1538 kWh |

Across 10,512 held-out observations, ML MAE was 28.05% lower than persistence. The three fold-level improvements were 27.67%, 30.67%, and 24.45%.

The third fold is the final post-training holdout for the active v1 artifact: 3.2133 kWh MAE for ML versus 4.2532 kWh for persistence, a 24.45% reduction.

These are historical out-of-sample results. They are not measurements from a live plant deployment.

## System Correctness

### Time and Leakage

The central temporal invariant is:

```text
feature_source_time <= forecast_cutoff < target_time
```

The primary feature set uses historical Usage and target-time calendar information. Contemporaneous target-interval physical measurements are excluded, `target_usage_kwh` cannot enter the feature matrix, and feature order is checked against the canonical 19-feature contract.

Retraining partitions are separated by `target_timestamp`, and the model configuration disables internal `early_stopping`.

The dataset's daily `00:00` closing-row convention is also tested because naïve timestamp sorting would break the 15-minute sequence around midnight.

### State and Feedback

The feature generator requires 96 contiguous observations before the first prediction. Pending forecasts are matched to actuals by exact target timestamp. Duplicate and out-of-order delivery is rejected.

## Operational Behavior

A clean replay of the complete 35,040-row dataset produced:

| Operational check | Result |
| --- | ---: |
| Telemetry consumed | 35,040 |
| Predictions generated | 34,945 |
| Completed feedback | 34,944 |
| Pending at end | 1 |
| Unmatched feedback events | 0 |
| Duplicate feedback events | 0 |
| Sequence failures | 0 |
| Gap incidents | 0 |
| Invalidated predictions | 0 |

The one pending prediction is expected: the final source row produces a forecast for a target interval that is outside the available dataset.

## Missing Telemetry Tests

Fault-injection tests remove one or more intervals from a short replay slice. The recovery path detects the missing interval, quarantines predictions whose targets are unobservable, resets feature state, starts a new segment, and waits for 96 contiguous observations before forecasting again.

No Usage value is imputed. Tests also cover duplicate and out-of-order delivery, state isolation across the gap, feedback isolation, and midnight gap handling.

## Retraining Validation

The 2018 Q3 candidate was evaluated using target-time partitions:

| Quantity | Result |
| --- | ---: |
| Training samples | 17,279 |
| Validation samples | 8,832 |
| Candidate MAE | 4.0639 kWh |
| Persistence MAE | 5.4154 kWh |
| Improvement | 24.96% |
| Gate result | **ACCEPTED** |

The existing v1 artifact was not used as a reference for this Q3 comparison because its training target boundary overlaps the Q3 evaluation period. Treating it as a reference would make the comparison in-sample.

## Automated Validation

The repository contains nine focused test modules covering ingestion contracts, replay order, temporal boundaries, causal feature generation, model validation, operational feedback, monitoring, missing-data recovery, retraining, and Streamlit UI behavior.

Run the full suite locally with:

```bash
python -m pytest
```

The documentation does not rely on a fixed pass-count claim; the executable test suite is the source of truth for the current result.

## Operational Diagnostics

Short windows do not always favor the ML model. In the final 24-hour diagnostic, ML MAE was 0.9815 kWh while persistence was 0.2030 kWh, with an ML win rate of 29.17%. The 7-day window favored ML on aggregate: 1.8664 kWh MAE versus 2.0194 kWh for persistence.

These diagnostics are descriptive. They are not used to tune the frozen model or redefine the evaluation set.

## Limits of the Evidence

The repository demonstrates a reproducible historical forecasting and ML lifecycle workflow. It does not prove live plant performance, distributed-system scalability, business impact, probabilistic forecast quality, or universal superiority over persistence.
