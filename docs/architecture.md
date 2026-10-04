# Architecture

ForgeCast is a stateful, causal forecasting pipeline built around a fixed 15-minute prediction cycle.

## System flow

```text
Telemetry replay
      ↓
Schema + value validation
      ↓
Temporal continuity validation
      ↓
96-observation state
      ↓
Causal feature generation
      ↓
Frozen ML model + persistence baseline
      ↓
Prediction event
      ↓
Delayed feedback pairing
      ↓
Residual and rolling metrics
      ↓
Retraining candidate
      ↓
Versioned artifact
```

The Streamlit application sits on top of these components as a read-only demonstration surface. It does not reimplement feature engineering, prediction, or monitoring calculations.

## Forecast contract

For an observation at time `t`, ForgeCast predicts `Usage_kWh(t + 15 min)`.

The hard temporal rule is:

```text
feature_source_time <= forecast_cutoff < target_time
```

The primary feature contract contains 19 values:

- recent `Usage_kWh` lags at the forecast origin
- 4-step and 96-step rolling means
- target-time calendar features known in advance
- cyclical encodings for time of day, day of week, and month

Target-interval physical measurements such as contemporaneous reactive power, power factor, and CO2 are excluded. `Load_Type` is also outside the primary contract.

## Stateful inference

The feature generator maintains a 96-observation contiguous history in memory. This supports the longest lag and 24-hour rolling feature without rebuilding the full dataset for every prediction.

The replay layer emits one validated record at a time. Future rows are never loaded into the feature state before their logical arrival.

## Data quality and failure handling

Validation is fail-closed at the state boundary.

- Missing, duplicate, or out-of-order intervals are detected explicitly.
- A positive gap can enter the configured recovery path.
- Missing measurements are **not imputed**.
- Pending predictions whose actual target was never observed are quarantined instead of being scored.
- Feature state is reset after a gap and the system re-warms for 96 contiguous observations before forecasting resumes.

This prevents stale history from contaminating post-gap features.

## Prediction and feedback

Each prediction carries the information needed for later audit, including:

- model version
- origin timestamp
- target timestamp
- prediction timestamp
- ML forecast
- persistence baseline
- feature values / origin metadata

When the target interval arrives, the matching prediction is paired by target timestamp. Residuals are then computed as:

```text
error     = actual - prediction
abs_error = |actual - prediction|
```

The baseline error is retained alongside the ML error.

## Monitoring

The operational monitor tracks:

- completed feedback count
- ML MAE / RMSE
- persistence MAE / RMSE
- ML-vs-baseline improvement
- ML win rate
- 24-hour and 7-day rolling windows
- active model version
- gap and invalidation counts
- sequence failures

Monitoring uses completed ground-truth pairs. Unobserved targets are not turned into synthetic errors.

## Model and artifacts

The active model is a serialized `HistGradientBoostingRegressor` artifact with companion metadata capturing the model version and training boundary.

Model lifecycle state is represented through versioned repository artifacts rather than an external registry. v1 remains the active replay model; later candidates are evaluated and stored separately.

## Retraining boundary

Retraining partitions are defined by `target_timestamp`, not `origin_timestamp`:

```text
training target <= training cutoff
validation start <= target <= validation end
```

This matters at one-step boundaries because an origin at the end of the training period can legitimately predict the first validation target. Partition membership therefore follows the ground-truth target itself.

## Deployment surface

The browser UI is implemented with Streamlit and uses repository-relative paths. It provides a bounded replay demonstration, current prediction/feedback state, rolling metrics, historical evaluation evidence, retraining status, and operational health.

The project deliberately does not require:

- a database
- Kafka or Spark
- Kubernetes
- a separate REST API
- a background daemon
- a distributed model-serving stack

Those components would add complexity without representing a requirement of this workload.

