# Model Lifecycle

ForgeCast treats the forecast model as a versioned artifact moving through a controlled loop: define the forecast contract, build features causally, evaluate chronologically, operate with delayed feedback, monitor against a simple baseline, and evaluate retraining candidates without silently replacing the active artifact.

## Forecast Contract

The prediction task is fixed:

```text
telemetry available through t -> forecast Usage(t + 15 min)
```

The target is always one 15-minute interval ahead. Each `PredictionEvent` stores the origin timestamp, target timestamp, prediction timestamp, model version, ML forecast, and persistence baseline. The target timestamp must equal the origin plus 15 minutes.

The primary model uses 19 features: historical Usage lags and rolling Usage statistics plus target-time calendar variables and cyclical encodings. Contemporaneous target-interval physical measurements are not used because they are unavailable at the forecast cutoff. `Load_Type` is outside the primary feature contract.

## Feature and Model Pipeline

The feature generator maintains 96 contiguous Usage observations in memory. It emits no feature vector until warm-up is complete. The resulting features are validated against a fixed canonical column order before fitting or inference.

The active estimator is scikit-learn `HistGradientBoostingRegressor` with frozen hyperparameters:

- `loss="squared_error"`
- `learning_rate=0.05`
- `max_iter=200`
- `max_leaf_nodes=31`
- `l2_regularization=1.0`
- `random_state=42`
- `early_stopping=False`

`early_stopping=False` removes an internal non-chronological validation split from model fitting.

## Offline Evaluation

Model evaluation is chronological rather than random. Three expanding training/evaluation windows each contain 3,504 held-out observations. Persistence is the primary baseline because it is directly available from the forecast origin:

```text
persistence forecast = Usage(t)
```

The frozen v1 artifact achieved:

| Metric | ML | Persistence |
| --- | ---: | ---: |
| Pooled MAE | 3.8626 kWh | 5.3688 kWh |
| Pooled RMSE | 8.2464 kWh | 12.1538 kWh |

Across the 10,512 pooled held-out observations, ML MAE was 28.05% lower than persistence.

The final chronological window also provides the clearest post-training evidence: 3,504 held-out intervals with ML MAE of 3.2133 kWh versus 4.2532 kWh for persistence, a 24.45% reduction.

## Operational Inference

The runtime loop is:

```text
validate telemetry
    -> resolve prior prediction feedback
    -> update monitoring
    -> update feature state
    -> generate next-target features
    -> predict
    -> register PredictionEvent
```

A prediction is not scored immediately because the target is not yet known. The runtime therefore separates prediction registration from feedback completion.

## Delayed Feedback

When the next target telemetry arrives, `DelayedFeedbackTracker` searches by exact logical `target_timestamp`. A successful match becomes a `FeedbackRecord` containing the actual Usage, ML forecast, persistence baseline, and both absolute errors.

Duplicate completed targets fail closed. Unmatched actuals are tracked rather than converted into synthetic model scores. Predictions whose target falls inside a detected telemetry gap are quarantined and never enter the metric accumulators.

## Monitoring

Monitoring consumes only completed feedback. It maintains cumulative metrics plus bounded 24-hour and 7-day windows, corresponding to 96 and 672 fifteen-minute observations.

Tracked quantities include ML MAE/RMSE, persistence MAE/RMSE, relative improvement, ML win rate, model version, and target-time bounds. Signed residual and error-concentration diagnostics are descriptive diagnostics, not automated tuning objectives.

The active model's training target boundary is also recorded so that post-training performance can be separated from full-replay historical performance.

## Retraining and Promotion

Retraining is organized around explicit historical checkpoints. A sample enters training only when its `target_timestamp` is at or before the checkpoint cutoff. Validation samples must fall inside the subsequent target-time evaluation window.

The Q3 candidate run used:

- training cutoff: `2018-06-30 23:45`
- training samples: 17,279
- validation samples: 8,832
- candidate MAE: 4.0639 kWh
- persistence MAE: 5.4154 kWh
- improvement vs persistence: 24.96%

The promotion policy requires at least 5.0% improvement over persistence and permits no regression against a valid reference model. When the existing v1 artifact would be an in-sample reference for the Q3 window, that reference comparison is explicitly excluded.

The Q3 candidate passed the implemented promotion gate and is stored as `v2` candidate evidence. It does not replace the active v1 artifact used by the hosted replay.

## Model Artifacts

The active artifact is `artifacts/models/forgecast_v1_replay_ready.pkl` with companion metadata. Metadata records the model version, feature contract, training boundaries, training sample count, dataset identity, and runtime versions.

Candidate artifacts follow the same pattern and are kept separately. Predictions retain the generating model version, supporting later attribution and lifecycle analysis.

The current system is a controlled historical lifecycle workflow, not a claim of continuous automated retraining or enterprise model serving.