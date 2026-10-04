# Model Lifecycle

ForgeCast follows the model from its forecast definition through evaluation, operation, feedback, monitoring, and retraining. The important point is that the forecast is treated as an event with a future label, not as an immediate score.

## Forecast Definition

```text
Telemetry available through t
        ↓
Forecast Usage(t + 15 min)
        ↓
Receive actual Usage(t + 15 min)
        ↓
Compute ML vs persistence error
```

The forecast horizon is fixed at 15 minutes. The prediction object records the origin time, target time, model version, model output, persistence baseline, and feature provenance.

## Feature Generation

The stateful feature generator requires 96 contiguous observations before it produces the first feature vector. The active feature contract contains 19 values:

- historical Usage lags and rolling means;
- target-time hour, quarter slot, day-of-week, weekend, month, and seconds-from-midnight features;
- cyclical encodings for time of day, day of week, and month.

Target-interval physical measurements are excluded because they are not available when the forecast is made. `Load_Type` is also outside the primary model contract.

## Training

The active estimator is scikit-learn `HistGradientBoostingRegressor` with frozen settings:

| Parameter | Value |
| --- | --- |
| `loss` | `squared_error` |
| `learning_rate` | `0.05` |
| `max_iter` | `200` |
| `max_leaf_nodes` | `31` |
| `l2_regularization` | `1.0` |
| `random_state` | `42` |
| `early_stopping` | `False` |

Disabling `early_stopping` keeps model fitting from creating a hidden internal validation split that would not respect the project's chronological evaluation design.

## Chronological Evaluation

Instead of a random split, ForgeCast uses three expanding time-ordered evaluation windows. Each window contains 3,504 held-out observations.

| Fold | ML MAE | Persistence MAE | ML improvement |
| --- | ---: | ---: | ---: |
| 1 | 3.7735 kWh | 5.2173 kWh | 27.67% |
| 2 | 4.6011 kWh | 6.6360 kWh | 30.67% |
| 3 | 3.2133 kWh | 4.2532 kWh | 24.45% |
| **Pooled** | **3.8626 kWh** | **5.3688 kWh** | **28.05%** |

The pooled comparison covers 10,512 held-out observations. Fold 3 is also the final post-training replay window for the active v1 artifact.

## Operational Inference

At runtime, the system validates the next telemetry record, resolves feedback for the interval that just closed, updates monitoring, advances the feature state, and generates the next prediction when the history is warm.

A forecast is stored before its label exists. This is why prediction registration and feedback completion are separate parts of the lifecycle.

## Delayed Feedback

When the target interval arrives, the feedback tracker matches it by exact logical target timestamp. The completed record contains:

- actual Usage;
- the original ML forecast;
- the persistence forecast;
- ML absolute error and persistence absolute error;
- the model version and temporal indices.

Duplicate completed targets fail closed. Unmatched actuals are tracked rather than converted into synthetic scores. A prediction whose target falls inside a detected telemetry gap is invalidated instead of scored.

## Monitoring

Monitoring consumes completed feedback only. It keeps cumulative metrics and bounded windows for the most recent 96 observations (about 24 hours) and 672 observations (about 7 days).

Core measures include ML MAE/RMSE, persistence MAE/RMSE, MAE improvement, and ML win rate. Signed residual and error-concentration diagnostics are used for description, not for tuning the active model on the same evidence.

## Retraining

Retraining is checkpoint-based rather than continuously automatic. Samples are partitioned by `target_timestamp`:

```text
training target <= checkpoint cutoff
validation start <= target <= validation end
```

The Q3 candidate used a `2018-06-30 23:45` training target cutoff, 17,279 training samples, and 8,832 validation samples. Candidate MAE was 4.0639 kWh versus 5.4154 kWh for persistence, a 24.96% improvement.

## Candidate Evaluation and Promotion

The implemented promotion policy requires at least 5.0% improvement over persistence. A reference-model comparison is used only when that reference is temporally valid; otherwise it is excluded rather than treated as evidence.

The Q3 `v2` candidate passed the implemented gate and was saved as a candidate artifact. The hosted replay still uses the active `v1` artifact. Acceptance of a candidate is therefore separate from silently replacing the deployed demonstration model.

## Artifacts and Traceability

The active model is stored as `artifacts/models/forgecast_v1_replay_ready.pkl` with companion metadata. The metadata records model version, feature contract, training boundaries, training sample count, dataset identity, and runtime versions.

Candidate models use the same artifact-plus-metadata pattern. Because predictions carry their model version, later feedback can be traced back to the artifact that generated it.

The current lifecycle is a controlled historical workflow. It demonstrates candidate evaluation and model traceability, but it is not continuous production retraining.