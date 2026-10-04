# Model Lifecycle

ForgeCast treats the forecasting model as a versioned artifact that moves through evaluation, operation, monitoring, and retraining without losing temporal traceability.

## Active model

**v1** is the active replay model.

- Estimator: `HistGradientBoostingRegressor`
- Horizon: 15 minutes
- Feature contract: 19 Usage + Calendar features
- Baseline: persistence (`ŷ(t+1) = Usage(t)`)
- `early_stopping=False` to avoid hidden non-chronological validation inside model fitting

The serialized artifact and its metadata are the runtime source of truth.

## Training and evaluation

The forecasting problem is evaluated chronologically rather than with a random split.

The evaluation harness uses three time-ordered windows and reports both fold-level and pooled metrics. Final evaluation results are not used to tune the model, features, thresholds, or architecture.

The strongest current evidence is:

| Measure | ML | Persistence |
|---|---:|---:|
| Pooled MAE | 3.8626 kWh | 5.3688 kWh |
| Pooled RMSE | 8.2464 kWh | 12.1538 kWh |

The pooled evaluation contains 10,512 held-out observations.

## Artifact metadata

The model metadata records the provenance needed to interpret the artifact, including the model version, training boundary, feature contract, runtime information, and source-data identity.

Predictions retain the model version that generated them, so later feedback can be attributed to the correct model.

## Operational feedback

The runtime follows a fixed loop:

```text
observe telemetry
    ↓
validate
    ↓
update state
    ↓
predict next interval
    ↓
store prediction event
    ↓
receive next actual
    ↓
pair by target timestamp
    ↓
compute residual
    ↓
update monitoring
```

This makes model performance observable shortly after each prediction because the target becomes available one interval later.

## Retraining

Retraining uses explicit target-time boundaries. A sample belongs to training only when its ground-truth target is at or before the training cutoff.

The Q3 candidate run used:

- training cutoff: `2018-06-30 23:45`
- validation targets: `2018-07-01 00:00` through `2018-09-30 23:45`
- training samples: `17,279`
- validation samples: `8,832`
- candidate MAE: `4.0639 kWh`
- persistence MAE: `5.4154 kWh`
- improvement vs persistence: `24.96%`
- gate decision: **ACCEPTED**

The candidate is stored separately from the active v1 artifact. Acceptance does not silently replace the active model.

## Promotion guardrail

The implemented gate requires:

- minimum improvement over persistence: **5.0%**
- maximum allowed regression against a valid reference: **0.0%**
- valid artifact and temporal checks

A reference-model comparison is excluded when the reference training target overlaps the candidate validation window; this prevents an in-sample comparison from being presented as evidence.

## Monitoring and lifecycle decisions

Residual performance is the primary lifecycle signal because actual labels arrive quickly. Input drift can be supplementary evidence, but it is not treated as proof of model degradation on its own.

The project uses periodic candidate evaluation rather than continuous automatic retraining. No numeric production retraining trigger is claimed beyond the implemented checkpoint workflow.

## Rollback and traceability

Model versions are kept as separate artifacts. The system can identify which model version generated a prediction, and candidate artifacts do not overwrite the active v1 artifact.

