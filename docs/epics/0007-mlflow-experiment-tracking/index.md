# Epic 0007: MLflow Experiment Tracking

## Status

Planned

## Objective

Standardize MLflow logging across the pipeline so every run captures consistent, forecasting-specific metadata and visualizations. No abstraction layer — build directly on MLflow per PR feedback.

## Scope

- Centralize MLflow calls from `forecaster.py` into well-defined helper functions (not an adapter/abstraction).
- Standardize logged metadata: dataset hash, model name, config params, horizon, frequency, backtest strategy, per-horizon metrics, and aggregate metrics.
- Log forecast vs. actual plots and residual diagnostics (ACF/PACF) as MLflow artifacts automatically.
- Remove the scattered inline `mlflow.log_metric` / `mlflow.set_tag` calls and replace with a consistent logging flow.

## Tasks

- [ ] [Task 0016: Centralize and standardize MLflow logging](tasks/0016-mlflow-logging-standardization.md)
- [ ] [Task 0017: Log forecast plots and residual diagnostics as artifacts](tasks/0017-forecast-plot-logging.md)

## Decisions

- [ ] [ADR 0007: MLflow artifact organization for forecasting runs](../../ADRs/0007-mlflow-artifact-organization.md)

## Notes

Reference: [Time Series Forecasting and Experiment Tracking with MLflow](https://medium.com/@pavansingu007/time-series-forecasting-and-experiment-tracking-with-mlflow-9e12cc31f9c1)

No `ExperimentTracker` abstraction is built — we use MLflow directly. A future swap to Neptune/W&B would be a separate effort.

Depends on Epic 0006 (backtesting decoupling) so the logging layer works against the new `BacktestEngine`.
