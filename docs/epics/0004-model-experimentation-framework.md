# Epic 0004: Model Experimentation Framework

## Status

Planned

## Objective

Extend forecast_forge beyond local statistical models to support global deep learning models, foundation time-series models, hyperparameter search, and structured experiment tracking. This closes the gap with Databricks MMF's three-tier model system (local, global, foundation) and adds capabilities MMF lacks (walk-forward strategy options, an experiment tracker adapter).

## Scope

- Add support for "global" models from NeuralForecast (NBEATSx, NHITS, LSTM, PatchTST, TiDE), training one model across all time series with GPU support.
- Add support for "foundation" models: Chronos-Bolt, Chronos-2, TimesFM 2.5 for zero-shot inference.
- Implement hyperparameter search via config (Optuna-based, like MMF "Auto" variants).
- Create an `ExperimentTracker` abstraction wrapping MLflow with a clean interface, enabling future swap to Neptune/W&B.
- Standardize logged metadata per run: dataset hash, model config, horizon, frequency, CV strategy, per-horizon metrics, residual plots, forecast vs. actual plots.
- Decouple backtesting from model classes into a standalone `BacktestEngine` supporting expanding window, sliding window, and time-series cross-validation.
- Add cross-framework model registration: a model entry in YAML should work regardless of whether it comes from StatsForecast, NeuralForecast, Darts, or PyTorch Forecasting.
- Add typed exception hierarchy (ported from MMF's `exceptions.py`) for debuggable pipeline failures.

## Tasks

- [ ] [Task 0013: Decouple backtesting into standalone BacktestEngine](../tasks/0013-backtest-engine.md)
- [ ] [Task 0014: Add support for global models (NeuralForecast)](../tasks/0014-global-models.md)
- [ ] [Task 0015: Add support for foundation models (Chronos, TimesFM)](../tasks/0015-foundation-models.md)
- [ ] [Task 0016: Implement hyperparameter search via config](../tasks/0016-hyperparameter-search.md)
- [ ] [Task 0017: Build ExperimentTracker abstraction over MLflow](../tasks/0017-experiment-tracker-abstraction.md)
- [ ] [Task 0018: Add cross-framework model registration support](../tasks/0018-cross-framework-models.md)
- [ ] [Task 0019: Add typed exception hierarchy](../tasks/0019-typed-exception-hierarchy.md)

## Decisions

- [ ] [ADR 0005: Experiment tracker adapter design](../ADRs/0005-experiment-tracker-design.md)
- [ ] [ADR 0006: Backtest engine walk-forward strategy design](../ADRs/0006-backtest-engine-design.md)
- [ ] [ADR 0007: Global model GPU execution strategy](../ADRs/0007-global-model-gpu-strategy.md)

## Notes

MMF's three-tier model system is the reference: local (per-series, StatsForecast/SKTime), global (cross-series, NeuralForecast), foundation (zero-shot, Chronos/TimesFM). This epic targets full parity with MMF's model catalog plus the following improvements:

- MMF logs MLflow metrics/artifacts inline throughout `forecaster.py`. Our `ExperimentTracker` adapter will centralize this.
- MMF only supports expanding-window backtesting. We'll support expanding, sliding, and time-series CV by making the strategy a config parameter.
- MMF has a rich exception hierarchy (30+ types under `MMFError`); the current codebase uses bare `raise Exception`. Task 0019 adds typed exceptions for debugging and pipeline observability.
- MMF's data quality checks have been moved to Epic 0003 (Task 0012) since they validate data before it reaches modeling.

**Dependency:** The `BacktestEngine` decoupling (Task 0013) should come first — models should be pure `fit`/`predict` containers without embedded backtesting logic. This unblocks all subsequent model work.

**Testing:** Each new model framework (NeuralForecast, Chronos, TimesFM) needs an integration test with a small synthetic multi-series dataset. The `ExperimentTracker` needs unit tests verifying correct metric/artifact logging for a known run.
