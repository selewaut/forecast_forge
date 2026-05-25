# Epic 0006: Backtesting Decoupling

## Status

Planned

## Objective

Extract walk-forward validation from model classes into a standalone `BacktestEngine` so models become pure `fit`/`predict` containers and backtesting strategies are configurable.

## Scope

- Decouple `backtest()` and `calculate_metrics()` from `ForecastingRegressor` into a standalone `BacktestEngine`.
- Support expanding window and sliding window strategies as a config parameter.
- Add typed exception hierarchy for debuggable pipeline failures.
- Clean up the `Forecaster.score_models()` and `evaluate_models()` methods to use the new engine.

## Tasks

- [ ] [Task 0014: Decouple backtesting into standalone BacktestEngine](tasks/0014-backtest-engine.md)
- [ ] [Task 0015: Add typed exception hierarchy](tasks/0015-typed-exception-hierarchy.md)

## Decisions

- [ ] [ADR 0006: Backtest engine walk-forward strategy design](../../ADRs/0006-backtest-engine-design.md)

## Notes

Currently `backtest()` and `calculate_metrics()` live inside `abstract_model.py`, making models impure and hard to test. The `BacktestEngine` extracts this into a single reusable component.

This epic does not add new model types (NeuralForecast, Chronos, TimesFM are deferred indefinitely per PR feedback).
