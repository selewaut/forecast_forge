# Epic 0005: UI & Experiment Visualization

## Status

Planned

## Objective

Build a web-based user interface for forecast_forge that enables dataset and model selection, run configuration, forecast visualization, and experiment comparison without touching the command line. This addresses the two biggest gaps versus Databricks MMF (which has no UI) and the hardcoded MLflow logging in the current codebase.

## Scope

- Build a Streamlit dashboard with four tabs: Data Preview, Forecast Comparison, Metrics Leaderboard, and Run History.
- Add a Run Configuration form that writes out a YAML config file and submits a forecasting run via `run_forecast()` as a subprocess or Spark submit.
- Implement forecast visualization: overlay forecasts from multiple models per time series, display prediction intervals, zoom to horizon.
- Implement a metrics leaderboard: sortable table of models × metrics (SMAPE, MAE, RMSE) with per-horizon breakdown.
- Query the MLflow tracking server from the dashboard and render forecasting-specific charts (residual diagnostics, forecast vs. actual, residual ACF/PACF).
- Log forecast plots and residual diagnostics automatically as MLflow artifacts via the `ExperimentTracker` (Epic 0004).
- Replace the current inline MLflow calls in `forecaster.py` with the `ExperimentTracker` adapter so all logging is centralized and consistent.

## Tasks

- [ ] [Task 0020: Build Streamlit dashboard shell with tab navigation](../tasks/0020-streamlit-dashboard-shell.md)
- [ ] [Task 0021: Implement run configuration form and submission](../tasks/0021-run-config-form.md)
- [ ] [Task 0022: Implement forecast comparison view](../tasks/0022-forecast-comparison-view.md)
- [ ] [Task 0023: Implement metrics leaderboard](../tasks/0023-metrics-leaderboard.md)
- [ ] [Task 0024: Integrate MLflow experiment browser](../tasks/0024-mlflow-experiment-browser.md)
- [ ] [Task 0025: Add automatic forecast plot logging as MLflow artifacts](../tasks/0025-automatic-plot-logging.md)
- [ ] [Task 0026: Replace inline MLflow calls with ExperimentTracker adapter](../tasks/0026-mlflow-adapter-migration.md)

## Decisions

- [ ] [ADR 0008: Streamlit as UI framework](../ADRs/0008-streamlit-ui-framework.md)
- [ ] [ADR 0009: MLflow artifact organization for forecasting plots](../ADRs/0009-mlflow-artifact-organization.md)

## Notes

Streamlit is recommended over alternatives (Gradio, Dash, custom Flask) because:
- Minimal boilerplate for data-science UIs — one Python file can produce a full dashboard.
- Native integration with Pandas, Plotly, and Altair for forecast visualization.
- Easy to extend with custom components if needed.

MLflow integration should query the existing tracking server (not require a separate database). The `ExperimentTracker` abstraction from Epic 0004 will provide a clean API for both the CLI and the UI to log and retrieve experiment data.

**Dependency:** This epic depends on Epic 0004 (specifically the `ExperimentTracker` adapter, Task 0017) for clean MLflow integration. Task 0026 (replacing inline MLflow calls) should be sequenced after the adapter exists.

**Testing:** The dashboard can be smoke-tested with a known MLflow experiment from a previous run. Use Playwright or Selenium for optional UI-level tests on the Streamlit app.
