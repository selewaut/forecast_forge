# Epic 0008: UI Dashboard

## Status

Planned

## Objective

Build a lightweight Streamlit dashboard for exploring forecasts and comparing experiment results by browsing MLflow runs — no model training from the UI, just visualization and analysis.

## Scope

- Build a Streamlit dashboard with: experiment/run selector, forecast vs. actual overlay chart (with prediction intervals), metrics comparison table, and residual diagnostics.
- Query the MLflow tracking server to display run history and logged artifacts.
- Keep the dashboard read-only for now (no run submission form).

## Tasks

- [ ] [Task 0018: Build Streamlit dashboard with MLflow browsing](../tasks/0018-streamlit-mlflow-dashboard.md)

## Decisions

- [ ] [ADR 0008: Streamlit as UI framework](../ADRs/0008-streamlit-ui-framework.md)

## Notes

This is intentionally minimal — a single Streamlit app that browses existing MLflow experiments and visualizes forecasts. Run configuration and submission are deferred to avoid scope creep.

Depends on Epic 0007 (MLflow experiment tracking) so there are well-structured artifacts and metrics to display.
