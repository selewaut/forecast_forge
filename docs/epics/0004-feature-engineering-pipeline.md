# Epic 0004: Feature Engineering Pipeline

## Status

Planned

## Objective

Abstract feature engineering into a configurable, composable pipeline so that dataset-specific transforms are modular and reusable, and generic time-series features can be added through YAML configuration.

## Scope

- Create a `FeaturePipeline` abstraction: a chain of configurable transformers, each registered by name and parameterized via YAML.
- Move Walmart-specific feature transforms (temperature bins, markdown imputation, week number) into a `walmart/` feature module.
- Implement built-in generic transformers: calendar features (day-of-week, month, quarter, holiday flags via `holidays` library), lagged values, rolling window statistics, and Fourier terms for seasonality.

## Tasks

- [ ] [Task 0009: Implement pluggable FeaturePipeline abstraction](../tasks/0009-feature-pipeline-abstraction.md)
- [ ] [Task 0010: Build generic feature transformers (calendar, lags, rolling, Fourier)](../tasks/0010-generic-feature-transformers.md)

## Decisions

- [ ] [ADR 0004: Feature pipeline design — composable transformer chain pattern](../ADRs/0004-feature-pipeline-design.md)

## Notes

MMF has no feature engineering abstraction — it assumes raw `unique_id`/`ds`/`y` columns. This epic is an opportunity to differentiate while keeping scope minimal.

Depends on Epic 0003 (data loading) for the schema adapter that feeds normalized DataFrames into the feature pipeline.
