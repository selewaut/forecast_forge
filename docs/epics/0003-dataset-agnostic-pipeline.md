# Epic 0003: Dataset-Agnostic Pipeline

## Status

Planned

## Objective

Decouple the forecasting pipeline from any single dataset so that new datasets can be onboarded through configuration alone — no code changes. This is the primary architectural differentiator versus Databricks MMF, which is tied to Delta tables and the Databricks runtime.

## Scope

- Define a `BaseDataLoader` interface that supports CSV, Parquet, Delta, SQL, and API sources through a config-driven schema mapping.
- Create a pluggable `FeaturePipeline` abstraction with built-in generic transformers (calendar features, lagged values, rolling windows, Fourier terms, STL decomposition).
- Build a schema adapter that maps arbitrary column names to internal conventions (`unique_id`, `ds`, `y`), with support for exogenous regressors.
- Create frequency-specific config presets (`forecasting_conf_daily.yaml`, `_weekly.yaml`, `_monthly.yaml`, `_hourly.yaml`) with appropriate defaults.
- Move Walmart-specific data loading and feature engineering into a dedicated `walmart/` module, implementing the new interfaces.
- Add a `datasets/` config directory where each dataset declares its loader, schema map, and feature pipeline.
- Port data quality checks from Databricks MMF (`DataQualityChecks`) as a mandatory pre-modeling stage: configuration validation, missing date detection, negative value threshold, training-length ratio check, and external regressor null checks.

## Tasks

- [ ] [Task 0006: Define BaseDataLoader interface and schema adapter](../tasks/0006-base-dataloader-interface.md)
- [ ] [Task 0007: Refactor Walmart data loading into adapter module](../tasks/0007-walmart-loader-adapter.md)
- [ ] [Task 0008: Implement pluggable FeaturePipeline abstraction](../tasks/0008-feature-pipeline-abstraction.md)
- [ ] [Task 0009: Build generic feature transformers (calendar, lags, rolling, Fourier)](../tasks/0009-generic-feature-transformers.md)
- [ ] [Task 0010: Create frequency-specific config presets](../tasks/0010-freq-specific-configs.md)
- [ ] [Task 0011: Add datasets config directory and schema validation](../tasks/0011-datasets-config-directory.md)
- [ ] [Task 0012: Port data quality checks from MMF](../tasks/0012-data-quality-checks.md)

## Decisions

- [ ] [ADR 0003: DataLoader interface design — three-source pattern (pandas, Spark, file path)](../ADRs/0003-dataloader-interface-design.md)
- [ ] [ADR 0004: Feature pipeline design — composable transformer chain pattern](../ADRs/0004-feature-pipeline-design.md)

## Notes

This epic draws inspiration from the Databricks MMF `resolve_source()` pattern, which already supports `Union[str, pd.DataFrame, DataFrame]`. The key addition is file-path and SQL-source support that MMF lacks, which is essential for running outside Databricks.

MMF has no feature engineering abstraction — it assumes raw `unique_id`/`ds`/`y` columns. This epic's `FeaturePipeline` is an opportunity to innovate beyond MMF.

**Naming alignment:** MMF uses `backtest_length` and `prediction_length`. The current codebase uses `backtest_periods`. The frequency-specific config presets (Task 0010) should adopt MMF's naming convention for compatibility.

**Testing:** Each task in this epic should include unit tests (pytest with synthetic fixtures) per the AGENTS.md guidelines, and an integration test that loads a small synthetic dataset end-to-end through the new pipeline.

Epic 0003 is the foundation that Epics 0004 and 0005 build on. All phases of the planning exercise (infrastructure analysis, MMF comparison, best-practices research) converged on this as the highest-impact first step.
