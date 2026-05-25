# Epic 0005: Data Quality & Configuration Presets

## Status

Planned

## Objective

Add data validation as a mandatory pipeline stage and create frequency-specific configuration presets so datasets of different granularities (daily, weekly, monthly) work out of the box.

## Scope

- Port data quality checks from Databricks MMF (`DataQualityChecks`): configuration validation, missing date detection, negative value threshold, training-length ratio check, and external regressor null checks.
- Create frequency-specific config presets (`forecasting_conf_daily.yaml`, `_weekly.yaml`, `_monthly.yaml`, `_hourly.yaml`) with appropriate defaults.
- Align config parameter naming with MMF convention (`backtest_periods` → `backtest_length`, `prediction_length` stays).

## Tasks

- [ ] [Task 0011: Port data quality checks from MMF](../tasks/0011-data-quality-checks.md)
- [ ] [Task 0012: Create frequency-specific config presets](../tasks/0012-freq-specific-configs.md)
- [ ] [Task 0013: Align config naming to MMF conventions](../tasks/0013-config-naming-alignment.md)

## Decisions

- [ ] [ADR 0005: Data quality check design and threshold configuration](../ADRs/0005-data-quality-thresholds.md)

## Notes

Data quality checks are mandatory rather than optional (MMF defaults them to off). This prevents silent pipeline failures from bad input data.

Depends on Epic 0003 (data loading) for the schema adapter and Epic 0004 (feature engineering) for post-transform validation.
