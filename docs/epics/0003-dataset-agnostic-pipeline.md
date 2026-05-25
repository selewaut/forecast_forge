# Epic 0003: Data Loading Abstraction

## Status

Planned

## Objective

Decouple data loading from any single dataset so new datasets can be added through configuration alone. This is the primary differentiator versus Databricks MMF, which is tied to Delta tables and the Databricks runtime.

## Scope

- Define a `BaseDataLoader` interface supporting CSV, Parquet, Delta, and SQL sources through a config-driven schema mapping.
- Build a schema adapter that maps arbitrary column names to internal conventions (`unique_id`, `ds`, `y`), with support for exogenous regressors.
- Move Walmart-specific data loading into a dedicated `walmart/` adapter module implementing the interface.
- Add a `datasets/` config directory where each dataset declares its loader class, file paths, and column mappings.

## Tasks

- [ ] [Task 0006: Define BaseDataLoader interface and schema adapter](../tasks/0006-base-dataloader-interface.md)
- [ ] [Task 0007: Refactor Walmart data loading into adapter module](../tasks/0007-walmart-loader-adapter.md)
- [ ] [Task 0008: Add datasets config directory with validation](../tasks/0008-datasets-config-directory.md)

## Decisions

- [ ] [ADR 0003: DataLoader interface design — three-source pattern (pandas, Spark, file path)](../ADRs/0003-dataloader-interface-design.md)

## Notes

This epic replaces the current hardcoded `data_loader.py` and `config.py` modules with a pluggable system. The schema adapter is inspired by MMF's `promoted_props` pattern and the Nixtla `unique_id`/`ds`/`y` convention.

Feature engineering and data quality checks are moved to separate epics (0004, 0005) to keep each epic focused.
