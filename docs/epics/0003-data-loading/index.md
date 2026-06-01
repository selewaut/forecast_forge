# Epic 0003: Data Loading Abstraction

## Status

**Superseded** — the loader abstraction (BaseDataLoader, SchemaAdapter, DatasetRegistry) was removed in favor of a simpler per-dataset script pattern modeled after Databricks MMF.

## What Changed

- Removed `src/forecast_forge/loaders/` (BaseDataLoader, SchemaAdapter, DatasetRegistry, WalmartDataLoader class).
- Removed `src/forecast_forge/data_processing.py` (transforms absorbed into the Walmart script).
- Replaced with `datasets/walmart/__init__.py` — a standalone `load_walmart_data()` function.
- `Forecaster.resolve_source()` now handles only `str | pd.DataFrame | DataFrame` (like MMF).
- `Forecaster._resolve_loader()` and `Forecaster.load_data()` removed.
- `run_forecast()` accepts data directly; no dataset-name indirection.

## Lesson

The MMF approach — passing column name configs (`group_id`, `date_col`, `target`) and providing data as a DataFrame — is simpler than a loader abstraction. The pipeline doesn't need to know how data was loaded; it only needs to know the column names.

See [ADR 0004 (superseding ADR 0003)](../../ADRs/0003-dataloader-interface-design.md) for the full rationale.
