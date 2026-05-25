# Task 0007: Refactor Walmart Data Loading into Adapter Module

## Status

Planned

## Epic

[Epic 0003: Data Loading Abstraction](../index.md)

## Objective

The Walmart-specific data loading and download logic is moved out of the top-level `src/forecast_forge/` into a dedicated `src/forecast_forge/loaders/walmart/` module implementing the `BaseDataLoader` interface from Task 0006. The old `config.py` and `data_loader.py` are removed.

## Scope

- Create `src/forecast_forge/loaders/walmart/__init__.py` with a `WalmartDataLoader` class implementing `BaseDataLoader`.
- Move Kaggle download logic from `data.py` into `WalmartDataLoader.download()`.
- Move column mapping and `group_id` construction from `data_loader.py` into the `SchemaAdapter`.
- Move path constants from `config.py` into `WalmartDataLoader` or a `walmart.yaml` config.
- Register the Walmart schema mapping in `datasets/walmart.yaml` (to be created in Task 0008).
- Remove old files: `src/forecast_forge/config.py`, `src/forecast_forge/data.py`, `src/forecast_forge/data_loader.py`.
- Update `Forecaster.load_data()` to use the new loader interface.
- Update `Makefile` and any scripts referencing the old module paths.

## Acceptance Criteria

- `WalmartDataLoader().load()` returns the same four DataFrames (`df_train`, `df_test`, `df_features`, `df_stores`) as the current `load_data()`.
- Column names are already mapped through the `SchemaAdapter` to canonical names.
- Running `make data` still downloads and prepares the Walmart dataset.
- Running `make run` still produces the same evaluation output.
- Old `config.py`, `data.py`, and `data_loader.py` are deleted.

## Validation

```sh
make data
make run
```

Then compare a sample of `weekly_evaluation_output/` against a previous run to verify numerical equivalence within floating-point tolerance.

## Documentation Updates

- README updated if dataset path configuration changed.
- ADR 0003 updated if the loader interface changed during implementation.

## Notes

The Kaggle download logic (`data.py`) should be preserved as-is behind the `WalmartDataLoader` — no need to rewrite the download/unzip flow, just encapsulate it.

The old `config.py` path constants (`DATA_DIR`, `TRAIN_PATH`, etc.) are used by `data.py` and `data_loader.py`. When moving, ensure the new module has equivalent path resolution — preferably relative to the project root, not hardcoded.

`Forecaster.load_data()` currently hardcodes Walmart. After this task it should look up the loader by dataset name from config.
