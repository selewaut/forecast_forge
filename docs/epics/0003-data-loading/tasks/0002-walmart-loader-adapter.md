# Task 0002: Refactor Walmart Data Loading into Adapter Module

## Status

Done

## Epic

[Epic 0003: Data Loading Abstraction](../index.md)

## Objective

The Walmart-specific data loading and download logic is moved out of the top-level `src/forecast_forge/` into a dedicated `src/forecast_forge/loaders/walmart/` module implementing the `BaseDataLoader` interface from Task 0001. The old `config.py`, `data.py`, and `data_loader.py` are removed.

## Scope

- Create `src/forecast_forge/loaders/walmart/__init__.py` with a `WalmartDataLoader` class implementing `BaseDataLoader`.
- Move Kaggle download logic from `data.py` into `WalmartDataLoader.download()`.
- Move path constants from `config.py` into `WalmartDataLoader`.
- Move group_id construction and date parsing from `data_loader.py` into `WalmartDataLoader.load()`.
- Merge features and stores into the training DataFrame inside `WalmartDataLoader.load()` — returns a single merged DataFrame instead of four separate ones.
- Remove old files: `src/forecast_forge/config.py`, `src/forecast_forge/data.py`, `src/forecast_forge/data_loader.py`.
- Update `Forecaster.load_data()` to use `WalmartDataLoader`.
- Update `Makefile` to use the new module path.
- Register the Walmart schema mapping in `datasets/walmart.yaml` (deferred to Task 0003).
- SchemaAdapter integration deferred — column names retain Walmart conventions for now (`group_id`, `date`, `weekly_sales`).

## Acceptance Criteria

- `WalmartDataLoader().load()` returns a single merged `pd.DataFrame` with Walmart column names (features and stores already joined).
- Running `make data` still downloads and prepares the Walmart dataset.
- Running `make run` still produces the same evaluation output.
- Old `config.py`, `data.py`, and `data_loader.py` are deleted.
- All 21 existing tests pass.

## Validation

```sh
uv run pytest tests/ -v
```

Expected: 21 tests pass.

```sh
make data
make run
```

Expected: pipeline runs successfully.

## Documentation Updates

- Task doc itself updated to reflect incremental approach.

## Incremental Approach

SchemaAdapter mapping (`unique_id`, `ds`, `y`) and canonical column names are deferred to avoid rippling changes through `Forecaster`, `pre_process_data`, `run_forecast`, `univariate_weekly.py`, and model configs in a single task. The `WalmartDataLoader` keeps the existing Walmart column names (`group_id`, `date`, `weekly_sales`, etc.) and the downstream pipeline remains unchanged.

The `SchemaAdapter` is still available and tested — new dataset loaders can use it directly. Applying it to Walmart data will happen in a follow-up task after the config layer is updated to reference canonical names.

## Notes

The Kaggle download logic (`data.py`) is preserved as-is behind `WalmartDataLoader.download()` — no rewrite of the download/unzip flow, just encapsulation.

The old `config.py` path constants (`DATA_DIR`, `TRAIN_PATH`, etc.) are now module-level constants in `src/forecast_forge/loaders/walmart/__init__.py` with equivalent project-root-relative path resolution.

`Forecaster.load_data()` creates a `WalmartDataLoader` internally. Future work (Task 0003) will make this dataset-agnostic via a `dataset_name` config parameter.
