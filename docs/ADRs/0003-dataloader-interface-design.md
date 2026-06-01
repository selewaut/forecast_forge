# ADR 0003: DataLoader Interface Design

## Status

**Superseded** — replaced by ADR 0004 (simplified per-dataset scripts).

## Superseded By

ADR 0004: Per-Dataset Data Preparation Scripts (simplified MMF-like approach).

## Context

The original ADR proposed an abstract base class pattern (`BaseDataLoader`, `SchemaAdapter`, `DatasetRegistry`) to decouple data loading from any single dataset. After implementing and reviewing against Databricks MMF, this proved over-engineered for the project's needs.

## Lessons Learned

1. **SchemaAdapter (`unique_id`/`ds`/`y` canonical columns) was unnecessary** — the pipeline already references columns by config keys (`group_id`, `date_col`, `target`). No renaming layer is needed.
2. **`BaseDataLoader` ABC added ceremony without benefit** — each dataset has unique loading logic; a shared interface doesn't reduce code.
3. **`DatasetRegistry` + `datasets/{name}.yaml` was indirect** — passing data directly to `run_forecast()` is simpler and more transparent.
4. **MMF's approach is simpler and sufficient**: pass column names as config params, provide data as a DataFrame, write a per-dataset script for preparation.

## Decision (Current)

Use a per-dataset script pattern instead of a loader abstraction:

1. Each dataset lives in `datasets/{name}/` with a `load_{name}_data() -> pd.DataFrame` function.
2. The function handles download, loading, merging, and any dataset-specific transforms.
3. The pipeline (`Forecaster`, `run_forecast()`) only accepts `str | pd.DataFrame | Spark DataFrame` — no loader interface.
4. Column names are configured at the `run_forecast()` call site via `group_id`, `date_col`, `target` params (same as MMF).

## Consequences

- Simpler: no abstract classes, no registry, no schema adapter.
- Simpler: adding a dataset means writing one function and passing its output to `run_forecast()`.
- Clearer: the data preparation contract is "return a DataFrame with the columns your config references."
- Removed: `src/forecast_forge/loaders/`, `src/forecast_forge/data_processing.py`, `datasets/walmart.yaml`.
